// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MapView,
  type MapViewState,
  type PickingInfo,
  type ViewStateChangeParameters
} from '@deck.gl/core';
import type {Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import {ArrowDeck} from '../arrow-deck';
import {type DeckExampleDeviceOptions, getDeckExampleProps} from '../deck-example-device';
import {createBasemapContainer, createMapGraphsPanel, createStandaloneContainer} from './app-ui';
import {createMapGraphsDataCatalog} from './map-graphs-data';
import {MapGraphsDeckEffect} from './map-graphs-effect';
import type {
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent,
  MapGraphsViewState
} from './map-graphs-mode';
import {MAP_GRAPHS_MODES} from './modes/index';

const BASEMAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
/** Frames a mode renders before the explorer reports it as ready (used by headless captures). */
const READY_FRAME_COUNT = 20;
const CONTROLLER = {
  dragPan: true,
  scrollZoom: {smooth: true, speed: 0.02},
  doubleClickZoom: true,
  touchZoom: true,
  dragRotate: false
};

type ExplorerViewState = MapViewState & Required<MapGraphsViewState>;

/** Options for {@link createMapGraphsExplorerDeck}. */
export type MapGraphsExplorerOptions = DeckExampleDeviceOptions & {
  /** Initial mode ID. Defaults to the `?mode=` URL parameter, then the first registered mode. */
  modeId?: string;
  /** Skip network requests and use deterministic synthetic data. Defaults to `?data=synthetic`. */
  forceSyntheticData?: boolean;
  /** Mode registry. Defaults to {@link MAP_GRAPHS_MODES}. */
  modes?: readonly MapGraphsModeDefinition[];
};

/** Observable explorer state, mirrored on `document.body.dataset` for headless captures. */
export type MapGraphsExplorerState = {
  modeId: string | null;
  ready: boolean;
  error: string | null;
  instance: MapGraphsModeInstance | null;
  /** Number of distinct compiled graph objects seen per mode beyond the first set. */
  rebuildCount: number;
};

declare global {
  // eslint-disable-next-line no-var
  var mapGraphsExplorer: (MapGraphsExplorerState & {selectMode: (id: string) => void}) | undefined;
}

/**
 * Creates the WebGPU map-graphs explorer: a MapLibre basemap under a Deck MapView whose effect
 * encodes the active mode's compiled `GPUCommandGraph`s into Deck's frame encoder.
 */
export function createMapGraphsExplorerDeck(
  parent?: HTMLDivElement,
  options: MapGraphsExplorerOptions = {}
): ArrowDeck<MapView> {
  const {modeId, forceSyntheticData, modes = MAP_GRAPHS_MODES, ...deviceOptions} = options;
  const searchParameters =
    typeof window === 'undefined'
      ? new URLSearchParams()
      : new URLSearchParams(window.location.search);
  const initialModeId = modeId ?? searchParameters.get('mode') ?? modes[0].id;
  const data = createMapGraphsDataCatalog({
    forceSynthetic: forceSyntheticData ?? searchParameters.get('data') === 'synthetic'
  });
  const ownsContainer = !parent;
  const container = parent ?? createStandaloneContainer();
  if (getComputedStyle(container).position === 'static') container.style.position = 'relative';

  const initialDefinition = modes.find(mode => mode.id === initialModeId) ?? modes[0];
  let viewState = toExplorerViewState(initialDefinition.initialViewState);
  const basemapContainer = createBasemapContainer(container);
  const map = new maplibregl.Map({
    container: basemapContainer,
    style: BASEMAP_STYLE,
    center: [viewState.longitude, viewState.latitude],
    zoom: viewState.zoom,
    interactive: false,
    attributionControl: false
  });
  map.addControl(new maplibregl.AttributionControl({compact: true}), 'bottom-right');
  const resizeObserver = new ResizeObserver(() => map.resize());
  resizeObserver.observe(container);

  const state: MapGraphsExplorerState = {
    modeId: null,
    ready: false,
    error: null,
    instance: null,
    rebuildCount: 0
  };
  let device: Device | null = null;
  let effect: MapGraphsDeckEffect | null = null;
  let activeDefinition: MapGraphsModeDefinition | null = null;
  let activeAbortController: AbortController | null = null;
  let generation = 0;
  let draggingMode: MapGraphsModeInstance | null = null;
  let finalized = false;
  let seenGraphs = new Set<CompiledGPUCommandGraph<unknown>>();
  let deck: ArrowDeck<MapView>;

  const panel = createMapGraphsPanel(
    container,
    modes.map(mode => ({id: mode.id, title: mode.title})),
    id => activateMode(id)
  );
  map.on('error', () =>
    panel.setBasemapStatus('Basemap tiles unavailable; GPU outputs still render.')
  );
  map.on('load', () => panel.setBasemapStatus(''));

  const footerTimer = setInterval(updateFooter, 500);

  deck = new ArrowDeck<MapView>({
    parent: container,
    ...getExplorerDeviceProps(deviceOptions),
    views: new MapView({id: 'map', repeat: false}),
    viewState,
    controller: CONTROLLER,
    style: {background: 'transparent'},
    _animate: true,
    layers: [],
    effects: [],
    getTooltip: info => {
      const event = toPointerEvent(info);
      return state.instance?.getTooltip?.(event) ?? null;
    },
    onClick: info => {
      state.instance?.onClick?.(toPointerEvent(info));
    },
    onDragStart: (info, event) => {
      const instance = state.instance;
      if (instance?.onDragStart?.(toPointerEvent(info))) {
        draggingMode = instance;
        event.stopPropagation();
      }
    },
    onDrag: (info, event) => {
      if (draggingMode && draggingMode === state.instance) {
        draggingMode.onDrag?.(toPointerEvent(info));
        event.stopPropagation();
      }
    },
    onDragEnd: (info, event) => {
      if (draggingMode && draggingMode === state.instance) {
        draggingMode.onDragEnd?.(toPointerEvent(info));
        event.stopPropagation();
      }
      draggingMode = null;
    },
    onViewStateChange: ({viewState: nextViewState}: ViewStateChangeParameters) => {
      viewState = nextViewState as ExplorerViewState;
      deck.setProps({viewState});
      synchronizeBasemap(map, viewState);
    },
    onLoad: ({device: loadedDevice}) => {
      if (loadedDevice.type !== 'webgpu') {
        reportError(new Error('The map-graphs explorer requires WebGPU'));
        return;
      }
      device = loadedDevice;
      const canvas = loadedDevice.getDefaultCanvasContext().canvas;
      if (canvas instanceof HTMLCanvasElement) {
        Object.assign(canvas.style, {
          position: 'absolute',
          inset: '0',
          zIndex: '1',
          background: 'transparent'
        });
      }
      effect = new MapGraphsDeckEffect(loadedDevice, reportError);
      deck.setProps({effects: [effect]});
      activateMode(initialDefinition.id);
    },
    onAfterRender: () => {
      if (
        !state.ready &&
        state.instance &&
        effect &&
        effect.stats.frameCount >= READY_FRAME_COUNT
      ) {
        state.ready = true;
        document.body.dataset.mapGraphsReady = 'true';
      }
    },
    onFinalize: () => {
      finalized = true;
      generation++;
      activeAbortController?.abort();
      effect?.setMode(null);
      const instance = state.instance;
      state.instance = null;
      instance?.destroy();
      clearInterval(footerTimer);
      resizeObserver.disconnect();
      panel.destroy();
      map.remove();
      if (ownsContainer) container.remove();
      if (globalThis.mapGraphsExplorer?.instance === instance)
        globalThis.mapGraphsExplorer = undefined;
    }
  });

  globalThis.mapGraphsExplorer = Object.assign(state, {
    selectMode: (id: string) => activateMode(id)
  });
  return deck;

  function activateMode(id: string): void {
    if (finalized || !device || !effect) return;
    const definition = modes.find(mode => mode.id === id);
    if (!definition) {
      reportError(new Error(`Unknown map-graphs mode "${id}"`));
      return;
    }
    const currentGeneration = ++generation;
    activeAbortController?.abort();
    const abortController = new AbortController();
    activeAbortController = abortController;

    // Stop encoding and drawing the previous mode, then destroy it once Deck has dropped its layers.
    const previous = state.instance;
    effect.setMode(null);
    state.instance = null;
    state.ready = false;
    state.error = null;
    state.modeId = definition.id;
    delete document.body.dataset.mapGraphsReady;
    delete document.body.dataset.mapGraphsError;
    document.body.dataset.mapGraphsMode = definition.id;
    deck.setProps({layers: [], controller: CONTROLLER});
    if (previous) destroyAfterFrames(previous);

    activeDefinition = definition;
    seenGraphs = new Set();
    state.rebuildCount = 0;
    panel.setActiveTab(definition.id);
    const controls = panel.beginMode(definition);
    viewState = toExplorerViewState(definition.initialViewState);
    deck.setProps({viewState});
    synchronizeBasemap(map, viewState);
    panel.setStatus('Loading data and compiling GPU graphs…');

    const activeDevice = device;
    definition
      .create({
        device: activeDevice,
        data,
        controls,
        signal: abortController.signal,
        getViewport: () => effect?.viewport ?? null,
        updateLayers: () => {
          if (currentGeneration === generation && state.instance) {
            deck.setProps({layers: state.instance.getLayers()});
          }
        },
        setStatus: message => {
          if (currentGeneration === generation) panel.setStatus(message);
        },
        setMapDragEnabled: enabled => {
          if (currentGeneration === generation) {
            deck.setProps({controller: {...CONTROLLER, dragPan: enabled}});
          }
        }
      })
      .then(instance => {
        if (currentGeneration !== generation || finalized) {
          instance.destroy();
          return;
        }
        state.instance = instance;
        for (const graph of instance.getCompiledGraphs()) seenGraphs.add(graph);
        effect?.setMode(instance);
        deck.setProps({layers: instance.getLayers()});
        panel.setStatus('');
        updateFooter();
      })
      .catch(error => {
        if (currentGeneration === generation) reportError(error);
      });
  }

  function destroyAfterFrames(instance: MapGraphsModeInstance): void {
    // Deck applies layer changes on its next animation frame; wait two frames before freeing the
    // buffers those layers bind.
    requestAnimationFrame(() => requestAnimationFrame(() => instance.destroy()));
  }

  function reportError(error: Error): void {
    state.error = error.message;
    document.body.dataset.mapGraphsError = error.message;
    panel.setStatus(`Error: ${error.message}`);
  }

  function updateFooter(): void {
    const instance = state.instance;
    const lines: string[] = [];
    if (instance && activeDefinition) {
      const graphs = instance.getCompiledGraphs();
      let rebuilt = 0;
      for (const graph of graphs) {
        if (!seenGraphs.has(graph)) {
          seenGraphs.add(graph);
          rebuilt++;
        }
      }
      state.rebuildCount += rebuilt;
      const nodeCount = graphs.reduce((total, graph) => total + graph.stats.nodeOrder.length, 0);
      lines.push(
        `${graphs.length} compiled graph${graphs.length === 1 ? '' : 's'} · ${nodeCount} nodes · ` +
          `${state.rebuildCount} rebuild${state.rebuildCount === 1 ? '' : 's'} (compile-time changes only)`
      );
    }
    if (effect) {
      const {framesPerSecond, encodeMilliseconds, frameCount} = effect.stats;
      lines.push(
        `${frameCount} frames encoded · ${framesPerSecond.toFixed(0)} fps · ${encodeMilliseconds.toFixed(2)} ms CPU encode`
      );
    }
    lines.push(
      data.forceSynthetic
        ? 'Data: deterministic synthetic (?data=synthetic)'
        : 'Data: deck.gl-data (synthetic fallback offline)'
    );
    panel.setFooter(lines);
  }
}

/**
 * Deck device props for the explorer. When Deck creates the device, `timestamp-query` is requested
 * if the adapter exposes it, so compiled graphs can report per-node GPU time
 * (`encoding.readTimings()`); an adapter without it still creates the same device.
 */
function getExplorerDeviceProps(deviceOptions: DeckExampleDeviceOptions) {
  const props = getDeckExampleProps({...deviceOptions, deviceType: 'webgpu'});
  return 'deviceProps' in props
    ? {deviceProps: {...props.deviceProps, optionalFeatures: ['timestamp-query' as const]}}
    : props;
}

function toExplorerViewState(view: MapGraphsViewState): ExplorerViewState {
  return {
    longitude: view.longitude,
    latitude: view.latitude,
    zoom: view.zoom,
    pitch: view.pitch ?? 0,
    bearing: view.bearing ?? 0,
    minZoom: 3,
    maxZoom: 19
  };
}

function toPointerEvent(info: PickingInfo): MapGraphsPointerEvent {
  const coordinate =
    info.coordinate && info.coordinate.length >= 2
      ? ([info.coordinate[0], info.coordinate[1]] as const)
      : null;
  return {info, coordinate, pixel: [info.x, info.y]};
}

function synchronizeBasemap(map: maplibregl.Map, viewState: ExplorerViewState): void {
  map.jumpTo({
    center: [viewState.longitude, viewState.latitude],
    zoom: viewState.zoom,
    pitch: viewState.pitch,
    bearing: viewState.bearing
  });
}
