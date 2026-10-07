// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  FlyToInterpolator,
  MapView,
  type Layer,
  type MapViewState,
  type PickingInfo,
  type Viewport
} from '@deck.gl/core';
import type {Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import {ArrowDeck} from '../../../arrow-deck';
import {getDeckExampleProps} from '../../../deck-example-device';
import {SpatialAnalysisDeckEffect, type SpatialAnalysisEffectStats} from './analysis-effect';
import type {EncodableInstance, SceneFrame, ScenePointerEvent, ViewState} from './types';

/** Device limits the host reports to scenes (see `ctx.limits`). */
export type HostLimits = {
  maxStorageBuffersPerShaderStage: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
};

/** Storage buffers per stage the host asks for (when the adapter has them). */
export const REQUESTED_STORAGE_BUFFERS_PER_STAGE = 16;

/** Frames a scene renders before the host reports it as ready (used by headless captures). */
export const READY_FRAME_COUNT = 20;

/** Basemap themes the host can switch between. */
export type HostTheme = 'light' | 'dark';

const BASEMAP_STYLES: Record<HostTheme, string> = {
  light: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
  dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
};

const CONTROLLER = {
  dragPan: true,
  scrollZoom: {smooth: true, speed: 0.02},
  doubleClickZoom: true,
  touchZoom: true,
  dragRotate: false
};

type HostViewState = MapViewState & Required<ViewState>;

/** What the host needs from a running scene instance. */
export type HostInstance = EncodableInstance & {
  getCompiledGraphs: () => readonly CompiledGPUCommandGraph<never>[];
  getLayers: () => Layer[];
  onClick?: (event: ScenePointerEvent) => boolean;
  onDragStart?: (event: ScenePointerEvent) => boolean;
  onDrag?: (event: ScenePointerEvent) => void;
  onDragEnd?: (event: ScenePointerEvent) => void;
  getTooltip?: (event: ScenePointerEvent) => string | null;
  destroy: () => void;
};

/** Services the host passes to a scene factory. */
export type HostServices = {
  device: Device;
  signal: AbortSignal;
  getViewport: () => Viewport | null;
  /** Moves the camera (animated when `transitionMs` > 0); omitted fields keep their value. */
  flyTo: (view: Partial<ViewState>, options?: {transitionMs?: number}) => void;
  /** Current camera. */
  getViewState: () => ViewState;
  /** Re-runs the instance's `getTooltip` for the last hover position. */
  refreshTooltip: () => void;
  /** Limits granted to the device. */
  limits: HostLimits;
  /** Re-reads `getLayers()` of the active instance and hands them to Deck. */
  updateLayers: () => void;
  setMapDragEnabled: (enabled: boolean) => void;
};

/** One scene activation request. */
export type HostActivation<Instance extends HostInstance> = {
  id: string;
  view: ViewState;
  create: (services: HostServices) => Promise<Instance>;
};

/** A text marker anchored to a map coordinate. */
export type HostCallout = {coordinate: readonly [number, number]; text: string};

/** Observable host status, mirrored on `document.body.dataset` by the shell. */
export type HostState = {
  sceneId: string | null;
  ready: boolean;
  error: string | null;
  /** Number of new compiled-graph objects seen after the first set (compile-time changes). */
  rebuildCount: number;
};

/**
 * Owns the one Deck, MapLibre basemap and WebGPU device of the showcase, and runs scenes on them.
 *
 * Scenes are activated with a generation counter and an AbortController so a slow `create` never
 * overwrites a newer scene, and a replaced instance is destroyed only two animation frames after
 * Deck has dropped its layers (Deck applies layer changes on its next frame).
 */
export class DeckHost {
  readonly element: HTMLDivElement;
  readonly state: HostState = {sceneId: null, ready: false, error: null, rebuildCount: 0};
  /** Called when the state changes (ready, error, scene). */
  onStateChange: (() => void) | null = null;
  /** Called with a one-line status while a scene loads. */
  onStatus: ((message: string) => void) | null = null;

  private readonly basemapElement: HTMLDivElement;
  private readonly calloutElement: HTMLDivElement;
  private readonly map: maplibregl.Map;
  private readonly resizeObserver: ResizeObserver;
  private readonly deviceReady: Promise<Device>;
  private rejectDevice: (error: Error) => void = () => {};
  private resolveDevice: (device: Device) => void = () => {};
  private deck!: ArrowDeck<MapView>;
  private device: Device | null = null;
  private effect: SpatialAnalysisDeckEffect | null = null;
  private viewState: HostViewState;
  private instance: HostInstance | null = null;
  private abortController: AbortController | null = null;
  private generation = 0;
  private draggingInstance: HostInstance | null = null;
  private seenGraphs = new Set<CompiledGPUCommandGraph<never>>();
  private callout: HostCallout | null = null;
  private theme: HostTheme;
  private finalized = false;
  private destroyed = false;
  private started = false;
  private lastHover: {info: PickingInfo; event: unknown} | null = null;
  private modifiers = {shiftKey: false, altKey: false, ctrlKey: false, metaKey: false};
  private readonly trackModifiers = (event: Event) => {
    const {shiftKey, altKey, ctrlKey, metaKey} = event as KeyboardEvent;
    if (shiftKey === undefined) return;
    this.modifiers = {shiftKey, altKey, ctrlKey, metaKey};
  };

  constructor(options: {theme: HostTheme; initialView: ViewState}) {
    this.theme = options.theme;
    this.viewState = toHostViewState(options.initialView);
    this.element = document.createElement('div');
    this.element.className = 'map-stage';
    this.basemapElement = document.createElement('div');
    this.basemapElement.className = 'map-basemap';
    this.calloutElement = document.createElement('div');
    this.calloutElement.className = 'map-callout';
    this.calloutElement.hidden = true;
    this.element.append(this.basemapElement, this.calloutElement);
    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'keydown', 'keyup'] as const) {
      // Capture so modifier state is current before Deck dispatches its own handlers.
      (type.startsWith('key') ? window : this.element).addEventListener(type, this.trackModifiers, {
        capture: true,
        passive: true
      });
    }

    this.deviceReady = new Promise<Device>((resolve, reject) => {
      this.resolveDevice = resolve;
      this.rejectDevice = reject;
    });
    // A rejected device promise is reported through `state.error`; avoid an unhandled rejection.
    this.deviceReady.catch(() => {});

    this.map = new maplibregl.Map({
      container: this.basemapElement,
      style: BASEMAP_STYLES[this.theme],
      center: [this.viewState.longitude, this.viewState.latitude],
      zoom: this.viewState.zoom,
      interactive: false,
      attributionControl: false
    });
    this.map.addControl(new maplibregl.AttributionControl({compact: true}), 'bottom-right');
    this.resizeObserver = new ResizeObserver(() => this.map.resize());
    this.resizeObserver.observe(this.element);
    this.map.on('error', () =>
      this.onStatus?.('Basemap tiles unavailable; GPU outputs still render.')
    );
  }

  /** Creates Deck (which requests the WebGPU device). Call once the element is in the page. */
  start(): void {
    if (this.started) return;
    this.started = true;
    // The adapter is queried first so the device can be created with raised limits.
    getHostDeviceProps()
      .then(props => {
        if (!this.destroyed) this.createDeck(props);
      })
      .catch(error => {
        const failure = error instanceof Error ? error : new Error(String(error));
        this.reportError(failure);
        this.rejectDevice(failure);
      });
  }

  private createDeck(deviceProps: Awaited<ReturnType<typeof getHostDeviceProps>>): void {
    this.deck = new ArrowDeck<MapView>({
      parent: this.element,
      ...deviceProps,
      views: new MapView({id: 'map', repeat: false}),
      viewState: this.viewState,
      controller: CONTROLLER,
      style: {background: 'transparent'},
      _animate: true,
      layers: [],
      effects: [],
      getTooltip: info => this.instance?.getTooltip?.(this.toEvent(info)) ?? null,
      onHover: (info, event) => {
        this.lastHover = {info, event};
      },
      onClick: info => {
        this.instance?.onClick?.(this.toEvent(info));
      },
      onDragStart: (info, event) => {
        const instance = this.instance;
        if (instance?.onDragStart?.(this.toEvent(info))) {
          this.draggingInstance = instance;
          event.stopPropagation();
        }
      },
      onDrag: (info, event) => {
        if (this.draggingInstance && this.draggingInstance === this.instance) {
          this.draggingInstance.onDrag?.(this.toEvent(info));
          event.stopPropagation();
        }
      },
      onDragEnd: (info, event) => {
        if (this.draggingInstance && this.draggingInstance === this.instance) {
          this.draggingInstance.onDragEnd?.(this.toEvent(info));
          event.stopPropagation();
        }
        this.draggingInstance = null;
      },
      onViewStateChange: ({viewState}) => {
        // Drop transition props so an interpolated state never restarts a transition.
        this.viewState = toHostViewState(viewState as ViewState);
        this.deck.setProps({viewState: this.viewState});
        this.synchronizeBasemap();
        this.positionCallout();
      },
      onLoad: ({device}) => {
        if (device.type !== 'webgpu') {
          const error = new Error('The spatial-analysis showcase requires WebGPU');
          this.reportError(error);
          this.rejectDevice(error);
          return;
        }
        this.device = device;
        const canvas = device.getDefaultCanvasContext().canvas;
        if (canvas instanceof HTMLCanvasElement) {
          Object.assign(canvas.style, {
            position: 'absolute',
            inset: '0',
            zIndex: '1',
            background: 'transparent'
          });
        }
        this.effect = new SpatialAnalysisDeckEffect(device, error => this.reportError(error));
        this.deck.setProps({effects: [this.effect]});
        this.resolveDevice(device);
      },
      onAfterRender: () => {
        this.positionCallout();
        if (
          !this.state.ready &&
          this.instance &&
          this.effect &&
          this.effect.stats.frameCount >= READY_FRAME_COUNT
        ) {
          this.state.ready = true;
          this.onStateChange?.();
        }
      },
      onFinalize: () => {
        this.finalized = true;
      }
    });
  }

  /** Limits granted to the device (spec defaults until the device exists). */
  get limits(): HostLimits {
    const limits = this.device?.limits;
    return {
      maxStorageBuffersPerShaderStage: limits?.maxStorageBuffersPerShaderStage ?? 8,
      maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize ?? 134217728,
      maxBufferSize: limits?.maxBufferSize ?? 268435456
    };
  }

  /** Current camera. */
  getViewState(): ViewState {
    const {longitude, latitude, zoom, pitch, bearing} = this.viewState;
    return {longitude, latitude, zoom, pitch, bearing};
  }

  /**
   * Re-evaluates `getTooltip` for the last hovered position and updates the tooltip. Use it after
   * an asynchronous probe resolves while the pointer rests on the map.
   */
  refreshTooltip(): void {
    const hover = this.lastHover;
    if (!hover || !this.instance?.getTooltip) return;
    const widgetManager = (
      this.deck as unknown as {
        widgetManager?: {onHover?: (info: PickingInfo, event: unknown) => void};
      }
    ).widgetManager;
    try {
      widgetManager?.onHover?.(hover.info, hover.event);
    } catch {
      // The tooltip is cosmetic; a stale picking info must never break the scene.
    }
  }

  /** Resolves with the WebGPU device, or rejects when WebGPU is unavailable. */
  whenDeviceReady(): Promise<Device> {
    return this.deviceReady;
  }

  /** CPU-side frame statistics of the active scene. */
  get stats(): SpatialAnalysisEffectStats {
    return this.effect?.stats ?? {frameCount: 0, framesPerSecond: 0, encodeMilliseconds: 0};
  }

  /** WebGPU device once Deck has created it. */
  get gpuDevice(): Device | null {
    return this.device;
  }

  /** The active scene instance, if any. */
  get activeInstance(): HostInstance | null {
    return this.instance;
  }

  /** Latest Deck viewport. */
  get viewport(): Viewport | null {
    return this.effect?.viewport ?? null;
  }

  /**
   * Replaces the active scene. Resolves with the new instance, or `null` when a newer activation
   * superseded this one.
   */
  async activate<Instance extends HostInstance>(
    activation: HostActivation<Instance>
  ): Promise<Instance | null> {
    const currentGeneration = ++this.generation;
    this.abortController?.abort();
    const abortController = new AbortController();
    this.abortController = abortController;
    this.clearActive();
    this.lastHover = null;
    this.state.sceneId = activation.id;
    this.state.ready = false;
    this.state.error = null;
    this.state.rebuildCount = 0;
    this.seenGraphs = new Set();
    this.setCallout(null);
    this.setViewState(activation.view);
    this.onStateChange?.();
    let device: Device;
    try {
      device = await this.deviceReady;
    } catch {
      return null;
    }
    if (currentGeneration !== this.generation || this.finalized) return null;
    try {
      const instance = await activation.create({
        device,
        signal: abortController.signal,
        getViewport: () => this.viewport,
        flyTo: (view, options) => {
          if (currentGeneration === this.generation) {
            this.setViewState({...view, transitionMs: options?.transitionMs ?? 0});
          }
        },
        getViewState: () => this.getViewState(),
        refreshTooltip: () => {
          if (currentGeneration === this.generation) this.refreshTooltip();
        },
        limits: this.limits,
        updateLayers: () => {
          if (currentGeneration === this.generation && this.instance) {
            this.deck.setProps({layers: this.instance.getLayers()});
          }
        },
        setMapDragEnabled: enabled => {
          if (currentGeneration === this.generation) {
            this.deck.setProps({controller: {...CONTROLLER, dragPan: enabled}});
          }
        }
      });
      if (currentGeneration !== this.generation || this.finalized) {
        instance.destroy();
        return null;
      }
      this.instance = instance;
      for (const graph of instance.getCompiledGraphs()) this.seenGraphs.add(graph);
      this.effect?.setMode(instance);
      this.deck.setProps({layers: instance.getLayers()});
      return instance;
    } catch (error) {
      if (currentGeneration === this.generation && !abortController.signal.aborted) {
        this.reportError(error instanceof Error ? error : new Error(String(error)));
      }
      return null;
    }
  }

  /** Stops encoding and drawing the active scene and destroys it. */
  deactivate(): void {
    this.generation++;
    this.abortController?.abort();
    this.abortController = null;
    this.clearActive();
    this.state.sceneId = null;
    this.state.ready = false;
    this.state.error = null;
    this.setCallout(null);
    this.onStateChange?.();
  }

  /** Counts compiled graph objects that appeared since the last call (compile-time rebuilds). */
  pollRebuilds(): {graphCount: number; nodeCount: number; rebuildCount: number} {
    const graphs = this.instance?.getCompiledGraphs() ?? [];
    let rebuilt = 0;
    for (const graph of graphs) {
      if (!this.seenGraphs.has(graph)) {
        this.seenGraphs.add(graph);
        rebuilt++;
      }
    }
    this.state.rebuildCount += rebuilt;
    return {
      graphCount: graphs.length,
      nodeCount: graphs.reduce((total, graph) => total + graph.stats.nodeOrder.length, 0),
      rebuildCount: this.state.rebuildCount
    };
  }

  /** Moves the camera, animating when `transitionMs` is positive. */
  setViewState(view: Partial<ViewState> & {transitionMs?: number}): void {
    const defined = Object.fromEntries(
      Object.entries(view).filter(([, value]) => value !== undefined)
    );
    const next = toHostViewState({...this.viewState, ...defined});
    const transitionMs = view.transitionMs ?? 0;
    this.viewState = next;
    if (!this.deck) return;
    if (transitionMs > 0) {
      this.deck.setProps({
        viewState: {
          ...next,
          transitionDuration: transitionMs,
          transitionInterpolator: new FlyToInterpolator()
        }
      });
    } else {
      this.deck.setProps({viewState: next});
      this.synchronizeBasemap();
    }
    this.positionCallout();
  }

  /** Shows a text marker anchored to a coordinate, or hides it with `null`. */
  setCallout(callout: HostCallout | null): void {
    this.callout = callout;
    this.calloutElement.hidden = !callout;
    if (callout) this.calloutElement.textContent = callout.text;
    this.positionCallout();
  }

  /** Switches the basemap style. */
  setTheme(theme: HostTheme): void {
    if (theme === this.theme) return;
    this.theme = theme;
    this.map.setStyle(BASEMAP_STYLES[theme]);
  }

  /** Stops the render loop while the map is off screen. */
  setActive(active: boolean): void {
    this.deck?.setProps({_animate: active});
    if (active) this.map.resize();
  }

  /** Releases the scene, Deck, the basemap and the device. */
  destroy(): void {
    this.destroyed = true;
    for (const type of ['pointerdown', 'pointermove', 'pointerup'] as const) {
      this.element.removeEventListener(type, this.trackModifiers, {capture: true});
    }
    window.removeEventListener('keydown', this.trackModifiers, {capture: true});
    window.removeEventListener('keyup', this.trackModifiers, {capture: true});
    this.deactivate();
    this.resizeObserver.disconnect();
    this.deck?.finalize();
    this.map.remove();
    this.element.remove();
  }

  private clearActive(): void {
    const previous = this.instance;
    this.effect?.setMode(null);
    this.instance = null;
    this.draggingInstance = null;
    this.deck?.setProps({layers: [], controller: CONTROLLER});
    // Deck applies layer changes on its next animation frame; wait two frames before freeing the
    // buffers those layers bind.
    if (previous) requestAnimationFrame(() => requestAnimationFrame(() => previous.destroy()));
  }

  private toEvent(info: PickingInfo): ScenePointerEvent {
    return {...toPointerEvent(info), ...this.modifiers};
  }

  private reportError(error: Error): void {
    this.state.error = error.message;
    this.onStateChange?.();
  }

  private synchronizeBasemap(): void {
    this.map.jumpTo({
      center: [this.viewState.longitude, this.viewState.latitude],
      zoom: this.viewState.zoom,
      pitch: this.viewState.pitch,
      bearing: this.viewState.bearing
    });
  }

  private positionCallout(): void {
    if (!this.callout) return;
    const viewport = this.deck?.getViewports()[0];
    if (!viewport) return;
    const [x, y] = viewport.project([this.callout.coordinate[0], this.callout.coordinate[1]]);
    this.calloutElement.style.transform = `translate(${x}px, ${y}px)`;
  }
}

/**
 * Deck device props. `timestamp-query` is requested if the adapter exposes it, so compiled graphs
 * can report per-node GPU time. The adapter's storage-buffer limits are raised (up to
 * {@link REQUESTED_STORAGE_BUFFERS_PER_STAGE} buffers per stage, the adapter's maximum binding and
 * buffer sizes) because several analyses bind more than the default 8 storage buffers; an adapter
 * that cannot provide more still creates the same device.
 */
async function getHostDeviceProps() {
  const props = getDeckExampleProps({deviceType: 'webgpu'});
  if (!('deviceProps' in props)) return props;
  let requiredLimits: Record<string, number> | undefined;
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) {
      requiredLimits = {
        maxStorageBuffersPerShaderStage: Math.min(
          adapter.limits.maxStorageBuffersPerShaderStage,
          REQUESTED_STORAGE_BUFFERS_PER_STAGE
        ),
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
      };
    }
  } catch {
    // No adapter information: fall back to default limits.
  }
  return {
    deviceProps: {
      ...props.deviceProps,
      optionalFeatures: ['timestamp-query' as const],
      ...(requiredLimits ? {requiredLimits} : {})
    }
  };
}

function toHostViewState(view: ViewState): HostViewState {
  return {
    longitude: view.longitude,
    latitude: view.latitude,
    zoom: view.zoom,
    pitch: view.pitch ?? 0,
    bearing: view.bearing ?? 0,
    minZoom: 2,
    maxZoom: 19
  };
}

function toPointerEvent(
  info: PickingInfo
): Omit<ScenePointerEvent, 'shiftKey' | 'altKey' | 'ctrlKey' | 'metaKey'> {
  const coordinate =
    info.coordinate && info.coordinate.length >= 2
      ? ([info.coordinate[0], info.coordinate[1]] as const)
      : null;
  return {info, coordinate, pixel: [info.x, info.y]};
}

export type {SceneFrame};
