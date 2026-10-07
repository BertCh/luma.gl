// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  FlyToInterpolator,
  MapView,
  WebMercatorViewport,
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
import {getBasemapGround, resolveBasemap} from '../cartography/grounds';
import type {
  BasemapSpec,
  BasemapStyleName,
  MapAnnotation,
  MapHighlight,
  ResolvedBasemap
} from '../cartography/types';
import {getMetersPerPixel} from '../cartography/zoom';
import type {TooltipContent} from '../scenes/scene';
import {SpatialAnalysisDeckEffect, type SpatialAnalysisEffectStats} from './analysis-effect';
import {AnnotationOverlay} from './annotation-overlay';
import {
  BASEMAP_STYLE_URLS,
  EMPTY_STYLE,
  createFlatStyle,
  createGroundTransform,
  createLabelTransform,
  getGroundStyleKey,
  getLabelStyleKey,
  needsFlatStyle
} from './basemap-style';
import {prefersReducedMotion} from './motion';
import type {EncodableInstance, SceneFrame, ScenePointerEvent, ViewState} from './types';

export {getBasemapGround, resolveBasemap, prefersReducedMotion};

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

/** Longest wait for the new ground's tiles before a cross-fade comes back in, in milliseconds. */
const FADE_IN_TIMEOUT_MS = 1800;

/** Default length of an animated `fitBounds`, in milliseconds. */
const DEFAULT_FIT_TRANSITION_MS = 900;

/** Zoom limits of the host camera. */
const MIN_ZOOM = 2;
const MAX_ZOOM = 19;

/** CSS pixels the shell keeps free of map content (the phone sheet, the step panel, furniture). */
export type ViewPadding = {top: number; right: number; bottom: number; left: number};

/** Pointer position of a tooltip or map click, in CSS pixels relative to the map stage. */
export type MapPointerPosition = {x: number; y: number; pointerType: string};

/** A rectangle in CSS pixels relative to the map stage (annotation obstacles). */
export type StageRect = {x: number; y: number; width: number; height: number};

const NO_PADDING: ViewPadding = {top: 0, right: 0, bottom: 0, left: 0};

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
  /** Hover content: structured (preferred), plain text, or `null` for no tooltip. */
  getTooltip?: (event: ScenePointerEvent) => TooltipContent | string | null;
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
  /** Frames `[west, south, east, north]` in the free map area; see {@link DeckHost.fitBounds}. */
  fitBounds: DeckHost['fitBounds'];
  /** Ground metres per CSS pixel at the current camera centre. */
  getMetersPerPixel: () => number;
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
  /**
   * Called on every hover of the map with the scene's tooltip content (structured, plain text, or
   * `null` to hide), and with `null` when the pointer leaves the map or starts a drag. Positions
   * are CSS pixels relative to the map stage; `pointerType` is `mouse`, `pen` or `touch`. A tap on
   * the map reports its tooltip the same way (before `onMapClick`) so touch can pin it.
   */
  onTooltip:
    | ((content: TooltipContent | string | null, position: MapPointerPosition) => void)
    | null = null;
  /** Called for a map click or tap that the active scene did not handle (pin or unpin a tooltip). */
  onMapClick: ((position: MapPointerPosition) => void) | null = null;

  private readonly basemapElement: HTMLDivElement;
  private readonly labelsElement: HTMLDivElement;
  private readonly annotations = new AnnotationOverlay();
  private readonly annotationGroups = new Map<string, readonly MapAnnotation[]>();
  private readonly highlightGroups = new Map<string, readonly MapHighlight[]>();
  private readonly viewListeners = new Set<(view: ViewState) => void>();
  private lastViewKey = '';
  private readonly map: maplibregl.Map;
  private labelsMap: maplibregl.Map | null = null;
  private basemapSpec: BasemapSpec | null = null;
  /** Style key last applied to the base map and the labels map, to skip redundant reloads. */
  private appliedBaseStyle = '';
  private appliedLabelsStyle = '';
  /** Incremented by every basemap change; a pending cross-fade step stops when it is stale. */
  private fadeToken = 0;
  private fadeTimer: number | undefined;
  private obstaclesKey = '[]';
  private viewPadding: ViewPadding = NO_PADDING;
  private lastPointer: MapPointerPosition = {x: 0, y: 0, pointerType: 'mouse'};
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
  private theme: HostTheme;
  private finalized = false;
  private destroyed = false;
  private started = false;
  private lastHover: {info: PickingInfo; event: unknown} | null = null;
  private readonly hideTooltip = () => this.onTooltip?.(null, this.lastPointer);
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
    this.labelsElement = document.createElement('div');
    this.labelsElement.className = 'map-labels';
    this.labelsElement.hidden = true;
    this.annotations.element.classList.add('map-annotations');
    this.element.append(this.basemapElement, this.labelsElement, this.annotations.element);
    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'keydown', 'keyup'] as const) {
      // Capture so modifier state is current before Deck dispatches its own handlers.
      (type.startsWith('key') ? window : this.element).addEventListener(type, this.trackModifiers, {
        capture: true,
        passive: true
      });
    }

    this.element.addEventListener('pointerleave', this.hideTooltip);

    this.deviceReady = new Promise<Device>((resolve, reject) => {
      this.resolveDevice = resolve;
      this.rejectDevice = reject;
    });
    // A rejected device promise is reported through `state.error`; avoid an unhandled rejection.
    this.deviceReady.catch(() => {});

    this.map = new maplibregl.Map({
      container: this.basemapElement,
      style: EMPTY_STYLE,
      center: [this.viewState.longitude, this.viewState.latitude],
      zoom: this.viewState.zoom,
      interactive: false,
      attributionControl: false
    });
    this.map.addControl(new maplibregl.AttributionControl({compact: true}), 'bottom-right');
    this.resizeObserver = new ResizeObserver(() => {
      this.map.resize();
      this.labelsMap?.resize();
    });
    this.resizeObserver.observe(this.element);
    this.map.on('error', () =>
      this.onStatus?.('Basemap tiles unavailable; GPU outputs still render.')
    );
    this.applyBasemap();
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
      onHover: (info, event) => {
        this.lastHover = {info, event};
        this.reportTooltip(info, event);
      },
      onClick: (info, event) => {
        const handled = this.instance?.onClick?.(this.toEvent(info)) ?? false;
        if (handled) return;
        const position = this.toPointerPosition(info, event);
        // A tap has no hover before it: show the tooltip of the tapped feature first.
        if (position.pointerType !== 'mouse') this.reportTooltip(info, event);
        this.onMapClick?.(position);
      },
      onDragStart: (info, event) => {
        this.hideTooltip();
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
        this.handleViewChange();
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
        this.handleViewChange();
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
   * Re-runs the active scene's `getTooltip` for the last hovered position and reports it through
   * `onTooltip`. Use it after an asynchronous probe resolves while the pointer rests on the map.
   */
  refreshTooltip(): void {
    const hover = this.lastHover;
    if (!hover) return;
    this.reportTooltip(hover.info, hover.event);
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
    this.annotationGroups.clear();
    this.annotations.setHaloWeight('normal');
    this.highlightGroups.clear();
    this.annotations.setTime(null);
    this.refreshAnnotations();
    this.refreshHighlights();
    this.hideTooltip();
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
        fitBounds: (bounds, options) => {
          if (currentGeneration === this.generation) this.fitBounds(bounds, options);
        },
        getMetersPerPixel: () => this.getMetersPerPixel(),
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
    this.annotationGroups.clear();
    this.annotations.setHaloWeight('normal');
    this.highlightGroups.clear();
    this.annotations.setTime(null);
    this.refreshAnnotations();
    this.refreshHighlights();
    this.hideTooltip();
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

  /**
   * Moves the camera, animating when `transitionMs` is positive (a jump cut under
   * `prefers-reduced-motion`). Omitted fields keep their value.
   *
   * `padding` is the CSS pixels to keep free on each side (default: the standing padding of
   * {@link setViewPadding}; `null` or all zeros for none): the requested centre is placed at the
   * centre of the free rectangle, so the camera centre is offset by half the padding difference.
   * Without any padding the requested centre is the camera centre, as before.
   */
  setViewState(
    view: Partial<ViewState> & {transitionMs?: number; padding?: Partial<ViewPadding> | null}
  ): void {
    const {transitionMs: requestedTransition, padding, ...fields} = view;
    const defined = Object.fromEntries(
      Object.entries(fields).filter(([, value]) => value !== undefined)
    );
    let next = toHostViewState({...this.viewState, ...defined});
    const effectivePadding =
      padding === null ? NO_PADDING : completePadding(padding ?? this.viewPadding);
    // Only a requested centre is padded; zoom-only moves keep the (already padded) centre.
    const moved = 'longitude' in defined && 'latitude' in defined;
    if (moved) next = this.offsetViewForPadding(next, effectivePadding);
    const transitionMs = prefersReducedMotion() ? 0 : (requestedTransition ?? 0);
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
    this.handleViewChange();
  }

  /**
   * The camera {@link setViewState} would apply for `view` and `padding`, without moving. Use it
   * to compare the live camera with a step's camera ("Modified" chip) when padding is in force.
   */
  getPaddedViewState(view: Partial<ViewState>, padding?: Partial<ViewPadding> | null): ViewState {
    const defined = Object.fromEntries(
      Object.entries(view).filter(([, value]) => value !== undefined)
    );
    const base = toHostViewState({...this.viewState, ...defined});
    const resolved = padding === null ? NO_PADDING : completePadding(padding ?? this.viewPadding);
    const {longitude, latitude, zoom, pitch, bearing} = this.offsetViewForPadding(base, resolved);
    return {longitude, latitude, zoom, pitch, bearing};
  }

  /**
   * Sets the standing padding: the CSS pixels the shell keeps free on each side (a phone sheet at
   * the bottom, the step panel on the left, furniture). It applies to every later `setViewState`
   * and `fitBounds`. With `recenter` the current view slides by the change in padding so the
   * content stays centred in the free area (animated over `transitionMs`, a jump cut under
   * reduced motion); without it the camera stays where it is.
   */
  setViewPadding(
    padding: Partial<ViewPadding>,
    options: {recenter?: boolean; transitionMs?: number} = {}
  ): void {
    const previous = this.viewPadding;
    this.viewPadding = completePadding(padding);
    if (!options.recenter || !this.deck) return;
    const shift = {
      x: (this.viewPadding.left - this.viewPadding.right - (previous.left - previous.right)) / 2,
      y: (this.viewPadding.top - this.viewPadding.bottom - (previous.top - previous.bottom)) / 2
    };
    if (shift.x === 0 && shift.y === 0) return;
    const shifted = this.shiftViewCenter(this.viewState, shift.x, shift.y);
    this.setViewState({
      longitude: shifted.longitude,
      latitude: shifted.latitude,
      transitionMs: options.transitionMs ?? 0,
      padding: null
    });
  }

  /** The standing padding set by {@link setViewPadding}. */
  getViewPadding(): ViewPadding {
    return {...this.viewPadding};
  }

  /**
   * Frames `[west, south, east, north]` in the free map rectangle: the viewport minus the standing
   * padding minus `padding` (a number for every side, or per side). Zoom is capped at `maxZoom`
   * (default 19). Animated over `transitionMs` (default 900; a jump cut under reduced motion).
   * Keeps pitch and bearing.
   */
  fitBounds(
    bounds: readonly [number, number, number, number],
    options: {padding?: number | Partial<ViewPadding>; maxZoom?: number; transitionMs?: number} = {}
  ): void {
    const {width, height} = this.getStageSize();
    if (width <= 0 || height <= 0) return;
    const extra = completePadding(
      typeof options.padding === 'number'
        ? {
            top: options.padding,
            right: options.padding,
            bottom: options.padding,
            left: options.padding
          }
        : options.padding
    );
    const total = {
      top: this.viewPadding.top + extra.top,
      right: this.viewPadding.right + extra.right,
      bottom: this.viewPadding.bottom + extra.bottom,
      left: this.viewPadding.left + extra.left
    };
    const [west, south, east, north] = bounds;
    const viewport = new WebMercatorViewport({width, height, longitude: 0, latitude: 0, zoom: 0});
    const maxZoom = Math.min(options.maxZoom ?? MAX_ZOOM, MAX_ZOOM);
    let fitted: {longitude: number; latitude: number; zoom: number};
    try {
      fitted = viewport.fitBounds(
        [
          [west, south],
          [east, north]
        ],
        {padding: clampPadding(total, width, height), maxZoom}
      );
    } catch {
      // Degenerate bounds or a rectangle with no free area: centre on the bounds at the current zoom.
      fitted = {
        longitude: (west + east) / 2,
        latitude: (south + north) / 2,
        zoom: Math.min(this.viewState.zoom, maxZoom)
      };
    }
    this.setViewState({
      longitude: fitted.longitude,
      latitude: fitted.latitude,
      zoom: Math.min(Math.max(fitted.zoom, MIN_ZOOM), maxZoom),
      transitionMs: options.transitionMs ?? DEFAULT_FIT_TRANSITION_MS,
      padding: null
    });
  }

  /** Ground metres per CSS pixel at the current camera centre (Web Mercator). */
  getMetersPerPixel(): number {
    return getMetersPerPixel(this.viewState.zoom, this.viewState.latitude);
  }

  private getStageSize(): {width: number; height: number} {
    const viewport = this.deck?.getViewports()[0];
    return {
      width: this.element.clientWidth || viewport?.width || 0,
      height: this.element.clientHeight || viewport?.height || 0
    };
  }

  /**
   * Moves a camera so the ground point that is at its centre appears `(dx, dy)` CSS pixels
   * further right and down: the new centre is the point `(dx, dy)` up and left of the old one.
   */
  private shiftViewCenter(
    view: HostViewState,
    dx: number,
    dy: number
  ): {longitude: number; latitude: number} {
    const {width, height} = this.getStageSize();
    if (width <= 0 || height <= 0 || (dx === 0 && dy === 0)) {
      return {longitude: view.longitude, latitude: view.latitude};
    }
    const viewport = new WebMercatorViewport({
      width,
      height,
      longitude: view.longitude,
      latitude: view.latitude,
      zoom: view.zoom,
      pitch: view.pitch,
      bearing: view.bearing
    });
    const [longitude, latitude] = viewport.unproject([width / 2 - dx, height / 2 - dy]);
    return {longitude, latitude};
  }

  /** Places the camera's requested centre at the centre of the free rectangle. */
  private offsetViewForPadding(view: HostViewState, padding: ViewPadding): HostViewState {
    const dx = (padding.left - padding.right) / 2;
    const dy = (padding.top - padding.bottom) / 2;
    if (dx === 0 && dy === 0) return view;
    return {...view, ...this.shiftViewCenter(view, dx, dy)};
  }

  /** Shows a text marker anchored to a coordinate, or hides it with `null`. */
  setCallout(callout: HostCallout | null): void {
    this.setAnnotationGroup(
      '\u0000callout',
      callout ? [{kind: 'callout', coordinate: callout.coordinate, text: callout.text}] : null
    );
  }

  /**
   * Sets one named group of annotations (`null` removes it). Groups are merged in insertion
   * order and share one collision pass. Cleared when a scene is activated or deactivated.
   */
  setAnnotationGroup(key: string, annotations: readonly MapAnnotation[] | null): void {
    if (annotations?.length) this.annotationGroups.set(key, annotations);
    else this.annotationGroups.delete(key);
    this.refreshAnnotations();
  }

  /**
   * Sets one named group of highlights (outlines above the data: the hovered feature, a chart or
   * legend selection), or removes it with `null`. Groups are merged in insertion order. Cleared
   * when a scene is activated or deactivated.
   */
  setHighlightGroup(key: string, highlights: readonly MapHighlight[] | null): void {
    const previous = this.highlightGroups.get(key);
    if (!highlights?.length) {
      if (!previous) return;
      this.highlightGroups.delete(key);
    } else {
      // Hover reports the same outline on every pointer move: skip identical content.
      if (previous && JSON.stringify(previous) === JSON.stringify(highlights)) return;
      this.highlightGroups.set(key, highlights);
    }
    this.refreshHighlights();
  }

  /** Halo weight of the annotation labels (`'heavy'` over dense or saturated data). */
  setAnnotationHalo(weight: 'normal' | 'heavy'): void {
    this.annotations.setHaloWeight(weight);
  }

  /** Sets the time that annotations with a `timeRange` show at; `null` shows all of them. */
  setAnnotationTime(time: number | null): void {
    this.annotations.setTime(time);
  }

  /**
   * Sets the screen rectangles (CSS pixels relative to the map stage: legend, cartouche, scale
   * bar) that annotation labels must avoid.
   */
  setAnnotationObstacles(rects: readonly StageRect[]): void {
    const key = JSON.stringify(rects);
    if (key === this.obstaclesKey) return;
    this.obstaclesKey = key;
    this.annotations.setObstacles(rects);
  }

  /** Calls `listener` with the camera whenever it changes (including flights and resizes). */
  addViewListener(listener: (view: ViewState) => void): () => void {
    this.viewListeners.add(listener);
    listener(this.getViewState());
    return () => this.viewListeners.delete(listener);
  }

  /**
   * Applies a basemap treatment (`null` = defaults: theme-paired style, labels in the basemap, no
   * dimming). Style changes reload tiles only when the style or label placement really changed.
   */
  setBasemap(spec: BasemapSpec | null): void {
    this.basemapSpec = spec;
    this.applyBasemap();
  }

  /** The basemap treatment in effect. */
  get basemap(): ResolvedBasemap {
    return resolveBasemap(this.basemapSpec, this.theme);
  }

  /**
   * Luminance of the ground under the data: `dark` for dark-matter, `light` for positron and
   * voyager; for `none` the luminance of the palette's background (below 0.2 is `dark`), or the
   * page theme when there is no palette. Map labels and data colours follow it.
   */
  get ground(): HostTheme {
    return getBasemapGround(this.basemap, this.theme);
  }

  /** Switches the page theme of the basemap (and of the labels map). */
  setTheme(theme: HostTheme): void {
    if (theme === this.theme) return;
    this.theme = theme;
    this.applyBasemap();
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
    this.element.removeEventListener('pointerleave', this.hideTooltip);
    window.clearTimeout(this.fadeTimer);
    window.removeEventListener('keydown', this.trackModifiers, {capture: true});
    window.removeEventListener('keyup', this.trackModifiers, {capture: true});
    this.deactivate();
    this.resizeObserver.disconnect();
    this.deck?.finalize();
    this.map.remove();
    this.labelsMap?.remove();
    this.annotations.destroy();
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

  private toPointerPosition(info: PickingInfo, event: unknown): MapPointerPosition {
    const source = (event as {srcEvent?: {pointerType?: string}} | null)?.srcEvent;
    const position = {x: info.x, y: info.y, pointerType: source?.pointerType || 'mouse'};
    this.lastPointer = position;
    return position;
  }

  /** Asks the active scene for the tooltip at a hover or tap and reports it through `onTooltip`. */
  private reportTooltip(info: PickingInfo, event: unknown): void {
    const position = this.toPointerPosition(info, event);
    const callback = this.onTooltip;
    if (!callback) return;
    callback(this.instance?.getTooltip?.(this.toEvent(info)) ?? null, position);
  }

  private reportError(error: Error): void {
    this.state.error = error.message;
    this.onStateChange?.();
  }

  private synchronizeBasemap(): void {
    const camera = {
      center: [this.viewState.longitude, this.viewState.latitude] as [number, number],
      zoom: this.viewState.zoom,
      pitch: this.viewState.pitch,
      bearing: this.viewState.bearing
    };
    this.map.jumpTo(camera);
    if (this.labelsMap && !this.labelsElement.hidden) this.labelsMap.jumpTo(camera);
  }

  /**
   * Brings the base map and the labels map to the current basemap spec. CSS-only changes (dim,
   * desaturate, visibility) apply at once and ease over `transitionMs`; a style change cross-fades
   * (the maps fade out over half the time, load the new style, and fade back in once its tiles
   * are drawn). The first style, reduced motion and `transitionMs` 0 apply without a fade.
   */
  private applyBasemap(): void {
    const resolved = resolveBasemap(this.basemapSpec, this.theme);
    const tone = getBasemapGround(resolved, this.theme);
    const showLabels = resolved.labels === 'above' && resolved.labelPreset !== 'none';
    const baseKey = getGroundStyleKey(resolved, tone, resolved.labels === 'basemap');
    const labelStyleName = this.getLabelStyleName(resolved);
    const labelsKey = showLabels ? getLabelStyleKey(resolved, tone, labelStyleName) : '';
    const fadeMs =
      prefersReducedMotion() || !this.appliedBaseStyle ? 0 : Math.round(resolved.transitionMs);
    const baseOpacity = resolved.dim > 0 ? 1 - resolved.dim : 1;
    this.fadeToken++;
    window.clearTimeout(this.fadeTimer);

    // Dim and desaturate are plain CSS and ease on their own.
    const easeMs = prefersReducedMotion() ? 0 : Math.round(resolved.transitionMs);
    this.basemapElement.style.filter =
      resolved.desaturate > 0 ? `saturate(${(1 - resolved.desaturate).toFixed(3)})` : '';
    this.setFadeTransition(easeMs);

    const unchanged = baseKey === this.appliedBaseStyle && labelsKey === this.appliedLabelsStyle;
    if (unchanged) {
      this.showBasemapElements(resolved, baseOpacity);
      return;
    }
    const commit = () =>
      this.commitBasemapStyles(resolved, tone, baseKey, labelsKey, labelStyleName);
    if (fadeMs <= 0) {
      commit();
      this.showBasemapElements(resolved, baseOpacity);
      return;
    }
    const token = this.fadeToken;
    const halfMs = Math.round(fadeMs / 2);
    this.setFadeTransition(halfMs);
    this.basemapElement.style.opacity = '0';
    this.labelsElement.style.opacity = '0';
    this.fadeTimer = window.setTimeout(() => {
      if (token !== this.fadeToken || this.destroyed) return;
      const restyled = commit();
      this.waitForMapsIdle(restyled, () => {
        if (token !== this.fadeToken || this.destroyed) return;
        this.showBasemapElements(resolved, baseOpacity);
      });
    }, halfMs);
  }

  /** Sets the opacity transition of the two map elements. */
  private setFadeTransition(durationMs: number): void {
    const transition =
      durationMs > 0 ? `opacity ${durationMs}ms var(--ease, ease-out), filter 0.3s` : 'none';
    this.basemapElement.style.transition = transition;
    this.labelsElement.style.transition =
      durationMs > 0 ? `opacity ${durationMs}ms var(--ease, ease-out)` : 'none';
  }

  /** Shows the maps the resolved basemap uses, at their resting opacity. */
  private showBasemapElements(resolved: ResolvedBasemap, baseOpacity: number): void {
    const drawsBase = resolved.style !== 'none' || needsFlatStyle(resolved);
    this.basemapElement.hidden = !drawsBase;
    this.basemapElement.style.opacity = baseOpacity < 1 ? String(baseOpacity) : '';
    const showLabels = resolved.labels === 'above' && resolved.labelPreset !== 'none';
    this.labelsElement.hidden = !showLabels;
    this.labelsElement.style.opacity = '';
  }

  /** Calls `done` when the restyled `maps` have drawn their new styles, or after a timeout. */
  private waitForMapsIdle(maps: readonly maplibregl.Map[], done: () => void): void {
    let pending = 0;
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      window.clearTimeout(timer);
      done();
    };
    const timer = window.setTimeout(finish, FADE_IN_TIMEOUT_MS);
    for (const map of maps) {
      pending++;
      map.once('idle', () => {
        if (--pending === 0) finish();
      });
    }
    if (pending === 0) finish();
  }

  /** The CARTO style that supplies labels: the base style, or the theme's own over a flat ground. */
  private getLabelStyleName(resolved: ResolvedBasemap): Exclude<BasemapStyleName, 'none'> {
    if (resolved.style !== 'none') return resolved.style;
    return getBasemapGround(resolved, this.theme) === 'dark' ? 'dark-matter' : 'positron';
  }

  /** Loads the base and labels styles of a basemap (the part of a change that is not CSS). */
  private commitBasemapStyles(
    resolved: ResolvedBasemap,
    tone: HostTheme,
    baseKey: string,
    labelsKey: string,
    labelStyleName: Exclude<BasemapStyleName, 'none'>
  ): maplibregl.Map[] {
    const restyled: maplibregl.Map[] = [];
    // Base map: the ground, plus labels only when they stay in the basemap.
    this.basemapElement.hidden = resolved.style === 'none' && !needsFlatStyle(resolved);
    if (baseKey !== this.appliedBaseStyle) {
      this.appliedBaseStyle = baseKey;
      if (!this.basemapElement.hidden) restyled.push(this.map);
      if (resolved.style === 'none') {
        this.map.setStyle(
          needsFlatStyle(resolved) ? createFlatStyle(resolved, tone) : EMPTY_STYLE,
          {
            diff: false
          }
        );
      } else {
        this.map.setStyle(BASEMAP_STYLE_URLS[resolved.style], {
          diff: false,
          transformStyle: createGroundTransform(resolved, tone, resolved.labels === 'basemap')
        });
      }
    }
    // Labels map: a second, label-only MapLibre map stacked over the Deck canvas.
    const showLabels = labelsKey !== '';
    this.labelsElement.hidden = !showLabels;
    if (!showLabels) {
      this.appliedLabelsStyle = '';
      return restyled;
    }
    if (!this.labelsMap) {
      this.labelsMap = new maplibregl.Map({
        container: this.labelsElement,
        style: EMPTY_STYLE,
        center: [this.viewState.longitude, this.viewState.latitude],
        zoom: this.viewState.zoom,
        pitch: this.viewState.pitch,
        bearing: this.viewState.bearing,
        interactive: false,
        attributionControl: false
      });
    }
    if (labelsKey !== this.appliedLabelsStyle) {
      this.appliedLabelsStyle = labelsKey;
      restyled.push(this.labelsMap);
      this.labelsMap.setStyle(BASEMAP_STYLE_URLS[labelStyleName], {
        diff: false,
        transformStyle: createLabelTransform(resolved, tone)
      });
    }
    this.labelsMap.resize();
    this.synchronizeBasemap();
    return restyled;
  }

  private refreshHighlights(): void {
    const merged: MapHighlight[] = [];
    for (const group of this.highlightGroups.values()) merged.push(...group);
    // The overlay redraws the highlights itself; no view update or listener notification needed.
    this.annotations.setHighlights(merged);
  }

  private refreshAnnotations(): void {
    const merged: MapAnnotation[] = [];
    for (const group of this.annotationGroups.values()) merged.push(...group);
    this.annotations.setAnnotations(merged);
    this.lastViewKey = '';
    this.handleViewChange();
  }

  /**
   * Re-projects annotations and notifies view listeners when Deck's viewport changed. Keyed on
   * the viewport Deck last drew (not the requested view state), so overlays never run a frame
   * ahead of the data during flights.
   */
  private handleViewChange(): void {
    const viewport = this.deck?.getViewports()[0] as
      | (Viewport & {longitude?: number; latitude?: number; pitch?: number; bearing?: number})
      | undefined;
    if (!viewport) return;
    const view: ViewState = {
      longitude: viewport.longitude ?? this.viewState.longitude,
      latitude: viewport.latitude ?? this.viewState.latitude,
      zoom: viewport.zoom,
      pitch: viewport.pitch ?? 0,
      bearing: viewport.bearing ?? 0
    };
    const {width, height} = viewport;
    const key = `${view.longitude},${view.latitude},${view.zoom},${view.pitch},${view.bearing},${width},${height}`;
    if (key === this.lastViewKey) return;
    this.lastViewKey = key;
    this.annotations.update({
      width,
      height,
      zoom: viewport.zoom,
      project: coordinate => {
        const [x, y] = viewport.project([coordinate[0], coordinate[1]]);
        return [x, y];
      }
    });
    for (const listener of this.viewListeners) listener(view);
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

/** Fills the missing sides of a padding with 0 and clamps negatives. */
function completePadding(padding: Partial<ViewPadding> | undefined): ViewPadding {
  return {
    top: Math.max(padding?.top ?? 0, 0),
    right: Math.max(padding?.right ?? 0, 0),
    bottom: Math.max(padding?.bottom ?? 0, 0),
    left: Math.max(padding?.left ?? 0, 0)
  };
}

/** Shrinks a padding that would leave under 20 % of the viewport free, keeping its proportions. */
function clampPadding(padding: ViewPadding, width: number, height: number): ViewPadding {
  const keep = 0.2;
  const horizontal = padding.left + padding.right;
  const vertical = padding.top + padding.bottom;
  const horizontalScale = horizontal > width * (1 - keep) ? (width * (1 - keep)) / horizontal : 1;
  const verticalScale = vertical > height * (1 - keep) ? (height * (1 - keep)) / vertical : 1;
  return {
    top: padding.top * verticalScale,
    bottom: padding.bottom * verticalScale,
    left: padding.left * horizontalScale,
    right: padding.right * horizontalScale
  };
}

function toHostViewState(view: ViewState): HostViewState {
  return {
    longitude: view.longitude,
    latitude: view.latitude,
    zoom: view.zoom,
    pitch: view.pitch ?? 0,
    bearing: view.bearing ?? 0,
    minZoom: MIN_ZOOM,
    maxZoom: MAX_ZOOM
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
