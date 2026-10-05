// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, PickingInfo, Viewport} from '@deck.gl/core';
import type {CommandEncoder, Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {SpatialAnalysisControlSection} from './app-ui';
import type {SpatialAnalysisDataCatalog} from './spatial-analysis-data';

/** Camera a mode starts from. Longitude/latitude in degrees, standard Web Mercator zoom. */
export type SpatialAnalysisViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch?: number;
  bearing?: number;
};

/** Per-frame values passed to {@link SpatialAnalysisModeInstance.encode}. */
export type SpatialAnalysisFrame = {
  /** Deck viewport used for this frame. */
  viewport: Viewport;
  /** Monotonic wall-clock time in seconds (`performance.now() / 1000`). */
  timeSeconds: number;
  /** Seconds since the previous encoded frame, clamped to `[0, 0.25]`. */
  deltaSeconds: number;
  /** Number of frames encoded since this mode instance was created. */
  frameIndex: number;
};

/** Pointer event forwarded to a mode. `coordinate` is `[longitude, latitude]` when known. */
export type SpatialAnalysisPointerEvent = {
  info: PickingInfo;
  /** `[longitude, latitude]` under the pointer, or `null` off the map. */
  coordinate: readonly [number, number] | null;
  /** Pointer position in CSS pixels relative to the deck canvas. */
  pixel: readonly [number, number];
};

/** Services the shell provides to a mode while it is active. */
export type SpatialAnalysisModeContext = {
  /** The WebGPU device owned by Deck. Modes create their buffers and graphs on it. */
  device: Device;
  /** Shared, lazily loaded datasets with deterministic synthetic fallbacks. */
  data: SpatialAnalysisDataCatalog;
  /** This mode's control section. It is cleared automatically when the mode is destroyed. */
  controls: SpatialAnalysisControlSection;
  /** Aborted when the user switches away before `create` resolves or after destroy. */
  signal: AbortSignal;
  /** Returns the current Deck viewport, or `null` before the first frame. */
  getViewport: () => Viewport | null;
  /** Requests that the shell rebuild layers by calling {@link SpatialAnalysisModeInstance.getLayers}. */
  updateLayers: () => void;
  /** Sets a one-line status message under the mode title. */
  setStatus: (message: string) => void;
  /**
   * Enables or disables Deck's map drag-pan controller, for example while drawing a lasso or
   * dragging an observer. Always re-enable it when the gesture ends.
   */
  setMapDragEnabled: (enabled: boolean) => void;
};

/**
 * A running mode: owns compiled graphs, parameter buffers, caller-owned output buffers, and the
 * Deck layers that render those buffers directly.
 */
export type SpatialAnalysisModeInstance = {
  /**
   * Compiled graphs this mode encodes. The shell reports node counts and checks that the set never
   * changes while parameters move (the "no recompilation" proof shown in the panel). A mode that
   * deliberately rebuilds a graph for a compile-time change (for example grid vs. hexagon) must
   * return the new objects; the shell counts those rebuilds separately.
   */
  getCompiledGraphs: () => readonly CompiledGPUCommandGraph<never>[];
  /**
   * Writes per-frame parameter buffers and encodes compiled graphs into Deck's frame encoder.
   * Deck owns queue submission. Never compile, submit, or synchronously read back here.
   */
  encode: (commandEncoder: CommandEncoder, frame: SpatialAnalysisFrame) => void;
  /** Deck layers rendering this mode's GPU outputs. Called whenever the shell rebuilds layers. */
  getLayers: () => Layer[];
  /** Map click. Return `true` when handled. */
  onClick?: (event: SpatialAnalysisPointerEvent) => boolean;
  /** Pointer drag start on the map. Return `true` to capture the gesture (pan is suppressed). */
  onDragStart?: (event: SpatialAnalysisPointerEvent) => boolean;
  /** Pointer drag while captured. */
  onDrag?: (event: SpatialAnalysisPointerEvent) => void;
  /** Pointer drag end while captured. */
  onDragEnd?: (event: SpatialAnalysisPointerEvent) => void;
  /** Hover tooltip text, or `null`. */
  getTooltip?: (event: SpatialAnalysisPointerEvent) => string | null;
  /** Releases every GPU resource the mode created (graphs first, then buffers). */
  destroy: () => void;
};

/**
 * One demo mode. Each mode lives in its own file under `modes/` and is listed once in
 * `modes/index.ts`; the shell needs no other change to add a mode.
 */
export type SpatialAnalysisModeDefinition = {
  /** Stable kebab-case ID, also accepted by the `?mode=` URL parameter. */
  id: string;
  /** Short tab label. */
  title: string;
  /** Contributor class names exercised, shown in the panel. */
  contributors: readonly string[];
  /** One or two sentences describing what the mode shows and how to interact. */
  description: string;
  /** Camera applied when the mode is activated. */
  initialViewState: SpatialAnalysisViewState;
  /** Loads data, builds and compiles graphs, and returns the running instance. */
  create: (context: SpatialAnalysisModeContext) => Promise<SpatialAnalysisModeInstance>;
};
