// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, Viewport} from '@deck.gl/core';
import type {Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {LoadedDataset} from '../data/catalog';
import type {RampName} from '../engine/ramps';
import type {SceneFrame, ScenePointerEvent, ViewState} from '../engine/types';
import type {CommandEncoder} from '@luma.gl/core';
import type {ChartData} from './chart-types';

export type {SceneFrame, ScenePointerEvent, ViewState};
export type {
  BarChartData,
  ChartData,
  ChartSeries,
  LineChartData,
  SparklineData
} from './chart-types';

/** Value of one option in a scene's state. */
export type OptionValue = number | string | boolean | readonly [number, number];

/** Scene option state: option id to value. Buttons carry no state. */
export type OptionState = Record<string, OptionValue>;

/**
 * How a change is applied.
 *
 * - `param`: a parameter-buffer write; no graph is recompiled.
 * - `compile`: a compile-time contributor option; the scene rebuilds (or switches between)
 *   compiled graphs. The panel marks these so readers see which options cost a rebuild.
 */
export type OptionApply = 'param' | 'compile';

type OptionBase<O, K extends keyof O & string = keyof O & string> = {
  /** Key in the scene's option state `O`. Also used in the URL. */
  id: K;
  /** Control label. */
  label: string;
  /** Help text: what the option does and why you would change it. Shown under the control. */
  help?: string;
  /** Whether the option is a buffer write or a graph rebuild. */
  apply: OptionApply;
  /** Disables the control when this returns true for the current state. */
  disabledWhen?: (state: O) => boolean;
  /** Controls are grouped under this heading in the panel. */
  group?: string;
};

/** Number slider. */
export type SliderOption<O> = OptionBase<O> & {
  kind: 'slider';
  min: number;
  max: number;
  step: number;
  default: number;
  /** Suffix shown after the value, for example `'m'`. */
  unit?: string;
  /** Overrides the value label. */
  format?: (value: number) => string;
};

/** Dropdown. */
export type SelectOption<O> = OptionBase<O> & {
  kind: 'select';
  options: readonly {value: string; label: string; help?: string}[];
  default: string;
};

/** Checkbox switch. */
export type ToggleOption<O> = OptionBase<O> & {
  kind: 'toggle';
  default: boolean;
};

/** Two-handle range, stored as `[low, high]`. */
export type RangeOption<O> = OptionBase<O> & {
  kind: 'range';
  min: number;
  max: number;
  step: number;
  default: readonly [number, number];
  unit?: string;
  format?: (value: number) => string;
};

/** Action button. Calls `instance.onAction(id, state)`; it has no state value. */
export type ButtonOption = {
  kind: 'button';
  id: string;
  label: string;
  help?: string;
  apply?: OptionApply;
  group?: string;
  disabledWhen?: (state: never) => boolean;
};

/** One declarative option; the shell generates the control, the URL state and the reset button. */
export type OptionSpec<O> =
  | SliderOption<O>
  | SelectOption<O>
  | ToggleOption<O>
  | RangeOption<O>
  | ButtonOption;

/** Legend entry color, RGBA 0-255. */
export type LegendColor = readonly [number, number, number, number?];

/** What a legend draws. Legends are recomputed from the option state. */
export type LegendSpec =
  | {
      kind: 'ramp';
      /** Needed when `extent` is `'gpu'`: the scene calls `ctx.setLegendExtent(id, [min, max])`. */
      id?: string;
      title: string;
      /** Ramp name from `RAMP_STOPS`; must match the layer's `colormap`. */
      ramp: RampName;
      /** Custom gradient stops, low value first. Overrides `ramp` for the swatch (for example wind speed). */
      colors?: readonly LegendColor[];
      /**
       * Value scale of the end labels and ticks. `'log'` puts decade ticks (`1`, `10`, `10^2`...)
       * between a positive `extent`; the gradient is the layer's own normalized ramp.
       */
      scale?: 'linear' | 'log';
      /** Value unit, for example `'trips / cell'`. */
      unit?: string;
      /** Fixed `[min, max]`, or `'gpu'` to show the extent the scene reports. */
      extent: 'gpu' | readonly [number, number];
      /** Match the layer's `sqrtScale`. */
      sqrtScale?: boolean;
      /** Replace the min and max labels, for example `['fewer', 'more']`. */
      labels?: readonly [string, string];
      /** Number formatter for the end labels. */
      format?: (value: number) => string;
    }
  | {
      kind: 'categories';
      title: string;
      entries: readonly {color: LegendColor; label: string}[];
      /** Short note under the entries. */
      note?: string;
    }
  | {
      kind: 'size';
      title: string;
      entries: readonly {radiusPixels: number; label: string}[];
      color?: LegendColor;
    };

/**
 * A typed readout. `kind: 'value'` (default) is filled with `ctx.setReadout`; `kind: 'chart'` is a
 * small inline chart filled with `ctx.setChart`.
 */
export type ReadoutSpec = {
  id: string;
  label: string;
  /** `'chart'` renders a chart widget under the label instead of a value. Default `'value'`. */
  kind?: 'value' | 'chart';
  /**
   * `'inline'` shows label and value on one row; `'block'` shows the value under the label in a
   * monospace block that keeps line breaks. Defaults to `'block'` when the value contains `\n`.
   */
  layout?: 'inline' | 'block';
  /** How numbers are formatted. Strings are shown as given. Defaults to `'text'`. */
  format?: 'text' | 'integer' | 'decimal' | 'percent' | 'milliseconds' | 'bytes' | 'meters';
  /** Tooltip explaining what the number means. */
  help?: string;
};

/** Camera move of a story step. */
export type StepCamera = Partial<ViewState> & {
  /** Fly time in milliseconds. Defaults to 1600. */
  transitionMs?: number;
};

/** One step of a scene's guided narrative. */
export type StoryStep<O = OptionState> = {
  /** Stable id used in the URL (`#/story/<scene>/<stepId>`). Kebab-case. */
  id: string;
  title: string;
  /** Markdown. Explain what is shown, what the tool computes, why it matters, how to read it. */
  body: string;
  /** Camera move applied when the step opens. */
  camera?: StepCamera;
  /** Options set when the step opens (applied on top of defaults and earlier steps). */
  options?: Partial<O>;
  /**
   * Option ids (including button ids) this step asks the reader to use, in display order. The
   * shell renders these controls inside the step card, right under the text, so the step never
   * sends the reader hunting through the full options list. Keep it to the one to four controls
   * the text talks about; an empty array means "just look". Defaults to the ids in `options`.
   */
  controls?: readonly string[];
  /** Readout ids shown inside the step card. Defaults to `highlight.readout`. */
  readouts?: readonly string[];
  /** Marker pinned to a coordinate while the step is open. */
  callout?: {coordinate: readonly [number, number]; text: string};
  /** Readout to emphasise in the panel while the step is open. */
  highlight?: {readout?: string};
};

/** Optional "About this analysis" panel. All fields are markdown. */
export type SceneAbout = {
  /** What the contributor computes. */
  what?: string;
  /** Why an analyst cares. */
  why?: string;
  /** How to read the map. */
  howToRead?: string;
};

/** Datasets a scene uses, resolved before `create`. */
export type DatasetRef = {id: string; role?: string};

/** Loaded datasets by id. */
export type ResolvedDatasets = {
  /** Returns a dataset the scene declared in `datasets`. */
  get: (id: string) => LoadedDataset;
};

/** Theme the page is currently showing. */
export type SceneTheme = 'light' | 'dark';

/** Device limits a scene may branch on, as granted to the device (see `ctx.limits`). */
export type SceneLimits = {
  /** Storage buffers per shader stage (the shell requests up to 16). */
  maxStorageBuffersPerShaderStage: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
};

/** Services the shell gives a scene. Scenes never touch the DOM. */
export type SceneContext<O> = {
  /** The WebGPU device owned by Deck. Scenes create their buffers and graphs on it. */
  device: Device;
  /** Current option state. A live view: read it when you need the latest value. */
  readonly options: Readonly<O>;
  /** Datasets declared in `Scene.datasets`. */
  datasets: ResolvedDatasets;
  /** Aborted when the user switches away before `create` resolves or after destroy. */
  signal: AbortSignal;
  /** True when `?data=synthetic` forces deterministic synthetic data. */
  forceSynthetic: boolean;
  /** Current page theme. */
  theme: () => SceneTheme;
  /** Current Deck viewport, or `null` before the first frame. */
  getViewport: () => Viewport | null;
  /** Rebuilds Deck layers by calling `instance.getLayers()`. */
  requestLayers: () => void;
  /** Sets a declared readout. `null` shows a dash. */
  setReadout: (id: string, value: string | number | null) => void;
  /** Reports the data range of a `'gpu'` ramp legend by its `id`. */
  setLegendExtent: (legendId: string, extent: readonly [number, number]) => void;
  /** Re-evaluates `Scene.legends` now (after state the scene owns changed, for example a loaded dataset). */
  refreshLegends: () => void;
  /**
   * Stores GPU- or data-derived legend input (class breaks, ranges, labels) under `key` and
   * refreshes legends. `Scene.legends(state, data)` receives all stored values as `data`.
   */
  setLegendData: (key: string, value: unknown) => void;
  /**
   * Writes option values from code (animated sliders, click-to-select). The panel and the URL
   * update; `instance.setOption` is NOT called for these writes unless `notify` is true, so there
   * is no compile loop. Unknown ids are ignored. Rapid writes are coalesced per frame.
   */
  setOptions: (values: Partial<O>, options?: {notify?: boolean}) => void;
  /** Moves the camera. Unspecified fields keep their current value. Animates when `transitionMs` > 0. */
  flyTo: (view: Partial<ViewState>, options?: {transitionMs?: number}) => void;
  /** Current camera. */
  getViewState: () => ViewState;
  /** Fills a `kind: 'chart'` readout. `null` clears it. */
  setChart: (readoutId: string, chart: ChartData | null) => void;
  /** Re-runs `getTooltip` for the last hovered position, for example after an async probe resolved. */
  refreshTooltip: () => void;
  /** Limits granted to the WebGPU device; use them to enable options conditionally. */
  limits: SceneLimits;
  /** One-line status shown under the title while loading. */
  setStatus: (message: string) => void;
  /**
   * Enables or disables Deck's map drag-pan, for example while drawing a lasso or dragging an
   * observer. Always re-enable it when the gesture ends.
   */
  setMapDragEnabled: (enabled: boolean) => void;
};

/**
 * A running scene: owns compiled graphs, parameter buffers, caller-owned output buffers and the
 * Deck layers that render those buffers directly.
 */
export type SceneInstance<O = OptionState> = {
  /**
   * Compiled graphs this scene encodes. The shell reports node counts and counts new graph objects
   * as rebuilds (the "compile once, rewrite parameters" proof in the "Under the hood" drawer).
   */
  getCompiledGraphs: () => readonly CompiledGPUCommandGraph<never>[];
  /**
   * Writes per-frame parameter buffers and encodes compiled graphs into Deck's frame encoder.
   * Deck owns queue submission. Never compile, submit, or synchronously read back here.
   */
  encode: (commandEncoder: CommandEncoder, frame: SceneFrame) => void;
  /** Deck layers rendering this scene's GPU outputs. Called whenever the shell rebuilds layers. */
  getLayers: () => Layer[];
  /** Called when an option changes (control, step, URL or reset). `state` is the new full state. */
  setOption?: (id: keyof O & string, value: O[keyof O & string], state: O) => void;
  /** Called when a button option is pressed. */
  onAction?: (id: string, state: O) => void;
  /** Called after the page theme changed. Call `ctx.requestLayers()` if layer colors depend on it. */
  onThemeChange?: (theme: SceneTheme) => void;
  /** Hover tooltip text, or `null`. */
  getTooltip?: (event: ScenePointerEvent) => string | null;
  /** Map click. Return `true` when handled. */
  onClick?: (event: ScenePointerEvent) => boolean;
  /** Pointer drag start on the map. Return `true` to capture the gesture (pan is suppressed). */
  onDragStart?: (event: ScenePointerEvent) => boolean;
  /** Pointer drag while captured. */
  onDrag?: (event: ScenePointerEvent) => void;
  /** Pointer drag end while captured. */
  onDragEnd?: (event: ScenePointerEvent) => void;
  /** Releases every GPU resource the scene created (graphs first, then buffers). */
  destroy: () => void;
};

/** One showcase scene: a dataset, a narrative and the controls to explore it. */
export type Scene<O extends object = OptionState> = {
  /** Kebab-case id; must equal the file name `<id>.scene.ts`. */
  id: string;
  title: string;
  /** Chapter id from `chapters.ts`. */
  chapter: string;
  /** Sort position within the chapter (ascending). */
  order: number;
  /** One or two sentences for the gallery card. */
  summary: string;
  /** Contributor class names, linked to the reference docs (`#/reference/<Name>`). */
  contributors: readonly string[];
  /** Dataset ids the scene uses; loaded before `create` and credited in the panel. */
  datasets: readonly DatasetRef[];
  /** Declarative options for every showcase-worthy option of the contributors involved. */
  options: readonly OptionSpec<O>[];
  /** Guided narrative, four to six steps. */
  story: readonly StoryStep<O>[];
  /** Optional what/why/how-to-read panel. */
  about?: SceneAbout;
  /** Legends for the current state. */
  legends: (state: O, data: Readonly<Record<string, unknown>>) => readonly LegendSpec[];
  /** Typed readouts the scene fills with `ctx.setReadout`. */
  readouts?: readonly ReadoutSpec[];
  /** Code sample, optionally derived from the state. */
  snippet?: string | ((state: O) => string);
  /** Camera before the first step moves it. */
  initialView: ViewState;
  /** Loads nothing itself: datasets are in `ctx.datasets`. Builds graphs and returns the instance. */
  create: (ctx: SceneContext<O>) => Promise<SceneInstance<O>>;
};

/** A scene whose option type is erased, as held by the registry and the shell. */
// biome-ignore lint/suspicious/noExplicitAny: option state is erased in the registry
export type AnyScene = Scene<any>;

/** Declares a scene. Identity function that checks the shape and infers `O`. */
export function defineScene<O extends object>(scene: Scene<O>): Scene<O> {
  return scene;
}

/** Declares an option list for state `O`. Identity function for inference. */
export function defineOptions<O extends object>(
  options: readonly OptionSpec<O>[]
): readonly OptionSpec<O>[] {
  return options;
}
