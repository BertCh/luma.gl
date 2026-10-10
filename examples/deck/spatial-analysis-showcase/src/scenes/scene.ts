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
import type {
  BasemapSpec,
  ClassTable,
  FurnitureSpec,
  LngLat,
  MapAnnotation,
  MapHighlight
} from '../cartography/types';

export type {SceneFrame, ScenePointerEvent, ViewState};
export type {
  AnnotationTone,
  BasemapLabels,
  BasemapSpec,
  BasemapStyleName,
  CartoucheSpec,
  ClassColor,
  ClassTable,
  FurnitureSpec,
  GroundName,
  GroundPalette,
  LabelAnchor,
  LabelPreset,
  LngLat,
  MapAnnotation,
  MapClockSpec,
  MapHighlight,
  ScaleBarSpec,
  ScaleBarUnits
} from '../cartography/types';
export type {
  BarChartData,
  ChartData,
  ChartSeries,
  DiagramChartData,
  DumbbellChartData,
  ForestChartData,
  LineChartData,
  LorenzChartData,
  MatrixChartData,
  MultiplesChartData,
  RoseChartData,
  ScatterChartData,
  SlopeChartData,
  SparklineData,
  StackedBarChartData,
  TimelineChartData
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
  /**
   * Engine detail most readers never need (capacities, workgroup sizes, accumulation modes): the
   * "All controls" tab folds these under a closed "Expert" group. Never list them in a step.
   */
  expert?: boolean;
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
  /**
   * A consequence line under the slider computed from the value and the state, for example
   * `"23 areas hidden"` or `"100 m = 7 px at this zoom"`.
   */
  describe?: (value: number, state: O) => string;
  /** Labelled ticks under the track (the default, a published threshold). */
  marks?: readonly {value: number; label?: string}[];
  /** A range of the track drawn as a warning ("too small to resolve", "slow"). */
  danger?: readonly [number, number];
  /**
   * `'slider'` (default) or `'stepper'`: previous and next buttons around the value, for
   * discrete ladders ("round 3 of 12", k, resolution).
   */
  display?: 'slider' | 'stepper';
  /**
   * Adds a Play button that sweeps the value from `from` to `to` (defaults: min to max) over
   * `durationMs`, eased, then stops (or loops). Pause and step buttons appear while it runs. Under
   * `prefers-reduced-motion` Play steps once per click instead of animating.
   */
  autoSweep?: {
    from?: number;
    to?: number;
    durationMs: number;
    ease?: 'linear' | 'in-out';
    loop?: boolean;
  };
};

/** Dropdown, segmented control or chips. */
export type SelectOption<O> = OptionBase<O> & {
  kind: 'select';
  options: readonly {value: string; label: string; help?: string}[];
  default: string;
  /**
   * `'dropdown'` (default), `'segmented'` (2-4 short choices side by side; the right control for
   * compile-time variants, which keeps the rebuild badge), or `'chips'` (wrapping pills).
   */
  display?: 'dropdown' | 'segmented' | 'chips';
};

/** Option values a preset writes (typed per scene; erased to a record in the shell). */
export type PresetValues<O> = [O] extends [never]
  ? Readonly<Record<string, OptionValue>>
  : Partial<O>;

/**
 * Preset chips: each chip writes several options at once (a traveller, a set of weights, a time
 * window). No state of its own; the chip whose values all match the current state is shown as
 * selected. `resets` lists options returned to their defaults when any chip is pressed.
 */
export type PresetOption<O> = {
  kind: 'preset';
  id: string;
  label: string;
  help?: string;
  group?: string;
  presets: readonly {label: string; values: PresetValues<O>; help?: string}[];
  resets?: readonly (keyof O & string)[];
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
  expert?: boolean;
};

/** One declarative option; the shell generates the control, the URL state and the reset button. */
export type OptionSpec<O> =
  | SliderOption<O>
  | SelectOption<O>
  | ToggleOption<O>
  | RangeOption<O>
  | ButtonOption
  | PresetOption<O>;

/** Legend entry color, RGBA 0-255. */
export type LegendColor = readonly [number, number, number, number?];

/** Fields every legend kind accepts. */
export type LegendCommon = {
  /**
   * Stable id. Needed for `'gpu'` ramp extents and for interactive legends (`interactive: true`
   * sends `instance.onLegendFilter(id, classes)`).
   */
  id?: string;
  /**
   * The normalisation basis, shown after the unit ("per km²", "per 1,000 residents"). Counts on
   * unequal areas without a basis are the "wrong map" by definition.
   */
  basis?: string;
};

/** What a legend draws. Legends are recomputed from the option state. */
export type LegendSpec = LegendCommon &
  (
    | {
        kind: 'ramp';
        /** Needed when `extent` is `'gpu'`: the scene calls `ctx.setLegendExtent(id, [min, max])`. */
        id?: string;
        title: string;
        /** Ramp name from `RAMP_STOPS`; must match the layer's `colormap`. */
        ramp: RampName;
        /** Match the layer's `reverseRamp`: the gradient runs from the ramp's high end. */
        reverse?: boolean;
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
        /** Number formatter for the end labels and ticks. */
        format?: (value: number) => string;
        /**
         * Diverging ramps: the value of the neutral midpoint (zero, a mean, a ratio of 1). Drawn as a
         * labelled tick; `midpointLabel` replaces the number (for example `'no change'`).
         */
        midpoint?: number;
        midpointLabel?: string;
        /** Extra labelled ticks, in value units inside `extent` (linear placement). */
        ticks?: readonly number[];
        /**
         * Optional distribution drawn as small bars over the ramp: bin counts spanning `extent`
         * evenly (for example a contributor `histogram` output read back with a `SummaryReader`).
         */
        histogram?: readonly number[];
        /**
         * Ramp trim `[t0, t1]` matching the layer's `rampRange` (for example `[0.15, 1]` so the
         * lowest value is not black on a dark ground).
         */
        range?: readonly [number, number];
        /** A value marked on the bar (the hovered feature's value), in value units. */
        marker?: number;
        /** Short note under the ramp ("Top 2 % clipped", "Square-root scale"). */
        note?: string;
      }
    | {
        /**
         * Classed (stepped) colour: `breaks.length + 1` classes, as drawn by a layer with the same
         * `classBreaks` and `colormap`. Swatch colours come from `getClassColors(ramp, n, reverse)`
         * unless `colors` is given (one per class).
         */
        kind: 'classes';
        title: string;
        unit?: string;
        ramp?: RampName;
        reverse?: boolean;
        /** One colour per class, low class first. Overrides `ramp`. */
        colors?: readonly LegendColor[];
        /**
         * Interior class breaks, ascending: the same array the layer gets as `classBreaks`.
         * Required unless `table` is given.
         */
        breaks?: readonly number[];
        /** Data `[min, max]` for the outer labels. Without it the ends read `< b0` and `>= bN`. */
        extent?: readonly [number, number];
        /** One label per class, replacing the generated ranges (for example `'Hot 99%'`). */
        labels?: readonly string[];
        /** Features per class, shown next to each class (and as bar lengths in `'list'` layout). */
        counts?: readonly number[];
        /**
         * `'bar'` (default without `counts`): equal blocks with the break values between them.
         * `'list'` (default with `counts`): one row per class with range, swatch and count.
         */
        layout?: 'bar' | 'list';
        /** Number formatter for the break labels. */
        format?: (value: number) => string;
        /** Optional distribution under the classes: bin counts spanning `extent` evenly. */
        histogram?: readonly number[];
        /** Short note under the classes, for example the classification method. */
        note?: string;
        /**
         * The class table the layer draws (`makeClassTable`). When given, `breaks`, `colors`,
         * `labels`, `extent`, `unit`, `hatched` and `noData` default to the table's.
         */
        table?: ClassTable;
        /** Classes drawn hatched, by index. */
        hatched?: readonly number[];
        /** A "No data" swatch after the classes (always add one when rows can be missing). */
        noData?: {color?: LegendColor; label?: string; count?: number; hatched?: boolean};
        /** A value marked on the bar or list (the hovered feature's value). */
        marker?: number;
        /**
         * Class swatches act as filters: hover isolates a class, click locks it, Esc shows all.
         * The scene receives `onLegendFilter(id, classIndices | null)` and dims the others.
         */
        interactive?: boolean;
      }
    | {
        /**
         * Two-variable classed colour: an `n x n` grid. `colors[row * n + column]`, column = class of
         * the x variable (low to high, left to right), row = class of the y variable (low to high,
         * bottom to top). Use the same list as the layer's category `palette` (see
         * `BIVARIATE_PALETTES`).
         */
        kind: 'bivariate';
        title: string;
        size: number;
        colors: readonly LegendColor[];
        xLabel: string;
        yLabel: string;
        /** Optional low and high end labels of each axis. */
        xEnds?: readonly [string, string];
        yEnds?: readonly [string, string];
        note?: string;
      }
    | {
        kind: 'categories';
        title: string;
        entries: readonly {
          color: LegendColor;
          label: string;
          /** Second text column ("~ every 10 min each way"). */
          detail?: string;
          /** Count shown right-aligned. */
          count?: number;
          /** Swatch shape. Default `'swatch'`. */
          shape?: 'swatch' | 'dot' | 'ring' | 'line' | 'hatch';
        }[];
        /** Short note under the entries. */
        note?: string;
        /** Entries act as filters (see `classes`); indices are entry positions. */
        interactive?: boolean;
        /** `'grid'` (default, wrapping columns) or `'list'` (one row each, for details and counts). */
        layout?: 'grid' | 'list';
      }
    | {
        /**
         * A two-way key: `rows x columns` swatches (hue x tier, aspect x slope), `colors[row *
         * columns + column]`, row 0 at the top.
         */
        kind: 'matrix';
        title: string;
        rows: readonly string[];
        columns: readonly string[];
        colors: readonly LegendColor[];
        rowTitle?: string;
        columnTitle?: string;
        note?: string;
      }
    | {
        /**
         * Cyclic key drawn as a ring: hour of day, day of year, aspect or direction. The ring starts
         * at the top (00:00 / north) and runs clockwise.
         */
        kind: 'cyclic';
        title: string;
        /** A cyclic ramp (`romao`), or one colour per class (classed ring). */
        ramp?: RampName;
        colors?: readonly LegendColor[];
        /** Labels placed evenly around the ring, starting at the top ("N E S W", "0 6 12 18"). */
        labels: readonly string[];
        /** Extra ticks, as fractions of the cycle (dawn, dusk). */
        marks?: readonly {at: number; label: string}[];
        note?: string;
      }
    | {
        /**
         * Value-by-alpha key: each colour row fades from `alphaRange[0]` to `alphaRange[1]` across
         * `steps` swatches ("estimate" down, "reliability" across).
         */
        kind: 'alpha';
        title: string;
        colors: readonly LegendColor[];
        /** Row labels, one per colour (optional). */
        rowLabels?: readonly string[];
        alphaRange?: readonly [number, number];
        steps?: number;
        /** Labels of the low and high alpha ends ("unreliable", "reliable"). */
        ends: readonly [string, string];
        note?: string;
      }
    | {
        /** Line symbols: stroke colour and width per entry (road classes, flow widths, boundaries). */
        kind: 'line';
        title: string;
        entries: readonly {
          color: LegendColor;
          widthPixels: number;
          label: string;
          dashed?: boolean;
        }[];
        note?: string;
      }
    | {
        kind: 'size';
        title: string;
        entries: readonly {radiusPixels: number; label: string}[];
        color?: LegendColor;
        /**
         * `'row'` (default): discs side by side. `'nested'`: concentric circles sharing a baseline
         * with leader lines to the labels, the classic proportional-symbol legend.
         */
        layout?: 'row' | 'nested';
        /** Outline of the discs; defaults to a theme ink when `layout` is `'nested'`. */
        outline?: LegendColor;
        unit?: string;
        note?: string;
      }
  );

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
  /** Unit appended to numeric values ("km²", "trips"). */
  unit?: string;
  /**
   * Engine detail (capacity, overflow, status flags, sum order): listed in the collapsed
   * "Under the hood" group instead of the main readouts, and never in a step by default.
   */
  hood?: boolean;
  /**
   * `'tile'` shows a key figure: a big tabular number with the label under it (use for the one
   * to three numbers a step is about).
   */
  emphasis?: 'tile';
  /**
   * Charts only: `'panel'` (default) draws the chart in the step card and the readout list;
   * `'map'` draws it in a screen-anchored inset card on the map (a calendar matrix, a Hovmoeller
   * key) while a step lists the readout, in the given corner (default top-right).
   */
  placement?: 'panel' | 'map';
  mapCorner?: 'top-right' | 'bottom-right' | 'top-left' | 'bottom-left';
};

/** One stage of a scene's pipeline strip. */
export type PipelineStage = {
  id: string;
  /** Chip label, two or three words ("Grid build", "Gaussian pass"). */
  label: string;
  /** One-line explanation shown on hover and for the active stage. */
  detail?: string;
  /**
   * Clicking the chip writes this option value, so the map shows that intermediate product
   * (the stage helper: the scene keeps both layers mounted and cross-fades).
   */
  show?: {option: string; value: OptionValue};
};

/** Row of a structured tooltip. */
export type TooltipRow = {
  label: string;
  value: string | number;
  unit?: string;
  /** Class swatch next to the value (the mapped value's class colour). */
  swatch?: LegendColor;
  /** Draw the row stronger (the mapped value). */
  emphasis?: boolean;
};

/**
 * A structured tooltip. The first row should be the mapped value with its class swatch, the
 * second its rank or percentile. The card follows the pointer, flips at the edges, and pins on
 * tap (touch) or click with Shift.
 */
export type TooltipContent = {
  title: string;
  subtitle?: string;
  rows?: readonly TooltipRow[];
  /** A 48 x 16 sparkline (a time series of the feature). */
  spark?: readonly number[];
  /** Index of the current value in `spark` (dot). */
  sparkHighlight?: number;
  /** Muted note at the foot ("Suppressed: fewer than 20 events"). */
  note?: string;
  /** Anchor the card here (a polygon's centroid) instead of at the pointer. */
  anchor?: LngLat;
  /** Outline drawn on the map while the card shows (the hovered feature). */
  highlight?: MapHighlight;
};

/** A swipe or toggle comparison of two states inside one step. */
export type CompareSpec = {
  /**
   * `'swipe'` (default): a draggable vertical divider; layers with `compareSide: 'a'` draw left
   * of it and `'b'` right of it. `'toggle'`: a "Hold to compare" button shows only side `b`
   * while held.
   */
  mode?: 'swipe' | 'toggle';
  /** Labels of the two sides ("Equal interval", "Natural breaks"). */
  labels: readonly [string, string];
  /** Initial divider position as a fraction of the map width. Default 0.5. */
  position?: number;
};

/** A docked time bar bound to the scene's playback options (see `engine/playback.ts`). */
export type TimelineSpec = {
  /** Option holding the playhead (a slider). */
  time: string;
  /** Toggle option that plays and pauses. */
  play?: string;
  /** Slider or select option of the speed multiplier. */
  speed?: string;
  /** Range option of a time window (drawn as handles with a fade). */
  window?: string;
  /** Formats a playhead value for the scrubber label. Defaults to the time slider's format. */
  format?: (value: number) => string;
  /** Season or period bands under the scrubber, in time-option units. */
  bands?: readonly {from: number; to: number; label?: string}[];
  /** Month or day ticks, in time-option units. */
  ticks?: readonly {at: number; label: string}[];
};

/** Camera move of a story step. */
export type StepCamera = Partial<ViewState> & {
  /** Fly time in milliseconds. Defaults to 1600 (0 under `prefers-reduced-motion`). */
  transitionMs?: number;
  /**
   * Frame these bounds `[west, south, east, north]` instead of (or before) `longitude`,
   * `latitude` and `zoom`; zoom is fitted to the free map area.
   */
  bounds?: readonly [number, number, number, number];
  /**
   * Extra CSS pixels kept free around the subject. The shell already keeps the phone sheet and
   * the map corners' furniture free; use this for a subject that must clear a large legend.
   */
  padding?: number | {top?: number; right?: number; bottom?: number; left?: number};
};

/** One step of a scene's guided narrative. */
export type StoryStep<O = OptionState> = {
  /** Stable id used in the URL (`#/story/<scene>/<stepId>`). Kebab-case. */
  id: string;
  title: string;
  /**
   * The finding, as a sentence of nine words or fewer ("Hot spots cluster along the lake, not the
   * river"), shown under the title in the active card and announced to screen readers.
   */
  headline?: string;
  /**
   * One-sentence text alternative of what the map shows in this step (screen readers read it as
   * the map's label).
   */
  textAlternative?: string;
  /** Markdown. Explain what is shown, what the tool computes, why it matters, how to read it. */
  body: string;
  /**
   * Markdown summary of the observation or readout that supports the headline. Keep this distinct
   * from interpretation: state what the reader can see or measure. Supports `{{readoutId}}` slots.
   */
  evidence?: string;
  /**
   * Markdown boundary on the claim: a data limitation, uncertainty or plausible alternative
   * explanation. Supports `{{readoutId}}` slots.
   */
  caveat?: string;
  /** Camera move applied when the step opens. */
  camera?: StepCamera;
  /** Options set when the step opens (applied on top of defaults and earlier steps). */
  options?: Partial<O>;
  /**
   * `'cumulative'` (default): defaults, then every earlier step's options, then this step's.
   * `'fresh'`: defaults then this step's options only, so no hidden state leaks in from earlier
   * steps.
   */
  optionsMode?: 'cumulative' | 'fresh';
  /**
   * Option ids (including button ids) this step asks the reader to use, in display order. The
   * shell renders these controls inside the step card, right under the text, so the step never
   * sends the reader hunting through the full options list. Keep it to the one to four controls
   * the text talks about; an empty array means "just look". Defaults to the ids in `options`.
   */
  controls?: readonly string[];
  /** Readout ids shown inside the step card. Defaults to `highlight.readout`. */
  readouts?: readonly string[];
  /** Marker pinned to a coordinate while the step is open (an accent bubble, never culled). */
  callout?: {coordinate: readonly [number, number]; text: string};
  /**
   * Annotations shown while this step is open, on top of the scene's `annotations` (labels,
   * rings, arrows). See CARTOGRAPHY-GUIDE.md.
   */
  annotations?: readonly MapAnnotation[];
  /** Basemap treatment from this step on (merged over the scene's and earlier steps'). */
  basemap?: BasemapSpec;
  /** Map furniture from this step on (merged over the scene's and earlier steps'). */
  furniture?: FurnitureSpec;
  /** Readout to emphasise in the panel while the step is open. */
  highlight?: {readout?: string};
  /** A swipe or hold-to-compare of two layer sets in this step. */
  compare?: CompareSpec;
  /** Pipeline stage id highlighted in the pipeline strip while this step is open. */
  stage?: string;
  /** Inline SVG diagram (or chart) under the text, for mechanisms prose cannot show. */
  diagram?: ChartData;
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
  /**
   * Luminance of the map ground under the data for the current step: `'dark'` on dark-matter,
   * `'light'` on positron or voyager, the page theme on `'none'`. Equals `theme()` for scenes
   * that keep the default basemap. Pick data colours (outlines, halos, point inks) from this, not
   * from the page theme, when a scene forces a basemap style.
   */
  ground: () => SceneTheme;
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
   * Shows data-driven annotations under `key` (for example the top five hot spots read back from
   * the GPU), or removes them with `null`. Groups from different keys are merged with the scene
   * and step annotations and share the collision pass.
   */
  setAnnotations: (key: string, annotations: readonly MapAnnotation[] | null) => void;
  /**
   * Enables or disables Deck's map drag-pan, for example while drawing a lasso or dragging an
   * observer. Always re-enable it when the gesture ends.
   */
  setMapDragEnabled: (enabled: boolean) => void;
  /**
   * Outlines features on the map above the data (linked highlighting from a chart, a legend or a
   * list; programmatic selection with `pulse`). `null` clears.
   */
  setHighlight: (highlight: MapHighlight | readonly MapHighlight[] | null) => void;
  /** Time for annotations with a `timeRange` (in the scene's time units); `null` shows all. */
  setAnnotationTime: (time: number | null) => void;
  /**
   * Frames `[west, south, east, north]` in the free map area (clear of the phone sheet and the
   * map furniture). Animated unless `transitionMs` is 0 or the reader prefers reduced motion.
   */
  fitBounds: (
    bounds: readonly [number, number, number, number],
    options?: {transitionMs?: number; maxZoom?: number; padding?: number}
  ) => void;
  /**
   * The cost line under the step headline: records processed and GPU passes per frame (the shell
   * adds GPU time and rebuild count). Call when they change.
   */
  setCost: (cost: {records?: number; passes?: number; note?: string} | null) => void;
  /**
   * Data for the docked time bar of `Scene.timeline`: a histogram over `domain` (events per
   * bin) and event ticks.
   */
  setTimelineData: (data: {
    domain?: readonly [number, number];
    histogram?: readonly number[];
    events?: readonly {at: number; label?: string}[];
  }) => void;
  /** Whether the reader asked for reduced motion (skip camera flights, auto-play, glyph motion). */
  reducedMotion: () => boolean;
  /** Ground metres per CSS pixel at the current camera (scale bar arithmetic for step text). */
  getMetersPerPixel: () => number;
  /**
   * Furniture fields computed from data or the camera, merged over the scene's and the step's
   * (for example the cartouche `sample` line counted from the loaded rows, or `scaleBar.ticks` at
   * a bandwidth whose metres follow the zoom). Kept across steps until replaced; `null` clears.
   */
  setFurniture: (overrides: FurnitureSpec | null) => void;
  /**
   * Halo weight of the annotation labels: `'heavy'` (3 px) over dense or saturated data,
   * `'normal'` (2.5 px) otherwise. Reset to normal when the scene changes.
   */
  setAnnotationHalo: (weight: 'normal' | 'heavy') => void;
  /** Divider position of the step's `compare` swipe as a fraction of the map width, or `null`. */
  getCompare: () => {position: number; showing: 'a' | 'b' | 'both'} | null;
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
  /**
   * Called when the map ground changed luminance (a step switched the basemap, or the theme
   * changed a theme-paired basemap). Call `ctx.requestLayers()` if colours depend on it.
   */
  onGroundChange?: (ground: SceneTheme) => void;
  /**
   * Hover tooltip: a structured {@link TooltipContent} (preferred), plain text, or `null`. The
   * shell renders it in its own card (theme- and ground-aware); deck's built-in tooltip is unused.
   */
  getTooltip?: (event: ScenePointerEvent) => TooltipContent | string | null;
  /**
   * An interactive legend's selection changed: `classes` are the isolated class (or entry)
   * indices, `null` when all are shown again. Dim the others (layer `highlightClasses`).
   */
  onLegendFilter?: (legendId: string, classes: readonly number[] | null) => void;
  /** The compare divider moved or the hold-to-compare state changed. Request layers if needed. */
  onCompareChange?: (state: {position: number; showing: 'a' | 'b' | 'both'}) => void;
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
  /** Contributor names. Documented APIs link to the reference; `Frontier*` prototypes stay local. */
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
  /**
   * Basemap treatment (style, label placement, dimming). Omit for the default theme-paired
   * basemap with labels underneath. Steps can override fields with `step.basemap`.
   */
  basemap?: BasemapSpec;
  /** Map furniture: scale bar, north arrow, title cartouche, source credit. Off when omitted. */
  furniture?: FurnitureSpec;
  /** Annotations shown in every step (place names that orient the reader). */
  annotations?: readonly MapAnnotation[];
  /** The computation as a strip of stage chips in the step card (see `PipelineStage`). */
  pipeline?: readonly PipelineStage[];
  /** A docked time bar for playback scenes, bound to the scene's time options. */
  timeline?: TimelineSpec;
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
