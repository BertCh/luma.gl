// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number formatter for chart axes and hover values. */
export type ChartFormatter = (value: number) => string;

/** RGBA colour with 0-255 channels (alpha optional). */
export type ChartColor = readonly [number, number, number, number?];

/** A shaded band across the plot (season, LEO/MEO shell, "not significant" zone). */
export type ChartBand = {
  from: number;
  to: number;
  /** Axis the band spans. Default `'x'`. */
  axis?: 'x' | 'y';
  label?: string;
  /** `'muted'` (default grey), `'signal'` (the controlled thing), or a palette slot 0-5. */
  tone?: 'muted' | 'signal' | number;
};

/**
 * Links a chart to an option: the option's value is drawn as the `.chart-marker` (it moves when
 * the slider moves) and clicking or dragging on the chart writes the option (snapped to the
 * option's step). The shell wires this; scenes only declare it.
 */
export type ChartLink = {
  option: string;
  /** Axis carrying the option value. Default `'x'`. */
  axis?: 'x' | 'y';
  /** Marker label, for example `(value) => \`r = ${value} m\``. */
  label?: (value: number) => string;
};

/** One polyline of a line chart. */
export type ChartSeries = {
  /** Legend and hover label. */
  label?: string;
  /** Values on the y axis. */
  y: ArrayLike<number>;
  /** Values on the x axis. Defaults to `0, 1, 2, ...`. */
  x?: ArrayLike<number>;
  /** Palette slot `0`-`5` (`--chart-1` to `--chart-6`). Defaults to the series index. */
  color?: number;
  /** Draw dashed, for a reference or expected curve. */
  dashed?: boolean;
  /** Fill down to the axis. */
  area?: boolean;
  /** Thin grey context line (another region, the global curve) drawn under the others. */
  ghost?: boolean;
  /** Stroke width in viewBox units. Default 1.75 (ghost 1). */
  width?: number;
  /** Plot against the secondary (right) axis. */
  axis?: 'y' | 'y2';
  /** Label the series at its last point instead of in a legend (default when <= 3 series). */
  directLabel?: boolean;
  /** Point markers at every vertex. */
  points?: boolean;
};

/** Options shared by the chart kinds. */
export type ChartCommon = {
  /** Height in viewBox units (the chart is 320 wide and scales to the panel). Default 140. */
  height?: number;
  xLabel?: string;
  yLabel?: string;
  /** Overrides the computed x range. */
  xDomain?: readonly [number, number];
  /** Overrides the computed y range. */
  yDomain?: readonly [number, number];
  formatX?: ChartFormatter;
  formatY?: ChartFormatter;
  /** Vertical rules at x values (for example the observed statistic in a permutation histogram). */
  markers?: readonly {x: number; label?: string}[];
  /** Horizontal rules at y values (for example the expected value or a significance threshold). */
  guides?: readonly {y: number; label?: string}[];
  /** Accessible description of what the chart shows. */
  description?: string;
  /** Short title drawn above the plot. */
  title?: string;
  /** Axis scales. `'log'` needs a positive domain and draws decade ticks. */
  xScale?: 'linear' | 'log';
  yScale?: 'linear' | 'log';
  /** Shaded bands. */
  bands?: readonly ChartBand[];
  /** Two-way link to an option value (slider marker, click to set). */
  link?: ChartLink;
  /** Adds a "Show values" disclosure with the data as a table. Default true. */
  table?: boolean;
};

/** Line chart with optional envelope band, for example K(d) against a simulation envelope. */
export type LineChartData = ChartCommon & {
  kind: 'line';
  series: readonly ChartSeries[];
  /** Shaded range between `low` and `high` (same x as `x`, or index). */
  band?: LineEnvelope;
  /** More envelopes (each with its own colour slot). */
  envelopes?: readonly (LineEnvelope & {color?: number})[];
  /** Secondary y axis on the right, for series with `axis: 'y2'`. */
  y2Domain?: readonly [number, number];
  y2Label?: string;
  formatY2?: ChartFormatter;
};

/** A shaded envelope between two curves (simulation envelope, interquartile range). */
export type LineEnvelope = {
  x?: ArrayLike<number>;
  low: ArrayLike<number>;
  high: ArrayLike<number>;
  label?: string;
};

/**
 * Bar chart over categories, or a histogram when `xDomain` is given (bars then span equal bins
 * between `xDomain[0]` and `xDomain[1]` and the axis is numeric).
 */
export type BarChartData = ChartCommon & {
  kind: 'bars' | 'histogram';
  values: ArrayLike<number>;
  /** Category names, one per bar. Ignored for histograms. */
  labels?: readonly string[];
  /** Bars drawn in the accent color; the rest are muted. */
  highlight?: readonly number[];
  /** Palette slot of the bars. Default `0`. */
  color?: number;
  /**
   * Histogram class breaks: ticks at the breaks under the axis, and bars coloured by the class
   * of the bin centre with `classColors` (the map's class table).
   */
  breaks?: readonly number[];
  classColors?: readonly ChartColor[];
  /** A "now" marker at this x (the hovered value, the observed statistic). */
  now?: number;
  nowLabel?: string;
  /** Explicit bar colours, one per bar (overrides `color`, `highlight`, `classColors`). */
  colors?: readonly ChartColor[];
  /** Horizontal bars with labels on the left (ranked lists). Bars only. */
  horizontal?: boolean;
  /** Called with the bar index when a bar is clicked (linked highlighting). */
  onBarClick?: (index: number) => void;
};

/** Scatter plot (Moran scatter, fitted vs observed, feature space). */
export type ScatterChartData = ChartCommon & {
  kind: 'scatter';
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  /** Class index per point into `palette` (or palette slot index when `palette` is omitted). */
  colorIndex?: ArrayLike<number>;
  palette?: readonly ChartColor[];
  /** Point radius in viewBox units. Default 2. */
  radius?: number;
  /** Point opacity. Default 0.7 (lower automatically above 2,000 points). */
  opacity?: number;
  /** Points drawn with a ring on top (the selected or named ones). */
  ringed?: readonly number[];
  /** The y = x line. */
  diagonal?: boolean;
  /**
   * Reference lines at `x` and `y` dividing four quadrants, with optional corner labels in
   * mathematical quadrant order I-IV: top-right, top-left, bottom-left, bottom-right (the Moran
   * scatter's HH, LH, LL, HL).
   */
  quadrants?: {
    x: number;
    y: number;
    labels?: readonly [string, string, string, string];
  };
  /** A fitted line `y = slope * x + intercept`, labelled ("slope = I = 0.41"). */
  fit?: {slope: number; intercept: number; label?: string};
  /** A shaded reference rectangle (the CSR region, the null box). */
  referenceRect?: {x0: number; x1: number; y0: number; y1: number; label?: string};
  /** Brush: called with the selected point indices (or `null` when cleared). */
  onBrush?: (indices: readonly number[] | null) => void;
  /** Called when a point is clicked. */
  onPointClick?: (index: number) => void;
};

/** A time strip: bars or a line over time, with a window bracket, a playhead and season bands. */
export type TimelineChartData = ChartCommon & {
  kind: 'timeline';
  /** Bin start (or sample) times, ascending, in the scene's time units. */
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  mode?: 'bars' | 'line' | 'area';
  /** Bracketed window `[from, to]` with faded outside. */
  window?: readonly [number, number];
  /** Playhead position. */
  playhead?: number;
  /** Event ticks ("CNC 28 Apr", "landfall"). */
  events?: readonly {at: number; label?: string}[];
  /** Click or drag sets the playhead (`link` does the same through an option). */
  onScrub?: (time: number) => void;
};

/** A K x K (or 24 x 7) heat matrix with labels and marginal bars. */
export type MatrixChartData = Omit<ChartCommon, 'xScale' | 'yScale'> & {
  kind: 'matrix';
  /** Row-major values, `rows * columns`. `NaN` draws an empty cell. */
  values: ArrayLike<number>;
  rows: number;
  columns: number;
  rowLabels?: readonly string[];
  columnLabels?: readonly string[];
  /** Colour: a ramp name with optional class breaks, or one colour per cell. */
  ramp?: string;
  reverse?: boolean;
  breaks?: readonly number[];
  /** Diverging ramps: the neutral midpoint value. */
  midpoint?: number;
  cellColors?: readonly ChartColor[];
  /** Marginal sums as bars along the right (`rows`), the bottom (`columns`) or both. */
  marginals?: 'rows' | 'columns' | 'both' | 'none';
  /** Outline the diagonal (transition matrices). */
  diagonal?: boolean;
  /** Highlighted row / column / cell. */
  highlight?: {row?: number; column?: number};
  /** Cell text formatter (values printed in cells when the matrix is <= 8 x 8). */
  formatCell?: ChartFormatter;
  onCellClick?: (row: number, column: number) => void;
};

/** Lorenz curve with the shaded Gini area. */
export type LorenzChartData = ChartCommon & {
  kind: 'lorenz';
  /** Cumulative population share (0-1), ascending, starting at 0. */
  x: ArrayLike<number>;
  /** Cumulative share of the quantity (0-1). */
  y: ArrayLike<number>;
  /** Gini coefficient label (computed from x, y when omitted). */
  gini?: number;
};

/** Forest plot: estimates with intervals around a reference. */
export type ForestChartData = Omit<ChartCommon, 'yScale'> & {
  kind: 'forest';
  rows: readonly {
    label: string;
    estimate: number;
    low: number;
    high: number;
    highlight?: boolean;
  }[];
  /** Reference value line (0, 1, the global estimate). */
  reference?: number;
  referenceLabel?: string;
};

/** Dumbbell (two values per row) or slopegraph (two columns joined by lines). */
export type DumbbellChartData = Omit<ChartCommon, 'yScale'> & {
  kind: 'dumbbell';
  rows: readonly {label: string; a: number; b: number; highlight?: boolean}[];
  aLabel: string;
  bLabel: string;
};

/** Slopegraph: two columns of values joined per row. */
export type SlopeChartData = Omit<ChartCommon, 'xScale'> & {
  kind: 'slope';
  rows: readonly {label: string; a: number; b: number; highlight?: boolean}[];
  aLabel: string;
  bLabel: string;
};

/** Rose / polar bar chart for cyclic data (hours, directions, months). */
export type RoseChartData = Pick<ChartCommon, 'height' | 'description' | 'title' | 'table'> & {
  kind: 'rose';
  values: ArrayLike<number>;
  /** Labels placed around the circle, evenly from the top clockwise. */
  labels?: readonly string[];
  /** Colours per sector (a cyclic ramp sampled per class). */
  colors?: readonly ChartColor[];
  /** Highlighted sector. */
  highlight?: number;
  /** A second, ghost series (the citywide baseline). */
  baseline?: ArrayLike<number>;
};

/** One stacked share bar (composition). */
export type StackedBarChartData = Pick<
  ChartCommon,
  'height' | 'description' | 'title' | 'table'
> & {
  kind: 'stacked';
  segments: readonly {label: string; value: number; color?: ChartColor}[];
  /** Show shares as percentages (default) or raw values. */
  format?: 'percent' | 'value';
};

/** Small multiples: 2-6 charts in a grid sharing their domains (bandwidth, k, season). */
export type MultiplesChartData = {
  kind: 'multiples';
  charts: readonly Exclude<ChartData, MultiplesChartData>[];
  titles?: readonly string[];
  /** Columns in the grid. Default 2 (3 for 6 panels). */
  columns?: number;
  /** Share x and y domains across panels (default true). */
  shareDomains?: boolean;
  /** Index of the highlighted panel (the current option). */
  highlight?: number;
  description?: string;
};

/**
 * An inline SVG diagram for mechanisms (a leash, a clip walk, a shoelace fan). `svg` is markup
 * of the SVG's children in a `viewBox` of `width x height`; use `currentColor` and the classes
 * `diagram-ink`, `diagram-muted`, `diagram-signal`, `diagram-accent`, `diagram-fill` so it follows
 * the theme.
 */
export type DiagramChartData = {
  kind: 'diagram';
  svg: string;
  width: number;
  height: number;
  description: string;
};

/** Tiny axis-free trend line for a readout row. */
export type SparklineData = {
  kind: 'sparkline';
  values: ArrayLike<number>;
  /** Index drawn as a dot (the current value). */
  highlight?: number;
  color?: number;
  /** Height in viewBox units. Default 36. */
  height?: number;
  description?: string;
};

/** Data accepted by `ctx.setChart`. */
export type ChartData =
  | LineChartData
  | BarChartData
  | SparklineData
  | ScatterChartData
  | TimelineChartData
  | MatrixChartData
  | LorenzChartData
  | ForestChartData
  | DumbbellChartData
  | SlopeChartData
  | RoseChartData
  | StackedBarChartData
  | MultiplesChartData
  | DiagramChartData;
