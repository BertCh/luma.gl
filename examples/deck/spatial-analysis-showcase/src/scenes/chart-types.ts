// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number formatter for chart axes and hover values. */
export type ChartFormatter = (value: number) => string;

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
};

/** Line chart with optional envelope band, for example K(d) against a simulation envelope. */
export type LineChartData = ChartCommon & {
  kind: 'line';
  series: readonly ChartSeries[];
  /** Shaded range between `low` and `high` (same x as `x`, or index). */
  band?: {
    x?: ArrayLike<number>;
    low: ArrayLike<number>;
    high: ArrayLike<number>;
    label?: string;
  };
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
export type ChartData = LineChartData | BarChartData | SparklineData;
