// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ChartColor, ScatterChartData} from '../chart-types';
import {FUNNEL_LIMITS, formatCompactNumber, RATE_SCALE} from './rate-smoothing.style';

/**
 * CPU companions of the GPU rate contributors: the empirical-Bayes weight `w` (the library does
 * not expose it), the neighbourhood-pooled weight and rate of a county, quantiles and ranks for
 * tooltips, per-state cluster tallies and the funnel plot. Pure functions, no luma.gl.
 */

/** The sorted finite values of an array (a copy). */
export function sortFinite(values: ArrayLike<number>): Float64Array {
  const finite: number[] = [];
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) finite.push(values[index]);
  }
  return Float64Array.from(finite).sort();
}

/** Linear-interpolated quantile `q` in `[0, 1]` of a sorted array; NaN when empty. */
export function getSortedQuantile(sorted: ArrayLike<number>, q: number): number {
  if (sorted.length === 0) return Number.NaN;
  const position = Math.min(Math.max(q, 0), 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/** Share of a sorted array at or below `value`, as a percentile 0-100. */
export function getPercentile(sorted: ArrayLike<number>, value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return sorted.length ? (low / sorted.length) * 100 : Number.NaN;
}

/**
 * The empirical-Bayes weight of every county, `w = a / (a + m / b)`: how much of its smoothed rate
 * is its own. `a` is the estimated variance of the true rates, `m` the pooled rate (per woman-year)
 * and `b` the woman-years. esda does not clamp `a`, so a negative `a` is clamped to 0 here for display.
 */
export function getEmpiricalBayesWeights(
  priorVariance: number,
  pooledRate: number,
  womenYears: ArrayLike<number>
): Float32Array {
  const weights = new Float32Array(womenYears.length);
  const variance = Math.max(priorVariance, 0);
  for (let row = 0; row < womenYears.length; row++) {
    const noise = pooledRate / womenYears[row];
    weights[row] = variance + noise > 0 ? variance / (variance + noise) : 0;
  }
  return weights;
}

/** Per-county result of pooling a county with its neighbours. */
export type LocalPooling = {
  /** Pooled rate of the neighbourhood (county included), per woman-year. */
  pooledRates: Float32Array;
  /** Weight on the county's own rate under the neighbourhood prior. */
  weights: Float32Array;
};

/**
 * `GPUSpatialEmpiricalBayesRates` on the CPU for display: the neighbourhood of a county is the
 * county plus its listed neighbours; `m = E / B`, `s2 = sum b (e / b - m)^2 / B`,
 * `a = max(s2 - m / (B / n), 0)` and `w = a / (a + m / b)`.
 */
export function getLocalPooling(
  events: ArrayLike<number>,
  womenYears: ArrayLike<number>,
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>
): LocalPooling {
  const count = events.length;
  const pooledRates = new Float32Array(count);
  const weights = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    const first = offsets[row];
    const end = offsets[row + 1];
    let eventSum = events[row];
    let populationSum = womenYears[row];
    for (let slot = first; slot < end; slot++) {
      eventSum += events[neighbors[slot]];
      populationSum += womenYears[neighbors[slot]];
    }
    const members = end - first + 1;
    const pooled = populationSum > 0 ? eventSum / populationSum : 0;
    let weightedVariance = womenYears[row] * (events[row] / womenYears[row] - pooled) ** 2;
    for (let slot = first; slot < end; slot++) {
      const member = neighbors[slot];
      weightedVariance += womenYears[member] * (events[member] / womenYears[member] - pooled) ** 2;
    }
    const variance = Math.max(
      populationSum > 0 ? weightedVariance / populationSum - pooled / (populationSum / members) : 0,
      0
    );
    const noise = pooled / womenYears[row];
    pooledRates[row] = pooled;
    weights[row] = variance + noise > 0 ? variance / (variance + noise) : 0;
  }
  return {pooledRates, weights};
}

/** Counts of a value per class of `breaks` (the number of breaks `<= value` is the class). */
export function getClassCounts(values: ArrayLike<number>, breaks: readonly number[]): number[] {
  const counts = new Array<number>(breaks.length + 1).fill(0);
  for (let row = 0; row < values.length; row++) {
    const value = values[row];
    if (!Number.isFinite(value)) continue;
    let index = 0;
    while (index < breaks.length && value >= breaks[index]) index++;
    counts[index]++;
  }
  return counts;
}

/** Class index of a value among `breaks`, or -1 for a non-finite value. */
export function getClassIndexAmong(value: number, breaks: readonly number[]): number {
  if (!Number.isFinite(value)) return -1;
  let index = 0;
  while (index < breaks.length && value >= breaks[index]) index++;
  return index;
}

/** One state's tally of a cluster quadrant. */
export type StateTally = {state: string; count: number; total: number};

/**
 * The states with the most counties in one cluster quadrant (1 high-high, 3 low-low), most first.
 * `states[row]` is the postal code of the county; ties go to the larger share.
 */
export function getStateTallies(
  quadrants: ArrayLike<number>,
  states: readonly string[],
  quadrant: number
): StateTally[] {
  const tallies = new Map<string, StateTally>();
  for (let row = 0; row < states.length; row++) {
    const state = states[row];
    let tally = tallies.get(state);
    if (!tally) {
      tally = {state, count: 0, total: 0};
      tallies.set(state, tally);
    }
    tally.total++;
    if (quadrants[row] === quadrant) tally.count++;
  }
  return [...tallies.values()]
    .filter(tally => tally.count > 0)
    .sort((a, b) => b.count - a.count || b.count / b.total - a.count / a.total);
}

/** Inputs of {@link buildFunnelChart}. */
export type FunnelInput = {
  /** Woman-years at risk of every county (the x axis, log scale). */
  womenYears: ArrayLike<number>;
  /** Raw rate of every county, per 1,000. */
  rawRates: ArrayLike<number>;
  /** Rate drawn as the coloured dots, per 1,000, or `null` to colour the raw dots themselves. */
  shownRates: ArrayLike<number> | null;
  /** Class index of every coloured dot into `palette`. */
  colorIndex: ArrayLike<number>;
  /** One colour per class, then the ghost colour, then the limit-curve ink. */
  palette: readonly ChartColor[];
  /** Pooled rate, per woman-year. */
  pooledRate: number;
  /** Variance of the true rates: 0 draws the Poisson funnel, `a` the overdispersed one. */
  priorVariance: number;
  /** Rows to ring (the selected county). */
  selectedRows: readonly number[];
  /** Called with a county row when a dot is clicked. */
  onSelect: (row: number) => void;
  /** Top of the y axis, per 1,000. */
  yMaximum: number;
};

const CURVE_POINTS = 150;

/**
 * The funnel plot as a scatter chart: rate against woman-years on a log axis, one dot per county
 * coloured by the map's class, and the control-limit curves `m +- z sqrt(a + m / b)` drawn as
 * dense dots (the scatter chart has no line series). Small counties fan out; large ones cluster
 * at the pooled rate.
 */
export function buildFunnelChart(input: FunnelInput): ScatterChartData {
  const count = input.womenYears.length;
  const ghostIndex = input.palette.length - 2;
  const curveIndex = input.palette.length - 1;
  const x: number[] = [];
  const y: number[] = [];
  const colorIndex: number[] = [];
  if (input.shownRates) {
    for (let row = 0; row < count; row++) {
      x.push(input.womenYears[row]);
      y.push(input.rawRates[row]);
      colorIndex.push(ghostIndex);
    }
  }
  const shown = input.shownRates ?? input.rawRates;
  for (let row = 0; row < count; row++) {
    x.push(input.womenYears[row]);
    y.push(shown[row]);
    colorIndex.push(input.colorIndex[row]);
  }
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = 0;
  for (let row = 0; row < count; row++) {
    const value = input.womenYears[row];
    if (value > 0 && value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  const logMinimum = Math.log(minimum);
  const logStep = (Math.log(maximum) - logMinimum) / (CURVE_POINTS - 1);
  for (const limit of FUNNEL_LIMITS) {
    for (const sign of [-1, 1]) {
      for (let step = 0; step < CURVE_POINTS; step++) {
        const population = Math.exp(logMinimum + step * logStep);
        const rate =
          (input.pooledRate +
            sign * limit * Math.sqrt(input.priorVariance + input.pooledRate / population)) *
          RATE_SCALE;
        if (rate < 0 || rate > input.yMaximum) continue;
        x.push(population);
        y.push(rate);
        colorIndex.push(curveIndex);
      }
    }
  }
  const offset = input.shownRates ? count : 0;
  return {
    kind: 'scatter',
    title: 'Rate against county size',
    x,
    y,
    colorIndex,
    palette: input.palette,
    radius: 1.5,
    opacity: 0.75,
    xScale: 'log',
    xDomain: [minimum, maximum],
    yDomain: [0, input.yMaximum],
    xLabel: 'Woman-years at risk, 2021-2023 (log scale)',
    yLabel: 'Births per 1,000 women a year',
    formatX: formatCompactNumber,
    formatY: value => value.toFixed(0),
    guides: [{y: input.pooledRate * RATE_SCALE, label: 'pooled'}],
    ringed: input.selectedRows.map(row => row + offset),
    table: false,
    description:
      'Scatter plot of every county: birth rate against the woman-years at risk on a log axis, with dotted control-limit curves around the pooled rate. Small counties scatter far outside the curves; large ones sit near the pooled rate.',
    onPointClick: index => {
      const row = index >= offset ? index - offset : index;
      if (row >= 0 && row < count) input.onSelect(row);
    }
  };
}
