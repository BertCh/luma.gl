// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassIndex, getQuantileBreaks} from '../../cartography/breaks';
import {createSeededRandom} from '../../engine/projection';

/**
 * CPU helpers of the global-autocorrelation scene: the Moran scatterplot, the shuffled map, the
 * bivariate classes, the null-distribution chart and the ground-distance conversion. Everything
 * here is derived from values the GPU already produced (the lag, the mean, the permutation
 * histogram); none of it replaces a GPU statistic. Pure TypeScript.
 */

/** Sentinel of a county with no bivariate class (a missing value). */
export const NO_CLASS = 0xffffffff;

/** The Moran scatterplot of a map: one point per county that has neighbours. */
export type MoranScatter = {
  /** County row of every point. */
  rows: Uint32Array;
  /** Standard score of the county's value. */
  x: Float32Array;
  /** Standard score of the average of its neighbours (the spatial lag). */
  y: Float32Array;
  /** Quadrant of every point in chart order I-IV: 0 High-High, 1 Low-High, 2 Low-Low, 3 High-Low. */
  quadrant: Uint8Array;
  /** Points per quadrant in chart order. */
  counts: [number, number, number, number];
  /** Least-squares slope of the lag on the value through the origin. */
  slope: number;
};

/**
 * Builds the Moran scatterplot from the values, the row-standardised spatial lag and the mean and
 * variance the GPU reduced. Counties without a finite value or without neighbours (islands) are
 * left out; quadrants use the same `>= mean` rule as the GPU classification kernel, in f32.
 *
 * @param values Variable of every county.
 * @param lag Row-standardised lag of every county, read back from `GPUSpatialLag`.
 * @param offsets Weights CSR row offsets (`count + 1`); a row with no slots is an island.
 * @param mean Mean of the included values, from the GPU summary.
 * @param variance Population variance of the included values, from the GPU summary.
 */
export function buildMoranScatter(
  values: ArrayLike<number>,
  lag: ArrayLike<number>,
  offsets: ArrayLike<number>,
  mean: number,
  variance: number
): MoranScatter {
  const standardDeviation = Math.sqrt(variance);
  const rows: number[] = [];
  for (let row = 0; row < values.length; row++) {
    if (Number.isFinite(values[row]) && offsets[row + 1] > offsets[row]) rows.push(row);
  }
  const x = new Float32Array(rows.length);
  const y = new Float32Array(rows.length);
  const quadrant = new Uint8Array(rows.length);
  const counts: [number, number, number, number] = [0, 0, 0, 0];
  let crossSum = 0;
  let squareSum = 0;
  rows.forEach((row, index) => {
    const zValue = (values[row] - mean) / standardDeviation;
    const zLag = (lag[row] - mean) / standardDeviation;
    x[index] = zValue;
    y[index] = zLag;
    const high = values[row] >= mean;
    const neighborsHigh = lag[row] >= mean;
    const chartQuadrant = high ? (neighborsHigh ? 0 : 3) : neighborsHigh ? 1 : 2;
    quadrant[index] = chartQuadrant;
    counts[chartQuadrant]++;
    crossSum += zValue * zLag;
    squareSum += zValue * zValue;
  });
  return {
    rows: Uint32Array.from(rows),
    x,
    y,
    quadrant,
    counts,
    slope: squareSum > 0 ? crossSum / squareSum : Number.NaN
  };
}

/**
 * The values dealt at random over the counties: a seeded Fisher-Yates shuffle of the finite
 * values among the rows that have one. Missing values stay where they are, so the shuffled map
 * has the same set of values and the same holes as the real one.
 */
export function shuffleValues(values: ArrayLike<number>, seed: number): Float32Array {
  const finiteRows: number[] = [];
  for (let row = 0; row < values.length; row++) {
    if (Number.isFinite(values[row])) finiteRows.push(row);
  }
  const pool = finiteRows.map(row => values[row]);
  const random = createSeededRandom(seed * 2654435761);
  for (let index = pool.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [pool[index], pool[other]] = [pool[other], pool[index]];
  }
  const shuffled = Float32Array.from(values as ArrayLike<number>);
  finiteRows.forEach((row, index) => {
    shuffled[row] = pool[index];
  });
  return shuffled;
}

/** Tertile breaks (two) of the finite values of a variable. */
export function getTertileBreaks(values: ArrayLike<number>): number[] {
  return getQuantileBreaks(values, 3);
}

/**
 * Bivariate class of every county for a 3 x 3 key: `row * 3 + column`, where `column` is the
 * tertile of `x` (low to high) and `row` the tertile of `y` (low to high, or high to low when
 * `invertSecond`, so that "bad" corners meet). A county missing either value gets {@link NO_CLASS}.
 */
export function getBivariateClasses(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  xBreaks: readonly number[],
  yBreaks: readonly number[],
  invertSecond: boolean
): Uint32Array {
  const classes = new Uint32Array(x.length);
  for (let row = 0; row < x.length; row++) {
    if (!Number.isFinite(x[row]) || !Number.isFinite(y[row])) {
      classes[row] = NO_CLASS;
      continue;
    }
    const column = Math.min(2, getClassIndex(x[row], xBreaks));
    const secondClass = Math.min(2, getClassIndex(y[row], yBreaks));
    classes[row] = (invertSecond ? 2 - secondClass : secondClass) * 3 + column;
  }
  return classes;
}

/** A histogram redrawn over a wider range so that one more value fits on its axis. */
export type RebinnedHistogram = {
  values: number[];
  domain: [number, number];
};

/**
 * Re-bins the permutation histogram (bins equal in width over `[minimum, maximum]` of the
 * simulated statistic) over a domain that also holds `observed`, so the observed statistic can be
 * drawn on the same axis. Each simulated bin lands in the display bin of its centre; when the
 * observed value is inside the simulated range the bins are returned unchanged.
 */
export function rebinHistogram(
  counts: ArrayLike<number>,
  minimum: number,
  maximum: number,
  observed: number,
  displayBins = 48
): RebinnedHistogram {
  const span = maximum - minimum;
  const original: number[] = Array.from(counts as ArrayLike<number>);
  if (!(span > 0) || !Number.isFinite(observed)) {
    return {values: original, domain: [minimum, maximum]};
  }
  const padding = 0.04 * Math.max(span, Math.abs(observed - minimum), Math.abs(observed - maximum));
  const low = Math.min(minimum, observed - padding);
  const high = Math.max(maximum, observed + padding);
  if (low === minimum && high === maximum) return {values: original, domain: [minimum, maximum]};
  const binWidth = span / original.length;
  const values = new Array<number>(displayBins).fill(0);
  original.forEach((count, bin) => {
    const center = minimum + (bin + 0.5) * binWidth;
    const target = Math.min(
      displayBins - 1,
      Math.max(0, Math.floor(((center - low) / (high - low)) * displayBins))
    );
    values[target] += count;
  });
  return {values, domain: [low, high]};
}

/**
 * Ratio of true ground distance at `latitude` to a planar metre of the map frame whose scale is
 * true at `originLatitude`: `cos(latitude) / cos(originLatitude)`. The planar frame is Web
 * Mercator scaled to be correct at its origin, so a planar kilometre is a shorter ground
 * kilometre north of the origin and a longer one south of it.
 */
export function getGroundDistanceRatio(latitude: number, originLatitude: number): number {
  return Math.cos((latitude * Math.PI) / 180) / Math.cos((originLatitude * Math.PI) / 180);
}

/** Value of `value` among the ascending `sorted` as a 0-1 rank (share at or below). */
export function getRank(sorted: ArrayLike<number>, value: number): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return sorted.length ? low / sorted.length : Number.NaN;
}

/** A p-value of at most two significant digits, for a pseudo p-value or its floor (`0.001`). */
export function formatPseudoPValue(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  return String(Number(value.toPrecision(2)));
}
