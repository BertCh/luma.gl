// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OraclePolygonFeature} from '../spatial-join/spatial-join-oracle';

/** Row value that marks a point without a feature. */
export const NO_FEATURE = 0xffffffff;

/** Inputs of {@link computeZonalStatistics}. Values are float32-representable numbers. */
export type ZonalOracleInput = {
  /** Feature row per point; values `>= featureCount` are unassigned. */
  featureRows: number[];
  featureCount: number;
  values?: number[];
  weights?: number[];
  /** Per-feature areas used for densities. */
  areas?: number[];
};

/** Per-feature results computed in float64. */
export type ZonalOracleResult = {
  counts: number[];
  valueCounts: number[];
  sums: number[];
  /** Sum of the absolute contributions, the scale of float32 summation error. */
  absoluteSums: number[];
  weightSums: number[];
  means: number[];
  minima: number[];
  maxima: number[];
  densities: number[];
};

/** Returns whether `value` is a finite number. */
const isFiniteNumber = (value: number): boolean => Number.isFinite(value);

/** CPU reference for every statistic of `GPUZonalStatistics`, following its documented semantics. */
export function computeZonalStatistics(input: ZonalOracleInput): ZonalOracleResult {
  const {featureRows, featureCount, values, weights, areas} = input;
  const zeros = () => new Array<number>(featureCount).fill(0);
  const result: ZonalOracleResult = {
    counts: zeros(),
    valueCounts: zeros(),
    sums: zeros(),
    absoluteSums: zeros(),
    weightSums: zeros(),
    means: new Array<number>(featureCount).fill(Number.NaN),
    minima: new Array<number>(featureCount).fill(Number.NaN),
    maxima: new Array<number>(featureCount).fill(Number.NaN),
    densities: new Array<number>(featureCount).fill(Number.NaN)
  };
  for (const [pointIndex, row] of featureRows.entries()) {
    if (row >= featureCount) {
      continue;
    }
    result.counts[row]++;
    if (!values) {
      continue;
    }
    const value = values[pointIndex];
    const weight = weights ? weights[pointIndex] : 1;
    if (!isFiniteNumber(value) || !isFiniteNumber(weight)) {
      continue;
    }
    const product = Math.fround(weight * value);
    result.valueCounts[row]++;
    if (isFiniteNumber(product)) {
      result.sums[row] += product;
      result.absoluteSums[row] += Math.abs(product);
    }
    if (weights) {
      result.weightSums[row] += weight;
    }
    result.minima[row] = Number.isNaN(result.minima[row])
      ? value
      : Math.min(result.minima[row], value);
    result.maxima[row] = Number.isNaN(result.maxima[row])
      ? value
      : Math.max(result.maxima[row], value);
  }
  for (let row = 0; row < featureCount; row++) {
    if (weights) {
      result.means[row] =
        result.weightSums[row] === 0 ? Number.NaN : result.sums[row] / result.weightSums[row];
    } else {
      result.means[row] =
        result.valueCounts[row] === 0 ? Number.NaN : result.sums[row] / result.valueCounts[row];
    }
    const area = areas?.[row];
    if (area !== undefined && isFiniteNumber(area) && area > 0) {
      result.densities[row] = result.counts[row] / area;
    }
  }
  return result;
}

/**
 * Float64 area of a polygon feature: `|shell| - sum |holes|` per polygon, summed. Each ring is
 * measured relative to its first finite vertex; non-finite vertices are skipped.
 */
export function computePolygonFeatureArea(feature: OraclePolygonFeature): number {
  let area = 0;
  for (const polygon of feature) {
    for (const [ringIndex, ring] of polygon.entries()) {
      const finite = ring.filter(([x, y]) => isFiniteNumber(x) && isFiniteNumber(y));
      let twiceArea = 0;
      for (let index = 1; index < finite.length; index++) {
        const [originX, originY] = finite[0];
        const [previousX, previousY] = finite[index - 1];
        const [currentX, currentY] = finite[index];
        twiceArea +=
          (previousX - originX) * (currentY - originY) -
          (previousY - originY) * (currentX - originX);
      }
      const ringArea = Math.abs(twiceArea) / 2;
      area += ringIndex === 0 ? ringArea : -ringArea;
    }
  }
  return area;
}

/** `[min, max]` of `statistic` over features with positive counts and finite statistics, else `[0, 0]`. */
export function computeExtent(statistic: number[], counts: number[]): [number, number] {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const [row, value] of statistic.entries()) {
    if (counts[row] > 0 && isFiniteNumber(value)) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  return minimum > maximum ? [0, 0] : [minimum, maximum];
}
