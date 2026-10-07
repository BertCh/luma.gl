// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {haversineDistance} from '../../cartography/reference-geometry';
import type {LngLat} from '../../cartography/types';

/**
 * The CPU twin of `GPUSimilarLocations`, used for what the GPU does not return: the standardised
 * values of the profile chart, the contribution of each attribute to a distance, the baseline
 * ranking the "persist" readout compares against, and a cross-check of the GPU matches. Pure
 * TypeScript, `Float64`: small float32 differences can only swap rows whose distances are equal
 * to about seven digits.
 */

/** How the attribute columns are put on a common scale (the contributor's `standardization`). */
export type Standardization = 'zscore' | 'rank';

/** Standardised columns, `columns[attribute][row]`; `NaN` for rows that cannot be ranked. */
export type StandardizedColumns = readonly Float64Array[];

/** One attribute's share of the distance between a row and the reference. */
export type AttributeContribution = {
  attribute: number;
  /** Standardised value of the row minus the reference's. */
  delta: number;
  /** `weight * delta^2`: the attribute's term of the squared distance. */
  contribution: number;
};

/** Ranking result of {@link rankRows}. */
export type CpuRanking = {
  /** Weighted distance per row; `NaN` when the row is not ranked. */
  distances: Float64Array;
  /** Ranked rows, best first. */
  order: Int32Array;
};

/** True when every attribute of `row` is finite (the contributor's validity rule). */
export function isRowValid(columns: readonly ArrayLike<number>[], row: number): boolean {
  for (const column of columns) if (!Number.isFinite(column[row])) return false;
  return true;
}

/**
 * Standardises each column over the valid rows: the population z-score `(x - mean) / sd`, or the
 * average-tie percentile `(rank - 1) / (n - 1)` of the contributor's `'rank'` mode.
 */
export function standardizeColumns(
  columns: readonly ArrayLike<number>[],
  mode: Standardization,
  rowCount: number
): Float64Array[] {
  const validRows: number[] = [];
  for (let row = 0; row < rowCount; row++) if (isRowValid(columns, row)) validRows.push(row);
  return columns.map(column => {
    const out = new Float64Array(rowCount).fill(Number.NaN);
    const count = validRows.length;
    if (mode === 'zscore') {
      let sum = 0;
      for (const row of validRows) sum += column[row];
      const mean = sum / count;
      let squares = 0;
      for (const row of validRows) squares += (column[row] - mean) ** 2;
      const deviation = Math.sqrt(squares / count);
      for (const row of validRows) out[row] = deviation > 0 ? (column[row] - mean) / deviation : 0;
      return out;
    }
    const sorted = Float64Array.from(validRows, row => column[row]).sort();
    for (const row of validRows) {
      const value = column[row];
      let low = 0;
      let high = count;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (sorted[middle] < value) low = middle + 1;
        else high = middle;
      }
      let end = low;
      while (end < count && sorted[end] === value) end++;
      // Average-tie rank, zero-based: rows below plus half of the tied rows.
      out[row] = count > 1 ? (low + (end - low - 1) / 2) / (count - 1) : 0;
    }
    return out;
  });
}

/** The reference vector: the mean of the standardised rows in `references` (one row: that row). */
export function getReferenceProfile(
  standardized: StandardizedColumns,
  references: readonly number[]
): Float64Array {
  return Float64Array.from(standardized, column => {
    let sum = 0;
    for (const row of references) sum += column[row];
    return references.length ? sum / references.length : Number.NaN;
  });
}

/** Options of {@link rankRows}. */
export type RankRowsOptions = {
  direction: 'most' | 'least';
  /** Rows left out of the ranking (the references when they are excluded). */
  excluded: ReadonlySet<number>;
};

/**
 * Ranks rows by `sqrt(sum_k w_k (z_k - r_k)^2)`, ties by the lowest row, as the contributor's
 * stable sort does. `least` lists the farthest first.
 */
export function rankRows(
  standardized: StandardizedColumns,
  profile: ArrayLike<number>,
  weights: ArrayLike<number>,
  options: RankRowsOptions
): CpuRanking {
  const rowCount = standardized[0]?.length ?? 0;
  const distances = new Float64Array(rowCount).fill(Number.NaN);
  const rows: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    if (options.excluded.has(row) || Number.isNaN(standardized[0][row])) continue;
    let squares = 0;
    for (let attribute = 0; attribute < standardized.length; attribute++) {
      squares += weights[attribute] * (standardized[attribute][row] - profile[attribute]) ** 2;
    }
    if (!Number.isFinite(squares)) continue;
    distances[row] = Math.sqrt(squares);
    rows.push(row);
  }
  const sign = options.direction === 'most' ? 1 : -1;
  rows.sort((a, b) => sign * (distances[a] - distances[b]) || a - b);
  return {distances, order: Int32Array.from(rows)};
}

/**
 * The terms of the squared distance of `row`, largest first. Attributes with zero weight are
 * left out.
 */
export function getContributions(
  standardized: StandardizedColumns,
  profile: ArrayLike<number>,
  weights: ArrayLike<number>,
  row: number
): AttributeContribution[] {
  const terms: AttributeContribution[] = [];
  for (let attribute = 0; attribute < standardized.length; attribute++) {
    if (!(weights[attribute] > 0)) continue;
    const delta = standardized[attribute][row] - profile[attribute];
    terms.push({attribute, delta, contribution: weights[attribute] * delta * delta});
  }
  return terms.sort((a, b) => b.contribution - a.contribution);
}

/** Great-circle distance in kilometres. */
export function getKilometers(a: LngLat, b: LngLat): number {
  return haversineDistance(a, b) / 1000;
}

/** Median of a list (the mean of the two middle values for an even length); `NaN` when empty. */
export function getMedian(values: readonly number[]): number {
  if (!values.length) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * The `count` rows nearest to any of `references` by great-circle distance between label points,
 * nearest first. Rows in `references` are skipped.
 */
export function getNearestRows(
  points: readonly (readonly [number, number])[],
  references: readonly number[],
  count: number
): {row: number; kilometers: number}[] {
  const referenceSet = new Set(references);
  const found: {row: number; kilometers: number}[] = [];
  for (let row = 0; row < points.length; row++) {
    if (referenceSet.has(row) || !Number.isFinite(points[row][0])) continue;
    let nearest = Number.POSITIVE_INFINITY;
    for (const reference of references) {
      nearest = Math.min(
        nearest,
        getKilometers(points[reference] as LngLat, points[row] as LngLat)
      );
    }
    found.push({row, kilometers: nearest});
  }
  found.sort((a, b) => a.kilometers - b.kilometers || a.row - b.row);
  return found.slice(0, count);
}

/** Number of values of `a` that are also in `b`. */
export function countShared(a: readonly number[], b: readonly number[]): number {
  const set = new Set(b);
  let shared = 0;
  for (const value of a) if (set.has(value)) shared++;
  return shared;
}

/**
 * The most correlated pair among the attributes that carry weight (Pearson correlation of the
 * z-scores over the valid rows), or `null` with fewer than two weighted attributes. Correlated
 * attributes count the same evidence twice in a Euclidean distance.
 */
export function getStrongestCorrelation(
  zScores: StandardizedColumns,
  weights: ArrayLike<number>
): {a: number; b: number; r: number} | null {
  const weighted = zScores.map((_, index) => index).filter(index => weights[index] > 0);
  let best: {a: number; b: number; r: number} | null = null;
  const rowCount = zScores[0]?.length ?? 0;
  for (let i = 0; i < weighted.length; i++) {
    for (let j = i + 1; j < weighted.length; j++) {
      let sum = 0;
      let count = 0;
      for (let row = 0; row < rowCount; row++) {
        const product = zScores[weighted[i]][row] * zScores[weighted[j]][row];
        if (Number.isFinite(product)) {
          sum += product;
          count++;
        }
      }
      // z-scores have unit variance, so the mean product is the Pearson correlation.
      const r = count ? sum / count : 0;
      if (!best || Math.abs(r) > Math.abs(best.r)) best = {a: weighted[i], b: weighted[j], r};
    }
  }
  return best;
}
