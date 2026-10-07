// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getShuffledOrder} from '../../engine/draw-order';
import {
  getDominanceTier,
  GROUP_COUNT,
  NO_GROUP_CLASS,
  SEGREGATION_GROUPS
} from './segregation.style';

/**
 * CPU side of the segregation scene: dominance classes, the aspatial dissimilarity of any unit
 * table (community areas), the seeded row shuffle and the members of a radius. All of it is small
 * and exact; the GPU contributors compute the spatial indices. Pure functions: no luma.gl.
 */

/** Aspatial multigroup and per-group dissimilarity of a unit-by-group count table. */
export type AspatialIndices = {
  /** Multigroup dissimilarity D (Reardon and Firebaugh). */
  multigroup: number;
  /** Dissimilarity of each group against all others. */
  perGroup: Float64Array;
  /** Citywide share of each group. */
  shares: Float64Array;
  /** Total population. */
  total: number;
};

/**
 * The aspatial indices of a count table `counts[unit * GROUP_COUNT + group]`, the formulas of
 * `GPUSegregation` at its aspatial scale in double precision: `D = sum_i sum_m t_i |p_im - P_m| /
 * (2 T I)` with `I = sum_m P_m (1 - P_m)`, and `D_g = sum_i t_i |p_ig - P_g| / (2 T P_g (1 - P_g))`.
 */
export function getAspatialIndices(counts: ArrayLike<number>, unitCount: number): AspatialIndices {
  const groupTotals = new Float64Array(GROUP_COUNT);
  let total = 0;
  for (let unit = 0; unit < unitCount; unit++) {
    for (let group = 0; group < GROUP_COUNT; group++) {
      const count = counts[unit * GROUP_COUNT + group];
      groupTotals[group] += count;
      total += count;
    }
  }
  const shares = groupTotals.map(count => (total > 0 ? count / total : 0));
  const interaction = shares.reduce((sum, share) => sum + share * (1 - share), 0);
  const perGroup = new Float64Array(GROUP_COUNT);
  let multigroup = 0;
  for (let unit = 0; unit < unitCount; unit++) {
    let unitTotal = 0;
    for (let group = 0; group < GROUP_COUNT; group++)
      unitTotal += counts[unit * GROUP_COUNT + group];
    if (unitTotal <= 0) continue;
    for (let group = 0; group < GROUP_COUNT; group++) {
      const gap = Math.abs(counts[unit * GROUP_COUNT + group] / unitTotal - shares[group]);
      multigroup += (unitTotal * gap) / (2 * total * interaction);
      const spread = 2 * total * shares[group] * (1 - shares[group]);
      if (spread > 0) perGroup[group] += (unitTotal * gap) / spread;
    }
  }
  return {multigroup: interaction > 0 ? multigroup : 0, perGroup, shares, total};
}

/** Sums the rows of a count table into `unitCount` coarser units (`unitOf[row]` is 0-based). */
export function sumRowsByUnit(
  counts: ArrayLike<number>,
  rowCount: number,
  unitOf: ArrayLike<number>,
  unitCount: number
): Float64Array {
  const sums = new Float64Array(unitCount * GROUP_COUNT);
  for (let row = 0; row < rowCount; row++) {
    const unit = unitOf[row];
    if (unit < 0 || unit >= unitCount) continue;
    for (let group = 0; group < GROUP_COUNT; group++) {
      sums[unit * GROUP_COUNT + group] += counts[row * GROUP_COUNT + group];
    }
  }
  return sums;
}

/** The largest group of a unit, its share of the unit and the unit total. */
export type LargestGroup = {group: number; share: number; total: number};

/** The largest group of row `row` of a count table, or `null` for a row without residents. */
export function getLargestGroup(counts: ArrayLike<number>, row: number): LargestGroup | null {
  let total = 0;
  let best = 0;
  for (let group = 0; group < GROUP_COUNT; group++) {
    const count = counts[row * GROUP_COUNT + group];
    total += count;
    if (count > counts[row * GROUP_COUNT + best]) best = group;
  }
  return total > 0 ? {group: best, share: counts[row * GROUP_COUNT + best] / total, total} : null;
}

/**
 * Class of the largest-group map, `registryGroup * 3 + tier`, or the no-data sentinel for a tract
 * without residents.
 */
export function getDominanceClasses(counts: ArrayLike<number>, rowCount: number): Uint32Array {
  const classes = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const largest = getLargestGroup(counts, row);
    classes[row] = largest
      ? SEGREGATION_GROUPS[largest.group].registryIndex * 3 + getDominanceTier(largest.share)
      : NO_GROUP_CLASS;
  }
  return classes;
}

/**
 * Tracts per cell of the dominance key: `cells[class]` with class `registryGroup * 3 + tier`
 * (legend rows are the registry order), plus the number of tracts without residents.
 */
export function countDominanceCells(classes: ArrayLike<number>): {cells: number[]; empty: number} {
  const cells = new Array<number>(GROUP_COUNT * 3).fill(0);
  let empty = 0;
  for (let row = 0; row < classes.length; row++) {
    if (classes[row] === NO_GROUP_CLASS) empty++;
    else cells[classes[row]]++;
  }
  return {cells, empty};
}

/**
 * Rows of a count table with every row of the populated units moved to another populated unit by
 * a seeded permutation (whole rows, so every unit keeps its own composition and total; units
 * without residents stay where they are). The same seed gives the same shuffle.
 */
export function shuffleRows(
  counts: ArrayLike<number>,
  rowCount: number,
  seed: number
): Float32Array {
  const populated: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    let total = 0;
    for (let group = 0; group < GROUP_COUNT; group++) total += counts[row * GROUP_COUNT + group];
    if (total > 0) populated.push(row);
  }
  const order = getShuffledOrder(populated.length, seed);
  const shuffled = Float32Array.from(counts as ArrayLike<number>);
  populated.forEach((target, position) => {
    const source = populated[order[position]];
    for (let group = 0; group < GROUP_COUNT; group++) {
      shuffled[target * GROUP_COUNT + group] = counts[source * GROUP_COUNT + group];
    }
  });
  return shuffled;
}

/**
 * Rows whose point lies within `radiusMeters` of the point of `row` (the row itself excluded):
 * the members of its distance band, as `GPUNeighborSearch` in radius mode lists them.
 * `points` are planar metres, `x, y` interleaved.
 */
export function getRadiusMembers(
  points: ArrayLike<number>,
  rowCount: number,
  row: number,
  radiusMeters: number
): number[] {
  const members: number[] = [];
  const centerX = points[row * 2];
  const centerY = points[row * 2 + 1];
  const limit = radiusMeters * radiusMeters;
  for (let other = 0; other < rowCount; other++) {
    if (other === row) continue;
    const dx = points[other * 2] - centerX;
    const dy = points[other * 2 + 1] - centerY;
    if (dx * dx + dy * dy <= limit) members.push(other);
  }
  return members;
}

/** Population-weighted mean of a per-row value over `rows` (rows with a non-finite value are skipped). */
export function getWeightedMean(
  values: ArrayLike<number>,
  weights: ArrayLike<number>,
  rows: readonly number[]
): number {
  let sum = 0;
  let weightSum = 0;
  for (const row of rows) {
    if (!Number.isFinite(values[row]) || !(weights[row] > 0)) continue;
    sum += values[row] * weights[row];
    weightSum += weights[row];
  }
  return weightSum > 0 ? sum / weightSum : Number.NaN;
}

/** Share of the values (finite ones only) that are at most `value`, as a 0-1 percentile. */
export function getPercentile(sortedValues: ArrayLike<number>, value: number): number {
  const count = sortedValues.length;
  if (count === 0 || !Number.isFinite(value)) return Number.NaN;
  let low = 0;
  let high = count;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sortedValues[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return low / count;
}

/** Counts of finite values per class of `breaks` (class `k` = number of breaks at or below the value). */
export function countClasses(values: ArrayLike<number>, breaks: readonly number[]): number[] {
  const counts = new Array<number>(breaks.length + 1).fill(0);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    let classIndex = 0;
    while (classIndex < breaks.length && value >= breaks[classIndex]) classIndex++;
    counts[classIndex]++;
  }
  return counts;
}

/** The class index of one value, or `-1` for a non-finite value. */
export function getClassIndex(value: number, breaks: readonly number[]): number {
  if (!Number.isFinite(value)) return -1;
  let classIndex = 0;
  while (classIndex < breaks.length && value >= breaks[classIndex]) classIndex++;
  return classIndex;
}
