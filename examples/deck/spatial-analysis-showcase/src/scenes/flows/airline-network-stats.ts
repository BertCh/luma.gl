// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CONTINENT_NAMES} from './b11-geography';
import {BETWEEN_GROUPS_INDEX, OTHER_GROUP_INDEX} from './airline-network-palette';

/**
 * CPU bookkeeping for the airline-network scene. The graph algorithms run on the GPU; these
 * functions only rank, bin and label the small per-airport summaries read back from it.
 */

/** Airport rows sorted by `values`, largest first (ties keep the lower row first). */
export function sortRowsDescending(values: ArrayLike<number>): Uint32Array {
  const rows = Uint32Array.from({length: values.length}, (_, row) => row);
  return rows.sort((a, b) => values[b] - values[a] || a - b);
}

/** Average ranks (1 is the smallest) with ties sharing their mean rank. */
function getAverageRanks(values: ArrayLike<number>): Float64Array {
  const order = Array.from({length: values.length}, (_, row) => row).sort(
    (a, b) => values[a] - values[b]
  );
  const ranks = new Float64Array(values.length);
  let start = 0;
  while (start < order.length) {
    let end = start;
    while (end + 1 < order.length && values[order[end + 1]] === values[order[start]]) end++;
    const rank = (start + end) / 2 + 1;
    for (let index = start; index <= end; index++) ranks[order[index]] = rank;
    start = end + 1;
  }
  return ranks;
}

/** Pearson correlation of two equally long columns, `NaN` when either has no spread. */
export function getPearsonCorrelation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const count = a.length;
  let meanA = 0;
  let meanB = 0;
  for (let index = 0; index < count; index++) {
    meanA += a[index];
    meanB += b[index];
  }
  meanA /= count;
  meanB /= count;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let index = 0; index < count; index++) {
    const deltaA = a[index] - meanA;
    const deltaB = b[index] - meanB;
    covariance += deltaA * deltaB;
    varianceA += deltaA * deltaA;
    varianceB += deltaB * deltaB;
  }
  return covariance / Math.sqrt(varianceA * varianceB);
}

/** Spearman rank correlation (Pearson on average ranks). */
export function getSpearmanCorrelation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return getPearsonCorrelation(getAverageRanks(a), getAverageRanks(b));
}

/** Counts of `log10(value)` in equal bins from `0` (value 1) to `log10(maximum)`. */
export function getLogHistogram(
  values: ArrayLike<number>,
  binCount: number
): {counts: number[]; maximumLog: number} {
  let maximum = 1;
  for (let index = 0; index < values.length; index++) maximum = Math.max(maximum, values[index]);
  const maximumLog = Math.max(Math.log10(maximum), 1e-6);
  const counts = new Array<number>(binCount).fill(0);
  for (let index = 0; index < values.length; index++) {
    if (values[index] < 1) continue;
    const bin = Math.min(
      binCount - 1,
      Math.floor((Math.log10(values[index]) / maximumLog) * binCount)
    );
    counts[bin]++;
  }
  return {counts, maximumLog};
}

/** One group (a community or a continent) of airports. */
export type GroupSummary = {
  label: number;
  size: number;
  /** Row of the airport with the highest PageRank in the group. */
  topAirport: number;
  /** Index into `CONTINENT_NAMES` holding most of the group's airports. */
  dominantContinent: number;
  /** Share of the group's airports on that continent. */
  dominantShare: number;
};

/** Groups, their sizes and how well they line up with continents. */
export type GroupAnalysis = {
  /** Groups largest first. */
  groups: GroupSummary[];
  /** Palette index of every airport: the six largest groups are 0 to 5, the rest 6. */
  colorIndex: Uint32Array;
  /** Share of airports that sit on their group's dominant continent. */
  continentPurity: number;
};

/** Summarises a partition: sizes, hubs, continent purity and the palette index of every airport. */
export function analyzeGroups(
  labels: ArrayLike<number>,
  pageRank: ArrayLike<number>,
  continent: ArrayLike<number>
): GroupAnalysis {
  const continentCount = CONTINENT_NAMES.length;
  const byLabel = new Map<number, {size: number; top: number; byContinent: Uint32Array}>();
  for (let row = 0; row < labels.length; row++) {
    let entry = byLabel.get(labels[row]);
    if (!entry) {
      entry = {size: 0, top: row, byContinent: new Uint32Array(continentCount)};
      byLabel.set(labels[row], entry);
    }
    entry.size++;
    entry.byContinent[continent[row]]++;
    if (pageRank[row] > pageRank[entry.top]) entry.top = row;
  }
  const groups: GroupSummary[] = [];
  let pure = 0;
  for (const [label, entry] of byLabel) {
    let best = 0;
    for (let index = 1; index < continentCount; index++) {
      if (entry.byContinent[index] > entry.byContinent[best]) best = index;
    }
    pure += entry.byContinent[best];
    groups.push({
      label,
      size: entry.size,
      topAirport: entry.top,
      dominantContinent: best,
      dominantShare: entry.byContinent[best] / entry.size
    });
  }
  groups.sort((a, b) => b.size - a.size || a.label - b.label);
  const paletteOfLabel = new Map<number, number>();
  groups.forEach((group, rank) => {
    paletteOfLabel.set(group.label, Math.min(rank, OTHER_GROUP_INDEX));
  });
  const colorIndex = new Uint32Array(labels.length);
  for (let row = 0; row < labels.length; row++) {
    colorIndex[row] = paletteOfLabel.get(labels[row]) ?? OTHER_GROUP_INDEX;
  }
  return {groups, colorIndex, continentPurity: labels.length > 0 ? pure / labels.length : 0};
}

/** Palette index of a route: its group when both ends share one, else the "between" index. */
export function getEdgeIndex(
  labels: ArrayLike<number>,
  colorIndex: ArrayLike<number>,
  source: number,
  target: number
): number {
  return labels[source] === labels[target] ? colorIndex[source] : BETWEEN_GROUPS_INDEX;
}
