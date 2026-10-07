// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CONTINENT_NAMES} from './b11-geography';
import {matchByOverlap} from '../../cartography/stable-hues';
import {BETWEEN_GROUPS_INDEX, GROUP_HUE_COUNT, OTHER_GROUP_INDEX} from './airline-network-palette';

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
  /** Palette slot of the group: a hue (`0` to `6`) or {@link OTHER_GROUP_INDEX}. */
  slot: number;
};

/** A partition of the airports with its palette slots and its edge categories. */
export type Coloring = {
  /** Groups largest first. */
  groups: GroupSummary[];
  /** Palette slot of every airport. */
  nodeSlots: Uint32Array;
  /** Share of airports that sit on their group's dominant continent. */
  continentPurity: number;
  /** Group label of every airport (the partition itself). */
  labels: Uint32Array;
  /** Palette slot of every route: its group when both ends share one, else the between slot. */
  edgeSlots: Uint32Array;
  /** Routes whose two airports are in different groups. */
  betweenCount: number;
};

/**
 * Summarises a partition and gives it palette slots. Identity-stable hues (no colouring by rank):
 * `kind: 'continent'` gives continent `i` hue `i`; `kind: 'community'` gives each of the seven
 * largest communities the hue of the continent it overlaps most (`matchByOverlap`, one hue per
 * continent), so the community that is mostly Europe wears Europe's hue. A community that matches
 * no continent takes the lowest unused hue; communities beyond the seven largest are grey.
 */
export function colorPartition(
  kind: 'continent' | 'community',
  labels: Uint32Array,
  pageRank: ArrayLike<number>,
  continent: ArrayLike<number>,
  source: ArrayLike<number>,
  target: ArrayLike<number>
): Coloring {
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
      dominantShare: entry.byContinent[best] / entry.size,
      slot: OTHER_GROUP_INDEX
    });
  }
  groups.sort((a, b) => b.size - a.size || a.label - b.label);

  if (kind === 'continent') {
    for (const group of groups) group.slot = Math.min(group.label, OTHER_GROUP_INDEX);
  } else {
    // Only the largest communities own a hue; the others are compacted out (label -1).
    const hued = groups.slice(0, GROUP_HUE_COUNT);
    const compactOfLabel = new Map<number, number>();
    hued.forEach((group, index) => compactOfLabel.set(group.label, index));
    const compactLabels = Int32Array.from(labels, label => compactOfLabel.get(label) ?? -1);
    const slots = matchByOverlap(continent, compactLabels, GROUP_HUE_COUNT);
    hued.forEach((group, index) => {
      group.slot = slots[index] ?? OTHER_GROUP_INDEX;
    });
  }

  const slotOfLabel = new Map<number, number>();
  for (const group of groups) slotOfLabel.set(group.label, group.slot);
  const nodeSlots = new Uint32Array(labels.length);
  for (let row = 0; row < labels.length; row++) {
    nodeSlots[row] = slotOfLabel.get(labels[row]) ?? OTHER_GROUP_INDEX;
  }
  const edgeSlots = new Uint32Array(source.length);
  let betweenCount = 0;
  for (let edge = 0; edge < source.length; edge++) {
    const a = source[edge];
    const b = target[edge];
    if (labels[a] === labels[b]) {
      edgeSlots[edge] = nodeSlots[a];
    } else {
      edgeSlots[edge] = BETWEEN_GROUPS_INDEX;
      betweenCount++;
    }
  }
  return {
    groups,
    nodeSlots,
    labels,
    edgeSlots,
    betweenCount,
    continentPurity: labels.length > 0 ? pure / labels.length : 0
  };
}

/**
 * Between-group routes per airport (how many of its routes leave its group), the size metric of
 * the bridges step and the ranking of the bridge labels.
 */
export function getBridgeCounts(
  labels: ArrayLike<number>,
  source: ArrayLike<number>,
  target: ArrayLike<number>
): Float32Array {
  const counts = new Float32Array(labels.length);
  for (let edge = 0; edge < source.length; edge++) {
    if (labels[source[edge]] !== labels[target[edge]]) {
      counts[source[edge]]++;
      counts[target[edge]]++;
    }
  }
  return counts;
}

/** Share of routes that touch at least one of the `top` airports (given as a set of rows). */
export function getTouchShare(
  top: ReadonlySet<number>,
  source: ArrayLike<number>,
  target: ArrayLike<number>
): number {
  let touching = 0;
  for (let edge = 0; edge < source.length; edge++) {
    if (top.has(source[edge]) || top.has(target[edge])) touching++;
  }
  return source.length > 0 ? touching / source.length : 0;
}
