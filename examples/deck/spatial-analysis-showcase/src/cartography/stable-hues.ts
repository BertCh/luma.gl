// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Stable categorical identity: when k changes (k-means, DBSCAN, regionalisation) a cluster that is
 * "the same place" keeps its hue, and new clusters take unused palette slots. Colours are palette
 * slot indices; the caller maps them through the palette (`OKABE_ITO_LIGHT[slot]`).
 *
 * Both matchers use an optimal (Hungarian) assignment, not a greedy pass, so two clusters that
 * swap places do not trade colours.
 */

/**
 * Minimum-cost assignment of `rowCount` rows to `columnCount >= rowCount` columns (Hungarian
 * algorithm with potentials, O(rows² x columns)). Returns the column of each row.
 */
function solveAssignment(
  cost: readonly (readonly number[])[],
  rowCount: number,
  columnCount: number
) {
  const infinity = Number.POSITIVE_INFINITY;
  const rowPotential = new Float64Array(rowCount + 1);
  const columnPotential = new Float64Array(columnCount + 1);
  const columnRow = new Int32Array(columnCount + 1); // 1-based row matched to each column
  const way = new Int32Array(columnCount + 1);
  for (let row = 1; row <= rowCount; row++) {
    columnRow[0] = row;
    let column0 = 0;
    const minimum = new Float64Array(columnCount + 1).fill(infinity);
    const used = new Uint8Array(columnCount + 1);
    do {
      used[column0] = 1;
      const row0 = columnRow[column0];
      let delta = infinity;
      let column1 = 0;
      for (let column = 1; column <= columnCount; column++) {
        if (used[column]) continue;
        const current = cost[row0 - 1][column - 1] - rowPotential[row0] - columnPotential[column];
        if (current < minimum[column]) {
          minimum[column] = current;
          way[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = column;
        }
      }
      for (let column = 0; column <= columnCount; column++) {
        if (used[column]) {
          rowPotential[columnRow[column]] += delta;
          columnPotential[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (columnRow[column0] !== 0);
    do {
      const column1 = way[column0];
      columnRow[column0] = columnRow[column1];
      column0 = column1;
    } while (column0);
  }
  const rowColumn = new Array<number>(rowCount).fill(-1);
  for (let column = 1; column <= columnCount; column++) {
    if (columnRow[column]) rowColumn[columnRow[column] - 1] = column - 1;
  }
  return rowColumn;
}

/** For each of `count` new items, the matched index of the other side (or `-1`), minimising `cost`. */
function matchRectangular(
  cost: (nextIndex: number, previousIndex: number) => number,
  nextCount: number,
  previousCount: number
): number[] {
  if (nextCount === 0 || previousCount === 0) return new Array<number>(nextCount).fill(-1);
  if (nextCount <= previousCount) {
    const matrix = Array.from({length: nextCount}, (_, next) =>
      Array.from({length: previousCount}, (_, previous) => cost(next, previous))
    );
    return solveAssignment(matrix, nextCount, previousCount);
  }
  const matrix = Array.from({length: previousCount}, (_, previous) =>
    Array.from({length: nextCount}, (_, next) => cost(next, previous))
  );
  const previousToNext = solveAssignment(matrix, previousCount, nextCount);
  const result = new Array<number>(nextCount).fill(-1);
  previousToNext.forEach((next, previous) => {
    if (next >= 0) result[next] = previous;
  });
  return result;
}

/** Gives every unmatched entry the lowest unused palette slot (wrapping when the palette is full). */
function fillFreeSlots(colors: number[], paletteSize: number): number[] {
  const used = new Set(colors.filter(color => color >= 0));
  let cursor = 0;
  let wrapped = 0;
  return colors.map(color => {
    if (color >= 0) return color;
    while (cursor < paletteSize && used.has(cursor)) cursor++;
    if (cursor < paletteSize) {
      used.add(cursor);
      return cursor;
    }
    return wrapped++ % paletteSize;
  });
}

/** A cluster of the previous solution: where it is, and the palette slot it had. */
export type StableHuePrevious = {
  /** Optional caller identity, ignored by the matcher (handy to carry the key through). */
  key?: string;
  /** Cluster centre (any dimension: `[lng, lat]`, feature means). */
  center: readonly number[];
  /** Palette slot the cluster had. */
  color: number;
};

/** Options of {@link matchStableColors}. */
export type StableHueOptions = {
  /**
   * A new cluster farther than this from every previous centre (same units as `center`) is treated
   * as new and takes an unused slot instead of inheriting a distant colour. Default: no limit.
   */
  maximumDistance?: number;
};

/**
 * Colours for a new set of cluster centres that keep the hue of the previous solution: previous
 * and new centres are matched one-to-one by the minimum total squared distance (optimal
 * assignment), a matched cluster inherits the previous slot, and unmatched clusters take unused
 * slots in ascending order. Returns one palette slot per entry of `next`.
 *
 * @param paletteSize Number of slots in the palette (at most 7 distinct hues is the rule).
 *
 * @example
 * ```ts
 * let previous = [];
 * function recolor(centres: number[][]): number[] {
 *   const slots = matchStableColors(previous, centres.map(center => ({center})), 7);
 *   previous = centres.map((center, i) => ({center, color: slots[i]}));
 *   return slots; // palette[slots[i]] colours cluster i; k 4 -> 5 keeps the first four hues
 * }
 * ```
 */
export function matchStableColors(
  previous: readonly StableHuePrevious[],
  next: readonly {center: readonly number[]}[],
  paletteSize: number,
  options: StableHueOptions = {}
): number[] {
  const distance = (nextIndex: number, previousIndex: number) => {
    const a = next[nextIndex].center;
    const b = previous[previousIndex].center;
    let sum = 0;
    for (let i = 0; i < Math.max(a.length, b.length); i++) sum += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2;
    return sum;
  };
  const matches = matchRectangular(distance, next.length, previous.length);
  const limit = options.maximumDistance;
  const colors = matches.map((previousIndex, nextIndex) => {
    if (previousIndex < 0) return -1;
    if (limit !== undefined && distance(nextIndex, previousIndex) > limit * limit) return -1;
    return ((previous[previousIndex].color % paletteSize) + paletteSize) % paletteSize;
  });
  return fillFreeSlots(colors, paletteSize);
}

/**
 * Colours for new cluster LABELS that keep the hue of the previous labelling by maximum overlap:
 * item `i` has label `previousLabels[i]` before and `nextLabels[i]` now (negative labels are noise
 * and ignored), each new label is matched to the previous label sharing the most items (optimal
 * one-to-one assignment), inherits its slot, and unmatched labels take unused slots. Returns one
 * slot per new label id (`0..max(nextLabels)`).
 *
 * @param previousColors Slot of each previous label id (default: `id % paletteSize`).
 *
 * @example
 * ```ts
 * const slots = matchByOverlap(labelsAtK4, labelsAtK5, 7, slotsAtK4);
 * const color = (item: number) => palette[slots[labelsAtK5[item]]];
 * ```
 */
export function matchByOverlap(
  previousLabels: ArrayLike<number>,
  nextLabels: ArrayLike<number>,
  paletteSize: number,
  previousColors?: readonly number[]
): number[] {
  const length = Math.min(previousLabels.length, nextLabels.length);
  let previousCount = 0;
  let nextCount = 0;
  for (let i = 0; i < length; i++) {
    if (previousLabels[i] >= 0) previousCount = Math.max(previousCount, previousLabels[i] + 1);
    if (nextLabels[i] >= 0) nextCount = Math.max(nextCount, nextLabels[i] + 1);
  }
  const overlap = Array.from({length: nextCount}, () => new Float64Array(previousCount));
  for (let i = 0; i < length; i++) {
    if (previousLabels[i] >= 0 && nextLabels[i] >= 0) overlap[nextLabels[i]][previousLabels[i]]++;
  }
  const matches = matchRectangular(
    (nextIndex, previousIndex) => -overlap[nextIndex][previousIndex],
    nextCount,
    previousCount
  );
  const colors = matches.map((previousIndex, nextIndex) => {
    if (previousIndex < 0 || overlap[nextIndex][previousIndex] === 0) return -1;
    const slot = previousColors?.[previousIndex] ?? previousIndex;
    return ((slot % paletteSize) + paletteSize) % paletteSize;
  });
  return fillFreeSlots(colors, paletteSize);
}
