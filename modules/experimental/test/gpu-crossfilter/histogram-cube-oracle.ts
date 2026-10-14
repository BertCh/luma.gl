// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU oracle for the Falcon-style histogram cubes of `GPUCrossfilter` (Moritz, Howe, Heer, CHI
 * 2019). Test-only: it defines the cube semantics the GPU kernels must reproduce, using exact
 * integer arithmetic and binning on values that float32 represents exactly.
 */

/** Equal-width inclusive domain or irregular edges, plus the bin count. */
export type OracleBinning = {
  binCount: number;
  domain?: readonly [number, number];
  edges?: readonly number[];
};

/** Bin of `value`, or `undefined` when the histogram would reject it (`GPUHistogram` rules). */
export function binValue(value: number, binning: OracleBinning): number | undefined {
  if (!Number.isFinite(value)) return undefined;
  if (binning.edges) {
    const edges = binning.edges;
    if (!(value >= edges[0] && value <= edges[binning.binCount])) return undefined;
    let lower = 0;
    let upper = binning.binCount;
    while (lower < upper) {
      const middle = lower + Math.floor((upper - lower) / 2);
      if (value < edges[middle + 1]) upper = middle;
      else lower = middle + 1;
    }
    return Math.min(lower, binning.binCount - 1);
  }
  const [minimum, maximum] = binning.domain!;
  if (!(maximum >= minimum) || value < minimum || value > maximum) return undefined;
  if (maximum === minimum) return 0;
  if (value === maximum) return binning.binCount - 1;
  return Math.min(
    Math.floor(((value - minimum) / (maximum - minimum)) * binning.binCount),
    binning.binCount - 1
  );
}

/** Exact masked histogram: the path the cube must reproduce. */
export function computeExactHistogram(
  values: readonly number[],
  accepted: readonly boolean[],
  binning: OracleBinning
): number[] {
  const counts = new Array<number>(binning.binCount).fill(0);
  values.forEach((value, row) => {
    if (!accepted[row]) return;
    const bin = binValue(value, binning);
    if (bin !== undefined) counts[bin]++;
  });
  return counts;
}

/**
 * Prefix-summed cube `cube[a][b]`: rows in `accepted` whose active value has bin `<= a` and whose
 * linked value has bin `b`. `accepted` excludes the active dimension (and the linked view's own).
 */
export function buildCubeOracle(
  activeValues: readonly number[],
  linkedValues: readonly number[],
  accepted: readonly boolean[],
  activeBinning: OracleBinning,
  linkedBinning: OracleBinning
): number[][] {
  const cube = Array.from({length: activeBinning.binCount}, () =>
    new Array<number>(linkedBinning.binCount).fill(0)
  );
  activeValues.forEach((activeValue, row) => {
    if (!accepted[row]) return;
    const activeBin = binValue(activeValue, activeBinning);
    const linkedBin = binValue(linkedValues[row], linkedBinning);
    if (activeBin !== undefined && linkedBin !== undefined) cube[activeBin][linkedBin]++;
  });
  for (let activeBin = 1; activeBin < activeBinning.binCount; activeBin++) {
    for (let linkedBin = 0; linkedBin < linkedBinning.binCount; linkedBin++) {
      cube[activeBin][linkedBin] += cube[activeBin - 1][linkedBin];
    }
  }
  return cube;
}

/** Bin range `[first, last]` an inclusive brush snaps to, or `undefined` when it selects nothing. */
export function snapBrush(
  brush: readonly [number, number],
  binning: OracleBinning
): [number, number] | undefined {
  const minimum = binning.edges ? binning.edges[0] : binning.domain![0];
  const maximum = binning.edges ? binning.edges[binning.binCount] : binning.domain![1];
  if (!(maximum >= minimum) || brush[1] < minimum || brush[0] > maximum) return undefined;
  const first = brush[0] > minimum ? binValue(brush[0], binning) : 0;
  const last = brush[1] < maximum ? binValue(brush[1], binning) : binning.binCount - 1;
  if (first === undefined || last === undefined || first > last) return undefined;
  return [first, last];
}

/** Linked histogram read from a cube for a snapped brush: `C[last] - C[first - 1]`. */
export function readCubeOracle(
  cube: readonly (readonly number[])[],
  brush: readonly [number, number],
  activeBinning: OracleBinning
): number[] {
  const linkedBinCount = cube[0].length;
  const snapped = snapBrush(brush, activeBinning);
  if (!snapped) return new Array<number>(linkedBinCount).fill(0);
  const [first, last] = snapped;
  return Array.from(
    {length: linkedBinCount},
    (_, bin) => cube[last][bin] - (first > 0 ? cube[first - 1][bin] : 0)
  );
}
