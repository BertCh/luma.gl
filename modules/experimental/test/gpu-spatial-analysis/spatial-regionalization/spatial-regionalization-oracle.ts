// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type RegionalizationCase = {
  name: string;
  rows: number;
  columns: number;
  offsets: number[];
  neighbors: number[];
  values: number[];
  mstEdges: [number, number][];
  mstCost: number;
  components: number;
  /**
   * `pseudoF` is `sklearn.metrics.calinski_harabasz_score(values, labels)` (scikit-learn 1.9.1),
   * written into skater-reference.json by fix/work/C/U6/pseudo_f.py.
   */
  runs: {k: number; floor: number | null; labels: number[]; score: number; pseudoF: number}[];
};

/** Squared Euclidean distance of two attribute rows. */
export function getSquaredDistance(
  values: ArrayLike<number>,
  columns: number,
  first: number,
  second: number
): number {
  let sum = 0;
  for (let column = 0; column < columns; column++) {
    const delta = values[first * columns + column] - values[second * columns + column];
    sum += delta * delta;
  }
  return sum;
}

/**
 * Kruskal minimum spanning forest under the contributor's total order: ascending cost, then
 * ascending edge ID (the lower-row CSR slot). Returns the edge slots in ascending order.
 */
export function computeKruskalForest(
  offsets: number[],
  neighbors: number[],
  getCost: (row: number, neighbor: number) => number
): number[] {
  const rows = offsets.length - 1;
  const edges: {slot: number; row: number; neighbor: number; cost: number}[] = [];
  for (let row = 0; row < rows; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (row < neighbor) {
        edges.push({slot, row, neighbor, cost: Math.fround(getCost(row, neighbor))});
      }
    }
  }
  edges.sort((a, b) => a.cost - b.cost || a.slot - b.slot);
  const parent = Array.from({length: rows}, (_, index) => index);
  const find = (node: number): number => {
    while (parent[node] !== node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  };
  const chosen: number[] = [];
  for (const edge of edges) {
    const first = find(edge.row);
    const second = find(edge.neighbor);
    if (first !== second) {
      parent[Math.max(first, second)] = Math.min(first, second);
      chosen.push(edge.slot);
    }
  }
  return chosen.sort((a, b) => a - b);
}

/** Labels each row with the smallest row index of its component of the given edge set. */
export function computeComponentLabels(rows: number, edges: [number, number][]): number[] {
  const parent = Array.from({length: rows}, (_, index) => index);
  const find = (node: number): number => {
    while (parent[node] !== node) {
      parent[node] = parent[parent[node]];
      node = parent[node];
    }
    return node;
  };
  for (const [first, second] of edges) {
    const a = find(first);
    const b = find(second);
    if (a !== b) {
      parent[Math.max(a, b)] = Math.min(a, b);
    }
  }
  return Array.from({length: rows}, (_, row) => find(row));
}

/** Renumbers labels by first occurrence so partitions compare regardless of label values. */
export function canonicalizePartition(labels: ArrayLike<number>): number[] {
  const map = new Map<number, number>();
  const result: number[] = [];
  for (let index = 0; index < labels.length; index++) {
    if (!map.has(labels[index])) {
      map.set(labels[index], map.size);
    }
    result.push(map.get(labels[index])!);
  }
  return result;
}

/** CPU partition evaluation in f64: within SSD, total SSD, sizes and cross-link fraction. */
export function evaluatePartition(
  values: ArrayLike<number>,
  columns: number,
  labels: ArrayLike<number>,
  offsets?: number[],
  neighbors?: number[]
) {
  const rows = labels.length;
  const groups = new Map<number, number[]>();
  for (let row = 0; row < rows; row++) {
    groups.set(labels[row], [...(groups.get(labels[row]) ?? []), row]);
  }
  const ssd = (members: number[]) => {
    let total = 0;
    for (let column = 0; column < columns; column++) {
      const mean =
        members.reduce((sum, row) => sum + values[row * columns + column], 0) / members.length;
      for (const row of members) {
        total += (values[row * columns + column] - mean) ** 2;
      }
    }
    return total;
  };
  let withinSsd = 0;
  const sizes: number[] = [];
  for (const members of groups.values()) {
    withinSsd += ssd(members);
    sizes.push(members.length);
  }
  const totalSsd = ssd(Array.from({length: rows}, (_, row) => row));
  let cross = 0;
  let total = 0;
  if (offsets && neighbors) {
    for (let row = 0; row < rows; row++) {
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        total++;
        if (labels[neighbors[slot]] !== labels[row]) {
          cross++;
        }
      }
    }
  }
  return {
    regionCount: groups.size,
    withinSsd,
    totalSsd,
    minimumSize: Math.min(...sizes),
    maximumSize: Math.max(...sizes),
    crossLinkFraction: total ? cross / total : 0
  };
}

/**
 * Per-column z-scores with the population standard deviation (divisor `n`), the
 * `sklearn.preprocessing.StandardScaler` convention used by the regionalization contributors'
 * `standardize` option. Non-finite entries are left out of the statistics and kept as they are;
 * a constant column becomes all zero.
 */
export function standardizeColumns(values: ArrayLike<number>, columns: number): number[] {
  const rows = values.length / columns;
  const result = Array.from(values);
  for (let column = 0; column < columns; column++) {
    let sum = 0;
    let count = 0;
    for (let row = 0; row < rows; row++) {
      const value = values[row * columns + column];
      if (Number.isFinite(value)) {
        sum += value;
        count++;
      }
    }
    const mean = sum / Math.max(count, 1);
    let squares = 0;
    for (let row = 0; row < rows; row++) {
      const value = values[row * columns + column];
      if (Number.isFinite(value)) {
        squares += (value - mean) ** 2;
      }
    }
    const deviation = Math.sqrt(squares / Math.max(count, 1));
    for (let row = 0; row < rows; row++) {
      const value = values[row * columns + column];
      result[row * columns + column] = deviation > 1e-20 ? (value - mean) / deviation : 0;
    }
  }
  return result;
}

/** One finished max-p construction, as the selection rule sees it. */
export type MaxPStartSummary = {
  /** Regions of the construction, including under-threshold leftovers. */
  regionCount: number;
  /** Regions that stay below the threshold. */
  infeasibleCount: number;
  /** Within-region sum of squares of the construction. */
  withinSquares: number;
};

/**
 * CPU mirror of the `GPUMaxPRegions` start selection: fewest under-threshold leftover regions,
 * then most regions, then smallest within-region sum of squares, then lowest start index
 * (comparisons are strict, so the earlier start wins a full tie). Returns the winning index.
 */
export function selectMaxPStart(starts: readonly MaxPStartSummary[]): number {
  let best = 0;
  for (let start = 1; start < starts.length; start++) {
    const candidate = starts[start];
    const current = starts[best];
    if (
      candidate.infeasibleCount < current.infeasibleCount ||
      (candidate.infeasibleCount === current.infeasibleCount &&
        (candidate.regionCount > current.regionCount ||
          (candidate.regionCount === current.regionCount &&
            candidate.withinSquares < current.withinSquares)))
    ) {
      best = start;
    }
  }
  return best;
}

/**
 * CPU pseudo-F (Calinski-Harabasz) in f64 with the `sklearn.metrics.calinski_harabasz_score`
 * convention: 1 when the within-region sum of squares is 0, NaN when it is undefined
 * (`k < 2` or `n <= k`; sklearn raises there).
 */
export function computePseudoF(
  values: ArrayLike<number>,
  columns: number,
  labels: ArrayLike<number>
): number {
  const {regionCount, withinSsd, totalSsd} = evaluatePartition(values, columns, labels);
  const rows = labels.length;
  if (regionCount < 2 || rows <= regionCount) {
    return Number.NaN;
  }
  if (withinSsd === 0) {
    return 1;
  }
  return ((totalSsd - withinSsd) * (rows - regionCount)) / (withinSsd * (regionCount - 1));
}
