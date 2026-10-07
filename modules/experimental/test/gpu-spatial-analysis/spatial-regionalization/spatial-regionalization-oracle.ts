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
  runs: {k: number; floor: number | null; labels: number[]; score: number}[];
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
