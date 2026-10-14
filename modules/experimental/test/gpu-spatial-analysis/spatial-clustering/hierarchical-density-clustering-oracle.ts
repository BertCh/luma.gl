// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/spatial-clustering-hdbscan.json';

/** One scene of `test/oracles/spatial-clustering-hdbscan.py`. */
export type HierarchicalDensityScene = {
  name: string;
  minimumClusterSize: number;
  minimumSamples: number;
  positions: [number, number][];
  /** scikit-learn labels canonicalized by first occurrence; -1 is noise. */
  labels: number[];
  coreDistances: number[];
  /** Sorted minimum spanning tree weights (unique even where ties allow several trees). */
  spanningTreeWeights: number[];
};

/** The decoded scikit-learn fixture. */
export const HIERARCHICAL_DENSITY_FIXTURE = decodeOracleFixture<{
  tolerances: {coreRelative: number; coreAbsolute: number};
  cpuSeconds: {blobs100k: number};
  scenes: HierarchicalDensityScene[];
}>(fixtureJson);

const NONE = 0xffffffff;

/** Canonical labels by first occurrence, noise (`0xffffffff` or negative) as -1. */
export function canonicalizeLabels(labels: ArrayLike<number>): number[] {
  const mapping = new Map<number, number>();
  return Array.from(labels, label => {
    if (label < 0 || label === NONE) {
      return -1;
    }
    if (!mapping.has(label)) {
      mapping.set(label, mapping.size);
    }
    return mapping.get(label) as number;
  });
}

/**
 * float64 model of the GPU pipeline of `GPUHierarchicalDensityClustering`, stage by stage: core
 * distances, Boruvka with the `(weight, low, high)` edge order, the Kruskal dendrogram over the
 * sorted tree, condensing by "cluster start" and "lowest big ancestor", condensed-tree stability
 * and the excess-of-mass selection in cluster-ID order. The node spec checks it against
 * scikit-learn, which validates the formulation the kernels implement.
 */
export function computeHierarchicalDensityModel(
  positions: readonly (readonly number[])[],
  minimumClusterSize: number,
  minimumSamples: number
): {
  labels: number[];
  coreDistances: number[];
  spanningTreeWeights: number[];
  sortedEdges: [number, number, number][];
} {
  const rows = positions.length;
  const distance = (a: number, b: number) =>
    Math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]);
  const coreDistances = positions.map((_, row) => {
    const distances = positions.map((__, other) => distance(row, other)).sort((x, y) => x - y);
    return distances.length >= minimumSamples ? distances[minimumSamples - 1] : Infinity;
  });
  const weight = (a: number, b: number) =>
    Math.max(coreDistances[a], coreDistances[b], distance(a, b));
  // Boruvka with the strict (weight, low, high) order.
  const parents = Array.from({length: rows}, (_, row) => row);
  const find = (row: number): number => {
    let current = row;
    while (parents[current] !== current) {
      current = parents[current];
    }
    return current;
  };
  const edges: [number, number, number][] = [];
  const isBetter = (a: [number, number, number], b: [number, number, number] | undefined) =>
    !b || a[2] < b[2] || (a[2] === b[2] && (a[0] < b[0] || (a[0] === b[0] && a[1] < b[1])));
  while (edges.length < rows - 1) {
    const components = parents.map((_, row) => find(row));
    const best = new Map<number, [number, number, number]>();
    for (let a = 0; a < rows; a++) {
      for (let b = 0; b < rows; b++) {
        if (components[a] === components[b]) {
          continue;
        }
        const edge: [number, number, number] = [Math.min(a, b), Math.max(a, b), weight(a, b)];
        if (isBetter(edge, best.get(components[a]))) {
          best.set(components[a], edge);
        }
      }
    }
    for (const edge of best.values()) {
      const [first, second] = [find(edge[0]), find(edge[1])];
      if (first !== second) {
        parents[Math.max(first, second)] = Math.min(first, second);
        edges.push(edge);
      }
    }
  }
  edges.sort((a, b) => a[2] - b[2] || a[0] - b[0] || a[1] - b[1]);
  // Kruskal dendrogram: node rows + k merges the tops of the two sets.
  const nodeParent = new Array(2 * rows).fill(NONE);
  const nodeSize: number[] = Array.from({length: 2 * rows}, (_, node) => (node < rows ? 1 : 0));
  const nodeWeight: number[] = new Array(2 * rows).fill(0);
  const nodeChildren: [number, number][] = new Array(2 * rows).fill(null).map(() => [NONE, NONE]);
  const sets = Array.from({length: rows}, (_, row) => row);
  const tops = Array.from({length: rows}, (_, row) => row);
  const findSet = (row: number): number => {
    let current = row;
    while (sets[current] !== current) {
      current = sets[current];
    }
    return current;
  };
  edges.forEach(([low, high, edgeWeight], rank) => {
    const [first, second] = [findSet(low), findSet(high)];
    const node = rows + rank;
    const [firstTop, secondTop] = [tops[first], tops[second]];
    nodeSize[node] = nodeSize[firstTop] + nodeSize[secondTop];
    nodeWeight[node] = edgeWeight;
    nodeChildren[node] = [firstTop, secondTop];
    nodeParent[firstTop] = node;
    nodeParent[secondTop] = node;
    const [root, child] =
      nodeSize[firstTop] >= nodeSize[secondTop] ? [first, second] : [second, first];
    sets[child] = root;
    tops[root] = node;
  });
  const lambda = (node: number) => (nodeWeight[node] > 0 ? 1 / nodeWeight[node] : 3.402823e38);
  const isBig = (node: number) => node !== NONE && nodeSize[node] >= minimumClusterSize;
  const isStart = (node: number) => {
    if (!isBig(node)) {
      return false;
    }
    const parent = nodeParent[node];
    return parent === NONE || (isBig(nodeChildren[parent][0]) && isBig(nodeChildren[parent][1]));
  };
  const nodeCount = rows + edges.length;
  const clusterStart = (node: number) => {
    let current = node;
    while (!isStart(current)) {
      current = nodeParent[current];
    }
    return current;
  };
  const lowestBig = (node: number) => {
    let current = node;
    while (current !== NONE && !isBig(current)) {
      current = nodeParent[current];
    }
    return current;
  };
  const clusterIds = new Map<number, number>();
  for (let node = 0; node < nodeCount; node++) {
    if (isStart(node)) {
      clusterIds.set(node, clusterIds.size);
    }
  }
  const clusterCount = clusterIds.size;
  const birth = new Array(clusterCount).fill(0);
  const parentCluster = new Array(clusterCount).fill(NONE);
  for (const [node, cluster] of clusterIds) {
    const parent = nodeParent[node];
    if (parent !== NONE) {
      birth[cluster] = lambda(parent);
      parentCluster[cluster] = clusterIds.get(clusterStart(parent)) as number;
    }
  }
  const stability = new Array(clusterCount).fill(0);
  const lowClusters = new Array(rows).fill(NONE);
  for (let row = 0; row < rows; row++) {
    const fall = lowestBig(row);
    if (fall === NONE) {
      continue;
    }
    const cluster = clusterIds.get(clusterStart(fall)) as number;
    lowClusters[row] = cluster;
    stability[cluster] += lambda(fall) - birth[cluster];
  }
  for (const [node, cluster] of clusterIds) {
    if (parentCluster[cluster] !== NONE) {
      stability[parentCluster[cluster]] +=
        nodeSize[node] * (birth[cluster] - birth[parentCluster[cluster]]);
    }
  }
  // Excess of mass in cluster-ID order (children first).
  const childBest = new Array(clusterCount).fill(0);
  const selected = new Array(clusterCount).fill(false);
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    if (parentCluster[cluster] === NONE) {
      continue;
    }
    let best = stability[cluster];
    if (childBest[cluster] > stability[cluster]) {
      best = childBest[cluster];
    } else {
      selected[cluster] = true;
    }
    childBest[parentCluster[cluster]] += best;
  }
  const representative = new Array(clusterCount).fill(NONE);
  for (let cluster = clusterCount - 1; cluster >= 0; cluster--) {
    const parent = parentCluster[cluster];
    const inherited = parent === NONE ? NONE : representative[parent];
    if (inherited !== NONE) {
      selected[cluster] = false;
      representative[cluster] = inherited;
    } else if (selected[cluster]) {
      representative[cluster] = cluster;
    }
  }
  const labels = lowClusters.map(cluster =>
    cluster === NONE || representative[cluster] === NONE ? NONE : representative[cluster]
  );
  return {
    labels: canonicalizeLabels(labels),
    coreDistances,
    spanningTreeWeights: edges.map(edge => edge[2]),
    sortedEdges: edges
  };
}

/**
 * Adjusted Rand index of two labelings (noise is one more label), 1 for identical partitions.
 * Used where tied mutual reachability distances let a few points condense differently.
 */
export function getAdjustedRandIndex(first: readonly number[], second: readonly number[]): number {
  const choose = (value: number) => (value * (value - 1)) / 2;
  const table = new Map<string, number>();
  const rows = new Map<number, number>();
  const columns = new Map<number, number>();
  first.forEach((label, index) => {
    const key = `${label}|${second[index]}`;
    table.set(key, (table.get(key) ?? 0) + 1);
    rows.set(label, (rows.get(label) ?? 0) + 1);
    columns.set(second[index], (columns.get(second[index]) ?? 0) + 1);
  });
  const sumTable = [...table.values()].reduce((sum, value) => sum + choose(value), 0);
  const sumRows = [...rows.values()].reduce((sum, value) => sum + choose(value), 0);
  const sumColumns = [...columns.values()].reduce((sum, value) => sum + choose(value), 0);
  const expected = (sumRows * sumColumns) / choose(first.length);
  const maximum = (sumRows + sumColumns) / 2;
  return maximum === expected ? 1 : (sumTable - expected) / (maximum - expected);
}
