// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixture from '../../oracles/fixtures/network-centrality.json';
import {
  buildCSR,
  dijkstra,
  type NetworkCSR,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';

/** One networkx centrality scene (see `test/oracles/network-centrality.py`). */
export type CentralityReferenceScene = {
  nodeCount: number;
  /** Directed `[from, to, weight]` edges with integer weights in `[1, 9]`. */
  edges: NetworkEdge[];
  /** Inclusive radius of the bounded references. */
  radius: number;
  /** Pivots of the sampled references; `nodeCount / sampledPivots.length` is exact. */
  sampledPivots: number[];
  sampledScale: number;
  betweenness: {
    /** networkx `betweenness_centrality`, every node a pivot. */
    exact: number[];
    /** networkx `betweenness_centrality_subset` from `sampledPivots`, unscaled. */
    sampled: number[];
    /** Explicit Brandes within `radius`, every node a pivot. */
    radius: number[];
    /** Explicit Brandes within `radius` from `sampledPivots`. */
    radiusSampled: number[];
  };
  /** Outward closeness per node used as the pivot. */
  closeness: {
    unbounded: CentralityReferenceClosenessRows;
    radius: CentralityReferenceClosenessRows;
  };
  maximumSigma: number;
  maximumDepth: number;
  maximumOutDegree: number;
  tolerance: {
    deltaRelative: number;
    betweennessRelative: number;
    closenessRelative: number;
    harmonicRelative: number;
  };
};

/** Per-pivot closeness references of a scene. */
export type CentralityReferenceClosenessRows = {
  reachedCounts: number[];
  closeness: number[];
  harmonicCloseness: number[];
};

/** The whole fixture. */
export type CentralityReferenceFixture = {
  generator: string;
  versions: Record<string, string>;
  scenes: Record<string, CentralityReferenceScene>;
};

/** The committed networkx fixture. */
export const CENTRALITY_REFERENCE = fixture as unknown as CentralityReferenceFixture;

/** Forward and reverse CSR of an edge list; the reverse CSR lists the same edges transposed. */
export function buildCentralityCSRs(
  nodeCount: number,
  edges: readonly NetworkEdge[]
): {forward: NetworkCSR; reverse: NetworkCSR} {
  return {
    forward: buildCSR(nodeCount, edges),
    reverse: buildCSR(
      nodeCount,
      edges.map(([from, to, weight]): NetworkEdge => [to, from, weight])
    )
  };
}

/** Values of {@link mirrorCentrality}. */
export type CentralityMirrorResult = {
  betweenness: Float32Array;
  reachedCounts: Uint32Array;
  closeness: Float32Array;
  harmonicCloseness: Float32Array;
};

/** Options of {@link mirrorCentrality}. */
export type CentralityMirrorOptions = {
  nodeCount: number;
  edges: readonly NetworkEdge[];
  /** Pivot nodes; out-of-range nodes are ignored. */
  pivots: readonly number[];
  /** Inclusive cost limit. Defaults to none. */
  radius?: number;
  /** Pivots per batch; sets the grouping of the betweenness sums. */
  laneCount: number;
  /** Factor applied to each batch sum. Defaults to 1. */
  scale?: number;
};

/**
 * CPU mirror of the arithmetic and schedule of `GPUNetworkCentrality`, in f32 where the kernels
 * are: f32 Dijkstra costs, tight edges `cost[u] < cost[w]` with the f32 sum equal to `cost[w]`,
 * sigma summed over reverse-CSR tight in-edges, delta summed over forward-CSR tight out-edges as
 * `sigma(v) / sigma(w) * (1 + delta(w))`, lane-ordered sums per batch scaled and added in batch
 * order. Nodes are processed in cost order, which is a valid topological order of the tight graph;
 * the kernels' push-ready schedule gives every node the same fixed-order sums. Reached nodes without
 * a tight in-edge are roots with sigma 1.
 */
export function mirrorCentrality(options: CentralityMirrorOptions): CentralityMirrorResult {
  const {nodeCount, pivots, laneCount} = options;
  const radius = options.radius ?? Infinity;
  const scale = Math.fround(options.scale ?? 1);
  const {forward, reverse} = buildCentralityCSRs(nodeCount, options.edges);
  const isTight = (sourceCost: number, weight: number, targetCost: number) =>
    weight >= 0 &&
    sourceCost < targetCost &&
    Number.isFinite(targetCost) &&
    Math.fround(sourceCost + weight) === targetCost;
  const betweenness = new Float32Array(nodeCount);
  const reachedCounts = new Uint32Array(pivots.length);
  const closeness = new Float32Array(pivots.length);
  const harmonicCloseness = new Float32Array(pivots.length);
  const deltas: Float32Array[] = [];
  const flushBatch = (batchPivots: readonly number[]) => {
    for (let node = 0; node < nodeCount; node++) {
      let total = 0;
      batchPivots.forEach((pivot, lane) => {
        if (pivot !== node && pivot < nodeCount) {
          total = Math.fround(total + deltas[lane][node]);
        }
      });
      betweenness[node] = Math.fround(betweenness[node] + Math.fround(scale * total));
    }
    deltas.length = 0;
  };
  for (let row = 0; row < pivots.length; row++) {
    const pivot = pivots[row];
    const delta = new Float32Array(nodeCount);
    if (pivot < nodeCount) {
      const costs = dijkstra(forward, nodeCount, [{node: pivot, cost: 0}], radius);
      const order = Array.from({length: nodeCount}, (_, node) => node)
        .filter(node => Number.isFinite(costs[node]))
        .sort((left, right) => costs[left] - costs[right] || left - right);
      let costSum = 0;
      let harmonicSum = 0;
      for (const node of order) {
        costSum += costs[node];
        if (costs[node] > 0) {
          harmonicSum += 1 / costs[node];
        }
      }
      reachedCounts[row] = order.length;
      closeness[row] = costSum > 0 ? Math.fround((order.length - 1) / costSum) : 0;
      harmonicCloseness[row] = Math.fround(harmonicSum);
      const sigma = new Float32Array(nodeCount);
      for (const node of order) {
        let tightInCount = 0;
        let total = 0;
        for (let edge = reverse.offsets[node]; edge < reverse.offsets[node + 1]; edge++) {
          const source = reverse.neighbors[edge];
          if (isTight(costs[source], reverse.weights[edge], costs[node])) {
            tightInCount++;
            total = Math.fround(total + sigma[source]);
          }
        }
        sigma[node] = tightInCount > 0 ? total : 1;
      }
      for (const node of [...order].reverse()) {
        let total = 0;
        for (let edge = forward.offsets[node]; edge < forward.offsets[node + 1]; edge++) {
          const target = forward.neighbors[edge];
          if (!isTight(costs[node], forward.weights[edge], costs[target]) || !(sigma[target] > 0)) {
            continue;
          }
          const ratio = Math.fround(sigma[node] / sigma[target]);
          const term = Math.fround(ratio * Math.fround(1 + delta[target]));
          total = Math.fround(total + term);
        }
        delta[node] = total;
      }
    }
    deltas.push(delta);
    if (deltas.length === laneCount || row === pivots.length - 1) {
      flushBatch(pivots.slice(row + 1 - deltas.length, row + 1));
    }
  }
  return {betweenness, reachedCounts, closeness, harmonicCloseness};
}

/**
 * Textbook Brandes betweenness in float64 with a binary-heap Dijkstra: the CPU speed bar of the
 * bench. Same conventions as the GPU contributor (endpoints excluded, tight edges with an exact
 * distance match; weights must be positive).
 */
export function computeBrandesBetweenness(
  csr: NetworkCSR,
  nodeCount: number,
  pivots: readonly number[]
): Float64Array {
  const betweenness = new Float64Array(nodeCount);
  const distances = new Float64Array(nodeCount);
  const sigma = new Float64Array(nodeCount);
  const delta = new Float64Array(nodeCount);
  const settled = new Uint8Array(nodeCount);
  for (const pivot of pivots) {
    distances.fill(Infinity);
    sigma.fill(0);
    delta.fill(0);
    settled.fill(0);
    distances[pivot] = 0;
    sigma[pivot] = 1;
    const order: number[] = [];
    const heap: [number, number][] = [[0, pivot]];
    while (heap.length > 0) {
      const [distance, node] = popHeap(heap);
      if (settled[node] || distance > distances[node]) {
        continue;
      }
      settled[node] = 1;
      order.push(node);
      for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
        const target = csr.neighbors[edge];
        const candidate = distance + csr.weights[edge];
        if (candidate < distances[target]) {
          distances[target] = candidate;
          sigma[target] = sigma[node];
          pushHeap(heap, [candidate, target]);
        } else if (candidate === distances[target] && !settled[target]) {
          sigma[target] += sigma[node];
        }
      }
    }
    for (let index = order.length - 1; index >= 0; index--) {
      const node = order[index];
      for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
        const target = csr.neighbors[edge];
        if (distances[node] + csr.weights[edge] === distances[target]) {
          delta[node] += (sigma[node] / sigma[target]) * (1 + delta[target]);
        }
      }
      if (node !== pivot) {
        betweenness[node] += delta[node];
      }
    }
  }
  return betweenness;
}

function pushHeap(heap: [number, number][], entry: [number, number]): void {
  heap.push(entry);
  let index = heap.length - 1;
  while (index > 0) {
    const parent = (index - 1) >> 1;
    if (heap[parent][0] <= heap[index][0]) {
      break;
    }
    [heap[parent], heap[index]] = [heap[index], heap[parent]];
    index = parent;
  }
}

function popHeap(heap: [number, number][]): [number, number] {
  const top = heap[0];
  const last = heap.pop()!;
  if (heap.length > 0) {
    heap[0] = last;
    let index = 0;
    for (;;) {
      const left = 2 * index + 1;
      const right = left + 1;
      let smallest = index;
      if (left < heap.length && heap[left][0] < heap[smallest][0]) {
        smallest = left;
      }
      if (right < heap.length && heap[right][0] < heap[smallest][0]) {
        smallest = right;
      }
      if (smallest === index) {
        break;
      }
      [heap[smallest], heap[index]] = [heap[index], heap[smallest]];
      index = smallest;
    }
  }
  return top;
}

/**
 * Compares values with a relative tolerance (exact zeros must stay zero). Returns the first
 * violation as text, or `undefined`.
 */
export function compareCentralityValues(
  name: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  relative: number,
  scale: number = 1
): string | undefined {
  if (actual.length !== expected.length) {
    return `${name}: length ${actual.length}, expected ${expected.length}`;
  }
  for (let index = 0; index < expected.length; index++) {
    const reference = expected[index] * scale;
    if (!(Math.abs(actual[index] - reference) <= relative * Math.abs(reference))) {
      return `${name}[${index}]: expected ${reference}, got ${actual[index]}`;
    }
  }
  return undefined;
}

/**
 * Bounds the dependency rounds for the bidirectional benchmark grid with integer weights >= 1.
 * Any pair has a row-then-column path costing at most the largest full row plus the largest full
 * column. A shortest path costs no more; each tight edge costs at least 1, so its hop count is at
 * most that sum. Processing the root as well requires one additional round.
 */
export function getCentralityGridMaximumIterations(
  width: number,
  edges: readonly NetworkEdge[]
): number {
  const rowCosts = new Float64Array(width);
  const columnCosts = new Float64Array(width);
  for (const [source, target, weight] of edges) {
    if (target === source + 1) {
      rowCosts[Math.floor(source / width)] += weight;
    } else if (target === source + width) {
      columnCosts[source % width] += weight;
    }
  }
  return Math.max(...rowCosts) + Math.max(...columnCosts) + 1;
}

/** Test fixtures: a grid with both directions and integer weights, for benches. */
export function createCentralityGridEdges(width: number, seed: number): NetworkEdge[] {
  let state = seed >>> 0;
  const nextWeight = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 1 + Math.floor((state / 2 ** 32) * 3);
  };
  const edges: NetworkEdge[] = [];
  for (let y = 0; y < width; y++) {
    for (let x = 0; x < width; x++) {
      const node = y * width + x;
      if (x + 1 < width) {
        const weight = nextWeight();
        edges.push([node, node + 1, weight], [node + 1, node, weight]);
      }
      if (y + 1 < width) {
        const weight = nextWeight();
        edges.push([node, node + width, weight], [node + width, node, weight]);
      }
    }
  }
  return edges;
}
