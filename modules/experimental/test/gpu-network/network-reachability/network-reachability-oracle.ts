// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const NONE = 0xffffffff;

/** Directed edge `[from, to, weight]`. */
export type NetworkEdge = readonly [number, number, number];

/** Packed CSR arrays. */
export type NetworkCSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Builds a CSR with a stable counting sort by source node. */
export function buildCSR(nodeCount: number, edges: readonly NetworkEdge[]): NetworkCSR {
  const offsets = new Uint32Array(nodeCount + 1);
  for (const [from] of edges) {
    offsets[from + 1]++;
  }
  for (let node = 0; node < nodeCount; node++) {
    offsets[node + 1] += offsets[node];
  }
  const cursor = offsets.slice(0, nodeCount);
  const neighbors = new Uint32Array(edges.length);
  const weights = new Float32Array(edges.length);
  for (const [from, to, weight] of edges) {
    const slot = cursor[from]++;
    neighbors[slot] = to;
    weights[slot] = weight;
  }
  return {offsets, neighbors, weights};
}

/** O(n^2) Dijkstra with the kernel's validity rules and f32 sums. */
export function dijkstra(
  csr: NetworkCSR,
  nodeCount: number,
  sources: readonly {node: number; cost: number}[],
  costLimit: number = Infinity
): Float32Array {
  const distances = new Float32Array(nodeCount).fill(Infinity);
  for (const {node, cost} of sources) {
    if (node < nodeCount && cost >= 0 && Number.isFinite(cost) && cost <= costLimit) {
      distances[node] = Math.min(distances[node], Math.fround(cost));
    }
  }
  const done = new Uint8Array(nodeCount);
  for (;;) {
    let current = -1;
    for (let node = 0; node < nodeCount; node++) {
      if (
        !done[node] &&
        Number.isFinite(distances[node]) &&
        (current < 0 || distances[node] < distances[current])
      ) {
        current = node;
      }
    }
    if (current < 0) {
      break;
    }
    done[current] = 1;
    for (let edge = csr.offsets[current]; edge < csr.offsets[current + 1]; edge++) {
      const target = csr.neighbors[edge];
      const weight = csr.weights[edge];
      if (target >= nodeCount || !(weight >= 0)) {
        continue;
      }
      const candidate = Math.fround(distances[current] + weight);
      if (!Number.isFinite(candidate) || candidate > costLimit) {
        continue;
      }
      if (candidate < distances[target]) {
        distances[target] = candidate;
      }
    }
  }
  return distances;
}

/** One accepted-or-not source row, as passed to the seed kernel. */
export type OracleSource = {node: number; cost: number};

/**
 * Cycle-safe shortest-path predecessors and tie levels, bit-exact with `GPUNetworkReachability`.
 *
 * Tight edge `u -> v`: valid edge (`v` in range, weight `>= 0`, both costs finite) with the f32 sum
 * `costs[u] + w === costs[v]`. Strict: tight and `costs[u] < costs[v]`. Tie: tight and equal costs.
 * `levels[v] = 0` when `v` has a strict tight in-edge or is a root source (an accepted source row
 * whose seeded f32 cost equals `costs[v]`); otherwise `1 + min levels[u]` over tie in-edges, or
 * `NONE` when no level-0 node reaches it across tie edges.
 * `predecessors[v]` is the smallest strict tight in-neighbor when one exists, else the smallest tie
 * in-neighbor `u` with `levels[u] + 1 === levels[v]`, else `NONE`. Along links cost never increases
 * and, at equal cost, the level strictly decreases, so following predecessors always ends at a root.
 *
 * `sources` must already be limited to the active `sourceCount`.
 */
export function tieLevelOracle(
  csr: NetworkCSR,
  costs: Float32Array,
  sources: readonly OracleSource[],
  costLimit: number = Infinity,
  maxLevel: number = Infinity
): {predecessors: Uint32Array; levels: Uint32Array} {
  const nodeCount = costs.length;
  const predecessors = new Uint32Array(nodeCount).fill(NONE);
  const levels = new Uint32Array(nodeCount).fill(NONE);
  const isTight = (from: number, to: number, weight: number) =>
    to < nodeCount &&
    weight >= 0 &&
    Number.isFinite(costs[from]) &&
    Number.isFinite(costs[to]) &&
    Math.fround(costs[from] + weight) === costs[to];
  for (let from = 0; from < nodeCount; from++) {
    for (let edge = csr.offsets[from]; edge < csr.offsets[from + 1]; edge++) {
      const to = csr.neighbors[edge];
      if (isTight(from, to, csr.weights[edge]) && costs[from] < costs[to]) {
        predecessors[to] = Math.min(predecessors[to], from);
        levels[to] = 0;
      }
    }
  }
  for (const {node, cost} of sources) {
    if (
      node < nodeCount &&
      cost >= 0 &&
      Number.isFinite(cost) &&
      Math.fround(cost) <= Math.fround(costLimit) &&
      Math.fround(cost) === costs[node]
    ) {
      levels[node] = 0;
    }
  }
  let frontier = Array.from({length: nodeCount}, (_, node) => node).filter(
    node => levels[node] === 0
  );
  for (let level = 0; frontier.length && level < maxLevel; level++) {
    const next: number[] = [];
    for (const from of frontier) {
      for (let edge = csr.offsets[from]; edge < csr.offsets[from + 1]; edge++) {
        const to = csr.neighbors[edge];
        if (
          isTight(from, to, csr.weights[edge]) &&
          costs[from] === costs[to] &&
          levels[to] === NONE
        ) {
          levels[to] = level + 1;
          next.push(to);
        }
      }
    }
    frontier = next;
  }
  for (let from = 0; from < nodeCount; from++) {
    for (let edge = csr.offsets[from]; edge < csr.offsets[from + 1]; edge++) {
      const to = csr.neighbors[edge];
      if (
        isTight(from, to, csr.weights[edge]) &&
        costs[from] === costs[to] &&
        levels[to] !== 0 &&
        levels[to] !== NONE &&
        levels[from] + 1 === levels[to]
      ) {
        predecessors[to] = Math.min(predecessors[to], from);
      }
    }
  }
  return {predecessors, levels};
}

/** Predecessors of {@link tieLevelOracle}. */
export function predecessorOracle(
  csr: NetworkCSR,
  costs: Float32Array,
  sources: readonly OracleSource[],
  costLimit: number = Infinity
): Uint32Array {
  return tieLevelOracle(csr, costs, sources, costLimit).predecessors;
}

/**
 * Independent check of a predecessor array: every reached node's predecessor walk ends at a root in
 * fewer than `nodeCount` steps and every link is a tight edge (`costs[u] + w === costs[v]`) of the
 * CSR. Returns the first violation as text, or `undefined`.
 */
export function checkPredecessorWalks(
  csr: NetworkCSR,
  costs: Float32Array,
  predecessors: ArrayLike<number>,
  isRoot: (node: number) => boolean
): string | undefined {
  const nodeCount = costs.length;
  for (let start = 0; start < nodeCount; start++) {
    if (!Number.isFinite(costs[start])) {
      continue;
    }
    let node = start;
    let steps = 0;
    while (predecessors[node] !== NONE) {
      const from = predecessors[node];
      if (from >= nodeCount || ++steps >= nodeCount) {
        return `walk from ${start} does not terminate`;
      }
      let linked = false;
      for (let edge = csr.offsets[from]; edge < csr.offsets[from + 1]; edge++) {
        linked ||=
          csr.neighbors[edge] === node &&
          csr.weights[edge] >= 0 &&
          Math.fround(costs[from] + csr.weights[edge]) === costs[node];
      }
      if (!linked) {
        return `link ${from} -> ${node} is not a tight edge`;
      }
      node = from;
    }
    if (!isRoot(node)) {
      return `walk from ${start} ends at non-root ${node}`;
    }
  }
  return undefined;
}

/** Number of thresholds below each cost, or `NONE`. */
export function bandOracle(costs: Float32Array, thresholds: readonly number[]): Uint32Array {
  return Uint32Array.from(costs, cost => {
    if (!Number.isFinite(cost)) {
      return NONE;
    }
    const count = thresholds.filter(threshold => Math.fround(threshold) < cost).length;
    return count < thresholds.length ? count : NONE;
  });
}

/** Deterministic random directed network; the last 5% of nodes get no in-edges. */
export function createRandomNetwork(
  seed: number,
  nodeCount: number,
  edgeCount: number
): NetworkEdge[] {
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const reachableCount = Math.floor(nodeCount * 0.95);
  return Array.from({length: edgeCount}, () => [
    Math.floor(next() * nodeCount),
    Math.floor(next() * reachableCount),
    1 + Math.floor(next() * 9)
  ]);
}

/** Cyclic, disconnected 8-node fixture. */
export const F1_EDGES: NetworkEdge[] = [
  [0, 1, 4],
  [0, 2, 1],
  [2, 1, 2],
  [1, 3, 1],
  [2, 3, 5],
  [3, 4, 3],
  [4, 2, 1],
  [4, 0, 10],
  [5, 6, 1],
  [6, 5, 1]
];

/** Directed path `0 -> 1 -> ... -> nodeCount - 1` with unit weights. */
export function createPathNetwork(nodeCount: number): NetworkEdge[] {
  return Array.from({length: nodeCount - 1}, (_, node) => [node, node + 1, 1] as NetworkEdge);
}

/**
 * Deterministic 4-neighbor grid with both directions per link and integer weights in `[1, 9]`.
 * Node `(x, y)` has index `y * width + x`; the longest shortest path is hundreds of hops.
 */
export function createGridNetwork(seed: number, width: number, height: number): NetworkEdge[] {
  let state = seed >>> 0;
  const nextWeight = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 1 + Math.floor((state / 2 ** 32) * 9);
  };
  const edges: NetworkEdge[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const node = y * width + x;
      if (x + 1 < width) {
        edges.push([node, node + 1, nextWeight()], [node + 1, node, nextWeight()]);
      }
      if (y + 1 < height) {
        edges.push([node, node + width, nextWeight()], [node + width, node, nextWeight()]);
      }
    }
  }
  return edges;
}
