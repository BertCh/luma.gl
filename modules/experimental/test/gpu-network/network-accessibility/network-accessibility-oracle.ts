// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** u32 "no edge" / "no node" sentinel. */
export const NONE = 0xffffffff;

/** Directed edge `[from, to, weight]`. */
export type AccessibilityEdge = readonly [number, number, number];

/** Packed CSR arrays plus the COO source of every CSR slot. */
export type AccessibilityCSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  sources: Uint32Array;
};

/** One seed of a search: node and initial cost. */
export type AccessibilitySeed = {node: number; cost: number};

/** Builds a CSR with a stable counting sort by source node. */
export function buildCSR(nodeCount: number, edges: readonly AccessibilityEdge[]): AccessibilityCSR {
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
  const sources = new Uint32Array(edges.length);
  for (const [from, to, weight] of edges) {
    const slot = cursor[from]++;
    neighbors[slot] = to;
    weights[slot] = weight;
    sources[slot] = from;
  }
  return {offsets, neighbors, weights, sources};
}

/** Reverses every edge of an edge list. */
export function reverseEdges(edges: readonly AccessibilityEdge[]): AccessibilityEdge[] {
  return edges.map(([from, to, weight]) => [to, from, weight]);
}

/**
 * O(n^2) Dijkstra with f32 path sums, the reachability seed rules (negative, NaN, infinite or
 * over-limit seeds are ignored), and impassable negative edges. With non-negative weights and
 * monotone f32 addition this equals the GPU relaxation fixpoint bit for bit.
 */
export function dijkstra(
  csr: AccessibilityCSR,
  nodeCount: number,
  seeds: readonly AccessibilitySeed[],
  costLimit: number = Infinity
): Float32Array {
  const limit = Math.fround(costLimit);
  const distances = new Float32Array(nodeCount).fill(Infinity);
  for (const {node, cost} of seeds) {
    const seedCost = Math.fround(cost);
    if (node < nodeCount && seedCost >= 0 && Number.isFinite(seedCost) && seedCost <= limit) {
      distances[node] = Math.min(distances[node], seedCost);
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
      if (Number.isFinite(candidate) && candidate <= limit && candidate < distances[target]) {
        distances[target] = candidate;
      }
    }
  }
  return distances;
}

/** Row-major `[rows.length x nodeCount]` matrix of one Dijkstra per row of seeds. */
export function costMatrixOracle(
  csr: AccessibilityCSR,
  nodeCount: number,
  rows: readonly (readonly AccessibilitySeed[])[],
  costLimit: number = Infinity
): Float32Array {
  const matrix = new Float32Array(rows.length * nodeCount);
  rows.forEach((seeds, row) =>
    matrix.set(dijkstra(csr, nodeCount, seeds, costLimit), row * nodeCount)
  );
  return matrix;
}

/** Scoring parameters mirrored from `GPUNetworkAccessibilityParameters`. */
export type OracleParameters = {
  threshold: number;
  decay: 'none' | 'exponential' | 'power';
  beta: number;
  minimumCost: number;
};

function isWithin(cost: number, threshold: number): boolean {
  return Number.isFinite(cost) && cost >= 0 && cost <= Math.fround(threshold);
}

function getDecay(cost: number, parameters: OracleParameters): number {
  if (parameters.decay === 'exponential') {
    return Math.exp(-parameters.beta * cost);
  }
  if (parameters.decay === 'power') {
    return Math.max(cost, parameters.minimumCost) ** -parameters.beta;
  }
  return 1;
}

/**
 * Cumulative and gravity accessibility in f64. `'opportunity-rows'`: rows are opportunities with
 * `weights` per row, outputs per column. `'origin-rows'`: weights per column, outputs per row.
 */
export function accessibilityOracle(
  matrix: Float32Array,
  rowCount: number,
  nodeCount: number,
  orientation: 'opportunity-rows' | 'origin-rows',
  weights: ArrayLike<number>,
  parameters: OracleParameters
): {cumulative: number[]; gravity: number[]} {
  const originCount = orientation === 'opportunity-rows' ? nodeCount : rowCount;
  const cumulative = new Array<number>(originCount).fill(0);
  const gravity = new Array<number>(originCount).fill(0);
  for (let row = 0; row < rowCount; row++) {
    for (let node = 0; node < nodeCount; node++) {
      const cost = matrix[row * nodeCount + node];
      if (!isWithin(cost, parameters.threshold)) {
        continue;
      }
      const [origin, weight] =
        orientation === 'opportunity-rows' ? [node, weights[row]] : [row, weights[node]];
      cumulative[origin] += weight;
      gravity[origin] += weight * getDecay(cost, parameters);
    }
  }
  return {cumulative, gravity};
}

/**
 * Two-step floating catchment area in f64: `R_j = S_j / sum_i P_i f(c_ij)` (0 without demand),
 * `A_i = sum_j R_j f(c_ij)`, over entries within the threshold. Rows are facilities.
 */
export function catchmentOracle(
  matrix: Float32Array,
  rowCount: number,
  nodeCount: number,
  supply: ArrayLike<number>,
  demand: ArrayLike<number>,
  parameters: OracleParameters
): {ratios: number[]; accessibility: number[]} {
  const ratios = new Array<number>(rowCount).fill(0);
  for (let row = 0; row < rowCount; row++) {
    let denominator = 0;
    for (let node = 0; node < nodeCount; node++) {
      const cost = matrix[row * nodeCount + node];
      if (isWithin(cost, parameters.threshold)) {
        denominator += demand[node] * getDecay(cost, parameters);
      }
    }
    ratios[row] = denominator > 0 ? supply[row] / denominator : 0;
  }
  const accessibility = new Array<number>(nodeCount).fill(0);
  for (let row = 0; row < rowCount; row++) {
    for (let node = 0; node < nodeCount; node++) {
      const cost = matrix[row * nodeCount + node];
      if (isWithin(cost, parameters.threshold)) {
        accessibility[node] += ratios[row] * getDecay(cost, parameters);
      }
    }
  }
  return {ratios, accessibility};
}

/** One snapped point of {@link snapOracle}. */
export type OracleSnap = {
  edge: number;
  fraction: number;
  distance: number;
  sourceCost: number;
  targetCost: number;
};

/**
 * Exhaustive nearest-edge snapping in f64: smallest distance, ties to the smallest edge row;
 * edges with an out-of-range endpoint are skipped. Edge cost defaults to the planar length.
 */
export function snapOracle(
  points: ArrayLike<number>,
  positions: ArrayLike<number>,
  edgeSources: ArrayLike<number>,
  edgeTargets: ArrayLike<number>,
  options: {edgeCosts?: ArrayLike<number>; maxSnapDistance?: number} = {}
): OracleSnap[] {
  const nodeCount = positions.length / 2;
  const limit = options.maxSnapDistance ?? Infinity;
  const result: OracleSnap[] = [];
  for (let point = 0; point < points.length / 2; point++) {
    const px = points[point * 2];
    const py = points[point * 2 + 1];
    let best: OracleSnap = {
      edge: NONE,
      fraction: -1,
      distance: -1,
      sourceCost: -1,
      targetCost: -1
    };
    for (let edge = 0; edge < edgeTargets.length; edge++) {
      const source = edgeSources[edge];
      const target = edgeTargets[edge];
      if (source >= nodeCount || target >= nodeCount) {
        continue;
      }
      const ax = positions[source * 2];
      const ay = positions[source * 2 + 1];
      const dx = positions[target * 2] - ax;
      const dy = positions[target * 2 + 1] - ay;
      const lengthSquared = dx * dx + dy * dy;
      const fraction =
        lengthSquared > 0
          ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / lengthSquared))
          : 0;
      const distance = Math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy));
      if (distance <= limit && (best.edge === NONE || distance < best.distance - 1e-9)) {
        const edgeCost = options.edgeCosts ? options.edgeCosts[edge] : Math.sqrt(lengthSquared);
        best = {
          edge,
          fraction,
          distance,
          sourceCost: fraction * edgeCost,
          targetCost: (1 - fraction) * edgeCost
        };
      }
    }
    result.push(best);
  }
  return result;
}

/**
 * Undirected `width x height` grid with unit spacing, both directions per link, and integer
 * weights in `[1, 9]` (the same weight both ways). Node `(x, y)` is `y * width + x`.
 */
export function createGridFixture(
  seed: number,
  width: number,
  height: number
): {positions: Float32Array; edges: AccessibilityEdge[]} {
  let state = seed >>> 0;
  const nextWeight = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 1 + Math.floor((state / 2 ** 32) * 9);
  };
  const positions = new Float32Array(width * height * 2);
  const edges: AccessibilityEdge[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const node = y * width + x;
      positions[node * 2] = x;
      positions[node * 2 + 1] = y;
      if (x + 1 < width) {
        const weight = nextWeight();
        edges.push([node, node + 1, weight], [node + 1, node, weight]);
      }
      if (y + 1 < height) {
        const weight = nextWeight();
        edges.push([node, node + width, weight], [node + width, node, weight]);
      }
    }
  }
  return {positions, edges};
}

/**
 * Directed 9-node fixture with a zero-weight chain (2 -> 3 -> 4 at weight 0, plus a zero-weight
 * cycle 4 -> 2), a one-way edge, and a disconnected component {6, 7, 8}.
 */
export const ZERO_WEIGHT_EDGES: AccessibilityEdge[] = [
  [0, 1, 2],
  [1, 0, 2],
  [1, 2, 3],
  [2, 3, 0],
  [3, 4, 0],
  [4, 2, 0],
  [4, 5, 1.5],
  [5, 1, 4],
  [0, 5, 10],
  [6, 7, 1],
  [7, 6, 1],
  [7, 8, 0]
];
