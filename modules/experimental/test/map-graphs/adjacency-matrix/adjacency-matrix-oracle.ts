// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAdjacencyMatrixFixedWeight} from '../../../src/map-graphs/adjacency-matrix';

/** Forward CSR with weights. */
export type MatrixCSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Builds a forward CSR from `[from, to, weight]` edges, keeping edge order within a row. */
export function buildMatrixCSR(
  nodeCount: number,
  edges: readonly (readonly [number, number, number])[]
): MatrixCSR {
  const offsets = new Uint32Array(nodeCount + 1);
  for (const [from] of edges) offsets[from + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
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

/** Deterministic pseudo-random edges with weights that are multiples of 1/8 (exact in f32). */
export function createMatrixEdges(
  seed: number,
  nodeCount: number,
  edgeCount: number
): [number, number, number][] {
  let state = seed;
  const next = (limit: number) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % limit;
  };
  return Array.from({length: edgeCount}, () => [next(nodeCount), next(nodeCount), next(64) / 8]);
}

export type MatrixOracleOptions = {
  nodeCount: number;
  csr: MatrixCSR;
  resolution: number;
  directed?: boolean;
  mirrorSlots?: boolean;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  order?: Uint32Array;
  window?: {
    rowStart: number;
    rowEnd: number;
    colStart: number;
    colEnd: number;
  };
  weightScale?: number;
};

/** CPU reference for `GPUAdjacencyMatrix`. */
export function computeAdjacencyMatrixOracle(options: MatrixOracleOptions) {
  const {nodeCount, csr, resolution} = options;
  const window = options.window ?? {
    rowStart: 0,
    rowEnd: nodeCount,
    colStart: 0,
    colEnd: nodeCount
  };
  const scale = options.weightScale ?? 1024;
  const mirror = !options.directed && Boolean(options.mirrorSlots);
  const counts = new Uint32Array(resolution * resolution);
  const weightSums = new Uint32Array(resolution * resolution);
  const place = (row: number, column: number, fixed: number) => {
    if (
      row < window.rowStart ||
      row >= window.rowEnd ||
      column < window.colStart ||
      column >= window.colEnd
    ) {
      return;
    }
    const rowBin = Math.floor(
      ((row - window.rowStart) * resolution) / (window.rowEnd - window.rowStart)
    );
    const columnBin = Math.floor(
      ((column - window.colStart) * resolution) / (window.colEnd - window.colStart)
    );
    const cell = rowBin * resolution + columnBin;
    counts[cell]++;
    weightSums[cell] = (weightSums[cell] + fixed) >>> 0;
  };
  for (let vertex = 0; vertex < nodeCount; vertex++) {
    if (options.vertexMask && !options.vertexMask[vertex]) continue;
    for (let slot = csr.offsets[vertex]; slot < csr.offsets[vertex + 1]; slot++) {
      const neighbor = csr.neighbors[slot];
      if (neighbor >= nodeCount) continue;
      if (options.vertexMask && !options.vertexMask[neighbor]) continue;
      if (options.edgeMask && !options.edgeMask[slot]) continue;
      const source = options.order ? options.order[vertex] : vertex;
      const target = options.order ? options.order[neighbor] : neighbor;
      const fixed = getGPUAdjacencyMatrixFixedWeight(csr.weights[slot], scale);
      place(source, target, fixed);
      if (mirror && source !== target) place(target, source, fixed);
    }
  }
  return {
    counts,
    weightSums,
    maxCount: counts.reduce((a, b) => Math.max(a, b), 0),
    maxWeightSum: weightSums.reduce((a, b) => Math.max(a, b), 0)
  };
}
