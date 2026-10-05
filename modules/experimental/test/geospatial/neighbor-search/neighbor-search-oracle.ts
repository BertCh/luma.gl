// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_NEIGHBOR_SEARCH_KERNEL,
  type GPUNeighborSearchParameters
} from '../../../src/geospatial/neighbor-search';

const f32 = Math.fround;

/** Inputs of {@link computeNeighborSearchOracle}. */
export type NeighborSearchOracleInput = {
  mode: 'knn' | 'radius';
  positions: Float32Array;
  queryPositions?: Float32Array;
  mask?: Uint32Array;
  queryMask?: Uint32Array;
  parameters: GPUNeighborSearchParameters;
  k?: number;
};

/** Unclamped CSR of every query row, the neighbors of each row ascending by ID. */
export type NeighborSearchOracleResult = {
  offsets: number[];
  neighbors: number[];
  /** f32 distances (`fround(sqrt(d^2))`). */
  distances: number[];
  /** f64 weights computed from the f32 distances. */
  weights: number[];
  counts: number[];
  /** f32 squared distance per slot, for tie diagnostics. */
  distancesSquared: number[];
};

/** mulberry32 generator with values in [0, 1). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Kernel or inverse-distance weight of one neighbor, as the WGSL `getNeighborWeight`. */
export function getNeighborWeight(
  distance: number,
  bandwidth: number,
  parameters: GPUNeighborSearchParameters
): number {
  const weightKind = parameters.weightKind ?? 'binary';
  let weight = 1;
  if (weightKind === 'inverseDistance') {
    weight = Math.max(distance, f32(parameters.distanceFloor ?? 0)) ** -f32(parameters.power ?? 1);
  } else if (weightKind === 'kernel') {
    const z = bandwidth > 0 ? distance / bandwidth : 0;
    switch (GPU_NEIGHBOR_SEARCH_KERNEL[parameters.kernel ?? 'triangular']) {
      case GPU_NEIGHBOR_SEARCH_KERNEL.gaussian:
        weight = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        break;
      case GPU_NEIGHBOR_SEARCH_KERNEL.triangular:
        weight = Math.max(1 - z, 0);
        break;
      case GPU_NEIGHBOR_SEARCH_KERNEL.epanechnikov:
        weight = 0.75 * Math.max(1 - z * z, 0);
        break;
      case GPU_NEIGHBOR_SEARCH_KERNEL.bisquare:
        weight = 0.9375 * Math.max(1 - z * z, 0) ** 2;
        break;
      default:
        weight = 0.5;
    }
  }
  return Number.isFinite(weight) && weight >= 0 && weight <= 3.4e38 ? weight : 0;
}

function isInside(x: number, y: number, bounds: readonly number[]): boolean {
  return (
    Number.isFinite(x) &&
    Number.isFinite(y) &&
    x >= bounds[0] &&
    x <= bounds[2] &&
    y >= bounds[1] &&
    y <= bounds[3]
  );
}

/**
 * Brute-force reference of `GPUNeighborSearch` with f32 squared distances, so neighbor IDs and
 * ordering match the GPU exactly.
 */
export function computeNeighborSearchOracle(
  input: NeighborSearchOracleInput
): NeighborSearchOracleResult {
  const {mode, positions, parameters} = input;
  const crossJoin = Boolean(input.queryPositions);
  const queries = input.queryPositions ?? positions;
  const queryMask = input.queryMask ?? (crossJoin ? undefined : input.mask);
  const bounds = parameters.bounds.map(f32);
  const radius = f32(parameters.radius ?? Infinity);
  const radiusSquared = f32(radius * radius);
  const radiusValid = Number.isFinite(radius) && Number.isFinite(radiusSquared) && radius > 0;
  const boundsValid =
    bounds.every(Number.isFinite) && bounds[2] - bounds[0] >= 0 && bounds[3] - bounds[1] >= 0;
  const latticeValid = boundsValid && (mode === 'knn' || radiusValid);
  const targetRows = positions.length / 2;
  const queryRows = queries.length / 2;
  const validTargets: number[] = [];
  for (let row = 0; row < targetRows; row++) {
    if (
      latticeValid &&
      (!input.mask || input.mask[row] !== 0) &&
      isInside(positions[row * 2], positions[row * 2 + 1], bounds)
    ) {
      validTargets.push(row);
    }
  }
  const result: NeighborSearchOracleResult = {
    offsets: [0],
    neighbors: [],
    distances: [],
    weights: [],
    counts: [],
    distancesSquared: []
  };
  for (let query = 0; query < queryRows; query++) {
    const x = queries[query * 2];
    const y = queries[query * 2 + 1];
    let candidates: {id: number; distanceSquared: number}[] = [];
    if (latticeValid && (!queryMask || queryMask[query] !== 0) && isInside(x, y, bounds)) {
      for (const target of validTargets) {
        if (!crossJoin && target === query) {
          continue;
        }
        const deltaX = f32(positions[target * 2] - x);
        const deltaY = f32(positions[target * 2 + 1] - y);
        const distanceSquared = f32(f32(deltaX * deltaX) + f32(deltaY * deltaY));
        const bounded = mode === 'radius' || radiusValid;
        if (!bounded || distanceSquared <= radiusSquared) {
          candidates.push({id: target, distanceSquared});
        }
      }
      if (mode === 'knn') {
        candidates.sort((a, b) => a.distanceSquared - b.distanceSquared || a.id - b.id);
        candidates = candidates.slice(0, input.k ?? 1);
      }
      candidates.sort((a, b) => a.id - b.id);
    }
    const distances = candidates.map(candidate => f32(Math.sqrt(candidate.distanceSquared)));
    const bandwidth = mode === 'radius' ? radius : Math.max(0, ...distances);
    let weights = distances.map(distance => getNeighborWeight(distance, bandwidth, parameters));
    const sum = weights.reduce((total, weight) => total + weight, 0);
    if (parameters.rowStandardize && sum > 0) {
      weights = weights.map(weight => weight / sum);
    }
    result.counts.push(candidates.length);
    result.neighbors.push(...candidates.map(candidate => candidate.id));
    result.distancesSquared.push(...candidates.map(candidate => candidate.distanceSquared));
    result.distances.push(...distances);
    result.weights.push(...weights);
    result.offsets.push(result.neighbors.length);
  }
  return result;
}
