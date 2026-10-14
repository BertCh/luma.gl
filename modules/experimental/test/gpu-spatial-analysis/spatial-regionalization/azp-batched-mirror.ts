// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU mirror (test oracle only) of the local articulation test and the batched move selection of
 * `GPUAZPRegions`. It follows the WGSL step by step with `Set`s in place of the private hash table,
 * and float64 arithmetic in place of float32.
 */
import {
  GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY,
  GPU_AZP_MAXIMUM_SCAN_DEGREE
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions';

/** Undirected adjacency lists without duplicates or self links. */
export type AzpAdjacency = readonly (readonly number[])[];

/** Converts a CSR to deduplicated, symmetrized adjacency lists. */
export function getAdjacency(offsets: Uint32Array, neighbors: Uint32Array): number[][] {
  const rowCount = offsets.length - 1;
  const sets = Array.from({length: rowCount}, () => new Set<number>());
  for (let row = 0; row < rowCount; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (neighbor !== row && neighbor < rowCount) {
        sets[row].add(neighbor);
        sets[neighbor].add(row);
      }
    }
  }
  return sets.map(set => [...set].sort((a, b) => a - b));
}

/** Packs adjacency lists into a CSR. */
export function getCSRFromAdjacency(adjacency: AzpAdjacency): {
  offsets: Uint32Array;
  neighbors: Uint32Array;
} {
  const offsets = new Uint32Array(adjacency.length + 1);
  adjacency.forEach((list, row) => (offsets[row + 1] = offsets[row] + list.length));
  return {offsets, neighbors: Uint32Array.from(adjacency.flat())};
}

/** Whether the rows of `region` form one connected set (full breadth-first search). */
export function isRegionConnected(
  adjacency: AzpAdjacency,
  labels: ArrayLike<number>,
  region: number,
  excludedRow: number = -1
): boolean {
  const members: number[] = [];
  for (let row = 0; row < adjacency.length; row++) {
    if (labels[row] === region && row !== excludedRow) members.push(row);
  }
  if (members.length === 0) return true;
  const seen = new Set<number>([members[0]]);
  const stack = [members[0]];
  while (stack.length) {
    const current = stack.pop()!;
    for (const neighbor of adjacency[current]) {
      if (neighbor !== excludedRow && labels[neighbor] === region && !seen.has(neighbor)) {
        seen.add(neighbor);
        stack.push(neighbor);
      }
    }
  }
  return seen.size === members.length;
}

/**
 * Mirror of `isRemovalSafe`: true only when the same-region neighbors of `row` all lie in one
 * component of the subgraph induced by themselves plus their same-region neighbors (without `row`).
 */
export function isRemovalSafeMirror(
  adjacency: AzpAdjacency,
  labels: ArrayLike<number>,
  row: number,
  capacity: number = GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY,
  maximumScanDegree: number = GPU_AZP_MAXIMUM_SCAN_DEGREE
): boolean {
  const source = labels[row];
  const list: number[] = [];
  const index = new Map<number, number>();
  const add = (vertex: number): boolean => {
    if (index.has(vertex)) return true;
    if (list.length >= capacity) return false;
    index.set(vertex, list.length);
    list.push(vertex);
    return true;
  };
  for (const neighbor of adjacency[row]) {
    if (labels[neighbor] === source && !add(neighbor)) return false;
  }
  const ringCount = list.length;
  if (ringCount < 2) return ringCount === 1;
  for (let position = 0; position < ringCount; position++) {
    const member = list[position];
    if (adjacency[member].length > maximumScanDegree) return false;
    for (const neighbor of adjacency[member]) {
      if (neighbor !== row && labels[neighbor] === source && !add(neighbor)) return false;
    }
  }
  const visited = new Set<number>([0]);
  const stack = [0];
  let reachedRing = 1;
  while (stack.length) {
    const current = list[stack.pop()!];
    if (adjacency[current].length > maximumScanDegree) return false;
    for (const neighbor of adjacency[current]) {
      const found = index.get(neighbor);
      if (found !== undefined && !visited.has(found)) {
        visited.add(found);
        stack.push(found);
        if (found < ringCount) reachedRing++;
      }
    }
    if (reachedRing === ringCount) return true;
  }
  return false;
}

/** Mirror of the WGSL `tieKey`, a bijection of the row. */
export function getTieKey(row: number, seed: number): number {
  let value = (row ^ seed) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

/** One candidate move of a round. */
export type AzpMove = {row: number; source: number; destination: number; gain: number};

/**
 * Mirror of the claim-gain, claim-key and select kernels: the moves holding the largest
 * (gain, tie key) of both their regions.
 */
export function selectIndependentMoves(
  moves: readonly AzpMove[],
  seed: number,
  regionCapacity: number
): AzpMove[] {
  const claimGain = new Float64Array(regionCapacity);
  const claimKey = new Float64Array(regionCapacity);
  for (const move of moves) {
    claimGain[move.source] = Math.max(claimGain[move.source], move.gain);
    claimGain[move.destination] = Math.max(claimGain[move.destination], move.gain);
  }
  const contenders = moves.filter(
    move => claimGain[move.source] === move.gain && claimGain[move.destination] === move.gain
  );
  for (const move of contenders) {
    const key = getTieKey(move.row, seed);
    claimKey[move.source] = Math.max(claimKey[move.source], key);
    claimKey[move.destination] = Math.max(claimKey[move.destination], key);
  }
  return contenders.filter(move => {
    const key = getTieKey(move.row, seed);
    return claimKey[move.source] === key && claimKey[move.destination] === key;
  });
}

/** Within-region sum of squares of one column. */
export function getWithinSumOfSquares(
  values: ArrayLike<number>,
  labels: ArrayLike<number>,
  regionCapacity: number
): number {
  const sums = new Float64Array(regionCapacity);
  const counts = new Float64Array(regionCapacity);
  let total = 0;
  for (let row = 0; row < labels.length; row++) {
    sums[labels[row]] += values[row];
    counts[labels[row]]++;
    total += values[row] * values[row];
  }
  for (let region = 0; region < regionCapacity; region++) {
    if (counts[region] > 0) total -= (sums[region] * sums[region]) / counts[region];
  }
  return total;
}

export type AzpMirrorRound = {
  moves: AzpMove[];
  selected: AzpMove[];
  objectiveBefore: number;
  objectiveAfter: number;
  /** Safe-by-full-search moves the local test refused this round. */
  refusedSafeCount: number;
  acceptedCount: number;
};

/** Runs the batched local search on the CPU (one column, no extensive constraint). */
export function runBatchedAzpMirror(props: {
  adjacency: AzpAdjacency;
  values: ArrayLike<number>;
  labels: ArrayLike<number>;
  regionCapacity: number;
  seed?: number;
  minimumRegionSize?: number;
  maximumRounds?: number;
}): {labels: number[]; rounds: AzpMirrorRound[]; converged: boolean} {
  const {adjacency, values, regionCapacity} = props;
  const seed = props.seed ?? 0;
  const minimumRegionSize = props.minimumRegionSize ?? 1;
  const labels = Array.from(props.labels);
  const rounds: AzpMirrorRound[] = [];
  for (let round = 0; round < (props.maximumRounds ?? 256); round++) {
    const sums = new Float64Array(regionCapacity);
    const counts = new Float64Array(regionCapacity);
    labels.forEach((label, row) => {
      sums[label] += values[row];
      counts[label]++;
    });
    const moves: AzpMove[] = [];
    let refusedSafeCount = 0;
    let acceptedCount = 0;
    for (let row = 0; row < labels.length; row++) {
      const source = labels[row];
      if (!(counts[source] > minimumRegionSize)) continue;
      const value = values[row];
      let best: {delta: number; scale: number; destination: number} | undefined;
      for (const neighbor of adjacency[row]) {
        const destination = labels[neighbor];
        if (destination === source) continue;
        const terms = [
          (sums[source] * sums[source]) / counts[source],
          (sums[destination] * sums[destination]) / counts[destination],
          ((sums[source] - value) * (sums[source] - value)) / (counts[source] - 1),
          ((sums[destination] + value) * (sums[destination] + value)) / (counts[destination] + 1)
        ];
        const delta = terms[0] + terms[1] - terms[2] - terms[3];
        const scale = terms[0] + terms[1] + terms[2] + terms[3];
        if (
          !best ||
          delta < best.delta ||
          (delta === best.delta && destination < best.destination)
        ) {
          best = {delta, scale, destination};
        }
      }
      if (!best || !(best.delta < -4e-6 * best.scale)) continue;
      if (isRemovalSafeMirror(adjacency, labels, row)) {
        acceptedCount++;
        moves.push({row, source, destination: best.destination, gain: -best.delta});
      } else if (isRegionConnected(adjacency, labels, source, row)) {
        refusedSafeCount++;
      }
    }
    const objectiveBefore = getWithinSumOfSquares(values, labels, regionCapacity);
    const selected = selectIndependentMoves(moves, seed, regionCapacity);
    for (const move of selected) labels[move.row] = move.destination;
    const objectiveAfter = getWithinSumOfSquares(values, labels, regionCapacity);
    rounds.push({
      moves,
      selected,
      objectiveBefore,
      objectiveAfter,
      refusedSafeCount,
      acceptedCount
    });
    if (selected.length === 0) return {labels, rounds, converged: true};
  }
  return {labels, rounds, converged: false};
}

/**
 * Mirror of the initial-contiguity kernels (hook, apply, jump rounds, final verification). Returns
 * the `invalidCount` contribution and the number of hook rounds that changed something.
 */
export function runContiguityMirror(
  adjacency: AzpAdjacency,
  labels: ArrayLike<number>,
  regionCapacity: number,
  roundBound: number = 2 * Math.ceil(Math.log2(Math.max(adjacency.length, 2))) + 8
): {invalidCount: number; activeRounds: number; parent: number[]} {
  const rowCount = adjacency.length;
  const jumps = Math.ceil(Math.log2(Math.max(rowCount, 2))) + 1;
  const parent = Array.from({length: rowCount}, (_, row) => row);
  let activeRounds = 0;
  for (let round = 0; round < roundBound; round++) {
    const hooks = Array.from({length: rowCount}, (_, row) => row);
    for (let row = 0; row < rowCount; row++) {
      if (labels[row] >= regionCapacity) continue;
      for (const neighbor of adjacency[row]) {
        if (labels[neighbor] !== labels[row] || parent[neighbor] === parent[row]) continue;
        const high = Math.max(parent[row], parent[neighbor]);
        hooks[high] = Math.min(hooks[high], Math.min(parent[row], parent[neighbor]));
      }
    }
    let changed = false;
    for (let row = 0; row < rowCount; row++) {
      if (hooks[row] < row) {
        parent[row] = hooks[row];
        changed = true;
      }
    }
    if (!changed) break;
    activeRounds++;
    for (let jump = 0; jump < jumps; jump++) {
      const next = parent.map(own => parent[own]);
      next.forEach((value, row) => (parent[row] = value));
    }
  }
  const firstRow = new Map<number, number>();
  for (let row = 0; row < rowCount; row++) {
    if (labels[row] < regionCapacity && !firstRow.has(labels[row])) firstRow.set(labels[row], row);
  }
  let invalidCount = 0;
  for (let row = 0; row < rowCount; row++) {
    if (labels[row] >= regionCapacity) continue;
    const root = parent[row];
    let invalid = parent[root] !== root ? 1 : 0;
    for (const neighbor of adjacency[row]) {
      if (labels[neighbor] === labels[row] && parent[neighbor] !== root) invalid = 1;
    }
    if (root === row && firstRow.get(labels[row]) !== row) invalid++;
    invalidCount += invalid;
  }
  return {invalidCount, activeRounds, parent};
}
