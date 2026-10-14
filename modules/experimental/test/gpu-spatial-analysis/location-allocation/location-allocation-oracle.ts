// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU oracles for `GPULocationAllocation`: greedy, the Teitz-Bart swap phase and brute force. */

/** Mirrors `GPU_LOCATION_ALLOCATION_SWAP_TOLERANCE` (a swap must beat this fraction of the scale). */
const SWAP_TOLERANCE = 1e-6;
const LARGE_COST = 1e30;

export type LocationAllocationOracleOperation = 'p-median' | 'maximum-coverage';

export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** Integer cost matrix, `demandCount` rows by `facilityCount` columns, values in [0, 30]. */
export function createCostMatrix(
  seed: number,
  demandCount: number,
  facilityCount: number
): Float32Array {
  const random = createSeededRandom(seed);
  return Float32Array.from({length: demandCount * facilityCount}, () => Math.floor(random() * 31));
}

/**
 * Cost matrix with unreachable (`+Infinity`) entries: about a third of the entries at random, and
 * every entry of row 0, so at least one demand row is unreachable from every facility set.
 */
export function createPartlyReachableCostMatrix(
  seed: number,
  demandCount: number,
  facilityCount: number
): Float32Array {
  const random = createSeededRandom(seed ^ 0x5eed);
  const costs = createCostMatrix(seed, demandCount, facilityCount);
  for (let index = 0; index < costs.length; index++) {
    if (index < facilityCount || random() < 0.35) {
      costs[index] = Infinity;
    }
  }
  return costs;
}

/**
 * The classic float32 sentinel failure: row 0 is unreachable, rows 1 and 2 cost 9 at facility 0 and
 * 1 at facility 1. Summing a 1e30 sentinel makes both one-facility scores equal in float32, so the
 * lowest index (facility 0, cost 18) wins; the lexicographic rule picks facility 1 (cost 2).
 */
export const UNREACHABLE_SENTINEL_CASE = {
  costs: Float32Array.from([Infinity, Infinity, 9, 1, 9, 1]),
  facilityCount: 2,
  expectedFacility: 1,
  expectedObjective: 2
};

/**
 * p-median objective of a facility set as the GPU reports it: the cost of the reachable rows, plus
 * the number of rows with no reachable selected facility (summed separately, never as a sentinel).
 */
export function getPMedianObjectiveParts(
  costs: Float32Array,
  facilityCount: number,
  selected: readonly number[]
): {unreachable: number; cost: number} {
  let unreachable = 0;
  let cost = 0;
  for (let row = 0; row < costs.length / facilityCount; row++) {
    const nearest = Math.min(
      LARGE_COST,
      ...selected.map(facility => costs[row * facilityCount + facility])
    );
    if (nearest < LARGE_COST) {
      cost += nearest;
    } else {
      unreachable += 1;
    }
  }
  return {unreachable, cost};
}

/**
 * Objective of a facility set: weighted cost of the reachable rows (p-median) or covered weight
 * (maximum coverage).
 */
export function getObjective(
  operation: LocationAllocationOracleOperation,
  costs: Float32Array,
  facilityCount: number,
  selected: readonly number[],
  distance: number
): number {
  if (operation === 'p-median') {
    return getPMedianObjectiveParts(costs, facilityCount, selected).cost;
  }
  let total = 0;
  for (let row = 0; row < costs.length / facilityCount; row++) {
    if (selected.some(facility => costs[row * facilityCount + facility] <= distance)) {
      total += 1;
    }
  }
  return total;
}

/**
 * The documented greedy rule: best marginal gain, ties to the lowest facility row. p-median
 * compares (rows left unreachable, reachable cost) lexicographically.
 */
export function runGreedyOnCPU(
  operation: LocationAllocationOracleOperation,
  costs: Float32Array,
  facilityCount: number,
  maximumFacilityCount: number,
  distance: number
): number[] {
  const demandCount = costs.length / facilityCount;
  const best = new Array<number>(demandCount).fill(LARGE_COST);
  const selected: number[] = [];
  for (let step = 0; step < maximumFacilityCount; step++) {
    let chosen = -1;
    let chosenScore = operation === 'p-median' ? Infinity : 0;
    let chosenCost = Infinity;
    for (let facility = 0; facility < facilityCount; facility++) {
      if (selected.includes(facility)) continue;
      let score = 0;
      let cost = 0;
      for (let row = 0; row < demandCount; row++) {
        const candidate = costs[row * facilityCount + facility];
        if (operation === 'p-median') {
          const next = Math.min(best[row], candidate);
          if (next < LARGE_COST) {
            cost += next;
          } else {
            score += 1;
          }
        } else if (best[row] > distance && candidate <= distance) {
          score += 1;
        }
      }
      const improves =
        operation === 'p-median'
          ? score < chosenScore || (score === chosenScore && cost < chosenCost)
          : score > chosenScore;
      if (improves) {
        chosen = facility;
        chosenScore = score;
        chosenCost = cost;
      }
    }
    if (chosen < 0) break;
    selected.push(chosen);
    for (let row = 0; row < demandCount; row++) {
      best[row] = Math.min(best[row], costs[row * facilityCount + chosen]);
    }
  }
  return selected;
}

export type SwapOracleResult = {
  /** Selected facilities, ascending. */
  selected: number[];
  swapIterations: number;
  converged: boolean;
  iterationLimitReached: boolean;
};

/**
 * CPU mirror of the GPU swap phase. Each iteration scores every (selected facility, unselected
 * candidate) pair by objective improvement, applies the best pair whose improvement exceeds
 * `SWAP_TOLERANCE * scale`, with ties to the lowest selected facility and then the lowest
 * candidate, and stops when none qualifies or after `maximumSwapIterations` applied swaps.
 */
export function runSwapPhaseOnCPU(
  operation: LocationAllocationOracleOperation,
  costs: Float32Array,
  facilityCount: number,
  initialSelected: readonly number[],
  distance: number,
  maximumSwapIterations: number
): SwapOracleResult {
  const selected = [...initialSelected].sort((a, b) => a - b);
  const demandCount = costs.length / facilityCount;
  let swapIterations = 0;
  let converged = false;
  while (swapIterations < maximumSwapIterations) {
    const before = getObjective(operation, costs, facilityCount, selected, distance);
    let scale = 0;
    for (let row = 0; row < demandCount; row++) {
      if (operation === 'p-median') {
        const current = Math.min(
          ...selected.map(f => Math.min(costs[row * facilityCount + f], LARGE_COST))
        );
        if (current < 1e29) scale += current;
      } else {
        scale += 1;
      }
    }
    const beforeUnreachable =
      operation === 'p-median'
        ? getPMedianObjectiveParts(costs, facilityCount, selected).unreachable
        : 0;
    // Lexicographic (reach gain, cost improvement) for p-median, as the GPU scores pairs.
    let bestPrimary = -1;
    let bestImprovement = -1;
    let bestOut = -1;
    let bestIn = -1;
    for (const removed of selected) {
      for (let candidate = 0; candidate < facilityCount; candidate++) {
        if (selected.includes(candidate)) continue;
        const next = selected.filter(f => f !== removed).concat(candidate);
        const after = getObjective(operation, costs, facilityCount, next, distance);
        let primary = -1;
        let improvement = operation === 'p-median' ? before - after : after - before;
        if (operation === 'p-median') {
          const reachGain =
            beforeUnreachable - getPMedianObjectiveParts(costs, facilityCount, next).unreachable;
          if (reachGain > SWAP_TOLERANCE * demandCount) {
            primary = reachGain;
          } else if (
            Math.abs(reachGain) <= SWAP_TOLERANCE * demandCount &&
            improvement > SWAP_TOLERANCE * scale
          ) {
            primary = 0;
          }
        } else if (improvement > SWAP_TOLERANCE * scale) {
          primary = improvement;
          improvement = 0;
        }
        if (primary < 0) continue;
        if (primary > bestPrimary || (primary === bestPrimary && improvement > bestImprovement)) {
          bestPrimary = primary;
          bestImprovement = improvement;
          bestOut = removed;
          bestIn = candidate;
        }
      }
    }
    if (bestOut < 0) {
      converged = true;
      break;
    }
    selected.splice(selected.indexOf(bestOut), 1, bestIn);
    selected.sort((a, b) => a - b);
    swapIterations++;
  }
  return {
    selected,
    swapIterations,
    converged,
    iterationLimitReached: !converged && swapIterations >= maximumSwapIterations
  };
}

export function getSubsets(facilityCount: number, size: number): number[][] {
  const subsets: number[][] = [];
  const visit = (start: number, chosen: number[]) => {
    if (chosen.length === size) {
      subsets.push(chosen);
      return;
    }
    for (let facility = start; facility < facilityCount; facility++) {
      visit(facility + 1, [...chosen, facility]);
    }
  };
  visit(0, []);
  return subsets;
}

/** Brute-force optimum over every facility subset of the given size. */
export function getBruteForceOptimum(
  operation: LocationAllocationOracleOperation,
  costs: Float32Array,
  facilityCount: number,
  size: number,
  distance: number
): number {
  const values = getSubsets(facilityCount, size).map(subset =>
    getObjective(operation, costs, facilityCount, subset, distance)
  );
  return operation === 'p-median' ? Math.min(...values) : Math.max(...values);
}
