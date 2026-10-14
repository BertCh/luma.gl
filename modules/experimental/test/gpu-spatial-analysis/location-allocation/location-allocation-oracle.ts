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

/** One instance of `p-median-fixture.json`, with `null` costs decoded to `+Infinity`. */
export type PMedianFixtureInstance = {
  demandCount: number;
  facilityCount: number;
  maximumFacilityCount: number;
  costs: Float32Array;
  demandWeights: Float32Array;
  /** Exact optimum (scipy.optimize.milp): demand weight left unreachable, then reachable cost. */
  optimalUnreachableWeight: number;
  optimalCost: number;
};

/** Decodes the instances of `p-median-fixture.json` (written by `p-median-oracle.py`). */
export function decodePMedianFixture(fixture: {
  instances: readonly {
    demandCount: number;
    facilityCount: number;
    maximumFacilityCount: number;
    costs: readonly (number | null)[];
    demandWeights: readonly number[];
    optimalUnreachableWeight: number;
    optimalCost: number;
  }[];
}): PMedianFixtureInstance[] {
  return fixture.instances.map(instance => ({
    ...instance,
    costs: Float32Array.from(instance.costs, value => (value === null ? Infinity : value)),
    demandWeights: Float32Array.from(instance.demandWeights)
  }));
}

/** Weighted p-median objective parts of a facility set: unreachable weight, reachable cost. */
export function getWeightedPMedianParts(
  costs: Float32Array,
  facilityCount: number,
  demandWeights: Float32Array,
  selected: readonly number[]
): {unreachableWeight: number; cost: number} {
  let unreachableWeight = 0;
  let cost = 0;
  for (let row = 0; row < demandWeights.length; row++) {
    const nearest = Math.min(...selected.map(facility => costs[row * facilityCount + facility]));
    if (Number.isFinite(nearest)) {
      cost += demandWeights[row] * nearest;
    } else {
      unreachableWeight += demandWeights[row];
    }
  }
  return {unreachableWeight, cost};
}

/** Lexicographic comparison of weighted p-median parts: negative when `left` is better. */
export function compareWeightedPMedianParts(
  left: {unreachableWeight: number; cost: number},
  right: {unreachableWeight: number; cost: number}
): number {
  return left.unreachableWeight - right.unreachableWeight || left.cost - right.cost;
}

/** Exact weighted p-median optimum over every facility subset of the given size. */
export function getWeightedPMedianBruteForceOptimum(
  costs: Float32Array,
  facilityCount: number,
  demandWeights: Float32Array,
  size: number
): {unreachableWeight: number; cost: number} {
  let best: {unreachableWeight: number; cost: number} | null = null;
  for (const subset of getSubsets(facilityCount, size)) {
    const parts = getWeightedPMedianParts(costs, facilityCount, demandWeights, subset);
    if (!best || compareWeightedPMedianParts(parts, best) < 0) {
      best = parts;
    }
  }
  return best!;
}

/** True when no single (selected, unselected) swap strictly improves the weighted objective. */
export function isWeightedPMedianSwapLocalOptimum(
  costs: Float32Array,
  facilityCount: number,
  demandWeights: Float32Array,
  selected: readonly number[]
): boolean {
  const current = getWeightedPMedianParts(costs, facilityCount, demandWeights, selected);
  for (const removed of selected) {
    for (let candidate = 0; candidate < facilityCount; candidate++) {
      if (selected.includes(candidate)) continue;
      const next = selected.filter(facility => facility !== removed).concat(candidate);
      const parts = getWeightedPMedianParts(costs, facilityCount, demandWeights, next);
      if (compareWeightedPMedianParts(parts, current) < 0) {
        return false;
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Swap gain decomposition (Whitaker; Resende and Werneck) and sparse costs: CPU mirrors.
// ---------------------------------------------------------------------------------------------

/** CSR sparse costs as `GPULocationAllocation` takes them. */
export type SparseCostsArrays = {
  offsets: Uint32Array;
  facilities: Uint32Array;
  costs: Float32Array;
};

/**
 * Converts a dense matrix to CSR, dropping every `+Infinity` entry (an unlisted pair is
 * unreachable). `keepInfinite` lists the entries selected by the predicate with their `+Infinity`
 * cost, which must behave the same as unlisted ones.
 */
export function toSparseCosts(
  costs: Float32Array,
  facilityCount: number,
  keepInfinite: (row: number, facility: number) => boolean = () => false
): SparseCostsArrays {
  const demandCount = costs.length / facilityCount;
  const offsets: number[] = [0];
  const facilities: number[] = [];
  const values: number[] = [];
  for (let row = 0; row < demandCount; row++) {
    for (let facility = 0; facility < facilityCount; facility++) {
      const cost = costs[row * facilityCount + facility];
      if (Number.isFinite(cost) || keepInfinite(row, facility)) {
        facilities.push(facility);
        values.push(cost);
      }
    }
    offsets.push(facilities.length);
  }
  return {
    offsets: Uint32Array.from(offsets),
    facilities: Uint32Array.from(facilities),
    costs: Float32Array.from(values)
  };
}

/** Expands CSR costs to the dense matrix (`+Infinity` where unlisted). */
export function toDenseCosts(sparse: SparseCostsArrays, facilityCount: number): Float32Array {
  const demandCount = sparse.offsets.length - 1;
  const costs = new Float32Array(demandCount * facilityCount).fill(Infinity);
  for (let row = 0; row < demandCount; row++) {
    for (let entry = sparse.offsets[row]; entry < sparse.offsets[row + 1]; entry++) {
      costs[row * facilityCount + sparse.facilities[entry]] = sparse.costs[entry];
    }
  }
  return costs;
}

/** Facility-major transpose of CSR costs with ascending demand rows, as the GPU builds it. */
export function transposeSparseCosts(
  sparse: SparseCostsArrays,
  facilityCount: number
): {offsets: number[]; entries: {demand: number; cost: number}[]} {
  const columns: {demand: number; cost: number}[][] = Array.from({length: facilityCount}, () => []);
  for (let row = 0; row < sparse.offsets.length - 1; row++) {
    for (let entry = sparse.offsets[row]; entry < sparse.offsets[row + 1]; entry++) {
      columns[sparse.facilities[entry]].push({demand: row, cost: sparse.costs[entry]});
    }
  }
  const offsets = [0];
  for (const column of columns) offsets.push(offsets[offsets.length - 1] + column.length);
  return {offsets, entries: columns.flat()};
}

/** A swap's improvement: p-median (reachable weight gained, cost removed) or coverage (weight, 0). */
export type SwapScore = {primary: number; secondary: number};

/** Weighted objective parts of a facility set: p-median (unreachable weight, cost) or coverage. */
function getWeightedParts(
  operation: LocationAllocationOracleOperation,
  costs: Float32Array,
  facilityCount: number,
  weights: Float32Array,
  selected: readonly number[],
  distance: number
): {unreachableWeight: number; cost: number; coveredWeight: number} {
  let unreachableWeight = 0;
  let cost = 0;
  let coveredWeight = 0;
  for (let row = 0; row < weights.length; row++) {
    const nearest = Math.min(
      LARGE_COST,
      ...selected.map(facility => costs[row * facilityCount + facility])
    );
    if (nearest < LARGE_COST) cost += weights[row] * nearest;
    else unreachableWeight += weights[row];
    if (operation === 'maximum-coverage' && nearest <= distance) coveredWeight += weights[row];
  }
  return {unreachableWeight, cost, coveredWeight};
}

/**
 * Swap score from the definition: the objective of the swapped set against the current one. p-median
 * returns (unreachable weight removed, reachable cost removed); coverage returns (covered weight
 * gained, 0).
 */
export function getBruteForceSwapScore(
  operation: LocationAllocationOracleOperation,
  costs: Float32Array,
  facilityCount: number,
  weights: Float32Array,
  selected: readonly number[],
  removed: number,
  candidate: number,
  distance: number
): SwapScore {
  const before = getWeightedParts(operation, costs, facilityCount, weights, selected, distance);
  const next = selected.filter(facility => facility !== removed).concat(candidate);
  const after = getWeightedParts(operation, costs, facilityCount, weights, next, distance);
  return operation === 'p-median'
    ? {
        primary: before.unreachableWeight - after.unreachableWeight,
        secondary: before.cost - after.cost
      }
    : {primary: after.coveredWeight - before.coveredWeight, secondary: 0};
}

/** The facility entries the decomposition scans: `(demand, cost)` per listed pair. */
export type EntryColumns = readonly (readonly {demand: number; cost: number}[])[];

/** All demand rows of every facility column (dense), as the dense GPU score kernel scans them. */
export function getDenseEntryColumns(costs: Float32Array, facilityCount: number): EntryColumns {
  const demandCount = costs.length / facilityCount;
  return Array.from({length: facilityCount}, (_, facility) =>
    Array.from({length: demandCount}, (_, demand) => ({
      demand,
      cost: costs[demand * facilityCount + facility]
    }))
  );
}

/** The listed entries of every facility column (sparse), ascending by demand row. */
export function getSparseEntryColumns(
  sparse: SparseCostsArrays,
  facilityCount: number
): EntryColumns {
  const {offsets, entries} = transposeSparseCosts(sparse, facilityCount);
  return Array.from({length: facilityCount}, (_, facility) =>
    entries.slice(offsets[facility], offsets[facility + 1])
  );
}

/**
 * CPU mirror of the GPU swap scoring (Resende and Werneck, "A fast swap-based local search
 * procedure for location problems", Annals of Operations Research 2007, which restates Whitaker's
 * fast interchange): per demand row the nearest selected cost `d1` with its facility `r1`, the
 * second nearest `d2` and the weight `w`, then
 *
 *   profit(i, r) = gain(i) - loss(r) + extra(i, r)
 *   gain(i)      = sum_j w * max(0, d1 - c_ij)
 *   loss(r)      = sum_{j: r1(j) = r} w * (d2 - d1)
 *   extra(i, r)  = sum_{j: r1(j) = r, c_ij < d2} w * (d2 - max(c_ij, d1))
 *
 * extended with the library's lexicographic unreachable-demand rule: rows whose `d1` or `d2` is
 * unreachable contribute to a separate reachable-weight term (primary) and a cost term
 * (secondary); the dense kernel evaluates the net per-row form (`options.netTerms`), the sparse one
 * the loss/extra form. `columns` is what the kernel scans, so dense and sparse columns must agree. Maximum
 * coverage uses the same shape with `gain(i)` = weight of the uncovered rows `i` covers,
 * `loss(r)` = weight covered only by `r` and `extra(i, r)` = the part of that weight `i` covers.
 */
export function getDecomposedSwapScores(
  operation: LocationAllocationOracleOperation,
  columns: EntryColumns,
  facilityCount: number,
  weights: Float32Array,
  selected: readonly number[],
  distance: number,
  options: {
    /**
     * Net form (dense kernel): every row served by the removed facility adds its own difference
     * `ct(min(d1, c)) - ct(min(d2, c))`, with no separate loss(r) and extra(i, r). Needs columns
     * that list every demand row.
     */
    netTerms?: boolean;
    /** Round every operation to float32 as the kernels do. */
    float32?: boolean;
  } = {}
): {scores: Map<string, SwapScore>; scale: number; totalWeight: number} {
  const sorted = [...selected].sort((a, b) => a - b);
  const demandCount = weights.length;
  // Per row: nearest and second-nearest selected cost through the same columns.
  const costOf = (facility: number, demand: number): number => {
    const entry = columns[facility].find(candidate => candidate.demand === demand);
    return entry ? Math.min(entry.cost, LARGE_COST) : LARGE_COST;
  };
  const nearest = new Array<number>(demandCount).fill(LARGE_COST);
  const second = new Array<number>(demandCount).fill(LARGE_COST);
  const slotOfNearest = new Array<number>(demandCount).fill(-1);
  const coveringCount = new Array<number>(demandCount).fill(0);
  const soleSlot = new Array<number>(demandCount).fill(-1);
  for (let demand = 0; demand < demandCount; demand++) {
    for (const [slot, facility] of sorted.entries()) {
      const cost = costOf(facility, demand);
      if (cost < nearest[demand]) {
        second[demand] = nearest[demand];
        nearest[demand] = cost;
        slotOfNearest[demand] = slot;
      } else if (cost < second[demand]) {
        second[demand] = cost;
      }
      if (cost <= distance) {
        coveringCount[demand]++;
        soleSlot[demand] = slot;
      }
    }
  }
  const reachable = (cost: number) => cost < LARGE_COST;
  // float32 mode rounds every product and sum like the kernels do (Math.fround).
  const round = options.float32 ? Math.fround : (value: number) => value;
  let scale = 0;
  let totalWeight = 0;
  const removalReach = new Array<number>(sorted.length).fill(0);
  const removalCost = new Array<number>(sorted.length).fill(0);
  for (let demand = 0; demand < demandCount; demand++) {
    const weight = weights[demand];
    totalWeight = round(totalWeight + weight);
    if (operation === 'maximum-coverage') {
      scale = round(scale + weight);
      if (coveringCount[demand] === 1 && !options.netTerms) {
        removalReach[soleSlot[demand]] = round(removalReach[soleSlot[demand]] - weight);
      }
    } else {
      if (reachable(nearest[demand])) scale = round(scale + round(weight * nearest[demand]));
      const slot = slotOfNearest[demand];
      if (slot >= 0 && !options.netTerms) {
        if (reachable(second[demand])) {
          removalCost[slot] = round(
            removalCost[slot] + round(weight * round(nearest[demand] - second[demand]))
          );
        } else {
          removalReach[slot] = round(removalReach[slot] - weight);
          removalCost[slot] = round(removalCost[slot] + round(weight * nearest[demand]));
        }
      }
    }
  }
  const scores = new Map<string, SwapScore>();
  for (let candidate = 0; candidate < facilityCount; candidate++) {
    if (sorted.includes(candidate)) continue;
    let gainReach = 0;
    let gainCost = 0;
    const extraReach = new Array<number>(sorted.length).fill(0);
    const extraCost = new Array<number>(sorted.length).fill(0);
    for (const {demand, cost} of columns[candidate]) {
      const weight = weights[demand];
      const c = Math.min(cost, LARGE_COST);
      const slot = operation === 'maximum-coverage' ? soleSlot[demand] : slotOfNearest[demand];
      if (operation === 'maximum-coverage') {
        if (coveringCount[demand] === 0) {
          if (c <= distance) gainReach = round(gainReach + weight);
        } else if (coveringCount[demand] === 1) {
          if (options.netTerms) {
            if (c > distance) extraReach[slot] = round(extraReach[slot] - weight);
          } else if (c <= distance) {
            extraReach[slot] = round(extraReach[slot] + weight);
          }
        }
        continue;
      }
      if (c < nearest[demand]) {
        if (reachable(nearest[demand])) {
          gainCost = round(gainCost + round(weight * round(nearest[demand] - c)));
        } else {
          gainReach = round(gainReach + weight);
          gainCost = round(gainCost - round(weight * c));
        }
      } else if (options.netTerms) {
        // Net form: the row falls back to min(second, c); every term is a modest difference.
        const kept = Math.min(second[demand], c);
        if (slot >= 0 && kept > nearest[demand]) {
          if (reachable(kept)) {
            extraCost[slot] = round(
              extraCost[slot] + round(weight * round(nearest[demand] - kept))
            );
          } else {
            extraReach[slot] = round(extraReach[slot] - weight);
            extraCost[slot] = round(extraCost[slot] + round(weight * nearest[demand]));
          }
        }
        continue;
      }
      if (!options.netTerms && slot >= 0 && c < second[demand]) {
        if (reachable(second[demand])) {
          extraCost[slot] = round(
            extraCost[slot] + round(weight * round(second[demand] - Math.max(c, nearest[demand])))
          );
        } else {
          extraReach[slot] = round(extraReach[slot] + weight);
          extraCost[slot] = round(
            extraCost[slot] +
              round(weight * round(round(Math.min(nearest[demand], c) - c) - nearest[demand]))
          );
        }
      }
    }
    for (const [slot, removed] of sorted.entries()) {
      scores.set(`${removed}:${candidate}`, {
        primary: round(round(gainReach + removalReach[slot]) + extraReach[slot]),
        secondary: round(round(gainCost + removalCost[slot]) + extraCost[slot])
      });
    }
  }
  return {scores, scale, totalWeight};
}

/**
 * The swap phase driven by {@link getDecomposedSwapScores}: the GPU thresholds and tie rules (best
 * `(primary, secondary)`, then the lowest selected facility, then the lowest candidate).
 */
export function runDecomposedSwapPhaseOnCPU(
  operation: LocationAllocationOracleOperation,
  columns: EntryColumns,
  facilityCount: number,
  weights: Float32Array,
  initialSelected: readonly number[],
  distance: number,
  maximumSwapIterations: number,
  options: {netTerms?: boolean; float32?: boolean} = {}
): SwapOracleResult {
  const selected = [...initialSelected].sort((a, b) => a - b);
  let swapIterations = 0;
  let converged = false;
  while (swapIterations < maximumSwapIterations) {
    const {scores, scale, totalWeight} = getDecomposedSwapScores(
      operation,
      columns,
      facilityCount,
      weights,
      selected,
      distance,
      options
    );
    let best: {primary: number; secondary: number; removed: number; candidate: number} | null =
      null;
    for (const removed of selected) {
      for (let candidate = 0; candidate < facilityCount; candidate++) {
        const score = scores.get(`${removed}:${candidate}`);
        if (!score) continue;
        let primary = -1;
        let secondary = -1;
        if (operation === 'p-median') {
          if (score.primary > SWAP_TOLERANCE * totalWeight) {
            primary = score.primary;
            secondary = score.secondary;
          } else if (
            Math.abs(score.primary) <= SWAP_TOLERANCE * totalWeight &&
            score.secondary > SWAP_TOLERANCE * scale
          ) {
            primary = 0;
            secondary = score.secondary;
          }
        } else if (score.primary > SWAP_TOLERANCE * scale) {
          primary = score.primary;
          secondary = 0;
        }
        if (primary < 0) continue;
        if (
          !best ||
          primary > best.primary ||
          (primary === best.primary && secondary > best.secondary)
        ) {
          best = {primary, secondary, removed, candidate};
        }
      }
    }
    if (!best) {
      converged = true;
      break;
    }
    selected.splice(selected.indexOf(best.removed), 1, best.candidate);
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

/**
 * Greedy construction through facility columns (the sparse GPU score): the candidate maximizing
 * (reachable weight won, cost saved) lexicographically for p-median, or the uncovered weight it
 * covers for coverage, ties to the lowest facility. Equals {@link runGreedyOnCPU} for unit weights.
 */
export function runColumnGreedyOnCPU(
  operation: LocationAllocationOracleOperation,
  columns: EntryColumns,
  facilityCount: number,
  weights: Float32Array,
  maximumFacilityCount: number,
  distance: number
): number[] {
  const best = new Array<number>(weights.length).fill(LARGE_COST);
  const selected: number[] = [];
  for (let step = 0; step < maximumFacilityCount; step++) {
    let chosen = -1;
    let chosenReach = -Infinity;
    let chosenCost = -Infinity;
    for (let facility = 0; facility < facilityCount; facility++) {
      if (selected.includes(facility)) continue;
      let reach = 0;
      let saved = 0;
      for (const {demand, cost} of columns[facility]) {
        const c = Math.min(cost, LARGE_COST);
        const weight = weights[demand];
        if (operation === 'p-median') {
          if (c < best[demand]) {
            if (best[demand] < LARGE_COST) {
              saved += weight * (best[demand] - c);
            } else {
              reach += weight;
              saved -= weight * c;
            }
          }
        } else if (best[demand] > distance && c <= distance) {
          reach += weight;
        }
      }
      const improves =
        operation === 'p-median'
          ? reach > chosenReach || (reach === chosenReach && saved > chosenCost)
          : reach > Math.max(chosenReach, 0);
      if (improves) {
        chosen = facility;
        chosenReach = reach;
        chosenCost = saved;
      }
    }
    if (chosen < 0) break;
    selected.push(chosen);
    for (const {demand, cost} of columns[chosen]) {
      best[demand] = Math.min(best[demand], Math.min(cost, LARGE_COST));
    }
  }
  return selected;
}

/**
 * Integer costs in [0, 6] (many exact ties), integer weights in [0, 4] (some zero) and about a
 * third of the entries unreachable, so every double sum is exact and the decomposition must equal
 * the definition exactly, not just within a tolerance.
 */
export function createWeightedInstance(seed: number, demandCount: number, facilityCount: number) {
  const random = createSeededRandom(seed);
  const costs = Float32Array.from({length: demandCount * facilityCount}, () =>
    random() < 0.33 ? Infinity : Math.floor(random() * 7)
  );
  const weights = Float32Array.from({length: demandCount}, () => Math.floor(random() * 5));
  return {costs, weights};
}
