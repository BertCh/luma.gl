// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS,
  type GPULocationAllocationOperation
} from '../../../src/gpu-spatial-analysis/location-allocation';
import {
  createPartlyReachableCostMatrix,
  createSeededRandom as createOracleRandom,
  createWeightedInstance,
  getDenseEntryColumns,
  getBruteForceOptimum,
  runColumnGreedyOnCPU,
  runDecomposedSwapPhaseOnCPU,
  toSparseCosts,
  getObjective,
  getPMedianObjectiveParts,
  runGreedyOnCPU as runReachabilityAwareGreedyOnCPU,
  runSwapPhaseOnCPU,
  UNREACHABLE_SENTINEL_CASE,
  type LocationAllocationOracleOperation,
  compareWeightedPMedianParts,
  decodePMedianFixture,
  getWeightedPMedianParts,
  isWeightedPMedianSwapLocalOptimum
} from './location-allocation-oracle';
import pMedianFixture from './p-median-fixture.json';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLegacyPairSwapNodes} from './legacy-pair-swap';

async function runAllocation(
  operation: GPULocationAllocationOperation,
  costs: Float32Array,
  facilityCount: number,
  maximumFacilityCount: number,
  coverageDistance?: number,
  // The greedy specs below pin the pure greedy rule, so the swap phase is off unless asked for.
  maximumSwapIterations = 0,
  demandWeights?: Float32Array,
  // Pass the same matrix as CSR sparse costs (unreachable entries unlisted).
  sparse = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const demandCount = costs.length / facilityCount;
  const sparseArrays = sparse ? toSparseCosts(costs, facilityCount) : undefined;
  const costsBuffer = createInputBuffer(device, sparseArrays ? sparseArrays.costs : costs);
  const offsetsBuffer = sparseArrays ? createInputBuffer(device, sparseArrays.offsets) : undefined;
  const facilitiesBuffer = sparseArrays
    ? createInputBuffer(device, sparseArrays.facilities)
    : undefined;
  const coverageBuffer =
    coverageDistance === undefined
      ? undefined
      : createInputBuffer(device, Float32Array.from([coverageDistance]));
  const weightsBuffer = demandWeights ? createInputBuffer(device, demandWeights) : undefined;
  const selectedBuffer = createOutputBuffer(device, facilityCount);
  const assignmentsBuffer = createOutputBuffer(device, demandCount);
  const assignedCostsBuffer = createOutputBuffer(device, demandCount);
  const objectiveBuffer = createOutputBuffer(device, 1);
  const statusBuffer = createOutputBuffer(device, GPU_LOCATION_ALLOCATION_STATUS.length);
  const graph = new GPUCommandGraph(device, {id: `location-allocation-${operation}`});
  graph.add(
    new GPULocationAllocation({
      ...(sparseArrays
        ? {
            sparseCosts: {
              offsets: importGraphBuffer(
                graph,
                'sparse-offsets',
                offsetsBuffer!,
                'uint32',
                sparseArrays.offsets.length
              ),
              facilities: importGraphBuffer(
                graph,
                'sparse-facilities',
                facilitiesBuffer!,
                'uint32',
                sparseArrays.facilities.length
              ),
              costs: importGraphBuffer(
                graph,
                'sparse-costs',
                costsBuffer,
                'float32',
                sparseArrays.costs.length
              )
            }
          }
        : {costs: importGraphBuffer(graph, 'costs', costsBuffer, 'float32', costs.length)}),
      facilityCount,
      demandWeights: weightsBuffer
        ? importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', demandCount)
        : undefined,
      operation,
      maximumFacilityCount,
      maximumSwapIterations: operation === 'set-covering' ? undefined : maximumSwapIterations,
      coverageDistance: coverageBuffer
        ? importGraphBuffer(graph, 'coverage', coverageBuffer, 'float32', 1)
        : undefined,
      output: {
        selectedFacilities: importGraphBuffer(
          graph,
          'selected',
          selectedBuffer,
          'uint32',
          facilityCount
        ),
        assignments: importGraphBuffer(
          graph,
          'assignments',
          assignmentsBuffer,
          'uint32',
          demandCount
        ),
        assignedCosts: importGraphBuffer(
          graph,
          'assigned-costs',
          assignedCostsBuffer,
          'float32',
          demandCount
        ),
        objective: importGraphBuffer(graph, 'objective', objectiveBuffer, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          statusBuffer,
          'uint32',
          GPU_LOCATION_ALLOCATION_STATUS.length
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    selected: await readUint32(selectedBuffer, facilityCount),
    assignments: await readUint32(assignmentsBuffer, demandCount),
    assignedCosts: await readFloat32(assignedCostsBuffer, demandCount),
    objective: (await readFloat32(objectiveBuffer, 1))[0],
    status: await readUint32(statusBuffer, GPU_LOCATION_ALLOCATION_STATUS.length)
  };
  compiled.destroy();
  costsBuffer.destroy();
  offsetsBuffer?.destroy();
  facilitiesBuffer?.destroy();
  coverageBuffer?.destroy();
  weightsBuffer?.destroy();
  selectedBuffer.destroy();
  assignmentsBuffer.destroy();
  assignedCostsBuffer.destroy();
  objectiveBuffer.destroy();
  statusBuffer.destroy();
  return result;
}

it('GPULocationAllocation greedily minimizes p-median cost', async () => {
  const result = await runAllocation(
    'p-median',
    Float32Array.from([0, 10, 20, 10, 0, 10, 20, 10, 0]),
    3,
    1
  );
  if (!result) return;
  expect(result.selected).toEqual([0, 1, 0]);
  expect(result.assignments).toEqual([1, 1, 1]);
  expect(result.assignedCosts).toEqual([10, 0, 10]);
  expect(result.objective).toBe(20);
  expect(result.status).toEqual([1, 0, 1, 0, 0, 0, 0]);
});

it('GPULocationAllocation reports maximum-coverage budget completion', async () => {
  const result = await runAllocation(
    'maximum-coverage',
    Float32Array.from([0, 10, 10, 0, 10, 10]),
    2,
    1,
    0
  );
  if (!result) return;
  expect(result.selected).toEqual([1, 0]);
  expect(result.objective).toBe(1);
  expect(result.status).toEqual([1, 2, 1, 0, 0, 0, 0]);
});

it('GPULocationAllocation completes deterministic set covering', async () => {
  const result = await runAllocation('set-covering', Float32Array.from([0, 10, 10, 0]), 2, 2, 0);
  if (!result) return;
  expect(result.selected).toEqual([1, 1]);
  expect(result.assignments).toEqual([0, 1]);
  expect(result.objective).toBe(2);
  expect(result.status).toEqual([2, 0, 1, 0, 0, 0, 0]);
});

// Exact checks. The TSDoc promises the deterministic greedy heuristics (lowest facility row on
// ties), not an integer-programming optimum, so each instance is checked against (a) a CPU copy of
// the greedy rule, which must match bit for bit because the integer costs make every f32 sum exact,
// and (b) a brute-force optimum, against which only the classical greedy guarantees are asserted.

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** Integer cost matrix, `demandCount` rows by `facilityCount` columns, values in [0, 30]. */
function createCostMatrix(seed: number, demandCount: number, facilityCount: number): Float32Array {
  const random = createSeededRandom(seed);
  return Float32Array.from({length: demandCount * facilityCount}, () => Math.floor(random() * 31));
}

function getSubsets(facilityCount: number, size: number): number[][] {
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

function getMedianCost(costs: Float32Array, facilityCount: number, subset: number[]): number {
  let total = 0;
  for (let row = 0; row < costs.length / facilityCount; row++) {
    total += Math.min(...subset.map(facility => costs[row * facilityCount + facility]));
  }
  return total;
}

function getCoveredCount(
  costs: Float32Array,
  facilityCount: number,
  distance: number,
  subset: number[]
): number {
  let covered = 0;
  for (let row = 0; row < costs.length / facilityCount; row++) {
    covered += subset.some(facility => costs[row * facilityCount + facility] <= distance) ? 1 : 0;
  }
  return covered;
}

/** The documented greedy rule: take the best marginal gain, ties to the lowest facility row. */
function runGreedyOnCPU(
  operation: GPULocationAllocationOperation,
  costs: Float32Array,
  facilityCount: number,
  maximumFacilityCount: number,
  distance: number
): number[] {
  const demandCount = costs.length / facilityCount;
  const best = new Array<number>(demandCount).fill(1e30);
  const selected: number[] = [];
  for (let step = 0; step < maximumFacilityCount; step++) {
    let chosen = -1;
    let chosenScore = operation === 'p-median' ? Infinity : 0;
    for (let facility = 0; facility < facilityCount; facility++) {
      if (selected.includes(facility)) continue;
      let score = 0;
      for (let row = 0; row < demandCount; row++) {
        const candidate = costs[row * facilityCount + facility];
        if (operation === 'p-median') {
          score += Math.min(best[row], candidate);
        } else if (best[row] > distance && candidate <= distance) {
          score += 1;
        }
      }
      const improves = operation === 'p-median' ? score < chosenScore : score > chosenScore;
      if (improves) {
        chosen = facility;
        chosenScore = score;
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

function getSelectedFacilities(flags: ArrayLike<number>): number[] {
  return Array.from(flags, (flag, facility) => (flag ? facility : -1)).filter(f => f >= 0);
}

const EXACT_DEMAND_COUNT = 14;
const EXACT_FACILITY_COUNT = 8;

it('GPULocationAllocation p-median equals the CPU greedy rule and the optimum for one facility', async () => {
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const costs = createCostMatrix(seed, EXACT_DEMAND_COUNT, EXACT_FACILITY_COUNT);
    for (const count of [1, 3]) {
      const result = await runAllocation('p-median', costs, EXACT_FACILITY_COUNT, count);
      if (!result) return;
      const label = `seed ${seed} p ${count}`;
      const greedy = runGreedyOnCPU('p-median', costs, EXACT_FACILITY_COUNT, count, 0);
      const selected = getSelectedFacilities(result.selected);
      expect(selected, label).toEqual([...greedy].sort((a, b) => a - b));
      const objective = getMedianCost(costs, EXACT_FACILITY_COUNT, selected);
      expect(result.objective, label).toBe(objective);
      const optimum = Math.min(
        ...getSubsets(EXACT_FACILITY_COUNT, count).map(subset =>
          getMedianCost(costs, EXACT_FACILITY_COUNT, subset)
        )
      );
      // Greedy never beats the optimum; with one facility the greedy step is the optimum. For
      // p > 1 the p-median greedy has no constant-factor guarantee, so only the CPU copy is exact.
      expect(objective, label).toBeGreaterThanOrEqual(optimum);
      if (count === 1) {
        expect(objective, label).toBe(optimum);
      }
    }
  }
});

it('GPULocationAllocation maximum coverage equals the CPU greedy rule and keeps the 1 - 1/e bound', async () => {
  const distance = 8;
  for (const seed of [11, 12, 13, 14, 15, 16]) {
    const costs = createCostMatrix(seed, EXACT_DEMAND_COUNT, EXACT_FACILITY_COUNT);
    for (const count of [1, 2, 3]) {
      const result = await runAllocation(
        'maximum-coverage',
        costs,
        EXACT_FACILITY_COUNT,
        count,
        distance
      );
      if (!result) return;
      const label = `seed ${seed} p ${count}`;
      const greedy = runGreedyOnCPU(
        'maximum-coverage',
        costs,
        EXACT_FACILITY_COUNT,
        count,
        distance
      );
      const selected = getSelectedFacilities(result.selected);
      expect(selected, label).toEqual([...greedy].sort((a, b) => a - b));
      const covered = getCoveredCount(costs, EXACT_FACILITY_COUNT, distance, selected);
      expect(result.objective, label).toBe(covered);
      const optimum = Math.max(
        ...getSubsets(EXACT_FACILITY_COUNT, count).map(subset =>
          getCoveredCount(costs, EXACT_FACILITY_COUNT, distance, subset)
        )
      );
      // Nemhauser-Wolsey-Fisher: greedy maximum coverage reaches at least (1 - 1/e) of the optimum.
      expect(covered, label).toBeLessThanOrEqual(optimum);
      expect(covered, label).toBeGreaterThanOrEqual((1 - 1 / Math.E) * optimum - 1e-9);
      if (count === 1) {
        expect(covered, label).toBe(optimum);
      }
    }
  }
});

it('GPULocationAllocation set covering covers every reachable row within the harmonic bound', async () => {
  const distance = 10;
  for (const seed of [21, 22, 23, 24, 25, 26]) {
    const costs = createCostMatrix(seed, EXACT_DEMAND_COUNT, EXACT_FACILITY_COUNT);
    const result = await runAllocation(
      'set-covering',
      costs,
      EXACT_FACILITY_COUNT,
      EXACT_FACILITY_COUNT,
      distance
    );
    if (!result) return;
    const label = `seed ${seed}`;
    const greedy = runGreedyOnCPU(
      'set-covering',
      costs,
      EXACT_FACILITY_COUNT,
      EXACT_FACILITY_COUNT,
      distance
    );
    const selected = getSelectedFacilities(result.selected);
    expect(selected, label).toEqual([...greedy].sort((a, b) => a - b));
    const allFacilities = Array.from({length: EXACT_FACILITY_COUNT}, (_, facility) => facility);
    const reachable = getCoveredCount(costs, EXACT_FACILITY_COUNT, distance, allFacilities);
    expect(getCoveredCount(costs, EXACT_FACILITY_COUNT, distance, selected), label).toBe(reachable);
    expect(result.objective, label).toBe(selected.length);
    let optimumSize = 0;
    for (let size = 1; size <= EXACT_FACILITY_COUNT && !optimumSize; size++) {
      if (
        getSubsets(EXACT_FACILITY_COUNT, size).some(
          subset => getCoveredCount(costs, EXACT_FACILITY_COUNT, distance, subset) === reachable
        )
      ) {
        optimumSize = size;
      }
    }
    // Chvatal: greedy set cover uses at most H(n) times the optimal number of sets, n = largest set.
    const largestSet = Math.max(
      ...allFacilities.map(facility =>
        getCoveredCount(costs, EXACT_FACILITY_COUNT, distance, [facility])
      )
    );
    let harmonic = 0;
    for (let k = 1; k <= largestSet; k++) harmonic += 1 / k;
    expect(selected.length, label).toBeGreaterThanOrEqual(optimumSize);
    expect(selected.length, label).toBeLessThanOrEqual(harmonic * optimumSize + 1e-9);
  }
});

// Swap phase (Teitz-Bart). The CPU mirror in location-allocation-oracle.ts applies the same rules
// (best improving pair, ties to the lowest selected facility then candidate, relative tolerance),
// and the integer costs make every f32 sum exact, so the GPU must match it exactly.

it('GPULocationAllocation swap phase equals the CPU swap mirror and never worsens greedy', async () => {
  const distance = 8;
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const costs = createCostMatrix(seed, EXACT_DEMAND_COUNT, EXACT_FACILITY_COUNT);
      for (const count of [2, 3, 4]) {
        const label = `${operation} seed ${seed} p ${count}`;
        const greedy = runGreedyOnCPU(operation, costs, EXACT_FACILITY_COUNT, count, distance);
        const mirror = runSwapPhaseOnCPU(
          operation,
          costs,
          EXACT_FACILITY_COUNT,
          greedy,
          distance,
          100
        );
        const result = await runAllocation(
          operation,
          costs,
          EXACT_FACILITY_COUNT,
          count,
          operation === 'maximum-coverage' ? distance : undefined,
          100
        );
        if (!result) return;
        expect(getSelectedFacilities(result.selected), label).toEqual(mirror.selected);
        const objective = getObjective(
          operation,
          costs,
          EXACT_FACILITY_COUNT,
          mirror.selected,
          distance
        );
        expect(result.objective, label).toBe(objective);
        const optimum = getBruteForceOptimum(
          operation,
          costs,
          EXACT_FACILITY_COUNT,
          count,
          distance
        );
        if (operation === 'p-median') {
          expect(objective, label).toBeGreaterThanOrEqual(optimum);
        } else {
          expect(objective, label).toBeLessThanOrEqual(optimum);
        }
        expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.swapIterationCount], label).toBe(
          mirror.swapIterations
        );
        expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.converged], label).toBe(
          mirror.converged ? 1 : 0
        );
        expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.iterationLimitReached], label).toBe(
          mirror.iterationLimitReached ? 1 : 0
        );
      }
    }
  }
});

it('GPULocationAllocation swap phase closes the classic p-median greedy gap and honors its budget', async () => {
  const costs = Float32Array.from([5, 0, 100, 5, 1, 100, 5, 100, 0, 5, 100, 1]);
  const full = await runAllocation('p-median', costs, 3, 2, undefined, 10);
  if (!full) return;
  expect(full.selected).toEqual([0, 1, 1]);
  expect(full.objective).toBe(2);
  expect(full.status[GPU_LOCATION_ALLOCATION_STATUS.converged]).toBe(1);
  const greedyOnly = await runAllocation('p-median', costs, 3, 2, undefined, 0);
  if (!greedyOnly) return;
  expect(greedyOnly.objective).toBeGreaterThan(full.objective);
  expect(greedyOnly.status.slice(4)).toEqual([0, 0, 0]);
  const limited = await runAllocation('p-median', costs, 3, 2, undefined, 1);
  if (!limited) return;
  expect(limited.status[GPU_LOCATION_ALLOCATION_STATUS.swapIterationCount]).toBe(1);
  expect(limited.status[GPU_LOCATION_ALLOCATION_STATUS.iterationLimitReached]).toBe(1);
  expect(limited.status[GPU_LOCATION_ALLOCATION_STATUS.converged]).toBe(0);
});

it('GPULocationAllocation rejects a NaN cost instead of treating it as a valid cost', async () => {
  // A NaN built at runtime from its bit pattern: the kernel must flag it through the bit test,
  // not through a self-comparison that shader compilers may fold away.
  const costs = Float32Array.from([1, 2, 3, 4]);
  new Uint32Array(costs.buffer)[1] = 0x7fc00000;
  const result = await runAllocation('p-median', costs, 2, 1);
  if (!result) return;
  expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.invalidCount]).toBeGreaterThan(0);
});

it('GPULocationAllocation p-median never sums unreachable rows into the objective', async () => {
  const {costs, facilityCount, expectedFacility, expectedObjective} = UNREACHABLE_SENTINEL_CASE;
  const result = await runAllocation('p-median', costs, facilityCount, 1);
  if (!result) return;
  // A summed 1e30 sentinel ties both facilities in float32 and picks facility 0 (cost 18).
  expect(getSelectedFacilities(result.selected)).toEqual([expectedFacility]);
  expect(result.objective).toBe(expectedObjective);
  expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.uncoveredDemandCount]).toBe(1);
  expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.invalidCount]).toBe(0);
  expect(result.assignments[0]).toBe(0xffffffff);
  expect(result.assignedCosts[0]).toBe(Infinity);

  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const sparse = createPartlyReachableCostMatrix(seed, EXACT_DEMAND_COUNT, EXACT_FACILITY_COUNT);
    for (const count of [1, 2, 3]) {
      const label = `seed ${seed} p ${count}`;
      const greedy = runReachabilityAwareGreedyOnCPU(
        'p-median',
        sparse,
        EXACT_FACILITY_COUNT,
        count,
        0
      );
      const mirror = runSwapPhaseOnCPU('p-median', sparse, EXACT_FACILITY_COUNT, greedy, 0, 100);
      const parts = getPMedianObjectiveParts(sparse, EXACT_FACILITY_COUNT, mirror.selected);
      const swapped = await runAllocation(
        'p-median',
        sparse,
        EXACT_FACILITY_COUNT,
        count,
        undefined,
        100
      );
      if (!swapped) return;
      expect(getSelectedFacilities(swapped.selected), label).toEqual(mirror.selected);
      // Integer costs: the reachable-row sum is exact in float32.
      expect(swapped.objective, label).toBe(parts.cost);
      expect(swapped.status[GPU_LOCATION_ALLOCATION_STATUS.uncoveredDemandCount], label).toBe(
        parts.unreachable
      );
    }
  }
});

// Property fixture (rule 22): random weighted instances with +Infinity entries, solved exactly by
// scipy.optimize.milp in p-median-oracle.py (versions and tolerance 0 recorded in the fixture).
// Teitz-Bart has no optimality guarantee, so the properties are: never better than the exact
// optimum, a swap-local optimum, converged, and an objective that equals the weighted reachable
// cost of the returned set exactly (integer data, so float32 sums are exact).
it('GPULocationAllocation p-median meets the scipy MILP optimum bound on the property fixture', async () => {
  expect(pMedianFixture.tolerance).toBe(0);
  let optimalCount = 0;
  const instances = decodePMedianFixture(pMedianFixture);
  for (const [index, instance] of instances.entries()) {
    const label = `instance ${index}`;
    const result = await runAllocation(
      'p-median',
      instance.costs,
      instance.facilityCount,
      instance.maximumFacilityCount,
      undefined,
      100,
      instance.demandWeights
    );
    if (!result) return;
    const selected = getSelectedFacilities(result.selected);
    expect(selected.length, label).toBe(instance.maximumFacilityCount);
    const parts = getWeightedPMedianParts(
      instance.costs,
      instance.facilityCount,
      instance.demandWeights,
      selected
    );
    const optimum = {
      unreachableWeight: instance.optimalUnreachableWeight,
      cost: instance.optimalCost
    };
    expect(compareWeightedPMedianParts(parts, optimum), label).toBeGreaterThanOrEqual(0);
    expect(result.objective, label).toBe(parts.cost);
    expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.invalidCount], label).toBe(0);
    expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.complete], label).toBe(1);
    expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.converged], label).toBe(1);
    expect(
      isWeightedPMedianSwapLocalOptimum(
        instance.costs,
        instance.facilityCount,
        instance.demandWeights,
        selected
      ),
      label
    ).toBe(true);
    for (let row = 0; row < instance.demandCount; row++) {
      const reachable = selected.some(facility =>
        Number.isFinite(instance.costs[row * instance.facilityCount + facility])
      );
      expect(result.assignments[row] === 0xffffffff, `${label} row ${row}`).toBe(!reachable);
    }
    if (compareWeightedPMedianParts(parts, optimum) === 0) optimalCount++;
  }
  // Not a guarantee, a regression tripwire: a CPU best-improvement Teitz-Bart run reaches the exact
  // optimum on 23 of the 24 instances (instance 21 stops at a swap-local optimum, 548 vs 500).
  expect(optimalCount).toBeGreaterThanOrEqual(instances.length - 3);
});

// ---------------------------------------------------------------------------------------------
// Swap gain decomposition and sparse costs. Integer costs and weights keep every float32 sum
// exact, so the GPU must pick exactly the swaps of the CPU mirrors (no near-tie wiggle room).
// ---------------------------------------------------------------------------------------------

const WEIGHTED_DEMAND_COUNT = 16;
const WEIGHTED_FACILITY_COUNT = 8;
const WEIGHTED_DISTANCE = 3;

it('GPULocationAllocation swap phase equals the decomposed CPU mirror on weighted, partly reachable instances', async () => {
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) {
      const {costs, weights} = createWeightedInstance(
        seed * 31 + 7,
        WEIGHTED_DEMAND_COUNT,
        WEIGHTED_FACILITY_COUNT
      );
      const columns = getDenseEntryColumns(costs, WEIGHTED_FACILITY_COUNT);
      for (const count of [2, 3, 4]) {
        const label = `${operation} seed ${seed} p ${count}`;
        const greedy = runColumnGreedyOnCPU(
          operation,
          columns,
          WEIGHTED_FACILITY_COUNT,
          weights,
          count,
          WEIGHTED_DISTANCE
        );
        const mirror = runDecomposedSwapPhaseOnCPU(
          operation,
          columns,
          WEIGHTED_FACILITY_COUNT,
          weights,
          greedy,
          WEIGHTED_DISTANCE,
          100
        );
        for (const sparse of [false, true]) {
          const result = await runAllocation(
            operation,
            costs,
            WEIGHTED_FACILITY_COUNT,
            count,
            operation === 'maximum-coverage' ? WEIGHTED_DISTANCE : undefined,
            100,
            weights,
            sparse
          );
          if (!result) return;
          expect(getSelectedFacilities(result.selected), `${label} sparse ${sparse}`).toEqual(
            mirror.selected
          );
          expect(
            result.status[GPU_LOCATION_ALLOCATION_STATUS.swapIterationCount],
            `${label} sparse ${sparse}`
          ).toBe(mirror.swapIterations);
          expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.converged], label).toBe(
            mirror.converged ? 1 : 0
          );
          expect(result.status[GPU_LOCATION_ALLOCATION_STATUS.invalidCount], label).toBe(0);
        }
      }
    }
  }
});

it('GPULocationAllocation sparseCosts equal the dense matrix for every operation (greedy and swap)', async () => {
  for (const operation of [
    'p-median',
    'maximum-coverage',
    'set-covering'
  ] as GPULocationAllocationOperation[]) {
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const {costs, weights} = createWeightedInstance(
        seed * 17 + 3,
        WEIGHTED_DEMAND_COUNT,
        WEIGHTED_FACILITY_COUNT
      );
      for (const swaps of [0, 100]) {
        for (const count of [2, 4]) {
          const label = `${operation} seed ${seed} p ${count} swaps ${swaps}`;
          const run = (sparse: boolean) =>
            runAllocation(
              operation,
              costs,
              WEIGHTED_FACILITY_COUNT,
              count,
              operation === 'p-median' ? undefined : WEIGHTED_DISTANCE,
              swaps,
              weights,
              sparse
            );
          const dense = await run(false);
          if (!dense) return;
          expect(await run(true), label).toEqual(dense);
        }
      }
    }
  }
});

it('GPULocationAllocation sparseCosts reproduce the property fixture results', async () => {
  for (const [index, instance] of decodePMedianFixture(pMedianFixture).entries()) {
    const run = (sparse: boolean) =>
      runAllocation(
        'p-median',
        instance.costs,
        instance.facilityCount,
        instance.maximumFacilityCount,
        undefined,
        100,
        instance.demandWeights,
        sparse
      );
    const dense = await run(false);
    if (!dense) return;
    const sparse = await run(true);
    expect(sparse, `instance ${index}`).toEqual(dense);
    // And the dense result is the fixture's swap-local optimum within the MILP bound.
    const selected = getSelectedFacilities(dense.selected);
    const parts = getWeightedPMedianParts(
      instance.costs,
      instance.facilityCount,
      instance.demandWeights,
      selected
    );
    expect(dense.objective, `instance ${index}`).toBe(parts.cost);
    expect(
      compareWeightedPMedianParts(parts, {
        unreachableWeight: instance.optimalUnreachableWeight,
        cost: instance.optimalCost
      }),
      `instance ${index}`
    ).toBeGreaterThanOrEqual(0);
  }
});

it('GPULocationAllocation rejects unsorted, duplicated or out-of-range sparse rows and bad offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const facilityCount = 3;
  const cases: {name: string; offsets: number[]; facilities: number[]}[] = [
    {name: 'descending row', offsets: [0, 2, 3], facilities: [2, 1, 0]},
    {name: 'duplicate in a row', offsets: [0, 2, 3], facilities: [1, 1, 0]},
    {name: 'facility out of range', offsets: [0, 2, 3], facilities: [0, 3, 1]},
    {name: 'offsets decrease', offsets: [0, 3, 2], facilities: [0, 1, 2]},
    {name: 'offsets end short', offsets: [0, 1, 2], facilities: [0, 1, 2]},
    {name: 'offsets do not start at zero', offsets: [1, 2, 3], facilities: [0, 1, 2]}
  ];
  for (const {name, offsets, facilities} of cases) {
    const graph = new GPUCommandGraph(device, {id: `sparse-invalid-${name}`});
    const buffers = [
      createInputBuffer(device, Uint32Array.from(offsets)),
      createInputBuffer(device, Uint32Array.from(facilities)),
      createInputBuffer(device, Float32Array.from([1, 2, 3])),
      createOutputBuffer(device, facilityCount),
      createOutputBuffer(device, 2),
      createOutputBuffer(device, 1),
      createOutputBuffer(device, GPU_LOCATION_ALLOCATION_STATUS.length)
    ];
    graph.add(
      new GPULocationAllocation({
        sparseCosts: {
          offsets: importGraphBuffer(graph, 'offsets', buffers[0], 'uint32', 3),
          facilities: importGraphBuffer(graph, 'facilities', buffers[1], 'uint32', 3),
          costs: importGraphBuffer(graph, 'costs', buffers[2], 'float32', 3)
        },
        facilityCount,
        operation: 'p-median',
        maximumFacilityCount: 1,
        output: {
          selectedFacilities: importGraphBuffer(
            graph,
            'selected',
            buffers[3],
            'uint32',
            facilityCount
          ),
          assignments: importGraphBuffer(graph, 'assignments', buffers[4], 'uint32', 2),
          objective: importGraphBuffer(graph, 'objective', buffers[5], 'float32', 1),
          status: importGraphBuffer(
            graph,
            'status',
            buffers[6],
            'uint32',
            GPU_LOCATION_ALLOCATION_STATUS.length
          )
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const status = await readUint32(buffers[6], GPU_LOCATION_ALLOCATION_STATUS.length);
    expect(status[GPU_LOCATION_ALLOCATION_STATUS.invalidCount], name).toBeGreaterThan(0);
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

// A/B (algorithms roadmap P2): the swap phase before (one thread per selected-slot x candidate pair
// scanning every demand row, `legacy-pair-swap.ts`) against the candidate-parallel Whitaker /
// Resende-Werneck decomposition, plus the sparse input on the same truncated cost matrix. Same
// greedy start, same graph shape, so the difference is the swap phase. Prints the timings; the
// results must agree (float32 sums are ordered differently, so a near-tie may pick another swap).
it('A/B swap phase: pair-per-thread (old) vs decomposition (new) vs sparse, 2000 candidates x 20000 demands, p = 20', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const facilityCount = 2000;
  const maximumFacilityCount = 20;
  // A dense 2000 x 20000 matrix is 160 MB; shrink the demand count on devices with smaller bindings.
  const demandCount = Math.min(
    20000,
    Math.floor(device.limits.maxStorageBufferBindingSize / 4 / facilityCount)
  );
  const random = createOracleRandom(2026);
  const facilityPoints = Float32Array.from({length: 2 * facilityCount}, () => random());
  const demandPoints = Float32Array.from({length: 2 * demandCount}, () => random());
  const cutoff = 0.08;
  const costs = new Float32Array(demandCount * facilityCount);
  for (let demand = 0; demand < demandCount; demand++) {
    for (let facility = 0; facility < facilityCount; facility++) {
      const distance = Math.hypot(
        demandPoints[2 * demand] - facilityPoints[2 * facility],
        demandPoints[2 * demand + 1] - facilityPoints[2 * facility + 1]
      );
      costs[demand * facilityCount + facility] = distance < cutoff ? distance : Infinity;
    }
  }
  const sparseArrays = toSparseCosts(costs, facilityCount);

  type Variant = 'old' | 'new' | 'sparse';
  const build = (variant: Variant) => {
    const graph = new GPUCommandGraph(device, {id: `swap-ab-${variant}`});
    const buffers = {
      costs: createInputBuffer(device, variant === 'sparse' ? sparseArrays.costs : costs),
      offsets: createInputBuffer(device, sparseArrays.offsets),
      facilities: createInputBuffer(device, sparseArrays.facilities),
      selected: createOutputBuffer(device, facilityCount),
      assignments: createOutputBuffer(device, demandCount),
      objective: createOutputBuffer(device, 1),
      status: createOutputBuffer(device, GPU_LOCATION_ALLOCATION_STATUS.length)
    };
    const costsView =
      variant === 'sparse'
        ? undefined
        : importGraphBuffer(graph, 'costs', buffers.costs, 'float32', costs.length);
    const selected = importGraphBuffer(
      graph,
      'selected',
      buffers.selected,
      'uint32',
      facilityCount
    );
    const status = importGraphBuffer(
      graph,
      'status',
      buffers.status,
      'uint32',
      GPU_LOCATION_ALLOCATION_STATUS.length
    );
    const allocation = new GPULocationAllocation({
      id: 'ab',
      ...(costsView
        ? {costs: costsView}
        : {
            sparseCosts: {
              offsets: importGraphBuffer(
                graph,
                'offsets',
                buffers.offsets,
                'uint32',
                sparseArrays.offsets.length
              ),
              facilities: importGraphBuffer(
                graph,
                'facilities',
                buffers.facilities,
                'uint32',
                sparseArrays.facilities.length
              ),
              costs: importGraphBuffer(
                graph,
                'sparse-costs',
                buffers.costs,
                'float32',
                sparseArrays.costs.length
              )
            }
          }),
      facilityCount,
      operation: 'p-median',
      maximumFacilityCount,
      maximumSwapIterations: variant === 'old' ? 0 : 100,
      output: {
        selectedFacilities: selected,
        assignments: importGraphBuffer(
          graph,
          'assignments',
          buffers.assignments,
          'uint32',
          demandCount
        ),
        objective: importGraphBuffer(graph, 'objective', buffers.objective, 'float32', 1),
        status
      }
    });
    const nodes = [...allocation.getCommandNodes(graph)];
    if (variant === 'old') {
      // Run the old swap phase between the greedy construction and the final assignment.
      const {nodes: legacy} = createLegacyPairSwapNodes(graph, {
        id: 'ab-legacy',
        costs: costsView!,
        facilityCount,
        demandCount,
        maximumFacilityCount,
        maximumSwapIterations: 100,
        selectedFacilities: selected,
        status
      });
      nodes.splice(
        nodes.findIndex(node => node.id === 'ab-assign'),
        0,
        ...legacy
      );
    }
    graph.add(nodes);
    return {compiled: graph.compile(), buffers};
  };

  const results = new Map<Variant, {objective: number; selected: number[]; median: number}>();
  for (const variant of ['old', 'new', 'sparse'] as const) {
    const {compiled, buffers} = build(variant);
    const timings: number[] = [];
    let objective = 0;
    let selected: number[] = [];
    for (let run = 0; run < 4; run++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      objective = (await readFloat32(buffers.objective, 1))[0];
      timings.push(performance.now() - start);
      if (run === 3)
        selected = getSelectedFacilities(await readUint32(buffers.selected, facilityCount));
    }
    // The first run compiles pipelines; report the median of the other three.
    const rest = timings.slice(1).sort((a, b) => a - b);
    results.set(variant, {objective, selected, median: rest[1]});
    compiled.destroy();
    for (const buffer of Object.values(buffers)) buffer.destroy();
  }
  const old = results.get('old')!;
  const next = results.get('new')!;
  const sparse = results.get('sparse')!;
  // eslint-disable-next-line no-console
  console.log(
    `[A/B location-allocation swap] candidates=${facilityCount} demand=${demandCount} p=${maximumFacilityCount} ` +
      `old ${old.median.toFixed(1)} ms, new ${next.median.toFixed(1)} ms, sparse ${sparse.median.toFixed(1)} ms ` +
      `(objective old ${old.objective}, new ${next.objective}, sparse ${sparse.objective}; ` +
      `same selection old/new ${String(old.selected.join() === next.selected.join())}, ` +
      `new/sparse ${String(next.selected.join() === sparse.selected.join())})`
  );
  expect(Math.abs(next.objective - old.objective)).toBeLessThanOrEqual(0.02 * old.objective);
  expect(Math.abs(sparse.objective - next.objective)).toBeLessThanOrEqual(0.02 * next.objective);
});
