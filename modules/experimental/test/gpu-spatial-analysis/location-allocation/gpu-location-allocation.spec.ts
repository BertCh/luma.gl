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
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

async function runAllocation(
  operation: GPULocationAllocationOperation,
  costs: Float32Array,
  facilityCount: number,
  maximumFacilityCount: number,
  coverageDistance?: number
) {
  const device = await getWebGPUTestDevice();
  if (!device) return null;
  const demandCount = costs.length / facilityCount;
  const costsBuffer = createInputBuffer(device, costs);
  const coverageBuffer =
    coverageDistance === undefined
      ? undefined
      : createInputBuffer(device, Float32Array.from([coverageDistance]));
  const selectedBuffer = createOutputBuffer(device, facilityCount);
  const assignmentsBuffer = createOutputBuffer(device, demandCount);
  const assignedCostsBuffer = createOutputBuffer(device, demandCount);
  const objectiveBuffer = createOutputBuffer(device, 1);
  const statusBuffer = createOutputBuffer(device, GPU_LOCATION_ALLOCATION_STATUS.length);
  const graph = new GPUCommandGraph(device, {id: `location-allocation-${operation}`});
  graph.add(
    new GPULocationAllocation({
      costs: importGraphBuffer(graph, 'costs', costsBuffer, 'float32', costs.length),
      facilityCount,
      operation,
      maximumFacilityCount,
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
  coverageBuffer?.destroy();
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
  expect(result.status).toEqual([1, 0, 1, 0]);
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
  expect(result.status).toEqual([1, 2, 1, 0]);
});

it('GPULocationAllocation completes deterministic set covering', async () => {
  const result = await runAllocation('set-covering', Float32Array.from([0, 10, 10, 0]), 2, 2, 0);
  if (!result) return;
  expect(result.selected).toEqual([1, 1]);
  expect(result.assignments).toEqual([0, 1]);
  expect(result.objective).toBe(2);
  expect(result.status).toEqual([2, 0, 1, 0]);
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
