// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS
} from '../../../src/gpu-spatial-analysis/location-allocation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCostMatrix,
  createPartlyReachableCostMatrix,
  getBruteForceOptimum,
  getObjective,
  getPMedianObjectiveParts,
  getSubsets,
  runGreedyOnCPU,
  runSwapPhaseOnCPU,
  type LocationAllocationOracleOperation,
  UNREACHABLE_SENTINEL_CASE
} from './location-allocation-oracle';

const DEMAND_COUNT = 14;
const FACILITY_COUNT = 8;
const DISTANCE = 8;

// Oracle: spopt is not installed in the oracle venv, so brute force over every facility subset is
// the optimum reference (instances are small enough to enumerate).

it('swap phase never worsens greedy, ends at a swap-local optimum and stays above the optimum', () => {
  let improvedCount = 0;
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (let seed = 1; seed <= 40; seed++) {
      const costs = createCostMatrix(seed, DEMAND_COUNT, FACILITY_COUNT);
      for (const count of [2, 3, 4]) {
        const label = `${operation} seed ${seed} p ${count}`;
        const greedy = runGreedyOnCPU(operation, costs, FACILITY_COUNT, count, DISTANCE);
        const greedyObjective = getObjective(operation, costs, FACILITY_COUNT, greedy, DISTANCE);
        const result = runSwapPhaseOnCPU(operation, costs, FACILITY_COUNT, greedy, DISTANCE, 100);
        const objective = getObjective(operation, costs, FACILITY_COUNT, result.selected, DISTANCE);
        const optimum = getBruteForceOptimum(operation, costs, FACILITY_COUNT, count, DISTANCE);
        expect(result.selected.length, label).toBe(greedy.length);
        if (operation === 'p-median') {
          expect(objective, label).toBeLessThanOrEqual(greedyObjective);
          expect(objective, label).toBeGreaterThanOrEqual(optimum);
        } else {
          expect(objective, label).toBeGreaterThanOrEqual(greedyObjective);
          expect(objective, label).toBeLessThanOrEqual(optimum);
        }
        if (objective !== greedyObjective) improvedCount++;
        expect(result.converged, label).toBe(true);
        // Local optimality: no single swap improves the final set.
        for (const removed of result.selected) {
          for (let candidate = 0; candidate < FACILITY_COUNT; candidate++) {
            if (result.selected.includes(candidate)) continue;
            const next = result.selected.filter(f => f !== removed).concat(candidate);
            const swapped = getObjective(operation, costs, FACILITY_COUNT, next, DISTANCE);
            if (operation === 'p-median') {
              expect(swapped, label).toBeGreaterThanOrEqual(objective);
            } else {
              expect(swapped, label).toBeLessThanOrEqual(objective);
            }
          }
        }
      }
    }
  }
  // The scenes include instances where greedy is not swap-optimal, so the phase must improve some.
  expect(improvedCount).toBeGreaterThan(0);
});

it('swap phase closes the p-median greedy gap on a classic instance', () => {
  // Facility 0 sits between two demand clusters (greedy picks it first for p = 2), but the
  // optimum serves each cluster from its own facility.
  const costs = Float32Array.from([
    // facilities:  0   1   2
    5,
    0,
    100, //   demand A1
    5,
    1,
    100, //   demand A2
    5,
    100,
    0, //   demand B1
    5,
    100,
    1 //    demand B2
  ]);
  const greedy = runGreedyOnCPU('p-median', costs, 3, 2, 0);
  expect(greedy[0]).toBe(0);
  const greedyObjective = getObjective('p-median', costs, 3, greedy, 0);
  const swapped = runSwapPhaseOnCPU('p-median', costs, 3, greedy, 0, 10);
  const optimum = getBruteForceOptimum('p-median', costs, 3, 2, 0);
  expect(optimum).toBe(2);
  expect(getObjective('p-median', costs, 3, swapped.selected, 0)).toBe(optimum);
  expect(getObjective('p-median', costs, 3, swapped.selected, 0)).toBeLessThan(greedyObjective);
  expect(swapped.selected).toEqual([1, 2]);
  expect(swapped.swapIterations).toBeGreaterThan(0);
});

it('swap phase reports the iteration limit when the budget runs out', () => {
  const costs = Float32Array.from([5, 0, 100, 5, 1, 100, 5, 100, 0, 5, 100, 1]);
  const limited = runSwapPhaseOnCPU('p-median', costs, 3, [0, 1], 0, 1);
  expect(limited.swapIterations).toBe(1);
  expect(limited.iterationLimitReached).toBe(true);
  expect(limited.converged).toBe(false);
  const disabled = runSwapPhaseOnCPU('p-median', costs, 3, [0, 1], 0, 0);
  expect(disabled.selected).toEqual([0, 1]);
  expect(disabled.swapIterations).toBe(0);
});

it('oracle subsets enumerate all combinations', () => {
  expect(getSubsets(5, 2).length).toBe(10);
});

it('GPULocationAllocation validates maximumSwapIterations and declares swap nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'la-swap-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `la-view-${serial++}`, format, length);
  const make = (extra: Record<string, unknown> = {}) => ({
    costs: view('float32', 4 * 3),
    facilityCount: 3,
    operation: 'p-median' as const,
    maximumFacilityCount: 2,
    output: {
      selectedFacilities: view('uint32', 3),
      assignments: view('uint32', 4),
      objective: view('float32', 1),
      status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
    },
    ...extra
  });
  expect(GPU_LOCATION_ALLOCATION_STATUS.length).toBe(7);
  expect(new GPULocationAllocation(make()).maximumSwapIterations).toBe(100);
  expect(new GPULocationAllocation(make({maximumSwapIterations: 0})).maximumSwapIterations).toBe(0);
  expect(() => new GPULocationAllocation(make({maximumSwapIterations: -1}))).toThrow(
    /maximumSwapIterations/
  );
  expect(() => new GPULocationAllocation(make({maximumSwapIterations: 1.5}))).toThrow(
    /maximumSwapIterations/
  );
  expect(() => new GPULocationAllocation(make({maximumSwapIterations: 5000}))).toThrow(
    /maximumSwapIterations/
  );
  expect(
    () =>
      new GPULocationAllocation(
        make({
          operation: 'set-covering',
          coverageDistance: view('float32', 1),
          maximumSwapIterations: 3
        })
      )
  ).toThrow(/set-covering has no swap phase/);
  expect(
    new GPULocationAllocation(
      make({operation: 'set-covering', coverageDistance: view('float32', 1)})
    ).maximumSwapIterations
  ).toBe(0);
  const statusShort = make();
  statusShort.output.status = view('uint32', 4);
  expect(() => new GPULocationAllocation(statusShort)).toThrow(/status must hold 7 words/);

  const ids = new GPULocationAllocation(make({id: 'swap', maximumSwapIterations: 2}))
    .getCommandNodes(graph)
    .map(node => node.id);
  for (const iteration of [0, 1]) {
    for (const stage of ['prepare', 'list', 'score', 'choose', 'apply']) {
      expect(ids).toContain(`swap-swap-${iteration}-${stage}`);
    }
  }
  expect(ids.indexOf('swap-swap-1-apply')).toBeLessThan(ids.indexOf('swap-assign'));
  expect(ids).not.toContain('swap-swap-2-apply');
  const plain = new GPULocationAllocation(make({id: 'plain', maximumSwapIterations: 0}))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plain.some(id => id.includes('-swap-'))).toBe(false);
});

it('p-median oracle never sums unreachable rows and compares reachability first', () => {
  const {costs, facilityCount, expectedFacility, expectedObjective} = UNREACHABLE_SENTINEL_CASE;
  const greedy = runGreedyOnCPU('p-median', costs, facilityCount, 1, 0);
  expect(greedy).toEqual([expectedFacility]);
  expect(getObjective('p-median', costs, facilityCount, greedy, 0)).toBe(expectedObjective);
  expect(getPMedianObjectiveParts(costs, facilityCount, greedy).unreachable).toBe(1);

  for (let seed = 1; seed <= 30; seed++) {
    const sparse = createPartlyReachableCostMatrix(seed, DEMAND_COUNT, FACILITY_COUNT);
    for (const count of [1, 2, 3]) {
      const label = `seed ${seed} p ${count}`;
      const selected = runGreedyOnCPU('p-median', sparse, FACILITY_COUNT, count, 0);
      const greedyParts = getPMedianObjectiveParts(sparse, FACILITY_COUNT, selected);
      expect(Number.isFinite(greedyParts.cost), label).toBe(true);
      expect(greedyParts.cost, label).toBeLessThan(1e6);
      expect(greedyParts.unreachable, label).toBeGreaterThanOrEqual(1);
      const swapped = runSwapPhaseOnCPU('p-median', sparse, FACILITY_COUNT, selected, 0, 100);
      const swappedParts = getPMedianObjectiveParts(sparse, FACILITY_COUNT, swapped.selected);
      // Lexicographic never-worse: fewer unreachable rows, or as many and no higher cost.
      expect(
        swappedParts.unreachable < greedyParts.unreachable ||
          (swappedParts.unreachable === greedyParts.unreachable &&
            swappedParts.cost <= greedyParts.cost),
        label
      ).toBe(true);
      expect(swapped.converged, label).toBe(true);
    }
  }
});
