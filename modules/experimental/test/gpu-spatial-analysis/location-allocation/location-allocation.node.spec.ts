// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  getGPULocationAllocationMaximumSparseEntryCount,
  GPU_LOCATION_ALLOCATION_STATUS
} from '../../../src/gpu-spatial-analysis/location-allocation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCostMatrix,
  createSeededRandom,
  createWeightedInstance,
  getBruteForceSwapScore,
  getDecomposedSwapScores,
  getDenseEntryColumns,
  getSparseEntryColumns,
  runColumnGreedyOnCPU,
  runDecomposedSwapPhaseOnCPU,
  toDenseCosts,
  toSparseCosts,
  transposeSparseCosts,
  createPartlyReachableCostMatrix,
  getBruteForceOptimum,
  getObjective,
  getPMedianObjectiveParts,
  getSubsets,
  runGreedyOnCPU,
  runSwapPhaseOnCPU,
  type LocationAllocationOracleOperation,
  UNREACHABLE_SENTINEL_CASE,
  decodePMedianFixture,
  getWeightedPMedianBruteForceOptimum
} from './location-allocation-oracle';
import pMedianFixture from './p-median-fixture.json';

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
    for (const stage of ['list', 'prepare', 'totals', 'score', 'choose', 'apply']) {
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

it('p-median fixture: recorded metadata, and the JS brute force matches the scipy MILP optimum', () => {
  expect(pMedianFixture.versions.scipy).toMatch(/^\d+\.\d+/);
  expect(pMedianFixture.versions.numpy).toMatch(/^\d+\.\d+/);
  // Integer costs and weights: every objective is an exact float32 integer.
  expect(pMedianFixture.tolerance).toBe(0);
  const instances = decodePMedianFixture(pMedianFixture);
  expect(instances.length).toBeGreaterThanOrEqual(20);
  expect(instances.some(instance => instance.optimalUnreachableWeight > 0)).toBe(true);
  for (const [index, instance] of instances.entries()) {
    const optimum = getWeightedPMedianBruteForceOptimum(
      instance.costs,
      instance.facilityCount,
      instance.demandWeights,
      instance.maximumFacilityCount
    );
    expect(optimum, `instance ${index}`).toEqual({
      unreachableWeight: instance.optimalUnreachableWeight,
      cost: instance.optimalCost
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Whitaker / Resende-Werneck gain decomposition against the swap definition.
// ---------------------------------------------------------------------------------------------

it('swap gain decomposition equals the brute-force swap delta (ties, unreachable rows, zero weights)', () => {
  const facilityCount = 7;
  const demandCount = 18;
  let checked = 0;
  let unreachableRowsSeen = 0;
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (let seed = 1; seed <= 60; seed++) {
      const {costs, weights} = createWeightedInstance(seed * 7 + 1, demandCount, facilityCount);
      const columns = getDenseEntryColumns(costs, facilityCount);
      const random = createSeededRandom(seed);
      for (const count of [1, 2, 3, 4]) {
        const selected = Array.from({length: facilityCount}, (_, facility) => facility)
          .sort(() => random() - 0.5)
          .slice(0, count)
          .sort((a, b) => a - b);
        const forms = [
          getDecomposedSwapScores(operation, columns, facilityCount, weights, selected, 3, {
            netTerms: true
          }).scores,
          getDecomposedSwapScores(operation, columns, facilityCount, weights, selected, 3).scores
        ];
        if (selected.some(() => true)) {
          for (let row = 0; row < demandCount; row++) {
            if (selected.every(facility => costs[row * facilityCount + facility] === Infinity)) {
              unreachableRowsSeen++;
            }
          }
        }
        for (const removed of selected) {
          for (let candidate = 0; candidate < facilityCount; candidate++) {
            if (selected.includes(candidate)) continue;
            const expected = getBruteForceSwapScore(
              operation,
              costs,
              facilityCount,
              weights,
              selected,
              removed,
              candidate,
              3
            );
            for (const [form, scores] of forms.entries()) {
              expect(
                scores.get(`${removed}:${candidate}`),
                `${operation} seed ${seed} S ${selected} swap ${removed}->${candidate} form ${form}`
              ).toEqual(expected);
            }
            checked++;
          }
        }
      }
    }
  }
  expect(checked).toBeGreaterThan(2000);
  expect(unreachableRowsSeen).toBeGreaterThan(0);
});

it('decomposed swap phase applies the same swaps as the brute-force swap phase', () => {
  const facilityCount = 8;
  const demandCount = 14;
  const unitWeights = new Float32Array(demandCount).fill(1);
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (let seed = 1; seed <= 40; seed++) {
      for (const matrix of [
        createCostMatrix(seed, demandCount, facilityCount),
        createPartlyReachableCostMatrix(seed, demandCount, facilityCount)
      ]) {
        const columns = getDenseEntryColumns(matrix, facilityCount);
        for (const count of [2, 3, 4]) {
          const label = `${operation} seed ${seed} p ${count}`;
          const greedy = runGreedyOnCPU(operation, matrix, facilityCount, count, 8);
          const expected = runSwapPhaseOnCPU(operation, matrix, facilityCount, greedy, 8, 100);
          const decomposed = runDecomposedSwapPhaseOnCPU(
            operation,
            columns,
            facilityCount,
            unitWeights,
            greedy,
            8,
            100
          );
          expect(decomposed, label).toEqual(expected);
        }
      }
    }
  }
});

it('sparse costs give the dense scores, greedy selection and swap phase (CPU mirrors)', () => {
  const facilityCount = 8;
  const demandCount = 16;
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (let seed = 1; seed <= 30; seed++) {
      const {costs, weights} = createWeightedInstance(seed * 13 + 5, demandCount, facilityCount);
      // Listing some +Infinity entries must not change anything.
      const sparse = toSparseCosts(
        costs,
        facilityCount,
        (row, facility) => (row + facility) % 5 === 0
      );
      expect(toDenseCosts(sparse, facilityCount), `${operation} ${seed} round trip`).toEqual(costs);
      const dense = getDenseEntryColumns(costs, facilityCount);
      const listed = getSparseEntryColumns(sparse, facilityCount);
      const selected = [1, 4, 6];
      const denseScores = getDecomposedSwapScores(
        operation,
        dense,
        facilityCount,
        weights,
        selected,
        3
      );
      const sparseScores = getDecomposedSwapScores(
        operation,
        listed,
        facilityCount,
        weights,
        selected,
        3
      );
      expect(sparseScores, `${operation} ${seed} scores`).toEqual(denseScores);
      const denseGreedy = runColumnGreedyOnCPU(operation, dense, facilityCount, weights, 3, 3);
      expect(runColumnGreedyOnCPU(operation, listed, facilityCount, weights, 3, 3)).toEqual(
        denseGreedy
      );
      expect(
        runDecomposedSwapPhaseOnCPU(operation, listed, facilityCount, weights, denseGreedy, 3, 50),
        `${operation} ${seed} swap`
      ).toEqual(
        runDecomposedSwapPhaseOnCPU(operation, dense, facilityCount, weights, denseGreedy, 3, 50)
      );
    }
  }
});

it('column greedy equals the dense greedy rule for unit weights', () => {
  const facilityCount = 8;
  const demandCount = 14;
  const unitWeights = new Float32Array(demandCount).fill(1);
  for (const operation of ['p-median', 'maximum-coverage'] as LocationAllocationOracleOperation[]) {
    for (let seed = 1; seed <= 40; seed++) {
      for (const matrix of [
        createCostMatrix(seed, demandCount, facilityCount),
        createPartlyReachableCostMatrix(seed, demandCount, facilityCount)
      ]) {
        const columns = getSparseEntryColumns(toSparseCosts(matrix, facilityCount), facilityCount);
        for (const count of [1, 3, 5]) {
          expect(
            runColumnGreedyOnCPU(operation, columns, facilityCount, unitWeights, count, 8),
            `${operation} seed ${seed} p ${count}`
          ).toEqual(runGreedyOnCPU(operation, matrix, facilityCount, count, 8));
        }
      }
    }
  }
});

it('transposeSparseCosts groups entries by facility with ascending demand rows', () => {
  const costs = Float32Array.from([1, Infinity, 3, Infinity, 5, 6, 7, 8, Infinity]);
  const transposed = transposeSparseCosts(toSparseCosts(costs, 3), 3);
  expect(transposed.offsets).toEqual([0, 2, 4, 6]);
  expect(transposed.entries.map(entry => entry.demand)).toEqual([0, 2, 1, 2, 0, 1]);
});

// ---------------------------------------------------------------------------------------------
// Sparse input validation and kernel structure.
// ---------------------------------------------------------------------------------------------

function createSparseHarness() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'la-sparse'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `la-sparse-view-${serial++}`, format, length);
  const make = (
    extra: Record<string, unknown> = {},
    demandCount = 6,
    facilityCount = 4,
    entryCount = 12
  ) => ({
    sparseCosts: {
      offsets: view('uint32', demandCount + 1),
      facilities: view('uint32', entryCount),
      costs: view('float32', entryCount)
    },
    facilityCount,
    operation: 'p-median' as const,
    maximumFacilityCount: 2,
    output: {
      selectedFacilities: view('uint32', facilityCount),
      assignments: view('uint32', demandCount),
      objective: view('float32', 1),
      status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
    },
    ...extra
  });
  return {graph, view, make};
}

it('GPULocationAllocation validates sparseCosts', () => {
  const {graph, view, make} = createSparseHarness();
  const allocation = new GPULocationAllocation(make());
  expect(allocation.demandCount).toBe(6);
  expect(getGPULocationAllocationMaximumSparseEntryCount(graph.device)).toBe(2 ** 24);
  expect(() => new GPULocationAllocation(make({costs: view('float32', 24)}))).toThrow(
    /mutually exclusive/
  );
  expect(() => new GPULocationAllocation(make({sparseCosts: undefined}))).toThrow(
    /requires costs or sparseCosts/
  );
  const sparse = make().sparseCosts;
  expect(
    () => new GPULocationAllocation(make({sparseCosts: {...sparse, costs: view('float32', 11)}}))
  ).toThrow(/equally long/);
  expect(
    () =>
      new GPULocationAllocation(
        make(
          {sparseCosts: {...sparse, facilities: view('uint32', 1), costs: view('float32', 1)}},
          6,
          4,
          1
        )
      )
  ).not.toThrow();
  expect(
    () => new GPULocationAllocation(make({sparseCosts: {...sparse, offsets: view('uint32', 1)}}))
  ).toThrow(/offsets/);
  expect(
    () => new GPULocationAllocation(make({sparseCosts: {...sparse, costs: view('uint32', 12)}}))
  ).toThrow(/sparseCosts.costs/);
  // The sparse views must belong to the graph the contributor is added to.
  const other = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'other'});
  expect(() => new GPULocationAllocation(make()).getCommandNodes(other)).toThrow(
    /same graph|belong/
  );
  // Outputs may not alias the sparse inputs.
  const base = make();
  expect(
    () =>
      new GPULocationAllocation({
        ...base,
        output: {...base.output, assignments: base.sparseCosts.facilities}
      })
  ).toThrow(/outputs must not share buffers/);
  expect(graph).toBeDefined();
});

it('sparse swap scoring does one pass per candidate, not one thread per pair', () => {
  for (const sparse of [false, true]) {
    const {graph, view, make} = createSparseHarness();
    const demandCount = 30;
    const facilityCount = 9;
    const input = sparse
      ? make({}, demandCount, facilityCount, 80)
      : {
          ...make({}, demandCount, facilityCount, 80),
          sparseCosts: undefined,
          costs: view('float32', demandCount * facilityCount)
        };
    const nodes = new GPULocationAllocation({
      ...input,
      id: 'la',
      maximumFacilityCount: 3,
      maximumSwapIterations: 1
    } as never).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    const constantsOf = (id: string) => sources.find(source => source.id === id)!.constants;
    // Candidate-parallel scoring: facilityCount invocations (pairs would be 3 * 9).
    expect(constantsOf('la-swap-0-score')['INVOCATION_COUNT'], `sparse ${sparse}`).toBe(
      facilityCount
    );
    // Sparse: one workgroup of 256 per slot (loss) plus one for the totals. Dense: totals only,
    // the candidate pass adds each row's net term itself.
    expect(constantsOf(`la-swap-0-${sparse ? 'removal' : 'totals'}`)['INVOCATION_COUNT']).toBe(
      sparse ? (3 + 1) * 256 : 256
    );
    expect(constantsOf('la-swap-0-prepare')['INVOCATION_COUNT']).toBe(demandCount);
  }
});

it('sparse setup runs validation, then the facility-major transpose, before the greedy steps', () => {
  const {graph, view, make} = createSparseHarness();
  const ids = new GPULocationAllocation(
    make({id: 'sp', maximumSwapIterations: 1, demandWeights: view('float32', 6)})
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  for (const id of [
    'sp-validate-offsets',
    'sp-validate-rows',
    'sp-validate-inputs',
    'sp-transpose-rows',
    'sp-transpose-offsets',
    'sp-transpose-gather',
    'sp-assign',
    'sp-objective'
  ]) {
    expect(ids, id).toContain(id);
  }
  expect(ids.indexOf('sp-validate-rows')).toBeLessThan(ids.indexOf('sp-transpose-rows'));
  expect(ids.indexOf('sp-transpose-gather')).toBeLessThan(ids.indexOf('sp-score-0'));
  expect(ids.indexOf('sp-assign')).toBeLessThan(ids.indexOf('sp-objective'));
  expect(ids).not.toContain('sp-validate');
  // No weights and no coverage distance: nothing for the input validation to check.
  const bare = new GPULocationAllocation(make({id: 'bare'})).getCommandNodes(graph).map(n => n.id);
  expect(bare).not.toContain('bare-validate-inputs');
});

it('net per-row terms keep float32 accuracy where loss(r) and extra(i, r) cancel (d2 ~ 1e7 * d1)', () => {
  // Rows served by facility 0 have d1 = 1 and d2 = 1e7 (facility 1); candidate 2 reaches them at
  // costs 1..3, so removing 0 for 2 costs about 1 per row. The loss/extra split sums +-1e7-sized
  // terms in float32 (spacing 1 near 1e7) and loses the answer; the net form stays exact.
  const demandCount = 64;
  const facilityCount = 3;
  const costs = new Float32Array(demandCount * facilityCount);
  for (let row = 0; row < demandCount; row++) {
    costs[row * facilityCount] = 1;
    costs[row * facilityCount + 1] = 1e7;
    costs[row * facilityCount + 2] = 1 + (row % 3);
  }
  const weights = new Float32Array(demandCount).fill(1);
  const columns = getDenseEntryColumns(costs, facilityCount);
  const exact = getBruteForceSwapScore('p-median', costs, facilityCount, weights, [0, 1], 0, 2, 0);
  const score = (netTerms: boolean, float32: boolean) =>
    getDecomposedSwapScores('p-median', columns, facilityCount, weights, [0, 1], 0, {
      netTerms,
      float32
    }).scores.get('0:2')!;
  expect(score(true, false)).toEqual(exact);
  expect(score(false, false)).toEqual(exact);
  const netError = Math.abs(score(true, true).secondary - exact.secondary);
  const splitError = Math.abs(score(false, true).secondary - exact.secondary);
  // Net terms are small integers here, so float32 is exact; the split form is off by whole units.
  expect(netError).toBe(0);
  expect(splitError).toBeGreaterThan(0);
});
