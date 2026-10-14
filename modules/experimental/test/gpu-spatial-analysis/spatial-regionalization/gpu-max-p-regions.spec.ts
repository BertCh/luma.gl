// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMaxPRegions,
  GPU_MAX_P_DEFAULT_START_COUNT,
  GPU_MAX_P_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createRandomValues,
  createRookLattice,
  findDisconnectedLabels
} from './large-lattice-invariants';
import {
  canonicalizePartition,
  evaluatePartition,
  standardizeColumns
} from './spatial-regionalization-oracle';

it('GPUMaxPRegions grows contiguous threshold-feasible regions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const offsets = Uint32Array.from([0, 1, 3, 5, 6]);
  const neighbors = Uint32Array.from([1, 0, 2, 1, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const values = Float32Array.from([0, 1, 10, 11]);
  const extensiveValues = Float32Array.from([1, 1, 1, 1]);
  const threshold = Float32Array.from([2]);
  const graph = new GPUCommandGraph(device, {id: 'max-p-regions-oracle'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    extensiveValues: createInputBuffer(device, extensiveValues),
    threshold: createInputBuffer(device, threshold)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_MAX_P_STATUS.length)
  };
  graph.add(
    new GPUMaxPRegions({
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', inputs.offsets, 'uint32', offsets.length),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', values.length),
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
      extensiveValues: importGraphBuffer(
        graph,
        'extensive-values',
        inputs.extensiveValues,
        'float32',
        extensiveValues.length
      ),
      threshold: importGraphBuffer(graph, 'threshold', inputs.threshold, 'float32', 1),
      seed: 0,
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', 4),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          outputs.status,
          'uint32',
          GPU_MAX_P_STATUS.length
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  // Label numbers follow the selected start's random seed order; the partition is what is pinned.
  expect(canonicalizePartition(await readUint32(outputs.labels, 4))).toEqual([0, 0, 1, 1]);
  expect((await readFloat32(outputs.objective, 1))[0]).toBeCloseTo(1, 5);
  const status = await readUint32(outputs.status, GPU_MAX_P_STATUS.length);
  expect(status.slice(0, 4)).toEqual([2, 1, 0, 0]);
  expect(status[GPU_MAX_P_STATUS.selectedStart]).toBeLessThan(GPU_MAX_P_DEFAULT_START_COUNT);
  expect(status[GPU_MAX_P_STATUS.improvementMoveCount]).toBe(0);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUMaxPRegions labels every row of a 900-row lattice with contiguous regions above the floor', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 900 rows exceed the 256-lane workgroup, so the strided scans and argmin run several tiles.
  const width = 30;
  const rowCount = width * width;
  const {offsets, neighbors, weights: edgeWeights} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 101);
  const extensiveValues = Float32Array.from(createRandomValues(rowCount, 202, 5), value =>
    Math.floor(value + 1)
  );
  const floor = 12;
  const graph = new GPUCommandGraph(device, {id: 'max-p-regions-large'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    extensiveValues: createInputBuffer(device, extensiveValues),
    threshold: createInputBuffer(device, Float32Array.from([floor]))
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_MAX_P_STATUS.length)
  };
  graph.add(
    new GPUMaxPRegions({
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', inputs.offsets, 'uint32', offsets.length),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', rowCount),
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
      extensiveValues: importGraphBuffer(
        graph,
        'extensive-values',
        inputs.extensiveValues,
        'float32',
        rowCount
      ),
      threshold: importGraphBuffer(graph, 'threshold', inputs.threshold, 'float32', 1),
      seed: 3,
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          outputs.status,
          'uint32',
          GPU_MAX_P_STATUS.length
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, rowCount);
  const status = await readUint32(outputs.status, GPU_MAX_P_STATUS.length);
  const [objective] = await readFloat32(outputs.objective, 1);
  const NONE = 0xffffffff;
  expect(labels.every(label => label !== NONE)).toBe(true);
  expect(status[GPU_MAX_P_STATUS.invalidCount]).toBe(0);
  expect(status[GPU_MAX_P_STATUS.feasible]).toBe(1);
  expect(status[GPU_MAX_P_STATUS.infeasibleRegionCount]).toBe(0);
  expect(Number.isFinite(objective)).toBe(true);
  // Region count matches the reported count and every region meets the floor.
  const regionSums = new Map<number, number>();
  labels.forEach((label, row) =>
    regionSums.set(label, (regionSums.get(label) ?? 0) + extensiveValues[row])
  );
  expect(regionSums.size).toBe(status[GPU_MAX_P_STATUS.regionCount]);
  expect(regionSums.size).toBeGreaterThan(1);
  for (const sum of regionSums.values()) {
    expect(sum).toBeGreaterThanOrEqual(floor);
  }
  expect(findDisconnectedLabels(offsets, neighbors, labels)).toEqual([]);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

type MaxPRun = {
  labels: number[];
  objective: number;
  status: number[];
};

/** Runs one `GPUMaxPRegions` graph and reads the labels, objective and status back. */
async function runMaxP(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  scene: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    values: Float32Array;
    columnCount?: number;
    extensiveValues: Float32Array;
    threshold: number;
    seed: number;
    startCount?: number;
    improvementIterations?: number;
    standardize?: boolean;
  }
): Promise<MaxPRun> {
  const rowCount = scene.offsets.length - 1;
  const edgeWeights = Float32Array.from({length: scene.neighbors.length}, () => 1);
  const graph = new GPUCommandGraph(device, {id: 'max-p-regions-run'});
  const inputs = {
    offsets: createInputBuffer(device, scene.offsets),
    neighbors: createInputBuffer(device, scene.neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, scene.values),
    extensiveValues: createInputBuffer(device, scene.extensiveValues),
    threshold: createInputBuffer(device, Float32Array.from([scene.threshold]))
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_MAX_P_STATUS.length)
  };
  graph.add(
    new GPUMaxPRegions({
      weights: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          inputs.offsets,
          'uint32',
          scene.offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          inputs.neighbors,
          'uint32',
          scene.neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', edgeWeights.length)
      },
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', scene.values.length),
      columnCount: scene.columnCount,
      standardize: scene.standardize,
      extensiveValues: importGraphBuffer(
        graph,
        'extensive-values',
        inputs.extensiveValues,
        'float32',
        rowCount
      ),
      threshold: importGraphBuffer(graph, 'threshold', inputs.threshold, 'float32', 1),
      startCount: scene.startCount,
      improvementIterations: scene.improvementIterations,
      seed: scene.seed,
      output: {
        labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          outputs.status,
          'uint32',
          GPU_MAX_P_STATUS.length
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    labels: await readUint32(outputs.labels, rowCount),
    objective: (await readFloat32(outputs.objective, 1))[0],
    status: await readUint32(outputs.status, GPU_MAX_P_STATUS.length)
  };
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
  return result;
}

function createLatticeScene(width: number, valueSeed: number, extensiveSeed: number) {
  const rowCount = width * width;
  const {offsets, neighbors} = createRookLattice(width, width);
  return {
    offsets,
    neighbors,
    values: createRandomValues(rowCount, valueSeed),
    extensiveValues: Float32Array.from(createRandomValues(rowCount, extensiveSeed, 5), value =>
      Math.floor(value + 1)
    )
  };
}

it('GPUMaxPRegions multi-start never loses to its own first start', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Start 0 of a 16-start run uses the same Philox key (seed, 0) as a one-start run, so the
  // selection rule (fewest leftovers, then most regions) can only match or beat it.
  const scene = {...createLatticeScene(20, 11, 22), threshold: 10, seed: 9};
  const single = await runMaxP(device, {...scene, startCount: 1});
  const multi = await runMaxP(device, {...scene, startCount: 16});
  expect(multi.status[GPU_MAX_P_STATUS.invalidCount]).toBe(0);
  expect(single.status[GPU_MAX_P_STATUS.selectedStart]).toBe(0);
  expect(multi.status[GPU_MAX_P_STATUS.selectedStart]).toBeLessThan(16);
  const singleLeftovers = single.status[GPU_MAX_P_STATUS.infeasibleRegionCount];
  const multiLeftovers = multi.status[GPU_MAX_P_STATUS.infeasibleRegionCount];
  expect(multiLeftovers).toBeLessThanOrEqual(singleLeftovers);
  if (multiLeftovers === singleLeftovers) {
    expect(multi.status[GPU_MAX_P_STATUS.regionCount]).toBeGreaterThanOrEqual(
      single.status[GPU_MAX_P_STATUS.regionCount]
    );
  }
  expect(new Set(multi.labels).size).toBe(multi.status[GPU_MAX_P_STATUS.regionCount]);
  const {offsets, neighbors} = scene;
  expect(findDisconnectedLabels(offsets, neighbors, multi.labels)).toEqual([]);
  // Reproducible: the same seed and start count give the same labels.
  const again = await runMaxP(device, {...scene, startCount: 16});
  expect(again.labels).toEqual(multi.labels);
});

it('GPUMaxPRegions symmetrizes directed weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // A chain listed in one direction only (row i lists row i + 1). Growth still reaches both ways.
  const result = await runMaxP(device, {
    offsets: Uint32Array.from([0, 1, 2, 3, 4, 5, 5]),
    neighbors: Uint32Array.from([1, 2, 3, 4, 5]),
    values: Float32Array.from([0, 0, 0, 10, 10, 10]),
    extensiveValues: Float32Array.from([1, 1, 1, 1, 1, 1]),
    threshold: 3,
    seed: 1,
    standardize: false
  });
  expect(canonicalizePartition(result.labels)).toEqual([0, 0, 0, 1, 1, 1]);
  expect(result.status.slice(0, 4)).toEqual([2, 1, 0, 0]);
  expect(result.objective).toBeCloseTo(0, 5);
});

it('GPUMaxPRegions local improvement keeps the region count and the threshold', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const scene = {...createLatticeScene(16, 31, 32), threshold: 8, seed: 4, startCount: 4};
  const constructed = await runMaxP(device, scene);
  const improved = await runMaxP(device, {...scene, improvementIterations: 64});
  expect(improved.status[GPU_MAX_P_STATUS.invalidCount]).toBe(0);
  expect(improved.status[GPU_MAX_P_STATUS.selectedStart]).toBe(
    constructed.status[GPU_MAX_P_STATUS.selectedStart]
  );
  expect(improved.status[GPU_MAX_P_STATUS.regionCount]).toBe(
    constructed.status[GPU_MAX_P_STATUS.regionCount]
  );
  expect(improved.objective).toBeLessThanOrEqual(constructed.objective * (1 + 1e-5));
  expect(new Set(improved.labels).size).toBe(constructed.status[GPU_MAX_P_STATUS.regionCount]);
  const sums = new Map<number, number>();
  improved.labels.forEach((label, row) =>
    sums.set(label, (sums.get(label) ?? 0) + scene.extensiveValues[row])
  );
  if (improved.status[GPU_MAX_P_STATUS.infeasibleRegionCount] === 0) {
    for (const sum of sums.values()) {
      expect(sum).toBeGreaterThanOrEqual(scene.threshold);
    }
  }
  expect(findDisconnectedLabels(scene.offsets, scene.neighbors, improved.labels)).toEqual([]);
  // The reported objective is the within-region sum of squares of the standardized table.
  const table = standardizeColumns(scene.values, 1);
  const expected = evaluatePartition(table, 1, improved.labels).withinSsd;
  expect(Math.abs(improved.objective - expected)).toBeLessThan(1e-3 * Math.max(expected, 1));
});

it('GPUMaxPRegions standardizes columns by default', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Two columns in very different units: only the standardized table makes the objective match
  // the CPU within-region sum of squares of z-scores (population standard deviation).
  const rowCount = 12 * 12;
  const {offsets, neighbors} = createRookLattice(12, 12);
  const small = createRandomValues(rowCount, 41, 1);
  const large = createRandomValues(rowCount, 42, 1000);
  const values = Float32Array.from({length: rowCount * 2}, (_, index) =>
    index % 2 === 0 ? small[index >> 1] : large[index >> 1]
  );
  const result = await runMaxP(device, {
    offsets,
    neighbors,
    values,
    columnCount: 2,
    extensiveValues: Float32Array.from({length: rowCount}, () => 1),
    threshold: 9,
    seed: 2,
    startCount: 4
  });
  const expected = evaluatePartition(standardizeColumns(values, 2), 2, result.labels).withinSsd;
  expect(Math.abs(result.objective - expected)).toBeLessThan(1e-3 * Math.max(expected, 1));
});
