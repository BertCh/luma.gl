// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUAZPRegions,
  GPU_AZP_STATUS
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

it('GPUAZPRegions reproducibly improves a contiguous partition and reports convergence', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const offsets = Uint32Array.from([0, 1, 3, 5, 6]);
  const neighbors = Uint32Array.from([1, 0, 2, 1, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const values = Float32Array.from([0, 10, 11, 20]);
  const initialLabels = Uint32Array.from([0, 0, 1, 1]);
  const graph = new GPUCommandGraph(device, {id: 'azp-oracle'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    initialLabels: createInputBuffer(device, initialLabels)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 17;
  graph.add(
    new GPUAZPRegions({
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
      initialLabels: importGraphBuffer(
        graph,
        'initial-labels',
        inputs.initialLabels,
        'uint32',
        initialLabels.length
      ),
      regionCapacity: 2,
      maximumIterations: 8,
      seed,
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', 4),
      optimization: {
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', GPU_AZP_STATUS.length),
        seed
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, 4);
  const [objective] = await readFloat32(outputs.objective, 1);
  const status = await readUint32(outputs.status, GPU_AZP_STATUS.length);
  expect(labels).toEqual([0, 1, 1, 1]);
  const mean = (10 + 11 + 20) / 3;
  const expectedObjective = (10 - mean) ** 2 + (11 - mean) ** 2 + (20 - mean) ** 2;
  expect(objective).toBeCloseTo(expectedObjective, 4);
  expect(status[GPU_AZP_STATUS.iterationCount]).toBe(1);
  expect(status[GPU_AZP_STATUS.converged]).toBe(1);
  expect(status[GPU_AZP_STATUS.iterationLimitReached]).toBe(0);
  expect(status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUAZPRegions accepts a non-leaf move when exact source connectivity is preserved', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Rows 0, 1 and 2 form a triangle in region 0. Row 0 has two same-region neighbors, but it is
  // not an articulation point because rows 1 and 2 remain directly connected after it moves.
  const offsets = Uint32Array.from([0, 3, 5, 7, 8]);
  const neighbors = Uint32Array.from([1, 2, 3, 0, 2, 0, 1, 0]);
  const edgeWeights = Float32Array.from({length: neighbors.length}, () => 1);
  const values = Float32Array.from([10, 0, 0, 10]);
  const initialLabels = Uint32Array.from([0, 0, 0, 1]);
  const graph = new GPUCommandGraph(device, {id: 'azp-articulation-oracle'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    initialLabels: createInputBuffer(device, initialLabels)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 5;
  graph.add(
    new GPUAZPRegions({
      weights: {
        offsets: importGraphBuffer(
          graph,
          'cycle-offsets',
          inputs.offsets,
          'uint32',
          offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'cycle-neighbors',
          inputs.neighbors,
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'cycle-weights',
          inputs.weights,
          'float32',
          edgeWeights.length
        )
      },
      values: importGraphBuffer(graph, 'cycle-values', inputs.values, 'float32', values.length),
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
      initialLabels: importGraphBuffer(
        graph,
        'cycle-initial-labels',
        inputs.initialLabels,
        'uint32',
        initialLabels.length
      ),
      regionCapacity: 2,
      maximumIterations: 4,
      seed,
      labels: importGraphBuffer(graph, 'cycle-labels', outputs.labels, 'uint32', 4),
      optimization: {
        objective: importGraphBuffer(graph, 'cycle-objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'cycle-status',
          outputs.status,
          'uint32',
          GPU_AZP_STATUS.length
        ),
        seed
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(outputs.labels, 4)).toEqual([1, 0, 0, 1]);
  expect((await readFloat32(outputs.objective, 1))[0]).toBeCloseTo(0, 4);
  expect(await readUint32(outputs.status, GPU_AZP_STATUS.length)).toEqual([1, 1, 0, 0]);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUAZPRegions keeps 36 contiguous regions on a 900-row lattice', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 900 rows exceed the 256-lane workgroup, so the candidate argmin needs a second reduction pass.
  const width = 30;
  const tile = 5;
  const rowCount = width * width;
  const regionCount = (width / tile) ** 2;
  const {offsets, neighbors, weights: edgeWeights} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 303);
  const initialLabels = Uint32Array.from({length: rowCount}, (_, row) => {
    const x = row % width;
    const y = Math.floor(row / width);
    return Math.floor(y / tile) * (width / tile) + Math.floor(x / tile);
  });
  const initialObjective = evaluatePartition(values, 1, initialLabels).withinSsd;
  const graph = new GPUCommandGraph(device, {id: 'azp-large'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    initialLabels: createInputBuffer(device, initialLabels)
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 11;
  graph.add(
    new GPUAZPRegions({
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
      initialLabels: importGraphBuffer(
        graph,
        'initial-labels',
        inputs.initialLabels,
        'uint32',
        rowCount
      ),
      regionCapacity: regionCount,
      maximumIterations: 16,
      seed,
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
      optimization: {
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', GPU_AZP_STATUS.length),
        seed
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, rowCount);
  const [objective] = await readFloat32(outputs.objective, 1);
  const status = await readUint32(outputs.status, GPU_AZP_STATUS.length);
  expect(labels.every(label => label < regionCount)).toBe(true);
  expect(new Set(labels).size).toBe(regionCount);
  expect(status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  expect(findDisconnectedLabels(offsets, neighbors, labels)).toEqual([]);
  // The reported objective is the within-region SSD of the returned labels, and AZP only improves.
  const finalObjective = evaluatePartition(values, 1, labels).withinSsd;
  expect(Math.abs(objective - finalObjective)).toBeLessThan(1e-3 * finalObjective);
  expect(finalObjective).toBeLessThanOrEqual(initialObjective * (1 + 1e-6));
  expect(status[GPU_AZP_STATUS.iterationCount]).toBeGreaterThan(0);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

type AzpRun = {labels: number[]; objective: number; status: number[]};

/** Runs one `GPUAZPRegions` graph on a unit-weight CSR and reads everything back. */
async function runAzp(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  scene: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    values: Float32Array;
    columnCount?: number;
    initialLabels: Uint32Array;
    regionCapacity: number;
    extensiveValues?: Float32Array;
    extensiveThreshold?: number;
    standardize?: boolean;
    maximumIterations?: number;
  }
): Promise<AzpRun> {
  const rowCount = scene.offsets.length - 1;
  const edgeWeights = Float32Array.from({length: scene.neighbors.length}, () => 1);
  const graph = new GPUCommandGraph(device, {id: 'azp-run'});
  const inputs = {
    offsets: createInputBuffer(device, scene.offsets),
    neighbors: createInputBuffer(device, scene.neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, scene.values),
    initialLabels: createInputBuffer(device, scene.initialLabels),
    ...(scene.extensiveValues
      ? {
          extensiveValues: createInputBuffer(device, scene.extensiveValues),
          threshold: createInputBuffer(device, Float32Array.from([scene.extensiveThreshold ?? 0]))
        }
      : {})
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_AZP_STATUS.length)
  };
  const seed = 3;
  graph.add(
    new GPUAZPRegions({
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
      extensiveValues:
        'extensiveValues' in inputs
          ? importGraphBuffer(graph, 'extensive', inputs.extensiveValues, 'float32', rowCount)
          : undefined,
      extensiveThreshold:
        'threshold' in inputs
          ? importGraphBuffer(graph, 'extensive-threshold', inputs.threshold, 'float32', 1)
          : undefined,
      initialLabels: importGraphBuffer(
        graph,
        'initial-labels',
        inputs.initialLabels,
        'uint32',
        rowCount
      ),
      regionCapacity: scene.regionCapacity,
      maximumIterations: scene.maximumIterations ?? 8,
      seed,
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
      optimization: {
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(graph, 'status', outputs.status, 'uint32', GPU_AZP_STATUS.length),
        seed
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    labels: await readUint32(outputs.labels, rowCount),
    objective: (await readFloat32(outputs.objective, 1))[0],
    status: await readUint32(outputs.status, GPU_AZP_STATUS.length)
  };
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
  return result;
}

it('GPUAZPRegions reports a disconnected initial region through invalidCount', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Path 0-1-2-3 with alternating labels: each region is two isolated rows, one extra component
  // per region.
  const result = await runAzp(device, {
    offsets: Uint32Array.from([0, 1, 3, 5, 6]),
    neighbors: Uint32Array.from([1, 0, 2, 1, 3, 2]),
    values: Float32Array.from([0, 1, 2, 3]),
    initialLabels: Uint32Array.from([0, 1, 0, 1]),
    regionCapacity: 2,
    standardize: false
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(2);
  expect(result.status[GPU_AZP_STATUS.iterationCount]).toBe(0);
});

it('GPUAZPRegions treats directed weights as undirected adjacency', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Each row lists only its lower neighbor (1 -> 0, 2 -> 1, 3 -> 2). Row 2 can move to region 1
  // only through the link that row 3 lists.
  const result = await runAzp(device, {
    offsets: Uint32Array.from([0, 0, 1, 2, 3]),
    neighbors: Uint32Array.from([0, 1, 2]),
    values: Float32Array.from([0, 0, 10, 10]),
    initialLabels: Uint32Array.from([0, 0, 0, 1]),
    regionCapacity: 2,
    standardize: false
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  expect(canonicalizePartition(result.labels)).toEqual([0, 0, 1, 1]);
  expect(result.objective).toBeCloseTo(0, 4);
});

it('GPUAZPRegions keeps a region at or above the extensive threshold', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const scene = {
    offsets: Uint32Array.from([0, 1, 3, 5, 6]),
    neighbors: Uint32Array.from([1, 0, 2, 1, 3, 2]),
    values: Float32Array.from([0, 0, 10, 10]),
    initialLabels: Uint32Array.from([0, 0, 0, 1]),
    regionCapacity: 2,
    extensiveValues: Float32Array.from([1, 1, 1, 1]),
    standardize: false
  };
  // Without the constraint row 2 moves next to its equal; with a floor of 3 region 0 may not shrink.
  const free = await runAzp(device, {...scene, extensiveThreshold: 0});
  expect(canonicalizePartition(free.labels)).toEqual([0, 0, 1, 1]);
  const bound = await runAzp(device, {...scene, extensiveThreshold: 3});
  expect(bound.labels).toEqual([0, 0, 0, 1]);
  expect(bound.status[GPU_AZP_STATUS.iterationCount]).toBe(0);
  expect(bound.status[GPU_AZP_STATUS.converged]).toBe(1);
  expect(bound.objective).toBeCloseTo(
    evaluatePartition(scene.values, 1, bound.labels).withinSsd,
    3
  );
});

it('GPUAZPRegions standardizes columns by default and reports the z-score objective', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 12;
  const rowCount = width * width;
  const {offsets, neighbors} = createRookLattice(width, width);
  const small = createRandomValues(rowCount, 51, 1);
  const large = createRandomValues(rowCount, 52, 1000);
  const values = Float32Array.from({length: rowCount * 2}, (_, index) =>
    index % 2 === 0 ? small[index >> 1] : large[index >> 1]
  );
  const tile = 4;
  const initialLabels = Uint32Array.from({length: rowCount}, (_, row) => {
    const x = row % width;
    const y = Math.floor(row / width);
    return Math.floor(y / tile) * (width / tile) + Math.floor(x / tile);
  });
  const result = await runAzp(device, {
    offsets,
    neighbors,
    values,
    columnCount: 2,
    initialLabels,
    regionCapacity: (width / tile) ** 2,
    maximumIterations: 16
  });
  expect(result.status[GPU_AZP_STATUS.invalidCount]).toBe(0);
  const expected = evaluatePartition(standardizeColumns(values, 2), 2, result.labels).withinSsd;
  expect(Math.abs(result.objective - expected)).toBeLessThan(1e-3 * Math.max(expected, 1));
});
