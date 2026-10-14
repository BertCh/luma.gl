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
import {evaluatePartition} from './spatial-regionalization-oracle';

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
