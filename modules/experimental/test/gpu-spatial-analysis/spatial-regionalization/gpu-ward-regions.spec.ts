// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUWardRegions,
  GPU_WARD_STATUS
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

it('GPUWardRegions merges the least-cost adjacent regions to its target', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const offsets = Uint32Array.from([0, 1, 3, 5, 6]);
  const neighbors = Uint32Array.from([1, 0, 2, 1, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const values = Float32Array.from([0, 1, 10, 11]);
  const graph = new GPUCommandGraph(device, {id: 'ward-regions-oracle'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_WARD_STATUS.length)
  };
  const seed = 0;
  graph.add(
    new GPUWardRegions({
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
      targetRegionCount: 2,
      maximumIterations: 2,
      seed,
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', 4),
      optimization: {
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          outputs.status,
          'uint32',
          GPU_WARD_STATUS.length
        ),
        seed
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(outputs.labels, 4)).toEqual([0, 0, 2, 2]);
  expect((await readFloat32(outputs.objective, 1))[0]).toBeCloseTo(1, 5);
  expect(await readUint32(outputs.status, GPU_WARD_STATUS.length)).toEqual([2, 1, 0, 0]);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUWardRegions merges a 900-row lattice to its target with contiguous regions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 900 rows exceed the 256-lane workgroup, so each merge selection reduces several tiles.
  const width = 30;
  const rowCount = width * width;
  const targetRegionCount = 860;
  const {offsets, neighbors, weights: edgeWeights} = createRookLattice(width, width);
  const values = createRandomValues(rowCount, 404);
  const graph = new GPUCommandGraph(device, {id: 'ward-regions-large'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values)
  };
  const outputs = {
    labels: createOutputBuffer(device, rowCount),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_WARD_STATUS.length)
  };
  const seed = 0;
  graph.add(
    new GPUWardRegions({
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
      targetRegionCount,
      maximumIterations: rowCount - targetRegionCount,
      seed,
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rowCount),
      optimization: {
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          outputs.status,
          'uint32',
          GPU_WARD_STATUS.length
        ),
        seed
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const labels = await readUint32(outputs.labels, rowCount);
  const [objective] = await readFloat32(outputs.objective, 1);
  const status = await readUint32(outputs.status, GPU_WARD_STATUS.length);
  expect(labels.every(label => label < rowCount)).toBe(true);
  expect(new Set(labels).size).toBe(targetRegionCount);
  expect(status[GPU_WARD_STATUS.invalidCount]).toBe(0);
  expect(status[GPU_WARD_STATUS.converged]).toBe(1);
  expect(findDisconnectedLabels(offsets, neighbors, labels)).toEqual([]);
  const finalObjective = evaluatePartition(values, 1, labels).withinSsd;
  expect(Math.abs(objective - finalObjective)).toBeLessThan(1e-3 * Math.max(finalObjective, 1));
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});

it('GPUWardRegions gates off merges once disconnected weights leave no adjacent pair', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Two components {0, 1} and {2, 3}: two merges are possible, the third iteration finds no
  // adjacent pair, disarms the indirect gate, and must not re-apply the stale winner.
  const offsets = Uint32Array.from([0, 1, 2, 3, 4]);
  const neighbors = Uint32Array.from([1, 0, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1]);
  const values = Float32Array.from([0, 1, 10, 11]);
  const graph = new GPUCommandGraph(device, {id: 'ward-regions-disconnected'});
  const inputs = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values)
  };
  const outputs = {
    labels: createOutputBuffer(device, 4),
    objective: createOutputBuffer(device, 1),
    status: createOutputBuffer(device, GPU_WARD_STATUS.length)
  };
  const seed = 7;
  graph.add(
    new GPUWardRegions({
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
      targetRegionCount: 1,
      maximumIterations: 3,
      seed,
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', 4),
      optimization: {
        objective: importGraphBuffer(graph, 'objective', outputs.objective, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          outputs.status,
          'uint32',
          GPU_WARD_STATUS.length
        ),
        seed
      }
    })
  );
  const compiled = graph.compile();
  // The second submission checks that initialization re-arms the gate.
  for (let submission = 0; submission < 2; submission++) {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputs.labels, 4)).toEqual([0, 0, 2, 2]);
    expect((await readFloat32(outputs.objective, 1))[0]).toBeCloseTo(1, 5);
    expect(await readUint32(outputs.status, GPU_WARD_STATUS.length)).toEqual([2, 0, 0, 0]);
  }
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
});
