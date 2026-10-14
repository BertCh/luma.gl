// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMaxPRegions,
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
  expect(await readUint32(outputs.labels, 4)).toEqual([0, 0, 1, 1]);
  expect((await readFloat32(outputs.objective, 1))[0]).toBeCloseTo(1, 5);
  expect(await readUint32(outputs.status, GPU_MAX_P_STATUS.length)).toEqual([2, 1, 0, 0]);
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
