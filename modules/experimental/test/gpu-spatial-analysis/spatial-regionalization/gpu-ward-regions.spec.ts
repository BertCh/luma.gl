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
import {
  canonicalizePartition,
  evaluatePartition,
  standardizeColumns
} from './spatial-regionalization-oracle';

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
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
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
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
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
      // Pinned raw-unit expectations: the default standardization is covered separately.
      standardize: false,
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

/** Runs one `GPUWardRegions` graph on a unit-weight chain and reads everything back. */
async function runWardChain(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  scene: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    values: Float32Array;
    standardize?: boolean;
  }
) {
  const rowCount = scene.offsets.length - 1;
  const edgeWeights = Float32Array.from({length: scene.neighbors.length}, () => 1);
  const graph = new GPUCommandGraph(device, {id: 'ward-regions-chain'});
  const inputs = {
    offsets: createInputBuffer(device, scene.offsets),
    neighbors: createInputBuffer(device, scene.neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, scene.values)
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
      values: importGraphBuffer(graph, 'values', inputs.values, 'float32', rowCount),
      standardize: scene.standardize,
      targetRegionCount: 2,
      maximumIterations: rowCount - 2,
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
  const result = {
    labels: await readUint32(outputs.labels, rowCount),
    objective: (await readFloat32(outputs.objective, 1))[0],
    status: await readUint32(outputs.status, GPU_WARD_STATUS.length)
  };
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  Object.values(outputs).forEach(buffer => buffer.destroy());
  return result;
}

it('GPUWardRegions treats directed weights as undirected adjacency', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Each row lists only its lower neighbor. A pair is only a candidate from its lower region's
  // side, so merges across a link that lists the higher region first need the transposed link.
  const result = await runWardChain(device, {
    offsets: Uint32Array.from([0, 0, 1, 2, 3]),
    neighbors: Uint32Array.from([0, 1, 2]),
    values: Float32Array.from([0, 1, 10, 11]),
    standardize: false
  });
  expect(canonicalizePartition(result.labels)).toEqual([0, 0, 1, 1]);
  expect(result.objective).toBeCloseTo(1, 5);
  expect(result.status.slice(0, 4)).toEqual([2, 1, 0, 0]);
});

it('GPUWardRegions standardizes columns by default', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const values = Float32Array.from([0, 1, 10, 11]);
  const result = await runWardChain(device, {
    offsets: Uint32Array.from([0, 1, 3, 5, 6]),
    neighbors: Uint32Array.from([1, 0, 2, 1, 3, 2]),
    values
  });
  expect(canonicalizePartition(result.labels)).toEqual([0, 0, 1, 1]);
  // Population z-scores: the within-region sum of squares is 1 / 25.25 for the raw pair split.
  const expected = evaluatePartition(standardizeColumns(values, 1), 1, result.labels).withinSsd;
  expect(result.objective).toBeCloseTo(expected, 5);
});
