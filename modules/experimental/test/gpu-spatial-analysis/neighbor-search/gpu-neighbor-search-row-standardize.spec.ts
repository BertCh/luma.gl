// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {
  GPUSpatialLag,
  isGPUSpatialWeightsRowStandardized
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

// NEVER COMPILED ON A DEVICE: the compile-time `rowStandardize` constant in the weights kernel and
// the normalized GPUSpatialLag kernel have not run on a GPU yet.
// Five planar points; the band radius 1.01 links point 0 to points 1, 2 and 3 (distance 1) and
// points 1, 2 and 3 back to point 0 only. Point 4 is isolated. The analysis mask removes point 3,
// so the lag of row 0 must be the mean of the remaining neighbors' values (re-normalized).
const POSITIONS = [0, 0, 1, 0, 0, 1, -1, 0, 10, 10];
const VALUES = [1, 2, 4, 8, 16];
const ANALYSIS_MASK = [1, 1, 1, 0, 1];
const RADIUS = 1.01;

function computeExpectedLag(): number[] {
  const rows = VALUES.length;
  const lag: number[] = [];
  for (let row = 0; row < rows; row++) {
    if (!ANALYSIS_MASK[row]) {
      lag.push(0);
      continue;
    }
    const neighbors: number[] = [];
    for (let other = 0; other < rows; other++) {
      const distance = Math.hypot(
        POSITIONS[2 * row] - POSITIONS[2 * other],
        POSITIONS[2 * row + 1] - POSITIONS[2 * other + 1]
      );
      if (other !== row && distance <= RADIUS && ANALYSIS_MASK[other]) {
        neighbors.push(VALUES[other]);
      }
    }
    lag.push(
      neighbors.length > 0 ? neighbors.reduce((sum, value) => sum + value, 0) / neighbors.length : 0
    );
  }
  return lag;
}

it('a compile-time row-standardized distance band re-normalizes over masked neighbors in GPUSpatialLag', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = VALUES.length;
  const capacity = 16;
  const graph = new GPUCommandGraph(device, {id: 'row-standardize-lag'});
  const buffers = {
    positions: createInputBuffer(device, Float32Array.from(POSITIONS)),
    values: createInputBuffer(device, Float32Array.from(VALUES)),
    mask: createInputBuffer(device, Uint32Array.from(ANALYSIS_MASK)),
    offsets: createOutputBuffer(device, rows + 1),
    neighbors: createOutputBuffer(device, capacity),
    weights: createOutputBuffer(device, capacity),
    distances: createOutputBuffer(device, capacity),
    overflow: createOutputBuffer(device, 1),
    lag: createOutputBuffer(device, rows)
  };
  // The per-frame field is deliberately left false: the compile-time prop is the source of truth.
  const parameters = new GPUParameterBuffer(device, {
    id: 'row-standardize-parameters',
    format: 'float32',
    length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    values: getGPUNeighborSearchParameterValues({
      bounds: [-2, -2, 12, 12],
      radius: RADIUS,
      weightKind: 'binary',
      rowStandardize: false
    })
  });
  const search = new GPUNeighborSearch({
    mode: 'radius',
    gridSize: [4, 4],
    rowStandardize: true,
    weightTransform: 'row',
    positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', rows),
    parameters: parameters.importToGraph(graph),
    weights: {
      offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', rows + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
      weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity),
      distances: importGraphBuffer(graph, 'distances', buffers.distances, 'float32', capacity)
    },
    overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
  });
  expect(isGPUSpatialWeightsRowStandardized(search.outputWeights)).toBe(true);
  graph.add(search);
  graph.add(
    new GPUSpatialLag({
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32', rows),
      weights: search.outputWeights,
      mask: importGraphBuffer(graph, 'analysis-mask', buffers.mask, 'uint32', rows),
      normalize: isGPUSpatialWeightsRowStandardized(search.outputWeights),
      output: importGraphBuffer(graph, 'lag', buffers.lag, 'float32', rows)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const lag = await readFloat32(buffers.lag, rows);
  const expected = computeExpectedLag();
  // Row 0 is the mean of points 1 and 2 only: the neighbor removed by the mask is not counted.
  expect(expected[0]).toBe(3);
  for (let row = 0; row < rows; row++) {
    // f32 rounding of the 1/3 weights: a few ulp, relative.
    expect(Math.abs(lag[row] - expected[row]), `row ${row} lag ${lag[row]}`).toBeLessThanOrEqual(
      4 * 2 ** -23 * Math.max(1, Math.abs(expected[row]))
    );
  }
  compiled.destroy();
  parameters.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
});
