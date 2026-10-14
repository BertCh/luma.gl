// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUAssuncaoReisStandardization,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  getGPUPermutationParameterValues,
  GPULocalPermutationTest,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getReferenceWeights, REFERENCE_ROW_COUNT} from './spatial-autocorrelation-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

/** Integer counts, exact in f32: the uint32 and float32 inputs hold the same numbers. */
const COUNTS = Uint32Array.from({length: REFERENCE_ROW_COUNT}, (_, row) => (row * 7) % 13);

/**
 * Runs local Moran, the local permutation test and the Assuncao-Reis standardization on the counts
 * given as `format`, and returns the raw output bits.
 */
async function runStatistics(device: Device, format: 'uint32' | 'float32') {
  const rows = REFERENCE_ROW_COUNT;
  const csr = getReferenceWeights({rowStandardize: true});
  const capacity = csr.neighbors.length;
  const data = format === 'uint32' ? COUNTS : Float32Array.from(COUNTS);
  const buffers = {
    offsets: createInputBuffer(device, csr.offsets),
    neighbors: createInputBuffer(device, csr.neighbors),
    weights: createInputBuffer(device, csr.weights),
    values: createInputBuffer(device, data),
    populations: createInputBuffer(
      device,
      Float32Array.from({length: rows}, () => 100)
    ),
    zScores: createOutputBuffer(device, rows),
    exceedances: createOutputBuffer(device, rows),
    pseudoPValues: createOutputBuffer(device, rows),
    overflow: createOutputBuffer(device, 1),
    standardizedRates: createOutputBuffer(device, rows)
  };
  const parameters = new GPUParameterBuffer(device, {
    id: `moran-parameters-${format}`,
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    values: getGPUSpatialAutocorrelationParameterValues()
  });
  const permutationParameters = new GPUParameterBuffer(device, {
    id: `permutation-parameters-${format}`,
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH,
    values: getGPUPermutationParameterValues({seed: 3, permutationCount: 99})
  });
  const graph = new GPUCommandGraph(device, {id: `uint32-values-${format}`});
  const weights = {
    information: {rowCount: rows, transform: 'row' as const, symmetric: 'unknown' as const},
    offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', rows + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity)
  };
  const values = importGraphBuffer(graph, 'values', buffers.values, format, rows);
  graph.add(
    new GPULocalMoran({
      weights,
      values,
      parameters: parameters.importToGraph(graph),
      islands: 'keep',
      output: {zScores: importGraphBuffer(graph, 'z', buffers.zScores, 'float32', rows)}
    })
  );
  graph.add(
    new GPULocalPermutationTest({
      weights,
      values,
      statistic: 'localMoran',
      parameters: permutationParameters.importToGraph(graph),
      maximumPermutations: 99,
      output: {
        exceedances: importGraphBuffer(graph, 'exceedances', buffers.exceedances, 'uint32', rows),
        pseudoPValues: importGraphBuffer(graph, 'p', buffers.pseudoPValues, 'float32', rows),
        status: {overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)}
      }
    })
  );
  graph.add(
    new GPUAssuncaoReisStandardization({
      events: values,
      populations: importGraphBuffer(graph, 'populations', buffers.populations, 'float32', rows),
      output: {
        standardizedRates: importGraphBuffer(
          graph,
          'rates',
          buffers.standardizedRates,
          'float32',
          rows
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    zScores: await readUint32(buffers.zScores, rows),
    exceedances: await readUint32(buffers.exceedances, rows),
    pseudoPValues: await readUint32(buffers.pseudoPValues, rows),
    standardizedRates: await readUint32(buffers.standardizedRates, rows)
  };
  compiled.destroy();
  parameters.destroy();
  permutationParameters.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
  return result;
}

it('uint32 counts give bit-identical statistics to the same counts as float32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fromUint32 = await runStatistics(device, 'uint32');
  const fromFloat32 = await runStatistics(device, 'float32');
  // The conversion node writes f32(u), which is exact below 2^24, so every kernel reads the same
  // f32 values: no tolerance.
  expect(fromUint32).toEqual(fromFloat32);
  expect(fromUint32.exceedances.some(count => count !== 0xffffffff)).toBe(true);
});
