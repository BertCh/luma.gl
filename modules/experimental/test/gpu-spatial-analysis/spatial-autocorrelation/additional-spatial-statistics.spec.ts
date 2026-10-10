// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUGammaStatistic,
  GPULocalGeary,
  GPUSpatialPearson
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('additional spatial statistics match direct CPU edge oracles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const offsets = Uint32Array.from([0, 1, 3, 5, 6]);
  const neighbors = Uint32Array.from([1, 0, 2, 1, 3, 2]);
  const edgeWeights = Float32Array.from([1, 1, 1, 1, 1, 1]);
  const values = Float32Array.from([1, 2, 4, 8]);
  const secondValues = Float32Array.from([2, 0, 5, 9]);
  const parameters = getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05});
  const graph = new GPUCommandGraph(device, {id: 'additional-spatial-statistics'});
  const inputBuffers = {
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, edgeWeights),
    values: createInputBuffer(device, values),
    secondValues: createInputBuffer(device, secondValues),
    parameters: createInputBuffer(device, parameters)
  };
  const outputBuffers = {
    localGeary: createOutputBuffer(device, values.length),
    pearson: createOutputBuffer(device, 1),
    gamma: createOutputBuffer(device, 1)
  };
  const weights = {
    offsets: importGraphBuffer(graph, 'offsets', inputBuffers.offsets, 'uint32', offsets.length),
    neighbors: importGraphBuffer(
      graph,
      'neighbors',
      inputBuffers.neighbors,
      'uint32',
      neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      'weights',
      inputBuffers.weights,
      'float32',
      edgeWeights.length
    )
  };
  const valueView = importGraphBuffer(
    graph,
    'values',
    inputBuffers.values,
    'float32',
    values.length
  );
  const secondView = importGraphBuffer(
    graph,
    'second-values',
    inputBuffers.secondValues,
    'float32',
    secondValues.length
  );
  graph.add(
    new GPULocalGeary({
      weights,
      values: valueView,
      parameters: importGraphBuffer(
        graph,
        'parameters',
        inputBuffers.parameters,
        'float32',
        parameters.length
      ),
      result: {
        statistic: importGraphBuffer(
          graph,
          'local-geary',
          outputBuffers.localGeary,
          'float32',
          values.length
        )
      }
    })
  );
  graph.add(
    new GPUSpatialPearson({
      weights,
      values: valueView,
      secondValues: secondView,
      result: {statistic: importGraphBuffer(graph, 'pearson', outputBuffers.pearson, 'float32', 1)}
    })
  );
  graph.add(
    new GPUGammaStatistic({
      weights,
      values: valueView,
      operation: 'squared-difference',
      result: {statistic: importGraphBuffer(graph, 'gamma', outputBuffers.gamma, 'float32', 1)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [localGeary, pearson, gamma] = await Promise.all([
    readFloat32(outputBuffers.localGeary, values.length),
    readFloat32(outputBuffers.pearson, 1),
    readFloat32(outputBuffers.gamma, 1)
  ]);

  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const expectedLocal = values.map((value, row) => {
    let result = 0;
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++)
      result += (value - values[neighbors[slot]]) ** 2 / variance;
    return result;
  });
  const edgePairs = Array.from(neighbors, (neighbor, slot) => {
    let row = 0;
    while (offsets[row + 1] <= slot) row++;
    return [values[row], secondValues[neighbor]] as const;
  });
  const meanX = edgePairs.reduce((sum, pair) => sum + pair[0], 0) / edgePairs.length;
  const meanY = edgePairs.reduce((sum, pair) => sum + pair[1], 0) / edgePairs.length;
  const covariance = edgePairs.reduce(
    (sum, pair) => sum + (pair[0] - meanX) * (pair[1] - meanY),
    0
  );
  const squareX = edgePairs.reduce((sum, pair) => sum + (pair[0] - meanX) ** 2, 0);
  const squareY = edgePairs.reduce((sum, pair) => sum + (pair[1] - meanY) ** 2, 0);
  const expectedPearson = covariance / Math.sqrt(squareX * squareY);
  const expectedGamma =
    Array.from(neighbors, (neighbor, slot) => {
      let row = 0;
      while (offsets[row + 1] <= slot) row++;
      return (values[row] - values[neighbor]) ** 2;
    }).reduce((sum, value) => sum + value, 0) / neighbors.length;
  localGeary.forEach((value, row) => expect(value).toBeCloseTo(expectedLocal[row], 5));
  expect(pearson[0]).toBeCloseTo(expectedPearson, 5);
  expect(gamma[0]).toBeCloseTo(expectedGamma, 5);
  compiled.destroy();
  Object.values(inputBuffers).forEach(buffer => buffer.destroy());
  Object.values(outputBuffers).forEach(buffer => buffer.destroy());
});
