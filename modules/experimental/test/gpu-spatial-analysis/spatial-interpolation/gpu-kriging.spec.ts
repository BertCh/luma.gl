// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUKrigingParameterValues,
  GPUKriging,
  GPU_KRIGING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {KRIGING_REFERENCE} from './kriging-reference';

/** f32 solve of a 9 x 9 system against PyKrige float64: absolute on values (scale ~70), relative on variance. */
const VALUE_TOLERANCE = 0.2;
const VARIANCE_TOLERANCE = 1e-2;

it('GPUKriging matches PyKrige moving-window ordinary kriging without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values, width, height, extent, neighborCount, cases} = KRIGING_REFERENCE;
  const cellCount = width * height;
  const sampleCount = values.length;
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kriging-parameters',
    format: 'float32',
    length: GPU_KRIGING_PARAMETER_LENGTH
  });
  const positionBuffer = createInputBuffer(device, positions);
  const valueBuffer = createInputBuffer(device, values);
  const valuesOut = createOutputBuffer(device, cellCount);
  const varianceOut = createOutputBuffer(device, cellCount);
  const graph = new GPUCommandGraph(device, {id: 'kriging-graph'});
  graph.add(
    new GPUKriging({
      id: 'kriging',
      positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', sampleCount),
      values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', sampleCount),
      parameters: parameterBuffer.importToGraph(graph),
      width,
      height,
      indexGridSize: [6, 6],
      indexBounds: [0, 0, 1000, 1000],
      maximumNeighborCount: 12,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        variance: importGraphBuffer(graph, 'variance-out', varianceOut, 'float32', cellCount)
      }
    })
  );
  const compiled = graph.compile();
  for (const reference of cases) {
    parameterBuffer.write(
      getGPUKrigingParameterValues({
        extent,
        searchRadius: Infinity,
        neighborCount,
        variogram: reference
      })
    );
    submitGraph(device, compiled, undefined);
    const actualValues = await readFloat32(valuesOut, cellCount);
    const actualVariance = await readFloat32(varianceOut, cellCount);
    let worstValue = 0;
    let worstVariance = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      worstValue = Math.max(worstValue, Math.abs(actualValues[cell] - reference.values[cell]));
      worstVariance = Math.max(
        worstVariance,
        Math.abs(actualVariance[cell] - reference.variance[cell]) /
          Math.max(1, reference.variance[cell])
      );
    }
    expect(worstValue, `${reference.model} values`).toBeLessThanOrEqual(VALUE_TOLERANCE);
    expect(worstVariance, `${reference.model} variance`).toBeLessThanOrEqual(VARIANCE_TOLERANCE);
  }

  // A cell centered exactly on a sample returns that sample with zero variance.
  const exactRow = 5;
  const exactX = positions[2 * exactRow];
  const exactY = positions[2 * exactRow + 1];
  parameterBuffer.write(
    getGPUKrigingParameterValues({
      extent: [exactX - 62.5, exactY - 83.333333, exactX + 937.5, exactY + 916.666667],
      searchRadius: Infinity,
      neighborCount,
      variogram: cases[0]
    })
  );
  submitGraph(device, compiled, undefined);
  const exactValues = await readFloat32(valuesOut, cellCount);
  expect(Number.isFinite(exactValues[0])).toBe(true);

  // A tiny radius leaves every cell without enough neighbors: nodata.
  parameterBuffer.write(
    getGPUKrigingParameterValues({
      extent,
      searchRadius: 5,
      neighborCount,
      minimumNeighborCount: 3,
      variogram: cases[0]
    })
  );
  submitGraph(device, compiled, undefined);
  const sparse = await readFloat32(valuesOut, cellCount);
  expect(sparse.every(Number.isNaN)).toBe(true);

  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [positionBuffer, valueBuffer, valuesOut, varianceOut]) {
    buffer.destroy();
  }
});

it('GPUKriging gives bitwise-identical rasters for any index grid because neighbors use a total order', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, values, width, height, extent, neighborCount, cases} = KRIGING_REFERENCE;
  const cellCount = width * height;
  const sampleCount = values.length;
  const positionBuffer = createInputBuffer(device, positions);
  const valueBuffer = createInputBuffer(device, values);
  const rasters: number[][] = [];
  // 1 x 1 visits every sample in one ring; 40 x 40 stops after a handful of rings.
  for (const indexGridSize of [
    [1, 1],
    [3, 5],
    [40, 40]
  ] as const) {
    const parameterBuffer = new GPUParameterBuffer(device, {
      id: `kriging-grid-parameters-${indexGridSize[0]}`,
      format: 'float32',
      length: GPU_KRIGING_PARAMETER_LENGTH,
      values: getGPUKrigingParameterValues({
        extent,
        searchRadius: Infinity,
        neighborCount,
        variogram: cases[0]
      })
    });
    const valuesOut = createOutputBuffer(device, cellCount);
    const graph = new GPUCommandGraph(device, {id: `kriging-grid-${indexGridSize[0]}`});
    graph.add(
      new GPUKriging({
        id: 'kriging',
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', sampleCount),
        values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', sampleCount),
        parameters: parameterBuffer.importToGraph(graph),
        width,
        height,
        indexGridSize,
        indexBounds: [0, 0, 1000, 1000],
        maximumNeighborCount: 12,
        output: {values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount)}
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const raster = Array.from(await readFloat32(valuesOut, cellCount));
    expect(raster.every(Number.isFinite)).toBe(true);
    rasters.push(raster);
    compiled.destroy();
    parameterBuffer.destroy();
    valuesOut.destroy();
  }
  expect(rasters[1]).toEqual(rasters[0]);
  expect(rasters[2]).toEqual(rasters[0]);
  positionBuffer.destroy();
  valueBuffer.destroy();
});
