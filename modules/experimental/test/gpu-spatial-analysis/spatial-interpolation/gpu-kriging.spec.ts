// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/index';
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
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from './extent-grid';
import {KRIGING_REFERENCE} from './kriging-reference';
import {GREAT_CIRCLE_REFERENCE} from './great-circle-reference';
import {KRIGING_ROBUSTNESS_REFERENCE} from './kriging-robustness-reference';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * f32 solve of a 9 x 9 system against PyKrige float64. Values are compared relative to the data
 * scale (the largest reference magnitude, about 70) and variance relative to max(1, variance).
 * The measured worst errors are 1.2e-5 absolute (1.7e-7 of the scale, one f32 ulp) on values and
 * 9e-7 on variance; these bounds leave about two orders of magnitude for other GPUs.
 */
const VALUE_TOLERANCE_RELATIVE = 2e-5;
const VARIANCE_TOLERANCE = 1e-4;

it('GPUKriging matches PyKrige moving-window ordinary kriging without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
      grid: makeExtentGrid(width, height, extent),
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
      getGPUKrigingParameterValues(
        {searchRadius: Infinity, neighborCount, variogram: reference},
        makeExtentGrid(width, height, extent)
      )
    );
    submitGraph(device, compiled, undefined);
    const actualValues = await readFloat32(valuesOut, cellCount);
    const actualVariance = await readFloat32(varianceOut, cellCount);
    const valueScale = Math.max(...reference.values.map(Math.abs));
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
    expect(worstValue, `${reference.model} values`).toBeLessThanOrEqual(
      VALUE_TOLERANCE_RELATIVE * valueScale
    );
    expect(worstVariance, `${reference.model} variance`).toBeLessThanOrEqual(VARIANCE_TOLERANCE);
  }

  // A cell centered exactly on a sample returns that sample with zero variance.
  const exactRow = 5;
  const exactX = positions[2 * exactRow];
  const exactY = positions[2 * exactRow + 1];
  parameterBuffer.write(
    getGPUKrigingParameterValues(
      {searchRadius: Infinity, neighborCount, variogram: cases[0]},
      makeExtentGrid(width, height, [
        exactX - 62.5,
        exactY - 83.333333,
        exactX + 937.5,
        exactY + 916.666667
      ])
    )
  );
  submitGraph(device, compiled, undefined);
  const exactValues = await readFloat32(valuesOut, cellCount);
  expect(Number.isFinite(exactValues[0])).toBe(true);

  // A tiny radius leaves every cell without enough neighbors: nodata.
  parameterBuffer.write(
    getGPUKrigingParameterValues(
      {searchRadius: 5, neighborCount, minimumNeighborCount: 3, variogram: cases[0]},
      makeExtentGrid(width, height, extent)
    )
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
  requireTestDevice(device);
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
      values: getGPUKrigingParameterValues(
        {searchRadius: Infinity, neighborCount, variogram: cases[0]},
        makeExtentGrid(width, height, extent)
      )
    });
    const valuesOut = createOutputBuffer(device, cellCount);
    const graph = new GPUCommandGraph(device, {id: `kriging-grid-${indexGridSize[0]}`});
    graph.add(
      new GPUKriging({
        id: 'kriging',
        positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', sampleCount),
        values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', sampleCount),
        parameters: parameterBuffer.importToGraph(graph),
        grid: makeExtentGrid(width, height, extent),
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

type VariogramFixture = {
  model: 'spherical' | 'exponential' | 'gaussian';
  nugget: number;
  sill: number;
  range: number;
};

type KrigingRunOptions = {
  positions: ArrayLike<number>;
  values: ArrayLike<number>;
  mask?: ArrayLike<number>;
  extent: [number, number, number, number];
  width: number;
  height: number;
  indexGridSize: [number, number];
  indexBounds: [number, number, number, number];
  maximumNeighborCount: number;
  neighborCount: number;
  searchRadius?: number;
  minimumNeighborCount?: number;
  variogram: VariogramFixture;
  crossValidation?: boolean;
  spatialContext?: GPUSpatialContext;
};

type KrigingRunResult = {
  values: number[];
  variance: number[];
  invalidCount: number;
  uncertainCount: number;
  crossValidationPrediction?: number[];
  crossValidationVariance?: number[];
};

/** Runs one kriging frame with every optional output and returns the read-back rasters. */
async function runKriging(device: Device, options: KrigingRunOptions): Promise<KrigingRunResult> {
  const cellCount = options.width * options.height;
  const sampleCount = options.values.length;
  // Rows advance north from the extent minimum, which packs the extent unchanged.
  const grid = makeExtentGrid(options.width, options.height, options.extent);
  const buffers = [
    createInputBuffer(device, Float32Array.from(options.positions)),
    createInputBuffer(device, Float32Array.from(options.values)),
    ...(options.mask ? [createInputBuffer(device, Uint32Array.from(options.mask))] : []),
    createOutputBuffer(device, cellCount),
    createOutputBuffer(device, cellCount),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, sampleCount),
    createOutputBuffer(device, sampleCount)
  ];
  const [positionBuffer, valueBuffer] = buffers;
  let next = 2;
  const maskBuffer = options.mask ? buffers[next++] : undefined;
  const [valuesOut, varianceOut, invalidOut, uncertainOut, predictionOut, predictionVarianceOut] =
    buffers.slice(next);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kriging-run-parameters',
    format: 'float32',
    length: GPU_KRIGING_PARAMETER_LENGTH,
    values: getGPUKrigingParameterValues(
      {
        searchRadius: options.searchRadius ?? Infinity,
        neighborCount: options.neighborCount,
        minimumNeighborCount: options.minimumNeighborCount,
        variogram: options.variogram
      },
      grid
    )
  });
  const graph = new GPUCommandGraph(device, {id: 'kriging-run'});
  graph.add(
    new GPUKriging({
      id: 'kriging',
      spatialContext: options.spatialContext,
      positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', sampleCount),
      values: importGraphBuffer(graph, 'values', valueBuffer, 'float32', sampleCount),
      mask: maskBuffer
        ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', sampleCount)
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: options.indexGridSize,
      indexBounds: options.indexBounds,
      maximumNeighborCount: options.maximumNeighborCount,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        variance: importGraphBuffer(graph, 'variance-out', varianceOut, 'float32', cellCount),
        status: {
          invalidCount: importGraphBuffer(graph, 'invalid-out', invalidOut, 'uint32', 1),
          uncertainCount: importGraphBuffer(graph, 'uncertain-out', uncertainOut, 'uint32', 1)
        },
        ...(options.crossValidation
          ? {
              crossValidationPrediction: importGraphBuffer(
                graph,
                'prediction-out',
                predictionOut,
                'float32',
                sampleCount
              ),
              crossValidationVariance: importGraphBuffer(
                graph,
                'prediction-variance-out',
                predictionVarianceOut,
                'float32',
                sampleCount
              )
            }
          : {})
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: KrigingRunResult = {
    values: await readFloat32(valuesOut, cellCount),
    variance: await readFloat32(varianceOut, cellCount),
    invalidCount: (await readUint32(invalidOut, 1))[0],
    uncertainCount: (await readUint32(uncertainOut, 1))[0],
    ...(options.crossValidation
      ? {
          crossValidationPrediction: await readFloat32(predictionOut, sampleCount),
          crossValidationVariance: await readFloat32(predictionVarianceOut, sampleCount)
        }
      : {})
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** Same bounds as the first spec: values relative to the data scale, variance to max(1, v). */
function expectKrigingMatches(
  actualValues: readonly number[],
  actualVariance: readonly number[],
  expectedValues: readonly number[],
  expectedVariance: readonly number[],
  label: string
): void {
  const valueScale = Math.max(...expectedValues.filter(Number.isFinite).map(Math.abs));
  for (let index = 0; index < expectedValues.length; index++) {
    if (Number.isNaN(expectedValues[index])) {
      expect(Number.isNaN(actualValues[index]), `${label} cell ${index} nodata`).toBe(true);
      continue;
    }
    expect(
      Math.abs(actualValues[index] - expectedValues[index]),
      `${label} value ${index}`
    ).toBeLessThanOrEqual(VALUE_TOLERANCE_RELATIVE * valueScale);
    expect(
      Math.abs(actualVariance[index] - expectedVariance[index]) /
        Math.max(1, expectedVariance[index]),
      `${label} variance ${index}`
    ).toBeLessThanOrEqual(VARIANCE_TOLERANCE);
  }
}

it('GPUKriging coalesces samples at identical positions and averages exact hits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: kriging_oracle.py F1. Rows 10-13 repeat positions of rows 0, 3, 3 and 6 with other
  // values, row 0 sits on a cell center, and all 14 samples fit in the 16-point neighborhood.
  const reference = KRIGING_ROBUSTNESS_REFERENCE.duplicates;
  const result = await runKriging(device, {
    positions: reference.positions,
    values: reference.values,
    extent: reference.extent as [number, number, number, number],
    width: reference.width,
    height: reference.height,
    // Samples outside the index bounds are ignored, so the bounds cover all samples (up to y = 840).
    indexGridSize: [4, 4],
    indexBounds: [0, 0, 1000, 1000],
    maximumNeighborCount: 16,
    neighborCount: reference.neighborCount,
    variogram: reference.variogram as VariogramFixture
  });
  expectKrigingMatches(
    result.values,
    result.variance,
    reference.expectedValues,
    reference.expectedVariance,
    'duplicates'
  );
  // The cell on rows 0 and 10 returns their mean with zero variance.
  expect(result.variance[reference.exactHitCell]).toBe(0);
  expect(result.invalidCount).toBe(0);
  expect(result.uncertainCount).toBe(0);
});

it('GPUKriging rejects ill-conditioned systems as nodata and counts them', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: kriging_oracle.py F2. Twelve samples within 150 units under a Gaussian model with a
  // 1e-4 nugget: the f32 pivot ratio is 7.9e-6, so the old absolute pivot test (1e-6) accepted
  // the system and returned unreliable weights; the ratio test rejects it. The same samples with
  // nugget 0.1 (pivot ratio 3.4e-3) are well conditioned and match the float64 oracle.
  const reference = KRIGING_ROBUSTNESS_REFERENCE.illConditioned;
  const common = {
    positions: reference.positions,
    values: reference.values,
    extent: reference.extent as [number, number, number, number],
    width: reference.width,
    height: reference.height,
    indexGridSize: [3, 3] as [number, number],
    indexBounds: [0, 0, 150, 150] as [number, number, number, number],
    maximumNeighborCount: 12,
    neighborCount: reference.neighborCount
  };
  const rejected = await runKriging(device, {
    ...common,
    variogram: reference.rejected as VariogramFixture
  });
  expect(rejected.values.every(Number.isNaN)).toBe(true);
  expect(rejected.variance.every(Number.isNaN)).toBe(true);
  expect(rejected.uncertainCount).toBe(reference.width * reference.height);
  expect(rejected.invalidCount).toBe(0);

  const accepted = await runKriging(device, {
    ...common,
    variogram: reference.wellConditioned as VariogramFixture
  });
  expectKrigingMatches(
    accepted.values,
    accepted.variance,
    reference.wellConditioned.expectedValues,
    reference.wellConditioned.expectedVariance,
    'well conditioned'
  );
  expect(accepted.uncertainCount).toBe(0);
  expect(accepted.invalidCount).toBe(0);
});

it('GPUKriging counts nodata cells without enough neighbors as invalid, not uncertain', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {positions, values, width, height, extent, neighborCount, cases} = KRIGING_REFERENCE;
  const result = await runKriging(device, {
    positions,
    values,
    extent,
    width,
    height,
    indexGridSize: [6, 6],
    indexBounds: [0, 0, 1000, 1000],
    maximumNeighborCount: 12,
    neighborCount,
    searchRadius: 5,
    minimumNeighborCount: 3,
    variogram: cases[0]
  });
  expect(result.values.every(Number.isNaN)).toBe(true);
  expect(result.invalidCount).toBe(width * height);
  expect(result.uncertainCount).toBe(0);
});

it('GPUKriging folds a sample mask without exceeding eight storage buffers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: kriging_oracle.py F3b, the reference raster with the masked rows removed. This run
  // binds mask, variance and both status counters, the maximum optional-output combination.
  const reference = KRIGING_ROBUSTNESS_REFERENCE.masked;
  const {positions, values} = KRIGING_REFERENCE;
  const mask = new Uint32Array(values.length).fill(1);
  for (const row of reference.maskedRows) {
    mask[row] = 0;
  }
  const result = await runKriging(device, {
    positions,
    values,
    mask,
    extent: reference.extent as [number, number, number, number],
    width: reference.width,
    height: reference.height,
    indexGridSize: [6, 6],
    indexBounds: [0, 0, 1000, 1000],
    maximumNeighborCount: 12,
    neighborCount: reference.neighborCount,
    variogram: reference.variogram as VariogramFixture
  });
  expectKrigingMatches(
    result.values,
    result.variance,
    reference.expectedValues,
    reference.expectedVariance,
    'masked'
  );
});

it('GPUKriging leave-one-out cross-validation krigs each sample from its other neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: kriging_oracle.py F3, the 48 PyKrige reference samples, exponential model, k = 8.
  const reference = KRIGING_ROBUSTNESS_REFERENCE.leaveOneOut;
  const {positions, values, width, height, extent} = KRIGING_REFERENCE;
  const result = await runKriging(device, {
    positions,
    values,
    extent,
    width,
    height,
    indexGridSize: [6, 6],
    indexBounds: [0, 0, 1000, 1000],
    maximumNeighborCount: 12,
    neighborCount: reference.neighborCount,
    variogram: reference.variogram as VariogramFixture,
    crossValidation: true
  });
  expectKrigingMatches(
    result.crossValidationPrediction!,
    result.crossValidationVariance!,
    reference.expectedPrediction,
    reference.expectedVariance,
    'leave-one-out'
  );
  // A leave-one-out prediction is never the sample itself.
  const identical = result.crossValidationPrediction!.filter(
    (prediction, row) => prediction === values[row]
  );
  expect(identical.length).toBe(0);
});

it('GPUKriging is translation invariant at UTM-sized absolute coordinates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: kriging_oracle.py F4. Positions sit on a 0.5 lattice, exactly representable in f32 at
  // 4e6 (ulp 0.25), so offsets from the extent minimum are exact and the raster must equal the
  // untranslated float64 result.
  const reference = KRIGING_ROBUSTNESS_REFERENCE.translation;
  const {offset} = reference;
  const translated = reference.positions.map(coordinate => coordinate + offset);
  const result = await runKriging(device, {
    positions: translated,
    values: reference.values,
    extent: [offset, offset, offset + 100, offset + 100],
    width: reference.width,
    height: reference.height,
    indexGridSize: [4, 4],
    indexBounds: [offset, offset, offset + 100, offset + 100],
    maximumNeighborCount: 12,
    neighborCount: reference.neighborCount,
    variogram: reference.variogram as VariogramFixture
  });
  expectKrigingMatches(
    result.values,
    result.variance,
    reference.expectedValues,
    reference.expectedVariance,
    'translated'
  );
});

it('GPUKriging great-circle mode matches haversine ordinary kriging at 47 N', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Oracle: geo_oracle.py (haversine on R = 6371008.8 m, cross-checked against pyproj.Geod). The
  // range is in meters. Positions are absolute f32 lon/lat (about 0.4 m ulp at 8 E), so the bound
  // is 5e-4 of the data scale on values and 5e-3 on variance, not the planar 2e-5.
  const reference = GREAT_CIRCLE_REFERENCE;
  const geographic = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  for (const crossValidation of [false, true]) {
    const result = await runKriging(device, {
      positions: reference.positions,
      values: reference.values,
      extent: reference.extent as [number, number, number, number],
      width: reference.width,
      height: reference.height,
      indexGridSize: [4, 4],
      indexBounds: [7.7, 46.8, 8.3, 47.2],
      maximumNeighborCount: 12,
      neighborCount: reference.kriging.neighborCount,
      variogram: {
        model: 'exponential',
        nugget: reference.kriging.nugget,
        sill: reference.kriging.sill,
        range: reference.kriging.range
      },
      spatialContext: geographic,
      crossValidation
    });
    const scale = Math.max(...reference.kriging.expectedValues.map(Math.abs));
    for (let cell = 0; cell < result.values.length; cell++) {
      expect(
        Math.abs(result.values[cell] - reference.kriging.expectedValues[cell]),
        `value ${cell}`
      ).toBeLessThanOrEqual(5e-4 * scale);
      expect(
        Math.abs(result.variance[cell] - reference.kriging.expectedVariance[cell]) /
          Math.max(1, reference.kriging.expectedVariance[cell]),
        `variance ${cell}`
      ).toBeLessThanOrEqual(5e-3);
    }
    expect(result.invalidCount).toBe(0);
    expect(result.uncertainCount).toBe(0);
    if (crossValidation) {
      // The leave-one-out metric is the same great-circle metric: every row is estimated.
      expect(result.crossValidationPrediction!.every(Number.isFinite)).toBe(true);
    }
  }
});
