// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {
  GPU_GWR_NONSTATIONARITY_SUMMARY,
  GPU_GWR_NONSTATIONARITY_TABLE,
  GPU_GWR_NONSTATIONARITY_TABLE_STRIDE
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-nonstationarity-parameters';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {MGWR_CONDITION_NUMBER_REFERENCE, MGWR_NONSTATIONARITY_REFERENCE} from './mgwr-reference';
import {
  createNonstationarityScene,
  type NonstationarityScene
} from './geographically-weighted-regression-nonstationarity-oracle';

const LADDER_CAPACITY = 4;
const MAXIMUM_PERMUTATIONS = 32;

type Result = {
  conditionNumbers: number[];
  table: number[];
  summary: number[];
  standardDeviations: number[];
};

/** Runs the regression and the Monte Carlo test in one graph, like the explorer does. */
async function runTest(
  device: Device,
  scene: NonstationarityScene,
  settings: GPUGeographicallyWeightedRegressionSettings & {kernel?: 'gaussian' | 'bisquare'},
  seed: number,
  permutations: number,
  tileRowCount?: number,
  mask?: Uint32Array,
  indexGridSize?: false | readonly [number, number]
): Promise<Result> {
  const rows = scene.response.length;
  const {predictorCount} = scene;
  const p = predictorCount + 1;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const bandwidthParameters = new GPUParameterBuffer(device, {
    id: 'bandwidth-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
  });
  const permutationParameters = new GPUParameterBuffer(device, {
    id: 'permutation-parameters',
    format: 'uint32',
    length: 4
  });
  bandwidthParameters.write(
    getGPUGeographicallyWeightedRegressionParameterValues(settings, LADDER_CAPACITY)
  );
  permutationParameters.write(getGPUPermutationParameterValues({seed, permutations}));
  const graph = new GPUCommandGraph(device, {id: 'nonstationarity-graph'});
  const outputs = {
    coefficients: track(createOutputBuffer(device, rows * p)),
    conditionNumber: track(createOutputBuffer(device, rows)),
    selectedBandwidth: track(createOutputBuffer(device, 2)),
    table: track(createOutputBuffer(device, p * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE)),
    summary: track(createOutputBuffer(device, GPU_GWR_NONSTATIONARITY_SUMMARY.length)),
    standardDeviations: track(createOutputBuffer(device, (MAXIMUM_PERMUTATIONS + 1) * p))
  };
  const input = (
    name: string,
    data: Float32Array | Uint32Array,
    format: 'float32' | 'uint32' | 'float32x2',
    length: number
  ) => importGraphBuffer(graph, name, track(createInputBuffer(device, data)), format, length);
  const positions = input('positions', scene.positions, 'float32x2', rows);
  const predictors = input('predictors', scene.predictors, 'float32', rows * predictorCount);
  const response = input('response', scene.response, 'float32', rows);
  const maskView = mask ? input('mask', mask, 'uint32', rows) : undefined;
  const float = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', length);
  const coefficients = float('coefficients', rows * p);
  const selectedBandwidth = float('selectedBandwidth', 2);
  const parameters = bandwidthParameters.importToGraph(graph);
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions,
      predictors,
      predictorCount,
      response,
      mask: maskView,
      parameters,
      maximumBandwidthCount: LADDER_CAPACITY,
      maximumNeighborCount: 32,
      indexGridSize,
      output: {
        coefficients,
        selectedBandwidth,
        localConditionNumber: float('conditionNumber', rows)
      }
    })
  );
  graph.add(
    new GPUGeographicallyWeightedRegressionNonstationarityTest({
      id: 'test',
      positions,
      predictors,
      predictorCount,
      response,
      mask: maskView,
      bandwidthParameters: parameters,
      selectedBandwidth,
      coefficients,
      parameters: permutationParameters.importToGraph(graph),
      maximumPermutations: MAXIMUM_PERMUTATIONS,
      maximumBandwidthCount: LADDER_CAPACITY,
      maximumNeighborCount: 32,
      tileRowCount,
      indexGridSize,
      output: {
        table: float('table', p * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE),
        summary: float('summary', GPU_GWR_NONSTATIONARITY_SUMMARY.length),
        standardDeviations: float('standardDeviations', (MAXIMUM_PERMUTATIONS + 1) * p)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    conditionNumbers: await readFloat32(outputs.conditionNumber, rows),
    table: await readFloat32(outputs.table, p * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE),
    summary: await readFloat32(outputs.summary, GPU_GWR_NONSTATIONARITY_SUMMARY.length),
    standardDeviations: await readFloat32(
      outputs.standardDeviations,
      (MAXIMUM_PERMUTATIONS + 1) * p
    )
  };
  compiled.destroy();
  bandwidthParameters.destroy();
  permutationParameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectRelativelyClose(actual: number, expected: number, tolerance: number, label: string) {
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance * Math.max(Math.abs(expected), 1)
  );
}

it('GPUGeographicallyWeightedRegressionNonstationarityTest matches mgwr permutation fits', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createNonstationarityScene(12, 77);
  const reference = MGWR_NONSTATIONARITY_REFERENCE;
  for (const [name, settings] of [
    ['fixed', {bandwidths: [2.2]}],
    ['adaptive', {bandwidths: [20], bandwidthMode: 'adaptive' as const}]
  ] as const) {
    const expected = reference[name];
    const actual = await runTest(device, scene, settings, reference.seed, reference.permutations);
    // Every permutation's spread of local estimates equals mgwr's.
    expected.standardDeviations.forEach((row, run) =>
      row.forEach((value, column) =>
        expectRelativelyClose(
          actual.standardDeviations[run * 3 + column],
          value,
          2e-3,
          `${name} sd ${run}/${column}`
        )
      )
    );
    const stride = GPU_GWR_NONSTATIONARITY_TABLE_STRIDE;
    expected.exceedances.forEach((count, column) => {
      expect(actual.table[column * stride + GPU_GWR_NONSTATIONARITY_TABLE.exceedances]).toBe(count);
      expectRelativelyClose(
        actual.table[column * stride + GPU_GWR_NONSTATIONARITY_TABLE.pseudoPValue],
        (count + 1) / (reference.permutations + 1),
        1e-6,
        `${name} p ${column}`
      );
    });
    // The varying slope is significant, the constant one is not.
    expect(actual.table[stride + GPU_GWR_NONSTATIONARITY_TABLE.pseudoPValue]).toBeLessThan(0.1);
    expect(actual.summary[GPU_GWR_NONSTATIONARITY_SUMMARY.permutations]).toBe(
      reference.permutations
    );
    expect(actual.summary[GPU_GWR_NONSTATIONARITY_SUMMARY.locationCount]).toBe(144);
    expect(actual.summary[GPU_GWR_NONSTATIONARITY_SUMMARY.failedFitCount]).toBe(0);
    // The regression's local condition number is mgwr's local_collinearity CN.
    expected.conditionNumbers.forEach((value, row) =>
      expectRelativelyClose(actual.conditionNumbers[row], value, 5e-3, `${name} CN ${row}`)
    );
  }
  // Reproducible for a seed (bitwise for one tiling, to rounding across tilings); the seed changes the draw.
  const settings = {bandwidths: [2.2]};
  const first = await runTest(device, scene, settings, 99, 16);
  expect((await runTest(device, scene, settings, 99, 16)).standardDeviations).toEqual(
    first.standardDeviations
  );
  const retiled = await runTest(device, scene, settings, 99, 16, 144);
  retiled.standardDeviations.forEach((value, index) => {
    const expected = first.standardDeviations[index];
    if (Number.isNaN(expected)) {
      expect(Number.isNaN(value)).toBe(true);
    } else {
      expectRelativelyClose(value, expected, 1e-4, `retiled sd ${index}`);
    }
  });
  const reseeded = await runTest(device, scene, settings, 100, 16);
  expect(reseeded.standardDeviations[3]).not.toBe(first.standardDeviations[3]);
  // A smaller permutation count leaves the unused runs NaN.
  expect(Number.isNaN(first.standardDeviations[17 * 3])).toBe(true);
});

it('GPUGeographicallyWeightedRegression localConditionNumber matches mgwr local_collinearity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const base = createNonstationarityScene(12, 77);
  const collinear = Float32Array.from(base.predictors);
  for (let row = 0; row < 144; row++) {
    collinear[row * 2 + 1] = Math.fround(0.8 * collinear[row * 2] + 0.25 * collinear[row * 2 + 1]);
  }
  const scene = {...base, predictors: collinear};
  for (const reference of Object.values(MGWR_CONDITION_NUMBER_REFERENCE)) {
    const actual = await runTest(
      device,
      scene,
      {
        kernel: reference.kernel as 'gaussian' | 'bisquare',
        bandwidths: [reference.bandwidth],
        bandwidthMode: reference.fixed ? 'fixed' : 'adaptive'
      },
      1,
      1
    );
    reference.conditionNumbers.forEach((value, row) =>
      expectRelativelyClose(
        actual.conditionNumbers[row],
        value,
        2e-2,
        `${reference.kernel} CN ${row}`
      )
    );
  }
});

it('GPUGeographicallyWeightedRegressionNonstationarityTest grid refits match the scan', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createNonstationarityScene(12, 77);
  const rows = scene.response.length;
  const mask = new Uint32Array(rows).fill(1);
  for (let row = 3; row < rows; row += 11) {
    mask[row] = 0;
  }
  for (const [name, settings] of [
    ['bisquare fixed', {bandwidths: [2.2]}],
    ['bisquare adaptive', {bandwidths: [20], bandwidthMode: 'adaptive' as const}],
    [
      'gaussian adaptive',
      {kernel: 'gaussian' as const, bandwidths: [25], bandwidthMode: 'adaptive' as const}
    ]
  ] as const) {
    const scan = await runTest(device, scene, settings, 7, 12, undefined, mask, false);
    for (const gridSize of [
      [6, 6],
      [2, 3]
    ] as const) {
      const grid = await runTest(device, scene, settings, 7, 12, undefined, mask, gridSize);
      scan.standardDeviations.forEach((value, index) => {
        if (Number.isNaN(value)) {
          expect(Number.isNaN(grid.standardDeviations[index])).toBe(true);
        } else {
          expectRelativelyClose(
            grid.standardDeviations[index],
            value,
            2e-3,
            `${name} ${gridSize} sd ${index}`
          );
        }
      });
      expect(grid.summary).toEqual(scan.summary);
      expect(grid.table[GPU_GWR_NONSTATIONARITY_TABLE.exceedances]).toBe(
        scan.table[GPU_GWR_NONSTATIONARITY_TABLE.exceedances]
      );
    }
  }
});
