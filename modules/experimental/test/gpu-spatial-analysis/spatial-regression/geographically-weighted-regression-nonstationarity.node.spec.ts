// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGeographicallyWeightedRegressionNonstationarityTest,
  type GPUGeographicallyWeightedRegressionNonstationarityTestProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {getGPUGeographicallyWeightedRegressionParameterLength} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {MGWR_NONSTATIONARITY_REFERENCE} from './mgwr-reference';
import {
  computeNonstationarityOnCPU,
  createNonstationarityScene
} from './geographically-weighted-regression-nonstationarity-oracle';

let serial = 0;

it('GPUGeographicallyWeightedRegressionNonstationarityTest validates its inputs and declares its kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const createProps = (
    overrides: Partial<GPUGeographicallyWeightedRegressionNonstationarityTestProps> = {}
  ): GPUGeographicallyWeightedRegressionNonstationarityTestProps => ({
    positions: view('float32x2', 10),
    predictors: view('float32', 20),
    predictorCount: 2,
    response: view('float32', 10),
    bandwidthParameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength()),
    selectedBandwidth: view('float32', 2),
    coefficients: view('float32', 30),
    parameters: view('uint32', 4),
    maximumPermutations: 9,
    output: {table: view('float32', 18), summary: view('float32', 4)},
    ...overrides
  });
  const expectThrows = (
    overrides: Partial<GPUGeographicallyWeightedRegressionNonstationarityTestProps>,
    message: RegExp
  ) =>
    expect(
      () => new GPUGeographicallyWeightedRegressionNonstationarityTest(createProps(overrides))
    ).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 8}, /predictorCount/);
  expectThrows({maximumPermutations: 0}, /maximumPermutations/);
  expectThrows({predictors: view('float32', 19)}, /predictors length/);
  expectThrows({response: view('float32', 9)}, /response length/);
  expectThrows({coefficients: view('float32', 29)}, /coefficients/);
  expectThrows({selectedBandwidth: view('float32', 1)}, /selectedBandwidth/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows({output: {table: view('float32', 17), summary: view('float32', 4)}}, /output.table/);
  expectThrows(
    {
      output: {
        table: view('float32', 18),
        summary: view('float32', 4),
        standardDeviations: view('float32', 29)
      }
    },
    /standardDeviations/
  );
  const nodes = new GPUGeographicallyWeightedRegressionNonstationarityTest(
    createProps({mask: view('uint32', 10)})
  ).getCommandNodes(graph);
  expect(nodes.map(node => node.id.replace('gwr-nonstationarity-test-', ''))).toEqual([
    'validate',
    'tile-counts',
    'tile-offsets',
    'compact',
    'bandwidths',
    'anchor',
    'observed-tiles',
    'permuted-tiles',
    'merge',
    'finish'
  ]);
  device.destroy();
});

it('nonstationarity oracle reproduces mgwr permutation fits (fixed and adaptive bandwidths)', () => {
  const scene = createNonstationarityScene(12, 77);
  const reference = MGWR_NONSTATIONARITY_REFERENCE;
  for (const [name, settings, tolerance] of [
    ['fixed', {bandwidths: [2.2]}, 1e-9],
    // The oracle uses the 1.00001 adaptive factor (f32 safe); mgwr uses 1.0000001.
    ['adaptive', {bandwidths: [20], bandwidthMode: 'adaptive' as const}, 1e-3]
  ] as const) {
    const oracle = computeNonstationarityOnCPU(
      scene,
      settings,
      reference.permutations,
      reference.seed
    );
    const expected = reference[name];
    expect(oracle.standardDeviations.length).toBe(expected.standardDeviations.length);
    expected.standardDeviations.forEach((row, run) =>
      row.forEach((value, column) =>
        expect(
          Math.abs(oracle.standardDeviations[run][column] - value),
          `${name} ${run}/${column}`
        ).toBeLessThan(tolerance)
      )
    );
    expect(oracle.exceedances).toEqual(expected.exceedances);
  }
});
