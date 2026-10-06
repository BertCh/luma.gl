// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialRegressionDiagnostics} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import type {GPUSpatialRegressionDiagnosticsProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {SPREG_DIAGNOSTICS_REFERENCE} from './spreg-reference';
import {
  computeSpatialRegressionDiagnosticsOnCPU,
  createDiagnosticsScene,
  createNearestNeighborScene,
  createLatticeWeights,
  invertMatrix
} from './spatial-regression-diagnostics-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSpatialRegressionDiagnosticsProps> = {}
): GPUSpatialRegressionDiagnosticsProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    },
    predictors: view('float32', 20),
    response: view('float32', 10),
    residuals: view('float32', 10),
    predictorCount: 2,
    output: {tests: view('float32', 18), summary: view('float32', 12), status: view('uint32', 1)},
    ...overrides
  };
}

it('GPUSpatialRegressionDiagnostics validates its inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `bad-${serial++}`, format, length);
  const expectThrows = (
    overrides: Partial<GPUSpatialRegressionDiagnosticsProps>,
    message: RegExp
  ) =>
    expect(() => new GPUSpatialRegressionDiagnostics(createProps(graph, overrides))).toThrow(
      message
    );
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 16}, /predictorCount/);
  expectThrows({response: view('float32', 9)}, /response length/);
  expectThrows({residuals: view('float32', 11)}, /residuals length/);
  expectThrows({predictors: view('float32', 19)}, /predictors length/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows(
    {output: {tests: view('float32', 17), summary: view('float32', 12), status: view('uint32', 1)}},
    /output.tests/
  );
  expectThrows(
    {output: {tests: view('float32', 18), summary: view('float32', 11), status: view('uint32', 1)}},
    /output.summary/
  );
  device.destroy();
});

it('GPUSpatialRegressionDiagnostics declares its kernels, including the weights transpose, with at most eight storage buffers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUSpatialRegressionDiagnostics(createProps(graph)).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 3)).toEqual([
    'spatial-regression-diagnostics-means-tiles',
    'spatial-regression-diagnostics-means-merge',
    'spatial-regression-diagnostics-row-table'
  ]);
  expect(ids.slice(-3)).toEqual([
    'spatial-regression-diagnostics-moments-tiles',
    'spatial-regression-diagnostics-moments-merge',
    'spatial-regression-diagnostics-finish'
  ]);
  expect(ids).toContain('spatial-regression-diagnostics-transpose-slot-keys');
  expect(ids).toContain('spatial-regression-diagnostics-transposed-lag');
  device.destroy();
});

it('spatial regression diagnostics oracle satisfies its own identities', () => {
  expect(
    invertMatrix([
      [2, 0],
      [0, 4]
    ])
  ).toEqual([
    [0.5, 0],
    [0, 0.25]
  ]);
  // Row-standardized weights: S0 = n. Binary rook weights on a 3x3 lattice: S0 = 24.
  expect(createLatticeWeights(3, false).offsets[9]).toBe(24);
  const scene = createDiagnosticsScene(6, 1, 0.5, true);
  const result = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(result.weightsSum).toBeCloseTo(36, 5);
  expect(result.lmSarma.statistic).toBeCloseTo(
    result.lmLag.statistic + result.robustLmError.statistic,
    9
  );
  // OLS residuals are orthogonal to the design and sum to zero.
  expect(result.residuals.reduce((sum, value) => sum + value, 0)).toBeCloseTo(0, 8);
  expect(result.lmError.statistic).toBeGreaterThan(0);
});

it('spatial regression diagnostics oracle reproduces spreg to float64 precision', () => {
  const cases = [
    [createDiagnosticsScene(8, 11, 0.6, true), SPREG_DIAGNOSTICS_REFERENCE.lattice],
    [createNearestNeighborScene(80, 4, 7, 0.6), SPREG_DIAGNOSTICS_REFERENCE.nearestNeighbor]
  ] as const;
  for (const [scene, reference] of cases) {
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    const pairs: [typeof result.lmLag, readonly number[]][] = [
      [result.lmLag, reference.lmLag],
      [result.lmError, reference.lmError],
      [result.robustLmLag, reference.robustLmLag],
      [result.robustLmError, reference.robustLmError],
      [result.lmSarma, reference.lmSarma]
    ];
    for (const [test, [statistic, pValue]] of pairs) {
      expect(test.statistic).toBeCloseTo(statistic, 6);
      expect(test.pValue).toBeCloseTo(pValue, 6);
    }
    expect(result.moranI).toBeCloseTo(reference.moran[0], 8);
    expect(result.moranZ).toBeCloseTo(reference.moran[1], 6);
    expect(result.moranPValue).toBeCloseTo(reference.moran[2], 6);
    expect(result.sigmaSquared).toBeCloseTo(reference.sigmaSquared, 8);
  }
});
