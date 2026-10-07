// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import type {GPUSpatialTwoStageLeastSquaresProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {
  SPREG_ASYMMETRIC_KNN_REFERENCE,
  SPREG_TWO_STAGE_ORDER_TWO_REFERENCE,
  SPREG_TWO_STAGE_REFERENCE
} from './spreg-reference';
import {createNearestNeighborScene} from './spatial-regression-diagnostics-oracle';
import {
  createLagScene,
  fitSpatialTwoStageLeastSquaresOnCPU
} from './spatial-two-stage-least-squares-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

it('GPUSpatialTwoStageLeastSquares validates its inputs and declares ten kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const createProps = (
    overrides: Partial<GPUSpatialTwoStageLeastSquaresProps> = {}
  ): GPUSpatialTwoStageLeastSquaresProps => ({
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    },
    predictors: view('float32', 20),
    response: view('float32', 10),
    predictorCount: 2,
    output: {table: view('float32', 16), summary: view('float32', 7), status: view('uint32', 1)},
    ...overrides
  });
  const expectThrows = (overrides: Partial<GPUSpatialTwoStageLeastSquaresProps>, message: RegExp) =>
    expect(() => new GPUSpatialTwoStageLeastSquares(createProps(overrides))).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 9}, /predictorCount/);
  expectThrows({response: view('float32', 9)}, /response length/);
  expectThrows({predictors: view('float32', 19)}, /predictors length/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows({instrumentOrder: 3 as 2}, /instrumentOrder/);
  expectThrows(
    {output: {table: view('float32', 15), summary: view('float32', 7), status: view('uint32', 1)}},
    /output.table/
  );
  const nodes = new GPUSpatialTwoStageLeastSquares(createProps()).getCommandNodes(graph);
  const ids = nodes.map(node => node.id.replace('spatial-two-stage-least-squares-', ''));
  expect(ids.filter(id => !id.startsWith('transpose-'))).toEqual([
    'means-tiles',
    'means-merge',
    'moments-tiles',
    'moments-merge',
    'solve',
    'residuals-tiles',
    'residuals-merge',
    'anselin-kelejian-tiles',
    'transposed-lag-residual',
    'anselin-kelejian-products',
    'anselin-kelejian-merge',
    'finish'
  ]);
  expect(ids).toContain('transpose-slot-keys');
  device.destroy();
});

it('spatial two-stage oracle reproduces spreg GM_Lag to float64 precision', () => {
  const scene = createLagScene(12, 21, 0.5, true);
  const result = fitSpatialTwoStageLeastSquaresOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const reference = SPREG_TWO_STAGE_REFERENCE;
  reference.betas.forEach((beta, index) => {
    expect(result.coefficients[index]).toBeCloseTo(beta, 6);
    expect(result.standardErrors[index]).toBeCloseTo(reference.standardErrors[index], 6);
    expect(result.zStatistics[index]).toBeCloseTo(reference.z[index], 4);
  });
  expect(result.sigmaSquared).toBeCloseTo(reference.sigma2, 8);
  expect(result.pseudoRSquared).toBeCloseTo(reference.pseudoRSquared, 8);
  expect(result.anselinKelejian).toBeCloseTo(reference.anselinKelejian[0], 6);
  expect(result.anselinKelejianPValue).toBeCloseTo(reference.anselinKelejian[1], 6);
});

it('spatial two-stage oracle reproduces spreg GM_Lag with w_lags = 2 (instrument order 2)', () => {
  const lattice = createLagScene(12, 21, 0.5, true);
  const scenes = [
    ['lattice', lattice, SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.lattice],
    [
      'knn 60',
      createNearestNeighborScene(60, 3, 13, 0.6, false),
      SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.scene_60_13
    ],
    [
      'knn 90',
      createNearestNeighborScene(90, 3, 29, 0.6, false),
      SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.scene_90_29
    ]
  ] as const;
  for (const [name, scene, reference] of scenes) {
    const result = fitSpatialTwoStageLeastSquaresOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount,
      2
    );
    reference.betas.forEach((beta, index) => {
      expect(result.coefficients[index], `${name} beta ${index}`).toBeCloseTo(beta, 6);
      expect(result.standardErrors[index], `${name} se ${index}`).toBeCloseTo(
        reference.standardErrors[index],
        6
      );
    });
    expect(result.sigmaSquared).toBeCloseTo(reference.sigma2, 8);
    expect(result.pseudoRSquared).toBeCloseTo(reference.pseudoRSquared, 8);
    expect(result.anselinKelejian).toBeCloseTo(reference.anselinKelejian[0], 5);
  }
  // Order 2 genuinely differs from order 1.
  const first = fitSpatialTwoStageLeastSquaresOnCPU(
    lattice.weights,
    lattice.predictors,
    lattice.response,
    lattice.predictorCount
  );
  expect(
    Math.abs(first.coefficients[0] - SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.lattice.betas[0])
  ).toBeGreaterThan(0.1);
  expect(SPREG_ASYMMETRIC_KNN_REFERENCE.scene_60_13.twoStage.betas.length).toBe(4);
});
