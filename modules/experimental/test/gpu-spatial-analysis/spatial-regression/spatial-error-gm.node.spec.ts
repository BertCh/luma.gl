// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import type {GPUSpatialErrorGMProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {SPREG_ERROR_GM_REFERENCE} from './spatial-error-gm-reference';
import {createErrorScene, fitSpatialErrorGMOnCPU} from './spatial-error-gm-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

it('GPUSpatialErrorGM validates its inputs and declares its nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const createProps = (
    overrides: Partial<GPUSpatialErrorGMProps> = {}
  ): GPUSpatialErrorGMProps => ({
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    },
    predictors: view('float32', 20),
    response: view('float32', 10),
    predictorCount: 2,
    output: {table: view('float32', 16), summary: view('float32', 13), status: view('uint32', 1)},
    ...overrides
  });
  const expectThrows = (overrides: Partial<GPUSpatialErrorGMProps>, message: RegExp) =>
    expect(() => new GPUSpatialErrorGM(createProps(overrides))).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 16}, /predictorCount/);
  expectThrows({estimator: 'ml' as 'homoskedastic'}, /estimator/);
  expectThrows({response: view('float32', 9)}, /response length/);
  expectThrows({predictors: view('float32', 19)}, /predictors length/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows(
    {output: {table: view('float32', 15), summary: view('float32', 13), status: view('uint32', 1)}},
    /output.table/
  );
  expectThrows(
    {output: {table: view('float32', 16), summary: view('float32', 12), status: view('uint32', 1)}},
    /output.summary/
  );
  // 15 predictors (the OLS and 16-wide Cholesky limit) are accepted.
  expect(
    () =>
      new GPUSpatialErrorGM(
        createProps({
          predictorCount: 15,
          predictors: view('float32', 150),
          output: {
            table: view('float32', 68),
            summary: view('float32', 13),
            status: view('uint32', 1)
          }
        })
      )
  ).not.toThrow();
  const kelejianPrucha = new GPUSpatialErrorGM(
    createProps({estimator: 'kelejian-prucha', id: 'kp'})
  );
  const kelejianPruchaNames = kelejianPrucha
    .getCommandNodes(graph)
    .map(node => node.id.replace('kp-', ''));
  expect(kelejianPruchaNames.filter(name => !name.startsWith('ols-'))).toEqual([
    'means-tiles',
    'means-merge',
    'weights-norm-tiles',
    'weights-norm-merge',
    'residual-lag',
    'moments-tiles',
    'moments-merge',
    'lambda-solve',
    'filtered-moments-tiles',
    'filtered-moments-merge',
    'filtered-solve',
    'residuals-tiles',
    'residuals-merge',
    'finish'
  ]);
  const homoskedastic = new GPUSpatialErrorGM(createProps());
  expect(homoskedastic.estimator).toBe('homoskedastic');
  const names = homoskedastic
    .getCommandNodes(graph)
    .map(node => node.id.replace('spatial-error-gm-', ''));
  expect(names.filter(name => !name.startsWith('ols-') && !name.startsWith('traces-'))).toEqual([
    'means-tiles',
    'means-merge',
    'weights-norm-tiles',
    'weights-norm-merge',
    'residual-lag',
    'moments-tiles',
    'moments-merge',
    'lambda-solve',
    'filtered-moments-tiles',
    'filtered-moments-merge',
    'filtered-solve',
    'residuals-tiles',
    'residuals-merge',
    'residual-lag-fgls',
    'moments-tiles-fgls',
    'moments-merge-fgls',
    'power-sums-tiles-first',
    'power-sums-merge-first',
    'lambda-solve-weighted',
    'power-sums-tiles-final',
    'power-sums-merge-final',
    'filtered-moments-tiles-final',
    'filtered-moments-merge-final',
    'finish'
  ]);
  expect(names.some(name => name.startsWith('ols-'))).toBe(true);
  expect(names.some(name => name.startsWith('traces-'))).toBe(true);
  device.destroy();
});

it('spatial error GM oracle recovers lambda and the betas on a spatial error scene', () => {
  for (const kind of ['lattice-row-standardized', 'knn'] as const) {
    const scene = createErrorScene(400, 31, 0.6, kind);
    const result = fitSpatialErrorGMOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    expect(result.lambda, kind).toBeGreaterThan(0.25);
    expect(result.lambda, kind).toBeLessThan(0.95);
    expect(result.coefficients[1], kind).toBeGreaterThan(1.7);
    expect(result.coefficients[1], kind).toBeLessThan(2.3);
    expect(result.moments.length).toBe(7);
  }
});

it('spatial error GM oracle returns lambda near zero for independent errors', () => {
  const scene = createErrorScene(400, 5, 0, 'lattice-row-standardized');
  const result = fitSpatialErrorGMOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expect(Math.abs(result.lambda)).toBeLessThan(0.3);
});

for (const name of ['lattice', 'knn'] as const) {
  it(`spatial error GM oracle reproduces spreg GM_Error on the ${name} fixture`, () => {
    const reference = SPREG_ERROR_GM_REFERENCE[name];
    const result = fitSpatialErrorGMOnCPU(
      {
        offsets: Uint32Array.from(reference.offsets),
        neighbors: Uint32Array.from(reference.neighbors),
        weights: Float32Array.from(reference.weights)
      },
      Float32Array.from(reference.predictors),
      Float32Array.from(reference.response),
      2
    );
    expect(result.lambda).toBeCloseTo(reference.lambda, 4);
    reference.betas.forEach((beta, index) => {
      expect(result.coefficients[index]).toBeCloseTo(beta, 3);
      expect(result.standardErrors[index]).toBeCloseTo(reference.standardErrors[index], 4);
      expect(result.zStatistics[index]).toBeCloseTo(reference.zStatistics[index], 2);
    });
    expect(result.sigmaSquared).toBeCloseTo(reference.sigma2, 4);
    expect(result.pseudoRSquared).toBeCloseTo(reference.pseudoRSquared, 4);
  });
}
