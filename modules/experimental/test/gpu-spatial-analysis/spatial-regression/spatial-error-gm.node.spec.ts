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
    output: {table: view('float32', 16), summary: view('float32', 6), status: view('uint32', 1)},
    ...overrides
  });
  const expectThrows = (overrides: Partial<GPUSpatialErrorGMProps>, message: RegExp) =>
    expect(() => new GPUSpatialErrorGM(createProps(overrides))).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 9}, /predictorCount/);
  expectThrows({response: view('float32', 9)}, /response length/);
  expectThrows({predictors: view('float32', 19)}, /predictors length/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows(
    {output: {table: view('float32', 15), summary: view('float32', 6), status: view('uint32', 1)}},
    /output.table/
  );
  expectThrows(
    {output: {table: view('float32', 16), summary: view('float32', 5), status: view('uint32', 1)}},
    /output.summary/
  );
  const nodes = new GPUSpatialErrorGM(createProps()).getCommandNodes(graph);
  const names = nodes.map(node => node.id.replace('spatial-error-gm-', ''));
  expect(names.filter(name => !name.startsWith('ols-'))).toEqual([
    'means-tiles',
    'means-merge',
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
  expect(names.some(name => name.startsWith('ols-'))).toBe(true);
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
