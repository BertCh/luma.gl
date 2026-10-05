// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUOrdinaryLeastSquares} from '../../../src/map-graphs/spatial-regression/gpu-ordinary-least-squares';
import type {GPUOrdinaryLeastSquaresProps} from '../../../src/map-graphs/spatial-regression/gpu-ordinary-least-squares';
import {
  getGPUOrdinaryLeastSquaresParameterValues,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '../../../src/map-graphs/spatial-regression/ordinary-least-squares-parameters';
import {
  getChiSquareSurvival,
  getLogGammaOfHalfInteger
} from '../../../src/map-graphs/spatial-regression/ordinary-least-squares-statistics';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  createOrdinaryLeastSquaresScene,
  fitOrdinaryLeastSquaresOnCPU
} from './ordinary-least-squares-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUOrdinaryLeastSquaresProps> = {}
): GPUOrdinaryLeastSquaresProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    predictors: view('float32', 40),
    response: view('float32', 20),
    predictorCount: 2,
    output: {
      coefficients: view('float32', 3),
      standardErrors: view('float32', 3),
      tStatistics: view('float32', 3),
      summary: view('float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
      status: view('uint32', 1)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUOrdinaryLeastSquaresProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUOrdinaryLeastSquares(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('GPUOrdinaryLeastSquares validates its inputs', () => {
  const view = <Format extends 'uint32' | 'float32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(() => ({predictorCount: 0}), /predictorCount/);
  expectThrows(() => ({predictorCount: 16}), /predictorCount/);
  expectThrows(() => ({predictorCount: 1.5}), /predictorCount/);
  expectThrows(graph => ({predictors: view(graph, 'float32', 39)}), /predictors length/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 19)}), /mask length/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 0)}), /parameters must hold/);
  expectThrows(() => ({tileRowCount: 0}), /tileRowCount/);
  expectThrows(
    graph => ({
      output: {
        coefficients: view(graph, 'float32', 2),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', 16),
        status: view(graph, 'uint32', 1)
      }
    }),
    /coefficients must hold/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: view(graph, 'float32', 3),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', 15),
        status: view(graph, 'uint32', 1)
      }
    }),
    /summary must hold/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: view(graph, 'float32', 3),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', 16),
        status: view(graph, 'uint32', 1),
        residuals: view(graph, 'float32', 19)
      }
    }),
    /residuals must hold/
  );
  expectThrows(graph => {
    const shared = view(graph, 'float32', 20);
    return {
      response: shared,
      output: {
        coefficients: view(graph, 'float32', 3),
        standardErrors: view(graph, 'float32', 3),
        tStatistics: view(graph, 'float32', 3),
        summary: view(graph, 'float32', 16),
        status: view(graph, 'uint32', 1),
        residuals: shared
      }
    };
  }, /must not share buffers/);
});

it('GPUOrdinaryLeastSquares emits a deterministic node chain with stable IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUOrdinaryLeastSquares(createProps(graph, {id: 'ols', tileRowCount: 8}));
  expect(recipe.tileCount).toBe(3);
  const nodes = recipe.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'ols-means-tiles',
    'ols-means-merge',
    'ols-moments-tiles',
    'ols-moments-merge',
    'ols-solve',
    'ols-residuals',
    'ols-residual-tiles',
    'ols-residual-merge',
    'ols-finish',
    'ols-bp-means-tiles',
    'ols-bp-means-merge',
    'ols-bp-moments-tiles',
    'ols-bp-moments-merge',
    'ols-bp-solve',
    'ols-bp-finish'
  ]);
  device.destroy();
});

it('default tile size caps the tile count at 4096', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = 3_000_000;
  const recipe = new GPUOrdinaryLeastSquares({
    predictors: createTransientView(graph, 'p', 'float32', rows),
    response: createTransientView(graph, 'r', 'float32', rows),
    predictorCount: 1,
    output: {
      coefficients: createTransientView(graph, 'c', 'float32', 2),
      standardErrors: createTransientView(graph, 's', 'float32', 2),
      tStatistics: createTransientView(graph, 't', 'float32', 2),
      summary: createTransientView(graph, 'm', 'float32', 16),
      status: createTransientView(graph, 'u', 'uint32', 1)
    }
  });
  expect(recipe.tileCount).toBeLessThanOrEqual(4096);
  expect(recipe.tileRowCount).toBe(Math.ceil(rows / 4096));
  device.destroy();
});

it('parameter helper packs the ridge lambda', () => {
  expect(Array.from(getGPUOrdinaryLeastSquaresParameterValues(2.5))).toEqual([2.5]);
  expect(() => getGPUOrdinaryLeastSquaresParameterValues(1, new Float32Array(0))).toThrow(/hold/);
});

it('chi-square survival matches known critical values and the df = 2 closed form', () => {
  expect(getChiSquareSurvival(3.841458820694124, 1)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(5.991464547107979, 2)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(7.814727903251179, 3)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(11.070497693516351, 5)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(18.307038053275146, 10)).toBeCloseTo(0.05, 10);
  expect(getChiSquareSurvival(6.634896601021213, 1)).toBeCloseTo(0.01, 10);
  expect(getChiSquareSurvival(0, 4)).toBe(1);
  expect(getChiSquareSurvival(NaN, 4)).toBeNaN();
  for (const value of [0.1, 1, 2.5, 7, 30]) {
    expect(getChiSquareSurvival(value, 2)).toBeCloseTo(Math.exp(-value / 2), 12);
  }
  expect(getLogGammaOfHalfInteger(1)).toBeCloseTo(0.5 * Math.log(Math.PI), 14);
  expect(getLogGammaOfHalfInteger(6)).toBeCloseTo(Math.log(2), 14);
  expect(() => getLogGammaOfHalfInteger(0)).toThrow(/degrees/);
});

it('oracle reproduces a textbook simple regression', () => {
  const result = fitOrdinaryLeastSquaresOnCPU({
    predictors: Float32Array.from([1, 2, 3, 4, 5]),
    response: Float32Array.from([2, 4, 5, 4, 5]),
    predictorCount: 1
  });
  expect(result.status).toBe(0);
  expect(result.coefficients[0]).toBeCloseTo(2.2, 12);
  expect(result.coefficients[1]).toBeCloseTo(0.6, 12);
  expect(result.summary[1]).toBeCloseTo(0.6, 12);
  expect(result.summary[11]).toBeCloseTo(2.4, 12);
  expect(result.summary[3]).toBeCloseTo(0.8, 12);
  expect(result.standardErrors[1]).toBeCloseTo(Math.sqrt(0.08), 12);
  expect(result.standardErrors[0]).toBeCloseTo(Math.sqrt(0.88), 12);
  expect(result.tStatistics[1]).toBeCloseTo(0.6 / Math.sqrt(0.08), 12);
  expect(result.summary[2]).toBeCloseTo(1 - (0.4 * 4) / 3, 12);
});

it('oracle recovers known coefficients, exact fits, and reports singular and short designs', () => {
  const exact = createOrdinaryLeastSquaresScene(1, 200, 2, [3, -1], 0);
  const exactFit = fitOrdinaryLeastSquaresOnCPU(exact);
  expect(exactFit.status).toBe(0);
  expect(exactFit.summary[1]).toBeGreaterThan(1 - 1e-9);
  expect(exactFit.coefficients[0]).toBeCloseTo(2, 4);
  expect(exactFit.coefficients[1]).toBeCloseTo(3, 6);
  expect(exactFit.coefficients[2]).toBeCloseTo(-1, 6);

  const noisy = createOrdinaryLeastSquaresScene(2, 4000, 2, [3, -1], 0.5);
  const noisyFit = fitOrdinaryLeastSquaresOnCPU(noisy);
  expect(Math.abs(noisyFit.coefficients[1] - 3)).toBeLessThan(0.05);
  expect(Math.abs(noisyFit.coefficients[2] + 1)).toBeLessThan(0.05);
  // Residuals are near normal and homoskedastic: large p-values on average.
  expect(noisyFit.summary[8]).toBeGreaterThan(0);
  expect(noisyFit.summary[1]).toBeGreaterThan(0.9);

  const duplicate = createOrdinaryLeastSquaresScene(3, 50, 1, [2, 0], 0.1);
  for (let row = 0; row < 50; row++) {
    duplicate.predictors[row * 2 + 1] = duplicate.predictors[row * 2];
  }
  expect(fitOrdinaryLeastSquaresOnCPU(duplicate).status).toBe(1);

  const short = createOrdinaryLeastSquaresScene(4, 3, 1, [2, 1], 0.1);
  const shortFit = fitOrdinaryLeastSquaresOnCPU(short);
  expect(shortFit.status).toBe(2);
  expect(shortFit.summary[0]).toBe(3);
});

it('oracle excludes masked and non-finite rows and applies ridge shrinkage', () => {
  const scene = createOrdinaryLeastSquaresScene(5, 300, 1, [2, -3], 0.2, {
    withMask: true,
    withNonFinite: true
  });
  const fit = fitOrdinaryLeastSquaresOnCPU(scene);
  // 60 masked rows (row % 5 == 3), plus rows 1, 7, 298 excluded unless already masked.
  const excluded = new Set<number>([1, 7, 298]);
  for (let row = 0; row < 300; row++) {
    if (row % 5 === 3) {
      excluded.add(row);
    }
  }
  expect(fit.rowCount).toBe(300 - excluded.size);
  expect(Number.isNaN(fit.residuals[1])).toBe(true);
  expect(Number.isNaN(fit.residuals[3])).toBe(true);
  expect(Number.isNaN(fit.residuals[0])).toBe(false);
  const ridge = fitOrdinaryLeastSquaresOnCPU({...scene, ridgeLambda: 5000});
  expect(Math.abs(ridge.coefficients[1])).toBeLessThan(Math.abs(fit.coefficients[1]));
  expect(ridge.summary[15]).toBe(5000);
  expect(ridge.summary[11]).toBeGreaterThan(fit.summary[11]);
});
