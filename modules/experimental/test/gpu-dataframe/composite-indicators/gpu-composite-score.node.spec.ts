// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCompositeScore,
  type GPUCompositeScoreProps
} from '../../../src/gpu-dataframe/composite-indicators/gpu-composite-score';
import {
  getGPUCompositeScoreParameterValues,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/composite-indicators/composite-score-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeCompositeScoreOnCPU, getLargestEigenpairByJacobi} from './composite-score-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCompositeScoreProps> = {}
): GPUCompositeScoreProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    indicators: view('float32', 30),
    indicatorCount: 3,
    parameters: view('float32', GPU_COMPOSITE_SCORE_PARAMETER_LENGTH),
    output: {score: view('float32', 10)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUCompositeScoreProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUCompositeScore(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUCompositeScoreParameterValues packs modes, weights and directions', () => {
  const values = getGPUCompositeScoreParameterValues({
    scaler: 'rank',
    aggregation: 'weighted-geometric-mean',
    weights: [2, 0.5],
    directions: [1, -3],
    epsilon: 0.25
  });
  expect(values.length).toBe(GPU_COMPOSITE_SCORE_PARAMETER_LENGTH);
  expect(Array.from(values.subarray(0, 6))).toEqual([2, 1, 0.25, 0, 2, 0.5]);
  expect(values[6]).toBe(0);
  expect(Array.from(values.subarray(20, 23))).toEqual([1, -1, 1]);
  expect(() => getGPUCompositeScoreParameterValues({weights: new Array(17).fill(1)})).toThrow(
    /at most 16/
  );
  expect(() => getGPUCompositeScoreParameterValues({weights: [NaN]})).toThrow(/finite/);
  expect(() => getGPUCompositeScoreParameterValues({weights: [1], epsilon: 0})).toThrow(/epsilon/);
  expect(() => getGPUCompositeScoreParameterValues({weights: [1]}, new Float32Array(4))).toThrow(
    /hold/
  );
});

it('GPUCompositeScore validates props', () => {
  expectThrows(() => ({indicatorCount: 0}), /indicatorCount/);
  expectThrows(() => ({indicatorCount: 17}), /indicatorCount/);
  expectThrows(() => ({indicatorCount: 4}), /multiple/);
  expectThrows(
    graph => ({parameters: createTransientView(graph, `p-${serial++}`, 'float32', 4)}),
    /parameters/
  );
  expectThrows(
    graph => ({mask: createTransientView(graph, `m-${serial++}`, 'uint32', 9)}),
    /mask length/
  );
  expectThrows(
    graph => ({output: {score: createTransientView(graph, `s-${serial++}`, 'float32', 9)}}),
    /score/
  );
  expectThrows(
    graph => ({
      output: {
        score: createTransientView(graph, `s-${serial++}`, 'float32', 10),
        loadings: createTransientView(graph, `l-${serial++}`, 'float32', 3)
      }
    }),
    /enablePrincipalComponent/
  );
  expectThrows(
    graph => ({
      output: {
        score: createTransientView(graph, `s-${serial++}`, 'float32', 10),
        scaled: createTransientView(graph, `c-${serial++}`, 'float32', 29)
      }
    }),
    /scaled/
  );
});

it('GPUCompositeScore emits deterministic node IDs and compiles optional passes only when enabled', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const basic = new GPUCompositeScore(createProps(graph, {id: 'basic'}));
  const basicIds = basic.getCommandNodes(graph).map(node => node.id);
  expect(basicIds).toEqual([
    'basic-validate',
    'basic-tile-sums',
    'basic-means',
    'basic-tile-moments',
    'basic-moments',
    'basic-score'
  ]);
  const full = new GPUCompositeScore(
    createProps(graph, {id: 'full', enableRank: true, enablePrincipalComponent: true})
  );
  const fullIds = full.getCommandNodes(graph).map(node => node.id);
  expect(fullIds).toContain('full-principal-component');
  expect(fullIds).toContain('full-rank-keys-0');
  expect(fullIds).toContain('full-rank-scatter-2');
  expect(fullIds[fullIds.length - 1]).toBe('full-score');
  expect(new Set(fullIds).size).toBe(fullIds.length);
  device.destroy();
});

it('computeCompositeScoreOnCPU power iteration matches a Jacobi eigen-solve', () => {
  const rows = 200;
  const d = 4;
  const indicators = new Float32Array(rows * d);
  let state = 7;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296 - 0.5;
  };
  for (let row = 0; row < rows; row++) {
    const latent = random();
    indicators[row * d] = latent + 0.1 * random();
    indicators[row * d + 1] = 2 * latent + 0.3 * random();
    indicators[row * d + 2] = -latent + 0.5 * random();
    indicators[row * d + 3] = random();
  }
  const result = computeCompositeScoreOnCPU({
    indicators,
    indicatorCount: d,
    settings: {aggregation: 'principal-component', weights: []}
  });
  // Correlation matrix of the raw columns for the Jacobi reference.
  const statistics = result.columnStatistics;
  const correlation = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      let sum = 0;
      for (let row = 0; row < rows; row++) {
        sum +=
          ((indicators[row * d + a] - statistics[a * 4 + 2]) / statistics[a * 4 + 3]) *
          ((indicators[row * d + b] - statistics[b * 4 + 2]) / statistics[b * 4 + 3]);
      }
      correlation[a * d + b] = sum / rows;
    }
  }
  const reference = getLargestEigenpairByJacobi(correlation, d);
  expect(result.principalComponent[0]).toBeCloseTo(reference.value, 8);
  const sign = Math.sign(reference.vector.reduce((sum, value) => sum + value, 0));
  for (let c = 0; c < d; c++) {
    expect(result.loadings[c]).toBeCloseTo(sign * reference.vector[c], 6);
  }
  expect(result.principalComponent[1]).toBeCloseTo(reference.value / d, 8);
});

it('computeCompositeScoreOnCPU averages tied ranks and honours directions', () => {
  const result = computeCompositeScoreOnCPU({
    indicators: Float32Array.from([1, 10, 2, 20, 2, 30, 3, 40, -0, 50, 0, 60]),
    indicatorCount: 2,
    settings: {scaler: 'rank', weights: [1, 1], directions: [1, -1]}
  });
  // Column 0 sorted: -0, 0, 1, 2, 2, 3 -> ranks 0.5, 0.5, 2, 3.5, 3.5, 5 over n - 1 = 5.
  expect(Array.from({length: 6}, (_, row) => result.scaled[row * 2])).toEqual([
    0.4, 0.7, 0.7, 1, 0.1, 0.1
  ]);
  // Column 1 is reversed: 1 - rank / 5.
  expect(result.scaled[1]).toBe(1);
  expect(result.scaled[11]).toBe(0);
});
