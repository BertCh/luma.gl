// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUVariogram,
  type GPUVariogramProps
} from '../../../src/map-graphs/pair-statistics/gpu-variogram';
import {
  evaluateVariogramModel,
  fitVariogramModel,
  getVariogramModelShape,
  type VariogramModelType
} from '../../../src/map-graphs/pair-statistics/variogram-model';
import {
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH
} from '../../../src/map-graphs/pair-statistics/variogram-parameters';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {computeVariogramOnCPU} from './variogram-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUVariogramProps> = {}
): GPUVariogramProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    values: view('float32', 10),
    parameters: view('float32', GPU_VARIOGRAM_PARAMETER_LENGTH),
    gridSize: [4, 4],
    lagCount: 8,
    semivariances: view('float32', 8),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUVariogramProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUVariogram(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUVariogramParameterValues packs and validates the layout', () => {
  expect(
    Array.from(getGPUVariogramParameterValues({bounds: [0, 1, 2, 3], maximumDistance: 4}))
  ).toEqual([0, 1, 2, 3, 4, 0, 0, 0]);
  expect(
    Array.from(
      getGPUVariogramParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 0.5,
        azimuthOffset: 0.25
      })
    )
  ).toEqual([0, 0, 1, 1, 0.5, 0.25, 0, 0]);
  const bounds = [0, 0, 1, 1] as const;
  expect(() => getGPUVariogramParameterValues({bounds, maximumDistance: 0})).toThrow(/positive/);
  expect(() => getGPUVariogramParameterValues({bounds, maximumDistance: NaN})).toThrow(/finite/);
  expect(() => getGPUVariogramParameterValues({bounds: [1, 0, 0, 1], maximumDistance: 1})).toThrow(
    /minX/
  );
  expect(() =>
    getGPUVariogramParameterValues({bounds, maximumDistance: 1, azimuthOffset: Infinity})
  ).toThrow(/azimuthOffset/);
  expect(() =>
    getGPUVariogramParameterValues({bounds, maximumDistance: 1}, new Float32Array(4))
  ).toThrow(/8/);
});

it('GPUVariogram validates its inputs and outputs', () => {
  const view = <Format extends 'float32' | 'uint32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(() => ({lagCount: 0}), /lagCount/);
  expectThrows(() => ({directionCount: 1.5}), /directionCount/);
  expectThrows(
    graph => ({lagCount: 64, directionCount: 16, semivariances: view(graph, 'float32', 1024)}),
    /512/
  );
  expectThrows(graph => ({values: view(graph, 'float32', 9)}), /values length/);
  expectThrows(graph => ({semivariances: view(graph, 'float32', 7)}), /semivariances/);
  expectThrows(graph => ({pairCounts: view(graph, 'uint32', 7)}), /pairCounts/);
  expectThrows(graph => ({statistics: view(graph, 'float32', 4)}), /statistics/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 4)}), /parameters/);
  expectThrows(() => ({gridSize: [0, 4]}), /gridSize/);
  expectThrows(graph => {
    const shared = view(graph, 'float32', 10);
    return {values: shared, semivariances: shared};
  }, /alias|overlap|disjoint|input/i);
});

it('GPUVariogram emits deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const ids = () => {
    const graph = new GPUCommandGraph(device);
    const recipe = new GPUVariogram({
      ...createProps(graph),
      id: 'v',
      pairCounts: createTransientView(graph, `counts-${serial++}`, 'uint32', 8),
      statistics: createTransientView(graph, `stats-${serial++}`, 'float32', 5)
    });
    return recipe.getCommandNodes(graph).map(node => node.id);
  };
  const first = ids();
  expect(ids()).toEqual(first);
  expect(first.every(id => id.startsWith('v-'))).toBe(true);
  expect(first).toContain('v-histogram-pairs');
  expect(first).toContain('v-finish');
  expect(first).toContain('v-statistics');
  device.destroy();
});

it('computeVariogramOnCPU matches a hand-computed transect', () => {
  // Points at x = 0..3 with values 0, 1, 3, 6: lag-1 pairs (0,1)(1,2)(2,3) differ by 1, 2, 3.
  const positions = new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]);
  const values = new Float32Array([0, 1, 3, 6]);
  const result = computeVariogramOnCPU(
    {positions, values},
    {bounds: [0, -1, 3, 1], maximumDistance: 3.5},
    7
  );
  // Lags are 0.5 wide: distance 1 -> lag 2, 2 -> lag 4, 3 -> lag 6.
  expect(result.pairCounts).toEqual([0, 0, 3, 0, 2, 0, 1]);
  expect(result.semivariances[2]).toBeCloseTo((1 + 4 + 9) / 6, 12);
  expect(result.semivariances[4]).toBeCloseTo((9 + 25) / 4, 12);
  expect(result.semivariances[6]).toBeCloseTo(36 / 2, 12);
  expect(result.meanDistances[2]).toBe(1);
  const robust = ((1 + Math.SQRT2 + Math.sqrt(3)) / 3) ** 4 / (2 * (0.457 + 0.494 / 3));
  expect(result.robustSemivariances[2]).toBeCloseTo(robust, 12);
  expect(result.statistics).toEqual([4, 2.5, 5.25, 0, 6]);
});

it('fitVariogramModel recovers noise-free model parameters', () => {
  const distances = Array.from({length: 24}, (_, bin) => (bin + 0.5) * 2);
  for (const model of ['spherical', 'exponential', 'gaussian'] as VariogramModelType[]) {
    const truth = {model, nugget: 0.3, sill: 2.5, range: 21};
    const semivariances = distances.map(distance => evaluateVariogramModel(truth, distance));
    const pairCounts = distances.map((_, bin) => 1000 - bin * 10);
    for (const weighting of ['cressie', 'pairs', 'none'] as const) {
      const fit = fitVariogramModel({distances, semivariances, pairCounts}, {model, weighting});
      expect(fit.model).toBe(model);
      expect(fit.nugget, `${model} ${weighting}`).toBeCloseTo(truth.nugget, 3);
      expect(fit.sill, `${model} ${weighting}`).toBeCloseTo(truth.sill, 3);
      expect(fit.range / truth.range, `${model} ${weighting}`).toBeCloseTo(1, 3);
      expect(fit.residual).toBeLessThan(1e-6);
    }
  }
});

it('fitVariogramModel keeps nugget and sill non-negative and skips empty bins', () => {
  // A decreasing empirical curve: the best non-negative fit is a pure nugget.
  const fit = fitVariogramModel(
    {
      distances: [1, 2, 3, 4, NaN],
      semivariances: [4, 3, 2, 1, 7],
      pairCounts: [10, 10, 10, 10, 0]
    },
    {model: 'spherical', weighting: 'pairs'}
  );
  expect(fit.sill).toBeGreaterThanOrEqual(0);
  expect(fit.nugget).toBeGreaterThanOrEqual(0);
  expect(fit.nugget).toBeCloseTo(2.5, 6);
  expect(() =>
    fitVariogramModel({distances: [1], semivariances: [1], pairCounts: [1]}, {model: 'gaussian'})
  ).toThrow(/two/);
  expect(() =>
    fitVariogramModel(
      {distances: [1, 2], semivariances: [1, 2], pairCounts: [1, 1]},
      {model: 'cubic' as VariogramModelType}
    )
  ).toThrow(/cubic/);
});

it('variogram model shapes follow the practical-range convention', () => {
  expect(getVariogramModelShape('spherical', 1)).toBe(1);
  expect(getVariogramModelShape('spherical', 0.5)).toBeCloseTo(0.6875, 12);
  expect(getVariogramModelShape('exponential', 1)).toBeCloseTo(1 - Math.exp(-3), 12);
  expect(getVariogramModelShape('gaussian', 1)).toBeCloseTo(1 - Math.exp(-3), 12);
  expect(evaluateVariogramModel({model: 'gaussian', nugget: 1, sill: 2, range: 5}, 0)).toBe(0);
});
