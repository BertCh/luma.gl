// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUHotSpotAnalysisProps,
  type GPULocalMoranProps
} from '../../../src/map-graphs/spatial-autocorrelation';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  computeHotSpotOracle,
  computeLocalMoranOracle,
  getConditionalLagMoments,
  getFalseDiscoveryRateLevels,
  getTwoSidedPValue
} from './spatial-autocorrelation-oracle';

let serial = 0;

function createHotSpotProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUHotSpotAnalysisProps> = {}
): GPUHotSpotAnalysisProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'sint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    values: view('float32', 10),
    parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
    gridSize: [4, 4],
    zScores: view('float32', 10),
    bins: view('sint32', 10),
    pValues: view('float32', 10),
    ...overrides
  };
}

function createLocalMoranProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULocalMoranProps> = {}
): GPULocalMoranProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    values: view('float32', 10),
    parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
    gridSize: [4, 4],
    zScores: view('float32', 10),
    localI: view('float32', 10),
    quadrants: view('uint32', 10),
    ...overrides
  };
}

function expectHotSpotThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUHotSpotAnalysisProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUHotSpotAnalysis(createHotSpotProps(graph, overrides(graph)))).toThrow(
    message
  );
  device.destroy();
}

it('getGPUSpatialAutocorrelationParameterValues packs and validates the layout', () => {
  expect(
    Array.from(getGPUSpatialAutocorrelationParameterValues({bounds: [0, 1, 2, 3], radius: 4}))
  ).toEqual([0, 1, 2, 3, 4, Math.fround(0.05), 1, 0, 0, 0, 0, 0]);
  expect(
    Array.from(
      getGPUSpatialAutocorrelationParameterValues({
        bounds: [0, 0, 1, 1],
        radius: 0.5,
        significanceLevel: 0.25,
        weightTransform: 'binary',
        fixedMoments: {count: 100, mean: 2, variance: 4}
      })
    )
  ).toEqual([0, 0, 1, 1, 0.5, 0.25, 0, 1, 100, 2, 4, 0]);
  const bounds = [0, 0, 1, 1] as const;
  expect(() => getGPUSpatialAutocorrelationParameterValues({bounds, radius: 0})).toThrow(/radius/);
  expect(() => getGPUSpatialAutocorrelationParameterValues({bounds, radius: NaN})).toThrow(
    /finite/
  );
  expect(() =>
    getGPUSpatialAutocorrelationParameterValues({bounds, radius: 1, significanceLevel: 1})
  ).toThrow(/significanceLevel/);
  expect(() =>
    getGPUSpatialAutocorrelationParameterValues({
      bounds,
      radius: 1,
      fixedMoments: {count: 1, mean: 0, variance: 1}
    })
  ).toThrow(/fixedMoments/);
  expect(() =>
    getGPUSpatialAutocorrelationParameterValues({bounds, radius: 1}, new Float32Array(4))
  ).toThrow(/12/);
});

it('getTwoSidedPValue matches known normal tail probabilities', () => {
  expect(getTwoSidedPValue(0)).toBeCloseTo(1, 6);
  for (const [zScore, pValue] of [
    [1.6448536269514722, 0.1],
    [1.959963984540054, 0.05],
    [2.5758293035489004, 0.01],
    [3.2905267314919255, 0.001],
    [5.326723886384, 1e-7]
  ]) {
    expect(Math.abs(getTwoSidedPValue(zScore) - pValue) / pValue).toBeLessThan(2e-6);
    expect(getTwoSidedPValue(-zScore)).toBe(getTwoSidedPValue(zScore));
  }
});

it('conditional lag moments equal the exhaustive permutation moments', () => {
  // Centered values of n = 6 rows; row 0 is the focus with k = 2 neighbors among 5 others.
  const raw = [3, -1, 4, 1, -5, 9];
  const mean = raw.reduce((sum, value) => sum + value, 0) / raw.length;
  const centered = raw.map(value => value - mean);
  const others = centered.slice(1);
  const sumOfSquares = centered.reduce((sum, value) => sum + value * value, 0);
  for (const neighborCount of [1, 2, 3, 4]) {
    const lags: number[] = [];
    const permute = (prefix: number[], rest: number[]): void => {
      if (prefix.length === neighborCount) {
        lags.push(prefix.reduce((sum, value) => sum + value, 0));
        return;
      }
      for (const [index, value] of rest.entries()) {
        permute([...prefix, value], [...rest.slice(0, index), ...rest.slice(index + 1)]);
      }
    };
    permute([], others);
    const expectedMean = lags.reduce((sum, value) => sum + value, 0) / lags.length;
    const expectedVariance =
      lags.reduce((sum, value) => sum + (value - expectedMean) ** 2, 0) / lags.length;
    const moments = getConditionalLagMoments(centered[0], raw.length, sumOfSquares, neighborCount);
    expect(moments.mean).toBeCloseTo(expectedMean, 10);
    expect(moments.variance).toBeCloseTo(expectedVariance, 10);
  }
});

it('the Gi* oracle equals the uncentered ArcGIS formula', () => {
  const positions = Float32Array.from([0, 0, 1, 0, 2, 0, 5, 5, 6, 5, 9, 9, 0, 1, 7, 7]);
  const values = Float32Array.from([10, 12, 11, 2, 3, 1, 9, 4]);
  const parameters = {bounds: [-1, -1, 10, 10] as const, radius: 1.5};
  const oracle = computeHotSpotOracle({positions, values, parameters});
  const count = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / count;
  const deviation = Math.sqrt(
    values.reduce((sum, value) => sum + value * value, 0) / count - mean * mean
  );
  for (let row = 0; row < count; row++) {
    let weightedSum = 0;
    let weightSum = 0;
    for (let other = 0; other < count; other++) {
      const distance = Math.hypot(
        positions[other * 2] - positions[row * 2],
        positions[other * 2 + 1] - positions[row * 2 + 1]
      );
      if (distance <= 1.5) {
        weightedSum += values[other];
        weightSum++;
      }
    }
    const expected =
      (weightedSum - mean * weightSum) /
      (deviation * Math.sqrt((count * weightSum - weightSum * weightSum) / (count - 1)));
    expect(oracle.zScores[row]).toBeCloseTo(expected, 9);
    expect(oracle.neighborCounts[row]).toBe(weightSum);
  }
});

it('the local Moran oracle scales I like esda and keeps z invariant to the weight transform', () => {
  const positions = Float32Array.from([0, 0, 1, 0, 2, 0, 5, 5, 6, 5, 9, 9, 0, 1, 7, 7]);
  const values = Float32Array.from([10, 12, 11, 2, 3, 1, 9, 4]);
  const bounds = [-1, -1, 10, 10] as const;
  const row = computeLocalMoranOracle({positions, values, parameters: {bounds, radius: 1.5}});
  const binary = computeLocalMoranOracle({
    positions,
    values,
    parameters: {bounds, radius: 1.5, weightTransform: 'binary'}
  });
  expect(binary.zScores).toEqual(row.zScores);
  // esda: I_i = (n - 1) z_i * sum_j w_ij z_j / sum z^2, with row-standardized weights.
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const centered = Array.from(values, value => value - mean);
  const denominator = centered.reduce((sum, value) => sum + value * value, 0);
  for (const [index, count] of row.neighborCounts.entries()) {
    const lag = count > 0 ? row.spatialLag[index] : 0;
    expect(row.localI[index]).toBeCloseTo(
      ((values.length - 1) * centered[index] * lag) / denominator,
      10
    );
    expect(binary.localI[index]).toBeCloseTo(row.localI[index] * count, 10);
  }
});

it('getFalseDiscoveryRateLevels applies the BH step-up rule', () => {
  // Two-sided p ~ [1e-4, 0.002, 0.03, 0.04, 0.5] with m = 5 (NaN is not tested). At 0.10 the
  // bounds are k * 0.02, so ranks 1-4 pass and p = 0.5 fails; at 0.01 the bounds are k * 0.002, so
  // only ranks 1-2 pass.
  const zScores = [3.8906, -3.0902, 2.1701, -2.0537, 0.6745, NaN];
  const levels = getFalseDiscoveryRateLevels(zScores, [0.1, 0.05, 0.01]);
  expect(levels).toEqual([3, 3, levels[2], levels[3], 0, 0]);
  expect(levels[2]).toBeGreaterThanOrEqual(1);
  expect(levels[2]).toBeLessThanOrEqual(2);
  expect(levels[3]).toBeGreaterThanOrEqual(1);
  expect(levels[3]).toBeLessThanOrEqual(2);
  // Step-up: rank 4 passes (p ~ 0.032 <= 0.05), which rejects ranks 1-3 even though rank 1 alone
  // fails (p ~ 0.021 > 0.0125).
  expect(getFalseDiscoveryRateLevels([2.3, 2.25, 2.2, 2.15], [0.05])).toEqual([1, 1, 1, 1]);
});

it('GPUHotSpotAnalysis and GPULocalMoran reject invalid properties', () => {
  expectHotSpotThrows(() => ({gridSize: [0, 4]}), /gridSize/);
  expectHotSpotThrows(
    graph => ({values: createTransientView(graph, 'short-values', 'float32', 9)}),
    /values length/
  );
  expectHotSpotThrows(
    graph => ({parameters: createTransientView(graph, 'short-parameters', 'float32', 11)}),
    /parameters must hold 12/
  );
  expectHotSpotThrows(
    graph => ({mask: createTransientView(graph, 'short-mask', 'uint32', 3)}),
    /mask length/
  );
  expectHotSpotThrows(
    graph => ({bins: createTransientView(graph, 'float-bins', 'float32', 10) as never}),
    /bins/
  );
  expectHotSpotThrows(
    graph => ({zScores: createTransientView(graph, 'short-z', 'float32', 4)}),
    /zScores length/
  );
  expectHotSpotThrows(
    graph => ({globalStatistics: createTransientView(graph, 'short-statistics', 'float32', 3)}),
    /globalStatistics/
  );
  expectHotSpotThrows(
    graph => ({
      positions: createTransientView(graph, 'no-rows', 'float32x2', 0),
      values: createTransientView(graph, 'no-values', 'float32', 0)
    }),
    /at least one row/
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createLocalMoranProps(graph);
  expect(() => new GPULocalMoran({...base, zScores: base.values})).toThrow(
    /outputs must not share/
  );
  expect(
    () =>
      new GPULocalMoran({
        ...base,
        quadrants: createTransientView(graph, 'short-quadrants', 'uint32', 2)
      })
  ).toThrow(/quadrants length/);
  device.destroy();
});

it('spatial-autocorrelation recipes create deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  for (const falseDiscoveryRate of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const hotSpot = new GPUHotSpotAnalysis(createHotSpotProps(graph, {falseDiscoveryRate}));
    const hotSpotIds = hotSpot.getCommandNodes(graph).map(node => node.id);
    for (const step of [
      'cell-keys',
      'cell-total',
      'block-offsets',
      'value-sum-blocks',
      'value-sum-total',
      'moments-mean',
      'center',
      'square-sum-total',
      'moments-variance',
      'neighbors',
      'classify'
    ]) {
      expect(hotSpotIds).toContain(`hot-spot-analysis-${step}`);
    }
    expect(hotSpotIds.includes('hot-spot-analysis-fdr-thresholds')).toBe(falseDiscoveryRate);
    expect(new Set(hotSpotIds).size).toBe(hotSpotIds.length);

    const moranGraph = new GPUCommandGraph(device);
    const moran = new GPULocalMoran(createLocalMoranProps(moranGraph, {falseDiscoveryRate}));
    const moranIds = moran.getCommandNodes(moranGraph).map(node => node.id);
    for (const step of ['neighbors', 'local-i', 'classify']) {
      expect(moranIds).toContain(`local-moran-${step}`);
    }
    expect(moranIds.includes('local-moran-fdr-ranks')).toBe(falseDiscoveryRate);
  }
  // Without bins or p-values there is no classify node.
  const graph = new GPUCommandGraph(device);
  const zOnly = new GPUHotSpotAnalysis(
    createHotSpotProps(graph, {bins: undefined, pValues: undefined})
  );
  expect(zOnly.getCommandNodes(graph).map(node => node.id)).not.toContain(
    'hot-spot-analysis-classify'
  );
  device.destroy();
});
