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
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeHotSpotOracle,
  createDistanceBandWeights,
  computeLocalMoranOracle,
  getConditionalLagMoments,
  getFalseDiscoveryRateLevels,
  getTwoSidedPValue
} from './spatial-autocorrelation-oracle';

let serial = 0;

function createWeightsViews(graph: GPUCommandGraph, rows = 10) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows * 4),
    weights: view('float32', rows * 4)
  };
}

function createHotSpotProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUHotSpotAnalysisProps> = {}
): GPUHotSpotAnalysisProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'sint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    weights: createWeightsViews(graph),
    values: view('float32', 10),
    parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
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
    weights: createWeightsViews(graph),
    values: view('float32', 10),
    parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
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
  expect(Array.from(getGPUSpatialAutocorrelationParameterValues())).toEqual([
    Math.fround(0.05),
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ]);
  expect(
    Array.from(
      getGPUSpatialAutocorrelationParameterValues({
        significanceLevel: 0.25,
        fixedMoments: {count: 100, mean: 2, variance: 4}
      })
    )
  ).toEqual([0.25, 1, 100, 2, 4, 0, 0, 0]);
  expect(() => getGPUSpatialAutocorrelationParameterValues({significanceLevel: NaN})).toThrow(
    /finite/
  );
  expect(() => getGPUSpatialAutocorrelationParameterValues({significanceLevel: 1})).toThrow(
    /significanceLevel/
  );
  expect(() =>
    getGPUSpatialAutocorrelationParameterValues({
      fixedMoments: {count: 1, mean: 0, variance: 1}
    })
  ).toThrow(/fixedMoments/);
  expect(() => getGPUSpatialAutocorrelationParameterValues({}, new Float32Array(4))).toThrow(/8/);
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
  // Binary weights on k neighbors, and a weighted case with distinct weights.
  const weightCases = [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1], [0.5, 2], [0.25, 1, 3]];
  for (const caseWeights of weightCases) {
    const neighborCount = caseWeights.length;
    const lags: number[] = [];
    const permute = (prefix: number[], rest: number[]): void => {
      if (prefix.length === neighborCount) {
        lags.push(prefix.reduce((sum, value, index) => sum + caseWeights[index] * value, 0));
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
    const moments = getConditionalLagMoments(
      centered[0],
      raw.length,
      sumOfSquares,
      caseWeights.reduce((sum, weight) => sum + weight, 0),
      caseWeights.reduce((sum, weight) => sum + weight * weight, 0)
    );
    expect(moments.mean).toBeCloseTo(expectedMean, 10);
    expect(moments.variance).toBeCloseTo(expectedVariance, 10);
  }
});

it('the Gi* oracle equals the uncentered Ord-Getis formula for weighted neighborhoods', () => {
  const positions = Float32Array.from([0, 0, 1, 0, 2, 0, 5, 5, 6, 5, 9, 9, 0, 1, 7, 7]);
  const values = Float32Array.from([10, 12, 11, 2, 3, 1, 9, 4]);
  const count = values.length;
  const base = createDistanceBandWeights(positions, 1.5);
  // Distance-decay weights on the same neighborhoods, so S1 differs from W.
  const weights = {
    ...base,
    weights: Float32Array.from(base.weights, (_, slot) => 1 / (1 + (slot % 3)))
  };
  const mean = values.reduce((sum, value) => sum + value, 0) / count;
  const deviation = Math.sqrt(
    values.reduce((sum, value) => sum + value * value, 0) / count - mean * mean
  );
  for (const selfWeight of [1, 0, 0.5]) {
    const oracle = computeHotSpotOracle({values, weights, selfWeight});
    for (let row = 0; row < count; row++) {
      let weightedSum = selfWeight * values[row];
      let weightSum = selfWeight;
      let squareSum = selfWeight * selfWeight;
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        weightedSum += weights.weights[slot] * values[weights.neighbors[slot]];
        weightSum += weights.weights[slot];
        squareSum += weights.weights[slot] ** 2;
      }
      // esda G_Local(star): (sum w x - X W) / (S sqrt((n S1 - W^2) / (n - 1))).
      const spread = (count * squareSum - weightSum ** 2) / (count - 1);
      const expected =
        spread > 0 ? (weightedSum - mean * weightSum) / (deviation * Math.sqrt(spread)) : NaN;
      if (Number.isNaN(expected)) {
        expect(oracle.zScores[row]).toBeNaN();
      } else {
        expect(oracle.zScores[row]).toBeCloseTo(expected, 9);
      }
    }
  }
});

it('the local Moran oracle scales I like esda and keeps z invariant to row scaling of the weights', () => {
  const positions = Float32Array.from([0, 0, 1, 0, 2, 0, 5, 5, 6, 5, 9, 9, 0, 1, 7, 7]);
  const values = Float32Array.from([10, 12, 11, 2, 3, 1, 9, 4]);
  const binary = createDistanceBandWeights(positions, 1.5);
  const standardized = createDistanceBandWeights(positions, 1.5, {rowStandardize: true});
  const binaryResult = computeLocalMoranOracle({weights: binary, values});
  const rowResult = computeLocalMoranOracle({weights: standardized, values});
  for (const [row, zScore] of binaryResult.zScores.entries()) {
    if (Number.isNaN(zScore)) {
      expect(rowResult.zScores[row]).toBeNaN();
    } else {
      expect(rowResult.zScores[row]).toBeCloseTo(zScore, 5);
    }
  }
  // esda: I_i = (n - 1) z_i * sum_j w_ij z_j / sum z^2 for the weights as given.
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const centered = Array.from(values, value => value - mean);
  const denominator = centered.reduce((sum, value) => sum + value * value, 0);
  for (const [index, count] of rowResult.neighborCounts.entries()) {
    expect(rowResult.localI[index]).toBeCloseTo(
      ((values.length - 1) * centered[index] * rowResult.spatialLag[index]) / denominator,
      10
    );
    // A binary lag is the neighbor sum, which is count times the row-standardized lag.
    expect(binaryResult.spatialLag[index]).toBeCloseTo(rowResult.spatialLag[index] * count, 5);
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
  expectHotSpotThrows(
    graph => ({
      weights: {
        ...createWeightsViews(graph),
        weights: createTransientView(graph, 'short-w', 'float32', 3)
      }
    }),
    /weights length/
  );
  expectHotSpotThrows(() => ({selfWeight: -1}), /selfWeight/);
  expectHotSpotThrows(() => ({selfWeight: NaN}), /selfWeight/);
  expectHotSpotThrows(
    graph => ({values: createTransientView(graph, 'short-values', 'float32', 9)}),
    /values length/
  );
  expectHotSpotThrows(
    graph => ({parameters: createTransientView(graph, 'short-parameters', 'float32', 7)}),
    /parameters must hold 8/
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
      weights: {
        ...createWeightsViews(graph),
        offsets: createTransientView(graph, 'one', 'uint32', 1)
      }
    }),
    /at least two entries/
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

it('spatial-autocorrelation contributors create deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  for (const falseDiscoveryRate of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const hotSpot = new GPUHotSpotAnalysis(createHotSpotProps(graph, {falseDiscoveryRate}));
    const hotSpotIds = hotSpot.getCommandNodes(graph).map(node => node.id);
    for (const step of [
      'moments-blocks',
      'moments-mean',
      'center',
      'moments-variance',
      'neighbors'
    ]) {
      expect(hotSpotIds).toContain(`hot-spot-analysis-${step}`);
    }
    // Without FDR the classification is fused into the neighbor kernel.
    expect(hotSpotIds.includes('hot-spot-analysis-classify')).toBe(falseDiscoveryRate);
    expect(hotSpotIds.some(id => id.includes('cell-'))).toBe(false);
    expect(hotSpotIds.includes('hot-spot-analysis-fdr-thresholds')).toBe(falseDiscoveryRate);
    expect(new Set(hotSpotIds).size).toBe(hotSpotIds.length);

    const moranGraph = new GPUCommandGraph(device);
    const moran = new GPULocalMoran(createLocalMoranProps(moranGraph, {falseDiscoveryRate}));
    const moranIds = moran.getCommandNodes(moranGraph).map(node => node.id);
    expect(moranIds).toContain('local-moran-neighbors');
    // The local-I epilogue is always fused; the classification only without FDR.
    expect(moranIds.includes('local-moran-local-i')).toBe(false);
    expect(moranIds.includes('local-moran-classify')).toBe(falseDiscoveryRate);
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
