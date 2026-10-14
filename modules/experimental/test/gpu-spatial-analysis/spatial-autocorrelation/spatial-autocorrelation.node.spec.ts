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
import {
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  GPULocalGeary
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
import {
  getReferenceMask,
  getReferenceValues,
  getReferenceWeights
} from './spatial-autocorrelation-fixture';
import {REFERENCE} from './spatial-autocorrelation-reference';

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
  const sum = values.reduce((total, value) => total + value, 0);
  const sumOfSquares = values.reduce((total, value) => total + value * value, 0);
  const mean = sum / count;
  const deviation = Math.sqrt(sumOfSquares / count - mean * mean);
  for (const selfWeight of [1, 0, 0.5]) {
    // Rows without neighbors have no z-score, so the island policy only changes the moments.
    const oracle = computeHotSpotOracle({values, weights, selfWeight, islands: 'keep'});
    for (let row = 0; row < count; row++) {
      let weightedSum = selfWeight * values[row];
      let weightSum = selfWeight;
      let squareSum = selfWeight * selfWeight;
      let neighborSum = 0;
      let neighborWeight = 0;
      let neighborSquare = 0;
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        weightedSum += weights.weights[slot] * values[weights.neighbors[slot]];
        weightSum += weights.weights[slot];
        squareSum += weights.weights[slot] ** 2;
        neighborSum += weights.weights[slot] * values[weights.neighbors[slot]];
        neighborWeight += weights.weights[slot];
        neighborSquare += weights.weights[slot] ** 2;
      }
      let expected = NaN;
      if (weights.offsets[row + 1] > weights.offsets[row]) {
        if (selfWeight > 0) {
          // esda G_Local(star): (sum w x - X W) / (S sqrt((n S1 - W^2) / (n - 1))).
          const spread = (count * squareSum - weightSum ** 2) / (count - 1);
          expected =
            spread > 0 ? (weightedSum - mean * weightSum) / (deviation * Math.sqrt(spread)) : NaN;
        } else {
          // Ord-Getis Gi: the mean and variance of the other n - 1 values.
          const otherMean = (sum - values[row]) / (count - 1);
          const otherVariance =
            (sumOfSquares - values[row] ** 2) / (count - 1) - otherMean * otherMean;
          const spread = ((count - 1) * neighborSquare - neighborWeight ** 2) / (count - 2);
          expected =
            spread > 0
              ? (neighborSum - neighborWeight * otherMean) / Math.sqrt(otherVariance * spread)
              : NaN;
        }
      }
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
  const binaryResult = computeLocalMoranOracle({weights: binary, values, islands: 'keep'});
  const rowResult = computeLocalMoranOracle({
    weights: standardized,
    values,
    islands: 'keep'
  });
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
  for (const multipleTesting of ['none', 'benjamini-hochberg'] as const) {
    const isCorrected = multipleTesting !== 'none';
    const graph = new GPUCommandGraph(device);
    const hotSpot = new GPUHotSpotAnalysis(createHotSpotProps(graph, {multipleTesting}));
    const hotSpotIds = hotSpot.getCommandNodes(graph).map(node => node.id);
    for (const step of [
      'row-states',
      'moments-blocks',
      'moments-mean',
      'moments-deviations',
      'moments-correct',
      'center',
      'moments-variance',
      'neighbors'
    ]) {
      expect(hotSpotIds).toContain(`hot-spot-analysis-${step}`);
    }
    // Without a correction the classification is fused into the neighbor kernel.
    expect(hotSpotIds.includes('hot-spot-analysis-classify')).toBe(isCorrected);
    expect(hotSpotIds.some(id => id.includes('cell-'))).toBe(false);
    expect(hotSpotIds.some(id => id.startsWith('hot-spot-analysis-multiple-testing'))).toBe(
      isCorrected
    );
    expect(new Set(hotSpotIds).size).toBe(hotSpotIds.length);

    const moranGraph = new GPUCommandGraph(device);
    const moran = new GPULocalMoran(createLocalMoranProps(moranGraph, {multipleTesting}));
    const moranIds = moran.getCommandNodes(moranGraph).map(node => node.id);
    expect(moranIds).toContain('local-moran-neighbors');
    // The local-I epilogue is always fused; the classification only without a correction.
    expect(moranIds.includes('local-moran-local-i')).toBe(false);
    expect(moranIds.includes('local-moran-classify')).toBe(isCorrected);
    expect(moranIds.some(id => id.startsWith('local-moran-multiple-testing'))).toBe(isCorrected);
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

it('multipleTesting, falseDiscoveryRate, islands and minimumNeighbors are validated', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  // The deprecated boolean is an alias of benjamini-hochberg.
  expect(
    new GPUHotSpotAnalysis(createHotSpotProps(graph, {falseDiscoveryRate: true})).multipleTesting
  ).toBe('benjamini-hochberg');
  expect(
    new GPULocalMoran(createLocalMoranProps(graph, {falseDiscoveryRate: true})).multipleTesting
  ).toBe('benjamini-hochberg');
  expect(
    new GPUHotSpotAnalysis(
      createHotSpotProps(graph, {falseDiscoveryRate: true, multipleTesting: 'benjamini-hochberg'})
    ).multipleTesting
  ).toBe('benjamini-hochberg');
  expect(new GPUHotSpotAnalysis(createHotSpotProps(graph)).multipleTesting).toBe('none');
  expect(
    () =>
      new GPUHotSpotAnalysis(
        createHotSpotProps(graph, {falseDiscoveryRate: true, multipleTesting: 'holm'})
      )
  ).toThrow(/falseDiscoveryRate/);
  expect(
    () =>
      new GPULocalMoran(
        createLocalMoranProps(graph, {falseDiscoveryRate: false, multipleTesting: 'bonferroni'})
      )
  ).toThrow(/falseDiscoveryRate/);
  expect(
    () =>
      new GPULocalMoran(
        createLocalMoranProps(graph, {quadrantGating: 'none', multipleTesting: 'holm'})
      )
  ).toThrow(/quadrantGating/);
  expect(
    () => new GPUHotSpotAnalysis(createHotSpotProps(graph, {multipleTesting: 'sidak' as never}))
  ).toThrow(/multipleTesting|method/);
  // Defaults: islands are excluded and one neighbor is enough.
  const defaults = new GPUHotSpotAnalysis(createHotSpotProps(graph));
  expect(defaults.islands).toBe('exclude');
  expect(defaults.minimumNeighbors).toBe(1);
  expect(
    new GPULocalMoran(createLocalMoranProps(graph, {islands: 'keep', minimumNeighbors: 8}))
  ).toMatchObject({islands: 'keep', minimumNeighbors: 8});
  for (const minimumNeighbors of [0, -1, 1.5, NaN]) {
    expect(() => new GPUHotSpotAnalysis(createHotSpotProps(graph, {minimumNeighbors}))).toThrow(
      /minimumNeighbors/
    );
    expect(() => new GPULocalMoran(createLocalMoranProps(graph, {minimumNeighbors}))).toThrow(
      /minimumNeighbors/
    );
  }
  expect(
    () => new GPUHotSpotAnalysis(createHotSpotProps(graph, {islands: 'drop' as never}))
  ).toThrow(/islands/);
  expect(
    () =>
      new GPULocalGeary({
        weights: createWeightsViews(graph),
        values: createTransientView(graph, 'geary-values', 'float32', 10),
        parameters: createTransientView(
          graph,
          'geary-parameters',
          'float32',
          GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        ),
        islands: 'drop' as never,
        result: {statistic: createTransientView(graph, 'geary-result', 'float32', 10)}
      })
  ).toThrow(/islands/);
  // globalStatistics reports [n, mean, variance, sd, islandCount].
  expect(GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH).toBe(5);
  device.destroy();
});

it('getInferenceMetadata describes the p-values for a legend', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const hotSpot = new GPUHotSpotAnalysis(
    createHotSpotProps(graph, {multipleTesting: 'holm', minimumNeighbors: 8})
  );
  expect(hotSpot.getInferenceMetadata()).toMatchObject({
    alternative: 'two-sided',
    permutationCount: 0,
    multipleTesting: 'holm'
  });
  expect(hotSpot.getInferenceMetadata().testedRowsDefinition).toMatch(/8 active neighbors/);
  expect(hotSpot.getInferenceMetadata().significanceLevel).toBeUndefined();
  const moran = new GPULocalMoran(createLocalMoranProps(graph, {islands: 'keep'}));
  expect(moran.getInferenceMetadata(0.01)).toMatchObject({
    alternative: 'two-sided',
    permutationCount: 0,
    multipleTesting: 'none',
    significanceLevel: 0.01
  });
  expect(moran.getInferenceMetadata().significanceLevel).toBe(0.05);
  expect(moran.getInferenceMetadata().testedRowsDefinition).not.toMatch(/islands excluded/);
  device.destroy();
});

/** CPU oracle outputs against the esda configurations of the reference fixture. */
it('the CPU oracles reproduce esda G_Local, Moran_Local and statsmodels on the island fixture', () => {
  // Oracle values: fix/work/A/oracle-u3-autocorrelation.py (esda 2.10.0, statsmodels 0.15.0).
  const values = getReferenceValues();
  const weights = getReferenceWeights();
  const rowWeights = getReferenceWeights({rowStandardize: true});
  const mask = getReferenceMask();
  const configurations = [
    {name: 'keep', input: {values, islands: 'keep' as const}, reference: REFERENCE.keep},
    {name: 'exclude', input: {values, islands: 'exclude' as const}, reference: REFERENCE.exclude},
    {
      name: 'masked',
      input: {values, mask, islands: 'exclude' as const},
      reference: REFERENCE.masked
    }
  ];
  for (const {name, input, reference} of configurations) {
    expect(computeHotSpotOracle({...input, weights}).moments.count, name).toBe(reference.count);
    expect(computeHotSpotOracle({...input, weights}).moments.mean, name).toBeCloseTo(
      reference.mean,
      9
    );
    for (const [selfWeight, expected] of [
      [1, reference.giStar],
      [0, reference.gi]
    ] as const) {
      const oracle = computeHotSpotOracle({...input, weights, selfWeight});
      for (const [row, zScore] of expected.entries()) {
        if (Number.isNaN(zScore)) {
          expect(oracle.zScores[row], `${name} self ${selfWeight} row ${row}`).toBeNaN();
        } else {
          expect(oracle.zScores[row], `${name} self ${selfWeight} row ${row}`).toBeCloseTo(
            zScore,
            9
          );
        }
      }
    }
    for (const [standardized, expectedI, expectedLag] of [
      [true, reference.localIRow, reference.lagRow],
      [false, reference.localIBinary, reference.lagBinary]
    ] as const) {
      const oracle = computeLocalMoranOracle({
        ...input,
        weights: standardized ? rowWeights : weights,
        rowStandardized: standardized
      });
      for (const [row, localI] of expectedI.entries()) {
        const label = `${name} ${standardized ? 'row' : 'binary'} row ${row}`;
        if (Number.isNaN(localI)) {
          expect(oracle.localI[row], label).toBeNaN();
        } else {
          expect(oracle.localI[row], label).toBeCloseTo(localI, 9);
          expect(oracle.spatialLag[row], label).toBeCloseTo(expectedLag[row], 9);
        }
      }
    }
  }
  // Islands: 'keep' counts the island, and the island has no z-score under either policy.
  expect(computeHotSpotOracle({values, weights, islands: 'keep'}).islandCount).toBe(1);
  expect(computeHotSpotOracle({values, weights}).zScores[REFERENCE.island]).toBeNaN();
  expect(
    computeHotSpotOracle({values, weights, islands: 'keep'}).zScores[REFERENCE.island]
  ).toBeNaN();
  // Multiple testing: bins from the adjusted p-values equal statsmodels.
  for (const method of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const oracle = computeHotSpotOracle(
      {values, weights, islands: 'keep'},
      {multipleTesting: method}
    );
    expect(oracle.bins, method).toEqual(REFERENCE.multipleTesting[method].bins);
    for (const [row, adjusted] of REFERENCE.multipleTesting[method].adjusted.entries()) {
      if (Number.isNaN(adjusted)) {
        expect(oracle.adjustedPValues[row], `${method} row ${row}`).toBeNaN();
      } else {
        expect(oracle.adjustedPValues[row], `${method} row ${row}`).toBeCloseTo(adjusted, 6);
      }
    }
  }
  expect(REFERENCE.multipleTesting.none.bins).toEqual(REFERENCE.noneBinsFromZ);
  expect(GPU_HOT_SPOT_SIGNIFICANCE_LEVELS).toEqual([0.1, 0.05, 0.01]);
});
