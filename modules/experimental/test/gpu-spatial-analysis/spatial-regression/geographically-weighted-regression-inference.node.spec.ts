// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import type {GPUGeographicallyWeightedRegressionProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {getGPUGeographicallyWeightedRegressionModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  getStudentTCriticalValue,
  getStudentTTwoSidedTail
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeGeographicallyWeightedRegressionOnCPU,
  type GeographicallyWeightedRegressionOracleResult
} from './geographically-weighted-regression-oracle';
import {
  createGeographicScene,
  createNonstationarityScene,
  createSingularRowScene
} from './geographically-weighted-regression-scenes';
import {
  MGWR_BANDWIDTH_SEARCH_REFERENCE,
  MGWR_INFERENCE_REFERENCE,
  MGWR_SPHERICAL_REFERENCE
} from './gwr-mgwr-inference-reference';

const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;
const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;

/**
 * `scipy.stats.t.ppf(1 - tail / 2, df)` as `[df, two-sided tail, quantile]`; produced by
 * fix/work/C/U4/pin-student-t.py (scipy 1.18.1).
 */
const SCIPY_STUDENT_T_QUANTILES: [number, number, number][] = [
  [1, 0.5, 1.0000000000000002],
  [1, 0.1, 6.313751514675037],
  [1, 0.05, 12.706204736174694],
  [1, 0.0025, 254.64659994875322],
  [1, 0.0001, 6366.197671316637],
  [1, 1e-6, 636619.7724194303],
  [2, 0.5, 0.8164965809277261],
  [2, 0.1, 2.9199855803537242],
  [2, 0.05, 4.302652729749462],
  [2, 0.0025, 19.962480451643184],
  [2, 0.0001, 99.99249984375004],
  [2, 1e-6, 999.9992500409771],
  [5, 0.5, 0.7266868438004227],
  [5, 0.1, 2.015048373333023],
  [5, 0.05, 2.5705818356363146],
  [5, 0.0025, 5.6041654784981985],
  [5, 0.0001, 11.177710070278868],
  [5, 1e-6, 28.47847346345525],
  [10, 0.5, 0.6998120613124317],
  [10, 0.1, 1.812461122811676],
  [10, 0.05, 2.228138851986274],
  [10, 0.0025, 4.004530447670987],
  [10, 0.0001, 6.211050891290746],
  [10, 1e-6, 10.516489957008547],
  [30, 0.5, 0.6827556933212926],
  [30, 0.1, 1.6972608865939571],
  [30, 0.05, 2.0422724563012378],
  [30, 0.0025, 3.299893643532204],
  [30, 0.0001, 4.482417175409787],
  [30, 1e-6, 6.119075620413732],
  [143, 0.5, 0.6762092644319704],
  [143, 0.1, 1.6555791434318061],
  [143, 0.05, 1.976692197929798],
  [143, 0.0025, 3.0778243235182834],
  [143, 0.0001, 4.003158505167718],
  [143, 1e-6, 5.113147061506942],
  [2000, 0.5, 0.6746124375310221],
  [2000, 0.1, 1.6456158666989074],
  [2000, 0.05, 1.9611508260994375],
  [2000, 0.0025, 3.027178217417666],
  [2000, 0.0001, 3.8984536633332953],
  [2000, 1e-6, 4.90692236544896],
  [20000, 0.5, 0.674502017141114],
  [20000, 0.1, 1.6449298189594808],
  [20000, 0.05, 1.9600826051581348],
  [20000, 0.0025, 3.023724715340209],
  [20000, 0.0001, 3.8913767940234796],
  [20000, 1e-6, 4.893163131942165]
];

function expectRelativelyClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string,
  floor = 1e-9
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const scale = Math.max(floor, Math.abs(expected[index]));
    expect(
      Math.abs(actual[index] - expected[index]) / scale,
      `${label}[${index}] ${actual[index]} vs ${expected[index]}`
    ).toBeLessThanOrEqual(tolerance);
  }
}

const planar = createNonstationarityScene(12, 77);

function runOracle(
  settings: GPUGeographicallyWeightedRegressionSettings,
  options: {
    scene?: typeof planar;
    sphereRadius?: number;
    maximumBandwidthCount?: number;
    refinementPassCount?: number;
  } = {}
): GeographicallyWeightedRegressionOracleResult {
  return computeGeographicallyWeightedRegressionOnCPU({
    ...(options.scene ?? planar),
    settings,
    maximumBandwidthCount: options.maximumBandwidthCount,
    refinementPassCount: options.refinementPassCount,
    sphereRadius: options.sphereRadius
  });
}

/** Summary array as the GPU writes it, from the float64 oracle. */
function getOracleSummary(result: GeographicallyWeightedRegressionOracleResult): number[] {
  const summary = new Array<number>(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH).fill(
    Number.NaN
  );
  summary[SUMMARY.RESIDUAL_SUM_OF_SQUARES] = result.residualSumOfSquares;
  summary[SUMMARY.TRACE_OF_HAT] = result.traceOfHat;
  summary[SUMMARY.AICC] = result.aicc;
  summary[SUMMARY.R_SQUARED] = result.rSquared;
  summary[SUMMARY.OBSERVATION_COUNT] = result.observationCount;
  summary[SUMMARY.HAS_VALID_CANDIDATE] = result.hasValidCandidate ? 1 : 0;
  summary[SUMMARY.SINGULAR_ROW_COUNT] = result.singularRowCount;
  summary[SUMMARY.SIGMA_SQUARED] = result.sigmaSquared;
  summary[SUMMARY.ADJUSTED_ALPHA] = result.adjustedAlpha;
  summary[SUMMARY.CRITICAL_T] = result.criticalT;
  summary[SUMMARY.LOG_LIKELIHOOD] = result.logLikelihood;
  summary[SUMMARY.INCLUDED_ROW_COUNT] = result.includedRowCount;
  return summary;
}

/**
 * Adaptive bandwidths: mgwr takes the k-th neighbour distance times 1.0000001, the contributor
 * times 1.00001 (float32 needs the larger margin to keep that neighbour's weight positive), which
 * moves hat traces and standard errors by about 1e-5 relative.
 */
const ADAPTIVE_TOLERANCE = 5e-4;

type MgwrFit = (typeof MGWR_INFERENCE_REFERENCE)['fixedBisquare22'];

function expectMatchesMgwr(
  result: GeographicallyWeightedRegressionOracleResult,
  reference: MgwrFit,
  label: string,
  tolerance = 1e-6
): void {
  expect(result.observationCount, `${label} n`).toBe(reference.n);
  expect(result.singularRowCount, `${label} singular`).toBe(0);
  expectRelativelyClose([result.traceOfHat], [reference.enp], tolerance, `${label} ENP`);
  expectRelativelyClose([result.sigmaSquared], [reference.sigma2], tolerance, `${label} sigma2`);
  expectRelativelyClose([result.aicc], [reference.aicc], tolerance, `${label} AICc`);
  expectRelativelyClose([result.logLikelihood], [reference.llf], tolerance, `${label} llf`);
  expectRelativelyClose([result.rSquared], [reference.r2], tolerance, `${label} R2`);
  expectRelativelyClose(
    [result.adjustedAlpha],
    [reference.adjAlpha05],
    tolerance,
    `${label} adj alpha`
  );
  expectRelativelyClose(
    [result.criticalT],
    [reference.criticalT],
    tolerance,
    `${label} critical t`
  );
}

function expectArraysMatchMgwr(
  result: GeographicallyWeightedRegressionOracleResult,
  reference: MgwrFit,
  label: string,
  tolerance = 1e-6
): void {
  expectRelativelyClose(
    result.coefficients,
    reference.coefficients,
    tolerance,
    `${label} coefficients`,
    0.1
  );
  expectRelativelyClose(
    result.standardErrors,
    reference.standardErrors,
    tolerance,
    `${label} bse`,
    0.01
  );
  expectRelativelyClose(result.tValues, reference.tValues, tolerance, `${label} tvalues`, 1);
  expect(result.significant, `${label} significance mask`).toEqual(reference.significant);
}

it('the float64 oracle reproduces mgwr local inference (fixed and adaptive bisquare)', () => {
  const fixed = runOracle({kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [2.2]});
  expectMatchesMgwr(fixed, MGWR_INFERENCE_REFERENCE.fixedBisquare22, 'fixed 2.2');
  expectArraysMatchMgwr(fixed, MGWR_INFERENCE_REFERENCE.fixedBisquare22, 'fixed 2.2');
  const adaptive = runOracle({kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [20]});
  expectMatchesMgwr(
    adaptive,
    MGWR_INFERENCE_REFERENCE.adaptiveBisquare20,
    'adaptive 20',
    ADAPTIVE_TOLERANCE
  );
  expectArraysMatchMgwr(
    adaptive,
    MGWR_INFERENCE_REFERENCE.adaptiveBisquare20,
    'adaptive 20',
    ADAPTIVE_TOLERANCE
  );
  const gaussian = runOracle({kernel: 'gaussian', bandwidthMode: 'fixed', bandwidths: [1.5]});
  expectMatchesMgwr(gaussian, MGWR_INFERENCE_REFERENCE.fixedGaussian15, 'gaussian 1.5');
});

it('the significance level setting changes the adjusted alpha like mgwr adj_alpha[0]', () => {
  const result = runOracle({
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    bandwidths: [2.2],
    significanceLevel: 0.1
  });
  expectRelativelyClose(
    [result.adjustedAlpha],
    [MGWR_INFERENCE_REFERENCE.adjustedAlpha10],
    1e-6,
    'adjusted alpha at 0.1'
  );
  const values = getGPUGeographicallyWeightedRegressionParameterValues(
    {bandwidths: [2.2], significanceLevel: 0.1},
    4
  );
  expect(values[3]).toBeCloseTo(0.1, 6);
  expect(() =>
    getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [2], significanceLevel: 1})
  ).toThrow(/significanceLevel/);
  expect(getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [2]})[3]).toBe(0);
});

it('the float64 oracle reproduces mgwr spherical (haversine) fits with a 6371 km sphere', () => {
  const geographic = createGeographicScene(planar);
  const options = {scene: geographic, sphereRadius: 6371000};
  const fixed = runOracle(
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [25000]},
    options
  );
  expectMatchesMgwr(fixed, MGWR_SPHERICAL_REFERENCE.fixedBisquare25km, 'spherical fixed 25 km');
  expectArraysMatchMgwr(fixed, MGWR_SPHERICAL_REFERENCE.fixedBisquare25km, 'spherical fixed');
  const adaptive = runOracle(
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [20]},
    options
  );
  expectMatchesMgwr(
    adaptive,
    MGWR_SPHERICAL_REFERENCE.adaptiveBisquare20,
    'spherical adaptive',
    ADAPTIVE_TOLERANCE
  );
  expectArraysMatchMgwr(
    adaptive,
    MGWR_SPHERICAL_REFERENCE.adaptiveBisquare20,
    'spherical adaptive',
    ADAPTIVE_TOLERANCE
  );
});

it('the Student t critical value matches scipy.stats.t.ppf', () => {
  for (const [degreesOfFreedom, tail, expected] of SCIPY_STUDENT_T_QUANTILES) {
    const actual = getStudentTCriticalValue(tail, degreesOfFreedom);
    expect(
      Math.abs(actual - expected) / expected,
      `df ${degreesOfFreedom} tail ${tail}: ${actual} vs ${expected}`
    ).toBeLessThan(1e-9);
    expect(getStudentTTwoSidedTail(expected, degreesOfFreedom) / tail).toBeCloseTo(1, 7);
  }
  expect(getStudentTCriticalValue(1, 10)).toBe(0);
  expect(getStudentTCriticalValue(3, 10)).toBe(0);
  expect(getStudentTCriticalValue(0, 10)).toBeNaN();
  expect(getStudentTCriticalValue(0.05, 0.5)).toBeNaN();
  // Degrees of freedom beyond the clamp are the normal quantile to 1e-4 relative.
  expect(getStudentTCriticalValue(0.05, 1e6)).toBeCloseTo(1.959964, 3);
});

it('getGPUGeographicallyWeightedRegressionModelResult follows the mgwr conventions', () => {
  const result = runOracle({kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [2.2]});
  const model = getGPUGeographicallyWeightedRegressionModelResult({
    summary: getOracleSummary(result),
    selectedBandwidth: [result.selectedIndex, result.selectedValue]
  });
  const reference = MGWR_INFERENCE_REFERENCE.fixedBisquare22;
  expect(model.kind).toBe('geographically-weighted-regression');
  expect(model.coefficients).toEqual([]);
  expect(model.status).toMatchObject({code: 0, ok: true});
  expect(model.rowCount).toBe(reference.n);
  expectRelativelyClose([model.parameterCount], [reference.enp], 1e-6, 'ENP');
  expectRelativelyClose([model.effectiveParameterCount], [reference.enp], 1e-6, 'effective');
  expectRelativelyClose([model.sigmaSquared], [reference.sigma2], 1e-6, 'sigma2');
  expect(model.standardErrorSigmaSquared).toBe('degrees-of-freedom');
  expectRelativelyClose(
    [model.sigmaSquaredMaximumLikelihood],
    [reference.rss / reference.n],
    1e-6,
    'ML sigma2'
  );
  expectRelativelyClose([model.residualSumOfSquares], [reference.rss], 1e-6, 'RSS');
  expectRelativelyClose([model.logLikelihood!], [reference.llf], 1e-6, 'llf');
  expectRelativelyClose([model.akaikeInformationCriterion!], [reference.aic], 1e-6, 'aic');
  expectRelativelyClose([model.schwarzCriterion!], [reference.bic], 1e-6, 'bic');
  expectRelativelyClose(
    [model.correctedAkaikeInformationCriterion!],
    [reference.aicc],
    1e-6,
    'aicc'
  );
  expectRelativelyClose([model.rSquared], [reference.r2], 1e-6, 'R2');
  expectRelativelyClose([model.adjustedAlpha], [reference.adjAlpha05], 1e-6, 'adjusted alpha');
  expectRelativelyClose([model.criticalT], [reference.criticalT], 1e-6, 'critical t');
  expect(model.singularRowCount).toBe(0);
  expect(model.selectedBandwidth).toEqual({index: 0, value: 2.2});
  expect(model.inference.alternative).toBe('two-sided');

  // No valid candidate: status 1, every statistic NaN, no likelihood.
  const none = getGPUGeographicallyWeightedRegressionModelResult({
    summary: getOracleSummary(runOracle({kernel: 'bisquare', bandwidths: [0.01]}))
  });
  expect(none.status).toMatchObject({code: 1, ok: false});
  expect(none.logLikelihood).toBeNull();
  expect(none.correctedAkaikeInformationCriterion).toBeNull();
  expect(none.selectedBandwidth).toBeNull();
  expect(() => getGPUGeographicallyWeightedRegressionModelResult({summary: [1, 2, 3]})).toThrow(
    /summary/
  );
});

it('bandwidth refinement reaches the exact mgwr integer optimum when the bracket is enumerable', () => {
  const adaptive = MGWR_BANDWIDTH_SEARCH_REFERENCE.adaptive;
  // mgwr's own bounds [40 + 2 p, n] = [46, 144] exclude the optimum and golden section stops at 47.
  expect(adaptive.goldenDefaultBounds).toBe(47);
  expect(adaptive.goldenBounds6to144).toBe(adaptive.argminK);
  const curve = (k: number) => adaptive.aicc[adaptive.ks.indexOf(k)];
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [10, 20, 40, 80, 130]
  };
  // Coarse ladder alone: the best caller candidate is k = 20.
  const coarse = runOracle(settings, {maximumBandwidthCount: 32, refinementPassCount: 0});
  expect(coarse.selectedValue).toBe(20);
  // One pass: the 29 integers strictly inside (10, 40) fit the 32-slot ladder, so all are tried.
  const refined = runOracle(settings, {maximumBandwidthCount: 32, refinementPassCount: 1});
  expect(refined.selectedValue).toBe(adaptive.argminK);
  expect(refined.selectedIndex).toBeGreaterThanOrEqual(32);
  expectRelativelyClose([refined.aicc], [curve(adaptive.argminK)], 1e-6, 'refined AICc');
  // A narrow ladder samples the bracket instead; more passes zoom in on the same optimum.
  const narrow = runOracle(settings, {maximumBandwidthCount: 8, refinementPassCount: 3});
  expect(narrow.selectedValue).toBe(adaptive.argminK);
  // Every evaluated candidate agrees with mgwr's AICc for that k.
  for (const candidate of refined.candidates.filter(entry => Number.isFinite(entry.aicc))) {
    if (candidate.value >= 10) {
      expectRelativelyClose(
        [candidate.aicc],
        [curve(candidate.value)],
        ADAPTIVE_TOLERANCE,
        `AICc at k = ${candidate.value}`
      );
    }
  }
});

it('fixed-bandwidth refinement lands within the mgwr golden-section optimum', () => {
  const fixed = MGWR_BANDWIDTH_SEARCH_REFERENCE.fixed;
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    bandwidths: [1.5, 2, 3, 4, 6]
  };
  const coarse = runOracle(settings, {maximumBandwidthCount: 8, refinementPassCount: 0});
  expect(coarse.selectedValue).toBe(3);
  const refined = runOracle(settings, {maximumBandwidthCount: 8, refinementPassCount: 2});
  expect(refined.aicc).toBeLessThan(coarse.aicc);
  expect(Math.abs(refined.selectedValue - fixed.goldenBounds1to6)).toBeLessThan(0.05);
  // At least as good as mgwr's golden section result (which reports 2.9 on its tolerance).
  expect(refined.aicc).toBeLessThanOrEqual(fixed.goldenAicc + 0.01);
  expect(refined.aicc).toBeGreaterThan(Math.min(...fixed.aicc) - 0.05);
});

it('a one-value ladder or refinement off keeps the caller ladder', () => {
  const single = runOracle(
    {kernel: 'bisquare', bandwidths: [2.2]},
    {maximumBandwidthCount: 8, refinementPassCount: 2}
  );
  expect(single.candidates.filter(entry => Number.isFinite(entry.value))).toHaveLength(1);
  expect(single.selectedValue).toBe(2.2);
});

it('singular rows become NaN rows instead of voiding the candidate', () => {
  const scene = createSingularRowScene(12, 5, 4, 3.5, 30);
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    bandwidths: [4.5, 5, 6, 7, 8]
  };
  const result = runOracle(settings, {scene, maximumBandwidthCount: 8});
  const candidates = result.candidates.slice(0, 5);
  expect(candidates.map(candidate => candidate.singularRowCount)).toEqual([32, 13, 2, 0, 0]);
  // Excluding the singular (noisy) rows gives the smallest AICc to the most singular candidate.
  const aiccArgmin = candidates.reduce(
    (best, candidate, index) => (candidate.aicc < candidates[best].aicc ? index : best),
    0
  );
  expect(aiccArgmin).toBe(0);
  // The rule prefers the candidates with the fewest singular rows: index 3 and 4 tie on zero
  // singular rows and the smaller AICc, k = 8, wins.
  expect(result.selectedIndex).toBe(4);
  expect(result.singularRowCount).toBe(0);
  expect(result.localStatus.every(status => status === STATUS.OK)).toBe(true);

  // With only candidates that all have singular rows, the fewest wins, not the lowest AICc.
  const onlySingular = runOracle(
    {...settings, bandwidths: [4.5, 5, 6]},
    {scene, maximumBandwidthCount: 8}
  );
  expect(onlySingular.selectedIndex).toBe(2);
  expect(onlySingular.singularRowCount).toBe(2);
  const singularRows = onlySingular.localStatus
    .map((status, row) => [status, row])
    .filter(([status]) => status === STATUS.SINGULAR)
    .map(([, row]) => row);
  expect(singularRows).toHaveLength(2);
  for (const row of singularRows) {
    for (const values of [
      onlySingular.coefficients,
      onlySingular.standardErrors,
      onlySingular.tValues
    ]) {
      expect(values.slice(row * 3, row * 3 + 3).every(Number.isNaN)).toBe(true);
    }
    expect(onlySingular.fitted[row]).toBeNaN();
    expect(onlySingular.residuals[row]).toBeNaN();
    expect(onlySingular.hatDiagonal[row]).toBeNaN();
    expect(onlySingular.significant.slice(row * 3, row * 3 + 3)).toEqual([0, 0, 0]);
  }
  // Statistics are over the 142 non-singular rows.
  expect(onlySingular.observationCount).toBe(142);
  expect(onlySingular.includedRowCount).toBe(144);
  expect(Number.isFinite(onlySingular.aicc)).toBe(true);
});

it('invalid candidates stay NaN: bandwidth <= 0, k out of range, no degrees of freedom', () => {
  const nonPositive = runOracle({kernel: 'bisquare', bandwidths: [0, -1, 3]});
  expect(nonPositive.bandwidthScores[0]).toBeNaN();
  expect(nonPositive.bandwidthScores[1]).toBeNaN();
  expect(nonPositive.selectedValue).toBe(3);
  const outOfRange = runOracle({
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [1, 500, 20]
  });
  expect(outOfRange.bandwidthScores[0]).toBeNaN();
  expect(outOfRange.bandwidthScores[1]).toBeNaN();
  expect(outOfRange.selectedValue).toBe(20);
  // A very wide Gaussian is nearly global: n - 2 - tr(S) stays positive, so it is valid; the
  // bandwidth that leaves n - 2 - tr(S) <= 0 is the tiny one with a huge trace.
  const saturated = runOracle({kernel: 'gaussian', bandwidths: [1e-3]});
  expect(saturated.hasValidCandidate).toBe(false);
});

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  rows: number,
  predictorCount: number,
  overrides: Partial<GPUGeographicallyWeightedRegressionProps> = {}
): GPUGeographicallyWeightedRegressionProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `inference-${serial++}`, format, length);
  return {
    positions: view('float32x2', rows),
    predictors: view('float32', rows * predictorCount),
    predictorCount,
    response: view('float32', rows),
    parameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength()),
    output: {coefficients: view('float32', rows * (predictorCount + 1))},
    ...overrides
  };
}

function expectConstructionThrows(
  message: RegExp,
  overrides: (graph: GPUCommandGraph) => Partial<GPUGeographicallyWeightedRegressionProps>,
  rows = 10,
  predictorCount = 2
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPUGeographicallyWeightedRegression(
        createProps(graph, rows, predictorCount, overrides(graph))
      )
  ).toThrow(message);
  device.destroy();
}

it('GPUGeographicallyWeightedRegression validates spatial context, refinement and the new outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const planarFrame = new GPUGeographicallyWeightedRegression(
    createProps(graph, 10, 2, {
      spatialContext: {
        coordinateSpace: 'planar',
        metric: 'native',
        units: 'native',
        frame: {id: 'local', kind: 'local-azimuthal-equidistant'}
      }
    })
  );
  expect(planarFrame.spatialContext.coordinateSpace).toBe('planar');
  const greatCircle = new GPUGeographicallyWeightedRegression(
    createProps(graph, 10, 2, {
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      }
    })
  );
  // The planar grid index is never built for a great-circle context.
  expect(greatCircle.indexGridSize).toBeNull();
  const defaults = new GPUGeographicallyWeightedRegression(createProps(graph, 10, 2));
  expect(defaults.refinementPassCount).toBe(2);
  expect(
    new GPUGeographicallyWeightedRegression(createProps(graph, 10, 2, {refinement: false}))
      .refinementPassCount
  ).toBe(0);
  expect(
    new GPUGeographicallyWeightedRegression(createProps(graph, 10, 2, {refinement: 3}))
      .refinementPassCount
  ).toBe(3);
  expect(
    new GPUGeographicallyWeightedRegression(createProps(graph, 10, 2, {maximumBandwidthCount: 1}))
      .refinementPassCount
  ).toBe(0);
  // 11 predictors (12 design columns) is the new limit.
  const widest = new GPUGeographicallyWeightedRegression(createProps(graph, 10, 11));
  expect(widest.coefficientCount).toBe(12);
  device.destroy();

  expectConstructionThrows(/predictorCount/, () => ({}), 10, 12);
  expectConstructionThrows(/predictorCount/, () => ({}), 10, 0);
  expectConstructionThrows(/refinement/, () => ({refinement: 5}));
  expectConstructionThrows(/refinement/, () => ({refinement: 1.5}));
  expectConstructionThrows(/project positions to a planar frame first|great-circle|metric/, () => ({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
  }));
  expectConstructionThrows(/does not support discrete-grid/, () => ({
    spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
  }));
  expectConstructionThrows(/indexGridSize is planar-only/, () => ({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    indexGridSize: [4, 4]
  }));
  expectConstructionThrows(/at most 65536 rows without a grid index/, graph => ({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    positions: createTransientView(graph, `big-${serial++}`, 'float32x2', 65537),
    predictors: createTransientView(graph, `big-${serial++}`, 'float32', 65537 * 2),
    response: createTransientView(graph, `big-${serial++}`, 'float32', 65537),
    output: {coefficients: createTransientView(graph, `big-${serial++}`, 'float32', 65537 * 3)}
  }));
  expectConstructionThrows(/output.standardErrors must hold/, graph => ({
    output: {
      coefficients: createTransientView(graph, `c-${serial++}`, 'float32', 30),
      standardErrors: createTransientView(graph, `s-${serial++}`, 'float32', 29)
    }
  }));
  expectConstructionThrows(/output.significant/, graph => ({
    output: {
      coefficients: createTransientView(graph, `c-${serial++}`, 'float32', 30),
      significant: createTransientView(graph, `s-${serial++}`, 'float32', 30) as never
    }
  }));
  expectConstructionThrows(/output.candidateTable must hold at least 672/, graph => ({
    output: {
      coefficients: createTransientView(graph, `c-${serial++}`, 'float32', 30),
      candidateTable: createTransientView(graph, `t-${serial++}`, 'float32', 100)
    }
  }));
});

it('the summary layout stays backwards compatible in its first six slots', () => {
  expect(SUMMARY.RESIDUAL_SUM_OF_SQUARES).toBe(0);
  expect(SUMMARY.TRACE_OF_HAT).toBe(1);
  expect(SUMMARY.AICC).toBe(2);
  expect(SUMMARY.R_SQUARED).toBe(3);
  expect(SUMMARY.OBSERVATION_COUNT).toBe(4);
  expect(SUMMARY.HAS_VALID_CANDIDATE).toBe(5);
  expect(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH).toBe(12);
});
