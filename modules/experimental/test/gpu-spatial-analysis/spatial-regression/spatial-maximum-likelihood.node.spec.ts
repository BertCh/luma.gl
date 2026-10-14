// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  fitSpatialErrorMaximumLikelihoodOnCPU,
  fitSpatialLagMaximumLikelihoodOnCPU,
  getSpatialFrobeniusTraceOnCPU,
  searchSpatialMaximumLikelihoodParameter
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-cpu';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  getSpatialLogDeterminantDerivativeFromTraces,
  getSpatialMaximumLikelihoodSeriesLimit,
  getSpatialMaximumLikelihoodTraceOne,
  getSpatialMaximumLikelihoodTraceTwo
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {
  getSpatialLagImpactsFromTraces,
  getSpatialLogDeterminantFromTraces,
  getSpatialLogDeterminantTruncationBound,
  getSpatialWeightsTracesOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {invertMatrix, multiply, toDenseWeights} from './spatial-regression-diagnostics-oracle';
import {
  SPATIAL_MAXIMUM_LIKELIHOOD_SCENES,
  SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE,
  SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE,
  type SpatialMaximumLikelihoodFixture
} from './spatial-maximum-likelihood-reference';

// spreg minimizes with scipy `minimize_scalar` (bounded, xatol 1e-7), so its `rho` / `lambda`
// are only the exact maximizer to about 1e-7; every value derived from it (betas, standard
// errors, likelihood) inherits that. Measured agreement of the float64 mirror with the pins:
// parameter 4e-8, coefficients 3e-7, standard errors 1e-7, z 4e-7, log-likelihood 2e-13 relative.
// The tolerances below are 10 to 100 times those.
const PARAMETER_TOLERANCE = 2e-6;
const RELATIVE_TOLERANCE = 1e-5;

function expectRelative(label: string, actual: number, expected: number, tolerance: number) {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    tolerance * Math.max(Math.abs(expected), 1e-3)
  );
}

function expectFixture(
  fit: {
    coefficients: number[];
    standardErrors: number[];
    zStatistics: number[];
    pValues: number[];
    sigmaSquared: number;
    logLikelihood: number;
    akaikeInformationCriterion: number;
    schwarzCriterion: number;
    pseudoRSquared: number;
    conditioning: {
      conditionNumber: number;
      correlationConditionNumber: number;
      maximumVarianceInflationFactor: number;
    };
  },
  parameter: number,
  fixture: SpatialMaximumLikelihoodFixture
) {
  expect(Math.abs(parameter - fixture.parameter)).toBeLessThan(PARAMETER_TOLERANCE);
  fixture.betas.forEach((value, index) => {
    expectRelative(`coefficient ${index}`, fit.coefficients[index], value, RELATIVE_TOLERANCE);
    expectRelative(
      `se ${index}`,
      fit.standardErrors[index],
      fixture.standardErrors[index],
      RELATIVE_TOLERANCE
    );
    expectRelative(`z ${index}`, fit.zStatistics[index], fixture.z[index], RELATIVE_TOLERANCE);
    expect(Math.abs(fit.pValues[index] - fixture.p[index])).toBeLessThan(1e-7);
  });
  expectRelative('sigma2', fit.sigmaSquared, fixture.sigma2, RELATIVE_TOLERANCE);
  expectRelative('logll', fit.logLikelihood, fixture.logLikelihood, 1e-8);
  // AIC / Schwarz use spreg's k: rho counted for the lag model, lambda not counted for the error.
  expectRelative('aic', fit.akaikeInformationCriterion, fixture.aic, 1e-8);
  expectRelative('schwarz', fit.schwarzCriterion, fixture.schwarz, 1e-8);
  expectRelative('pr2', fit.pseudoRSquared, fixture.pseudoRSquared, RELATIVE_TOLERANCE);
  expectRelative(
    'condition',
    fit.conditioning.conditionNumber,
    fixture.conditioning.conditionNumber,
    1e-6
  );
  expectRelative(
    'correlation condition',
    fit.conditioning.correlationConditionNumber,
    fixture.conditioning.correlationConditionNumber,
    1e-6
  );
  expectRelative(
    'vif',
    fit.conditioning.maximumVarianceInflationFactor,
    fixture.conditioning.maximumVarianceInflationFactor,
    1e-6
  );
}

for (const name of Object.keys(SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE) as Array<
  keyof typeof SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE
>) {
  it(`the float64 spatial lag ML mirror matches spreg ML_Lag on ${name}`, () => {
    const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES[name]();
    const fixture = SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE[name];
    const fit = fitSpatialLagMaximumLikelihoodOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    expect(fit).not.toBeNull();
    if (!fit) {
      return;
    }
    expectFixture(fit, fit.rho, fixture);
    expect(fit.atBoundary).toBe(false);
    expect(fit.localMaximumCount).toBe(1);
    expectRelative(
      'pr2_e',
      fit.pseudoRSquaredSpatial,
      fixture.pseudoRSquaredSpatial,
      RELATIVE_TOLERANCE
    );
    // The estimate is the last coefficient and the bound is 0.99-or-series-limit over ||W||_inf.
    expect(fit.coefficients[fit.coefficients.length - 1]).toBe(fit.rho);
    expectRelative(
      'bound',
      fit.bound,
      Math.min(0.99, getSpatialMaximumLikelihoodSeriesLimit(128)) / fixture.rowSumNorm,
      1e-12
    );
    // tr1, tr2, tr3 of W (I - rho W)^-1 against numpy at spreg's rho.
    const {powerTraces: traces, infinityNorm} = getSpatialWeightsTracesOnCPU(scene.weights, 128);
    expectRelative(
      'tr1',
      getSpatialMaximumLikelihoodTraceOne(fixture.parameter, traces, infinityNorm),
      fixture.traces[0],
      1e-6
    );
    expectRelative(
      'tr2',
      getSpatialMaximumLikelihoodTraceTwo(fixture.parameter, traces, infinityNorm),
      fixture.traces[1],
      1e-6
    );
    expectRelative(
      'tr3',
      getSpatialFrobeniusTraceOnCPU(scene.weights, fixture.parameter, 128),
      fixture.traces[2],
      1e-6
    );
    // Impacts: spreg spmultiplier(method='full') from the same rho and betas.
    const rho = fixture.parameter;
    const impacts = getSpatialLagImpactsFromTraces(
      rho,
      1,
      traces,
      getSpatialWeightsTracesOnCPU(scene.weights, 128).powerSums,
      scene.response.length,
      infinityNorm
    );
    expectRelative('direct multiplier', impacts.direct, fixture.impacts.multipliers[0], 1e-5);
    // spreg's total multiplier is 1 / (1 - rho), which assumes a row-standardized W; the exact
    // 1'W^q 1 sums used here agree with it only then (see the GPUSpatialLagImpacts notes).
    if (Math.abs(fixture.rowSumNorm - 1) < 1e-6) {
      expectRelative('indirect multiplier', impacts.indirect, fixture.impacts.multipliers[1], 1e-5);
      expectRelative('total multiplier', impacts.total, fixture.impacts.multipliers[2], 1e-5);
    }
  });
}

for (const name of Object.keys(SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE) as Array<
  keyof typeof SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE
>) {
  it(`the float64 spatial error ML mirror matches spreg ML_Error on ${name}`, () => {
    const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES[name]();
    const fixture = SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE[name];
    const fit = fitSpatialErrorMaximumLikelihoodOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    expect(fit).not.toBeNull();
    if (!fit) {
      return;
    }
    expectFixture(fit, fit.lambda, fixture);
    expect(fit.atBoundary).toBe(false);
    expect(fit.localMaximumCount).toBe(1);
    expectRelative(
      'bound',
      fit.bound,
      Math.min(0.99, getSpatialMaximumLikelihoodSeriesLimit(128)) / fixture.rowSumNorm,
      1e-12
    );
    // spreg ML_Error does not count lambda in k (AIC = 2 (predictors + 1) - 2 logll).
    expect(fixture.specK).toBe(scene.predictorCount + 1);
  });
}

it('spreg k: the lag model counts rho, the error model does not count lambda', () => {
  expect(SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE.lag_12_21.specK).toBe(4);
  expect(SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE.err_lattice_400_31.specK).toBe(3);
});

it('the series limit keeps the worst-case log-determinant remainder below 1e-5 per observation', () => {
  for (const maximumPower of [2, 8, 32, 128, 256]) {
    const limit = getSpatialMaximumLikelihoodSeriesLimit(maximumPower);
    expect(limit).toBeGreaterThan(0);
    expect(limit).toBeLessThanOrEqual(0.99);
    // The remainder bound per observation at the limit is at most the tolerance ...
    expect(getSpatialLogDeterminantTruncationBound(limit, maximumPower, 1)).toBeLessThanOrEqual(
      1.0000001e-5
    );
    // ... and a little beyond the limit it is not (unless the limit is the 0.99 cap).
    if (limit < 0.99) {
      expect(
        getSpatialLogDeterminantTruncationBound(limit + 1e-3, maximumPower, 1)
      ).toBeGreaterThan(1e-5);
    }
  }
  // The documented defaults.
  expect(getSpatialMaximumLikelihoodSeriesLimit(128)).toBeGreaterThan(0.92);
  expect(getSpatialMaximumLikelihoodSeriesLimit(128)).toBeLessThan(0.94);
  expect(getSpatialMaximumLikelihoodSeriesLimit(256)).toBeGreaterThan(0.95);
  expect(getSpatialMaximumLikelihoodSeriesLimit(256)).toBeLessThan(0.97);
  // Default Q = 128: truncation below 1e-6 |LL| (|LL| >= n for unit-variance data) at |rho| <= 0.9.
  expect(getSpatialLogDeterminantTruncationBound(0.9, 128, 1)).toBeLessThan(1e-6);
});

it('the trace series match dense algebra and the log-determinant derivative', () => {
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const {powerTraces: traces, infinityNorm} = getSpatialWeightsTracesOnCPU(scene.weights, 128);
  const w = toDenseWeights(scene.weights);
  const n = w.length;
  const rho = 0.6;
  const inverse = invertMatrix(
    w.map((row, i) => row.map((value, j) => (i === j ? 1 : 0) - rho * value))
  );
  const wa = multiply(w, inverse);
  const trace = (matrix: number[][]) => matrix.reduce((sum, row, index) => sum + row[index], 0);
  expectRelative(
    'tr1',
    getSpatialMaximumLikelihoodTraceOne(rho, traces, infinityNorm),
    trace(wa),
    1e-9
  );
  expectRelative(
    'tr2',
    getSpatialMaximumLikelihoodTraceTwo(rho, traces, infinityNorm),
    trace(multiply(wa, wa)),
    1e-9
  );
  // d/d rho ln|I - rho W| = -tr(W (I - rho W)^-1), by a central difference of the series.
  const step = 1e-5;
  const difference =
    (getSpatialLogDeterminantFromTraces(rho + step, traces, infinityNorm) -
      getSpatialLogDeterminantFromTraces(rho - step, traces, infinityNorm)) /
    (2 * step);
  expectRelative(
    'derivative',
    getSpatialLogDeterminantDerivativeFromTraces(rho, traces, infinityNorm),
    difference,
    1e-7
  );
  expect(getSpatialLogDeterminantDerivativeFromTraces(rho, traces, infinityNorm)).toBeCloseTo(
    -trace(wa),
    8
  );
  expect(n).toBe(144);
});

it('the parameter search finds the global maximum, boundary maxima and reports multimodality', () => {
  // Unimodal: maximum at 0.3.
  const unimodal = searchSpatialMaximumLikelihoodParameter(
    x => -((x - 0.3) ** 2),
    x => -2 * (x - 0.3),
    0.9
  );
  expect(unimodal.parameter).toBeCloseTo(0.3, 9);
  expect(unimodal.localMaximumCount).toBe(1);
  expect(unimodal.atBoundary).toBe(false);
  // Bimodal: maxima at -0.5 (height 0.2) and 0.6 (height 0.5); the taller one wins.
  const objective = (x: number) =>
    0.2 * Math.exp(-((x + 0.5) ** 2) / 0.02) + 0.5 * Math.exp(-((x - 0.6) ** 2) / 0.02);
  const derivative = (x: number) =>
    (0.2 * Math.exp(-((x + 0.5) ** 2) / 0.02) * (-2 * (x + 0.5))) / 0.02 +
    (0.5 * Math.exp(-((x - 0.6) ** 2) / 0.02) * (-2 * (x - 0.6))) / 0.02;
  const bimodal = searchSpatialMaximumLikelihoodParameter(objective, derivative, 0.9);
  expect(bimodal.parameter).toBeCloseTo(0.6, 7);
  expect(bimodal.localMaximumCount).toBe(2);
  // Increasing everywhere: the maximum is the right end.
  const increasing = searchSpatialMaximumLikelihoodParameter(
    x => x,
    () => 1,
    0.9
  );
  expect(increasing.parameter).toBe(0.9);
  expect(increasing.atBoundary).toBe(true);
  // Decreasing everywhere: the left end.
  const decreasing = searchSpatialMaximumLikelihoodParameter(
    x => -x,
    () => -1,
    0.9
  );
  expect(decreasing.parameter).toBe(-0.9);
  expect(decreasing.atBoundary).toBe(true);
  // A non-finite derivative produces no candidate.
  const invalid = searchSpatialMaximumLikelihoodParameter(
    () => NaN,
    () => NaN,
    0.9
  );
  expect(Number.isNaN(invalid.parameter)).toBe(true);
  expect(invalid.localMaximumCount).toBe(0);
});

it('a short series limits the interval and reports a boundary solution', () => {
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  for (const fit of [
    fitSpatialLagMaximumLikelihoodOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount,
      {
        maximumPower: 8
      }
    ),
    fitSpatialErrorMaximumLikelihoodOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount,
      {
        maximumPower: 8
      }
    )
  ]) {
    expect(fit).not.toBeNull();
    // Q = 8 limits |rho| to about 0.34 (< the unrestricted 0.47): the estimate sits on the bound.
    expect(fit?.seriesLimited).toBe(true);
    expect(fit?.atBoundary).toBe(true);
    // ||W||_inf of the float32 row-standardized weights is 1 + 3e-8.
    expectRelative('bound', fit?.bound ?? NaN, getSpatialMaximumLikelihoodSeriesLimit(8), 1e-6);
  }
});

it('Hutchinson traces approximate the exact fit within the measured sampling error', () => {
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const exact = fitSpatialLagMaximumLikelihoodOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  // Measured on seeds 0..2 with 256 probes: |rho - exact| <= 1e-3 and the same standard error to
  // 4 digits; the bound below is 10 times the worst rho error.
  for (const seed of [0, 1, 2]) {
    const stochastic = fitSpatialLagMaximumLikelihoodOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount,
      {traceMode: 'stochastic', probeCount: 256, seed}
    );
    expect(Math.abs((stochastic?.rho ?? NaN) - (exact?.rho ?? NaN))).toBeLessThan(0.01);
    expectRelative(
      'se rho',
      stochastic?.standardErrors.at(-1) ?? NaN,
      exact?.standardErrors.at(-1) ?? NaN,
      0.01
    );
  }
});

it('the CPU fits return null for a singular design', () => {
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const predictors = Float32Array.from(scene.predictors);
  for (let row = 0; row < scene.response.length; row++) {
    predictors[row * 2 + 1] = 2 * predictors[row * 2];
  }
  expect(
    fitSpatialLagMaximumLikelihoodOnCPU(scene.weights, predictors, scene.response, 2)
  ).toBeNull();
  expect(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH).toBe(22);
});
