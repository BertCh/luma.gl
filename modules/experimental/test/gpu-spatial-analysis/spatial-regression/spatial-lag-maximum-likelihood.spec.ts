// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialLagImpacts} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {fitSpatialLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-cpu';
import {
  getGPUSpatialLagMaximumLikelihoodModelResult,
  getGPUSpatialMaximumLikelihoodDiagnostics
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result';
import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_SINGULAR,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_TOO_FEW_ROWS,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED,
  getSpatialMaximumLikelihoodSeriesLimit
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import type {GPUSpatialWeightsTracesMode} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createLatticeWeights} from './spatial-regression-diagnostics-oracle';
import {
  SPATIAL_MAXIMUM_LIKELIHOOD_SCENES,
  SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE,
  type SpatialLagMaximumLikelihoodFixture
} from './spatial-maximum-likelihood-reference';

type Scene = ReturnType<(typeof SPATIAL_MAXIMUM_LIKELIHOOD_SCENES)['lag_12_21']>;
type Result = {
  status: number;
  converged: number;
  table: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  impacts: Float32Array;
};

type RunOptions = {
  tileRowCount?: number;
  maximumPower?: number;
  traceMode?: GPUSpatialWeightsTracesMode;
  traceProbeCount?: number;
  withImpacts?: boolean;
};

/** Runs the ML lag fit (and optionally the impacts node on its table and traces) in one graph. */
async function runLagMaximumLikelihood(
  device: Device,
  scene: Scene,
  {
    tileRowCount,
    maximumPower = 128,
    traceMode,
    traceProbeCount,
    withImpacts = false
  }: RunOptions = {}
): Promise<Result> {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const tableLength = (k + 2) * 4;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'lag-ml-graph'});
  const weights = scene.weights;
  const tableBuffer = track(createOutputBuffer(device, tableLength));
  const summaryBuffer = track(
    createOutputBuffer(device, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
  );
  const statusBuffer = track(createOutputBuffer(device, 1));
  const convergedBuffer = track(createOutputBuffer(device, 1));
  const residualBuffer = track(createOutputBuffer(device, rows));
  const impactsBuffer = track(createOutputBuffer(device, k * 3));
  const table = importGraphBuffer(graph, 'table', tableBuffer, 'float32', tableLength);
  const status = importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1);
  const powerTraces = importGraphBuffer(
    graph,
    'power-traces',
    track(createOutputBuffer(device, maximumPower)),
    'float32',
    maximumPower
  );
  const powerSums = importGraphBuffer(
    graph,
    'power-sums',
    track(createOutputBuffer(device, maximumPower)),
    'float32',
    maximumPower
  );
  const infinityNorm = importGraphBuffer(
    graph,
    'infinity-norm',
    track(createOutputBuffer(device, 1)),
    'float32',
    1
  );
  graph.add(
    new GPUSpatialLagMaximumLikelihood({
      id: 'lag-ml',
      weights: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          track(createInputBuffer(device, weights.offsets)),
          'uint32',
          weights.offsets.length
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          track(createInputBuffer(device, weights.neighbors)),
          'uint32',
          weights.neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'weights',
          track(createInputBuffer(device, weights.weights)),
          'float32',
          weights.weights.length
        )
      },
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device, scene.predictors)),
        'float32',
        rows * k
      ),
      response: importGraphBuffer(
        graph,
        'response',
        track(createInputBuffer(device, scene.response)),
        'float32',
        rows
      ),
      predictorCount: k,
      maximumPower,
      tileRowCount,
      traceMode,
      traceProbeCount,
      output: {
        table,
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
        ),
        status,
        converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1),
        residuals: importGraphBuffer(graph, 'residuals', residualBuffer, 'float32', rows),
        powerTraces,
        powerSums,
        infinityNorm
      }
    })
  );
  if (withImpacts) {
    graph.add(
      new GPUSpatialLagImpacts({
        id: 'lag-ml-impacts',
        coefficients: table,
        coefficientStride: 4,
        rhoIndex: k + 1,
        predictorCount: k,
        rowCount: rows,
        maximumPower,
        traces: {powerTraces, powerSums, infinityNorm},
        status,
        output: {impacts: importGraphBuffer(graph, 'impacts-out', impactsBuffer, 'float32', k * 3)}
      })
    );
  }
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    status: (await readUint32(statusBuffer, 1))[0],
    converged: (await readUint32(convergedBuffer, 1))[0],
    table: Float32Array.from(await readFloat32(tableBuffer, tableLength)),
    summary: Float32Array.from(
      await readFloat32(summaryBuffer, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
    ),
    residuals: Float32Array.from(await readFloat32(residualBuffer, rows)),
    impacts: Float32Array.from(await readFloat32(impactsBuffer, k * 3))
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectClose(
  label: string,
  actual: number,
  expected: number,
  relative: number,
  floor: number
) {
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.max(Math.abs(expected), floor)
  );
}

// Float32 tolerances (not measured yet: the first headless run should print the worst ratios and
// tighten them). They follow from the error sources: the OLS fits that feed the concentrated
// likelihood agree with float64 to ~1e-5 relative; the likelihood derivative is a ratio of f32
// sums (1e-6 relative) so `rho` moves by well under 1e-3; the log-determinant is a Horner series
// of f32 traces (1e-6 relative); the standard errors add the tr3 probe sums and a Schur
// complement with mild cancellation (1e-3 relative); coefficient and `rho` bounds follow the
// error-GM spec (2e-4 coefficients, 5e-3 `rho`) but are widened 10x for the sums of squares.
const PARAMETER_TOLERANCE = 2e-3;
const COEFFICIENT_TOLERANCE = 2e-3;
const STANDARD_ERROR_TOLERANCE = 1e-2;
const LOG_LIKELIHOOD_TOLERANCE = 0.02;

function expectParity(actual: Result, fixture: SpatialLagMaximumLikelihoodFixture, k: number) {
  expect(actual.status).toBe(0);
  expect(actual.converged).toBe(1);
  expect(
    Math.abs(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER] - fixture.parameter)
  ).toBeLessThan(PARAMETER_TOLERANCE);
  for (let row = 0; row < k + 2; row++) {
    expectClose(
      `coefficient ${row}`,
      actual.table[row * 4],
      fixture.betas[row],
      COEFFICIENT_TOLERANCE,
      0.05
    );
    expectClose(
      `se ${row}`,
      actual.table[row * 4 + 1],
      fixture.standardErrors[row],
      STANDARD_ERROR_TOLERANCE,
      1e-3
    );
    expectClose(`z ${row}`, actual.table[row * 4 + 2], fixture.z[row], 2e-2, 0.2);
    expect(Math.abs(actual.table[row * 4 + 3] - fixture.p[row])).toBeLessThan(5e-3);
  }
  expect(actual.summary[0]).toBe(actual.residuals.length);
  expectClose('sigma2', actual.summary[2], fixture.sigma2, 1e-3, 1e-3);
  expect(Math.abs(actual.summary[3] - fixture.logLikelihood)).toBeLessThan(
    LOG_LIKELIHOOD_TOLERANCE
  );
  expect(Math.abs(actual.summary[4] - fixture.aic)).toBeLessThan(2 * LOG_LIKELIHOOD_TOLERANCE);
  expect(Math.abs(actual.summary[5] - fixture.schwarz)).toBeLessThan(2 * LOG_LIKELIHOOD_TOLERANCE);
  expectClose('pr2', actual.summary[6], fixture.pseudoRSquared, 1e-3, 1e-3);
  expectClose('pr2_e', actual.summary[7], fixture.pseudoRSquaredSpatial, 1e-3, 1e-3);
  expectClose('condition number', actual.summary[9], fixture.conditioning.conditionNumber, 5e-2, 1);
  expectClose(
    'vif',
    actual.summary[11],
    fixture.conditioning.maximumVarianceInflationFactor,
    5e-2,
    1
  );
  expect(actual.summary[12]).toBe(0);
  // Exact traces below 4096 rows, an interior maximum, a defined variance.
  expect(actual.summary[14]).toBe(0);
  expect(actual.summary[16]).toBe(0);
  expect(actual.summary[19]).toBe(1);
  expect(actual.summary[20]).toBe(1);
  // The truncation bound at the estimate is far below the likelihood.
  expect(actual.summary[18]).toBeLessThan(1e-3);
}

for (const name of Object.keys(SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE) as Array<
  keyof typeof SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE
>) {
  it(`GPUSpatialLagMaximumLikelihood matches spreg ML_Lag on ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES[name]();
    const fixture = SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE[name];
    const actual = await runLagMaximumLikelihood(device, scene);
    expectParity(actual, fixture, scene.predictorCount);
    // The fit must be a real computation, not a failed-compile zero.
    expect(Math.abs(actual.summary[1])).toBeGreaterThan(0.01);
    expect(actual.summary[2]).toBeGreaterThan(0.1);
    // Residuals are the fit's `u = e0 - rho e1`: their sum of squares is `n sigma2`.
    let sum = 0;
    for (const value of actual.residuals) {
      sum += value * value;
    }
    expectClose('residual sum of squares', sum, actual.summary[8], 1e-3, 1e-3);
    // The model result built from the read-back arrays reproduces spreg's AIC and Schwarz.
    const result = getGPUSpatialLagMaximumLikelihoodModelResult(toReadback(actual));
    expect(Math.abs((result.akaikeInformationCriterion ?? NaN) - fixture.aic)).toBeLessThan(0.05);
    expect(Math.abs((result.schwarzCriterion ?? NaN) - fixture.schwarz)).toBeLessThan(0.05);
    expect(result.parameterCount).toBe(fixture.specK);
    expect(result.status.ok).toBe(true);
  });
}

it('GPUSpatialLagMaximumLikelihood is deterministic and tile-size independent within tolerance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const fixture = SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE.lag_12_21;
  const first = await runLagMaximumLikelihood(device, scene);
  const second = await runLagMaximumLikelihood(device, scene);
  // Fixed-order tile sums and Kahan merges, no atomics: bitwise identical.
  expect(Array.from(second.table)).toEqual(Array.from(first.table));
  expect(Array.from(second.summary)).toEqual(Array.from(first.summary));
  for (const tileRowCount of [7, 144]) {
    expectParity(
      await runLagMaximumLikelihood(device, scene, {tileRowCount}),
      fixture,
      scene.predictorCount
    );
  }
});

it('GPUSpatialLagMaximumLikelihood agrees with the float64 mirror under Hutchinson traces', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const options = {traceMode: 'stochastic', traceProbeCount: 256} as const;
  const actual = await runLagMaximumLikelihood(device, scene, options);
  // The GPU and the CPU mirror draw the same Philox probes (seed 0), so they agree to float32.
  const expected = fitSpatialLagMaximumLikelihoodOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount,
    {traceMode: 'stochastic', probeCount: 256}
  );
  expect(actual.status).toBe(0);
  expect(actual.summary[14]).toBe(1);
  expect(Math.abs(actual.summary[1] - (expected?.rho ?? NaN))).toBeLessThan(PARAMETER_TOLERANCE);
  expectClose(
    'se rho',
    actual.table[3 * 4 + 1],
    expected?.standardErrors[3] ?? NaN,
    STANDARD_ERROR_TOLERANCE,
    1e-3
  );
});

it('GPUSpatialLagMaximumLikelihood reports a boundary solution when the series limits the interval', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  // Q = 8 limits |rho| to about 0.34, below the unrestricted maximum 0.47.
  const actual = await runLagMaximumLikelihood(device, scene, {maximumPower: 8});
  const bound = actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND];
  expect(actual.status).toBe(0);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED]).toBe(1);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY]).toBe(1);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED]).toBe(0);
  expect(actual.converged).toBe(0);
  expectClose('bound', bound, getSpatialMaximumLikelihoodSeriesLimit(8), 1e-5, 1e-3);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER]).toBeCloseTo(bound, 5);
  const diagnostics = getGPUSpatialMaximumLikelihoodDiagnostics(actual.summary);
  expect(diagnostics.atBoundary).toBe(true);
  expect(getGPUSpatialLagMaximumLikelihoodModelResult(toReadback(actual)).status.converged).toBe(
    false
  );
});

it('GPUSpatialLagMaximumLikelihood reports singular, too-small and non-finite inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_12_21();
  const singular = {...scene, predictors: Float32Array.from(scene.predictors)};
  for (let row = 0; row < scene.response.length; row++) {
    singular.predictors[row * 2 + 1] = 2 * singular.predictors[row * 2];
  }
  const collinear = await runLagMaximumLikelihood(device, singular);
  expect(collinear.status).toBe(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_SINGULAR);
  expect(Number.isNaN(collinear.table[0])).toBe(true);
  expect(Number.isNaN(collinear.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER])).toBe(
    true
  );
  expect(collinear.summary[0]).toBe(144);

  // 4 rows with 2 predictors: n <= predictorCount + 2.
  const small = {
    weights: createLatticeWeights(2, true),
    predictors: Float32Array.of(1, 5, 2, 3, 4, 1, 3, 2),
    response: Float32Array.of(1, 2, 4, 3),
    predictorCount: 2
  };
  const tooFew = await runLagMaximumLikelihood(device, small);
  expect(tooFew.status).toBe(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_TOO_FEW_ROWS);
  expect(Number.isNaN(tooFew.table[0])).toBe(true);

  const missing = {...scene, response: Float32Array.from(scene.response)};
  missing.response[5] = Number.NaN;
  const nonFinite = await runLagMaximumLikelihood(device, missing);
  expect(nonFinite.status).toBe(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  expect(Number.isNaN(nonFinite.table[0])).toBe(true);
  const result = getGPUSpatialLagMaximumLikelihoodModelResult(toReadback(nonFinite));
  expect(result.status.ok).toBe(false);
});

for (const name of ['lag_12_21', 'lag_6pred_10_7'] as const) {
  it(`ML lag chains into GPUSpatialLagImpacts and matches spreg ML_Lag spat_impacts='full' on ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES[name]();
    const fixture = SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE[name];
    const actual = await runLagMaximumLikelihood(device, scene, {withImpacts: true});
    expect(actual.status).toBe(0);
    // spreg's total multiplier is 1 / (1 - rho), exact for these row-standardized weights. The
    // tolerance is the `rho` and beta tolerances propagated through the multipliers (the
    // multiplier derivative is 1 / (1 - rho)^2 <= 4 here).
    fixture.impacts.values.forEach((expected, predictor) => {
      for (let column = 0; column < 3; column++) {
        expectClose(
          `impact ${predictor}/${column}`,
          actual.impacts[predictor * 3 + column],
          expected[column],
          2e-2,
          0.05
        );
      }
    });
  });
}

it('ML lag impacts on a raw binary W use the scaled series (direct impacts match spreg)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.lag_binary_10_5();
  const fixture = SPREG_LAG_MAXIMUM_LIKELIHOOD_REFERENCE.lag_binary_10_5;
  const actual = await runLagMaximumLikelihood(device, scene, {withImpacts: true});
  expect(actual.status).toBe(0);
  // spreg's total multiplier 1 / (1 - rho) assumes a row-standardized W, so only the direct
  // impacts (tr((I - rho W)^-1) / n, valid for any W) are compared; the total must be finite and
  // exceed 1 / (1 - rho) because the row sums reach 4.
  fixture.impacts.values.forEach((expected, predictor) => {
    expectClose('direct', actual.impacts[predictor * 3], expected[0], 2e-2, 0.05);
    expect(Number.isFinite(actual.impacts[predictor * 3 + 2])).toBe(true);
  });
});

/** The model-result readers take the status as a one-element array, as read back from the GPU. */
function toReadback(result: Result) {
  return {table: result.table, summary: result.summary, status: Uint32Array.of(result.status)};
}
