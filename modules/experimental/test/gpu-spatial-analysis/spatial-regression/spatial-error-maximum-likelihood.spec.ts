// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {fitSpatialErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-cpu';
import {
  getGPUSpatialErrorMaximumLikelihoodModelResult,
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
  SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE,
  type SpatialMaximumLikelihoodFixture
} from './spatial-maximum-likelihood-reference';

type Scene = ReturnType<(typeof SPATIAL_MAXIMUM_LIKELIHOOD_SCENES)['err_knn_400_31']>;
type Result = {
  status: number;
  converged: number;
  table: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
};

type RunOptions = {
  tileRowCount?: number;
  maximumPower?: number;
  traceMode?: GPUSpatialWeightsTracesMode;
  traceProbeCount?: number;
};

async function runErrorMaximumLikelihood(
  device: Device,
  scene: Scene,
  {tileRowCount, maximumPower = 128, traceMode, traceProbeCount}: RunOptions = {}
): Promise<Result> {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const tableLength = (k + 2) * 4;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'error-ml-graph'});
  const weights = scene.weights;
  const tableBuffer = track(createOutputBuffer(device, tableLength));
  const summaryBuffer = track(
    createOutputBuffer(device, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
  );
  const statusBuffer = track(createOutputBuffer(device, 1));
  const convergedBuffer = track(createOutputBuffer(device, 1));
  const residualBuffer = track(createOutputBuffer(device, rows));
  graph.add(
    new GPUSpatialErrorMaximumLikelihood({
      id: 'error-ml',
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
        table: importGraphBuffer(graph, 'table', tableBuffer, 'float32', tableLength),
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH
        ),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1),
        converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1),
        residuals: importGraphBuffer(graph, 'residuals', residualBuffer, 'float32', rows)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    status: (await readUint32(statusBuffer, 1))[0],
    converged: (await readUint32(convergedBuffer, 1))[0],
    table: Float32Array.from(await readFloat32(tableBuffer, tableLength)),
    summary: Float32Array.from(
      await readFloat32(summaryBuffer, GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH)
    ),
    residuals: Float32Array.from(await readFloat32(residualBuffer, rows))
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
// tighten them). The filtered Gram matrices are f32 tile sums merged with Kahan compensation
// (1e-6 relative); each grid point solves a unit-scaled 3 x 3 to 16 x 16 system, so `ee` and its
// derivative carry ~1e-5 relative error and `lambda` (a root of a ratio of them) well under 1e-3;
// the standard errors add the tr3 probe sums. Bounds follow the error-GM spec (5e-3 for lambda,
// 2e-4 coefficients, 5e-5 standard errors) widened for the extra solve.
const PARAMETER_TOLERANCE = 2e-3;
const COEFFICIENT_TOLERANCE = 2e-3;
const STANDARD_ERROR_TOLERANCE = 1e-2;
const LOG_LIKELIHOOD_TOLERANCE = 0.05;

function expectParity(actual: Result, fixture: SpatialMaximumLikelihoodFixture, k: number) {
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
  expectClose('sigma2', actual.summary[2], fixture.sigma2, 2e-3, 1e-3);
  expect(Math.abs(actual.summary[3] - fixture.logLikelihood)).toBeLessThan(
    LOG_LIKELIHOOD_TOLERANCE
  );
  expect(Math.abs(actual.summary[4] - fixture.aic)).toBeLessThan(2 * LOG_LIKELIHOOD_TOLERANCE);
  expect(Math.abs(actual.summary[5] - fixture.schwarz)).toBeLessThan(2 * LOG_LIKELIHOOD_TOLERANCE);
  expectClose('pr2', actual.summary[6], fixture.pseudoRSquared, 1e-3, 1e-3);
  expect(Number.isNaN(actual.summary[7])).toBe(true);
  expectClose('condition number', actual.summary[9], fixture.conditioning.conditionNumber, 5e-2, 1);
  expectClose(
    'correlation condition number',
    actual.summary[10],
    fixture.conditioning.correlationConditionNumber,
    5e-2,
    1
  );
  expectClose(
    'vif',
    actual.summary[11],
    fixture.conditioning.maximumVarianceInflationFactor,
    5e-2,
    1
  );
  expect(actual.summary[14]).toBe(0);
  expect(actual.summary[16]).toBe(0);
  expect(actual.summary[19]).toBe(1);
  expect(actual.summary[20]).toBe(1);
  expect(actual.summary[18]).toBeLessThan(1e-3);
}

for (const name of Object.keys(SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE) as Array<
  keyof typeof SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE
>) {
  it(`GPUSpatialErrorMaximumLikelihood matches spreg ML_Error on ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES[name]();
    const fixture = SPREG_ERROR_MAXIMUM_LIKELIHOOD_REFERENCE[name];
    const actual = await runErrorMaximumLikelihood(device, scene);
    expectParity(actual, fixture, scene.predictorCount);
    expect(Math.abs(actual.summary[1])).toBeGreaterThan(0.02);
    // Residuals are the unfiltered `u = y - X b`: their sum of squares is the summary slot.
    let sum = 0;
    for (const value of actual.residuals) {
      sum += value * value;
    }
    expectClose('residual sum of squares', sum, actual.summary[8], 1e-3, 1e-3);
    // The model result reproduces spreg's AIC and Schwarz, which do not count lambda.
    const result = getGPUSpatialErrorMaximumLikelihoodModelResult(toReadback(actual));
    expect(Math.abs((result.akaikeInformationCriterion ?? NaN) - fixture.aic)).toBeLessThan(0.1);
    expect(Math.abs((result.schwarzCriterion ?? NaN) - fixture.schwarz)).toBeLessThan(0.1);
    expect(result.status.ok).toBe(true);
    if (name === 'err_knn_400_31') {
      for (const tileRowCount of [7, 400]) {
        expectParity(
          await runErrorMaximumLikelihood(device, scene, {tileRowCount}),
          fixture,
          scene.predictorCount
        );
      }
    }
  });
}

it('GPUSpatialErrorMaximumLikelihood is deterministic', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.err_lattice_400_31();
  const first = await runErrorMaximumLikelihood(device, scene);
  const second = await runErrorMaximumLikelihood(device, scene);
  expect(Array.from(second.table)).toEqual(Array.from(first.table));
  expect(Array.from(second.summary)).toEqual(Array.from(first.summary));
});

it('GPUSpatialErrorMaximumLikelihood agrees with the float64 mirror under Hutchinson traces', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.err_knn_400_31();
  const actual = await runErrorMaximumLikelihood(device, scene, {
    traceMode: 'stochastic',
    traceProbeCount: 256
  });
  const expected = fitSpatialErrorMaximumLikelihoodOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount,
    {traceMode: 'stochastic', probeCount: 256}
  );
  expect(actual.status).toBe(0);
  expect(actual.summary[14]).toBe(1);
  expect(Math.abs(actual.summary[1] - (expected?.lambda ?? NaN))).toBeLessThan(PARAMETER_TOLERANCE);
  expectClose(
    'se lambda',
    actual.table[3 * 4 + 1],
    expected?.standardErrors[3] ?? NaN,
    STANDARD_ERROR_TOLERANCE,
    1e-3
  );
});

it('GPUSpatialErrorMaximumLikelihood reports a boundary solution when the series limits the interval', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.err_lattice_400_31();
  // Q = 8 limits |lambda| to about 0.34, below the unrestricted maximum 0.60.
  const actual = await runErrorMaximumLikelihood(device, scene, {maximumPower: 8});
  const bound = actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND];
  expect(actual.status).toBe(0);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED]).toBe(1);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY]).toBe(1);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED]).toBe(0);
  expect(actual.converged).toBe(0);
  expectClose('bound', bound, getSpatialMaximumLikelihoodSeriesLimit(8), 1e-5, 1e-3);
  expect(actual.summary[GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER]).toBeCloseTo(bound, 5);
  expect(getGPUSpatialMaximumLikelihoodDiagnostics(actual.summary).atBoundary).toBe(true);
});

it('GPUSpatialErrorMaximumLikelihood reports singular, too-small and non-finite inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = SPATIAL_MAXIMUM_LIKELIHOOD_SCENES.err_lattice_400_31();
  const singular = {...scene, predictors: Float32Array.from(scene.predictors)};
  for (let row = 0; row < scene.response.length; row++) {
    singular.predictors[row * 2 + 1] = 2 * singular.predictors[row * 2];
  }
  const collinear = await runErrorMaximumLikelihood(device, singular);
  expect(collinear.status).toBe(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_SINGULAR);
  expect(Number.isNaN(collinear.table[0])).toBe(true);
  expect(collinear.summary[0]).toBe(400);

  const small = {
    weights: createLatticeWeights(2, true),
    predictors: Float32Array.of(1, 5, 2, 3, 4, 1, 3, 2),
    response: Float32Array.of(1, 2, 4, 3),
    predictorCount: 2
  };
  const tooFew = await runErrorMaximumLikelihood(device, small);
  expect(tooFew.status).toBe(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_TOO_FEW_ROWS);
  expect(Number.isNaN(tooFew.table[0])).toBe(true);

  const missing = {...scene, response: Float32Array.from(scene.response)};
  missing.response[5] = Number.NaN;
  const nonFinite = await runErrorMaximumLikelihood(device, missing);
  expect(nonFinite.status).toBe(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_STATUS_NON_FINITE);
  expect(Number.isNaN(nonFinite.table[0])).toBe(true);
  expect(getGPUSpatialErrorMaximumLikelihoodModelResult(toReadback(nonFinite)).status.ok).toBe(
    false
  );
});

/** The model-result readers take the status as a one-element array, as read back from the GPU. */
function toReadback(result: Result) {
  return {table: result.table, summary: result.summary, status: Uint32Array.of(result.status)};
}
