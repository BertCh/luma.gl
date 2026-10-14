// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {
  GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION,
  GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_AT_BOUND,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_ERROR_GM_SUMMARY_PRECISION_WARNING,
  type GPUSpatialErrorGMEstimator
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-parameters';
import {getGPUSpatialErrorGMModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-model-result';
import type {GPUSpatialWeightsTracesMode} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {SPREG_ERROR_GM_HOM_REFERENCE, SPREG_ERROR_GM_REFERENCE} from './spatial-error-gm-reference';
import {
  createErrorScene,
  fitSpatialErrorGMHomOnCPU,
  fitSpatialErrorGMOnCPU,
  type ErrorGMHomOracleResult,
  type ErrorGMOracleResult
} from './spatial-error-gm-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Scene = ReturnType<typeof createErrorScene>;
type Result = {status: number; table: Float32Array; summary: Float32Array; residuals: Float32Array};

type RunOptions = {
  tileRowCount?: number;
  estimator?: GPUSpatialErrorGMEstimator;
  traceMode?: GPUSpatialWeightsTracesMode;
  traceProbeCount?: number;
};

async function runErrorGM(
  device: Device,
  scene: Scene,
  {tileRowCount, estimator, traceMode, traceProbeCount}: RunOptions = {}
): Promise<Result> {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const tableLength = (k + 2) * 4;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'error-gm-graph'});
  const weights = scene.weights;
  const tableBuffer = track(createOutputBuffer(device, tableLength));
  const summaryBuffer = track(createOutputBuffer(device, GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH));
  const statusBuffer = track(createOutputBuffer(device, 1));
  const residualBuffer = track(createOutputBuffer(device, rows));
  graph.add(
    new GPUSpatialErrorGM({
      id: 'error-gm',
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
      tileRowCount,
      estimator,
      traceMode,
      traceProbeCount,
      output: {
        table: importGraphBuffer(graph, 'table', tableBuffer, 'float32', tableLength),
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH
        ),
        status: {code: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1)},
        residuals: importGraphBuffer(graph, 'residuals', residualBuffer, 'float32', rows)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [status] = await readUint32(statusBuffer, 1);
  const result = {
    status,
    table: Float32Array.from(await readFloat32(tableBuffer, tableLength)),
    summary: Float32Array.from(
      await readFloat32(summaryBuffer, GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH)
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

// Relative bounds are 10 to 100 times the worst error measured against the float64 oracle and
// spreg (about 7e-6 for coefficients, 1e-6 for standard errors and variances, 1e-5 for z).
function expectParity(actual: Result, expected: ErrorGMOracleResult): void {
  expect(actual.status).toBe(0);
  const count = expected.coefficients.length;
  // Oracle order is (intercept, X...); the GPU table adds a lambda row last.
  for (let row = 0; row < count; row++) {
    expectClose(
      `coefficient ${row}`,
      actual.table[row * 4],
      expected.coefficients[row],
      2e-4,
      0.05
    );
    expectClose(`se ${row}`, actual.table[row * 4 + 1], expected.standardErrors[row], 5e-5, 1e-3);
    expectClose(`z ${row}`, actual.table[row * 4 + 2], expected.zStatistics[row], 5e-4, 0.2);
    expect(Math.abs(actual.table[row * 4 + 3] - expected.pValues[row])).toBeLessThan(5e-3);
  }
  expect(Math.abs(actual.table[count * 4] - expected.lambda)).toBeLessThan(5e-3);
  expect(Number.isNaN(actual.table[count * 4 + 1])).toBe(true);
  expect(Math.abs(actual.summary[1] - expected.lambda)).toBeLessThan(5e-3);
  expectClose('sigma2', actual.summary[2], expected.sigmaSquared, 2e-5, 1e-3);
  expectClose('pr2', actual.summary[3], expected.pseudoRSquared, 1e-5, 1e-3);
  expectClose('rss', actual.summary[4], expected.residualSumOfSquares, 2e-5, 1e-3);
  expect(actual.summary[0]).toBe(expected.residuals.length);
  for (let row = 0; row < expected.residuals.length; row++) {
    expect(Math.abs(actual.residuals[row] - expected.residuals[row])).toBeLessThan(
      0.05 * Math.sqrt(expected.sigmaSquared) + 1e-3
    );
  }
}

for (const [kind, count, seed, lambda] of [
  ['lattice-row-standardized', 400, 31, 0.6],
  ['knn', 300, 17, 0.5],
  ['lattice-binary', 324, 9, 0.5]
] as const) {
  it(`GPUSpatialErrorGM matches the dense float64 oracle on ${kind} weights`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = createErrorScene(count, seed, lambda, kind);
    const expected = fitSpatialErrorGMOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    const actual = await runErrorGM(device, scene, {estimator: 'kelejian-prucha'});
    expectParity(actual, expected);
    // The moment solve must be a real computation, not a failed-compile zero.
    expect(Math.abs(actual.summary[1])).toBeGreaterThan(0.02);
    expect(actual.summary[2]).toBeGreaterThan(0.1);
    if (kind === 'lattice-row-standardized') {
      expect(actual.summary[1]).toBeGreaterThan(0.25);
      expect(actual.summary[1]).toBeLessThan(0.95);
      for (const tileRowCount of [7, 400]) {
        expectParity(
          await runErrorGM(device, scene, {tileRowCount, estimator: 'kelejian-prucha'}),
          expected
        );
      }
    }
  });
}

it('GPUSpatialErrorGM reports a singular design', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createErrorScene(100, 3, 0.5, 'lattice-row-standardized');
  for (let row = 0; row < scene.response.length; row++) {
    scene.predictors[row * 2 + 1] = 2 * scene.predictors[row * 2];
  }
  for (const estimator of ['kelejian-prucha', 'homoskedastic'] as const) {
    const actual = await runErrorGM(device, scene, {estimator});
    expect(actual.status, estimator).toBe(1);
    expect(Number.isNaN(actual.table[0]), estimator).toBe(true);
  }
});

for (const name of ['lattice', 'knn'] as const) {
  it(`GPUSpatialErrorGM matches spreg GM_Error on the ${name} fixture`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const reference = SPREG_ERROR_GM_REFERENCE[name];
    const actual = await runErrorGM(
      device,
      {
        weights: {
          offsets: Uint32Array.from(reference.offsets),
          neighbors: Uint32Array.from(reference.neighbors),
          weights: Float32Array.from(reference.weights)
        },
        predictors: Float32Array.from(reference.predictors),
        response: Float32Array.from(reference.response),
        predictorCount: 2
      },
      {estimator: 'kelejian-prucha'}
    );
    expect(actual.status).toBe(0);
    expect(Math.abs(actual.table[3 * 4] - reference.lambda)).toBeLessThan(5e-3);
    reference.betas.forEach((beta, row) => {
      expectClose(`spreg beta ${row}`, actual.table[row * 4], beta, 5e-4, 0.05);
      expectClose(
        `spreg se ${row}`,
        actual.table[row * 4 + 1],
        reference.standardErrors[row],
        5e-5,
        1e-3
      );
      expectClose(
        `spreg z ${row}`,
        actual.table[row * 4 + 2],
        reference.zStatistics[row],
        5e-4,
        0.2
      );
    });
    expectClose('spreg sigma2', actual.summary[2], reference.sigma2, 2e-5, 1e-3);
    expectClose('spreg pr2', actual.summary[3], reference.pseudoRSquared, 1e-5, 1e-3);
    expect(actual.summary[1]).toBeGreaterThan(0.3);
  });
}

// ---- GM_Error_Hom (the default estimator) ------------------------------------------------------
// Tolerances are chosen from the float32 error budget (psi enters through a 2 x 2 inverse and the
// weighted minimization), not measured: loosen only after inspecting a run.

function expectHomParity(actual: Result, expected: ErrorGMHomOracleResult): void {
  expect(actual.status).toBe(0);
  const count = expected.coefficients.length;
  for (let row = 0; row < count; row++) {
    expectClose(`beta ${row}`, actual.table[row * 4], expected.coefficients[row], 5e-4, 0.05);
    expectClose(`se ${row}`, actual.table[row * 4 + 1], expected.standardErrors[row], 5e-3, 1e-3);
    expectClose(`z ${row}`, actual.table[row * 4 + 2], expected.zStatistics[row], 5e-3, 0.2);
  }
  expect(Math.abs(actual.table[count * 4] - expected.lambda)).toBeLessThan(5e-3);
  expectClose('lambda se', actual.table[count * 4 + 1], expected.lambdaStandardError, 2e-2, 1e-3);
  expectClose('lambda z', actual.table[count * 4 + 2], expected.lambdaZStatistic, 2e-2, 0.2);
  expect(Math.abs(actual.table[count * 4 + 3] - expected.lambdaPValue)).toBeLessThan(5e-3);
  expect(Math.abs(actual.summary[1] - expected.lambda)).toBeLessThan(5e-3);
  expectClose('sigma2', actual.summary[2], expected.sigmaSquared, 5e-3, 1e-3);
  expectClose('pr2', actual.summary[3], expected.pseudoRSquared, 1e-4, 1e-3);
  expectClose('rss', actual.summary[4], expected.residualSumOfSquares, 1e-4, 1e-3);
  expectClose(
    'bound',
    actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND],
    expected.lambdaBound,
    1e-5,
    1e-3
  );
}

for (const [kind, count, seed, lambda] of [
  ['lattice-row-standardized', 400, 31, 0.6],
  ['knn', 300, 17, 0.5],
  ['lattice-binary', 324, 9, 0.5]
] as const) {
  it(`GPUSpatialErrorGM homoskedastic matches the float64 GM_Error_Hom oracle on ${kind} weights`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = createErrorScene(count, seed, lambda, kind);
    const expected = fitSpatialErrorGMHomOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    // The default estimator is homoskedastic.
    const actual = await runErrorGM(device, scene);
    expectHomParity(actual, expected);
    expect(actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION]).toBe(0);
    expect(actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_AT_BOUND]).toBe(0);
    // The standard error of lambda must be a real computation, not a failed-compile NaN or zero.
    expect(actual.table[3 * 4 + 1]).toBeGreaterThan(0.005);
    expect(actual.table[3 * 4 + 1]).toBeLessThan(0.5);
    if (kind === 'lattice-row-standardized') {
      for (const tileRowCount of [7, 400]) {
        expectHomParity(await runErrorGM(device, scene, {tileRowCount}), expected);
      }
    }
  });
}

for (const name of ['lattice', 'knn', 'binary'] as const) {
  it(`GPUSpatialErrorGM homoskedastic matches spreg GM_Error_Hom and its conditioning on ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const reference = SPREG_ERROR_GM_HOM_REFERENCE[name];
    const scene =
      name === 'binary'
        ? createErrorScene(324, 9, 0.5, 'lattice-binary')
        : {
            weights: {
              offsets: Uint32Array.from(SPREG_ERROR_GM_REFERENCE[name].offsets),
              neighbors: Uint32Array.from(SPREG_ERROR_GM_REFERENCE[name].neighbors),
              weights: Float32Array.from(SPREG_ERROR_GM_REFERENCE[name].weights)
            },
            predictors: Float32Array.from(SPREG_ERROR_GM_REFERENCE[name].predictors),
            response: Float32Array.from(SPREG_ERROR_GM_REFERENCE[name].response),
            predictorCount: 2
          };
    const actual = await runErrorGM(device, scene, {estimator: 'homoskedastic'});
    expect(actual.status).toBe(0);
    const hom = reference.hom;
    const count = hom.betas.length;
    hom.betas.forEach((beta, row) => {
      expectClose(`spreg beta ${row}`, actual.table[row * 4], beta, 5e-4, 0.05);
      expectClose(
        `spreg se ${row}`,
        actual.table[row * 4 + 1],
        hom.standardErrors[row],
        5e-3,
        1e-3
      );
      expectClose(`spreg z ${row}`, actual.table[row * 4 + 2], hom.z[row], 5e-3, 0.2);
    });
    // spreg minimizes with a local L-BFGS-B, hence the absolute lambda tolerance.
    expect(Math.abs(actual.table[count * 4] - hom.lambda)).toBeLessThan(5e-3);
    expectClose(
      'spreg lambda se',
      actual.table[count * 4 + 1],
      hom.standardErrors[count],
      3e-2,
      1e-3
    );
    expectClose('spreg lambda z', actual.table[count * 4 + 2], hom.z[count], 3e-2, 0.2);
    expect(Math.abs(actual.table[count * 4 + 3] - hom.p[count])).toBeLessThan(5e-3);
    expectClose('spreg sigma2', actual.summary[2], hom.sigma2, 5e-3, 1e-3);
    expectClose('spreg pr2', actual.summary[3], hom.pseudoRSquared, 1e-4, 1e-3);
    expectClose(
      'bound',
      actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND],
      0.99 / reference.rowSumNorm,
      1e-5,
      1e-3
    );
    // Conditioning of the filtered regression at the reported lambda (numpy pins, f32 Jacobi).
    expectClose(
      'condition number',
      actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER],
      hom.conditioning.conditionNumber,
      5e-2,
      1
    );
    expectClose(
      'correlation condition',
      actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER],
      hom.conditioning.correlationConditionNumber,
      5e-2,
      1
    );
    expectClose(
      'vif',
      actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR],
      hom.conditioning.maximumVarianceInflationFactor,
      5e-2,
      1
    );
    expect(actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_PRECISION_WARNING]).toBe(0);
    // The model-result reader consumes the read-back arrays.
    const result = getGPUSpatialErrorGMModelResult({
      table: actual.table,
      summary: actual.summary,
      status: Uint32Array.of(actual.status)
    });
    expect(result.coefficients[count].role).toBe('spatial-error');
    expect(result.coefficients[count].standardError).toBeGreaterThan(0);
    expect(result.status.ok).toBe(true);
  });
}

it('GPUSpatialErrorGM kelejian-prucha keeps NaN for the lambda statistics and reports conditioning', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const reference = SPREG_ERROR_GM_HOM_REFERENCE.binary;
  const scene = createErrorScene(324, 9, 0.5, 'lattice-binary');
  const actual = await runErrorGM(device, scene, {estimator: 'kelejian-prucha'});
  expect(actual.status).toBe(0);
  // Non-row-standardized weights: spreg GM_Error pins, lambda bound 0.99 / 4.
  expect(Math.abs(actual.table[3 * 4] - reference.kelejianPrucha.lambda)).toBeLessThan(5e-3);
  expect(Number.isNaN(actual.table[3 * 4 + 1])).toBe(true);
  expect(Number.isNaN(actual.table[3 * 4 + 2])).toBe(true);
  expect(Number.isNaN(actual.table[3 * 4 + 3])).toBe(true);
  expectClose('kp sigma2', actual.summary[2], reference.kelejianPrucha.sigma2, 5e-3, 1e-3);
  expectClose(
    'bound',
    actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND],
    0.2475,
    1e-5,
    1e-3
  );
  expectClose(
    'condition number',
    actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER],
    reference.kelejianPrucha.conditioning.conditionNumber,
    5e-2,
    1
  );
  expect(actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION]).toBe(0);
});

it('GPUSpatialErrorGM flags a lambda at the search bound without failing the fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Strong dependence on binary weights: lambda / 4 = 0.24 sits next to the 0.2475 bound, and
  // the bound (not the spreg 0.99) is what keeps I - lambda W invertible for ||W||_inf = 4.
  const scene = createErrorScene(324, 9, 0.99, 'lattice-binary');
  const actual = await runErrorGM(device, scene, {estimator: 'kelejian-prucha'});
  expect(actual.status).toBe(0);
  expect(Math.abs(actual.summary[1])).toBeLessThanOrEqual(0.2475 * 1.0001);
  expectClose(
    'bound',
    actual.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND],
    0.2475,
    1e-5,
    1e-3
  );
});

it('GPUSpatialErrorGM with stochastic traces marks the lambda standard error as approximate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createErrorScene(400, 31, 0.6, 'lattice-row-standardized');
  const exact = await runErrorGM(device, scene, {traceMode: 'exact'});
  const stochastic = await runErrorGM(device, scene, {
    traceMode: 'stochastic',
    traceProbeCount: 256
  });
  expect(exact.status).toBe(0);
  expect(stochastic.status).toBe(0);
  expect(exact.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION]).toBe(0);
  expect(stochastic.summary[GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION]).toBe(1);
  // Betas come from the first-step lambda and are unaffected by the traces. The lambda and its
  // standard error use the Hutchinson estimates: bounds are generous (error ~ sqrt(2 / 256) ||A||_F).
  for (let row = 0; row < 3; row++) {
    expect(stochastic.table[row * 4]).toBe(exact.table[row * 4]);
  }
  expect(Math.abs(stochastic.table[3 * 4] - exact.table[3 * 4])).toBeLessThan(0.05);
  expectClose(
    'stochastic lambda se',
    stochastic.table[3 * 4 + 1],
    exact.table[3 * 4 + 1],
    0.3,
    1e-3
  );
});

it('GPUSpatialErrorGM fits 15 predictors with both estimators', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createErrorScene(484, 7, 0.5, 'lattice-row-standardized', 15);
  const expectedHom = fitSpatialErrorGMHomOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const hom = await runErrorGM(device, scene);
  expect(hom.status).toBe(0);
  for (let row = 0; row < 16; row++) {
    expectClose(`beta ${row}`, hom.table[row * 4], expectedHom.coefficients[row], 5e-3, 0.1);
    expectClose(`se ${row}`, hom.table[row * 4 + 1], expectedHom.standardErrors[row], 2e-2, 1e-3);
  }
  expect(Math.abs(hom.table[16 * 4] - expectedHom.lambda)).toBeLessThan(2e-2);
  const expectedKp = fitSpatialErrorGMOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const kp = await runErrorGM(device, scene, {estimator: 'kelejian-prucha'});
  expect(kp.status).toBe(0);
  for (let row = 0; row < 16; row++) {
    expectClose(`kp beta ${row}`, kp.table[row * 4], expectedKp.coefficients[row], 5e-3, 0.1);
  }
});
