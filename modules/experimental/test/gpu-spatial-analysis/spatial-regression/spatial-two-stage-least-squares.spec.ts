// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PRECISION_WARNING,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {
  getGPUSpatialTwoStageLeastSquaresDiagnostics,
  getGPUSpatialTwoStageLeastSquaresModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  SPREG_ASYMMETRIC_KNN_REFERENCE,
  SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE,
  SPREG_TWO_STAGE_ORDER_TWO_REFERENCE,
  SPREG_TWO_STAGE_REFERENCE,
  TWO_STAGE_ILL_CONDITIONED_REFERENCE
} from './spreg-reference';
import {createNearestNeighborScene} from './spatial-regression-diagnostics-oracle';
import {
  createIllConditionedLagScene,
  createLagScene,
  createWeakInstrumentScene,
  fitSpatialTwoStageLeastSquaresOnCPU,
  type TwoStageOracleResult
} from './spatial-two-stage-least-squares-oracle';

type Scene = ReturnType<typeof createLagScene>;
type Result = {status: number; table: Float32Array; summary: Float32Array; residuals: Float32Array};

async function runTwoStage(
  device: Device,
  scene: Scene,
  tileRowCount?: number,
  instrumentOrder?: 1 | 2
): Promise<Result> {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const tableLength = (k + 2) * 4;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'two-stage-graph'});
  const weights = scene.weights;
  const tableBuffer = track(createOutputBuffer(device, tableLength));
  const summaryBuffer = track(
    createOutputBuffer(device, GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH)
  );
  const statusBuffer = track(createOutputBuffer(device, 1));
  const residualBuffer = track(createOutputBuffer(device, rows));
  graph.add(
    new GPUSpatialTwoStageLeastSquares({
      id: 'two-stage',
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
      instrumentOrder,
      output: {
        table: importGraphBuffer(graph, 'table', tableBuffer, 'float32', tableLength),
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
        ),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1),
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
      await readFloat32(summaryBuffer, GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH)
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

function expectParity(actual: Result, expected: TwoStageOracleResult): void {
  expect(actual.status).toBe(0);
  const count = expected.coefficients.length;
  // Oracle order is (intercept, X..., rho), the GPU table order.
  for (let row = 0; row < count; row++) {
    expectClose(
      `coefficient ${row}`,
      actual.table[row * 4],
      expected.coefficients[row],
      5e-3,
      0.05
    );
    expectClose(`se ${row}`, actual.table[row * 4 + 1], expected.standardErrors[row], 1e-2, 1e-3);
    expectClose(`z ${row}`, actual.table[row * 4 + 2], expected.zStatistics[row], 2e-2, 0.2);
    expect(Math.abs(actual.table[row * 4 + 3] - expected.pValues[row])).toBeLessThan(5e-3);
  }
  expectClose('sigma2', actual.summary[1], expected.sigmaSquared, 5e-3, 1e-3);
  expectClose('rss', actual.summary[2], expected.residualSumOfSquares, 5e-3, 1e-3);
  expectClose('pr2', actual.summary[3], expected.pseudoRSquared, 2e-3, 1e-3);
  expectClose('moran I', actual.summary[4], expected.moranI, 1e-2, 1e-3);
  expectClose('anselin-kelejian', actual.summary[5], expected.anselinKelejian, 2e-2, 0.05);
  expect(Math.abs(actual.summary[6] - expected.anselinKelejianPValue)).toBeLessThan(1e-2);
  expect(actual.summary[0]).toBe(expected.residuals.length);
  for (let row = 0; row < expected.residuals.length; row++) {
    expect(Math.abs(actual.residuals[row] - expected.residuals[row])).toBeLessThan(
      0.05 * Math.sqrt(expected.sigmaSquared) + 1e-3
    );
  }
}

it('GPUSpatialTwoStageLeastSquares matches the dense float64 oracle and recovers rho', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createLagScene(12, 21, 0.5, true);
  const expected = fitSpatialTwoStageLeastSquaresOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const actual = await runTwoStage(device, scene);
  expectParity(actual, expected);
  // The generating rho is 0.5 and the instruments are valid: the estimate is in its neighborhood.
  expect(actual.table[3 * 4]).toBeGreaterThan(0.2);
  expect(actual.table[3 * 4]).toBeLessThan(0.8);
  // The slope of x1 is 2 and strongly significant.
  expect(Math.abs(actual.table[4 + 2])).toBeGreaterThan(5);
  for (const tileRowCount of [5, 144]) {
    expectParity(await runTwoStage(device, scene, tileRowCount), expected);
  }
});

it('GPUSpatialTwoStageLeastSquares matches spreg GM_Lag including Anselin-Kelejian', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const reference = SPREG_TWO_STAGE_REFERENCE;
  const actual = await runTwoStage(device, createLagScene(12, 21, 0.5, true));
  expect(actual.status).toBe(0);
  reference.betas.forEach((beta, row) => {
    expectClose(`spreg beta ${row}`, actual.table[row * 4], beta, 5e-3, 0.05);
    expectClose(
      `spreg se ${row}`,
      actual.table[row * 4 + 1],
      reference.standardErrors[row],
      1e-2,
      1e-3
    );
    expectClose(`spreg z ${row}`, actual.table[row * 4 + 2], reference.z[row], 2e-2, 0.2);
  });
  expectClose('spreg sigma2', actual.summary[1], reference.sigma2, 5e-3, 1e-3);
  expectClose('spreg pr2', actual.summary[3], reference.pseudoRSquared, 2e-3, 1e-3);
  expectClose('spreg AK', actual.summary[5], reference.anselinKelejian[0], 2e-2, 0.05);
  expect(Math.abs(actual.summary[6] - reference.anselinKelejian[1])).toBeLessThan(1e-2);
  // The statistic must be a real computation, not a failed-compile zero.
  expect(actual.summary[5]).toBeGreaterThan(0.5);
});

it('GPUSpatialTwoStageLeastSquares matches the oracle on binary weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createLagScene(10, 8, 0.12, false);
  const expected = fitSpatialTwoStageLeastSquaresOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  expectParity(await runTwoStage(device, scene), expected);
});

it('GPUSpatialTwoStageLeastSquares matches spreg GM_Lag on asymmetric directed kNN weights (k = 3)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [count, seed, reference] of [
    [60, 13, SPREG_ASYMMETRIC_KNN_REFERENCE.scene_60_13.twoStage],
    [90, 29, SPREG_ASYMMETRIC_KNN_REFERENCE.scene_90_29.twoStage]
  ] as const) {
    const scene = createNearestNeighborScene(count, 3, seed, 0.6, false);
    const actual = await runTwoStage(device, scene);
    expect(actual.status).toBe(0);
    reference.betas.forEach((beta, row) => {
      expectClose(`knn ${count} beta ${row}`, actual.table[row * 4], beta, 5e-3, 0.05);
      expectClose(
        `knn ${count} se ${row}`,
        actual.table[row * 4 + 1],
        reference.standardErrors[row],
        1e-2,
        1e-3
      );
      expectClose(`knn ${count} z ${row}`, actual.table[row * 4 + 2], reference.z[row], 2e-2, 0.2);
    });
    expectClose(`knn ${count} sigma2`, actual.summary[1], reference.sigma2, 5e-3, 1e-3);
    expectClose(`knn ${count} pr2`, actual.summary[3], reference.pseudoRSquared, 2e-3, 1e-3);
    expectClose(`knn ${count} AK`, actual.summary[5], reference.anselinKelejian[0], 2e-2, 0.05);
    expect(Math.abs(actual.summary[6] - reference.anselinKelejian[1])).toBeLessThan(1e-2);
    expect(actual.summary[5]).toBeGreaterThan(0.5);
  }
});

it('GPUSpatialTwoStageLeastSquares instrumentOrder 2 matches spreg GM_Lag(w_lags=2) and the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scenes = [
    [createLagScene(12, 21, 0.5, true), SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.lattice],
    [
      createNearestNeighborScene(60, 3, 13, 0.6, false),
      SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.scene_60_13
    ],
    [
      createNearestNeighborScene(90, 3, 29, 0.6, false),
      SPREG_TWO_STAGE_ORDER_TWO_REFERENCE.scene_90_29
    ]
  ] as const;
  for (const [scene, reference] of scenes) {
    const actual = await runTwoStage(device, scene, undefined, 2);
    expect(actual.status).toBe(0);
    reference.betas.forEach((beta, row) => {
      expectClose(`order 2 beta ${row}`, actual.table[row * 4], beta, 5e-3, 0.05);
      expectClose(
        `order 2 se ${row}`,
        actual.table[row * 4 + 1],
        reference.standardErrors[row],
        1e-2,
        1e-3
      );
    });
    expectClose('order 2 sigma2', actual.summary[1], reference.sigma2, 5e-3, 1e-3);
    expectClose('order 2 pr2', actual.summary[3], reference.pseudoRSquared, 2e-3, 1e-3);
    expectClose('order 2 AK', actual.summary[5], reference.anselinKelejian[0], 2e-2, 0.05);
  }
  const lattice = createLagScene(12, 21, 0.5, true);
  expectParity(
    await runTwoStage(device, lattice, 7, 2),
    fitSpatialTwoStageLeastSquaresOnCPU(
      lattice.weights,
      lattice.predictors,
      lattice.response,
      lattice.predictorCount,
      2
    )
  );
});

// ---- Diagnostics: conditioning, first-stage strength, rho admissibility --------------------------
// Pins: numpy float64 definitions (SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE, generated by
// fix/work/C/U2/pin-two-stage-diagnostics.py). Tolerances are chosen from the float32 error budget
// of the Jacobi eigenvalues and the first-stage sums, not measured: loosen only after a run.

const DIAGNOSTIC_SCENES = {
  lag_12_21: () => createLagScene(12, 21, 0.5, true),
  lag_binary_10_5_high: () => createLagScene(10, 5, 0.255, false),
  weak_80_7: () => createWeakInstrumentScene(80, 7),
  knn_60_13: () => createNearestNeighborScene(60, 3, 13, 0.6, false)
} as const;

for (const name of Object.keys(DIAGNOSTIC_SCENES) as (keyof typeof DIAGNOSTIC_SCENES)[]) {
  for (const order of [1, 2] as const) {
    it(`GPUSpatialTwoStageLeastSquares diagnostics match the numpy pins on ${name}, order ${order}`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) {
        return;
      }
      const reference = SPREG_TWO_STAGE_DIAGNOSTICS_REFERENCE[name];
      const fixture = order === 1 ? reference.order1 : reference.order2;
      const actual = await runTwoStage(device, DIAGNOSTIC_SCENES[name](), undefined, order);
      expect(actual.status).toBe(0);
      const summary = actual.summary;
      expectClose(
        'norm',
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_WEIGHTS_INFINITY_NORM],
        reference.weightsInfinityNorm,
        1e-5,
        1e-3
      );
      expectClose(
        'first-stage F',
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F],
        fixture.firstStageF,
        5e-2,
        0.2
      );
      expectClose(
        'partial R2',
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_PARTIAL_R_SQUARED],
        fixture.firstStagePartialRSquared,
        2e-2,
        0.02
      );
      expectClose(
        'condition number',
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER],
        fixture.conditionNumber,
        5e-2,
        1
      );
      expectClose(
        'correlation condition',
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER],
        fixture.correlationConditionNumber,
        5e-2,
        1
      );
      expectClose(
        'VIF',
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR],
        fixture.maximumVarianceInflationFactor,
        5e-2,
        1
      );
      expectClose(
        'instrument condition',
        summary[
          GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_INSTRUMENT_CORRELATION_CONDITION_NUMBER
        ],
        fixture.instrumentCorrelationConditionNumber,
        5e-2,
        1
      );
      expect(summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PRECISION_WARNING]).toBe(0);
      // The rho flag follows |rho| ||W||_inf >= 1; the fixture rho comes from spreg GM_Lag.
      expect(
        summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_RHO_OUTSIDE_ADMISSIBLE_RANGE] === 1,
        `rho ${actual.table[3 * 4]}`
      ).toBe(fixture.rhoOutsideAdmissibleRange);
      // A flagged rho is still reported, not failed.
      expect(Number.isFinite(actual.table[3 * 4])).toBe(true);
      const diagnostics = getGPUSpatialTwoStageLeastSquaresDiagnostics(summary);
      expect(diagnostics.rhoOutsideAdmissibleRange).toBe(fixture.rhoOutsideAdmissibleRange);
      const result = getGPUSpatialTwoStageLeastSquaresModelResult({
        table: actual.table,
        summary,
        status: Uint32Array.of(actual.status)
      });
      expect(result.coefficients[3].role).toBe('spatial-lag');
      expect(result.status.ok).toBe(true);
    });
  }
}

it('GPUSpatialTwoStageLeastSquares separates a weak first stage from a strong one', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const weak = await runTwoStage(device, createWeakInstrumentScene(80, 7));
  const strong = await runTwoStage(device, createNearestNeighborScene(60, 3, 13, 0.6, false));
  expect(weak.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F]).toBeLessThan(10);
  expect(strong.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F]).toBeGreaterThan(
    100
  );
});

it('GPUSpatialTwoStageLeastSquares fits 8 predictors with order 1 and 5 with order 2', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [predictorCount, order] of [
    [8, 1],
    [5, 2]
  ] as const) {
    const scene = createLagScene(14, 5, 0.4, true, predictorCount);
    const expected = fitSpatialTwoStageLeastSquaresOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount,
      order
    );
    const actual = await runTwoStage(device, scene, undefined, order);
    expect(actual.status, `k ${predictorCount} order ${order}`).toBe(0);
    for (let row = 0; row < predictorCount + 2; row++) {
      expectClose(
        `k ${predictorCount} coefficient ${row}`,
        actual.table[row * 4],
        expected.coefficients[row],
        2e-2,
        0.1
      );
      expectClose(
        `k ${predictorCount} se ${row}`,
        actual.table[row * 4 + 1],
        expected.standardErrors[row],
        5e-2,
        1e-3
      );
    }
    expectClose(
      'first-stage F',
      actual.summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_FIRST_STAGE_F],
      expected.diagnostics.firstStageF,
      0.1,
      1
    );
  }
});

// Belsley condition number in the thousands. Pin: np.linalg.cond of the unit-column [1, P Z] design
// (fix/work/C/U2/pin-ill-conditioned.py). The raw Gram route cannot resolve this in float32
// (lambdaMin / lambdaMax ~ 1.5e-8 is below float32 epsilon); the Jacobi route on the correlation
// Cholesky factor has relative error about epsilon * cond ~ 1e-3, so rtol 1e-2 leaves a factor 10.
for (const order of [1, 2] as const) {
  it(`GPUSpatialTwoStageLeastSquares Belsley condition number stays accurate near 8000, order ${order}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const reference =
      order === 1
        ? TWO_STAGE_ILL_CONDITIONED_REFERENCE.order1
        : TWO_STAGE_ILL_CONDITIONED_REFERENCE.order2;
    const actual = await runTwoStage(
      device,
      createIllConditionedLagScene(12, 21, 0.5),
      undefined,
      order
    );
    expect(actual.status).toBe(0);
    const summary = actual.summary;
    expectClose(
      'condition number',
      summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER],
      reference.conditionNumber,
      1e-2,
      0
    );
    expectClose(
      'correlation condition',
      summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_CORRELATION_CONDITION_NUMBER],
      reference.correlationConditionNumber,
      5e-2,
      1
    );
    expect(summary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PRECISION_WARNING]).toBe(0);
  });
}
