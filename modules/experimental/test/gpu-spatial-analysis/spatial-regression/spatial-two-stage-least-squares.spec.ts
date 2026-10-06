// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {SPREG_ASYMMETRIC_KNN_REFERENCE, SPREG_TWO_STAGE_REFERENCE} from './spreg-reference';
import {createNearestNeighborScene} from './spatial-regression-diagnostics-oracle';
import {
  createLagScene,
  fitSpatialTwoStageLeastSquaresOnCPU,
  type TwoStageOracleResult
} from './spatial-two-stage-least-squares-oracle';

type Scene = ReturnType<typeof createLagScene>;
type Result = {status: number; table: Float32Array; summary: Float32Array; residuals: Float32Array};

async function runTwoStage(device: Device, scene: Scene, tileRowCount?: number): Promise<Result> {
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
