// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUSpatialRegressionDiagnostics} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-parameters';
import {
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-diagnostics-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {SPREG_ASYMMETRIC_KNN_REFERENCE, SPREG_DIAGNOSTICS_REFERENCE} from './spreg-reference';
import {
  computeSpatialRegressionDiagnosticsOnCPU,
  createDiagnosticsScene,
  createNearestNeighborScene,
  type DiagnosticsOracleResult
} from './spatial-regression-diagnostics-oracle';

type Scene = ReturnType<typeof createDiagnosticsScene>;

type Result = {status: number; tests: Float32Array; summary: Float32Array};

/** Fits OLS and the diagnostics in one graph, so the residuals never leave the GPU. */
async function runDiagnostics(
  device: Device,
  scene: Scene,
  tileRowCount?: number,
  residualsOverride?: Float32Array
): Promise<Result> {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'diagnostics-graph'});
  const predictors = importGraphBuffer(
    graph,
    'predictors',
    track(createInputBuffer(device, scene.predictors)),
    'float32',
    rows * k
  );
  const response = importGraphBuffer(
    graph,
    'response',
    track(createInputBuffer(device, scene.response)),
    'float32',
    rows
  );
  const residualBuffer = track(
    residualsOverride
      ? createInputBuffer(device, residualsOverride)
      : createOutputBuffer(device, rows)
  );
  const residuals = importGraphBuffer(graph, 'residuals', residualBuffer, 'float32', rows);
  if (!residualsOverride) {
    graph.add(
      new GPUOrdinaryLeastSquares({
        id: 'ols',
        predictors,
        response,
        predictorCount: k,
        output: {
          coefficients: importGraphBuffer(
            graph,
            'coef',
            track(createOutputBuffer(device, k + 1)),
            'float32',
            k + 1
          ),
          standardErrors: importGraphBuffer(
            graph,
            'se',
            track(createOutputBuffer(device, k + 1)),
            'float32',
            k + 1
          ),
          tStatistics: importGraphBuffer(
            graph,
            'ts',
            track(createOutputBuffer(device, k + 1)),
            'float32',
            k + 1
          ),
          summary: importGraphBuffer(
            graph,
            'olsSummary',
            track(createOutputBuffer(device, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH)),
            'float32',
            GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
          ),
          status: importGraphBuffer(
            graph,
            'olsStatus',
            track(createOutputBuffer(device, 1)),
            'uint32',
            1
          ),
          residuals
        }
      })
    );
  }
  const weights = scene.weights;
  const testsBuffer = track(
    createOutputBuffer(device, GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH)
  );
  const summaryBuffer = track(
    createOutputBuffer(device, GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH)
  );
  const statusBuffer = track(createOutputBuffer(device, 1));
  graph.add(
    new GPUSpatialRegressionDiagnostics({
      id: 'diagnostics',
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
      predictors,
      response,
      residuals,
      predictorCount: k,
      tileRowCount,
      output: {
        tests: importGraphBuffer(
          graph,
          'tests',
          testsBuffer,
          'float32',
          GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
        ),
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
        ),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [status] = await readUint32(statusBuffer, 1);
  const result = {
    status,
    tests: Float32Array.from(
      await readFloat32(testsBuffer, GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH)
    ),
    summary: Float32Array.from(
      await readFloat32(summaryBuffer, GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH)
    )
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
  floor = 1e-3
) {
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.max(Math.abs(expected), floor)
  );
}

function expectParity(actual: Result, expected: DiagnosticsOracleResult): void {
  expect(actual.status).toBe(0);
  const tests = [
    expected.lmLag,
    expected.lmError,
    expected.robustLmLag,
    expected.robustLmError,
    expected.lmSarma
  ];
  const names = ['lmLag', 'lmError', 'robustLmLag', 'robustLmError', 'lmSarma'];
  tests.forEach((test, index) => {
    expectClose(`${names[index]} statistic`, actual.tests[index * 3], test.statistic, 2e-3, 0.1);
    expect(actual.tests[index * 3 + 1]).toBe(test.degreesOfFreedom);
    expect(Math.abs(actual.tests[index * 3 + 2] - test.pValue)).toBeLessThan(2e-3);
  });
  const s = actual.summary;
  expect(s[0]).toBe(expected.residuals.length);
  expectClose('sigma2', s[1], expected.sigmaSquared, 1e-3);
  expectClose('trace', s[2], expected.trace, 1e-4);
  expectClose('information', s[3], expected.information, 2e-3);
  expectClose('weightsSum', s[4], expected.weightsSum, 1e-5);
  expectClose('moranI', s[5], expected.moranI, 2e-3, 1e-2);
  expectClose('moranExpectation', s[6], expected.moranExpectation, 2e-3, 1e-4);
  expectClose('moranVariance', s[7], expected.moranVariance, 1e-2, 1e-6);
  expectClose('moranZ', s[8], expected.moranZ, 1e-2, 0.1);
  expect(actual.tests[15]).toBe(s[8]);
  expect(actual.tests[16]).toBe(0);
  expect(Math.abs(s[9] - expected.moranPValue)).toBeLessThan(2e-3);
}

it('GPUSpatialRegressionDiagnostics matches the dense float64 oracle (row-standardized lattice)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createDiagnosticsScene(8, 11, 0.6, true);
  const expected = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const actual = await runDiagnostics(device, scene);
  // The scene is built with dependent errors: the error tests must be clearly nonzero.
  expect(expected.lmError.statistic).toBeGreaterThan(5);
  expect(actual.tests[3]).toBeGreaterThan(1);
  expectParity(actual, expected);
  // SARMA is LM-lag plus robust LM-error by definition.
  expectClose('sarma identity', actual.tests[12], actual.tests[0] + actual.tests[9], 1e-4);
});

it('GPUSpatialRegressionDiagnostics matches the oracle on binary weights and several tile sizes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createDiagnosticsScene(12, 5, 0.45, false);
  const expected = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const reference = await runDiagnostics(device, scene);
  expectParity(reference, expected);
  expect(reference.tests[3]).toBeGreaterThan(1);
  for (const tileRowCount of [1, 7, 144]) {
    const result = await runDiagnostics(device, scene, tileRowCount);
    expectParity(result, expected);
  }
  // Same tile size, same bits.
  const again = await runDiagnostics(device, scene);
  expect(Array.from(new Uint32Array(again.tests.buffer))).toEqual(
    Array.from(new Uint32Array(reference.tests.buffer))
  );
});

it('GPUSpatialRegressionDiagnostics reports weak evidence for independent errors', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createDiagnosticsScene(10, 99, 0, true);
  const expected = computeSpatialRegressionDiagnosticsOnCPU(
    scene.weights,
    scene.predictors,
    scene.response,
    scene.predictorCount
  );
  const actual = await runDiagnostics(device, scene);
  expectParity(actual, expected);
  expect(expected.lmError.pValue).toBeGreaterThan(0.01);
  expect(actual.tests[5]).toBeGreaterThan(0.01);
});

it('GPUSpatialRegressionDiagnostics reports non-finite residuals as status 4 with NaN', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createDiagnosticsScene(6, 3, 0.3, true);
  const residuals = new Float32Array(36).fill(0.5);
  residuals[4] = NaN;
  const result = await runDiagnostics(device, scene, undefined, residuals);
  expect(result.status).toBe(4);
  expect(Number.isNaN(result.tests[0])).toBe(true);
  expect(Number.isNaN(result.summary[5])).toBe(true);
});

function expectMatchesSpreg(
  actual: Result,
  reference: (typeof SPREG_DIAGNOSTICS_REFERENCE)['lattice']
): void {
  expect(actual.status).toBe(0);
  const pairs = [
    reference.lmLag,
    reference.lmError,
    reference.robustLmLag,
    reference.robustLmError,
    reference.lmSarma
  ];
  pairs.forEach(([statistic, pValue], index) => {
    expectClose(`spreg statistic ${index}`, actual.tests[index * 3], statistic, 2e-3, 0.1);
    expect(Math.abs(actual.tests[index * 3 + 2] - pValue)).toBeLessThan(2e-3);
  });
  expectClose('spreg sigma2', actual.summary[1], reference.sigmaSquared, 1e-3);
  expectClose('spreg moran I', actual.summary[5], reference.moran[0], 2e-3, 1e-2);
  expectClose('spreg moran z', actual.summary[8], reference.moran[1], 1e-2, 0.1);
  expect(Math.abs(actual.summary[9] - reference.moran[2])).toBeLessThan(2e-3);
}

it('GPUSpatialRegressionDiagnostics matches spreg (OLS spat_diag, moran) on the lattice scene', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createDiagnosticsScene(8, 11, 0.6, true);
  expectMatchesSpreg(await runDiagnostics(device, scene), SPREG_DIAGNOSTICS_REFERENCE.lattice);
});

it('GPUSpatialRegressionDiagnostics matches spreg on symmetrized kNN weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createNearestNeighborScene(80, 4, 7, 0.6);
  const actual = await runDiagnostics(device, scene);
  expectMatchesSpreg(actual, SPREG_DIAGNOSTICS_REFERENCE.nearestNeighbor);
  // Strong error dependence: LM-error far above its chi-square(1) critical value.
  expect(actual.tests[3]).toBeGreaterThan(20);
});

it('GPUSpatialRegressionDiagnostics matches the dense oracle on asymmetric kNN weights (k = 3)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [count, seed] of [
    [60, 13],
    [90, 29]
  ] as const) {
    const scene = createNearestNeighborScene(count, 3, seed, 0.6, false);
    // The directed kNN pattern must really be asymmetric, or the case proves nothing.
    const {offsets, neighbors} = scene.weights;
    const pairs = new Set<number>();
    for (let row = 0; row < count; row++) {
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        pairs.add(row * count + neighbors[slot]);
      }
    }
    let oneWay = 0;
    for (const pair of pairs) {
      if (!pairs.has((pair % count) * count + Math.floor(pair / count))) oneWay++;
    }
    expect(oneWay).toBeGreaterThan(count / 4);
    // Dense float64 oracle from the textbook definitions (spreg is not installed here, so no spreg pin).
    const expected = computeSpatialRegressionDiagnosticsOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    const actual = await runDiagnostics(device, scene);
    expectParity(actual, expected);
    expect(actual.tests[3]).toBeGreaterThan(2);
    // Other tile sizes agree too.
    expectParity(await runDiagnostics(device, scene, 7), expected);
  }
});

it('GPUSpatialRegressionDiagnostics matches spreg on asymmetric directed kNN weights (k = 3)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [count, seed, reference] of [
    [60, 13, SPREG_ASYMMETRIC_KNN_REFERENCE.scene_60_13.diagnostics],
    [90, 29, SPREG_ASYMMETRIC_KNN_REFERENCE.scene_90_29.diagnostics]
  ] as const) {
    const scene = createNearestNeighborScene(count, 3, seed, 0.6, false);
    expectMatchesSpreg(await runDiagnostics(device, scene), reference);
  }
});
