// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {SPREG_ERROR_GM_REFERENCE} from './spatial-error-gm-reference';
import {
  createErrorScene,
  fitSpatialErrorGMOnCPU,
  type ErrorGMOracleResult
} from './spatial-error-gm-oracle';

type Scene = ReturnType<typeof createErrorScene>;
type Result = {status: number; table: Float32Array; summary: Float32Array; residuals: Float32Array};

async function runErrorGM(device: Device, scene: Scene, tileRowCount?: number): Promise<Result> {
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
      output: {
        table: importGraphBuffer(graph, 'table', tableBuffer, 'float32', tableLength),
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH
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

function expectParity(actual: Result, expected: ErrorGMOracleResult): void {
  expect(actual.status).toBe(0);
  const count = expected.coefficients.length;
  // Oracle order is (intercept, X...); the GPU table adds a lambda row last.
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
  expect(Math.abs(actual.table[count * 4] - expected.lambda)).toBeLessThan(5e-3);
  expect(Number.isNaN(actual.table[count * 4 + 1])).toBe(true);
  expect(Math.abs(actual.summary[1] - expected.lambda)).toBeLessThan(5e-3);
  expectClose('sigma2', actual.summary[2], expected.sigmaSquared, 1e-2, 1e-3);
  expectClose('pr2', actual.summary[3], expected.pseudoRSquared, 5e-3, 1e-3);
  expectClose('rss', actual.summary[4], expected.residualSumOfSquares, 1e-2, 1e-3);
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
    if (!device) {
      return;
    }
    const scene = createErrorScene(count, seed, lambda, kind);
    const expected = fitSpatialErrorGMOnCPU(
      scene.weights,
      scene.predictors,
      scene.response,
      scene.predictorCount
    );
    const actual = await runErrorGM(device, scene);
    expectParity(actual, expected);
    // The moment solve must be a real computation, not a failed-compile zero.
    expect(Math.abs(actual.summary[1])).toBeGreaterThan(0.02);
    expect(actual.summary[2]).toBeGreaterThan(0.1);
    if (kind === 'lattice-row-standardized') {
      expect(actual.summary[1]).toBeGreaterThan(0.25);
      expect(actual.summary[1]).toBeLessThan(0.95);
      for (const tileRowCount of [7, 400]) {
        expectParity(await runErrorGM(device, scene, tileRowCount), expected);
      }
    }
  });
}

it('GPUSpatialErrorGM reports a singular design', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createErrorScene(100, 3, 0.5, 'lattice-row-standardized');
  for (let row = 0; row < scene.response.length; row++) {
    scene.predictors[row * 2 + 1] = 2 * scene.predictors[row * 2];
  }
  const actual = await runErrorGM(device, scene);
  expect(actual.status).toBe(1);
  expect(Number.isNaN(actual.table[0])).toBe(true);
});

for (const name of ['lattice', 'knn'] as const) {
  it(`GPUSpatialErrorGM matches spreg GM_Error on the ${name} fixture`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const reference = SPREG_ERROR_GM_REFERENCE[name];
    const actual = await runErrorGM(device, {
      weights: {
        offsets: Uint32Array.from(reference.offsets),
        neighbors: Uint32Array.from(reference.neighbors),
        weights: Float32Array.from(reference.weights)
      },
      predictors: Float32Array.from(reference.predictors),
      response: Float32Array.from(reference.response),
      predictorCount: 2
    });
    expect(actual.status).toBe(0);
    expect(Math.abs(actual.table[3 * 4] - reference.lambda)).toBeLessThan(5e-3);
    reference.betas.forEach((beta, row) => {
      expectClose(`spreg beta ${row}`, actual.table[row * 4], beta, 5e-3, 0.05);
      expectClose(
        `spreg se ${row}`,
        actual.table[row * 4 + 1],
        reference.standardErrors[row],
        1e-2,
        1e-3
      );
      expectClose(
        `spreg z ${row}`,
        actual.table[row * 4 + 2],
        reference.zStatistics[row],
        2e-2,
        0.2
      );
    });
    expectClose('spreg sigma2', actual.summary[2], reference.sigma2, 1e-2, 1e-3);
    expectClose('spreg pr2', actual.summary[3], reference.pseudoRSquared, 5e-3, 1e-3);
    expect(actual.summary[1]).toBeGreaterThan(0.3);
  });
}
