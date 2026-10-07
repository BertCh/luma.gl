// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GWR grid paths against the all-rows scan (`indexGridSize: false`) on scenes with masked and
 * non-finite rows and clustered locations: adaptive bisquare (ring k-th neighbour search, grid
 * weights, cell-order invocations), adaptive Gaussian (ring k-th neighbour search, scanned
 * weights) and fixed bisquare. Coefficients agree up to the float summation order.
 */

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const LADDER_CAPACITY = 4;

function createScene(rowCount: number) {
  let state = 24681357;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const positions = new Float32Array(rowCount * 2);
  const predictors = new Float32Array(rowCount * 2);
  const response = new Float32Array(rowCount);
  const mask = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    // Half the rows form a dense cluster, so cells differ strongly in population.
    const isCluster = random() < 0.5;
    const x = isCluster ? 10 + random() * 15 : random() * 100;
    const y = isCluster ? 60 + random() * 15 : random() * 100;
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    predictors[2 * row] = random() * 4 - 2;
    predictors[2 * row + 1] = random() * 4 - 2;
    response[row] =
      1 + 0.02 * x * predictors[2 * row] + 0.01 * y * predictors[2 * row + 1] + 0.2 * random();
    mask[row] = random() < 0.08 ? 0 : 1;
  }
  // Non-finite inputs are excluded like masked rows.
  positions[2 * 5] = Number.NaN;
  response[9] = Number.NaN;
  return {positions, predictors, response, mask};
}

async function fit(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  scene: ReturnType<typeof createScene>,
  indexGridSize: false | readonly [number, number],
  settings: Parameters<typeof getGPUGeographicallyWeightedRegressionParameterValues>[0]
) {
  const rowCount = scene.response.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => (buffers.push(buffer), buffer);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'gwr-adaptive-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
  });
  const coefficients = track(createOutputBuffer(device, rowCount * 3));
  const status = track(createOutputBuffer(device, rowCount));
  const scores = track(createOutputBuffer(device, LADDER_CAPACITY));
  const summary = track(createOutputBuffer(device, 6));
  const graph = new GPUCommandGraph(device, {id: 'gwr-adaptive-grid'});
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, scene.positions)),
        'float32x2',
        rowCount
      ),
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device, scene.predictors)),
        'float32',
        rowCount * 2
      ),
      predictorCount: 2,
      response: importGraphBuffer(
        graph,
        'response',
        track(createInputBuffer(device, scene.response)),
        'float32',
        rowCount
      ),
      mask: importGraphBuffer(
        graph,
        'mask',
        track(createInputBuffer(device, scene.mask)),
        'uint32',
        rowCount
      ),
      parameters: parameterBuffer.importToGraph(graph),
      maximumBandwidthCount: LADDER_CAPACITY,
      maximumNeighborCount: 64,
      indexGridSize,
      output: {
        coefficients: importGraphBuffer(
          graph,
          'coefficients',
          coefficients,
          'float32',
          rowCount * 3
        ),
        localStatus: importGraphBuffer(graph, 'status', status, 'uint32', rowCount),
        bandwidthScores: importGraphBuffer(graph, 'scores', scores, 'float32', LADDER_CAPACITY),
        summary: importGraphBuffer(graph, 'summary', summary, 'float32', 6)
      }
    })
  );
  const compiled = graph.compile();
  parameterBuffer.write(
    getGPUGeographicallyWeightedRegressionParameterValues(settings, LADDER_CAPACITY)
  );
  submitGraph(device, compiled, undefined);
  const result = {
    coefficients: await readFloat32(coefficients, rowCount * 3),
    status: await readUint32(status, rowCount),
    scores: await readFloat32(scores, LADDER_CAPACITY),
    summary: await readFloat32(summary, 6)
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

const CASES = [
  {name: 'adaptive bisquare', kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [20, 40]},
  {name: 'adaptive gaussian', kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [30, 60]},
  {name: 'fixed bisquare', kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [9, 14]}
] as const;

for (const {name, ...settings} of CASES) {
  it(`GWR grid ${name} agrees with the scan on clustered, masked rows`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const scene = createScene(1500);
    const scan = await fit(device, scene, false, settings);
    // A grid finer than the cluster and a coarse one: both must agree with the scan.
    for (const gridSize of [
      [24, 24],
      [5, 7]
    ] as const) {
      const grid = await fit(device, scene, gridSize, settings);
      expect(Array.from(grid.status), `${name} ${gridSize} status`).toEqual(
        Array.from(scan.status)
      );
      expect(
        grid.status.filter(code => code === GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED)
          .length
      ).toBeGreaterThan(50);
      let worst = 0;
      for (let index = 0; index < scan.coefficients.length; index++) {
        const difference = Math.abs(grid.coefficients[index] - scan.coefficients[index]);
        if (Number.isNaN(scan.coefficients[index])) {
          expect(Number.isNaN(grid.coefficients[index])).toBe(true);
        } else {
          worst = Math.max(worst, difference);
        }
      }
      expect(worst, `${name} ${gridSize} coefficients`).toBeLessThan(2e-3);
      for (let candidate = 0; candidate < 2; candidate++) {
        expect(Math.abs(grid.scores[candidate] - scan.scores[candidate])).toBeLessThan(0.5);
      }
      expect(grid.summary[5]).toBe(scan.summary[5]);
    }
  }, 120000);
}
