// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GWR grid candidate path against the all-rows scan (`indexGridSize: false`) on the same uniform
 * scene: coefficients must agree (only the float summation order differs). Bounded fixed bandwidth
 * covering about 60 rows.
 */

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const DOMAIN_SIZE = 100;
const LADDER_CAPACITY = 8;
const WARMUP_COUNT = 0;
const SAMPLE_COUNT = 1;

function createScene(rowCount: number) {
  let state = 987654321;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const positions = new Float32Array(rowCount * 2);
  const predictors = new Float32Array(rowCount * 2);
  const response = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const x = random() * DOMAIN_SIZE;
    const y = random() * DOMAIN_SIZE;
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    predictors[2 * row] = random() * 4 - 2;
    predictors[2 * row + 1] = random() * 4 - 2;
    response[row] =
      1 + 0.02 * x * predictors[2 * row] + 0.01 * y * predictors[2 * row + 1] + 0.2 * random();
  }
  return {positions, predictors, response};
}

async function measure(
  device: Awaited<ReturnType<typeof getWebGPUTestDevice>>,
  scene: ReturnType<typeof createScene>,
  indexGridSize: false | undefined,
  bandwidths: number[]
) {
  const rowCount = scene.response.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => (buffers.push(buffer), buffer);
  const parameterBuffer = new GPUParameterBuffer(device!, {
    id: 'gwr-bench-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
  });
  const coefficients = track(createOutputBuffer(device!, rowCount * 3));
  const summary = track(createOutputBuffer(device!, 6));
  const graph = new GPUCommandGraph(device!, {id: 'gwr-bench'});
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device!, scene.positions)),
        'float32x2',
        rowCount
      ),
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device!, scene.predictors)),
        'float32',
        rowCount * 2
      ),
      predictorCount: 2,
      response: importGraphBuffer(
        graph,
        'response',
        track(createInputBuffer(device!, scene.response)),
        'float32',
        rowCount
      ),
      parameters: parameterBuffer.importToGraph(graph),
      maximumBandwidthCount: LADDER_CAPACITY,
      ...(indexGridSize === false ? {indexGridSize} : {}),
      output: {
        coefficients: importGraphBuffer(
          graph,
          'coefficients',
          coefficients,
          'float32',
          rowCount * 3
        ),
        summary: importGraphBuffer(graph, 'summary', summary, 'float32', 6)
      }
    })
  );
  const compiled = graph.compile();
  parameterBuffer.write(
    getGPUGeographicallyWeightedRegressionParameterValues(
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths},
      LADDER_CAPACITY
    )
  );
  const samples: number[] = [];
  for (let iteration = 0; iteration < WARMUP_COUNT + SAMPLE_COUNT; iteration++) {
    const start = performance.now();
    submitGraph(device!, compiled, undefined);
    await readFloat32(summary, 6);
    if (iteration >= WARMUP_COUNT) samples.push(performance.now() - start);
  }
  const result = {
    median: samples.sort((a, b) => a - b)[Math.floor(samples.length / 2)],
    coefficients: await readFloat32(coefficients, rowCount * 3),
    summary: await readFloat32(summary, 6)
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

for (const [rowCount, ladderLength, runBrute] of [
  [4096, 1, true],
  [16384, 1, true],
  [16384, 8, true]
] as const) {
  it(`GWR ${rowCount} rows, ${ladderLength} bandwidths: grid vs scan`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const scene = createScene(rowCount);
    const baseBandwidth = Math.sqrt((60 * DOMAIN_SIZE * DOMAIN_SIZE) / (rowCount * Math.PI));
    const bandwidths = Array.from(
      {length: ladderLength},
      (_, index) => baseBandwidth * (0.6 + 0.2 * index)
    );
    const grid = await measure(device, scene, undefined, bandwidths);
    let line = `GWR n=${rowCount} ladder=${ladderLength}: grid ${grid.median.toFixed(1)} ms`;
    if (runBrute) {
      const scan = await measure(device, scene, false, bandwidths);
      let maximumDifference = 0;
      for (let index = 0; index < grid.coefficients.length; index++) {
        const difference = Math.abs(grid.coefficients[index] - scan.coefficients[index]);
        if (!Number.isNaN(difference)) maximumDifference = Math.max(maximumDifference, difference);
      }
      line += `, scan ${scan.median.toFixed(1)} ms (x${(scan.median / grid.median).toFixed(1)}), max |coef diff| ${maximumDifference.toExponential(2)}, summary AICc ${grid.summary[2]} vs ${scan.summary[2]}`;
      expect(maximumDifference).toBeLessThan(1e-3);
      expect(Math.abs(grid.summary[2] - scan.summary[2])).toBeLessThan(0.1);
    }
    // eslint-disable-next-line no-console
    console.log(line);
  }, 300000);
}
