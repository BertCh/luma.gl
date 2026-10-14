// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity
} from '../../../src/gpu-spatial-analysis/point-density';
import type {GPUKernelDensityMethod} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

// Same-run A/B of the exact gather against the binned blur at a large bandwidth.
// Opt in with `VITE_KERNEL_DENSITY_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_KERNEL_DENSITY_BENCHMARK'
  ] === 'true';

const POINT_COUNT = 100000;
const SIZE = 512;
const CELL_COUNT = SIZE * SIZE;
const BANDWIDTH_CELLS = 64;
const SAMPLE_COUNT = 5;

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

/** Deterministic clustered points on the grid: a uniform background and 20 gaussian clusters. */
function createPoints(): Float32Array {
  let seed = 20261014;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  const points = new Float32Array(2 * POINT_COUNT);
  const centers = Array.from({length: 20}, () => [random() * SIZE, random() * SIZE]);
  for (let index = 0; index < POINT_COUNT; index++) {
    if (index % 5 === 0) {
      points[2 * index] = random() * SIZE;
      points[2 * index + 1] = random() * SIZE;
    } else {
      const [centerX, centerY] = centers[index % 20];
      const gaussian = () => random() + random() + random() - 1.5;
      points[2 * index] = Math.min(Math.max(centerX + 30 * gaussian(), 0), SIZE);
      points[2 * index + 1] = Math.min(Math.max(centerY + 30 * gaussian(), 0), SIZE);
    }
  }
  return points;
}

async function measure(
  device: Device,
  points: Float32Array,
  method: GPUKernelDensityMethod
): Promise<{median: number; first: number; values: Float32Array}> {
  // Units are cells, so the bandwidth in cells equals the bandwidth in position units.
  const grid = makeExtentGrid(SIZE, SIZE, [0, 0, SIZE, SIZE]);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kde-bench-parameters',
    format: 'float32',
    length: 9,
    values: getGPUKernelDensityParameterValues({bandwidth: BANDWIDTH_CELLS}, grid)
  });
  const valuesOut = track(createOutputBuffer(device, CELL_COUNT));
  const graph = new GPUCommandGraph(device, {id: `kde-bench-${method}`});
  graph.add(
    new GPUKernelDensity({
      id: 'kde-bench',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, points)),
        'float32x2',
        POINT_COUNT
      ),
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: [128, 128],
      indexBounds: [0, 0, SIZE, SIZE],
      method,
      // 2.5 bandwidths of margin (160) plus the guard cell.
      binnedPadding: Math.ceil(2.5 * BANDWIDTH_CELLS),
      edgeCorrection: 'uniform',
      output: {values: importGraphBuffer(graph, 'values', valuesOut, 'float32', CELL_COUNT)}
    })
  );
  const compileStart = performance.now();
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  await waitForQueue(device);
  const first = performance.now() - compileStart;
  const samples: number[] = [];
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const start = performance.now();
    submitGraph(device, compiled, undefined);
    await waitForQueue(device);
    samples.push(performance.now() - start);
  }
  samples.sort((left, right) => left - right);
  const values = Float32Array.from(await readFloat32(valuesOut, CELL_COUNT));
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {median: samples[Math.floor(samples.length / 2)], first, values};
}

it.skipIf(!RUN_BENCHMARK)(
  'kernel density exact versus binned, 1e5 points, 512^2, h = 64 cells',
  {timeout: 900000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const points = createPoints();
    const lines: string[] = [];
    let exactValues: Float32Array | undefined;
    // Exact, binned, then both again in reverse order so warm-up does not favor one.
    for (const method of ['exact', 'binned', 'binned', 'exact'] as const) {
      const result = await measure(device, points, method);
      lines.push(
        `${method}: median ${result.median.toFixed(1)} ms, first ${result.first.toFixed(0)} ms`
      );
      if (method === 'exact') {
        exactValues = result.values;
      } else if (exactValues) {
        let worst = 0;
        let peak = 0;
        for (const [index, value] of exactValues.entries()) {
          worst = Math.max(worst, Math.abs(value - result.values[index]));
          peak = Math.max(peak, Math.abs(value));
        }
        lines.push(
          `  binned versus exact: max |difference| ${worst.toExponential(2)}, peak ${peak.toExponential(2)}`
        );
      }
    }
    console.log(`kernel density A/B\n${lines.join('\n')}`);
  }
);
