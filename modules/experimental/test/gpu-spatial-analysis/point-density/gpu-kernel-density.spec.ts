// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity,
  type GPUKernelDensityEdgeCorrection,
  type GPUKernelDensityKernel,
  type GPUKernelDensityNormalization
} from '../../../src/gpu-spatial-analysis/point-density';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {KERNEL_DENSITY_ORACLE} from './kernel-density-oracle-values';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * GPU versus float64 oracle (sklearn / numpy / scipy dblquad, see kernel-density-oracle-values.ts):
 * `|gpu - oracle| <= 2e-4 * max|oracle|`. Covers f32 accumulation, the f32 erfc fit (1.2e-7), and
 * the f32 quadrant mass cancellation of the edge correction.
 */
const RELATIVE_TOLERANCE = 2e-4;

const {field} = KERNEL_DENSITY_ORACLE;
const WINDOW = field.window as unknown as [number, number, number, number];

type Run = {
  values: Float32Array;
  weightSum?: number;
  invalidCount?: number;
};

async function runKernelDensity(
  device: Device,
  options: {
    kernel: GPUKernelDensityKernel;
    edgeCorrection: GPUKernelDensityEdgeCorrection;
    normalization?: GPUKernelDensityNormalization;
    weighted: boolean;
    extraRows?: {positions: number[]; weights: number[]};
    readStatus?: boolean;
  }
): Promise<Run> {
  const positions = [...field.points, ...(options.extraRows?.positions ?? [])];
  const weights = [...field.weights, ...(options.extraRows?.weights ?? [])];
  const pointCount = positions.length / 2;
  const cellCount = field.columns * field.rows;
  // Rows advance north from the window minimum, as the old extent props did.
  const grid = makeExtentGrid(field.columns, field.rows, WINDOW);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kernel-density-parameters',
    format: 'float32',
    length: 9
  });
  const valuesOut = track(createOutputBuffer(device, cellCount));
  const weightSumOut = track(createOutputBuffer(device, 1));
  const invalidCountOut = track(createOutputBuffer(device, 1));
  const graph = new GPUCommandGraph(device, {id: 'kernel-density-graph'});
  graph.add(
    new GPUKernelDensity({
      id: 'kde',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, Float32Array.from(positions))),
        'float32x2',
        pointCount
      ),
      weights: options.weighted
        ? importGraphBuffer(
            graph,
            'weights',
            track(createInputBuffer(device, Float32Array.from(weights))),
            'float32',
            pointCount
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: [4, 3],
      indexBounds: WINDOW,
      kernel: options.kernel,
      edgeCorrection: options.edgeCorrection,
      normalization: options.normalization,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        weightSum: options.readStatus
          ? importGraphBuffer(graph, 'weight-sum-out', weightSumOut, 'float32', 1)
          : undefined,
        status: options.readStatus
          ? {
              invalidCount: importGraphBuffer(
                graph,
                'invalid-count-out',
                invalidCountOut,
                'uint32',
                1
              )
            }
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  try {
    parameterBuffer.write(getGPUKernelDensityParameterValues({bandwidth: field.bandwidth}, grid));
    submitGraph(device, compiled, undefined);
    return {
      values: Float32Array.from(await readFloat32(valuesOut, cellCount)),
      weightSum: options.readStatus ? (await readFloat32(weightSumOut, 1))[0] : undefined,
      invalidCount: options.readStatus ? (await readUint32(invalidCountOut, 1))[0] : undefined
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function expectField(actual: Float32Array, expected: readonly number[], label: string): void {
  const scale = Math.max(...expected.map(Math.abs));
  let worst = 0;
  for (const [cell, value] of expected.entries()) {
    worst = Math.max(worst, Math.abs(actual[cell] - value));
  }
  expect(worst, label).toBeLessThanOrEqual(RELATIVE_TOLERANCE * scale);
}

it('GPUKernelDensity matches sklearn, numpy and dblquad for every kernel and edge correction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of field.cases) {
    const run = await runKernelDensity(device, {
      kernel: testCase.kernel as GPUKernelDensityKernel,
      edgeCorrection: testCase.edgeCorrection as GPUKernelDensityEdgeCorrection,
      weighted: testCase.weighted
    });
    expectField(
      run.values,
      testCase.values,
      `${testCase.kernel} ${testCase.edgeCorrection} weighted=${testCase.weighted}`
    );
  }
});

it('GPUKernelDensity probability output divides by the included weight', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const testCase = field.cases.find(
    candidate =>
      candidate.kernel === 'gaussian' && candidate.edgeCorrection === 'none' && candidate.weighted
  )!;
  const run = await runKernelDensity(device, {
    kernel: 'gaussian',
    edgeCorrection: 'none',
    normalization: 'probability',
    weighted: true,
    readStatus: true
  });
  expectField(
    run.values,
    testCase.values.map(value => value / testCase.weightSum),
    'probability'
  );
  expect(run.weightSum!).toBeCloseTo(testCase.weightSum, 4);
  expect(run.invalidCount).toBe(0);
});

it('GPUKernelDensity excludes non-finite, negative and out-of-domain rows and counts them', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const testCase = field.cases.find(
    candidate =>
      candidate.kernel === 'gaussian' && candidate.edgeCorrection === 'none' && candidate.weighted
  )!;
  // NaN position, negative weight, and a position outside indexBounds: all excluded.
  const run = await runKernelDensity(device, {
    kernel: 'gaussian',
    edgeCorrection: 'none',
    weighted: true,
    readStatus: true,
    extraRows: {positions: [Number.NaN, 1, 2, 2, 50, 50], weights: [1, -1, 1]}
  });
  expectField(run.values, testCase.values, 'excluded rows leave the field unchanged');
  expect(run.invalidCount).toBe(3);
  expect(run.weightSum!).toBeCloseTo(testCase.weightSum, 4);
});
