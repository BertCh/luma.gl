// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURipleyDistanceFunctions} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {GPUVariogram} from '../../../src/gpu-dataframe/pair-statistics/gpu-variogram';
import {
  getGPURipleyDistanceFunctionsParameterValues,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/variogram-parameters';
import {requireTestDevice} from '../../utils/require-test-device';
import {createPairStatisticsHarness} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyDistanceFunctionsOnCPU} from './ripley-distance-functions-oracle';
import {computeVariogramOnCPU} from './variogram-oracle';

// Report B section 5: the variogram at 512 bins (409 with the robust channel) and the Ripley
// distance functions at 256 radii sit at the 16384-byte workgroup limit. The node spec
// `pair-statistics-limits.node.spec.ts` counts the declarations; this spec compiles and runs both
// shapes on a real device inside a validation error scope and checks the totals against the f64
// oracles (a failed pipeline leaves zeros).

const BOUNDS = [0, 0, 100, 100] as const;

async function captureValidationErrors(
  device: Device,
  callback: () => Promise<void>
): Promise<string[]> {
  const gpuDevice = device.handle as GPUDevice;
  gpuDevice.pushErrorScope('validation');
  await callback();
  await gpuDevice.queue.onSubmittedWorkDone();
  const error = await gpuDevice.popErrorScope();
  return error ? [error.message] : [];
}

function createScene(seed: number, rows: number) {
  const random = createRandom(seed);
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = random() * 100;
    positions[row * 2 + 1] = random() * 100;
    values[row] = 10 * Math.sin(positions[row * 2] / 6) + random();
  }
  return {positions, values};
}

it('GPUVariogram compiles and runs at the maximal workgroup shapes on a real device', async () => {
  // A 'core' device keeps the default limits, so 16384 bytes is the real ceiling here.
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  expect(device.limits.maxComputeWorkgroupStorageSize).toBe(16384);
  const scene = createScene(3, 1500);
  const frame = {bounds: BOUNDS, maximumDistance: 40};
  for (const [lagCount, directionCount, robust] of [
    [128, 4, false],
    [409, 1, true]
  ] as const) {
    const binCount = lagCount * directionCount;
    let pairTotal = 0;
    let robustFinite = 0;
    const errors = await captureValidationErrors(device, async () => {
      const harness = createPairStatisticsHarness(device, {
        ...scene,
        parameterLength: GPU_VARIOGRAM_PARAMETER_LENGTH,
        outputs: {
          semivariances: {format: 'float32', length: binCount},
          pairCounts: {format: 'uint32', length: binCount},
          meanDistances: {format: 'float32', length: binCount},
          // Bound only with the robust channel; otherwise it stays unwritten.
          robustSemivariances: {format: 'float32', length: binCount},
          statistics: {format: 'float32', length: GPU_VARIOGRAM_STATISTICS_LENGTH},
          overflow: {format: 'uint32', length: 1}
        },
        createContributor: views =>
          new GPUVariogram({
            positions: views.positions,
            values: views.values!,
            parameters: views.parameters,
            gridSize: [32, 32],
            lagCount,
            directionCount,
            output: {
              semivariances: views.outputs.semivariances as never,
              pairCounts: views.outputs.pairCounts as never,
              meanDistances: views.outputs.meanDistances as never,
              ...(robust ? {robustSemivariances: views.outputs.robustSemivariances as never} : {}),
              statistics: views.outputs.statistics as never,
              status: {overflow: views.outputs.overflow as never}
            }
          })
      });
      try {
        const result = await harness.run(getGPUVariogramParameterValues(frame));
        pairTotal = result.pairCounts.reduce((sum, count) => sum + count, 0);
        robustFinite = robust
          ? result.robustSemivariances.filter(value => Number.isFinite(value)).length
          : 0;
      } finally {
        harness.destroy();
      }
    });
    const label = `${lagCount} x ${directionCount}${robust ? ' robust' : ''}`;
    expect(errors, `${label} validation errors`).toEqual([]);
    const oracle = computeVariogramOnCPU(scene, frame, lagCount, directionCount);
    expect(pairTotal, `${label} pair total`).toBe(
      oracle.pairCounts.reduce((sum, count) => sum + count, 0)
    );
    expect(pairTotal).toBeGreaterThan(10000);
    if (robust) {
      expect(robustFinite, `${label} robust bins`).toBeGreaterThan(binCount / 2);
    }
  }
});

it('GPURipleyDistanceFunctions compiles and runs at 256 radii on a real device', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  expect(device.limits.maxComputeWorkgroupStorageSize).toBe(16384);
  const radiusCount = 256;
  const referenceGrid = [37, 31] as const;
  const scene = {positions: createScene(7, 700).positions};
  const frame = {bounds: BOUNDS, maximumDistance: 12, edgeCorrection: 'none' as const};
  let result: Record<'g' | 'f', number[]> | undefined;
  const errors = await captureValidationErrors(device, async () => {
    const harness = createPairStatisticsHarness(device, {
      ...scene,
      parameterLength: GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
      outputs: {
        g: {format: 'float32', length: radiusCount},
        f: {format: 'float32', length: radiusCount}
      },
      createContributor: views =>
        new GPURipleyDistanceFunctions({
          positions: views.positions,
          parameters: views.parameters,
          gridSize: [24, 24],
          referenceGrid,
          radiusCount,
          output: {g: views.outputs.g as never, f: views.outputs.f as never}
        })
    });
    try {
      result = await harness.run(getGPURipleyDistanceFunctionsParameterValues(frame));
    } finally {
      harness.destroy();
    }
  });
  expect(errors, 'validation errors').toEqual([]);
  const oracle = computeRipleyDistanceFunctionsOnCPU(scene, frame, radiusCount, referenceGrid);
  // The last radius is the maximum distance: both CDFs are compared there and at the midpoint.
  for (const radius of [radiusCount / 2 - 1, radiusCount - 1]) {
    expect(Math.abs(result!.g[radius] - oracle.g[radius])).toBeLessThan(0.01);
    expect(Math.abs(result!.f[radius] - oracle.f[radius])).toBeLessThan(0.01);
  }
  expect(result!.g[radiusCount - 1]).toBeGreaterThan(0.5);
});
