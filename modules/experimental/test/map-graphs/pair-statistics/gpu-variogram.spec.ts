// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUVariogram} from '../../../src/map-graphs/pair-statistics/gpu-variogram';
import {
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH,
  type GPUVariogramParameters
} from '../../../src/map-graphs/pair-statistics/variogram-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeVariogramOnCPU} from './variogram-oracle';

const BOUNDS = [0, 0, 100, 100] as const;

/** Anisotropic field: strong variation along x, weak along y, with masked, NaN and outside rows. */
function createAnisotropicScene(seed: number, rows: number) {
  const random = createRandom(seed);
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const x = random() * 104 - 2;
    const y = random() * 104 - 2;
    positions[row * 2] = x;
    positions[row * 2 + 1] = y;
    values[row] = 10 * Math.sin(x / 6) + 0.05 * y + random() - 0.5 + 40;
    mask[row] = random() < 0.92 ? 1 : 0;
    if (random() < 0.02) {
      values[row] = NaN;
    }
  }
  return {positions, values, mask};
}

function createHarness(
  device: Device,
  scene: ReturnType<typeof createAnisotropicScene>,
  lagCount: number,
  directionCount: number
) {
  const binCount = lagCount * directionCount;
  return createPairStatisticsHarness(device, {
    ...scene,
    parameterLength: GPU_VARIOGRAM_PARAMETER_LENGTH,
    outputs: {
      semivariances: {format: 'float32', length: binCount},
      pairCounts: {format: 'uint32', length: binCount},
      meanDistances: {format: 'float32', length: binCount},
      robustSemivariances: {format: 'float32', length: binCount},
      statistics: {format: 'float32', length: GPU_VARIOGRAM_STATISTICS_LENGTH}
    },
    createRecipe: views =>
      new GPUVariogram({
        positions: views.positions,
        values: views.values!,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [32, 32],
        lagCount,
        directionCount,
        semivariances: views.outputs.semivariances as never,
        pairCounts: views.outputs.pairCounts as never,
        meanDistances: views.outputs.meanDistances as never,
        robustSemivariances: views.outputs.robustSemivariances as never,
        statistics: views.outputs.statistics as never
      })
  });
}

function expectClose(label: string, actual: number, expected: number, tolerance: number): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  if (!(Math.abs(actual - expected) <= tolerance)) {
    throw new Error(`${label}: ${actual} != ${expected} (tolerance ${tolerance})`);
  }
}

it('GPUVariogram matches the f64 oracle across per-frame distances without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAnisotropicScene(5, 2500);
  const lagCount = 12;
  const harness = createHarness(device, scene, lagCount, 1);
  try {
    const frames: GPUVariogramParameters[] = [
      {bounds: BOUNDS, maximumDistance: 6},
      {bounds: BOUNDS, maximumDistance: 30},
      {bounds: BOUNDS, maximumDistance: 400},
      {bounds: [20, 10, 80, 70], maximumDistance: 15}
    ];
    for (const frame of frames) {
      const label = `maximumDistance ${frame.maximumDistance}`;
      const result = await harness.run(getGPUVariogramParameterValues(frame));
      const oracle = computeVariogramOnCPU(scene, frame, lagCount);
      const total = (counts: number[]) => counts.reduce((sum, count) => sum + count, 0);
      expect(total(result.pairCounts), label).toBe(total(oracle.pairCounts));
      expect(total(oracle.pairCounts)).toBeGreaterThan(1000);
      for (let bin = 0; bin < lagCount; bin++) {
        // GPU division rounding may move a pair lying on a lag edge into the neighboring lag.
        const countDifference = Math.abs(result.pairCounts[bin] - oracle.pairCounts[bin]);
        expect(countDifference, `${label} count ${bin}`).toBeLessThanOrEqual(
          oracle.edgePairCounts[bin]
        );
        if (countDifference > 0) {
          continue;
        }
        const expected = oracle.semivariances[bin];
        expectClose(
          `${label} gamma ${bin}`,
          result.semivariances[bin],
          expected,
          oracle.semivarianceTolerance + 2e-5 * Math.abs(expected)
        );
        expectClose(
          `${label} distance ${bin}`,
          result.meanDistances[bin],
          oracle.meanDistances[bin],
          frame.maximumDistance * 2 ** -24 + 2e-6 * oracle.meanDistances[bin]
        );
        expectClose(
          `${label} robust ${bin}`,
          result.robustSemivariances[bin],
          oracle.robustSemivariances[bin],
          oracle.semivarianceTolerance * 8 + 1e-4 * Math.abs(oracle.robustSemivariances[bin])
        );
      }
      expect(result.statistics[0]).toBe(oracle.statistics[0]);
      for (let slot = 1; slot < GPU_VARIOGRAM_STATISTICS_LENGTH; slot++) {
        expectClose(
          `${label} statistics ${slot}`,
          result.statistics[slot],
          oracle.statistics[slot],
          1e-4 * Math.abs(oracle.statistics[slot]) + 1e-5
        );
      }
      const again = await harness.run(getGPUVariogramParameterValues(frame));
      expect(getFloatBits(again.semivariances)).toEqual(getFloatBits(result.semivariances));
      expect(getFloatBits(again.robustSemivariances)).toEqual(
        getFloatBits(result.robustSemivariances)
      );
    }
    expect(harness.rebuildCount).toBe(0);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUVariogram splits pairs into direction sectors that resolve anisotropy', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAnisotropicScene(9, 2000);
  const lagCount = 6;
  const directionCount = 4;
  const binCount = lagCount * directionCount;
  const harness = createHarness(device, scene, lagCount, directionCount);
  try {
    for (const azimuthOffset of [-Math.PI / 8, 0.3]) {
      const frame = {bounds: BOUNDS, maximumDistance: 12, azimuthOffset};
      const result = await harness.run(getGPUVariogramParameterValues(frame));
      const oracle = computeVariogramOnCPU(scene, frame, lagCount, directionCount);
      // f32 atan2 may move a pair lying within a few ulps of a sector edge.
      let countDifference = 0;
      for (let bin = 0; bin < binCount; bin++) {
        countDifference += Math.abs(result.pairCounts[bin] - oracle.pairCounts[bin]);
        if (result.pairCounts[bin] === oracle.pairCounts[bin]) {
          expectClose(
            `offset ${azimuthOffset} gamma ${bin}`,
            result.semivariances[bin],
            oracle.semivariances[bin],
            oracle.semivarianceTolerance + 2e-5 * Math.abs(oracle.semivariances[bin])
          );
        }
      }
      expect(countDifference).toBeLessThanOrEqual(4);
      expect(
        result.pairCounts.reduce((sum, count) => sum + count, 0),
        'every pair lands in exactly one sector'
      ).toBe(oracle.pairCounts.reduce((sum, count) => sum + count, 0));
    }
    // With offset -pi/8, sector 0 is centred on +x (strong variation) and sector 2 on +y (weak).
    const result = await harness.run(
      getGPUVariogramParameterValues({
        bounds: BOUNDS,
        maximumDistance: 12,
        azimuthOffset: -Math.PI / 8
      })
    );
    const along = result.semivariances[0 * lagCount + 4];
    const across = result.semivariances[2 * lagCount + 4];
    expect(along).toBeGreaterThan(across * 5);
    expect(harness.rebuildCount).toBe(0);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUVariogram reports NaN for empty lags and zero for constant values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Four points on a line 1 apart: lags of width 1 over [0, 8] see pairs only at 1, 2 and 3.
  const positions = new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]);
  const values = new Float32Array([5, 5, 5, 5]);
  const harness = createPairStatisticsHarness(device, {
    positions,
    values,
    parameterLength: GPU_VARIOGRAM_PARAMETER_LENGTH,
    outputs: {
      semivariances: {format: 'float32', length: 8},
      pairCounts: {format: 'uint32', length: 8}
    },
    createRecipe: views =>
      new GPUVariogram({
        positions: views.positions,
        values: views.values!,
        parameters: views.parameters,
        gridSize: [4, 4],
        lagCount: 8,
        semivariances: views.outputs.semivariances as never,
        pairCounts: views.outputs.pairCounts as never
      })
  });
  try {
    const result = await harness.run(
      getGPUVariogramParameterValues({bounds: [0, -1, 3, 1], maximumDistance: 7.999})
    );
    expect(result.pairCounts).toEqual([0, 3, 2, 1, 0, 0, 0, 0]);
    expect(result.semivariances[0]).toBeNaN();
    expect(result.semivariances.slice(1, 4)).toEqual([0, 0, 0]);
    expect(result.semivariances[7]).toBeNaN();
  } finally {
    harness.destroy();
  }
});
