// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpatialCorrelogram,
  type GPUSpatialCorrelogramBandMode
} from '../../../src/gpu-dataframe/pair-statistics/gpu-spatial-correlogram';
import {
  getGPUSpatialCorrelogramParameterValues,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH,
  type GPUSpatialCorrelogramParameters
} from '../../../src/gpu-dataframe/pair-statistics/spatial-correlogram-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeSpatialCorrelogramOnCPU} from './spatial-correlogram-oracle';

const BOUNDS = [0, 0, 100, 100] as const;

/** Smooth field with noise, plus masked, NaN and out-of-bounds rows. */
function createSmoothScene(seed: number, rows: number) {
  const random = createRandom(seed);
  const positions = new Float32Array(rows * 2);
  const values = new Float32Array(rows);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const x = random() * 104 - 2;
    const y = random() * 104 - 2;
    positions[row * 2] = x;
    positions[row * 2 + 1] = y;
    values[row] = 5 * Math.sin(x / 10) * Math.cos(y / 12) + 2 * (random() - 0.5) + 100;
    mask[row] = random() < 0.92 ? 1 : 0;
    if (random() < 0.02) {
      values[row] = NaN;
    }
  }
  return {positions, values, mask};
}

function createHarness(
  device: Device,
  scene: ReturnType<typeof createSmoothScene>,
  bandCount: number,
  bandMode: GPUSpatialCorrelogramBandMode
) {
  return createPairStatisticsHarness(device, {
    ...scene,
    parameterLength: GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
    outputs: {
      moransI: {format: 'float32', length: bandCount},
      zScores: {format: 'float32', length: bandCount},
      pValues: {format: 'float32', length: bandCount},
      expectedI: {format: 'float32', length: bandCount},
      varianceI: {format: 'float32', length: bandCount},
      pairCounts: {format: 'uint32', length: bandCount},
      peakBands: {format: 'uint32', length: 2},
      statistics: {format: 'float32', length: GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH}
    },
    createContributor: views =>
      new GPUSpatialCorrelogram({
        positions: views.positions,
        values: views.values!,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [32, 32],
        bandCount,
        bandMode,
        moransI: views.outputs.moransI as never,
        zScores: views.outputs.zScores as never,
        pValues: views.outputs.pValues as never,
        expectedI: views.outputs.expectedI as never,
        varianceI: views.outputs.varianceI as never,
        pairCounts: views.outputs.pairCounts as never,
        peakBands: views.outputs.peakBands as never,
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

for (const bandMode of ['cumulative', 'annulus'] as const) {
  it(`GPUSpatialCorrelogram (${bandMode}) matches the f64 oracle across per-frame changes without rebuilding`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createSmoothScene(bandMode === 'cumulative' ? 3 : 4, 2000);
    const bandCount = 10;
    const harness = createHarness(device, scene, bandCount, bandMode);
    try {
      const frames: GPUSpatialCorrelogramParameters[] = [
        {bounds: BOUNDS, maximumDistance: 25},
        {bounds: BOUNDS, maximumDistance: 25, varianceAssumption: 'normality'},
        {bounds: BOUNDS, maximumDistance: 6},
        {bounds: BOUNDS, maximumDistance: 300},
        {bounds: [10, 20, 90, 70], maximumDistance: 40}
      ];
      for (const frame of frames) {
        const label = `${bandMode} ${frame.maximumDistance} ${frame.varianceAssumption ?? ''}`;
        const result = await harness.run(getGPUSpatialCorrelogramParameterValues(frame));
        const oracle = computeSpatialCorrelogramOnCPU(scene, frame, bandCount, bandMode);
        let comparedBands = 0;
        for (let band = 0; band < bandCount; band++) {
          const countDifference = Math.abs(result.pairCounts[band] - oracle.pairCounts[band]);
          expect(countDifference, `${label} count ${band}`).toBeLessThanOrEqual(
            Math.ceil(oracle.edgePairCounts[band])
          );
          if (countDifference > 0) {
            continue;
          }
          comparedBands++;
          const expectedI = oracle.moransI[band];
          expectClose(
            `${label} I ${band}`,
            result.moransI[band],
            expectedI,
            oracle.moransITolerance + 1e-4 * Math.abs(expectedI) + 1e-6
          );
          expectClose(`${label} E ${band}`, result.expectedI[band], oracle.expectedI[band], 1e-9);
          expectClose(
            `${label} Var ${band}`,
            result.varianceI[band],
            oracle.varianceI[band],
            2e-3 * Math.abs(oracle.varianceI[band])
          );
          expectClose(
            `${label} z ${band}`,
            result.zScores[band],
            oracle.zScores[band],
            2e-3 + 2e-3 * Math.abs(oracle.zScores[band])
          );
          expectClose(
            `${label} p ${band}`,
            result.pValues[band],
            oracle.pValues[band],
            1e-5 + 5e-3 * oracle.pValues[band]
          );
        }
        expect(comparedBands, label).toBeGreaterThanOrEqual(bandCount - 2);
        expect(result.statistics[0], label).toBe(oracle.statistics[0]);
        for (let slot = 1; slot < GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH; slot++) {
          expectClose(
            `${label} statistics ${slot}`,
            result.statistics[slot],
            oracle.statistics[slot],
            1e-4 * Math.abs(oracle.statistics[slot]) + 1e-5
          );
        }
        if (comparedBands === bandCount) {
          expect(result.peakBands, label).toEqual(oracle.peakBands);
        }
        const again = await harness.run(getGPUSpatialCorrelogramParameterValues(frame));
        expect(getFloatBits(again.moransI)).toEqual(getFloatBits(result.moransI));
        expect(getFloatBits(again.zScores)).toEqual(getFloatBits(result.zScores));
      }
      // The smooth field is strongly positively autocorrelated at short range.
      const short = await harness.run(
        getGPUSpatialCorrelogramParameterValues({bounds: BOUNDS, maximumDistance: 25})
      );
      expect(short.moransI[0]).toBeGreaterThan(0.5);
      expect(short.zScores[0]).toBeGreaterThan(10);
      expect(harness.rebuildCount).toBe(0);
    } finally {
      harness.destroy();
    }
  }, 120000);
}

it('GPUSpatialCorrelogram reports NaN for empty bands and too few rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = {
    positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
    values: new Float32Array([1, 2, 3, 4]),
    mask: new Uint32Array([1, 1, 1, 1])
  };
  const harness = createHarness(device, scene, 4, 'annulus');
  try {
    // Annuli of width 1 over [0, 4]: lag-1 pairs sit in annulus 1, none in annulus 0.
    const frame = {
      bounds: [0, -1, 3, 1] as const,
      maximumDistance: 4,
      varianceAssumption: 'normality' as const
    };
    const result = await harness.run(getGPUSpatialCorrelogramParameterValues(frame));
    const oracle = computeSpatialCorrelogramOnCPU(scene, frame, 4, 'annulus');
    expect(result.pairCounts).toEqual([0, 3, 2, 1]);
    expect(result.moransI[0]).toBeNaN();
    expect(result.moransI[1]).toBeCloseTo(oracle.moransI[1], 5);
    expect(result.varianceI[1]).toBeCloseTo(oracle.varianceI[1], 5);
    // Two rows in a narrow window: n < 3 leaves every band undefined.
    const narrow = await harness.run(
      getGPUSpatialCorrelogramParameterValues({bounds: [0, -1, 1, 1], maximumDistance: 4})
    );
    expect(narrow.moransI.every(Number.isNaN)).toBe(true);
    expect(narrow.peakBands).toEqual([0xffffffff, 0xffffffff]);
  } finally {
    harness.destroy();
  }
});
