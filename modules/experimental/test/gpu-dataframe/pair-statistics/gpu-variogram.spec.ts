// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUVariogram} from '../../../src/gpu-dataframe/pair-statistics/gpu-variogram';
import {getPairStatisticsUnitVarianceOnCPU} from './pair-statistics-unit-oracle';
import {
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH,
  type GPUVariogramSettings
} from '../../../src/gpu-dataframe/pair-statistics/variogram-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom, forEachPair, isNearBinEdge} from './pair-statistics-oracle';
import {computeVariogramOnCPU} from './variogram-oracle';
import {createVariogramOutlierScene, VARIOGRAM_OUTLIER_FRAME} from './variogram-outlier-fixture';
import {VARIOGRAM_OUTLIER_REFERENCES} from './variogram-outlier-reference';
import {getVariogramReferenceMismatches, VARIOGRAM_REFERENCE} from './variogram-reference';
import {VARIOGRAM_EDGE_POLICY} from '../dataframe-edge-policy';
import {requireTestDevice} from '../../utils/require-test-device';

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
  directionCount: number,
  gridSize: readonly [number, number] = [32, 32]
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
      statistics: {format: 'float32', length: GPU_VARIOGRAM_STATISTICS_LENGTH},
      overflow: {format: 'uint32', length: 1}
    },
    createContributor: views =>
      new GPUVariogram({
        positions: views.positions,
        values: views.values!,
        mask: views.mask,
        parameters: views.parameters,
        gridSize,
        lagCount,
        directionCount,
        output: {
          semivariances: views.outputs.semivariances as never,
          pairCounts: views.outputs.pairCounts as never,
          meanDistances: views.outputs.meanDistances as never,
          robustSemivariances: views.outputs.robustSemivariances as never,
          statistics: views.outputs.statistics as never,
          status: {overflow: views.outputs.overflow as never}
        }
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
  requireTestDevice(device);
  const scene = createAnisotropicScene(5, 2500);
  const lagCount = 12;
  const harness = createHarness(device, scene, lagCount, 1);
  try {
    const frames: GPUVariogramSettings[] = [
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
      expect(result.overflow, `${label} overflow`).toEqual([0]);
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
  requireTestDevice(device);
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
      // A-25: f32 atan2 (and f32 lag division) may move a pair lying within a few ulps of a sector
      // (or lag) edge into the neighboring bin, changing two counts by one. Bound the difference by
      // twice the pairs within 1e-5 rad of a sector edge (about 80 f32 ulps of pi; the GPU angle
      // error is a few ulps) or near a lag edge, instead of an unexplained 4.
      const sectorWidth = Math.PI / directionCount;
      let edgePairs = 0;
      forEachPair(scene, frame, 'unordered', ({deltaX, deltaY, distance}) => {
        const angle = Math.atan2(deltaY, deltaX) - Math.fround(azimuthOffset);
        const scaled = (angle - Math.floor(angle / Math.PI) * Math.PI) / sectorWidth;
        const nearSectorEdge = Math.abs(scaled - Math.round(scaled)) * sectorWidth < 1e-5;
        if (nearSectorEdge || isNearBinEdge(distance, frame.maximumDistance, lagCount)) {
          edgePairs++;
        }
      });
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
      expect(
        countDifference,
        `offset ${azimuthOffset} edge pairs ${edgePairs}`
      ).toBeLessThanOrEqual(2 * edgePairs);
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
  requireTestDevice(device);
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
    createContributor: views =>
      new GPUVariogram({
        positions: views.positions,
        values: views.values!,
        parameters: views.parameters,
        gridSize: [4, 4],
        lagCount: 8,
        output: {
          semivariances: views.outputs.semivariances as never,
          pairCounts: views.outputs.pairCounts as never
        }
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

it('GPUVariogram bins are bitwise independent of the cell lattice, including direction sectors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The pair loop walks the forward half of the 3x3 cell stencil. Any lattice must meet every pair
  // exactly once: a single cell (plain all-pairs), a thin strip, a lattice finer than the points
  // and a coarse one all have to give the same integer pair sums.
  const scene = createAnisotropicScene(11, 1800);
  const lagCount = 8;
  const directionCount = 4;
  const frame: GPUVariogramSettings = {bounds: BOUNDS, maximumDistance: 9, azimuthOffset: 0.2};
  let reference: Awaited<ReturnType<ReturnType<typeof createHarness>['run']>> | undefined;
  for (const gridSize of [
    [1, 1],
    [1, 40],
    [40, 1],
    [5, 3],
    [64, 64]
  ] as const) {
    const harness = createHarness(device, scene, lagCount, directionCount, gridSize);
    try {
      const result = await harness.run(getGPUVariogramParameterValues(frame));
      const total = result.pairCounts.reduce((sum, count) => sum + count, 0);
      expect(total, `grid ${gridSize}`).toBeGreaterThan(500);
      if (!reference) {
        reference = result;
        continue;
      }
      expect(result.pairCounts, `grid ${gridSize} counts`).toEqual(reference.pairCounts);
      expect(getFloatBits(result.semivariances), `grid ${gridSize} gamma`).toEqual(
        getFloatBits(reference.semivariances)
      );
      expect(getFloatBits(result.meanDistances), `grid ${gridSize} distance`).toEqual(
        getFloatBits(reference.meanDistances)
      );
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUVariogram keeps the typical semivariance when one outlier or sentinel inflates the range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // N(0, 1) values on 4000 seeded points plus one outlier (2e4 or 1e5 standard deviations), either
  // isolated (it joins no pair but stretches the value range: a range-pegged unit rounded every
  // typical term to 0 and lost 83% to 100% of the semivariance) or embedded among its neighbors
  // (huge pair terms go to the exact second channel). Reference: scikit-gstat 1.0.24 Matheron,
  // see variogram-outlier-reference.ts.
  const {bounds, maximumDistance, lagCount} = VARIOGRAM_OUTLIER_FRAME;
  const scenes = {
    clean: createVariogramOutlierScene(),
    'isolated-2e4': createVariogramOutlierScene(2e4, 'isolated'),
    'isolated-1e5': createVariogramOutlierScene(1e5, 'isolated'),
    'embedded-2e4': createVariogramOutlierScene(2e4, 'embedded'),
    'embedded-1e5': createVariogramOutlierScene(1e5, 'embedded')
  };
  for (const [name, scene] of Object.entries(scenes)) {
    const reference = VARIOGRAM_OUTLIER_REFERENCES[name];
    const harness = createPairStatisticsHarness(device, {
      ...scene,
      parameterLength: GPU_VARIOGRAM_PARAMETER_LENGTH,
      outputs: {
        semivariances: {format: 'float32', length: lagCount},
        pairCounts: {format: 'uint32', length: lagCount},
        overflow: {format: 'uint32', length: 1}
      },
      createContributor: views =>
        new GPUVariogram({
          positions: views.positions,
          values: views.values!,
          parameters: views.parameters,
          gridSize: [32, 32],
          lagCount,
          output: {
            semivariances: views.outputs.semivariances as never,
            pairCounts: views.outputs.pairCounts as never,
            status: {overflow: views.outputs.overflow as never}
          }
        })
    });
    try {
      const result = await harness.run(
        getGPUVariogramParameterValues({bounds: [...bounds], maximumDistance})
      );
      expect(result.overflow, name).toEqual([0]);
      const unit = getPairStatisticsUnitVarianceOnCPU(scene.values);
      let compared = 0;
      for (let lag = 0; lag < lagCount; lag++) {
        // GPU division rounding may move a pair lying on a lag edge into the neighboring lag.
        if (result.pairCounts[lag] !== reference.pairCounts[lag]) {
          continue;
        }
        compared++;
        const expected = reference.semivariances[lag];
        expectClose(
          `${name} gamma ${lag}`,
          result.semivariances[lag],
          expected,
          unit * 2 ** -25 + 2e-5 * Math.abs(expected)
        );
      }
      expect(compared, name).toBeGreaterThanOrEqual(6);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUVariogram matches the scikit-gstat references, including an embedded 2e4 sentinel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fixture: test/oracles/dataframe-variogram.py (scikit-gstat 1.0.24). Lattice positions keep
  // every pair at least nearestEdgeDistance from a lag edge, so pair counts must match exactly;
  // value tolerances are the derived fixed-point bounds stored with each scene.
  const {positions, bounds, maximumDistance, lagCount, scenes} = VARIOGRAM_REFERENCE;
  const mismatches: string[] = [];
  for (const [name, scene] of Object.entries(scenes)) {
    const harness = createHarness(
      device,
      {
        positions: Float32Array.from(positions),
        values: Float32Array.from(scene.values),
        mask: new Uint32Array(scene.values.length).fill(1)
      },
      lagCount,
      1
    );
    try {
      const result = await harness.run(
        getGPUVariogramParameterValues({bounds: [...bounds], maximumDistance})
      );
      expect(result.overflow, name).toEqual([0]);
      mismatches.push(...getVariogramReferenceMismatches(name, result, scene));
    } finally {
      harness.destroy();
    }
  }
  expect(mismatches).toEqual([]);
}, 120000);

it('GPUVariogram follows the variogram edge policy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const row of VARIOGRAM_EDGE_POLICY) {
    const harness = createHarness(
      device,
      {
        positions: Float32Array.from(row.positions),
        values: Float32Array.from(row.values),
        mask: Uint32Array.from(row.mask ?? row.values.map(() => 1))
      },
      2,
      1,
      [4, 4]
    );
    try {
      // Raw parameters: the CPU packer refuses a non-positive maximum distance.
      const parameters = new Float32Array(GPU_VARIOGRAM_PARAMETER_LENGTH);
      parameters.set(row.parameters);
      const result = await harness.run(parameters);
      expect(result.pairCounts, row.name).toEqual(row.expectedPairCounts);
      for (const [lag, expected] of row.expectedSemivariances.entries()) {
        const actual = result.semivariances[lag];
        expect(Object.is(actual, expected), `${row.name} lag ${lag}: ${actual}`).toBe(true);
      }
    } finally {
      harness.destroy();
    }
  }
});
