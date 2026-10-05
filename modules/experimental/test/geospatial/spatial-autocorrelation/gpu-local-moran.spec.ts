// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_LOCAL_MORAN_QUADRANT,
  type GPUSpatialAutocorrelationParameters
} from '../../../src/geospatial/spatial-autocorrelation';
import {
  computeLocalMoranOracle,
  createAutocorrelatedScene,
  createSeededRandom,
  getFalseDiscoveryRateLevels,
  getFalseDiscoveryRateMargin,
  getQuadrant,
  type LocalMoranOracleResult
} from './spatial-autocorrelation-oracle';
import {
  createSpatialAutocorrelationHarness,
  isClose,
  type SpatialAutocorrelationReadback
} from './spatial-autocorrelation-harness';

/** z-scores agree with the double-precision oracle within `2e-3 + 1e-3 |z|`. */
const Z_ABSOLUTE_TOLERANCE = 2e-3;
const Z_RELATIVE_TOLERANCE = 1e-3;

/** Rows whose oracle centered value is closer than this to 0 are compared by |z| only. */
const CENTERED_GUARD = 1e-3;

const BOUNDS = [-1, -1, 101, 101] as const;
const RADII = [0.8, 3, 7.5, 15, 40] as const;

function expectMatchesOracle(
  result: SpatialAutocorrelationReadback,
  oracle: LocalMoranOracleResult,
  oracleValues: Float32Array,
  significanceLevel: number,
  label: string
): void {
  expect(result.neighborCounts, label).toEqual(oracle.neighborCounts);
  expect(result.globalStatistics[0], label).toBe(oracle.moments.count);
  let comparedQuadrants = 0;
  for (const [row, expected] of oracle.zScores.entries()) {
    // z takes the sign of the centered value, so it flips for values within f32 rounding of the
    // mean (the GPU mean is f32). |z| and p are continuous there and are always compared.
    const centered = oracle.included[row] ? oracleValues[row] - oracle.moments.mean : NaN;
    const nearMean = Math.abs(centered) < CENTERED_GUARD;
    const actualZ = nearMean ? Math.abs(result.zScores[row]) : result.zScores[row];
    const expectedZ = nearMean ? Math.abs(expected) : expected;
    if (!isClose(actualZ, expectedZ, Z_ABSOLUTE_TOLERANCE, Z_RELATIVE_TOLERANCE)) {
      throw new Error(`${label}: row ${row} z ${result.zScores[row]} != oracle ${expected}`);
    }
    if (!isClose(result.pValues[row], oracle.pValues[row], 1e-5, 2e-3)) {
      throw new Error(`${label}: row ${row} p ${result.pValues[row]} != ${oracle.pValues[row]}`);
    }
    if (!isClose(result.spatialLag[row], oracle.spatialLag[row], 2e-4, 1e-3)) {
      throw new Error(
        `${label}: row ${row} lag ${result.spatialLag[row]} != ${oracle.spatialLag[row]}`
      );
    }
    if (!isClose(result.localI[row], oracle.localI[row], 2e-3 + (nearMean ? 1e-3 : 0), 2e-3)) {
      throw new Error(`${label}: row ${row} I ${result.localI[row]} != ${oracle.localI[row]}`);
    }
    // Quadrants are compared where neither the p-value nor a sign sits on a decision boundary.
    const pValue = oracle.pValues[row];
    const nearBoundary =
      nearMean ||
      (Number.isFinite(pValue) &&
        Math.abs(pValue - significanceLevel) < 2e-3 * significanceLevel + 1e-5) ||
      // Islands have a lag of exactly 0 on both sides, so only nonzero small lags are ambiguous.
      (oracle.neighborCounts[row] > 0 && Math.abs(oracle.spatialLag[row]) < 1e-3);
    if (!nearBoundary) {
      expect(result.quadrants[row], `${label}: row ${row} quadrant`).toBe(oracle.quadrants[row]);
      comparedQuadrants++;
    }
  }
  expect(comparedQuadrants, label).toBeGreaterThan(oracle.zScores.length * 0.95);
}

it('GPULocalMoran matches the oracle across per-frame radius, level, and weight changes without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(11, 2000, RADII);
  const frames: GPUSpatialAutocorrelationParameters[] = [
    ...RADII.map(radius => ({bounds: BOUNDS, radius})),
    {bounds: BOUNDS, radius: 7.5, significanceLevel: 0.01},
    {bounds: BOUNDS, radius: 7.5, weightTransform: 'binary'},
    {bounds: [10, 10, 70, 90], radius: 5}
  ];
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'local-moran',
    scene,
    parameters: frames[0]
  });
  try {
    for (const frame of frames) {
      const oracle = computeLocalMoranOracle({...scene, parameters: frame});
      const result = await harness.run(frame);
      expectMatchesOracle(
        result,
        oracle,
        scene.values,
        frame.significanceLevel ?? 0.05,
        `radius ${frame.radius}`
      );
      expect((await harness.run(frame)).zScoreBits).toEqual(result.zScoreBits);
    }
    const oracle = computeLocalMoranOracle({...scene, parameters: frames[2]});
    const {HIGH_HIGH, LOW_LOW} = GPU_LOCAL_MORAN_QUADRANT;
    expect(oracle.quadrants.some(quadrant => quadrant === HIGH_HIGH)).toBe(true);
    expect(oracle.quadrants.some(quadrant => quadrant === LOW_LOW)).toBe(true);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPULocalMoran finds a planted spatial outlier and honors the mask and fixed moments', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(23, 1500, [8]);
  const values = scene.values.slice();
  // Plant a low outlier at the row closest to the strongest hot bump.
  let outlier = 0;
  let closest = Infinity;
  for (let row = 0; row < values.length; row++) {
    const distance =
      (scene.positions[row * 2] - 25) ** 2 + (scene.positions[row * 2 + 1] - 30) ** 2;
    if (distance < closest) {
      closest = distance;
      outlier = row;
    }
  }
  values[outlier] = 80;
  const random = createSeededRandom(9);
  const mask = Uint32Array.from({length: values.length}, () => (random() < 0.7 ? 1 : 0));
  mask[outlier] = 1;
  const parameters: GPUSpatialAutocorrelationParameters = {bounds: BOUNDS, radius: 8};
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'local-moran',
    scene: {positions: scene.positions, values, mask},
    parameters
  });
  try {
    const result = await harness.run(parameters);
    const oracle = computeLocalMoranOracle({positions: scene.positions, values, mask, parameters});
    expectMatchesOracle(result, oracle, values, 0.05, 'masked');
    expect(result.quadrants[outlier]).toBe(GPU_LOCAL_MORAN_QUADRANT.LOW_HIGH);
    expect(result.localI[outlier]).toBeLessThan(0);
    expect(mask.some((flag, row) => flag === 0 && !Number.isNaN(result.zScores[row]))).toBe(false);

    const fixedParameters: GPUSpatialAutocorrelationParameters = {
      ...parameters,
      fixedMoments: {count: 2000, mean: 99, variance: 9}
    };
    const fixed = await harness.run(fixedParameters);
    expectMatchesOracle(
      fixed,
      computeLocalMoranOracle({
        positions: scene.positions,
        values,
        mask,
        parameters: fixedParameters
      }),
      values,
      0.05,
      'fixed moments'
    );
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPULocalMoran applies Benjamini-Hochberg FDR to quadrant significance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(41, 2500, [6]);
  for (const significanceLevel of [0.05, 0.1]) {
    const parameters: GPUSpatialAutocorrelationParameters = {
      bounds: BOUNDS,
      radius: 6,
      significanceLevel
    };
    const harness = createSpatialAutocorrelationHarness(device, {
      contributor: 'local-moran',
      scene,
      parameters,
      falseDiscoveryRate: true
    });
    try {
      const result = await harness.run(parameters);
      expect(getFalseDiscoveryRateMargin(result.zScores, [significanceLevel])).toBeGreaterThan(
        1e-4
      );
      const levels = getFalseDiscoveryRateLevels(result.zScores, [significanceLevel]);
      const centeredValues = Array.from(scene.values, value => value - result.globalStatistics[1]);
      const expected = levels.map((level, row) =>
        level > 0 ? getQuadrant(centeredValues[row], result.spatialLag[row]) : 0
      );
      expect(result.quadrants).toEqual(expected);
      const corrected = expected.filter(quadrant => quadrant !== 0).length;
      const uncorrected = result.pValues.filter(pValue => pValue <= significanceLevel).length;
      expect(corrected).toBeGreaterThan(0);
      expect(corrected).toBeLessThanOrEqual(uncorrected);
    } finally {
      harness.destroy();
    }
  }
}, 120000);
