// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  type GPUSpatialAutocorrelationParameters
} from '../../../src/map-graphs/spatial-autocorrelation';
import {
  computeHotSpotOracle,
  createAutocorrelatedScene,
  createSeededRandom,
  getFalseDiscoveryRateLevels,
  getFalseDiscoveryRateMargin,
  type HotSpotOracleResult
} from './spatial-autocorrelation-oracle';
import {
  createSpatialAutocorrelationHarness,
  isClose,
  type SpatialAutocorrelationReadback
} from './spatial-autocorrelation-harness';

/** z-scores agree with the double-precision oracle within `2e-3 + 1e-3 |z|`. */
const Z_ABSOLUTE_TOLERANCE = 2e-3;
const Z_RELATIVE_TOLERANCE = 1e-3;
/** Bins are compared only where the oracle z is farther than this from a critical value. */
const BIN_GUARD = 1e-2;

const BOUNDS = [-1, -1, 101, 101] as const;
const RADII = [0.8, 3, 7.5, 15, 40] as const;

function expectMatchesOracle(
  result: SpatialAutocorrelationReadback,
  oracle: HotSpotOracleResult,
  label: string
): void {
  expect(result.neighborCounts, label).toEqual(oracle.neighborCounts);
  expect(result.globalStatistics[0], label).toBe(oracle.moments.count);
  expect(isClose(result.globalStatistics[1], oracle.moments.mean, 1e-4, 1e-6), label).toBe(true);
  expect(isClose(result.globalStatistics[2], oracle.moments.variance, 1e-5, 1e-4), label).toBe(
    true
  );
  let comparedBins = 0;
  for (const [row, expected] of oracle.zScores.entries()) {
    const actual = result.zScores[row];
    if (!isClose(actual, expected, Z_ABSOLUTE_TOLERANCE, Z_RELATIVE_TOLERANCE)) {
      throw new Error(`${label}: row ${row} z ${actual} != oracle ${expected}`);
    }
    if (!isClose(result.pValues[row], oracle.pValues[row], 1e-5, 2e-3)) {
      throw new Error(`${label}: row ${row} p ${result.pValues[row]} != ${oracle.pValues[row]}`);
    }
    const nearCritical = GPU_HOT_SPOT_CRITICAL_Z_SCORES.some(
      critical => Math.abs(Math.abs(expected) - critical) < BIN_GUARD
    );
    if (!nearCritical) {
      expect(result.bins[row], `${label}: row ${row} bin`).toBe(oracle.bins[row]);
      comparedBins++;
    }
  }
  expect(comparedBins, label).toBeGreaterThan(oracle.zScores.length * 0.95);
}

it('GPUHotSpotAnalysis matches the Gi* oracle across per-frame radius and bounds changes without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(7, 2000, RADII);
  const frames: GPUSpatialAutocorrelationParameters[] = [
    ...RADII.map(radius => ({bounds: BOUNDS, radius})),
    // A tighter extent excludes rows outside it.
    {bounds: [20, 20, 80, 80], radius: 7.5}
  ];
  const harness = createSpatialAutocorrelationHarness(device, {
    recipe: 'hot-spot',
    scene,
    parameters: frames[0]
  });
  try {
    let significantRows = 0;
    for (const frame of frames) {
      const oracle = computeHotSpotOracle({...scene, parameters: frame});
      const result = await harness.run(frame);
      expectMatchesOracle(result, oracle, `radius ${frame.radius}`);
      significantRows += oracle.bins.filter(bin => bin !== 0).length;
      // Repeated encodings are bitwise identical.
      expect((await harness.run(frame)).zScoreBits).toEqual(result.zScoreBits);
    }
    // The scene has real hot and cold spots.
    const oracle = computeHotSpotOracle({...scene, parameters: frames[2]});
    expect(oracle.bins.some(bin => bin === 3)).toBe(true);
    expect(oracle.bins.some(bin => bin === -3)).toBe(true);
    expect(significantRows).toBeGreaterThan(0);
    // New values in the same compiled graph.
    const shifted = scene.values.map((value, row) => value * 0.5 + (row % 7));
    harness.writeValues(shifted);
    const oracleShifted = computeHotSpotOracle({...scene, values: shifted, parameters: frames[2]});
    expectMatchesOracle(await harness.run(frames[2]), oracleShifted, 'new values');
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUHotSpotAnalysis honors the mask, excluded values, and fixed moments', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const base = createAutocorrelatedScene(19, 1500, [6]);
  const values = base.values.slice();
  values[3] = NaN;
  values[10] = Infinity;
  const positions = base.positions.slice();
  positions[2 * 20] = NaN;
  const random = createSeededRandom(5);
  const mask = Uint32Array.from({length: values.length}, () => (random() < 0.6 ? 1 : 0));
  const scene = {positions, values, mask};
  const parameters: GPUSpatialAutocorrelationParameters = {bounds: BOUNDS, radius: 6};
  const harness = createSpatialAutocorrelationHarness(device, {
    recipe: 'hot-spot',
    scene,
    parameters
  });
  try {
    const masked = computeHotSpotOracle({...scene, parameters});
    const maskedResult = await harness.run(parameters);
    expectMatchesOracle(maskedResult, masked, 'masked');
    for (const row of [3, 10, 20]) {
      expect(Number.isNaN(maskedResult.zScores[row])).toBe(true);
      expect(maskedResult.bins[row]).toBe(0);
      expect(maskedResult.neighborCounts[row]).toBe(0);
    }
    expect(mask.some((flag, row) => flag === 0 && !Number.isNaN(maskedResult.zScores[row]))).toBe(
      false
    );

    // Full selection: its global statistics become the fixed moments of a viewport-like subset.
    const everyRow = new Uint32Array(values.length).fill(1);
    harness.writeMask(everyRow);
    const full = await harness.run(parameters);
    expectMatchesOracle(full, computeHotSpotOracle({...scene, mask: everyRow, parameters}), 'full');
    harness.writeMask(mask);
    const fixedParameters: GPUSpatialAutocorrelationParameters = {
      ...parameters,
      fixedMoments: {
        count: full.globalStatistics[0],
        mean: full.globalStatistics[1],
        variance: full.globalStatistics[2]
      }
    };
    const fixed = await harness.run(fixedParameters);
    const fixedOracle = computeHotSpotOracle({...scene, parameters: fixedParameters});
    expectMatchesOracle(fixed, fixedOracle, 'fixed moments');
    expect(fixed.globalStatistics.slice(0, 3)).toEqual(full.globalStatistics.slice(0, 3));
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUHotSpotAnalysis applies Benjamini-Hochberg FDR to the bins', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(31, 2500, [5, 9]);
  const harness = createSpatialAutocorrelationHarness(device, {
    recipe: 'hot-spot',
    scene,
    parameters: {bounds: BOUNDS, radius: 5},
    falseDiscoveryRate: true
  });
  try {
    for (const radius of [5, 9]) {
      const parameters = {bounds: BOUNDS, radius};
      const result = await harness.run(parameters);
      const oracle = computeHotSpotOracle({...scene, parameters});
      for (const [row, expected] of oracle.zScores.entries()) {
        expect(
          isClose(result.zScores[row], expected, Z_ABSOLUTE_TOLERANCE, Z_RELATIVE_TOLERANCE)
        ).toBe(true);
      }
      // The decision is exact given the GPU z-scores, as long as no p-value sits on its BH bound.
      expect(
        getFalseDiscoveryRateMargin(result.zScores, GPU_HOT_SPOT_SIGNIFICANCE_LEVELS)
      ).toBeGreaterThan(1e-4);
      const levels = getFalseDiscoveryRateLevels(result.zScores, GPU_HOT_SPOT_SIGNIFICANCE_LEVELS);
      const expectedBins = levels.map(
        (level, row) => (result.zScores[row] < 0 ? -level : level) || 0
      );
      expect(result.bins, `radius ${radius}`).toEqual(expectedBins);
      const corrected = expectedBins.filter(bin => bin !== 0).length;
      const uncorrected = oracle.bins.filter(bin => bin !== 0).length;
      expect(corrected).toBeGreaterThan(0);
      expect(corrected).toBeLessThanOrEqual(uncorrected);
      expect((await harness.run(parameters)).bins).toEqual(result.bins);
    }
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUHotSpotAnalysis writes NaN for invalid parameters and degenerate selections', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(3, 64, [10]);
  const mask = new Uint32Array(64);
  mask[5] = 1;
  const harness = createSpatialAutocorrelationHarness(device, {
    recipe: 'hot-spot',
    scene: {...scene, mask},
    parameters: {bounds: BOUNDS, radius: 10}
  });
  try {
    // One included row: n < 2, so even that row has no z-score.
    const single = await harness.run({bounds: BOUNDS, radius: 10});
    expect(single.zScores.every(Number.isNaN)).toBe(true);
    expect(single.neighborCounts[5]).toBe(1);
    expect(single.globalStatistics[0]).toBe(1);
    harness.writeMask(new Uint32Array(64).fill(1));
    // Radius 0 bypassing the packer excludes every row.
    const zeroRadius = new Float32Array([...BOUNDS, 0, 0.05, 1, 0, 0, 0, 0, 0]);
    const invalid = await harness.run(zeroRadius);
    expect(invalid.zScores.every(Number.isNaN)).toBe(true);
    expect(invalid.bins.every(bin => bin === 0)).toBe(true);
    expect(invalid.neighborCounts.every(count => count === 0)).toBe(true);
    expect(invalid.globalStatistics[0]).toBe(0);
    // Every row within the band of every other: k = n, so z is undefined.
    const everything = await harness.run({bounds: BOUNDS, radius: 1000});
    expect(everything.zScores.every(Number.isNaN)).toBe(true);
    expect(everything.neighborCounts.every(count => count === 64)).toBe(true);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 60000);
