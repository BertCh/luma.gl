// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  type GPUSpatialAutocorrelationParameters
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  computeHotSpotOracle,
  createAutocorrelatedScene,
  createDistanceBandWeights,
  createSeededRandom,
  getFalseDiscoveryRateLevels,
  getFalseDiscoveryRateMargin,
  type HotSpotOracleResult,
  type OracleWeights
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

function toCsr(weights: OracleWeights) {
  return {kind: 'csr' as const, ...weights};
}
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

it('GPUHotSpotAnalysis matches the Gi* oracle for hand-built CSR weights, self weights and new values without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(7, 2000, RADII);
  let significantRows = 0;
  for (const [radius, selfWeight, rowStandardize] of [
    [0.8, 1, false],
    [3, 1, false],
    [7.5, 1, false],
    [7.5, 0, false],
    [15, 1, false],
    [7.5, 0.5, true],
    [40, 1, false]
  ] as const) {
    const weights = createDistanceBandWeights(scene.positions, radius, {rowStandardize});
    const harness = createSpatialAutocorrelationHarness(device, {
      contributor: 'hot-spot',
      scene,
      weights: toCsr(weights),
      selfWeight
    });
    try {
      const input = {...scene, weights, selfWeight};
      const label = `radius ${radius} self ${selfWeight} standardized ${rowStandardize}`;
      const result = await harness.run();
      expectMatchesOracle(result, computeHotSpotOracle(input), label);
      significantRows += computeHotSpotOracle(input).bins.filter(bin => bin !== 0).length;
      // Repeated encodings are bitwise identical.
      expect((await harness.run()).zScoreBits).toEqual(result.zScoreBits);
      if (radius === 7.5 && selfWeight === 1) {
        // The scene has real hot and cold spots.
        const oracle = computeHotSpotOracle(input);
        expect(oracle.bins.some(bin => bin === 3)).toBe(true);
        expect(oracle.bins.some(bin => bin === -3)).toBe(true);
        // New values in the same compiled graph.
        const shifted = scene.values.map((value, row) => value * 0.5 + (row % 7));
        harness.writeValues(shifted);
        expectMatchesOracle(
          await harness.run(),
          computeHotSpotOracle({...input, values: shifted}),
          'new values'
        );
        expect(harness.buildCount).toBe(1);
      }
    } finally {
      harness.destroy();
    }
  }
  expect(significantRows).toBeGreaterThan(0);
}, 240000);

it('GPUHotSpotAnalysis consumes weights written by GPUNeighborSearch in the same graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(13, 2000, [3, 7.5, 12]);
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene,
    weights: {
      kind: 'neighbor-search',
      positions: scene.positions,
      parameters: {bounds: BOUNDS, radius: 3},
      capacity: 2000 * 160
    }
  });
  try {
    // The search radius and the weight transform change per frame without a rebuild.
    for (const [radius, rowStandardize] of [
      [3, false],
      [7.5, false],
      [12, false],
      [7.5, true]
    ] as const) {
      const result = await harness.run(undefined, {bounds: BOUNDS, radius, rowStandardize});
      const expected = createDistanceBandWeights(scene.positions, radius, {rowStandardize});
      expect(result.csr.offsets, `radius ${radius} offsets`).toEqual(Array.from(expected.offsets));
      expect(result.csr.neighbors, `radius ${radius} neighbors`).toEqual(
        Array.from(expected.neighbors)
      );
      const weights = {
        offsets: Uint32Array.from(result.csr.offsets),
        neighbors: Uint32Array.from(result.csr.neighbors),
        weights: Float32Array.from(result.csr.weights)
      };
      const oracle = computeHotSpotOracle({...scene, weights});
      expectMatchesOracle(result, oracle, `radius ${radius} standardized ${rowStandardize}`);
      expect(oracle.bins.some(bin => bin !== 0)).toBe(true);
    }
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
  const random = createSeededRandom(5);
  const mask = Uint32Array.from({length: values.length}, () => (random() < 0.6 ? 1 : 0));
  const scene = {values, mask};
  const weights = createDistanceBandWeights(base.positions, 6);
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene,
    weights: toCsr(weights)
  });
  try {
    const masked = computeHotSpotOracle({...scene, weights});
    const maskedResult = await harness.run();
    expectMatchesOracle(maskedResult, masked, 'masked');
    for (const row of [3, 10]) {
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
    const full = await harness.run();
    expectMatchesOracle(full, computeHotSpotOracle({...scene, mask: everyRow, weights}), 'full');
    harness.writeMask(mask);
    const fixedParameters: GPUSpatialAutocorrelationParameters = {
      fixedMoments: {
        count: full.globalStatistics[0],
        mean: full.globalStatistics[1],
        variance: full.globalStatistics[2]
      }
    };
    const fixed = await harness.run(fixedParameters);
    const fixedOracle = computeHotSpotOracle({...scene, weights, parameters: fixedParameters});
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
  for (const radius of [5, 9]) {
    const weights = createDistanceBandWeights(scene.positions, radius);
    const harness = createSpatialAutocorrelationHarness(device, {
      contributor: 'hot-spot',
      scene,
      weights: toCsr(weights),
      falseDiscoveryRate: true
    });
    try {
      const result = await harness.run();
      const oracle = computeHotSpotOracle({...scene, weights});
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
      expect((await harness.run()).bins).toEqual(result.bins);
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUHotSpotAnalysis handles islands, degenerate selections and out-of-range neighbors', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createAutocorrelatedScene(3, 64, [10]);
  const mask = new Uint32Array(64);
  mask[5] = 1;
  const clique = createDistanceBandWeights(scene.positions, 1000);
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene: {...scene, mask},
    weights: toCsr(clique)
  });
  try {
    // One included row: n < 2, so even that row has no z-score.
    const single = await harness.run();
    expect(single.zScores.every(Number.isNaN)).toBe(true);
    expect(single.neighborCounts[5]).toBe(1);
    expect(single.globalStatistics[0]).toBe(1);
    harness.writeMask(new Uint32Array(64).fill(1));
    // Every row weighs every other row: k = n, so z is undefined.
    const everything = await harness.run();
    expect(everything.zScores.every(Number.isNaN)).toBe(true);
    expect(everything.neighborCounts.every(count => count === 64)).toBe(true);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }

  // No neighbors at all (plus one out-of-range ID, which is ignored): with the focal weight each
  // row is its own neighborhood, so z = (x - X) / S; without it z is undefined.
  const rows = 64;
  const islands: OracleWeights = {
    offsets: Uint32Array.from({length: rows + 1}, (_, row) => (row > 0 ? 1 : 0)),
    neighbors: Uint32Array.from([9999]),
    weights: Float32Array.from([1])
  };
  for (const selfWeight of [1, 0]) {
    const islandHarness = createSpatialAutocorrelationHarness(device, {
      contributor: 'hot-spot',
      scene,
      weights: toCsr(islands),
      selfWeight
    });
    try {
      const result = await islandHarness.run();
      const oracle = computeHotSpotOracle({...scene, weights: islands, selfWeight});
      expectMatchesOracle(result, oracle, `islands self ${selfWeight}`);
      expect(result.zScores.every(Number.isNaN)).toBe(selfWeight === 0);
    } finally {
      islandHarness.destroy();
    }
  }
}, 60000);
