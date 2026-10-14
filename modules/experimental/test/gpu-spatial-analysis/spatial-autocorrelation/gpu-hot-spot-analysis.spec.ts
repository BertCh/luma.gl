// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  type GPUSpatialAutocorrelationSettings
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts';
import {
  computeHotSpotOracle,
  createAutocorrelatedScene,
  createDistanceBandWeights,
  createSeededRandom,
  getAdjustedPValueMargin,
  getAdjustedPValuesFromZScores,
  type HotSpotOracleResult,
  type OracleWeights
} from './spatial-autocorrelation-oracle';
import {
  getReferenceMask,
  getReferenceValues,
  getReferenceWeights,
  REFERENCE_ROW_COUNT
} from './spatial-autocorrelation-fixture';
import {REFERENCE} from './spatial-autocorrelation-reference';
import {
  createSpatialAutocorrelationHarness,
  isClose,
  type SpatialAutocorrelationReadback
} from './spatial-autocorrelation-harness';
import {requireTestDevice} from '../../utils/require-test-device';

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
  expect(result.globalStatistics[4], `${label} islandCount`).toBe(oracle.islandCount);
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
  requireTestDevice(device);
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
      selfWeight,
      rowStandardized: rowStandardize
    });
    try {
      const input = {...scene, weights, selfWeight, rowStandardized: rowStandardize};
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
  requireTestDevice(device);
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
  requireTestDevice(device);
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
    const fixedParameters: GPUSpatialAutocorrelationSettings = {
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

it('GPUHotSpotAnalysis bins follow the adjusted p-value of every multiple-testing method', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createAutocorrelatedScene(31, 2500, [5, 9]);
  const levels = GPU_HOT_SPOT_SIGNIFICANCE_LEVELS;
  for (const radius of [5, 9]) {
    const weights = createDistanceBandWeights(scene.positions, radius);
    const uncorrected = new Map<GPUMultipleTestingMethod, number>();
    for (const method of [
      'none',
      'bonferroni',
      'holm',
      'benjamini-hochberg',
      'benjamini-yekutieli'
    ] as const) {
      const harness = createSpatialAutocorrelationHarness(device, {
        contributor: 'hot-spot',
        scene,
        weights: toCsr(weights),
        multipleTesting: method
      });
      try {
        const result = await harness.run();
        const oracle = computeHotSpotOracle({...scene, weights}, {multipleTesting: method});
        for (const [row, expected] of oracle.zScores.entries()) {
          expect(
            isClose(result.zScores[row], expected, Z_ABSOLUTE_TOLERANCE, Z_RELATIVE_TOLERANCE)
          ).toBe(true);
        }
        // The decision is exact given the GPU z-scores, as long as no adjusted p-value sits on a
        // level.
        const adjusted = getAdjustedPValuesFromZScores(result.zScores, method);
        if (method !== 'none') {
          expect(getAdjustedPValueMargin(adjusted, levels)).toBeGreaterThan(1e-4);
          for (const [row, expectedAdjusted] of adjusted.entries()) {
            expect(
              isClose(result.adjustedPValues[row], expectedAdjusted, 1e-6, 1e-4),
              `${method} radius ${radius} row ${row} adjusted ${result.adjustedPValues[row]} != ${expectedAdjusted}`
            ).toBe(true);
          }
          const expectedBins = adjusted.map((value, row) => {
            const level = Number.isFinite(value) ? levels.filter(l => value <= l).length : 0;
            return (result.zScores[row] < 0 ? -level : level) || 0;
          });
          expect(result.bins, `${method} radius ${radius}`).toEqual(expectedBins);
        }
        const significant = result.bins.filter(bin => bin !== 0).length;
        uncorrected.set(method, significant);
        expect((await harness.run()).bins).toEqual(result.bins);
        expect(harness.buildCount).toBe(1);
      } finally {
        harness.destroy();
      }
    }
    // A correction never adds significant rows to the uncorrected map; Bonferroni and Holm
    // control the family-wise error, so they are the strictest.
    const none = uncorrected.get('none') ?? 0;
    for (const method of [
      'bonferroni',
      'holm',
      'benjamini-hochberg',
      'benjamini-yekutieli'
    ] as const) {
      expect(uncorrected.get(method) ?? 0, `${method} radius ${radius}`).toBeLessThanOrEqual(none);
    }
    expect(uncorrected.get('holm') ?? 0).toBeGreaterThanOrEqual(uncorrected.get('bonferroni') ?? 0);
    expect(uncorrected.get('benjamini-hochberg') ?? 0).toBeGreaterThan(0);
    expect(uncorrected.get('benjamini-hochberg') ?? 0).toBeGreaterThanOrEqual(
      uncorrected.get('benjamini-yekutieli') ?? 0
    );
  }
}, 240000);

it('GPUHotSpotAnalysis handles degenerate selections and out-of-range neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
    // One included row has no included neighbor: an island, removed from the analysis.
    const single = await harness.run();
    expect(single.zScores.every(Number.isNaN)).toBe(true);
    expect(single.neighborCounts[5]).toBe(0);
    expect(single.globalStatistics[0]).toBe(0);
    expect(single.globalStatistics[4]).toBe(1);
    harness.writeMask(new Uint32Array(64).fill(1));
    // Every row weighs every other row: k = n - 1, so with the focal weight z is undefined.
    const everything = await harness.run();
    expect(everything.zScores.every(Number.isNaN)).toBe(true);
    expect(everything.neighborCounts.every(count => count === 63)).toBe(true);
    expect(everything.globalStatistics[4]).toBe(0);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }

  // No neighbors at all (plus one out-of-range ID, which is ignored). A row never gets a z-score
  // from its own value alone: every output is NaN under both policies, with or without the focal
  // weight. 'exclude' removes the rows from n; 'keep' leaves them in.
  const rows = 64;
  const islands: OracleWeights = {
    offsets: Uint32Array.from({length: rows + 1}, (_, row) => (row > 0 ? 1 : 0)),
    neighbors: Uint32Array.from([9999]),
    weights: Float32Array.from([1])
  };
  for (const policy of ['exclude', 'keep'] as const) {
    for (const selfWeight of [1, 0]) {
      const islandHarness = createSpatialAutocorrelationHarness(device, {
        contributor: 'hot-spot',
        scene,
        weights: toCsr(islands),
        selfWeight,
        islands: policy
      });
      try {
        const result = await islandHarness.run();
        const label = `islands ${policy} self ${selfWeight}`;
        expectMatchesOracle(
          result,
          computeHotSpotOracle({...scene, weights: islands, selfWeight, islands: policy}),
          label
        );
        expect(result.zScores.every(Number.isNaN), label).toBe(true);
        expect(
          result.bins.every(bin => bin === 0),
          label
        ).toBe(true);
        expect(result.globalStatistics[0], label).toBe(policy === 'keep' ? rows : 0);
        expect(result.globalStatistics[4], label).toBe(rows);
      } finally {
        islandHarness.destroy();
      }
    }
  }
}, 60000);

/** Compares a GPU result with an esda configuration on the shared 60-row fixture. */
function expectMatchesEsdaZScores(
  result: SpatialAutocorrelationReadback,
  expected: readonly number[],
  label: string
): void {
  expect(result.zScores.length, label).toBe(expected.length);
  for (const [row, zScore] of expected.entries()) {
    if (!isClose(result.zScores[row], zScore, Z_ABSOLUTE_TOLERANCE, Z_RELATIVE_TOLERANCE)) {
      throw new Error(`${label}: row ${row} z ${result.zScores[row]} != esda ${zScore}`);
    }
  }
}

it('GPUHotSpotAnalysis matches esda G_Local on a fixture with an island (islands keep and exclude)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = getReferenceValues();
  const weights = getReferenceWeights();
  // Oracle: test/oracles/statistics-local-autocorrelation.py (esda 2.10.0, G_Local transform 'B').
  // esda keeps the island in n (islands: 'keep'); its own z for the island is not compared.
  for (const [policy, mask, expectedStar, expectedGi, count] of [
    ['keep', undefined, REFERENCE.keep.giStar, REFERENCE.keep.gi, REFERENCE_ROW_COUNT],
    ['exclude', undefined, REFERENCE.exclude.giStar, REFERENCE.exclude.gi, REFERENCE_ROW_COUNT - 1],
    [
      'exclude',
      getReferenceMask(),
      REFERENCE.masked.giStar,
      REFERENCE.masked.gi,
      REFERENCE.masked.count
    ]
  ] as const) {
    for (const [selfWeight, expected] of [
      [1, expectedStar],
      [0, expectedGi]
    ] as const) {
      const harness = createSpatialAutocorrelationHarness(device, {
        contributor: 'hot-spot',
        scene: {values, mask},
        weights: toCsr(weights),
        selfWeight,
        islands: policy
      });
      try {
        const label = `esda ${policy}${mask ? ' masked' : ''} self ${selfWeight}`;
        const result = await harness.run();
        expectMatchesEsdaZScores(result, expected, label);
        expect(result.globalStatistics[0], label).toBe(count);
        expect(result.globalStatistics[4], label).toBe(mask ? 0 : 1);
        expect(result.zScores[REFERENCE.island], `${label} island`).toBeNaN();
        expect(result.bins[REFERENCE.island], `${label} island bin`).toBe(0);
        expect(isClose(result.globalStatistics[1], expectedMean(policy, mask), 1e-4, 1e-6)).toBe(
          true
        );
      } finally {
        harness.destroy();
      }
    }
  }
}, 120000);

function expectedMean(policy: 'keep' | 'exclude', mask: Uint32Array | undefined): number {
  return mask
    ? REFERENCE.masked.mean
    : policy === 'keep'
      ? REFERENCE.keep.mean
      : REFERENCE.exclude.mean;
}

it('GPUHotSpotAnalysis counts positive-weight neighbors only and honors minimumNeighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = getReferenceValues();
  const weights = getReferenceWeights();
  // The same neighborhoods with a zero-weight slot inserted before every row's last neighbor and a
  // self slot: neither is a neighbor, so every output is bitwise identical.
  const offsets = [0];
  const neighbors: number[] = [];
  const padded: number[] = [];
  for (let row = 0; row < REFERENCE_ROW_COUNT; row++) {
    const slots: [number, number][] = [];
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      slots.push([weights.neighbors[slot], weights.weights[slot]]);
    }
    if (slots.length > 0) {
      slots.push([row, 1]);
      slots.push([(row + 7) % REFERENCE_ROW_COUNT, 0]);
      slots.sort((left, right) => left[0] - right[0]);
    }
    for (const [neighbor, weight] of slots) {
      neighbors.push(neighbor);
      padded.push(weight);
    }
    offsets.push(neighbors.length);
  }
  const paddedWeights = {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(padded)
  };
  const plain = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene: {values},
    weights: toCsr(weights),
    islands: 'keep'
  });
  const withZeroSlots = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene: {values},
    weights: toCsr(paddedWeights),
    islands: 'keep'
  });
  const minimum = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene: {values},
    weights: toCsr(weights),
    islands: 'keep',
    minimumNeighbors: 4
  });
  try {
    const expected = await plain.run();
    const padding = await withZeroSlots.run();
    expect(padding.zScoreBits).toEqual(expected.zScoreBits);
    expect(padding.neighborCounts).toEqual(expected.neighborCounts);
    expect(padding.neighborCounts[REFERENCE.island]).toBe(0);
    // Interior rows have 4 neighbors, rows next to the ends 3 or 2, the island 0.
    const strict = await minimum.run();
    for (let row = 0; row < REFERENCE_ROW_COUNT; row++) {
      if (expected.neighborCounts[row] >= 4) {
        expect(strict.zScoreBits[row], `row ${row}`).toBe(expected.zScoreBits[row]);
      } else {
        expect(strict.zScores[row], `row ${row}`).toBeNaN();
        expect(strict.pValues[row], `row ${row}`).toBeNaN();
        expect(strict.bins[row], `row ${row}`).toBe(0);
      }
    }
    expect(strict.zScores.filter(Number.isNaN).length).toBeGreaterThan(1);
    // Moments are unchanged: minimumNeighbors only removes rows from the output.
    expect(strict.globalStatistics.slice(0, 4)).toEqual(expected.globalStatistics.slice(0, 4));
  } finally {
    plain.destroy();
    withZeroSlots.destroy();
    minimum.destroy();
  }
}, 120000);

it('GPUHotSpotAnalysis multipleTesting bins equal statsmodels on the esda Gi* p-values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = getReferenceValues();
  const weights = getReferenceWeights();
  // Oracle: test/oracles/statistics-local-autocorrelation.py (esda 2.10.0 G_Local Zs, statsmodels 0.15.0
  // multipletests bonferroni / holm / fdr_bh / fdr_by). The family is the 59 non-island rows.
  expect(REFERENCE.minimumMargin).toBeGreaterThan(1e-3);
  for (const method of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const harness = createSpatialAutocorrelationHarness(device, {
      contributor: 'hot-spot',
      scene: {values},
      weights: toCsr(weights),
      islands: 'keep',
      multipleTesting: method
    });
    try {
      const result = await harness.run();
      const reference = REFERENCE.multipleTesting[method];
      expect(result.bins, method).toEqual(reference.bins);
      for (const [row, adjusted] of reference.adjusted.entries()) {
        expect(
          isClose(result.adjustedPValues[row], adjusted, 1e-6, 2e-3),
          `${method} row ${row} adjusted ${result.adjustedPValues[row]} != ${adjusted}`
        ).toBe(true);
      }
    } finally {
      harness.destroy();
    }
  }
  // The uncorrected bins keep using the critical z-scores.
  expect(REFERENCE.multipleTesting.none.bins).toEqual(REFERENCE.noneBinsFromZ);
}, 120000);

it('GPUHotSpotAnalysis re-standardizes row-standardized weights over the unmasked neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createAutocorrelatedScene(53, 1500, [6]);
  const weights = createDistanceBandWeights(scene.positions, 6, {rowStandardize: true});
  const random = createSeededRandom(77);
  const mask = Uint32Array.from({length: scene.values.length}, () => (random() < 0.7 ? 1 : 0));
  for (const selfWeight of [1, 0]) {
    const harness = createSpatialAutocorrelationHarness(device, {
      contributor: 'hot-spot',
      scene: {...scene, mask},
      weights: toCsr(weights),
      selfWeight,
      rowStandardized: true
    });
    try {
      const result = await harness.run();
      const oracle = computeHotSpotOracle({
        ...scene,
        mask,
        weights,
        selfWeight,
        rowStandardized: true
      });
      expectMatchesOracle(result, oracle, `re-standardized self ${selfWeight}`);
      // Masked neighbors really were removed from the row sums. Without a self weight, z is
      // invariant to rescaling a row of weights (numerator and denominator are both linear in w),
      // so re-standardizing cannot change it. With selfWeight 1 the self term keeps its absolute
      // weight next to neighbors that were rescaled, so the declaration changes z.
      const asGiven = computeHotSpotOracle({...scene, mask, weights, selfWeight});
      const changed = oracle.zScores.some(
        (zScore, row) => Number.isFinite(zScore) && Math.abs(zScore - asGiven.zScores[row]) > 0.05
      );
      expect(changed, `re-standardized self ${selfWeight} changes z`).toBe(selfWeight !== 0);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUHotSpotAnalysis centers large-offset values with a second-pass mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createAutocorrelatedScene(61, 3000, [6]);
  // Mean about 1e6 with a spread of a few units: a single f32 tree sum loses the low bits.
  const values = Float32Array.from(scene.values, value => 1e6 + (value - 100));
  const weights = createDistanceBandWeights(scene.positions, 6);
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'hot-spot',
    scene: {values},
    weights: toCsr(weights)
  });
  try {
    const result = await harness.run();
    // Exact moments of the f32 inputs the GPU reads, over the non-island rows.
    const oracle = computeHotSpotOracle({values, weights});
    expect(result.globalStatistics[0]).toBe(oracle.moments.count);
    expect(Math.abs(result.globalStatistics[1] - oracle.moments.mean)).toBeLessThan(0.0625);
    expect(isClose(result.globalStatistics[2], oracle.moments.variance, 1e-3, 5e-3)).toBe(true);
  } finally {
    harness.destroy();
  }
}, 120000);
