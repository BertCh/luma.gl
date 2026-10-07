// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSpatialScanParameterValues,
  GPUSpatialScanStatistic,
  GPU_SCAN_STATISTIC_CLUSTER,
  GPU_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SCAN_STATISTIC_SUMMARY,
  GPU_SCAN_STATISTIC_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/scan-statistics/index';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {computeScanOracle, type ScanOracleInput} from './scan-statistic-oracle';

/** A 7x7 lattice (spacing 10, exact in f32) with a planted space-time hot spot and a weaker one. */
function createScene(timeBuckets: number, seed: number, side = 7): ScanOracleInput {
  const random = createSeededRandom(seed);
  const zones = side * side;
  const positions = new Float32Array(zones * 2);
  const cases = new Uint32Array(zones * timeBuckets);
  const baseline = new Float32Array(zones * timeBuckets);
  for (let zone = 0; zone < zones; zone++) {
    const column = zone % side;
    const row = Math.floor(zone / side);
    positions[zone * 2] = column * 10;
    positions[zone * 2 + 1] = row * 10;
    const population = 100 + Math.floor(random() * 200);
    for (let bucket = 0; bucket < timeBuckets; bucket++) {
      const cell = zone * timeBuckets + bucket;
      baseline[cell] = population;
      let rate = 0.05;
      if (column >= 1 && column <= 2 && row >= 1 && row <= 2 && bucket >= 1 && bucket <= 2) {
        rate = 0.15;
      }
      if (column >= 5 && row >= 4 && row <= 5 && bucket === 0) rate = 0.1;
      // Binomial draw: deterministic, integer, close to Poisson.
      let count = 0;
      for (let person = 0; person < population; person++) if (random() < rate) count++;
      cases[cell] = count;
    }
  }
  return {
    positions,
    cases,
    baseline,
    timeBuckets,
    maximumWindowZones: 16,
    maximumPopulationFraction: 0.25,
    maximumTimeBuckets: 2,
    windowShape: 'circle',
    maximumClusters: 4,
    seed: 12345,
    permutations: 99
  };
}

async function runScan(scene: ScanOracleInput) {
  const device = await getWebGPUTestDevice();
  if (!device) return null;
  const rig = new AnalysisRig(device);
  const maximumPermutations = 128;
  const zones = scene.positions.length / 2;
  const clusterIndices = rig.output(
    'uint32',
    scene.maximumClusters * GPU_SCAN_STATISTIC_INDEX_WORDS
  );
  const clusterStatistics = rig.output(
    'float32',
    scene.maximumClusters * GPU_SCAN_STATISTIC_STATISTIC_WORDS
  );
  const statistics = rig.output('float32', maximumPermutations + 1);
  const summary = rig.output('uint32', GPU_SCAN_STATISTIC_SUMMARY_LENGTH);
  const zoneStatistics = rig.output('float32', zones);
  rig.run(
    new GPUSpatialScanStatistic({
      positions: rig.input(scene.positions, 'float32x2', zones),
      cases: rig.input(scene.cases, 'uint32'),
      baseline: rig.input(scene.baseline, 'float32'),
      timeBuckets: scene.timeBuckets,
      maximumWindowZones: scene.maximumWindowZones,
      maximumPermutations,
      maximumClusters: scene.maximumClusters,
      parameters: rig.input(
        getGPUSpatialScanParameterValues({
          seed: scene.seed,
          permutations: scene.permutations,
          maximumPopulationFraction: scene.maximumPopulationFraction,
          maximumTimeBuckets: scene.maximumTimeBuckets,
          windowShape: scene.windowShape
        }),
        'uint32'
      ),
      clusterIndices,
      clusterStatistics,
      statistics,
      summary,
      zoneStatistics
    })
  );
  const result = {
    indices: await rig.readUint(clusterIndices),
    clusterStats: await rig.readFloat(clusterStatistics),
    maxima: await rig.readFloat(statistics),
    summary: await rig.readUint(summary),
    zoneStats: await rig.readFloat(zoneStatistics)
  };
  rig.destroy();
  return result;
}

function expectMatches(
  scene: ScanOracleInput,
  actual: NonNullable<Awaited<ReturnType<typeof runScan>>>
) {
  const expected = computeScanOracle(scene);
  const {center, zoneCount, firstBucket, lastBucket} = GPU_SCAN_STATISTIC_CLUSTER_INDEX;
  const clusterCount = actual.summary[GPU_SCAN_STATISTIC_SUMMARY.clusterCount];
  expect(actual.summary[GPU_SCAN_STATISTIC_SUMMARY.totalCases]).toBe(expected.totalCases);
  expect(actual.summary[GPU_SCAN_STATISTIC_SUMMARY.permutations]).toBe(scene.permutations);
  expect(expected.clusters.length).toBeGreaterThanOrEqual(2);
  expect(clusterCount).toBe(expected.clusters.length);
  for (let index = 0; index < clusterCount; index++) {
    const record = index * GPU_SCAN_STATISTIC_INDEX_WORDS;
    const statistic = index * GPU_SCAN_STATISTIC_STATISTIC_WORDS;
    const oracle = expected.clusters[index];
    expect(actual.indices[record + center]).toBe(oracle.center);
    expect(actual.indices[record + zoneCount]).toBe(oracle.zoneCount);
    expect(actual.indices[record + firstBucket]).toBe(oracle.firstBucket);
    expect(actual.indices[record + lastBucket]).toBe(oracle.lastBucket);
    const stats = actual.clusterStats;
    expect(
      Math.abs(
        stats[statistic + GPU_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio] - oracle.logLikelihoodRatio
      )
    ).toBeLessThan(2e-3 * oracle.logLikelihoodRatio + 1e-3);
    expect(stats[statistic + GPU_SCAN_STATISTIC_CLUSTER.observedCases]).toBe(oracle.observedCases);
    expect(stats[statistic + GPU_SCAN_STATISTIC_CLUSTER.expectedCases]).toBeCloseTo(
      oracle.expectedCases,
      2
    );
    expect(stats[statistic + GPU_SCAN_STATISTIC_CLUSTER.radius]).toBeCloseTo(oracle.radius, 3);
    expect(
      Math.abs(stats[statistic + GPU_SCAN_STATISTIC_CLUSTER.pValue] - oracle.pValue)
    ).toBeLessThanOrEqual(0.03);
  }
  // Replicate maxima follow the same Philox draws; f32 weight rounding may move a rare draw.
  let matching = 0;
  for (let slot = 0; slot <= scene.permutations; slot++) {
    if (
      Math.abs(actual.maxima[slot] - expected.maxima[slot]) <
      2e-3 * expected.maxima[slot] + 1e-3
    ) {
      matching++;
    }
  }
  expect(matching).toBeGreaterThanOrEqual(0.9 * (scene.permutations + 1));
  expect(actual.maxima.slice(scene.permutations + 1).every(value => value === 0)).toBe(true);
  // Per-zone best LLR agrees with the oracle's per-center maxima.
  for (let zone = 0; zone < expected.observed.length; zone++) {
    expect(Math.abs(actual.zoneStats[zone] - expected.observed[zone].ratio)).toBeLessThan(
      2e-3 * expected.observed[zone].ratio + 1e-3
    );
  }
  return expected;
}

it('GPUSpatialScanStatistic finds the planted space-time cylinder and matches the oracle', async () => {
  const scene = createScene(4, 21);
  const actual = await runScan(scene);
  if (!actual) return;
  const expected = expectMatches(scene, actual);
  // The planted cluster covers buckets 1-2 around zones (1..2, 1..2).
  const primary = expected.clusters[0];
  expect(primary.firstBucket).toBeGreaterThanOrEqual(1);
  expect(primary.lastBucket).toBeLessThanOrEqual(2);
  expect(primary.pValue).toBeLessThanOrEqual(0.05);
});

it('GPUSpatialScanStatistic scans space only with k-nearest windows', async () => {
  const scene: ScanOracleInput = {
    ...createScene(1, 5),
    windowShape: 'nearest',
    maximumTimeBuckets: 1
  };
  const actual = await runScan(scene);
  if (!actual) return;
  expectMatches(scene, actual);
});

it('GPUSpatialScanStatistic reduces replicate maxima across several workgroups of centers', async () => {
  // 18 x 18 = 324 zones: more than the 256 centers one evaluate workgroup reduces.
  const scene: ScanOracleInput = {...createScene(2, 8, 18), maximumTimeBuckets: 2};
  const actual = await runScan(scene);
  if (!actual) return;
  expectMatches(scene, actual);
});
