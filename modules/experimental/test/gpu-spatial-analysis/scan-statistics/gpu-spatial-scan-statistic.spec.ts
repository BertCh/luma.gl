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
import {
  computeScanOracle,
  createRegionalClusterScene,
  type ScanOracleInput
} from './scan-statistic-oracle';

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
      model: scene.model,
      cases: rig.input(scene.cases, 'uint32'),
      ...(scene.baseline ? {baseline: rig.input(scene.baseline, 'float32')} : {}),
      ...(scene.controls ? {controls: rig.input(scene.controls, 'uint32')} : {}),
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
          windowShape: scene.windowShape,
          rateDirection: scene.rateDirection
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
  actual: NonNullable<Awaited<ReturnType<typeof runScan>>>,
  minimumClusters = 1
) {
  const expected = computeScanOracle(scene);
  const {center, zoneCount, firstBucket, lastBucket} = GPU_SCAN_STATISTIC_CLUSTER_INDEX;
  const clusterCount = actual.summary[GPU_SCAN_STATISTIC_SUMMARY.clusterCount];
  expect(actual.summary[GPU_SCAN_STATISTIC_SUMMARY.totalCases]).toBe(expected.totalCases);
  expect(actual.summary[GPU_SCAN_STATISTIC_SUMMARY.permutations]).toBe(scene.permutations);
  expect(actual.summary[GPU_SCAN_STATISTIC_SUMMARY.overflow]).toBe(expected.overflow);
  expect(expected.clusters.length).toBeGreaterThanOrEqual(minimumClusters);
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
    expect(stats[statistic + GPU_SCAN_STATISTIC_CLUSTER.truncatedByZoneCap]).toBe(
      oracle.truncatedByZoneCap ? 1 : 0
    );
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
  // The secondary-cluster path must be exercised (the oracle finds 4 clusters here).
  const expected = expectMatches(scene, actual, 2);
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
  expectMatches(scene, actual, 2);
});

it('GPUSpatialScanStatistic reduces replicate maxima across several workgroups of centers', async () => {
  // 18 x 18 = 324 zones: more than the 256 centers one evaluate workgroup reduces.
  const scene: ScanOracleInput = {...createScene(2, 8, 18), maximumTimeBuckets: 2};
  const actual = await runScan(scene);
  if (!actual) return;
  expectMatches(scene, actual, 2);
});

it('GPUSpatialScanStatistic lets the population share govern windows beyond 32 zones', async () => {
  // 144 zones, a 60-zone regional cluster: one cluster, no truncation.
  const scene = createRegionalClusterScene(12, 5, 11);
  const actual = await runScan(scene);
  if (!actual) return;
  const expected = expectMatches(scene, actual);
  expect(expected.clusters[0].zoneCount).toBeGreaterThanOrEqual(50);
  expect(actual.summary[GPU_SCAN_STATISTIC_SUMMARY.overflow]).toBe(0);
});

it('GPUSpatialScanStatistic reports windows cut by the zone capacity', async () => {
  const scene = {...createRegionalClusterScene(12, 5, 11), maximumWindowZones: 32};
  const actual = await runScan(scene);
  if (!actual) return;
  const expected = expectMatches(scene, actual);
  expect(expected.overflow).toBeGreaterThan(0);
});

it('GPUSpatialScanStatistic scans low rates and both directions', async () => {
  for (const rateDirection of ['low', 'both'] as const) {
    const scene: ScanOracleInput = {...createScene(2, 8), maximumTimeBuckets: 2, rateDirection};
    const actual = await runScan(scene);
    if (!actual) return;
    expectMatches(scene, actual);
  }
});

it('GPUSpatialScanStatistic defaults the time window to half the study period', async () => {
  const scene = createScene(4, 21);
  delete (scene as {maximumTimeBuckets?: number}).maximumTimeBuckets;
  const actual = await runScan(scene);
  if (!actual) return;
  const expected = expectMatches(scene, actual);
  for (const cluster of expected.clusters) {
    expect(cluster.lastBucket - cluster.firstBucket + 1).toBeLessThanOrEqual(2);
  }
});

it('GPUSpatialScanStatistic scans the Bernoulli model', async () => {
  const base = createRegionalClusterScene(8, 2, 3);
  const controls = new Uint32Array(base.cases.length);
  for (let zone = 0; zone < controls.length; zone++) controls[zone] = 100 - base.cases[zone];
  const scene: ScanOracleInput = {
    ...base,
    model: 'bernoulli',
    baseline: undefined,
    controls,
    permutations: 49
  };
  const actual = await runScan(scene);
  if (!actual) return;
  expectMatches(scene, actual);
});

it('GPUSpatialScanStatistic scans the space-time permutation model', async () => {
  const scene = createScene(4, 21);
  // Cases only: the expectation comes from their margins.
  const spaceTime: ScanOracleInput = {
    ...scene,
    model: 'space-time-permutation',
    baseline: undefined,
    maximumTimeBuckets: 2,
    permutations: 49
  };
  const actual = await runScan(spaceTime);
  if (!actual) return;
  expectMatches(spaceTime, actual);
});
