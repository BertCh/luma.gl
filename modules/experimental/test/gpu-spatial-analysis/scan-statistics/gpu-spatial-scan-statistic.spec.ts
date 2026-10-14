// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSpatialScanStatisticParameterValues,
  GPUSpatialScanStatistic,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/scan-statistics/index';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {
  computeScanOracle,
  createGreatCircleScanScene,
  createRegionalClusterScene,
  type ScanOracleInput
} from './scan-statistic-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

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
    permutationCount: 99
  };
}

async function runScan(
  scene: ScanOracleInput,
  options: {maximumPermutations?: number; batchReplicates?: number} = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const maximumPermutations = options.maximumPermutations ?? 128;
  const zones = scene.positions.length / 2;
  const clusterIndices = rig.output(
    'uint32',
    scene.maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS
  );
  const clusterStatistics = rig.output(
    'float32',
    scene.maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS
  );
  const statistics = rig.output('float32', maximumPermutations + 1);
  const summary = rig.output('uint32', GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH);
  const zoneStatistics = rig.output('float32', zones);
  rig.run(
    new GPUSpatialScanStatistic({
      positions: rig.input(scene.positions, 'float32x2', zones),
      spatialContext: scene.spatialContext,
      model: scene.model,
      cases: rig.input(scene.cases, 'uint32'),
      ...(scene.baseline ? {baseline: rig.input(scene.baseline, 'float32')} : {}),
      ...(scene.controls ? {controls: rig.input(scene.controls, 'uint32')} : {}),
      timeBuckets: scene.timeBuckets,
      maximumWindowZones: scene.maximumWindowZones,
      maximumPermutations,
      // Forces the replicate batch size (@internal); the baseline code ignores it.
      ...(options.batchReplicates ? {batchReplicates: options.batchReplicates} : {}),
      maximumClusters: scene.maximumClusters,
      parameters: rig.input(
        getGPUSpatialScanStatisticParameterValues({
          seed: scene.seed,
          permutationCount: scene.permutationCount,
          maximumPopulationFraction: scene.maximumPopulationFraction,
          maximumTimeBuckets: scene.maximumTimeBuckets,
          windowShape: scene.windowShape,
          rateDirection: scene.rateDirection
        }),
        'uint32'
      ),
      output: {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics}
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
  const {center, zoneCount, firstBucket, lastBucket} = GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX;
  const clusterCount = actual.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.clusterCount];
  expect(actual.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.totalCases]).toBe(expected.totalCases);
  expect(actual.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.permutationCount]).toBe(
    scene.permutationCount
  );
  expect(actual.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.overflow]).toBe(expected.overflow);
  expect(expected.clusters.length).toBeGreaterThanOrEqual(minimumClusters);
  expect(clusterCount).toBe(expected.clusters.length);
  for (let index = 0; index < clusterCount; index++) {
    const record = index * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS;
    const statistic = index * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS;
    const oracle = expected.clusters[index];
    expect(actual.indices[record + center]).toBe(oracle.center);
    expect(actual.indices[record + zoneCount]).toBe(oracle.zoneCount);
    expect(actual.indices[record + firstBucket]).toBe(oracle.firstBucket);
    expect(actual.indices[record + lastBucket]).toBe(oracle.lastBucket);
    const stats = actual.clusterStats;
    expect(
      Math.abs(
        stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio] -
          oracle.logLikelihoodRatio
      )
    ).toBeLessThan(2e-3 * oracle.logLikelihoodRatio + 1e-3);
    expect(stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.observedCases]).toBe(
      oracle.observedCases
    );
    expect(stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.expectedCases]).toBeCloseTo(
      oracle.expectedCases,
      2
    );
    if (scene.spatialContext?.metric === 'great-circle') {
      expect(
        Math.abs(stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.radius] - oracle.radius)
      ).toBeLessThan(2e-5 * oracle.radius + 1e-6);
    } else {
      expect(stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.radius]).toBeCloseTo(
        oracle.radius,
        3
      );
    }
    expect(stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.truncatedByZoneCap]).toBe(
      oracle.truncatedByZoneCap ? 1 : 0
    );
    expect(
      Math.abs(stats[statistic + GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.pValue] - oracle.pValue)
    ).toBeLessThanOrEqual(0.03);
  }
  // Replicate maxima follow the same Philox draws; f32 weight rounding may move a rare draw.
  let matching = 0;
  for (let slot = 0; slot <= scene.permutationCount; slot++) {
    if (
      Math.abs(actual.maxima[slot] - expected.maxima[slot]) <
      2e-3 * expected.maxima[slot] + 1e-3
    ) {
      matching++;
    }
  }
  expect(matching).toBeGreaterThanOrEqual(0.9 * (scene.permutationCount + 1));
  expect(actual.maxima.slice(scene.permutationCount + 1).every(value => value === 0)).toBe(true);
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
  expect(actual.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.overflow]).toBe(0);
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
    permutationCount: 49
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
    permutationCount: 49
  };
  const actual = await runScan(spaceTime);
  if (!actual) return;
  expectMatches(spaceTime, actual);
});

it('GPUSpatialScanStatistic keeps near-null LLRs precise (no cancellation at c close to e)', async () => {
  // 49 zones of population 1000 with about 500 cases each, so every window has c within a few
  // cases of e while C is about 24500: the LLRs are 1e-4 to 1e-1. With the old
  // `c ln(c/e) + (C - c) ln((C - c)/(C - e))` the f32 logarithms of ratios near 1 carried an absolute
  // error of about `C * 2^-24` (1.5e-3), larger than most of these LLRs.
  //
  // Bound for the `e ((1 + d) ln(1 + d) - d)` form: baselines are integers, so window sums are exact
  // in f32 and `e = baseline * C / totalBaseline` carries two roundings (relative 2^-23). The LLR moves
  // by about `|c - e| * 2^-23 * (1 + e / (C - e))` <= 48 * 1.2e-7 * 2 = 1.2e-5 (|c - e| <= 2 per zone,
  // at most 24 zones per window), plus a few ulp of the LLR itself; the tolerance is 3e-5 + 1e-4 LLR.
  const side = 7;
  const zones = side * side;
  const positions = new Float32Array(zones * 2);
  const cases = new Uint32Array(zones);
  const baseline = new Float32Array(zones);
  for (let zone = 0; zone < zones; zone++) {
    positions[zone * 2] = (zone % side) * 10;
    positions[zone * 2 + 1] = Math.floor(zone / side) * 10;
    baseline[zone] = 1000;
    cases[zone] = 500 + ((zone * 3) % 5) - 2;
  }
  for (const rateDirection of ['high', 'both'] as const) {
    const scene: ScanOracleInput = {
      positions,
      cases,
      baseline,
      timeBuckets: 1,
      maximumPopulationFraction: 0.5,
      windowShape: 'circle',
      maximumClusters: 2,
      seed: 99,
      permutationCount: 9,
      rateDirection
    };
    const actual = await runScan(scene);
    if (!actual) return;
    const expected = computeScanOracle(scene);
    let smallest = Number.POSITIVE_INFINITY;
    for (let zone = 0; zone < zones; zone++) {
      const ratio = expected.observed[zone].ratio;
      if (ratio > 0) smallest = Math.min(smallest, ratio);
      expect(
        Math.abs(actual.zoneStats[zone] - ratio),
        `${rateDirection} zone ${zone}: ${actual.zoneStats[zone]} vs ${ratio}`
      ).toBeLessThanOrEqual(3e-5 + 1e-4 * ratio);
    }
    // The scene does reach the regime the old form got wrong.
    expect(smallest).toBeLessThan(1e-2);
  }
});

/** Bits of every output, for bit-exact comparison (float equality would hide -0 and NaN payloads). */
function getBits(result: NonNullable<Awaited<ReturnType<typeof runScan>>>): number[][] {
  const bits = (values: ArrayLike<number>, floating: boolean) =>
    Array.from(floating ? new Uint32Array(new Float32Array(values).buffer) : (values as number[]));
  return [
    bits(result.indices, false),
    bits(result.clusterStats, true),
    bits(result.maxima, true),
    bits(result.summary, false),
    bits(result.zoneStats, true)
  ];
}

function createBernoulliScene(): ScanOracleInput {
  const base = createRegionalClusterScene(8, 2, 3);
  const controls = new Uint32Array(base.cases.length);
  for (let zone = 0; zone < controls.length; zone++) controls[zone] = 100 - base.cases[zone];
  return {...base, model: 'bernoulli', baseline: undefined, controls, permutationCount: 49};
}

const BATCH_SCENES: [string, () => ScanOracleInput][] = [
  ['poisson space-time', () => createScene(4, 21)],
  ['bernoulli', createBernoulliScene],
  [
    'space-time permutation',
    () => ({
      ...createScene(4, 21),
      model: 'space-time-permutation',
      baseline: undefined,
      maximumTimeBuckets: 2,
      permutationCount: 49
    })
  ]
];

it.each(
  BATCH_SCENES
)('GPUSpatialScanStatistic %s: batched replicates equal the single-batch result bit for bit', async (_, createBatchScene) => {
  const scene = createBatchScene();
  // 128 replicates in one batch (the former single pass) against batches of 1, 7 and 40, where the
  // last batch is partial and the batches past `permutationCount` are disarmed.
  const single = await runScan(scene, {batchReplicates: 128});
  if (!single) return;
  const expected = getBits(single);
  expectMatches(scene, single);
  for (const batchReplicates of [1, 7, 40]) {
    const batched = await runScan(scene, {batchReplicates});
    expect(getBits(batched!), `batchReplicates ${batchReplicates}`).toEqual(expected);
  }
});

it('GPUSpatialScanStatistic runs more replicates than the former replicate-table cap', async () => {
  // 3500 zones, one bucket: (9999 + 1) * 3500 = 35M words > 2^25, which the single pass rejected.
  const side = 59;
  const zones = 3500;
  const random = createSeededRandom(77);
  const positions = new Float32Array(zones * 2);
  const cases = new Uint32Array(zones);
  const baseline = new Float32Array(zones).fill(100);
  for (let zone = 0; zone < zones; zone++) {
    positions[zone * 2] = (zone % side) * 10;
    positions[zone * 2 + 1] = Math.floor(zone / side) * 10;
    const column = zone % side;
    const row = Math.floor(zone / side);
    const rate = column < 4 && row < 4 ? 0.06 : 0.01;
    for (let person = 0; person < 100; person++) if (random() < rate) cases[zone]++;
  }
  const scene: ScanOracleInput = {
    positions,
    cases,
    baseline,
    timeBuckets: 1,
    maximumWindowZones: 16,
    maximumPopulationFraction: 0.25,
    windowShape: 'circle',
    maximumClusters: 2,
    seed: 4242,
    permutationCount: 9999
  };
  const full = await runScan(scene, {maximumPermutations: 9999});
  if (!full) return;
  expect(full.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.permutationCount]).toBe(9999);
  const maxima = Array.from(full.maxima);
  expect(maxima.length).toBe(10000);
  expect(maxima.slice(1).every(value => Number.isFinite(value) && value > 0)).toBe(true);
  // The planted corner is a significant cluster against ten thousand replicates.
  const primary = full.clusterStats[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.pValue];
  expect(primary).toBeLessThanOrEqual(0.01);
  // The minimum p = 1 / (P + 1) is stored as f32. WGSL division allows 2.5 ULP;
  // at 1e-4 the ULP is 2^-37, so the lower bound includes precisely that error budget.
  expect(primary).toBeGreaterThanOrEqual(1 / 10000 - 2.5 * 2 ** -37);

  // Streams are keyed by the global replicate index: another batch size and another replicate
  // count give the same maxima for the replicates they share.
  const smallBatches = await runScan(scene, {maximumPermutations: 9999, batchReplicates: 1000});
  expect(getBits(smallBatches!)).toEqual(getBits(full));
  const first99 = await runScan({...scene, permutationCount: 99}, {maximumPermutations: 128});
  expect(Array.from(first99!.maxima.slice(0, 100))).toEqual(maxima.slice(0, 100));
}, 600_000);

it('GPUSpatialScanStatistic A/B: replicate batching at P = 999 on the spec scene', async () => {
  // The batching splits the former single pass into batch-control, batch-clear, draw and
  // batch-evaluate per batch. Compare one batch (the former pass) with batches of 100 on the same
  // device, and print a checksum of the outputs. Run this same spec in a baseline checkout (where
  // `batchReplicates` is ignored and every run is the single pass) to compare the checksum and the
  // time with the pre-change code: bit-identical outputs mean equal checksums.
  const scene: ScanOracleInput = {...createScene(4, 21, 12), permutationCount: 999};
  const timings: Record<string, number> = {};
  const checksums: Record<string, number> = {};
  for (const [label, batchReplicates] of [
    ['single batch', 999],
    ['batches of 100', 100],
    ['single batch again', 999]
  ] as const) {
    const start = performance.now();
    const result = await runScan(scene, {maximumPermutations: 999, batchReplicates});
    if (!result) return;
    timings[label] = performance.now() - start;
    let checksum = 2166136261;
    for (const word of getBits(result).flat()) {
      checksum = Math.imul(checksum ^ word, 16777619) >>> 0;
    }
    checksums[label] = checksum;
  }
  console.log('scan statistic P = 999 A/B (ms incl. setup)', timings, 'checksums', checksums);
  expect(checksums['batches of 100']).toBe(checksums['single batch']);
  expect(checksums['single batch again']).toBe(checksums['single batch']);
}, 120_000);

it.each([
  {units: 'meters' as const},
  {units: 'kilometers' as const},
  {units: 'kilometers' as const, sphereRadius: 1000}
])('GPUSpatialScanStatistic matches great-circle clusters and radii in $units with radius $sphereRadius', async context => {
  const scene = createGreatCircleScanScene({
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    ...context
  });
  const actual = await runScan(scene);
  const expected = expectMatches(scene, actual);
  expect(expected.clusters[0].center).toBe(0);
  expect(expected.clusters[0].zoneCount).toBe(2);
  expect(expected.clusters[0].observedCases).toBe(70);
});
