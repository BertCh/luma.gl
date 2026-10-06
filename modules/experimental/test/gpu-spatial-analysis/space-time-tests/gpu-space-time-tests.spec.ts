// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSpaceTimeParameterValues,
  getKnoxPoissonPValue,
  GPUKnoxTest,
  GPUMantelTest,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {AnalysisRig, createSeededRandom, type CPUWeights} from '../catchment-accessibility/rig';
import {computeKnoxOracle, computeMantelOracle} from './space-time-oracle';

/** Events: a space-time clustered core plus uniform background (so interaction exists). */
function createEvents(count: number, seed: number) {
  const random = createSeededRandom(seed);
  const positions = new Float32Array(count * 2);
  const times = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const clustered = index % 3 === 0;
    const center = Math.floor(random() * 4);
    positions[index * 2] = Math.fround(clustered ? center * 25 + random() * 6 : random() * 100);
    positions[index * 2 + 1] = Math.fround(clustered ? center * 20 + random() * 6 : random() * 100);
    times[index] = Math.fround(clustered ? center * 30 + random() * 4 : random() * 120);
  }
  return {positions, times};
}

function buildBand(positions: Float32Array, radius: number): CPUWeights & {distances: number[]} {
  const rows = positions.length / 2;
  const csr = {
    offsets: [0],
    neighbors: [] as number[],
    weights: [] as number[],
    distances: [] as number[]
  };
  for (let row = 0; row < rows; row++) {
    for (let other = 0; other < rows; other++) {
      const distance = Math.hypot(
        positions[row * 2] - positions[other * 2],
        positions[row * 2 + 1] - positions[other * 2 + 1]
      );
      if (other !== row && distance <= radius) {
        csr.neighbors.push(other);
        csr.weights.push(1);
        csr.distances.push(Math.fround(distance));
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

function uploadPairs(rig: AnalysisRig, csr: CPUWeights & {distances: number[]}) {
  const capacity = csr.neighbors.length + 4;
  const distances = new Float32Array(capacity);
  distances.set(csr.distances);
  return {
    ...rig.weights(csr, 4),
    distances: rig.input(distances, 'float32')
  };
}

it('GPUKnoxTest matches the Feistel permutation oracle exactly', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const {count, radius, threshold, permutations, seed} of [
    {count: 240, radius: 9, threshold: 3, permutations: 99, seed: 7},
    {count: 37, radius: 40, threshold: 25.5, permutations: 31, seed: 123456789012}
  ]) {
    const {positions, times} = createEvents(count, 5);
    const csr = buildBand(positions, radius);
    const rig = new AnalysisRig(device);
    const maximumPermutations = 128;
    const statistics = rig.output('uint32', maximumPermutations + 1);
    const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    rig.run(
      new GPUKnoxTest({
        pairs: rig.weights(csr, 4),
        times: rig.input(times, 'float32'),
        parameters: rig.input(
          getGPUSpaceTimeParameterValues({seed, permutations, timeThreshold: threshold}),
          'uint32'
        ),
        maximumPermutations,
        statistics,
        summary
      })
    );
    const expected = computeKnoxOracle(csr, times, seed, permutations, threshold);
    const actual = await rig.readUint(statistics);
    expect(expected.statistics[0]).toBeGreaterThan(0);
    expect(new Set(expected.statistics).size).toBeGreaterThan(3);
    expect(actual.slice(0, permutations + 1)).toEqual(expected.statistics);
    expect(actual.slice(permutations + 1).every(value => value === 0)).toBe(true);
    const result = await rig.readFloat(summary);
    const index = GPU_SPACE_TIME_SUMMARY;
    expect(result[index.observed]).toBe(expected.statistics[0]);
    expect(result[index.pairCount]).toBe(expected.pairCount);
    expect(result[index.timeClosePairs]).toBe(expected.timeClose);
    expect(Math.abs(result[index.expected] - expected.expected)).toBeLessThan(
      1e-4 * expected.expected
    );
    expect(result[index.greaterCount]).toBe(expected.greater);
    expect(result[index.lesserCount]).toBe(expected.lesser);
    expect(result[index.pseudoPGreater]).toBeCloseTo(
      (expected.greater + 1) / (permutations + 1),
      6
    );
    const permuted = expected.statistics.slice(1);
    const mean = permuted.reduce((sum, value) => sum + value, 0) / permutations;
    expect(Math.abs(result[index.permutationMean] - mean)).toBeLessThan(1e-3 * (1 + mean));
    // Space-time clustering is built in, so the observed count exceeds the permutation mean.
    if (count > 100) {
      expect(result[index.observed]).toBeGreaterThan(result[index.permutationMean]);
      expect(result[index.pseudoPGreater]).toBeLessThan(0.05);
    }
    rig.destroy();
  }
});

it('GPUKnoxTest re-evaluates a new threshold and seed without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {positions, times} = createEvents(150, 9);
  const csr = buildBand(positions, 10);
  const rig = new AnalysisRig(device);
  const statistics = rig.output('uint32', 65);
  const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const parameters = rig.input(
    getGPUSpaceTimeParameterValues({seed: 1, permutations: 16, timeThreshold: 2}),
    'uint32'
  );
  rig.run(
    new GPUKnoxTest({
      pairs: rig.weights(csr, 4),
      times: rig.input(times, 'float32'),
      parameters,
      maximumPermutations: 64,
      statistics,
      summary
    })
  );
  const first = await rig.readUint(statistics);
  expect(first.slice(0, 17)).toEqual(computeKnoxOracle(csr, times, 1, 16, 2).statistics);
  rig.destroy();
  // A different seed changes permuted counts but not the observed count.
  const second = new AnalysisRig(device);
  const statistics2 = second.output('uint32', 65);
  second.run(
    new GPUKnoxTest({
      pairs: second.weights(csr, 4),
      times: second.input(times, 'float32'),
      parameters: second.input(
        getGPUSpaceTimeParameterValues({seed: 2, permutations: 16, timeThreshold: 2}),
        'uint32'
      ),
      maximumPermutations: 64,
      statistics: statistics2,
      summary: second.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH)
    })
  );
  const other = await second.readUint(statistics2);
  expect(other[0]).toBe(first[0]);
  expect(other.slice(1, 17)).not.toEqual(first.slice(1, 17));
  second.destroy();
});

it('GPUMantelTest matches the double-precision oracle with permuted times', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {positions, times} = createEvents(200, 11);
  const csr = buildBand(positions, 14);
  const permutations = 49;
  const seed = 99;
  const rig = new AnalysisRig(device);
  const statistics = rig.output('float32', 64 + 1);
  const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  rig.run(
    new GPUMantelTest({
      pairs: uploadPairs(rig, csr),
      times: rig.input(times, 'float32'),
      parameters: rig.input(getGPUSpaceTimeParameterValues({seed, permutations}), 'uint32'),
      maximumPermutations: 64,
      statistics,
      summary
    })
  );
  const expected = computeMantelOracle(csr, times, seed, permutations);
  const actual = await rig.readFloat(statistics);
  expect(Math.abs(expected[0])).toBeGreaterThan(0.05);
  for (let slot = 0; slot <= permutations; slot++) {
    expect(Math.abs(actual[slot] - expected[slot]), `slot ${slot}`).toBeLessThan(2e-4);
  }
  expect(actual.slice(permutations + 1).every(value => value === 0)).toBe(true);
  const result = await rig.readFloat(summary);
  const index = GPU_SPACE_TIME_SUMMARY;
  expect(result[index.pairCount]).toBe(csr.neighbors.length / 2);
  expect(Math.abs(result[index.observed] - expected[0])).toBeLessThan(2e-4);
  const greater = actual.slice(1, permutations + 1).filter(value => value >= actual[0]).length;
  expect(result[index.greaterCount]).toBe(greater);
  expect(result[index.pseudoPGreater]).toBeCloseTo((greater + 1) / (permutations + 1), 6);
  expect(result[index.zScore]).toBeGreaterThan(2);
  rig.destroy();
});

it('GPUKnoxTest runs on pairs written by GPUNeighborSearch', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {positions, times} = createEvents(200, 13);
  const radius = 8.3;
  const rig = new AnalysisRig(device);
  const capacity = 200 * 200;
  const offsets = rig.output('uint32', 201);
  const neighbors = rig.output('uint32', capacity);
  const weights = rig.output('float32', capacity);
  const statistics = rig.output('uint32', 33);
  const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  rig.run(
    new GPUNeighborSearch({
      mode: 'radius',
      gridSize: [8, 8],
      positions: rig.input(positions, 'float32x2', 200),
      parameters: rig.input(
        getGPUNeighborSearchParameterValues({bounds: [-1, -1, 101, 101], radius}),
        'float32'
      ),
      weights: {offsets, neighbors, weights},
      overflow: rig.output('uint32', 1)
    }),
    new GPUKnoxTest({
      pairs: {offsets, neighbors, weights},
      times: rig.input(times, 'float32'),
      parameters: rig.input(
        getGPUSpaceTimeParameterValues({seed: 3, permutations: 32, timeThreshold: 3}),
        'uint32'
      ),
      maximumPermutations: 32,
      statistics,
      summary
    })
  );
  const csrOffsets = await rig.readUint(offsets);
  const csrNeighbors = (await rig.readUint(neighbors)).slice(0, csrOffsets[200]);
  expect(csrNeighbors.length).toBeGreaterThan(100);
  const expected = computeKnoxOracle(
    {offsets: csrOffsets, neighbors: csrNeighbors},
    times,
    3,
    32,
    3
  );
  expect(await rig.readUint(statistics)).toEqual(expected.statistics);
  rig.destroy();
});

it('getKnoxPoissonPValue returns the Poisson upper tail', () => {
  expect(getKnoxPoissonPValue(0, 3)).toBe(1);
  // P(X >= 3 | mean 3) = 1 - (e^-3)(1 + 3 + 4.5) = 0.5768...
  expect(getKnoxPoissonPValue(3, 3)).toBeCloseTo(1 - Math.exp(-3) * 8.5, 10);
  expect(getKnoxPoissonPValue(20, 3)).toBeLessThan(1e-8);
});

// Reference values from pointpats 'Knox.from_dataframe' (delta=3, tau=4, permutations=0) on the
// fixture below, generated with the PySAL venv: statistic 29, expected 31.76091954, p_poisson 0.64655.
const POINTPATS_XY = [
  9.431, 5.113, 9.762, 0.808, 6.074, 3.765, 8.019, 1.745, 8.716, 5.439, 9.022, 4.772, 4.305, 7.889,
  9.842, 3.697, 9.689, 9.29, 1.777, 6.089, 7.049, 9.428, 6.657, 1.334, 4.979, 4.936, 5.002, 9.586,
  3.499, 2.238, 5.221, 6.412, 9.391, 5.82, 2.678, 9.298, 4.917, 6.758, 4.76, 2.17, 6.926, 7.706,
  1.908, 4.599, 3.618, 1.707, 2.214, 9.625, 8.842, 3.823, 7.395, 0.42, 9.151, 5.372, 8.203, 2.759,
  3.761, 3.48, 9.724, 4.295
];
const POINTPATS_TIMES = [
  9.994, 19.12, 18.064, 7.9, 6.147, 16.141, 2.149, 7.844, 17.529, 18.353, 4.67, 1.574, 15.755,
  12.436, 4.75, 10.118, 1.428, 14.002, 10.535, 14.093, 1.763, 12.896, 7.407, 9.986, 6.71, 12.682,
  11.365, 12.342, 1.268, 4.503
];

it('GPUKnoxTest reproduces the pointpats Knox statistic and expectation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const positions = Float32Array.from(POINTPATS_XY);
  const times = Float32Array.from(POINTPATS_TIMES);
  const csr = buildBand(positions, 3);
  const rig = new AnalysisRig(device);
  const statistics = rig.output('uint32', 17);
  const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  rig.run(
    new GPUKnoxTest({
      pairs: rig.weights(csr, 4),
      times: rig.input(times, 'float32'),
      parameters: rig.input(
        getGPUSpaceTimeParameterValues({seed: 1, permutations: 16, timeThreshold: 4}),
        'uint32'
      ),
      maximumPermutations: 16,
      statistics,
      summary
    })
  );
  const result = await rig.readFloat(summary);
  expect(result[GPU_SPACE_TIME_SUMMARY.observed]).toBe(29);
  expect(Math.abs(result[GPU_SPACE_TIME_SUMMARY.expected] - 31.76091954)).toBeLessThan(1e-3);
  // pointpats reports P(X > observed) (scipy poisson.sf); getKnoxPoissonPValue is P(X >= observed).
  const expected = result[GPU_SPACE_TIME_SUMMARY.expected];
  expect(Math.abs(getKnoxPoissonPValue(30, expected) - 0.6465526)).toBeLessThan(1e-3);
  rig.destroy();
});
