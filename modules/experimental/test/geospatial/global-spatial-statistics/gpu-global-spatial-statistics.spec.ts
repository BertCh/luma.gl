// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
} from '../../../src/geospatial/neighbor-search';
import {
  GPU_GLOBAL_JOIN_COUNT_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPUGlobalSpatialStatistics,
  type GPUGlobalSpatialStatistic
} from '../../../src/geospatial/global-spatial-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computeGlobalSpatialStatisticsOracle,
  createGridWeights,
  createKnnWeights,
  createSeededRandom,
  createWeights,
  type CPUSpatialWeights,
  type GlobalSpatialStatisticsOracle
} from './global-spatial-statistics-oracle';

const ALL: GPUGlobalSpatialStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'joinCount'
];
const LAYOUT = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
const FIELD = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;

type Scene = {
  weights: CPUSpatialWeights;
  values: Float32Array;
  secondValues?: Float32Array;
  mask?: Uint32Array;
};

/** Builds one compiled graph; `run` rewrites the inputs and returns results and join counts. */
function createHarness(device: Device, scene: Scene, statistics = ALL) {
  const rows = scene.values.length;
  const capacity = scene.weights.neighbors.length;
  const buffers = {
    offsets: createInputBuffer(device, scene.weights.offsets),
    neighbors: createInputBuffer(device, scene.weights.neighbors),
    weights: createInputBuffer(device, scene.weights.weights),
    values: createInputBuffer(device, scene.values),
    secondValues: scene.secondValues && createInputBuffer(device, scene.secondValues),
    mask: scene.mask && createInputBuffer(device, scene.mask),
    results: createOutputBuffer(device, LAYOUT.length),
    joinCounts: createOutputBuffer(device, 3)
  };
  const graph = new GPUCommandGraph(device, {id: 'global-statistics-test'});
  const recipe = new GPUGlobalSpatialStatistics({
    weights: {
      offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', rows + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
      weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity)
    },
    values: importGraphBuffer(graph, 'values', buffers.values, 'float32', rows),
    secondValues:
      buffers.secondValues &&
      importGraphBuffer(graph, 'second', buffers.secondValues, 'float32', rows),
    mask: buffers.mask && importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', rows),
    statistics,
    results: importGraphBuffer(graph, 'results', buffers.results, 'float32', LAYOUT.length),
    joinCounts: importGraphBuffer(graph, 'join-counts', buffers.joinCounts, 'uint32', 3)
  });
  let buildCount = 0;
  const getCommandNodes = recipe.getCommandNodes.bind(recipe);
  recipe.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof recipe.getCommandNodes;
  graph.add(recipe);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return buildCount;
    },
    async run(update?: Partial<Scene>) {
      if (update?.values) {
        buffers.values.write(update.values);
      }
      if (update?.mask && buffers.mask) {
        buffers.mask.write(update.mask);
      }
      if (update?.weights) {
        buffers.offsets.write(update.weights.offsets);
        buffers.neighbors.write(update.weights.neighbors);
        buffers.weights.write(update.weights.weights);
      }
      buffers.results.write(new Float32Array(LAYOUT.length).fill(12345));
      submitGraph(device, compiled, undefined);
      return {
        results: await readFloat32(buffers.results, LAYOUT.length),
        resultBits: await readUint32(buffers.results, LAYOUT.length),
        joinCounts: await readUint32(buffers.joinCounts, 3)
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of Object.values(buffers)) {
        buffer?.destroy();
      }
    }
  };
}

function isClose(actual: number, expected: number, absolute: number, relative: number): boolean {
  if (Number.isNaN(expected)) {
    return Number.isNaN(actual);
  }
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}

function expectField(
  actual: number,
  expected: number,
  label: string,
  absolute = 1e-5,
  relative = 1e-3
): void {
  if (!isClose(actual, expected, absolute, relative)) {
    throw new Error(`${label}: GPU ${actual} != oracle ${expected}`);
  }
}

function expectMatchesOracle(
  results: number[],
  joinCounts: number[],
  oracle: GlobalSpatialStatisticsOracle,
  label: string,
  statistics = ALL
): void {
  const summary = GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY;
  expect(results[summary.count], `${label} n`).toBe(oracle.count);
  expectField(results[summary.s0], oracle.s0, `${label} S0`, 1e-5, 1e-5);
  expectField(results[summary.s1], oracle.s1, `${label} S1`, 1e-5, 1e-5);
  expectField(results[summary.s2], oracle.s2, `${label} S2`, 1e-5, 1e-5);
  expectField(results[summary.mean], oracle.mean, `${label} mean`, 1e-5, 1e-5);
  expectField(results[summary.variance], oracle.variance, `${label} variance`, 1e-6, 1e-4);
  expect(results[summary.blackCount], `${label} blacks`).toBe(oracle.blackCount);
  expect(results[summary.islandCount], `${label} islands`).toBe(oracle.islandCount);
  for (const statistic of ['moran', 'geary', 'getisOrdG', 'bivariateMoran'] as const) {
    const block = LAYOUT[statistic];
    if (!statistics.includes(statistic)) {
      for (let field = 0; field < 8; field++) {
        expect(results[block + field], `${label} ${statistic} disabled`).toBeNaN();
      }
      continue;
    }
    const expected = oracle[statistic];
    for (const [name, field] of Object.entries(FIELD)) {
      const tolerance = name.startsWith('p')
        ? [2e-5, 5e-3]
        : name.startsWith('z')
          ? [2e-3, 2e-3]
          : [1e-6, 1e-3];
      expectField(
        results[block + field],
        expected[name as keyof typeof expected],
        `${label} ${statistic}.${name}`,
        tolerance[0],
        tolerance[1]
      );
    }
  }
  if (statistics.includes('joinCount')) {
    expect(joinCounts, `${label} ordered join counts`).toEqual(oracle.joinCount.ordered);
    for (const [name, field] of Object.entries(GPU_GLOBAL_JOIN_COUNT_FIELD)) {
      const expected = oracle.joinCount[name as keyof typeof GPU_GLOBAL_JOIN_COUNT_FIELD];
      const tolerance = name.startsWith('p')
        ? [2e-5, 5e-3]
        : name.startsWith('z')
          ? [2e-3, 2e-3]
          : [1e-5, 1e-3];
      expectField(
        results[LAYOUT.joinCount + field],
        expected,
        `${label} join.${name}`,
        tolerance[0],
        tolerance[1]
      );
    }
  }
}

function createRandomPositions(count: number, seed: number): Float64Array {
  const random = createSeededRandom(seed);
  return Float64Array.from({length: count * 2}, () => random() * 100);
}

it('GPUGlobalSpatialStatistics matches the oracle on a rook checkerboard and a smooth grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const columns = 20;
  const weights = createGridWeights(columns, 15);
  const rows = columns * 15;
  const checkerboard = Float32Array.from(
    {length: rows},
    (_, index) => (Math.floor(index / columns) + (index % columns)) % 2
  );
  const secondValues = Float32Array.from({length: rows}, (_, index) => Math.sin(index * 0.37));
  const harness = createHarness(device, {weights, values: checkerboard, secondValues});
  const result = await harness.run();
  expect(result.results[LAYOUT.moran]).toBeCloseTo(-1, 5);
  expect(result.results[LAYOUT.joinCount + GPU_GLOBAL_JOIN_COUNT_FIELD.blackBlack]).toBe(0);
  expectMatchesOracle(
    result.results,
    result.joinCounts,
    computeGlobalSpatialStatisticsOracle({weights, values: checkerboard, secondValues}),
    'checkerboard'
  );
  // Smooth surface: strong positive autocorrelation.
  const smooth = Float32Array.from(
    {length: rows},
    (_, index) => 10 + (index % columns) + Math.floor(index / columns)
  );
  const smoothResult = await harness.run({values: smooth});
  expect(smoothResult.results[LAYOUT.moran + FIELD.zRandomization]).toBeGreaterThan(10);
  expectMatchesOracle(
    smoothResult.results,
    smoothResult.joinCounts,
    computeGlobalSpatialStatisticsOracle({weights, values: smooth, secondValues}),
    'smooth'
  );
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});

it('GPUGlobalSpatialStatistics matches the oracle on asymmetric kNN, row-standardized and inverse-distance weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPositions(600, 3);
  const random = createSeededRandom(4);
  const values = Float32Array.from(
    {length: 600},
    (_, index) => 5 + positions[index * 2] * 0.05 + random() * 2
  );
  const secondValues = Float32Array.from(
    {length: 600},
    (_, index) => positions[index * 2 + 1] * 0.1 + random()
  );
  for (const [label, weights] of [
    ['knn binary', createKnnWeights(positions, 6, false)],
    ['knn row standardized', createKnnWeights(positions, 8, true)],
    ['knn inverse distance', createKnnWeights(positions, 5, false, true)]
  ] as const) {
    const harness = createHarness(device, {weights, values, secondValues});
    const result = await harness.run();
    expectMatchesOracle(
      result.results,
      result.joinCounts,
      computeGlobalSpatialStatisticsOracle({weights, values, secondValues}),
      label
    );
    // Join counts on a binary column with the same weights.
    const binary = Float32Array.from(values, value => (value > 7 ? 1 : 0));
    const binaryResult = await harness.run({values: binary});
    expectMatchesOracle(
      binaryResult.results,
      binaryResult.joinCounts,
      computeGlobalSpatialStatisticsOracle({weights, values: binary, secondValues}),
      `${label} binary`
    );
    harness.destroy();
  }
});

it('GPUGlobalSpatialStatistics excludes masked and NaN rows, handles islands and ignores capacity slack', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = createRandomPositions(400, 8);
  const knn = createKnnWeights(positions, 4, false);
  // Three appended island rows, and capacity slack holding garbage past offsets[rows].
  const lists = Array.from({length: 403}, (_, row) =>
    row < 400
      ? Array.from(
          knn.neighbors.subarray(knn.offsets[row], knn.offsets[row + 1]),
          (neighbor, slot): [number, number] => [neighbor, knn.weights[knn.offsets[row] + slot]]
        )
      : []
  );
  const weights = createWeights(lists, 50);
  const random = createSeededRandom(9);
  const values = Float32Array.from({length: 403}, () => random() * 10);
  values[5] = NaN;
  values[17] = Infinity;
  const mask = Uint32Array.from({length: 403}, (_, row) => (row >= 400 || random() < 0.85 ? 1 : 0));
  const harness = createHarness(device, {weights, values, mask}, [
    'moran',
    'geary',
    'getisOrdG',
    'joinCount'
  ]);
  const result = await harness.run();
  const oracle = computeGlobalSpatialStatisticsOracle({weights, values, mask});
  expect(oracle.islandCount).toBeGreaterThan(0);
  expectMatchesOracle(result.results, result.joinCounts, oracle, 'masked', [
    'moran',
    'geary',
    'getisOrdG',
    'joinCount'
  ]);
  // A new mask between submissions, without rebuilding.
  const nextMask = Uint32Array.from({length: 403}, (_, index) => (index % 3 === 0 ? 0 : 1));
  const next = await harness.run({mask: nextMask});
  expectMatchesOracle(
    next.results,
    next.joinCounts,
    computeGlobalSpatialStatisticsOracle({weights, values, mask: nextMask}),
    'next mask',
    ['moran', 'geary', 'getisOrdG', 'joinCount']
  );
  expect(harness.buildCount).toBe(1);
  harness.destroy();
});

it('GPUGlobalSpatialStatistics reports NaN inference for degenerate inputs and is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const weights = createGridWeights(3, 1);
  const tiny = createHarness(device, {
    weights,
    values: new Float32Array([1, 2, 3]),
    secondValues: new Float32Array([1, 0, 1])
  });
  const tinyResult = await tiny.run();
  expect(tinyResult.results[LAYOUT.moran + FIELD.varianceRandomization]).toBeNaN();
  expect(tinyResult.results[LAYOUT.moran + FIELD.zNormality]).toBeNaN();
  // Constant values: inference stays undefined (the GPU mean may differ from 2 by one f32 ulp).
  const constant = await tiny.run({values: new Float32Array([2, 2, 2])});
  expect(constant.results[LAYOUT.moran + FIELD.zRandomization]).toBeNaN();
  tiny.destroy();

  const positions = createRandomPositions(5000, 12);
  const random = createSeededRandom(13);
  const values = Float32Array.from(
    {length: 5000},
    (_, index) => positions[index * 2] + random() * 30
  );
  const gridWeights = createGridWeights(100, 50, true, true);
  const harness = createHarness(device, {
    weights: gridWeights,
    values,
    secondValues: values.map(value => -value)
  });
  const first = await harness.run();
  const second = await harness.run();
  expect(second.resultBits).toEqual(first.resultBits);
  expectMatchesOracle(
    first.results,
    first.joinCounts,
    computeGlobalSpatialStatisticsOracle({
      weights: gridWeights,
      values,
      secondValues: values.map(value => -value)
    }),
    'queen 100x50'
  );
  harness.destroy();
});

it('GPUGlobalSpatialStatistics consumes the CSR written by GPUNeighborSearch in the same graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const count = 2000;
  const k = 8;
  const random = createSeededRandom(21);
  const positions = Float32Array.from({length: count * 2}, () => random() * 1000);
  const values = Float32Array.from(
    {length: count},
    (_, index) => Math.sin(positions[index * 2] / 150) + random() * 0.5
  );
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'neighbor-parameters',
    format: 'float32',
    length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    values: getGPUNeighborSearchParameterValues({bounds: [0, 0, 1000, 1000], rowStandardize: true})
  });
  const buffers = {
    positions: createInputBuffer(device, positions),
    values: createInputBuffer(device, values),
    offsets: createOutputBuffer(device, count + 1),
    neighbors: createOutputBuffer(device, count * k),
    weights: createOutputBuffer(device, count * k),
    overflow: createOutputBuffer(device, 1),
    results: createOutputBuffer(device, LAYOUT.length)
  };
  const graph = new GPUCommandGraph(device, {id: 'neighbors-to-statistics'});
  const weights = {
    offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', count + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', count * k),
    weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', count * k)
  };
  graph.add(
    new GPUNeighborSearch({
      mode: 'knn',
      k,
      gridSize: [32, 32],
      positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', count),
      parameters: parameterBuffer.importToGraph(graph),
      weights,
      overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
    })
  );
  graph.add(
    new GPUGlobalSpatialStatistics({
      weights,
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32', count),
      statistics: ['moran', 'geary', 'getisOrdG'],
      results: importGraphBuffer(graph, 'results', buffers.results, 'float32', LAYOUT.length)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const results = await readFloat32(buffers.results, LAYOUT.length);
  const csr: CPUSpatialWeights = {
    offsets: Uint32Array.from(await readUint32(buffers.offsets, count + 1)),
    neighbors: Uint32Array.from(await readUint32(buffers.neighbors, count * k)),
    weights: Float32Array.from(await readFloat32(buffers.weights, count * k))
  };
  expect((await readUint32(buffers.overflow, 1))[0]).toBe(0);
  expect(csr.offsets[count]).toBe(count * k);
  expectMatchesOracle(
    results,
    [0, 0, 0],
    computeGlobalSpatialStatisticsOracle({weights: csr, values}),
    'chained',
    ['moran', 'geary', 'getisOrdG']
  );
  expect(results[LAYOUT.moran + FIELD.zRandomization]).toBeGreaterThan(20);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
});
