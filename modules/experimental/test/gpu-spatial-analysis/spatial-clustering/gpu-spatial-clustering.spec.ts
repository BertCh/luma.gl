// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSpatialClusteringParameterValues,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  type GPUSpatialClusteringParameters
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  clusterPointsOracle,
  createClusteredPoints,
  createSeededRandom,
  separateNearEpsilonPairs,
  type SpatialClusteringOracleResult
} from './spatial-clustering-oracle';

const GARBAGE = 0x7f7f7f7f;

type HarnessOptions = {
  positions: Float32Array;
  parameters: GPUSpatialClusteringParameters;
  gridSize?: readonly [number, number];
  capacity?: number;
  sourceIds?: Uint32Array;
  sumOrder?: 'atomic' | 'sorted';
};

type HarnessResult = {
  labels: number[];
  rootRows: number[];
  coreFlags: number[];
  clusterCount: number;
  ids: number[];
  count: number;
  overflow: number;
  totalCount: number;
  sizes: number[];
  centroids: number[];
};

/** Builds one compiled graph with every optional output so tests can resubmit it per frame. */
function createHarness(device: Device, options: HarnessOptions) {
  const rows = options.positions.length / 2;
  const capacity = options.capacity ?? 64;
  const positionsBuffer = createInputBuffer(device, options.positions);
  const sourceIdsBuffer = options.sourceIds && createInputBuffer(device, options.sourceIds);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'spatial-clustering-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: getGPUSpatialClusteringParameterValues(options.parameters)
  });
  const outputs: Record<
    | 'labels'
    | 'rootRows'
    | 'coreFlags'
    | 'clusterCount'
    | 'ids'
    | 'count'
    | 'overflow'
    | 'totalCount'
    | 'sizes'
    | 'centroids',
    Buffer
  > = {
    labels: createOutputBuffer(device, rows),
    rootRows: createOutputBuffer(device, rows),
    coreFlags: createOutputBuffer(device, rows),
    clusterCount: createOutputBuffer(device, 1),
    ids: createOutputBuffer(device, capacity),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    totalCount: createOutputBuffer(device, 1),
    sizes: createOutputBuffer(device, capacity),
    centroids: createOutputBuffer(device, capacity * 2)
  };
  const graph = new GPUCommandGraph(device, {id: 'spatial-clustering-test'});
  const contributor = new GPUSpatialClustering({
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
    parameters: parameterBuffer.importToGraph(graph),
    gridSize: options.gridSize ?? [32, 32],
    sumOrder: options.sumOrder,
    sourceIds:
      sourceIdsBuffer && importGraphBuffer(graph, 'source-ids', sourceIdsBuffer, 'uint32', rows),
    labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', rows),
    rootRows: importGraphBuffer(graph, 'root-rows', outputs.rootRows, 'uint32', rows),
    coreFlags: importGraphBuffer(graph, 'core-flags', outputs.coreFlags, 'uint32', rows),
    clusterCount: importGraphBuffer(graph, 'cluster-count', outputs.clusterCount, 'uint32', 1),
    clusters: {
      ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', capacity),
      count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
      totalCount: importGraphBuffer(graph, 'total-count', outputs.totalCount, 'uint32', 1)
    },
    clusterSizes: importGraphBuffer(graph, 'sizes', outputs.sizes, 'uint32', capacity),
    clusterCentroids: importGraphBuffer(
      graph,
      'centroids',
      outputs.centroids,
      'float32x2',
      capacity
    )
  });
  graph.add(contributor);
  const compiled: CompiledGPUCommandGraph<void> = graph.compile();

  return {
    capacity,
    /** Writes `parameters` (a raw array when given as one), poisons every output, and runs. */
    async run(parameters: GPUSpatialClusteringParameters | Float32Array): Promise<HarnessResult> {
      parameterBuffer.write(
        parameters instanceof Float32Array
          ? parameters
          : getGPUSpatialClusteringParameterValues(parameters)
      );
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      return {
        labels: await readUint32(outputs.labels, rows),
        rootRows: await readUint32(outputs.rootRows, rows),
        coreFlags: await readUint32(outputs.coreFlags, rows),
        clusterCount: (await readUint32(outputs.clusterCount, 1))[0],
        ids: await readUint32(outputs.ids, count),
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        totalCount: (await readUint32(outputs.totalCount, 1))[0],
        sizes: await readUint32(outputs.sizes, capacity),
        centroids: await readFloat32(outputs.centroids, capacity * 2)
      };
    },
    writePositions(positions: Float32Array) {
      positionsBuffer.write(positions);
    },
    destroy() {
      compiled.destroy();
      positionsBuffer.destroy();
      sourceIdsBuffer?.destroy();
      parameterBuffer.destroy();
      for (const buffer of Object.values(outputs)) {
        buffer.destroy();
      }
    }
  };
}

/** Asserts exact agreement with the oracle, including bounded cluster outputs. */
function expectMatchesOracle(
  result: HarnessResult,
  oracle: SpatialClusteringOracleResult,
  capacity: number
): void {
  expect(result.labels).toEqual(oracle.labels);
  expect(result.rootRows).toEqual(oracle.rootRows);
  expect(result.coreFlags).toEqual(oracle.coreFlags);
  expect(result.clusterCount).toBe(oracle.clusterCount);
  const bounded = Math.min(oracle.clusterCount, capacity);
  expect(result.count).toBe(bounded);
  expect(result.totalCount).toBe(oracle.clusterCount);
  expect(result.overflow).toBe(oracle.clusterCount > capacity ? 1 : 0);
  expect(result.ids).toEqual(oracle.clusterRoots.slice(0, bounded));
  expect(result.sizes.slice(0, bounded)).toEqual(oracle.clusterSizes.slice(0, bounded));
  expect(result.sizes.slice(bounded)).toEqual(new Array(capacity - bounded).fill(0));
  for (let cluster = 0; cluster < capacity; cluster++) {
    for (const axis of [0, 1]) {
      const expected = cluster < bounded ? oracle.clusterCentroids[cluster * 2 + axis] : 0;
      expect(Math.abs(result.centroids[cluster * 2 + axis] - expected)).toBeLessThan(
        1e-3 + 1e-4 * Math.abs(expected)
      );
    }
  }
}

function getBlobProps(pointCount: number) {
  return {
    pointCount,
    blobCount: 6,
    extent: 100,
    blobSigma: 3,
    noiseFraction: 0.15
  };
}

it('GPUSpatialClustering matches the oracle on random clustered points and frame changes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const bounds = [-10, -10, 110, 110] as const;
  const frames: GPUSpatialClusteringParameters[] = [
    {bounds, epsilon: 1.5, minimumPoints: 5},
    // Much smaller than bounds / gridSize (3.75): many more points per cell than the lattice implies.
    {bounds, epsilon: 0.4, minimumPoints: 3},
    // Much larger than bounds / gridSize: the active lattice shrinks to fewer, wider cells.
    {bounds, epsilon: 12, minimumPoints: 40},
    {bounds, epsilon: 1.5, minimumPoints: 1},
    {bounds, epsilon: 1.5, minimumPoints: 25},
    {bounds: [20, 20, 80, 80], epsilon: 1.5, minimumPoints: 5},
    {bounds, epsilon: 1.5, minimumPoints: 5}
  ];
  const epsilons = [...new Set(frames.map(frame => frame.epsilon))];
  const first = separateNearEpsilonPairs(
    createClusteredPoints(11, getBlobProps(1500)),
    epsilons,
    3
  );
  const second = separateNearEpsilonPairs(
    createClusteredPoints(29, getBlobProps(1500)),
    epsilons,
    5
  );
  const sourceIds = Uint32Array.from({length: 1500}, (_, row) => 5000 + row * 3);
  const harness = createHarness(device, {
    positions: first,
    parameters: frames[0],
    sourceIds
  });
  try {
    for (const frame of frames) {
      const result = await harness.run(frame);
      expectMatchesOracle(result, clusterPointsOracle(first, frame, sourceIds), harness.capacity);
    }
    expect(clusterPointsOracle(first, frames[0]).clusterCount).toBeGreaterThan(2);
    // Same compiled graph, new point contents.
    harness.writePositions(second);
    for (const frame of [frames[0], frames[1], frames[2]]) {
      const result = await harness.run(frame);
      expectMatchesOracle(result, clusterPointsOracle(second, frame, sourceIds), harness.capacity);
    }
    // Invalid epsilon (bypassing the packer) excludes every point.
    const invalid = new Float32Array([...bounds, 0, 3, 0, 0]);
    const result = await harness.run(invalid);
    expect(result.labels.every(label => label === GPU_SPATIAL_CLUSTERING_NOISE)).toBe(true);
    expect(result.coreFlags.every(flag => flag === 0)).toBe(true);
    expect(result.clusterCount).toBe(0);
    expect(result.count).toBe(0);
    expect(result.overflow).toBe(0);
    const nan = new Float32Array([...bounds, NaN, 3, 0, 0]);
    expect((await harness.run(nan)).clusterCount).toBe(0);
    const inverted = new Float32Array([5, 5, 1, 1, 1.5, 3, 0, 0]);
    expect((await harness.run(inverted)).clusterCount).toBe(0);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUSpatialClustering assigns a border point to the adjacent cluster with the smaller root', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const clusterA = [0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1];
  const clusterB = [6, 0, 7, 0, 8, 0, 6, 1, 7, 1, 8, 1];
  const border = [4, 0];
  const parameters = {
    bounds: [-2, -2, 12, 4],
    epsilon: 2.5,
    minimumPoints: 6
  } as const;
  for (const [name, rowsInOrder] of [
    ['B first', [...clusterB, ...border, ...clusterA]],
    ['A first', [...clusterA, ...border, ...clusterB]]
  ] as const) {
    const positions = Float32Array.from(rowsInOrder);
    const harness = createHarness(device, {
      positions,
      parameters,
      gridSize: [4, 4]
    });
    try {
      const result = await harness.run(parameters);
      const oracle = clusterPointsOracle(positions, parameters);
      expectMatchesOracle(result, oracle, harness.capacity);
      expect(oracle.clusterCount, name).toBe(2);
      const borderRow = 6;
      expect(oracle.coreFlags[borderRow]).toBe(0);
      // Whichever cluster starts at row 0 has the smaller root and wins the tie.
      expect(result.labels[borderRow], name).toBe(0);
      expect(result.rootRows[borderRow], name).toBe(0);
    } finally {
      harness.destroy();
    }
  }
}, 60000);

it('GPUSpatialClustering finds one cluster for a long shuffled chain', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const count = 1000;
  const random = createSeededRandom(77);
  const order = Array.from({length: count}, (_, index) => index);
  for (let index = count - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  const positions = new Float32Array(count * 2);
  for (const [row, position] of order.entries()) {
    positions[row * 2] = position * 0.9;
    positions[row * 2 + 1] = 0.5;
  }
  for (const gridSize of [
    [16, 16],
    [4, 1]
  ] as const) {
    const parameters = {
      bounds: [-1, -1, 1000, 2],
      epsilon: 1,
      minimumPoints: 2
    } as const;
    const harness = createHarness(device, {positions, parameters, gridSize});
    try {
      const result = await harness.run(parameters);
      expectMatchesOracle(result, clusterPointsOracle(positions, parameters), harness.capacity);
      expect(result.clusterCount).toBe(1);
      expect(new Set(result.rootRows)).toEqual(new Set([0]));
      expect(new Set(result.labels)).toEqual(new Set([0]));
      expect(result.sizes[0]).toBe(count);
    } finally {
      harness.destroy();
    }
  }
}, 60000);

it('GPUSpatialClustering excludes NaN, infinite, and out-of-bounds points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = Float32Array.from([
    // A cluster that is truly dense.
    2,
    2,
    2.2,
    2,
    2,
    2.2,
    2.2,
    2.2,
    // Two valid points near the right edge, made core only if the outside point counted.
    9.9,
    5,
    9.95,
    5.3,
    10.4,
    5.1, // outside the bounds, within epsilon of both
    NaN,
    5,
    5,
    Infinity,
    -Infinity,
    1,
    5,
    NaN,
    -0.5,
    2.1 // outside on the left, within epsilon of the cluster
  ]);
  const parameters = {
    bounds: [0, 0, 10, 10],
    epsilon: 1,
    minimumPoints: 3
  } as const;
  const harness = createHarness(device, {
    positions,
    parameters,
    gridSize: [8, 8]
  });
  try {
    const result = await harness.run(parameters);
    const oracle = clusterPointsOracle(positions, parameters);
    expectMatchesOracle(result, oracle, harness.capacity);
    expect(result.clusterCount).toBe(1);
    expect(result.labels.slice(0, 4)).toEqual([0, 0, 0, 0]);
    expect(result.labels.slice(4)).toEqual(
      new Array(positions.length / 2 - 4).fill(GPU_SPATIAL_CLUSTERING_NOISE)
    );
  } finally {
    harness.destroy();
  }
}, 60000);

it('GPUSpatialClustering reports cluster capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const parameters = {
    bounds: [-10, -10, 110, 110],
    epsilon: 1.5,
    minimumPoints: 5
  } as const;
  const positions = separateNearEpsilonPairs(
    createClusteredPoints(11, getBlobProps(1500)),
    [parameters.epsilon],
    3
  );
  const oracle = clusterPointsOracle(positions, parameters);
  expect(oracle.clusterCount).toBeGreaterThan(2);
  const harness = createHarness(device, {positions, parameters, capacity: 2});
  try {
    const result = await harness.run(parameters);
    expectMatchesOracle(result, oracle, 2);
    expect(result.count).toBe(2);
    expect(result.overflow).toBe(1);
    expect(result.totalCount).toBe(oracle.clusterCount);
    expect(result.clusterCount).toBe(oracle.clusterCount);
  } finally {
    harness.destroy();
  }
}, 60000);

it('GPUSpatialClustering handles empty, single-point, and all-noise inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const parameters = {
    bounds: [0, 0, 10, 10],
    epsilon: 1,
    minimumPoints: 1
  } as const;
  const empty = createHarness(device, {
    positions: new Float32Array(0),
    parameters,
    capacity: 4
  });
  try {
    const result = await empty.run(parameters);
    expect(result.labels).toEqual([]);
    expect(result.clusterCount).toBe(0);
    expect(result.count).toBe(0);
    expect(result.totalCount).toBe(0);
    expect(result.overflow).toBe(0);
    expect(result.sizes).toEqual([0, 0, 0, 0]);
    expect(result.centroids).toEqual(new Array(8).fill(0));
  } finally {
    empty.destroy();
  }

  const single = createHarness(device, {
    positions: Float32Array.from([3, 4]),
    parameters,
    capacity: 4
  });
  try {
    const one = await single.run(parameters);
    expectMatchesOracle(one, clusterPointsOracle([3, 4], parameters), 4);
    expect(one.labels).toEqual([0]);
    expect(one.clusterCount).toBe(1);
    expect(one.sizes[0]).toBe(1);
    expect(one.centroids.slice(0, 2)).toEqual([3, 4]);
    const noise = await single.run({...parameters, minimumPoints: 2});
    expectMatchesOracle(noise, clusterPointsOracle([3, 4], {...parameters, minimumPoints: 2}), 4);
    expect(noise.labels).toEqual([GPU_SPATIAL_CLUSTERING_NOISE]);
    expect(noise.clusterCount).toBe(0);
  } finally {
    single.destroy();
  }

  // Sparse points with minimumPoints larger than the point count: all noise.
  const sparse = Float32Array.from([1, 1, 5, 5, 9, 9, 1, 9]);
  const sparseParameters = {
    bounds: [0, 0, 10, 10],
    epsilon: 1,
    minimumPoints: 10
  } as const;
  const sparseHarness = createHarness(device, {
    positions: sparse,
    parameters: sparseParameters
  });
  try {
    const result = await sparseHarness.run(sparseParameters);
    expectMatchesOracle(result, clusterPointsOracle(sparse, sparseParameters), 64);
    expect(result.clusterCount).toBe(0);
  } finally {
    sparseHarness.destroy();
  }
}, 60000);

it('GPUSpatialClustering maps clusters.ids through sourceIds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Two tight groups, the second listed first so root rows differ from label order.
  const positions = Float32Array.from([8, 8, 8.2, 8, 8, 8.2, 1, 1, 1.2, 1, 1, 1.2, 5, 5]);
  const sourceIds = Uint32Array.from([900, 901, 902, 700, 701, 702, 703]);
  const parameters = {
    bounds: [0, 0, 10, 10],
    epsilon: 0.5,
    minimumPoints: 3
  } as const;
  const harness = createHarness(device, {
    positions,
    parameters,
    sourceIds,
    gridSize: [4, 4]
  });
  try {
    const result = await harness.run(parameters);
    expectMatchesOracle(result, clusterPointsOracle(positions, parameters, sourceIds), 64);
    expect(result.ids).toEqual([900, 700]);
    expect(result.labels).toEqual([0, 0, 0, 1, 1, 1, GPU_SPATIAL_CLUSTERING_NOISE]);
  } finally {
    harness.destroy();
  }
}, 60000);

const BITS_PER_CENTROID_ROW = 2;

function getCentroidBits(centroids: number[]): number[] {
  const view = new DataView(new ArrayBuffer(4));
  return centroids.map(value => {
    view.setFloat32(0, value);
    return view.getUint32(0);
  });
}

it("GPUSpatialClustering sumOrder 'sorted' gives bitwise-identical centroids that match the oracle", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Large offset coordinates make f32 sums order-sensitive; few clusters make segments long.
  const offset = 4096;
  const pointCount = 30000;
  const base = createClusteredPoints(5, {
    pointCount,
    blobCount: 4,
    extent: 100,
    blobSigma: 3,
    noiseFraction: 0
  });
  const positions = base.map(value => value + offset);
  const parameters = {
    bounds: [offset - 10, offset - 10, offset + 110, offset + 110],
    epsilon: 0.6,
    minimumPoints: 5
  } as const;
  const capacity = 8;
  const results: Record<string, HarnessResult[]> = {};
  const timings: Record<string, number> = {};
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const harness = createHarness(device, {
      positions,
      parameters,
      capacity,
      sumOrder
    });
    try {
      await harness.run(parameters);
      const start = performance.now();
      results[sumOrder] = [];
      for (let run = 0; run < 5; run++) {
        results[sumOrder].push(await harness.run(parameters));
      }
      timings[sumOrder] = (performance.now() - start) / 5;
    } finally {
      harness.destroy();
    }
  }
  console.log(
    `spatial-clustering centroid sums, ${pointCount} points: atomic ${timings.atomic.toFixed(1)} ms, sorted ${timings.sorted.toFixed(1)} ms per submit+readback`
  );
  const sorted = results.sorted;
  expect(sorted[0].clusterCount).toBeGreaterThanOrEqual(2);
  // Long segments, so atomic accumulation order matters; stragglers beyond capacity are excluded.
  expect(Math.max(...sorted[0].sizes)).toBeGreaterThan(2000);
  const firstBits = getCentroidBits(sorted[0].centroids);
  expect(firstBits.length).toBe(capacity * BITS_PER_CENTROID_ROW);
  for (const result of sorted.slice(1)) {
    expect(getCentroidBits(result.centroids)).toEqual(firstBits);
    expect(result.labels).toEqual(sorted[0].labels);
  }
  // Atomic and sorted agree on labels and sizes, and on centroids within tolerance.
  expect(results.atomic[0].labels).toEqual(sorted[0].labels);
  expect(results.atomic[0].sizes).toEqual(sorted[0].sizes);
  for (let index = 0; index < sorted[0].centroids.length; index++) {
    expect(Math.abs(sorted[0].centroids[index] - results.atomic[0].centroids[index])).toBeLessThan(
      1e-2
    );
  }
  // Float64 mean of the GPU labels (the O(n^2) oracle is too slow at this size).
  const sums = new Float64Array(capacity * 2);
  const sizes = new Array(capacity).fill(0);
  sorted[0].labels.forEach((label, row) => {
    if (label < capacity) {
      sums[label * 2] += Math.fround(positions[row * 2]);
      sums[label * 2 + 1] += Math.fround(positions[row * 2 + 1]);
      sizes[label]++;
    }
  });
  expect(sorted[0].sizes).toEqual(sizes);
  for (let cluster = 0; cluster < capacity; cluster++) {
    for (const axis of [0, 1]) {
      const expected = sizes[cluster] ? sums[cluster * 2 + axis] / sizes[cluster] : 0;
      expect(Math.abs(sorted[0].centroids[cluster * 2 + axis] - expected)).toBeLessThan(
        1e-3 + 1e-4 * Math.abs(expected)
      );
    }
  }
}, 240000);

it("GPUSpatialClustering sumOrder 'sorted' matches the oracle with capacity overflow and noise", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const bounds = [-10, -10, 110, 110] as const;
  const parameters = {bounds, epsilon: 1.5, minimumPoints: 5} as const;
  const positions = separateNearEpsilonPairs(
    createClusteredPoints(11, getBlobProps(1500)),
    [parameters.epsilon],
    3
  );
  const oracle = clusterPointsOracle(positions, parameters);
  expect(oracle.clusterCount).toBeGreaterThan(2);
  for (const capacity of [64, 2]) {
    const harness = createHarness(device, {
      positions,
      parameters,
      capacity,
      sumOrder: 'sorted'
    });
    try {
      expectMatchesOracle(await harness.run(parameters), oracle, capacity);
    } finally {
      harness.destroy();
    }
  }
}, 120000);
