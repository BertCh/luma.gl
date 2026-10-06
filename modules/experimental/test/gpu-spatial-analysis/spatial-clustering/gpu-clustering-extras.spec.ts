// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSpatialClusteringParameterValues,
  GPUKMeans,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  type GPUKMeansInitialization,
  type GPUSpatialClusteringParameters
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {computeKMeansOracle} from './kmeans-oracle';
import {
  clusterPointsOracle,
  createClusteredPoints,
  createSeededRandom,
  separateNearEpsilonPairs
} from './spatial-clustering-oracle';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

async function runDbscan(
  device: Device,
  positions: Float32Array,
  parameters: GPUSpatialClusteringParameters,
  denseBoxShortcut: boolean
) {
  const rows = positions.length / 2;
  const rig = new WeightsRig(device);
  const labels = rig.output('uint32', rows);
  const rootRows = rig.output('uint32', rows);
  const coreFlags = rig.output('uint32', rows);
  const clusterCount = rig.output('uint32', 1);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'dbscan-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: getGPUSpatialClusteringParameterValues(parameters)
  });
  rig.run(
    new GPUSpatialClustering({
      positions: rig.input(positions, 'float32x2', rows),
      parameters: parameterBuffer.importToGraph(rig.graph),
      gridSize: [32, 32],
      denseBoxShortcut,
      labels: labels.view,
      rootRows: rootRows.view,
      coreFlags: coreFlags.view,
      clusterCount: clusterCount.view
    })
  );
  const result = {
    labels: await readUint32(labels.buffer, rows),
    rootRows: await readUint32(rootRows.buffer, rows),
    coreFlags: await readUint32(coreFlags.buffer, rows),
    clusterCount: (await readUint32(clusterCount.buffer, 1))[0]
  };
  parameterBuffer.destroy();
  rig.destroy();
  return result;
}

/** Number of points whose dense box (side 0.7 epsilon) holds at least `minimumPoints` points. */
function countDenseBoxPoints(
  positions: Float32Array,
  epsilon: number,
  minimumPoints: number,
  minimum: number
) {
  const side = Math.fround(epsilon) * 0.7;
  const boxes = new Map<string, number>();
  const keyOf = (row: number) =>
    `${Math.floor((positions[2 * row] - minimum) / side)},${Math.floor((positions[2 * row + 1] - minimum) / side)}`;
  const rows = positions.length / 2;
  for (let row = 0; row < rows; row++) boxes.set(keyOf(row), (boxes.get(keyOf(row)) ?? 0) + 1);
  let covered = 0;
  for (let row = 0; row < rows; row++) if ((boxes.get(keyOf(row)) ?? 0) >= minimumPoints) covered++;
  return covered;
}

it('GPUSpatialClustering with denseBoxShortcut gives identical results to the plain algorithm', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const bounds = [-10, -10, 110, 110] as const;
  const frames: GPUSpatialClusteringParameters[] = [
    {bounds, epsilon: 12, minimumPoints: 40},
    {bounds, epsilon: 6, minimumPoints: 15},
    {bounds, epsilon: 1.5, minimumPoints: 5},
    {bounds, epsilon: 4, minimumPoints: 1},
    {bounds: [20, 20, 80, 80], epsilon: 8, minimumPoints: 20}
  ];
  const epsilons = [...new Set(frames.map(frame => frame.epsilon))];
  const positions = separateNearEpsilonPairs(
    createClusteredPoints(11, {
      pointCount: 1500,
      blobCount: 6,
      extent: 100,
      blobSigma: 3,
      noiseFraction: 0.15
    }),
    epsilons,
    3
  );
  let shortcutHelped = false;
  for (const frame of frames) {
    const oracle = clusterPointsOracle(positions, frame);
    const plain = await runDbscan(device, positions, frame, false);
    const dense = await runDbscan(device, positions, frame, true);
    expect(plain.labels).toEqual(oracle.labels);
    expect(dense.labels).toEqual(oracle.labels);
    expect(dense.rootRows).toEqual(oracle.rootRows);
    expect(dense.coreFlags).toEqual(oracle.coreFlags);
    expect(dense.clusterCount).toBe(oracle.clusterCount);
    // Make sure the shortcut decided real core flags on at least one frame.
    if (countDenseBoxPoints(positions, frame.epsilon, frame.minimumPoints, -10) > 100) {
      shortcutHelped = true;
      expect(oracle.coreFlags.some(flag => flag === 1)).toBe(true);
    }
  }
  expect(shortcutHelped).toBe(true);
  expect(clusterPointsOracle(positions, frames[1]).clusterCount).toBeGreaterThan(1);
});

function createBlobs(seed: number, pointsPerBlob: number): Float32Array {
  const random = createSeededRandom(seed);
  const centers = [
    [0, 0],
    [100, 10],
    [50, 90]
  ];
  const points: number[] = [];
  for (let index = 0; index < pointsPerBlob * centers.length; index++) {
    const center = centers[index % centers.length];
    const gaussian = () => random() + random() + random() - 1.5;
    points.push(center[0] + gaussian() * 6, center[1] + gaussian() * 6);
  }
  const positions = new Float32Array(points);
  positions[10] = NaN;
  positions[23] = Infinity;
  return positions;
}

async function runKMeans(
  device: Device,
  positions: Float32Array,
  options: {k: number; iterations: number; initialization?: GPUKMeansInitialization; seed?: number}
) {
  const rows = positions.length / 2;
  const rig = new WeightsRig(device);
  const labels = rig.output('uint32', rows);
  const centers = rig.output('float32x2', options.k);
  const sizes = rig.output('uint32', options.k);
  const squaredDistances = rig.output('float32', rows);
  rig.run(
    new GPUKMeans({
      positions: rig.input(positions, 'float32x2', rows),
      ...options,
      labels: labels.view,
      centers: centers.view,
      sizes: sizes.view,
      squaredDistances: squaredDistances.view
    })
  );
  const result = {
    labels: await readUint32(labels.buffer, rows),
    centers: await readFloat32(centers.buffer, options.k * 2),
    sizes: await readUint32(sizes.buffer, options.k),
    squaredDistances: await readFloat32(squaredDistances.buffer, rows)
  };
  rig.destroy();
  return result;
}

it("GPUKMeans 'first-valid' matches the f64 Lloyd oracle and ignores non-finite rows", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const positions = createBlobs(5, 150);
  for (const [k, iterations] of [
    [3, 5],
    [5, 8]
  ]) {
    const expected = computeKMeansOracle(positions, k, iterations);
    const result = await runKMeans(device, positions, {k, iterations});
    expect(result.labels).toEqual(expected.labels);
    expect(result.sizes).toEqual(expected.sizes);
    expect(result.labels[5]).toBe(GPU_SPATIAL_CLUSTERING_NOISE);
    expect(result.sizes.reduce((sum, size) => sum + size, 0)).toBe(positions.length / 2 - 2);
    expect(result.sizes.every(size => size > 0)).toBe(true);
    for (let index = 0; index < 2 * k; index++) {
      expect(Math.abs(result.centers[index] - expected.centers[index])).toBeLessThan(
        1e-3 * (1 + Math.abs(expected.centers[index]))
      );
    }
    for (let row = 0; row < expected.labels.length; row++) {
      if (Number.isNaN(expected.squaredDistances[row])) {
        expect(result.squaredDistances[row]).toBeNaN();
      } else {
        expect(
          Math.abs(result.squaredDistances[row] - expected.squaredDistances[row])
        ).toBeLessThan(1e-3 * (1 + expected.squaredDistances[row]));
      }
    }
  }
});

it('GPUKMeans leaves centers empty when there are fewer valid points than k', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const positions = new Float32Array([1, 1, NaN, 2, 5, 5, 9, 9, 3, NaN]);
  const result = await runKMeans(device, positions, {k: 5, iterations: 3});
  expect(result.sizes).toEqual([1, 1, 1, 0, 0]);
  expect(result.centers[6]).toBeNaN();
  expect(result.centers[9]).toBeNaN();
  expect(result.labels).toEqual([
    0,
    GPU_SPATIAL_CLUSTERING_NOISE,
    1,
    2,
    GPU_SPATIAL_CLUSTERING_NOISE
  ]);
  expect(result.squaredDistances[0]).toBe(0);
});

it("GPUKMeans 'kmeans++' is deterministic per seed and recovers separated blobs", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const positions = createBlobs(8, 120);
  const outcomes = [];
  for (const seed of [1, 2, 3, 4]) {
    const first = await runKMeans(device, positions, {
      k: 3,
      iterations: 6,
      initialization: 'kmeans++',
      seed
    });
    const second = await runKMeans(device, positions, {
      k: 3,
      iterations: 6,
      initialization: 'kmeans++',
      seed
    });
    expect(second.labels).toEqual(first.labels);
    expect(Array.from(new Uint32Array(new Float32Array(second.centers).buffer))).toEqual(
      Array.from(new Uint32Array(new Float32Array(first.centers).buffer))
    );
    // Blobs are 100 apart with sigma 6: D-squared seeding picks one point per blob, so every
    // cluster is exactly one blob (rows alternate blobs by index modulo 3).
    for (let cluster = 0; cluster < 3; cluster++) {
      const members = first.labels
        .map((label, row) => (label === cluster ? row % 3 : -1))
        .filter(blob => blob >= 0);
      expect(new Set(members).size).toBe(1);
      expect(members.length).toBeGreaterThan(100);
    }
    expect(first.centers.every(Number.isFinite)).toBe(true);
    outcomes.push(first.centers.slice(0, 2).join(','));
  }
  // Seeds permute which blob seeds which cluster ID, so the result is seed dependent.
  expect(new Set(outcomes).size).toBeGreaterThan(1);
});
