// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUGeographicDistributionParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/geographic-distribution';
import {
  GPU_GROUP_GEOMETRY_NO_MEDOID,
  GPUGroupGeometry
} from '../../../src/gpu-spatial-analysis/group-geometry';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeGeographicDistribution} from '../geographic-distribution/geographic-distribution-oracle';
import {
  computeGroupGeometryOracle,
  createRandom,
  getOracleGroupKeys
} from './group-geometry-oracle';

const ORIGIN: [number, number] = [100000, 200000];

type Scene = {
  positions: Float32Array;
  labels: Uint32Array;
  weights: Float32Array;
  groupCount: number;
  noiseLabel?: number;
};

/** Clusters at a large offset, DBSCAN-style noise, a NaN row, and an empty last group. */
function createScene(seed: number, rows: number, groupCount: number, noiseLabel?: number): Scene {
  const random = createRandom(seed);
  const positions = new Float32Array(rows * 2);
  const labels = new Uint32Array(rows);
  const weights = new Float32Array(rows);
  const usedGroups = Math.max(groupCount - 1, 1);
  for (let row = 0; row < rows; row++) {
    const group = Math.floor(random() * usedGroups);
    positions[row * 2] = ORIGIN[0] + group * 60 + (random() - 0.5) * 40 * (1 + group * 0.1);
    positions[row * 2 + 1] = ORIGIN[1] - group * 30 + (random() - 0.5) * 12;
    const roll = random();
    labels[row] =
      roll < 0.08 ? 0xffffffff : roll < 0.1 && noiseLabel !== undefined ? noiseLabel : group;
    weights[row] = random() < 0.05 ? -1 : 0.5 + random() * 2;
  }
  positions[6] = NaN;
  return {positions, labels, weights, groupCount, noiseLabel};
}

type Result = {
  counts: number[];
  bounds: number[];
  meanCenters: number[];
  weightedCenters: number[];
  weightSums: number[];
  medoids: number[];
  ellipses: number[];
  standardDistances: number[];
  overflow: number[];
};

async function run(
  device: Device,
  scene: Scene,
  options: {weighted: boolean; medoidMaximumGroupSize?: number}
): Promise<Result> {
  const rows = scene.labels.length;
  const groups = scene.groupCount;
  const graph = new GPUCommandGraph(device, {id: 'group-geometry-graph'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const out = {
    counts: output(groups),
    bounds: output(groups * 4),
    meanCenters: output(groups * 2),
    weightedCenters: output(groups * 2),
    weightSums: output(groups),
    medoids: output(groups),
    ellipses: output(groups * 3),
    standardDistances: output(groups),
    overflow: output(1)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
    values: getGPUGeographicDistributionParameterValues({origin: ORIGIN})
  });
  const weighted = options.weighted;
  graph.add(
    new GPUGroupGeometry({
      id: 'geometry',
      positions: importGraphBuffer(graph, 'positions', input(scene.positions), 'float32x2', rows),
      labels: importGraphBuffer(graph, 'labels', input(scene.labels), 'uint32', rows),
      weights: weighted
        ? importGraphBuffer(graph, 'weights', input(scene.weights), 'float32', rows)
        : undefined,
      groupCount: groups,
      noiseLabel: scene.noiseLabel,
      medoidMaximumGroupSize: options.medoidMaximumGroupSize,
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        counts: importGraphBuffer(graph, 'o-counts', out.counts, 'uint32', groups),
        bounds: importGraphBuffer(graph, 'o-bounds', out.bounds, 'float32x4', groups),
        meanCenters: importGraphBuffer(graph, 'o-mean', out.meanCenters, 'float32x2', groups),
        weightedCenters: weighted
          ? importGraphBuffer(graph, 'o-weighted', out.weightedCenters, 'float32x2', groups)
          : undefined,
        weightSums: weighted
          ? importGraphBuffer(graph, 'o-weight-sums', out.weightSums, 'float32', groups)
          : undefined,
        medoidIndices: importGraphBuffer(graph, 'o-medoids', out.medoids, 'uint32', groups),
        ellipses: importGraphBuffer(graph, 'o-ellipses', out.ellipses, 'float32', groups * 3),
        standardDistances: importGraphBuffer(
          graph,
          'o-std',
          out.standardDistances,
          'float32',
          groups
        ),
        overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: Result = {
    counts: await readUint32(out.counts, groups),
    bounds: await readFloat32(out.bounds, groups * 4),
    meanCenters: await readFloat32(out.meanCenters, groups * 2),
    weightedCenters: await readFloat32(out.weightedCenters, groups * 2),
    weightSums: await readFloat32(out.weightSums, groups),
    medoids: await readUint32(out.medoids, groups),
    ellipses: await readFloat32(out.ellipses, groups * 3),
    standardDistances: await readFloat32(out.standardDistances, groups),
    overflow: await readUint32(out.overflow, 1)
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectClose(name: string, actual: number[], expected: number[], tolerance: number) {
  expect(actual.length, name).toBeGreaterThanOrEqual(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(actual[index], `${name}[${index}]`).toBeNaN();
    } else {
      expect(Math.abs(actual[index] - expected[index]), `${name}[${index}]`).toBeLessThanOrEqual(
        tolerance
      );
    }
  }
}

for (const [seed, groupCount, noiseLabel, weighted] of [
  [1, 6, undefined, true],
  [2, 9, 2, false],
  [3, 1, undefined, false]
] as const) {
  it(`GPUGroupGeometry matches the CPU oracle (seed ${seed}, ${groupCount} groups, noise ${noiseLabel}, weighted ${weighted})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createScene(seed, 2000, groupCount, noiseLabel);
    const actual = await run(device, scene, {weighted});
    const keys = getOracleGroupKeys(scene.positions, scene.labels, groupCount, noiseLabel);
    const expected = computeGroupGeometryOracle(scene.positions, keys, groupCount);
    expect(expected.counts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(1200);
    expect(actual.counts).toEqual(expected.counts);
    // Bounds are exact: min and max of the same float32 values.
    expectClose('bounds', actual.bounds, expected.bounds, 0);
    expect(actual.medoids).toEqual(expected.medoids);
    expect(actual.overflow[0]).toBe(0);
    const unweighted = computeGeographicDistribution({
      positions: scene.positions,
      groupIds: Uint32Array.from(keys),
      groupCount,
      origin: ORIGIN
    });
    expectClose('meanCenters', actual.meanCenters, unweighted.meanCenters, 0.05);
    if (weighted) {
      const reference = computeGeographicDistribution({
        positions: scene.positions,
        weights: scene.weights,
        groupIds: Uint32Array.from(keys),
        groupCount,
        origin: ORIGIN
      });
      expectClose('weightedCenters', actual.weightedCenters, reference.meanCenters, 0.05);
      expectClose('weightSums', actual.weightSums, reference.weightSums, 0.1);
      expectClose('standardDistances', actual.standardDistances, reference.standardDistances, 0.1);
      expect(actual.weightedCenters[0]).toBeGreaterThan(ORIGIN[0] - 100);
    } else {
      expectClose('standardDistances', actual.standardDistances, unweighted.standardDistances, 0.1);
      expectClose(
        'ellipses[sigma]',
        actual.ellipses.filter((_, i) => i % 3 !== 0),
        unweighted.ellipses.filter((_, i) => i % 3 !== 0),
        0.2
      );
    }
    if (groupCount > 1) {
      const last = groupCount - 1;
      expect(actual.counts[last]).toBe(0);
      expect(actual.medoids[last]).toBe(GPU_GROUP_GEOMETRY_NO_MEDOID);
      expect(actual.bounds[last * 4]).toBeNaN();
      expect(actual.meanCenters[last * 2]).toBeNaN();
    }
    if (noiseLabel !== undefined) {
      expect(actual.counts[noiseLabel]).toBe(expected.counts[noiseLabel]);
    }
  });
}

it('GPUGroupGeometry medoid breaks exact ties with the lowest row index and flags oversize groups', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Group 0: rows 1 and 3 are the identical central point (tie, lowest index 1 wins); group 1 is
  // large enough to exceed the medoid cap of 5 rows.
  const points: number[] = [-4, 0, 0, 0, 4, 0, 0, 0, 0, 3];
  const labels: number[] = [0, 0, 0, 0, 0];
  for (let index = 0; index < 8; index++) {
    points.push(100 + index, 50);
    labels.push(1);
  }
  const scene: Scene = {
    positions: Float32Array.from(points),
    labels: Uint32Array.from(labels),
    weights: new Float32Array(labels.length).fill(1),
    groupCount: 2
  };
  const actual = await run(device, scene, {weighted: false, medoidMaximumGroupSize: 5});
  expect(actual.counts).toEqual([5, 8]);
  // Rows 1 and 3 are both (0, 0) and have equal summed distance, row 1 is the lowest index.
  expect(actual.medoids[0]).toBe(1);
  expect(actual.medoids[1]).toBe(GPU_GROUP_GEOMETRY_NO_MEDOID);
  expect(actual.overflow[0]).toBe(1);
  expect(actual.bounds.slice(0, 4)).toEqual([-4, 0, 4, 3]);
  expect(actual.bounds.slice(4, 8)).toEqual([100, 50, 107, 50]);
});
