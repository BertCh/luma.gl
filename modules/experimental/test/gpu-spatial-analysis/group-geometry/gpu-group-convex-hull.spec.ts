// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW,
  GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW,
  GPUGroupConvexHull
} from '../../../src/gpu-spatial-analysis/group-geometry';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeGroupConvexHullOracle,
  createRandom,
  getOracleGroupKeys
} from './group-geometry-oracle';

type Hulls = {
  hulls: number[][];
  sizes: number[];
  overflow: number;
  vertexIndices: number[];
  vertexPositions: number[];
  offsets: number[];
};

async function runHull(
  device: Device,
  positions: Float32Array,
  labels: Uint32Array,
  groupCount: number,
  options: {
    maximumVerticesPerGroup?: number;
    totalCapacity?: number;
    noiseLabel?: number;
    prefilterLevels?: number;
  } = {}
): Promise<Hulls> {
  const rows = labels.length;
  const capacity = options.totalCapacity ?? rows + groupCount;
  const graph = new GPUCommandGraph(device, {id: 'hull-graph'});
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
    vertexIndices: output(capacity),
    vertexPositions: output(capacity * 2),
    offsets: output(groupCount + 1),
    counts: output(groupCount),
    sizes: output(groupCount),
    overflow: output(1)
  };
  graph.add(
    new GPUGroupConvexHull({
      id: 'hull',
      positions: importGraphBuffer(graph, 'positions', input(positions), 'float32x2', rows),
      labels: importGraphBuffer(graph, 'labels', input(labels), 'uint32', rows),
      groupCount,
      noiseLabel: options.noiseLabel,
      prefilterLevels: options.prefilterLevels,
      maximumVerticesPerGroup: options.maximumVerticesPerGroup ?? 1024,
      totalCapacity: capacity,
      output: {
        vertexIndices: importGraphBuffer(graph, 'o-indices', out.vertexIndices, 'uint32', capacity),
        vertexPositions: importGraphBuffer(
          graph,
          'o-positions',
          out.vertexPositions,
          'float32x2',
          capacity
        ),
        offsets: importGraphBuffer(graph, 'o-offsets', out.offsets, 'uint32', groupCount + 1),
        counts: importGraphBuffer(graph, 'o-counts', out.counts, 'uint32', groupCount),
        sizes: importGraphBuffer(graph, 'o-sizes', out.sizes, 'uint32', groupCount),
        overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const counts = await readUint32(out.counts, groupCount);
  const offsets = await readUint32(out.offsets, groupCount + 1);
  const vertexIndices = await readUint32(out.vertexIndices, capacity);
  const result: Hulls = {
    hulls: counts.map((count, group) =>
      vertexIndices.slice(offsets[group], offsets[group] + count)
    ),
    sizes: await readUint32(out.sizes, groupCount),
    overflow: (await readUint32(out.overflow, 1))[0],
    vertexIndices,
    vertexPositions: await readFloat32(out.vertexPositions, capacity * 2),
    offsets
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** Random clusters, a grid with many collinear points, duplicates, a line, one point, two points. */
function createScene(seed: number) {
  const random = createRandom(seed);
  const points: number[] = [];
  const labels: number[] = [];
  const add = (x: number, y: number, label: number) => {
    points.push(x, y);
    labels.push(label);
  };
  // Group 0: gaussian-ish blob at a large offset (float32 cross products would misjudge).
  for (let index = 0; index < 1500; index++) {
    const radius = Math.sqrt(random()) * 50;
    const angle = random() * Math.PI * 2;
    add(1e6 + radius * Math.cos(angle), 2e6 + radius * Math.sin(angle), 0);
  }
  // Group 1: integer grid, so many exactly collinear boundary points.
  for (let x = 0; x < 12; x++) {
    for (let y = 0; y < 9; y++) {
      add(1e6 + 200 + x, 2e6 + y, 1);
    }
  }
  // Group 2: all collinear (including duplicates). Hull is the two extremes.
  for (let index = 0; index < 20; index++) {
    add(1e6 + index * 2, 2e6 - 300 + index * 2, 2);
  }
  add(1e6 + 6, 2e6 - 294, 2);
  // Group 3: a single point repeated.
  add(1e6 + 500, 2e6, 3);
  add(1e6 + 500, 2e6, 3);
  // Group 4: two distinct points.
  add(1e6 + 600, 2e6, 4);
  add(1e6 + 601, 2e6 + 1, 4);
  // Group 5: nearly collinear triple that float32 cross products may misclassify.
  add(1e6 + 700, 2e6, 5);
  add(1e6 + 700.5, 2e6 + 0.25, 5);
  add(1e6 + 701, 2e6 + 0.5, 5);
  add(1e6 + 700.5, 2e6 + 2, 5);
  // Group 6: empty. Noise rows and a NaN row.
  add(1e6, 2e6, 0xffffffff);
  add(NaN, 0, 0);
  add(1e6 + 10, 2e6 + 10, 7);
  return {positions: Float32Array.from(points), labels: Uint32Array.from(labels), groupCount: 7};
}

it('GPUGroupConvexHull matches the exact CPU monotone chain, including collinear and duplicate points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seed of [1, 7]) {
    const {positions, labels, groupCount} = createScene(seed);
    const actual = await runHull(device, positions, labels, groupCount);
    const keys = getOracleGroupKeys(positions, labels, groupCount);
    const expected = computeGroupConvexHullOracle(positions, keys, groupCount);
    expect(actual.overflow).toBe(0);
    for (let group = 0; group < groupCount; group++) {
      expect(actual.hulls[group], `hull ${group}`).toEqual(expected.hulls[group]);
      expect(actual.sizes[group]).toBe(expected.hulls[group].length);
    }
    // Sanity on the semantics so a silently failed compile (all zeros) cannot pass.
    expect(actual.hulls[0].length).toBeGreaterThan(8);
    expect(actual.hulls[1].length).toBe(4);
    expect(actual.hulls[2].length).toBe(2);
    expect(actual.hulls[3].length).toBe(1);
    expect(actual.hulls[4].length).toBe(2);
    expect(actual.hulls[6]).toEqual([]);
    // Counter-clockwise: positive signed area (shifted to the origin to stay in float64 range).
    for (const group of [0, 1]) {
      const ring = actual.hulls[group].map(row => [
        positions[row * 2] - 1e6,
        positions[row * 2 + 1] - 2e6
      ]);
      let area = 0;
      for (let index = 0; index < ring.length; index++) {
        const next = ring[(index + 1) % ring.length];
        area += ring[index][0] * next[1] - next[0] * ring[index][1];
      }
      expect(area).toBeGreaterThan(0);
    }
    // Duplicate point: lowest row index represents it.
    const pointRows = labels.reduce<number[]>((rows, label, row) => {
      if (label === 3) {
        rows.push(row);
      }
      return rows;
    }, []);
    expect(actual.hulls[3]).toEqual([pointRows[0]]);
    // Vertex positions follow the indices.
    const first = actual.hulls[0][0];
    expect(actual.vertexPositions[actual.offsets[0] * 2]).toBe(positions[first * 2]);
  }
});

it('GPUGroupConvexHull drops hulls over the per-group cap or total capacity and flags overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, labels, groupCount} = createScene(3);
  const keys = getOracleGroupKeys(positions, labels, groupCount);
  const expected = computeGroupConvexHullOracle(positions, keys, groupCount);
  expect(expected.hulls[0].length).toBeGreaterThan(10);

  const capped = await runHull(device, positions, labels, groupCount, {
    maximumVerticesPerGroup: 10
  });
  expect(capped.overflow & GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW).toBe(
    GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW
  );
  expect(capped.hulls[0]).toEqual([]);
  expect(capped.sizes[0]).toBe(expected.hulls[0].length);
  expect(capped.hulls[1]).toEqual(expected.hulls[1]);

  // Groups are admitted in group order; group 1 no longer fits, so it and every later group drop.
  const small = await runHull(device, positions, labels, groupCount, {
    totalCapacity: expected.hulls[0].length + expected.hulls[1].length - 1
  });
  expect(small.overflow & GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW).toBe(
    GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW
  );
  expect(small.hulls[0]).toEqual(expected.hulls[0]);
  for (let group = 1; group < groupCount; group++) {
    expect(small.hulls[group]).toEqual([]);
    expect(small.sizes[group]).toBe(expected.hulls[group].length);
  }
});

it('GPUGroupConvexHull honors noiseLabel', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Group 0 is a triangle plus a far noise point labeled 1 (noise), which must not enter any hull.
  const positions = Float32Array.from([0, 0, 4, 0, 0, 4, 100, 100, 1, 1]);
  const labels = Uint32Array.from([0, 0, 0, 1, 0]);
  const actual = await runHull(device, positions, labels, 2, {noiseLabel: 1});
  expect(actual.hulls[0]).toEqual([0, 1, 2]);
  expect(actual.hulls[1]).toEqual([]);
});

it('GPUGroupConvexHull matches the oracle above the chunk prefilter threshold', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 30k rows in 3 groups with a large offset, a duplicated block and a collinear group, so
  // chunks straddle group boundaries and duplicate points cross chunk boundaries.
  const random = createRandom(11);
  const rows = 30000;
  const groupCount = 3;
  const positions = new Float32Array(rows * 2);
  const labels = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const group = row < 15000 ? 0 : row < 15300 ? 2 : 1;
    labels[row] = group;
    if (group === 2) {
      positions[row * 2] = 1e6 + (row % 7);
      positions[row * 2 + 1] = 2e6 + (row % 7) * 2;
    } else {
      const radius = Math.sqrt(random()) * 50;
      const angle = random() * Math.PI * 2;
      positions[row * 2] = 1e6 + Math.round(radius * Math.cos(angle) * 4) / 4;
      positions[row * 2 + 1] = 2e6 + Math.round(radius * Math.sin(angle) * 4) / 4;
    }
  }
  const expected = computeGroupConvexHullOracle(
    positions,
    getOracleGroupKeys(positions, labels, groupCount),
    groupCount
  );
  const actual = await runHull(device, positions, labels, groupCount);
  expect(actual.overflow).toBe(0);
  expect(actual.hulls).toEqual(expected.hulls);
});

it('GPUGroupConvexHull gives the same hulls for every prefilter level count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Group 0: a large disk, group 1: points on a circle (almost all are hull vertices, so the
  // prefilter barely shrinks them), group 2: collinear, group 3: empty, group 4: a small blob
  // that straddles chunk boundaries. Lattice-rounded coordinates produce duplicates.
  const random = createRandom(23);
  const rows = 40000;
  const groupCount = 5;
  const positions = new Float32Array(rows * 2);
  const labels = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const group = row < 20000 ? 0 : row < 22000 ? 1 : row < 22300 ? 2 : 4;
    labels[row] = group;
    const angle = random() * Math.PI * 2;
    if (group === 2) {
      positions[row * 2] = 1e6 + (row % 11);
      positions[row * 2 + 1] = 2e6 + (row % 11) * 3;
    } else if (group === 1) {
      positions[row * 2] = 1e6 + 500 + Math.round(Math.cos(angle) * 4000) / 8;
      positions[row * 2 + 1] = 2e6 + Math.round(Math.sin(angle) * 4000) / 8;
    } else {
      const radius = Math.sqrt(random()) * (group === 4 ? 5 : 50);
      positions[row * 2] = 1e6 + 1000 + Math.round(radius * Math.cos(angle) * 4) / 4;
      positions[row * 2 + 1] = 2e6 + Math.round(radius * Math.sin(angle) * 4) / 4;
    }
  }
  const expected = computeGroupConvexHullOracle(
    positions,
    getOracleGroupKeys(positions, labels, groupCount),
    groupCount
  );
  expect(expected.hulls[1].length, 'the circle keeps many vertices').toBeGreaterThan(200);
  for (const prefilterLevels of [0, 1, 2, 3, 4]) {
    const actual = await runHull(device, positions, labels, groupCount, {
      prefilterLevels,
      maximumVerticesPerGroup: 4096
    });
    expect(actual.overflow, `levels ${prefilterLevels}`).toBe(0);
    expect(actual.hulls, `levels ${prefilterLevels}`).toEqual(expected.hulls);
  }
});

it('GPUGroupConvexHull handles sparse groups: empty first, middle and last groups, excluded labels', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 1000 groups of which only 0, 17, 18 and 998 own rows; labels 1000 and 5000 (out of range) are
  // excluded, so the last group 999 stays empty. Group boundaries come from binary search in the sorted keys.
  const random = createRandom(5);
  const groupCount = 1000;
  const used = [998, 17, 0, 18, 5000, 17, 0, 998];
  const rows = 600;
  const positions = new Float32Array(rows * 2);
  const labels = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    labels[row] = used[row % used.length];
    positions[row * 2] = 100 * (row % used.length) + random() * 50;
    positions[row * 2 + 1] = random() * 50;
  }
  labels[3] = 1000;
  const expected = computeGroupConvexHullOracle(
    positions,
    getOracleGroupKeys(positions, labels, groupCount),
    groupCount
  );
  const actual = await runHull(device, positions, labels, groupCount, {totalCapacity: rows});
  expect(actual.overflow).toBe(0);
  expect(actual.hulls).toEqual(expected.hulls);
  expect(actual.hulls[998].length).toBeGreaterThan(2);
  expect(actual.hulls[999]).toEqual([]);
  expect(actual.hulls[500]).toEqual([]);
});
