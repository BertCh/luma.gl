// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_SKATER_NO_CUT,
  GPURegionPartitionEvaluation,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {createSeededRandom} from '../spatial-weights/spatial-weights-oracle';
import {
  canonicalizePartition,
  computeKruskalForest,
  evaluatePartition,
  getSquaredDistance
} from './spatial-regionalization-oracle';

type Csr = {offsets: number[]; neighbors: number[]};

/** Rook lattice, neighbors ascending per row. */
function createLattice(width: number, height: number): Csr {
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < width * height; row++) {
    const x = row % width;
    for (const candidate of [
      row - width,
      x > 0 ? row - 1 : -1,
      x < width - 1 ? row + 1 : -1,
      row + width
    ]) {
      if (candidate >= 0 && candidate < width * height) neighbors.push(candidate);
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

/** Path graph `0 - 1 - ... - (rows - 1)`: the deepest possible tree and longest Boruvka hook chain. */
function createPath(rows: number): Csr {
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < rows; row++) {
    if (row > 0) neighbors.push(row - 1);
    if (row < rows - 1) neighbors.push(row + 1);
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

async function buildForest(device: Device, csr: Csr, values: number[], columns: number) {
  const rows = csr.offsets.length - 1;
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(
    {
      offsets: csr.offsets,
      neighbors: csr.neighbors,
      weights: csr.neighbors.map(() => 1),
      distances: []
    },
    0
  );
  const flags = rig.output('uint32', csr.neighbors.length);
  const componentLabels = rig.output('uint32', rows);
  rig.graph.add(
    new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values: rig.input(new Float32Array(values), 'float32', values.length),
      columnCount: columns,
      standardize: false,
      treeEdgeFlags: flags.view,
      componentLabels: componentLabels.view
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const treeFlags = await readUint32(flags.buffer, csr.neighbors.length);
  const labels = await readUint32(componentLabels.buffer, rows);
  compiled.destroy();
  rig.destroy();
  return {slots: treeFlags.flatMap((flag, slot) => (flag ? [slot] : [])), labels};
}

it('GPUSpatialWeightsMinimumSpanningTree handles a monotone path, random lattices and forests', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Costs (i + 1)^2 grow along the path, so every row's cheapest link points down the path and
  // Boruvka hooks one chain of every row: the pointer-jumping depth is at its maximum.
  for (const rows of [2, 3, 255, 257, 1500]) {
    const csr = createPath(rows);
    const values = Array.from({length: rows}, (_, row) => (row * (row + 1)) / 2);
    const forest = await buildForest(device, csr, values, 1);
    expect(forest.slots, `path ${rows}`).toEqual(
      computeKruskalForest(csr.offsets, csr.neighbors, (row, neighbor) =>
        getSquaredDistance(values, 1, row, neighbor)
      )
    );
    expect(new Set(forest.labels).size).toBe(1);
  }
  // Random continuous costs on lattices whose size is not a multiple of the workgroup.
  const random = createSeededRandom(23);
  for (const [width, height] of [
    [2, 1],
    [23, 17],
    [40, 31]
  ] as const) {
    const csr = createLattice(width, height);
    const rows = width * height;
    const values = Array.from({length: rows * 2}, () => Math.fround(random() * 10));
    const forest = await buildForest(device, csr, values, 2);
    const expected = computeKruskalForest(csr.offsets, csr.neighbors, (row, neighbor) =>
      getSquaredDistance(values, 2, row, neighbor)
    );
    expect(forest.slots, `lattice ${width}x${height}`).toEqual(expected);
  }
  // Two disjoint paths and an isolated row: three trees, labelled by their smallest row.
  const offsets = [0, 1, 3, 4, 5, 7, 8, 8];
  const neighbors = [1, 0, 2, 1, 4, 3, 5, 4];
  const forest = await buildForest(device, {offsets, neighbors}, [0, 1, 3, 10, 12, 15, 99], 1);
  expect(Array.from(forest.labels)).toEqual([0, 0, 0, 3, 3, 3, 6]);
});

/** Greedy SKATER on a given tree in f64: best SSD reduction, ties to the lowest edge slot. */
function runGreedyOracle(
  csr: Csr,
  treeSlots: number[],
  values: number[],
  cutCount: number,
  minimumSize: number
): {cuts: number[]; labels: number[]} {
  const rows = csr.offsets.length - 1;
  const edgeOf = new Map<string, number>();
  for (const slot of treeSlots) {
    let row = 0;
    while (csr.offsets[row + 1] <= slot) row++;
    edgeOf.set(`${row}-${csr.neighbors[slot]}`, slot);
  }
  const live = new Set(treeSlots);
  const adjacency: [number, number][][] = Array.from({length: rows}, () => []);
  for (const slot of treeSlots) {
    let row = 0;
    while (csr.offsets[row + 1] <= slot) row++;
    adjacency[row].push([csr.neighbors[slot], slot]);
    adjacency[csr.neighbors[slot]].push([row, slot]);
  }
  const collect = (start: number, blocked: number): number[] => {
    const seen = new Set([start]);
    const stack = [start];
    while (stack.length) {
      const node = stack.pop()!;
      for (const [other, slot] of adjacency[node]) {
        if (slot !== blocked && live.has(slot) && !seen.has(other)) {
          seen.add(other);
          stack.push(other);
        }
      }
    }
    return [...seen];
  };
  const ssd = (members: number[]) => {
    const mean = members.reduce((sum, row) => sum + values[row], 0) / members.length;
    return members.reduce((sum, row) => sum + (values[row] - mean) ** 2, 0);
  };
  const cuts: number[] = [];
  for (let step = 0; step < cutCount; step++) {
    let best = {gain: -1, slot: -1};
    for (const slot of [...live].sort((a, b) => a - b)) {
      let row = 0;
      while (csr.offsets[row + 1] <= slot) row++;
      const first = collect(row, slot);
      const second = collect(csr.neighbors[slot], slot);
      if (first.length < minimumSize || second.length < minimumSize) continue;
      const gain = ssd([...first, ...second]) - ssd(first) - ssd(second);
      if (gain > best.gain * (1 + 1e-9)) best = {gain, slot};
    }
    if (best.slot < 0) {
      cuts.push(GPU_SKATER_NO_CUT);
      continue;
    }
    cuts.push(best.slot);
    live.delete(best.slot);
  }
  const labels = Array.from({length: rows}, () => -1);
  for (let row = 0; row < rows; row++) {
    if (labels[row] < 0) for (const member of collect(row, -1)) labels[member] = row;
  }
  return {cuts, labels};
}

it('GPUSkaterRegions matches a greedy f64 oracle on a deep path and a lattice tree', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createSeededRandom(5);
  const scenes: {name: string; csr: Csr; minimumSize: number}[] = [
    {name: 'path-600', csr: createPath(600), minimumSize: 1},
    {name: 'path-600-floor-20', csr: createPath(600), minimumSize: 20},
    {name: 'lattice-18x14', csr: createLattice(18, 14), minimumSize: 3}
  ];
  const cutCount = 11;
  for (const {name, csr, minimumSize} of scenes) {
    const rows = csr.offsets.length - 1;
    // Piecewise-constant mean shifts plus noise give well separated greedy gains.
    const values = Array.from({length: rows}, (_, row) =>
      Math.fround(Math.floor(row / 37) * 3 + random() * 2)
    );
    const forest = await buildForest(device, csr, values, 1);
    const expected = runGreedyOracle(csr, forest.slots, values, cutCount, minimumSize);

    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(
      {
        offsets: csr.offsets,
        neighbors: csr.neighbors,
        weights: csr.neighbors.map(() => 1),
        distances: []
      },
      0
    );
    const flags = rig.input(
      Uint32Array.from(csr.neighbors.map((_, slot) => (forest.slots.includes(slot) ? 1 : 0))),
      'uint32',
      csr.neighbors.length
    );
    const parameters = rig.input(new Uint32Array([cutCount + 1, minimumSize]), 'uint32', 2);
    const labels = rig.output('uint32', rows);
    const regionCount = rig.output('uint32', 1);
    const cutEdges = rig.output('uint32', cutCount);
    rig.graph.add(
      new GPUSkaterRegions({
        weights,
        treeEdgeFlags: flags,
        componentLabels: rig.input(new Uint32Array(forest.labels), 'uint32', rows),
        values: rig.input(new Float32Array(values), 'float32', rows),
        maximumRegionCount: cutCount + 1,
        parameters,
        labels: labels.view,
        regionCount: regionCount.view,
        cutEdges: cutEdges.view
      })
    );
    const compiled = rig.graph.compile();
    submitGraph(device, compiled, undefined);
    const actualCuts = Array.from(await readUint32(cutEdges.buffer, cutCount));
    const actualLabels = await readUint32(labels.buffer, rows);
    expect(actualCuts, `${name} cuts`).toEqual(expected.cuts);
    expect(canonicalizePartition(actualLabels), `${name} labels`).toEqual(
      canonicalizePartition(expected.labels)
    );
    const applied = expected.cuts.filter(cut => cut !== GPU_SKATER_NO_CUT).length;
    expect((await readUint32(regionCount.buffer, 1))[0]).toBe(1 + applied);
    compiled.destroy();
    rig.destroy();
  }
});

it('GPURegionPartitionEvaluation scales with the default label capacity and skewed region sizes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createSeededRandom(31);
  const rows = 3001;
  const columns = 2;
  const values = Array.from({length: rows * columns}, () => Math.fround(random() * 20 - 5));
  // One region holds most rows, a few are tiny, some labels are unused, some rows are unlabelled
  // or beyond labelCapacity (default capacity is the row count, so only 0xffffffff is ignored).
  const labels = Array.from({length: rows}, (_, row) => {
    if (row % 97 === 0) return 0xffffffff;
    const draw = random();
    if (draw < 0.7) return 5;
    if (draw < 0.9) return 2999;
    return Math.floor(random() * 40) * 71;
  });
  const rig = new WeightsRig(device);
  const csr = createLattice(1, rows);
  const weights = rig.uploadWeights(
    {
      offsets: csr.offsets,
      neighbors: csr.neighbors,
      weights: csr.neighbors.map(() => 1),
      distances: []
    },
    0
  );
  const summary = rig.output('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
  const sizes = rig.output('uint32', rows);
  const within = rig.output('float32', rows);
  rig.graph.add(
    new GPURegionPartitionEvaluation({
      values: rig.input(new Float32Array(values), 'float32', values.length),
      columnCount: columns,
      labels: rig.input(new Uint32Array(labels), 'uint32', rows),
      weights,
      summary: summary.view,
      regionSizes: sizes.view,
      regionWithinSsd: within.view
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const words = await readFloat32(summary.buffer, GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
  const regionSizes = await readUint32(sizes.buffer, rows);
  const kept = labels.flatMap((label, row) => (label === 0xffffffff ? [] : [row]));
  const expected = evaluatePartition(
    kept.flatMap(row => [values[row * 2], values[row * 2 + 1]]),
    columns,
    kept.map(row => labels[row]),
    undefined,
    undefined
  );
  const layout = GPU_REGION_PARTITION_EVALUATION_LAYOUT;
  expect(words[layout.regionCount]).toBe(expected.regionCount);
  expect(words[layout.ignoredRows]).toBe(rows - kept.length);
  expect(words[layout.minimumSize]).toBe(expected.minimumSize);
  expect(words[layout.maximumSize]).toBe(expected.maximumSize);
  expect(Math.abs(words[layout.withinSsd] - expected.withinSsd)).toBeLessThan(
    1e-4 * expected.withinSsd
  );
  expect(Math.abs(words[layout.totalSsd] - expected.totalSsd)).toBeLessThan(
    1e-4 * expected.totalSsd
  );
  const histogram = new Map<number, number>();
  for (const row of kept) histogram.set(labels[row], (histogram.get(labels[row]) ?? 0) + 1);
  for (const [label, count] of histogram) {
    expect(regionSizes[label]).toBe(count);
  }
  compiled.destroy();
  rig.destroy();
});
