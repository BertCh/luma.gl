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
import referenceCases from './skater-reference.json';
import {
  canonicalizePartition,
  computeComponentLabels,
  computeKruskalForest,
  evaluatePartition,
  getSquaredDistance,
  type RegionalizationCase
} from './spatial-regionalization-oracle';

const CASES = referenceCases as RegionalizationCase[];

const SPARE_SLOTS = 5;

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

/** Row-major attribute table where the first row is the anchor for ties to rows with equal values. */
function createRig(
  device: Device,
  csr: {offsets: number[]; neighbors: number[]},
  values: number[] | Float32Array,
  columns: number,
  options: {standardize: boolean; maximumRegionCount?: number; withSkater?: boolean}
) {
  const rows = csr.offsets.length - 1;
  const rig = new WeightsRig(device);
  // Spare slots past the last row end exercise the unused-capacity path.
  const weights = rig.uploadWeights(
    {
      offsets: csr.offsets,
      neighbors: csr.neighbors,
      weights: csr.neighbors.map(() => 1),
      distances: []
    },
    SPARE_SLOTS
  );
  const valuesView = rig.input(new Float32Array(values), 'float32', values.length);
  const flags = rig.output('uint32', csr.neighbors.length + SPARE_SLOTS);
  const componentLabels = rig.output('uint32', rows);
  const edgeIds = rig.output('uint32', rows);
  const edgeCount = rig.output('uint32', 1);
  const edgeOverflow = rig.output('uint32', 1);
  const endpoints = rig.output('uint32', 2 * rows);
  const costs = rig.output('float32', rows);
  const standardized = options.standardize ? rig.output('float32', rows * columns) : undefined;
  rig.graph.add(
    new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values: valuesView,
      columnCount: columns,
      standardize: options.standardize,
      treeEdgeFlags: flags.view,
      componentLabels: componentLabels.view,
      standardizedValues: standardized?.view,
      edges: {ids: edgeIds.view, count: edgeCount.view, overflow: edgeOverflow.view},
      edgeEndpoints: endpoints.view,
      edgeCosts: costs.view
    })
  );
  let skater;
  if (options.withSkater) {
    const maximumRegionCount = options.maximumRegionCount ?? 8;
    const parameters = rig.input(new Uint32Array([2, 1]), 'uint32', 2);
    const labels = rig.output('uint32', rows);
    const regionCount = rig.output('uint32', 1);
    const cutEdges = rig.output('uint32', maximumRegionCount - 1);
    const cutGains = rig.output('float32', maximumRegionCount - 1);
    const summary = rig.output('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
    const sizes = rig.output('uint32', rows);
    const within = rig.output('float32', rows);
    rig.graph.add(
      new GPUSkaterRegions({
        weights,
        treeEdgeFlags: flags.view,
        componentLabels: componentLabels.view,
        values: standardized?.view ?? valuesView,
        columnCount: columns,
        maximumRegionCount,
        parameters,
        labels: labels.view,
        regionCount: regionCount.view,
        cutEdges: cutEdges.view,
        cutGains: cutGains.view
      })
    );
    rig.graph.add(
      new GPURegionPartitionEvaluation({
        values: standardized?.view ?? valuesView,
        columnCount: columns,
        labels: labels.view,
        weights,
        summary: summary.view,
        regionSizes: sizes.view,
        regionWithinSsd: within.view
      })
    );
    skater = {parameters, labels, regionCount, cutEdges, cutGains, summary, sizes, within};
  }
  const compiled = rig.graph.compile();
  return {
    rig,
    compiled,
    rows,
    weights,
    flags,
    componentLabels,
    edgeIds,
    edgeCount,
    edgeOverflow,
    endpoints,
    costs,
    standardized,
    skater,
    encode: () => submitGraph(device, compiled, undefined),
    destroy: () => {
      compiled.destroy();
      rig.destroy();
    }
  };
}

async function readEdges(setup: ReturnType<typeof createRig>): Promise<[number, number][]> {
  const [count] = await readUint32(setup.edgeCount.buffer, 1);
  const pairs = await readUint32(setup.endpoints.buffer, 2 * count);
  const edges: [number, number][] = [];
  for (let index = 0; index < count; index++) {
    edges.push([
      Math.min(pairs[2 * index], pairs[2 * index + 1]),
      Math.max(pairs[2 * index], pairs[2 * index + 1])
    ]);
  }
  return edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

it('GPUSpatialWeightsMinimumSpanningTree matches scipy on lattices and an island forest', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const data of CASES) {
    const setup = createRig(device, data, data.values, data.columns, {standardize: false});
    setup.encode();
    const edges = await readEdges(setup);
    expect(edges, `${data.name} edges`).toEqual(data.mstEdges);
    const labels = await readUint32(setup.componentLabels.buffer, data.rows);
    expect(labels, `${data.name} labels`).toEqual(computeComponentLabels(data.rows, data.mstEdges));
    expect(new Set(labels).size).toBe(data.components);
    const [overflow] = await readUint32(setup.edgeOverflow.buffer, 1);
    expect(overflow).toBe(0);
    const costs = await readFloat32(setup.costs.buffer, edges.length);
    const total = costs.reduce((sum, cost) => sum + cost, 0);
    expect(Math.abs(total - data.mstCost)).toBeLessThan(1e-3 * data.mstCost);
    setup.destroy();
  }
});

it('GPUSpatialWeightsMinimumSpanningTree breaks cost ties by the lowest edge slot', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // A 9x9 rook lattice with attributes in {0, 1, 2}: cost ties everywhere. Exact integer costs.
  const width = 9;
  const rows = width * width;
  const random = createSeededRandom(17);
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < rows; row++) {
    const x = row % width;
    const candidates = [
      row - width,
      x > 0 ? row - 1 : -1,
      x < width - 1 ? row + 1 : -1,
      row + width
    ];
    for (const candidate of candidates) {
      if (candidate >= 0 && candidate < rows) neighbors.push(candidate);
    }
    offsets.push(neighbors.length);
  }
  const values = Array.from({length: rows * 2}, () => Math.floor(random() * 3));
  const expected = computeKruskalForest(offsets, neighbors, (row, neighbor) =>
    getSquaredDistance(values, 2, row, neighbor)
  );
  const setup = createRig(device, {offsets, neighbors}, values, 2, {standardize: false});
  setup.encode();
  const flags = await readUint32(setup.flags.buffer, neighbors.length);
  const slots = flags.flatMap((flag, slot) => (flag ? [slot] : []));
  expect(slots).toEqual(expected);
  expect(slots).toHaveLength(rows - 1);
  const [count] = await readUint32(setup.edgeCount.buffer, 1);
  expect(await readUint32(setup.edgeIds.buffer, count)).toEqual(expected);
  setup.destroy();
});

it('GPUSkaterRegions and GPURegionPartitionEvaluation match spopt SpanningForest', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const data of CASES) {
    const maximumRegionCount = data.components + 9;
    const setup = createRig(device, data, data.values, data.columns, {
      standardize: false,
      withSkater: true,
      maximumRegionCount
    });
    const skater = setup.skater!;
    for (const run of data.runs) {
      // spopt's islands='increase' adds the tree count to n_clusters when the graph is disconnected.
      const target = run.k + (data.components > 1 ? data.components : 0);
      setup.rig.bufferOf(skater.parameters).write(new Uint32Array([target, run.floor ?? 1]));
      setup.encode();
      const labels = await readUint32(skater.labels.buffer, data.rows);
      const label = `${data.name} k=${run.k} floor=${run.floor}`;
      expect(canonicalizePartition(labels), label).toEqual(canonicalizePartition(run.labels));
      const regionCount = new Set(run.labels).size;
      expect((await readUint32(skater.regionCount.buffer, 1))[0], `${label} count`).toBe(
        regionCount
      );
      // Evaluation on the same labels equals the spopt score and the CPU oracle.
      const summary = await readFloat32(
        skater.summary.buffer,
        GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
      );
      const expected = evaluatePartition(
        data.values,
        data.columns,
        run.labels,
        data.offsets,
        data.neighbors
      );
      const {
        regionCount: countIndex,
        withinSsd,
        betweenSsd,
        totalSsd,
        minimumSize,
        maximumSize,
        crossLinkFraction,
        ignoredRows
      } = GPU_REGION_PARTITION_EVALUATION_LAYOUT;
      expect(summary[countIndex], `${label} regions`).toBe(expected.regionCount);
      expect(summary[withinSsd]).toBeCloseTo(run.score, 2);
      expect(summary[withinSsd]).toBeCloseTo(expected.withinSsd, 2);
      expect(summary[totalSsd]).toBeCloseTo(expected.totalSsd, 2);
      expect(summary[betweenSsd]).toBeCloseTo(expected.totalSsd - expected.withinSsd, 2);
      expect(summary[minimumSize]).toBe(expected.minimumSize);
      expect(summary[maximumSize]).toBe(expected.maximumSize);
      expect(summary[crossLinkFraction]).toBeCloseTo(expected.crossLinkFraction, 5);
      expect(summary[ignoredRows]).toBe(0);
      // The cut log: applied cuts equal the extra regions, and their gains sum to the SSD drop.
      const cuts = await readUint32(skater.cutEdges.buffer, maximumRegionCount - 1);
      const gains = await readFloat32(skater.cutGains.buffer, maximumRegionCount - 1);
      const applied = cuts.filter(edge => edge !== GPU_SKATER_NO_CUT).length;
      expect(applied, `${label} cuts`).toBe(regionCount - data.components);
      const gainSum = gains.slice(0, applied).reduce((sum, gain) => sum + gain, 0);
      const forestSsd = evaluatePartition(
        data.values,
        data.columns,
        computeComponentLabels(data.rows, data.mstEdges),
        data.offsets,
        data.neighbors
      ).withinSsd;
      expect(gainSum).toBeCloseTo(forestSsd - expected.withinSsd, 1);
    }
    setup.destroy();
  }
});

it('GPURegionPartitionEvaluation ignores unlabelled rows and honours labelCapacity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rig = new WeightsRig(device);
  const values = [0, 0, 1, 1, 10, 10, 11, 11, 100, 100];
  const labels = [0, 0, 1, 1, 0xffffffff, 7, 1, 1, 0, 0];
  const summary = rig.output('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
  rig.graph.add(
    new GPURegionPartitionEvaluation({
      values: rig.input(new Float32Array(values), 'float32', 10),
      labels: rig.input(new Uint32Array(labels), 'uint32', 10),
      labelCapacity: 4,
      summary: summary.view
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const words = await readFloat32(summary.buffer, GPU_REGION_PARTITION_EVALUATION_LAYOUT.length);
  // Rows 0,1,8,9 form label 0 (SSD of {0,0,100,100}); rows 2,3,6,7 label 1.
  const kept = [0, 1, 2, 3, 6, 7, 8, 9];
  const expected = evaluatePartition(
    kept.map(row => values[row]),
    1,
    kept.map(row => labels[row])
  );
  expect(words[GPU_REGION_PARTITION_EVALUATION_LAYOUT.regionCount]).toBe(2);
  expect(words[GPU_REGION_PARTITION_EVALUATION_LAYOUT.ignoredRows]).toBe(2);
  expect(words[GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSsd]).toBeCloseTo(
    expected.withinSsd,
    2
  );
  expect(words[GPU_REGION_PARTITION_EVALUATION_LAYOUT.totalSsd]).toBeCloseTo(expected.totalSsd, 1);
  compiled.destroy();
  rig.destroy();
});
