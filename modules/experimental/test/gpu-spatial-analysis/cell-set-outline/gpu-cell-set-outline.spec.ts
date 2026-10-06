// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {getPentagons, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {
  h3ToBigInt,
  joinCellKey,
  quadbinTileToCell,
  splitCellKey
} from '../cell-aggregation/cell-aggregation-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  getMultiPolygonSegments,
  getSegmentKey,
  matchSegments,
  outlineH3OnCPU,
  outlineQuadbinOnCPU,
  type OutlineSegment
} from './cell-set-outline-oracle';

type OutlineRun = {
  count: number;
  overflow: number;
  total: number;
  rows: number[];
  edgeIndices: number[];
  groups: number[];
  segments: OutlineSegment[];
};

function sortKeys(cells: bigint[]): bigint[] {
  return [...new Set(cells)].sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
}

async function runOutline(
  device: Device,
  family: 'h3' | 'quadbin',
  cells: bigint[],
  options: {groups?: Map<bigint, number>; capacity: number; count?: number}
): Promise<OutlineRun> {
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const words = new Uint32Array(cells.length * 2);
  cells.forEach((cell, row) => {
    const [low, high] = splitCellKey(cell);
    words[2 * row] = low;
    words[2 * row + 1] = high;
  });
  const {capacity} = options;
  const out = {
    rows: track(createOutputBuffer(device, capacity)),
    cells: track(createOutputBuffer(device, 2 * capacity)),
    edges: track(createOutputBuffer(device, capacity)),
    endpoints: track(createOutputBuffer(device, 4 * capacity)),
    groups: track(createOutputBuffer(device, capacity)),
    count: track(createOutputBuffer(device, 1)),
    overflow: track(createOutputBuffer(device, 1)),
    total: track(createOutputBuffer(device, 1))
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-set-outline-graph'});
  const groupValues = options.groups
    ? Uint32Array.from(cells.map(cell => options.groups!.get(cell) ?? 0))
    : undefined;
  graph.add(
    new GPUCellSetOutline({
      family,
      cells: importGraphBuffer(
        graph,
        'cells',
        track(createInputBuffer(device, words)),
        'uint32x2',
        cells.length
      ),
      count:
        options.count === undefined
          ? undefined
          : importGraphBuffer(
              graph,
              'count-in',
              track(createInputBuffer(device, Uint32Array.of(options.count))),
              'uint32',
              1
            ),
      groups: groupValues
        ? importGraphBuffer(
            graph,
            'groups',
            track(createInputBuffer(device, groupValues)),
            'uint32',
            cells.length
          )
        : undefined,
      output: {
        rows: importGraphBuffer(graph, 'out-rows', out.rows, 'uint32', capacity),
        cells: importGraphBuffer(graph, 'out-cells', out.cells, 'uint32x2', capacity),
        edgeIndices: importGraphBuffer(graph, 'out-edges', out.edges, 'uint32', capacity),
        endpoints: importGraphBuffer(graph, 'out-endpoints', out.endpoints, 'float32x4', capacity),
        groups: groupValues
          ? importGraphBuffer(graph, 'out-groups', out.groups, 'uint32', capacity)
          : undefined,
        count: importGraphBuffer(graph, 'out-count', out.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'out-overflow', out.overflow, 'uint32', 1),
        totalCount: importGraphBuffer(graph, 'out-total', out.total, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const [overflow] = await readUint32(out.overflow, 1);
  const [total] = await readUint32(out.total, 1);
  const rows = (await readUint32(out.rows, capacity)).slice(0, count);
  const cellWords = await readUint32(out.cells, 2 * capacity);
  const edgeIndices = (await readUint32(out.edges, capacity)).slice(0, count);
  const endpoints = await readFloat32(out.endpoints, 4 * capacity);
  const groups = (await readUint32(out.groups, capacity)).slice(0, count);
  const segments: OutlineSegment[] = [];
  for (let row = 0; row < count; row++) {
    segments.push({
      cell: joinCellKey(cellWords[2 * row], cellWords[2 * row + 1]),
      a: [endpoints[4 * row], endpoints[4 * row + 1]],
      b: [endpoints[4 * row + 2], endpoints[4 * row + 3]]
    });
  }
  compiled.destroy?.();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {count, overflow, total, rows, edgeIndices, groups, segments};
}

/** Disk of H3 cells with scattered holes plus a second blob. */
function createH3Scene(resolution: number): bigint[] {
  const first = gridDisk(latLngToCell(47.2, 8.5, resolution), 6).filter(
    (_, index) => index % 5 !== 3
  );
  const second = gridDisk(latLngToCell(39.7, -100.2, resolution), 3);
  return sortKeys([...first, ...second].map(h3ToBigInt));
}

function createGroups(cells: bigint[], groupCount: number): Map<bigint, number> {
  return new Map(cells.map(cell => [cell, Number((cell >> 3n) % BigInt(groupCount))]));
}

function expectSameSegments(actual: OutlineSegment[], expected: OutlineSegment[], label: string) {
  const {unmatchedActual, unmatchedExpected} = matchSegments(actual, expected);
  expect(unmatchedExpected, `${label} expected but missing`).toEqual([]);
  expect(unmatchedActual, `${label} unexpected extra`).toEqual([]);
}

it('GPUCellSetOutline h3 matches the h3-js directed-edge oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const resolution of [4, 6]) {
    const cells = createH3Scene(resolution);
    const expected = outlineH3OnCPU(cells);
    expect(expected.length).toBeGreaterThan(20);
    const run = await runOutline(device, 'h3', cells, {capacity: expected.length + 16});
    const label = `res ${resolution}`;
    expect(run.overflow, label).toBe(0);
    expect(run.total, label).toBe(expected.length);
    expect(run.count, label).toBe(expected.length);
    expectSameSegments(run.segments, expected, label);
    // Deterministic order: by input row, then by edge index.
    for (let index = 1; index < run.count; index++) {
      const previous = run.rows[index - 1] * 16 + run.edgeIndices[index - 1];
      expect(run.rows[index] * 16 + run.edgeIndices[index], `${label} order`).toBeGreaterThan(
        previous
      );
    }
    // Every emitted edge belongs to the cell its row names.
    run.rows.forEach((row, index) => expect(cells[row]).toBe(run.segments[index].cell));
  }
});

it('GPUCellSetOutline h3 outline equals the cellsToMultiPolygon ring edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cells = createH3Scene(5);
  const expected = getMultiPolygonSegments(cells);
  const run = await runOutline(device, 'h3', cells, {capacity: 4096});
  expect(expected.length).toBeGreaterThan(20);
  expectSameSegments(
    run.segments.map(segment => ({...segment, cell: 0n})),
    expected,
    'cellsToMultiPolygon'
  );
});

it('GPUCellSetOutline h3 handles pentagons and icosahedron-edge cells', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pentagon = getPentagons(3)[0];
  const cells = sortKeys(gridDisk(pentagon, 2).map(h3ToBigInt));
  const expected = outlineH3OnCPU(cells);
  const run = await runOutline(device, 'h3', cells, {capacity: expected.length + 32});
  expect(run.count).toBeGreaterThan(10);
  // Compare by edge identity per cell: distortion vertices may split an edge differently, so
  // compare the union of covered endpoints rather than the segment lists.
  const endpointSet = (segments: OutlineSegment[]) =>
    new Set(
      segments.flatMap(segment =>
        [segment.a, segment.b].map(
          point =>
            `${segment.cell.toString(16)}|${Math.round(point[0] * 100)},${Math.round(point[1] * 100)}`
        )
      )
    );
  const actualPoints = endpointSet(run.segments);
  const expectedPoints = endpointSet(expected);
  let missing = 0;
  for (const point of expectedPoints) {
    if (!actualPoints.has(point)) {
      missing++;
    }
  }
  expect(missing, 'expected outline points missing from the GPU outline').toBeLessThanOrEqual(
    Math.ceil(expectedPoints.size * 0.05)
  );
  const cellsWithEdges = new Set(run.segments.map(segment => segment.cell));
  const expectedCells = new Set(expected.map(segment => segment.cell));
  expect(cellsWithEdges).toEqual(expectedCells);
});

it('GPUCellSetOutline emits group borders when groups are given', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const cells = createH3Scene(5);
  const groups = createGroups(cells, 3);
  const plain = outlineH3OnCPU(cells);
  const expected = outlineH3OnCPU(cells, groups);
  expect(expected.length).toBeGreaterThan(plain.length);
  const run = await runOutline(device, 'h3', cells, {groups, capacity: expected.length + 16});
  expectSameSegments(run.segments, expected, 'h3 groups');
  run.segments.forEach((segment, index) => {
    expect(run.groups[index]).toBe(groups.get(segment.cell));
  });
});

it('GPUCellSetOutline kernels fit the default eight storage buffers with groups', async () => {
  // Regression: group-aware outlines once bound nine buffers in one kernel and failed on devices
  // without raised limits. `core` requests default limits, so createWGSLKernelNode would throw.
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
  const cells = createH3Scene(3);
  const groups = createGroups(cells, 2);
  const expected = outlineH3OnCPU(cells, groups);
  const run = await runOutline(device, 'h3', cells, {groups, capacity: expected.length + 16});
  expect(run.segments.length).toBeGreaterThan(0);
  expectSameSegments(run.segments, expected, 'h3 groups core');
});

it('GPUCellSetOutline quadbin matches the tile oracle including wrap and poles', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const z = 4;
  const n = 2 ** z;
  const tiles: [number, number][] = [];
  for (let y = 3; y <= 9; y++) {
    for (let x = 0; x < n; x++) {
      if (x >= 3 && x <= 12) {
        continue;
      }
      if ((x * 7 + y * 3) % 5 === 0) {
        continue;
      }
      tiles.push([x, y]);
    }
  }
  // Top row (north map edge) and a tile next to the antimeridian seam from the other side.
  tiles.push([8, 0], [9, 0], [15, 0], [0, 0]);
  const cells = sortKeys(tiles.map(([x, y]) => quadbinTileToCell(x, y, z)));
  for (const withGroups of [false, true]) {
    const groups = withGroups ? createGroups(cells, 2) : undefined;
    const expected = outlineQuadbinOnCPU(cells, groups);
    const run = await runOutline(device, 'quadbin', cells, {groups, capacity: expected.length + 8});
    expect(run.overflow).toBe(0);
    expect(run.count).toBeGreaterThan(20);
    expect(run.count).toBe(expected.length);
    expectSameSegments(run.segments, expected, `quadbin groups ${withGroups}`);
  }
});

it('GPUCellSetOutline reports overflow, clamps count and honors the row count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cells = createH3Scene(5);
  const expected = outlineH3OnCPU(cells);
  const capacity = 24;
  expect(expected.length).toBeGreaterThan(capacity);
  const run = await runOutline(device, 'h3', cells, {capacity});
  expect(run.overflow).toBe(1);
  expect(run.total).toBe(expected.length);
  expect(run.count).toBe(capacity);
  // The kept prefix is the first segments in (row, edge) order and a subset of the full outline.
  const expectedKeys = new Set(expected.map(item => getSegmentKey(item.cell, item.a, item.b)));
  for (const item of run.segments) {
    expect(expectedKeys.has(getSegmentKey(item.cell, item.a, item.b))).toBe(true);
  }
  const full = await runOutline(device, 'h3', cells, {capacity: expected.length});
  expect(full.rows.slice(0, capacity)).toEqual(run.rows);
  expect(full.edgeIndices.slice(0, capacity)).toEqual(run.edgeIndices);

  // A smaller `count` leaves the trailing rows out of the set and out of the output.
  const half = Math.floor(cells.length / 2);
  const limited = await runOutline(device, 'h3', cells, {
    capacity: expected.length,
    count: half
  });
  expectSameSegments(limited.segments, outlineH3OnCPU(cells.slice(0, half)), 'count rows');
});
