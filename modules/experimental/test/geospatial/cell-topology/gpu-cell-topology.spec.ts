// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {getPentagons, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getCellTopologyStride,
  GPUCellTopology,
  type GPUCellTopologyOperation
} from '../../../src/geospatial/cell-topology/gpu-cell-topology';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  h3ToBigInt,
  joinCellKey,
  quadbinTileToCell,
  splitCellKey
} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {getCellTopologyRowOnCPU} from './cell-topology-oracle';

type Family = 'quadbin' | 'h3';

/** One compiled topology graph with rewritable cells and mask. */
function createTopology(
  device: Device,
  family: Family,
  operation: GPUCellTopologyOperation,
  rows: number,
  withMask = false
) {
  const stride = getCellTopologyStride(family, operation);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-topology-graph'});
  const cellsBuffer = track(createInputBuffer(device, new Uint32Array(2 * rows)));
  const maskBuffer = track(createInputBuffer(device, new Uint32Array(rows).fill(1)));
  const outCells = track(createOutputBuffer(device, 2 * rows * stride));
  const outDistances = track(createOutputBuffer(device, rows * stride));
  const outCounts = track(createOutputBuffer(device, rows));
  graph.add(
    new GPUCellTopology({
      family,
      operation,
      cells: importGraphBuffer(graph, 'cells', cellsBuffer, 'uint32x2', rows),
      mask: withMask ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
      output: {
        cells: importGraphBuffer(graph, 'out-cells', outCells, 'uint32x2', rows * stride),
        distances:
          operation.type === 'disk'
            ? importGraphBuffer(graph, 'out-distances', outDistances, 'uint32', rows * stride)
            : undefined,
        counts: importGraphBuffer(graph, 'out-counts', outCounts, 'uint32', rows)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    compiled,
    stride,
    get rebuildCount() {
      return compileCount - 1;
    },
    async run(cells: bigint[], mask?: Uint32Array) {
      const words = new Uint32Array(2 * rows);
      cells.forEach((cell, row) => words.set(splitCellKey(cell), 2 * row));
      cellsBuffer.write(words);
      if (mask) {
        maskBuffer.write(mask);
      }
      submitGraph(device, compiled, undefined);
      return {
        cells: await readUint32(outCells, 2 * rows * stride),
        distances: await readUint32(outDistances, rows * stride),
        counts: await readUint32(outCounts, rows)
      };
    },
    destroy() {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

type Actual = Awaited<ReturnType<ReturnType<typeof createTopology>['run']>>;

/** Compares every row bit exactly, including zero padding, distances and counts. */
function expectRows(
  family: Family,
  operation: GPUCellTopologyOperation,
  stride: number,
  cells: bigint[],
  actual: Actual,
  label: string,
  mask?: Uint32Array
) {
  for (const [row, cell] of cells.entries()) {
    const expected =
      mask && mask[row] === 0
        ? {cells: [], distances: []}
        : getCellTopologyRowOnCPU(family, operation, cell);
    const expectedCells: string[] = [];
    const actualCells: string[] = [];
    const expectedDistances: number[] = [];
    const actualDistances: number[] = [];
    for (let slot = 0; slot < stride; slot++) {
      const index = row * stride + slot;
      expectedCells.push((expected.cells[slot] ?? 0n).toString(16));
      actualCells.push(
        joinCellKey(actual.cells[2 * index], actual.cells[2 * index + 1]).toString(16)
      );
      expectedDistances.push(expected.distances[slot] ?? 0);
      actualDistances.push(actual.distances[index]);
    }
    expect(actualCells, `${label} row ${row} cell ${cell.toString(16)}`).toEqual(expectedCells);
    if (operation.type === 'disk') {
      expect(actualDistances, `${label} row ${row} distances`).toEqual(expectedDistances);
    }
    expect(actual.counts[row], `${label} row ${row} count`).toBe(expected.cells.length);
  }
}

const H3_INVALID = [0n, 1n, 0x8f283080dcb0d3bn, 0xffffffffffffffffn];

/** H3 cells: random at resolutions 3, 7, 12, 15, pentagons and cells within 3 of them at 2-5. */
function getH3Cells(): bigint[] {
  const random = createRandom(2024);
  const cells = new Set<bigint>();
  for (const resolution of [3, 7, 12, 15]) {
    for (let i = 0; i < 30; i++) {
      const latitude = (Math.asin(2 * random() - 1) * 180) / Math.PI;
      cells.add(h3ToBigInt(latLngToCell(latitude, 360 * random() - 180, resolution)));
    }
  }
  for (const resolution of [2, 3, 4, 5]) {
    for (const pentagon of getPentagons(resolution)) {
      cells.add(h3ToBigInt(pentagon));
      const disk = gridDisk(pentagon, 3);
      for (let i = 0; i < 8; i++) {
        cells.add(h3ToBigInt(disk[Math.floor(random() * disk.length)]));
      }
    }
  }
  return [...cells, ...H3_INVALID];
}

/** Quadbin tiles: corners and edges at several zooms, all tiles of z <= 2, random elsewhere. */
function getQuadbinCells(): bigint[] {
  const random = createRandom(77);
  const cells = new Set<bigint>();
  for (const z of [0, 1, 2]) {
    for (let x = 0; x < 2 ** z; x++) {
      for (let y = 0; y < 2 ** z; y++) {
        cells.add(quadbinTileToCell(x, y, z));
      }
    }
  }
  for (const z of [3, 5, 9, 16, 26]) {
    const max = 2 ** z - 1;
    for (const [x, y] of [
      [0, 0],
      [max, 0],
      [0, max],
      [max, max],
      [1, 1],
      [max - 1, 1],
      [0, Math.floor(max / 2)],
      [max, Math.floor(max / 2)]
    ]) {
      cells.add(quadbinTileToCell(x, y, z));
    }
    for (let i = 0; i < 12; i++) {
      cells.add(quadbinTileToCell(Math.floor(random() * 2 ** z), Math.floor(random() * 2 ** z), z));
    }
  }
  return [...cells, 0n, 1n, 0x4800000000000000n, 0xffffffffffffffffn];
}

async function checkOperation(
  device: Device,
  family: Family,
  operation: GPUCellTopologyOperation,
  cells: bigint[]
) {
  const topology = createTopology(device, family, operation, cells.length);
  const actual = await topology.run(cells);
  expectRows(
    family,
    operation,
    topology.stride,
    cells,
    actual,
    `${family} ${JSON.stringify(operation)}`
  );
  topology.destroy();
}

for (const family of ['quadbin', 'h3'] as const) {
  it(`GPUCellTopology ${family} disks and rings match the CPU reference`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const cells = family === 'h3' ? getH3Cells() : getQuadbinCells();
    for (const k of [0, 1, 2, 4]) {
      await checkOperation(device, family, {type: 'disk', k}, cells);
      await checkOperation(device, family, {type: 'ring', k}, cells);
    }
  }, 120_000);
}

it('GPUCellTopology parents match the CPU reference', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Resolution 12 and 15 cells, plus coarser ones that yield zero rows.
  const h3Cells = getH3Cells();
  for (const resolution of [0, 3, 7, 12]) {
    await checkOperation(device, 'h3', {type: 'parent', resolution}, h3Cells);
  }
  const quadbinCells = getQuadbinCells();
  for (const resolution of [0, 1, 4, 9]) {
    await checkOperation(device, 'quadbin', {type: 'parent', resolution}, quadbinCells);
  }
}, 120_000);

it('GPUCellTopology children match the CPU reference including pentagons', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const h3Cells = getH3Cells();
  for (const [inputResolution, resolution] of [
    [3, 3],
    [3, 4],
    [3, 5],
    [2, 6],
    [5, 7]
  ]) {
    await checkOperation(device, 'h3', {type: 'children', inputResolution, resolution}, h3Cells);
  }
  const quadbinCells = getQuadbinCells();
  for (const [inputResolution, resolution] of [
    [3, 3],
    [0, 6],
    [1, 5],
    [3, 7],
    [26, 26]
  ]) {
    await checkOperation(
      device,
      'quadbin',
      {type: 'children', inputResolution, resolution},
      quadbinCells
    );
  }
}, 180_000);

it('GPUCellTopology follows masks and rewritten cells without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const family of ['quadbin', 'h3'] as const) {
    const all = family === 'h3' ? getH3Cells() : getQuadbinCells();
    const rows = 40;
    const operation: GPUCellTopologyOperation = {type: 'disk', k: 2};
    const topology = createTopology(device, family, operation, rows, true);
    const compiled = topology.compiled;
    for (const offset of [0, 40, 80, 0]) {
      const cells = all.slice(offset, offset + rows);
      const mask = Uint32Array.from({length: rows}, (_, row) => (row % 3 === 0 ? 0 : 1));
      const actual = await topology.run(cells, mask);
      expectRows(
        family,
        operation,
        topology.stride,
        cells,
        actual,
        `${family} offset ${offset}`,
        mask
      );
      expect(topology.compiled).toBe(compiled);
    }
    expect(topology.rebuildCount).toBe(0);
    topology.destroy();
  }
}, 120_000);

it('GPUCellTopology accepts high-low word order', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cells = getQuadbinCells().slice(0, 30);
  const operation: GPUCellTopologyOperation = {type: 'ring', k: 1};
  const stride = getCellTopologyStride('quadbin', operation);
  const graph = new GPUCommandGraph(device, {id: 'word-order'});
  const words = new Uint32Array(2 * cells.length);
  cells.forEach((cell, row) => words.set(splitCellKey(cell).reverse(), 2 * row));
  const input = createInputBuffer(device, words);
  const output = createOutputBuffer(device, 2 * cells.length * stride);
  graph.add(
    new GPUCellTopology({
      family: 'quadbin',
      operation,
      wordOrder: 'high-low',
      cells: importGraphBuffer(graph, 'cells', input, 'uint32x2', cells.length),
      output: {
        cells: importGraphBuffer(graph, 'out', output, 'uint32x2', cells.length * stride)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const raw = await readUint32(output, 2 * cells.length * stride);
  const actual: Actual = {cells: raw, distances: [], counts: cells.map(() => 0)};
  for (const [row, cell] of cells.entries()) {
    const expected = getCellTopologyRowOnCPU('quadbin', operation, cell).cells;
    const got: bigint[] = [];
    for (let slot = 0; slot < stride; slot++) {
      const index = row * stride + slot;
      got.push(joinCellKey(actual.cells[2 * index], actual.cells[2 * index + 1]));
    }
    expect(got.slice(0, expected.length)).toEqual(expected);
    expect(got.slice(expected.length).every(cell => cell === 0n)).toBe(true);
  }
  compiled.destroy();
  input.destroy();
  output.destroy();
});
