// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellToChildren, cellToParent, compactCells, getPentagons, polygonToCells} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUCellCompaction,
  type GPUCellCompactionProps
} from '../../../src/geospatial/cell-topology/gpu-cell-compaction';
import {
  bigIntToH3,
  h3ToBigInt,
  joinCellKey,
  quadbinTileToCell,
  splitCellKey,
  type CellFamily
} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';
import {
  compactCellsOnCPU,
  compareBigInt,
  getCellDescendants,
  uncompactCellsOnCPU
} from './cell-compaction-oracle';

type RunResult = {
  cells: bigint[];
  count: number;
  overflow: number;
  total: number;
  dropped: number;
  /** True when every row past count is zero. */
  zeroTail: boolean;
};

type RunOptions = {
  family: CellFamily;
  operation: GPUCellCompactionProps['operation'];
  cells: readonly bigint[];
  capacity?: number;
  wordOrder?: 'little-endian' | 'high-low';
  mask?: number[];
  count?: number;
};

function createWords(cells: readonly bigint[], wordOrder?: string): Uint32Array {
  const words = new Uint32Array(2 * cells.length);
  for (const [row, cell] of cells.entries()) {
    const [low, high] = splitCellKey(cell);
    words[2 * row] = wordOrder === 'high-low' ? high : low;
    words[2 * row + 1] = wordOrder === 'high-low' ? low : high;
  }
  return words;
}

/** Builds, compiles, and runs one compaction graph; returns the decoded output. */
async function runCompaction(device: Device, options: RunOptions): Promise<RunResult> {
  const rows = options.cells.length;
  const capacity = options.capacity ?? Math.max(rows, 1) * 4;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-compaction-graph'});
  const importView = <Format extends 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, track(buffer), format, length);
  const cellsOut = createOutputBuffer(device, 2 * capacity);
  const countOut = createOutputBuffer(device, 1);
  const overflowOut = createOutputBuffer(device, 1);
  const totalOut = createOutputBuffer(device, 1);
  const droppedOut = createOutputBuffer(device, 1);
  graph.add(
    new GPUCellCompaction({
      family: options.family,
      operation: options.operation,
      wordOrder: options.wordOrder,
      cells: importView(
        'cells',
        createInputBuffer(device, createWords(options.cells, options.wordOrder)),
        'uint32x2',
        rows
      ),
      mask: options.mask
        ? importView(
            'mask',
            createInputBuffer(device, Uint32Array.from(options.mask)),
            'uint32',
            rows
          )
        : undefined,
      count:
        options.count === undefined
          ? undefined
          : importView(
              'count',
              createInputBuffer(device, Uint32Array.of(options.count)),
              'uint32',
              1
            ),
      output: {
        cells: importView('cells-out', cellsOut, 'uint32x2', capacity),
        count: importView('count-out', countOut, 'uint32', 1),
        overflow: importView('overflow-out', overflowOut, 'uint32', 1),
        totalCount: importView('total-out', totalOut, 'uint32', 1),
        droppedCount: importView('dropped-out', droppedOut, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(countOut, 1);
  const [overflow] = await readUint32(overflowOut, 1);
  const [total] = await readUint32(totalOut, 1);
  const [dropped] = await readUint32(droppedOut, 1);
  const words = await readUint32(cellsOut, 2 * capacity);
  const cells: bigint[] = [];
  for (let row = 0; row < count; row++) {
    cells.push(joinCellKey(words[2 * row], words[2 * row + 1]));
  }
  const zeroTail = words.slice(2 * count).every(word => word === 0);
  for (const buffer of [...buffers, cellsOut, countOut, overflowOut, totalOut, droppedOut]) {
    buffer.destroy();
  }
  return {cells, count, overflow, total, dropped, zeroTail};
}

function shuffle<T>(values: readonly T[], seed: number): T[] {
  const random = createRandom(seed);
  const result = [...values];
  for (let index = result.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [result[index], result[other]] = [result[other], result[index]];
  }
  return result;
}

const POLYGON = [
  [37.5, -122.8],
  [38.1, -122.8],
  [38.1, -122.0],
  [37.5, -122.0]
];

function getQuadbinBlock(x: number, y: number, resolution: number, side: number): bigint[] {
  const cells: bigint[] = [];
  for (let dx = 0; dx < side; dx++) {
    for (let dy = 0; dy < side; dy++) {
      cells.push(quadbinTileToCell(x + dx, y + dy, resolution));
    }
  }
  return cells;
}

async function expectCompaction(
  device: Device,
  options: RunOptions & {resolution: number; minimumResolution?: number},
  label: string
): Promise<RunResult> {
  const {family, resolution, minimumResolution} = options;
  const operation = {
    type: 'compact' as const,
    resolution,
    minimumResolution,
    ...(options.operation.type === 'compact' ? {sorted: options.operation.sorted} : {})
  };
  const result = await runCompaction(device, {...options, operation});
  const expected = compactCellsOnCPU(family, options.cells, resolution, minimumResolution ?? 0);
  expect(result.cells.map(String), `${label} cells`).toEqual(expected.map(String));
  expect(result.total, `${label} total`).toBe(expected.length);
  expect(result.overflow, `${label} overflow`).toBe(0);
  expect(result.zeroTail, `${label} zero tail`).toBe(true);
  return result;
}

const COMPACT_STUB = {type: 'compact', resolution: 0} as const;

it('GPUCellCompaction compacts H3 polygon fills like h3-js compactCells', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const resolution of [5, 6, 7, 8]) {
    const strings = polygonToCells(POLYGON, resolution);
    const cells = strings.map(h3ToBigInt);
    const h3Compacted = compactCells(strings).map(h3ToBigInt).sort(compareBigInt);
    const result = await expectCompaction(
      device,
      {family: 'h3', operation: COMPACT_STUB, resolution, cells: shuffle(cells, resolution)},
      `h3 r${resolution}`
    );
    expect(result.cells.map(String)).toEqual(h3Compacted.map(String));
    expect(result.dropped).toBe(0);
  }
  // Duplicates collapse; rows of another resolution, invalid keys and masked rows are dropped.
  const cells = polygonToCells(POLYGON, 7).map(h3ToBigInt);
  const coarse = polygonToCells(POLYGON, 5).map(h3ToBigInt).slice(0, 3);
  const noisy = [...cells, ...cells.slice(0, 50), ...coarse, 0n, 0xffffffffffffffffn];
  const mask = noisy.map((_, row) => (row === 7 ? 0 : 1));
  const result = await runCompaction(device, {
    family: 'h3',
    operation: {type: 'compact', resolution: 7},
    cells: noisy,
    mask
  });
  const expected = compactCellsOnCPU(
    'h3',
    noisy.filter((_, row) => row !== 7),
    7
  );
  expect(result.cells.map(String)).toEqual(expected.map(String));
  expect(result.dropped).toBe(coarse.length + 2 + 1);
  // minimumResolution stops the merging.
  await expectCompaction(
    device,
    {
      family: 'h3',
      operation: COMPACT_STUB,
      resolution: 8,
      minimumResolution: 7,
      cells: polygonToCells(POLYGON, 8).map(h3ToBigInt)
    },
    'h3 min 7'
  );
  device.destroy();
});

it('GPUCellCompaction handles H3 pentagon subtrees', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pentagons = getPentagons(2);
  // Full child sets of every pentagon at depth 1 and 2 (6 and 41 descendants), then partial sets.
  for (const [depth, label] of [
    [1, 'children'],
    [2, 'grandchildren']
  ] as const) {
    const full = pentagons.flatMap(pentagon => cellToChildren(pentagon, 2 + depth)).map(h3ToBigInt);
    expect(full.length).toBe(pentagons.length * (depth === 1 ? 6 : 41));
    await expectCompaction(
      device,
      {family: 'h3', operation: COMPACT_STUB, resolution: 2 + depth, cells: shuffle(full, depth)},
      `pentagon ${label}`
    );
    const partial = full.filter((_, row) => row % 7 !== 3);
    await expectCompaction(
      device,
      {family: 'h3', operation: COMPACT_STUB, resolution: 2 + depth, cells: partial},
      `pentagon ${label} partial`
    );
  }
  // A pentagon plus a hexagon neighbor's full child set at the same time.
  const hexagonChildren = cellToChildren(cellToParent(polygonToCells(POLYGON, 7)[0], 3), 4);
  const pentagonChildren = cellToChildren(pentagons[0], 4);
  await expectCompaction(
    device,
    {
      family: 'h3',
      operation: COMPACT_STUB,
      resolution: 4,
      cells: [...hexagonChildren, ...pentagonChildren].map(h3ToBigInt)
    },
    'pentagon plus hexagon'
  );
  device.destroy();
});

it('GPUCellCompaction compacts Quadbin blocks, partial blocks and random sets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A full 8 x 8 block (4^3 cells at resolution 8) compacts to a res 5 parent.
  const block = getQuadbinBlock(16, 32, 8, 8);
  const full = await expectCompaction(
    device,
    {family: 'quadbin', operation: COMPACT_STUB, resolution: 8, cells: shuffle(block, 1)},
    'quadbin block'
  );
  expect(full.cells).toEqual([quadbinTileToCell(2, 4, 5)]);
  await expectCompaction(
    device,
    {family: 'quadbin', operation: COMPACT_STUB, resolution: 8, minimumResolution: 7, cells: block},
    'quadbin min 7'
  );
  await expectCompaction(
    device,
    {family: 'quadbin', operation: COMPACT_STUB, resolution: 8, cells: block.slice(5)},
    'quadbin partial'
  );
  // Several blocks and loose cells, sorted input with the sorted flag.
  const mixed = [
    ...getQuadbinBlock(0, 0, 10, 4),
    ...getQuadbinBlock(40, 8, 10, 2),
    ...getQuadbinBlock(100, 100, 10, 3)
  ];
  const sortedMixed = [...mixed].sort(compareBigInt);
  await expectCompaction(
    device,
    {family: 'quadbin', operation: COMPACT_STUB, resolution: 10, cells: shuffle(mixed, 4)},
    'quadbin mixed'
  );
  await expectCompaction(
    device,
    {
      family: 'quadbin',
      operation: {type: 'compact', resolution: 10, sorted: true},
      resolution: 10,
      cells: sortedMixed
    },
    'quadbin sorted flag'
  );
  // Resolution 0 and a single cell.
  await expectCompaction(
    device,
    {
      family: 'quadbin',
      operation: COMPACT_STUB,
      resolution: 0,
      cells: [quadbinTileToCell(0, 0, 0)]
    },
    'quadbin res 0'
  );
  // Deep keys that need both key words.
  const deep = getQuadbinBlock(1000, 2000, 24, 4);
  await expectCompaction(
    device,
    {family: 'quadbin', operation: COMPACT_STUB, resolution: 24, cells: deep},
    'quadbin res 24'
  );
  // Random subset of one parent's descendants.
  const random = createRandom(9);
  const subset = getQuadbinBlock(64, 64, 9, 16).filter(() => random() < 0.9);
  await expectCompaction(
    device,
    {family: 'quadbin', operation: COMPACT_STUB, resolution: 9, cells: subset},
    'quadbin random subset'
  );
  device.destroy();
});

it('GPUCellCompaction supports high-low words, masks, row counts and capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cells = [...getQuadbinBlock(0, 0, 9, 8), ...getQuadbinBlock(100, 4, 9, 3)];
  const expected = compactCellsOnCPU('quadbin', cells, 9);
  const highLow = await runCompaction(device, {
    family: 'quadbin',
    operation: {type: 'compact', resolution: 9},
    cells,
    wordOrder: 'high-low'
  });
  expect(highLow.cells).toEqual(expected);
  // count limits the rows read; trailing rows are neither used nor counted as dropped.
  const limited = await runCompaction(device, {
    family: 'quadbin',
    operation: {type: 'compact', resolution: 9},
    cells,
    count: 64
  });
  expect(limited.cells).toEqual(compactCellsOnCPU('quadbin', cells.slice(0, 64), 9));
  expect(limited.dropped).toBe(0);
  // Capacity overflow: count is clamped, the ascending prefix is kept, the flag is set.
  const small = await runCompaction(device, {
    family: 'quadbin',
    operation: {type: 'compact', resolution: 9},
    cells,
    capacity: 3
  });
  expect(expected.length).toBeGreaterThan(3);
  expect(small.count).toBe(3);
  expect(small.total).toBe(expected.length);
  expect(small.overflow).toBe(1);
  expect(small.cells).toEqual(expected.slice(0, 3));
  device.destroy();
});

it('GPUCellCompaction uncompact round trips', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [family, resolution, cells] of [
    ['h3', 7, polygonToCells(POLYGON, 7).map(h3ToBigInt)],
    ['quadbin', 10, [...getQuadbinBlock(0, 0, 10, 8), ...getQuadbinBlock(32, 8, 10, 3)]]
  ] as const) {
    const compacted = await expectCompaction(
      device,
      {family, operation: COMPACT_STUB, resolution, cells: shuffle(cells, 5)},
      `${family} round trip compact`
    );
    const expanded = await runCompaction(device, {
      family,
      operation: {type: 'uncompact', resolution},
      cells: compacted.cells,
      capacity: cells.length + 5
    });
    expect(expanded.overflow).toBe(0);
    expect(expanded.total).toBe(cells.length);
    expect([...expanded.cells].sort(compareBigInt).map(String)).toEqual(
      [...cells].sort(compareBigInt).map(String)
    );
    expect(expanded.cells.map(String)).toEqual(
      uncompactCellsOnCPU(family, compacted.cells, resolution).map(String)
    );
  }
  device.destroy();
});

it('GPUCellCompaction uncompacts pentagons, mixed resolutions, caps and overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pentagons = getPentagons(1).map(h3ToBigInt);
  for (const resolution of [2, 3, 4]) {
    const result = await runCompaction(device, {
      family: 'h3',
      operation: {type: 'uncompact', resolution},
      cells: pentagons,
      capacity: 12 * 300
    });
    const expected = uncompactCellsOnCPU('h3', pentagons, resolution);
    expect(result.cells.map(String), `pentagon r${resolution}`).toEqual(expected.map(String));
    // Ascending within each pentagon's descendants.
    for (let start = 0; start < expected.length; start += expected.length / 12) {
      const slice = result.cells.slice(start, start + expected.length / 12);
      expect(slice.every((cell, row) => row === 0 || slice[row - 1] < cell)).toBe(true);
    }
  }
  // Mixed resolutions, a coarser-than-target cell, a finer cell (dropped) and an invalid key.
  const hexagon = h3ToBigInt(cellToParent(polygonToCells(POLYGON, 7)[0], 3));
  const fine = h3ToBigInt(polygonToCells(POLYGON, 7)[0]);
  const cells = [hexagon, h3ToBigInt(polygonToCells(POLYGON, 5)[0]), fine, 0n];
  const mixed = await runCompaction(device, {
    family: 'h3',
    operation: {type: 'uncompact', resolution: 5},
    cells,
    capacity: 1000
  });
  expect(mixed.cells.map(String)).toEqual(uncompactCellsOnCPU('h3', cells, 5).map(String));
  expect(mixed.dropped).toBe(2);
  // maximumDepth drops deeper cells.
  const capped = await runCompaction(device, {
    family: 'quadbin',
    operation: {type: 'uncompact', resolution: 8, maximumDepth: 2},
    cells: [quadbinTileToCell(0, 0, 3), quadbinTileToCell(1, 1, 6)],
    capacity: 100
  });
  expect(capped.cells.map(String)).toEqual(
    getCellDescendants('quadbin', quadbinTileToCell(1, 1, 6), 8).map(String)
  );
  expect(capped.dropped).toBe(1);
  // Overflow keeps the ascending prefix.
  const overflow = await runCompaction(device, {
    family: 'quadbin',
    operation: {type: 'uncompact', resolution: 8},
    cells: [quadbinTileToCell(0, 0, 5), quadbinTileToCell(1, 1, 6)],
    capacity: 20
  });
  const all = uncompactCellsOnCPU(
    'quadbin',
    [quadbinTileToCell(0, 0, 5), quadbinTileToCell(1, 1, 6)],
    8
  );
  expect(overflow.total).toBe(all.length);
  expect(overflow.count).toBe(20);
  expect(overflow.overflow).toBe(1);
  expect(overflow.cells.map(String)).toEqual(all.slice(0, 20).map(String));
  expect(bigIntToH3(pentagons[0]).length).toBeGreaterThan(0);
  device.destroy();
});
