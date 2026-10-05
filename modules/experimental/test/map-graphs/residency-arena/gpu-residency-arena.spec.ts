// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUResidencyArena,
  type GPUResidencyArenaTileData
} from '../../../src/map-graphs/residency-arena/gpu-residency-arena';
import {ResidencyArenaFullError} from '../../../src/map-graphs/residency-arena/residency-arena-allocator';
import {GPU_RESIDENCY_ARENA_DEAD_SLOT} from '../../../src/map-graphs/residency-arena/residency-arena-types';

const PAGE_ROW_COUNT = 64;
const ROW_CAPACITY = PAGE_ROW_COUNT * 32;

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function readBack<ArrayType extends Float32Array | Uint32Array>(
  buffer: {readAsync(): Promise<Uint8Array>},
  Constructor: new (buffer: ArrayBuffer, byteOffset: number, length: number) => ArrayType
): Promise<ArrayType> {
  const bytes = await buffer.readAsync();
  const copy = bytes.slice();
  return new Constructor(copy.buffer as ArrayBuffer, 0, copy.byteLength / 4);
}

it('GPUResidencyArena churn matches the CPU image and creates no buffers', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const arena = new GPUResidencyArena(device, {
    rowCapacity: ROW_CAPACITY,
    pageRowCount: PAGE_ROW_COUNT,
    maxTileCount: 12,
    columns: [
      {name: 'positions', format: 'float32x2', deadValue: NaN},
      {name: 'time', format: 'float32'},
      {name: 'ids', format: 'uint32', deadValue: 0}
    ]
  });

  const expectedPositions = new Float32Array(ROW_CAPACITY * 2).fill(NaN);
  const expectedTime = new Float32Array(ROW_CAPACITY);
  const expectedIds = new Uint32Array(ROW_CAPACITY);
  const random = createRandom(777);
  let serial = 1;

  const makeData = (rowCount: number): GPUResidencyArenaTileData => {
    const positions = new Float32Array(rowCount * 2);
    const time = new Float32Array(rowCount);
    const ids = new Uint32Array(rowCount);
    for (let i = 0; i < rowCount; i++) {
      positions[i * 2] = serial + i;
      positions[i * 2 + 1] = -(serial + i);
      time[i] = serial * 0.5 + i;
      ids[i] = serial * 1000 + i;
    }
    serial++;
    return {rowCount, columns: {positions, time, ids}};
  };
  const applyToImage = (key: string, data: GPUResidencyArenaTileData): void => {
    const tile = arena.allocator.getTile(key)!;
    let tileRow = 0;
    for (const range of tile.ranges) {
      for (let i = 0; i < range.rowCount; i++, tileRow++) {
        const row = range.firstRow + i;
        expectedPositions.set(
          data.columns.positions.subarray(tileRow * 2, tileRow * 2 + 2),
          row * 2
        );
        expectedTime[row] = data.columns.time[tileRow];
        expectedIds[row] = data.columns.ids[tileRow];
      }
    }
  };
  const killRows = (ranges: readonly {firstRow: number; rowCount: number}[]): void => {
    for (const range of ranges) {
      for (let row = range.firstRow; row < range.firstRow + range.rowCount; row++) {
        expectedPositions[row * 2] = NaN;
        expectedPositions[row * 2 + 1] = NaN;
        expectedIds[row] = 0;
      }
    }
  };

  const verify = async (): Promise<void> => {
    const allocator = arena.allocator;
    expect(await readBack(arena.liveMaskBuffer, Uint32Array)).toEqual(allocator.getLiveMask());
    expect(await readBack(arena.rowTileSlotsBuffer, Uint32Array)).toEqual(
      allocator.getRowTileSlots()
    );
    const positions = await readBack(arena.getColumnBuffer('positions'), Float32Array);
    const time = await readBack(arena.getColumnBuffer('time'), Float32Array);
    const ids = await readBack(arena.getColumnBuffer('ids'), Uint32Array);
    const mask = allocator.getLiveMask();
    for (let row = 0; row < ROW_CAPACITY; row++) {
      if (mask[row]) {
        expect(positions[row * 2]).toBe(expectedPositions[row * 2]);
        expect(positions[row * 2 + 1]).toBe(expectedPositions[row * 2 + 1]);
        expect(time[row]).toBe(expectedTime[row]);
        expect(ids[row]).toBe(expectedIds[row]);
      } else {
        expect(Number.isNaN(positions[row * 2])).toBe(true);
        expect(Number.isNaN(positions[row * 2 + 1])).toBe(true);
        expect(ids[row]).toBe(0);
      }
    }
  };

  await verify(); // initial dead state

  const originalCreateBuffer = device.createBuffer.bind(device);
  let createBufferCalls = 0;
  device.createBuffer = ((...args: Parameters<typeof originalCreateBuffer>) => {
    createBufferCalls++;
    return originalCreateBuffer(...args);
  }) as typeof device.createBuffer;

  let keyCounter = 0;
  try {
    for (let op = 0; op < 200; op++) {
      const keys = arena.allocator.getTiles().map(tile => tile.key);
      const choice = random();
      const rowCount = random() < 0.08 ? 0 : 1 + Math.floor(random() * PAGE_ROW_COUNT * 4);
      try {
        if (choice < 0.45 || keys.length === 0) {
          const key = `t${keyCounter++}`;
          const data = makeData(rowCount);
          arena.insertTile(key, data);
          applyToImage(key, data);
        } else if (choice < 0.7) {
          const tile = arena.evictTile(keys[Math.floor(random() * keys.length)]);
          killRows(tile.ranges);
        } else {
          const key = keys[Math.floor(random() * keys.length)];
          const previousRanges = arena.allocator.getTile(key)!.ranges;
          const data = makeData(rowCount);
          arena.replaceTile(key, data);
          killRows(previousRanges);
          applyToImage(key, data);
        }
      } catch (error) {
        expect(error).toBeInstanceOf(ResidencyArenaFullError);
      }
      if (op % 40 === 39) {
        // readAsync may stage buffers, so verify outside the counter
        const callsBefore = createBufferCalls;
        device.createBuffer = originalCreateBuffer;
        await verify();
        createBufferCalls = callsBefore;
        device.createBuffer = ((...args: Parameters<typeof originalCreateBuffer>) => {
          createBufferCalls++;
          return originalCreateBuffer(...args);
        }) as typeof device.createBuffer;
      }
    }
  } finally {
    device.createBuffer = originalCreateBuffer;
  }
  expect(createBufferCalls).toBe(0);
  expect(arena.residentTileCount).toBeGreaterThan(0);
  await verify();
  arena.destroy();
});

it('GPUResidencyArena validates tile data before mutating and shrinks to dead tails', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const arena = new GPUResidencyArena(device, {
    rowCapacity: PAGE_ROW_COUNT * 2,
    pageRowCount: PAGE_ROW_COUNT,
    columns: [{name: 'weight', format: 'float32', deadValue: 0}]
  });
  const bad = {rowCount: 3, columns: {weight: new Float32Array(2)}};
  expect(() => arena.insertTile('x', bad)).toThrow();
  expect(() => arena.insertTile('x', {rowCount: 1, columns: {}})).toThrow();
  expect(() =>
    arena.insertTile('x', {
      rowCount: 1,
      columns: {weight: new Uint32Array(1)}
    })
  ).toThrow();
  expect(arena.version).toBe(0);

  arena.insertTile('a', {
    rowCount: 100,
    columns: {weight: new Float32Array(100).fill(7)}
  });
  arena.replaceTile('a', {
    rowCount: 10,
    columns: {weight: new Float32Array(10).fill(3)}
  });
  const weights = await readBack(arena.getColumnBuffer('weight'), Float32Array);
  const mask = await readBack(arena.liveMaskBuffer, Uint32Array);
  const slots = await readBack(arena.rowTileSlotsBuffer, Uint32Array);
  for (let row = 0; row < arena.rowCapacity; row++) {
    expect(weights[row]).toBe(row < 10 ? 3 : 0);
    expect(mask[row]).toBe(row < 10 ? 1 : 0);
    expect(slots[row]).toBe(row < 10 ? 0 : GPU_RESIDENCY_ARENA_DEAD_SLOT);
  }
  arena.destroy();
});

it('GPUResidencyArena.importToGraph returns full-capacity views with stable IDs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const arena = new GPUResidencyArena(device, {
    id: 'tiles',
    rowCapacity: ROW_CAPACITY,
    pageRowCount: PAGE_ROW_COUNT,
    columns: [
      {name: 'positions', format: 'float32x2'},
      {name: 'ids', format: 'uint32'}
    ]
  });
  const graph = new GPUCommandGraph(device);
  const views = arena.importToGraph(graph);
  expect(views.rowCapacity).toBe(ROW_CAPACITY);
  expect(Object.keys(views.columns)).toEqual(['positions', 'ids']);
  expect(views.columns.positions.length).toBe(ROW_CAPACITY);
  expect(views.columns.ids.length).toBe(ROW_CAPACITY);
  expect(views.liveMask.length).toBe(ROW_CAPACITY);
  expect(views.rowTileSlots.length).toBe(ROW_CAPACITY);
  expect(views.columns.positions.buffer.id).toBe('tiles-positions');
  expect(views.liveMask.buffer.id).toBe('tiles-live-mask');
  expect(views.rowTileSlots.buffer.id).toBe('tiles-row-tile-slots');
  arena.destroy();
});
