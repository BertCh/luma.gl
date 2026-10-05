// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  ResidencyArenaAllocator,
  ResidencyArenaFullError
} from '../../../src/gpu-tables/residency-arena/residency-arena-allocator';
import {GPU_RESIDENCY_ARENA_DEAD_SLOT} from '../../../src/gpu-tables/residency-arena/residency-arena-types';

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

/** Cheap assertion; vitest expect is too slow for per-row loops. */
function check(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`invariant violated: ${message}`);
  }
}

/** Element-wise comparison without per-element matcher overhead. */
function checkSameValues(actual: Uint32Array, expected: Uint32Array, name: string): void {
  check(actual.length === expected.length, `${name} length`);
  for (let index = 0; index < expected.length; index++) {
    check(actual[index] === expected[index], `${name} differs at ${index}`);
  }
}

/** Brute-force invariant check against the public records only. */
function checkInvariants(allocator: ResidencyArenaAllocator): void {
  const pageOwner = new Map<number, string>();
  const expectedMask = new Uint32Array(allocator.rowCapacity);
  const expectedSlots = new Uint32Array(allocator.rowCapacity).fill(GPU_RESIDENCY_ARENA_DEAD_SLOT);
  const expectedLive: number[] = [];
  const slotsSeen = new Set<number>();
  let liveRows = 0;
  const tiles = allocator.getTiles();
  for (const tile of tiles) {
    check(!slotsSeen.has(tile.slot), 'duplicate slot');
    slotsSeen.add(tile.slot);
    expect(tile.slot).toBeLessThan(allocator.maxTileCount);
    expect(tile.pages.length).toBe(Math.ceil(tile.rowCount / allocator.pageRowCount));
    for (const page of tile.pages) {
      check(!pageOwner.has(page), 'page owned twice');
      pageOwner.set(page, tile.key);
    }
    expect([...tile.pages]).toEqual([...tile.pages].sort((a, b) => a - b));
    const rows: number[] = [];
    for (const range of tile.ranges) {
      expect(range.rowCount).toBeGreaterThan(0);
      for (let row = range.firstRow; row < range.firstRow + range.rowCount; row++) {
        rows.push(row);
        check(
          pageOwner.get(Math.floor(row / allocator.pageRowCount)) === tile.key,
          'row outside pages'
        );
        expectedMask[row] = 1;
        expectedSlots[row] = tile.slot;
      }
    }
    expect(rows.length).toBe(tile.rowCount);
    rows.forEach((row, rowInTile) => {
      const resolved = allocator.resolveRow(row);
      check(
        resolved?.key === tile.key &&
          resolved.slot === tile.slot &&
          resolved.rowInTile === rowInTile,
        'resolveRow round trip'
      );
      expectedLive.push(row);
    });
    liveRows += tile.rowCount;
  }
  expect(tiles.map(tile => tile.slot)).toEqual([...slotsSeen].sort((a, b) => a - b));
  expect(allocator.residentTileCount).toBe(tiles.length);
  expect(allocator.liveRowCount).toBe(liveRows);
  expect(allocator.freePageCount).toBe(allocator.pageCount - pageOwner.size);
  checkSameValues(allocator.getLiveMask(), expectedMask, 'live mask');
  checkSameValues(allocator.getRowTileSlots(), expectedSlots, 'row tile slots');
  checkSameValues(
    allocator.getLiveRows(),
    Uint32Array.from(expectedLive.sort((a, b) => a - b)),
    'live rows'
  );
  for (let row = 0; row < allocator.rowCapacity; row++) {
    if (!expectedMask[row]) {
      check(allocator.resolveRow(row) === null, 'dead row resolves');
    }
  }
}

describe('ResidencyArenaAllocator', () => {
  it('validates construction', () => {
    expect(() => new ResidencyArenaAllocator({rowCapacity: 100})).toThrow();
    expect(() => new ResidencyArenaAllocator({rowCapacity: 128, pageRowCount: 100})).toThrow();
    expect(() => new ResidencyArenaAllocator({rowCapacity: 0, pageRowCount: 64})).toThrow();
    const allocator = new ResidencyArenaAllocator({
      rowCapacity: 256,
      pageRowCount: 64
    });
    expect(allocator.pageCount).toBe(4);
    expect(allocator.maxTileCount).toBe(4);
  });

  it('inserts, evicts and replaces', () => {
    const allocator = new ResidencyArenaAllocator({
      rowCapacity: 64 * 8,
      pageRowCount: 64
    });
    const a = allocator.insert('a', 100);
    expect(a.pages).toEqual([0, 1]);
    expect(a.ranges).toEqual([{firstRow: 0, rowCount: 100}]);
    const b = allocator.insert('b', 64);
    expect(b.slot).toBe(1);
    expect(b.ranges).toEqual([{firstRow: 128, rowCount: 64}]);
    expect(allocator.version).toBe(2);
    expect(() => allocator.insert('a', 1)).toThrow(/already/);
    expect(() => allocator.insert('c', -1)).toThrow();
    expect(() => allocator.evict('zzz')).toThrow();
    expect(allocator.evict('a')).toBe(a);
    expect(allocator.has('a')).toBe(false);
    expect(allocator.insert('c', 1).slot).toBe(0);

    // shrink keeps the lowest pages and the slot
    const {previous, next} = allocator.replace('b', 10);
    expect(previous).toBe(b);
    expect(next.slot).toBe(1);
    expect(next.pages).toEqual([2]);
    // grow keeps owned pages and takes the lowest free ones
    const grown = allocator.replace('b', 64 * 3).next;
    expect(grown.pages).toEqual([1, 2, 3]);
    expect(grown.ranges).toEqual([{firstRow: 64, rowCount: 192}]);
    checkInvariants(allocator);
  });

  it('throws a full error and leaves state unchanged', () => {
    const allocator = new ResidencyArenaAllocator({
      rowCapacity: 64 * 4,
      pageRowCount: 64,
      maxTileCount: 3
    });
    allocator.insert('a', 64 * 3);
    allocator.insert('b', 1);
    const version = allocator.version;
    expect(allocator.canInsert(1)).toBe(false);
    expect(() => allocator.insert('c', 1)).toThrow(ResidencyArenaFullError);
    allocator.evict('b');
    expect(() => allocator.insert('d', 65)).toThrow(ResidencyArenaFullError);
    allocator.insert('e', 0);
    allocator.insert('f', 64);
    expect(() => allocator.insert('g', 0)).toThrow(ResidencyArenaFullError); // out of slots
    const before = JSON.stringify(allocator.getTiles());
    const versionBefore = allocator.version;
    expect(() => allocator.replace('f', 65)).toThrow(ResidencyArenaFullError);
    expect(JSON.stringify(allocator.getTiles())).toBe(before);
    expect(allocator.version).toBe(versionBefore);
    expect(versionBefore).toBeGreaterThan(version);
    checkInvariants(allocator);
  });

  it('allocates non-contiguously after fragmentation', () => {
    const allocator = new ResidencyArenaAllocator({
      rowCapacity: 64 * 6,
      pageRowCount: 64
    });
    for (const key of ['a', 'b', 'c', 'd', 'e', 'f']) {
      allocator.insert(key, 64);
    }
    allocator.evict('b');
    allocator.evict('d');
    const big = allocator.insert('big', 128);
    expect(big.pages).toEqual([1, 3]);
    expect(big.ranges).toEqual([
      {firstRow: 64, rowCount: 64},
      {firstRow: 192, rowCount: 64}
    ]);
    expect(allocator.resolveRow(192)).toEqual({
      key: 'big',
      slot: big.slot,
      rowInTile: 64
    });
    checkInvariants(allocator);
  });

  it('supports 0-row tiles', () => {
    const allocator = new ResidencyArenaAllocator({
      rowCapacity: 64,
      pageRowCount: 64,
      maxTileCount: 4
    });
    const empty = allocator.insert('empty', 0);
    expect(empty.pages).toEqual([]);
    expect(empty.ranges).toEqual([]);
    expect(allocator.freePageCount).toBe(1);
    allocator.insert('full', 64);
    allocator.replace('full', 0);
    expect(allocator.freePageCount).toBe(1);
    allocator.replace('empty', 5);
    expect(allocator.liveRowCount).toBe(5);
    checkInvariants(allocator);
  });

  it('survives a 2000-operation fuzz', () => {
    const random = createRandom(12345);
    const allocator = new ResidencyArenaAllocator({
      rowCapacity: 64 * 24,
      pageRowCount: 64,
      maxTileCount: 10
    });
    let keyCounter = 0;
    for (let op = 0; op < 2000; op++) {
      const keys = allocator.getTiles().map(tile => tile.key);
      const version = allocator.version;
      const choice = random();
      const rowCount = random() < 0.1 ? 0 : Math.floor(random() * 64 * 6);
      let succeeded = true;
      try {
        if (choice < 0.45 || keys.length === 0) {
          const canInsert = allocator.canInsert(rowCount);
          allocator.insert(`t${keyCounter++}`, rowCount);
          expect(canInsert).toBe(true);
        } else if (choice < 0.75) {
          allocator.evict(keys[Math.floor(random() * keys.length)]);
        } else {
          allocator.replace(keys[Math.floor(random() * keys.length)], rowCount);
        }
      } catch (error) {
        expect(error).toBeInstanceOf(ResidencyArenaFullError);
        succeeded = false;
      }
      expect(allocator.version).toBe(succeeded ? version + 1 : version);
      checkInvariants(allocator);
    }
  });
});
