// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUResidencyArena,
  ResidencyArenaFullError,
  type GPUResidencyArenaColumnSpec,
  type GPUResidencyArenaTileData
} from '../../../src/gpu-tables/residency-arena';

type ColumnArray = Float32Array | Uint32Array | Int32Array;

const COLUMN_COMPONENTS: Record<GPUResidencyArenaColumnSpec['format'], number> = {
  float32: 1,
  float32x2: 2,
  float32x3: 3,
  float32x4: 4,
  uint32: 1,
  uint32x2: 2,
  sint32: 1
};

/** Seeded mulberry32 generator. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Creates the tile data of a new tile of `rowCount` rows. */
export type MakeTileData = (rowCount: number) => GPUResidencyArenaTileData;

/**
 * A `GPUResidencyArena` plus an independent CPU copy of every column buffer.
 *
 * The CPU copy follows the arena's documented invariant: live rows hold tile data, rows that stop
 * being live get the column's `deadValue` when it declares one, and otherwise KEEP their stale
 * data. That makes it the oracle for what the GPU buffers contain, stale rows included.
 */
export class ArenaMirror {
  readonly arena: GPUResidencyArena;
  readonly columns: Record<string, ColumnArray> = {};
  private tileCounter = 0;

  constructor(
    device: Device,
    props: {
      id: string;
      rowCapacity: number;
      pageRowCount: number;
      maxTileCount?: number;
      columns: readonly GPUResidencyArenaColumnSpec[];
    }
  ) {
    this.arena = new GPUResidencyArena(device, props);
    for (const column of props.columns) {
      const length = props.rowCapacity * COLUMN_COMPONENTS[column.format];
      const values =
        column.format === 'sint32'
          ? new Int32Array(length)
          : column.format.startsWith('uint')
            ? new Uint32Array(length)
            : new Float32Array(length);
      if (column.deadValue !== undefined) {
        values.fill(column.deadValue);
      }
      this.columns[column.name] = values;
    }
  }

  get rowCapacity(): number {
    return this.arena.rowCapacity;
  }

  /** Keys of the resident tiles. */
  getKeys(): string[] {
    return this.arena.allocator.getTiles().map(tile => tile.key);
  }

  /** CPU live mask. */
  getLiveMask(): Uint32Array {
    return this.arena.allocator.getLiveMask();
  }

  /** Inserts a tile under a fresh key and returns the key, or undefined when it does not fit. */
  insert(data: GPUResidencyArenaTileData): string | undefined {
    if (!this.arena.allocator.canInsert(data.rowCount)) {
      return undefined;
    }
    const liveBefore = this.getLiveMask();
    const tile = this.arena.insertTile(`tile-${this.tileCounter++}`, data);
    this.writeTile(tile.ranges, data);
    this.killRowsThatStoppedBeingLive(liveBefore);
    return tile.key;
  }

  /** Evicts a resident tile. */
  evict(key: string): void {
    const liveBefore = this.getLiveMask();
    this.arena.evictTile(key);
    this.killRowsThatStoppedBeingLive(liveBefore);
  }

  /** Replaces a resident tile, or returns false when the larger tile does not fit. */
  replace(key: string, data: GPUResidencyArenaTileData): boolean {
    const liveBefore = this.getLiveMask();
    try {
      const tile = this.arena.replaceTile(key, data);
      this.writeTile(tile.ranges, data);
    } catch (error) {
      if (error instanceof ResidencyArenaFullError) {
        return false;
      }
      throw error;
    }
    this.killRowsThatStoppedBeingLive(liveBefore);
    return true;
  }

  /**
   * One churn step: evicts some tiles, replaces one sometimes, then inserts tiles of random size
   * 1..200 rows until one no longer fits (at most 8 attempts).
   */
  churn(random: () => number, makeTileData: MakeTileData, evictProbability = 0.4): void {
    for (const key of this.getKeys()) {
      if (random() < evictProbability) {
        this.evict(key);
      }
    }
    const keys = this.getKeys();
    if (keys.length > 0 && random() < 0.3) {
      this.replace(keys[Math.floor(random() * keys.length)], makeTileData(randomTileSize(random)));
    }
    for (let attempt = 0; attempt < 8; attempt++) {
      if (this.insert(makeTileData(randomTileSize(random))) === undefined) {
        break;
      }
    }
  }

  destroy(): void {
    this.arena.destroy();
  }

  private writeTile(
    ranges: readonly {firstRow: number; rowCount: number}[],
    data: GPUResidencyArenaTileData
  ): void {
    let tileRow = 0;
    for (const range of ranges) {
      for (const column of this.arena.columns) {
        const components = COLUMN_COMPONENTS[column.format];
        this.columns[column.name].set(
          data.columns[column.name].subarray(
            tileRow * components,
            (tileRow + range.rowCount) * components
          ),
          range.firstRow * components
        );
      }
      tileRow += range.rowCount;
    }
  }

  private killRowsThatStoppedBeingLive(liveBefore: Uint32Array): void {
    const liveAfter = this.getLiveMask();
    for (let row = 0; row < liveAfter.length; row++) {
      if (liveBefore[row] && !liveAfter[row]) {
        for (const column of this.arena.columns) {
          if (column.deadValue !== undefined) {
            const components = COLUMN_COMPONENTS[column.format];
            this.columns[column.name].fill(
              column.deadValue,
              row * components,
              (row + 1) * components
            );
          }
        }
      }
    }
  }
}

/** Random tile size in 1..200 rows. */
export function randomTileSize(random: () => number): number {
  return 1 + Math.floor(random() * 200);
}

/** Ascending arena rows that are live. */
export function getLiveRows(mirror: ArenaMirror): number[] {
  return Array.from(mirror.arena.allocator.getLiveRows());
}
