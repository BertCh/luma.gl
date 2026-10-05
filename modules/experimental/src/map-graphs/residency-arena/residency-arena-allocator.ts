// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_RESIDENCY_ARENA_DEAD_SLOT,
  type GPUResidencyArenaResolvedRow,
  type GPUResidencyArenaRowRange,
  type GPUResidencyArenaTile
} from './residency-arena-types';

/** Thrown by {@link ResidencyArenaAllocator} when pages or tile slots are insufficient. */
export class ResidencyArenaFullError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ResidencyArenaFullError';
  }
}

/** Properties for {@link ResidencyArenaAllocator}. */
export type ResidencyArenaAllocatorProps = {
  /** Total arena rows. A positive multiple of `pageRowCount`. */
  rowCapacity: number;
  /** Rows per page. A positive multiple of 64. Defaults to 1024. */
  pageRowCount?: number;
  /** Maximum simultaneously resident tiles. Defaults to the page count. */
  maxTileCount?: number;
};

/**
 * Pure CPU page allocator for a fixed-capacity row arena. It is also the CPU oracle of
 * `GPUResidencyArena`.
 *
 * Allocation policy: the arena is split into fixed-size pages. A tile of `n` rows takes
 * `ceil(n / pageRowCount)` pages (a 0-row tile takes no pages but still a slot). Pages are chosen
 * lowest-index-first from ANY free pages, not necessarily contiguous, so fragmentation never
 * blocks an insert while enough pages are free. Row ranges coalesce adjacent pages; the unused
 * tail of a tile's last page is not part of its ranges. Tile slots are the lowest free slot in
 * `[0, maxTileCount)`. Every operation is O(pageCount).
 *
 * Rejected: contiguous first-fit (fragmentation blocks inserts and needs compaction copies).
 */
export class ResidencyArenaAllocator {
  /** Total arena rows. */
  readonly rowCapacity: number;
  /** Rows per page. */
  readonly pageRowCount: number;
  /** Number of pages. */
  readonly pageCount: number;
  /** Maximum simultaneously resident tiles. */
  readonly maxTileCount: number;

  private tilesByKey = new Map<string, GPUResidencyArenaTile>();
  private tilesBySlot: (GPUResidencyArenaTile | undefined)[];
  /** Slot owning each page, or -1 when free. */
  private pageSlots: Int32Array;
  /** Index of each page within its tile's page list. */
  private pageIndicesInTile: Uint32Array;
  private freePages: number;
  private liveRows = 0;
  private currentVersion = 0;

  constructor(props: ResidencyArenaAllocatorProps) {
    const pageRowCount = props.pageRowCount ?? 1024;
    if (!Number.isInteger(pageRowCount) || pageRowCount <= 0 || pageRowCount % 64 !== 0) {
      throw new Error('pageRowCount must be a positive multiple of 64');
    }
    if (
      !Number.isInteger(props.rowCapacity) ||
      props.rowCapacity <= 0 ||
      props.rowCapacity % pageRowCount !== 0
    ) {
      throw new Error('rowCapacity must be a positive multiple of pageRowCount');
    }
    this.rowCapacity = props.rowCapacity;
    this.pageRowCount = pageRowCount;
    this.pageCount = props.rowCapacity / pageRowCount;
    const maxTileCount = props.maxTileCount ?? this.pageCount;
    if (!Number.isInteger(maxTileCount) || maxTileCount <= 0) {
      throw new Error('maxTileCount must be a positive integer');
    }
    this.maxTileCount = maxTileCount;
    this.tilesBySlot = new Array(maxTileCount).fill(undefined);
    this.pageSlots = new Int32Array(this.pageCount).fill(-1);
    this.pageIndicesInTile = new Uint32Array(this.pageCount);
    this.freePages = this.pageCount;
  }

  /** Pages not owned by any tile. */
  get freePageCount(): number {
    return this.freePages;
  }

  /** Number of resident tiles. */
  get residentTileCount(): number {
    return this.tilesByKey.size;
  }

  /** Sum of the row counts of all resident tiles. */
  get liveRowCount(): number {
    return this.liveRows;
  }

  /** Increments on every successful insert, evict, or replace. */
  get version(): number {
    return this.currentVersion;
  }

  /** Returns whether a tile of `rowCount` rows would currently fit (pages and a slot). */
  canInsert(rowCount: number): boolean {
    return (
      Number.isInteger(rowCount) &&
      rowCount >= 0 &&
      this.tilesByKey.size < this.maxTileCount &&
      this.getPagesNeeded(rowCount) <= this.freePages
    );
  }

  /**
   * Inserts a tile.
   *
   * @throws {ResidencyArenaFullError} when pages or slots are insufficient.
   * @throws {Error} on a duplicate key or invalid `rowCount`.
   */
  insert(key: string, rowCount: number): GPUResidencyArenaTile {
    this.validateRowCount(rowCount);
    if (this.tilesByKey.has(key)) {
      throw new Error(`residency arena already holds tile "${key}"`);
    }
    const slot = this.tilesBySlot.indexOf(undefined);
    if (slot < 0) {
      throw new ResidencyArenaFullError(`residency arena has no free tile slot for "${key}"`);
    }
    const pagesNeeded = this.getPagesNeeded(rowCount);
    if (pagesNeeded > this.freePages) {
      throw new ResidencyArenaFullError(
        `residency arena needs ${pagesNeeded} pages for "${key}" but only ${this.freePages} are free`
      );
    }
    const pages = this.takeFreePages(pagesNeeded);
    const tile = this.assignTile(key, slot, rowCount, pages);
    this.tilesByKey.set(key, tile);
    this.liveRows += rowCount;
    this.currentVersion++;
    return tile;
  }

  /** Evicts a tile and returns its removed record. Throws on an unknown key. */
  evict(key: string): GPUResidencyArenaTile {
    const tile = this.requireTile(key);
    for (const page of tile.pages) {
      this.pageSlots[page] = -1;
    }
    this.freePages += tile.pages.length;
    this.tilesBySlot[tile.slot] = undefined;
    this.tilesByKey.delete(key);
    this.liveRows -= tile.rowCount;
    this.currentVersion++;
    return tile;
  }

  /**
   * Resizes a resident tile in place. The slot is kept; the lowest-indexed pages it already owns
   * that it still needs are kept, the rest are freed, and new pages are allocated only when it
   * needs more.
   *
   * @throws {ResidencyArenaFullError} when it cannot fit; state is left unchanged.
   */
  replace(
    key: string,
    rowCount: number
  ): {previous: GPUResidencyArenaTile; next: GPUResidencyArenaTile} {
    this.validateRowCount(rowCount);
    const previous = this.requireTile(key);
    const pagesNeeded = this.getPagesNeeded(rowCount);
    const extraPages = pagesNeeded - previous.pages.length;
    if (extraPages > this.freePages) {
      throw new ResidencyArenaFullError(
        `residency arena needs ${extraPages} more pages for "${key}" but only ${this.freePages} are free`
      );
    }
    let pages: number[];
    if (extraPages <= 0) {
      pages = previous.pages.slice(0, pagesNeeded);
      for (const page of previous.pages.slice(pagesNeeded)) {
        this.pageSlots[page] = -1;
      }
      this.freePages -= extraPages;
    } else {
      pages = [...previous.pages, ...this.takeFreePages(extraPages)].sort((a, b) => a - b);
    }
    const next = this.assignTile(key, previous.slot, rowCount, pages);
    this.tilesByKey.set(key, next);
    this.liveRows += rowCount - previous.rowCount;
    this.currentVersion++;
    return {previous, next};
  }

  /** Returns whether a tile with `key` is resident. */
  has(key: string): boolean {
    return this.tilesByKey.has(key);
  }

  /** Returns the resident tile record, or `undefined`. */
  getTile(key: string): GPUResidencyArenaTile | undefined {
    return this.tilesByKey.get(key);
  }

  /** Returns all resident tiles in slot order. */
  getTiles(): GPUResidencyArenaTile[] {
    return this.tilesBySlot.filter((tile): tile is GPUResidencyArenaTile => tile !== undefined);
  }

  /** Resolves an arena row to its tile and row within the tile, or `null` when the row is dead. */
  resolveRow(row: number): GPUResidencyArenaResolvedRow | null {
    if (!Number.isInteger(row) || row < 0 || row >= this.rowCapacity) {
      return null;
    }
    const page = Math.floor(row / this.pageRowCount);
    const slot = this.pageSlots[page];
    if (slot < 0) {
      return null;
    }
    const tile = this.tilesBySlot[slot]!;
    const rowInTile =
      this.pageIndicesInTile[page] * this.pageRowCount + (row - page * this.pageRowCount);
    return rowInTile < tile.rowCount ? {key: tile.key, slot, rowInTile} : null;
  }

  /** Oracle: `rowCapacity` values, 1 for live rows and 0 otherwise. */
  getLiveMask(): Uint32Array {
    const mask = new Uint32Array(this.rowCapacity);
    for (const tile of this.tilesByKey.values()) {
      for (const range of tile.ranges) {
        mask.fill(1, range.firstRow, range.firstRow + range.rowCount);
      }
    }
    return mask;
  }

  /** Oracle: tile slot of each row, or `GPU_RESIDENCY_ARENA_DEAD_SLOT` for dead rows. */
  getRowTileSlots(): Uint32Array {
    const slots = new Uint32Array(this.rowCapacity).fill(GPU_RESIDENCY_ARENA_DEAD_SLOT);
    for (const tile of this.tilesByKey.values()) {
      for (const range of tile.ranges) {
        slots.fill(tile.slot, range.firstRow, range.firstRow + range.rowCount);
      }
    }
    return slots;
  }

  /** Oracle: ascending arena rows that are live. */
  getLiveRows(): Uint32Array {
    const mask = this.getLiveMask();
    const rows = new Uint32Array(this.liveRows);
    let count = 0;
    for (let row = 0; row < mask.length; row++) {
      if (mask[row]) {
        rows[count++] = row;
      }
    }
    return rows;
  }

  private getPagesNeeded(rowCount: number): number {
    return Math.ceil(rowCount / this.pageRowCount);
  }

  private validateRowCount(rowCount: number): void {
    if (!Number.isInteger(rowCount) || rowCount < 0) {
      throw new Error('rowCount must be a non-negative integer');
    }
  }

  private requireTile(key: string): GPUResidencyArenaTile {
    const tile = this.tilesByKey.get(key);
    if (!tile) {
      throw new Error(`residency arena has no tile "${key}"`);
    }
    return tile;
  }

  /** Takes the `count` lowest-indexed free pages, marking them with a placeholder owner. */
  private takeFreePages(count: number): number[] {
    const pages: number[] = [];
    for (let page = 0; page < this.pageCount && pages.length < count; page++) {
      if (this.pageSlots[page] < 0) {
        pages.push(page);
        this.pageSlots[page] = 0;
      }
    }
    this.freePages -= count;
    return pages;
  }

  private assignTile(
    key: string,
    slot: number,
    rowCount: number,
    pages: number[]
  ): GPUResidencyArenaTile {
    const ranges: GPUResidencyArenaRowRange[] = [];
    let remaining = rowCount;
    pages.forEach((page, indexInTile) => {
      this.pageSlots[page] = slot;
      this.pageIndicesInTile[page] = indexInTile;
      const rows = Math.min(remaining, this.pageRowCount);
      remaining -= rows;
      const last = ranges[ranges.length - 1];
      if (last && last.firstRow + last.rowCount === page * this.pageRowCount) {
        last.rowCount += rows;
      } else {
        ranges.push({firstRow: page * this.pageRowCount, rowCount: rows});
      }
    });
    const tile: GPUResidencyArenaTile = {key, slot, rowCount, pages, ranges};
    this.tilesBySlot[slot] = tile;
    return tile;
  }
}
