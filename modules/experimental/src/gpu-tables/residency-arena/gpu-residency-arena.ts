// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../utils/gpu-contributor-utils';
import {ResidencyArenaAllocator} from './residency-arena-allocator';
import {
  GPU_RESIDENCY_ARENA_DEAD_SLOT,
  type GPUResidencyArenaColumnFormat,
  type GPUResidencyArenaColumnSpec,
  type GPUResidencyArenaGraphViews,
  type GPUResidencyArenaRowRange,
  type GPUResidencyArenaTile
} from './residency-arena-types';

/** Properties for {@link GPUResidencyArena}. */
export type GPUResidencyArenaProps = {
  /** Buffer ID prefix. Defaults to `'residency-arena'`. */
  id?: string;
  /** Total arena rows. A positive multiple of `pageRowCount`. */
  rowCapacity: number;
  /** Rows per page. A positive multiple of 64. Defaults to 1024. */
  pageRowCount?: number;
  /** Maximum simultaneously resident tiles. Defaults to the page count. */
  maxTileCount?: number;
  /** Column layout. Names must be unique. */
  columns: readonly GPUResidencyArenaColumnSpec[];
  /** Extra `Buffer` usage bits ORed onto the column buffers. */
  usage?: number;
};

/** Typed values of one tile: every arena column must be present. */
export type GPUResidencyArenaTileData = {
  /** Number of rows in the tile. */
  rowCount: number;
  /** Per-column values, `rowCount * components` elements each. */
  columns: Record<string, Float32Array | Uint32Array | Int32Array>;
};

const COLUMN_FORMAT_INFO: Record<
  GPUResidencyArenaColumnFormat,
  {components: number; scalar: 'f32' | 'u32' | 'i32'}
> = {
  float32: {components: 1, scalar: 'f32'},
  float32x2: {components: 2, scalar: 'f32'},
  float32x3: {components: 3, scalar: 'f32'},
  float32x4: {components: 4, scalar: 'f32'},
  uint32: {components: 1, scalar: 'u32'},
  uint32x2: {components: 2, scalar: 'u32'},
  sint32: {components: 1, scalar: 'i32'}
};

const SCALAR_ARRAY_CONSTRUCTORS = {
  f32: Float32Array,
  u32: Uint32Array,
  i32: Int32Array
};

/**
 * A fixed-capacity, never-reallocated set of column buffers whose rows are paged out to resident
 * tiles by a {@link ResidencyArenaAllocator}.
 *
 * Invariant: every row that is not live has `liveMask` 0, `rowTileSlots`
 * `GPU_RESIDENCY_ARENA_DEAD_SLOT`, and the declared `deadValue` in every column that declares one.
 * Construction initializes that state and eviction restores it, so an insert never needs to touch
 * the unused tail of its last page. No operation creates a buffer.
 */
export class GPUResidencyArena {
  /** Buffer ID prefix. */
  readonly id: string;
  /** CPU page allocator and oracle. */
  readonly allocator: ResidencyArenaAllocator;
  /** Column layout. */
  readonly columns: readonly GPUResidencyArenaColumnSpec[];
  /** `rowCapacity` uint32 rows: 1 for live rows. */
  readonly liveMaskBuffer: Buffer;
  /** `rowCapacity` uint32 rows: tile slot of live rows, `GPU_RESIDENCY_ARENA_DEAD_SLOT` otherwise. */
  readonly rowTileSlotsBuffer: Buffer;

  private readonly columnBuffers = new Map<string, Buffer>();
  private scratch = new ArrayBuffer(0);

  constructor(device: Device, props: GPUResidencyArenaProps) {
    this.id = props.id ?? 'residency-arena';
    this.allocator = new ResidencyArenaAllocator(props);
    this.columns = props.columns;
    const rowCapacity = this.allocator.rowCapacity;
    const usage =
      Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC | Buffer.VERTEX | (props.usage ?? 0);
    for (const column of props.columns) {
      if (this.columnBuffers.has(column.name)) {
        throw new Error(`${this.id} duplicate column "${column.name}"`);
      }
      const info = COLUMN_FORMAT_INFO[column.format];
      const buffer = device.createBuffer({
        id: `${this.id}-${column.name}`,
        byteLength: rowCapacity * info.components * 4,
        usage
      });
      this.columnBuffers.set(column.name, buffer);
      if (column.deadValue !== undefined) {
        const values = new SCALAR_ARRAY_CONSTRUCTORS[info.scalar](rowCapacity * info.components);
        values.fill(column.deadValue);
        buffer.write(values);
      }
    }
    const stateUsage = Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC;
    this.liveMaskBuffer = device.createBuffer({
      id: `${this.id}-live-mask`,
      byteLength: rowCapacity * 4,
      usage: stateUsage
    });
    this.rowTileSlotsBuffer = device.createBuffer({
      id: `${this.id}-row-tile-slots`,
      byteLength: rowCapacity * 4,
      usage: stateUsage
    });
    this.liveMaskBuffer.write(new Uint32Array(rowCapacity));
    this.rowTileSlotsBuffer.write(new Uint32Array(rowCapacity).fill(GPU_RESIDENCY_ARENA_DEAD_SLOT));
  }

  /** Fixed row capacity. */
  get rowCapacity(): number {
    return this.allocator.rowCapacity;
  }
  /** Allocator version; increments on every successful insert, evict, or replace. */
  get version(): number {
    return this.allocator.version;
  }
  /** Number of resident tiles. */
  get residentTileCount(): number {
    return this.allocator.residentTileCount;
  }
  /** Total live rows. */
  get liveRowCount(): number {
    return this.allocator.liveRowCount;
  }

  /** Returns the buffer of a column. Throws on an unknown name. */
  getColumnBuffer(name: string): Buffer {
    const buffer = this.columnBuffers.get(name);
    if (!buffer) {
      throw new Error(`${this.id} has no column "${name}"`);
    }
    return buffer;
  }

  /** Inserts a tile and writes its data, live mask, and slots. Validates before mutating. */
  insertTile(key: string, data: GPUResidencyArenaTileData): GPUResidencyArenaTile {
    this.validateTileData(data);
    const tile = this.allocator.insert(key, data.rowCount);
    this.writeLiveTile(tile, data);
    return tile;
  }

  /** Evicts a tile, restoring the dead state over its rows. Returns the removed record. */
  evictTile(key: string): GPUResidencyArenaTile {
    const tile = this.allocator.evict(key);
    this.writeDeadRows(tile.ranges);
    return tile;
  }

  /**
   * Replaces a resident tile's data in place, keeping its slot and reusing its pages where
   * possible. Rows that are no longer part of the tile end up dead.
   */
  replaceTile(key: string, data: GPUResidencyArenaTileData): GPUResidencyArenaTile {
    this.validateTileData(data);
    const {previous, next} = this.allocator.replace(key, data.rowCount);
    this.writeLiveTile(next, data);
    this.writeDeadRows(subtractRanges(previous.ranges, next.ranges));
    return next;
  }

  /**
   * Imports every arena buffer into a graph, one packed view per buffer of length `rowCapacity`.
   * Resource IDs are `${id}-${column}`, `${id}-live-mask`, and `${id}-row-tile-slots`.
   */
  importToGraph<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    id: string = this.id
  ): GPUResidencyArenaGraphViews {
    const columns: Record<string, GraphDataView> = {};
    for (const column of this.columns) {
      columns[column.name] = importGraphBuffer(
        graph,
        `${id}-${column.name}`,
        this.getColumnBuffer(column.name),
        column.format,
        this.rowCapacity
      );
    }
    return {
      columns,
      liveMask: importGraphBuffer(
        graph,
        `${id}-live-mask`,
        this.liveMaskBuffer,
        'uint32',
        this.rowCapacity
      ),
      rowTileSlots: importGraphBuffer(
        graph,
        `${id}-row-tile-slots`,
        this.rowTileSlotsBuffer,
        'uint32',
        this.rowCapacity
      ),
      rowCapacity: this.rowCapacity
    };
  }

  /** Destroys the owned buffers. Destroy compiled graphs that import them first. */
  destroy(): void {
    for (const buffer of this.columnBuffers.values()) {
      buffer.destroy();
    }
    this.liveMaskBuffer.destroy();
    this.rowTileSlotsBuffer.destroy();
  }

  private validateTileData(data: GPUResidencyArenaTileData): void {
    if (!Number.isInteger(data.rowCount) || data.rowCount < 0) {
      throw new Error(`${this.id} rowCount must be a non-negative integer`);
    }
    for (const column of this.columns) {
      const info = COLUMN_FORMAT_INFO[column.format];
      const values = data.columns[column.name];
      if (!values) {
        throw new Error(`${this.id} tile data is missing column "${column.name}"`);
      }
      if (!(values instanceof SCALAR_ARRAY_CONSTRUCTORS[info.scalar])) {
        throw new Error(
          `${this.id} column "${column.name}" must be a ${SCALAR_ARRAY_CONSTRUCTORS[info.scalar].name}`
        );
      }
      if (values.length !== data.rowCount * info.components) {
        throw new Error(
          `${this.id} column "${column.name}" needs ${data.rowCount * info.components} values, got ${values.length}`
        );
      }
    }
  }

  private writeLiveTile(tile: GPUResidencyArenaTile, data: GPUResidencyArenaTileData): void {
    let tileRow = 0;
    for (const range of tile.ranges) {
      for (const column of this.columns) {
        const components = COLUMN_FORMAT_INFO[column.format].components;
        this.getColumnBuffer(column.name).write(
          data.columns[column.name].subarray(
            tileRow * components,
            (tileRow + range.rowCount) * components
          ),
          range.firstRow * components * 4
        );
      }
      this.liveMaskBuffer.write(this.fillScratch('u32', range.rowCount, 1), range.firstRow * 4);
      this.rowTileSlotsBuffer.write(
        this.fillScratch('u32', range.rowCount, tile.slot),
        range.firstRow * 4
      );
      tileRow += range.rowCount;
    }
  }

  private writeDeadRows(ranges: readonly GPUResidencyArenaRowRange[]): void {
    for (const range of ranges) {
      this.liveMaskBuffer.write(this.fillScratch('u32', range.rowCount, 0), range.firstRow * 4);
      this.rowTileSlotsBuffer.write(
        this.fillScratch('u32', range.rowCount, GPU_RESIDENCY_ARENA_DEAD_SLOT),
        range.firstRow * 4
      );
      for (const column of this.columns) {
        if (column.deadValue === undefined) {
          continue;
        }
        const info = COLUMN_FORMAT_INFO[column.format];
        this.getColumnBuffer(column.name).write(
          this.fillScratch(info.scalar, range.rowCount * info.components, column.deadValue),
          range.firstRow * info.components * 4
        );
      }
    }
  }

  /** Fills the grow-only write-staging scratch with `value` and returns a typed view of it. */
  private fillScratch(
    scalar: 'f32' | 'u32' | 'i32',
    length: number,
    value: number
  ): Float32Array | Uint32Array | Int32Array {
    if (this.scratch.byteLength < length * 4) {
      this.scratch = new ArrayBuffer(length * 4);
    }
    const view = new SCALAR_ARRAY_CONSTRUCTORS[scalar](this.scratch, 0, length);
    view.fill(value);
    return view;
  }
}

/** Returns the parts of ascending `ranges` that are not covered by ascending `subtrahend`. */
function subtractRanges(
  ranges: readonly GPUResidencyArenaRowRange[],
  subtrahend: readonly GPUResidencyArenaRowRange[]
): GPUResidencyArenaRowRange[] {
  const result: GPUResidencyArenaRowRange[] = [];
  let cursor = 0;
  for (const range of ranges) {
    let start = range.firstRow;
    const end = range.firstRow + range.rowCount;
    while (
      cursor < subtrahend.length &&
      subtrahend[cursor].firstRow + subtrahend[cursor].rowCount <= start
    ) {
      cursor++;
    }
    let index = cursor;
    while (start < end) {
      const cut = subtrahend[index];
      if (!cut || cut.firstRow >= end) {
        result.push({firstRow: start, rowCount: end - start});
        break;
      }
      if (cut.firstRow > start) {
        result.push({firstRow: start, rowCount: cut.firstRow - start});
      }
      start = Math.max(start, cut.firstRow + cut.rowCount);
      index++;
    }
  }
  return result;
}
