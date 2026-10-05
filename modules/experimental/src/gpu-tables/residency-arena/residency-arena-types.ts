// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';

/** `rowTileSlots` value of a row that belongs to no resident tile. */
export const GPU_RESIDENCY_ARENA_DEAD_SLOT = 0xffffffff;

/**
 * Fixed-width column formats an arena column may use. Every component is 32 bits, so a column is
 * a plain WGSL `array<f32>`, `array<u32>`, or `array<i32>` indexed by `row * components`.
 */
export type GPUResidencyArenaColumnFormat =
  | 'float32'
  | 'float32x2'
  | 'float32x3'
  | 'float32x4'
  | 'uint32'
  | 'uint32x2'
  | 'sint32';

/** One arena column: one buffer of `rowCapacity` rows that never reallocates. */
export type GPUResidencyArenaColumnSpec = {
  /** Column name, unique within the arena. Also the suffix of the graph resource ID. */
  name: string;
  /** Fixed-width row format. */
  format: GPUResidencyArenaColumnFormat;
  /**
   * Optional value written to every component of rows that hold no live data: at construction,
   * on eviction, and into the unused tail of a tile's last page. Use `NaN` for positions so
   * contributors without a mask input (for example `GPUPointDensity`, which ignores non-finite
   * positions) skip dead rows, and `0` for weights. When omitted, dead rows keep stale data and
   * only the live mask excludes them.
   */
  deadValue?: number;
};

/** One contiguous run of arena rows. */
export type GPUResidencyArenaRowRange = {
  /** First arena row of the run. */
  firstRow: number;
  /** Number of rows in the run. */
  rowCount: number;
};

/** CPU record of one resident tile. Ranges are in ascending row order. */
export type GPUResidencyArenaTile = {
  /** Caller key, unique among resident tiles. */
  key: string;
  /** Dense tile slot in `[0, maxTileCount)`, written to `rowTileSlots` for every live row. */
  slot: number;
  /** Live rows of the tile. */
  rowCount: number;
  /** Ascending page indices the tile occupies. */
  pages: readonly number[];
  /**
   * Live row runs, ascending, coalesced across adjacent pages, covering exactly `rowCount` rows.
   * Tile row `i` is the `i`-th row of the concatenated ranges.
   */
  ranges: readonly GPUResidencyArenaRowRange[];
};

/** Result of resolving one arena row back to its tile. */
export type GPUResidencyArenaResolvedRow = {
  /** Tile key. */
  key: string;
  /** Tile slot. */
  slot: number;
  /** Row index within the tile's own data. */
  rowInTile: number;
};

/**
 * Graph views over one arena, all imported once per graph. Each column is ONE packed view over
 * its whole buffer (`length === rowCapacity`), so a contributor over arena columns emits the same nodes
 * whatever the number or placement of resident tiles.
 */
export type GPUResidencyArenaGraphViews = {
  /** One packed view per column, keyed by column name. */
  columns: Record<string, GraphDataView>;
  /** `rowCapacity` uint32 rows: 1 for live rows, 0 for dead rows. */
  liveMask: GraphDataView<'uint32'>;
  /** `rowCapacity` uint32 rows: tile slot of each live row, `GPU_RESIDENCY_ARENA_DEAD_SLOT` otherwise. */
  rowTileSlots: GraphDataView<'uint32'>;
  /** Fixed row capacity, equal to every view's length. */
  rowCapacity: number;
};
