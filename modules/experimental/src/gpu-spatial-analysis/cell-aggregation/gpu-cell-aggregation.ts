// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  CELL_KEY_WGSL,
  getCellKeyLayout,
  QUADBIN_TILE_WGSL,
  type CellKeyLayout,
  type GPUCellFamily
} from './cell-keys';
import {
  getCellTableNodes,
  getCellTableViews,
  GPU_CELL_DEFAULT_SUM_SCALE,
  hasCellKeyHighWord,
  validateCellTable,
  validateSumScale,
  type GPUCellTable
} from './cell-table';

const OPERATION = 'GPUCellAggregation';

/** Word order of pre-keyed 64-bit cell rows. */
export type GPUCellWordOrder = 'little-endian' | 'high-low';

/**
 * Properties for {@link GPUCellAggregation}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `mask` and `values`.
 * Topology (needs a new graph): `family`, `resolution`, `sumScale`, view lengths, which optional
 * views are present, and the output capacity.
 */
export type GPUCellAggregationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-aggregation'`. */
  id?: string;
  /** Grid family of the keys. Points (`positions`) are supported for `'quadbin'` only. */
  family: GPUCellFamily;
  /** Resolution of the output table: Quadbin 0-26, H3 0-15. */
  resolution: number;
  /**
   * Longitude/latitude degrees per row (Quadbin only). Longitude is clipped to [-180, 180] and
   * wraps 180 to column 0; latitude is clipped to ±85.051129. Rows with a NaN coordinate are
   * skipped.
   */
  positions?: GraphDataView<'float32x2'>;
  /**
   * Pre-keyed rows: one 64-bit Quadbin or H3 index per row as two `uint32` words, for example an
   * Arrow `Uint64` column. Rows finer than `resolution` are rolled up to it; invalid rows and rows
   * coarser than `resolution` are skipped.
   */
  cells?: GraphDataView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellWordOrder;
  /** Optional per-row value. Required for sums and extremes; rows with a non-finite value are skipped. */
  values?: GraphDataView<'float32'>;
  /** Optional per-row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Fixed-point scale of `output.sums`, a positive float32 value; a power of two keeps the
   * scaling exact. Defaults to 65536 (16 fractional bits).
   */
  sumScale?: number;
  /** Caller-owned cell table; its capacity is `output.cells.length`. */
  output: GPUCellTable;
  /** Optional extra one-row destinations of the clamped output count. @internal */
  extraCounts?: readonly GraphDataView<'uint32'>[];
};

/**
 * Bins rows into discrete global grid cells and aggregates them per cell into a compact, sorted,
 * capacity-bounded cell table.
 *
 * Keys are 64-bit Quadbin or H3 indexes handled as two `u32` words. Point rows get Quadbin keys
 * in WGSL with integer arithmetic only (exact tile column; fixed-point Mercator row, see
 * `QUADBIN_TILE_WGSL`), so every device and the BigInt CPU reference produce the same key; f32
 * polynomials would not, because compilers may fuse multiply-adds. Pre-keyed rows (Quadbin or H3,
 * for example CARTO tables) are validated (H3 through the gpu-dggs `dggs_h3_is_valid_cell_id`)
 * and truncated to `resolution`.
 *
 * Rows are sorted by key with stable radix sorts (one when the key path fits 31 bits: Quadbin up
 * to resolution 15, H3 up to 8), and cell boundaries come from a scan of segment heads. Each row
 * then adds into its table row with integer atomics: counts, 64-bit fixed-point sums (two words
 * with an explicit carry), and extremes as order-preserving u32 keys, so every output is exact and
 * independent of thread order. The table keeps the smallest keys when it overflows. Inputs must
 * be single packed views.
 */
export class GPUCellAggregation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellAggregationProps;
  /** Key layout of the output resolution. */
  readonly layout: CellKeyLayout;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;

  constructor(props: GPUCellAggregationProps) {
    this.id = props.id ?? 'cell-aggregation';
    this.props = props;
    const id = this.id;
    if (props.family !== 'quadbin' && props.family !== 'h3') {
      throw new Error(`${id} family must be 'quadbin' or 'h3'`);
    }
    this.layout = getCellKeyLayout(props.family, props.resolution);
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    if (Boolean(props.positions) === Boolean(props.cells)) {
      throw new Error(`${id} needs exactly one of positions or cells`);
    }
    if (props.positions && props.family !== 'quadbin') {
      throw new Error(`${id} positions are supported for the quadbin family only`);
    }
    for (const [name, view] of [
      ['positions', props.positions],
      ['cells', props.cells],
      ['values', props.values],
      ['mask', props.mask]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rows = (props.positions ?? props.cells)!.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.positions) {
      validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    }
    if (props.cells) {
      validatePackedView(props.cells, ['uint32x2'], `${id} cells`);
    }
    if (props.values) {
      validatePackedView(props.values, ['float32'], `${id} values`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
    }
    for (const [name, view] of [
      ['values', props.values],
      ['mask', props.mask]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the row count`);
      }
    }
    validateCellTable(id, 'output', props.output);
    const {output} = props;
    if (!props.values && (output.sums || output.sumValues || output.minimums || output.maximums)) {
      throw new Error(`${id} needs values for sums and extremes`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [...getCellTableViews(output), ...(props.extraCounts ?? [])],
      [props.positions, props.cells, props.values, props.mask]
    );
  }

  /** Returns key, sort, boundary, table, reduction, and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, layout} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.cells,
      props.values,
      props.mask,
      ...getCellTableViews(props.output),
      ...(props.extraCounts ?? [])
    ]);
    const rows = (props.positions ?? props.cells)!.length;
    const twoWords = hasCellKeyHighWord(layout);
    const keyLow = createTransientView(graph, `${id}-key-low`, 'uint32', rows);
    const keyHigh = twoWords
      ? createTransientView(graph, `${id}-key-high`, 'uint32', rows)
      : undefined;
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
    const bindings: WGSLKernelBinding[] = [
      props.positions
        ? {
            name: 'positions',
            view: props.positions,
            type: 'f32',
            access: 'read'
          }
        : {name: 'cells', view: props.cells!, type: 'u32', access: 'read'},
      ...(props.mask
        ? [
            {
              name: 'rowMask',
              view: props.mask,
              type: 'u32',
              access: 'read'
            } as const
          ]
        : []),
      ...(props.values
        ? [
            {
              name: 'values',
              view: props.values,
              type: 'f32',
              access: 'read'
            } as const
          ]
        : []),
      {name: 'keyLow', view: keyLow, type: 'u32', access: 'read_write'},
      ...(keyHigh
        ? [
            {
              name: 'keyHigh',
              view: keyHigh,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : []),
      {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
    ];
    const invalidKey = `cellShiftLeft(vec2u(0u, 1u), ${layout.width}u)`;
    const keyBody = props.positions
      ? `let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  if (isValid && !isNanBits(longitude) && !isNanBits(latitude)) {
    let x = quadbinGetTileX(longitude, RESOLUTION);
    let y = quadbinGetTileY(latitude, RESOLUTION);
    compact = quadbinGetCompactKey(x, y);
  }`
      : `let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  ${
    props.family === 'h3'
      ? 'let isValidKey = dggs_h3_is_valid_cell_id(key) && dggs_h3_get_resolution(key) >= RESOLUTION;'
      : 'let isValidKey = cellIsValidQuadbin(key) && ((key.x >> 20u) & 0x1fu) >= RESOLUTION;'
  }
  if (isValid && isValidKey) {
    compact = cellGetCompactKey(key, ${layout.lowBit}u, ${layout.width}u);
  }`;
    const keyNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-keys`,
      operation: OPERATION,
      variant: props.positions ? 'quadbin-points' : `${props.family}-cells`,
      bindings,
      invocationCount: rows,
      declarations: `const RESOLUTION: u32 = ${layout.resolution}u;
${props.cells && props.family === 'h3' ? dggs.source : ''}
${CELL_KEY_WGSL}
${props.positions ? QUADBIN_TILE_WGSL : ''}
fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}`,
      body: `var isValid = true;
  ${props.mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${
    props.values
      ? `let value = values[valuesOffset + index];
  isValid = isValid && (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;`
      : ''
  }
  var compact = ${invalidKey};
  ${keyBody}
  keyLow[keyLowOffset + index] = compact.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + index] = compact.x;' : ''}
  rowIds[rowIdsOffset + index] = index;`
    });
    return [
      keyNode,
      ...getCellTableNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        layout,
        keyLow,
        keyHigh,
        rowIds,
        sorted: false,
        source: {kind: 'rows', values: props.values},
        sumScale: this.sumScale,
        output: props.output,
        extraCounts: props.extraCounts
      })
    ];
  }
}
