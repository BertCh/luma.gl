// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {CELL_KEY_WGSL, type CellKeyLayout} from './cell-keys';

/** Default fixed-point scale of {@link GPUCellTable.sums}: 16 fractional bits. */
export const GPU_CELL_DEFAULT_SUM_SCALE = 65536;

/** Key word written to every word of an unused cell table row. */
export const GPU_CELL_EMPTY_KEY_WORD = 0xffffffff;

/**
 * Caller-owned, capacity-bounded cell table written by {@link GPUCellAggregation} and
 * {@link GPUCellRollup}.
 *
 * Capacity is `cells.length`. Rows `[0, count)` hold one occupied cell each, in ascending key
 * order; later rows hold the empty key (`0xffffffff` in both words), a zero count and sum, and
 * NaN extremes. When more cells exist than fit, the cells with the largest keys are dropped and
 * `overflow` is 1. Every column must hold `cells.length` rows.
 */
export type GPUCellTable = {
  /**
   * 64-bit cell keys as little-endian `(low, high)` `uint32` words, the layout of Arrow `Uint64`
   * columns and of `GPUH3CellProjection` input, so an H3 table feeds the projection directly.
   */
  cells: GraphDataView<'uint32x2'>;
  /** Rows aggregated into each cell. */
  counts: GraphDataView<'uint32'>;
  /**
   * Exact fixed-point sums: signed 64-bit two's-complement integers as little-endian words, equal
   * to the sum over rows of `roundHalfEven(fround(value * sumScale))`. Integer sums are
   * associative, so a roll-up equals a direct aggregation bit for bit. Required on a source
   * table to roll sums up to a coarser level.
   */
  sums?: GraphDataView<'uint32x2'>;
  /** `sums / sumScale` as f32, for styling. */
  sumValues?: GraphDataView<'float32'>;
  /** Minimum value per cell. */
  minimums?: GraphDataView<'float32'>;
  /** Maximum value per cell. */
  maximums?: GraphDataView<'float32'>;
  /** One-row scalar receiving the occupied row count, `min(totalCount, cells.length)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when cells were dropped here or in any source level. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of occupied cells. */
  totalCount?: GraphDataView<'uint32'>;
};

/** Returns every view of a cell table, for graph membership and aliasing checks. @internal */
export function getCellTableViews(table: GPUCellTable): (GraphDataView | undefined)[] {
  return [
    table.cells,
    table.counts,
    table.sums,
    table.sumValues,
    table.minimums,
    table.maximums,
    table.count,
    table.overflow,
    table.totalCount
  ];
}

/** Returns the first row of `table.cells` within its buffer, for instanced draws. */
export function getCellTableFirstRow(table: GPUCellTable): number {
  return table.cells.byteOffset / table.cells.rowByteLength;
}

/** Validates formats, packing and row counts of a cell table. @internal */
export function validateCellTable(id: string, name: string, table: GPUCellTable): void {
  for (const [column, view] of getCellTableViews(table).entries()) {
    if ((view as unknown) instanceof GraphVectorView) {
      throw new Error(`${id} ${name} column ${column} must be a single packed view`);
    }
  }
  validatePackedView(table.cells, ['uint32x2'], `${id} ${name}.cells`);
  const capacity = table.cells.length;
  if (capacity < 1) {
    throw new Error(`${id} ${name}.cells must hold at least one row`);
  }
  validatePackedUint32View(table.counts, `${id} ${name}.counts`);
  if (table.sums) {
    validatePackedView(table.sums, ['uint32x2'], `${id} ${name}.sums`);
  }
  for (const column of ['sumValues', 'minimums', 'maximums'] as const) {
    const view = table[column];
    if (view) {
      validatePackedView(view, ['float32'], `${id} ${name}.${column}`);
    }
  }
  for (const column of ['counts', 'sums', 'sumValues', 'minimums', 'maximums'] as const) {
    const view = table[column];
    if (view && view.length !== capacity) {
      throw new Error(`${id} ${name}.${column} must have the same length as ${name}.cells`);
    }
  }
  for (const column of ['count', 'overflow', 'totalCount'] as const) {
    const view = table[column];
    if (!view) {
      continue;
    }
    validatePackedUint32View(view, `${id} ${name}.${column}`);
    if (view.length < 1) {
      throw new Error(`${id} ${name}.${column} must contain one uint32 row`);
    }
  }
}

/** Validates a fixed-point sum scale. @internal */
export function validateSumScale(id: string, sumScale: number): void {
  if (!Number.isFinite(sumScale) || sumScale <= 0 || Math.fround(sumScale) !== sumScale) {
    throw new Error(`${id} sumScale must be a positive finite float32 value`);
  }
}

/** Per-row source of the reductions: raw rows or the cells of a finer table. @internal */
export type CellTableReductionSource =
  | {
      kind: 'rows';
      /** Per-row values; required for sums and extremes. */
      values?: GraphDataView<'float32'>;
    }
  | {kind: 'cells'; table: GPUCellTable};

/** Properties of {@link getCellTableNodes}. @internal */
export type CellTableNodesProps = {
  id: string;
  operation: string;
  /** Layout of the table's resolution. */
  layout: CellKeyLayout;
  /** Compact keys (low words) per row; the invalid key has bit `layout.width` set. */
  keyLow: GraphDataView<'uint32'>;
  /** High words of the compact keys; present when `layout.width + 1 > 32`. */
  keyHigh?: GraphDataView<'uint32'>;
  /** Row indices `0..n-1`, needed when `sorted` is false. */
  rowIds?: GraphDataView<'uint32'>;
  /** True when rows are already in ascending key order with invalid rows last. */
  sorted: boolean;
  source: CellTableReductionSource;
  sumScale: number;
  output: GPUCellTable;
  /** One-row flags ORed into `output.overflow`. */
  overflowSources?: readonly GraphDataView<'uint32'>[];
  /** Extra one-row destinations of the clamped count. */
  extraCounts?: readonly GraphDataView<'uint32'>[];
};

/** Returns whether compact keys at this layout need a high word. @internal */
export function hasCellKeyHighWord(layout: CellKeyLayout): boolean {
  return layout.width + 1 > 32;
}

/** WGSL statement assigning `isInvalid` for the compact key `(high, low)`. */
function getInvalidKeyExpression(layout: CellKeyLayout, high: string, low: string): string {
  return hasCellKeyHighWord(layout)
    ? `(${high} >> ${layout.width - 32}u) != 0u`
    : `(${low} >> ${layout.width}u) != 0u`;
}

/**
 * Sorts compact keys (unless already sorted), finds cell boundaries, and writes the cell table:
 * keys, counts, fixed-point sums, f32 sums, extremes, count and overflow. Rows accumulate into
 * their table row with integer atomics (64-bit sums as two words with an explicit carry, extremes
 * as order-preserving u32 keys), so every result is independent of thread order. Work is
 * O(rows + capacity).
 *
 * @internal
 */
export function getCellTableNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CellTableNodesProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, layout, output, source} = props;
  const rowCount = props.keyLow.length;
  const capacity = output.cells.length;
  const twoWords = hasCellKeyHighWord(layout);
  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
    ({name, view, type, access: 'read'}) as WGSLKernelBinding;
  const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
    ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
  const nodes: GPUCommandNode<Parameters>[] = [];

  // 1. Ascending (high, low) order by two stable LSD sorts, or one when the key fits a word.
  let sortedLow = props.keyLow;
  let sortedHigh = props.keyHigh;
  let permutation: GraphDataView<'uint32'> | undefined;
  if (!props.sorted) {
    if (!props.rowIds) {
      throw new Error(`${id} needs row IDs to sort`);
    }
    const lowKeyBits = twoWords ? 32 : layout.width + 1;
    const firstKeys = u32('sorted-low-first', rowCount);
    const firstRows = u32('sorted-rows-first', rowCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-low`,
        keys: props.keyLow,
        values: props.rowIds,
        outputKeys: firstKeys,
        outputValues: firstRows,
        keyBits: lowKeyBits
      }).getCommandNodes(graph)
    );
    if (twoWords && props.keyHigh) {
      const highByFirst = u32('high-by-low-order', rowCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-gather-high`,
          operation,
          variant: 'gather-high',
          bindings: [
            read('rows', firstRows),
            read('keyHigh', props.keyHigh),
            write('gathered', highByFirst)
          ],
          invocationCount: rowCount,
          body: `gathered[gatheredOffset + index] = keyHigh[keyHighOffset + rows[rowsOffset + index]];`
        })
      );
      sortedHigh = u32('sorted-high', rowCount);
      permutation = u32('permutation', rowCount);
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort-high`,
          keys: highByFirst,
          values: firstRows,
          outputKeys: sortedHigh,
          outputValues: permutation,
          keyBits: layout.width + 1 - 32
        }).getCommandNodes(graph)
      );
      sortedLow = u32('sorted-low', rowCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-gather-low`,
          operation,
          variant: 'gather-low',
          bindings: [
            read('rows', permutation),
            read('keyLow', props.keyLow),
            write('gathered', sortedLow)
          ],
          invocationCount: rowCount,
          body: `gathered[gatheredOffset + index] = keyLow[keyLowOffset + rows[rowsOffset + index]];`
        })
      );
    } else {
      sortedLow = firstKeys;
      permutation = firstRows;
    }
  }

  const sortedKeyBindings = (): WGSLKernelBinding[] => [
    read('sortedLow', sortedLow),
    ...(twoWords && sortedHigh ? [read('sortedHigh', sortedHigh)] : [])
  ];
  const keyAt = (row: string) =>
    twoWords
      ? `vec2u(sortedHigh[sortedHighOffset + ${row}], sortedLow[sortedLowOffset + ${row}])`
      : `vec2u(0u, sortedLow[sortedLowOffset + ${row}])`;

  // 2. Segment heads and the number of valid rows (valid rows are a prefix of the sorted order).
  const heads = u32('heads', rowCount);
  const validCount = u32('valid-count', 1);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-heads`,
      operation,
      variant: 'heads',
      bindings: [...sortedKeyBindings(), write('heads', heads), write('validCount', validCount)],
      invocationCount: rowCount,
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;
fn isInvalidKey(key: vec2u) -> bool {
  return ${getInvalidKeyExpression(layout, 'key.x', 'key.y')};
}`,
      body: `let key = ${keyAt('index')};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${keyAt('index - 1u')});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${keyAt('index + 1u')}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`
    })
  );

  // 3. Exclusive scan of heads gives each head its cell index.
  const cellIndices = u32('cell-indices', rowCount);
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan`,
      input: heads,
      output: cellIndices,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );

  // 4. Cell row offsets for the first `capacity + 1` cells, the unclamped cell total, and each
  // row's table row (`NO_CELL` for invalid rows and cells past the capacity).
  const offsets = u32('offsets', capacity + 1);
  const cellTotal = u32('cell-total', 1);
  const rowCells = u32('row-cells', rowCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation,
      variant: 'offsets',
      bindings: [
        read('heads', heads),
        read('cellIndices', cellIndices),
        read('validCount', validCount),
        write('offsets', offsets),
        write('cellTotal', cellTotal),
        write('rowCells', rowCells)
      ],
      invocationCount: rowCount,
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const CAPACITY: u32 = ${capacity}u;`,
      body: `let isHead = heads[headsOffset + index] != 0u;
  let cell = cellIndices[cellIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  rowCells[rowCellsOffset + index] = select(0xffffffffu, cell, isValid && cell < CAPACITY);
  if (isHead && cell <= CAPACITY) {
    offsets[offsetsOffset + cell] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = cellIndices[cellIndicesOffset + last] + heads[headsOffset + last];
    cellTotal[cellTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`
    })
  );

  const offsetsDeclarations = `const CAPACITY: u32 = ${capacity}u;
fn getCellCount() -> u32 {
  return min(cellTotal[cellTotalOffset], CAPACITY);
}`;

  // 5. Keys, and counts of raw rows, one invocation per table row.
  const childTable = source.kind === 'cells' ? source.table : undefined;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cells`,
      operation,
      variant: 'cells',
      bindings: [
        read('offsets', offsets),
        read('cellTotal', cellTotal),
        ...sortedKeyBindings(),
        write('cellsOut', output.cells),
        write('countsOut', output.counts)
      ],
      invocationCount: capacity,
      declarations: `${offsetsDeclarations}
${CELL_KEY_WGSL}`,
      body: `if (index >= getCellCount()) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = cellGetKey(${keyAt('begin')}, ${layout.headerHigh}u, ${layout.resolution}u, ${layout.lowBit}u);
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  // Roll-ups sum the child counts in the accumulation pass.
  countsOut[countsOutOffset + index] = ${childTable ? '0u' : `offsets[offsetsOffset + index + 1u] - begin`};`
    })
  );

  const wantsSums = Boolean(output.sums || output.sumValues);
  const wantsExtremes = Boolean(output.minimums || output.maximums);
  const rowValues = source.kind === 'rows' ? source.values : undefined;
  if ((wantsSums || wantsExtremes) && source.kind === 'rows' && !rowValues) {
    throw new Error(`${id} needs values to compute sums and extremes`);
  }
  if (wantsSums && childTable && !childTable.sums) {
    throw new Error(`${id} source table needs sums to roll up sums`);
  }
  if (
    childTable &&
    ((output.minimums && !childTable.minimums) || (output.maximums && !childTable.maximums))
  ) {
    throw new Error(`${id} source table needs the extremes it rolls up`);
  }
  // Integer accumulators: 64-bit sums as two atomic words with an explicit carry, and extremes as
  // order-preserving u32 keys. Integer atomics give the same result in any order.
  const sums = wantsSums
    ? (output.sums ?? createTransientView(graph, `${id}-sums-scratch`, 'uint32x2', capacity))
    : undefined;
  const minimumKeys = output.minimums ? u32('minimum-keys', capacity) : undefined;
  const maximumKeys = output.maximums ? u32('maximum-keys', capacity) : undefined;
  const sourceRow = permutation ? 'permutation[permutationOffset + index]' : 'index';
  const valueBindings = (): WGSLKernelBinding[] =>
    rowValues
      ? [
          ...(permutation ? [read('permutation', permutation)] : []),
          read('values', rowValues, 'f32')
        ]
      : [];

  // 6. Clear the accumulators.
  if (sums || minimumKeys || maximumKeys) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation,
        variant: 'clear',
        bindings: [
          ...(sums ? [write('sums', sums)] : []),
          ...(minimumKeys ? [write('minimumKeys', minimumKeys)] : []),
          ...(maximumKeys ? [write('maximumKeys', maximumKeys)] : [])
        ],
        invocationCount: capacity,
        body: `${sums ? 'sums[sumsOffset + 2u * index] = 0u;\n  sums[sumsOffset + 2u * index + 1u] = 0u;' : ''}
  ${minimumKeys ? 'minimumKeys[minimumKeysOffset + index] = 0xffffffffu;' : ''}
  ${maximumKeys ? 'maximumKeys[maximumKeysOffset + index] = 0u;' : ''}`
      })
    );
  }

  // 7. Accumulate sums (and child counts), then extremes, one invocation per source row.
  if (sums || childTable) {
    const contribution = rowValues
      ? `cellScaleValue(values[valuesOffset + ${sourceRow}])`
      : 'vec2u(childSums[childSumsOffset + 2u * index + 1u], childSums[childSumsOffset + 2u * index])';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-accumulate-sums`,
        operation,
        variant: 'accumulate-sums',
        bindings: [
          read('rowCells', rowCells),
          ...valueBindings(),
          ...(childTable && sums ? [read('childSums', childTable.sums!)] : []),
          ...(childTable
            ? [
                read('childCounts', childTable.counts),
                {
                  name: 'countsOut',
                  view: output.counts,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(sums
            ? [{name: 'sums', view: sums, type: 'atomic<u32>', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: rowCount,
        declarations: getFixedPointWGSL(props.sumScale),
        body: `let cell = rowCells[rowCellsOffset + index];
  if (cell == 0xffffffffu) {
    return;
  }
  ${childTable ? 'atomicAdd(&countsOut[countsOutOffset + cell], childCounts[childCountsOffset + index]);' : ''}
  ${
    sums
      ? `let value = ${contribution};
  let previous = atomicAdd(&sums[sumsOffset + 2u * cell], value.y);
  let high = value.x + select(0u, 1u, previous + value.y < previous);
  if (high != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * cell + 1u], high);
  }`
      : ''
  }`
      })
    );
  }
  if (minimumKeys || maximumKeys) {
    const minimumSource = rowValues
      ? `values[valuesOffset + ${sourceRow}]`
      : 'childMinimums[childMinimumsOffset + index]';
    const maximumSource = rowValues
      ? `values[valuesOffset + ${sourceRow}]`
      : 'childMaximums[childMaximumsOffset + index]';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-accumulate-extremes`,
        operation,
        variant: 'accumulate-extremes',
        bindings: [
          read('rowCells', rowCells),
          ...valueBindings(),
          ...(childTable && minimumKeys
            ? [read('childMinimums', childTable.minimums!, 'f32')]
            : []),
          ...(childTable && maximumKeys
            ? [read('childMaximums', childTable.maximums!, 'f32')]
            : []),
          ...(minimumKeys
            ? [
                {
                  name: 'minimumKeys',
                  view: minimumKeys,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(maximumKeys
            ? [
                {
                  name: 'maximumKeys',
                  view: maximumKeys,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: rowCount,
        declarations: ORDERED_KEY_WGSL,
        body: `let cell = rowCells[rowCellsOffset + index];
  if (cell == 0xffffffffu) {
    return;
  }
  ${minimumKeys ? `atomicMin(&minimumKeys[minimumKeysOffset + cell], getOrderedKey(${minimumSource}));` : ''}
  ${maximumKeys ? `atomicMax(&maximumKeys[maximumKeysOffset + cell], getOrderedKey(${maximumSource}));` : ''}`
      })
    );
  }

  // 8. Finish: f32 sums and decoded extremes (NaN for empty rows).
  if (output.sumValues || minimumKeys || maximumKeys) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation,
        variant: 'finish',
        bindings: [
          read('cellTotal', cellTotal),
          ...(output.sumValues && sums ? [read('sums', sums)] : []),
          ...(minimumKeys ? [read('minimumKeys', minimumKeys)] : []),
          ...(maximumKeys ? [read('maximumKeys', maximumKeys)] : []),
          ...(output.sumValues ? [write('sumValuesOut', output.sumValues, 'f32')] : []),
          ...(output.minimums ? [write('minimumsOut', output.minimums, 'f32')] : []),
          ...(output.maximums ? [write('maximumsOut', output.maximums, 'f32')] : [])
        ],
        invocationCount: capacity,
        declarations: `${offsetsDeclarations}
${ORDERED_KEY_WGSL}
${getFixedPointWGSL(props.sumScale)}`,
        body: `let isOccupied = index < getCellCount();
  let nan = getNaN();
  ${
    output.sumValues
      ? `let total = vec2u(sums[sumsOffset + 2u * index + 1u], sums[sumsOffset + 2u * index]);
  sumValuesOut[sumValuesOutOffset + index] = cellI64ToF32(total) / SUM_SCALE;`
      : ''
  }
  ${
    output.minimums
      ? 'minimumsOut[minimumsOutOffset + index] = select(nan, decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), isOccupied);'
      : ''
  }
  ${
    output.maximums
      ? 'maximumsOut[maximumsOutOffset + index] = select(nan, decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), isOccupied);'
      : ''
  }`
      })
    );
  }

  // 9. Clamped count, overflow and total.
  nodes.push(
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation,
      totalCount: cellTotal,
      output: {
        ids: output.counts,
        count: output.count,
        overflow: output.overflow,
        totalCount: output.totalCount
      },
      overflowSources: props.overflowSources,
      extraCounts: props.extraCounts
    })
  );
  return nodes;
}

/**
 * Order-preserving u32 encoding of f32 (`-0` below `+0`) and a NaN builder.
 *
 * @internal
 */
export const ORDERED_KEY_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`;

/** Largest f32 below 2^62: scaled values saturate here so 64-bit sums of 2^31 rows cannot wrap far. */
export const CELL_MAXIMUM_SCALED_VALUE = Math.fround(2 ** 62 - 2 ** 38);

/** Signed 64-bit fixed-point helpers on `vec2u(high, low)`. @internal */
export function getFixedPointWGSL(sumScale: number): string {
  return /* wgsl */ `
const SUM_SCALE: f32 = ${getWGSLFloatLiteral(sumScale)};
const MAXIMUM_SCALED_VALUE: f32 = ${getWGSLFloatLiteral(CELL_MAXIMUM_SCALED_VALUE)};

fn cellAddI64(left: vec2u, right: vec2u) -> vec2u {
  let low = left.y + right.y;
  return vec2u(left.x + right.x + select(0u, 1u, low < left.y), low);
}

fn cellNegateI64(value: vec2u) -> vec2u {
  let low = ~value.y + 1u;
  return vec2u(~value.x + select(0u, 1u, low == 0u), low);
}

/** roundHalfEven(value * SUM_SCALE), saturated, as a two's-complement 64-bit integer. */
fn cellScaleValue(value: f32) -> vec2u {
  let scaled = clamp(round(value * SUM_SCALE), -MAXIMUM_SCALED_VALUE, MAXIMUM_SCALED_VALUE);
  let magnitude = abs(scaled);
  let highFloat = floor(magnitude * 2.3283064365386963e-10);
  let lowFloat = magnitude - highFloat * 4294967296.0;
  let result = vec2u(u32(highFloat), u32(lowFloat));
  return select(result, cellNegateI64(result), scaled < 0.0);
}

fn cellI64ToF32(value: vec2u) -> f32 {
  let isNegative = (value.x & 0x80000000u) != 0u;
  let magnitude = select(value, cellNegateI64(value), isNegative);
  let result = f32(magnitude.x) * 4294967296.0 + f32(magnitude.y);
  return select(result, -result, isNegative);
}
`;
}
