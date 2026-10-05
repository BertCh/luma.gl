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
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  getFixedPointWGSL,
  GPU_CELL_DEFAULT_SUM_SCALE,
  ORDERED_KEY_WGSL,
  validateSumScale
} from '../../geospatial/cell-aggregation/cell-table';

const OPERATION = 'GPUKeyJoin';
const MAXIMUM_GATHER_COLUMNS_PER_KERNEL = 3;

/** Join kind: `'left'` keeps every left row aligned, `'inner'` also compacts the matched rows. */
export type GPUKeyJoinKind = 'left' | 'inner';

/** Operation of a 1:n {@link GPUKeyJoinAggregate} over all right rows matching a left row's key. */
export type GPUKeyJoinAggregateOperation = 'count' | 'sum' | 'mean' | 'minimum' | 'maximum';

/** One 1:1 gathered right column: the first (smallest-row) matching right row's value. */
export type GPUKeyJoinGather = {
  /** Right-aligned source column. */
  column: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /**
   * Left-aligned destination of the same format. Unmatched rows receive NaN (`float32`) or
   * `0xffffffff` (`uint32`). The value is copied bit for bit.
   */
  output: GraphDataView<'float32'> | GraphDataView<'uint32'>;
};

/** One 1:n aggregate over every right row that matches a left row's key. */
export type GPUKeyJoinAggregate = {
  /**
   * Right-aligned values. Required except for `'count'`. Rows with a non-finite value are
   * excluded from this aggregate only.
   */
  column?: GraphDataView<'float32'>;
  /**
   * `'count'` counts matching right rows (finite values when a column is given), `'sum'` and
   * `'mean'` use exact fixed-point sums, `'minimum'` and `'maximum'` are exact. `'mean'` is the
   * decoded fixed-point sum divided by the finite count.
   */
  operation: GPUKeyJoinAggregateOperation;
  /**
   * Left-aligned destination. Unmatched rows receive 0 for `'count'` and NaN otherwise; a matched
   * row without a finite value receives NaN for `'mean'`, `'minimum'` and `'maximum'`, and 0 for
   * `'sum'`.
   */
  output: GraphDataView<'float32'>;
  /**
   * Optional left-aligned exact fixed-point sums (`'sum'` and `'mean'` only), signed 64-bit
   * little-endian words equal to the sum of `roundHalfEven(fround(value * sumScale))`. Zero for
   * unmatched rows.
   */
  sums?: GraphDataView<'uint32x2'>;
};

/** Outputs of {@link GPUKeyJoin}. Left-aligned unless noted. */
export type GPUKeyJoinOutput = {
  /** Smallest matching right row per left row, `0xffffffff` when unmatched. */
  rightRows?: GraphDataView<'uint32'>;
  /** Number of right rows matching each left row's key. */
  matchCounts?: GraphDataView<'uint32'>;
  /** 1 when the left row matched, else 0; usable as a crossfilter live mask. */
  matched?: GraphDataView<'uint32'>;
  /** Right-aligned: 1 when the right row's key is matched by some valid left row. */
  rightMatched?: GraphDataView<'uint32'>;
  /** `'inner'` only (required there): ascending matched left row IDs, capacity bounded. */
  rows?: GPUCompactOutput;
};

/**
 * Properties for {@link GPUKeyJoin}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `kind`, `sumScale`, view lengths, key format, and which optional views and aggregates exist.
 */
export type GPUKeyJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'key-join'`. */
  id?: string;
  /**
   * Left keys: `uint32` or 64-bit `uint32x2` little-endian `(low, high)` words. The all-ones key
   * (`0xffffffff`, or both words `0xffffffff`) means "no key" and never matches.
   */
  leftKeys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
  /** Right keys, in the same format as `leftKeys`. */
  rightKeys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
  /** Optional left mask; zero rows never match. */
  leftMask?: GraphDataView<'uint32'>;
  /** Optional right mask; zero rows are ignored by every output. */
  rightMask?: GraphDataView<'uint32'>;
  /** Join kind. Defaults to `'left'`. */
  kind?: GPUKeyJoinKind;
  /** 1:1 gathered right columns (first matching right row wins). */
  gather?: readonly GPUKeyJoinGather[];
  /** 1:n aggregates over all matching right rows. */
  aggregates?: readonly GPUKeyJoinAggregate[];
  /** Fixed-point scale of sums and means, a positive float32. Defaults to 65536. */
  sumScale?: number;
  /** Caller-owned outputs. */
  output: GPUKeyJoinOutput;
};

/**
 * Attribute join of a left table onto a right table by key, entirely on the GPU.
 *
 * Valid right rows (mask set, key not all-ones) are sorted by key with a stable radix sort
 * (one pass per key word), so equal keys stay in ascending right-row order. Segment heads and a
 * scan give the unique right keys and their row ranges. Each left row binary-searches the unique
 * keys on the unsigned `(high, low)` order, then gathers the first matching right row's columns
 * and 1:n aggregates. Aggregates accumulate once per unique key with integer atomics (64-bit
 * fixed-point sums with explicit carry, ordered-key extremes), so every output is exact,
 * independent of thread order, and bitwise reproducible. All inputs must be single packed views.
 * Left-aligned outputs keep the left row order for both kinds; `'inner'` additionally compacts the
 * matched left row IDs in ascending order.
 */
export class GPUKeyJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKeyJoinProps;
  /** Join kind. */
  readonly kind: GPUKeyJoinKind;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;

  constructor(props: GPUKeyJoinProps) {
    this.id = props.id ?? 'key-join';
    this.props = props;
    this.kind = props.kind ?? 'left';
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    const id = this.id;
    validateSumScale(id, this.sumScale);
    if (this.kind !== 'left' && this.kind !== 'inner') {
      throw new Error(`${id} kind must be 'left' or 'inner'`);
    }
    const {output} = props;
    const gather = props.gather ?? [];
    const aggregates = props.aggregates ?? [];
    const allViews: [string, unknown][] = [
      ['leftKeys', props.leftKeys],
      ['rightKeys', props.rightKeys],
      ['leftMask', props.leftMask],
      ['rightMask', props.rightMask],
      ...gather.flatMap((entry, index): [string, unknown][] => [
        [`gather[${index}].column`, entry.column],
        [`gather[${index}].output`, entry.output]
      ]),
      ...aggregates.flatMap((entry, index): [string, unknown][] => [
        [`aggregates[${index}].column`, entry.column],
        [`aggregates[${index}].output`, entry.output],
        [`aggregates[${index}].sums`, entry.sums]
      ]),
      ['output.rightRows', output.rightRows],
      ['output.matchCounts', output.matchCounts],
      ['output.matched', output.matched],
      ['output.rightMatched', output.rightMatched]
    ];
    for (const [name, view] of allViews) {
      if (view instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const keyFormat = props.leftKeys.format;
    if (keyFormat !== 'uint32' && keyFormat !== 'uint32x2') {
      throw new Error(`${id} leftKeys must be uint32 or uint32x2`);
    }
    validatePackedView(props.leftKeys, [keyFormat], `${id} leftKeys`);
    validatePackedView(props.rightKeys, [keyFormat], `${id} rightKeys`);
    const leftCount = props.leftKeys.length;
    const rightCount = props.rightKeys.length;
    if (leftCount < 1 || rightCount < 1) {
      throw new Error(`${id} needs at least one left row and one right row`);
    }
    for (const [name, view, count] of [
      ['leftMask', props.leftMask, leftCount],
      ['rightMask', props.rightMask, rightCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== count) {
          throw new Error(`${id} ${name} length must equal the ${name.slice(0, -4)} row count`);
        }
      }
    }
    for (const [index, entry] of gather.entries()) {
      const name = `gather[${index}]`;
      const format = entry.column.format;
      validatePackedView(entry.column, ['float32', 'uint32'], `${id} ${name}.column`);
      validatePackedView(entry.output, [format], `${id} ${name}.output`);
      if (entry.column.length !== rightCount) {
        throw new Error(`${id} ${name}.column length must equal the right row count`);
      }
      if (entry.output.length !== leftCount) {
        throw new Error(`${id} ${name}.output length must equal the left row count`);
      }
    }
    for (const [index, entry] of aggregates.entries()) {
      const name = `aggregates[${index}]`;
      if (!['count', 'sum', 'mean', 'minimum', 'maximum'].includes(entry.operation)) {
        throw new Error(`${id} ${name}.operation is not supported`);
      }
      if (entry.operation !== 'count' && !entry.column) {
        throw new Error(`${id} ${name} needs a column for ${entry.operation}`);
      }
      if (entry.column) {
        validatePackedView(entry.column, ['float32'], `${id} ${name}.column`);
        if (entry.column.length !== rightCount) {
          throw new Error(`${id} ${name}.column length must equal the right row count`);
        }
      }
      validatePackedView(entry.output, ['float32'], `${id} ${name}.output`);
      if (entry.output.length !== leftCount) {
        throw new Error(`${id} ${name}.output length must equal the left row count`);
      }
      if (entry.sums) {
        if (entry.operation !== 'sum' && entry.operation !== 'mean') {
          throw new Error(`${id} ${name}.sums needs the sum or mean operation`);
        }
        validatePackedView(entry.sums, ['uint32x2'], `${id} ${name}.sums`);
        if (entry.sums.length !== leftCount) {
          throw new Error(`${id} ${name}.sums length must equal the left row count`);
        }
      }
    }
    for (const name of ['rightRows', 'matchCounts', 'matched'] as const) {
      const view = output[name];
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== leftCount) {
          throw new Error(`${id} output.${name} length must equal the left row count`);
        }
      }
    }
    if (output.rightMatched) {
      validatePackedUint32View(output.rightMatched, `${id} output.rightMatched`);
      if (output.rightMatched.length !== rightCount) {
        throw new Error(`${id} output.rightMatched length must equal the right row count`);
      }
    }
    if (this.kind === 'inner') {
      if (!output.rows) {
        throw new Error(`${id} output.rows is required for the inner join`);
      }
      validateCompactOutput(id, output.rows);
      if (output.rows.ids.length < 1) {
        throw new Error(`${id} output.rows.ids must hold at least one row`);
      }
    } else if (output.rows) {
      throw new Error(`${id} output.rows is only supported for the inner join`);
    }
    const inputs = [
      props.leftKeys,
      props.rightKeys,
      props.leftMask,
      props.rightMask,
      ...gather.map(entry => entry.column),
      ...aggregates.map(entry => entry.column)
    ];
    const outputs = [
      output.rightRows,
      output.matchCounts,
      output.matched,
      output.rightMatched,
      output.rows?.ids,
      output.rows?.count,
      output.rows?.overflow,
      output.rows?.totalCount,
      ...gather.map(entry => entry.output),
      ...aggregates.flatMap(entry => [entry.output, entry.sums])
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  }

  /** Returns build, probe, gather, aggregate, right-matched and compaction nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, kind} = this;
    const output = props.output;
    const gather = props.gather ?? [];
    const aggregates = props.aggregates ?? [];
    validateGraphViewsBelongToGraph(id, graph, [
      props.leftKeys,
      props.rightKeys,
      props.leftMask,
      props.rightMask,
      ...gather.flatMap(entry => [entry.column, entry.output]),
      ...aggregates.flatMap(entry => [entry.column, entry.output, entry.sums]),
      output.rightRows,
      output.matchCounts,
      output.matched,
      output.rightMatched,
      output.rows?.ids,
      output.rows?.count,
      output.rows?.overflow,
      output.rows?.totalCount
    ]);
    const twoWords = props.leftKeys.format === 'uint32x2';
    const leftCount = props.leftKeys.length;
    const rightCount = props.rightKeys.length;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const atomic = (name: string, view: GraphDataView) =>
      ({
        name,
        view,
        type: 'atomic<u32>',
        access: 'read_write'
      }) as WGSLKernelBinding;
    const node = (
      step: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount,
        declarations,
        body
      });
    const nodes: GPUCommandNode<Parameters>[] = [];
    const keyDeclarations = /* wgsl */ `
const NO_KEY: u32 = 0xffffffffu;
fn isNoKey(key: vec2u) -> bool {
  return key.y == NO_KEY && ${twoWords ? 'key.x == NO_KEY' : 'true'};
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`;
    // Keys as vec2u(high, low) from an input key view.
    const readKey = (name: string, row: string) =>
      twoWords
        ? `vec2u(${name}[${name}Offset + 2u * (${row}) + 1u], ${name}[${name}Offset + 2u * (${row})])`
        : `vec2u(0u, ${name}[${name}Offset + (${row})])`;

    // 1. Right keys: invalid rows (mask or reserved key) become the all-ones key and sort last.
    const keyLow = u32('key-low', rightCount);
    const keyHigh = twoWords ? u32('key-high', rightCount) : undefined;
    const rowIds = u32('row-ids', rightCount);
    nodes.push(
      node(
        'right-keys',
        [
          read('rightKeys', props.rightKeys),
          ...(props.rightMask ? [read('rightMask', props.rightMask)] : []),
          write('keyLow', keyLow),
          ...(keyHigh ? [write('keyHigh', keyHigh)] : []),
          write('rowIds', rowIds)
        ],
        rightCount,
        `var key = ${readKey('rightKeys', 'index')};
  ${props.rightMask ? 'if (rightMask[rightMaskOffset + index] == 0u) {\n    key = vec2u(NO_KEY, NO_KEY);\n  }' : ''}
  if (isNoKey(key)) {
    key = vec2u(NO_KEY, NO_KEY);
  }
  keyLow[keyLowOffset + index] = key.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + index] = key.x;' : ''}
  rowIds[rowIdsOffset + index] = index;`,
        keyDeclarations
      )
    );

    // 2. Stable ascending (high, low) order: low-word sort, then high-word sort of that order.
    const firstKeys = u32('sorted-low-first', rightCount);
    const firstRows = u32('sorted-rows-first', rightCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-low`,
        keys: keyLow,
        values: rowIds,
        outputKeys: firstKeys,
        outputValues: firstRows,
        keyBits: 32
      }).getCommandNodes(graph)
    );
    let sortedLow = firstKeys;
    let sortedHigh: GraphDataView<'uint32'> | undefined;
    let sortedRows = firstRows;
    if (twoWords && keyHigh) {
      const highByFirst = u32('high-by-low-order', rightCount);
      nodes.push(
        node(
          'gather-high',
          [read('rows', firstRows), read('keyHigh', keyHigh), write('gathered', highByFirst)],
          rightCount,
          'gathered[gatheredOffset + index] = keyHigh[keyHighOffset + rows[rowsOffset + index]];'
        )
      );
      sortedHigh = u32('sorted-high', rightCount);
      sortedRows = u32('sorted-rows', rightCount);
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort-high`,
          keys: highByFirst,
          values: firstRows,
          outputKeys: sortedHigh,
          outputValues: sortedRows,
          keyBits: 32
        }).getCommandNodes(graph)
      );
      sortedLow = u32('sorted-low', rightCount);
      nodes.push(
        node(
          'gather-low',
          [read('rows', sortedRows), read('keyLow', keyLow), write('gathered', sortedLow)],
          rightCount,
          'gathered[gatheredOffset + index] = keyLow[keyLowOffset + rows[rowsOffset + index]];'
        )
      );
    }
    const sortedKeyBindings = (): WGSLKernelBinding[] => [
      read('sortedLow', sortedLow),
      ...(sortedHigh ? [read('sortedHigh', sortedHigh)] : [])
    ];
    const sortedKeyAt = (row: string) =>
      twoWords
        ? `vec2u(sortedHigh[sortedHighOffset + ${row}], sortedLow[sortedLowOffset + ${row}])`
        : `vec2u(0u, sortedLow[sortedLowOffset + ${row}])`;

    // 3. Segment heads of valid rows (a prefix of the sorted order) and the valid row count.
    const heads = u32('heads', rightCount);
    const validCount = u32('valid-count', 1);
    nodes.push(
      node(
        'heads',
        [...sortedKeyBindings(), write('heads', heads), write('validCount', validCount)],
        rightCount,
        `let key = ${sortedKeyAt('index')};
  let isValid = !isNoKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${sortedKeyAt('index - 1u')});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isNoKey(${sortedKeyAt('index + 1u')}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`,
        `const ROW_COUNT: u32 = ${rightCount}u;\n${keyDeclarations}`
      )
    );

    // 4. Exclusive scan of heads gives every head its unique-key index.
    const uniqueIndices = u32('unique-indices', rightCount);
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: heads,
        output: uniqueIndices,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 5. Unique keys with the first sorted position of each; `uniqueStart[U]` is the valid count.
    const uniqueLow = u32('unique-low', rightCount);
    const uniqueHigh = twoWords ? u32('unique-high', rightCount) : undefined;
    const uniqueStart = u32('unique-start', rightCount + 1);
    nodes.push(
      node(
        'unique-keys',
        [
          ...sortedKeyBindings(),
          read('heads', heads),
          read('uniqueIndices', uniqueIndices),
          read('validCount', validCount),
          write('uniqueLow', uniqueLow),
          ...(uniqueHigh ? [write('uniqueHigh', uniqueHigh)] : []),
          write('uniqueStart', uniqueStart)
        ],
        rightCount,
        `if (heads[headsOffset + index] != 0u) {
    let unique = uniqueIndices[uniqueIndicesOffset + index];
    let key = ${sortedKeyAt('index')};
    uniqueLow[uniqueLowOffset + unique] = key.y;
    ${uniqueHigh ? 'uniqueHigh[uniqueHighOffset + unique] = key.x;' : ''}
    uniqueStart[uniqueStartOffset + unique] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let uniqueTotal = uniqueIndices[uniqueIndicesOffset + last] + heads[headsOffset + last];
    uniqueStart[uniqueStartOffset + uniqueTotal] = validCount[validCountOffset];
  }`,
        `const ROW_COUNT: u32 = ${rightCount}u;`
      )
    );
    // Unique-key total and the unique index of every sorted row (NO_UNIQUE for invalid rows).
    const uniqueTotal = u32('unique-total', 1);
    const rowUnique = u32('row-unique', rightCount);
    nodes.push(
      node(
        'row-unique',
        [
          read('heads', heads),
          read('uniqueIndices', uniqueIndices),
          read('validCount', validCount),
          write('rowUnique', rowUnique),
          write('uniqueTotal', uniqueTotal)
        ],
        rightCount,
        `let isValid = index < validCount[validCountOffset];
  rowUnique[rowUniqueOffset + index] = select(
    0xffffffffu,
    uniqueIndices[uniqueIndicesOffset + index] + heads[headsOffset + index] - 1u,
    isValid
  );
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    uniqueTotal[uniqueTotalOffset] =
      uniqueIndices[uniqueIndicesOffset + last] + heads[headsOffset + last];
  }`,
        `const ROW_COUNT: u32 = ${rightCount}u;`
      )
    );

    // 6. Probe: binary search of every left key over the unique keys.
    const leftUnique = u32('left-unique', leftCount);
    const uniqueMatched = output.rightMatched ? u32('unique-matched', rightCount) : undefined;
    if (uniqueMatched) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-unique-matched`,
          operation: OPERATION,
          view: uniqueMatched,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
      node(
        'probe',
        [
          read('leftKeys', props.leftKeys),
          ...(props.leftMask ? [read('leftMask', props.leftMask)] : []),
          read('uniqueLow', uniqueLow),
          ...(uniqueHigh ? [read('uniqueHigh', uniqueHigh)] : []),
          read('uniqueTotal', uniqueTotal),
          write('leftUnique', leftUnique),
          ...(uniqueMatched ? [atomic('uniqueMatched', uniqueMatched)] : [])
        ],
        leftCount,
        `var key = ${readKey('leftKeys', 'index')};
  ${props.leftMask ? 'if (leftMask[leftMaskOffset + index] == 0u) {\n    key = vec2u(NO_KEY, NO_KEY);\n  }' : ''}
  var found = 0xffffffffu;
  if (!isNoKey(key)) {
    let total = uniqueTotal[uniqueTotalOffset];
    var low = 0u;
    var high = total;
    while (low < high) {
      let middle = (low + high) >> 1u;
      let candidate = ${
        twoWords
          ? 'vec2u(uniqueHigh[uniqueHighOffset + middle], uniqueLow[uniqueLowOffset + middle])'
          : 'vec2u(0u, uniqueLow[uniqueLowOffset + middle])'
      };
      if (isKeyLess(candidate, key)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    if (low < total) {
      let candidate = ${
        twoWords
          ? 'vec2u(uniqueHigh[uniqueHighOffset + low], uniqueLow[uniqueLowOffset + low])'
          : 'vec2u(0u, uniqueLow[uniqueLowOffset + low])'
      };
      if (all(candidate == key)) {
        found = low;
      }
    }
  }
  leftUnique[leftUniqueOffset + index] = found;
  ${uniqueMatched ? 'if (found != 0xffffffffu) {\n    atomicMax(&uniqueMatched[uniqueMatchedOffset + found], 1u);\n  }' : ''}`,
        keyDeclarations
      )
    );

    // 7. Left-aligned flags, first rows and match counts.
    const needsMatched = Boolean(output.matched) || kind === 'inner';
    const needsFirst = Boolean(output.rightRows) || gather.length > 0;
    const matchedView = output.matched ?? (needsMatched ? u32('matched', leftCount) : undefined);
    const firstView = output.rightRows ?? (needsFirst ? u32('first-rows', leftCount) : undefined);
    if (matchedView || firstView || output.matchCounts) {
      nodes.push(
        node(
          'left-rows',
          [
            read('leftUnique', leftUnique),
            read('uniqueStart', uniqueStart),
            ...(firstView ? [read('sortedRows', sortedRows)] : []),
            ...(matchedView ? [write('matchedOut', matchedView)] : []),
            ...(firstView ? [write('firstOut', firstView)] : []),
            ...(output.matchCounts ? [write('countsOut', output.matchCounts)] : [])
          ],
          leftCount,
          `let unique = leftUnique[leftUniqueOffset + index];
  let isMatched = unique != 0xffffffffu;
  ${matchedView ? 'matchedOut[matchedOutOffset + index] = select(0u, 1u, isMatched);' : ''}
  var begin = 0u;
  var end = 0u;
  if (isMatched) {
    begin = uniqueStart[uniqueStartOffset + unique];
    end = uniqueStart[uniqueStartOffset + unique + 1u];
  }
  ${firstView ? 'firstOut[firstOutOffset + index] = select(0xffffffffu, sortedRows[sortedRowsOffset + begin], isMatched);' : ''}
  ${output.matchCounts ? 'countsOut[countsOutOffset + index] = end - begin;' : ''}`
        )
      );
    }

    // 8. 1:1 gathers, at most three columns per kernel (bit copies; NaN or all-ones if unmatched).
    for (let first = 0; first < gather.length; first += MAXIMUM_GATHER_COLUMNS_PER_KERNEL) {
      const group = gather.slice(first, first + MAXIMUM_GATHER_COLUMNS_PER_KERNEL);
      nodes.push(
        node(
          `gather-${first}`,
          [
            read('firstRows', firstView!),
            ...group.flatMap((entry, index) => [
              read(`column${index}`, entry.column),
              write(`out${index}`, entry.output)
            ])
          ],
          leftCount,
          `let row = firstRows[firstRowsOffset + index];
  let isMatched = row != 0xffffffffu;
  ${group
    .map(
      (entry, index) =>
        `out${index}[out${index}Offset + index] = select(${entry.column.format === 'float32' ? '0x7fc00000u' : '0xffffffffu'}, column${index}[column${index}Offset + select(0u, row, isMatched)], isMatched);`
    )
    .join('\n  ')}`
        )
      );
    }

    // 9. 1:n aggregates: per-unique integer accumulation, then a left-aligned finish.
    for (const [index, aggregate] of aggregates.entries()) {
      const name = `aggregate-${index}`;
      const {operation, column} = aggregate;
      const wantsSums = operation === 'sum' || operation === 'mean';
      const wantsCounts = operation === 'mean' || (operation === 'count' && Boolean(column));
      const isMinimum = operation === 'minimum';
      const isMaximum = operation === 'maximum';
      const sums = wantsSums
        ? createTransientView(graph, `${id}-${name}-sums`, 'uint32x2', rightCount)
        : undefined;
      const counts = wantsCounts ? u32(`${name}-counts`, rightCount) : undefined;
      const extremes = isMinimum || isMaximum ? u32(`${name}-extremes`, rightCount) : undefined;
      if (sums || counts || extremes) {
        nodes.push(
          node(
            `${name}-clear`,
            [
              ...(sums ? [write('sums', sums)] : []),
              ...(counts ? [write('counts', counts)] : []),
              ...(extremes ? [write('extremes', extremes)] : [])
            ],
            rightCount,
            `${sums ? 'sums[sumsOffset + 2u * index] = 0u;\n  sums[sumsOffset + 2u * index + 1u] = 0u;' : ''}
  ${counts ? 'counts[countsOffset + index] = 0u;' : ''}
  ${extremes ? `extremes[extremesOffset + index] = ${isMinimum ? '0xffffffffu' : '0u'};` : ''}`
          )
        );
        nodes.push(
          node(
            `${name}-accumulate`,
            [
              read('sortedRows', sortedRows),
              read('rowUnique', rowUnique),
              read('values', column!, 'f32'),
              ...(sums ? [atomic('sums', sums)] : []),
              ...(counts ? [atomic('counts', counts)] : []),
              ...(extremes ? [atomic('extremes', extremes)] : [])
            ],
            rightCount,
            `let unique = rowUnique[rowUniqueOffset + index];
  if (unique == 0xffffffffu) {
    return;
  }
  let value = values[valuesOffset + sortedRows[sortedRowsOffset + index]];
  if ((bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u) {
    return;
  }
  ${counts ? 'atomicAdd(&counts[countsOffset + unique], 1u);' : ''}
  ${
    sums
      ? `let scaled = cellScaleValue(value);
  let previous = atomicAdd(&sums[sumsOffset + 2u * unique], scaled.y);
  let carried = scaled.x + select(0u, 1u, previous + scaled.y < previous);
  if (carried != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * unique + 1u], carried);
  }`
      : ''
  }
  ${isMinimum ? 'atomicMin(&extremes[extremesOffset + unique], getOrderedKey(value));' : ''}
  ${isMaximum ? 'atomicMax(&extremes[extremesOffset + unique], getOrderedKey(value));' : ''}`,
            `${getFixedPointWGSL(this.sumScale)}\n${ORDERED_KEY_WGSL}`
          )
        );
      }
      let finishBody: string;
      if (operation === 'count') {
        finishBody = counts
          ? 'let value = f32(counts[countsOffset + unique]);'
          : 'let value = f32(uniqueStart[uniqueStartOffset + unique + 1u] - uniqueStart[uniqueStartOffset + unique]);';
      } else if (operation === 'sum') {
        finishBody = 'let value = cellI64ToF32(total) / SUM_SCALE;';
      } else if (operation === 'mean') {
        finishBody = `let n = counts[countsOffset + unique];
    let value = select(getNaN(), cellI64ToF32(total) / SUM_SCALE / f32(n), n != 0u);`;
      } else {
        finishBody = `let key = extremes[extremesOffset + unique];
    let value = select(decodeOrderedKey(key), getNaN(), key == ${isMinimum ? '0xffffffffu' : '0u'});`;
      }
      const needsUniqueStart = operation === 'count' && !counts;
      nodes.push(
        node(
          `${name}-finish`,
          [
            read('leftUnique', leftUnique),
            ...(needsUniqueStart ? [read('uniqueStart', uniqueStart)] : []),
            ...(sums ? [read('sums', sums)] : []),
            ...(counts ? [read('counts', counts)] : []),
            ...(extremes ? [read('extremes', extremes)] : []),
            write('valueOut', aggregate.output, 'f32'),
            ...(aggregate.sums ? [write('sumsOut', aggregate.sums)] : [])
          ],
          leftCount,
          `let unique = leftUnique[leftUniqueOffset + index];
  if (unique == 0xffffffffu) {
    valueOut[valueOutOffset + index] = ${operation === 'count' ? '0.0' : 'getNaN()'};
    ${aggregate.sums ? 'sumsOut[sumsOutOffset + 2u * index] = 0u;\n    sumsOut[sumsOutOffset + 2u * index + 1u] = 0u;' : ''}
    return;
  }
  ${sums ? 'let total = vec2u(sums[sumsOffset + 2u * unique + 1u], sums[sumsOffset + 2u * unique]);' : ''}
  ${finishBody}
  valueOut[valueOutOffset + index] = value;
  ${aggregate.sums ? 'sumsOut[sumsOutOffset + 2u * index] = total.y;\n  sumsOut[sumsOutOffset + 2u * index + 1u] = total.x;' : ''}`,
          `${getFixedPointWGSL(this.sumScale)}\n${ORDERED_KEY_WGSL}`
        )
      );
    }

    // 10. Right-aligned matched flags: every sorted position writes its source row once.
    if (output.rightMatched && uniqueMatched) {
      nodes.push(
        node(
          'right-matched',
          [
            read('sortedRows', sortedRows),
            read('rowUnique', rowUnique),
            read('uniqueMatched', uniqueMatched),
            write('rightMatchedOut', output.rightMatched)
          ],
          rightCount,
          `let unique = rowUnique[rowUniqueOffset + index];
  let isMatched = unique != 0xffffffffu && uniqueMatched[uniqueMatchedOffset + unique] != 0u;
  rightMatchedOut[rightMatchedOutOffset + sortedRows[sortedRowsOffset + index]] =
    select(0u, 1u, isMatched);`
        )
      );
    }

    // 11. Inner join: ascending matched left row IDs through a scan, bounded by the capacity.
    if (kind === 'inner' && output.rows && matchedView) {
      const positions = u32('positions', leftCount);
      const total = u32('matched-total', 1);
      const capacity = output.rows.ids.length;
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan-matched`,
          input: matchedView,
          output: positions,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
      nodes.push(
        node(
          'compact',
          [
            read('matched', matchedView),
            read('positions', positions),
            write('idsOut', output.rows.ids),
            write('totalOut', total)
          ],
          Math.max(leftCount, capacity),
          `let last = LEFT_COUNT - 1u;
  let total = positions[positionsOffset + last] + matched[matchedOffset + last];
  if (index == 0u) {
    totalOut[totalOutOffset] = total;
  }
  if (index < LEFT_COUNT && matched[matchedOffset + index] != 0u) {
    let position = positions[positionsOffset + index];
    if (position < CAPACITY) {
      idsOut[idsOutOffset + position] = index;
    }
  }
  if (index < CAPACITY && index >= total) {
    idsOut[idsOutOffset + index] = 0xffffffffu;
  }`,
          `const LEFT_COUNT: u32 = ${leftCount}u;\nconst CAPACITY: u32 = ${capacity}u;`
        )
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          totalCount: total,
          output: output.rows
        })
      );
    }
    return nodes;
  }
}
