// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getCellTableViews,
  getFixedPointWGSL,
  GPU_CELL_DEFAULT_SUM_SCALE,
  ORDERED_KEY_WGSL,
  validateCellTable,
  validateSumScale,
  type GPUCellTable
} from '../cell-aggregation/cell-table';

const OPERATION = 'GPUCellTableCompare';
const STATS_WORKGROUP_SIZE = 256;

/** `presence` bit set when the cell occurs in the `before` table. */
export const GPU_CELL_COMPARE_PRESENT_BEFORE = 1;

/** `presence` bit set when the cell occurs in the `after` table. */
export const GPU_CELL_COMPARE_PRESENT_AFTER = 2;

/** Cell table column that {@link GPUCellTableCompare} compares. */
export type GPUCellTableCompareMeasure = 'count' | 'sum';

/**
 * Score written to `output.zScore`: `'poisson'` is `delta / sqrt(before + after)` (0 when both
 * are 0), `'standardized'` is `(delta - mean) / sd` over the union rows.
 */
export type GPUCellTableCompareZScore = 'poisson' | 'standardized';

/** Caller-owned, capacity-bounded outputs of {@link GPUCellTableCompare}. */
export type GPUCellTableCompareOutput = {
  /** Union cell keys ascending as little-endian `(low, high)` words; capacity is its length. */
  cells: GraphDataView<'uint32x2'>;
  /** Per cell: bit 1 (`GPU_CELL_COMPARE_PRESENT_BEFORE`) in `before`, bit 2 in `after`. */
  presence?: GraphDataView<'uint32'>;
  /** Measure in the `before` table, 0 when absent. */
  before?: GraphDataView<'float32'>;
  /** Measure in the `after` table, 0 when absent. */
  after?: GraphDataView<'float32'>;
  /**
   * `after - before`, computed exactly in integers (64-bit fixed point for sums) and then rounded
   * to f32.
   */
  delta?: GraphDataView<'float32'>;
  /** `after / before`, NaN when `before` is 0. */
  ratio?: GraphDataView<'float32'>;
  /** `100 * (after - before) / before`, NaN when `before` is 0. */
  percentChange?: GraphDataView<'float32'>;
  /** The configured z-score, see {@link GPUCellTableCompareZScore}. */
  zScore?: GraphDataView<'float32'>;
  /** One-row scalar receiving `min(totalCount, cells.length)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when union cells were dropped or either input table overflowed. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of union cells. */
  totalCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellTableCompare}.
 *
 * Per-frame (no recompile): the contents of both tables. Topology: table capacities, which
 * columns are present, `measure`, `sumScale`, `zScore`, and the output capacity.
 */
export type GPUCellTableCompareProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-table-compare'`. */
  id?: string;
  /**
   * First table, written by {@link GPUCellAggregation} or {@link GPUCellRollup}: ascending keys in
   * rows `[0, count)`. Both tables must use the same grid family and resolution; the recipe
   * cannot check that.
   */
  before: GPUCellTable;
  /** Second table, same family and resolution as `before`. */
  after: GPUCellTable;
  /** Compared column. Defaults to `'count'`; `'sum'` needs `sums` on both tables. */
  measure?: GPUCellTableCompareMeasure;
  /** Fixed-point scale of both tables' `sums` (for `'sum'`). Defaults to 65536. */
  sumScale?: number;
  /** Defaults to `'poisson'` for counts and `'standardized'` for sums. */
  zScore?: GPUCellTableCompareZScore;
  /** Caller-owned union table. */
  output: GPUCellTableCompareOutput;
};

/**
 * Outer-join diff of two sorted cell tables (period-over-period or A-versus-B compare maps).
 *
 * Both tables are sorted ascending with distinct keys, so the union order follows from binary
 * searches and one scan of matched `before` rows: a `before` row `i` lands at
 * `i + lowerBound(after, key) - matchesBelow(i)` and an `after` row `j` whose key is absent from
 * `before` at `j + lowerBound(before, key) - matchesBelow(lowerBound(before, key))`, where
 * `matchesBelow(k)` counts `before` rows below `k` that also occur in `after`. Equal keys are
 * emitted once, by the `before` side, which also reads the matching `after` row. Output rows are
 * written by exactly one invocation each, so results do not depend on thread order. `delta` is an
 * exact integer difference (64-bit fixed point for sums) before it is rounded to f32. The
 * standardized z-score mean and standard deviation come from one workgroup in a fixed order (strided partial
 * sums then a binary tree), so a second encoding is bitwise identical.
 *
 * The union keeps the smallest keys when it exceeds the output capacity, and the overflow flag
 * includes the input tables' own overflow flags. Rows past `count` hold the empty key, presence 0,
 * `before`/`after` 0 and NaN for derived columns. Keys that are the reserved all-ones key inside
 * `[0, count)` are not supported. Inputs must be single packed views.
 */
export class GPUCellTableCompare implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellTableCompareProps;
  /** Compared column. */
  readonly measure: GPUCellTableCompareMeasure;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;
  /** Resolved z-score kind. */
  readonly zScoreKind: GPUCellTableCompareZScore;

  constructor(props: GPUCellTableCompareProps) {
    this.id = props.id ?? 'cell-table-compare';
    this.props = props;
    const id = this.id;
    this.measure = props.measure ?? 'count';
    if (this.measure !== 'count' && this.measure !== 'sum') {
      throw new Error(`${id} measure must be 'count' or 'sum'`);
    }
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    this.zScoreKind = props.zScore ?? (this.measure === 'sum' ? 'standardized' : 'poisson');
    if (this.zScoreKind !== 'poisson' && this.zScoreKind !== 'standardized') {
      throw new Error(`${id} zScore must be 'poisson' or 'standardized'`);
    }
    validateCellTable(id, 'before', props.before);
    validateCellTable(id, 'after', props.after);
    if (this.measure === 'sum' && (!props.before.sums || !props.after.sums)) {
      throw new Error(`${id} measure 'sum' needs sums on both before and after`);
    }
    const {output} = props;
    for (const [name, view] of Object.entries(output)) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} output.${name} must be a single packed view`);
      }
    }
    validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    const capacity = output.cells.length;
    if (capacity < 1) {
      throw new Error(`${id} output.cells must hold at least one row`);
    }
    if (output.presence) {
      validatePackedUint32View(output.presence, `${id} output.presence`);
    }
    for (const name of ['before', 'after', 'delta', 'ratio', 'percentChange', 'zScore'] as const) {
      const view = output[name];
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
      }
    }
    for (const name of [
      'presence',
      'before',
      'after',
      'delta',
      'ratio',
      'percentChange',
      'zScore'
    ] as const) {
      const view = output[name];
      if (view && view.length !== capacity) {
        throw new Error(`${id} output.${name} must have the same length as output.cells`);
      }
    }
    for (const name of ['count', 'overflow', 'totalCount'] as const) {
      const view = output[name];
      if (!view) {
        continue;
      }
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} output.${name} must contain one uint32 row`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputViews(output), [
      ...getCellTableViews(props.before),
      ...getCellTableViews(props.after)
    ]);
  }

  /** Returns match, scan, merge, measure, statistics, derived-column, tail, and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, measure} = this;
    const {before, after, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getCellTableViews(before),
      ...getCellTableViews(after),
      ...getOutputViews(output)
    ]);
    const capacity = output.cells.length;
    const capacityBefore = before.cells.length;
    const capacityAfter = after.cells.length;
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const total = transient('total', 'uint32', 1);
    const beforeRows = transient('before-rows', 'uint32', capacity);
    const afterRows = transient('after-rows', 'uint32', capacity);
    const beforeValues = output.before ?? transient('before-values', 'float32', capacity);
    const afterValues = output.after ?? transient('after-values', 'float32', capacity);
    const deltaValues = output.delta ?? transient('delta-values', 'float32', capacity);

    // 1. Matches: before rows whose key also occurs in `after`, an exclusive scan of them, and
    // the clamped row counts. The scan has one extra row, so `prefix[capacityBefore]` is the total.
    const flags = transient('matched-flags', 'uint32', capacityBefore + 1);
    const prefix = transient('matched-prefix', 'uint32', capacityBefore + 1);
    const counts = transient('counts', 'uint32', 2);
    const keyDeclarations = `const CAPACITY: u32 = ${capacity}u;
const CAPACITY_BEFORE: u32 = ${capacityBefore}u;
const CAPACITY_AFTER: u32 = ${capacityAfter}u;
const NO_ROW: u32 = 0xffffffffu;
fn getBeforeKey(row: u32) -> vec2u {
  return vec2u(beforeCells[beforeCellsOffset + 2u * row + 1u], beforeCells[beforeCellsOffset + 2u * row]);
}
fn getAfterKey(row: u32) -> vec2u {
  return vec2u(afterCells[afterCellsOffset + 2u * row + 1u], afterCells[afterCellsOffset + 2u * row]);
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`;
    const searchFunction = (
      name: string,
      table: string
    ) => `fn ${name}(key: vec2u, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isKeyLess(get${table}Key(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flags`,
        operation: OPERATION,
        variant: 'matches',
        bindings: [
          read('beforeCells', before.cells),
          read('beforeCount', before.count),
          read('afterCells', after.cells),
          read('afterCount', after.count),
          write('flags', flags),
          write('counts', counts)
        ],
        invocationCount: capacityBefore + 1,
        declarations: `${keyDeclarations}
${searchFunction('lowerBoundAfter', 'After')}`,
        body: `let countBefore = min(beforeCount[beforeCountOffset], CAPACITY_BEFORE);
  let countAfter = min(afterCount[afterCountOffset], CAPACITY_AFTER);
  if (index == 0u) {
    counts[countsOffset] = countBefore;
    counts[countsOffset + 1u] = countAfter;
  }
  var isMatch = false;
  if (index < countBefore) {
    let key = getBeforeKey(index);
    let bound = lowerBoundAfter(key, countAfter);
    isMatch = bound < countAfter && all(getAfterKey(bound) == key);
  }
  flags[flagsOffset + index] = select(0u, 1u, isMatch);`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: flags,
        output: prefix,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 2. Merge path. A `before` row lands after the `before` rows below it and the `after` rows
    // below it, minus the matched pairs below it (counted on both sides). An unmatched `after`
    // row likewise subtracts the matched pairs below its key.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-merge`,
        operation: OPERATION,
        variant: 'merge-path',
        bindings: [
          read('beforeCells', before.cells),
          read('afterCells', after.cells),
          read('counts', counts),
          read('prefix', prefix),
          write('cellsOut', output.cells),
          write('beforeRows', beforeRows),
          write('afterRows', afterRows),
          write('total', total)
        ],
        invocationCount: capacityBefore + capacityAfter,
        declarations: `${keyDeclarations}
${searchFunction('lowerBoundBefore', 'Before')}
${searchFunction('lowerBoundAfter', 'After')}`,
        body: `let countBefore = counts[countsOffset];
  let countAfter = counts[countsOffset + 1u];
  if (index == 0u) {
    total[totalOffset] = countBefore + countAfter - prefix[prefixOffset + CAPACITY_BEFORE];
  }
  var key = vec2u(0u);
  var position = 0u;
  var beforeRow = NO_ROW;
  var afterRow = NO_ROW;
  if (index < CAPACITY_BEFORE) {
    if (index >= countBefore) {
      return;
    }
    key = getBeforeKey(index);
    let bound = lowerBoundAfter(key, countAfter);
    position = index + bound - prefix[prefixOffset + index];
    beforeRow = index;
    if (bound < countAfter && all(getAfterKey(bound) == key)) {
      afterRow = bound;
    }
  } else {
    let row = index - CAPACITY_BEFORE;
    if (row >= countAfter) {
      return;
    }
    key = getAfterKey(row);
    let bound = lowerBoundBefore(key, countBefore);
    if (bound < countBefore && all(getBeforeKey(bound) == key)) {
      return;
    }
    position = row + bound - prefix[prefixOffset + bound];
    afterRow = row;
  }
  if (position < CAPACITY) {
    cellsOut[cellsOutOffset + 2u * position] = key.y;
    cellsOut[cellsOutOffset + 2u * position + 1u] = key.x;
    beforeRows[beforeRowsOffset + position] = beforeRow;
    afterRows[afterRowsOffset + position] = afterRow;
  }`
      })
    );

    // 3. Measures and the exact delta.
    const isSum = measure === 'sum';
    const beforeMeasure = isSum ? before.sums! : before.counts;
    const afterMeasure = isSum ? after.sums! : after.counts;
    const measureAt = (table: string, row: string) =>
      isSum
        ? `vec2u(${table}Measure[${table}MeasureOffset + 2u * ${row} + 1u], ${table}Measure[${table}MeasureOffset + 2u * ${row}])`
        : `vec2u(0u, ${table}Measure[${table}MeasureOffset + ${row}])`;
    const toFloat = (value: string) =>
      isSum ? `cellI64ToF32(${value}) / SUM_SCALE` : `cellI64ToF32(${value})`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measures`,
        operation: OPERATION,
        variant: isSum ? 'sum-measures' : 'count-measures',
        bindings: [
          read('beforeRows', beforeRows),
          read('afterRows', afterRows),
          read('beforeMeasure', beforeMeasure),
          read('afterMeasure', afterMeasure),
          read('total', total),
          write('beforeOut', beforeValues, 'f32'),
          write('afterOut', afterValues, 'f32'),
          write('deltaOut', deltaValues, 'f32')
        ],
        invocationCount: capacity,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const NO_ROW: u32 = 0xffffffffu;
${ORDERED_KEY_WGSL}
${getFixedPointWGSL(this.sumScale)}`,
        body: `if (index >= min(total[totalOffset], CAPACITY)) {
    beforeOut[beforeOutOffset + index] = 0.0;
    afterOut[afterOutOffset + index] = 0.0;
    deltaOut[deltaOutOffset + index] = getNaN();
    return;
  }
  let beforeRow = beforeRows[beforeRowsOffset + index];
  let afterRow = afterRows[afterRowsOffset + index];
  var beforeValue = vec2u(0u);
  var afterValue = vec2u(0u);
  if (beforeRow != NO_ROW) {
    beforeValue = ${measureAt('before', 'beforeRow')};
  }
  if (afterRow != NO_ROW) {
    afterValue = ${measureAt('after', 'afterRow')};
  }
  let difference = cellAddI64(afterValue, cellNegateI64(beforeValue));
  beforeOut[beforeOutOffset + index] = ${toFloat('beforeValue')};
  afterOut[afterOutOffset + index] = ${toFloat('afterValue')};
  deltaOut[deltaOutOffset + index] = ${toFloat('difference')};`
      })
    );

    // 4. Standardized z-score statistics: one workgroup, fixed order.
    const wantsStandardized = Boolean(output.zScore) && this.zScoreKind === 'standardized';
    const statistics = wantsStandardized ? transient('statistics', 'float32', 2) : undefined;
    if (statistics) {
      const tree = (name: string) => `partials[localInvocationIndex] = ${name};
  workgroupBarrier();
  for (var stride = ${STATS_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics`,
          operation: OPERATION,
          variant: 'statistics',
          bindings: [
            read('deltas', deltaValues, 'f32'),
            read('total', total),
            write('statistics', statistics, 'f32')
          ],
          workgroupSize: STATS_WORKGROUP_SIZE,
          invocationCount: STATS_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `const CAPACITY: u32 = ${capacity}u;
var<workgroup> partials: array<f32, ${STATS_WORKGROUP_SIZE}>;`,
          body: `let rowCount = min(total[totalOffset], CAPACITY);
  var partialSum = 0.0;
  for (var row = localInvocationIndex; row < rowCount; row += ${STATS_WORKGROUP_SIZE}u) {
    partialSum += deltas[deltasOffset + row];
  }
  ${tree('partialSum')}
  let mean = select(0.0, partials[0] / f32(rowCount), rowCount > 0u);
  workgroupBarrier();
  var partialSquares = 0.0;
  for (var row = localInvocationIndex; row < rowCount; row += ${STATS_WORKGROUP_SIZE}u) {
    let centered = deltas[deltasOffset + row] - mean;
    partialSquares += centered * centered;
  }
  ${tree('partialSquares')}
  if (localInvocationIndex == 0u) {
    statistics[statisticsOffset] = mean;
    statistics[statisticsOffset + 1u] = select(0.0, sqrt(partials[0] / f32(rowCount)), rowCount > 0u);
  }`
        })
      );
    }

    // 5. Ratio, percent change and z-score.
    if (output.ratio || output.percentChange || output.zScore) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-derived`,
          operation: OPERATION,
          variant: 'derived',
          bindings: [
            read('beforeValues', beforeValues, 'f32'),
            read('afterValues', afterValues, 'f32'),
            read('deltaValues', deltaValues, 'f32'),
            read('total', total),
            ...(output.ratio ? [write('ratioOut', output.ratio, 'f32')] : []),
            ...(output.percentChange ? [write('percentOut', output.percentChange, 'f32')] : []),
            ...(output.zScore ? [write('zScoreOut', output.zScore, 'f32')] : []),
            ...(statistics ? [read('statistics', statistics, 'f32')] : [])
          ],
          invocationCount: capacity,
          declarations: `const CAPACITY: u32 = ${capacity}u;
${ORDERED_KEY_WGSL}`,
          body: `let isOccupied = index < min(total[totalOffset], CAPACITY);
  let nan = getNaN();
  let beforeValue = beforeValues[beforeValuesOffset + index];
  let afterValue = afterValues[afterValuesOffset + index];
  let delta = deltaValues[deltaValuesOffset + index];
  ${output.ratio ? 'ratioOut[ratioOutOffset + index] = select(nan, afterValue / beforeValue, isOccupied && beforeValue != 0.0);' : ''}
  ${output.percentChange ? 'percentOut[percentOutOffset + index] = select(nan, (100.0 * delta) / beforeValue, isOccupied && beforeValue != 0.0);' : ''}
  ${
    output.zScore
      ? this.zScoreKind === 'poisson'
        ? `let denominator = beforeValue + afterValue;
  zScoreOut[zScoreOutOffset + index] = select(nan, select(0.0, delta / sqrt(denominator), denominator != 0.0), isOccupied);`
        : `let mean = statistics[statisticsOffset];
  let deviation = statistics[statisticsOffset + 1u];
  zScoreOut[zScoreOutOffset + index] = select(nan, select(0.0, (delta - mean) / deviation, deviation != 0.0), isOccupied);`
      : ''
  }`
        })
      );
    }

    // 6. Presence, and empty keys past the union count.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tail`,
        operation: OPERATION,
        variant: 'tail',
        bindings: [
          read('beforeRows', beforeRows),
          read('afterRows', afterRows),
          read('total', total),
          write('cellsOut', output.cells),
          ...(output.presence ? [write('presenceOut', output.presence)] : [])
        ],
        invocationCount: capacity,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const NO_ROW: u32 = 0xffffffffu;`,
        body: `if (index >= min(total[totalOffset], CAPACITY)) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    ${output.presence ? 'presenceOut[presenceOutOffset + index] = 0u;' : ''}
    return;
  }
  ${
    output.presence
      ? `presenceOut[presenceOutOffset + index] =
    select(0u, ${GPU_CELL_COMPARE_PRESENT_BEFORE}u, beforeRows[beforeRowsOffset + index] != NO_ROW) |
    select(0u, ${GPU_CELL_COMPARE_PRESENT_AFTER}u, afterRows[afterRowsOffset + index] != NO_ROW);`
      : ''
  }`
      })
    );

    // 7. Clamped count and overflow, including the input tables' own overflow.
    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        totalCount: total,
        // Only the capacity of `ids` is used: nothing is copied without `compactIds`.
        output: {
          ids: output.cells as unknown as GraphDataView<'uint32'>,
          count: output.count,
          overflow: output.overflow,
          totalCount: output.totalCount
        },
        overflowSources: [before.overflow, after.overflow]
      })
    );
    return nodes;
  }
}

function getOutputViews(output: GPUCellTableCompareOutput): (GraphDataView | undefined)[] {
  return [
    output.cells,
    output.presence,
    output.before,
    output.after,
    output.delta,
    output.ratio,
    output.percentChange,
    output.zScore,
    output.count,
    output.overflow,
    output.totalCount
  ];
}
