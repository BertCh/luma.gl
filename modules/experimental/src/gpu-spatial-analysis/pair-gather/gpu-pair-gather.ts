// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {validateGPUCompactPairPort} from '../contracts/index';
import type {GPUSpatialJoinPairs} from '../spatial-join/spatial-join-types';

const OPERATION = 'GPUPairGather';
const NO_FEATURE = 0xffffffff;

/** Value written to `rightRows` for unmatched left rows of a left join (and to unused output rows). */
export const GPU_PAIR_GATHER_NO_ROW = 0xffffffff;

/** Join type of {@link GPUPairGather}: `'inner'` keeps matched pairs, `'left'` also keeps unmatched left rows. */
export type GPUPairGatherHow = 'inner' | 'left';

/** One attribute column gathered by {@link GPUPairGather}. */
export type GPUPairGatherColumn = {
  /** Packed source column. */
  source: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Caller-owned destination with the source format and one row per output row capacity. */
  output: GraphDataView<'float32'> | GraphDataView<'uint32'>;
};

/**
 * Dense `[query, slot]` neighbor lists as written by `GPUNearestFeatureJoin` in neighbors mode;
 * flattened to one pair row per filled slot (GeoPandas `sjoin_nearest`).
 */
export type GPUPairGatherNeighbors = {
  /** Neighbor feature row at `query * capacity + slot`, or `GPU_SPATIAL_JOIN_NO_FEATURE`. */
  ids: GraphDataView<'uint32'>;
  /** Number of filled slots per query. Its length is the query (left row) count. */
  counts: GraphDataView<'uint32'>;
  /** Slots per query: `ids.length` is `counts.length * capacity`. */
  capacity: number;
};

/** Caller-owned row outputs of {@link GPUPairGather}. */
export type GPUPairGatherOutput = {
  /** Left row per output row. Its length is the output row capacity. */
  leftRows: GraphDataView<'uint32'>;
  /** Right row per output row, {@link GPU_PAIR_GATHER_NO_ROW} for unmatched left rows. Same length. */
  rightRows: GraphDataView<'uint32'>;
  /** One-row scalar receiving `min(matched + unmatched, leftRows.length)`. */
  rowCount: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when any capacity (including the inputs') overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the matched pair count that fed the output. */
  matchedCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUPairGather}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths,
 * `how`, `neighbors.capacity` and which optional views are present.
 */
export type GPUPairGatherProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'pair-gather'`. */
  id?: string;
  /** `'inner'` (default) or `'left'`. */
  how?: GPUPairGatherHow;
  /** Matched pairs of a predicate join. Give exactly one of `pairs` and `neighbors`. */
  pairs?: GPUSpatialJoinPairs;
  /** Dense neighbor lists of a nearest join. Give exactly one of `pairs` and `neighbors`. */
  neighbors?: GPUPairGatherNeighbors;
  /**
   * Left rows without a match, for `how: 'left'`: the `unmatched` output of a predicate join with
   * `how: 'anti'`. With `neighbors` it is optional: queries with no neighbor are derived.
   */
  unmatchedLeft?: GPUCompactOutput;
  /** Left attribute columns, gathered by `leftRows`. Unmatched rows are never missing on the left. */
  leftColumns?: readonly GPUPairGatherColumn[];
  /** Right attribute columns, gathered by `rightRows`; unmatched left rows get NaN or `0xffffffff`. */
  rightColumns?: readonly GPUPairGatherColumn[];
  /**
   * Columns aligned with the input pair slots (`pairs.leftIds` rows, or `ids` rows for
   * `neighbors`) and carried to the output pair rows, for example `neighborDistances`
   * (`sjoin_nearest(distance_col=...)`) or join weights. Unmatched rows get NaN or `0xffffffff`.
   */
  slotColumns?: readonly GPUPairGatherColumn[];
  /** Extra one-row flags ORed into `output.overflow`, for example a nearest join's `overflow`. */
  overflow?: readonly GraphDataView<'uint32'>[];
  /** Caller-owned bounded row outputs. */
  output: GPUPairGatherOutput;
};

/**
 * Merges join output with attribute columns: the dataframe half of GeoPandas `sjoin` and
 * `sjoin_nearest` (`how='inner'` and `how='left'`, `distance_col`).
 *
 * Output rows `[0, matched)` are the matched pairs in input order (a predicate join emits them
 * sorted by `(left, right)`; neighbor lists by query then distance), followed, for `how: 'left'`,
 * by one row per unmatched left row in ascending order with `rightRows = GPU_PAIR_GATHER_NO_ROW`
 * and NaN (`float32`) or `0xffffffff` (`uint32`) in every right and slot column. This is a
 * concatenation, not GeoPandas' left-row order; sort by `(leftRows, rightRows)` to compare.
 *
 * With `neighbors`, filled slots (`slot < counts[query]`, feature not `GPU_SPATIAL_JOIN_NO_FEATURE`)
 * are flattened by a stable compaction. Columns are copied as raw 32-bit words, so NaN payloads
 * survive. Bounded by `output.leftRows.length`; surplus rows set `overflow`. Deterministic, no atomics.
 */
export class GPUPairGather implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPairGatherProps;
  /** Resolved join type. */
  readonly how: GPUPairGatherHow;

  constructor(props: GPUPairGatherProps) {
    this.id = props.id ?? 'pair-gather';
    this.props = props;
    this.how = props.how ?? 'inner';
    const {id, how} = this;
    const {output} = props;
    if (how !== 'inner' && how !== 'left') {
      throw new Error(`${id} how must be 'inner' or 'left'`);
    }
    if (Boolean(props.pairs) === Boolean(props.neighbors)) {
      throw new Error(`${id} needs exactly one of pairs and neighbors`);
    }
    if (how === 'left' && !props.unmatchedLeft && !props.neighbors) {
      throw new Error(
        `${id} how 'left' with pairs needs unmatchedLeft (a predicate join anti output)`
      );
    }
    if (how === 'inner' && props.unmatchedLeft) {
      throw new Error(`${id} unmatchedLeft needs how 'left'`);
    }
    for (const name of ['leftRows', 'rightRows', 'rowCount', 'overflow', 'matchedCount'] as const) {
      if (output[name]) {
        validatePackedUint32View(output[name], `${id} output.${name}`);
      }
    }
    const capacity = output.leftRows.length;
    if (capacity < 1) {
      throw new Error(`${id} output.leftRows must contain at least one row`);
    }
    if (output.rightRows.length !== capacity) {
      throw new Error(`${id} output.rightRows length must equal output.leftRows length`);
    }
    for (const name of ['rowCount', 'overflow', 'matchedCount'] as const) {
      if (output[name] && output[name].length < 1) {
        throw new Error(`${id} output.${name} must contain one uint32 row`);
      }
    }
    let slotCount: number;
    if (props.pairs) {
      const {pairs} = props;
      validateGPUCompactPairPort(id, pairs);
      validatePackedUint32View(pairs.leftIds, `${id} pairs.leftIds`);
      validatePackedUint32View(pairs.rightIds, `${id} pairs.rightIds`);
      validatePackedUint32View(pairs.count, `${id} pairs.count`);
      validatePackedUint32View(pairs.overflow, `${id} pairs.overflow`);
      if (pairs.rightIds.length !== pairs.leftIds.length) {
        throw new Error(`${id} pairs.rightIds length must equal pairs.leftIds length`);
      }
      slotCount = pairs.leftIds.length;
    } else {
      const {neighbors} = props;
      validatePackedUint32View(neighbors!.ids, `${id} neighbors.ids`);
      validatePackedUint32View(neighbors!.counts, `${id} neighbors.counts`);
      if (!Number.isInteger(neighbors!.capacity) || neighbors!.capacity < 1) {
        throw new Error(`${id} neighbors.capacity must be a positive integer`);
      }
      if (neighbors!.ids.length !== neighbors!.counts.length * neighbors!.capacity) {
        throw new Error(`${id} neighbors.ids length must equal counts length * capacity`);
      }
      slotCount = neighbors!.ids.length;
    }
    if (slotCount < 1) {
      throw new Error(`${id} needs at least one pair slot`);
    }
    if (props.unmatchedLeft) {
      validateCompactOutput(id, props.unmatchedLeft);
    }
    for (const [name, columns, rows] of [
      ['leftColumns', props.leftColumns, capacity],
      ['rightColumns', props.rightColumns, capacity],
      ['slotColumns', props.slotColumns, capacity]
    ] as const) {
      for (const [index, column] of (columns ?? []).entries()) {
        const label = `${id} ${name}[${index}]`;
        validatePackedView(column.source, [column.source.format as 'float32' | 'uint32'], label);
        if (column.source.format !== 'float32' && column.source.format !== 'uint32') {
          throw new Error(`${label} must be float32 or uint32`);
        }
        if (column.output.format !== column.source.format) {
          throw new Error(`${label} output format must equal source format`);
        }
        validatePackedView(column.output, [column.output.format], `${label} output`);
        if (column.output.length !== rows) {
          throw new Error(`${label} output length must equal output.leftRows length`);
        }
        if (name === 'slotColumns' && column.source.length < slotCount) {
          throw new Error(`${label} source must hold one row per pair slot (${slotCount})`);
        }
      }
    }
    for (const flag of props.overflow ?? []) {
      validatePackedUint32View(flag, `${id} overflow`);
    }
    validateGraphOutputsDisjointFromInputs(id, this._getOutputViews(), this._getInputViews());
  }

  private _getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [
      output.leftRows,
      output.rightRows,
      output.rowCount,
      output.overflow,
      ...(output.matchedCount ? [output.matchedCount] : []),
      ...[
        ...(this.props.leftColumns ?? []),
        ...(this.props.rightColumns ?? []),
        ...(this.props.slotColumns ?? [])
      ].map(column => column.output as GraphDataView)
    ];
  }

  private _getInputViews(): GraphDataView[] {
    const {props} = this;
    return [
      ...(props.pairs
        ? [props.pairs.leftIds, props.pairs.rightIds, props.pairs.count, props.pairs.overflow]
        : [props.neighbors!.ids, props.neighbors!.counts]),
      ...(props.unmatchedLeft
        ? [props.unmatchedLeft.ids, props.unmatchedLeft.count, props.unmatchedLeft.overflow]
        : []),
      ...(props.overflow ?? []),
      ...[
        ...(props.leftColumns ?? []),
        ...(props.rightColumns ?? []),
        ...(props.slotColumns ?? [])
      ].map(column => column.source as GraphDataView)
    ];
  }

  /** Returns the optional flatten nodes, the row kernel, one kernel per column and the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, how} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...this._getInputViews(),
      ...this._getOutputViews()
    ]);
    const capacity = output.leftRows.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Stage 1 (neighbors): flatten filled slots to a compact slot list and count them.
    let slotIds: GraphDataView<'uint32'> | undefined;
    let matched: GraphDataView<'uint32'>;
    const neighbors = props.neighbors;
    if (neighbors) {
      const slotCount = neighbors.ids.length;
      const slotMask = createTransientView(graph, `${id}-slot-mask`, 'uint32', slotCount);
      slotIds = createTransientView(graph, `${id}-slot-ids`, 'uint32', slotCount);
      matched = createTransientView(graph, `${id}-matched`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-mask`,
          operation: OPERATION,
          variant: 'neighbors',
          bindings: [
            {name: 'neighborIds', view: neighbors.ids, type: 'u32', access: 'read'},
            {name: 'neighborCounts', view: neighbors.counts, type: 'u32', access: 'read'},
            {name: 'maskOut', view: slotMask, type: 'u32', access: 'read_write'}
          ],
          invocationCount: slotCount,
          declarations: `const CAPACITY: u32 = ${neighbors.capacity}u;
const NO_FEATURE: u32 = ${NO_FEATURE}u;`,
          body: `let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let filled = slot < neighborCounts[neighborCountsOffset + query] &&
    neighborIds[neighborIdsOffset + index] != NO_FEATURE;
  maskOut[maskOutOffset + index] = select(0u, 1u, filled);`
        }),
        ...new GPUVisibilityWorkflow({
          id: `${id}-slots`,
          predicates: [{kind: 'selection', mask: slotMask}],
          output: slotIds,
          count: matched
        }).getCommandNodes(graph)
      );
    } else {
      matched = props.pairs!.count;
    }

    // Unmatched left rows: given, or derived from queries with no neighbor.
    let unmatchedIds: GraphDataView<'uint32'> | undefined;
    let unmatchedCount: GraphDataView<'uint32'> | undefined;
    const overflowFlags: GraphDataView<'uint32'>[] = [
      ...(props.pairs ? [props.pairs.overflow] : []),
      ...(props.overflow ?? [])
    ];
    if (how === 'left') {
      if (props.unmatchedLeft) {
        unmatchedIds = props.unmatchedLeft.ids;
        unmatchedCount = props.unmatchedLeft.count;
        overflowFlags.push(props.unmatchedLeft.overflow);
      } else {
        const queryCount = neighbors!.counts.length;
        const emptyMask = createTransientView(graph, `${id}-empty-mask`, 'uint32', queryCount);
        unmatchedIds = createTransientView(graph, `${id}-unmatched-ids`, 'uint32', queryCount);
        unmatchedCount = createTransientView(graph, `${id}-unmatched-count`, 'uint32', 1);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-empty-mask`,
            operation: OPERATION,
            variant: 'unmatched',
            bindings: [
              {name: 'neighborCounts', view: neighbors!.counts, type: 'u32', access: 'read'},
              {name: 'maskOut', view: emptyMask, type: 'u32', access: 'read_write'}
            ],
            invocationCount: queryCount,
            body: `maskOut[maskOutOffset + index] =
    select(0u, 1u, neighborCounts[neighborCountsOffset + index] == 0u);`
          }),
          ...new GPUVisibilityWorkflow({
            id: `${id}-unmatched`,
            predicates: [{kind: 'selection', mask: emptyMask}],
            output: unmatchedIds,
            count: unmatchedCount
          }).getCommandNodes(graph)
        );
      }
    }

    // Stage 2: left and right row per output row.
    const rowBindings: WGSLKernelBinding[] = [
      {name: 'matchedIn', view: matched, type: 'u32', access: 'read'}
    ];
    let pairLookup: string;
    if (slotIds) {
      rowBindings.push({name: 'slotIds', view: slotIds, type: 'u32', access: 'read'});
    }
    if (neighbors) {
      rowBindings.push({name: 'neighborIds', view: neighbors.ids, type: 'u32', access: 'read'});
      pairLookup = `let slot = slotIds[slotIdsOffset + index];
    left = slot / ${neighbors.capacity}u;
    right = neighborIds[neighborIdsOffset + slot];`;
    } else {
      rowBindings.push(
        {name: 'pairLeft', view: props.pairs!.leftIds, type: 'u32', access: 'read'},
        {name: 'pairRight', view: props.pairs!.rightIds, type: 'u32', access: 'read'}
      );
      pairLookup = `left = pairLeft[pairLeftOffset + index];
    right = pairRight[pairRightOffset + index];`;
    }
    if (unmatchedIds && unmatchedCount) {
      rowBindings.push(
        {name: 'unmatchedIds', view: unmatchedIds, type: 'u32', access: 'read'},
        {name: 'unmatchedCount', view: unmatchedCount, type: 'u32', access: 'read'}
      );
    }
    rowBindings.push(
      {name: 'leftOut', view: output.leftRows, type: 'u32', access: 'read_write'},
      {name: 'rightOut', view: output.rightRows, type: 'u32', access: 'read_write'}
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows`,
        operation: OPERATION,
        variant: 'rows',
        bindings: rowBindings,
        invocationCount: capacity,
        declarations: `const NO_ROW: u32 = ${GPU_PAIR_GATHER_NO_ROW}u;`,
        body: `let matched = matchedIn[matchedInOffset];
  var left = NO_ROW;
  var right = NO_ROW;
  if (index < matched) {
    ${pairLookup}
  }${
    unmatchedIds
      ? ` else {
    let unmatchedIndex = index - matched;
    if (unmatchedIndex < ${unmatchedIds.length}u &&
        unmatchedIndex < unmatchedCount[unmatchedCountOffset]) {
      left = unmatchedIds[unmatchedIdsOffset + unmatchedIndex];
    }
  }`
      : ''
  }
  leftOut[leftOutOffset + index] = left;
  rightOut[rightOutOffset + index] = right;`
      })
    );

    // Stage 3: columns, copied as raw words.
    const columnNode = (name: string, column: GPUPairGatherColumn, rows: GraphDataView<'uint32'>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: 'column',
        bindings: [
          {name: 'rowsIn', view: rows, type: 'u32', access: 'read'},
          {name: 'sourceIn', view: column.source as GraphDataView, type: 'u32', access: 'read'},
          {
            name: 'columnOut',
            view: column.output as GraphDataView,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: capacity,
        declarations: `const NO_ROW: u32 = ${GPU_PAIR_GATHER_NO_ROW}u;
const SOURCE_LENGTH: u32 = ${column.source.length}u;
const FILL: u32 = ${column.source.format === 'float32' ? '0x7fc00000u' : '0xffffffffu'};`,
        body: `let row = rowsIn[rowsInOffset + index];
  var value = FILL;
  if (row != NO_ROW && row < SOURCE_LENGTH) {
    value = sourceIn[sourceInOffset + row];
  }
  columnOut[columnOutOffset + index] = value;`
      });
    (props.leftColumns ?? []).forEach((column, index) =>
      nodes.push(columnNode(`left-column-${index}`, column, output.leftRows))
    );
    (props.rightColumns ?? []).forEach((column, index) =>
      nodes.push(columnNode(`right-column-${index}`, column, output.rightRows))
    );
    (props.slotColumns ?? []).forEach((column, index) => {
      const slotBindings: WGSLKernelBinding[] = [
        {name: 'matchedIn', view: matched, type: 'u32', access: 'read'},
        {name: 'sourceIn', view: column.source as GraphDataView, type: 'u32', access: 'read'},
        {name: 'columnOut', view: column.output as GraphDataView, type: 'u32', access: 'read_write'}
      ];
      if (slotIds) {
        slotBindings.push({name: 'slotIds', view: slotIds, type: 'u32', access: 'read'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-column-${index}`,
          operation: OPERATION,
          variant: 'slot-column',
          bindings: slotBindings,
          invocationCount: capacity,
          declarations: `const FILL: u32 = ${column.source.format === 'float32' ? '0x7fc00000u' : '0xffffffffu'};`,
          body: `var value = FILL;
  if (index < matchedIn[matchedInOffset]) {
    ${slotIds ? 'let slot = slotIds[slotIdsOffset + index];' : 'let slot = index;'}
    value = sourceIn[sourceInOffset + slot];
  }
  columnOut[columnOutOffset + index] = value;`
        })
      );
    });

    // Stage 4: clamp counts and publish flags.
    const publishBindings: WGSLKernelBinding[] = [
      {name: 'matchedIn', view: matched, type: 'u32', access: 'read'}
    ];
    if (unmatchedCount) {
      publishBindings.push({
        name: 'unmatchedCount',
        view: unmatchedCount,
        type: 'u32',
        access: 'read'
      });
    }
    overflowFlags.forEach((flag, flagIndex) =>
      publishBindings.push({name: `flagIn${flagIndex}`, view: flag, type: 'u32', access: 'read'})
    );
    publishBindings.push(
      {name: 'rowCountOut', view: output.rowCount, type: 'u32', access: 'read_write'},
      {name: 'overflowOut', view: output.overflow, type: 'u32', access: 'read_write'}
    );
    if (output.matchedCount) {
      publishBindings.push({
        name: 'matchedOut',
        view: output.matchedCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    const overflowExpression = [
      'total > CAPACITY',
      ...overflowFlags.map((_, flagIndex) => `flagIn${flagIndex}[flagIn${flagIndex}Offset] != 0u`)
    ].join(' || ');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: publishBindings,
        invocationCount: 1,
        declarations: `const CAPACITY: u32 = ${capacity}u;`,
        body: `let matched = matchedIn[matchedInOffset];
  let total = matched + ${unmatchedCount ? 'unmatchedCount[unmatchedCountOffset]' : '0u'};
  rowCountOut[rowCountOutOffset] = min(total, CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, ${overflowExpression});
  ${output.matchedCount ? 'matchedOut[matchedOutOffset] = matched;' : ''}`
      })
    );
    return nodes;
  }
}
