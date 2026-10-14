// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  GraphVectorView,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUVisibilityPredicate,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput, GPUUint32Rows} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createPublishNode,
  createTransientUint32Rows
} from '../../utils/wgsl-kernel-nodes';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGraphViewTopology,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {getTimeWindowClassifyNodes} from './time-window-classify-node';
import {
  createSortedTimeWindowIdsNode,
  createSortedTimeWindowOrderCheckNode,
  createSortedTimeWindowRangeNode
} from './time-window-sorted-nodes';
import {GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} from './time-window-parameters';
import {
  GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH,
  type GPUInt64TimeWordRows
} from './time-words';

type Float32Rows = GraphDataView<'float32'> | GraphVectorView<'float32'>;

/**
 * Caller-owned outputs of {@link GPUTimeWindowFilter}: the bounded compact list of accepted rows
 * (`ids`, `count`, `overflow`, optional `requiredCount`; `ids.length` is the capacity) plus
 * optional per-row and per-track views. Writable views must not alias each other or any input.
 */
export type GPUTimeWindowFilterOutput = GPUCompactOutput & {
  /** Optional canonical 0/1 mask of accepted rows (time AND additional predicates). */
  outputMask?: GPUUint32Rows;
  /** Optional per-row fade weight in `[0, 1]`, 0 for rejected rows. Reflects the time test only. */
  fadeWeights?: Float32Rows;
  /**
   * Optional per-row `[clipStart, clipEnd]` fractions of each accepted interval inside the window.
   * Rejected rows get `[0, 0]`. Requires `endTimestamps`. Reflects the time test only.
   */
  clipFractions?: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /** Optional per-track count of accepted rows. Track IDs `>= length` are ignored. Requires `trackIds`. */
  trackVisibleCounts?: GraphDataView<'uint32'>;
  /**
   * Optional packed one-row view that receives the clamped count, typically an indirect draw
   * record's `instanceCount` imported with `graph.importGPUData(id, drawCommands.getInstanceCountData(0))`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /**
   * Sorted mode only. Optional packed two-word view `[firstRow, rowCount]` of the accepted
   * contiguous row range. `rowCount` is the unclamped number of accepted rows, so it can exceed
   * `ids.length`; the rows are `firstRow .. firstRow + rowCount - 1`. A caller that reads the range
   * alone passes a small `ids` view and skips the O(capacity) ID write; `drawInstanceCount` and
   * `status.count` stay clamped to `ids.length`, so keep `ids` large enough when they drive a draw.
   */
  rowRange?: GraphDataView<'uint32'>;
  /**
   * Sorted mode only. Optional packed one-row view that receives `firstRow`, typically an indirect
   * draw record's `firstInstance` word imported as a one-row packed `GPUData` at the record's
   * last four bytes (the instance index is then the row index).
   */
  drawFirstInstance?: GraphDataView<'uint32'>;
  /**
   * Optional packed one-row view that receives, when {@link GPUTimeWindowFilterProps.validateOrder}
   * is set, the number of adjacent row pairs whose times decrease (or are NaN). Zero means the
   * sorted precondition holds. Required with `validateOrder`.
   */
  orderViolations?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTimeWindowFilter}.
 *
 * Per-frame (no recompile): the contents of `window` and of every input buffer. Topology (needs a
 * new graph): view lengths and chunking, `output.ids.length`, `output.trackVisibleCounts.length`, which
 * optional views are present, and the number of `additionalPredicates`.
 */
export type GPUTimeWindowFilterProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'time-window-filter'`. */
  id?: string;
  /**
   * Row timestamps (instant mode), or row start times when `endTimestamps` is given (interval
   * mode). Either f32 relative to an application epoch (or the high part of `splitTimestamps`),
   * or exact Int64 words: a `uint32x2` view of `(low, high)` rows, for example an Arrow `Int64` or
   * `Timestamp` column uploaded with `getInt64TimeWords`. Word mode requires a `uint32` window.
   */
  timestamps: Float32Rows | GPUInt64TimeWordRows;
  /** Optional low parts aligned with `timestamps` for double-single precision. Float32 timestamps only. */
  timestampsLow?: Float32Rows;
  /**
   * Optional row end times. When given, rows are intervals such as trail segments. Must have the
   * same format as `timestamps` (float32 or `uint32x2` words).
   */
  endTimestamps?: Float32Rows | GPUInt64TimeWordRows;
  /** Optional low parts aligned with `endTimestamps`. Requires `endTimestamps`. Float32 timestamps only. */
  endTimestampsLow?: Float32Rows;
  /**
   * Per-frame window. Float32 timestamps: packed float32 view of at least
   * {@link GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} elements written with `getGPUTimeWindowFilterParameterValues`.
   * Word timestamps: packed `uint32` view of at least {@link GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH}
   * elements written with `getGPUTimeWindowFilterWordParameterValues`.
   */
  window: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Optional extra source-aligned predicate masks (bounds, LOD, selection) ANDed with time. */
  additionalPredicates?: readonly GPUVisibilityPredicate[];
  /** Optional stable IDs aligned with rows. Zero-based row indices are emitted when omitted. */
  sourceIds?: GPUUint32Rows;
  /** Optional dense track IDs per row, for example the track that owns each segment. */
  trackIds?: GPUUint32Rows;
  /** Caller-owned outputs. `output.ids.length` is the capacity. */
  output: GPUTimeWindowFilterOutput;
  /**
   * Compile-time, default false. Declares that `timestamps` (with `timestampsLow`) is non-decreasing
   * in row order and has no NaN. The accepted rows are then one contiguous range found by two binary
   * searches in a single small kernel, O(log n) per frame and no compaction, written to
   * `output.rowRange`, `output.drawFirstInstance` and the status words; `output.ids` receives only
   * its first `min(count, capacity)` entries. Acceptance is identical to the mask path, which uses the
   * same comparison helpers.
   *
   * Restrictions, each a constructor error: instant timestamps only (no `endTimestamps`), no
   * `additionalPredicates`, `trackIds`, `output.trackVisibleCounts` or `output.clipFractions`, and
   * single packed (unchunked) views. Optional `output.outputMask` and `output.fadeWeights` are
   * still written for every row (O(n)), so omit them when scrubbing very large data.
   *
   * The order is not checked on the default path. Unsorted input gives an unspecified range; use
   * {@link GPUTimeWindowFilterProps.validateOrder} while developing.
   */
  sorted?: boolean;
  /**
   * Compile-time, default false, sorted mode only. Adds one O(n) pass that counts adjacent
   * descents into `output.orderViolations` each encoding. Opt-in because the pass costs as much as
   * the mask path that sorted mode avoids.
   */
  validateOrder?: boolean;
};

/**
 * Filters timestamped rows or time intervals against a per-frame time window.
 *
 * Publishes stable compacted IDs, a clamped count with overflow, and optional mask, fade-weight,
 * trail-clip, per-track-count, and indirect-draw outputs, without recompiling the graph when the
 * window moves. Composition: one classify kernel, `GPUVisibilityWorkflow`, optional
 * `GPUGroupAggregation`, and one publish kernel that clamps the count. With `sorted: true` the
 * composition is instead one binary-search kernel, an IDs kernel over the capacity and the publish
 * kernel (see {@link GPUTimeWindowFilterProps.sorted}).
 *
 * Time input modes:
 * - float32 (cheap path): relative times, optionally with `timestampsLow` for double-single
 *   precision. Window written with `getGPUTimeWindowFilterParameterValues`.
 * - exact Int64 words: `uint32x2` rows `(low, high)` such as Arrow `Int64` epoch milliseconds, with
 *   a `uint32` window written with `getGPUTimeWindowFilterWordParameterValues`. Acceptance comparisons
 *   are exact for any Int64 value and any playhead fraction. Fade and clip differences are
 *   subtracted exactly and then rounded to f32: exact below 2^24 units, rounded above. There is no
 *   epoch to rebase. `timestampsLow` is not allowed in this mode.
 */
export class GPUTimeWindowFilter implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTimeWindowFilterProps;

  constructor(props: GPUTimeWindowFilterProps) {
    this.id = props.id ?? 'time-window-filter';
    this.props = props;
    const id = this.id;
    const rows = props.timestamps.length;
    const {output} = props;

    const isWordMode = props.timestamps.format === 'uint32x2';
    if (props.endTimestamps && (props.endTimestamps.format === 'uint32x2') !== isWordMode) {
      throw new Error(`${id} endTimestamps must have the same format as timestamps`);
    }
    if (isWordMode && (props.timestampsLow || props.endTimestampsLow)) {
      throw new Error(`${id} timestampsLow and endTimestampsLow require float32 timestamps`);
    }
    for (const [name, view] of [
      ['timestamps', props.timestamps],
      ['timestampsLow', props.timestampsLow],
      ['endTimestamps', props.endTimestamps],
      ['endTimestampsLow', props.endTimestampsLow],
      ['output.fadeWeights', output.fadeWeights]
    ] as const) {
      const isTimeWords = isWordMode && (name === 'timestamps' || name === 'endTimestamps');
      for (const chunk of view ? getGraphViewChunks(view as GraphDataView | GraphVectorView) : []) {
        validatePackedView(chunk, [isTimeWords ? 'uint32x2' : 'float32'], `${id} ${name}`);
      }
      if (name !== 'timestamps') {
        validateGraphViewTopology(id, name, props.timestamps, view);
      }
    }
    for (const chunk of output.clipFractions ? getGraphViewChunks(output.clipFractions) : []) {
      validatePackedView(chunk, ['float32x2'], `${id} output.clipFractions`);
    }
    validateGraphViewTopology(id, 'output.clipFractions', props.timestamps, output.clipFractions);
    for (const chunk of output.outputMask ? getGraphViewChunks(output.outputMask) : []) {
      validatePackedUint32View(chunk, `${id} output.outputMask`);
    }
    validateGraphViewTopology(id, 'output.outputMask', props.timestamps, output.outputMask);

    if (isWordMode) {
      if (props.window.format !== 'uint32') {
        throw new Error(`${id} uint32x2 word timestamps require a packed uint32 word window`);
      }
      validatePackedView(props.window, ['uint32'], `${id} window`);
      if (props.window.length < GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH) {
        throw new Error(
          `${id} window must hold ${GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH} uint32 values`
        );
      }
    } else {
      if (props.window.format !== 'float32') {
        throw new Error(`${id} float32 timestamps require a float32 window`);
      }
      validatePackedView(props.window, ['float32'], `${id} window`);
      if (props.window.length < GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH) {
        throw new Error(
          `${id} window must hold ${GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    for (const [name, view] of [
      ['sourceIds', props.sourceIds],
      ['trackIds', props.trackIds],
      ...(props.additionalPredicates ?? []).map(
        (predicate, index) => [`additionalPredicates[${index}]`, predicate.mask] as const
      )
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal timestamps length`);
      }
    }
    if (props.endTimestampsLow && !props.endTimestamps) {
      throw new Error(`${id} endTimestampsLow requires endTimestamps`);
    }
    if (output.clipFractions && !props.endTimestamps) {
      throw new Error(`${id} output.clipFractions requires endTimestamps`);
    }
    if (Boolean(props.trackIds) !== Boolean(output.trackVisibleCounts)) {
      throw new Error(`${id} trackIds and output.trackVisibleCounts must be given together`);
    }
    if (output.trackVisibleCounts) {
      validatePackedUint32View(output.trackVisibleCounts, `${id} output.trackVisibleCounts`);
      if (output.trackVisibleCounts.length < 1) {
        throw new Error(`${id} output.trackVisibleCounts must contain at least one track`);
      }
    }
    validateCompactOutput(id, props.output);
    this.validateSorted();
    if (output.drawInstanceCount) {
      validatePackedUint32View(output.drawInstanceCount, `${id} output.drawInstanceCount`);
      if (output.drawInstanceCount.length < 1) {
        throw new Error(`${id} output.drawInstanceCount must contain one uint32 row`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.ids,
        props.output.status.count,
        props.output.status.overflow,
        props.output.status.requiredCount,
        output.outputMask,
        output.fadeWeights,
        output.clipFractions,
        output.trackVisibleCounts,
        output.drawInstanceCount,
        output.rowRange,
        output.drawFirstInstance,
        output.orderViolations
      ],
      [
        props.timestamps,
        props.timestampsLow,
        props.endTimestamps,
        props.endTimestampsLow,
        props.window,
        props.sourceIds,
        props.trackIds,
        ...(props.additionalPredicates ?? []).map(predicate => predicate.mask)
      ]
    );
  }

  /** Validates the `sorted`, `validateOrder` and sorted-only output props. */
  private validateSorted(): void {
    const {props, id} = this;
    const {output} = props;
    const sortedOnly = [
      ['output.rowRange', output.rowRange],
      ['output.drawFirstInstance', output.drawFirstInstance],
      ['validateOrder', props.validateOrder]
    ] as const;
    if (!props.sorted) {
      for (const [name, value] of sortedOnly) {
        if (value) {
          throw new Error(`${id} ${name} requires sorted`);
        }
      }
      if (output.orderViolations) {
        throw new Error(`${id} output.orderViolations requires sorted and validateOrder`);
      }
      return;
    }
    if (props.endTimestamps) {
      throw new Error(`${id} sorted requires instant timestamps`);
    }
    if (props.additionalPredicates?.length || props.trackIds) {
      throw new Error(`${id} sorted does not support additionalPredicates or trackIds`);
    }
    for (const [name, view] of [
      ['timestamps', props.timestamps],
      ['timestampsLow', props.timestampsLow],
      ['sourceIds', props.sourceIds]
    ] as const) {
      if (view instanceof GraphVectorView) {
        throw new Error(`${id} sorted requires a single packed ${name} view`);
      }
    }
    if (output.rowRange) {
      validatePackedUint32View(output.rowRange, `${id} output.rowRange`);
      if (output.rowRange.length < 2) {
        throw new Error(`${id} output.rowRange must contain two uint32 rows`);
      }
    }
    if (output.drawFirstInstance) {
      validatePackedUint32View(output.drawFirstInstance, `${id} output.drawFirstInstance`);
      if (output.drawFirstInstance.length < 1) {
        throw new Error(`${id} output.drawFirstInstance must contain one uint32 row`);
      }
    }
    if (Boolean(props.validateOrder) !== Boolean(output.orderViolations)) {
      throw new Error(`${id} validateOrder and output.orderViolations must be given together`);
    }
    if (output.orderViolations) {
      validatePackedUint32View(output.orderViolations, `${id} output.orderViolations`);
      if (output.orderViolations.length < 1) {
        throw new Error(`${id} output.orderViolations must contain one uint32 row`);
      }
    }
  }

  /** Returns classify, visibility, optional track-count, and publish nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output, trackIds} = props;
    const {outputMask, trackVisibleCounts, drawInstanceCount} = output;
    const additionalPredicates = props.additionalPredicates ?? [];
    validateGraphViewsBelongToGraph(id, graph, [
      props.timestamps,
      props.timestampsLow,
      props.endTimestamps,
      props.endTimestampsLow,
      props.window,
      props.sourceIds,
      outputMask,
      output.fadeWeights,
      output.clipFractions,
      trackIds,
      trackVisibleCounts,
      drawInstanceCount,
      output.rowRange,
      output.drawFirstInstance,
      output.orderViolations,
      output.ids,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount,
      ...additionalPredicates.map(predicate => predicate.mask)
    ]);
    const rows = props.timestamps.length;
    if (props.sorted) {
      return this.getSortedCommandNodes(graph);
    }
    const hasExtraPredicates = additionalPredicates.length > 0;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const timeMask =
      !hasExtraPredicates && outputMask
        ? outputMask
        : createTransientUint32Rows(graph, `${id}-time-mask`, props.timestamps);
    nodes.push(
      ...getTimeWindowClassifyNodes(graph, {
        id: `${id}-classify`,
        timestamps: props.timestamps,
        timestampsLow: props.timestampsLow,
        endTimestamps: props.endTimestamps,
        endTimestampsLow: props.endTimestampsLow,
        window: props.window,
        mask: timeMask,
        fadeWeights: output.fadeWeights,
        clipFractions: output.clipFractions
      })
    );

    let finalMask: GPUUint32Rows | undefined = timeMask;
    if (hasExtraPredicates) {
      finalMask =
        outputMask ??
        (trackVisibleCounts
          ? createTransientUint32Rows(graph, `${id}-mask`, props.timestamps)
          : undefined);
    }

    // Compact straight into the caller's IDs when they can hold every row; otherwise compact into
    // full-size scratch and let the publish kernel copy the bounded prefix.
    const direct = rows > 0 && output.ids.length >= rows;
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const compactIds = direct
      ? output.ids
      : createTransientView(graph, `${id}-compact-ids`, 'uint32', rows);
    nodes.push(
      ...new GPUVisibilityWorkflow({
        id: `${id}-visibility`,
        predicates: [{kind: 'time-range', mask: timeMask}, ...additionalPredicates],
        output: compactIds,
        count: total,
        // Passing the time mask itself as the output mask skips the redundant compose pass.
        outputMask: finalMask,
        sourceIds: props.sourceIds
      }).getCommandNodes(graph)
    );

    if (trackIds && trackVisibleCounts && finalMask) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-track-counts`,
          keys: trackIds,
          mask: finalMask,
          output: trackVisibleCounts
        }).getCommandNodes(graph)
      );
    }

    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUTimeWindowFilter',
        requiredCount: total,
        compactIds: direct ? undefined : compactIds,
        output,
        extraCounts: drawInstanceCount ? [drawInstanceCount] : []
      })
    );
    return nodes;
  }

  /**
   * Sorted mode: optional order check, the range search, optional mask and fade classify, the
   * IDs kernel over the capacity, and the publish kernel (clamped count, overflow, draw count).
   */
  private getSortedCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const timestamps = props.timestamps as GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    const timestampsLow = props.timestampsLow as GraphDataView<'float32'> | undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (output.orderViolations) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-order-fill`,
          operation: 'GPUTimeWindowFilter',
          view: output.orderViolations,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
      if (timestamps.length >= 2) {
        nodes.push(
          createSortedTimeWindowOrderCheckNode<Parameters>(graph, {
            id: `${id}-order-check`,
            timestamps,
            timestampsLow,
            violations: output.orderViolations
          })
        );
      }
    }

    const range = output.rowRange ?? createTransientView(graph, `${id}-range`, 'uint32', 2);
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    nodes.push(
      createSortedTimeWindowRangeNode<Parameters>(graph, {
        id: `${id}-range-search`,
        timestamps,
        timestampsLow,
        window: props.window,
        range,
        total,
        firstInstance: output.drawFirstInstance
      })
    );

    if (output.outputMask || output.fadeWeights) {
      nodes.push(
        ...getTimeWindowClassifyNodes(graph, {
          id: `${id}-classify`,
          timestamps: props.timestamps,
          timestampsLow: props.timestampsLow,
          window: props.window,
          mask:
            output.outputMask ?? createTransientUint32Rows(graph, `${id}-time-mask`, timestamps),
          fadeWeights: output.fadeWeights
        })
      );
    }

    // With no rows the count is always 0 and there is nothing to write.
    if (output.ids.length > 0 && timestamps.length > 0) {
      nodes.push(
        createSortedTimeWindowIdsNode<Parameters>(graph, {
          id: `${id}-ids`,
          range,
          ids: output.ids,
          sourceIds: props.sourceIds as GraphDataView<'uint32'> | undefined
        })
      );
    }

    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUTimeWindowFilter',
        requiredCount: total,
        output,
        extraCounts: output.drawInstanceCount ? [output.drawInstanceCount] : []
      })
    );
    return nodes;
  }
}
