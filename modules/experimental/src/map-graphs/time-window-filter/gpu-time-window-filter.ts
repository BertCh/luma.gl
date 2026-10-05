// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUVisibilityPredicate,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUMapGraphCompactOutput,
  GPUMapGraphRecipe,
  GPUMapGraphUint32Rows
} from '../map-graph-types';
import {createMapGraphPublishNode, createTransientUint32Rows} from '../map-graph-kernels';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGraphViewTopology,
  validateMapGraphCompactOutput
} from '../map-graph-utils';
import {getTimeWindowClassifyNodes} from './time-window-classify-node';
import {GPU_TIME_WINDOW_PARAMETER_LENGTH} from './time-window-parameters';
import {GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH, type GPUInt64TimeWordRows} from './time-words';

type Float32Rows = GraphDataView<'float32'> | GraphVectorView<'float32'>;

/**
 * Properties for {@link GPUTimeWindowFilter}.
 *
 * Per-frame (no recompile): the contents of `window` and of every input buffer. Topology (needs a
 * new graph): view lengths and chunking, `output.ids.length`, `trackVisibleCounts.length`, which
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
   * {@link GPU_TIME_WINDOW_PARAMETER_LENGTH} elements written with `getGPUTimeWindowParameterValues`.
   * Word timestamps: packed `uint32` view of at least {@link GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH}
   * elements written with `getGPUTimeWindowWordParameterValues`.
   */
  window: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Optional extra source-aligned predicate masks (bounds, LOD, selection) ANDed with time. */
  additionalPredicates?: readonly GPUVisibilityPredicate[];
  /** Optional stable IDs aligned with rows. Zero-based row indices are emitted when omitted. */
  sourceIds?: GPUMapGraphUint32Rows;
  /** Caller-owned bounded result. `output.ids.length` is the capacity. */
  output: GPUMapGraphCompactOutput;
  /** Optional caller-owned canonical 0/1 mask of accepted rows (time AND additional predicates). */
  outputMask?: GPUMapGraphUint32Rows;
  /** Optional per-row fade weight in `[0, 1]`, 0 for rejected rows. Reflects the time test only. */
  fadeWeights?: Float32Rows;
  /**
   * Optional per-row `[clipStart, clipEnd]` fractions of each accepted interval inside the window.
   * Rejected rows get `[0, 0]`. Requires `endTimestamps`. Reflects the time test only.
   */
  clipFractions?: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /** Optional dense track IDs per row, for example the track that owns each segment. */
  trackIds?: GPUMapGraphUint32Rows;
  /** Optional per-track count of accepted rows. Track IDs `>= length` are ignored. Requires `trackIds`. */
  trackVisibleCounts?: GraphDataView<'uint32'>;
  /**
   * Optional packed one-row view that receives the clamped count, typically an indirect draw
   * record's `instanceCount` imported with `graph.importGPUData(id, drawCommands.getInstanceCountData(0))`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
};

/**
 * Filters timestamped rows or time intervals against a per-frame time window.
 *
 * Publishes stable compacted IDs, a clamped count with overflow, and optional mask, fade-weight,
 * trail-clip, per-track-count, and indirect-draw outputs, without recompiling the graph when the
 * window moves. Composition: one classify kernel, `GPUVisibilityWorkflow`, optional
 * `GPUGroupAggregation`, and one publish kernel that clamps the count.
 *
 * Time input modes:
 * - float32 (cheap path): relative times, optionally with `timestampsLow` for double-single
 *   precision. Window written with `getGPUTimeWindowParameterValues`.
 * - exact Int64 words: `uint32x2` rows `(low, high)` such as Arrow `Int64` epoch milliseconds, with
 *   a `uint32` window written with `getGPUTimeWindowWordParameterValues`. Acceptance comparisons
 *   are exact for any Int64 value and any playhead fraction. Fade and clip differences are
 *   subtracted exactly and then rounded to f32: exact below 2^24 units, rounded above. There is no
 *   epoch to rebase. `timestampsLow` is not allowed in this mode.
 */
export class GPUTimeWindowFilter implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'time-window-filter';
  /** Validated properties. */
  readonly props: GPUTimeWindowFilterProps;

  constructor(props: GPUTimeWindowFilterProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    const rows = props.timestamps.length;

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
      ['fadeWeights', props.fadeWeights]
    ] as const) {
      const isTimeWords = isWordMode && (name === 'timestamps' || name === 'endTimestamps');
      for (const chunk of view ? getGraphViewChunks(view as GraphDataView | GraphVectorView) : []) {
        validatePackedView(chunk, [isTimeWords ? 'uint32x2' : 'float32'], `${id} ${name}`);
      }
      if (name !== 'timestamps') {
        validateGraphViewTopology(id, name, props.timestamps, view);
      }
    }
    for (const chunk of props.clipFractions ? getGraphViewChunks(props.clipFractions) : []) {
      validatePackedView(chunk, ['float32x2'], `${id} clipFractions`);
    }
    validateGraphViewTopology(id, 'clipFractions', props.timestamps, props.clipFractions);
    for (const chunk of props.outputMask ? getGraphViewChunks(props.outputMask) : []) {
      validatePackedUint32View(chunk, `${id} outputMask`);
    }
    validateGraphViewTopology(id, 'outputMask', props.timestamps, props.outputMask);

    if (isWordMode) {
      if (props.window.format !== 'uint32') {
        throw new Error(`${id} uint32x2 word timestamps require a packed uint32 word window`);
      }
      validatePackedView(props.window, ['uint32'], `${id} window`);
      if (props.window.length < GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH) {
        throw new Error(
          `${id} window must hold ${GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH} uint32 values`
        );
      }
    } else {
      if (props.window.format !== 'float32') {
        throw new Error(`${id} float32 timestamps require a float32 window`);
      }
      validatePackedView(props.window, ['float32'], `${id} window`);
      if (props.window.length < GPU_TIME_WINDOW_PARAMETER_LENGTH) {
        throw new Error(
          `${id} window must hold ${GPU_TIME_WINDOW_PARAMETER_LENGTH} float32 values`
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
    if (props.clipFractions && !props.endTimestamps) {
      throw new Error(`${id} clipFractions requires endTimestamps`);
    }
    if (Boolean(props.trackIds) !== Boolean(props.trackVisibleCounts)) {
      throw new Error(`${id} trackIds and trackVisibleCounts must be given together`);
    }
    if (props.trackVisibleCounts) {
      validatePackedUint32View(props.trackVisibleCounts, `${id} trackVisibleCounts`);
      if (props.trackVisibleCounts.length < 1) {
        throw new Error(`${id} trackVisibleCounts must contain at least one track`);
      }
    }
    validateMapGraphCompactOutput(id, props.output);
    if (props.drawInstanceCount) {
      validatePackedUint32View(props.drawInstanceCount, `${id} drawInstanceCount`);
      if (props.drawInstanceCount.length < 1) {
        throw new Error(`${id} drawInstanceCount must contain one uint32 row`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.ids,
        props.output.count,
        props.output.overflow,
        props.output.totalCount,
        props.outputMask,
        props.fadeWeights,
        props.clipFractions,
        props.trackVisibleCounts,
        props.drawInstanceCount
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

  /** Returns classify, visibility, optional track-count, and publish nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output, outputMask, trackIds, trackVisibleCounts, drawInstanceCount} = props;
    const additionalPredicates = props.additionalPredicates ?? [];
    validateGraphViewsBelongToGraph(id, graph, [
      props.timestamps,
      props.timestampsLow,
      props.endTimestamps,
      props.endTimestampsLow,
      props.window,
      props.sourceIds,
      outputMask,
      props.fadeWeights,
      props.clipFractions,
      trackIds,
      trackVisibleCounts,
      drawInstanceCount,
      output.ids,
      output.count,
      output.overflow,
      output.totalCount,
      ...additionalPredicates.map(predicate => predicate.mask)
    ]);
    const rows = props.timestamps.length;
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
        fadeWeights: props.fadeWeights,
        clipFractions: props.clipFractions
      })
    );

    let finalMask: GPUMapGraphUint32Rows | undefined = timeMask;
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
      createMapGraphPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUTimeWindowFilter',
        totalCount: total,
        compactIds: direct ? undefined : compactIds,
        output,
        extraCounts: drawInstanceCount ? [drawInstanceCount] : []
      })
    );
    return nodes;
  }
}
