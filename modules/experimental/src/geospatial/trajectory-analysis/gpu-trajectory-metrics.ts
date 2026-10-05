// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUGroupAggregation,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {
  createTrajectoryCentroidOffsetsNode,
  createTrajectoryFinalizeNode,
  createTrajectoryQualifyNodes,
  createTrajectoryRowIdsNode,
  createTrajectorySplitPositionsNode,
  createTrajectoryStepsNodes,
  createTrajectoryStopGatherNodes,
  getSegmentedReductionNodes
} from './trajectory-metrics-kernels';
import {GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH} from './trajectory-metrics-parameters';

/**
 * Caller-owned, capacity-bounded stop (dwell) outputs of {@link GPUTrajectoryMetrics}.
 *
 * `output.ids.length` is the stop capacity, fixed at compile time. Every column that is present
 * must have exactly that many rows. Stops are ordered by start row, which is track order and then
 * time. Rows at and after `output.count` are rewritten with sentinels: `0xffffffff` for IDs and
 * rows, `0` for durations and centroids.
 */
export type GPUTrajectoryStopOutput = {
  /** Bounded compact result. `ids` holds the track index of each stop. */
  output: GPUCompactOutput;
  /** Optional first row of each dwell (the sample where the slow run begins). */
  startRows?: GraphDataView<'uint32'>;
  /** Optional last row of each dwell, inclusive. */
  endRows?: GraphDataView<'uint32'>;
  /** Optional mean position of the rows `startRow..endRow` of each dwell. */
  centroids?: GraphDataView<'float32x2'>;
  /** Optional `timestamps[endRow] - timestamps[startRow]` of each dwell. */
  durations?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUTrajectoryMetrics}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `trackOffsets` and
 * `parameters`. Compile-time (needs a new graph): view lengths, the track count
 * (`trackOffsets.length - 1`), the stop capacity (`stops.output.ids.length`), and which optional
 * views are present.
 */
export type GPUTrajectoryMetricsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-metrics'`. */
  id?: string;
  /** Packed planar positions, one row per sample, sorted by track and then time. Compile-time length. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Packed sample times, same length as `positions`. Three representations:
   *
   * - `float32` relative to an application epoch (default, cheapest). Absolute epoch times lose
   *   precision in f32.
   * - `float32` high parts with `timestampsLow` (double-single), for absolute times from
   *   `splitTimestamps`.
   * - `uint32x2` Int64 `(low, high)` words, for example an Arrow `Int64` epoch-ms column uploaded
   *   zero-copy. Every difference is subtracted exactly and then converted to f32.
   *
   * All durations, speeds, `stopMinimumDuration`, and `stopSpeedThreshold` use the timestamps'
   * unit (milliseconds for epoch-ms words).
   */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /**
   * Optional double-single low parts, aligned with `timestamps`, which must then be the f32 high
   * parts (see `splitTimestamps`). Differences are `(aHigh - bHigh) + (aLow - bLow)` and ordering
   * compares high parts first. Not allowed with `uint32x2` word timestamps.
   */
  timestampsLow?: GraphDataView<'float32'>;
  /**
   * `trackCount + 1` monotonic row offsets. Track `t` owns rows
   * `[trackOffsets[t], trackOffsets[t + 1])`. The length is compile-time and the contents are
   * per-frame. Rows outside `[trackOffsets[0], trackOffsets[trackCount])` are ignored.
   */
  trackOffsets: GraphDataView<'uint32'>;
  /**
   * Per-frame packed float32 view of at least {@link GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH}
   * elements written with `getGPUTrajectoryMetricsParameterValues`. Required when `stops` or
   * `trackStopCounts` is requested.
   */
  parameters?: GraphDataView<'float32'>;
  /** Optional per-track path length, `trackCount` rows. */
  trackLengths?: GraphDataView<'float32'>;
  /** Optional per-track duration `t[last] - t[first]`, `trackCount` rows. */
  trackDurations?: GraphDataView<'float32'>;
  /** Optional per-track `trackLength / trackDuration` (0 when the duration is not positive). */
  averageSpeeds?: GraphDataView<'float32'>;
  /** Optional per-track maximum step speed (0 without steps). */
  maximumSpeeds?: GraphDataView<'float32'>;
  /** Optional per-track number of qualifying stops, `trackCount` rows. Not clamped by the stop capacity. */
  trackStopCounts?: GraphDataView<'uint32'>;
  /** Optional bounded stop list. */
  stops?: GPUTrajectoryStopOutput;
};

/**
 * Computes per-track movement metrics and stop (dwell) detection for GPS-style tracks.
 *
 * Tracks are stored as rows sorted by track and then time, delimited by `trackOffsets`. For a
 * track with rows `[s, e)`:
 *
 * - Step `i` for `s < i < e` has `distance_i = length(p_i - p_{i-1})` and
 *   `deltaTime_i = t_i - t_{i-1}`.
 * - `trackLength` is the sum of step distances (0 with fewer than two rows). `trackDuration` is
 *   `t[e-1] - t[s]` (0 with fewer than two rows).
 * - `speed_i = distance_i / deltaTime_i` when `deltaTime_i > 0`, otherwise 0, so non-increasing
 *   timestamps never create speed spikes. `maximumSpeed` is the largest step speed and
 *   `averageSpeed` is `trackLength / trackDuration` when the duration is positive, otherwise 0.
 * - A step is slow when `deltaTime_i >= 0 && (distance_i == 0 || distance_i < stopSpeedThreshold * deltaTime_i)`.
 *   The comparison does not divide, and zero-distance steps (including duplicate samples) are
 *   always slow.
 * - A dwell is a maximal run of consecutive slow steps `a..b` inside one track. It covers rows
 *   `a - 1` through `b` and lasts `t_b - t_{a-1}`. It is a stop when that duration is at least
 *   `stopMinimumDuration`. Its centroid is the mean position of those `b - a + 2` rows. Dwells
 *   never merge across a track boundary.
 *
 * Composition: one per-row step kernel, `GPUSegmentedReduction` for lengths, maximum speeds and
 * centroid sums, one per-track finalize kernel, and for stops `GPUCompaction` of run starts and
 * ends, a per-run qualify kernel, a stable compaction of qualifying runs, gather kernels that fill
 * the bounded outputs, `GPUGroupAggregation` for per-track stop counts, and one publish kernel
 * that clamps the count. Stop thresholds are read from `parameters` every encoding.
 *
 * Limitations: segmented reductions are split into chunks of `maxComputeWorkgroupsPerDimension`
 * segments. Scratch memory is a handful of `uint32` or `float32` columns of one row each.
 *
 * Time: durations and speeds are in the timestamps' unit. With Int64 word timestamps every time
 * difference is an exact integer subtraction converted to f32 afterwards, so durations are exact
 * below 2^24 units and equal times give exactly 0; with double-single times differences are
 * `(aHigh - bHigh) + (aLow - bLow)` and the ordering test is exact. The step and qualify kernels
 * are split in two only when double-single times would push them past eight storage bindings.
 *
 * Non-goals: geodesic (haversine) distances (project upstream); map
 * matching; trajectory simplification; resampling; tracks given as unsorted rows (sort upstream
 * with `GPUSort` or derive offsets with `GPUSegmentedLayout` or `GPURunLengthEncode`); chunked
 * inputs.
 */
export class GPUTrajectoryMetrics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryMetricsProps;

  constructor(props: GPUTrajectoryMetricsProps) {
    this.id = props.id ?? 'trajectory-metrics';
    this.props = props;
    const id = this.id;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    if (props.timestampsLow) {
      if (props.timestamps.format !== 'float32') {
        throw new Error(`${id} timestampsLow requires float32 timestamps, not Int64 words`);
      }
      validatePackedView(props.timestampsLow, ['float32'], `${id} timestampsLow`);
      if (props.timestampsLow.length !== props.positions.length) {
        throw new Error(`${id} timestampsLow length must equal positions length`);
      }
    }
    validatePackedUint32View(props.trackOffsets, `${id} trackOffsets`);
    if (props.timestamps.length !== props.positions.length) {
      throw new Error(`${id} timestamps length must equal positions length`);
    }
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    const trackCount = props.trackOffsets.length - 1;
    for (const [name, view] of [
      ['trackLengths', props.trackLengths],
      ['trackDurations', props.trackDurations],
      ['averageSpeeds', props.averageSpeeds],
      ['maximumSpeeds', props.maximumSpeeds]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== trackCount) {
        throw new Error(`${id} ${name} length must equal the track count`);
      }
    }
    if (props.trackStopCounts) {
      validatePackedUint32View(props.trackStopCounts, `${id} trackStopCounts`);
      if (props.trackStopCounts.length !== trackCount) {
        throw new Error(`${id} trackStopCounts length must equal the track count`);
      }
    }
    const {stops} = props;
    if (stops) {
      validateCompactOutput(id, stops.output);
      const capacity = stops.output.ids.length;
      for (const [name, view, format] of [
        ['startRows', stops.startRows, 'uint32'],
        ['endRows', stops.endRows, 'uint32'],
        ['centroids', stops.centroids, 'float32x2'],
        ['durations', stops.durations, 'float32']
      ] as const) {
        if (!view) {
          continue;
        }
        validatePackedView(view, [format], `${id} stops.${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} stops.${name} length must equal the stop capacity`);
        }
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH} float32 values`
        );
      }
    }
    if ((stops || props.trackStopCounts) && !props.parameters) {
      throw new Error(`${id} parameters are required for stops and trackStopCounts`);
    }
    if (
      !(
        props.trackLengths ||
        props.trackDurations ||
        props.averageSpeeds ||
        props.maximumSpeeds ||
        props.trackStopCounts ||
        stops
      )
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.trackLengths,
        props.trackDurations,
        props.averageSpeeds,
        props.maximumSpeeds,
        props.trackStopCounts,
        stops?.output.ids,
        stops?.output.count,
        stops?.output.overflow,
        stops?.output.totalCount,
        stops?.startRows,
        stops?.endRows,
        stops?.centroids,
        stops?.durations
      ],
      [props.positions, props.timestamps, props.timestampsLow, props.trackOffsets, props.parameters]
    );
  }

  /**
   * Returns step, reduction, finalize, and (when stop outputs are requested) run-detection,
   * centroid, gather, count, and publish nodes in dependency order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {stops} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.timestamps,
      props.timestampsLow,
      props.trackOffsets,
      props.parameters,
      props.trackLengths,
      props.trackDurations,
      props.averageSpeeds,
      props.maximumSpeeds,
      props.trackStopCounts,
      stops?.output.ids,
      stops?.output.count,
      stops?.output.overflow,
      stops?.output.totalCount,
      stops?.startRows,
      stops?.endRows,
      stops?.centroids,
      stops?.durations
    ]);
    const rowCount = props.positions.length;
    const trackCount = props.trackOffsets.length - 1;
    const maximumSegmentCount = graph.device.limits.maxComputeWorkgroupsPerDimension;
    const hasStops = Boolean(stops || props.trackStopCounts);
    const nodes: GPUCommandNode<Parameters>[] = [];

    const needsLengths = Boolean(props.trackLengths || props.averageSpeeds);
    const stepDistances = needsLengths
      ? createTransientView(graph, `${id}-step-distances`, 'float32', rowCount)
      : undefined;
    const stepSpeeds = props.maximumSpeeds
      ? createTransientView(graph, `${id}-step-speeds`, 'float32', rowCount)
      : undefined;
    // Scratch for runs. Compaction outputs need input-length capacity; everything else per run.
    const runCapacity = Math.max(1, Math.ceil(rowCount / 2));
    const runStartFlags = hasStops
      ? createTransientView(graph, `${id}-run-start-flags`, 'uint32', rowCount)
      : undefined;
    const runEndFlags = hasStops
      ? createTransientView(graph, `${id}-run-end-flags`, 'uint32', rowCount)
      : undefined;

    nodes.push(
      ...createTrajectoryStepsNodes<Parameters>(graph, {
        id: `${id}-steps`,
        positions: props.positions,
        timestamps: props.timestamps,
        timestampsLow: props.timestampsLow,
        trackOffsets: props.trackOffsets,
        parameters: hasStops ? props.parameters : undefined,
        stepDistances,
        stepSpeeds,
        runStartFlags,
        runEndFlags
      })
    );

    const trackLengths =
      props.trackLengths ??
      (props.averageSpeeds
        ? createTransientView(graph, `${id}-lengths`, 'float32', trackCount)
        : undefined);
    if (stepDistances && trackLengths) {
      nodes.push(
        ...getSegmentedReductionNodes(
          graph,
          {
            id: `${id}-track-lengths`,
            input: stepDistances,
            segmentOffsets: props.trackOffsets,
            output: trackLengths,
            operation: 'sum'
          },
          maximumSegmentCount
        )
      );
    }
    if (stepSpeeds && props.maximumSpeeds) {
      nodes.push(
        ...getSegmentedReductionNodes(
          graph,
          {
            id: `${id}-maximum-speeds`,
            input: stepSpeeds,
            segmentOffsets: props.trackOffsets,
            output: props.maximumSpeeds,
            operation: 'max'
          },
          maximumSegmentCount
        )
      );
    }
    if (props.trackDurations || props.averageSpeeds) {
      nodes.push(
        createTrajectoryFinalizeNode<Parameters>(graph, {
          id: `${id}-track-finalize`,
          timestamps: props.timestamps,
          timestampsLow: props.timestampsLow,
          trackOffsets: props.trackOffsets,
          trackLengths,
          trackDurations: props.trackDurations,
          averageSpeeds: props.averageSpeeds
        })
      );
    }

    if (!hasStops || !runStartFlags || !runEndFlags || !props.parameters) {
      return nodes;
    }

    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rowCount);
    const runIndices = createTransientView(graph, `${id}-run-indices`, 'uint32', runCapacity);
    // Compaction outputs must hold input.length rows even though at most runCapacity are written.
    const runStarts = createTransientView(
      graph,
      `${id}-run-starts`,
      'uint32',
      Math.max(rowCount, 1)
    );
    const runEnds = createTransientView(graph, `${id}-run-ends`, 'uint32', Math.max(rowCount, 1));
    const runStartCount = createTransientView(graph, `${id}-run-start-count`, 'uint32', 1);
    const runEndCount = createTransientView(graph, `${id}-run-end-count`, 'uint32', 1);
    const qualifyFlags = createTransientView(graph, `${id}-qualify-flags`, 'uint32', runCapacity);
    const runTracks = createTransientView(graph, `${id}-run-tracks`, 'uint32', runCapacity);

    nodes.push(
      createTrajectoryRowIdsNode<Parameters>(graph, {
        id: `${id}-row-ids`,
        rowIds,
        rowCount,
        runIndices,
        runCapacity
      }),
      ...new GPUCompaction({
        id: `${id}-run-starts`,
        input: rowIds,
        flags: runStartFlags,
        output: runStarts,
        count: runStartCount
      }).getCommandNodes(graph),
      ...new GPUCompaction({
        id: `${id}-run-ends`,
        input: rowIds,
        flags: runEndFlags,
        output: runEnds,
        count: runEndCount
      }).getCommandNodes(graph),
      ...createTrajectoryQualifyNodes<Parameters>(graph, {
        id: `${id}-run-qualify`,
        timestamps: props.timestamps,
        timestampsLow: props.timestampsLow,
        trackOffsets: props.trackOffsets,
        parameters: props.parameters,
        runStarts,
        runEnds,
        runCount: runStartCount,
        qualifyFlags,
        runTracks,
        runCapacity
      })
    );

    let centroidSums:
      | {sumsX: GraphDataView<'float32'>; sumsY: GraphDataView<'float32'>}
      | undefined;
    if (stops?.centroids) {
      const centroidOffsets = createTransientView(
        graph,
        `${id}-centroid-offsets`,
        'uint32',
        2 * runCapacity + 1
      );
      const xs = createTransientView(graph, `${id}-xs`, 'float32', rowCount);
      const ys = createTransientView(graph, `${id}-ys`, 'float32', rowCount);
      const sumsX = createTransientView(graph, `${id}-sums-x`, 'float32', 2 * runCapacity);
      const sumsY = createTransientView(graph, `${id}-sums-y`, 'float32', 2 * runCapacity);
      nodes.push(
        createTrajectoryCentroidOffsetsNode<Parameters>(graph, {
          id: `${id}-centroid-offsets`,
          runStarts,
          runEnds,
          runCount: runStartCount,
          centroidOffsets,
          runCapacity,
          rowCount
        }),
        createTrajectorySplitPositionsNode<Parameters>(graph, {
          id: `${id}-split-positions`,
          positions: props.positions,
          xs,
          ys
        }),
        ...getSegmentedReductionNodes(
          graph,
          {
            id: `${id}-centroid-sums-x`,
            input: xs,
            segmentOffsets: centroidOffsets,
            output: sumsX,
            operation: 'sum'
          },
          maximumSegmentCount
        ),
        ...getSegmentedReductionNodes(
          graph,
          {
            id: `${id}-centroid-sums-y`,
            input: ys,
            segmentOffsets: centroidOffsets,
            output: sumsY,
            operation: 'sum'
          },
          maximumSegmentCount
        )
      );
      centroidSums = {sumsX, sumsY};
    }

    if (stops) {
      const qualifiedRuns = createTransientView(
        graph,
        `${id}-qualified-runs`,
        'uint32',
        runCapacity
      );
      const stopTotal = createTransientView(graph, `${id}-stop-total`, 'uint32', 1);
      nodes.push(
        ...new GPUCompaction({
          id: `${id}-stop-indices`,
          input: runIndices,
          flags: qualifyFlags,
          output: qualifiedRuns,
          count: stopTotal
        }).getCommandNodes(graph),
        ...createTrajectoryStopGatherNodes<Parameters>(graph, {
          id: `${id}-stop-gather`,
          capacity: stops.output.ids.length,
          stopTotal,
          qualifiedRuns,
          runStarts,
          runEnds,
          runTracks,
          timestamps: props.timestamps,
          timestampsLow: props.timestampsLow,
          sumsX: centroidSums?.sumsX,
          sumsY: centroidSums?.sumsY,
          ids: stops.output.ids,
          startRows: stops.startRows,
          endRows: stops.endRows,
          centroids: stops.centroids,
          durations: stops.durations
        })
      );
      if (props.trackStopCounts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-stop-counts`,
            keys: runTracks,
            output: props.trackStopCounts
          }).getCommandNodes(graph)
        );
      }
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: 'GPUTrajectoryMetrics',
          totalCount: stopTotal,
          output: stops.output
        })
      );
    } else if (props.trackStopCounts) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-stop-counts`,
          keys: runTracks,
          output: props.trackStopCounts
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
