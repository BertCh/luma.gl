// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUCompaction,
  GPUGroupAggregation,
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
  validateCompactOutput,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../analysis-core/contracts/index';
import {
  createTrajectoryCentroidOffsetsNode,
  createTrajectoryFinalizeNode,
  createTrajectoryQualifyNodes,
  createTrajectoryRowIdsNode,
  createTrajectoryStepColumnsNode,
  createTrajectorySplitPositionsNode,
  createTrajectoryStepsNodes,
  createTrajectoryStopGatherNodes,
  getSegmentedReductionNodes
} from './trajectory-metrics-kernels';
import {GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH} from './trajectory-metrics-parameters';
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';

/**
 * Caller-owned, capacity-bounded stop (dwell) outputs of {@link GPUTrajectoryMetrics}: a
 * {@link GPUCompactOutput} extended with optional per-stop columns.
 *
 * `ids.length` is the stop capacity, fixed at compile time. Every column that is present must have
 * exactly that many rows. Stops are ordered by start row, which is track order and then time. Rows
 * at and after `count` are rewritten with sentinels: `0xffffffff` for IDs and rows, `0` for
 * durations and centroids. `ids` holds the track index of each stop.
 */
export type GPUTrajectoryStopOutput = GPUCompactOutput & {
  /**
   * Optional one row receiving the clamped stop count (`count`), typically an indirect draw
   * record's `instanceCount` imported with `graph.importGPUData(id, drawCommands.getInstanceCountData(0))`,
   * so a stop layer needs no count copy.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
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
 * Caller-owned outputs of {@link GPUTrajectoryMetrics}. At least one view must be present.
 * Status: the stop list reports capacity overflow through `stops.overflow`.
 */
export type GPUTrajectoryMetricsOutput = {
  /** Optional per-track path length, `trackCount` rows. */
  trackLengths?: GraphDataView<'float32'>;
  /** Optional per-track duration `t[last] - t[first]`, `trackCount` rows. */
  trackDurations?: GraphDataView<'float32'>;
  /** Optional per-track `trackLength / trackDuration` (0 when the duration is not positive). */
  averageSpeeds?: GraphDataView<'float32'>;
  /** Optional per-track maximum step speed (0 without steps). */
  maximumSpeeds?: GraphDataView<'float32'>;
  /**
   * Optional per-row (per-vertex) speed of the step that ends at the row, `positions.length`
   * rows: `distance / deltaTime` when `deltaTime > 0`, otherwise 0. The first row of every track
   * and rows outside every track are 0. A segment between rows `i - 1` and `i` takes its value
   * from row `i`.
   */
  stepSpeeds?: GraphDataView<'float32'>;
  /**
   * Optional per-row heading in radians of the step that ends at the row, `positions.length`
   * rows, always in `(-pi, pi]`. Planar: `atan2(deltaY, deltaX)`, counter-clockwise from +x.
   * Great-circle: the initial bearing of the step in the same convention, counterclockwise from
   * east (`pi / 2 - bearing`, where `GPUGeodesicPairs` reports `bearing` in degrees clockwise from
   * north), so the column means the same in every context and matches a run on positions
   * projected with `createGPULocalFrame`. Zero-distance steps, the first row of every track and
   * rows outside every track are 0.
   */
  stepHeadings?: GraphDataView<'float32'>;
  /**
   * Optional per-row acceleration of the step that ends at the row, `positions.length` rows:
   * `(speed_i - speed_{i-1}) / deltaTime_i` in position units per time unit squared. 0 for the
   * first two rows of every track, rows outside every track, and steps with a non-positive
   * `deltaTime`.
   */
  stepAccelerations?: GraphDataView<'float32'>;
  /** Optional per-track number of qualifying stops, `trackCount` rows. Not clamped by the stop capacity. */
  trackStopCounts?: GraphDataView<'uint32'>;
  /** Optional bounded stop list. */
  stops?: GPUTrajectoryStopOutput;
};

/**
 * Properties for {@link GPUTrajectoryMetrics}: the trajectory layout ({@link GPUTrajectoryGeometryFields}:
 * `positions`, `timestamps`, `trackOffsets`) plus the options below.
 *
 * `positions` are packed rows sorted by track and then time, planar `(x, y)` or geographic
 * `(longitude, latitude)` degrees, see `spatialContext`. `timestamps` have three representations:
 *
 * - `float32` relative to an application epoch (default, cheapest). Absolute epoch times lose
 *   precision in f32.
 * - `float32` high parts with `timestampsLow` (double-single), for absolute times from
 *   `splitTimestamps`.
 * - `uint32x2` Int64 `(low, high)` words, for example an Arrow `Int64` epoch-ms column uploaded
 *   zero-copy. Every difference is subtracted exactly and then converted to f32.
 *
 * All durations, speeds, `stopMinimumDuration`, and `stopSpeedThreshold` use the timestamps'
 * unit (milliseconds for epoch-ms words). `trackOffsets` are `trackCount + 1` monotonic row
 * offsets. Track `t` owns rows `[trackOffsets[t], trackOffsets[t + 1])`. Rows outside
 * `[trackOffsets[0], trackOffsets[trackCount])` are ignored.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `trackOffsets` and
 * `parameters`. Compile-time (needs a new graph): view lengths, the track count
 * (`trackOffsets.length - 1`), the stop capacity (`output.stops.ids.length`), and which optional
 * views are present.
 */
export type GPUTrajectoryMetricsProps = GPUTrajectoryGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-metrics'`. */
  id?: string;
  /**
   * Coordinate space and metric of the positions. Omitted means native planar coordinates and
   * Euclidean distances in position units. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` means positions are `(longitude, latitude)` degrees and distances are
   * great-circle meters on a sphere of `sphereRadius` (default `GPU_EARTH_MEAN_RADIUS_METERS`),
   * so speeds are meters per timestamp unit. Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Optional double-single low parts, aligned with `timestamps`, which must then be the f32 high
   * parts (see `splitTimestamps`). Differences are `(aHigh - bHigh) + (aLow - bLow)` and ordering
   * compares high parts first. Not allowed with `uint32x2` word timestamps.
   */
  timestampsLow?: GraphDataView<'float32'>;
  /**
   * Per-frame packed float32 view of at least {@link GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH}
   * elements written with `getGPUTrajectoryMetricsParameterValues`. Required when `output.stops`
   * or `output.trackStopCounts` is requested.
   */
  parameters?: GraphDataView<'float32'>;
  /** Caller-owned output views. */
  output: GPUTrajectoryMetricsOutput;
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
 * - Optional per-row columns describe the step that ends at each row: `stepSpeeds`,
 *   `stepHeadings` (`atan2(deltaY, deltaX)`, 0 for zero-distance steps) and `stepAccelerations`
 *   (`(speed_i - speed_{i-1}) / deltaTime_i`, 0 without a previous step in the track). They are
 *   0 for the first row of a track, so a drawn segment takes the value of its end row.
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
 * Spatial context: without `spatialContext` (or with a planar one) distances are Euclidean in
 * position units and headings are `atan2(deltaY, deltaX)`. With `{coordinateSpace:
 * 'longitude-latitude', metric: 'great-circle'}`, positions are `(longitude, latitude)` degrees,
 * step distances are great-circle meters (`geodesicHaversineDistance` on `sphereRadius`, default
 * `GPU_EARTH_MEAN_RADIUS_METERS`), speeds are meters per timestamp unit, and headings are initial
 * bearings expressed counterclockwise from east (`pi / 2 - bearing`) in `(-pi, pi]`. Stop `centroids` are arithmetic means of
 * the coordinates, so a dwell that straddles the antimeridian or a pole has a meaningless centroid
 * in great-circle mode. Positions are f32 degrees (about 1 m resolution at mid-latitudes).
 *
 * Non-goals: ellipsoidal (Vincenty) distances; map
 * matching; trajectory simplification; resampling; tracks given as unsorted rows (sort upstream
 * with `GPUSort` or derive offsets with `GPUSegmentedLayout` or `GPURunLengthEncode`); chunked
 * inputs.
 */
export class GPUTrajectoryMetrics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryMetricsProps;
  /** Coordinate and metric contract shared with downstream movement contributors. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUTrajectoryMetricsProps) {
    this.id = props.id ?? 'trajectory-metrics';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT
    );
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
    validatePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
    if (props.timestamps.length !== props.positions.length) {
      throw new Error(`${id} timestamps length must equal positions length`);
    }
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    const trackCount = props.trackOffsets.length - 1;
    const {output} = props;
    for (const [name, view] of [
      ['trackLengths', output.trackLengths],
      ['trackDurations', output.trackDurations],
      ['averageSpeeds', output.averageSpeeds],
      ['maximumSpeeds', output.maximumSpeeds]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== trackCount) {
        throw new Error(`${id} ${name} length must equal the track count`);
      }
    }
    for (const [name, view] of [
      ['stepSpeeds', output.stepSpeeds],
      ['stepHeadings', output.stepHeadings],
      ['stepAccelerations', output.stepAccelerations]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['float32'], `${id} ${name}`);
      if (view.length !== props.positions.length) {
        throw new Error(`${id} ${name} length must equal positions length`);
      }
    }
    if (output.trackStopCounts) {
      validatePackedView(output.trackStopCounts, ['uint32'], `${id} trackStopCounts`);
      if (output.trackStopCounts.length !== trackCount) {
        throw new Error(`${id} trackStopCounts length must equal the track count`);
      }
    }
    const {stops} = output;
    if (stops) {
      validateCompactOutput(id, stops);
      if (stops.drawInstanceCount) {
        validatePackedView(stops.drawInstanceCount, ['uint32'], `${id} stops.drawInstanceCount`);
        if (stops.drawInstanceCount.length < 1) {
          throw new Error(`${id} stops.drawInstanceCount must contain one uint32 row`);
        }
      }
      const capacity = stops.ids.length;
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
    if ((stops || output.trackStopCounts) && !props.parameters) {
      throw new Error(`${id} parameters are required for stops and trackStopCounts`);
    }
    if (
      !(
        output.trackLengths ||
        output.trackDurations ||
        output.averageSpeeds ||
        output.maximumSpeeds ||
        output.trackStopCounts ||
        output.stepSpeeds ||
        output.stepHeadings ||
        output.stepAccelerations ||
        stops
      )
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.trackLengths,
        output.trackDurations,
        output.averageSpeeds,
        output.maximumSpeeds,
        output.stepSpeeds,
        output.stepHeadings,
        output.stepAccelerations,
        output.trackStopCounts,
        stops?.ids,
        stops?.status.count,
        stops?.status.overflow,
        stops?.status.requiredCount,
        stops?.drawInstanceCount,
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
    const {output} = props;
    const {stops} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.timestamps,
      props.timestampsLow,
      props.trackOffsets,
      props.parameters,
      output.trackLengths,
      output.trackDurations,
      output.averageSpeeds,
      output.maximumSpeeds,
      output.stepSpeeds,
      output.stepHeadings,
      output.stepAccelerations,
      output.trackStopCounts,
      stops?.ids,
      stops?.status.count,
      stops?.status.overflow,
      stops?.status.requiredCount,
      stops?.drawInstanceCount,
      stops?.startRows,
      stops?.endRows,
      stops?.centroids,
      stops?.durations
    ]);
    const rowCount = props.positions.length;
    const trackCount = props.trackOffsets.length - 1;
    const maximumSegmentCount = graph.device.limits.maxComputeWorkgroupsPerDimension;
    const hasStops = Boolean(stops || output.trackStopCounts);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const geodesicRadius = getGPUGreatCircleRadius(this.spatialContext);

    const needsLengths = Boolean(output.trackLengths || output.averageSpeeds);
    const stepDistances = needsLengths
      ? createTransientView(graph, `${id}-step-distances`, 'float32', rowCount)
      : undefined;
    const stepSpeeds = output.maximumSpeeds
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
        geodesicRadius,
        stepDistances,
        stepSpeeds,
        runStartFlags,
        runEndFlags
      })
    );

    if (output.stepSpeeds || output.stepHeadings || output.stepAccelerations) {
      nodes.push(
        createTrajectoryStepColumnsNode<Parameters>(graph, {
          id: `${id}-step-columns`,
          positions: props.positions,
          timestamps: props.timestamps,
          timestampsLow: props.timestampsLow,
          trackOffsets: props.trackOffsets,
          geodesicRadius,
          stepSpeeds: output.stepSpeeds,
          stepHeadings: output.stepHeadings,
          stepAccelerations: output.stepAccelerations
        })
      );
    }

    const trackLengths =
      output.trackLengths ??
      (output.averageSpeeds
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
    if (stepSpeeds && output.maximumSpeeds) {
      nodes.push(
        ...getSegmentedReductionNodes(
          graph,
          {
            id: `${id}-maximum-speeds`,
            input: stepSpeeds,
            segmentOffsets: props.trackOffsets,
            output: output.maximumSpeeds,
            operation: 'max'
          },
          maximumSegmentCount
        )
      );
    }
    if (output.trackDurations || output.averageSpeeds) {
      nodes.push(
        createTrajectoryFinalizeNode<Parameters>(graph, {
          id: `${id}-track-finalize`,
          timestamps: props.timestamps,
          timestampsLow: props.timestampsLow,
          trackOffsets: props.trackOffsets,
          trackLengths,
          trackDurations: output.trackDurations,
          averageSpeeds: output.averageSpeeds
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
          capacity: stops.ids.length,
          stopTotal,
          qualifiedRuns,
          runStarts,
          runEnds,
          runTracks,
          timestamps: props.timestamps,
          timestampsLow: props.timestampsLow,
          sumsX: centroidSums?.sumsX,
          sumsY: centroidSums?.sumsY,
          ids: stops.ids,
          startRows: stops.startRows,
          endRows: stops.endRows,
          centroids: stops.centroids,
          durations: stops.durations
        })
      );
      if (output.trackStopCounts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-stop-counts`,
            keys: runTracks,
            output: output.trackStopCounts
          }).getCommandNodes(graph)
        );
      }
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: 'GPUTrajectoryMetrics',
          requiredCount: stopTotal,
          output: stops,
          extraCounts: stops.drawInstanceCount ? [stops.drawInstanceCount] : []
        })
      );
    } else if (output.trackStopCounts) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-stop-counts`,
          keys: runTracks,
          output: output.trackStopCounts
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
