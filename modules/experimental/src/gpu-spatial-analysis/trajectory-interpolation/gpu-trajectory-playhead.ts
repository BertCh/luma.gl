// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createTrajectoryBracketNode,
  createTrajectoryColumnsNode,
  createTrajectoryGeometryNode,
  TRAJECTORY_BRACKET_STRIDE
} from './trajectory-interpolation-kernels';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH} from './trajectory-playhead-parameters';
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';

/**
 * Largest track count (`trackOffsets.length - 1`) of {@link GPUTrajectoryPlayhead} on `device`: the
 * internal bracket scratch keeps 5 uint32 words (20 bytes) per track, read by every output kernel,
 * in one storage binding, so 6710886 on a default device (128 MiB bindings).
 */
export function getGPUTrajectoryPlayheadMaximumTrackCount(device: Device): number {
  return getStorageBindingElementLimit(
    device,
    TRAJECTORY_BRACKET_STRIDE * Uint32Array.BYTES_PER_ELEMENT
  );
}

/**
 * Caller-owned outputs of {@link GPUTrajectoryPlayhead}. At least one view must be present.
 */
export type GPUTrajectoryPlayheadOutput = {
  /** Optional interpolated position per track, `trackCount` rows. */
  currentPositions?: GraphDataView<'float32x2'>;
  /** Optional interpolated elevation per track. Requires `elevations`. */
  currentElevations?: GraphDataView<'float32'>;
  /**
   * Optional heading per track in radians, in `(-pi, pi]`. Planar: `atan2(dy, dx)` of the
   * bracketing segment, 0 along +x, growing counterclockwise. Great-circle: the initial bearing of
   * the bracketing segment in the same convention (0 due east, growing counterclockwise, i.e.
   * `pi / 2 - bearing`). 0 for an empty track or a zero-length
   * segment.
   */
  headings?: GraphDataView<'float32'>;
  /**
   * Optional speed of the bracketing segment for active tracks, otherwise 0: planar distance, or
   * great-circle meters, per timestamp unit.
   */
  speeds?: GraphDataView<'float32'>;
  /** Optional `GPU_TRAJECTORY_PLAYHEAD_STATUS` value per track, `trackCount` rows. */
  trackStatus?: GraphDataView<'uint32'>;
  /** Optional first row of the bracketing segment per track, `0xffffffff` for an empty track. */
  segmentRows?: GraphDataView<'uint32'>;
  /** Optional interpolation fraction inside the bracketing segment, in `[0, 1]`. */
  segmentFractions?: GraphDataView<'float32'>;
  /** Optional compact, ascending list of active track indices with a clamped count. */
  activeTracks?: GPUCompactOutput;
  /**
   * Optional packed one-row view that receives the clamped active count, typically an indirect draw
   * record's `instanceCount`. Requires `activeTracks`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTrajectoryPlayhead}: the trajectory layout
 * ({@link GPUTrajectoryGeometryFields}: `positions`, `timestamps`, `trackOffsets`) plus the options
 * below.
 *
 * `positions` are packed rows (planar, or longitude/latitude degrees), sorted by track and then
 * time. `timestamps` have the same length as `positions` and are non-decreasing inside each track:
 *
 * - `float32` relative to an application epoch, with a float32 `parameters` view.
 * - `uint32x2` Int64 `(low, high)` words such as an Arrow `Int64` epoch-ms column
 *   (`getInt64TimeWords`), with a uint32 `parameters` view. Comparisons with the playhead are
 *   exact; differences are subtracted exactly and then rounded to f32.
 *
 * `trackOffsets` are `trackCount + 1` monotonic row offsets; track `t` owns rows
 * `[trackOffsets[t], trackOffsets[t + 1])`. Offsets past `positions.length` are clamped.
 *
 * Per-frame (no recompile): the contents of `parameters` (playhead and `maximumGap`) and of every
 * input buffer, including `trackOffsets`. Compile-time (needs a new graph): view lengths, the
 * track count (`trackOffsets.length - 1`, at most {@link getGPUTrajectoryPlayheadMaximumTrackCount}), the time format, `output.activeTracks.ids.length`, and
 * which optional views are present.
 */
export type GPUTrajectoryPlayheadProps = GPUTrajectoryGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-playhead'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Omitted means native planar coordinates with
   * linear interpolation. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` means
   * `(longitude, latitude)` degrees interpolated along great circles (spherical linear
   * interpolation) on a sphere of `sphereRadius` (default `GPU_EARTH_MEAN_RADIUS_METERS`); lengths
   * and speeds are then meters. Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /** Optional packed elevation (z) per sample, same length as `positions`. */
  elevations?: GraphDataView<'float32'>;
  /**
   * Per-frame playhead, at least {@link GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH} elements: float32
   * from `getGPUTrajectoryPlayheadParameterValues` for float32 timestamps, or uint32 from
   * `getGPUTrajectoryPlayheadWordParameterValues` for word timestamps.
   */
  parameters: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Caller-owned output views. */
  output: GPUTrajectoryPlayheadOutput;
};

/**
 * Interpolates every track at a per-frame playhead time: the "current positions" of a trips map.
 *
 * One invocation per track runs an upper-bound binary search over the track's sorted timestamps
 * (first row whose time is greater than the playhead) and interpolates the bracketing segment
 * `[segmentRow, segmentRow + 1]`. Rules, for a track with rows `[s, e)` and playhead `p`:
 *
 * - Empty track (`e == s`): status `empty`, zero position, heading, and speed.
 * - `p < t[s]`: `beforeStart`, position of the first sample, heading of the first segment.
 * - `p > t[e - 1]`: `afterEnd`, position of the last sample, heading of the last segment.
 * - Otherwise `active`: the segment is `[j - 1, j]` with `t[j - 1] <= p < t[j]`, or the last
 *   segment when `p == t[e - 1]`. The fraction is `(p - t[j - 1]) / (t[j] - t[j - 1])`. A single
 *   sample track is active only at exactly its time.
 * - Gap: when `maximumGap > 0` and the playhead lies strictly inside a bracketing interval longer than
 *   `maximumGap`, the status is `gap` and the position holds at the last fix before the gap.
 * - Duplicate timestamps: the search picks the last of several rows with equal time, so at exactly
 *   that time the track reports the last duplicate ("latest row wins"), and the interval used for
 *   interpolation always has a positive duration. Zero-duration segments never divide.
 * - Exact sample times: `p == t[k]` gives fraction 0 on segment `[k, k + 1]` and position
 *   `positions[k]` bit-exactly.
 *
 * Speed is segment length (planar, or great-circle meters) over duration in the timestamps' unit
 * and is only non-zero for active tracks.
 *
 * Spatial context: with a great-circle `spatialContext`, positions are interpolated along the
 * great-circle arc of each segment instead of linearly in degrees. Interpolated longitudes are
 * continuous with the segment's start row, so a segment crossing the antimeridian produces
 * longitudes beyond +-180 (`end = b` shifted by a multiple of 360) rather than jumping; wrap them
 * downstream if needed. Fraction 0 returns the start row exactly and fraction 1 the end row
 * exactly (up to that 360-degree shift).
 *
 * Heading and speed come from the bracketing segment, so they are piecewise constant.
 *
 * Composition: one search kernel writing an internal per-track bracket, one geometry kernel, one
 * column kernel, and for `activeTracks` a stable `GPUCompaction` of active track indices and one
 * publish kernel that clamps the count. Moving the playhead only rewrites `parameters`.
 *
 * Non-goals: ellipsoidal (geodesic on the ellipsoid) interpolation, curved
 * (spline) interpolation, unsorted rows, chunked inputs, double-single timestamps.
 */
export class GPUTrajectoryPlayhead implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryPlayheadProps;
  /** Resolved coordinate and metric context; native planar when `props.spatialContext` is omitted. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUTrajectoryPlayheadProps) {
    this.id = props.id ?? 'trajectory-playhead';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT
    );
    const {output} = props;
    const rowCount = props.positions.length;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
    if (props.timestamps.length !== rowCount) {
      throw new Error(`${id} timestamps length must equal positions length`);
    }
    if (props.elevations) {
      validatePackedView(props.elevations, ['float32'], `${id} elevations`);
      if (props.elevations.length !== rowCount) {
        throw new Error(`${id} elevations length must equal positions length`);
      }
    }
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    const isWordMode = props.timestamps.format === 'uint32x2';
    const parameterFormat = isWordMode ? 'uint32' : 'float32';
    if (props.parameters.format !== parameterFormat) {
      throw new Error(
        `${id} ${isWordMode ? 'uint32x2 word' : 'float32'} timestamps require ${parameterFormat} parameters`
      );
    }
    validatePackedView(props.parameters, [parameterFormat], `${id} parameters`);
    if (props.parameters.length < GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH} ${parameterFormat} values`
      );
    }
    const trackCount = props.trackOffsets.length - 1;
    for (const [name, view, format] of [
      ['currentPositions', output.currentPositions, 'float32x2'],
      ['currentElevations', output.currentElevations, 'float32'],
      ['headings', output.headings, 'float32'],
      ['speeds', output.speeds, 'float32'],
      ['trackStatus', output.trackStatus, 'uint32'],
      ['segmentRows', output.segmentRows, 'uint32'],
      ['segmentFractions', output.segmentFractions, 'float32']
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} ${name}`);
      if (view.length !== trackCount) {
        throw new Error(`${id} ${name} length must equal the track count`);
      }
    }
    if (output.currentElevations && !props.elevations) {
      throw new Error(`${id} currentElevations requires elevations`);
    }
    if (output.activeTracks) {
      validateCompactOutput(id, output.activeTracks);
    }
    if (output.drawInstanceCount) {
      if (!output.activeTracks) {
        throw new Error(`${id} drawInstanceCount requires activeTracks`);
      }
      validatePackedView(output.drawInstanceCount, ['uint32'], `${id} drawInstanceCount`);
      if (output.drawInstanceCount.length < 1) {
        throw new Error(`${id} drawInstanceCount must contain one uint32 row`);
      }
    }
    if (
      !(
        output.currentPositions ||
        output.currentElevations ||
        output.headings ||
        output.speeds ||
        output.trackStatus ||
        output.segmentRows ||
        output.segmentFractions ||
        output.activeTracks
      )
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), [
      props.positions,
      props.elevations,
      props.timestamps,
      props.trackOffsets,
      props.parameters
    ]);
  }

  /** Returns the search, geometry, column, and (with `activeTracks`) compaction and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.elevations,
      props.timestamps,
      props.trackOffsets,
      props.parameters,
      ...this.getOutputViews()
    ]);
    const rowCount = props.positions.length;
    const trackCount = props.trackOffsets.length - 1;
    validateMaximumCount(
      id,
      'tracks',
      trackCount,
      getGPUTrajectoryPlayheadMaximumTrackCount(graph.device)
    );
    const bracket = createTransientView(
      graph,
      `${id}-bracket`,
      'uint32',
      TRAJECTORY_BRACKET_STRIDE * trackCount
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      createTrajectoryBracketNode<Parameters>(graph, {
        id: `${id}-bracket`,
        timestamps: props.timestamps,
        trackOffsets: props.trackOffsets,
        parameters: props.parameters,
        bracket,
        rowCount,
        trackCount
      })
    ];
    if (output.currentPositions || output.currentElevations || output.headings || output.speeds) {
      nodes.push(
        createTrajectoryGeometryNode<Parameters>(graph, {
          id: `${id}-geometry`,
          bracket,
          positions: props.positions,
          elevations: props.elevations,
          currentPositions: output.currentPositions,
          currentElevations: output.currentElevations,
          headings: output.headings,
          speeds: output.speeds,
          geodesicRadius: getGPUGreatCircleRadius(this.spatialContext),
          trackCount
        })
      );
    }
    const {activeTracks} = output;
    const activeFlags = activeTracks
      ? createTransientView(graph, `${id}-active-flags`, 'uint32', trackCount)
      : undefined;
    const trackIndices = activeTracks
      ? createTransientView(graph, `${id}-track-indices`, 'uint32', trackCount)
      : undefined;
    if (output.trackStatus || output.segmentRows || output.segmentFractions || activeTracks) {
      nodes.push(
        createTrajectoryColumnsNode<Parameters>(graph, {
          id: `${id}-columns`,
          bracket,
          status: output.trackStatus,
          segmentRows: output.segmentRows,
          segmentFractions: output.segmentFractions,
          activeFlags,
          trackIndices,
          trackCount
        })
      );
    }
    if (activeTracks && activeFlags && trackIndices) {
      const compactIds = createTransientView(graph, `${id}-active-ids`, 'uint32', trackCount);
      const activeTotal = createTransientView(graph, `${id}-active-total`, 'uint32', 1);
      nodes.push(
        ...new GPUCompaction({
          id: `${id}-active`,
          input: trackIndices,
          flags: activeFlags,
          output: compactIds,
          count: activeTotal
        }).getCommandNodes(graph),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: 'GPUTrajectoryPlayhead',
          requiredCount: activeTotal,
          compactIds,
          output: activeTracks,
          extraCounts: output.drawInstanceCount ? [output.drawInstanceCount] : []
        })
      );
    }
    return nodes;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.currentPositions,
      output.currentElevations,
      output.headings,
      output.speeds,
      output.trackStatus,
      output.segmentRows,
      output.segmentFractions,
      output.activeTracks?.ids,
      output.activeTracks?.status.count,
      output.activeTracks?.status.overflow,
      output.activeTracks?.status.requiredCount,
      output.drawInstanceCount
    ];
  }
}
