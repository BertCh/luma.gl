// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphCompactOutput, GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphPublishNode} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMapGraphCompactOutput
} from '../map-graph-utils';
import {
  createTrajectoryBracketNode,
  createTrajectoryColumnsNode,
  createTrajectoryGeometryNode,
  TRAJECTORY_BRACKET_STRIDE
} from './trajectory-interpolation-kernels';
import {GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH} from './trajectory-playhead-parameters';

/**
 * Properties for {@link GPUTrajectoryPlayhead}.
 *
 * Per-frame (no recompile): the contents of `parameters` (playhead and `maxGap`) and of every input
 * buffer, including `trackOffsets`. Compile-time (needs a new graph): view lengths, the track count
 * (`trackOffsets.length - 1`), the time format, `activeTracks.ids.length`, and which optional views
 * are present.
 */
export type GPUTrajectoryPlayheadProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-playhead'`. */
  id?: string;
  /** Packed planar positions, one row per sample, sorted by track and then time. */
  positions: GraphDataView<'float32x2'>;
  /** Optional packed elevation (z) per sample, same length as `positions`. */
  elevations?: GraphDataView<'float32'>;
  /**
   * Packed sample times, same length as `positions`, non-decreasing inside each track:
   *
   * - `float32` relative to an application epoch, with a float32 `parameters` view.
   * - `uint32x2` Int64 `(low, high)` words such as an Arrow `Int64` epoch-ms column
   *   (`getInt64TimeWords`), with a uint32 `parameters` view. Comparisons with the playhead are
   *   exact; differences are subtracted exactly and then rounded to f32.
   */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /**
   * `trackCount + 1` monotonic row offsets; track `t` owns rows `[trackOffsets[t], trackOffsets[t + 1])`.
   * The same layout as `GPUTrajectoryMetrics`. Offsets past `positions.length` are clamped.
   */
  trackOffsets: GraphDataView<'uint32'>;
  /**
   * Per-frame playhead, at least {@link GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH} elements: float32
   * from `getGPUTrajectoryPlayheadParameterValues` for float32 timestamps, or uint32 from
   * `getGPUTrajectoryPlayheadWordParameterValues` for word timestamps.
   */
  parameters: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Optional interpolated position per track, `trackCount` rows. */
  currentPositions?: GraphDataView<'float32x2'>;
  /** Optional interpolated elevation per track. Requires `elevations`. */
  currentElevations?: GraphDataView<'float32'>;
  /**
   * Optional heading per track in radians, `atan2(dy, dx)` of the bracketing segment: 0 points
   * along +x and angles grow counterclockwise. 0 for an empty track or a zero-length segment.
   */
  headings?: GraphDataView<'float32'>;
  /** Optional planar speed of the bracketing segment for active tracks, otherwise 0. */
  speeds?: GraphDataView<'float32'>;
  /** Optional `GPU_TRAJECTORY_PLAYHEAD_STATUS` value per track. */
  status?: GraphDataView<'uint32'>;
  /** Optional first row of the bracketing segment per track, `0xffffffff` for an empty track. */
  segmentRows?: GraphDataView<'uint32'>;
  /** Optional interpolation fraction inside the bracketing segment, in `[0, 1]`. */
  segmentFractions?: GraphDataView<'float32'>;
  /** Optional compact, ascending list of active track indices with a clamped count. */
  activeTracks?: GPUMapGraphCompactOutput;
  /**
   * Optional packed one-row view that receives the clamped active count, typically an indirect draw
   * record's `instanceCount`. Requires `activeTracks`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
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
 * - Gap: when `maxGap > 0` and the playhead lies strictly inside a bracketing interval longer than
 *   `maxGap`, the status is `gap` and the position holds at the last fix before the gap.
 * - Duplicate timestamps: the search picks the last of several rows with equal time, so at exactly
 *   that time the track reports the last duplicate ("latest row wins"), and the interval used for
 *   interpolation always has a positive duration. Zero-duration segments never divide.
 * - Exact sample times: `p == t[k]` gives fraction 0 on segment `[k, k + 1]` and position
 *   `positions[k]` bit-exactly.
 *
 * Speed is planar distance over duration in the timestamps' unit and is only non-zero for active
 * tracks. Heading and speed come from the bracketing segment, so they are piecewise constant.
 *
 * Composition: one search kernel writing an internal per-track bracket, one geometry kernel, one
 * column kernel, and for `activeTracks` a stable `GPUCompaction` of active track indices and one
 * publish kernel that clamps the count. Moving the playhead only rewrites `parameters`.
 *
 * Non-goals: geodesic interpolation and antimeridian handling (project upstream), curved
 * (spline) interpolation, unsorted rows, chunked inputs, double-single timestamps.
 */
export class GPUTrajectoryPlayhead implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'trajectory-playhead';
  /** Validated properties. */
  readonly props: GPUTrajectoryPlayheadProps;

  constructor(props: GPUTrajectoryPlayheadProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    const rowCount = props.positions.length;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
    validatePackedUint32View(props.trackOffsets, `${id} trackOffsets`);
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
      ['currentPositions', props.currentPositions, 'float32x2'],
      ['currentElevations', props.currentElevations, 'float32'],
      ['headings', props.headings, 'float32'],
      ['speeds', props.speeds, 'float32'],
      ['status', props.status, 'uint32'],
      ['segmentRows', props.segmentRows, 'uint32'],
      ['segmentFractions', props.segmentFractions, 'float32']
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} ${name}`);
      if (view.length !== trackCount) {
        throw new Error(`${id} ${name} length must equal the track count`);
      }
    }
    if (props.currentElevations && !props.elevations) {
      throw new Error(`${id} currentElevations requires elevations`);
    }
    if (props.activeTracks) {
      validateMapGraphCompactOutput(id, props.activeTracks);
    }
    if (props.drawInstanceCount) {
      if (!props.activeTracks) {
        throw new Error(`${id} drawInstanceCount requires activeTracks`);
      }
      validatePackedUint32View(props.drawInstanceCount, `${id} drawInstanceCount`);
      if (props.drawInstanceCount.length < 1) {
        throw new Error(`${id} drawInstanceCount must contain one uint32 row`);
      }
    }
    if (
      !(
        props.currentPositions ||
        props.currentElevations ||
        props.headings ||
        props.speeds ||
        props.status ||
        props.segmentRows ||
        props.segmentFractions ||
        props.activeTracks
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
    if (props.currentPositions || props.currentElevations || props.headings || props.speeds) {
      nodes.push(
        createTrajectoryGeometryNode<Parameters>(graph, {
          id: `${id}-geometry`,
          bracket,
          positions: props.positions,
          elevations: props.elevations,
          currentPositions: props.currentPositions,
          currentElevations: props.currentElevations,
          headings: props.headings,
          speeds: props.speeds,
          trackCount
        })
      );
    }
    const {activeTracks} = props;
    const activeFlags = activeTracks
      ? createTransientView(graph, `${id}-active-flags`, 'uint32', trackCount)
      : undefined;
    const trackIndices = activeTracks
      ? createTransientView(graph, `${id}-track-indices`, 'uint32', trackCount)
      : undefined;
    if (props.status || props.segmentRows || props.segmentFractions || activeTracks) {
      nodes.push(
        createTrajectoryColumnsNode<Parameters>(graph, {
          id: `${id}-columns`,
          bracket,
          status: props.status,
          segmentRows: props.segmentRows,
          segmentFractions: props.segmentFractions,
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
        createMapGraphPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: 'GPUTrajectoryPlayhead',
          totalCount: activeTotal,
          compactIds,
          output: activeTracks,
          extraCounts: props.drawInstanceCount ? [props.drawInstanceCount] : []
        })
      );
    }
    return nodes;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.currentPositions,
      props.currentElevations,
      props.headings,
      props.speeds,
      props.status,
      props.segmentRows,
      props.segmentFractions,
      props.activeTracks?.ids,
      props.activeTracks?.count,
      props.activeTracks?.overflow,
      props.activeTracks?.totalCount,
      props.drawInstanceCount
    ];
  }
}
