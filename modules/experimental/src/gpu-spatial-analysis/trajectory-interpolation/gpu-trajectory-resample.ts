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
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH} from './trajectory-clock-parameters';
import {
  createTrajectoryArcLengthNode,
  createTrajectoryResampleNode
} from './trajectory-interpolation-kernels';
import type {GPUTrajectoryGeometryFields} from '../../analysis-core/contracts/index';

/** How {@link GPUTrajectoryResample} spaces samples along each track. */
export type GPUTrajectoryResampleSpacing = 'time' | 'arc-length' | 'clock';

/**
 * Caller-owned dense outputs of {@link GPUTrajectoryResample}.
 */
export type GPUTrajectoryResampleOutput = {
  /** Dense `[trackCount × sampleCount]` positions, row `track * sampleCount + k`. */
  samples: GraphDataView<'float32x2'>;
  /** Optional dense elevations aligned with `samples`. Requires `elevations`. */
  sampleElevations?: GraphDataView<'float32'>;
  /**
   * Optional dense sample times, in the timestamps' unit: relative to each track's first timestamp,
   * or with `spacing: 'clock'` relative to the clock start (`k * step`). Requires `timestamps`.
   */
  sampleTimes?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUTrajectoryResample}: the trajectory layout
 * ({@link GPUTrajectoryGeometryFields}: `positions`, `trackOffsets`, and `timestamps`, which is
 * optional here) plus the options below.
 *
 * `positions` are packed rows (planar, or longitude/latitude degrees), sorted by track and then
 * time. `timestamps` are `float32` relative or `uint32x2` Int64 words, non-decreasing inside each
 * track, and are required for `spacing: 'time'` and for `output.sampleTimes`. `trackOffsets` are
 * `trackCount + 1` monotonic row offsets, the same layout as `GPUTrajectoryMetrics`.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * track count, `sampleCount`, `spacing`, the time format, and which optional views are present.
 */
export type GPUTrajectoryResampleProps = Omit<GPUTrajectoryGeometryFields, 'timestamps'> & {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-resample'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Omitted means native planar coordinates, linear
   * interpolation and planar arc length. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` means `(longitude, latitude)` degrees interpolated along great circles
   * (spherical linear interpolation) on a sphere of `sphereRadius` (default
   * `GPU_EARTH_MEAN_RADIUS_METERS`), with `'arc-length'` spacing measured in great-circle meters.
   * Other contexts throw.
   */
  spatialContext?: GPUSpatialContext;
  /** Optional packed elevation (z) per sample. Required for `output.sampleElevations`. */
  elevations?: GraphDataView<'float32'>;
  /** Packed sample times, see above. Required for `spacing: 'time'` and `output.sampleTimes`. */
  timestamps?: GPUTrajectoryGeometryFields['timestamps'];
  /** Samples per track, at least 1. Sample `k` targets `k / (sampleCount - 1)` of the track. */
  sampleCount: number;
  /**
   * `'time'` (default) spaces samples uniformly between the first and last timestamp;
   * `'arc-length'` spaces them uniformly along the path length (planar, or great-circle meters); `'clock'` samples every
   * track at the shared instants `start + k * step` of `clock` (see below).
   */
  spacing?: GPUTrajectoryResampleSpacing;
  /**
   * Per-frame shared clock, required for `spacing: 'clock'` and invalid otherwise. Float32 from
   * `getGPUTrajectoryClockParameterValues` for float32 timestamps, or uint32 from
   * `getGPUTrajectoryClockWordParameterValues` for word timestamps. Changing the start or step
   * only rewrites this buffer; the compiled graph is reused.
   */
  clock?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Caller-owned output views. */
  output: GPUTrajectoryResampleOutput;
};

/**
 * Resamples every track to a fixed number of samples, the dense input for trajectory similarity,
 * k-means, and smooth fixed-vertex trails.
 *
 * Sample `k` of a track with `n > 0` rows targets progress `total * k / (sampleCount - 1)`, where
 * progress is time since the first sample (`spacing: 'time'`) or cumulative planar length
 * (`'arc-length'`), and `total` is the progress of the last row. The last sample is the last row
 * exactly; with `sampleCount == 1` the only sample targets progress 0. The bracketing segment is
 * found by the same upper-bound search as `GPUTrajectoryPlayhead`, so duplicate timestamps and
 * zero-length steps resolve to their last row and never divide by zero. A single-row track
 * repeats that row; an empty track writes zeros (derive validity from `trackOffsets`).
 *
 * Common clock: with `spacing: 'clock'` column `k` means the same instant `start + k * step` for
 * every track, the dense input `GPUTrajectoryEncounters` needs. A track is interpolated only
 * while that instant lies inside its own time range; before its first and after its last sample,
 * and for empty tracks, the position (and elevation) is NaN, which downstream contributors read as
 * "absent". `sampleCount` is then the number of clock steps.
 *
 * Spatial context: with a great-circle `spatialContext`, samples lie on the great-circle arc of
 * the bracketing segment and are interpolated with spherical linear interpolation instead of a
 * linear blend of degrees. Sample longitudes are continuous with the start row of the bracketing
 * segment, so a segment crossing the antimeridian yields longitudes beyond +-180 (the segment end
 * is `b` shifted by a multiple of 360); wrap downstream if needed. Absent clock samples stay NaN.
 *
 * Composition: in arc-length mode one per-track kernel sums step lengths sequentially in row order
 * (deterministic, but O(track length) per invocation); then one kernel with one invocation per
 * output sample.
 */
export class GPUTrajectoryResample implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrajectoryResampleProps;
  /** Resolved spacing. */
  readonly spacing: GPUTrajectoryResampleSpacing;
  /** Resolved coordinate and metric context; native planar when `props.spatialContext` is omitted. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUTrajectoryResampleProps) {
    this.id = props.id ?? 'trajectory-resample';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT
    );
    this.spacing = props.spacing ?? 'time';
    const id = this.id;
    const {output} = props;
    const rowCount = props.positions.length;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.trackOffsets, ['uint32'], `${id} trackOffsets`);
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    if (this.spacing !== 'time' && this.spacing !== 'arc-length' && this.spacing !== 'clock') {
      throw new Error(`${id} spacing must be 'time', 'arc-length' or 'clock'`);
    }
    if (this.spacing === 'clock') {
      if (!props.clock) {
        throw new Error(`${id} clock is required for clock spacing`);
      }
      if (!props.timestamps) {
        throw new Error(`${id} timestamps are required for clock spacing`);
      }
      const clockFormat = props.timestamps.format === 'uint32x2' ? 'uint32' : 'float32';
      validatePackedView(props.clock, [clockFormat], `${id} clock`);
      if (props.clock.length < GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH) {
        throw new Error(
          `${id} clock must hold ${GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH} ${clockFormat} values`
        );
      }
    } else if (props.clock) {
      throw new Error(`${id} clock requires spacing 'clock'`);
    }
    validateCapacity(id, 'sampleCount', props.sampleCount);
    if (props.timestamps) {
      validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
      if (props.timestamps.length !== rowCount) {
        throw new Error(`${id} timestamps length must equal positions length`);
      }
    } else if (this.spacing === 'time' || output.sampleTimes) {
      throw new Error(`${id} timestamps are required for time spacing and sampleTimes`);
    }
    if (props.elevations) {
      validatePackedView(props.elevations, ['float32'], `${id} elevations`);
      if (props.elevations.length !== rowCount) {
        throw new Error(`${id} elevations length must equal positions length`);
      }
    }
    if (output.sampleElevations && !props.elevations) {
      throw new Error(`${id} sampleElevations requires elevations`);
    }
    const trackCount = props.trackOffsets.length - 1;
    const outputLength = trackCount * props.sampleCount;
    if (outputLength > 0xffffffff) {
      throw new Error(`${id} trackCount * sampleCount must fit in 32 bits`);
    }
    for (const [name, view, format] of [
      ['samples', output.samples, 'float32x2'],
      ['sampleElevations', output.sampleElevations, 'float32'],
      ['sampleTimes', output.sampleTimes, 'float32']
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} ${name}`);
      if (view.length !== outputLength) {
        throw new Error(`${id} ${name} length must equal trackCount * sampleCount`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.samples, output.sampleElevations, output.sampleTimes],
      [props.positions, props.elevations, props.timestamps, props.trackOffsets, props.clock]
    );
  }

  /** Returns the optional arc-length node and the per-sample resample node. */
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
      props.clock,
      output.samples,
      output.sampleElevations,
      output.sampleTimes
    ]);
    const rowCount = props.positions.length;
    const trackCount = props.trackOffsets.length - 1;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const geodesicRadius = getGPUGreatCircleRadius(this.spatialContext);
    let cumulativeLengths: GraphDataView<'float32'> | undefined;
    if (this.spacing === 'arc-length') {
      cumulativeLengths = createTransientView(
        graph,
        `${id}-cumulative-lengths`,
        'float32',
        Math.max(rowCount, 1)
      );
      nodes.push(
        createTrajectoryArcLengthNode<Parameters>(graph, {
          id: `${id}-arc-length`,
          positions: props.positions,
          trackOffsets: props.trackOffsets,
          cumulativeLengths,
          geodesicRadius,
          rowCount,
          trackCount
        })
      );
    }
    nodes.push(
      createTrajectoryResampleNode<Parameters>(graph, {
        id: `${id}-samples`,
        positions: props.positions,
        elevations: props.elevations,
        timestamps:
          this.spacing !== 'arc-length' || output.sampleTimes ? props.timestamps : undefined,
        clock: props.clock,
        trackOffsets: props.trackOffsets,
        cumulativeLengths,
        samples: output.samples,
        sampleElevations: output.sampleElevations,
        sampleTimes: output.sampleTimes,
        geodesicRadius,
        rowCount,
        trackCount,
        sampleCount: props.sampleCount
      })
    );
    return nodes;
  }
}
