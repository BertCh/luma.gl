// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createTrajectoryArcLengthNode,
  createTrajectoryResampleNode
} from './trajectory-interpolation-kernels';

/** How {@link GPUTrajectoryResample} spaces samples along each track. */
export type GPUTrajectoryResampleSpacing = 'time' | 'arc-length';

/**
 * Properties for {@link GPUTrajectoryResample}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * track count, `sampleCount`, `spacing`, the time format, and which optional views are present.
 */
export type GPUTrajectoryResampleProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-resample'`. */
  id?: string;
  /** Packed planar positions, one row per sample, sorted by track and then time. */
  positions: GraphDataView<'float32x2'>;
  /** Optional packed elevation (z) per sample. Required for `sampleElevations`. */
  elevations?: GraphDataView<'float32'>;
  /**
   * Packed sample times (`float32` relative or `uint32x2` Int64 words), non-decreasing inside each
   * track. Required for `spacing: 'time'` and for `sampleTimes`.
   */
  timestamps?: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** `trackCount + 1` monotonic row offsets, the same layout as `GPUTrajectoryMetrics`. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Samples per track, at least 1. Sample `k` targets `k / (sampleCount - 1)` of the track. */
  sampleCount: number;
  /**
   * `'time'` (default) spaces samples uniformly between the first and last timestamp;
   * `'arc-length'` spaces them uniformly along the planar path length.
   */
  spacing?: GPUTrajectoryResampleSpacing;
  /** Dense `[trackCount × sampleCount]` positions, row `track * sampleCount + k`. */
  samples: GraphDataView<'float32x2'>;
  /** Optional dense elevations aligned with `samples`. Requires `elevations`. */
  sampleElevations?: GraphDataView<'float32'>;
  /**
   * Optional dense sample times relative to each track's first timestamp, in the timestamps' unit.
   * Requires `timestamps`.
   */
  sampleTimes?: GraphDataView<'float32'>;
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

  constructor(props: GPUTrajectoryResampleProps) {
    this.id = props.id ?? 'trajectory-resample';
    this.props = props;
    this.spacing = props.spacing ?? 'time';
    const id = this.id;
    const rowCount = props.positions.length;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedUint32View(props.trackOffsets, `${id} trackOffsets`);
    if (props.trackOffsets.length < 2) {
      throw new Error(`${id} trackOffsets must contain at least two rows`);
    }
    if (this.spacing !== 'time' && this.spacing !== 'arc-length') {
      throw new Error(`${id} spacing must be 'time' or 'arc-length'`);
    }
    if (!Number.isInteger(props.sampleCount) || props.sampleCount < 1) {
      throw new Error(`${id} sampleCount must be a positive integer`);
    }
    if (props.timestamps) {
      validatePackedView(props.timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
      if (props.timestamps.length !== rowCount) {
        throw new Error(`${id} timestamps length must equal positions length`);
      }
    } else if (this.spacing === 'time' || props.sampleTimes) {
      throw new Error(`${id} timestamps are required for time spacing and sampleTimes`);
    }
    if (props.elevations) {
      validatePackedView(props.elevations, ['float32'], `${id} elevations`);
      if (props.elevations.length !== rowCount) {
        throw new Error(`${id} elevations length must equal positions length`);
      }
    }
    if (props.sampleElevations && !props.elevations) {
      throw new Error(`${id} sampleElevations requires elevations`);
    }
    const trackCount = props.trackOffsets.length - 1;
    const outputLength = trackCount * props.sampleCount;
    if (outputLength > 0xffffffff) {
      throw new Error(`${id} trackCount * sampleCount must fit in 32 bits`);
    }
    for (const [name, view, format] of [
      ['samples', props.samples, 'float32x2'],
      ['sampleElevations', props.sampleElevations, 'float32'],
      ['sampleTimes', props.sampleTimes, 'float32']
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
      [props.samples, props.sampleElevations, props.sampleTimes],
      [props.positions, props.elevations, props.timestamps, props.trackOffsets]
    );
  }

  /** Returns the optional arc-length node and the per-sample resample node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.elevations,
      props.timestamps,
      props.trackOffsets,
      props.samples,
      props.sampleElevations,
      props.sampleTimes
    ]);
    const rowCount = props.positions.length;
    const trackCount = props.trackOffsets.length - 1;
    const nodes: GPUCommandNode<Parameters>[] = [];
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
        timestamps: this.spacing === 'time' || props.sampleTimes ? props.timestamps : undefined,
        trackOffsets: props.trackOffsets,
        cumulativeLengths,
        samples: props.samples,
        sampleElevations: props.sampleElevations,
        sampleTimes: props.sampleTimes,
        rowCount,
        trackCount,
        sampleCount: props.sampleCount
      })
    );
    return nodes;
  }
}
