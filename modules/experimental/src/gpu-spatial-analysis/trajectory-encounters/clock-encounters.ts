// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUTrajectoryResample} from '../trajectory-interpolation/gpu-trajectory-resample';
import {createClockBucketTimesNode} from './encounter-kernels';
import {
  GPUTrajectoryEncounters,
  type GPUTrajectoryEncountersProps
} from './gpu-trajectory-encounters';

/**
 * Properties for {@link addClockEncounters}.
 *
 * The encounter options (`distance`, `cellSize`, `bounds`, `hitCapacity`, `pairs`) are those of
 * {@link GPUTrajectoryEncountersProps}; `samples`, `trackCount`, `bucketCount` and `trackValid`
 * are derived. Per-frame (no recompile): every input buffer, `clock` (start and step) and
 * `distance`. Compile-time: view lengths, `bucketCount`, `cellSize`, `bounds`, `hitCapacity`, the
 * output capacity, and the time format.
 */
export type AddClockEncountersProps = Omit<
  GPUTrajectoryEncountersProps,
  'samples' | 'trackCount' | 'bucketCount' | 'trackValid' | 'bucketTimes'
> & {
  /** Prefix for generated node and transient IDs. Defaults to `'clock-encounters'`. */
  id?: string;
  /** Packed planar positions, one row per sample, sorted by track and then time. */
  positions: GraphDataView<'float32x2'>;
  /** Packed sample times, float32 relative or `uint32x2` Int64 words, sorted inside each track. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** `trackCount + 1` monotonic row offsets. */
  trackOffsets: GraphDataView<'uint32'>;
  /**
   * Per-frame shared clock: float32 from `getGPUTrajectoryClockParameterValues` for float32
   * timestamps, or uint32 from `getGPUTrajectoryClockWordParameterValues` for word timestamps.
   */
  clock: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Number of clock buckets, the dense table width. Bucket `k` is `start + k * step`. */
  bucketCount: number;
  /**
   * Optional output, `bucketCount` rows, receiving the time of each bucket relative to the clock
   * start (`k * step`). Created internally when omitted and `pairs.firstTimes` needs it.
   */
  bucketTimes?: GraphDataView<'float32'>;
  /**
   * Optional output, `trackCount * bucketCount` rows, receiving the shared-clock positions
   * (NaN while a track is absent). Created internally when omitted.
   */
  samples?: GraphDataView<'float32x2'>;
};

/** Result of {@link addClockEncounters}. */
export type ClockEncounters = {
  /** The shared-clock resample contributor that was added to the graph. */
  resample: GPUTrajectoryResample;
  /** The encounter contributor that was added to the graph. */
  encounters: GPUTrajectoryEncounters;
  /** Dense `[trackCount * bucketCount]` positions the encounters read. */
  samples: GraphDataView<'float32x2'>;
  /** Time of each bucket relative to the clock start, when it was requested or needed. */
  bucketTimes?: GraphDataView<'float32'>;
};

/**
 * Resamples every track onto one shared clock and finds encounters on it, in one call.
 *
 * Adds a `GPUTrajectoryResample` with `spacing: 'clock'` (every track is evaluated at the instants
 * `start + k * step`, NaN outside its own time range) and a `GPUTrajectoryEncounters` that reads
 * the result directly, so callers do not wire the dense table by hand. Changing the clock start
 * or step, or `distance`, only rewrites parameter buffers. The nodes are added to `graph` in
 * order; read pairs from `props.pairs`.
 *
 * Because absent samples are NaN, a pair only encounters while both tracks exist. The same
 * discrete-approximation caveat as `GPUTrajectoryEncounters` applies: pick a clock step small
 * compared with the distance a track travels.
 *
 * @param graph Graph that receives the nodes.
 * @param props Inputs, clock and encounter options.
 * @returns The added contributors and the intermediate views.
 */
export function addClockEncounters<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: AddClockEncountersProps
): ClockEncounters {
  const id = props.id ?? 'clock-encounters';
  const {positions, timestamps, trackOffsets, clock, bucketCount, ...encounterProps} = props;
  const trackCount = trackOffsets.length - 1;
  if (trackCount < 1) {
    throw new Error(`${id} trackOffsets must contain at least two rows`);
  }
  const sampleCount = trackCount * bucketCount;
  const samples =
    props.samples ??
    createTransientView(graph, `${id}-samples`, 'float32x2', Math.max(sampleCount, 1));
  const needsBucketTimes = Boolean(props.bucketTimes || props.pairs.firstTimes);
  const bucketTimes = needsBucketTimes
    ? (props.bucketTimes ??
      createTransientView(graph, `${id}-bucket-times`, 'float32', Math.max(bucketCount, 1)))
    : undefined;
  const resample = new GPUTrajectoryResample({
    id: `${id}-resample`,
    positions,
    timestamps,
    trackOffsets,
    sampleCount: bucketCount,
    spacing: 'clock',
    clock,
    samples
  });
  const encounters = new GPUTrajectoryEncounters({
    ...encounterProps,
    id: `${id}-encounters`,
    samples,
    trackCount,
    bucketCount,
    bucketTimes
  });
  graph.add(resample);
  if (bucketTimes) {
    graph.add(
      createClockBucketTimesNode<Parameters>(graph, {
        id: `${id}-bucket-times`,
        isWordMode: timestamps.format === 'uint32x2',
        bucketCount,
        clock,
        bucketTimes
      })
    );
  }
  graph.add(encounters);
  return {resample, encounters, samples, bucketTimes};
}
