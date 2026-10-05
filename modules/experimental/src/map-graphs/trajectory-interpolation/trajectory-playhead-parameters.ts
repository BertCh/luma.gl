// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {splitTimeWords} from '../time-window-filter/time-words';

/** Number of 32-bit elements in a trajectory playhead parameter view (both time modes). */
export const GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH = 4;

/**
 * Per-track playhead status written by `GPUTrajectoryPlayhead`.
 *
 * - `empty`: the track has no rows.
 * - `active`: the playhead lies inside the track's time range and its bracketing interval is not a
 *   gap. Position, heading, and speed are interpolated.
 * - `beforeStart`: the playhead is earlier than the first sample.
 * - `afterEnd`: the playhead is later than the last sample.
 * - `gap`: the playhead lies strictly inside a bracketing interval longer than `maxGap`.
 */
export const GPU_TRAJECTORY_PLAYHEAD_STATUS = {
  empty: 0,
  active: 1,
  beforeStart: 2,
  afterEnd: 3,
  gap: 4
} as const;

/** One value of {@link GPU_TRAJECTORY_PLAYHEAD_STATUS}. */
export type GPUTrajectoryPlayheadStatus =
  (typeof GPU_TRAJECTORY_PLAYHEAD_STATUS)[keyof typeof GPU_TRAJECTORY_PLAYHEAD_STATUS];

/**
 * CPU description of one per-frame playhead.
 *
 * Times use the unit of the timestamps: relative f32 time for float32 timestamps, or Int64 units
 * (for example epoch milliseconds) for word timestamps.
 */
export type GPUTrajectoryPlayheadTime = {
  /**
   * Playhead instant. Float32 mode: a number in the timestamps' relative epoch. Word mode: a
   * `bigint` (exact integer) or a `number` whose fractional part becomes the f32 sub-unit fraction.
   */
  playhead: number | bigint;
  /**
   * Largest bracketing interval that is still interpolated, in time units. When the playhead lies
   * strictly inside a longer interval the track status is `gap`. 0 (default) disables the rule.
   */
  maxGap?: number;
};

/**
 * Packs a playhead for float32 (relative) timestamps.
 *
 * Layout: `[playhead, maxGap, 0, 0]` as float32. Write it into a `GPUMapGraphParameterBuffer` with
 * `format: 'float32'` between encodings; the compiled graph is reused.
 *
 * @param playhead Playhead and optional gap limit.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `maxGap` is negative, `playhead` is a `bigint`, or `target` is
 * too short.
 */
export function getGPUTrajectoryPlayheadParameterValues(
  playhead: GPUTrajectoryPlayheadTime,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH)
): Float32Array {
  checkTarget(target);
  if (typeof playhead.playhead === 'bigint') {
    throw new Error('Float32 trajectory playheads must be numbers; use the word parameters');
  }
  const maxGap = getMaxGap(playhead);
  if (!Number.isFinite(playhead.playhead)) {
    throw new Error('Trajectory playhead must be finite');
  }
  target.set([playhead.playhead, maxGap, 0, 0]);
  return target;
}

/**
 * Packs a playhead for exact Int64 word timestamps.
 *
 * Layout: `[playheadLow, playheadHigh, playheadFraction, maxGap]` as uint32, where the last two are
 * f32 bit patterns. Write it into a `GPUMapGraphParameterBuffer` with `format: 'uint32'`.
 *
 * @param playhead Playhead and optional gap limit. The integer part may be any signed 64-bit value.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If the playhead cannot be split into words, `maxGap` is negative or not finite, or
 * `target` is too short.
 */
export function getGPUTrajectoryPlayheadWordParameterValues(
  playhead: GPUTrajectoryPlayheadTime,
  target: Uint32Array = new Uint32Array(GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH)
): Uint32Array {
  checkTarget(target);
  const maxGap = getMaxGap(playhead);
  const words = splitTimeWords(playhead.playhead);
  const floats = new Float32Array([words.fraction, maxGap]);
  target[0] = words.low;
  target[1] = words.high;
  target.set(new Uint32Array(floats.buffer), 2);
  return target;
}

function checkTarget(target: Float32Array | Uint32Array): void {
  if (target.length < GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH) {
    throw new Error(
      `Trajectory playhead target must hold ${GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH} elements`
    );
  }
}

function getMaxGap(playhead: GPUTrajectoryPlayheadTime): number {
  const maxGap = playhead.maxGap ?? 0;
  if (!Number.isFinite(maxGap) || maxGap < 0) {
    throw new Error('Trajectory playhead maxGap must be finite and non-negative');
  }
  return maxGap;
}
