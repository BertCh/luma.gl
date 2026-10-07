// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {splitTimeWords} from '../../gpu-dataframe/time-window-filter/time-words';

/** Number of 32-bit elements in a trajectory clock parameter view (both time modes). */
export const GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH = 4;

/**
 * CPU description of the shared clock of `GPUTrajectoryResample` with `spacing: 'clock'`.
 *
 * Sample `k` of every track is taken at `start + k * step`, in the unit of the timestamps.
 */
export type GPUTrajectoryClock = {
  /**
   * First clock instant. Float32 mode: a number in the timestamps' relative epoch. Word mode: a
   * `bigint` (exact integer) or a `number` whose fractional part becomes the f32 sub-unit fraction.
   */
  start: number | bigint;
  /** Distance between consecutive clock samples, finite and positive. */
  step: number;
};

/**
 * Packs a clock for float32 (relative) timestamps.
 *
 * Layout: `[start, step, 0, 0]` as float32. Write it into a `GPUParameterBuffer` with
 * `format: 'float32'` between encodings; the compiled graph is reused.
 *
 * @param clock Clock start and step.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `step` is not positive, `start` is a `bigint`, or `target` is
 * too short.
 */
export function getGPUTrajectoryClockParameterValues(
  clock: GPUTrajectoryClock,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH)
): Float32Array {
  checkTarget(target);
  checkStep(clock);
  if (typeof clock.start === 'bigint') {
    throw new Error('Float32 trajectory clocks must use numbers; use the word parameters');
  }
  if (!Number.isFinite(clock.start)) {
    throw new Error('Trajectory clock start must be finite');
  }
  target.set([clock.start, clock.step, 0, 0]);
  return target;
}

/**
 * Packs a clock for exact Int64 word timestamps.
 *
 * Layout: `[startLow, startHigh, startFraction, step]` as uint32, where the last two are f32 bit
 * patterns. Write it into a `GPUParameterBuffer` with `format: 'uint32'`.
 *
 * @param clock Clock start and step. The integer part of `start` may be any signed 64-bit value.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If the start cannot be split into words, `step` is not positive and finite, or `target`
 * is too short.
 */
export function getGPUTrajectoryClockWordParameterValues(
  clock: GPUTrajectoryClock,
  target: Uint32Array = new Uint32Array(GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH)
): Uint32Array {
  checkTarget(target);
  checkStep(clock);
  const words = splitTimeWords(clock.start);
  const floats = new Float32Array([words.fraction, clock.step]);
  target[0] = words.low;
  target[1] = words.high;
  target.set(new Uint32Array(floats.buffer), 2);
  return target;
}

function checkTarget(target: Float32Array | Uint32Array): void {
  if (target.length < GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH) {
    throw new Error(
      `Trajectory clock target must hold ${GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH} elements`
    );
  }
}

function checkStep(clock: GPUTrajectoryClock): void {
  if (!Number.isFinite(clock.step) || clock.step <= 0) {
    throw new Error('Trajectory clock step must be finite and positive');
  }
}
