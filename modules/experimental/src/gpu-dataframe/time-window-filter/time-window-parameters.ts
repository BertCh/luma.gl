// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a time-window parameter buffer. */
export const GPU_TIME_WINDOW_PARAMETER_LENGTH = 8;

/**
 * CPU description of one per-frame time window.
 *
 * Times use the same unit and epoch as the row timestamps, for example seconds since an
 * application epoch, or absolute epoch values when timestamps are split with
 * {@link splitTimestamps}.
 */
export type GPUTimeWindow = {
  /** Inclusive window start. */
  start: number;
  /** Inclusive window end. A window with `end < start` accepts nothing. */
  end: number;
  /** Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. */
  startFadeDuration?: number;
  /** Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. */
  endFadeDuration?: number;
};

/** High and low float32 parts of double-precision timestamps, `value ≈ high + low`. */
export type GPUSplitTimestamps = {
  /** `Math.fround(value)`. */
  high: Float32Array;
  /** `Math.fround(value - high)`. */
  low: Float32Array;
};

/**
 * Packs a time window into the 8-element float32 layout read by `GPUTimeWindowFilter`.
 *
 * Layout: `[startHigh, startLow, endHigh, endLow, startFadeDuration, endFadeDuration, 0, 0]`.
 * Start and end are split into high and low float32 parts so absolute epoch times keep sub-ulp
 * precision.
 *
 * @param window Window to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, a fade duration is negative, or `target` is too short.
 */
export function getGPUTimeWindowParameterValues(
  window: GPUTimeWindow,
  target: Float32Array = new Float32Array(GPU_TIME_WINDOW_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TIME_WINDOW_PARAMETER_LENGTH) {
    throw new Error(`Time window target must hold ${GPU_TIME_WINDOW_PARAMETER_LENGTH} elements`);
  }
  const startFadeDuration = window.startFadeDuration ?? 0;
  const endFadeDuration = window.endFadeDuration ?? 0;
  for (const value of [window.start, window.end, startFadeDuration, endFadeDuration]) {
    if (!Number.isFinite(value)) {
      throw new Error('Time window values must be finite');
    }
  }
  if (startFadeDuration < 0 || endFadeDuration < 0) {
    throw new Error('Time window fade durations must be non-negative');
  }
  const [startHigh, startLow] = splitTime(window.start);
  const [endHigh, endLow] = splitTime(window.end);
  target.set([startHigh, startLow, endHigh, endLow, startFadeDuration, endFadeDuration, 0, 0]);
  return target;
}

/**
 * Splits double-precision timestamps into float32 high and low parts for double-single comparison.
 *
 * `high[i] = Math.fround(values[i])` and `low[i] = Math.fround(values[i] - high[i])`. Values are not
 * validated; a non-finite timestamp is never accepted by the filter.
 */
export function splitTimestamps(values: ArrayLike<number>): GPUSplitTimestamps {
  const high = new Float32Array(values.length);
  const low = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    [high[index], low[index]] = splitTime(values[index]);
  }
  return {high, low};
}

/** Splits one value into float32 high and low parts. */
function splitTime(value: number): [number, number] {
  const high = Math.fround(value);
  return [high, Math.fround(value - high)];
}
