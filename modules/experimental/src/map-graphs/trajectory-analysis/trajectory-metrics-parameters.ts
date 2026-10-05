// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a trajectory-metrics parameter buffer. */
export const GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH = 4;

/**
 * CPU description of the per-frame stop-detection parameters of `GPUTrajectoryMetrics`.
 *
 * Both values use the units of the track data: positions per time unit for the speed threshold and
 * time units (the unit of the timestamps) for the minimum duration.
 */
export type GPUTrajectoryStopParameters = {
  /**
   * A step between two consecutive samples is slow when its speed is below this value. Must be
   * finite and non-negative. A threshold of 0 only treats zero-distance steps as slow.
   */
  stopSpeedThreshold: number;
  /**
   * A dwell (a maximal run of slow steps) is reported as a stop when its duration is at least this
   * value. Must be finite and non-negative.
   */
  stopMinimumDuration: number;
};

/**
 * Packs stop parameters into the 4-element float32 layout read by `GPUTrajectoryMetrics`.
 *
 * Layout: `[stopSpeedThreshold, stopMinimumDuration, 0, 0]`. Write the result into a
 * `GPUMapGraphParameterBuffer` between encodings to change the thresholds without recompiling.
 *
 * @param parameters Stop parameters to encode.
 * @param target Optional destination of at least 4 elements. A new array is returned when omitted.
 * @throws If a value is not finite, is negative, or `target` is too short.
 */
export function getGPUTrajectoryMetricsParameterValues(
  parameters: GPUTrajectoryStopParameters,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH) {
    throw new Error(
      `Trajectory metrics target must hold ${GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH} elements`
    );
  }
  for (const value of [parameters.stopSpeedThreshold, parameters.stopMinimumDuration]) {
    if (!Number.isFinite(value)) {
      throw new Error('Trajectory metrics parameters must be finite');
    }
    if (value < 0) {
      throw new Error('Trajectory metrics parameters must be non-negative');
    }
  }
  target.set([parameters.stopSpeedThreshold, parameters.stopMinimumDuration, 0, 0]);
  return target;
}
