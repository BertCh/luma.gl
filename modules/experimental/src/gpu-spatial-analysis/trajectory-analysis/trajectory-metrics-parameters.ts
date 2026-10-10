// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineGPUSpatialParameterSchema, packGPUSpatialParameterValues} from '../contracts/index';

/** Number of float32 elements in a trajectory-metrics parameter buffer. */
export const GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH = 4;

/** Declarative layout shared by trajectory-metrics parameter writers and the contributor. */
export const GPU_TRAJECTORY_METRICS_PARAMETER_SCHEMA = defineGPUSpatialParameterSchema({
  id: 'trajectory-metrics',
  format: 'float32',
  wordLength: GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  fields: [
    {
      name: 'stopSpeedThreshold',
      format: 'float32',
      wordOffset: 0,
      defaultValue: 0,
      minimum: 0,
      units: 'spatial-context-units-per-time-unit',
      dynamic: true
    },
    {
      name: 'stopMinimumDuration',
      format: 'float32',
      wordOffset: 1,
      defaultValue: 0,
      minimum: 0,
      units: 'time-units',
      dynamic: true
    }
  ]
});

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
 * `GPUParameterBuffer` between encodings to change the thresholds without recompiling.
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
  target.set(packGPUSpatialParameterValues(GPU_TRAJECTORY_METRICS_PARAMETER_SCHEMA, parameters));
  return target;
}
