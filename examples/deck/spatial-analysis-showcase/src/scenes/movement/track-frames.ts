// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma.gl/experimental/gpu-spatial-analysis';

/**
 * Great-circle path length in meters of every track (haversine on the sphere of
 * `GPU_EARTH_MEAN_RADIUS_METERS`), summed in float64 on the CPU for labels and rankings.
 */
export function getGreatCircleTrackLengths(
  lngLat: Float32Array,
  offsets: Uint32Array
): Float32Array {
  const toRadians = Math.PI / 180;
  const lengths = new Float32Array(offsets.length - 1);
  for (let track = 0; track < lengths.length; track++) {
    let length = 0;
    for (let vertex = offsets[track]; vertex < offsets[track + 1] - 1; vertex++) {
      const latitudeA = lngLat[vertex * 2 + 1] * toRadians;
      const latitudeB = lngLat[vertex * 2 + 3] * toRadians;
      const sinLatitude = Math.sin((latitudeB - latitudeA) / 2);
      const sinLongitude = Math.sin(
        ((lngLat[vertex * 2 + 2] - lngLat[vertex * 2]) * toRadians) / 2
      );
      const haversine =
        sinLatitude * sinLatitude +
        Math.cos(latitudeA) * Math.cos(latitudeB) * sinLongitude * sinLongitude;
      length += 2 * GPU_EARTH_MEAN_RADIUS_METERS * Math.asin(Math.min(1, Math.sqrt(haversine)));
    }
    lengths[track] = length;
  }
  return lengths;
}
