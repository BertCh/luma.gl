// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';

/**
 * Converts between `[longitude, latitude]` and planar meters around one origin using the same
 * high-precision distance scales deck.gl applies to `METER_OFFSETS` coordinates.
 */
export class LocalMetricProjection {
  /** `[longitude, latitude]` origin. */
  readonly origin: readonly [number, number];
  private readonly metersPerDegree: readonly [number, number];

  constructor(origin: readonly [number, number]) {
    this.origin = origin;
    const scales = getDistanceScales({
      longitude: origin[0],
      latitude: origin[1],
      highPrecision: true
    });
    this.metersPerDegree = [
      scales.unitsPerDegree[0] * scales.metersPerUnit[0],
      scales.unitsPerDegree[1] * scales.metersPerUnit[1]
    ];
  }

  /** Returns `[x, y]` meters for one `[longitude, latitude]`. */
  project(longitude: number, latitude: number): [number, number] {
    // Linear estimate refined once against deck.gl's own meters-to-lng/lat conversion.
    let x = (longitude - this.origin[0]) * this.metersPerDegree[0];
    let y = (latitude - this.origin[1]) * this.metersPerDegree[1];
    const [estimatedLongitude, estimatedLatitude] = this.unproject(x, y);
    x += (longitude - estimatedLongitude) * this.metersPerDegree[0];
    y += (latitude - estimatedLatitude) * this.metersPerDegree[1];
    return [x, y];
  }

  /** Returns `[longitude, latitude]` for `[x, y]` meters. */
  unproject(x: number, y: number): [number, number] {
    const [longitude, latitude] = addMetersToLngLat(this.origin as [number, number], [x, y]);
    return [longitude, latitude];
  }
}

/** Mulberry32: small deterministic PRNG returning values in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Projects interleaved `[longitude, latitude, ...]` pairs (stride in floats) to planar meters. */
export function projectLngLatArray(
  projection: LocalMetricProjection,
  lngLat: ArrayLike<number>,
  stride = 2
): Float32Array {
  const count = Math.floor(lngLat.length / stride);
  const meters = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    const [x, y] = projection.project(lngLat[index * stride], lngLat[index * stride + 1]);
    meters[index * 2] = x;
    meters[index * 2 + 1] = y;
  }
  return meters;
}

/** Returns the `[longitude, latitude]` center of a `[west, south, east, north]` box. */
export function getBboxCenter(bbox: readonly [number, number, number, number]): [number, number] {
  return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
}
