// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '@luma.gl/experimental/gpu-spatial-analysis';
import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';

/** Bbox half-diagonal beyond which {@link LocalMetricProjection} exceeds about 1% scale error. */
export const LOCAL_METRIC_MAXIMUM_HALF_DIAGONAL_METERS = 150_000;

/**
 * Converts between `[longitude, latitude]` and planar meters around one origin using the same
 * high-precision distance scales deck.gl applies to `METER_OFFSETS` coordinates.
 *
 * This is Web Mercator scaled to be true at the origin, so it must match deck.gl for drawing. Its
 * local scale error grows with distance from the origin: about +-0.3% over Chicago, +-3% over the
 * Netherlands rail network, and -15% to +21% over the contiguous US. It is the display frame:
 * regional analysis must not run in it; use an analysis frame from `analysis-frame.ts` and pass
 * its projection's `spatialContext` to every operator consuming frame meters. City-scale scenes
 * that still analyse in these meters pass {@link LocalMetricProjection.spatialContext}, a `'crs'`
 * frame that names this projection honestly (it is neither equidistant nor equal-area).
 */
export class LocalMetricProjection {
  /** `[longitude, latitude]` origin. */
  readonly origin: readonly [number, number];
  /**
   * Planar context of these meters: a `'crs'` frame `showcase-mercator-local(<lon>,<lat>)`, so data
   * projected about different origins is never mixed and both distance and area operators accept it.
   */
  readonly spatialContext: GPUSpatialContext;
  private readonly metersPerDegree: readonly [number, number];

  constructor(origin: readonly [number, number]) {
    this.origin = origin;
    this.spatialContext = Object.freeze({
      coordinateSpace: 'planar',
      datum: 'EPSG:4326',
      metric: 'native',
      units: 'meters',
      frame: Object.freeze({
        kind: 'crs',
        id: `showcase-mercator-local(${origin[0]},${origin[1]})`
      })
    });
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
