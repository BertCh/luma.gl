// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Viewport} from '@deck.gl/core';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma.gl/experimental/gpu-spatial-analysis';

/**
 * Analysis frames: planar meters about an origin on the sphere of radius
 * `GPU_EARTH_MEAN_RADIUS_METERS`, for analysis that must not run in the Mercator-local frame of
 * `LocalMetricProjection` (regional extents, where that frame's scale error passes about 1%).
 *
 * - `'azimuthal-equidistant'`: distances and bearings from the origin are exact; the tangential
 *   scale is `c / sin(c)` (c the angular distance), +0.65% at 1,250 km.
 * - `'lambert-equal-area'`: areas are exact everywhere; scales are `1/cos(c/2)` and `cos(c/2)`.
 * - `'albers-equal-area'`: conic, areas exact everywhere; scale is exact on the two standard
 *   parallels (`ALBERS_CONUS_FRAME`: +-1.44% over the contiguous US).
 *
 * The CPU formulas here are the ones the layers' WGSL inverts (`getAnalysisFrameWgsl` in
 * `layer-wgsl.ts`), so a layer with the same `analysisFrame` draws frame meters exactly where the
 * longitude/latitude they came from lies.
 */

/** Kind of an {@link AnalysisFrame}. */
export type AnalysisFrameKind = AnalysisFrame['kind'];

/** An azimuthal analysis frame about `origin` (`[longitude, latitude]` degrees). */
export type AzimuthalAnalysisFrame = {
  kind: 'azimuthal-equidistant' | 'lambert-equal-area';
  origin: readonly [number, number];
};

/**
 * A spherical Albers equal-area conic frame. `origin` is `[central meridian, latitude of origin]`
 * (the frame's `(0, 0)`), `standardParallels` the two latitudes of true scale, all in degrees.
 */
export type AlbersAnalysisFrame = {
  kind: 'albers-equal-area';
  origin: readonly [number, number];
  standardParallels: readonly [number, number];
};

/** A planar analysis frame (meters on the sphere of radius `GPU_EARTH_MEAN_RADIUS_METERS`). */
export type AnalysisFrame = AzimuthalAnalysisFrame | AlbersAnalysisFrame;

/**
 * Albers CONUS (USGS parameters: parallels 29.5 and 45.5 N, origin 37.5 N 96 W). Its CPU
 * projection is bit-identical to `projectAlbers` (scenes/regression/b6-polygons.ts), so layers
 * draw `projectAlbers` meters with `analysisFrame: ALBERS_CONUS_FRAME`.
 */
export const ALBERS_CONUS_FRAME: AlbersAnalysisFrame = {
  kind: 'albers-equal-area',
  origin: [-96, 37.5],
  standardParallels: [29.5, 45.5]
};

/** Forward and inverse of one {@link AnalysisFrame}. */
export type AnalysisFrameProjection = {
  frame: AnalysisFrame;
  /** `[x, y]` meters for one `[longitude, latitude]`. */
  project: (longitude: number, latitude: number) => [number, number];
  /** `[longitude, latitude]` for `[x, y]` meters. */
  unproject: (x: number, y: number) => [number, number];
};

const DEGREES_TO_RADIANS = Math.PI / 180;

/**
 * Spherical projection of `frame`: azimuthal (equidistant or Lambert equal-area) about
 * `frame.origin`, or Albers equal-area conic. The equidistant formulas are those of
 * `createAzimuthalEquidistant` (scenes/movement/b12-tracks.ts), the Albers ones those of
 * `projectAlbers` (scenes/regression/b6-polygons.ts), in the same operation order.
 */
export function createAnalysisFrameProjection(frame: AnalysisFrame): AnalysisFrameProjection {
  if (frame.kind === 'albers-equal-area') return createAlbersProjection(frame);
  const radius = GPU_EARTH_MEAN_RADIUS_METERS;
  const equalArea = frame.kind === 'lambert-equal-area';
  const [longitude0, latitude0] = frame.origin;
  const lambda0 = longitude0 * DEGREES_TO_RADIANS;
  const phi0 = latitude0 * DEGREES_TO_RADIANS;
  const sinPhi0 = Math.sin(phi0);
  const cosPhi0 = Math.cos(phi0);
  return {
    frame,
    project(longitude, latitude) {
      const deltaLambda = longitude * DEGREES_TO_RADIANS - lambda0;
      const phi = latitude * DEGREES_TO_RADIANS;
      const cosC = sinPhi0 * Math.sin(phi) + cosPhi0 * Math.cos(phi) * Math.cos(deltaLambda);
      let k: number;
      if (equalArea) {
        k = Math.sqrt(2 / Math.max(1 + cosC, 1e-15));
      } else {
        const c = Math.acos(Math.min(1, Math.max(-1, cosC)));
        k = c < 1e-9 ? 1 : c / Math.sin(c);
      }
      return [
        radius * k * Math.cos(phi) * Math.sin(deltaLambda),
        radius * k * (cosPhi0 * Math.sin(phi) - sinPhi0 * Math.cos(phi) * Math.cos(deltaLambda))
      ];
    },
    unproject(x, y) {
      const rho = Math.hypot(x, y);
      if (rho < 1e-9) return [longitude0, latitude0];
      const c = equalArea ? 2 * Math.asin(Math.min(1, rho / (2 * radius))) : rho / radius;
      const sinC = Math.sin(c);
      const cosC = Math.cos(c);
      const phi = Math.asin(
        Math.min(1, Math.max(-1, cosC * sinPhi0 + ((y * sinC) / rho) * cosPhi0))
      );
      const deltaLambda = Math.atan2(x * sinC, rho * cosPhi0 * cosC - y * sinPhi0 * sinC);
      return [longitude0 + deltaLambda / DEGREES_TO_RADIANS, phi / DEGREES_TO_RADIANS];
    }
  };
}

/** Constants of a spherical Albers frame, in units of the sphere radius. */
export type AlbersConstants = {
  /** Cone constant `n = (sin phi1 + sin phi2) / 2`. */
  n: number;
  /** `C = cos(phi1)^2 + 2 n sin(phi1)`. */
  c: number;
  /** Radius of the parallel of origin, `sqrt(C - 2 n sin(phi0)) / n`. */
  rho0: number;
};

/** The constants of an Albers frame, computed as `projectAlbers` computes them. */
export function getAlbersConstants(frame: AlbersAnalysisFrame): AlbersConstants {
  const [parallel1, parallel2] = frame.standardParallels;
  const n =
    (Math.sin(parallel1 * DEGREES_TO_RADIANS) + Math.sin(parallel2 * DEGREES_TO_RADIANS)) / 2;
  const c =
    Math.cos(parallel1 * DEGREES_TO_RADIANS) ** 2 +
    2 * n * Math.sin(parallel1 * DEGREES_TO_RADIANS);
  const rho0 = Math.sqrt(c - 2 * n * Math.sin(frame.origin[1] * DEGREES_TO_RADIANS)) / n;
  return {n, c, rho0};
}

/** Spherical Albers equal-area conic; `project` matches `projectAlbers` bit for bit. */
function createAlbersProjection(frame: AlbersAnalysisFrame): AnalysisFrameProjection {
  const radius = GPU_EARTH_MEAN_RADIUS_METERS;
  const {n, c, rho0} = getAlbersConstants(frame);
  const longitude0 = frame.origin[0];
  return {
    frame,
    project(longitude, latitude) {
      const rho = Math.sqrt(c - 2 * n * Math.sin(latitude * DEGREES_TO_RADIANS)) / n;
      const theta = n * (longitude - longitude0) * DEGREES_TO_RADIANS;
      return [radius * rho * Math.sin(theta), radius * (rho0 - rho * Math.cos(theta))];
    },
    unproject(x, y) {
      const east = x / radius;
      const north = rho0 - y / radius;
      // n > 0 for northern-hemisphere parallels: the cone opens to the south.
      const rho = Math.hypot(east, north);
      const theta = Math.atan2(east, north);
      const sinPhi = Math.min(1, Math.max(-1, (c - rho * rho * n * n) / (2 * n)));
      return [longitude0 + theta / n / DEGREES_TO_RADIANS, Math.asin(sinPhi) / DEGREES_TO_RADIANS];
    }
  };
}

/** Projects interleaved `[longitude, latitude, ...]` (stride in floats) to frame meters. */
export function projectLngLatToFrame(
  projection: AnalysisFrameProjection,
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

/**
 * `[minX, minY, maxX, maxY]` frame meters covering the viewport. Each screen edge is sampled at
 * `samplesPerEdge + 1` points, because straight screen edges are curves in an azimuthal or conic
 * frame (corners alone miss the bulge of the northern edge).
 */
export function getViewportFrameBounds(
  viewport: Viewport,
  projection: AnalysisFrameProjection,
  samplesPerEdge = 8
): [number, number, number, number] {
  const {width, height} = viewport;
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const add = (pixelX: number, pixelY: number) => {
    const [longitude, latitude] = viewport.unproject([pixelX, pixelY]);
    const [x, y] = projection.project(longitude, latitude);
    bounds[0] = Math.min(bounds[0], x);
    bounds[1] = Math.min(bounds[1], y);
    bounds[2] = Math.max(bounds[2], x);
    bounds[3] = Math.max(bounds[3], y);
  };
  for (let sample = 0; sample <= samplesPerEdge; sample++) {
    const t = sample / samplesPerEdge;
    add(t * width, 0);
    add(t * width, height);
    add(0, t * height);
    add(width, t * height);
  }
  return bounds;
}
