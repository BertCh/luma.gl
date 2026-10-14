// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Viewport} from '@deck.gl/core';
import {
  createGPULocalFrame,
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS,
  GPU_WGS84_FLATTENING
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {GPUSpatialContext} from '@luma.gl/experimental/gpu-spatial-analysis';

/**
 * Analysis frames project longitude/latitude to planar meters for GPU analysis. The display-only
 * Mercator-local `LocalMetricProjection` must not be used for analysis.
 *
 * - `'azimuthal-equidistant'`: the library's spherical projection on the mean Earth radius.
 *   Distances and bearings from the origin are exact; tangential scale is `c / sin(c)`.
 * - `'lambert-equal-area'`: the library's ellipsoidal WGS84 Lambert azimuthal equal-area.
 *   Areas are exact ellipsoidal areas.
 * - `'albers-equal-area'`: showcase-local spherical Albers on the mean Earth radius, with a
 *   `'crs'` frame id. Areas on that sphere are exact; scale is true on the standard parallels.
 *
 * Every analysis operator fed frame meters must receive `projection.spatialContext`. Distance
 * operators reject Lambert frames; area operators reject AEQD frames. Layers invert these same
 * frames in WGSL (`getAnalysisFrameWgsl` in `layer-wgsl.ts`) to draw their meters at longitude/latitude.
 */

/** Kind of an {@link AnalysisFrame}. */
export type AnalysisFrameKind = AnalysisFrame['kind'];

/** A library AEQD (spherical) or Lambert (WGS84 ellipsoidal) frame about `origin` in degrees. */
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

/** A planar analysis frame in meters: library azimuthal or showcase-local spherical Albers. */
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
  /** Pass this frame identity and metric contract to every operator consuming frame meters. */
  readonly spatialContext: GPUSpatialContext;
  /** `[x, y]` meters for one `[longitude, latitude]`. */
  project: (longitude: number, latitude: number) => [number, number];
  /** `[longitude, latitude]` for `[x, y]` meters, longitude continuous about the origin. */
  unproject: (x: number, y: number) => [number, number];
};

const DEGREES_TO_RADIANS = Math.PI / 180;

/**
 * Projects with the library's AEQD or ellipsoidal Lambert frame, or showcase-local Albers.
 * Albers preserves the operation order of `projectAlbers` (scenes/regression/b6-polygons.ts).
 * The inverse keeps longitude continuous about the origin, including across the antimeridian.
 */
export function createAnalysisFrameProjection(frame: AnalysisFrame): AnalysisFrameProjection {
  if (frame.kind === 'albers-equal-area') return createAlbersProjection(frame);
  const localFrame = createGPULocalFrame({origin: frame.origin, kind: frame.kind});
  return {
    frame,
    spatialContext: localFrame.spatialContext,
    project: localFrame.project,
    unproject(x, y) {
      const [longitude, latitude] = localFrame.unproject(x, y);
      const deltaLongitude = (longitude - frame.origin[0]) % 360;
      const wrappedDelta =
        deltaLongitude > 180
          ? deltaLongitude - 360
          : deltaLongitude < -180
            ? deltaLongitude + 360
            : deltaLongitude;
      return [frame.origin[0] + wrappedDelta, latitude];
    }
  };
}

/** Returns the same spatial context and frame id carried by this frame's projection. */
export function getAnalysisFrameSpatialContext(frame: AnalysisFrame): GPUSpatialContext {
  if (frame.kind !== 'albers-equal-area') {
    return createGPULocalFrame({origin: frame.origin, kind: frame.kind}).spatialContext;
  }
  return Object.freeze({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: Object.freeze({
      kind: 'crs',
      id: `showcase-albers-${[...frame.origin, ...frame.standardParallels].map(String).join('-')}`
    })
  });
}

/** Binary64 WGS84 constants for the ellipsoidal Lambert inverse used by layer WGSL. */
export function getLambertFrameConstants(frame: AzimuthalAnalysisFrame): {
  authalicRadius: number;
  scaleD: number;
  sinAuthalicOrigin: number;
  cosAuthalicOrigin: number;
  seriesA: number;
  seriesB: number;
  seriesC: number;
} {
  const eccentricitySquared = GPU_WGS84_FLATTENING * (2 - GPU_WGS84_FLATTENING);
  const eccentricity = Math.sqrt(eccentricitySquared);
  const getAuthalicQ = (sinLatitude: number): number => {
    const eccentricitySine = eccentricity * sinLatitude;
    return (
      (1 - eccentricitySquared) *
      (sinLatitude / (1 - eccentricitySine * eccentricitySine) -
        (1 / (2 * eccentricity)) * Math.log((1 - eccentricitySine) / (1 + eccentricitySine)))
    );
  };
  const polarAuthalicQ = getAuthalicQ(1);
  const authalicRadius = GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.sqrt(polarAuthalicQ / 2);
  const originLatitude = frame.origin[1] * DEGREES_TO_RADIANS;
  const sinGeodeticOrigin = Math.sin(originLatitude);
  const authalicOrigin = Math.asin(
    Math.max(-1, Math.min(1, getAuthalicQ(sinGeodeticOrigin) / polarAuthalicQ))
  );
  // Match the library's polar-origin threshold and exact polar components.
  const polar = Math.PI / 2 - Math.abs(originLatitude) < 1e-10;
  const sinAuthalicOrigin = polar ? Math.sign(originLatitude) : Math.sin(authalicOrigin);
  const cosAuthalicOrigin = polar ? 0 : Math.cos(authalicOrigin);
  const scaleD = polar
    ? 1
    : (GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.cos(originLatitude)) /
      Math.sqrt(1 - eccentricitySquared * sinGeodeticOrigin * sinGeodeticOrigin) /
      (authalicRadius * cosAuthalicOrigin);
  return {
    authalicRadius,
    scaleD,
    sinAuthalicOrigin,
    cosAuthalicOrigin,
    seriesA:
      eccentricitySquared / 3 +
      (31 * eccentricitySquared ** 2) / 180 +
      (517 * eccentricitySquared ** 3) / 5040,
    seriesB: (23 * eccentricitySquared ** 2) / 360 + (251 * eccentricitySquared ** 3) / 3780,
    seriesC: (761 * eccentricitySquared ** 3) / 45360
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
    spatialContext: getAnalysisFrameSpatialContext(frame),
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
