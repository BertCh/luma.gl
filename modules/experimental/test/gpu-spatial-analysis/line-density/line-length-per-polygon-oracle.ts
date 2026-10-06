// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getHaversine, type Point} from './line-density-oracle';

/** One polygon feature: polygons of rings (ring 0 shell, later rings holes), each `Point[]`. */
export type OraclePolygonFeature = Point[][][];

function isInsideFeature(point: Point, feature: OraclePolygonFeature): boolean {
  let inside = false;
  for (const polygon of feature) {
    for (const ring of polygon) {
      if (ring.length < 3) {
        continue;
      }
      for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
        const [cx, cy] = ring[index];
        const [px, py] = ring[previous];
        if (
          cy > point[1] !== py > point[1] &&
          point[0] < ((px - cx) * (point[1] - cy)) / (py - cy) + cx
        ) {
          inside = !inside;
        }
      }
    }
  }
  return inside;
}

/**
 * f64 reference: cuts every segment at all ring-edge intersections of every polygon feature
 * (no bounding-box prefilter, no crossing bound) and keeps the pieces whose midpoint is inside.
 */
export function computeLineLengthPerPolygon(
  paths: Point[][],
  pathWeights: number[] | undefined,
  features: OraclePolygonFeature[],
  spherical = false,
  radius = 6371008.8
): {lengths: Float64Array; weightedLengths: Float64Array; segmentCounts: Uint32Array} {
  const lengths = new Float64Array(features.length);
  const weightedLengths = new Float64Array(features.length);
  const segmentCounts = new Uint32Array(features.length);
  paths.forEach((path, pathIndex) => {
    const weight = pathWeights ? pathWeights[pathIndex] : 1;
    for (let vertex = 0; vertex + 1 < path.length; vertex++) {
      const a = path[vertex];
      const b = path[vertex + 1];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      if (dx === 0 && dy === 0) {
        continue;
      }
      features.forEach((feature, featureIndex) => {
        const ts = [0, 1];
        for (const polygon of feature) {
          for (const ring of polygon) {
            for (
              let index = 0, previous = ring.length - 1;
              index < ring.length;
              previous = index++
            ) {
              const p = ring[previous];
              const ex = ring[index][0] - p[0];
              const ey = ring[index][1] - p[1];
              const denominator = dx * ey - dy * ex;
              if (denominator === 0) {
                continue;
              }
              const ox = p[0] - a[0];
              const oy = p[1] - a[1];
              const t = (ox * ey - oy * ex) / denominator;
              const u = (ox * dy - oy * dx) / denominator;
              if (t > 0 && t < 1 && u >= 0 && u <= 1) {
                ts.push(t);
              }
            }
          }
        }
        ts.sort((x, y) => x - y);
        let inside = 0;
        for (let piece = 0; piece + 1 < ts.length; piece++) {
          const [t0, t1] = [ts[piece], ts[piece + 1]];
          if (t1 <= t0) {
            continue;
          }
          const mid = (t0 + t1) / 2;
          if (isInsideFeature([a[0] + dx * mid, a[1] + dy * mid], feature)) {
            const start: Point = [a[0] + dx * t0, a[1] + dy * t0];
            const end: Point = [a[0] + dx * t1, a[1] + dy * t1];
            inside += spherical ? getHaversine(start, end, radius) : (t1 - t0) * Math.hypot(dx, dy);
          }
        }
        if (inside > 0) {
          lengths[featureIndex] += inside;
          weightedLengths[featureIndex] += inside * weight;
          segmentCounts[featureIndex]++;
        }
      });
    }
  });
  return {lengths, weightedLengths, segmentCounts};
}
