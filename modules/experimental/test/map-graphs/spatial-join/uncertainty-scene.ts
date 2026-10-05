// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OraclePolygonFeature} from './spatial-join-oracle';

export type UncertaintyPoint = [number, number];

/** Next representable float32 above (or below, with a negative direction) `value`. */
function stepFloat32(value: number, direction: 1 | -1): number {
  const floats = new Float32Array([value]);
  const bits = new Uint32Array(floats.buffer);
  bits[0] += value > 0 === direction > 0 ? 1 : -1;
  return floats[0];
}

/**
 * Ring whose far edges (above the test point at `(ox + 500, 500)`) are nearly collinear with the
 * point: `(ox + 1000, 1000) -> (ox + 800, 800 + 1 ulp) -> (ox + 600, 600)`. The point is inside.
 * Their supporting lines pass within float32 ulps of the point although they cannot be crossed by
 * the +x ray or contain the point.
 */
function nearCollinearAboveRing(ox: number): number[][] {
  return [
    [ox, 0],
    [ox + 1000, 0],
    [ox + 1000, 1000],
    [ox + 800, stepFloat32(800, 1)],
    [ox + 600, 600],
    [ox, 1000]
  ];
}

/** Mirror image: far edges below the point, `(ox + 1000, 0) -> (ox + 800, 200 - 1 ulp) -> (ox + 600, 400)`. */
function nearCollinearBelowRing(ox: number): number[][] {
  return [
    [ox, 1000],
    [ox + 1000, 1000],
    [ox + 1000, 0],
    [ox + 800, stepFloat32(200, -1)],
    [ox + 600, 400],
    [ox, 0]
  ];
}

/** Hole whose far edges above the point `(ox + 500, 500)` are near-collinear with it. */
function nearCollinearHole(ox: number): number[][] {
  return [
    [ox + 950, 950],
    [ox + 800, stepFloat32(800, 1)],
    [ox + 600, 600],
    [ox + 300, 900],
    [ox + 900, 950]
  ];
}

const SHELL = (ox: number): number[][] => [
  [ox, 0],
  [ox + 1000, 0],
  [ox + 1000, 1000],
  [ox, 1000]
];

/** Rings with near-collinear far edges: above, below, in a hole, and in a multipolygon. */
export const UNCERTAINTY_FEATURES: OraclePolygonFeature[] = [
  [[nearCollinearAboveRing(0)]],
  [[nearCollinearBelowRing(2000)]],
  [[SHELL(4000), nearCollinearHole(4000)]],
  [[nearCollinearAboveRing(6000)], [nearCollinearBelowRing(8000)]]
];

export const UNCERTAINTY_POINTS: UncertaintyPoint[] = [
  // Near-collinear far edges.
  [500, 500],
  [2500, 500],
  [4500, 500],
  [6500, 500],
  [8500, 500],
  // Far inside, outside, and in the hole.
  [900, 100],
  [2100, 900],
  [4100, 100],
  [4400, 750],
  [6900, 100],
  [8900, 900],
  [500, 1500],
  [-500, 500],
  [5500, 500]
];
