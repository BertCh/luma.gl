// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type Ring = number[][];
export type Descriptors = {
  area: number;
  perimeter: number;
  polsbyPopper: number;
  schwartzberg: number;
  elongation: number;
  orientation: number;
  convexity: number;
  clockwise: number;
  sliver: number;
};

function getRingSums(ring: Ring, origin: number[]) {
  let twiceArea = 0;
  let momentX = 0;
  let momentY = 0;
  let xx = 0;
  let yy = 0;
  let xy = 0;
  let perimeter = 0;
  for (let index = 0; index < ring.length; index++) {
    const p = ring[(index + ring.length - 1) % ring.length];
    const c = ring[index];
    const px = p[0] - origin[0];
    const py = p[1] - origin[1];
    const cx = c[0] - origin[0];
    const cy = c[1] - origin[1];
    const cross = px * cy - cx * py;
    twiceArea += cross;
    momentX += cross * (px + cx);
    momentY += cross * (py + cy);
    xx += cross * (px * px + px * cx + cx * cx);
    yy += cross * (py * py + py * cy + cy * cy);
    xy += cross * (px * cy + 2 * px * py + 2 * cx * cy + cx * py);
    perimeter += Math.hypot(cx - px, cy - py);
  }
  return {twiceArea, momentX, momentY, xx, yy, xy, perimeter};
}

function getConvexHullArea(points: number[][]): number {
  const sorted = points.map(p => [p[0], p[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: number[][] = [];
  for (const point of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0
    ) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper: number[][] = [];
  for (const point of sorted.reverse()) {
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0
    ) {
      upper.pop();
    }
    upper.push(point);
  }
  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  let twiceArea = 0;
  for (let index = 0; index < hull.length; index++) {
    const a = hull[index];
    const b = hull[(index + 1) % hull.length];
    twiceArea += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(twiceArea) / 2;
}

/** f64 reference of every `GPUShapeDescriptors` column for one feature (`rings[0]` exterior). */
export function describeFeature(
  rings: Ring[],
  holeRule: 'winding' | 'first-ring-exterior',
  sliverThreshold: number
): Descriptors {
  const origin = rings[0][0];
  const sums = rings.map(ring => getRingSums(ring, origin));
  const signs = sums.map((sum, ring) =>
    holeRule === 'winding' ? 1 : (ring === 0 ? 1 : -1) * Math.sign(sum.twiceArea)
  );
  const twiceArea = sums.reduce((total, sum, ring) => total + signs[ring] * sum.twiceArea, 0);
  const area = Math.abs(twiceArea / 2);
  const perimeter = sums.reduce((total, sum) => total + sum.perimeter, 0);
  const centroidX =
    sums.reduce((total, sum, ring) => total + signs[ring] * sum.momentX, 0) / (3 * twiceArea);
  const centroidY =
    sums.reduce((total, sum, ring) => total + signs[ring] * sum.momentY, 0) / (3 * twiceArea);
  // Central moments from raw moments about the origin vertex (parallel axis theorem).
  const rawXX = sums.reduce((total, sum, ring) => total + (signs[ring] * sum.xx) / 12, 0);
  const rawYY = sums.reduce((total, sum, ring) => total + (signs[ring] * sum.yy) / 12, 0);
  const rawXY = sums.reduce((total, sum, ring) => total + (signs[ring] * sum.xy) / 24, 0);
  const signedArea = twiceArea / 2;
  const flip = signedArea < 0 ? -1 : 1;
  const xx = flip * (rawXX - signedArea * centroidX * centroidX);
  const yy = flip * (rawYY - signedArea * centroidY * centroidY);
  const xy = flip * (rawXY - signedArea * centroidX * centroidY);
  const mean = (xx + yy) / 2;
  const radius = Math.sqrt(((xx - yy) / 2) ** 2 + xy * xy);
  const major = mean + radius;
  const minor = Math.max(mean - radius, 0);
  const polsbyPopper = (4 * Math.PI * area) / (perimeter * perimeter);
  const hullArea = getConvexHullArea(rings.flat());
  return {
    area,
    perimeter,
    polsbyPopper,
    schwartzberg: Math.sqrt(polsbyPopper),
    elongation: 1 - Math.sqrt(minor / major),
    orientation: 0.5 * Math.atan2(2 * xy, xx - yy),
    convexity: Math.min(area / hullArea, 1),
    clockwise: sums[0].twiceArea < 0 ? 1 : 0,
    sliver: polsbyPopper < sliverThreshold ? 1 : 0
  };
}
