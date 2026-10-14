// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** f64 distance in meters from a point to the great-circle arc `a..b` (endpoints included). */
export function getArcDistanceOracle(
  point: readonly number[],
  a: readonly number[],
  b: readonly number[]
): number {
  const toVector = ([longitude, latitude]: readonly number[]) => {
    const lambda = (longitude * Math.PI) / 180;
    const phi = (latitude * Math.PI) / 180;
    return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
  };
  const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cross = (u: number[], v: number[]) => [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
  ];
  const angle = (u: number[], v: number[]) => Math.atan2(Math.hypot(...cross(u, v)), dot(u, v));
  const p = toVector(point);
  const start = toVector(a);
  const end = toVector(b);
  let best = Math.min(angle(p, start), angle(p, end));
  const normal = cross(start, end);
  const normalLength = Math.hypot(...normal);
  if (normalLength > 1e-12) {
    // The point's foot on the great circle lies on the arc when it is between both endpoints.
    const unit = normal.map(value => value / normalLength);
    const foot = cross(unit, cross(p, unit));
    const arcAngle = angle(start, end);
    if (angle(start, foot) <= arcAngle && angle(foot, end) <= arcAngle) {
      best = Math.min(best, Math.abs(Math.asin(Math.max(-1, Math.min(1, dot(p, unit))))));
    }
  }
  return best * 6371008.8;
}

/** Reproducible globe-spanning linework and queries for the spherical BVH regression. */
export function createSphericalBVHScene() {
  // 5000 arcs in 100 lines over the globe (antimeridian crossings included) plus three adversarial
  // lines: a long 60-degree-latitude arc whose great circle bulges about 14 degrees north of its
  // endpoints, a near-polar arc, and two identical segments (a tie that must go to the lower row).
  let state = 987;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const lineOffsets = [0];
  const vertices: number[][] = [];
  for (let line = 0; line < 100; line++) {
    let longitude = random() * 360 - 180;
    let latitude = random() * 150 - 75;
    for (let vertex = 0; vertex <= 50; vertex++) {
      vertices.push([longitude, latitude]);
      longitude = ((longitude + (random() - 0.4) * 1.5 + 540) % 360) - 180;
      latitude = Math.max(-80, Math.min(80, latitude + (random() - 0.5) * 1.5));
    }
    lineOffsets.push(vertices.length);
  }
  const bulgeLine = lineOffsets.length - 1;
  vertices.push([-60, 60], [60, 60]);
  lineOffsets.push(vertices.length);
  vertices.push([10, 87], [170, 88]);
  lineOffsets.push(vertices.length);
  const tieLine = lineOffsets.length - 1;
  vertices.push([30.25, -20.5], [30.75, -20.25]);
  lineOffsets.push(vertices.length);
  vertices.push([30.25, -20.5], [30.75, -20.25]);
  lineOffsets.push(vertices.length);
  const points: number[][] = [
    // Near the bulge's apex: the great circle reaches atan(tan 60 / cos 60) = 73.9 degrees at
    // longitude 0, 13.9 degrees north of both endpoints, so an endpoint-only box would prune it.
    [0, 73.5],
    // Near the polar arc.
    [90, 89.2],
    // Equidistant from the duplicated segments.
    [30.5, -20.0]
  ];
  for (let index = 0; index < 2000; index++) {
    const near = vertices[Math.floor(random() * vertices.length)];
    points.push(
      index % 4 === 0
        ? [random() * 360 - 180, random() * 160 - 80]
        : [
            ((near[0] + (random() - 0.5) * 3 + 540) % 360) - 180,
            Math.max(-89, Math.min(89, near[1] + (random() - 0.5) * 3))
          ]
    );
  }
  return {vertices, lineOffsets, points, bulgeLine, tieLine};
}

/** Brute-force nearest arc, runner-up arc, and nearest arc on a different line. */
export function getNearestArcCandidates(
  point: readonly number[],
  vertices: readonly (readonly number[])[],
  lineOffsets: readonly number[]
) {
  let best = {row: -1, line: -1, distance: Infinity};
  let secondDistance = Infinity;
  const lineDistances = new Array<number>(lineOffsets.length - 1).fill(Infinity);
  for (let line = 0; line + 1 < lineOffsets.length; line++) {
    for (let row = lineOffsets[line]; row + 1 < lineOffsets[line + 1]; row++) {
      const distance = getArcDistanceOracle(point, vertices[row], vertices[row + 1]);
      lineDistances[line] = Math.min(lineDistances[line], distance);
      if (distance < best.distance) {
        secondDistance = best.distance;
        best = {row, line, distance};
      } else if (distance < secondDistance) {
        secondDistance = distance;
      }
    }
  }
  const competingLineDistance = Math.min(...lineDistances.filter((_, line) => line !== best.line));
  return {best, secondDistance, competingLineDistance};
}
