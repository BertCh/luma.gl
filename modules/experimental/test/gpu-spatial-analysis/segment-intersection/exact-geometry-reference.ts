// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU reference of the shared exact point-in-ring rule in `src/utils/exact-geometry-wgsl.ts`, for
 * oracles and node specs. Coordinates are rounded to f32 first (as the GPU sees them), and every
 * orientation sign is exact (BigInt over the f32 values scaled by 2^149, so every f32 is an
 * integer). Ties follow the same Simulation of Simplicity rule: a query point `(x, y)` is read as
 * `(x - e^2, y - e)`.
 */

export type Point = readonly [number, number];

const SCALE = 2 ** 149;

function toInteger(value: number): bigint {
  // Every finite f32 is a multiple of 2^-149, so the scaled value is an exact integer double.
  return BigInt(Math.fround(value) * SCALE);
}

function compare(value: number, reference: number): number {
  return value > reference ? 1 : value < reference ? -1 : 0;
}

/** Exact sign of `cross(b - a, c - a)` over the f32-rounded coordinates. */
export function orientSignExact(a: Point, b: Point, c: Point): number {
  const ax = toInteger(a[0]);
  const ay = toInteger(a[1]);
  const bx = toInteger(b[0]);
  const by = toInteger(b[1]);
  const cx = toInteger(c[0]);
  const cy = toInteger(c[1]);
  const determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

/** Side (-1 right, 1 left) of the perturbed point against edge `c -> d`, given `orient(c, d, p)`. */
export function getSoSPointSide(sign: number, c: Point, d: Point): number {
  if (sign !== 0) {
    return sign;
  }
  const cx = Math.fround(c[0]);
  const dx = Math.fround(d[0]);
  if (dx !== cx) {
    return -compare(dx, cx);
  }
  return compare(Math.fround(d[1]), Math.fround(c[1]));
}

/** Side (-1 right, 1 left) of vertex `c` against the perturbed segment `p -> q`, given `orient(p, q, c)`. */
export function getSoSSegmentSide(sign: number, p: Point, q: Point): number {
  if (sign !== 0) {
    return sign;
  }
  const px = Math.fround(p[0]);
  const qx = Math.fround(q[0]);
  if (qx !== px) {
    return compare(qx, px);
  }
  return -compare(Math.fround(q[1]), Math.fround(p[1]));
}

/** Probe of `exactProbeCrossing`: `side` 0 is a point probe, 1 / -1 the point beside segment `start -> end`. */
export type ExactProbe = {point: Point; start: Point; end: Point; side: number};

/** Point probe at `point`. */
export function createPointProbe(point: Point): ExactProbe {
  return {point, start: point, end: point, side: 0};
}

/** Probe infinitesimally left (`side` 1) or right (`side` -1) of the midpoint of `start -> end`. */
export function createSideProbe(start: Point, end: Point, side: 1 | -1): ExactProbe {
  const midpoint: Point = [
    Math.fround(0.5 * Math.fround(Math.fround(start[0]) + Math.fround(end[0]))),
    Math.fround(0.5 * Math.fround(Math.fround(start[1]) + Math.fround(end[1])))
  ];
  return {point: midpoint, start, end, side};
}

/** A side probe of a zero-length segment is read as a point probe (mirror of `isExactPointProbe`). */
function isPointProbe(probe: ExactProbe): boolean {
  return (
    probe.side === 0 ||
    (Math.fround(probe.start[0]) === Math.fround(probe.end[0]) &&
      Math.fround(probe.start[1]) === Math.fround(probe.end[1]))
  );
}

function isAboveProbe(probe: ExactProbe, v: Point): boolean {
  const vy = Math.fround(v[1]);
  const py = Math.fround(probe.point[1]);
  if (vy !== py) {
    return vy > py;
  }
  if (isPointProbe(probe)) {
    return true;
  }
  return probe.side * compare(Math.fround(probe.end[0]), Math.fround(probe.start[0])) <= 0;
}

/** Mirror of WGSL `exactProbeCrossing`: signed crossing and whether a point probe is on the edge. */
export function getExactProbeCrossing(
  probe: ExactProbe,
  c: Point,
  d: Point
): {crossing: number; boundary: boolean} {
  // Non-finite edges contribute no crossing, matching the WGSL uncertainty path.
  if (![...c, ...d].every(value => Number.isFinite(Math.fround(value)))) {
    return {crossing: 0, boundary: false};
  }
  const p = probe.point;
  const straddles = isAboveProbe(probe, c) !== isAboveProbe(probe, d);
  const isPoint = isPointProbe(probe);
  const inBox =
    isPoint &&
    p[0] >= Math.min(c[0], d[0]) &&
    p[0] <= Math.max(c[0], d[0]) &&
    p[1] >= Math.min(c[1], d[1]) &&
    p[1] <= Math.max(c[1], d[1]);
  if (!straddles && !inBox) {
    return {crossing: 0, boundary: false};
  }
  let boundary = false;
  let side: number;
  if (isPoint) {
    const sign = orientSignExact(c, d, p);
    boundary = inBox && sign === 0;
    side = getSoSPointSide(sign, c, d);
  } else {
    const startSign = orientSignExact(c, d, probe.start);
    const endSign = orientSignExact(c, d, probe.end);
    if (startSign === 0 && endSign === 0) {
      const horizontal = Math.fround(probe.end[0]) !== Math.fround(probe.start[0]);
      const axis = horizontal ? 0 : 1;
      const same =
        compare(Math.fround(probe.end[axis]), Math.fround(probe.start[axis])) *
        compare(Math.fround(d[axis]), Math.fround(c[axis]));
      side = probe.side * (same > 0 ? 1 : -1);
    } else if (startSign >= 0 && endSign >= 0) {
      side = 1;
    } else if (startSign <= 0 && endSign <= 0) {
      side = -1;
    } else {
      side = orientSignExact(c, d, p);
      if (side === 0) {
        const dot =
          (d[0] - c[0]) * (probe.end[0] - probe.start[0]) +
          (d[1] - c[1]) * (probe.end[1] - probe.start[1]);
        side = probe.side * Math.sign(dot);
      }
      side = getSoSPointSide(side, c, d);
    }
  }
  if (!straddles) {
    return {crossing: 0, boundary};
  }
  const upward = Math.fround(d[1]) > Math.fround(c[1]);
  const crossing = upward && side > 0 ? 1 : !upward && side < 0 ? -1 : 0;
  return {crossing, boundary};
}

/** Winding number of a ring (implicitly closed) about the probe, and whether a point probe is on it. */
export function getExactRingWinding(
  probe: ExactProbe,
  ring: readonly Point[]
): {winding: number; boundary: boolean} {
  let winding = 0;
  let boundary = false;
  if (ring.length < 3) {
    return {winding, boundary};
  }
  let previous = ring[ring.length - 1];
  for (const current of ring) {
    const result = getExactProbeCrossing(probe, previous, current);
    winding += result.crossing;
    boundary ||= result.boundary;
    previous = current;
  }
  return {winding, boundary};
}

/** Even-odd containment of the SoS-perturbed point over all rings (the WGSL `coverContains`). */
export function isInsideRingsExact(rings: readonly (readonly Point[])[], point: Point): boolean {
  let winding = 0;
  for (const ring of rings) {
    winding += getExactRingWinding(createPointProbe(point), ring).winding;
  }
  return (winding & 1) !== 0;
}

/** Mirror of WGSL `exactSegmentsCross` for the SoS-perturbed segment `p -> q` against edge `c -> d`. */
export function doSegmentsCrossExact(p: Point, q: Point, c: Point, d: Point): boolean {
  if (Math.fround(p[0]) === Math.fround(q[0]) && Math.fround(p[1]) === Math.fround(q[1])) {
    return false;
  }
  const separatesTrack =
    getSoSPointSide(orientSignExact(c, d, p), c, d) !==
    getSoSPointSide(orientSignExact(c, d, q), c, d);
  const separatesEdge =
    getSoSSegmentSide(orientSignExact(p, q, c), p, q) !==
    getSoSSegmentSide(orientSignExact(p, q, d), p, q);
  return separatesTrack && separatesEdge;
}
