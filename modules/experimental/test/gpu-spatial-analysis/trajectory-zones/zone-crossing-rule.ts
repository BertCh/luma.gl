// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact JavaScript replay of the `GPUZoneEvents` boundary rule (`ZONE_TIE_RULE_WGSL` in
 * `zone-events-kernels.ts`). Every track sample `(x, y)` is the perturbed point `(x - e^2, y - e)`
 * with infinitesimal `e > 0`; the polygon is unperturbed. Orientation signs are exact (BigInt over
 * f32 inputs), zero cases are broken by the perturbation.
 */

const SCALE = 2 ** 149;

/** Exact sign of `cross(b - a, c - a)` for coordinates that are f32 values. */
export function orientSign(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number]
): number {
  const big = (value: number) => BigInt(value * SCALE);
  const [ax, ay, bx, by, cx, cy] = [a[0], a[1], b[0], b[1], c[0], c[1]].map(big);
  const determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

type Point = readonly [number, number];

/** Side of the perturbed point of edge c-d. */
export function pointSide(c: Point, d: Point, p: Point): number {
  const side = orientSign(c, d, p);
  if (side !== 0) {
    return side;
  }
  const [ex, ey] = [d[0] - c[0], d[1] - c[1]];
  return ex !== 0 ? -Math.sign(ex) : Math.sign(ey);
}

/** Side of vertex c of the perturbed segment p-q. */
export function segmentSide(p: Point, q: Point, c: Point): number {
  const side = orientSign(p, q, c);
  if (side !== 0) {
    return side;
  }
  const [dx, dy] = [q[0] - p[0], q[1] - p[1]];
  return dx !== 0 ? Math.sign(dx) : -Math.sign(dy);
}

/** True when the perturbed segment p-q crosses the edge a-b. */
export function segmentCrossesEdge(p: Point, q: Point, a: Point, b: Point): boolean {
  return pointSide(a, b, p) !== pointSide(a, b, q) && segmentSide(p, q, a) !== segmentSide(p, q, b);
}

/** True when the +x ray from the perturbed point crosses the edge a-b. */
export function rayCrossesEdge(p: Point, a: Point, b: Point): boolean {
  if (a[1] >= p[1] === b[1] >= p[1]) {
    return false;
  }
  const side = pointSide(a, b, p);
  return b[1] > a[1] ? side > 0 : side < 0;
}

/** Edge list `[[ax, ay], [bx, by]]` of a closed ring given as flat coordinates. */
export function getRingEdges(ring: readonly number[]): [Point, Point][] {
  const count = ring.length / 2;
  const edges: [Point, Point][] = [];
  for (let index = 0; index < count; index++) {
    const next = (index + 1) % count;
    edges.push([
      [ring[2 * index], ring[2 * index + 1]],
      [ring[2 * next], ring[2 * next + 1]]
    ]);
  }
  return edges;
}

/** Inside flag of the perturbed point for rings of one zone (even-odd). */
export function isInsideRule(p: Point, rings: readonly (readonly number[])[]): boolean {
  let parity = false;
  for (const ring of rings) {
    for (const [a, b] of getRingEdges(ring)) {
      if (rayCrossesEdge(p, a, b)) {
        parity = !parity;
      }
    }
  }
  return parity;
}

/** Number of crossings of the perturbed segment p-q with the rings. */
export function countSegmentCrossings(
  p: Point,
  q: Point,
  rings: readonly (readonly number[])[]
): number {
  let count = 0;
  for (const ring of rings) {
    for (const [a, b] of getRingEdges(ring)) {
      if (segmentCrossesEdge(p, q, a, b)) {
        count++;
      }
    }
  }
  return count;
}
