// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type Point = [number, number];
export type Rectangle = {minX: number; minY: number; maxX: number; maxY: number};

/** f64 Liang-Barsky. Returns `[t0, t1]` or undefined when the segment misses the rectangle. */
export function clipSegment(a: Point, b: Point, rect: Rectangle): [number, number] | undefined {
  const d = [b[0] - a[0], b[1] - a[1]];
  const p = [-d[0], d[0], -d[1], d[1]];
  const q = [a[0] - rect.minX, rect.maxX - a[0], a[1] - rect.minY, rect.maxY - a[1]];
  let t0 = 0;
  let t1 = 1;
  for (let side = 0; side < 4; side++) {
    if (p[side] === 0) {
      if (q[side] < 0) {
        return undefined;
      }
    } else {
      const r = q[side] / p[side];
      if (p[side] < 0) {
        if (r > t1) {
          return undefined;
        }
        t0 = Math.max(t0, r);
      } else {
        if (r < t0) {
          return undefined;
        }
        t1 = Math.min(t1, r);
      }
    }
  }
  if (t1 < t0 || (t1 === t0 && (d[0] !== 0 || d[1] !== 0))) {
    return undefined;
  }
  return [t0, t1];
}

/** Clips open paths; surviving segments join when neither end is clipped at the shared vertex. */
export function clipLines(
  paths: Point[][],
  rect: Rectangle
): {paths: Point[][]; sourcePaths: number[]} {
  const output: Point[][] = [];
  const sourcePaths: number[] = [];
  paths.forEach((path, pathIndex) => {
    let current: Point[] | undefined;
    let previousUnclippedEnd = false;
    for (let index = 0; index + 1 < path.length; index++) {
      const a = path[index];
      const b = path[index + 1];
      const clip = clipSegment(a, b, rect);
      if (!clip) {
        current = undefined;
        previousUnclippedEnd = false;
        continue;
      }
      const start: Point =
        clip[0] === 0 ? a : [a[0] + (b[0] - a[0]) * clip[0], a[1] + (b[1] - a[1]) * clip[0]];
      const end: Point =
        clip[1] === 1 ? b : [a[0] + (b[0] - a[0]) * clip[1], a[1] + (b[1] - a[1]) * clip[1]];
      if (current && previousUnclippedEnd && clip[0] === 0) {
        current.push(end);
      } else {
        current = [start, end];
        output.push(current);
        sourcePaths.push(pathIndex);
      }
      previousUnclippedEnd = clip[1] === 1;
    }
  });
  return {paths: output, sourcePaths};
}

/** f64 Sutherland-Hodgman for one ring in the same stage order as the GPU (left, right, bottom, top). */
export function clipRing(ring: Point[], rect: Rectangle): Point[] {
  let current = ring;
  const stages: [(p: Point) => boolean, (s: Point, e: Point) => Point][] = [
    [
      p => p[0] >= rect.minX,
      (s, e) => [rect.minX, s[1] + ((rect.minX - s[0]) / (e[0] - s[0])) * (e[1] - s[1])]
    ],
    [
      p => p[0] <= rect.maxX,
      (s, e) => [rect.maxX, s[1] + ((rect.maxX - s[0]) / (e[0] - s[0])) * (e[1] - s[1])]
    ],
    [
      p => p[1] >= rect.minY,
      (s, e) => [s[0] + ((rect.minY - s[1]) / (e[1] - s[1])) * (e[0] - s[0]), rect.minY]
    ],
    [
      p => p[1] <= rect.maxY,
      (s, e) => [s[0] + ((rect.maxY - s[1]) / (e[1] - s[1])) * (e[0] - s[0]), rect.maxY]
    ]
  ];
  for (const [isInside, intersect] of stages) {
    const next: Point[] = [];
    for (let index = 0; index < current.length; index++) {
      const e = current[index];
      const s = current[(index + current.length - 1) % current.length];
      if (isInside(e)) {
        if (!isInside(s)) {
          next.push(intersect(s, e));
        }
        next.push(e);
      } else if (isInside(s)) {
        next.push(intersect(s, e));
      }
    }
    current = next;
  }
  return current;
}

/** Shoelace area (absolute) of a ring. */
export function getRingArea(ring: Point[]): number {
  let twice = 0;
  for (let index = 0; index < ring.length; index++) {
    const a = ring[index];
    const b = ring[(index + 1) % ring.length];
    twice += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(twice) / 2;
}
