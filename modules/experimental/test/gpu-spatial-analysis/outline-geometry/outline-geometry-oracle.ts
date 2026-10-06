// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type Point = [number, number];

/** f64 reference of the planar triangle list of `GPUOutlineGeometry`. */
export function buildOutlineTriangles(
  vertices: Point[],
  pathOffsets: number[] | undefined,
  geometryType: 'points' | 'lines' | 'rings',
  distance: number,
  joinSegments: number
): Point[] {
  const output: Point[] = [];
  const findPath = (row: number) => {
    if (!pathOffsets) {
      return -1;
    }
    for (let path = 0; path + 1 < pathOffsets.length; path++) {
      if (row >= pathOffsets[path] && row < pathOffsets[path + 1]) {
        return path;
      }
    }
    return -1;
  };
  for (let row = 0; row < vertices.length; row++) {
    const center = vertices[row];
    for (let k = 0; k < joinSegments; k++) {
      const angleA = (2 * Math.PI * k) / joinSegments;
      const angleB = (2 * Math.PI * (k + 1)) / joinSegments;
      output.push(
        center,
        [center[0] + distance * Math.cos(angleA), center[1] + distance * Math.sin(angleA)],
        [center[0] + distance * Math.cos(angleB), center[1] + distance * Math.sin(angleB)]
      );
    }
    let next = -1;
    const path = geometryType === 'points' ? -1 : findPath(row);
    if (path >= 0 && pathOffsets) {
      const start = pathOffsets[path];
      const end = pathOffsets[path + 1];
      if (row + 1 < end) {
        next = row + 1;
      } else if (geometryType === 'rings' && end - start >= 2) {
        next = start;
      }
    }
    const quad: Point[] = [center, center, center, center, center, center];
    if (next >= 0) {
      const b = vertices[next];
      const dx = b[0] - center[0];
      const dy = b[1] - center[1];
      const length = Math.hypot(dx, dy);
      if (length > 0) {
        const nx = -dy / length;
        const ny = dx / length;
        const aLeft: Point = [center[0] + nx * distance, center[1] + ny * distance];
        const aRight: Point = [center[0] - nx * distance, center[1] - ny * distance];
        const bRight: Point = [b[0] - nx * distance, b[1] - ny * distance];
        const bLeft: Point = [b[0] + nx * distance, b[1] + ny * distance];
        quad.splice(0, 6, aLeft, aRight, bRight, aLeft, bRight, bLeft);
      }
    }
    output.push(...quad);
  }
  return output;
}

/** Distance from `point` to a segment. */
export function getPointSegmentDistance(point: Point, a: Point, b: Point): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  let t = lengthSquared > 0 ? ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSquared : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

/** Whether any triangle of a list contains `point` (closed). */
export function isCoveredByTriangles(point: Point, triangles: Point[]): boolean {
  for (let index = 0; index + 2 < triangles.length; index += 3) {
    const [a, b, c] = [triangles[index], triangles[index + 1], triangles[index + 2]];
    const cross = (p: Point, q: Point, r: Point) =>
      (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    const d1 = cross(a, b, point);
    const d2 = cross(b, c, point);
    const d3 = cross(c, a, point);
    // Degenerate triangles cover nothing.
    if (cross(a, b, c) > 0 && d1 >= 0 && d2 >= 0 && d3 >= 0) {
      return true;
    }
  }
  return false;
}
