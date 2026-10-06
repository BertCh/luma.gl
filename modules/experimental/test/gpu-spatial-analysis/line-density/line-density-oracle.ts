// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type Point = [number, number];

/** Haversine distance in meters between lon/lat degree points. */
export function getHaversine(a: Point, b: Point, radius: number): number {
  const toRadians = Math.PI / 180;
  const dLat = (b[1] - a[1]) * toRadians;
  const dLon = (b[0] - a[0]) * toRadians;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a[1] * toRadians) * Math.cos(b[1] * toRadians) * Math.sin(dLon / 2) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(h));
}

/**
 * f64 reference: splits each segment at every grid line and the grid boundary, then assigns each
 * piece to the cell holding its midpoint. Independent of the GPU's incremental traversal.
 */
export function computeLineLengthsPerCell(
  paths: Point[][],
  grid: {minX: number; minY: number; width: number; height: number; columns: number; rows: number},
  spherical = false,
  radius = 6371008.8
): {lengths: Float64Array; pieces: number} {
  const lengths = new Float64Array(grid.columns * grid.rows);
  let pieces = 0;
  for (const path of paths) {
    for (let index = 0; index + 1 < path.length; index++) {
      const a = path[index];
      const b = path[index + 1];
      const ts = new Set<number>([0, 1]);
      for (let column = 0; column <= grid.columns; column++) {
        const t = (grid.minX + column * grid.width - a[0]) / (b[0] - a[0]);
        if (t > 0 && t < 1) {
          ts.add(t);
        }
      }
      for (let row = 0; row <= grid.rows; row++) {
        const t = (grid.minY + row * grid.height - a[1]) / (b[1] - a[1]);
        if (t > 0 && t < 1) {
          ts.add(t);
        }
      }
      const sorted = [...ts].sort((left, right) => left - right);
      for (let k = 0; k + 1 < sorted.length; k++) {
        const t0 = sorted[k];
        const t1 = sorted[k + 1];
        const mid = (t0 + t1) / 2;
        const mx = a[0] + (b[0] - a[0]) * mid;
        const my = a[1] + (b[1] - a[1]) * mid;
        const column = Math.floor((mx - grid.minX) / grid.width);
        const row = Math.floor((my - grid.minY) / grid.height);
        if (column < 0 || column >= grid.columns || row < 0 || row >= grid.rows) {
          continue;
        }
        const p0: Point = [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0];
        const p1: Point = [a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1];
        lengths[row * grid.columns + column] += spherical
          ? getHaversine(p0, p1, radius)
          : (t1 - t0) * Math.hypot(b[0] - a[0], b[1] - a[1]);
        pieces++;
      }
    }
  }
  return {lengths, pieces};
}
