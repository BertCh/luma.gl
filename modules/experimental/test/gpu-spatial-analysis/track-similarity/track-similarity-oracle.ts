// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU references for `GPUTrackSimilarity` in float64, matching Shapely/GEOS semantics. */

/** Vertex list as `[x, y]` pairs. */
export type Polyline = readonly (readonly [number, number])[];

/** GEOS densification: `round(1 / densify)` equal parts per segment, plus the last vertex. */
export function densifyPolyline(polyline: Polyline, densify: number = 0): Polyline {
  if (!densify) {
    return polyline;
  }
  const parts = Math.round(1 / densify);
  const points: [number, number][] = [];
  for (let index = 0; index + 1 < polyline.length; index++) {
    const [x0, y0] = polyline[index];
    const [x1, y1] = polyline[index + 1];
    for (let part = 0; part < parts; part++) {
      points.push([x0 + ((x1 - x0) * part) / parts, y0 + ((y1 - y0) * part) / parts]);
    }
  }
  points.push(polyline[polyline.length - 1]);
  return points;
}

function distanceToPolyline(point: readonly [number, number], polyline: Polyline): number {
  if (polyline.length === 1) {
    return Math.hypot(point[0] - polyline[0][0], point[1] - polyline[0][1]);
  }
  let nearest = Infinity;
  for (let index = 0; index + 1 < polyline.length; index++) {
    const [x0, y0] = polyline[index];
    const [x1, y1] = polyline[index + 1];
    const [dx, dy] = [x1 - x0, y1 - y0];
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared
      ? Math.min(1, Math.max(0, ((point[0] - x0) * dx + (point[1] - y0) * dy) / lengthSquared))
      : 0;
    nearest = Math.min(nearest, Math.hypot(point[0] - x0 - t * dx, point[1] - y0 - t * dy));
  }
  return nearest;
}

/**
 * Symmetric Hausdorff distance as Shapely/GEOS computes it: the largest distance from a (densified)
 * vertex of one polyline to the other polyline's segments. NaN when either side is empty.
 */
export function computeHausdorffOracle(a: Polyline, b: Polyline, densify: number = 0): number {
  if (a.length === 0 || b.length === 0) {
    return Number.NaN;
  }
  const directed = (from: Polyline, to: Polyline) =>
    Math.max(...densifyPolyline(from, densify).map(point => distanceToPolyline(point, to)));
  return Math.max(directed(a, b), directed(b, a));
}

/** Largest vertex-to-vertex distance (Sedona `ST_MaxDistance`). NaN when either side is empty. */
export function computeMaxDistanceOracle(a: Polyline, b: Polyline): number {
  if (a.length === 0 || b.length === 0) {
    return Number.NaN;
  }
  return Math.max(...a.map(p => Math.max(...b.map(q => Math.hypot(p[0] - q[0], p[1] - q[1])))));
}

/** Discrete Frechet distance (Eiter and Mannila). NaN when either side is empty. */
export function computeFrechetOracle(rawA: Polyline, rawB: Polyline, densify: number = 0): number {
  const a = densifyPolyline(rawA, densify);
  const b = densifyPolyline(rawB, densify);
  if (a.length === 0 || b.length === 0) {
    return Number.NaN;
  }
  const table: number[][] = [];
  for (let i = 0; i < a.length; i++) {
    table.push([]);
    for (let j = 0; j < b.length; j++) {
      const cost = Math.hypot(a[i][0] - b[j][0], a[i][1] - b[j][1]);
      if (i === 0 && j === 0) {
        table[i].push(cost);
      } else if (i === 0) {
        table[i].push(Math.max(table[i][j - 1], cost));
      } else if (j === 0) {
        table[i].push(Math.max(table[i - 1][j], cost));
      } else {
        table[i].push(
          Math.max(Math.min(table[i - 1][j], table[i - 1][j - 1], table[i][j - 1]), cost)
        );
      }
    }
  }
  return table[a.length - 1][b.length - 1];
}
