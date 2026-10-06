// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU references for `GPUTrackSimilarity`: vertex-based discrete Hausdorff and Frechet in float64. */

/** Vertex list as `[x, y]` pairs. */
export type Polyline = readonly (readonly [number, number])[];

/** Symmetric discrete Hausdorff distance. NaN when either side is empty. */
export function computeHausdorffOracle(a: Polyline, b: Polyline): number {
  if (a.length === 0 || b.length === 0) {
    return Number.NaN;
  }
  const directed = (from: Polyline, to: Polyline) =>
    Math.max(
      ...from.map(point =>
        Math.min(...to.map(other => Math.hypot(point[0] - other[0], point[1] - other[1])))
      )
    );
  return Math.max(directed(a, b), directed(b, a));
}

/** Discrete Frechet distance (Eiter and Mannila). NaN when either side is empty. */
export function computeFrechetOracle(a: Polyline, b: Polyline): number {
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
