// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU neighbour measures of the point-patterns story, in local metres: the selected point with the
 * most selected neighbours inside r (where the r-ring is drawn), the share of records that sit on
 * an exact shared coordinate, and the distance from one point to its nearest neighbour. They read
 * the same positions and mask the GPU graphs read, so the map and the curves agree.
 */

import type {PatternBounds} from './point-patterns-envelope';

/** Candidate centres examined by {@link findDensestPoint}. */
const MAXIMUM_CENTERS = 6000;

/** The densest selected point: its row and how many other selected points lie within r. */
export type DensestPoint = {row: number; neighbours: number};

/**
 * Finds the selected point (mask `1`, inside `bounds`) with the most other selected points within
 * `radius` metres. A counting-sort grid of `radius`-wide cells keeps it linear: a point only looks
 * at the 3 x 3 cells around it.
 *
 * With more than about 6,000 selected points only every k-th point is tried as a centre (the
 * neighbour count of the point found is still exact).
 *
 * @returns The point, or `null` when nothing is selected inside the window.
 */
export function findDensestPoint(
  positions: Float32Array,
  mask: Uint32Array,
  bounds: Readonly<PatternBounds>,
  radius: number
): DensestPoint | null {
  const columns = Math.max(1, Math.ceil((bounds[2] - bounds[0]) / radius));
  const rows = Math.max(1, Math.ceil((bounds[3] - bounds[1]) / radius));
  const cellOf = new Int32Array(mask.length).fill(-1);
  const starts = new Uint32Array(columns * rows + 1);
  for (let row = 0; row < mask.length; row++) {
    if (!mask[row]) continue;
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    if (!(x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3])) continue;
    const column = Math.min(columns - 1, Math.floor((x - bounds[0]) / radius));
    const cellRow = Math.min(rows - 1, Math.floor((y - bounds[1]) / radius));
    const cell = cellRow * columns + column;
    cellOf[row] = cell;
    starts[cell + 1]++;
  }
  for (let cell = 0; cell < columns * rows; cell++) starts[cell + 1] += starts[cell];
  const fill = starts.slice(0, columns * rows);
  const sorted = new Uint32Array(starts[columns * rows]);
  for (let row = 0; row < mask.length; row++) {
    const cell = cellOf[row];
    if (cell >= 0) sorted[fill[cell]++] = row;
  }
  const radiusSquared = radius * radius;
  // Evaluate at most about MAXIMUM_CENTERS candidate centres (every k-th point); neighbours are
  // always counted over every point, so the count of the chosen point is exact.
  const stride = Math.max(1, Math.ceil(sorted.length / MAXIMUM_CENTERS));
  let seen = 0;
  let best: DensestPoint | null = null;
  for (let row = 0; row < mask.length; row++) {
    const cell = cellOf[row];
    if (cell < 0) continue;
    if (seen++ % stride !== 0) continue;
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    const column = cell % columns;
    const cellRow = Math.floor(cell / columns);
    let count = 0;
    for (let dy = -1; dy <= 1; dy++) {
      const neighbourRow = cellRow + dy;
      if (neighbourRow < 0 || neighbourRow >= rows) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const neighbourColumn = column + dx;
        if (neighbourColumn < 0 || neighbourColumn >= columns) continue;
        const neighbourCell = neighbourRow * columns + neighbourColumn;
        for (let slot = starts[neighbourCell]; slot < starts[neighbourCell + 1]; slot++) {
          const other = sorted[slot];
          if (other === row) continue;
          const offsetX = positions[other * 2] - x;
          const offsetY = positions[other * 2 + 1] - y;
          if (offsetX * offsetX + offsetY * offsetY <= radiusSquared) count++;
        }
      }
    }
    if (!best || count > best.neighbours) best = {row, neighbours: count};
  }
  return best;
}

/**
 * Share of the selected rows that sit on exactly the same coordinate as another selected row
 * (repeat visits to one spot, reused map pins). Read from the loaded positions, not typed.
 *
 * @returns `shared` rows, `total` selected rows and their `share`.
 */
export function countSharedCoordinates(
  positions: Float32Array,
  mask: Uint32Array
): {shared: number; total: number; share: number} {
  const counts = new Map<string, number>();
  let total = 0;
  for (let row = 0; row < mask.length; row++) {
    if (!mask[row]) continue;
    total++;
    const key = `${positions[row * 2]},${positions[row * 2 + 1]}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  let shared = 0;
  for (const count of counts.values()) if (count > 1) shared += count;
  return {shared, total, share: total > 0 ? shared / total : Number.NaN};
}

/**
 * Distance in metres from the point at `row` to its nearest other point among `candidates` (rows
 * already known to lie close by, for example from a grid index). `Infinity` when there is none.
 */
export function getNearestDistance(
  positions: Float32Array,
  row: number,
  candidates: readonly number[]
): number {
  let best = Number.POSITIVE_INFINITY;
  const x = positions[row * 2];
  const y = positions[row * 2 + 1];
  for (const other of candidates) {
    if (other === row) continue;
    const distance = Math.hypot(positions[other * 2] - x, positions[other * 2 + 1] - y);
    if (distance < best) best = distance;
  }
  return best;
}
