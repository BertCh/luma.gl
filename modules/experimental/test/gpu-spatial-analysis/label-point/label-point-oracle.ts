// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type Ring = number[][];

function getSegmentDistance(px: number, py: number, a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  let t = lengthSquared > 0 ? ((px - a[0]) * dx + (py - a[1]) * dy) / lengthSquared : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

/** f64 signed distance to the boundary of all rings, positive inside (even-odd). */
export function getSignedDistance(px: number, py: number, rings: Ring[]): number {
  let inside = false;
  let minimum = Infinity;
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index++) {
      const a = ring[index];
      const b = ring[(index + ring.length - 1) % ring.length];
      if (a[1] > py !== b[1] > py && px < ((b[0] - a[0]) * (py - a[1])) / (b[1] - a[1]) + a[0]) {
        inside = !inside;
      }
      minimum = Math.min(minimum, getSegmentDistance(px, py, a, b));
    }
  }
  return inside ? minimum : -minimum;
}

/**
 * f64 polylabel (Mapbox): best-first cell subdivision with the `d + h * sqrt(2)` upper bound.
 * Returns the pole of inaccessibility and its distance, accurate to `precision`.
 */
export function findPoleOfInaccessibility(
  rings: Ring[],
  precision: number
): {x: number; y: number; distance: number} {
  const points = rings.flat();
  const minX = Math.min(...points.map(p => p[0]));
  const maxX = Math.max(...points.map(p => p[0]));
  const minY = Math.min(...points.map(p => p[1]));
  const maxY = Math.max(...points.map(p => p[1]));
  const cellSize = Math.min(maxX - minX, maxY - minY);
  type Cell = {x: number; y: number; h: number; d: number; max: number};
  const makeCell = (x: number, y: number, h: number): Cell => {
    const d = getSignedDistance(x, y, rings);
    return {x, y, h, d, max: d + h * Math.SQRT2};
  };
  let best = makeCell((minX + maxX) / 2, (minY + maxY) / 2, 0);
  const queue: Cell[] = [];
  if (cellSize === 0) {
    return {x: minX, y: minY, distance: 0};
  }
  const half = cellSize / 2;
  for (let x = minX; x < maxX; x += cellSize) {
    for (let y = minY; y < maxY; y += cellSize) {
      queue.push(makeCell(x + half, y + half, half));
    }
  }
  while (queue.length > 0) {
    queue.sort((a, b) => a.max - b.max);
    const cell = queue.pop() as Cell;
    if (cell.d > best.d) {
      best = cell;
    }
    if (cell.max - best.d <= precision) {
      continue;
    }
    const h = cell.h / 2;
    queue.push(
      makeCell(cell.x - h, cell.y - h, h),
      makeCell(cell.x + h, cell.y - h, h),
      makeCell(cell.x - h, cell.y + h, h),
      makeCell(cell.x + h, cell.y + h, h)
    );
  }
  return {x: best.x, y: best.y, distance: best.d};
}
