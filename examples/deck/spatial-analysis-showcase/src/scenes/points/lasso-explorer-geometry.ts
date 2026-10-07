// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The CPU geometry of the lasso-explorer story: the shapes the GPU tests (rings in planar metres),
 * their interior triangulation for the translucent fill, the bounding box and the grid cells it
 * touches (what `GPUGridIndex` gathers), and the CPU recount that cross-checks the GPU summary.
 * Pure TypeScript, no DOM and no luma.gl.
 */

import earcut from 'earcut';

/** A point in planar metres (or screen pixels for a screen-space shape). */
export type Point = readonly [number, number];

/** `[minX, minY, maxX, maxY]`. */
export type Bounds = readonly [number, number, number, number];

/** Largest polygon the statistics contributor accepts. */
export const VERTEX_CAPACITY = 256;
/** Sides of the polygon that stands in for a circle (outline and the mask path). */
export const CIRCLE_SEGMENTS = 64;
/** Triangle vertices needed by an earcut fill of a ring of {@link VERTEX_CAPACITY} points. */
export const FILL_VERTEX_CAPACITY = (VERTEX_CAPACITY - 2) * 3;

/** Mean of the vertices (a stand-in centre for a circle placed on a lasso). */
export function getCentroid(polygon: readonly Point[]): Point {
  const sum = polygon.reduce((total, point) => [total[0] + point[0], total[1] + point[1]], [0, 0]);
  return [sum[0] / Math.max(polygon.length, 1), sum[1] / Math.max(polygon.length, 1)];
}

/** Bounding box of a ring, or `null` when it has no points. */
export function getRingBounds(ring: readonly Point[]): Bounds | null {
  if (ring.length === 0) return null;
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const [x, y] of ring) {
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }
  return [minimumX, minimumY, maximumX, maximumY];
}

/** A circle as a closed polygon of `segments` sides. */
export function createCircleRing(
  center: Point,
  radius: number,
  segments = CIRCLE_SEGMENTS
): Point[] {
  return Array.from({length: segments}, (_, index) => {
    const angle = (index / segments) * Math.PI * 2;
    return [center[0] + Math.cos(angle) * radius, center[1] + Math.sin(angle) * radius] as Point;
  });
}

/** Radius of the circle with the same area as `areaSquareKilometers`, in metres. */
export function getEqualAreaRadius(areaSquareKilometers: number): number {
  return Math.sqrt((areaSquareKilometers * 1e6) / Math.PI);
}

/** A cheap fingerprint of a ring, to notice when it changed (a drag, a pan under a screen shape). */
export function getRingSignature(ring: readonly Point[]): string {
  let hash = 7;
  for (const [x, y] of ring) {
    hash = (hash * 31 + Math.round(x * 4) * 7 + Math.round(y * 4)) % 1_000_000_007;
  }
  return `${ring.length}:${hash}`;
}

/**
 * Triangulates a ring for the interior tint (earcut, as `buildPolygonMesh` does for a feature).
 * Returns `x, y` per triangle vertex, three vertices per triangle; a self-intersecting lasso still
 * gets a plausible tint, while the GPU keeps the even-odd rule.
 */
export function triangulateRing(ring: readonly Point[]): number[] {
  if (ring.length < 3) return [];
  const flat: number[] = [];
  for (const [x, y] of ring) flat.push(x, y);
  const indices = earcut(flat, null, 2);
  const triangles: number[] = [];
  for (const index of indices) triangles.push(flat[index * 2], flat[index * 2 + 1]);
  return triangles;
}

/** Four segment rows `x0, y0, x1, y1` of a rectangle outline. */
export function getBoundsSegments(bounds: Bounds): number[] {
  const [x0, y0, x1, y1] = bounds;
  return [x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0];
}

/**
 * Segment rows of the grid-index cells that a shape's bounding box overlaps: the cells
 * `GPURegionStatistics` gathers candidates from instead of testing every record. The index is a
 * uniform `gridSize` grid over `gridBounds`; the lines run along the cell edges of the overlapped
 * block, clamped to the grid. Returns an empty array for a box outside the grid.
 */
export function getGridCellSegments(
  shapeBounds: Bounds,
  gridBounds: Bounds,
  gridSize: readonly [number, number]
): number[] {
  const cellWidth = (gridBounds[2] - gridBounds[0]) / gridSize[0];
  const cellHeight = (gridBounds[3] - gridBounds[1]) / gridSize[1];
  const column0 = Math.max(0, Math.floor((shapeBounds[0] - gridBounds[0]) / cellWidth));
  const column1 = Math.min(
    gridSize[0] - 1,
    Math.floor((shapeBounds[2] - gridBounds[0]) / cellWidth)
  );
  const row0 = Math.max(0, Math.floor((shapeBounds[1] - gridBounds[1]) / cellHeight));
  const row1 = Math.min(gridSize[1] - 1, Math.floor((shapeBounds[3] - gridBounds[1]) / cellHeight));
  if (column1 < column0 || row1 < row0) return [];
  const west = gridBounds[0] + column0 * cellWidth;
  const east = gridBounds[0] + (column1 + 1) * cellWidth;
  const south = gridBounds[1] + row0 * cellHeight;
  const north = gridBounds[1] + (row1 + 1) * cellHeight;
  const segments: number[] = [];
  for (let column = column0; column <= column1 + 1; column++) {
    const x = gridBounds[0] + column * cellWidth;
    segments.push(x, south, x, north);
  }
  for (let row = row0; row <= row1 + 1; row++) {
    const y = gridBounds[1] + row * cellHeight;
    segments.push(west, y, east, y);
  }
  return segments;
}

/** Number of grid segment rows needed for a grid of `gridSize` cells (one block of every cell). */
export function getGridSegmentCapacity(gridSize: readonly [number, number]): number {
  return gridSize[0] + 1 + gridSize[1] + 1;
}

/** Reduces a ring to at most `limit` vertices by Douglas-Peucker with a growing tolerance. */
export function simplifyRing(ring: Point[], limit: number): Point[] {
  const points =
    ring.length > 1 &&
    ring[0][0] === ring[ring.length - 1][0] &&
    ring[0][1] === ring[ring.length - 1][1]
      ? ring.slice(0, -1)
      : ring;
  if (points.length <= limit) return points;
  let tolerance = 5;
  let result = points;
  while (result.length > limit) {
    result = douglasPeucker(points, tolerance);
    tolerance *= 1.5;
  }
  return result;
}

function douglasPeucker(points: Point[], tolerance: number): Point[] {
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length > 0) {
    const [start, end] = stack.pop()!;
    let maximum = 0;
    let index = -1;
    const [ax, ay] = points[start];
    const [bx, by] = points[end];
    const length = Math.hypot(bx - ax, by - ay) || 1;
    for (let candidate = start + 1; candidate < end; candidate++) {
      const distance =
        Math.abs(
          (bx - ax) * (ay - points[candidate][1]) - (ax - points[candidate][0]) * (by - ay)
        ) / length;
      if (distance > maximum) {
        maximum = distance;
        index = candidate;
      }
    }
    if (index >= 0 && maximum > tolerance) {
      keep[index] = 1;
      stack.push([start, index], [index, end]);
    }
  }
  return points.filter((_, index) => keep[index]);
}

/** Padded bounds of a position buffer (the grid index covers these). */
export function getPaddedBounds(positions: Float32Array): [number, number, number, number] {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minimumX = Math.min(minimumX, positions[index]);
    maximumX = Math.max(maximumX, positions[index]);
    minimumY = Math.min(minimumY, positions[index + 1]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  const padding = 0.02 * Math.max(maximumX - minimumX, maximumY - minimumY, 1);
  return [minimumX - padding, minimumY - padding, maximumX + padding, maximumY + padding];
}

/** Records inside a shape, and how many of those were logged on a weekend. */
export type InsideSummary = {count: number; weekend: number};

/** CPU even-odd point-in-polygon recount, used only to cross-check the GPU summary. */
export function summariseInsidePolygon(
  positions: Float32Array,
  weekend: Uint8Array,
  polygon: readonly Point[]
): InsideSummary {
  const summary = {count: 0, weekend: 0};
  const bounds = polygon.length >= 3 ? getRingBounds(polygon) : null;
  if (!bounds) return summary;
  for (let row = 0; row < positions.length / 2; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) continue;
    let inside = false;
    for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
      const [xi, yi] = polygon[index];
      const [xj, yj] = polygon[previous];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    if (inside) {
      summary.count++;
      summary.weekend += weekend[row];
    }
  }
  return summary;
}

/** CPU recount inside an axis-aligned rectangle `[minX, minY, maxX, maxY]`. */
export function summariseInsideRectangle(
  positions: Float32Array,
  weekend: Uint8Array,
  bounds: readonly number[]
): InsideSummary {
  const summary = {count: 0, weekend: 0};
  for (let row = 0; row < positions.length / 2; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) {
      summary.count++;
      summary.weekend += weekend[row];
    }
  }
  return summary;
}

/** CPU recount inside a circle. */
export function summariseInsideCircle(
  positions: Float32Array,
  weekend: Uint8Array,
  center: Point,
  radius: number
): InsideSummary {
  const summary = {count: 0, weekend: 0};
  for (let row = 0; row < positions.length / 2; row++) {
    if (Math.hypot(positions[row * 2] - center[0], positions[row * 2 + 1] - center[1]) <= radius) {
      summary.count++;
      summary.weekend += weekend[row];
    }
  }
  return summary;
}
