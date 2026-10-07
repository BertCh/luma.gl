// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LngLat} from '../cartography/types';
import {MAX_MERCATOR_LATITUDE} from './annotation-geodesic';

/** An axis-aligned box in CSS pixels. */
export type Box = {x: number; y: number; width: number; height: number};

/** A screen point `[x, y]` in CSS pixels. */
export type ScreenPoint = readonly [number, number];

/** Projects `[longitude, latitude]` to CSS pixels from the top-left of the map. */
export type Projector = (coordinate: readonly [number, number]) => readonly [number, number];

/** Projects one coordinate, clamping latitude to the Web Mercator limit so poles stay finite. */
export function projectCoordinate(project: Projector, coordinate: LngLat): ScreenPoint {
  const latitude = Math.max(-MAX_MERCATOR_LATITUDE, Math.min(MAX_MERCATOR_LATITUDE, coordinate[1]));
  return project([coordinate[0], latitude]);
}

/** Projects a list of coordinates. */
export function projectCoordinates(
  project: Projector,
  coordinates: readonly LngLat[]
): ScreenPoint[] {
  return coordinates.map(coordinate => projectCoordinate(project, coordinate));
}

/** Projects several pieces of a path. */
export function projectPieces(
  project: Projector,
  pieces: readonly (readonly LngLat[])[]
): ScreenPoint[][] {
  return pieces.map(piece => projectCoordinates(project, piece));
}

/** SVG path data for pieces of screen points (`M x y L ...`), each closed with `Z` on request. */
export function piecesToPath(pieces: readonly (readonly ScreenPoint[])[], close: boolean): string {
  const parts: string[] = [];
  for (const piece of pieces) {
    if (piece.length === 0) continue;
    for (let i = 0; i < piece.length; i++) {
      parts.push(`${i === 0 ? 'M' : 'L'}${piece[i][0].toFixed(1)} ${piece[i][1].toFixed(1)}`);
    }
    if (close) parts.push('Z');
  }
  return parts.join('');
}

/** `true` when two boxes, each grown by `padding`, overlap. */
export function boxesOverlap(a: Box, b: Box, padding: number): boolean {
  return (
    a.x - padding < b.x + b.width &&
    a.x + a.width + padding > b.x &&
    a.y - padding < b.y + b.height &&
    a.y + a.height + padding > b.y
  );
}

/** `true` when `inner` lies completely inside `outer`. */
export function boxInside(inner: Box, outer: Box): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  );
}

/** Bounding box of screen points (`null` for an empty list). */
export function pointsBounds(points: readonly ScreenPoint[]): Box | null {
  if (points.length === 0) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return {x: minX, y: minY, width: maxX - minX, height: maxY - minY};
}

/** Area-weighted centroid of a ring in coordinate space; the vertex mean for a degenerate ring. */
export function polygonCentroid(ring: readonly LngLat[]): LngLat {
  let area = 0;
  let centroidX = 0;
  let centroidY = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x0, y0] = ring[i];
    const [x1, y1] = ring[(i + 1) % ring.length];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    centroidX += (x0 + x1) * cross;
    centroidY += (y0 + y1) * cross;
  }
  if (Math.abs(area) < 1e-12) {
    const count = Math.max(ring.length, 1);
    return [
      ring.reduce((sum, point) => sum + point[0], 0) / count,
      ring.reduce((sum, point) => sum + point[1], 0) / count
    ];
  }
  return [centroidX / (3 * area), centroidY / (3 * area)];
}

/** The point halfway along a polyline's length (`null` for an empty list). */
export function polylineMidpoint(points: readonly ScreenPoint[]): ScreenPoint | null {
  if (points.length === 0) return null;
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  let remaining = total / 2;
  for (let i = 1; i < points.length; i++) {
    const length = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    if (remaining <= length && length > 0) {
      const t = remaining / length;
      return [
        points[i - 1][0] + (points[i][0] - points[i - 1][0]) * t,
        points[i - 1][1] + (points[i][1] - points[i - 1][1]) * t
      ];
    }
    remaining -= length;
  }
  return points[points.length - 1];
}

/**
 * The ring of a `[west, south, east, north]` rectangle with `steps` points per edge, so a
 * non-linear projection bends the edges instead of cutting corners.
 */
export function boundsRing(
  bounds: readonly [number, number, number, number],
  steps: number
): LngLat[] {
  const [west, south, east, north] = bounds;
  const corners: LngLat[] = [
    [west, south],
    [east, south],
    [east, north],
    [west, north]
  ];
  const ring: LngLat[] = [];
  for (let edge = 0; edge < 4; edge++) {
    const from = corners[edge];
    const to = corners[(edge + 1) % 4];
    for (let step = 0; step < steps; step++) {
      const t = step / steps;
      ring.push([from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t]);
    }
  }
  return ring;
}

/** Path data of a five-point star centred on the origin with the given outer radius. */
export function starPath(radius: number): string {
  const parts: string[] = [];
  for (let i = 0; i < 10; i++) {
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    const distance = i % 2 === 0 ? radius : radius * 0.4;
    parts.push(
      `${i === 0 ? 'M' : 'L'}${(Math.cos(angle) * distance).toFixed(2)} ${(Math.sin(angle) * distance).toFixed(2)}`
    );
  }
  return `${parts.join('')}Z`;
}

/** Path data of a circle of radius `r` centred on `(cx, cy)` (two arcs, so it needs no element). */
export function circlePath(cx: number, cy: number, r: number): string {
  return `M${(cx - r).toFixed(1)} ${cy.toFixed(1)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
}
