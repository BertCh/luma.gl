// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellToBoundary} from 'h3-js';
import {bigIntToH3, quadbinCellToTile} from '../cell-aggregation/cell-aggregation-oracle';
import {webMercatorTileBounds} from '../cell-indexing/cell-indexing-oracle';
import {isInsideRings, type CoverFeature} from './cell-cover-oracle';

type Vertex = [number, number];

/** Counter-clockwise corner ring `[lng, lat]` (open) of a cell, in f64. */
export function getCellRing(family: 'quadbin' | 'h3', cell: bigint): Vertex[] {
  if (family === 'quadbin') {
    const {x, y, z} = quadbinCellToTile(cell);
    const [west, south, east, north] = webMercatorTileBounds(x, y, z);
    return [
      [west, south],
      [east, south],
      [east, north],
      [west, north]
    ];
  }
  return cellToBoundary(bigIntToH3(cell), true) as Vertex[];
}

function orientation(a: Vertex, b: Vertex, c: Vertex): number {
  return Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
}

/** Whether two segments cross at a point interior to both (touching and collinear overlap excluded). */
function segmentsCrossProperly(a: Vertex, b: Vertex, c: Vertex, d: Vertex): boolean {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  return o1 * o2 < 0 && o3 * o4 < 0;
}

/**
 * Exact f64 soundness check of a core cell: every cell corner and the cell midpoint of every cell
 * edge lie inside the feature, and no polygon edge properly crosses a cell edge. Together these
 * mean the cell lies inside the feature (up to boundary touching).
 */
export function isCellInsideFeature(
  feature: CoverFeature,
  family: 'quadbin' | 'h3',
  cell: bigint
): boolean {
  const ring = getCellRing(family, cell);
  const rings = feature.flatMap(polygon => polygon) as Vertex[][];
  for (let index = 0; index < ring.length; index++) {
    const a = ring[index];
    const b = ring[(index + 1) % ring.length];
    const midpoint: Vertex = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    if (!isInsideRings(rings, a, true) || !isInsideRings(rings, midpoint, true)) {
      return false;
    }
    for (const polygonRing of rings) {
      for (let edge = 0; edge < polygonRing.length; edge++) {
        if (
          segmentsCrossProperly(
            a,
            b,
            polygonRing[edge],
            polygonRing[(edge + 1) % polygonRing.length]
          )
        ) {
          return false;
        }
      }
    }
  }
  return true;
}
