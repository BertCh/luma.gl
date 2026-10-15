// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {S3Polygons} from './s3-lens-data';

/**
 * Closed shell rings of a four-level polygon dataset, one path per feature, for drawing. Feature `f`
 * is polygon `featureOffsets[f]` and its shell is that polygon's first ring (`polygonOffsets`);
 * holes and extra rings stay in the computation and are not drawn. Offsets are path starts plus the
 * final end, so the result view draws one path per feature in feature order.
 */
export function getFeatureShells(polygons: S3Polygons): {
  positions: Float32Array;
  offsets: Uint32Array;
} {
  const positions: number[] = [];
  const offsets = [0];
  for (let feature = 0; feature < polygons.featureCount; feature++) {
    const ring = polygons.polygonOffsets[polygons.featureOffsets[feature]];
    const start = polygons.ringOffsets[ring];
    const end = polygons.ringOffsets[ring + 1];
    for (let vertex = start; vertex < end; vertex++) {
      positions.push(
        polygons.positionsProjected[2 * vertex],
        polygons.positionsProjected[2 * vertex + 1]
      );
    }
    // Close the drawn ring with its first vertex.
    positions.push(
      polygons.positionsProjected[2 * start],
      polygons.positionsProjected[2 * start + 1]
    );
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

/** Projected extent `[minX, minY, maxX, maxY]` of interleaved x, y positions, snapped outwards to `cellSize`. */
export function getSnappedExtent(
  positions: Float32Array,
  cellSize: number
): [number, number, number, number] {
  const extent: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 2) {
    extent[0] = Math.min(extent[0], positions[index]);
    extent[1] = Math.min(extent[1], positions[index + 1]);
    extent[2] = Math.max(extent[2], positions[index]);
    extent[3] = Math.max(extent[3], positions[index + 1]);
  }
  return [
    Math.floor(extent[0] / cellSize) * cellSize - cellSize,
    Math.floor(extent[1] / cellSize) * cellSize - cellSize,
    Math.ceil(extent[2] / cellSize) * cellSize + cellSize,
    Math.ceil(extent[3] / cellSize) * cellSize + cellSize
  ];
}
