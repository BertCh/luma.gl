// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mode-local helpers shared by the areal-interpolation and segregation modes: a CPU ZIP-code
 * locator (setup and tooltips only; every analysis runs on the GPU).
 */

import type {SpatialAnalysisPolygons} from '../spatial-analysis-data';

/** Finds the ZIP polygon (feature row) containing a point, or -1. */
export type ZipLocator = (x: number, y: number) => number;

/** Creates an even-odd point-in-polygon locator over the ZIP-code features with bbox culling. */
export function createZipLocator(zips: SpatialAnalysisPolygons): ZipLocator {
  const {polygonPositions, featureOffsets, polygonOffsets, ringOffsets} = zips;
  const featureCount = featureOffsets.length - 1;
  const boxes = new Float64Array(featureCount * 4);
  for (let feature = 0; feature < featureCount; feature++) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const firstVertex = ringOffsets[polygonOffsets[featureOffsets[feature]]];
    const lastVertex = ringOffsets[polygonOffsets[featureOffsets[feature + 1]]];
    for (let vertex = firstVertex; vertex < lastVertex; vertex++) {
      const x = polygonPositions[vertex * 2];
      const y = polygonPositions[vertex * 2 + 1];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    boxes.set([minX, minY, maxX, maxY], feature * 4);
  }
  return (x, y) => {
    for (let feature = 0; feature < featureCount; feature++) {
      if (
        x < boxes[feature * 4] ||
        x > boxes[feature * 4 + 2] ||
        y < boxes[feature * 4 + 1] ||
        y > boxes[feature * 4 + 3]
      ) {
        continue;
      }
      let inside = false;
      for (
        let polygon = featureOffsets[feature];
        polygon < featureOffsets[feature + 1];
        polygon++
      ) {
        for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
          const first = ringOffsets[ring];
          const count = ringOffsets[ring + 1] - first;
          for (let vertex = 0, previous = count - 1; vertex < count; previous = vertex++) {
            const x0 = polygonPositions[(first + vertex) * 2];
            const y0 = polygonPositions[(first + vertex) * 2 + 1];
            const x1 = polygonPositions[(first + previous) * 2];
            const y1 = polygonPositions[(first + previous) * 2 + 1];
            if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) {
              inside = !inside;
            }
          }
        }
      }
      if (inside) return feature;
    }
    return -1;
  };
}

/** `[minX, minY, maxX, maxY]` of every polygon vertex, in the polygons' planar meters. */
export function getPolygonBounds(
  zips: Pick<SpatialAnalysisPolygons, 'polygonPositions'>
): [number, number, number, number] {
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  const positions = zips.polygonPositions;
  for (let index = 0; index < positions.length; index += 2) {
    bounds[0] = Math.min(bounds[0], positions[index]);
    bounds[1] = Math.min(bounds[1], positions[index + 1]);
    bounds[2] = Math.max(bounds[2], positions[index]);
    bounds[3] = Math.max(bounds[3], positions[index + 1]);
  }
  return bounds;
}
