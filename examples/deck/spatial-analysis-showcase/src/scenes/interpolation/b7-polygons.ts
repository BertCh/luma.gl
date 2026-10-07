// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {projectLngLatArray} from '../../engine/projection';

/** Polygon features of a binary-polygon dataset as planar meters in GeoArrow-style offsets. */
export type B7PolygonSet = {
  /** Number of features (zones). */
  featureCount: number;
  /** Ring vertices `x, y` in meters around the origin. Rings close implicitly (no repeated vertex). */
  polygonPositions: Float32Array;
  /** `featureCount + 1` feature-to-polygon (part) offsets. */
  featureOffsets: Uint32Array;
  /** `partCount + 1` part-to-ring offsets. */
  polygonOffsets: Uint32Array;
  /** `ringCount + 1` ring-to-vertex offsets. */
  ringOffsets: Uint32Array;
  /** Ring edges `x0, y0, x1, y1` for drawing outlines. */
  outlineSegments: Float32Array;
  /** `[minX, minY, maxX, maxY]` of every vertex. */
  bounds: [number, number, number, number];
  /** Finds the feature containing a point (meters), or -1. */
  locate: (x: number, y: number) => number;
};

/**
 * Reads the `vertices`, `ringOffsets`, `polygonRingOffsets` and `partFeature` columns of a
 * polygon dataset (see `public/data/chicago-tracts/manifest.json`) and projects them to meters
 * around `origin`. Closing vertices that repeat the first vertex are dropped.
 */
export function loadB7Polygons(
  dataset: LoadedDataset,
  origin: readonly [number, number]
): B7PolygonSet {
  const projection = dataset.getProjection(origin);
  const lngLat = dataset.column<Float32Array>('vertices');
  const sourceRingOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonOffsets = Uint32Array.from(dataset.column<Uint32Array>('polygonRingOffsets'));
  const partFeature = dataset.column<Uint32Array>('partFeature');
  const meters = projectLngLatArray(projection, lngLat, 2);
  const ringCount = sourceRingOffsets.length - 1;

  const positions: number[] = [];
  const ringOffsets: number[] = [0];
  const outline: number[] = [];
  for (let ring = 0; ring < ringCount; ring++) {
    let first = sourceRingOffsets[ring];
    let last = sourceRingOffsets[ring + 1];
    if (
      last - first > 1 &&
      lngLat[first * 2] === lngLat[(last - 1) * 2] &&
      lngLat[first * 2 + 1] === lngLat[(last - 1) * 2 + 1]
    ) {
      last--;
    }
    const start = positions.length / 2;
    for (let vertex = first; vertex < last; vertex++) {
      positions.push(meters[vertex * 2], meters[vertex * 2 + 1]);
    }
    const count = last - first;
    for (let index = 0; index < count; index++) {
      const a = (start + index) * 2;
      const b = (start + ((index + 1) % count)) * 2;
      outline.push(positions[a], positions[a + 1], positions[b], positions[b + 1]);
    }
    ringOffsets.push(positions.length / 2);
    first = last;
  }

  const partCount = partFeature.length;
  const featureCount = partCount === 0 ? 0 : partFeature[partCount - 1] + 1;
  const featureOffsets = new Uint32Array(featureCount + 1);
  for (let part = 0; part < partCount; part++) featureOffsets[partFeature[part] + 1]++;
  for (let feature = 0; feature < featureCount; feature++) {
    featureOffsets[feature + 1] += featureOffsets[feature];
  }

  const polygonPositions = Float32Array.from(positions);
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < polygonPositions.length; index += 2) {
    bounds[0] = Math.min(bounds[0], polygonPositions[index]);
    bounds[1] = Math.min(bounds[1], polygonPositions[index + 1]);
    bounds[2] = Math.max(bounds[2], polygonPositions[index]);
    bounds[3] = Math.max(bounds[3], polygonPositions[index + 1]);
  }
  const ringOffsetArray = Uint32Array.from(ringOffsets);
  return {
    featureCount,
    polygonPositions,
    featureOffsets,
    polygonOffsets,
    ringOffsets: ringOffsetArray,
    outlineSegments: Float32Array.from(outline),
    bounds,
    locate: createLocator(polygonPositions, featureOffsets, polygonOffsets, ringOffsetArray)
  };
}

function createLocator(
  positions: Float32Array,
  featureOffsets: Uint32Array,
  polygonOffsets: Uint32Array,
  ringOffsets: Uint32Array
): (x: number, y: number) => number {
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
      minX = Math.min(minX, positions[vertex * 2]);
      minY = Math.min(minY, positions[vertex * 2 + 1]);
      maxX = Math.max(maxX, positions[vertex * 2]);
      maxY = Math.max(maxY, positions[vertex * 2 + 1]);
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
      for (let part = featureOffsets[feature]; part < featureOffsets[feature + 1]; part++) {
        for (let ring = polygonOffsets[part]; ring < polygonOffsets[part + 1]; ring++) {
          const first = ringOffsets[ring];
          const count = ringOffsets[ring + 1] - first;
          for (let vertex = 0, previous = count - 1; vertex < count; previous = vertex++) {
            const x0 = positions[(first + vertex) * 2];
            const y0 = positions[(first + vertex) * 2 + 1];
            const x1 = positions[(first + previous) * 2];
            const y1 = positions[(first + previous) * 2 + 1];
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
