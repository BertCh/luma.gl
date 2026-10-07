// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/** Polygon inputs of the community areas in the layout `GPUCellCover` reads. */
export type AreaPolygons = {
  names: readonly string[];
  vertices: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Ring edges `x0, y0, x1, y1` in degrees, for the outline layer. */
  outlineSegments: Float32Array;
  /** Per part: bounding box and owning feature, for hover lookup. */
  partBounds: Float32Array;
  partFeature: Uint32Array;
};

export function readAreaPolygons(dataset: LoadedDataset): AreaPolygons {
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partFeature = dataset.column<Uint32Array>('partFeature');
  const names = (dataset.manifest as unknown as {names?: string[]}).names ?? [];
  const featureCount = names.length || dataset.count;
  const featureOffsets = new Uint32Array(featureCount + 1);
  let part = 0;
  for (let feature = 0; feature <= featureCount; feature++) {
    while (part < partFeature.length && partFeature[part] < feature) part++;
    featureOffsets[feature] = part;
  }
  const outline: number[] = [];
  for (let ring = 0; ring < ringOffsets.length - 1; ring++) {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1];
    for (let vertex = first; vertex < last; vertex++) {
      const next = vertex + 1 < last ? vertex + 1 : first;
      outline.push(
        vertices[vertex * 2],
        vertices[vertex * 2 + 1],
        vertices[next * 2],
        vertices[next * 2 + 1]
      );
    }
  }
  const partCount = partFeature.length;
  const partBounds = new Float32Array(partCount * 4);
  for (let partIndex = 0; partIndex < partCount; partIndex++) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const firstRing = polygonOffsets[partIndex];
    const lastRing = polygonOffsets[partIndex + 1];
    for (let vertex = ringOffsets[firstRing]; vertex < ringOffsets[lastRing]; vertex++) {
      minX = Math.min(minX, vertices[vertex * 2]);
      maxX = Math.max(maxX, vertices[vertex * 2]);
      minY = Math.min(minY, vertices[vertex * 2 + 1]);
      maxY = Math.max(maxY, vertices[vertex * 2 + 1]);
    }
    partBounds.set([minX, minY, maxX, maxY], partIndex * 4);
  }
  return {
    names,
    vertices,
    featureOffsets,
    polygonOffsets,
    ringOffsets,
    outlineSegments: Float32Array.from(outline),
    partBounds,
    partFeature
  };
}

/** Even-odd test of a point against every ring of one part. */
function isInsidePart(
  areas: AreaPolygons,
  part: number,
  longitude: number,
  latitude: number
): boolean {
  let inside = false;
  const firstRing = areas.polygonOffsets[part];
  const lastRing = areas.polygonOffsets[part + 1];
  for (let ring = firstRing; ring < lastRing; ring++) {
    const first = areas.ringOffsets[ring];
    const last = areas.ringOffsets[ring + 1];
    for (let current = first, previous = last - 1; current < last; previous = current++) {
      const x0 = areas.vertices[current * 2];
      const y0 = areas.vertices[current * 2 + 1];
      const x1 = areas.vertices[previous * 2];
      const y1 = areas.vertices[previous * 2 + 1];
      if (
        y0 > latitude !== y1 > latitude &&
        longitude < ((x1 - x0) * (latitude - y0)) / (y1 - y0) + x0
      ) {
        inside = !inside;
      }
    }
  }
  return inside;
}

/** Index of the community area under a coordinate, or -1. */
export function findArea(areas: AreaPolygons, longitude: number, latitude: number): number {
  for (let part = 0; part < areas.partFeature.length; part++) {
    const bounds = areas.partBounds.subarray(part * 4, part * 4 + 4);
    if (
      longitude < bounds[0] ||
      longitude > bounds[2] ||
      latitude < bounds[1] ||
      latitude > bounds[3]
    ) {
      continue;
    }
    if (isInsidePart(areas, part, longitude, latitude)) return areas.partFeature[part];
  }
  return -1;
}
