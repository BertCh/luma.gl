// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import earcut from 'earcut';
import type {LoadedDataset} from '../../data/catalog';

/**
 * GeoArrow-style polygon layout of a shipped polygon dataset, normalised so that county datasets
 * (feature to polygon offsets) and tract or community-area datasets (part to feature) look alike.
 * Coordinates are longitude and latitude degrees.
 */
export type PolygonLayout = {
  featureCount: number;
  /** Interleaved `lng, lat` vertices. */
  vertices: Float32Array;
  /** Ring to vertex offsets (rings are closed or implicitly closed). */
  ringOffsets: Uint32Array;
  /** Part (polygon) to ring offsets, with a terminal entry. */
  polygonRingOffsets: Uint32Array;
  /** Feature of every part. Parts of one feature are contiguous. */
  partFeature: Uint32Array;
  /** Feature to ring offsets (`featureCount + 1`), the layout `addRateClusterMapRecipe` reads. */
  featureRingOffsets: Uint32Array;
};

/** Reads the polygon columns of a dataset into a {@link PolygonLayout}. */
export function getPolygonLayout(dataset: LoadedDataset): PolygonLayout {
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partCount = polygonRingOffsets.length - 1;
  let partFeature: Uint32Array;
  let featureCount: number;
  if (dataset.hasColumn('partFeature')) {
    partFeature = dataset.column<Uint32Array>('partFeature');
    featureCount = dataset.count;
  } else {
    // Counties: feature i owns parts [countyPolygonOffsets[i], countyPolygonOffsets[i + 1]).
    const featureOffsets = dataset.column<Uint32Array>('countyPolygonOffsets');
    featureCount = featureOffsets.length - 1;
    partFeature = new Uint32Array(partCount);
    for (let feature = 0; feature < featureCount; feature++) {
      for (let part = featureOffsets[feature]; part < featureOffsets[feature + 1]; part++) {
        partFeature[part] = feature;
      }
    }
  }
  const featureRingOffsets = new Uint32Array(featureCount + 1);
  let part = 0;
  for (let feature = 0; feature < featureCount; feature++) {
    featureRingOffsets[feature] = polygonRingOffsets[part];
    while (part < partCount && partFeature[part] === feature) part++;
  }
  featureRingOffsets[featureCount] = polygonRingOffsets[partCount];
  return {
    featureCount,
    vertices,
    ringOffsets,
    polygonRingOffsets,
    partFeature,
    featureRingOffsets
  };
}

/** Triangulated polygon fill: three `lng, lat` vertices per triangle and the feature of each vertex. */
export type PolygonMesh = {
  /** Interleaved `lng, lat`, three vertices per triangle. */
  positions: Float32Array;
  /** Feature row of every vertex. */
  featureRows: Uint32Array;
  triangleCount: number;
  /** Outline segments `lng0, lat0, lng1, lat1`, one per ring edge. */
  outline: Float32Array;
};

/** Triangulates every polygon part (holes included) and collects the ring outline. */
export function buildPolygonMesh(layout: PolygonLayout): PolygonMesh {
  const {vertices, ringOffsets, polygonRingOffsets, partFeature} = layout;
  const positions: number[] = [];
  const featureRows: number[] = [];
  const outline: number[] = [];
  const partCount = polygonRingOffsets.length - 1;
  for (let part = 0; part < partCount; part++) {
    const feature = partFeature[part];
    const flat: number[] = [];
    const holes: number[] = [];
    for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
      let start = ringOffsets[ring];
      let end = ringOffsets[ring + 1];
      // Drop an explicit closing vertex.
      if (
        end - start > 1 &&
        vertices[start * 2] === vertices[(end - 1) * 2] &&
        vertices[start * 2 + 1] === vertices[(end - 1) * 2 + 1]
      ) {
        end--;
      }
      if (end - start < 3) continue;
      if (ring > polygonRingOffsets[part]) holes.push(flat.length / 2);
      for (let vertex = start; vertex < end; vertex++) {
        flat.push(vertices[vertex * 2], vertices[vertex * 2 + 1]);
        const next = vertex + 1 < end ? vertex + 1 : start;
        outline.push(
          vertices[vertex * 2],
          vertices[vertex * 2 + 1],
          vertices[next * 2],
          vertices[next * 2 + 1]
        );
      }
    }
    if (flat.length < 6) continue;
    const triangles = earcut(flat, holes.length ? holes : null, 2);
    for (const index of triangles) {
      positions.push(flat[index * 2], flat[index * 2 + 1]);
      featureRows.push(feature);
    }
  }
  return {
    positions: Float32Array.from(positions),
    featureRows: Uint32Array.from(featureRows),
    triangleCount: positions.length / 6,
    outline: Float32Array.from(outline)
  };
}

/** Bounding boxes and a point-in-polygon lookup for hover and click. */
export type FeatureLocator = {
  /** Feature row containing `[lng, lat]`, or -1. */
  locate: (lng: number, lat: number) => number;
  /** Bounding box centre of every feature (`lng, lat`). */
  centers: Float32Array;
};

/** Builds a {@link FeatureLocator} over a layout (bounding-box prefilter, even-odd ray test). */
export function createFeatureLocator(layout: PolygonLayout): FeatureLocator {
  const {featureCount, vertices, ringOffsets, featureRingOffsets} = layout;
  const boxes = new Float32Array(featureCount * 4);
  const centers = new Float32Array(featureCount * 2);
  for (let feature = 0; feature < featureCount; feature++) {
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    const first = ringOffsets[featureRingOffsets[feature]];
    const last = ringOffsets[featureRingOffsets[feature + 1]];
    for (let vertex = first; vertex < last; vertex++) {
      const x = vertices[vertex * 2];
      const y = vertices[vertex * 2 + 1];
      if (x < minimumX) minimumX = x;
      if (x > maximumX) maximumX = x;
      if (y < minimumY) minimumY = y;
      if (y > maximumY) maximumY = y;
    }
    boxes.set([minimumX, minimumY, maximumX, maximumY], feature * 4);
    centers.set([(minimumX + maximumX) / 2, (minimumY + maximumY) / 2], feature * 2);
  }
  const locate = (lng: number, lat: number): number => {
    for (let feature = 0; feature < featureCount; feature++) {
      if (
        lng < boxes[feature * 4] ||
        lng > boxes[feature * 4 + 2] ||
        lat < boxes[feature * 4 + 1] ||
        lat > boxes[feature * 4 + 3]
      ) {
        continue;
      }
      // Even-odd over all rings of the feature handles holes and multipart features.
      let inside = false;
      for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
        const start = ringOffsets[ring];
        const end = ringOffsets[ring + 1];
        for (let vertex = start, previous = end - 1; vertex < end; previous = vertex++) {
          const x0 = vertices[vertex * 2];
          const y0 = vertices[vertex * 2 + 1];
          const x1 = vertices[previous * 2];
          const y1 = vertices[previous * 2 + 1];
          if (y0 > lat !== y1 > lat && lng < ((x1 - x0) * (lat - y0)) / (y1 - y0) + x0) {
            inside = !inside;
          }
        }
      }
      if (inside) return feature;
    }
    return -1;
  };
  return {locate, centers};
}

/** Area-weighted-free planar centroids in local metric meters (mean of exterior vertices). */
export function getFeatureCentroidsMeters(
  layout: PolygonLayout,
  project: (lng: number, lat: number) => [number, number]
): Float32Array {
  const {featureCount, vertices, ringOffsets, featureRingOffsets} = layout;
  const centroids = new Float32Array(featureCount * 2);
  for (let feature = 0; feature < featureCount; feature++) {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    const first = ringOffsets[featureRingOffsets[feature]];
    const last = ringOffsets[featureRingOffsets[feature + 1]];
    for (let vertex = first; vertex < last; vertex++) {
      const [x, y] = project(vertices[vertex * 2], vertices[vertex * 2 + 1]);
      sumX += x;
      sumY += y;
      count++;
    }
    centroids[feature * 2] = sumX / Math.max(count, 1);
    centroids[feature * 2 + 1] = sumY / Math.max(count, 1);
  }
  return centroids;
}
