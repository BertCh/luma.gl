// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Polygon preparation shared by the regression chapter (builder B6): reads the GeoArrow-style
 * polygon columns of a catalog dataset, keeps a chosen subset of features, triangulates them once
 * for the GPU fill layer, builds outline segments, centroids, a hit-test grid and a queen
 * adjacency used only to prune islands and to draw links. All of this is one-time CPU work on
 * static input; every analysis result is computed by the GPU contributors.
 */

import earcut from 'earcut';
import type {LoadedDataset} from '../../data/catalog';
import {LocalMetricProjection} from '../../engine/projection';

/** Row marker of a feature that has no value row (drawn by no colored fill). */
export const NO_ROW = 0xffffffff;

/** Polygon features as flat arrays. Coordinates are longitude and latitude in degrees. */
export type PolygonGeometry = {
  featureCount: number;
  /** Interleaved `lng, lat` of every ring vertex (rings are closed: first vertex repeated). */
  vertices: Float32Array;
  /** Ring to vertex offsets, `ringCount + 1` entries. */
  ringOffsets: Uint32Array;
  /** Part (polygon) to ring offsets, `partCount + 1` entries. */
  partRingOffsets: Uint32Array;
  /** Feature of every part. Parts of a feature are contiguous. */
  partFeature: Uint32Array;
  /** Feature to ring offsets, `featureCount + 1` entries. */
  featureRingOffsets: Uint32Array;
};

/** Reads the polygon columns of a dataset (tracts, community areas, counties). */
export function readPolygonGeometry(dataset: LoadedDataset): PolygonGeometry {
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const partRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partCount = partRingOffsets.length - 1;
  let partFeature: Uint32Array;
  let featureCount: number;
  if (dataset.hasColumn('partFeature')) {
    partFeature = dataset.column<Uint32Array>('partFeature');
    featureCount = dataset.count;
  } else {
    // Counties: `countyPolygonOffsets` maps a feature to its polygon parts.
    const featureParts = dataset.column<Uint32Array>('countyPolygonOffsets');
    featureCount = featureParts.length - 1;
    partFeature = new Uint32Array(partCount);
    for (let feature = 0; feature < featureCount; feature++) {
      for (let part = featureParts[feature]; part < featureParts[feature + 1]; part++) {
        partFeature[part] = feature;
      }
    }
  }
  const featureRingOffsets = new Uint32Array(featureCount + 1);
  let part = 0;
  for (let feature = 0; feature < featureCount; feature++) {
    featureRingOffsets[feature] = partRingOffsets[part];
    while (part < partCount && partFeature[part] === feature) part++;
  }
  featureRingOffsets[featureCount] = partRingOffsets[partCount];
  return {featureCount, vertices, ringOffsets, partRingOffsets, partFeature, featureRingOffsets};
}

/** Copies a subset of features, in the given order, into a compact geometry. */
export function subsetPolygonGeometry(
  source: PolygonGeometry,
  features: ArrayLike<number>
): PolygonGeometry {
  const vertices: number[] = [];
  const ringOffsets: number[] = [0];
  const partRingOffsets: number[] = [0];
  const partFeature: number[] = [];
  const featureRingOffsets: number[] = [0];
  const partsOf = getFeatureParts(source);
  for (let row = 0; row < features.length; row++) {
    for (const part of partsOf[features[row]]) {
      for (
        let ring = source.partRingOffsets[part];
        ring < source.partRingOffsets[part + 1];
        ring++
      ) {
        for (
          let vertex = source.ringOffsets[ring];
          vertex < source.ringOffsets[ring + 1];
          vertex++
        ) {
          vertices.push(source.vertices[vertex * 2], source.vertices[vertex * 2 + 1]);
        }
        ringOffsets.push(vertices.length / 2);
      }
      partRingOffsets.push(ringOffsets.length - 1);
      partFeature.push(row);
    }
    featureRingOffsets.push(ringOffsets.length - 1);
  }
  return {
    featureCount: features.length,
    vertices: Float32Array.from(vertices),
    ringOffsets: Uint32Array.from(ringOffsets),
    partRingOffsets: Uint32Array.from(partRingOffsets),
    partFeature: Uint32Array.from(partFeature),
    featureRingOffsets: Uint32Array.from(featureRingOffsets)
  };
}

function getFeatureParts(geometry: PolygonGeometry): number[][] {
  const parts: number[][] = Array.from({length: geometry.featureCount}, () => []);
  for (let part = 0; part < geometry.partFeature.length; part++) {
    parts[geometry.partFeature[part]].push(part);
  }
  return parts;
}

/** Triangles of a polygon set; `owners[t]` is the value row of triangle `t`. */
export type PolygonTriangles = {
  /** `lng, lat` per corner, three corners per triangle. */
  corners: Float32Array;
  owners: Uint32Array;
  triangleCount: number;
};

/**
 * Triangulates every part with earcut (holes supported). `rowOfFeature[f]` is the value row drawn
 * for feature `f`, or `NO_ROW` to skip it. Default: the feature index.
 */
export function triangulatePolygonGeometry(
  geometry: PolygonGeometry,
  rowOfFeature?: ArrayLike<number>
): PolygonTriangles {
  const corners: number[] = [];
  const owners: number[] = [];
  for (let part = 0; part < geometry.partFeature.length; part++) {
    const feature = geometry.partFeature[part];
    const row = rowOfFeature ? rowOfFeature[feature] : feature;
    if (row === NO_ROW) continue;
    const coordinates: number[] = [];
    const holes: number[] = [];
    for (
      let ring = geometry.partRingOffsets[part];
      ring < geometry.partRingOffsets[part + 1];
      ring++
    ) {
      if (ring > geometry.partRingOffsets[part]) holes.push(coordinates.length / 2);
      const start = geometry.ringOffsets[ring];
      let end = geometry.ringOffsets[ring + 1];
      // earcut wants an open ring.
      if (
        end - start > 1 &&
        geometry.vertices[start * 2] === geometry.vertices[(end - 1) * 2] &&
        geometry.vertices[start * 2 + 1] === geometry.vertices[(end - 1) * 2 + 1]
      ) {
        end--;
      }
      for (let vertex = start; vertex < end; vertex++) {
        coordinates.push(geometry.vertices[vertex * 2], geometry.vertices[vertex * 2 + 1]);
      }
    }
    const indices = earcut(coordinates, holes, 2);
    for (const index of indices) corners.push(coordinates[index * 2], coordinates[index * 2 + 1]);
    for (let triangle = 0; triangle < indices.length / 3; triangle++) owners.push(row);
  }
  return {
    corners: Float32Array.from(corners),
    owners: Uint32Array.from(owners),
    triangleCount: owners.length
  };
}

/** Ring edges as `lng0, lat0, lng1, lat1` rows, for the segment layer. */
export function buildOutlineSegments(geometry: PolygonGeometry): Float32Array {
  const segments: number[] = [];
  for (let ring = 0; ring < geometry.ringOffsets.length - 1; ring++) {
    const start = geometry.ringOffsets[ring];
    const end = geometry.ringOffsets[ring + 1];
    for (let vertex = start; vertex + 1 < end; vertex++) {
      segments.push(
        geometry.vertices[vertex * 2],
        geometry.vertices[vertex * 2 + 1],
        geometry.vertices[vertex * 2 + 2],
        geometry.vertices[vertex * 2 + 3]
      );
    }
  }
  return Float32Array.from(segments);
}

/** Outline segments of the listed features only. */
export function buildFeatureOutline(geometry: PolygonGeometry, feature: number): Float32Array {
  const segments: number[] = [];
  for (
    let ring = geometry.featureRingOffsets[feature];
    ring < geometry.featureRingOffsets[feature + 1];
    ring++
  ) {
    for (
      let vertex = geometry.ringOffsets[ring];
      vertex + 1 < geometry.ringOffsets[ring + 1];
      vertex++
    ) {
      segments.push(
        geometry.vertices[vertex * 2],
        geometry.vertices[vertex * 2 + 1],
        geometry.vertices[vertex * 2 + 2],
        geometry.vertices[vertex * 2 + 3]
      );
    }
  }
  return Float32Array.from(segments);
}

/** Area-weighted centroid (longitude, latitude) of the largest ring of every feature. */
export function computeCentroids(geometry: PolygonGeometry): Float32Array {
  const centroids = new Float32Array(geometry.featureCount * 2);
  for (let feature = 0; feature < geometry.featureCount; feature++) {
    let bestArea = -1;
    let bestX = 0;
    let bestY = 0;
    for (
      let ring = geometry.featureRingOffsets[feature];
      ring < geometry.featureRingOffsets[feature + 1];
      ring++
    ) {
      const start = geometry.ringOffsets[ring];
      const end = geometry.ringOffsets[ring + 1];
      let area = 0;
      let cx = 0;
      let cy = 0;
      let meanX = 0;
      let meanY = 0;
      const originX = geometry.vertices[start * 2];
      const originY = geometry.vertices[start * 2 + 1];
      for (let vertex = start; vertex + 1 < end; vertex++) {
        const x0 = geometry.vertices[vertex * 2] - originX;
        const y0 = geometry.vertices[vertex * 2 + 1] - originY;
        const x1 = geometry.vertices[vertex * 2 + 2] - originX;
        const y1 = geometry.vertices[vertex * 2 + 3] - originY;
        const cross = x0 * y1 - x1 * y0;
        area += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
        meanX += x0;
        meanY += y0;
      }
      area *= 0.5;
      const absoluteArea = Math.abs(area);
      if (absoluteArea > bestArea) {
        bestArea = absoluteArea;
        if (absoluteArea > 1e-14) {
          bestX = originX + cx / (6 * area);
          bestY = originY + cy / (6 * area);
        } else {
          const count = Math.max(1, end - start - 1);
          bestX = originX + meanX / count;
          bestY = originY + meanY / count;
        }
      }
    }
    centroids[feature * 2] = bestX;
    centroids[feature * 2 + 1] = bestY;
  }
  return centroids;
}

/** Queen adjacency by exactly shared vertices (the definition `GPUContiguityWeights` implements). */
export function computeQueenAdjacency(geometry: PolygonGeometry): number[][] {
  const owners = new Map<string, number[]>();
  const owningFeature = new Uint32Array(geometry.ringOffsets.length - 1);
  for (let feature = 0; feature < geometry.featureCount; feature++) {
    for (
      let ring = geometry.featureRingOffsets[feature];
      ring < geometry.featureRingOffsets[feature + 1];
      ring++
    ) {
      owningFeature[ring] = feature;
    }
  }
  for (let ring = 0; ring < owningFeature.length; ring++) {
    const feature = owningFeature[ring];
    for (
      let vertex = geometry.ringOffsets[ring];
      vertex < geometry.ringOffsets[ring + 1];
      vertex++
    ) {
      const key = `${geometry.vertices[vertex * 2]},${geometry.vertices[vertex * 2 + 1]}`;
      const list = owners.get(key);
      if (!list) owners.set(key, [feature]);
      else if (list[list.length - 1] !== feature) list.push(feature);
    }
  }
  const sets = Array.from({length: geometry.featureCount}, () => new Set<number>());
  for (const list of owners.values()) {
    if (list.length < 2) continue;
    for (const a of list) for (const b of list) if (a !== b) sets[a].add(b);
  }
  return sets.map(set => [...set].sort((a, b) => a - b));
}

/** Returns the features of the largest connected component of `adjacency` among `keep`. */
export function getLargestComponent(adjacency: number[][], keep: ArrayLike<number>): number[] {
  const allowed = new Set<number>();
  for (let index = 0; index < keep.length; index++) allowed.add(keep[index]);
  const seen = new Set<number>();
  let best: number[] = [];
  for (const start of allowed) {
    if (seen.has(start)) continue;
    const component: number[] = [];
    const stack = [start];
    seen.add(start);
    while (stack.length) {
      const feature = stack.pop()!;
      component.push(feature);
      for (const neighbor of adjacency[feature]) {
        if (allowed.has(neighbor) && !seen.has(neighbor)) {
          seen.add(neighbor);
          stack.push(neighbor);
        }
      }
    }
    if (component.length > best.length) best = component;
  }
  return best.sort((a, b) => a - b);
}

/** Point-in-feature lookup over a coarse grid. */
export type FeatureLocator = (longitude: number, latitude: number) => number;

/** Builds a locator that returns the feature containing a point, or -1. */
export function createFeatureLocator(geometry: PolygonGeometry): FeatureLocator {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (let vertex = 0; vertex < geometry.vertices.length / 2; vertex++) {
    west = Math.min(west, geometry.vertices[vertex * 2]);
    east = Math.max(east, geometry.vertices[vertex * 2]);
    south = Math.min(south, geometry.vertices[vertex * 2 + 1]);
    north = Math.max(north, geometry.vertices[vertex * 2 + 1]);
  }
  const resolution = Math.max(
    16,
    Math.min(128, Math.round(Math.sqrt(geometry.featureCount) * 1.5))
  );
  const cellWidth = (east - west) / resolution || 1;
  const cellHeight = (north - south) / resolution || 1;
  const cells: number[][] = Array.from({length: resolution * resolution}, () => []);
  const featureBounds = new Float32Array(geometry.featureCount * 4);
  for (let feature = 0; feature < geometry.featureCount; feature++) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (
      let ring = geometry.featureRingOffsets[feature];
      ring < geometry.featureRingOffsets[feature + 1];
      ring++
    ) {
      for (
        let vertex = geometry.ringOffsets[ring];
        vertex < geometry.ringOffsets[ring + 1];
        vertex++
      ) {
        minX = Math.min(minX, geometry.vertices[vertex * 2]);
        maxX = Math.max(maxX, geometry.vertices[vertex * 2]);
        minY = Math.min(minY, geometry.vertices[vertex * 2 + 1]);
        maxY = Math.max(maxY, geometry.vertices[vertex * 2 + 1]);
      }
    }
    featureBounds.set([minX, minY, maxX, maxY], feature * 4);
    const column0 = Math.max(0, Math.floor((minX - west) / cellWidth));
    const column1 = Math.min(resolution - 1, Math.floor((maxX - west) / cellWidth));
    const row0 = Math.max(0, Math.floor((minY - south) / cellHeight));
    const row1 = Math.min(resolution - 1, Math.floor((maxY - south) / cellHeight));
    for (let row = row0; row <= row1; row++) {
      for (let column = column0; column <= column1; column++) {
        cells[row * resolution + column].push(feature);
      }
    }
  }
  const contains = (feature: number, x: number, y: number) => {
    let inside = false;
    for (
      let ring = geometry.featureRingOffsets[feature];
      ring < geometry.featureRingOffsets[feature + 1];
      ring++
    ) {
      const start = geometry.ringOffsets[ring];
      const end = geometry.ringOffsets[ring + 1];
      for (let vertex = start, previous = end - 1; vertex < end; previous = vertex++) {
        const xi = geometry.vertices[vertex * 2];
        const yi = geometry.vertices[vertex * 2 + 1];
        const xj = geometry.vertices[previous * 2];
        const yj = geometry.vertices[previous * 2 + 1];
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
    }
    return inside;
  };
  return (longitude, latitude) => {
    const column = Math.floor((longitude - west) / cellWidth);
    const row = Math.floor((latitude - south) / cellHeight);
    if (column < 0 || row < 0 || column >= resolution || row >= resolution) return -1;
    for (const feature of cells[row * resolution + column]) {
      const bounds = featureBounds.subarray(feature * 4, feature * 4 + 4);
      if (
        longitude < bounds[0] ||
        longitude > bounds[2] ||
        latitude < bounds[1] ||
        latitude > bounds[3]
      )
        continue;
      if (contains(feature, longitude, latitude)) return feature;
    }
    return -1;
  };
}

/** Albers equal-area conic (USGS contiguous-US parameters) on a sphere, in meters. */
export function projectAlbers(longitude: number, latitude: number): [number, number] {
  const radians = Math.PI / 180;
  const n = (Math.sin(29.5 * radians) + Math.sin(45.5 * radians)) / 2;
  const c = Math.cos(29.5 * radians) ** 2 + 2 * n * Math.sin(29.5 * radians);
  const rho0 = Math.sqrt(c - 2 * n * Math.sin(37.5 * radians)) / n;
  const rho = Math.sqrt(c - 2 * n * Math.sin(latitude * radians)) / n;
  const theta = n * (longitude + 96) * radians;
  const radius = 6371008.8;
  return [radius * rho * Math.sin(theta), radius * (rho0 - rho * Math.cos(theta))];
}

/** Projects `lng, lat` rows with a function. */
export function projectRows(
  lngLat: Float32Array,
  project: (longitude: number, latitude: number) => [number, number]
): Float32Array {
  const result = new Float32Array(lngLat.length);
  for (let row = 0; row < lngLat.length / 2; row++) {
    const [x, y] = project(lngLat[row * 2], lngLat[row * 2 + 1]);
    result[row * 2] = x;
    result[row * 2 + 1] = y;
  }
  return result;
}

/** Local meters around an origin, matching the rest of the showcase. */
export function createLocalProjector(
  origin: readonly [number, number]
): (longitude: number, latitude: number) => [number, number] {
  const projection = new LocalMetricProjection(origin);
  return (longitude, latitude) => projection.project(longitude, latitude);
}

// ---------------------------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------------------------

/** Value at a fraction of the sorted finite values. */
export function getQuantile(values: ArrayLike<number>, fraction: number): number {
  const finite: number[] = [];
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) finite.push(values[index]);
  }
  if (finite.length === 0) return 0;
  finite.sort((a, b) => a - b);
  return finite[
    Math.min(finite.length - 1, Math.max(0, Math.floor(fraction * (finite.length - 1))))
  ];
}

/** Symmetric `[-b, b]` range covering the 5th to 95th percentile magnitude. */
export function getSymmetricRange(
  values: ArrayLike<number>,
  lower = 0.05,
  upper = 0.95
): [number, number] {
  const bound = Math.max(
    1e-6,
    Math.abs(getQuantile(values, lower)),
    Math.abs(getQuantile(values, upper))
  );
  return [-bound, bound];
}

/** Mean and standard deviation (population) of finite values. */
export function getMoments(values: ArrayLike<number>): {mean: number; deviation: number} {
  let count = 0;
  let sum = 0;
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) {
      sum += values[index];
      count++;
    }
  }
  const mean = count ? sum / count : 0;
  let squares = 0;
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) squares += (values[index] - mean) ** 2;
  }
  return {mean, deviation: Math.sqrt(count ? squares / count : 0) || 1};
}

/** Formats a number with fixed digits, switching to exponent for tiny values. */
export function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return 'n/a';
  return Math.abs(value) < 0.01 && value !== 0 ? value.toExponential(2) : value.toFixed(digits);
}

/** Formats a p-value. */
export function formatP(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value < 0.001 ? 'p < 0.001' : `p = ${value.toFixed(3)}`;
}

/** Formats a count with thousands separators. */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/** Ring edges of GeoJSON Polygon and MultiPolygon features as `lng0, lat0, lng1, lat1` rows. */
export function buildGeoJsonOutline(collection: {
  features: {geometry: {type: string; coordinates: unknown} | null}[];
}): Float32Array {
  const segments: number[] = [];
  const addRing = (ring: number[][]) => {
    for (let index = 0; index + 1 < ring.length; index++) {
      segments.push(ring[index][0], ring[index][1], ring[index + 1][0], ring[index + 1][1]);
    }
  };
  for (const feature of collection.features) {
    if (!feature.geometry) continue;
    const {type, coordinates} = feature.geometry;
    if (type === 'Polygon') {
      for (const ring of coordinates as number[][][]) addRing(ring);
    } else if (type === 'MultiPolygon') {
      for (const polygon of coordinates as number[][][][])
        for (const ring of polygon) addRing(ring);
    }
  }
  return Float32Array.from(segments);
}
