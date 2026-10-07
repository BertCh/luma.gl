// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Geometry preparation shared by the "joins" scenes: GeoArrow-style polygon and line sets in
 * planar meters built from the catalog datasets, plus the small CPU lookups (hit testing and
 * bounding boxes) the scenes use for tooltips and clicks. Nothing here touches the GPU.
 */

import type {LoadedDataset} from '../../data/catalog';

/** Polygon features in the layout `GPUPointInPolygonJoin` and the other joins read, in meters. */
export type PolygonSet = {
  featureCount: number;
  /** Flattened ring vertices `x, y` (rings are open: the closing vertex is dropped). */
  positions: Float32Array;
  /** `featureCount + 1` feature-to-polygon offsets. */
  featureOffsets: Uint32Array;
  /** Polygon-to-ring offsets. */
  polygonOffsets: Uint32Array;
  /** Ring-to-vertex offsets. */
  ringOffsets: Uint32Array;
  /** Ring outline segments `x0, y0, x1, y1`. */
  outline: Float32Array;
  /** Feature row of every outline segment. */
  outlineRows: Uint32Array;
  /** `minX, minY, maxX, maxY` per feature. */
  featureBounds: Float32Array;
  /** Area-weighted centroid `x, y` per feature. */
  centroids: Float32Array;
  /** `[minX, minY, maxX, maxY]` over all features. */
  bounds: [number, number, number, number];
};

/** Polylines in the layout `GPUSpatialJoinLines` reads, in meters, plus drawable segments. */
export type LineSet = {
  featureCount: number;
  /** Flattened vertices `x, y`. */
  positions: Float32Array;
  /** `featureCount + 1` feature-to-vertex offsets. */
  lineOffsets: Uint32Array;
  /** Consecutive-vertex segments `x0, y0, x1, y1`. */
  segments: Float32Array;
  /** Feature row of every segment. */
  segmentRows: Uint32Array;
  /** `[minX, minY, maxX, maxY]`. */
  bounds: [number, number, number, number];
};

/** Longitude/latitude rings of one polygon part (shell first, then holes). */
type LngLatPolygon = readonly (readonly (readonly [number, number])[])[];

/** Reads the polygons of every feature from a dataset: GeoJSON when present (full precision), else binary columns. */
function readLngLatFeatures(dataset: LoadedDataset): LngLatPolygon[][] {
  const features: LngLatPolygon[][] = [];
  const collection = dataset.geojson;
  if (collection) {
    for (const feature of collection.features) {
      const geometry = feature.geometry as {type: string; coordinates: unknown};
      features.push(
        geometry.type === 'Polygon'
          ? [geometry.coordinates as LngLatPolygon]
          : (geometry.coordinates as LngLatPolygon[])
      );
    }
    return features;
  }
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partFeature = dataset.column<Uint32Array>('partFeature');
  for (let feature = 0; feature < dataset.count; feature++) features.push([]);
  for (let part = 0; part < partFeature.length; part++) {
    const rings: [number, number][][] = [];
    for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
      const coordinates: [number, number][] = [];
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        coordinates.push([vertices[vertex * 2], vertices[vertex * 2 + 1]]);
      }
      rings.push(coordinates);
    }
    features[partFeature[part]].push(rings);
  }
  return features;
}

/**
 * Builds a {@link PolygonSet} from a polygon dataset, projected around `origin`. The GeoJSON
 * (6-decimal doubles) is preferred over the float32 binary columns: a float32 longitude is only
 * accurate to about 0.8 m, which flips points that sit on tract edges.
 */
export function createPolygonSet(
  dataset: LoadedDataset,
  origin: readonly [number, number]
): PolygonSet {
  const projection = dataset.getProjection(origin);
  const lngLatFeatures = readLngLatFeatures(dataset);
  const featureCount = lngLatFeatures.length;
  const positions: number[] = [];
  const ringOffsets: number[] = [0];
  const polygonOffsets: number[] = [0];
  const featureOffsets: number[] = [0];
  const outline: number[] = [];
  const outlineRows: number[] = [];
  const featureBounds = new Float32Array(featureCount * 4);
  const centroids = new Float32Array(featureCount * 2);
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];

  lngLatFeatures.forEach((polygons, feature) => {
    const featureBox = [Infinity, Infinity, -Infinity, -Infinity];
    let weightSum = 0;
    let centroidX = 0;
    let centroidY = 0;
    for (const rings of polygons) {
      rings.forEach((lngLatRing, ringIndex) => {
        let coordinates = lngLatRing;
        const first = coordinates[0];
        const last = coordinates[coordinates.length - 1];
        if (coordinates.length > 1 && first[0] === last[0] && first[1] === last[1]) {
          coordinates = coordinates.slice(0, -1);
        }
        const meters = coordinates.map(([lng, lat]) => projection.project(lng, lat));
        let area = 0;
        let ringX = 0;
        let ringY = 0;
        meters.forEach(([x, y], index) => {
          const [nextX, nextY] = meters[(index + 1) % meters.length];
          positions.push(x, y);
          outline.push(x, y, nextX, nextY);
          outlineRows.push(feature);
          const cross = x * nextY - nextX * y;
          area += cross;
          ringX += (x + nextX) * cross;
          ringY += (y + nextY) * cross;
          featureBox[0] = Math.min(featureBox[0], x);
          featureBox[1] = Math.min(featureBox[1], y);
          featureBox[2] = Math.max(featureBox[2], x);
          featureBox[3] = Math.max(featureBox[3], y);
        });
        ringOffsets.push(positions.length / 2);
        // Shells contribute to the centroid; holes are ignored.
        if (ringIndex === 0 && Math.abs(area) > 1e-9) {
          const weight = Math.abs(area) / 2;
          centroidX += (ringX / (3 * area)) * weight;
          centroidY += (ringY / (3 * area)) * weight;
          weightSum += weight;
        }
      });
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
    featureBounds.set(featureBox, feature * 4);
    centroids[feature * 2] = centroidX / (weightSum || 1);
    centroids[feature * 2 + 1] = centroidY / (weightSum || 1);
    bounds[0] = Math.min(bounds[0], featureBox[0]);
    bounds[1] = Math.min(bounds[1], featureBox[1]);
    bounds[2] = Math.max(bounds[2], featureBox[2]);
    bounds[3] = Math.max(bounds[3], featureBox[3]);
  });
  return {
    featureCount,
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    outline: Float32Array.from(outline),
    outlineRows: Uint32Array.from(outlineRows),
    featureBounds,
    centroids,
    bounds
  };
}

/** Builds a {@link LineSet} from explicit polylines (each an array of `x, y` pairs). */
export function createLineSet(polylines: readonly Float32Array[]): LineSet {
  const lineOffsets = new Uint32Array(polylines.length + 1);
  let vertexCount = 0;
  let segmentCount = 0;
  polylines.forEach((line, row) => {
    vertexCount += line.length / 2;
    segmentCount += Math.max(0, line.length / 2 - 1);
    lineOffsets[row + 1] = vertexCount;
  });
  const positions = new Float32Array(vertexCount * 2);
  const segments = new Float32Array(segmentCount * 4);
  const segmentRows = new Uint32Array(segmentCount);
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  let segment = 0;
  polylines.forEach((line, row) => {
    positions.set(line, lineOffsets[row] * 2);
    for (let index = 0; index < line.length; index += 2) {
      bounds[0] = Math.min(bounds[0], line[index]);
      bounds[1] = Math.min(bounds[1], line[index + 1]);
      bounds[2] = Math.max(bounds[2], line[index]);
      bounds[3] = Math.max(bounds[3], line[index + 1]);
      if (index + 3 < line.length) {
        segments.set([line[index], line[index + 1], line[index + 2], line[index + 3]], segment * 4);
        segmentRows[segment++] = row;
      }
    }
  });
  return {featureCount: polylines.length, positions, lineOffsets, segments, segmentRows, bounds};
}

/** Returns true when `(x, y)` is inside feature `feature` (even-odd over all rings). */
export function isPointInFeature(set: PolygonSet, feature: number, x: number, y: number): boolean {
  const b = set.featureBounds;
  if (
    x < b[feature * 4] ||
    y < b[feature * 4 + 1] ||
    x > b[feature * 4 + 2] ||
    y > b[feature * 4 + 3]
  ) {
    return false;
  }
  let inside = false;
  const firstRing = set.polygonOffsets[set.featureOffsets[feature]];
  const lastRing = set.polygonOffsets[set.featureOffsets[feature + 1]];
  for (let ring = firstRing; ring < lastRing; ring++) {
    const start = set.ringOffsets[ring];
    const end = set.ringOffsets[ring + 1];
    for (let index = start, previous = end - 1; index < end; previous = index++) {
      const xi = set.positions[index * 2];
      const yi = set.positions[index * 2 + 1];
      const xj = set.positions[previous * 2];
      const yj = set.positions[previous * 2 + 1];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** Returns the lowest feature row containing `(x, y)`, or -1. */
export function findPolygonAt(set: PolygonSet, x: number, y: number): number {
  for (let feature = 0; feature < set.featureCount; feature++) {
    if (isPointInFeature(set, feature, x, y)) return feature;
  }
  return -1;
}

/** Returns the line feature closest to `(x, y)` within `maximumDistance` meters, or -1. */
export function findLineNear(set: LineSet, x: number, y: number, maximumDistance: number): number {
  let best = -1;
  let bestDistance = maximumDistance;
  const {segments, segmentRows} = set;
  for (let segment = 0; segment < segmentRows.length; segment++) {
    const x0 = segments[segment * 4];
    const y0 = segments[segment * 4 + 1];
    const dx = segments[segment * 4 + 2] - x0;
    const dy = segments[segment * 4 + 3] - y0;
    const lengthSquared = dx * dx + dy * dy;
    const t =
      lengthSquared === 0
        ? 0
        : Math.min(1, Math.max(0, ((x - x0) * dx + (y - y0) * dy) / lengthSquared));
    const distance = Math.hypot(x - (x0 + t * dx), y - (y0 + t * dy));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = segmentRows[segment];
    }
  }
  return best;
}

/** Formats an integer with thousands separators. */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/** Formats a number compactly: integers plain, thousands with `k`, small values with decimals. */
export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  const magnitude = Math.abs(value);
  if (magnitude >= 10000) return `${(value / 1000).toFixed(magnitude >= 100000 ? 0 : 1)}k`;
  if (magnitude >= 100) return Math.round(value).toString();
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

/** Median of a numeric array (copies and sorts). */
export function getMedian(values: ArrayLike<number>): number {
  const sorted = Float64Array.from(values).sort();
  if (sorted.length === 0) return NaN;
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Uniform-grid index over segments `x0, y0, x1, y1` for short-range nearest-distance queries. */
export class SegmentGridIndex {
  private readonly cells = new Map<number, number[]>();
  private readonly segments: Float32Array;
  private readonly cellSize: number;

  /**
   * @param segments Segments `x0, y0, x1, y1` in meters.
   * @param cellSize Grid cell size in meters.
   * @param margin Queries are exact for distances up to `margin` meters.
   */
  constructor(segments: Float32Array, cellSize = 200, margin = 10) {
    this.segments = segments;
    this.cellSize = cellSize;
    for (let segment = 0; segment < segments.length / 4; segment++) {
      const minX = Math.min(segments[segment * 4], segments[segment * 4 + 2]) - margin;
      const maxX = Math.max(segments[segment * 4], segments[segment * 4 + 2]) + margin;
      const minY = Math.min(segments[segment * 4 + 1], segments[segment * 4 + 3]) - margin;
      const maxY = Math.max(segments[segment * 4 + 1], segments[segment * 4 + 3]) + margin;
      for (let cx = Math.floor(minX / cellSize); cx <= Math.floor(maxX / cellSize); cx++) {
        for (let cy = Math.floor(minY / cellSize); cy <= Math.floor(maxY / cellSize); cy++) {
          const key = cx * 100003 + cy;
          const cell = this.cells.get(key);
          if (cell) cell.push(segment);
          else this.cells.set(key, [segment]);
        }
      }
    }
  }

  /** Distance from `(x, y)` to the nearest indexed segment, exact up to the margin, else Infinity. */
  getDistance(x: number, y: number): number {
    const cell = this.cells.get(
      Math.floor(x / this.cellSize) * 100003 + Math.floor(y / this.cellSize)
    );
    if (!cell) return Infinity;
    let best = Infinity;
    const {segments} = this;
    for (const segment of cell) {
      const x0 = segments[segment * 4];
      const y0 = segments[segment * 4 + 1];
      const dx = segments[segment * 4 + 2] - x0;
      const dy = segments[segment * 4 + 3] - y0;
      const lengthSquared = dx * dx + dy * dy;
      const t =
        lengthSquared === 0
          ? 0
          : Math.min(1, Math.max(0, ((x - x0) * dx + (y - y0) * dy) / lengthSquared));
      best = Math.min(best, Math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)));
    }
    return best;
  }
}

/** Distance from `(x, y)` to feature `feature` of a polygon set: 0 when inside, else to the nearest ring edge. */
export function getDistanceToFeature(
  set: PolygonSet,
  feature: number,
  x: number,
  y: number
): number {
  if (isPointInFeature(set, feature, x, y)) return 0;
  let best = Infinity;
  const firstRing = set.polygonOffsets[set.featureOffsets[feature]];
  const lastRing = set.polygonOffsets[set.featureOffsets[feature + 1]];
  for (let ring = firstRing; ring < lastRing; ring++) {
    const start = set.ringOffsets[ring];
    const end = set.ringOffsets[ring + 1];
    for (let index = start; index < end; index++) {
      const next = index + 1 < end ? index + 1 : start;
      const x0 = set.positions[index * 2];
      const y0 = set.positions[index * 2 + 1];
      const dx = set.positions[next * 2] - x0;
      const dy = set.positions[next * 2 + 1] - y0;
      const lengthSquared = dx * dx + dy * dy;
      const t =
        lengthSquared === 0
          ? 0
          : Math.min(1, Math.max(0, ((x - x0) * dx + (y - y0) * dy) / lengthSquared));
      best = Math.min(best, Math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)));
    }
  }
  return best;
}
