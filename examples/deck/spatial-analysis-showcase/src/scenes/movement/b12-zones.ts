// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GeoJsonCollection} from '../../data/loaders';

/** Zone kinds of the `ais-zones` dataset in the order of its manifest. */
export const ZONE_KINDS = ['anchorage', 'channel', 'terminal', 'gate', 'tourist', 'ferry'] as const;

export type ZoneKind = (typeof ZONE_KINDS)[number];

export const ZONE_KIND_LABELS: Record<ZoneKind, string> = {
  anchorage: 'Anchorage (official)',
  channel: 'Maintained channel (official)',
  terminal: 'Terminal (approximate)',
  gate: 'Bridge span gate (approximate)',
  tourist: 'Tour area (approximate)',
  ferry: 'Ferry lane (approximate)'
};

/** Outline colors by zone kind, readable on light and dark basemaps. */
export const ZONE_KIND_COLORS: readonly (readonly [number, number, number, number])[] = [
  [100, 150, 255, 255],
  [50, 200, 210, 255],
  [235, 105, 205, 255],
  [255, 85, 85, 255],
  [200, 225, 70, 255],
  [235, 235, 245, 255]
];

/** Polygon zones in planar meters, in the layouts the contributors and layers need. */
export type ZoneSet = {
  zoneCount: number;
  names: string[];
  /** Index into {@link ZONE_KINDS} per zone. */
  kinds: Uint32Array;
  /** GeoArrow-style polygon topology (feature, polygon, ring, vertex offsets). */
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Boundary edges (rings implicitly closed), with the zone of each edge. */
  edgeStarts: Float32Array;
  edgeEnds: Float32Array;
  edgeZones: Uint32Array;
  /** The same edges as `x0, y0, x1, y1` rows for drawing. */
  outlineSegments: Float32Array;
  /** Every ring of every zone as flat `x, y` arrays, for picking. */
  zoneRings: Float32Array[][];
  /** Absolute area of each zone in square meters (shell minus holes). */
  areas: Float64Array;
  /** `[minX, minY, maxX, maxY]`. */
  bounds: [number, number, number, number];
};

type Coordinate = [number, number];

/** Builds a {@link ZoneSet} from GeoJSON polygons and multipolygons, projecting with `project`. */
export function buildZoneSet(
  collection: GeoJsonCollection,
  project: (longitude: number, latitude: number) => [number, number]
): ZoneSet {
  const names: string[] = [];
  const kinds: number[] = [];
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  const edgeStarts: number[] = [];
  const edgeEnds: number[] = [];
  const edgeZones: number[] = [];
  const outline: number[] = [];
  const zoneRings: Float32Array[][] = [];
  const areas: number[] = [];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  collection.features.forEach((feature, zone) => {
    const properties = feature.properties as {name?: string; kind?: string} | null;
    names.push(properties?.name ?? `Zone ${zone + 1}`);
    kinds.push(Math.max(0, ZONE_KINDS.indexOf((properties?.kind ?? 'anchorage') as ZoneKind)));
    const geometry = feature.geometry;
    if (!geometry) throw new Error(`Zone ${zone} has no geometry`);
    const polygons: Coordinate[][][] =
      geometry.type === 'Polygon'
        ? [geometry.coordinates as Coordinate[][]]
        : (geometry.coordinates as Coordinate[][][]);
    const rings: Float32Array[] = [];
    let area = 0;
    for (const polygon of polygons) {
      polygon.forEach((ring, ringIndex) => {
        const projected: Coordinate[] = ring.map(([longitude, latitude]) =>
          project(longitude, latitude)
        );
        const first = projected[0];
        const last = projected[projected.length - 1];
        if (projected.length > 1 && first[0] === last[0] && first[1] === last[1]) projected.pop();
        const flat = new Float32Array(projected.length * 2);
        let ringArea = 0;
        projected.forEach(([x, y], vertex) => {
          positions.push(x, y);
          flat[vertex * 2] = x;
          flat[vertex * 2 + 1] = y;
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
          const [nextX, nextY] = projected[(vertex + 1) % projected.length];
          edgeStarts.push(x, y);
          edgeEnds.push(nextX, nextY);
          edgeZones.push(zone);
          outline.push(x, y, nextX, nextY);
          ringArea += x * nextY - nextX * y;
        });
        ringOffsets.push(positions.length / 2);
        rings.push(flat);
        area += (ringIndex === 0 ? 1 : -1) * Math.abs(ringArea / 2);
      });
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
    zoneRings.push(rings);
    areas.push(area);
  });

  return {
    zoneCount: collection.features.length,
    names,
    kinds: Uint32Array.from(kinds),
    polygonPositions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    edgeStarts: Float32Array.from(edgeStarts),
    edgeEnds: Float32Array.from(edgeEnds),
    edgeZones: Uint32Array.from(edgeZones),
    outlineSegments: Float32Array.from(outline),
    zoneRings,
    areas: Float64Array.from(areas),
    bounds: [minX, minY, maxX, maxY]
  };
}

/** Even-odd point-in-zone test over all rings of one zone. */
function isInsideZone(rings: readonly Float32Array[], x: number, y: number): boolean {
  let inside = false;
  for (const ring of rings) {
    const count = ring.length / 2;
    for (let index = 0, previous = count - 1; index < count; previous = index++) {
      const x0 = ring[index * 2];
      const y0 = ring[index * 2 + 1];
      const x1 = ring[previous * 2];
      const y1 = ring[previous * 2 + 1];
      if (y0 > y !== y1 > y && x < ((x1 - x0) * (y - y0)) / (y1 - y0) + x0) inside = !inside;
    }
  }
  return inside;
}

/**
 * Returns the zone containing a point, or -1. Where zones overlap the smallest one wins, matching
 * {@link rasterizeZones}.
 */
export function findZone(zones: ZoneSet, x: number, y: number): number {
  let best = -1;
  for (let zone = 0; zone < zones.zoneCount; zone++) {
    if (best >= 0 && zones.areas[zone] >= zones.areas[best]) continue;
    if (isInsideZone(zones.zoneRings[zone], x, y)) best = zone;
  }
  return best;
}

/**
 * Rasterizes the zones into a `width x height` grid of zone rows (`zoneCount` where no zone
 * covers the cell). Rows grow with y. Larger zones are painted first so smaller zones stay visible
 * where zones overlap.
 */
export function rasterizeZones(
  zones: ZoneSet,
  bounds: readonly [number, number, number, number],
  width: number,
  height: number
): Uint32Array {
  const cells = new Uint32Array(width * height).fill(zones.zoneCount);
  const cellWidth = (bounds[2] - bounds[0]) / width;
  const cellHeight = (bounds[3] - bounds[1]) / height;
  const order = Array.from({length: zones.zoneCount}, (_, zone) => zone).sort(
    (a, b) => zones.areas[b] - zones.areas[a]
  );
  for (const zone of order) {
    const rings = zones.zoneRings[zone];
    let zoneMinY = Infinity;
    let zoneMaxY = -Infinity;
    for (const ring of rings) {
      for (let index = 1; index < ring.length; index += 2) {
        zoneMinY = Math.min(zoneMinY, ring[index]);
        zoneMaxY = Math.max(zoneMaxY, ring[index]);
      }
    }
    const firstRow = Math.max(0, Math.floor((zoneMinY - bounds[1]) / cellHeight));
    const lastRow = Math.min(height - 1, Math.ceil((zoneMaxY - bounds[1]) / cellHeight));
    for (let row = firstRow; row <= lastRow; row++) {
      const y = bounds[1] + (row + 0.5) * cellHeight;
      const crossings: number[] = [];
      for (const ring of rings) {
        const count = ring.length / 2;
        for (let index = 0, previous = count - 1; index < count; previous = index++) {
          const y0 = ring[index * 2 + 1];
          const y1 = ring[previous * 2 + 1];
          if (y0 > y !== y1 > y) {
            const x0 = ring[index * 2];
            const x1 = ring[previous * 2];
            crossings.push(x0 + ((y - y0) / (y1 - y0)) * (x1 - x0));
          }
        }
      }
      crossings.sort((a, b) => a - b);
      for (let pair = 0; pair + 1 < crossings.length; pair += 2) {
        const startColumn = Math.max(0, Math.ceil((crossings[pair] - bounds[0]) / cellWidth - 0.5));
        const endColumn = Math.min(
          width - 1,
          Math.floor((crossings[pair + 1] - bounds[0]) / cellWidth - 0.5)
        );
        for (let column = startColumn; column <= endColumn; column++) {
          cells[row * width + column] = zone;
        }
      }
    }
  }
  return cells;
}
