// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import type {LngLat} from '../../cartography/types';
import type {GeoJsonCollection, GeoJsonGeometry} from '../../data/loaders';
import {buildZoneSet, ZONE_KINDS, type ZoneKind, type ZoneSet} from './b12-zones';
import {groupZones, type ZoneGroups} from './zone-dwell-names';

/** Line styles of the zone outlines: kind is told by line style, never by hue. */
export const OUTLINE_STYLE_NAMES = ['official', 'channel', 'approximate'] as const;

/** One of {@link OUTLINE_STYLE_NAMES}. */
export type OutlineStyleName = (typeof OUTLINE_STYLE_NAMES)[number];

/** The outline style of a zone kind: official anchorage, maintained channel, hand-drawn. */
export function getOutlineStyleName(kind: ZoneKind): OutlineStyleName {
  if (kind === 'anchorage') return 'official';
  if (kind === 'channel') return 'channel';
  return 'approximate';
}

/** Everything the zone-dwell scene draws and names, built once from the zone polygons. */
export type ZoneDwellGeometry = {
  /** Zone polygons in the layouts the contributors need (edges, rings, areas). */
  zones: ZoneSet;
  /** Pieces of one named zone share a group. */
  groups: ZoneGroups;
  /** Fill triangles, `float32x2`, largest zone first so small zones sit on top. */
  triangles: Float32Array;
  /** Zone row of every triangle vertex. */
  triangleFeatures: Uint32Array;
  /** Outline segments `x0, y0, x1, y1` of every zone by line style. */
  outlines: Record<OutlineStyleName, Float32Array>;
  /** Area of every zone in square kilometres. */
  areasKm2: Float64Array;
  /** Area of every group in square kilometres (the pieces added up). */
  groupAreasKm2: Float64Array;
  /** Label point `[longitude, latitude]` of every group (inside its largest piece). */
  groupLabelPoints: LngLat[];
  /** Every ring of every piece of every group, in longitude and latitude. */
  groupRings: LngLat[][][];
  /** Outer rings of every group, in longitude and latitude. */
  groupOuterRings: LngLat[][][];
};

type GeoPolygon = number[][][];

function getPolygons(geometry: GeoJsonGeometry | null): GeoPolygon[] {
  if (!geometry) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates as GeoPolygon];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates as GeoPolygon[];
  return [];
}

/**
 * Builds the {@link ZoneDwellGeometry} of the `ais-zones` collection: the zone set for the
 * contributors, the fill mesh drawn largest zone first, outlines split by line style, and the
 * label points and rings in degrees for annotations and highlights.
 */
export function buildZoneDwellGeometry(
  collection: GeoJsonCollection,
  project: (longitude: number, latitude: number) => [number, number]
): ZoneDwellGeometry {
  const zones = buildZoneSet(collection, project);
  const mesh = buildPolygonMesh(collection, project);
  const kinds = Array.from(zones.kinds, kind => ZONE_KINDS[kind]);
  const groups = groupZones(zones.names, kinds);
  const zoneCount = zones.zoneCount;

  // Triangles are emitted feature by feature; reorder those runs by area, largest first.
  const firstVertex = new Int32Array(zoneCount).fill(-1);
  const lastVertex = new Int32Array(zoneCount).fill(-1);
  for (let vertex = 0; vertex < mesh.triangleFeatures.length; vertex++) {
    const zone = mesh.triangleFeatures[vertex];
    if (firstVertex[zone] < 0) firstVertex[zone] = vertex;
    lastVertex[zone] = vertex + 1;
  }
  const drawOrder = Array.from({length: zoneCount}, (_, zone) => zone).sort(
    (a, b) => mesh.areas[b] - mesh.areas[a] || a - b
  );
  const triangles = new Float32Array(mesh.triangles.length);
  const triangleFeatures = new Uint32Array(mesh.triangleFeatures.length);
  let cursor = 0;
  for (const zone of drawOrder) {
    if (firstVertex[zone] < 0) continue;
    const length = lastVertex[zone] - firstVertex[zone];
    triangles.set(mesh.triangles.subarray(firstVertex[zone] * 2, lastVertex[zone] * 2), cursor * 2);
    triangleFeatures.set(
      mesh.triangleFeatures.subarray(firstVertex[zone], lastVertex[zone]),
      cursor
    );
    cursor += length;
  }

  const outlineRows: Record<OutlineStyleName, number[]> = {
    official: [],
    channel: [],
    approximate: []
  };
  for (let edge = 0; edge < zones.edgeZones.length; edge++) {
    const style = getOutlineStyleName(kinds[zones.edgeZones[edge]]);
    const row = outlineRows[style];
    for (let component = 0; component < 4; component++) {
      row.push(zones.outlineSegments[edge * 4 + component]);
    }
  }

  const areasKm2 = Float64Array.from(zones.areas, area => area / 1e6);
  const groupAreasKm2 = new Float64Array(groups.groupCount);
  const groupLabelPoints: LngLat[] = [];
  const groupRings: LngLat[][][] = [];
  const groupOuterRings: LngLat[][][] = [];
  groups.members.forEach((members, group) => {
    let largest = members[0];
    const rings: LngLat[][] = [];
    const outers: LngLat[][] = [];
    for (const zone of members) {
      groupAreasKm2[group] += areasKm2[zone];
      if (areasKm2[zone] > areasKm2[largest]) largest = zone;
      for (const polygon of getPolygons(collection.features[zone].geometry)) {
        polygon.forEach((ring, ringIndex) => {
          const lngLat = ring.map(([longitude, latitude]) => [longitude, latitude] as LngLat);
          rings.push(lngLat);
          if (ringIndex === 0) outers.push(lngLat);
        });
      }
    }
    groupLabelPoints.push(mesh.labelPoints[largest] as LngLat);
    groupRings.push(rings);
    groupOuterRings.push(outers);
  });

  return {
    zones,
    groups,
    triangles,
    triangleFeatures,
    outlines: {
      official: Float32Array.from(outlineRows.official),
      channel: Float32Array.from(outlineRows.channel),
      approximate: Float32Array.from(outlineRows.approximate)
    },
    areasKm2,
    groupAreasKm2,
    groupLabelPoints,
    groupRings,
    groupOuterRings
  };
}
