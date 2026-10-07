// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {projectLngLatArray, type LocalMetricProjection} from '../../engine/projection';
import {loadPolygonLayout, type PolygonLayout} from './b3-common';

/** Longitude and latitude of the Loop (Chicago's central business district). */
export const CHICAGO_LOOP: readonly [number, number] = [-87.6298, 41.8781];

/** Road classes of `chicago-roads`, in manifest order, and the palette index each one draws with. */
export const ROAD_CLASSES = [
  'motorway',
  'trunk',
  'primary',
  'secondary',
  'tertiary',
  'residential',
  'service / other'
] as const;

/** A set of polylines (one flat vertex array plus path offsets) in local meters and in degrees. */
export type PathSet = {
  /** Local planar meters around the city origin. */
  local: Float32Array;
  /** Interleaved longitude and latitude degrees. */
  lngLat: Float32Array;
  /** `pathCount + 1` vertex offsets. */
  offsets: Uint32Array;
  pathCount: number;
  vertexCount: number;
  /** Optional per-path class (road class). */
  classes?: Uint8Array;
  /** Optional: path row of every directed source edge (both directions of a street map to one path). */
  edgeToPath?: Uint32Array;
};

/** Point set in local meters and degrees. */
export type PointSet = {
  local: Float32Array;
  lngLat: Float32Array;
  count: number;
};

/** The Chicago layers the Geometry chapter's scenes draw on, prepared once per scene. */
export type CityData = {
  origin: readonly [number, number];
  projection: LocalMetricProjection;
  /** `[west, south, east, north]` of the city. */
  bbox: readonly [number, number, number, number];
  railStations: PointSet & {routeCount: Uint8Array};
  allStops: PointSet;
  railLines: PathSet;
  areas: PolygonLayout & {local: Float32Array};
};

function pathsFromRanges(
  lngLatVertices: Float32Array,
  offsets: Uint32Array,
  include: (path: number) => boolean,
  projection: LocalMetricProjection
): PathSet {
  const kept: number[] = [];
  let vertexCount = 0;
  for (let path = 0; path + 1 < offsets.length; path++) {
    if (!include(path)) continue;
    kept.push(path);
    vertexCount += offsets[path + 1] - offsets[path];
  }
  const lngLat = new Float32Array(vertexCount * 2);
  const outputOffsets = new Uint32Array(kept.length + 1);
  let cursor = 0;
  kept.forEach((path, row) => {
    const start = offsets[path];
    const end = offsets[path + 1];
    lngLat.set(lngLatVertices.subarray(start * 2, end * 2), cursor * 2);
    cursor += end - start;
    outputOffsets[row + 1] = cursor;
  });
  return {
    local: projectLngLatArray(projection, lngLat),
    lngLat,
    offsets: outputOffsets,
    pathCount: kept.length,
    vertexCount
  };
}

/** Rail stations, all transit stops, rail route shapes and community areas around one origin. */
export function loadCityData(transit: LoadedDataset, areasDataset: LoadedDataset): CityData {
  const origin = areasDataset.defaultOrigin;
  const projection = areasDataset.getProjection(origin);
  const stopLngLat = transit.column<Float32Array>('stopPosition');
  const stopMode = transit.column<Uint8Array>('stopMode');
  const stopRoutes = transit.column<Uint8Array>('stopRouteCount');
  const allStops: PointSet = {
    local: transit.projectColumn('stopPosition', origin),
    lngLat: stopLngLat,
    count: stopMode.length
  };
  const railRows: number[] = [];
  for (let row = 0; row < stopMode.length; row++) if (stopMode[row] === 1) railRows.push(row);
  const railLngLat = new Float32Array(railRows.length * 2);
  const railLocal = new Float32Array(railRows.length * 2);
  const routeCount = new Uint8Array(railRows.length);
  railRows.forEach((row, index) => {
    railLngLat[index * 2] = stopLngLat[row * 2];
    railLngLat[index * 2 + 1] = stopLngLat[row * 2 + 1];
    railLocal[index * 2] = allStops.local[row * 2];
    railLocal[index * 2 + 1] = allStops.local[row * 2 + 1];
    routeCount[index] = stopRoutes[row];
  });

  const routes = (transit.properties.routes ?? []) as {type: string}[];
  const shapeRoute = transit.column<Uint16Array>('shapeRoute');
  const shapeDirection = transit.column<Uint8Array>('shapeDirection');
  const railLines = pathsFromRanges(
    transit.column<Float32Array>('shapeVertices'),
    transit.column<Uint32Array>('shapePathOffsets'),
    path => routes[shapeRoute[path]]?.type === 'rail' && shapeDirection[path] === 0,
    projection
  );

  const areas = loadPolygonLayout(areasDataset);
  return {
    origin,
    projection,
    bbox: areasDataset.manifest.bbox,
    railStations: {local: railLocal, lngLat: railLngLat, count: railRows.length, routeCount},
    allStops,
    railLines,
    areas: {...areas, local: areasDataset.projectColumn('vertices', origin)}
  };
}

/**
 * One polyline per street (the two directions of a two-way street collapse to one), in local
 * meters, with the road class of each.
 */
export function loadStreetPaths(roads: LoadedDataset, projection: LocalMetricProjection): PathSet {
  const oneway = roads.column<Uint8Array>('edgeOneway');
  const reverse = roads.column<Uint32Array>('edgeReverse');
  const classes = roads.column<Uint8Array>('edgeClass');
  const paths = pathsFromRanges(
    roads.column<Float32Array>('edgeVertices'),
    roads.column<Uint32Array>('edgePathOffsets'),
    edge => oneway[edge] === 1 || reverse[edge] === 0xffffffff || edge < reverse[edge],
    projection
  );
  const kept = new Uint8Array(paths.pathCount);
  const edgeToPath = new Uint32Array(classes.length);
  let row = 0;
  for (let edge = 0; edge < classes.length; edge++) {
    if (oneway[edge] === 1 || reverse[edge] === 0xffffffff || edge < reverse[edge]) {
      edgeToPath[edge] = row;
      kept[row++] = classes[edge];
    }
  }
  for (let edge = 0; edge < classes.length; edge++) {
    if (!(oneway[edge] === 1 || reverse[edge] === 0xffffffff || edge < reverse[edge])) {
      edgeToPath[edge] = edgeToPath[reverse[edge]];
    }
  }
  return {...paths, classes: kept, edgeToPath};
}

/** The L route shapes (one direction each) in local meters around `projection`'s origin. */
export function loadRailPaths(transit: LoadedDataset, projection: LocalMetricProjection): PathSet {
  const routes = (transit.properties.routes ?? []) as {type: string}[];
  const shapeRoute = transit.column<Uint16Array>('shapeRoute');
  const shapeDirection = transit.column<Uint8Array>('shapeDirection');
  return pathsFromRanges(
    transit.column<Float32Array>('shapeVertices'),
    transit.column<Uint32Array>('shapePathOffsets'),
    path => routes[shapeRoute[path]]?.type === 'rail' && shapeDirection[path] === 0,
    projection
  );
}
