// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {projectLngLatArray, type LocalMetricProjection} from '../../engine/projection';
import type {PathSet} from './b3-city-data';
import {CHICAGO_LOOP} from './b3-city-data';
import type {PolygonLayout} from './b3-common';

/** Window around the Loop used for the bus-network noding input, in meters. */
export const DOWNTOWN_HALF_WIDTH = 3500;

/**
 * CTA route shapes ready for noding: one direction of every route inside a downtown window (or only
 * the eight L routes), clipped into runs of consecutive vertices inside the window, with exact
 * duplicates removed. Coordinates are local meters.
 */
export function buildNodingLines(
  transit: LoadedDataset,
  projection: LocalMetricProjection,
  scope: 'rail' | 'rail-bus'
): PathSet & {routeCount: number; duplicatesRemoved: number} {
  const routes = (transit.properties.routes ?? []) as {type: string}[];
  const shapeRoute = transit.column<Uint16Array>('shapeRoute');
  const shapeDirection = transit.column<Uint8Array>('shapeDirection');
  const offsets = transit.column<Uint32Array>('shapePathOffsets');
  const vertices = transit.column<Float32Array>('shapeVertices');
  const local = projectLngLatArray(projection, vertices);
  const loop = projection.project(CHICAGO_LOOP[0], CHICAGO_LOOP[1]);
  const inside = (vertex: number) =>
    Math.abs(local[vertex * 2] - loop[0]) <= DOWNTOWN_HALF_WIDTH &&
    Math.abs(local[vertex * 2 + 1] - loop[1]) <= DOWNTOWN_HALF_WIDTH;

  const positions: number[] = [];
  const lineOffsets: number[] = [0];
  const seen = new Set<string>();
  let duplicatesRemoved = 0;
  const usedRoutes = new Set<number>();
  for (let path = 0; path + 1 < offsets.length; path++) {
    const route = shapeRoute[path];
    const isRail = routes[route]?.type === 'rail';
    if (shapeDirection[path] !== 0) continue;
    if (scope === 'rail' && !isRail) continue;
    // Runs of consecutive vertices inside the window (rail uses the whole city).
    let run: number[] = [];
    const flush = () => {
      if (run.length >= 2) {
        const key = run
          .map(vertex => `${local[vertex * 2].toFixed(1)},${local[vertex * 2 + 1].toFixed(1)}`)
          .join(';');
        if (seen.has(key)) {
          duplicatesRemoved++;
        } else {
          seen.add(key);
          for (const vertex of run) positions.push(local[vertex * 2], local[vertex * 2 + 1]);
          lineOffsets.push(positions.length / 2);
          usedRoutes.add(route);
        }
      }
      run = [];
    };
    for (let vertex = offsets[path]; vertex < offsets[path + 1]; vertex++) {
      if (scope === 'rail' || inside(vertex)) run.push(vertex);
      else flush();
    }
    flush();
  }
  const localPositions = Float32Array.from(positions);
  return {
    local: localPositions,
    lngLat: new Float32Array(0),
    offsets: Uint32Array.from(lineOffsets),
    pathCount: lineOffsets.length - 1,
    vertexCount: localPositions.length / 2,
    routeCount: usedRoutes.size,
    duplicatesRemoved
  };
}

/** Directed ring edges of a polygon layer: `x0, y0, x1, y1` per edge and the feature row of each. */
export function buildDirectedEdges(layout: PolygonLayout): {
  endpoints: Float32Array;
  featureRows: Uint32Array;
  count: number;
} {
  const endpoints: number[] = [];
  const rows: number[] = [];
  const {lngLat, ringOffsets} = layout;
  for (let ring = 0; ring < layout.ringCount; ring++) {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1];
    for (let vertex = first; vertex < last; vertex++) {
      const next = vertex + 1 < last ? vertex + 1 : first;
      const x0 = lngLat[vertex * 2];
      const y0 = lngLat[vertex * 2 + 1];
      const x1 = lngLat[next * 2];
      const y1 = lngLat[next * 2 + 1];
      if (x0 === x1 && y0 === y1) continue;
      endpoints.push(x0, y0, x1, y1);
      rows.push(layout.ringFeature[ring]);
    }
  }
  return {
    endpoints: Float32Array.from(endpoints),
    featureRows: Uint32Array.from(rows),
    count: rows.length
  };
}
