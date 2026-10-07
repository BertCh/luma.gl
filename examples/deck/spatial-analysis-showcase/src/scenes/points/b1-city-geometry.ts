// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The City of Chicago as a study window: the city limit (dissolved community areas), Lake Michigan
 * and an "outside the city" mask, from the shared `chicago-boundary` dataset. Every `points` story
 * that has a data edge uses the same geometry and the same treatment, so the reader learns the edge
 * once: point-patterns (the window and the CSR null), street-density (no data outside the city),
 * nature-clusters (the share of a hull that is not city land).
 *
 * Scenes that call {@link loadCityGeometry} must list `chicago-boundary` in `datasets`.
 */

import {buildPolygonMesh, type PolygonMesh} from '../../cartography/polygon-mesh';
import {getInputPolygons, type PolygonRings} from '../../cartography/picking';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {LngLat} from '../../cartography/types';
import {fetchJson, type GeoJsonCollection, type GeoJsonFeature} from '../../data/loaders';
import type {SceneContext} from '../scene';

/** The box the mask covers (the extent of `chicago-boundary`), `[west, south, east, north]`. */
const MASK_BOX: readonly [number, number, number, number] = [-88.0, 41.55, -87.2, 42.15];

/** The city as a study window, in degrees and in the scene's local metres. */
export type CityGeometry = {
  /** City limit (3 parts: the city and two small O'Hare outliers). */
  city: GeoJsonCollection;
  /** Lake Michigan (Natural Earth 1:10m, generalised about 100 m), or null if it failed to load. */
  lake: GeoJsonCollection | null;
  /** Polygons of the city limit, outer ring first (degrees). */
  cityPolygons: PolygonRings[];
  /** Every city ring in local metres, flattened `x, y` pairs (for the even-odd test). */
  cityRingsMeters: Float64Array[];
  /** City bounding box in local metres `[minX, minY, maxX, maxY]`. */
  cityBoundsMeters: [number, number, number, number];
  /** City area in km² (planar, local metres). */
  cityAreaKm2: number;
  /** City-limit outline as segment rows `x0, y0, x1, y1` in local metres. */
  cityOutlineSegments: Float32Array;
  /** Lake shore as segment rows in local metres (empty if the lake did not load). */
  lakeOutlineSegments: Float32Array;
  /**
   * The world outside the city (a box with the city cut out, lake included), triangulated in local
   * metres for `SpatialAnalysisPolygonLayer`: paint it in ground ink or hatch to say "no data here".
   */
  outsideMesh: PolygonMesh;
  /** Lake Michigan triangulated in local metres (null if the lake did not load). */
  lakeMesh: PolygonMesh | null;
  /** True when local metres `(x, y)` are inside the city limit (even-odd over every ring). */
  containsMeters: (x: number, y: number) => boolean;
  /** True when `[lng, lat]` is inside the city limit. */
  contains: (lngLat: LngLat) => boolean;
};

/**
 * Loads the city limit and the lake and prepares the masks in the metres of `origin` (pass the same
 * origin the scene projects its points with, usually `dataset.defaultOrigin`).
 */
export async function loadCityGeometry(
  ctx: Pick<SceneContext<unknown>, 'datasets' | 'signal'>,
  origin: LngLat
): Promise<CityGeometry> {
  const boundary = ctx.datasets.get('chicago-boundary');
  const city = boundary.geojson ?? {type: 'FeatureCollection', features: []};
  let lake: GeoJsonCollection | null = null;
  try {
    lake = await fetchJson<GeoJsonCollection>(boundary.fileUrl('lake.geojson'), ctx.signal);
  } catch {
    // The lake is optional context: the city limit alone still defines the window.
  }
  return buildCityGeometry(city, lake, origin);
}

/** Builds {@link CityGeometry} from already-loaded GeoJSON (pure; used by {@link loadCityGeometry}). */
export function buildCityGeometry(
  city: GeoJsonCollection,
  lake: GeoJsonCollection | null,
  origin: LngLat
): CityGeometry {
  const project = getLocalProjector(origin);
  const cityPolygons = getInputPolygons(city).map(({polygon}) => polygon);
  const cityRingsMeters: Float64Array[] = [];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let areaSquareMeters = 0;
  for (const polygon of cityPolygons) {
    polygon.forEach((ring, ringIndex) => {
      const flat = new Float64Array(ring.length * 2);
      ring.forEach((vertex, index) => {
        const [x, y] = project(vertex);
        flat[index * 2] = x;
        flat[index * 2 + 1] = y;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      });
      cityRingsMeters.push(flat);
      const area = Math.abs(getSignedArea(flat));
      areaSquareMeters += ringIndex === 0 ? area : -area;
    });
  }

  const containsMeters = (x: number, y: number): boolean => {
    if (x < minX || x > maxX || y < minY || y > maxY) return false;
    let inside = false;
    for (const ring of cityRingsMeters) {
      const count = ring.length / 2;
      for (let i = 0, j = count - 1; i < count; j = i++) {
        const xi = ring[i * 2];
        const yi = ring[i * 2 + 1];
        const xj = ring[j * 2];
        const yj = ring[j * 2 + 1];
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
    }
    return inside;
  };

  // The outside mask: the box with every city part's outer ring as a hole.
  const [west, south, east, north] = MASK_BOX;
  const outsideFeature: GeoJsonFeature = {
    type: 'Feature',
    properties: null,
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [west, south],
          [east, south],
          [east, north],
          [west, north],
          [west, south]
        ],
        ...cityPolygons.map(polygon => polygon[0])
      ]
    }
  };
  const projectPair = (longitude: number, latitude: number) => project([longitude, latitude]);
  const outsideMesh = buildPolygonMesh([outsideFeature], projectPair);
  const lakeMesh = lake ? buildPolygonMesh(lake, projectPair) : null;
  const lakeRings = lake ? getInputPolygons(lake).flatMap(({polygon}) => polygon) : [];

  return {
    city,
    lake,
    cityPolygons,
    cityRingsMeters,
    cityBoundsMeters: [minX, minY, maxX, maxY],
    cityAreaKm2: areaSquareMeters / 1e6,
    cityOutlineSegments: projectRingsToSegments(
      cityPolygons.flatMap(polygon => polygon),
      project
    ),
    lakeOutlineSegments: lakeRings.length
      ? projectRingsToSegments(lakeRings, project)
      : new Float32Array(0),
    outsideMesh,
    lakeMesh,
    containsMeters,
    contains: lngLat => {
      const [x, y] = project(lngLat);
      return containsMeters(x, y);
    }
  };
}

/** Shoelace area of a flat `x, y` ring (positive counter-clockwise). */
function getSignedArea(ring: Float64Array): number {
  let sum = 0;
  const count = ring.length / 2;
  for (let i = 0, j = count - 1; i < count; j = i++) {
    sum += ring[j * 2] * ring[i * 2 + 1] - ring[i * 2] * ring[j * 2 + 1];
  }
  return sum / 2;
}
