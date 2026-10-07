// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {
  createFeatureLocator,
  type FeatureLocator,
  getGeometryPolygons,
  getInputPolygons
} from '../../cartography/picking';
import {buildPolygonMesh, type PolygonMesh} from '../../cartography/polygon-mesh';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {LngLat} from '../../cartography/types';
import type {LoadedDataset} from '../../data/catalog';
import type {GeoJsonFeature} from '../../data/loaders';
import {createPolygonMeshBuffers, type PolygonMeshBuffers} from '../../engine/polygon-buffers';
import {LocalMetricProjection} from '../../engine/projection';
import type {SpatialAnalysisResources} from '../../engine/resources';

/** GPU geometry of the county map: fill triangles, outlines, state lines and a selection outline. */
export type CountyGeometry = {
  /** `[longitude, latitude]` origin of every planar-metre buffer (the layers' coordinate origin). */
  origin: LngLat;
  features: readonly GeoJsonFeature[];
  mesh: PolygonMesh;
  buffers: PolygonMeshBuffers;
  locator: FeatureLocator;
  /** State boundary segments (the zone tier), projected around the same origin. */
  stateLines: {buffer: Buffer; segmentCount: number} | null;
  /** Rings of one county in longitude and latitude (tooltip highlight, label anchors). */
  getRings: (row: number) => LngLat[][];
  /**
   * Writes the outline of one county into the selection buffer and returns its segment count
   * (0 clears). The buffer is allocated once for the largest county.
   */
  setSelection: (row: number) => number;
  selectionBuffer: Buffer;
};

/**
 * Triangulates the counties once in planar metres around the dataset origin (the same metres as
 * `dataset.projectColumn`, so every layer of the scene shares one `coordinateOrigin`) and uploads
 * the static buffers. `states` adds the zone-boundary tier of the national map.
 */
export function createCountyGeometry(
  resources: SpatialAnalysisResources,
  counties: LoadedDataset,
  states: LoadedDataset
): CountyGeometry {
  const geojson = counties.geojson;
  if (!geojson) throw new Error('us-counties has no polygon geometry');
  const origin = counties.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const mesh = buildPolygonMesh(geojson, (longitude, latitude) =>
    projection.project(longitude, latitude)
  );
  const buffers = createPolygonMeshBuffers(resources, mesh, 'counties');
  const project = getLocalProjector(origin);

  const stateGeojson = states.geojson;
  const stateSegments = stateGeojson
    ? projectRingsToSegments(
        getInputPolygons(stateGeojson).flatMap(({polygon}) => polygon),
        project
      )
    : new Float32Array(0);
  const stateLines = stateSegments.length
    ? {
        buffer: resources.createBuffer('state-lines', stateSegments),
        segmentCount: stateSegments.length / 4
      }
    : null;

  const getRings = (row: number): LngLat[][] =>
    getGeometryPolygons(geojson.features[row]?.geometry).flatMap(polygon =>
      polygon.map(ring => ring.map(point => [point[0], point[1]] as LngLat))
    );

  // The selection buffer holds the outline of the county with the most outline segments.
  const segmentsPerRow = new Uint32Array(mesh.featureCount);
  for (const row of mesh.outlineFeatures) segmentsPerRow[row]++;
  const selectionBuffer = resources.createBuffer(
    'selection',
    (Math.max(1, ...segmentsPerRow) + 4) * 4 * 4
  );

  return {
    origin,
    features: geojson.features,
    mesh,
    buffers,
    locator: createFeatureLocator(geojson),
    stateLines,
    getRings,
    selectionBuffer,
    setSelection(row) {
      if (row < 0) return 0;
      const segments = projectRingsToSegments(getRings(row), project);
      selectionBuffer.write(segments);
      return segments.length / 4;
    }
  };
}
