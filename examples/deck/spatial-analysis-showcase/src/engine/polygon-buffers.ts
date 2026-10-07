// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {PolygonMesh} from '../cartography/polygon-mesh';
import type {SpatialAnalysisResources} from './resources';

/**
 * GPU buffers of a {@link PolygonMesh}, ready for `SpatialAnalysisPolygonLayer` (fill) and
 * `SpatialAnalysisSegmentLayer` (outlines). Features are the rows of the mesh's source features, so
 * one `values` buffer (one value per feature) colours both the fill and the outline.
 */
export type PolygonMeshBuffers = {
  /** `float32x2` triangle vertices (the polygon layer's `triangles`). */
  triangles: Buffer;
  /** `uint32` feature row per triangle vertex (the polygon layer's `features`). */
  triangleFeatures: Buffer;
  /** Triangle vertex count (the polygon layer's `vertexCount`). */
  vertexCount: number;
  /** `float32x4` outline segments (a segment layer's `segments`). */
  outline: Buffer;
  /** `uint32` feature row per outline segment (a segment layer's `valueIndices`). */
  outlineFeatures: Buffer;
  /** Outline segment count (a segment layer's `instanceCount`). */
  outlineCount: number;
};

/** GPU buffers of the outline segments of a {@link PolygonMesh}. */
export type PolygonOutlineBuffers = Pick<
  PolygonMeshBuffers,
  'outline' | 'outlineFeatures' | 'outlineCount'
>;

/**
 * Uploads only the outline segments of a polygon mesh, for scenes that draw boundaries alone (a
 * region frame over a raster, class boundaries) and never fill the polygons. Pass `outline` as a
 * segment layer's `segments`, `outlineFeatures` as its `valueIndices` and `outlineCount` as its
 * `instanceCount`.
 */
export function createPolygonOutlineBuffers(
  resources: SpatialAnalysisResources,
  mesh: PolygonMesh,
  id: string
): PolygonOutlineBuffers {
  return {
    outline: resources.createBuffer(`${id}-outline`, mesh.outlineSegments),
    outlineFeatures: resources.createBuffer(`${id}-outline-features`, mesh.outlineFeatures),
    outlineCount: mesh.outlineSegments.length / 4
  };
}

/**
 * Uploads a polygon mesh once (fill triangles and outline segments).
 *
 * Returns `triangles` (`float32x2`, three vertices per triangle), `triangleFeatures` (`uint32`
 * feature row per vertex), `vertexCount`, and `outline` / `outlineFeatures` / `outlineCount` (see
 * {@link PolygonMeshBuffers}). Recolouring the polygons later is a write to the value buffer the
 * layers read, never a re-upload of geometry.
 *
 * ```ts
 * const mesh = buildPolygonMesh(dataset.geojson!, (lng, lat) => projection.project(lng, lat));
 * const polygons = createPolygonMeshBuffers(resources, mesh, 'tracts');
 * new SpatialAnalysisPolygonLayer({id: 'tract-fill', coordinateOrigin, triangles: polygons.triangles,
 *   features: polygons.triangleFeatures, vertexCount: polygons.vertexCount, values, colormap: 'ylgnbu',
 *   classBreaks: breaks});
 * new SpatialAnalysisSegmentLayer({id: 'tract-outline', coordinateOrigin, segments: polygons.outline,
 *   instanceCount: polygons.outlineCount, widthPixels: 0.6, color: outlineColor});
 * ```
 */
export function createPolygonMeshBuffers(
  resources: SpatialAnalysisResources,
  mesh: PolygonMesh,
  id: string
): PolygonMeshBuffers {
  // Empty arrays become 4-byte buffers (see `createBuffer`), so layers can always bind them.
  return {
    triangles: resources.createBuffer(`${id}-triangles`, mesh.triangles),
    triangleFeatures: resources.createBuffer(`${id}-triangle-features`, mesh.triangleFeatures),
    vertexCount: mesh.triangleFeatures.length,
    outline: resources.createBuffer(`${id}-outline`, mesh.outlineSegments),
    outlineFeatures: resources.createBuffer(`${id}-outline-features`, mesh.outlineFeatures),
    outlineCount: mesh.outlineSegments.length / 4
  };
}
