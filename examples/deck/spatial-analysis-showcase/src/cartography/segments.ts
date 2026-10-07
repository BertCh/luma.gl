// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Lines and rings to GPU segment rows (`x0, y0, x1, y1`) for `SpatialAnalysisSegmentLayer`, plus
 * the local metric projector that matches `LoadedDataset.projectColumn`. Pure TypeScript.
 */

import {LocalMetricProjection} from '../engine/projection';
import type {LngLat} from './types';

/** Maps `[longitude, latitude]` to planar `[x, y]` metres. */
export type Projector = (lngLat: readonly number[]) => readonly [number, number];

/**
 * Returns a projector to planar metres around `origin` that is numerically identical to what
 * `LoadedDataset.projectColumn(name, origin)` applies to a column (both use
 * `LocalMetricProjection`: deck.gl's high-precision `METER_OFFSETS` distance scales with one
 * refinement step). Annotation-free geometry projected with it lines up with the GPU layers of
 * the same scene as long as the same `origin` is passed to both.
 *
 * When a scene does not pass an origin to `projectColumn`, the dataset default is the centre of
 * its manifest bbox: use `getBboxCenter(bbox)` from `engine/projection` (or `dataset.defaultOrigin`).
 *
 * @example
 * const project = getLocalProjector(dataset.defaultOrigin);
 * const segments = projectLinesToSegments(routes, project);
 */
export function getLocalProjector(origin: LngLat): Projector {
  const projection = new LocalMetricProjection(origin);
  return lngLat => projection.project(lngLat[0], lngLat[1]);
}

/**
 * Projects polylines to a flat `x0, y0, x1, y1` float32 array (one row per consecutive vertex
 * pair), the layout of `SpatialAnalysisSegmentLayer`. Lines with fewer than two vertices and
 * segments with a non-finite end are skipped.
 *
 * @example
 * const rows = projectLinesToSegments([[a, b, c]], getLocalProjector(origin)); // 2 segments
 */
export function projectLinesToSegments(
  lines: readonly (readonly LngLat[])[],
  project: Projector
): Float32Array {
  let segmentCount = 0;
  for (const line of lines) segmentCount += Math.max(0, line.length - 1);
  const rows = new Float32Array(segmentCount * 4);
  let written = 0;
  for (const line of lines) {
    if (line.length < 2) continue;
    let [x0, y0] = project(line[0]);
    for (let i = 1; i < line.length; i++) {
      const [x1, y1] = project(line[i]);
      if (
        Number.isFinite(x0) &&
        Number.isFinite(y0) &&
        Number.isFinite(x1) &&
        Number.isFinite(y1)
      ) {
        rows[written++] = x0;
        rows[written++] = y0;
        rows[written++] = x1;
        rows[written++] = y1;
      }
      x0 = x1;
      y0 = y1;
    }
  }
  return written === rows.length ? rows : rows.slice(0, written);
}

/**
 * Projects polygon rings (outer rings and holes alike) to segment rows. A ring whose last vertex
 * differs from its first is closed with one extra segment, so open and GeoJSON-closed rings both
 * produce a closed outline.
 *
 * @example
 * const outline = projectRingsToSegments(polygon.coordinates as LngLat[][], project);
 */
export function projectRingsToSegments(
  rings: readonly (readonly (readonly number[])[])[],
  project: Projector
): Float32Array {
  const closed = rings.map(ring => {
    if (ring.length < 2) return [] as LngLat[];
    const vertices = ring.map(vertex => [vertex[0], vertex[1]] as LngLat);
    const first = vertices[0];
    const last = vertices[vertices.length - 1];
    if (first[0] !== last[0] || first[1] !== last[1]) vertices.push(first);
    return vertices;
  });
  return projectLinesToSegments(closed, project);
}

/**
 * Projects flat `lng0, lat0, lng1, lat1` rows (for example the output of `dissolveBoundaries`) to
 * `x0, y0, x1, y1` float32 rows. Without `project` the numbers are only converted to float32.
 */
export function projectSegmentRows(rows: ArrayLike<number>, project?: Projector): Float32Array {
  const count = Math.floor(rows.length / 4);
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const base = i * 4;
    if (project) {
      const [x0, y0] = project([rows[base], rows[base + 1]]);
      const [x1, y1] = project([rows[base + 2], rows[base + 3]]);
      out[base] = x0;
      out[base + 1] = y0;
      out[base + 2] = x1;
      out[base + 3] = y1;
    } else {
      out[base] = rows[base];
      out[base + 1] = rows[base + 1];
      out[base + 2] = rows[base + 2];
      out[base + 3] = rows[base + 3];
    }
  }
  return out;
}
