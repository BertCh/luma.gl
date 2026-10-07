// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Boundary dissolve (G22): turns polygons with a grouping (counties with a state) into the
 * interior group boundaries and the outer edge of the union, by shared-edge detection. Pure
 * TypeScript.
 */

import type {GeoJsonCollection} from '../data/loaders';
import {getGeometryPolygons, getRingsBounds, type LngLatBounds} from './picking';
import {projectSegmentRows, type Projector} from './segments';

/** Options of {@link dissolveBoundaries}. */
export type DissolveOptions = {
  /**
   * Vertices closer than this many degrees are the same vertex when matching shared edges.
   * Default `1e-7` (about 1 cm), which absorbs float32 round-off of stored coordinates.
   */
  toleranceDegrees?: number;
};

/** A group of features and its bounding box. */
export type BoundaryGroup = {
  /** Feature indices (rows of the collection) in the group. */
  featureIndices: number[];
  /** `[west, south, east, north]` of the group's polygons. */
  bbox: LngLatBounds;
};

/** Output of {@link dissolveBoundaries}. */
export type DissolvedBoundaries = {
  /**
   * Segments between features of different groups, `lng0, lat0, lng1, lat1` per segment, each
   * shared edge once. Pass to `toSegmentPairs` to project for a segment layer.
   */
  interior: Float64Array;
  /** Segments on the outer edge of the union of all features (same layout). Hole edges count. */
  exterior: Float64Array;
  /** Group key to its features and bounding box, in first-seen order. */
  groups: Map<string | number, BoundaryGroup>;
};

type EdgeRecord = {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  group: string | number;
  count: number;
  crossesGroups: boolean;
};

/**
 * Dissolves a polygon collection by group. Every ring edge is quantised to the tolerance and
 * hashed as an undirected edge: an edge seen once lies on the outer edge of the union
 * (`exterior`), an edge shared by features of different groups is a group boundary (`interior`),
 * and an edge shared inside one group is dropped.
 *
 * Edges only match when both neighbours have the same two vertices, as in topologically clean
 * data (Census cartographic boundaries); a vertex that sits mid-edge of the neighbour leaves
 * both edges in `exterior`. Handles `MultiPolygon` and holes. Cost is linear in vertices
 * (3,109 counties in a few milliseconds).
 *
 * @example
 * const {interior, exterior} = dissolveBoundaries(counties, feature => feature.properties?.fips.slice(0, 2));
 * const stateLines = toSegmentPairs(interior, project); // heavy state borders
 * const nation = toSegmentPairs(exterior, project); // coastline and national border
 */
export function dissolveBoundaries(
  geojson: GeoJsonCollection,
  getGroup: (feature: GeoJsonCollection['features'][number], index: number) => string | number,
  options: DissolveOptions = {}
): DissolvedBoundaries {
  const scale = 1 / Math.max(1e-12, options.toleranceDegrees ?? 1e-7);
  const edges = new Map<string, EdgeRecord>();
  const groups = new Map<string | number, BoundaryGroup>();

  geojson.features.forEach((feature, featureIndex) => {
    const group = getGroup(feature, featureIndex);
    let entry = groups.get(group);
    if (!entry) {
      entry = {featureIndices: [], bbox: [Infinity, Infinity, -Infinity, -Infinity]};
      groups.set(group, entry);
    }
    entry.featureIndices.push(featureIndex);
    for (const polygon of getGeometryPolygons(feature.geometry)) {
      const bounds = getRingsBounds(polygon);
      if (bounds) {
        entry.bbox = [
          Math.min(entry.bbox[0], bounds[0]),
          Math.min(entry.bbox[1], bounds[1]),
          Math.max(entry.bbox[2], bounds[2]),
          Math.max(entry.bbox[3], bounds[3])
        ];
      }
      for (const ring of polygon) {
        for (let i = 1; i < ring.length; i++) {
          const ax = ring[i - 1][0];
          const ay = ring[i - 1][1];
          const bx = ring[i][0];
          const by = ring[i][1];
          const qax = Math.round(ax * scale);
          const qay = Math.round(ay * scale);
          const qbx = Math.round(bx * scale);
          const qby = Math.round(by * scale);
          if (qax === qbx && qay === qby) continue;
          const forward = qax < qbx || (qax === qbx && qay < qby);
          const key = forward ? `${qax},${qay},${qbx},${qby}` : `${qbx},${qby},${qax},${qay}`;
          const record = edges.get(key);
          if (!record) {
            edges.set(key, {ax, ay, bx, by, group, count: 1, crossesGroups: false});
          } else {
            record.count++;
            if (record.group !== group) record.crossesGroups = true;
          }
        }
      }
    }
  });

  let interiorCount = 0;
  let exteriorCount = 0;
  for (const record of edges.values()) {
    if (record.count === 1) exteriorCount++;
    else if (record.crossesGroups) interiorCount++;
  }
  const interior = new Float64Array(interiorCount * 4);
  const exterior = new Float64Array(exteriorCount * 4);
  let interiorAt = 0;
  let exteriorAt = 0;
  for (const record of edges.values()) {
    if (record.count === 1) {
      exterior.set([record.ax, record.ay, record.bx, record.by], exteriorAt);
      exteriorAt += 4;
    } else if (record.crossesGroups) {
      interior.set([record.ax, record.ay, record.bx, record.by], interiorAt);
      interiorAt += 4;
    }
  }
  return {interior, exterior, groups};
}

/**
 * State boundaries from a county collection, by the two leading digits of the county FIPS
 * (`properties.fips`, a five-character string such as `'01001'`; the `us-counties` dataset
 * stores `{fips, name, state}` per county). Pass `fipsProperty` for another property that begins
 * with the state FIPS (`GEOID` in raw Census files); numeric values are zero-padded to five
 * digits.
 *
 * For state outlines alone prefer the `us-states` dataset, which ships ready-made polygons; use
 * this when the story draws counties and wants the state lines between them from the same
 * vertices, so the two always line up.
 *
 * @example
 * const states = getStateBoundaries(counties);
 * states.groups.get('17')?.featureIndices.length; // Illinois: 102 counties
 */
export function getStateBoundaries(
  counties: GeoJsonCollection,
  fipsProperty = 'fips',
  options: DissolveOptions = {}
): DissolvedBoundaries {
  return dissolveBoundaries(
    counties,
    (feature, index) => {
      const value = feature.properties?.[fipsProperty] ?? feature.id;
      if (value === undefined || value === null) return `row-${index}`;
      return String(value).padStart(5, '0').slice(0, 2);
    },
    options
  );
}

/**
 * Converts flat `lng0, lat0, lng1, lat1` segments (from {@link dissolveBoundaries}) to the
 * `x0, y0, x1, y1` float32 rows of `SpatialAnalysisSegmentLayer`, projecting with `project`
 * (`getLocalProjector(origin)` from `segments.ts` to line up with the dataset's GPU layers).
 * Without `project` the values are returned as float32 degrees.
 *
 * @example
 * const project = getLocalProjector(dataset.defaultOrigin);
 * const rows = toSegmentPairs(getStateBoundaries(counties).interior, project);
 */
export function toSegmentPairs(lines: ArrayLike<number>, project?: Projector): Float32Array {
  return projectSegmentRows(lines, project);
}
