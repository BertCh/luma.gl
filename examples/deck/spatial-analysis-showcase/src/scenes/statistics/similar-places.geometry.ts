// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {greatCircleArc} from '../../cartography/reference-geometry';
import {
  getLocalProjector,
  projectLinesToSegments,
  projectRingsToSegments
} from '../../cartography/segments';
import type {LngLat} from '../../cartography/types';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {CountyGeometry} from './choropleth-classes.geometry';
import {getKilometers} from './similar-places.stats';

/** A GPU segment buffer that is rewritten when the matches change. */
export type SegmentWriter = {
  buffer: Buffer;
  /** Writes the segments and returns how many to draw. */
  write: (segments: Float32Array) => number;
};

/**
 * Segment buffer of a fixed capacity. Segments beyond the capacity are dropped (the capacity is
 * sized from the largest counties, so this only guards against a mistake).
 */
export function createSegmentWriter(
  resources: SpatialAnalysisResources,
  name: string,
  capacitySegments: number
): SegmentWriter {
  const buffer = resources.createBuffer(name, Math.max(1, capacitySegments) * 16);
  return {
    buffer,
    write(segments) {
      const count = Math.min(segments.length / 4, capacitySegments);
      if (count > 0) buffer.write(segments.subarray(0, count * 4));
      return count;
    }
  };
}

/**
 * Segment capacity for the outlines of any `rowCount` counties: the `rowCount` largest outlines
 * of the mesh plus a closing segment per ring.
 */
export function getOutlineCapacity(geometry: CountyGeometry, rowCount: number): number {
  const segmentsPerRow = new Uint32Array(geometry.mesh.featureCount);
  for (const row of geometry.mesh.outlineFeatures) segmentsPerRow[row]++;
  const largest = [...segmentsPerRow].sort((a, b) => b - a).slice(0, rowCount);
  return largest.reduce((sum, count) => sum + count + 8, 0);
}

/** Outline segments of several counties, projected around the geometry's origin. */
export function getOutlineSegments(
  geometry: CountyGeometry,
  rows: readonly number[]
): Float32Array {
  const project = getLocalProjector(geometry.origin);
  const parts = rows.map(row => projectRingsToSegments(geometry.getRings(row), project));
  const merged = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    merged.set(part, offset);
    offset += part.length;
  }
  return merged;
}

/** Kilometres per arc segment: dashes restart on every segment, so long arcs need several. */
const KILOMETERS_PER_ARC_SEGMENT = 400;
const MAXIMUM_ARC_SEGMENTS = 12;
/** Segment capacity of one arc. */
export const ARC_SEGMENT_CAPACITY = MAXIMUM_ARC_SEGMENTS;

/**
 * Great-circle arcs from `from` to every `to`, as projected segments. Each arc is cut into one
 * segment per 400 km (at most 12), so a dashed line keeps its rhythm at the national scale.
 */
export function getArcSegments(
  geometry: CountyGeometry,
  from: LngLat,
  targets: readonly LngLat[]
): Float32Array {
  const project = getLocalProjector(geometry.origin);
  const lines: LngLat[][] = [];
  for (const target of targets) {
    const pieces = Math.min(
      MAXIMUM_ARC_SEGMENTS,
      Math.max(1, Math.ceil(getKilometers(from, target) / KILOMETERS_PER_ARC_SEGMENT))
    );
    // The contiguous US never crosses the antimeridian, so the arc is one polyline.
    lines.push(...greatCircleArc(from, target, pieces));
  }
  return projectLinesToSegments(lines, project);
}
