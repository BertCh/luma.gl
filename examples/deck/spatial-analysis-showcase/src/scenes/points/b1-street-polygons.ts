// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {LocalMetricProjection} from '../../engine/projection';

/**
 * Polygons in the layout the `GPULineLengthPerPolygon` graph reads. Drawing and picking use the
 * triangulated GeoJSON mesh instead (`buildPolygonMesh`); feature rows are the same in both.
 */
export type StreetPolygonSet = {
  id: string;
  label: string;
  /** Ring vertices as local meters `x, y`. */
  positions: Float32Array;
  /** The same vertices as longitude and latitude degrees (for the spherical graphs). */
  degrees: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  names: readonly string[];
  /** Area per feature in square meters (shell minus holes). */
  areas: Float32Array;
  /** Residents per feature. */
  population: Float32Array;
};

/**
 * Builds a polygon set from a binary GeoArrow-style dataset (`vertices`, `ringOffsets`,
 * `polygonRingOffsets`, `partFeature`). Parts are grouped by feature, rings stay open.
 */
export function readPolygonSet(
  dataset: LoadedDataset,
  projection: LocalMetricProjection,
  options: {id: string; label: string; names: readonly string[]; population?: Float32Array}
): StreetPolygonSet {
  const vertices = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partFeature = dataset.column<Uint32Array>('partFeature');
  const featureCount = options.names.length;
  // Rebuild the arrays so rings are open (no repeated closing vertex) and parts follow features.
  const positions: number[] = [];
  const degrees: number[] = [];
  const newRingOffsets = [0];
  const newPolygonOffsets = [0];
  const newFeatureOffsets = [0];
  const areas = new Float32Array(featureCount);
  const partsByFeature: number[][] = Array.from({length: featureCount}, () => []);
  for (let part = 0; part < partFeature.length; part++)
    partsByFeature[partFeature[part]].push(part);
  for (let feature = 0; feature < featureCount; feature++) {
    for (const part of partsByFeature[feature]) {
      for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
        let start = ringOffsets[ring];
        let end = ringOffsets[ring + 1];
        if (
          end - start > 1 &&
          vertices[start * 2] === vertices[(end - 1) * 2] &&
          vertices[start * 2 + 1] === vertices[(end - 1) * 2 + 1]
        ) {
          end--;
        }
        const ringPoints: [number, number][] = [];
        for (let vertex = start; vertex < end; vertex++) {
          const longitude = vertices[vertex * 2];
          const latitude = vertices[vertex * 2 + 1];
          degrees.push(longitude, latitude);
          ringPoints.push(projection.project(longitude, latitude));
        }
        let twiceArea = 0;
        for (let index = 0; index < ringPoints.length; index++) {
          const [x0, y0] = ringPoints[index];
          const [x1, y1] = ringPoints[(index + 1) % ringPoints.length];
          positions.push(x0, y0);
          twiceArea += x0 * y1 - x1 * y0;
        }
        newRingOffsets.push(positions.length / 2);
        // The first ring of a part is its shell; later rings are holes.
        areas[feature] += (ring === polygonRingOffsets[part] ? 1 : -1) * Math.abs(twiceArea / 2);
        start = end;
      }
      newPolygonOffsets.push(newRingOffsets.length - 1);
    }
    newFeatureOffsets.push(newPolygonOffsets.length - 1);
  }
  return {
    id: options.id,
    label: options.label,
    positions: Float32Array.from(positions),
    degrees: Float32Array.from(degrees),
    featureOffsets: Uint32Array.from(newFeatureOffsets),
    polygonOffsets: Uint32Array.from(newPolygonOffsets),
    ringOffsets: Uint32Array.from(newRingOffsets),
    names: options.names,
    areas,
    population: options.population ?? new Float32Array(featureCount)
  };
}
