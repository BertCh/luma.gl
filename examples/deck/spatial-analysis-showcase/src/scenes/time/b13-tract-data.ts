// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {LocalMetricProjection} from '../../engine/projection';
import {getPolygonSource, type B13PolygonSource} from './b13-polygon-raster';

/** Area-weighted centroids of every feature of a polygon dataset, in planar meters. */
export function getFeatureCentroids(
  source: B13PolygonSource,
  projection: LocalMetricProjection
): Float32Array {
  const {vertices, ringOffsets, polygonRingOffsets, partFeature, featureCount} = source;
  const sums = new Float64Array(featureCount * 3);
  const fallbacks = new Float64Array(featureCount * 3);
  for (let part = 0; part + 1 < polygonRingOffsets.length; part++) {
    const feature = partFeature[part];
    for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
      const start = ringOffsets[ring];
      const end = ringOffsets[ring + 1];
      let area = 0;
      let cx = 0;
      let cy = 0;
      for (let vertex = start; vertex < end; vertex++) {
        const next = vertex + 1 < end ? vertex + 1 : start;
        const [x0, y0] = projection.project(vertices[vertex * 2], vertices[vertex * 2 + 1]);
        const [x1, y1] = projection.project(vertices[next * 2], vertices[next * 2 + 1]);
        const cross = x0 * y1 - x1 * y0;
        area += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
        fallbacks[feature * 3] += x0;
        fallbacks[feature * 3 + 1] += y0;
        fallbacks[feature * 3 + 2] += 1;
      }
      if (Math.abs(area) > 1e-6) {
        // Holes wind the other way and subtract naturally.
        sums[feature * 3] += cx;
        sums[feature * 3 + 1] += cy;
        sums[feature * 3 + 2] += area;
      }
    }
  }
  const centroids = new Float32Array(featureCount * 2);
  for (let feature = 0; feature < featureCount; feature++) {
    const area = sums[feature * 3 + 2];
    if (Math.abs(area) > 1e-6) {
      centroids[feature * 2] = sums[feature * 3] / (3 * area);
      centroids[feature * 2 + 1] = sums[feature * 3 + 1] / (3 * area);
    } else {
      centroids[feature * 2] = fallbacks[feature * 3] / Math.max(1, fallbacks[feature * 3 + 2]);
      centroids[feature * 2 + 1] =
        fallbacks[feature * 3 + 1] / Math.max(1, fallbacks[feature * 3 + 2]);
    }
  }
  return centroids;
}

/** Chicago census tracts with centroids (meters around `origin`) and residents. */
export type B13Tracts = {
  count: number;
  source: B13PolygonSource;
  centroids: Float32Array;
  population: Float32Array;
  geoid: string[];
  communityArea: Uint8Array;
};

/** Reads the tract dataset. */
export function readTracts(tracts: LoadedDataset, origin: readonly [number, number]): B13Tracts {
  const source = getPolygonSource(tracts);
  const projection = tracts.getProjection(origin as [number, number]);
  const features = tracts.geojson?.features ?? [];
  return {
    count: source.featureCount,
    source,
    centroids: getFeatureCentroids(source, projection),
    population: tracts.column<Float32Array>('population'),
    geoid: features.map(feature => String((feature.properties as {GEOID?: string})?.GEOID ?? '')),
    communityArea: tracts.column<Uint8Array>('communityArea')
  };
}
