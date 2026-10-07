// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/** Census tracts reduced to what the network scenes need: a centroid and a weight per tract. */
export type TractDemand = {
  count: number;
  /** Planar meters, `x, y` per tract. */
  centroids: Float32Array;
  /** Longitude and latitude, per tract. */
  centroidsLngLat: Float32Array;
  /** Residents per tract (`population`, NaN treated as zero). */
  population: Float32Array;
  /** Jobs per workplace tract (LEHD WAC 2021). */
  jobs: Float32Array;
  /** Workers who live in the tract (LEHD, 2021). */
  residentWorkers: Float32Array;
  /** Planar-meter rings per tract for rasterizing. */
  rings: Float32Array[][];
};

/** Reads `chicago-tracts`: centroid of the first ring and per-tract counts, in meters around `origin`. */
export function buildTractDemand(
  tracts: LoadedDataset,
  origin: readonly [number, number]
): TractDemand {
  const projection = tracts.getProjection(origin);
  const vertices = tracts.projectColumn('vertices', origin);
  const ringOffsets = tracts.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = tracts.column<Uint32Array>('polygonRingOffsets');
  const partFeature = tracts.column<Uint32Array>('partFeature');
  const count = tracts.count;
  const population = Float32Array.from(tracts.column<Float32Array>('population'), value =>
    Number.isFinite(value) ? value : 0
  );
  const jobs = Float32Array.from(tracts.column<Float32Array>('jobsWac2021'), value =>
    Number.isFinite(value) ? value : 0
  );
  const residentWorkers = Float32Array.from(
    tracts.column<Float32Array>('residentWorkers2021'),
    value => (Number.isFinite(value) ? value : 0)
  );
  const centroids = new Float32Array(count * 2);
  const centroidsLngLat = new Float32Array(count * 2);
  const rings: Float32Array[][] = Array.from({length: count}, () => []);
  for (let part = 0; part < partFeature.length; part++) {
    const feature = partFeature[part];
    for (let ring = polygonRingOffsets[part]; ring < polygonRingOffsets[part + 1]; ring++) {
      rings[feature].push(vertices.subarray(ringOffsets[ring] * 2, ringOffsets[ring + 1] * 2));
    }
  }
  for (let feature = 0; feature < count; feature++) {
    const ring = rings[feature][0];
    if (!ring) continue;
    // Area-weighted centroid of the outer ring; a plain vertex mean for degenerate rings.
    let area = 0;
    let cx = 0;
    let cy = 0;
    const n = ring.length / 2;
    for (let vertex = 0; vertex < n; vertex++) {
      const next = (vertex + 1) % n;
      const cross = ring[vertex * 2] * ring[next * 2 + 1] - ring[next * 2] * ring[vertex * 2 + 1];
      area += cross;
      cx += (ring[vertex * 2] + ring[next * 2]) * cross;
      cy += (ring[vertex * 2 + 1] + ring[next * 2 + 1]) * cross;
    }
    if (Math.abs(area) < 1e-3) {
      cx = 0;
      cy = 0;
      for (let vertex = 0; vertex < n; vertex++) {
        cx += ring[vertex * 2];
        cy += ring[vertex * 2 + 1];
      }
      centroids[feature * 2] = cx / n;
      centroids[feature * 2 + 1] = cy / n;
    } else {
      centroids[feature * 2] = cx / (3 * area);
      centroids[feature * 2 + 1] = cy / (3 * area);
    }
    const [longitude, latitude] = projection.unproject(
      centroids[feature * 2],
      centroids[feature * 2 + 1]
    );
    centroidsLngLat[feature * 2] = longitude;
    centroidsLngLat[feature * 2 + 1] = latitude;
  }
  return {count, centroids, centroidsLngLat, population, jobs, residentWorkers, rings};
}

/**
 * Spreads each tract's `values` uniformly over the raster cells whose centers fall inside it
 * (dasymetric-free areal weighting), so zonal sums over any zone layer approximate the people in
 * it. Row 0 is the minimum-y edge. A tract smaller than a cell lands in the cell of its centroid.
 */
export function rasterizeTractValues(
  demand: TractDemand,
  values: Float32Array,
  bounds: readonly [number, number, number, number],
  width: number,
  height: number
): Float32Array {
  const raster = new Float32Array(width * height);
  const cellWidth = (bounds[2] - bounds[0]) / width;
  const cellHeight = (bounds[3] - bounds[1]) / height;
  for (let feature = 0; feature < demand.count; feature++) {
    const rings = demand.rings[feature];
    if (!rings.length || !(values[feature] > 0)) continue;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const ring of rings) {
      for (let vertex = 0; vertex < ring.length; vertex += 2) {
        minX = Math.min(minX, ring[vertex]);
        maxX = Math.max(maxX, ring[vertex]);
        minY = Math.min(minY, ring[vertex + 1]);
        maxY = Math.max(maxY, ring[vertex + 1]);
      }
    }
    const column0 = Math.max(0, Math.floor((minX - bounds[0]) / cellWidth));
    const column1 = Math.min(width - 1, Math.floor((maxX - bounds[0]) / cellWidth));
    const row0 = Math.max(0, Math.floor((minY - bounds[1]) / cellHeight));
    const row1 = Math.min(height - 1, Math.floor((maxY - bounds[1]) / cellHeight));
    const covered: number[] = [];
    for (let row = row0; row <= row1; row++) {
      const y = bounds[1] + (row + 0.5) * cellHeight;
      for (let column = column0; column <= column1; column++) {
        const x = bounds[0] + (column + 0.5) * cellWidth;
        let inside = false;
        for (const ring of rings) {
          const n = ring.length / 2;
          for (let a = 0, b = n - 1; a < n; b = a++) {
            const ay = ring[a * 2 + 1];
            const by = ring[b * 2 + 1];
            if (ay > y !== by > y) {
              const crossX = ((ring[b * 2] - ring[a * 2]) * (y - ay)) / (by - ay) + ring[a * 2];
              if (x < crossX) inside = !inside;
            }
          }
        }
        if (inside) covered.push(row * width + column);
      }
    }
    if (covered.length === 0) {
      const column = Math.min(
        width - 1,
        Math.max(0, Math.floor((demand.centroids[feature * 2] - bounds[0]) / cellWidth))
      );
      const row = Math.min(
        height - 1,
        Math.max(0, Math.floor((demand.centroids[feature * 2 + 1] - bounds[1]) / cellHeight))
      );
      raster[row * width + column] += values[feature];
    } else {
      const share = values[feature] / covered.length;
      for (const cell of covered) raster[cell] += share;
    }
  }
  return raster;
}
