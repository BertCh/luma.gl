// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NetworkCSR} from '../network-reachability/network-reachability-oracle';

export type IsochroneRasterSettings = {
  width: number;
  height: number;
  extent: [number, number, number, number];
  bufferRadius: number;
  walkCostPerUnit: number;
  mode: 'min' | 'max';
  maximumBufferPixels: number;
  maximumSamplesPerEdge: number;
  unreachedCost: number;
};

/** f64 mirror of the splat kernel: samples edges, applies the buffer, min or max per pixel. */
export function splatIsochroneRasterOnCPU(
  csr: NetworkCSR,
  positions: Float32Array,
  costs: Float32Array,
  settings: IsochroneRasterSettings
): Float32Array {
  const {width, height, extent, mode} = settings;
  const [minX, minY, maxX, maxY] = extent;
  const cellX = (maxX - minX) / width;
  const cellY = (maxY - minY) / height;
  const radius = Math.max(settings.bufferRadius, 0.5 * Math.hypot(cellX, cellY));
  const reachX = Math.min(Math.ceil(radius / cellX), settings.maximumBufferPixels);
  const reachY = Math.min(Math.ceil(radius / cellY), settings.maximumBufferPixels);
  const values = new Float64Array(width * height).fill(NaN);
  const nodeCount = costs.length;
  for (let node = 0; node < nodeCount; node++) {
    if (!Number.isFinite(costs[node])) {
      continue;
    }
    for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
      const target = csr.neighbors[edge];
      const weight = csr.weights[edge];
      if (target >= nodeCount || !(weight >= 0)) {
        continue;
      }
      const startX = positions[2 * node];
      const startY = positions[2 * node + 1];
      const endX = positions[2 * target];
      const endY = positions[2 * target + 1];
      const length = Math.hypot(endX - startX, endY - startY);
      const sampleCount = Math.min(
        Math.max(Math.ceil(length / (0.5 * radius)) + 1, 2),
        settings.maximumSamplesPerEdge
      );
      for (let sample = 0; sample < sampleCount; sample++) {
        const t = sample / (sampleCount - 1);
        const pointX = startX + t * (endX - startX);
        const pointY = startY + t * (endY - startY);
        const edgeCost = costs[node] + t * weight;
        const centerColumn = Math.floor((pointX - minX) / cellX);
        const centerRow = Math.floor((pointY - minY) / cellY);
        for (let row = centerRow - reachY; row <= centerRow + reachY; row++) {
          for (let column = centerColumn - reachX; column <= centerColumn + reachX; column++) {
            if (row < 0 || row >= height || column < 0 || column >= width) {
              continue;
            }
            const distance = Math.hypot(
              minX + (column + 0.5) * cellX - pointX,
              minY + (row + 0.5) * cellY - pointY
            );
            if (distance > radius) {
              continue;
            }
            const cost = edgeCost + settings.walkCostPerUnit * distance;
            const slot = row * width + column;
            const current = values[slot];
            values[slot] = Number.isNaN(current)
              ? cost
              : mode === 'min'
                ? Math.min(current, cost)
                : Math.max(current, cost);
          }
        }
      }
    }
  }
  return Float32Array.from(values, value => (Number.isNaN(value) ? settings.unreachedCost : value));
}

/**
 * Bounds of the area of each band of the marching-squares isobands of a raster whose samples sit at
 * pixel centers: the area of cells whose four corners are all in the band (lower bound) and of
 * cells with any corner in the band (upper bound).
 */
export function getBandAreaBounds(
  values: Float32Array,
  width: number,
  height: number,
  breaks: number[],
  cellArea: number
): {lower: number[]; upper: number[]} {
  const bandOf = (value: number) => breaks.filter(breakValue => breakValue <= value).length;
  const lower = new Array(breaks.length + 1).fill(0);
  const upper = new Array(breaks.length + 1).fill(0);
  for (let row = 0; row + 1 < height; row++) {
    for (let column = 0; column + 1 < width; column++) {
      const bands = [
        values[row * width + column],
        values[row * width + column + 1],
        values[(row + 1) * width + column],
        values[(row + 1) * width + column + 1]
      ].map(bandOf);
      if (bands.every(band => band === bands[0])) {
        lower[bands[0]] += cellArea;
      }
      for (const band of new Set(bands)) {
        upper[band] += cellArea;
      }
    }
  }
  return {lower, upper};
}

/** Total area of triangles per band (counter-clockwise triangles are positive). */
export function getTriangleAreasPerBand(
  triangles: number[],
  bands: number[],
  bandCount: number
): number[] {
  const areas = new Array(bandCount).fill(0);
  bands.forEach((band, index) => {
    const [ax, ay, bx, by, cx, cy] = triangles.slice(6 * index, 6 * index + 6);
    areas[band] += 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
  });
  return areas;
}
