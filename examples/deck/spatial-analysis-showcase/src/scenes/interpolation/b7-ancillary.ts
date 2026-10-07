// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {projectLngLatArray} from '../../engine/projection';

/** Placement of a raster whose row 0 is its south edge: `[originX, originY]` and a square cell. */
export type B7RasterGrid = {
  width: number;
  height: number;
  originX: number;
  originY: number;
  cellSize: number;
};

/** Fits a square-cell raster of `width x height` cells around `bounds`, centered on them. */
export function fitRasterGrid(
  bounds: readonly [number, number, number, number],
  width: number,
  height: number,
  marginFraction = 0.015
): B7RasterGrid {
  const marginX = (bounds[2] - bounds[0]) * marginFraction;
  const marginY = (bounds[3] - bounds[1]) * marginFraction;
  const cellSize = Math.max(
    (bounds[2] - bounds[0] + 2 * marginX) / width,
    (bounds[3] - bounds[1] + 2 * marginY) / height
  );
  return {
    width,
    height,
    cellSize,
    originX: (bounds[0] + bounds[2]) / 2 - (width * cellSize) / 2,
    originY: (bounds[1] + bounds[3]) / 2 - (height * cellSize) / 2
  };
}

/**
 * Meters of OpenStreetMap street per raster cell from the `chicago-roads` network. Each physical
 * street is counted once (the reverse direction of a two-way street is skipped).
 */
export function rasterizeStreetLength(
  roads: LoadedDataset,
  origin: readonly [number, number],
  grid: B7RasterGrid
): Float32Array {
  const projection = roads.getProjection(origin);
  const vertices = projectLngLatArray(projection, roads.column<Float32Array>('edgeVertices'), 2);
  const pathOffsets = roads.column<Uint32Array>('edgePathOffsets');
  const reverse = roads.column<Uint32Array>('edgeReverse');
  const raster = new Float32Array(grid.width * grid.height);
  const step = grid.cellSize / 2;
  const edgeCount = pathOffsets.length - 1;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (reverse[edge] !== 0xffffffff && reverse[edge] < edge) continue;
    for (let vertex = pathOffsets[edge]; vertex + 1 < pathOffsets[edge + 1]; vertex++) {
      const x0 = vertices[vertex * 2];
      const y0 = vertices[vertex * 2 + 1];
      const x1 = vertices[vertex * 2 + 2];
      const y1 = vertices[vertex * 2 + 3];
      const length = Math.hypot(x1 - x0, y1 - y0);
      const samples = Math.max(1, Math.ceil(length / step));
      const piece = length / samples;
      for (let sample = 0; sample < samples; sample++) {
        const t = (sample + 0.5) / samples;
        addToCell(raster, grid, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, piece);
      }
    }
  }
  return raster;
}

/** Number of places (points of interest) per raster cell from the `chicago-places` dataset. */
export function rasterizePointCount(
  points: LoadedDataset,
  origin: readonly [number, number],
  grid: B7RasterGrid
): Float32Array {
  const meters = points.projectColumn('position', origin);
  const raster = new Float32Array(grid.width * grid.height);
  for (let point = 0; point < meters.length / 2; point++) {
    addToCell(raster, grid, meters[point * 2], meters[point * 2 + 1], 1);
  }
  return raster;
}

function addToCell(
  raster: Float32Array,
  grid: B7RasterGrid,
  x: number,
  y: number,
  amount: number
): void {
  const column = Math.floor((x - grid.originX) / grid.cellSize);
  const row = Math.floor((y - grid.originY) / grid.cellSize);
  if (column >= 0 && row >= 0 && column < grid.width && row < grid.height) {
    raster[row * grid.width + column] += amount;
  }
}

/** Box blur with a square window of `radius` cells, in place of a copy. */
export function blurRaster(
  raster: Float32Array,
  width: number,
  height: number,
  radius: number
): Float32Array {
  let source = raster;
  for (let pass = 0; pass < 2; pass++) {
    const target = new Float32Array(source.length);
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        let sum = 0;
        let count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const x = column + dx;
            const y = row + dy;
            if (x < 0 || y < 0 || x >= width || y >= height) continue;
            sum += source[y * width + x];
            count++;
          }
        }
        target[row * width + column] = sum / count;
      }
    }
    source = target;
  }
  return source;
}

/**
 * Turns a raw per-cell activity raster into dasymetric weights: blurred, divided by a high
 * percentile and floored at `floor` so no cell is exactly zero (a zero-weight cell carries no mass).
 */
export function makeDasymetricWeights(
  raw: Float32Array,
  width: number,
  height: number,
  blurRadius: number,
  floor: number
): Float32Array {
  const smooth = blurRaster(raw, width, height, blurRadius);
  const positive = Array.from(smooth).filter(value => value > 0);
  positive.sort((a, b) => a - b);
  const reference = positive.length > 0 ? positive[Math.floor(positive.length * 0.95)] : 1;
  return Float32Array.from(smooth, value => Math.min(1, value / reference) * (1 - floor) + floor);
}
