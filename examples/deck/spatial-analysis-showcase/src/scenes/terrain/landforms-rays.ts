// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The lines-of-sight overlay of the landforms story: the eight rays of one pinned cell, drawn to the
 * search radius with a paper casing and coloured by what they found (terrain rises, is level or
 * falls), a dot at the zenith (the highest angle) and a ring at the nadir (the lowest) of each ray,
 * and the reader's pin as a ring-and-dot in the signal colour. The rays come from the CPU twin of
 * the GPU contributor (`geomorphonRays` of `cpu-dem.ts`), so the picture is exactly what the GPU
 * does for every cell.
 */

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {MAP_INK} from '../../cartography/hue-registry';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {AlpsGrid} from './b14a-grid';
import type {GeomorphonResult} from './cpu-dem';
import type {TerrainGroundTone} from './terrain-palettes';

/** Ray colours by sign: terrain rises (+), is level (0) or falls (-), light and dark ground. */
export const RAY_COLORS: Record<
  TerrainGroundTone,
  {rise: readonly number[]; level: readonly number[]; fall: readonly number[]}
> = {
  light: {rise: [178, 24, 43, 255], level: [140, 140, 140, 255], fall: [33, 102, 172, 255]},
  dark: {rise: [255, 120, 130, 255], level: [176, 184, 196, 255], fall: [111, 176, 219, 255]}
};

const RAY_COUNT = 8;

/** The overlay: set the cell, then ask for its layers. */
export type RayOverlay = {
  /** Shows the rays of a result (or hides them with `null`). */
  setCell: (result: GeomorphonResult | null) => void;
  /** The layers, bottom to top: casing and rays, nadir rings, zenith dots, the pin. */
  getLayers: (ground: TerrainGroundTone) => Layer[];
};

/** Creates the overlay buffers on the scene's resources; all updates are small buffer writes. */
export function createRayOverlay(resources: SpatialAnalysisResources, grid: AlpsGrid): RayOverlay {
  const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
  const segments = resources.createBuffer('rays-segments', new Float32Array(RAY_COUNT * 4));
  const signs = resources.createBuffer('rays-signs', new Uint32Array(RAY_COUNT));
  const zeniths = resources.createBuffer('rays-zeniths', new Float32Array(RAY_COUNT * 2));
  const nadirs = resources.createBuffer('rays-nadirs', new Float32Array(RAY_COUNT * 2));
  const pin = resources.createBuffer('rays-pin', new Float32Array(2));
  let visible = false;

  const toMeters = (column: number, row: number): [number, number] => {
    const [longitude, latitude] = grid.getLongitudeLatitude(column, row);
    return grid.projection.project(longitude, latitude);
  };

  return {
    setCell(result) {
      visible = result !== null;
      if (!result) return;
      const segmentValues = new Float32Array(RAY_COUNT * 4);
      const signValues = new Uint32Array(RAY_COUNT);
      const zenithValues = new Float32Array(RAY_COUNT * 2);
      const nadirValues = new Float32Array(RAY_COUNT * 2);
      const centre = toMeters(result.column, result.row);
      result.rays.forEach((ray, index) => {
        const end = toMeters(ray.endCell[0], ray.endCell[1]);
        segmentValues.set([centre[0], centre[1], end[0], end[1]], index * 4);
        signValues[index] = ray.sign > 0 ? 0 : ray.sign === 0 ? 1 : 2;
        const zenith = ray.zenith ? toMeters(ray.zenith.cell[0], ray.zenith.cell[1]) : centre;
        const nadir = ray.nadir ? toMeters(ray.nadir.cell[0], ray.nadir.cell[1]) : centre;
        zenithValues.set(zenith, index * 2);
        nadirValues.set(nadir, index * 2);
      });
      segments.write(segmentValues);
      signs.write(signValues);
      zeniths.write(zenithValues);
      nadirs.write(nadirValues);
      pin.write(Float32Array.of(centre[0], centre[1]));
    },

    getLayers(ground) {
      if (!visible) return [];
      const ink = MAP_INK[ground];
      const colors = RAY_COLORS[ground];
      const paper = ground === 'dark' ? [9, 12, 16, 230] : [255, 255, 255, 235];
      const base = {coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS, coordinateOrigin: origin};
      const signal = hexToColor(ink.signal);
      const inkColor = hexToColor(ink.ink);
      const bufferOf = (buffer: Buffer) => buffer;
      return [
        new SpatialAnalysisSegmentLayer({
          ...base,
          id: 'landforms-rays',
          segments: bufferOf(segments),
          instanceCount: RAY_COUNT,
          values: signs,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: [colors.rise, colors.level, colors.fall] as [number, number, number, number][],
          widthPixels: 2.2,
          cap: 'round',
          outlineColor: paper as [number, number, number, number],
          outlineWidthPixels: 1.6
        }),
        new SpatialAnalysisPointLayer({
          ...base,
          id: 'landforms-rays-nadir',
          positions: nadirs,
          instanceCount: RAY_COUNT,
          shape: 'ring',
          radiusPixels: 4.5,
          outlineWidthPixels: 1.6,
          color: inkColor
        }),
        new SpatialAnalysisPointLayer({
          ...base,
          id: 'landforms-rays-zenith',
          positions: zeniths,
          instanceCount: RAY_COUNT,
          shape: 'circle',
          radiusPixels: 3.2,
          color: inkColor,
          outlineColor: paper as [number, number, number, number],
          outlineWidthPixels: 1.2
        }),
        new SpatialAnalysisPointLayer({
          ...base,
          id: 'landforms-pin-ring',
          positions: pin,
          instanceCount: 1,
          shape: 'ring',
          radiusPixels: 9,
          outlineWidthPixels: 2,
          color: signal
        }),
        new SpatialAnalysisPointLayer({
          ...base,
          id: 'landforms-pin-dot',
          positions: pin,
          instanceCount: 1,
          shape: 'circle',
          radiusPixels: 3,
          color: signal
        })
      ];
    }
  };
}

function hexToColor(hex: string): [number, number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255, 255];
}

// ---------------------------------------------------------------------------------------------
// The default pin
// ---------------------------------------------------------------------------------------------

/**
 * The most crest-like cell near a point: the cell within `searchMeters` of `lngLat` whose height
 * most exceeds the mean of the square window of `windowRadius` cells around it (the topographic
 * position at one small scale, computed directly on the CPU for a few thousand cells). Used to pin
 * the story's first cell on the Hörnli ridge. Returns `null` when the point misses the raster.
 */
export function findCrestCell(
  grid: AlpsGrid,
  lngLat: readonly [number, number],
  searchMeters: number,
  windowRadius: number
): {column: number; row: number} | null {
  const pixel = grid.getPixel(lngLat[0], lngLat[1]);
  if (!pixel) return null;
  const {width, height, cpuElevation} = grid;
  const cell = grid.getGroundCellSize(pixel[1]);
  const reach = Math.ceil(searchMeters / cell);
  let best: {column: number; row: number} | null = null;
  let bestPosition = Number.NEGATIVE_INFINITY;
  for (let row = Math.max(windowRadius, pixel[1] - reach); row <= pixel[1] + reach; row++) {
    for (
      let column = Math.max(windowRadius, pixel[0] - reach);
      column <= pixel[0] + reach;
      column++
    ) {
      if (column >= width - windowRadius || row >= height - windowRadius) continue;
      if (Math.hypot(column - pixel[0], row - pixel[1]) * cell > searchMeters) continue;
      let sum = 0;
      for (let rowOffset = -windowRadius; rowOffset <= windowRadius; rowOffset++) {
        const start = (row + rowOffset) * width + column;
        for (let columnOffset = -windowRadius; columnOffset <= windowRadius; columnOffset++) {
          sum += cpuElevation[start + columnOffset];
        }
      }
      const count = (2 * windowRadius + 1) ** 2;
      const position = cpuElevation[row * width + column] - sum / count;
      if (position > bestPosition) {
        bestPosition = position;
        best = {column, row};
      }
    }
  }
  return best;
}
