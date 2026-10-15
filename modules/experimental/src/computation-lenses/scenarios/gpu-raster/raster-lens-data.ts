// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../gpu-raster/raster-grid/index';
import type {GPURasterMetadata} from '../../../gpu-raster/types';

/** Version stamped into every raster/terrain lens run id; bump when the extracts change. */
export const T_DATA_VERSION = '2026-10-14';

/** Header shared by the stream-T data files (rows north → south, row-major). */
type TGridFile = {
  id: string;
  crs: string;
  width: number;
  height: number;
  cellSize: number;
  origin: readonly number[];
  unit: string;
};

/** A lens grid plus the facts the harness and captions need. */
export type TLensGrid = {
  grid: GPURasterMetadata;
  coordinateModel: 'projected' | 'geographic';
  /** Horizontal unit: `m` or `deg`. */
  unit: string;
  crs: string;
  cellSize: number;
};

function toLensGrid(file: TGridFile): TLensGrid {
  const geographic = file.crs === 'EPSG:4326';
  return {
    grid: makeGPURasterMetadata({
      width: file.width,
      height: file.height,
      cellSize: file.cellSize,
      origin: [file.origin[0], file.origin[1]]
    }),
    coordinateModel: geographic ? 'geographic' : 'projected',
    unit: geographic ? 'deg' : 'm',
    crs: file.crs,
    cellSize: file.cellSize
  };
}

/** 64 × 64 Dixie Fire DEM, 120 m UTM 10N cells, elevations in metres. */
export async function loadDixieDem(): Promise<TLensGrid & {elevation: Float32Array}> {
  const file = (await import('../../data/t-dixie-dem.json')).default as TGridFile & {
    values: number[];
  };
  return {...toLensGrid(file), elevation: Float32Array.from(file.values)};
}

/** Sentinel-2 reflectance (DN × 0.0001, 0 = no data) on the DEM grid. */
export async function loadDixieBands(): Promise<
  TLensGrid & {
    scale: number;
    noData: number;
    bands: Record<'nirBefore' | 'swirBefore' | 'nirAfter' | 'swirAfter', Float32Array>;
  }
> {
  const file = (await import('../../data/t-dixie-bands.json')).default as TGridFile & {
    scale: number;
    noData: number;
    bands: Record<'nirBefore' | 'swirBefore' | 'nirAfter' | 'swirAfter', number[]>;
  };
  return {
    ...toLensGrid(file),
    scale: file.scale,
    noData: file.noData,
    bands: {
      nirBefore: Float32Array.from(file.bands.nirBefore),
      swirBefore: Float32Array.from(file.bands.swirBefore),
      nirAfter: Float32Array.from(file.bands.nirAfter),
      swirAfter: Float32Array.from(file.bands.swirAfter)
    }
  };
}

/** WorldCover class codes on the DEM grid, with their names. */
export async function loadDixieLandCover(): Promise<
  TLensGrid & {classes: Uint32Array; classNames: Readonly<Record<string, string>>}
> {
  const file = (await import('../../data/t-dixie-landcover.json')).default as TGridFile & {
    values: number[];
    classes: Record<string, string>;
  };
  return {...toLensGrid(file), classes: Uint32Array.from(file.values), classNames: file.classes};
}

/** Dixie perimeter rings (closed, UTM metres) clipped to the DEM window. */
export async function loadDixiePerimeter(): Promise<
  TLensGrid & {rings: Float32Array[]; outer: boolean[]; coverage: number}
> {
  const file = (await import('../../data/t-dixie-perimeter.json')).default as TGridFile & {
    rings: number[][];
    inside: boolean[];
    coverage: number;
  };
  return {
    ...toLensGrid(file),
    rings: file.rings.map(ring => Float32Array.from(ring)),
    outer: file.inside,
    coverage: file.coverage
  };
}

/** GFS 10 m wind u/v (m/s) on a 64 × 48 lon/lat grid of 0.25° cells. */
export async function loadGFSWind(): Promise<TLensGrid & {u: Float32Array; v: Float32Array}> {
  const file = (await import('../../data/t-gfs-wind.json')).default as TGridFile & {
    u: number[];
    v: number[];
  };
  return {...toLensGrid(file), u: Float32Array.from(file.u), v: Float32Array.from(file.v)};
}

/** Raw Terrarium RGBA bytes (one value per channel) of a 32 × 32 block at 20 m. */
export async function loadDixieTerrarium(): Promise<TLensGrid & {rgba: Uint8Array}> {
  const file = (await import('../../data/t-dixie-terrarium.json')).default as TGridFile & {
    rgba: number[];
  };
  return {...toLensGrid(file), rgba: Uint8Array.from(file.rgba)};
}
