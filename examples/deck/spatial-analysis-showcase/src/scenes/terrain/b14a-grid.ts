// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {fetchBytes, getDataFileUrl} from '../../data/loaders';
import {LocalMetricProjection} from '../../engine/projection';

/** Circumference of the Web Mercator world in meters. */
const WORLD_METERS = 40075016.68557849;

/** A named observer of the alps-dem manifest. */
export type AlpsObserver = {
  name: string;
  longitude: number;
  latitude: number;
  elevationMeters: number;
};

/**
 * The alps-dem raster prepared for the terrain contributors.
 *
 * The PNG is Terrarium in Web Mercator, so every contributor runs with
 * `cellSizeMode: 'web-mercator'`: `cellSize` is the equatorial Mercator pixel size (about 9.55 m)
 * and `northEdge` / `southEdge` are the normalized Web Mercator y of the raster edges, from which
 * the contributor derives the true ground size of each row (about 6.6 m at 46 degrees north).
 */
export type AlpsGrid = {
  width: number;
  height: number;
  pixelCount: number;
  /** Packed little-endian RGBA8 words of the Terrarium PNG, exactly as `getImageData` returns them. */
  packedWords: Uint32Array;
  /** Elevation decoded on the CPU by the dataset loader (the reference for the GPU decode). */
  cpuElevation: Float32Array;
  /** `[minX, minY, maxX, maxY]` meters around `origin`, the outer cell edges for the layers. */
  bounds: [number, number, number, number];
  /** `[longitude, latitude]` layer coordinate origin. */
  origin: [number, number];
  projection: LocalMetricProjection;
  /** Web Mercator pixel size, equatorial meters. */
  mercatorCellSize: number;
  /** Normalized Web Mercator y of the top of row 0. */
  northEdge: number;
  /** Normalized Web Mercator y of the bottom of the last row. */
  southEdge: number;
  /** Ground meters per pixel at the raster's central row. */
  groundCellSize: number;
  /** `[minX, minY, maxX, maxY]` Web Mercator meters of the raster, from the manifest. */
  boundsMercator: readonly [number, number, number, number];
  observers: AlpsObserver[];
  /** Cell settings object every contributor's parameter packer accepts. */
  cellSettings: {cellSize: readonly [number, number]; northEdge: number; southEdge: number};
  /** Returns the `[column, row]` pixel containing a longitude/latitude, or `null` outside. */
  getPixel: (longitude: number, latitude: number) => [number, number] | null;
  /** Returns `[longitude, latitude]` of a pixel center. */
  getLongitudeLatitude: (column: number, row: number) => [number, number];
  /** Normalized Mercator pixel to ground meters per pixel for a row (cosine of latitude). */
  getGroundCellSize: (row: number) => number;
};

function mercatorYFromLatitude(latitude: number): number {
  const radians = (latitude * Math.PI) / 180;
  return (WORLD_METERS / (2 * Math.PI)) * Math.log(Math.tan(Math.PI / 4 + radians / 2));
}

function latitudeFromMercatorY(meters: number): number {
  return (
    ((2 * Math.atan(Math.exp(meters / (WORLD_METERS / (2 * Math.PI)))) - Math.PI / 2) * 180) /
    Math.PI
  );
}

/** Reads the Terrarium PNG of the alps-dem dataset as packed RGBA8 words plus georeferencing. */
export async function loadAlpsGrid(
  dataset: LoadedDataset,
  signal?: AbortSignal
): Promise<AlpsGrid> {
  const raster = dataset.raster;
  if (!raster) throw new Error('alps-dem has no raster');
  const spec = raster.spec as Record<string, unknown>;
  const boundsMercator = spec.boundsMercator as [number, number, number, number];
  const mercatorCellSize = spec.cellSizeMercatorM as number;
  const bytes = await fetchBytes(getDataFileUrl(dataset.info.id, String(spec.file)), signal);
  const bitmap = await createImageBitmap(new Blob([bytes], {type: 'image/png'}), {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none'
  });
  const {width, height} = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('2d', {willReadFrequently: true});
  if (!context) throw new Error('2D canvas unavailable for raster decoding');
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const pixels = context.getImageData(0, 0, width, height).data;
  const packedWords = new Uint32Array(pixels.buffer.slice(0));
  const [west, south, east, north] = raster.bounds;
  const origin: [number, number] = [(west + east) / 2, (south + north) / 2];
  const projection = new LocalMetricProjection(origin);
  const [minX, minY] = projection.project(west, south);
  const [maxX, maxY] = projection.project(east, north);
  const northEdge = 0.5 - boundsMercator[3] / WORLD_METERS;
  const southEdge = 0.5 - boundsMercator[1] / WORLD_METERS;
  const centerLatitude = latitudeFromMercatorY((boundsMercator[1] + boundsMercator[3]) / 2);
  const properties = dataset.properties as {
    observers?: Record<string, {lon: number; lat: number; elevationM: number}>;
  };
  const observers = Object.entries(properties.observers ?? {}).map(([name, value]) => ({
    name,
    longitude: value.lon,
    latitude: value.lat,
    elevationMeters: value.elevationM
  }));
  const getGroundCellSize = (row: number): number => {
    const y = boundsMercator[3] - (row + 0.5) * mercatorCellSize;
    return mercatorCellSize * Math.cos((latitudeFromMercatorY(y) * Math.PI) / 180);
  };
  return {
    width,
    height,
    pixelCount: width * height,
    packedWords,
    cpuElevation: raster.values as Float32Array,
    bounds: [minX, minY, maxX, maxY],
    origin,
    projection,
    mercatorCellSize,
    northEdge,
    southEdge,
    groundCellSize: mercatorCellSize * Math.cos((centerLatitude * Math.PI) / 180),
    boundsMercator,
    observers,
    cellSettings: {cellSize: [mercatorCellSize, mercatorCellSize], northEdge, southEdge},
    getPixel(longitude, latitude) {
      const x = ((longitude + 180) / 360) * WORLD_METERS - WORLD_METERS / 2;
      const y = mercatorYFromLatitude(latitude);
      const column = Math.floor((x - boundsMercator[0]) / mercatorCellSize);
      const row = Math.floor((boundsMercator[3] - y) / mercatorCellSize);
      if (column < 0 || row < 0 || column >= width || row >= height) return null;
      return [column, row];
    },
    getLongitudeLatitude(column, row) {
      const x = boundsMercator[0] + (column + 0.5) * mercatorCellSize;
      const y = boundsMercator[3] - (row + 0.5) * mercatorCellSize;
      return [((x + WORLD_METERS / 2) / WORLD_METERS) * 360 - 180, latitudeFromMercatorY(y)];
    },
    getGroundCellSize
  };
}
