// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared terrain preparation of the visibility chapter scenes (`viewshed`, `horizon`, `summits`).
 *
 * The alps-dem raster is Terrarium in Web Mercator. `LoadedDataset.raster` already decodes it to
 * float32 elevations on the CPU; this module optionally box-averages it by an integer stride
 * (coarser grids make the O(pixels x range) analyses interactive), and records every number a
 * contributor needs about the Mercator geometry:
 *
 * - contributors with a Web Mercator cell model (`GPUTerrainSummits`, `GPUTerrainPeakSnap`,
 *   `GPUTerrainDerivatives`, `GPUTerrainHorizon`) take `cellSize` (equatorial Mercator meters) with
 *   `northEdge` / `southEdge` (normalized Mercator y) and scale each row by the cosine of latitude;
 * - contributors with a planar cell model (`GPUTerrainViewshed`, `GPUTerrainLineOfSight`,
 *   `GPUTerrainCumulativeViewshed`) take one uniform ground cell size, here the ground size of the
 *   central row (the cosine of latitude varies by 0.1 percent across this 13.6 km window);
 * - `GPUPointHorizonProfile` and `GPUPointHorizonVisibility` have a true `'web-mercator'`
 *   projection: great-circle rays from `worldPixelSize` and the world pixel row of the window.
 */

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {Buffer} from '@luma.gl/core';
import type {LoadedDataset} from '../../data/catalog';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {LocalMetricProjection} from '../../engine/projection';
import type {SpatialAnalysisResources} from '../../engine/resources';

/** Circumference of the Web Mercator world in meters (2 pi R, R = 6378137). */
export const WORLD_METERS = 40075016.68557849;
const EARTH_RADIUS = 6378137;

/** One named place used as an observer or a peak catalogue entry. */
export type NamedPlace = {
  name: string;
  longitude: number;
  latitude: number;
  /** Published elevation in meters (catalogue value, not the DEM value). */
  elevationMeters: number;
};

/** The alps-dem prepared at one analysis resolution. */
export type AlpsTerrain = {
  /** Analysis grid size in pixels. */
  width: number;
  height: number;
  pixelCount: number;
  /** Source pixels averaged per analysis pixel along each axis (1 = full resolution). */
  stride: number;
  /** Elevation in meters, row 0 is the north edge. */
  elevation: Float32Array;
  /** 1 for every pixel (the DEM has no nodata); the contributors still want the view. */
  validity: Uint32Array;
  /** Lowest and highest elevation. */
  elevationRange: readonly [number, number];
  /** `[west, south, east, north]` degrees of the raster's outer edges, from the manifest. */
  lngLatBounds: readonly [number, number, number, number];
  /** `[longitude, latitude]` layer coordinate origin (the bbox center). */
  origin: [number, number];
  projection: LocalMetricProjection;
  /** `[minX, minY, maxX, maxY]` meters around `origin`: the outer pixel edges, for layers. */
  bounds: [number, number, number, number];
  /** Layer meters per pixel `[x, y]`. */
  layerCellSize: readonly [number, number];
  /** Equatorial Web Mercator meters per analysis pixel. */
  mercatorCellSize: number;
  /** Normalized Web Mercator y of the top of row 0 and of the bottom of the last row. */
  northEdge: number;
  southEdge: number;
  /** Ground meters per analysis pixel at the central row (uniform cell model). */
  groundCellSize: number;
  /** World size in analysis pixels (`512 * 2^13 / stride`) and the world row of the top edge. */
  worldPixelSize: number;
  originY: number;
  /** Cell settings every Mercator-aware parameter packer accepts. */
  mercatorCellSettings: {
    cellSize: readonly [number, number];
    northEdge: number;
    southEdge: number;
  };
  /** Observers recorded in the dataset manifest. */
  observers: NamedPlace[];
  /** Pixel-center index position `[column, row]` (fractional) of a longitude and latitude. */
  getPixel: (longitude: number, latitude: number) => [number, number];
  /** `[longitude, latitude]` of a pixel-center index position. */
  getLongitudeLatitude: (column: number, row: number) => [number, number];
  /** Layer meters `[x, y]` of a pixel-center index position. */
  getMeters: (column: number, row: number) => [number, number];
  /** Pixel-center index position of layer meters, clamped to the grid. */
  getPixelFromMeters: (x: number, y: number) => [number, number];
  /** Ground meters per pixel at a row (the cosine of latitude scaled Mercator size). */
  getGroundCellSize: (row: number) => number;
  /** Bilinear elevation at a pixel-center index position (meters). */
  sampleElevation: (column: number, row: number) => number;
};

function getLatitude(mercatorY: number): number {
  return ((2 * Math.atan(Math.exp(mercatorY / EARTH_RADIUS)) - Math.PI / 2) * 180) / Math.PI;
}

function getMercatorY(latitude: number): number {
  const radians = (latitude * Math.PI) / 180;
  return EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + radians / 2));
}

/**
 * Prepares the alps-dem for the terrain contributors.
 *
 * @param dataset The loaded `alps-dem` dataset.
 * @param stride Integer downsampling factor; the grid is box-averaged by `stride x stride`.
 */
export function prepareAlpsTerrain(dataset: LoadedDataset, stride = 1): AlpsTerrain {
  const raster = dataset.raster;
  if (!raster) throw new Error('alps-dem has no raster');
  const spec = raster.spec as Record<string, unknown>;
  const boundsMercator = spec.boundsMercator as [number, number, number, number];
  const sourceCellSize = spec.cellSizeMercatorM as number;
  const tile = spec.webMercatorTile as {zoom: number; tileSize: number; originPixel: number[]};
  const sourceWidth = raster.width;
  const sourceHeight = raster.height;
  const width = Math.floor(sourceWidth / stride);
  const height = Math.floor(sourceHeight / stride);
  const source = raster.values as Float32Array;
  let elevation: Float32Array;
  if (stride === 1) {
    elevation = Float32Array.from(source);
  } else {
    elevation = new Float32Array(width * height);
    const weight = 1 / (stride * stride);
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        let sum = 0;
        for (let dy = 0; dy < stride; dy++) {
          const base = (row * stride + dy) * sourceWidth + column * stride;
          for (let dx = 0; dx < stride; dx++) sum += source[base + dx];
        }
        elevation[row * width + column] = sum * weight;
      }
    }
  }
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < elevation.length; index++) {
    minimum = Math.min(minimum, elevation[index]);
    maximum = Math.max(maximum, elevation[index]);
  }
  const validity = new Uint32Array(width * height).fill(1);
  const [west, south, east, north] = raster.bounds;
  const origin: [number, number] = [(west + east) / 2, (south + north) / 2];
  const projection = new LocalMetricProjection(origin);
  const [minX, minY] = projection.project(west, south);
  const [maxX, maxY] = projection.project(east, north);
  const bounds: [number, number, number, number] = [minX, minY, maxX, maxY];
  const layerCellSize: [number, number] = [(maxX - minX) / width, (maxY - minY) / height];
  const mercatorCellSize = sourceCellSize * stride;
  const northEdge = 0.5 - boundsMercator[3] / WORLD_METERS;
  const southEdge = 0.5 - boundsMercator[1] / WORLD_METERS;
  const centerLatitude = getLatitude((boundsMercator[1] + boundsMercator[3]) / 2);
  const groundCellSize = mercatorCellSize * Math.cos((centerLatitude * Math.PI) / 180);
  const worldPixelSize = (tile.tileSize * 2 ** tile.zoom) / stride;
  const originY = tile.originPixel[1] / stride;

  const properties = dataset.properties as {
    observers?: Record<string, {lon: number; lat: number; elevationM: number}>;
  };
  const observers = Object.entries(properties.observers ?? {}).map(([name, value]) => ({
    name,
    longitude: value.lon,
    latitude: value.lat,
    elevationMeters: value.elevationM
  }));

  const getPixel = (longitude: number, latitude: number): [number, number] => {
    const x = (EARTH_RADIUS * longitude * Math.PI) / 180;
    const y = getMercatorY(latitude);
    return [
      (x - boundsMercator[0]) / mercatorCellSize - 0.5,
      (boundsMercator[3] - y) / mercatorCellSize - 0.5
    ];
  };
  const getLongitudeLatitude = (column: number, row: number): [number, number] => {
    const x = boundsMercator[0] + (column + 0.5) * mercatorCellSize;
    const y = boundsMercator[3] - (row + 0.5) * mercatorCellSize;
    return [(x / EARTH_RADIUS) * (180 / Math.PI), getLatitude(y)];
  };
  const getMeters = (column: number, row: number): [number, number] => [
    minX + (column + 0.5) * layerCellSize[0],
    maxY - (row + 0.5) * layerCellSize[1]
  ];
  const getPixelFromMeters = (x: number, y: number): [number, number] => [
    Math.min(Math.max((x - minX) / layerCellSize[0] - 0.5, 0), width - 1),
    Math.min(Math.max((maxY - y) / layerCellSize[1] - 0.5, 0), height - 1)
  ];
  const getGroundCellSize = (row: number): number =>
    mercatorCellSize *
    Math.cos((getLatitude(boundsMercator[3] - (row + 0.5) * mercatorCellSize) * Math.PI) / 180);
  const sampleElevation = (column: number, row: number): number => {
    const baseColumn = Math.min(Math.max(Math.floor(column), 0), width - 2);
    const baseRow = Math.min(Math.max(Math.floor(row), 0), height - 2);
    const fractionX = Math.min(Math.max(column - baseColumn, 0), 1);
    const fractionY = Math.min(Math.max(row - baseRow, 0), 1);
    const index = baseRow * width + baseColumn;
    const top = elevation[index] * (1 - fractionX) + elevation[index + 1] * fractionX;
    const bottom =
      elevation[index + width] * (1 - fractionX) + elevation[index + width + 1] * fractionX;
    return top * (1 - fractionY) + bottom * fractionY;
  };

  return {
    width,
    height,
    pixelCount: width * height,
    stride,
    elevation,
    validity,
    elevationRange: [minimum, maximum],
    lngLatBounds: [west, south, east, north],
    origin,
    projection,
    bounds,
    layerCellSize,
    mercatorCellSize,
    northEdge,
    southEdge,
    groundCellSize,
    worldPixelSize,
    originY,
    mercatorCellSettings: {
      cellSize: [mercatorCellSize, mercatorCellSize],
      northEdge,
      southEdge
    },
    observers,
    getPixel,
    getLongitudeLatitude,
    getMeters,
    getPixelFromMeters,
    getGroundCellSize,
    sampleElevation
  };
}

/** Named observers and viewpoints near Zermatt (catalogue coordinates, rounded to 0.001 degree). */
export const ZERMATT_VIEWPOINTS: readonly NamedPlace[] = [
  {name: 'Gornergrat', longitude: 7.7843, latitude: 45.9832, elevationMeters: 3089},
  {name: 'Klein Matterhorn', longitude: 7.73, latitude: 45.9384, elevationMeters: 3883},
  {name: 'Riffelhorn', longitude: 7.7588, latitude: 45.9814, elevationMeters: 2927},
  {name: 'Theodulhorn', longitude: 7.7084, latitude: 45.9493, elevationMeters: 3468},
  {name: 'Zermatt', longitude: 7.7491, latitude: 46.0207, elevationMeters: 1608},
  {name: 'Matterhorn', longitude: 7.6586, latitude: 45.9766, elevationMeters: 4478}
];

/**
 * A small peak catalogue of the window with published elevations. Longitudes and latitudes are
 * deliberately only catalogue-precise (three decimals, about 100 m), like OpenStreetMap
 * `natural=peak` nodes: `GPUTerrainPeakSnap` moves each onto the DEM summit.
 */
export const ALPS_PEAK_CATALOGUE: readonly NamedPlace[] = [
  {name: 'Matterhorn', longitude: 7.659, latitude: 45.977, elevationMeters: 4478},
  {name: 'Breithorn', longitude: 7.746, latitude: 45.941, elevationMeters: 4164},
  {name: 'Pollux', longitude: 7.786, latitude: 45.928, elevationMeters: 4092},
  {name: 'Obergabelhorn', longitude: 7.668, latitude: 46.039, elevationMeters: 4063},
  {name: 'Wellenkuppe', longitude: 7.678, latitude: 46.042, elevationMeters: 3903},
  {name: 'Klein Matterhorn', longitude: 7.73, latitude: 45.938, elevationMeters: 3883},
  {name: 'Theodulhorn', longitude: 7.708, latitude: 45.949, elevationMeters: 3468},
  {name: 'Riffelhorn', longitude: 7.759, latitude: 45.981, elevationMeters: 2927}
];

/** Elevation band over two imported buffers, as every terrain contributor takes it. */
export function createElevationBand<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  prefix: string,
  values: Buffer,
  validity: Buffer,
  pixelCount: number
) {
  return {
    id: `${prefix}-elevation`,
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, `${prefix}-values`, values, 'float32', pixelCount)
    },
    validity: importGraphBuffer(graph, `${prefix}-validity`, validity, 'uint32', pixelCount)
  };
}

/** Uploads the elevation and validity of a terrain as storage buffers. */
export function createTerrainBuffers(
  resources: SpatialAnalysisResources,
  terrain: AlpsTerrain,
  name: string
): {elevation: Buffer; validity: Buffer} {
  return {
    elevation: resources.createBuffer(`${name}-elevation`, terrain.elevation),
    validity: resources.createBuffer(`${name}-validity`, terrain.validity)
  };
}

/** Reads a typed array view of bytes returned by a ring ticket or `readAsync`. */
export function toFloat32(
  bytes: Uint8Array | ArrayBuffer,
  wordCount: number
): Float32Array<ArrayBuffer> {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return new Float32Array(view.slice(0, wordCount * 4).buffer as ArrayBuffer);
}

/** Reads uint32 words from bytes returned by a ring ticket or `readAsync`. */
export function toUint32(
  bytes: Uint8Array | ArrayBuffer,
  wordCount: number
): Uint32Array<ArrayBuffer> {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return new Uint32Array(view.slice(0, wordCount * 4).buffer as ArrayBuffer);
}

/** Returns the pixel-center position index views are keyed on; exported for typed helpers. */
export type FloatView = GraphDataView<'float32'>;

/** Formats a distance in meters as meters below 1 km and kilometers above. */
export function formatDistance(meters: number): string {
  return meters < 1000 ? `${meters.toFixed(0)} m` : `${(meters / 1000).toFixed(2)} km`;
}

/** Compass azimuth (degrees clockwise from north) of `[dx, dy]` where y grows north. */
export function getAzimuthDegrees(dx: number, dy: number): number {
  return ((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360;
}

const COMPASS = [
  'N',
  'NNE',
  'NE',
  'ENE',
  'E',
  'ESE',
  'SE',
  'SSE',
  'S',
  'SSW',
  'SW',
  'WSW',
  'W',
  'WNW',
  'NW',
  'NNW'
];

/** Compass point name of an azimuth. */
export function getCompassName(azimuthDegrees: number): string {
  return COMPASS[Math.round((((azimuthDegrees % 360) + 360) % 360) / 22.5) % 16];
}
