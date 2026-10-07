// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared helpers of the raster chapter (B16): the UTM display frame of the Sentinel-2 grids, a
 * raster layer that draws a UTM grid rotated onto the true-north basemap, small formatters and the
 * categorical colors of the severity classes.
 *
 * The Sentinel-2 datasets live on EPSG:32610 grids. UTM grid north differs from true north by the
 * meridian convergence (about 1.3 degrees here), which is 170 m at the edge of a 15 km window, so
 * an axis-aligned quad would visibly slide off the roads. Contributors run on the native grid
 * (no resampling, so the results match the CPU references exactly); only the display rotates.
 */

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import type {LoadedDataset} from '../../data/catalog';
import {decodeRasterImage, fetchBytes, getDataFileUrl} from '../../data/loaders';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import type {SpatialAnalysisRasterLayerProps} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';

export {PATCH_COLORS, SEVERITY_COLORS, SEVERITY_NAMES} from './b16-colors';

/** Inverse UTM (Snyder series, WGS84) to `[longitude, latitude]` degrees. */
export function utmToLngLat(easting: number, northing: number, zone: number): [number, number] {
  const a = 6378137;
  const flattening = 1 / 298.257223563;
  const k0 = 0.9996;
  const e2 = flattening * (2 - flattening);
  const ep2 = e2 / (1 - e2);
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const x = easting - 500000;
  const m = northing / k0;
  const mu = m / (a * (1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 * e2 * e2) / 256));
  const phi1 =
    mu +
    ((3 * e1) / 2 - (27 * e1 ** 3) / 32) * Math.sin(2 * mu) +
    ((21 * e1 * e1) / 16 - (55 * e1 ** 4) / 32) * Math.sin(4 * mu) +
    ((151 * e1 ** 3) / 96) * Math.sin(6 * mu) +
    ((1097 * e1 ** 4) / 512) * Math.sin(8 * mu);
  const sin1 = Math.sin(phi1);
  const cos1 = Math.cos(phi1);
  const tan1 = Math.tan(phi1);
  const n1 = a / Math.sqrt(1 - e2 * sin1 * sin1);
  const t1 = tan1 * tan1;
  const c1 = ep2 * cos1 * cos1;
  const r1 = (a * (1 - e2)) / (1 - e2 * sin1 * sin1) ** 1.5;
  const d = x / (n1 * k0);
  const latitude =
    phi1 -
    ((n1 * tan1) / r1) *
      ((d * d) / 2 -
        ((5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ep2) * d ** 4) / 24 +
        ((61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ep2 - 3 * c1 * c1) * d ** 6) / 720);
  const longitudeOffset =
    (d -
      ((1 + 2 * t1 + c1) * d ** 3) / 6 +
      ((5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ep2 + 24 * t1 * t1) * d ** 5) / 120) /
    cos1;
  const centralMeridian = (zone - 1) * 6 - 180 + 3;
  return [centralMeridian + (longitudeOffset * 180) / Math.PI, (latitude * 180) / Math.PI];
}

/** A UTM grid and the affine transform that places it on the local-meters (deck) frame. */
export type UtmFrame = {
  columns: number;
  rows: number;
  /** Ground meters per cell. */
  cellSize: number;
  /** Half of the grid extent in meters. */
  halfWidth: number;
  halfHeight: number;
  /** `[longitude, latitude]` origin of the local frame (the layer `coordinateOrigin`). */
  origin: [number, number];
  /** Column-major 2x2: local UTM `[x, y]` (east, north, relative to the center) to deck meters. */
  rotation: readonly [number, number, number, number];
  /** Deck meters of the grid center. */
  translation: readonly [number, number];
  /** Grid cell `[column, row]` (fractional) of a longitude and latitude. */
  lngLatToCell: (longitude: number, latitude: number) => [number, number];
  /** Deck meters of local UTM `[x, y]` (east, north, relative to the grid center). */
  localToMeters: (x: number, y: number) => [number, number];
  /** Deck meters `[x, y]` of a grid cell center (`row` 0 is north). */
  cellToMeters: (column: number, row: number) => [number, number];
  /** Longitude and latitude of a grid cell center. */
  cellToLngLat: (column: number, row: number) => [number, number];
};

/**
 * Builds the display frame of a UTM raster: the center and the two axis vectors are mapped through
 * the true inverse projection, so rotation and scale are measured, not assumed.
 */
export function createUtmFrame(
  origin: readonly [number, number],
  spec: {
    boundsProjected: readonly [number, number, number, number];
    cellSizeM: number;
    width: number;
    height: number;
  },
  zone = 10
): UtmFrame {
  const [minE, minN, maxE, maxN] = spec.boundsProjected;
  const centerE = (minE + maxE) / 2;
  const centerN = (minN + maxN) / 2;
  const projection = new LocalMetricProjection(origin);
  const centerMeters = projection.project(...utmToLngLat(centerE, centerN, zone));
  const eastMeters = projection.project(...utmToLngLat(centerE + 1000, centerN, zone));
  const northMeters = projection.project(...utmToLngLat(centerE, centerN + 1000, zone));
  const ax = (eastMeters[0] - centerMeters[0]) / 1000;
  const ay = (eastMeters[1] - centerMeters[1]) / 1000;
  const bx = (northMeters[0] - centerMeters[0]) / 1000;
  const by = (northMeters[1] - centerMeters[1]) / 1000;
  const determinant = ax * by - bx * ay;
  const halfWidth = (spec.width * spec.cellSizeM) / 2;
  const halfHeight = (spec.height * spec.cellSizeM) / 2;
  const toMeters = (x: number, y: number): [number, number] => [
    centerMeters[0] + ax * x + bx * y,
    centerMeters[1] + ay * x + by * y
  ];
  return {
    columns: spec.width,
    rows: spec.height,
    cellSize: spec.cellSizeM,
    halfWidth,
    halfHeight,
    origin: [origin[0], origin[1]],
    rotation: [ax, ay, bx, by],
    translation: centerMeters,
    lngLatToCell(longitude, latitude) {
      const [mx, my] = projection.project(longitude, latitude);
      const dx = mx - centerMeters[0];
      const dy = my - centerMeters[1];
      const x = (by * dx - bx * dy) / determinant;
      const y = (-ay * dx + ax * dy) / determinant;
      return [(x + halfWidth) / spec.cellSizeM, (halfHeight - y) / spec.cellSizeM];
    },
    localToMeters: toMeters,
    cellToMeters(column, row) {
      return toMeters(
        (column + 0.5) * spec.cellSizeM - halfWidth,
        halfHeight - (row + 0.5) * spec.cellSizeM
      );
    },
    cellToLngLat(column, row) {
      const [mx, my] = toMeters(
        (column + 0.5) * spec.cellSizeM - halfWidth,
        halfHeight - (row + 0.5) * spec.cellSizeM
      );
      return projection.unproject(mx, my);
    }
  };
}

/** Props of {@link UtmRasterLayer}. */
export type UtmRasterLayerProps = Omit<SpatialAnalysisRasterLayerProps, 'bounds' | 'gridSize'> & {
  frame: UtmFrame;
  /** Cell values, one per grid cell, row 0 north. */
  values: Buffer;
};

/**
 * Draws a UTM grid rotated onto the basemap. It reuses the spatial-analysis raster layer (cell
 * lookup, ramps, categories, discards) and only patches the vertex stage so the quad corners go
 * through the frame's rotation and translation; the fragment stage keeps reading cells in the
 * native, unrotated grid coordinates.
 */
export class UtmRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'UtmRasterLayer';

  constructor(props: UtmRasterLayerProps & Record<string, unknown>) {
    const {frame} = props;
    super({
      rowOrigin: 'north',
      coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
      ...props,
      gridSize: [frame.columns, frame.rows],
      bounds: [-frame.halfWidth, -frame.halfHeight, frame.halfWidth, frame.halfHeight]
    } as never);
  }

  protected override getShaderSource(): string {
    const {frame} = this.props as unknown as UtmRasterLayerProps;
    const source = super.getShaderSource();
    const [a, b, c, d] = frame.rotation;
    const constants = `const UTM_ROTATION = mat2x2<f32>(vec2<f32>(${a.toFixed(9)}, ${b.toFixed(9)}), vec2<f32>(${c.toFixed(9)}, ${d.toFixed(9)}));
const UTM_TRANSLATION = vec2<f32>(${frame.translation[0].toFixed(4)}, ${frame.translation[1].toFixed(4)});
`;
    const anchor = '@group(0) @binding(auto) var<storage, read> rasterBounds';
    const projectCall = 'output.position = projectSpatialAnalysisPosition(worldPosition);';
    if (!source.includes(anchor) || !source.includes(projectCall)) {
      throw new Error('UtmRasterLayer: the raster shader changed; update the patch');
    }
    return source
      .replace(anchor, `${constants}${anchor}`)
      .replace(
        projectCall,
        'output.position = projectSpatialAnalysisPosition(UTM_ROTATION * worldPosition + UTM_TRANSLATION);'
      );
  }
}

/** Reads a little-endian uint16 band (`.bin`) of a dataset as float32 (digital numbers). */
export async function loadUint16Band(
  datasetId: string,
  file: string,
  signal?: AbortSignal
): Promise<Float32Array> {
  const bytes = await fetchBytes(getDataFileUrl(datasetId, file), signal);
  const raw = new Uint16Array(bytes);
  const values = new Float32Array(raw.length);
  for (let index = 0; index < raw.length; index++) values[index] = raw[index];
  return values;
}

/** Reads an 8-bit class PNG of a dataset as float32 class codes. */
export async function loadClassRaster(
  datasetId: string,
  file: string,
  signal?: AbortSignal
): Promise<{values: Float32Array; classes: Uint8Array}> {
  const bytes = await fetchBytes(getDataFileUrl(datasetId, file), signal);
  const decoded = await decodeRasterImage(bytes, 'uint8-classes');
  const classes = decoded.values as Uint8Array;
  return {values: Float32Array.from(classes), classes};
}

/** The Sentinel-2 bands and class rasters of the Dixie Fire dataset, on one 750 x 750 grid. */
export type DixieRasters = {
  width: number;
  height: number;
  cellCount: number;
  /** Reflectance times 10000 as float32, 0 where there is no data. */
  redBefore: Float32Array;
  nirBefore: Float32Array;
  swirBefore: Float32Array;
  redAfter: Float32Array;
  nirAfter: Float32Array;
  swirAfter: Float32Array;
  /** Scene classification of the after scene as float32 class codes. */
  sclAfter: Float32Array;
  /** ESA WorldCover codes (10, 20, ...) as float32 and as bytes. */
  worldCover: Float32Array;
  worldCoverClasses: Uint8Array;
  /** Terrarium elevation in meters. */
  elevation: Float32Array;
  frame: UtmFrame;
};

/** Loads the Dixie Fire bands in parallel. */
export async function loadDixieRasters(
  dataset: LoadedDataset,
  signal?: AbortSignal
): Promise<DixieRasters> {
  const id = dataset.info.id;
  const spec = (dataset.manifest as unknown as {rasters: unknown}).rasters as Record<
    string,
    {
      width: number;
      height: number;
      boundsProjected: [number, number, number, number];
      cellSizeM: number;
    }
  >;
  const grid = spec.dem;
  const [redBefore, nirBefore, swirBefore, redAfter, nirAfter, swirAfter, scl, worldCover] =
    await Promise.all([
      loadUint16Band(id, 'B04_before.bin', signal),
      loadUint16Band(id, 'B08_before.bin', signal),
      loadUint16Band(id, 'B12_before.bin', signal),
      loadUint16Band(id, 'B04_after.bin', signal),
      loadUint16Band(id, 'B08_after.bin', signal),
      loadUint16Band(id, 'B12_after.bin', signal),
      loadClassRaster(id, 'scl_after.png', signal),
      loadClassRaster(id, 'worldcover.png', signal)
    ]);
  const elevation = dataset.raster?.values;
  if (!elevation || !(elevation instanceof Float32Array)) {
    throw new Error('dixie-fire needs its Terrarium DEM');
  }
  return {
    width: grid.width,
    height: grid.height,
    cellCount: grid.width * grid.height,
    redBefore,
    nirBefore,
    swirBefore,
    redAfter,
    nirAfter,
    swirAfter,
    sclAfter: scl.values,
    worldCover: worldCover.values,
    worldCoverClasses: worldCover.classes,
    elevation,
    frame: createUtmFrame(dataset.defaultOrigin, grid)
  };
}

/** Reference statistics the data builder computed on the CPU (`properties.validation`). */
export type DixieReference = {
  meanDNBR: number;
  maxDNBR: number;
  moderateFraction: number;
  highFraction: number;
  lowFraction: number;
  meanDNDVI: number;
  meanNDVIBefore: number;
  meanNDVIAfter: number;
  meanNBRBefore: number;
  meanNBRAfter: number;
  burnedByCover: Record<string, number>;
};

/** Reads {@link DixieReference} from the dataset manifest. */
export function readDixieReference(dataset: LoadedDataset): DixieReference {
  const properties = dataset.properties as {
    validation: {
      meanDNBR: number;
      maxDNBR: number;
      fractionAbove: Record<string, number>;
      meanDNDVI: number;
      'burnedFractionByWorldCover(dNBR>0.27)': Record<string, number>;
    };
    sceneStats: Record<'before' | 'after', {meanNDVI: number; meanNBR: number}>;
  };
  const validation = properties.validation;
  const fractions = validation.fractionAbove;
  const stats = properties.sceneStats;
  return {
    meanDNBR: validation.meanDNBR,
    maxDNBR: validation.maxDNBR,
    lowFraction: fractions['dNBR>0.1 (low+)'],
    moderateFraction: fractions['dNBR>0.27 (moderate-low+)'],
    highFraction: fractions['dNBR>0.66 (high)'],
    meanDNDVI: validation.meanDNDVI,
    meanNDVIBefore: stats.before.meanNDVI,
    meanNDVIAfter: stats.after.meanNDVI,
    meanNBRBefore: stats.before.meanNBR,
    meanNBRAfter: stats.after.meanNBR,
    burnedByCover: validation['burnedFractionByWorldCover(dNBR>0.27)']
  };
}

/** Formats a number with fixed digits and a typographic minus. */
export function formatFixed(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value.toFixed(digits).replace('-', '−');
}

/** Formats a fraction as a percentage. */
export function formatPercent(fraction: number, digits = 1): string {
  return Number.isFinite(fraction) ? `${(fraction * 100).toFixed(digits)}%` : 'n/a';
}

/** Formats hectares with thousands separators. */
export function formatHectares(hectares: number): string {
  if (!Number.isFinite(hectares)) return 'n/a';
  return `${hectares.toLocaleString('en-US', {maximumFractionDigits: hectares < 100 ? 1 : 0})} ha`;
}

/**
 * Returns a function that makes typed views of application buffers in one graph. A graph rejects
 * the same physical buffer imported twice, so the import is made once per buffer and every later
 * view (any format or length) shares its handle.
 */
export function createGraphViewer(graph: GPUCommandGraph<void>) {
  const handles = new Map<Buffer, ReturnType<GPUCommandGraph<void>['importBuffer']>>();
  return <Format extends GPUVectorFormat>(
    name: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ): GraphDataView<Format> => {
    let handle = handles.get(buffer);
    if (!handle) {
      handle = graph.importBuffer(
        {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      handles.set(buffer, handle);
    }
    return graph.createDataView(handle, {
      format,
      length: length ?? Math.floor(buffer.byteLength / getGPUVectorFormatInfo(format).byteLength)
    });
  };
}
