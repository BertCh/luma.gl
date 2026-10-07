// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared helpers of the hydrology chapter: terrain grids built from the shipped DEM datasets
 * (Grand Canyon in Web Mercator, Dixie fire in UTM), their display frame in local ground meters,
 * and the small WGSL display kernels that turn contributor outputs into drawable rasters.
 */

import type {GraphDataView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {LoadedDataset} from '../../data/catalog';
import {decodeRasterImage, fetchBytes, getDataFileUrl} from '../../data/loaders';
import {addKernelPass} from '../../engine/mode-kernels';
import {LocalMetricProjection} from '../../engine/projection';

/** Equatorial circumference of the Web Mercator sphere in meters. */
const WEB_MERCATOR_CIRCUMFERENCE = 2 * Math.PI * 6378137;

/** A terrain raster plus everything needed to analyse it and to draw it in local meters. */
export type TerrainGrid = {
  id: string;
  width: number;
  height: number;
  cellCount: number;
  /** Elevation in meters, row 0 at the north edge. */
  elevation: Float32Array;
  /** Optional land cover class per cell (ESA WorldCover codes), row 0 north. */
  landCover: Uint8Array | null;
  /** How contributors read `cellSize`: equatorial Web Mercator meters or plain ground meters. */
  cellSizeMode: 'uniform' | 'web-mercator';
  /** `[x, y]` cell size in the units of `cellSizeMode`. */
  cellSize: [number, number];
  /** Normalized Web Mercator y of the north and south edge (0 when `uniform`). */
  northEdge: number;
  southEdge: number;
  /** Nominal ground meters per cell (centre of the window). */
  groundCellSize: [number, number];
  /** `[longitude, latitude]` origin of the display frame. */
  origin: [number, number];
  projection: LocalMetricProjection;
  /** `[minX, minY, maxX, maxY]` of the raster in local ground meters around `origin`. */
  bounds: [number, number, number, number];
  /** Display meters per cell. */
  displayCellSize: [number, number];
  /** `[west, south, east, north]` degrees, for camera framing. */
  bbox: [number, number, number, number];
};

/** Average of `stride` x `stride` blocks; `stride` 1 returns the input. */
export function decimateRaster(
  values: Float32Array,
  width: number,
  height: number,
  stride: number
): {values: Float32Array; width: number; height: number} {
  if (stride <= 1) return {values, width, height};
  const outputWidth = Math.floor(width / stride);
  const outputHeight = Math.floor(height / stride);
  const output = new Float32Array(outputWidth * outputHeight);
  const area = stride * stride;
  for (let row = 0; row < outputHeight; row++) {
    for (let column = 0; column < outputWidth; column++) {
      let sum = 0;
      for (let dRow = 0; dRow < stride; dRow++) {
        const base = (row * stride + dRow) * width + column * stride;
        for (let dColumn = 0; dColumn < stride; dColumn++) sum += values[base + dColumn];
      }
      output[row * outputWidth + column] = sum / area;
    }
  }
  return {values: output, width: outputWidth, height: outputHeight};
}

/**
 * Grand Canyon grid. The Terrarium tiles are in Web Mercator, so contributors use
 * `cellSizeMode: 'web-mercator'` with the north and south edges of the window; ground spacing is
 * then evaluated per row (a 0.35 % effect over this window, but exact by construction).
 *
 * @param stride Block-average factor: 1 keeps the 15 m grid, 2 gives 31 m.
 */
export function createCanyonGrid(dataset: LoadedDataset, stride: number): TerrainGrid {
  const raster = dataset.raster;
  if (!raster) throw new Error('grand-canyon-dem has no raster');
  const spec = raster.spec as unknown as {
    boundsMercator: [number, number, number, number];
    cellSizeMercatorM: number;
    cellSizeGroundM: number;
  };
  const reduced = decimateRaster(
    raster.values as Float32Array,
    raster.width,
    raster.height,
    stride
  );
  const origin = dataset.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const [west, south, east, north] = raster.bounds;
  const [minX, minY] = projection.project(west, south);
  const [maxX, maxY] = projection.project(east, north);
  // Row 0 is the north edge; normalized Web Mercator y grows southward from 0 at the top.
  const northEdge = 0.5 - spec.boundsMercator[3] / WEB_MERCATOR_CIRCUMFERENCE;
  const southEdge = 0.5 - spec.boundsMercator[1] / WEB_MERCATOR_CIRCUMFERENCE;
  return {
    id: 'grand-canyon-dem',
    width: reduced.width,
    height: reduced.height,
    cellCount: reduced.width * reduced.height,
    elevation: reduced.values,
    landCover: null,
    cellSizeMode: 'web-mercator',
    cellSize: [spec.cellSizeMercatorM * stride, spec.cellSizeMercatorM * stride],
    northEdge,
    southEdge,
    groundCellSize: [spec.cellSizeGroundM * stride, spec.cellSizeGroundM * stride],
    origin,
    projection,
    bounds: [minX, minY, maxX, maxY],
    displayCellSize: [(maxX - minX) / reduced.width, (maxY - minY) / reduced.height],
    bbox: raster.bounds
  };
}

/** Inverse UTM (Krueger series, WGS84) to `[longitude, latitude]` degrees. */
export function utmToLngLat(easting: number, northing: number, zone: number): [number, number] {
  const a = 6378137;
  const flattening = 1 / 298.257223563;
  const k0 = 0.9996;
  const n = flattening / (2 - flattening);
  const rectifyingRadius = (a / (1 + n)) * (1 + (n * n) / 4 + n ** 4 / 64);
  const beta = [
    n / 2 - (2 * n * n) / 3 + (37 * n ** 3) / 96,
    (n * n) / 48 + n ** 3 / 15,
    (17 * n ** 3) / 480
  ];
  const delta = [
    2 * n - (2 * n * n) / 3 - 2 * n ** 3,
    (7 * n * n) / 3 - (8 * n ** 3) / 5,
    (56 * n ** 3) / 15
  ];
  const xi = northing / (k0 * rectifyingRadius);
  const eta = (easting - 500000) / (k0 * rectifyingRadius);
  let xiPrime = xi;
  let etaPrime = eta;
  for (let order = 1; order <= 3; order++) {
    xiPrime -= beta[order - 1] * Math.sin(2 * order * xi) * Math.cosh(2 * order * eta);
    etaPrime -= beta[order - 1] * Math.cos(2 * order * xi) * Math.sinh(2 * order * eta);
  }
  const chi = Math.asin(Math.sin(xiPrime) / Math.cosh(etaPrime));
  let latitude = chi;
  for (let order = 1; order <= 3; order++) latitude += delta[order - 1] * Math.sin(2 * order * chi);
  const centralMeridian = (zone - 1) * 6 - 180 + 3;
  const longitude =
    centralMeridian + (Math.atan2(Math.sinh(etaPrime), Math.cos(xiPrime)) * 180) / Math.PI;
  return [longitude, (latitude * 180) / Math.PI];
}

/**
 * Dixie fire grid: 20 m UTM 10N cells with an ESA WorldCover raster on the same grid. The raster is
 * drawn as an axis-aligned square around the true window centre; UTM grid north differs from true
 * north by about 1.3 degrees here, so the far corners can sit up to ~170 m from the basemap.
 */
export async function createDixieGrid(
  dataset: LoadedDataset,
  signal?: AbortSignal
): Promise<TerrainGrid> {
  const raster = dataset.raster;
  if (!raster) throw new Error('dixie-fire has no raster');
  const manifest = dataset.manifest as unknown as {
    rasters: {worldcover: {file: string; encoding: 'uint8-classes'}};
    raster: {boundsProjected: [number, number, number, number]; cellSizeM: number};
  };
  const bytes = await fetchBytes(
    getDataFileUrl('dixie-fire', manifest.rasters.worldcover.file),
    signal
  );
  const landCover = await decodeRasterImage(bytes, 'uint8-classes');
  const [minE, minN, maxE, maxN] = manifest.raster.boundsProjected;
  const origin = utmToLngLat((minE + maxE) / 2, (minN + maxN) / 2, 10);
  const projection = new LocalMetricProjection(origin);
  const halfWidth = (maxE - minE) / 2;
  const halfHeight = (maxN - minN) / 2;
  const cellSize = manifest.raster.cellSizeM;
  return {
    id: 'dixie-fire',
    width: raster.width,
    height: raster.height,
    cellCount: raster.width * raster.height,
    elevation: raster.values as Float32Array,
    landCover: landCover.values as Uint8Array,
    cellSizeMode: 'uniform',
    cellSize: [cellSize, cellSize],
    northEdge: 0,
    southEdge: 0,
    groundCellSize: [cellSize, cellSize],
    origin,
    projection,
    bounds: [-halfWidth, -halfHeight, halfWidth, halfHeight],
    displayCellSize: [(2 * halfWidth) / raster.width, (2 * halfHeight) / raster.height],
    bbox: raster.bounds
  };
}

/** Raster cell under a longitude/latitude, or `-1` outside the grid. */
export function getCellAt(grid: TerrainGrid, longitude: number, latitude: number): number {
  const [x, y] = grid.projection.project(longitude, latitude);
  const column = Math.floor((x - grid.bounds[0]) / grid.displayCellSize[0]);
  const row = Math.floor((grid.bounds[3] - y) / grid.displayCellSize[1]);
  if (column < 0 || column >= grid.width || row < 0 || row >= grid.height) return -1;
  return row * grid.width + column;
}

/** Same as {@link getCellAt} but clamps to the grid. */
export function getCellAtClamped(grid: TerrainGrid, longitude: number, latitude: number): number {
  const [x, y] = grid.projection.project(longitude, latitude);
  const column = Math.min(
    grid.width - 1,
    Math.max(0, Math.floor((x - grid.bounds[0]) / grid.displayCellSize[0]))
  );
  const row = Math.min(
    grid.height - 1,
    Math.max(0, Math.floor((grid.bounds[3] - y) / grid.displayCellSize[1]))
  );
  return row * grid.width + column;
}

/** Cell centre in local display meters (y up). */
export function getCellCenter(grid: TerrainGrid, cell: number): [number, number] {
  return [
    grid.bounds[0] + ((cell % grid.width) + 0.5) * grid.displayCellSize[0],
    grid.bounds[3] - (Math.floor(cell / grid.width) + 0.5) * grid.displayCellSize[1]
  ];
}

/** Cell centre as `[longitude, latitude]`. */
export function getCellLngLat(grid: TerrainGrid, cell: number): [number, number] {
  const [x, y] = getCellCenter(grid, cell);
  return grid.projection.unproject(x, y);
}

/** Settings prefix `{cellSize, northEdge, southEdge}` every raster grid contributor reads. */
export function getGridSettings(grid: TerrainGrid) {
  return {cellSize: grid.cellSize, northEdge: grid.northEdge, southEdge: grid.southEdge};
}

/** Flips a north-first raster to south-first (row 0 at the minimum y). */
export function flipRows<T extends Float32Array | Uint32Array | Uint8Array>(
  values: T,
  width: number,
  height: number
): T {
  const flipped = new (values.constructor as new (length: number) => T)(values.length);
  for (let row = 0; row < height; row++) {
    flipped.set(values.subarray(row * width, (row + 1) * width), (height - 1 - row) * width);
  }
  return flipped;
}

/** Formats square meters as square kilometers with a precision that suits the size. */
export function formatKilometers(squareMeters: number): string {
  const squareKilometers = squareMeters / 1e6;
  if (squareKilometers >= 100) return `${squareKilometers.toFixed(0)} km²`;
  if (squareKilometers >= 1) return `${squareKilometers.toFixed(1)} km²`;
  return `${squareKilometers.toFixed(2)} km²`;
}

/** Formats a count with thousands separators. */
export function formatInteger(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/** Formats a duration in minutes as `h:mm` for readouts. */
export function formatDuration(minutes: number): string {
  if (!Number.isFinite(minutes)) return 'unreached';
  const whole = Math.round(minutes);
  const hours = Math.floor(whole / 60);
  const remainder = whole % 60;
  return hours > 0 ? `${hours} h ${String(remainder).padStart(2, '0')} min` : `${remainder} min`;
}

// -------------------------------------------------------------------------------------------------
// Display kernels
// -------------------------------------------------------------------------------------------------

/** Display sentinel for NaN or infinite cells; hidden with `discardAtOrBelow`. */
export const HIDDEN_VALUE = -9999;

const NAN_DECLARATIONS = /* wgsl */ `
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn isInfiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u; }
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`;

/** `output = log10(max(value * scale, 1e-6))` with {@link HIDDEN_VALUE} for NaN and infinity. */
export function addLogDisplayPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    input: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
    useLog: boolean;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'source', view: props.input, type: 'f32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let value = source[sourceOffset + index];
  let invalid = isNaNValue(value) || isInfiniteValue(value);
  ${props.useLog ? 'let shown = log2(max(value, 0.000001)) * 0.30102999566;' : 'let shown = value;'}
  display[displayOffset + index] = select(shown, ${HIDDEN_VALUE.toFixed(1)}, invalid);`
  });
}

/** `depth = filled - elevation`, 0 where either is NaN: how much each cell was raised by the fill. */
export function addFillDepthPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    elevation: GraphDataView<'float32'>;
    filled: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'filled', view: props.filled, type: 'f32', access: 'read'},
      {name: 'depth', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let value = filled[filledOffset + index] - elevation[elevationOffset + index];
  depth[depthOffset + index] = select(value, 0.0, isNaNValue(value));`
  });
}

/** Formats a number as a WGSL `f32` literal. */
export function getFloatLiteral(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/**
 * One segment per cell from its centre to its D8 receiver's centre (display meters, y up), plus
 * four alpha weights selecting the segments of each Strahler width class, so one layer per class
 * draws stream lines of order-dependent width.
 */
export function addStreamSegmentPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    cellCount: number;
    grid: TerrainGrid;
    directions: GraphDataView<'uint32'>;
    streamOrder: GraphDataView<'uint32'>;
    segments: GraphDataView<'float32'>;
    widths: readonly GraphDataView<'float32'>[];
  }
): void {
  const {grid} = props;
  addKernelPass(graph, {
    id: 'stream-segments',
    bindings: [
      {name: 'directions', view: props.directions, type: 'u32', access: 'read'},
      {name: 'orders', view: props.streamOrder, type: 'u32', access: 'read'},
      {name: 'segments', view: props.segments, type: 'f32', access: 'read_write'},
      ...props.widths.map((view, index) => ({
        name: `width${index}`,
        view,
        type: 'f32' as const,
        access: 'read_write' as const
      }))
    ],
    invocationCount: props.cellCount,
    declarations: `
const GRID_WIDTH: u32 = ${grid.width}u;
const MINIMUM_X: f32 = ${getFloatLiteral(grid.bounds[0])};
const MAXIMUM_Y: f32 = ${getFloatLiteral(grid.bounds[3])};
const CELL_X: f32 = ${getFloatLiteral(grid.displayCellSize[0])};
const CELL_Y: f32 = ${getFloatLiteral(grid.displayCellSize[1])};`,
    body: `let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  let code = directions[directionsOffset + index];
  var columnStep = 0.0;
  var rowStep = 0.0;
  if (code == 1u) { columnStep = 1.0; }
  else if (code == 2u) { columnStep = 1.0; rowStep = 1.0; }
  else if (code == 4u) { rowStep = 1.0; }
  else if (code == 8u) { columnStep = -1.0; rowStep = 1.0; }
  else if (code == 16u) { columnStep = -1.0; }
  else if (code == 32u) { columnStep = -1.0; rowStep = -1.0; }
  else if (code == 64u) { rowStep = -1.0; }
  else if (code == 128u) { columnStep = 1.0; rowStep = -1.0; }
  let startX = MINIMUM_X + (f32(column) + 0.5) * CELL_X;
  let startY = MAXIMUM_Y - (f32(row) + 0.5) * CELL_Y;
  let base = segmentsOffset + 4u * index;
  segments[base] = startX;
  segments[base + 1u] = startY;
  segments[base + 2u] = startX + columnStep * CELL_X;
  segments[base + 3u] = startY - rowStep * CELL_Y;
  let order = orders[ordersOffset + index];
  let isStream = order > 0u && order != 0xffffffffu;
  width0[width0Offset + index] = select(0.0, 1.0, isStream && order == 1u);
  width1[width1Offset + index] = select(0.0, 1.0, isStream && order == 2u);
  width2[width2Offset + index] = select(0.0, 1.0, isStream && order == 3u);
  width3[width3Offset + index] = select(0.0, 1.0, isStream && order >= 4u);`
  });
}

/**
 * Turns a cell-index path (destination first) into `float32` segments `x0, y0, x1, y1` in display
 * meters. Segment `i` joins `ids[i]` to `ids[min(i + 1, count - 1)]`; rows at or after `count` are
 * NaN.
 */
export function addPathSegmentsPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: TerrainGrid;
    capacity: number;
    ids: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    segments: GraphDataView<'float32'>;
  }
): void {
  const {grid} = props;
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'ids', view: props.ids, type: 'u32', access: 'read'},
      {name: 'count', view: props.count, type: 'u32', access: 'read'},
      {name: 'segments', view: props.segments, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.capacity,
    declarations: /* wgsl */ `
const GRID_WIDTH: u32 = ${grid.width}u;
const ORIGIN_X: f32 = ${getFloatLiteral(grid.bounds[0])};
const ORIGIN_Y: f32 = ${getFloatLiteral(grid.bounds[3])};
const CELL_WIDTH: f32 = ${getFloatLiteral(grid.displayCellSize[0])};
const CELL_HEIGHT: f32 = ${getFloatLiteral(grid.displayCellSize[1])};
${NAN_DECLARATIONS}
fn getCellCenter(cell: u32) -> vec2<f32> {
  return vec2<f32>(
    ORIGIN_X + (f32(cell % GRID_WIDTH) + 0.5) * CELL_WIDTH,
    ORIGIN_Y - (f32(cell / GRID_WIDTH) + 0.5) * CELL_HEIGHT
  );
}`,
    body: `let pathCount = count[countOffset];
  let base = segmentsOffset + index * 4u;
  if (index < pathCount) {
    let start = getCellCenter(ids[idsOffset + index]);
    let end = getCellCenter(ids[idsOffset + min(index + 1u, pathCount - 1u)]);
    segments[base] = start.x;
    segments[base + 1u] = start.y;
    segments[base + 2u] = end.x;
    segments[base + 3u] = end.y;
  } else {
    let nan = getQuietNaN();
    segments[base] = nan;
    segments[base + 1u] = nan;
    segments[base + 2u] = nan;
    segments[base + 3u] = nan;
  }`
  });
}

export {NAN_DECLARATIONS};
