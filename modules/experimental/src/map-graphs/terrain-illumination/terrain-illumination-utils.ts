// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePackedView, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWGSLFloatLiteral} from '../map-graph-kernels';

/**
 * How terrain-illumination kernels convert settings cell sizes into ground meters per pixel.
 *
 * Same meaning as `GPUTerrainCellSizeMode` in `terrain-analysis`: `'uniform'` cell sizes are
 * meters; `'web-mercator'` cell sizes are equatorial Web Mercator meters scaled by the row's
 * latitude; `'geographic'` cell sizes are degrees scaled by meters per degree.
 */
export type GPUTerrainIlluminationCellSizeMode = 'uniform' | 'web-mercator' | 'geographic';

/** Meters per degree of latitude on the WGS84 equatorial sphere used by terrain kernels. @internal */
export const TERRAIN_ILLUMINATION_METERS_PER_DEGREE = 111319.49079327357;

/** WGSL constants shared by terrain-illumination kernels. @internal */
export const TERRAIN_ILLUMINATION_WGSL_CONSTANTS = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
const METERS_PER_DEGREE: f32 = ${TERRAIN_ILLUMINATION_METERS_PER_DEGREE};
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn getNaN(index: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (index & 0u)); }`;

/** Settings slots read by {@link getTerrainIlluminationGroundCellWGSL}. @internal */
export type TerrainIlluminationCellSizeSlots = {
  /** Name of the `f32` settings binding. */
  settingsName: string;
  /** Index of cell size x; cell size y is the next slot. */
  cellSizeIndex: number;
  /** Index of the north edge (top of row 0); the south edge is the next slot. */
  northEdgeIndex: number;
};

/**
 * Returns WGSL `fn getGroundCellSize(row: u32) -> vec2<f32>` for one cell size mode.
 *
 * `HEIGHT` must be declared by the caller. The latitude of a row is interpolated linearly between
 * the north and south edges at the row center, exactly as in `GPUTerrainDerivatives`.
 *
 * @internal
 */
export function getTerrainIlluminationGroundCellWGSL(
  mode: GPUTerrainIlluminationCellSizeMode,
  slots: TerrainIlluminationCellSizeSlots
): string {
  const settings = slots.settingsName;
  const offset = `${settings}Offset`;
  const cellSize = `vec2<f32>(${settings}[${offset} + ${slots.cellSizeIndex}u], ${settings}[${offset} + ${slots.cellSizeIndex + 1}u])`;
  if (mode === 'uniform') {
    return `fn getGroundCellSize(row: u32) -> vec2<f32> { return ${cellSize}; }`;
  }
  const edge = `mix(${settings}[${offset} + ${slots.northEdgeIndex}u], ${settings}[${offset} + ${slots.northEdgeIndex + 1}u], (f32(row) + 0.5) / f32(HEIGHT))`;
  const scaled =
    mode === 'web-mercator'
      ? // cos(latitude) = 1 / cosh(PI * (1 - 2y)) for normalized Web Mercator y.
        'cellSize / cosh(PI * (1.0 - 2.0 * edge))'
      : 'cellSize * METERS_PER_DEGREE * vec2<f32>(cos(edge * DEGREES_TO_RADIANS), 1.0)';
  return `fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = ${cellSize};
  let edge = ${edge};
  return ${scaled};
}`;
}

/**
 * CPU twin of {@link getTerrainIlluminationGroundCellWGSL} for oracles, in float64.
 *
 * @returns `[x, y]` ground meters per pixel for `row`.
 * @internal
 */
export function getTerrainIlluminationGroundCellSize(
  mode: GPUTerrainIlluminationCellSizeMode,
  cellSize: readonly [number, number],
  northEdge: number,
  southEdge: number,
  row: number,
  height: number
): [number, number] {
  if (mode === 'uniform') {
    return [cellSize[0], cellSize[1]];
  }
  const fraction = (row + 0.5) / height;
  const edge = northEdge + (southEdge - northEdge) * fraction;
  if (mode === 'web-mercator') {
    const scale = 1 / Math.cosh(Math.PI * (1 - 2 * edge));
    return [cellSize[0] * scale, cellSize[1] * scale];
  }
  return [
    cellSize[0] * TERRAIN_ILLUMINATION_METERS_PER_DEGREE * Math.cos((edge * Math.PI) / 180),
    cellSize[1] * TERRAIN_ILLUMINATION_METERS_PER_DEGREE
  ];
}

/** Throws unless `mode` is a known cell size mode. @internal */
export function validateTerrainIlluminationCellSizeMode(id: string, mode: string): void {
  if (!['uniform', 'web-mercator', 'geographic'].includes(mode)) {
    throw new Error(`${id} cellSizeMode must be uniform, web-mercator, or geographic`);
  }
}

/** Throws unless `rowDirection` is `'south'` or `'north'`. @internal */
export function validateTerrainIlluminationRowDirection(id: string, rowDirection: string): void {
  if (!['south', 'north'].includes(rowDirection)) {
    throw new Error(`${id} rowDirection must be south or north`);
  }
}

/** Throws unless `view` is a packed view of `format` with exactly `length` rows. @internal */
export function validateTerrainIlluminationView(
  id: string,
  name: string,
  view: GraphDataView | undefined,
  format: 'float32' | 'uint32',
  length: number
): void {
  if (!view) {
    return;
  }
  validatePackedView(view, [format], `${id} ${name}`);
  if (view.length !== length) {
    throw new Error(`${id} ${name} must contain ${length} ${format} values`);
  }
}

/**
 * Throws when a single storage binding would exceed the device's `maxStorageBufferBindingSize`.
 *
 * @internal
 */
export function validateTerrainIlluminationBindingSize(
  id: string,
  name: string,
  byteLength: number,
  maxStorageBufferBindingSize: number
): void {
  if (byteLength > maxStorageBufferBindingSize) {
    throw new Error(
      `${id} ${name} needs ${byteLength} bytes in one binding, above the device limit ${maxStorageBufferBindingSize}; use fewer directions or a smaller tile`
    );
  }
}

/**
 * Storage format of horizon-angle maps: `'float32'` stores one angle in degrees per value;
 * `'unorm16'` packs two 16-bit codes per `uint32` word, see `encodeGPUTerrainHorizonUnorm16`.
 */
export type GPUTerrainHorizonFormat = 'float32' | 'unorm16';

/** Degrees per unorm16 code step: `180 / 65534`. @internal */
export const TERRAIN_HORIZON_UNORM16_STEP_DEGREES = 180 / 65534;

/** Codes per degree used by the unorm16 encoder, rounded to float32: `65534 / 180`. @internal */
export const TERRAIN_HORIZON_UNORM16_SCALE = Math.fround(65534 / 180);

/** Throws unless `format` is `'float32'` or `'unorm16'`. @internal */
export function validateTerrainHorizonFormat(id: string, format: string): void {
  if (!['float32', 'unorm16'].includes(format)) {
    throw new Error(`${id} horizonFormat must be float32 or unorm16`);
  }
}

/**
 * Number of storage values (`float32` angles or packed `uint32` words) of a horizon map.
 *
 * @internal
 */
export function getTerrainHorizonStorageLength(
  format: GPUTerrainHorizonFormat,
  pixelCount: number,
  directionCount: number
): number {
  const elementCount = pixelCount * directionCount;
  return format === 'unorm16' ? Math.ceil(elementCount / 2) : elementCount;
}

/**
 * Throws unless `view` stores a horizon map of `format`: `float32` with one value per element, or
 * `uint32` with `ceil(pixelCount * directionCount / 2)` words for `'unorm16'`.
 *
 * @internal
 */
export function validateTerrainHorizonView(
  id: string,
  name: string,
  view: GraphDataView | undefined,
  format: GPUTerrainHorizonFormat,
  pixelCount: number,
  directionCount: number
): void {
  validateTerrainIlluminationView(
    id,
    name,
    view,
    format === 'unorm16' ? 'uint32' : 'float32',
    getTerrainHorizonStorageLength(format, pixelCount, directionCount)
  );
}

/**
 * Returns WGSL `fn readHorizonAngle(element: u32) -> f32` reading horizon angles in degrees from
 * the storage binding `bindingName` (element `pixel * directionCount + sector`).
 *
 * `'float32'` bindings are `array<f32>`; `'unorm16'` bindings are `array<u32>` holding two codes
 * per word (low half for even elements). Code 0 reads as NaN. The generated source needs
 * `getNaN` from {@link TERRAIN_ILLUMINATION_WGSL_CONSTANTS} and the `${bindingName}Offset`
 * constant that `createMapGraphKernelNode` declares.
 *
 * @internal
 */
export function getTerrainHorizonReadWGSL(
  format: GPUTerrainHorizonFormat,
  bindingName: string
): string {
  if (format === 'float32') {
    return `fn readHorizonAngle(element: u32) -> f32 {
  return ${bindingName}[${bindingName}Offset + element];
}`;
  }
  return `fn readHorizonAngle(element: u32) -> f32 {
  let word = ${bindingName}[${bindingName}Offset + (element >> 1u)];
  let code = (word >> ((element & 1u) * 16u)) & 0xffffu;
  if (code == 0u) { return getNaN(element); }
  return f32(code - 1u) * ${getWGSLFloatLiteral(TERRAIN_HORIZON_UNORM16_STEP_DEGREES)} - 90.0;
}`;
}
