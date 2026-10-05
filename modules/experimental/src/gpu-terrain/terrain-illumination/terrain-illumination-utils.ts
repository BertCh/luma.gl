// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePackedView, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  getTerrainGroundCellSize,
  TERRAIN_DEGREES_TO_RADIANS_WGSL,
  TERRAIN_METERS_PER_DEGREE_WGSL
} from '../terrain-grid-utils';

/** CPU twin of the terrain ground cell size WGSL, kept under its illumination name. @internal */
export const getTerrainIlluminationGroundCellSize = getTerrainGroundCellSize;

/** WGSL constants shared by terrain-illumination kernels. @internal */
export const TERRAIN_ILLUMINATION_WGSL_CONSTANTS = /* wgsl */ `
const PI: f32 = 3.141592653589793;
${TERRAIN_DEGREES_TO_RADIANS_WGSL}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${TERRAIN_METERS_PER_DEGREE_WGSL}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn getNaN(index: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (index & 0u)); }`;

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
 * constant that `createWGSLKernelNode` declares.
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

/**
 * Returns the WGSL shared by every kernel that turns a horizon map and a sun position into solar
 * visibility: `DIRECTION_COUNT`, `SECTOR_DEGREES`, `readHorizonAngle` (see
 * {@link getTerrainHorizonReadWGSL}), `fn getHorizonAtAzimuth(pixel: u32, azimuthDegrees: f32) -> f32`
 * and `fn getSolarDiskVisibility(altitudeDegrees: f32, radiusDegrees: f32, horizonAngle: f32) -> f32`.
 *
 * `getHorizonAtAzimuth` interpolates linearly between the two sectors bracketing the compass
 * azimuth, with wraparound. `getSolarDiskVisibility` is the area fraction of the solar disk above
 * the horizon: `1 - (acos(u) - u * sqrt(1 - u^2)) / PI` with
 * `u = clamp((altitude - horizon) / radius, -1, 1)`, a hard step when the radius is 0, and 0 when
 * the whole disk is below the astronomical horizon. Used by `GPUSolarShadowMask` and
 * `GPUSolarIrradiance`, so both agree exactly.
 *
 * Needs {@link TERRAIN_ILLUMINATION_WGSL_CONSTANTS} and the `${bindingName}Offset` constant.
 *
 * @internal
 */
export function getTerrainSolarVisibilityWGSL(
  format: GPUTerrainHorizonFormat,
  bindingName: string,
  directionCount: number
): string {
  return `const DIRECTION_COUNT: u32 = ${directionCount}u;
const SECTOR_DEGREES: f32 = ${getWGSLFloatLiteral(360 / directionCount)};
${getTerrainHorizonReadWGSL(format, bindingName)}
fn getHorizonAtAzimuth(pixel: u32, azimuthDegrees: f32) -> f32 {
  let wrappedAzimuth = azimuthDegrees - 360.0 * floor(azimuthDegrees / 360.0);
  let sectorPosition = wrappedAzimuth / SECTOR_DEGREES;
  let lowerSector = min(u32(floor(sectorPosition)), DIRECTION_COUNT - 1u);
  let upperSector = (lowerSector + 1u) % DIRECTION_COUNT;
  let blend = clamp(sectorPosition - f32(lowerSector), 0.0, 1.0);
  return mix(
    readHorizonAngle(pixel * DIRECTION_COUNT + lowerSector),
    readHorizonAngle(pixel * DIRECTION_COUNT + upperSector),
    blend
  );
}
fn getSolarDiskVisibility(altitudeDegrees: f32, radiusDegrees: f32, horizonAngle: f32) -> f32 {
  var visibility = 0.0;
  if (altitudeDegrees + radiusDegrees > 0.0) {
    if (radiusDegrees > 0.0) {
      let u = clamp((altitudeDegrees - horizonAngle) / radiusDegrees, -1.0, 1.0);
      visibility = clamp(1.0 - (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / PI, 0.0, 1.0);
    } else {
      visibility = select(0.0, 1.0, altitudeDegrees > horizonAngle);
    }
  }
  return visibility;
}`;
}
