// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, validatePackedView, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';

/** Throws unless `view` is a packed single view of `format` with at least `rowCount` rows. @internal */
export function validateRasterSamplingView(
  id: string,
  name: string,
  view: GraphDataView | GraphVectorView | undefined,
  format: 'float32' | 'uint32' | 'float32x2',
  rowCount: number
): void {
  if (!view) {
    throw new Error(`${id} needs ${name}`);
  }
  if (view instanceof GraphVectorView) {
    throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
  }
  validatePackedView(view, [format], `${id} ${name}`);
  if (view.length < rowCount) {
    throw new Error(`${id} ${name} must hold at least ${rowCount} rows`);
  }
}

/** Throws unless the raster description is valid. @internal */
export function validateRasterDescription(
  id: string,
  width: number,
  height: number,
  noDataValue: number | undefined
): void {
  for (const [name, value] of [
    ['width', width],
    ['height', height]
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new Error(`${id} ${name} must be a positive integer`);
    }
  }
  if (width * height > 0x7fffffff) {
    throw new Error(`${id} width * height must not exceed 2^31 - 1`);
  }
  if (noDataValue !== undefined && !Number.isFinite(noDataValue)) {
    throw new Error(`${id} noDataValue must be finite (NaN cells are always nodata)`);
  }
}

/** Options of {@link getRasterSamplingWGSL}. @internal */
export type RasterSamplingWGSLOptions = {
  width: number;
  height: number;
  noDataValue?: number;
  hasValidity: boolean;
};

/**
 * Shared WGSL sampling helper. Requires bindings `raster` (f32), `params` (f32) and, with
 * `hasValidity`, `validity` (u32). Provides `getNaN()`, `isNaNValue(v)`, `isFiniteValue(v)` and
 * `sampleRaster(x, y) -> f32` (NaN outside the extent or for nodata results).
 *
 * @internal
 */
export function getRasterSamplingWGSL(options: RasterSamplingWGSLOptions): string {
  const {width, height, noDataValue, hasValidity} = options;
  return /* wgsl */ `
const RASTER_WIDTH: i32 = ${width};
const RASTER_HEIGHT: i32 = ${height};
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }

// Cell value with every nodata convention folded into NaN.
fn fetchCell(column: i32, row: i32) -> f32 {
  let cell = u32(row * RASTER_WIDTH + column);
  var value = raster[rasterOffset + cell];
  ${hasValidity ? 'if (validity[validityOffset + cell] == 0u) { value = getNaN(); }' : ''}
  ${noDataValue !== undefined ? `if (value == ${getWGSLFloatLiteral(noDataValue)}) { value = getNaN(); }` : ''}
  return value;
}
fn fetchClamped(column: i32, row: i32) -> f32 {
  return fetchCell(clamp(column, 0, RASTER_WIDTH - 1), clamp(row, 0, RASTER_HEIGHT - 1));
}

fn getCubicWeights(t: f32) -> vec4<f32> {
  return vec4<f32>(
    t * (-0.5 + t * (1.0 - 0.5 * t)),
    1.0 + t * t * (-2.5 + 1.5 * t),
    t * (0.5 + t * (2.0 - 1.5 * t)),
    t * t * (-0.5 + 0.5 * t)
  );
}

// A sample participates only when both of its axis weights are nonzero.
fn sampleBilinear(iu: i32, iv: i32, fx: f32, fy: f32, renormalize: bool) -> f32 {
  var wx = array<f32, 2>(1.0 - fx, fx);
  var wy = array<f32, 2>(1.0 - fy, fy);
  var sum = 0.0;
  var weightSum = 0.0;
  var hasNoData = false;
  for (var j = 0; j < 2; j++) {
    for (var i = 0; i < 2; i++) {
      if (wx[i] != 0.0 && wy[j] != 0.0) {
        let value = fetchClamped(iu + i, iv + j);
        let weight = wx[i] * wy[j];
        if (isNaNValue(value)) {
          hasNoData = true;
        } else {
          sum += weight * value;
          weightSum += weight;
        }
      }
    }
  }
  if (!hasNoData) {
    return sum;
  }
  if (renormalize && weightSum > 0.0) {
    return sum / weightSum;
  }
  return getNaN();
}

fn sampleBicubic(iu: i32, iv: i32, fx: f32, fy: f32, renormalize: bool) -> f32 {
  let cx = getCubicWeights(fx);
  let cy = getCubicWeights(fy);
  var wx = array<f32, 4>(cx.x, cx.y, cx.z, cx.w);
  var wy = array<f32, 4>(cy.x, cy.y, cy.z, cy.w);
  var cells = array<f32, 16>();
  var hasNoData = false;
  for (var j = 0; j < 4; j++) {
    for (var i = 0; i < 4; i++) {
      if (wx[i] != 0.0 && wy[j] != 0.0) {
        let value = fetchClamped(iu - 1 + i, iv - 1 + j);
        cells[j * 4 + i] = value;
        if (isNaNValue(value)) {
          hasNoData = true;
        }
      }
    }
  }
  if (hasNoData) {
    return sampleBilinear(iu, iv, fx, fy, renormalize);
  }
  var result = 0.0;
  for (var j = 0; j < 4; j++) {
    if (wy[j] != 0.0) {
      var row = 0.0;
      for (var i = 0; i < 4; i++) {
        if (wx[i] != 0.0) {
          row += wx[i] * cells[j * 4 + i];
        }
      }
      result += wy[j] * row;
    }
  }
  return result;
}

fn sampleRaster(x: f32, y: f32) -> f32 {
  if (isNaNValue(x) || isNaNValue(y)) {
    return getNaN();
  }
  let minX = params[paramsOffset];
  let minY = params[paramsOffset + 1u];
  if (x < minX || x > params[paramsOffset + 2u] || y < minY || y > params[paramsOffset + 3u]) {
    return getNaN();
  }
  let method = u32(params[paramsOffset + 8u]);
  let renormalize = params[paramsOffset + 9u] != 0.0;
  let tx = (x - minX) * params[paramsOffset + 6u];
  let ty = (y - minY) * params[paramsOffset + 7u];
  if (method == 0u) {
    return fetchClamped(i32(floor(tx)), i32(floor(ty)));
  }
  let u = tx - 0.5;
  let v = ty - 0.5;
  let u0 = floor(u);
  let v0 = floor(v);
  if (method == 1u) {
    return sampleBilinear(i32(u0), i32(v0), u - u0, v - v0, renormalize);
  }
  return sampleBicubic(i32(u0), i32(v0), u - u0, v - v0, renormalize);
}`;
}
