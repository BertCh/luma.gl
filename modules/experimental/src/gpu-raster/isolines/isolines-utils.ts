// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, validatePackedView, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';

/**
 * Throws unless `view` is one packed view of `format` with at least `rowCount` rows.
 *
 * @internal
 */
export function validateIsolinesView(
  id: string,
  name: string,
  view: GraphDataView | GraphVectorView | undefined,
  format: 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x2',
  rowCount: number
): void {
  if (!view) {
    return;
  }
  if (view instanceof GraphVectorView) {
    throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
  }
  validatePackedView(view, [format], `${id} ${name}`);
  if (view.length < rowCount) {
    throw new Error(`${id} ${name} must hold at least ${rowCount} rows`);
  }
}

/**
 * Throws when two views of `outputs` share a buffer, or an output shares a buffer with `inputs`.
 *
 * @internal
 */
export function validateIsolinesAliasing(
  id: string,
  outputs: readonly (GraphDataView | undefined)[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
  if (new Set(outputBuffers).size !== outputBuffers.length) {
    throw new Error(`${id} outputs must not share buffers`);
  }
  const inputBuffers = new Set(inputs.filter(view => view !== undefined).map(view => view.buffer));
  if (outputBuffers.some(buffer => inputBuffers.has(buffer))) {
    throw new Error(`${id} outputs must not share buffers with inputs`);
  }
}

/**
 * WGSL `getNaN`, `Corners` and `loadCorners(cx, cy)` for a raster binding `raster` with optional
 * `validity`. A cell is nodata when it is NaN, equals the optional sentinel, or has a zero
 * validity word; infinities are values. Cell `(cx, cy)` reads samples v0 `(cx, cy)`, v1 `(cx + 1,
 * cy)`, v2 `(cx + 1, cy + 1)`, v3 `(cx, cy + 1)`; `valid` is false when any corner is nodata.
 *
 * @internal
 */
export function getIsolinesCornersWGSL(
  hasValidity: boolean,
  noDataValue: number | undefined
): string {
  return /* wgsl */ `
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: NaN is nodata, and infinities are kept as values.
  var isValid = (bitcast<u32>(sample) & 0x7fffffffu) <= 0x7f800000u;
  ${noDataValue !== undefined ? `isValid = isValid && sample != ${getWGSLFloatLiteral(noDataValue)};` : ''}
  ${hasValidity ? 'isValid = isValid && validity[validityOffset + cell] != 0u;' : ''}
  return isValid;
}

struct Corners {
  valid: bool,
  v: vec4<f32>
}

fn loadCorners(cx: u32, cy: u32) -> Corners {
  var v0 = 0.0;
  var v1 = 0.0;
  var v2 = 0.0;
  var v3 = 0.0;
  let ok0 = readCell(cy * WIDTH + cx, &v0);
  let ok1 = readCell(cy * WIDTH + cx + 1u, &v1);
  let ok2 = readCell((cy + 1u) * WIDTH + cx + 1u, &v2);
  let ok3 = readCell((cy + 1u) * WIDTH + cx, &v3);
  var corners: Corners;
  corners.valid = ok0 && ok1 && ok2 && ok3;
  corners.v = vec4<f32>(v0, v1, v2, v3);
  return corners;
}

// Bit i is set when corner i (v0..v3) is high, i.e. value >= level.
fn getHighMask(v: vec4<f32>, level: f32) -> u32 {
  return select(0u, 1u, v.x >= level) | select(0u, 2u, v.y >= level) |
    select(0u, 4u, v.z >= level) | select(0u, 8u, v.w >= level);
}

// Counter-clockwise boundary walk v0 -> v1 -> v2 -> v3 -> v0: edge k runs from corner k to corner
// (k + 1) % 4, i.e. edges e0 bottom, e1 right, e2 top, e3 left. An exit is high -> low (the
// segment starts there, keeping high on its left); an entry is low -> high (the segment ends).
fn getRotatedMask(mask: u32) -> u32 {
  return ((mask >> 1u) | (mask << 3u)) & 15u;
}
fn getExitMask(mask: u32) -> u32 { return mask & ~getRotatedMask(mask) & 15u; }
fn getEntryMask(mask: u32) -> u32 { return ~mask & getRotatedMask(mask) & 15u; }

fn getEdgeId(k: u32, cx: u32, cy: u32) -> u32 {
  let horizontalCount = HEIGHT * (WIDTH - 1u);
  if (k == 0u) { return cy * (WIDTH - 1u) + cx; }
  if (k == 2u) { return (cy + 1u) * (WIDTH - 1u) + cx; }
  if (k == 3u) { return horizontalCount + cy * WIDTH + cx; }
  return horizontalCount + cy * WIDTH + cx + 1u;
}`;
}
