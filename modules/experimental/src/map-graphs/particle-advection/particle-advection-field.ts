// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Private vector-field sampling shared by the particle-advection WGSL and its CPU oracle.
 *
 * A field is a packed row-major `(u, v)` raster of `width * height` cells placed by
 * `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. Sampling is manual
 * bilinear interpolation between cell centres (no filterable float texture needed), clamped to the
 * outermost centres inside the raster bounds. A position outside
 * `[origin, origin + cellSize * (width, height))`, a NaN position, or a NaN corner makes the sample
 * invalid.
 */

/**
 * Returns WGSL defining `sampleField(position: vec2<f32>, extent: vec4<f32>) -> vec3<f32>`, which
 * returns `(u, v, 1)` for a valid sample and `(0, 0, 0)` otherwise.
 *
 * @param bindingName Name of the `array<f32>` storage binding holding the packed field.
 * @param width Field width in cells.
 * @param height Field height in cells.
 */
export function getFieldSamplingWGSL(bindingName: string, width: number, height: number): string {
  return /* wgsl */ `
const FIELD_WIDTH: u32 = ${width}u;
const FIELD_HEIGHT: u32 = ${height}u;

fn isNanFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

fn readFieldVelocity(column: u32, row: u32) -> vec2<f32> {
  let base = ${bindingName}Offset + 2u * (row * FIELD_WIDTH + column);
  return vec2<f32>(${bindingName}[base], ${bindingName}[base + 1u]);
}

fn isNanVelocity(value: vec2<f32>) -> bool {
  return isNanFloat(value.x) || isNanFloat(value.y);
}

fn sampleField(position: vec2<f32>, extent: vec4<f32>) -> vec3<f32> {
  let local = (position - extent.xy) / extent.zw;
  if (isNanVelocity(local)) {
    return vec3<f32>(0.0);
  }
  if (local.x < 0.0 || local.y < 0.0 || local.x >= f32(FIELD_WIDTH) || local.y >= f32(FIELD_HEIGHT)) {
    return vec3<f32>(0.0);
  }
  let grid = clamp(
    local - vec2<f32>(0.5),
    vec2<f32>(0.0),
    vec2<f32>(f32(FIELD_WIDTH - 1u), f32(FIELD_HEIGHT - 1u))
  );
  let column0 = min(u32(grid.x), FIELD_WIDTH - 1u);
  let row0 = min(u32(grid.y), FIELD_HEIGHT - 1u);
  let column1 = min(column0 + 1u, FIELD_WIDTH - 1u);
  let row1 = min(row0 + 1u, FIELD_HEIGHT - 1u);
  let fraction = grid - vec2<f32>(f32(column0), f32(row0));
  let v00 = readFieldVelocity(column0, row0);
  let v10 = readFieldVelocity(column1, row0);
  let v01 = readFieldVelocity(column0, row1);
  let v11 = readFieldVelocity(column1, row1);
  if (isNanVelocity(v00) || isNanVelocity(v10) || isNanVelocity(v01) || isNanVelocity(v11)) {
    return vec3<f32>(0.0);
  }
  let bottom = v00 * (1.0 - fraction.x) + v10 * fraction.x;
  let top = v01 * (1.0 - fraction.x) + v11 * fraction.x;
  return vec3<f32>(bottom * (1.0 - fraction.y) + top * fraction.y, 1.0);
}
`;
}

/** CPU description of a packed vector field. */
export type FieldRaster = {
  /** Packed `(u, v)` rows, `width * height * 2` floats. */
  velocities: Float32Array;
  /** Width in cells. */
  width: number;
  /** Height in cells. */
  height: number;
};

/** Result of {@link sampleFieldOnCPU}: `[u, v]` or `undefined` when the sample is invalid. */
export type FieldSample = [number, number] | undefined;

const fround = Math.fround;

/**
 * CPU mirror of WGSL `sampleField`, rounding every intermediate to f32.
 *
 * @param field Packed field raster.
 * @param x Sample x.
 * @param y Sample y.
 * @param extent `[originX, originY, cellWidth, cellHeight]`.
 */
export function sampleFieldOnCPU(
  field: FieldRaster,
  x: number,
  y: number,
  extent: ArrayLike<number>
): FieldSample {
  const {velocities, width, height} = field;
  const localX = fround(fround(x - extent[0]) / extent[2]);
  const localY = fround(fround(y - extent[1]) / extent[3]);
  if (Number.isNaN(localX) || Number.isNaN(localY)) {
    return undefined;
  }
  if (localX < 0 || localY < 0 || localX >= width || localY >= height) {
    return undefined;
  }
  const gridX = Math.min(Math.max(fround(localX - 0.5), 0), width - 1);
  const gridY = Math.min(Math.max(fround(localY - 0.5), 0), height - 1);
  const column0 = Math.min(Math.floor(gridX), width - 1);
  const row0 = Math.min(Math.floor(gridY), height - 1);
  const column1 = Math.min(column0 + 1, width - 1);
  const row1 = Math.min(row0 + 1, height - 1);
  const fractionX = fround(gridX - column0);
  const fractionY = fround(gridY - row0);
  const read = (column: number, row: number): [number, number] => {
    const base = 2 * (row * width + column);
    return [velocities[base], velocities[base + 1]];
  };
  const corners = [
    read(column0, row0),
    read(column1, row0),
    read(column0, row1),
    read(column1, row1)
  ];
  if (corners.some(([u, v]) => Number.isNaN(u) || Number.isNaN(v))) {
    return undefined;
  }
  const [v00, v10, v01, v11] = corners;
  const inverseX = fround(1 - fractionX);
  const inverseY = fround(1 - fractionY);
  const result: [number, number] = [0, 0];
  for (let component = 0; component < 2; component++) {
    const bottom = fround(fround(v00[component] * inverseX) + fround(v10[component] * fractionX));
    const top = fround(fround(v01[component] * inverseX) + fround(v11[component] * fractionX));
    result[component] = fround(fround(bottom * inverseY) + fround(top * fractionY));
  }
  return result;
}
