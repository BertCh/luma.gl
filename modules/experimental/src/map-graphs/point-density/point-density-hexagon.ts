// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const SQRT3 = Math.sqrt(3);

/**
 * WGSL helpers shared by the hexagon-keys kernel and by render shaders that draw hexagon bins.
 *
 * Hexagons are pointy-top with odd rows shifted right by half a cell (odd-r offset layout, the
 * d3-hexbin orientation). Hexagon `(0, 0)` is centered at the origin passed to each function.
 */
export const GPU_POINT_DENSITY_HEXAGON_WGSL = /* wgsl */ `
const POINT_DENSITY_SQRT3: f32 = 1.7320508075688772;

// Odd-r offset [column, row] of the pointy-top hexagon containing (x, y).
fn getPointDensityHexagonCell(x: f32, y: f32, originX: f32, originY: f32, radius: f32) -> vec2<i32> {
  let localX = (x - originX) / radius;
  let localY = (y - originY) / radius;
  let q = (POINT_DENSITY_SQRT3 / 3.0) * localX - localY / 3.0;
  let r = (2.0 / 3.0) * localY;
  let s = -q - r;
  var roundedQ = round(q);
  var roundedR = round(r);
  let roundedS = round(s);
  let deltaQ = abs(roundedQ - q);
  let deltaR = abs(roundedR - r);
  let deltaS = abs(roundedS - s);
  if (deltaQ > deltaR && deltaQ > deltaS) {
    roundedQ = -roundedR - roundedS;
  } else if (deltaR > deltaS) {
    roundedR = -roundedQ - roundedS;
  }
  let axialQ = i32(roundedQ);
  let axialR = i32(roundedR);
  return vec2<i32>(axialQ + (axialR - (axialR & 1)) / 2, axialR);
}

// Center of odd-r hexagon (column, row).
fn getPointDensityHexagonCenter(column: i32, row: i32, originX: f32, originY: f32, radius: f32) -> vec2<f32> {
  return vec2<f32>(
    originX + radius * POINT_DENSITY_SQRT3 * (f32(column) + 0.5 * f32(row & 1)),
    originY + 1.5 * radius * f32(row)
  );
}
`;

/** CPU mirror of the WGSL `getPointDensityHexagonCell`: odd-r `[column, row]` containing `(x, y)`. */
export function getGPUPointDensityHexagonCell(
  x: number,
  y: number,
  originX: number,
  originY: number,
  radius: number
): [number, number] {
  const localX = (x - originX) / radius;
  const localY = (y - originY) / radius;
  const q = (SQRT3 / 3) * localX - localY / 3;
  const r = (2 / 3) * localY;
  const s = -q - r;
  let roundedQ = roundHalfToEven(q);
  let roundedR = roundHalfToEven(r);
  const roundedS = roundHalfToEven(s);
  const deltaQ = Math.abs(roundedQ - q);
  const deltaR = Math.abs(roundedR - r);
  const deltaS = Math.abs(roundedS - s);
  if (deltaQ > deltaR && deltaQ > deltaS) {
    roundedQ = -roundedR - roundedS;
  } else if (deltaR > deltaS) {
    roundedR = -roundedQ - roundedS;
  }
  const row = roundedR + 0;
  // Odd-r conversion with floor division, matching the WGSL two's-complement expression.
  const column = roundedQ + Math.trunc((row - (row & 1)) / 2) + 0;
  return [column, row];
}

/** Center of odd-r hexagon `(column, row)`. */
export function getGPUPointDensityHexagonCenter(
  column: number,
  row: number,
  originX: number,
  originY: number,
  radius: number
): [number, number] {
  return [originX + radius * SQRT3 * (column + 0.5 * (row & 1)), originY + 1.5 * radius * row];
}

/**
 * Returns the smallest `[columns, rows]` lattice that holds every point inside `bounds` when the
 * lattice origin is `(minX, minY)`.
 *
 * @throws For a non-positive or non-finite radius, or invalid bounds.
 */
export function getGPUPointDensityHexagonGridSize(
  bounds: readonly [number, number, number, number],
  radius: number
): [number, number] {
  const [minX, minY, maxX, maxY] = bounds;
  if (!Number.isFinite(radius) || radius <= 0) {
    throw new Error('Hexagon radius must be positive and finite');
  }
  if (!bounds.every(Number.isFinite) || minX > maxX || minY > maxY) {
    throw new Error('Hexagon bounds must be finite [minX, minY, maxX, maxY]');
  }
  return [
    Math.ceil((maxX - minX) / (SQRT3 * radius)) + 1,
    Math.ceil((maxY - minY) / (1.5 * radius)) + 1
  ];
}

/** Rounds like WGSL `round()`: halves go to the even neighbor. */
function roundHalfToEven(value: number): number {
  const rounded = Math.round(value);
  return Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}
