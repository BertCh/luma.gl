// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Forward (longitude/latitude to key) WGSL for the quadkey, geohash and S2 families. The Quadbin
// helpers it builds on live in `../cell-aggregation/cell-keys`; H3 lives in `h3-index-wgsl`.

import {CELL_KEY_WGSL, QUADBIN_TILE_WGSL} from '../cell-aggregation/cell-keys';

/**
 * S2 curve tables: `kIJToPos[orientation][ij]` with `ij = (i << 1) | j`, derived from the S2
 * reference `kPosToIJ` (orientation 0: ij of positions 0..3 is 0, 1, 3, 2; 1: 0, 2, 3, 1; 2: 3, 2,
 * 0, 1; 3: 3, 1, 0, 2) and `kPosToOrientation` (swap mask 1, invert mask 2: 1, 0, 0, 3).
 *
 * @internal
 */
export const S2_POSITION_TO_IJ: readonly (readonly number[])[] = [
  [0, 1, 3, 2],
  [0, 2, 3, 1],
  [3, 2, 0, 1],
  [3, 1, 0, 2]
];
/** S2 `kPosToOrientation`. @internal */
export const S2_POSITION_TO_ORIENTATION: readonly number[] = [1, 0, 0, 3];

function getInverseTable(): number[] {
  const table: number[] = [];
  for (const row of S2_POSITION_TO_IJ) {
    const inverse = [0, 0, 0, 0];
    row.forEach((ij, position) => {
      inverse[ij] = position;
    });
    table.push(...inverse);
  }
  return table;
}

/**
 * Quadkey (Web Mercator tile) forward indexing. Concatenate after `CELL_KEY_WGSL`; it includes
 * `QUADBIN_TILE_WGSL`.
 *
 * Defines `cellIndexQuadkeyFromLngLat(lng, lat, zoom) -> vec2u(high, low)`. Tile column and row are
 * the integer-exact Quadbin functions (see `QUADBIN_TILE_WGSL`): longitude 180 wraps to column 0
 * and latitude is clipped to the Mercator limit 85.051129. The digits are the Morton interleave
 * of column and row, so the key is `zoom << 58 | morton`.
 *
 * @internal
 */
export const QUADKEY_INDEX_WGSL = /* wgsl */ `
${CELL_KEY_WGSL}
${QUADBIN_TILE_WGSL}
fn cellIndexQuadkeyFromLngLat(longitude: f32, latitude: f32, zoom: u32) -> vec2u {
  let x = quadbinGetTileX(longitude, zoom);
  let y = quadbinGetTileY(latitude, zoom);
  let morton = quadbinGetCompactKey(x, y);
  return vec2u(morton.x | (zoom << 26u), morton.y);
}
`;

/**
 * Geohash forward indexing, integer-exact. Concatenate after `CELL_KEY_WGSL`; it includes
 * `QUADBIN_TILE_WGSL`.
 *
 * Defines `cellIndexGeohashFromLngLat(lng, lat, length) -> vec2u(high, low)`. Bisecting the f32
 * longitude in [-180, 180] with `n` bits is `floor((lng + 180) 2^n / 360)`, and the latitude in
 * [-90, 90] with `m` bits is `floor((2 lat + 180) 2^m / 360)`. Both are exact integer results (the
 * midpoints are dyadic), computed with the Quadbin column routine, which divides exactly. The
 * values 180 and 90 land in the last bin like the reference bisection, inputs beyond them clamp.
 * Longitude takes `ceil(5 length / 2)` bits and latitude `floor(5 length / 2)`; the two are
 * Morton-interleaved with the longitude bit first.
 *
 * @internal
 */
export const GEOHASH_INDEX_WGSL = /* wgsl */ `
${CELL_KEY_WGSL}
${QUADBIN_TILE_WGSL}
fn cellIndexGeohashFromLngLat(longitude: f32, latitude: f32, length: u32) -> vec2u {
  let totalBits = 5u * length;
  let longitudeBits = (totalBits + 1u) / 2u;
  let latitudeBits = totalBits / 2u;
  var longitudeIndex = cellMaskLow32(longitudeBits);
  if (longitude < 180.0) {
    longitudeIndex = quadbinGetTileX(longitude, longitudeBits);
  }
  var latitudeIndex = cellMaskLow32(latitudeBits);
  if (latitude < 90.0) {
    latitudeIndex = quadbinGetTileX(2.0 * max(latitude, -90.0), latitudeBits);
  }
  var interleaved = quadbinGetCompactKey(longitudeIndex, latitudeIndex);
  if ((totalBits & 1u) == 0u) {
    interleaved = quadbinGetCompactKey(latitudeIndex, longitudeIndex);
  }
  return vec2u(interleaved.x | (length << 28u), interleaved.y);
}
`;

/**
 * S2 forward indexing in f32. Concatenate after `CELL_KEY_WGSL`.
 *
 * Defines `cellIndexS2FromLngLat(lng, lat, level) -> vec2u(high, low)`: longitude/latitude to a unit
 * vector (range-reduced polynomial sine and cosine, no `sin`/`cos` builtins), face from the largest
 * absolute component, `(u, v)` by division, `st` with S2's quadratic transform, `ij = floor(st 2^30)`
 * clamped to `[0, 2^30 - 1]`, then the Hilbert position by the S2 `kIJToPos` orientation state
 * machine (initial orientation `face & 1`, swap mask 1, invert mask 2), truncated to `level`.
 *
 * Precision: `st` is an f32 (24 bit mantissa) so `ij` carries an absolute error of up to a few
 * hundred units of 2^-30; the cell is exact except within that distance of a cell edge, so the
 * mismatch rate against an f64 reference grows with the level (see the contributor documentation).
 *
 * @internal
 */
export const S2_INDEX_WGSL = /* wgsl */ `
${CELL_KEY_WGSL}
const CELL_INDEX_S2_IJ_TO_POSITION = array<u32, 16>(${getInverseTable()
  .map(value => `${value}u`)
  .join(', ')});
const CELL_INDEX_S2_POSITION_TO_ORIENTATION = array<u32, 4>(${S2_POSITION_TO_ORIENTATION.map(value => `${value}u`).join(', ')});
const CELL_INDEX_S2_RADIANS_PER_DEGREE: f32 = 0.017453292519943295;

/** (sin, cos) of an angle in degrees, exact quadrant reduction plus polynomials on [-45, 45]. */
fn cellIndexS2SinCosDegrees(degrees: f32) -> vec2f {
  let quadrant = round(degrees / 90.0);
  let x = (degrees - quadrant * 90.0) * CELL_INDEX_S2_RADIANS_PER_DEGREE;
  let x2 = x * x;
  let sine = x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))));
  let cosine = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0)))));
  let turn = u32(i32(quadrant)) & 3u;
  if (turn == 0u) {
    return vec2f(sine, cosine);
  }
  if (turn == 1u) {
    return vec2f(cosine, -sine);
  }
  if (turn == 2u) {
    return vec2f(-sine, -cosine);
  }
  return vec2f(-cosine, sine);
}

fn cellIndexS2UvToSt(uv: f32) -> f32 {
  if (uv >= 0.0) {
    return 0.5 * sqrt(1.0 + 3.0 * uv);
  }
  return 1.0 - 0.5 * sqrt(1.0 - 3.0 * uv);
}

fn cellIndexS2StToIj(st: f32) -> u32 {
  let scaled = floor(st * 1073741824.0);
  if (scaled >= 1073741824.0) {
    return 1073741823u;
  }
  return u32(max(scaled, 0.0));
}

fn cellIndexS2FromLngLat(longitude: f32, latitude: f32, level: u32) -> vec2u {
  let lngTrig = cellIndexS2SinCosDegrees(longitude);
  let latTrig = cellIndexS2SinCosDegrees(clamp(latitude, -90.0, 90.0));
  let cosLatitude = max(latTrig.y, 0.0);
  let p = vec3f(cosLatitude * lngTrig.y, cosLatitude * lngTrig.x, latTrig.x);
  let magnitude = abs(p);
  var axis = 2u;
  if (magnitude.x > magnitude.y) {
    if (magnitude.x > magnitude.z) {
      axis = 0u;
    }
  } else if (magnitude.y > magnitude.z) {
    axis = 1u;
  }
  let component = select(p.z, select(p.y, p.x, axis == 0u), axis < 2u);
  let face = axis + select(0u, 3u, component < 0.0);
  var uv: vec2f;
  if (face == 0u) {
    uv = vec2f(p.y / p.x, p.z / p.x);
  } else if (face == 1u) {
    uv = vec2f(-p.x / p.y, p.z / p.y);
  } else if (face == 2u) {
    uv = vec2f(-p.x / p.z, -p.y / p.z);
  } else if (face == 3u) {
    uv = vec2f(p.z / p.x, p.y / p.x);
  } else if (face == 4u) {
    uv = vec2f(p.z / p.y, -p.x / p.y);
  } else {
    uv = vec2f(-p.y / p.z, -p.x / p.z);
  }
  let i = cellIndexS2StToIj(cellIndexS2UvToSt(uv.x));
  let j = cellIndexS2StToIj(cellIndexS2UvToSt(uv.y));
  var orientation = face & 1u;
  // Face in bits 61..63 and the level marker at bit 60 - 2 level.
  var key = cellShiftLeft(vec2u(0u, face), 61u) | cellShiftLeft(vec2u(0u, 1u), 60u - 2u * level);
  for (var step = 0u; step < level; step++) {
    let bit = 29u - step;
    let ij = (((i >> bit) & 1u) << 1u) | ((j >> bit) & 1u);
    let position = CELL_INDEX_S2_IJ_TO_POSITION[orientation * 4u + ij];
    orientation = orientation ^ CELL_INDEX_S2_POSITION_TO_ORIENTATION[position];
    key = key | cellShiftLeft(vec2u(0u, position), 59u - 2u * step);
  }
  return key;
}
`;
