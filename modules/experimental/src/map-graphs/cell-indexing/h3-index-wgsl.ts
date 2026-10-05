// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// The forward H3 algorithm (`geoToFaceIjk`, `hex2dToCoordIjk`, `faceIjkToH3`) is derived from
// Uber's Apache-2.0 H3.

import {
  CELL_INDEX_H3_BASE_CELL_PACKED_TABLE,
  CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES
} from './h3-index-tables';

const BASE_CELL_TABLE_WGSL = CELL_INDEX_H3_BASE_CELL_PACKED_TABLE.map(value =>
  value < 0 ? '65535u' : `${value}u`
).join(', ');

const CW_OFFSET_CONDITION_WGSL = CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES.filter(
  ([, faceA]) => faceA >= 0
)
  .map(
    ([baseCell, faceA, faceB]) =>
      `(baseCell == ${baseCell}u && (face == ${faceA}u || face == ${faceB}u))`
  )
  .join(' ||\n    ');

const SQRT7_POWERS_WGSL = Array.from({length: 16}, (_, resolution) =>
  Math.fround(Math.sqrt(7) ** resolution).toPrecision(9)
).join(', ');

/**
 * WGSL implementing forward H3 `latLngToCell` for one point, to be concatenated AFTER
 * `dggs.source` from `@luma.gl/shadertools` (it reuses the `dggs_h3_*` and `dggs_u64_*` helpers).
 *
 * Defines `fn cellIndexH3FromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u`, which returns
 * the canonical `vec2u(high, low)` H3 index (mode 1, unused digits 7), or `vec2u(0u)` for non-finite
 * input, resolution above 15, or a point the lookup table cannot place. Every other identifier is
 * prefixed `cellIndexH3` / `CELL_INDEX_H3_`.
 *
 * Method: degrees to a unit vector with range-reduced polynomial sine/cosine (no `sin`/`cos`
 * builtins, whose accuracy WGSL leaves implementation-defined), nearest icosahedron face by maximum
 * dot product, gnomonic projection onto the face hex grid through the face basis vectors (no
 * `atan`/`acos`/`tan`), `hex2dToCoordIjk` quantization, then the `_faceIjkToH3` digit walk with the
 * generated `faceIjkBaseCells` table and pentagon rotation rules.
 *
 * Precision (float32): the input is f32 degrees and the unit vector, face projection and hex2d
 * scaling are f32, so the absolute position error is about 1e-7 of the sphere radius (roughly one
 * metre) plus a relative 6e-8 of the hex2d magnitude, which grows by sqrt(7) per resolution. The
 * result matches h3-js on the same f32-rounded input except for points within that error of a cell
 * edge, so the cell boundary mismatch rate grows with resolution: it is negligible at coarse
 * resolutions, still small through the middle resolutions, and by resolution 13 to 15 (edges of a
 * few metres to under a metre, hex2d coordinates beyond 2^20) most points are in the wrong cell,
 * though normally a grid neighbor. Resolution 15 is effectively unusable from f32 positions: the
 * f32 input itself is coarser than the cell. A mismatch is nearly always a grid neighbor.
 *
 * Latitude is expected in [-90, 90]; longitude may be any finite value (it wraps).
 */
export const H3_INDEX_WGSL: string = /* wgsl */ `\
const CELL_INDEX_H3_BASE_CELL_TABLE = array<u32, 540>(${BASE_CELL_TABLE_WGSL});
const CELL_INDEX_H3_SQRT7_POWERS = array<f32, 16>(${SQRT7_POWERS_WGSL});
const CELL_INDEX_H3_ROTATE_CCW = array<u32, 8>(0u, 5u, 3u, 1u, 6u, 4u, 2u, 7u);
const CELL_INDEX_H3_NO_CELL: u32 = 65535u;
// cos and sin of the aperture-7 rotation: 5 / (2 sqrt 7) and sqrt 3 / (2 sqrt 7).
const CELL_INDEX_H3_AP7_COS: f32 = 0.944911182523068;
const CELL_INDEX_H3_AP7_SIN: f32 = 0.3273268353539886;

fn cellIndexH3IsCwOffset(baseCell: u32, face: u32) -> bool {
  return ${CW_OFFSET_CONDITION_WGSL};
}

// (sin, cos) of an angle in degrees. The angle is reduced to [-45, 45] degrees exactly in degrees,
// then a Taylor polynomial gives full f32 accuracy independent of the builtin sin/cos precision.
fn cellIndexH3SinCosDegrees(degrees: f32) -> vec2f {
  let quadrant = round(degrees * (1.0 / 90.0));
  let reduced = (degrees - quadrant * 90.0) * 0.017453292519943295;
  let x2 = reduced * reduced;
  let sine = reduced * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 +
    x2 * (1.0 / 362880.0 + x2 * (-1.0 / 39916800.0))))));
  let cosine = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 +
    x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0))))));
  let turn = u32(i32(quadrant)) & 3u;
  if (turn == 1u) {
    return vec2f(cosine, -sine);
  }
  if (turn == 2u) {
    return vec2f(-sine, -cosine);
  }
  if (turn == 3u) {
    return vec2f(-cosine, sine);
  }
  return vec2f(sine, cosine);
}

fn cellIndexH3GetNearestFace(point: vec3f) -> u32 {
  var face = 0u;
  var bestAlignment = -2.0;
  for (var candidate = 0u; candidate < 20u; candidate++) {
    let alignment = dot(point, dggs_h3_get_face_unit_vector_basis(candidate, 0u));
    if (alignment > bestAlignment) {
      bestAlignment = alignment;
      face = candidate;
    }
  }
  return face;
}

// H3 _hex2dToCoordIJK: quantizes a hex2d point to the containing cell's normalized IJK.
fn cellIndexH3Hex2dToIjk(point: vec2f) -> vec3i {
  let a1 = abs(point.x);
  let a2 = abs(point.y);
  let x2 = a2 / DGGS_H3_SQRT3_2;
  let x1 = a1 + x2 * 0.5;
  let m1 = i32(x1);
  let m2 = i32(x2);
  let r1 = x1 - f32(m1);
  let r2 = x2 - f32(m2);
  var i = 0;
  var j = 0;
  if (r1 < 0.5) {
    if (r1 < 1.0 / 3.0) {
      i = m1;
      j = select(m2 + 1, m2, r2 < (1.0 + r1) * 0.5);
    } else {
      j = select(m2 + 1, m2, r2 < 1.0 - r1);
      i = select(m1, m1 + 1, (1.0 - r1) <= r2 && r2 < 2.0 * r1);
    }
  } else if (r1 < 2.0 / 3.0) {
    j = select(m2 + 1, m2, r2 < 1.0 - r1);
    i = select(m1 + 1, m1, (2.0 * r1 - 1.0) < r2 && r2 < 1.0 - r1);
  } else {
    i = m1 + 1;
    j = select(m2 + 1, m2, r2 < r1 * 0.5);
  }
  if (point.x < 0.0) {
    if ((j & 1) == 0) {
      i -= 2 * (i - j / 2);
    } else {
      i -= 2 * (i - (j + 1) / 2) + 1;
    }
  }
  if (point.y < 0.0) {
    i -= (2 * j + 1) / 2;
    j = -j;
  }
  return dggs_h3_ijk_normalize(vec3i(i, j, 0));
}

// H3 _upAp7: Class III child to Class II parent (counter-clockwise aperture 7).
fn cellIndexH3UpAp7(coord: vec3i) -> vec3i {
  let i = coord.x - coord.z;
  let j = coord.y - coord.z;
  return dggs_h3_ijk_normalize(vec3i(
    dggs_h3_divide_round_nearest_seven(3 * i - j),
    dggs_h3_divide_round_nearest_seven(i + 2 * j),
    0
  ));
}

fn cellIndexH3RotateDigitsCcw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = dggs_h3_get_digit(index, digitResolution);
    result = dggs_u64_set_bits(
      result,
      dggs_h3_digit_bit_offset(digitResolution),
      3u,
      CELL_INDEX_H3_ROTATE_CCW[digit]
    );
  }
  return result;
}

fn cellIndexH3RotateDigitsCw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = dggs_h3_get_digit(index, digitResolution);
    result = dggs_u64_set_bits(
      result,
      dggs_h3_digit_bit_offset(digitResolution),
      3u,
      dggs_h3_rotate_digit_60_cw(digit)
    );
  }
  return result;
}

// H3 _h3RotatePent60ccw: rotates counter-clockwise, skipping the deleted K subsequence.
fn cellIndexH3RotatePentagonDigitsCcw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  var foundFirstNonZeroDigit = false;
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = CELL_INDEX_H3_ROTATE_CCW[dggs_h3_get_digit(result, digitResolution)];
    result = dggs_u64_set_bits(result, dggs_h3_digit_bit_offset(digitResolution), 3u, digit);
    if (!foundFirstNonZeroDigit && digit != 0u) {
      foundFirstNonZeroDigit = true;
      if (dggs_h3_get_leading_non_zero_digit(result) == 1u) {
        result = cellIndexH3RotateDigitsCcw(result);
      }
    }
  }
  return result;
}

// H3 _faceIjkToH3 for an IJK at 'resolution' on 'face'. Returns vec2u(0u) when no base cell fits.
fn cellIndexH3FaceIjkToIndex(face: u32, coord: vec3i, resolution: u32) -> vec2u {
  // Header with base cell 0 and every digit unused (7); digits and base cell are filled in below.
  var index = vec2u((1u << 27u) | (resolution << 20u) | 0x1FFFu, 0xFFFFFFFFu);
  var current = coord;
  for (var parentResolution = resolution; parentResolution > 0u; parentResolution--) {
    let last = current;
    var lastCenter = vec3i(0);
    if (dggs_h3_is_resolution_class_iii(parentResolution)) {
      current = cellIndexH3UpAp7(current);
      lastCenter = dggs_h3_down_ap7(current);
    } else {
      current = dggs_h3_up_ap7r(current);
      lastCenter = dggs_h3_down_ap7r(current);
    }
    let difference = dggs_h3_ijk_normalize(last - lastCenter);
    index = dggs_u64_set_bits(
      index,
      dggs_h3_digit_bit_offset(parentResolution),
      3u,
      u32(4 * difference.x + 2 * difference.y + difference.z)
    );
  }
  if (any(current < vec3i(0)) || any(current > vec3i(2))) {
    return vec2u(0u);
  }
  let packed = CELL_INDEX_H3_BASE_CELL_TABLE[
    face * 27u + u32(current.x) * 9u + u32(current.y) * 3u + u32(current.z)
  ];
  if (packed == CELL_INDEX_H3_NO_CELL) {
    return vec2u(0u);
  }
  let baseCell = packed & 0xFFu;
  let rotations = packed >> 8u;
  index = dggs_u64_set_bits(index, 45u, 7u, baseCell);
  if (dggs_h3_is_base_cell_pentagon(baseCell)) {
    if (dggs_h3_get_leading_non_zero_digit(index) == 1u) {
      if (cellIndexH3IsCwOffset(baseCell, face)) {
        index = cellIndexH3RotateDigitsCw(index);
      } else {
        index = cellIndexH3RotateDigitsCcw(index);
      }
    }
    for (var rotation = 0u; rotation < rotations; rotation++) {
      index = cellIndexH3RotatePentagonDigitsCcw(index);
    }
  } else {
    for (var rotation = 0u; rotation < rotations; rotation++) {
      index = cellIndexH3RotateDigitsCcw(index);
    }
  }
  return index;
}

fn cellIndexH3FromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u {
  if (
    !(abs(lngLatDegrees.x) < 3.0e38) ||
    !(abs(lngLatDegrees.y) < 3.0e38) ||
    resolution > DGGS_H3_MAX_RESOLUTION
  ) {
    return vec2u(0u);
  }
  let longitudeSinCos = cellIndexH3SinCosDegrees(lngLatDegrees.x);
  let latitudeSinCos = cellIndexH3SinCosDegrees(lngLatDegrees.y);
  let point = vec3f(
    latitudeSinCos.y * longitudeSinCos.y,
    latitudeSinCos.y * longitudeSinCos.x,
    latitudeSinCos.x
  );
  let face = cellIndexH3GetNearestFace(point);
  let faceCenter = dggs_h3_get_face_unit_vector_basis(face, 0u);
  let inverseScale = 1.0 / (dot(point, faceCenter) * DGGS_H3_RES0_U_GNOMONIC);
  var hexPoint = vec2f(
    dot(point, dggs_h3_get_face_unit_vector_basis(face, 1u)),
    dot(point, dggs_h3_get_face_unit_vector_basis(face, 2u))
  ) * inverseScale;
  if (dggs_h3_is_resolution_class_iii(resolution)) {
    hexPoint = vec2f(
      CELL_INDEX_H3_AP7_COS * hexPoint.x + CELL_INDEX_H3_AP7_SIN * hexPoint.y,
      -CELL_INDEX_H3_AP7_SIN * hexPoint.x + CELL_INDEX_H3_AP7_COS * hexPoint.y
    );
  }
  hexPoint *= CELL_INDEX_H3_SQRT7_POWERS[resolution];
  return cellIndexH3FaceIjkToIndex(face, cellIndexH3Hex2dToIjk(hexPoint), resolution);
}
`;
