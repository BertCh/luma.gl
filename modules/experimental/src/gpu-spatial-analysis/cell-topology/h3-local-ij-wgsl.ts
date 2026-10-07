// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// H3 local IJK is derived from Uber's Apache-2.0 H3 (`cellToLocalIjk`, localij.c).

/**
 * Pentagon rotation counts of H3 local IJK, `-1` where the pair fails (the cell lies behind the
 * deleted K sector of a pentagon), row-major `[leadingDigit * 7 + other]`.
 *
 * GENERATED, not recalled: each entry is the value that reproduces h3-js `cellToLocalIj` for
 * every origin/cell pair that reaches it (hundreds to thousands of pairs per entry, resolutions 1-4,
 * pentagon neighborhoods) and its failures. The values count counter-clockwise IJK rotations
 * (which is what makes them equal to H3's own table). `gpu-cell-grid-path.spec.ts` re-verifies
 * them against python h3 on pentagon neighborhoods and random pairs.
 *
 * - `origin`: indexed `[originLeadingDigit][direction]` when the origin sits on a pentagon base
 *   cell, and `[originLeadingDigit][cellLeadingDigit]` when both share a pentagon base cell.
 * - `index`: indexed `[direction][rotatedCellLeadingDigit]` when the cell sits on a pentagon base
 *   cell and the origin does not.
 *
 * @internal
 */
export const H3_LOCAL_IJK_PENTAGON_ROTATIONS = {
  // prettier-ignore
  origin: [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 5, -1, 0, 0,
    -1, -1, 0, 0, 0, 0, 0, -1, 1, 0, 0, -1, 0, 0, 0, -1, 0, -1, 0
  ],
  // prettier-ignore
  index: [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, -1, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, -1, 5, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]
} as const;

function getI32ArraySource(name: string, values: readonly number[]): string {
  return `const ${name} = array<i32, ${values.length}>(${values.map(value => `${value}`).join(', ')});`;
}

/**
 * WGSL for H3 local IJK coordinates (`cellToLocalIjk`), grid distance of arbitrary cell pairs, and
 * the exact `gridPathCells` interpolation.
 *
 * Concatenate after `dggs.source` and {@link H3_NEIGHBOR_WGSL} (it reuses its base cell tables,
 * `cellTopologyH3Rotate60Clockwise`, `cellTopologyH3IsPentagon` and `cellTopologyH3Neighbor`).
 * Defines:
 * - `struct CellTopologyH3LocalIjk {ijk: vec3i, valid: u32}`
 * - `cellTopologyH3GetLocalIjk(origin, cell) -> CellTopologyH3LocalIjk`: the IJK of `cell` in the
 *   coordinate system anchored at `origin` (H3 `cellToLocalIjk`). `valid` is 0 when H3 itself
 *   fails: invalid cells, different resolutions, base cells that are not equal or adjacent, or a
 *   cell behind the deleted K sector of a pentagon. The local `(i, j)` of h3-js `cellToLocalIj`
 *   is `(ijk.x - ijk.z, ijk.y - ijk.z)`.
 * - `cellTopologyH3GetGridDistance(a: vec3i, b: vec3i) -> u32`: the IJK distance of two local
 *   coordinates of the same origin (H3 `gridDistance`).
 * - `cellTopologyH3GetCube(ijk) -> vec3i` and `cellTopologyH3RoundCubeStep(...) -> vec2i`: the
 *   cube-rounded linear interpolation of H3 `gridPathCells`, one step at a time, in exact integers.
 *
 * Everything is integer arithmetic, so results are identical on every device.
 *
 * @internal
 */
export const H3_LOCAL_IJ_WGSL = /* wgsl */ `
${getI32ArraySource('CELL_TOPOLOGY_H3_ORIGIN_PENTAGON_ROTATIONS', H3_LOCAL_IJK_PENTAGON_ROTATIONS.origin)}
${getI32ArraySource('CELL_TOPOLOGY_H3_INDEX_PENTAGON_ROTATIONS', H3_LOCAL_IJK_PENTAGON_ROTATIONS.index)}

struct CellTopologyH3LocalIjk {
  ijk : vec3i,
  valid : u32,
};

/** Direction (0-6) from a base cell to an adjacent base cell, or -1. */
fn cellTopologyH3GetBaseCellDirection(fromBaseCell: u32, toBaseCell: u32) -> i32 {
  for (var direction = 0u; direction < 7u; direction++) {
    if ((CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS[fromBaseCell * 7u + direction] & 0xffu) == toBaseCell) {
      return i32(direction);
    }
  }
  return -1;
}

/** H3 \`_h3RotatePent60cw\`: rotates again when the leading digit lands on the deleted K axis. */
fn cellTopologyH3RotatePentagon60Clockwise(cell: vec2u) -> vec2u {
  var result = cell;
  var foundFirstNonZeroDigit = false;
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    result = cellTopologyH3SetDigit(
      result,
      level,
      dggs_h3_rotate_digit_60_cw(dggs_h3_get_digit(result, level))
    );
    if (!foundFirstNonZeroDigit && dggs_h3_get_digit(result, level) != 0u) {
      foundFirstNonZeroDigit = true;
      if (dggs_h3_get_leading_non_zero_digit(result) == 1u) {
        result = cellTopologyH3Rotate60Clockwise(result);
      }
    }
  }
  return result;
}

/** IJK of a cell in its own base cell's coordinate system (base cell center at the origin). */
fn cellTopologyH3GetBaseCellIjk(cell: vec2u) -> vec3i {
  var ijk = vec3i(0);
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    if (dggs_h3_is_resolution_class_iii(level)) {
      ijk = dggs_h3_down_ap7(ijk);
    } else {
      ijk = dggs_h3_down_ap7r(ijk);
    }
    ijk = dggs_h3_neighbor(ijk, dggs_h3_get_digit(cell, level));
  }
  return ijk;
}

fn cellTopologyH3GetLocalIjk(origin: vec2u, cell: vec2u) -> CellTopologyH3LocalIjk {
  let invalidResult = CellTopologyH3LocalIjk(vec3i(0), 0u);
  if (!dggs_h3_is_valid_cell_id(origin) || !dggs_h3_is_valid_cell_id(cell)) {
    return invalidResult;
  }
  let resolution = dggs_h3_get_resolution(origin);
  if (dggs_h3_get_resolution(cell) != resolution) {
    return invalidResult;
  }
  let originBaseCell = dggs_h3_get_base_cell(origin);
  let cellBaseCell = dggs_h3_get_base_cell(cell);
  var direction = 0;
  if (originBaseCell != cellBaseCell) {
    direction = cellTopologyH3GetBaseCellDirection(originBaseCell, cellBaseCell);
    if (direction < 0) {
      return invalidResult;
    }
  }
  let originOnPentagon = dggs_h3_is_base_cell_pentagon(originBaseCell);
  let cellOnPentagon = dggs_h3_is_base_cell_pentagon(cellBaseCell);

  // Rotate the cell into the orientation of the origin base cell.
  var rotated = cell;
  if (direction != 0) {
    let rotations = CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS[originBaseCell * 7u + u32(direction)] >> 8u;
    for (var rotation = 0u; rotation < rotations; rotation++) {
      if (cellOnPentagon) {
        rotated = cellTopologyH3RotatePentagon60Clockwise(rotated);
      } else {
        rotated = cellTopologyH3Rotate60Clockwise(rotated);
      }
    }
  }
  var ijk = cellTopologyH3GetBaseCellIjk(rotated);

  if (direction != 0) {
    var pentagonRotations = 0;
    var directionRotations = 0;
    if (originOnPentagon) {
      let leading = dggs_h3_get_leading_non_zero_digit(origin);
      let rotations = CELL_TOPOLOGY_H3_ORIGIN_PENTAGON_ROTATIONS[leading * 7u + u32(direction)];
      if (rotations < 0) {
        return invalidResult;
      }
      pentagonRotations = rotations;
      directionRotations = rotations;
    } else if (cellOnPentagon) {
      let leading = dggs_h3_get_leading_non_zero_digit(rotated);
      let rotations = CELL_TOPOLOGY_H3_INDEX_PENTAGON_ROTATIONS[u32(direction) * 7u + leading];
      if (rotations < 0) {
        return invalidResult;
      }
      pentagonRotations = rotations;
    }
    for (var rotation = 0; rotation < pentagonRotations; rotation++) {
      ijk = dggs_h3_ijk_rotate_60_ccw(ijk);
    }
    // The neighboring base cell center, scaled to the cell resolution.
    var offset = dggs_h3_get_unit_vector(u32(direction));
    for (var level = resolution; level >= 1u; level--) {
      if (dggs_h3_is_resolution_class_iii(level)) {
        offset = dggs_h3_down_ap7(offset);
      } else {
        offset = dggs_h3_down_ap7r(offset);
      }
    }
    for (var rotation = 0; rotation < directionRotations; rotation++) {
      offset = dggs_h3_ijk_rotate_60_ccw(offset);
    }
    ijk = dggs_h3_ijk_normalize(ijk + offset);
  } else if (originOnPentagon && cellOnPentagon) {
    let originLeading = dggs_h3_get_leading_non_zero_digit(origin);
    let cellLeading = dggs_h3_get_leading_non_zero_digit(rotated);
    let rotations = CELL_TOPOLOGY_H3_ORIGIN_PENTAGON_ROTATIONS[originLeading * 7u + cellLeading];
    if (rotations < 0) {
      return invalidResult;
    }
    for (var rotation = 0; rotation < rotations; rotation++) {
      ijk = dggs_h3_ijk_rotate_60_ccw(ijk);
    }
  }
  return CellTopologyH3LocalIjk(ijk, 1u);
}

/** H3 \`gridDistance\` between two local IJK coordinates of the same origin. */
fn cellTopologyH3GetGridDistance(a: vec3i, b: vec3i) -> u32 {
  let difference = dggs_h3_ijk_normalize(a - b);
  return u32(max(difference.x, max(difference.y, difference.z)));
}

/** Cube coordinates \`(-I, J, I - J)\` of a local IJK, as H3 \`ijkToCube\`. */
fn cellTopologyH3GetCube(ijk: vec3i) -> vec3i {
  let i = ijk.x - ijk.z;
  let j = ijk.y - ijk.z;
  return vec3i(-i, j, i - j);
}

/**
 * Rounds the interpolation \`from + (to - from) * step / distance\` of two cube coordinates to
 * the nearest cube coordinate, with H3 \`cubeRound\` (half away from zero, largest rounding error
 * recomputed) evaluated in exact integer arithmetic. \`quotient\` and \`remainder\` carry
 * \`floor((to - from) * step / distance)\` and its remainder, advanced by the caller one step at a
 * time so nothing overflows for long paths. Returns the local \`(i, j)\`.
 */
fn cellTopologyH3RoundCubeStep(
  start: vec3i,
  quotient: vec3i,
  remainder: vec3i,
  pathDistance: i32
) -> vec2i {
  let base = start + quotient;
  var rounded = vec3i(0);
  var roundingError = vec3i(0);
  for (var axis = 0; axis < 3; axis++) {
    let whole = base[axis];
    let part = remainder[axis];
    // The value is whole + part / pathDistance with part in [0, pathDistance).
    let roundsUp = select(2 * part > pathDistance, 2 * part >= pathDistance, whole >= 0);
    rounded[axis] = whole + select(0, 1, roundsUp);
    roundingError[axis] = select(2 * part, 2 * pathDistance - 2 * part, roundsUp);
  }
  if (roundingError.x > roundingError.y && roundingError.x > roundingError.z) {
    rounded.x = -rounded.y - rounded.z;
  } else if (roundingError.y > roundingError.z) {
    rounded.y = -rounded.x - rounded.z;
  } else {
    rounded.z = -rounded.x - rounded.y;
  }
  return vec2i(-rounded.x, rounded.y);
}
`;
