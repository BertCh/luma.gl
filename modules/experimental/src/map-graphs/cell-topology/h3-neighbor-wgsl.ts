// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// H3 neighbor traversal is derived from Uber's Apache-2.0 H3 (`h3NeighborRotations`, algos.c).

import {
  H3_BASE_CELL_NEIGHBOR_ROTATIONS,
  H3_BASE_CELL_NEIGHBORS,
  H3_NEW_ADJUSTMENT_CLASS_II,
  H3_NEW_ADJUSTMENT_CLASS_III,
  H3_NEW_DIGIT_CLASS_II,
  H3_NEW_DIGIT_CLASS_III,
  H3_PENTAGON_CLOCKWISE_OFFSET_FACES
} from './h3-neighbor-tables';

/** Packs `neighbor | rotations << 8` per base cell and direction. */
function getPackedBaseCellNeighbors(): number[] {
  return H3_BASE_CELL_NEIGHBORS.map(
    (neighbor, index) => neighbor | (H3_BASE_CELL_NEIGHBOR_ROTATIONS[index] << 8)
  );
}

/** Packs the 7 digits of each table row into 3-bit fields. */
function getPackedDigitRows(table: readonly number[]): number[] {
  return Array.from({length: 7}, (_, row) =>
    table
      .slice(row * 7, row * 7 + 7)
      .reduce((packed, digit, column) => packed | (digit << (3 * column)), 0)
  );
}

/** Packs both clockwise-offset faces of each base cell as `face0 | face1 << 8`, `0xffff` if none. */
function getPackedClockwiseOffsetFaces(): number[] {
  return Array.from({length: 122}, (_, baseCell) => {
    const faces = H3_PENTAGON_CLOCKWISE_OFFSET_FACES[baseCell];
    return faces ? faces[0] | (faces[1] << 8) : 0xffff;
  });
}

function getU32ArraySource(name: string, values: readonly number[]): string {
  return `const ${name} = array<u32, ${values.length}>(${values.map(value => `${value >>> 0}u`).join(', ')});`;
}

/**
 * WGSL stepping from an H3 cell to its neighbor in one of the six H3 directions.
 *
 * Concatenate after `dggs.source` from `@luma.gl/shadertools`. Defines:
 * - `cellTopologyH3Neighbor(cell: vec2u, direction: u32) -> vec2u`: canonical `vec2u(high, low)`
 *   neighbor of a valid H3 cell for directions 1-6 (K, J, JK, I, IK, IJ), or `vec2u(0u)` for the
 *   deleted K direction at a pentagon center, an invalid cell, or an invalid direction. Directions are
 *   in the cell's own coordinate system, so the six results of a hexagon are its six distinct
 *   neighbors and the five nonzero results of a pentagon are its five neighbors (equal as a set to
 *   h3-js `gridDisk(cell, 1)` minus the cell). Integer only, so results are exact on every device.
 * - `cellTopologyH3IsPentagon(cell: vec2u) -> bool`.
 *
 * The rotation count returned by H3's `h3NeighborRotations` is not exposed: set-based traversal
 * (breadth-first disks) does not need it.
 *
 * @internal
 */
export const H3_NEIGHBOR_WGSL = /* wgsl */ `
${getU32ArraySource('CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS', getPackedBaseCellNeighbors())}
${getU32ArraySource('CELL_TOPOLOGY_H3_NEW_DIGIT_CLASS_III', getPackedDigitRows(H3_NEW_DIGIT_CLASS_III))}
${getU32ArraySource('CELL_TOPOLOGY_H3_NEW_ADJUSTMENT_CLASS_III', getPackedDigitRows(H3_NEW_ADJUSTMENT_CLASS_III))}
${getU32ArraySource('CELL_TOPOLOGY_H3_NEW_DIGIT_CLASS_II', getPackedDigitRows(H3_NEW_DIGIT_CLASS_II))}
${getU32ArraySource('CELL_TOPOLOGY_H3_NEW_ADJUSTMENT_CLASS_II', getPackedDigitRows(H3_NEW_ADJUSTMENT_CLASS_II))}
${getU32ArraySource('CELL_TOPOLOGY_H3_CLOCKWISE_OFFSET_FACES', getPackedClockwiseOffsetFaces())}
const CELL_TOPOLOGY_H3_INVALID_BASE_CELL: u32 = 127u;

fn cellTopologyH3GetTableDigit(packedRow: u32, column: u32) -> u32 {
  return (packedRow >> (3u * column)) & 7u;
}

fn cellTopologyH3SetDigit(cell: vec2u, resolution: u32, digit: u32) -> vec2u {
  return dggs_u64_set_bits(cell, dggs_h3_digit_bit_offset(resolution), 3u, digit);
}

fn cellTopologyH3RotateDigit60CounterClockwise(digit: u32) -> u32 {
  let values = array<u32, 8>(0u, 5u, 3u, 1u, 6u, 4u, 2u, 7u);
  return values[min(digit, 7u)];
}

fn cellTopologyH3Rotate60CounterClockwise(cell: vec2u) -> vec2u {
  var result = cell;
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    result = cellTopologyH3SetDigit(
      result,
      level,
      cellTopologyH3RotateDigit60CounterClockwise(dggs_h3_get_digit(result, level))
    );
  }
  return result;
}

fn cellTopologyH3Rotate60Clockwise(cell: vec2u) -> vec2u {
  var result = cell;
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    result = cellTopologyH3SetDigit(
      result,
      level,
      dggs_h3_rotate_digit_60_cw(dggs_h3_get_digit(result, level))
    );
  }
  return result;
}

/** H3 \`_h3RotatePent60ccw\`: rotates again when the leading digit lands on the deleted K axis. */
fn cellTopologyH3RotatePentagon60CounterClockwise(cell: vec2u) -> vec2u {
  var result = cell;
  var foundFirstNonZeroDigit = false;
  let resolution = dggs_h3_get_resolution(cell);
  for (var level = 1u; level <= resolution; level++) {
    result = cellTopologyH3SetDigit(
      result,
      level,
      cellTopologyH3RotateDigit60CounterClockwise(dggs_h3_get_digit(result, level))
    );
    if (!foundFirstNonZeroDigit && dggs_h3_get_digit(result, level) != 0u) {
      foundFirstNonZeroDigit = true;
      if (dggs_h3_get_leading_non_zero_digit(result) == 1u) {
        result = cellTopologyH3Rotate60CounterClockwise(result);
      }
    }
  }
  return result;
}

fn cellTopologyH3IsPentagon(cell: vec2u) -> bool {
  return dggs_h3_is_valid_cell_id(cell) &&
    dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) &&
    dggs_h3_get_leading_non_zero_digit(cell) == 0u;
}

fn cellTopologyH3IsClockwiseOffset(baseCell: u32, face: u32) -> bool {
  let faces = CELL_TOPOLOGY_H3_CLOCKWISE_OFFSET_FACES[min(baseCell, 121u)];
  return (faces & 0xffu) == face || ((faces >> 8u) & 0xffu) == face;
}

fn cellTopologyH3Neighbor(cell: vec2u, direction: u32) -> vec2u {
  if (direction == 0u || direction > 6u || !dggs_h3_is_valid_cell_id(cell)) {
    return vec2u(0u);
  }
  // The K direction is deleted at a pentagon. At resolution 0 H3 would return the IK neighbor
  // a second time; zero keeps the six results distinct.
  if (direction == 1u && cellTopologyH3IsPentagon(cell)) {
    return vec2u(0u);
  }
  var current = cell;
  var stepDirection = direction;
  var newRotations = 0u;
  let oldBaseCell = dggs_h3_get_base_cell(cell);
  let oldLeadingDigit = dggs_h3_get_leading_non_zero_digit(cell);

  // Adjust the digits from the finest resolution up, carrying into coarser digits.
  var level = dggs_h3_get_resolution(cell);
  loop {
    if (level == 0u) {
      var packed = CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS[oldBaseCell * 7u + stepDirection];
      if ((packed & 0xffu) == CELL_TOPOLOGY_H3_INVALID_BASE_CELL) {
        // The deleted K vertex at the base cell level: this edge borders the IK neighbor.
        packed = CELL_TOPOLOGY_H3_BASE_CELL_NEIGHBORS[oldBaseCell * 7u + 5u];
        current = cellTopologyH3Rotate60CounterClockwise(current);
      }
      current = dggs_u64_set_bits(current, 45u, 7u, packed & 0xffu);
      newRotations = packed >> 8u;
      break;
    }
    let oldDigit = dggs_h3_get_digit(current, level);
    var nextDirection = 0u;
    if (dggs_h3_is_resolution_class_iii(level)) {
      current = cellTopologyH3SetDigit(
        current,
        level,
        cellTopologyH3GetTableDigit(CELL_TOPOLOGY_H3_NEW_DIGIT_CLASS_III[oldDigit], stepDirection)
      );
      nextDirection = cellTopologyH3GetTableDigit(
        CELL_TOPOLOGY_H3_NEW_ADJUSTMENT_CLASS_III[oldDigit],
        stepDirection
      );
    } else {
      current = cellTopologyH3SetDigit(
        current,
        level,
        cellTopologyH3GetTableDigit(CELL_TOPOLOGY_H3_NEW_DIGIT_CLASS_II[oldDigit], stepDirection)
      );
      nextDirection = cellTopologyH3GetTableDigit(
        CELL_TOPOLOGY_H3_NEW_ADJUSTMENT_CLASS_II[oldDigit],
        stepDirection
      );
    }
    if (nextDirection == 0u) {
      break;
    }
    stepDirection = nextDirection;
    level -= 1u;
  }

  let newBaseCell = dggs_h3_get_base_cell(current);
  if (dggs_h3_is_base_cell_pentagon(newBaseCell)) {
    // Force the result out of the deleted K subsequence.
    if (dggs_h3_get_leading_non_zero_digit(current) == 1u) {
      if (oldBaseCell != newBaseCell) {
        if (cellTopologyH3IsClockwiseOffset(newBaseCell, u32(dggs_h3_get_base_cell_home(oldBaseCell).face))) {
          current = cellTopologyH3Rotate60Clockwise(current);
        } else {
          current = cellTopologyH3Rotate60CounterClockwise(current);
        }
      } else if (oldLeadingDigit == 0u) {
        // The K direction is deleted at a pentagon center.
        return vec2u(0u);
      } else if (oldLeadingDigit == 3u) {
        current = cellTopologyH3Rotate60CounterClockwise(current);
      } else if (oldLeadingDigit == 5u) {
        current = cellTopologyH3Rotate60Clockwise(current);
      } else {
        return vec2u(0u);
      }
    }
    for (var rotation = 0u; rotation < newRotations; rotation++) {
      current = cellTopologyH3RotatePentagon60CounterClockwise(current);
    }
  } else {
    for (var rotation = 0u; rotation < newRotations; rotation++) {
      current = cellTopologyH3Rotate60CounterClockwise(current);
    }
  }
  return current;
}
`;
