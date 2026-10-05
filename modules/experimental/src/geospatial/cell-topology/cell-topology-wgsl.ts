// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Quadbin tile helpers and per-thread neighborhood enumeration.
 *
 * Concatenate after `CELL_KEY_WGSL` and define `CELL_TOPOLOGY_K` (radius) and
 * `CELL_TOPOLOGY_CAPACITY` (private array size, at least the largest neighborhood) first. Defines:
 * - `cellTopologyQuadbinGetTile(key) -> vec2u(x, y)` and
 *   `cellTopologyQuadbinGetKey(x, y, z) -> vec2u` (canonical `vec2u(high, low)` keys).
 * - `cellTopologyQuadbinNeighborhood(cell, ringOnly) -> u32` fills the private arrays
 *   `cellTopologyCells` / `cellTopologyDistances` with the cells at Chebyshev distance
 *   `<= K` (or `== K` for rings) sorted by (distance, key) and returns the count. Longitude
 *   (x) wraps across the antimeridian, latitude (y) does not: rows outside `[0, 2^z)` are dropped.
 *   Each wrapped tile appears once, at its smallest circular x distance.
 * - `cellTopologyQuadbinGetChild(cell, targetResolution, depth, slot) -> vec2u` and
 *   `cellTopologyQuadbinGetParent(cell, targetResolution) -> vec2u`.
 *
 * @internal
 */
export const QUADBIN_TOPOLOGY_WGSL = /* wgsl */ `
const CELL_TOPOLOGY_QUADBIN_HEADER_HIGH: u32 = 0x48000000u;

fn cellTopologyQuadbinSpread16(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

fn cellTopologyQuadbinCompact16(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}

fn cellTopologyQuadbinGetTile(key: vec2u) -> vec2u {
  let z = (key.x >> 20u) & 0x1fu;
  let compact = cellGetCompactKey(key, 52u - 2u * z, 2u * z);
  return vec2u(
    cellTopologyQuadbinCompact16(compact.y) | (cellTopologyQuadbinCompact16(compact.x) << 16u),
    cellTopologyQuadbinCompact16(compact.y >> 1u) | (cellTopologyQuadbinCompact16(compact.x >> 1u) << 16u)
  );
}

fn cellTopologyQuadbinGetKey(x: u32, y: u32, z: u32) -> vec2u {
  let compact = vec2u(
    cellTopologyQuadbinSpread16(x >> 16u) | (cellTopologyQuadbinSpread16(y >> 16u) << 1u),
    cellTopologyQuadbinSpread16(x) | (cellTopologyQuadbinSpread16(y) << 1u)
  );
  return cellGetKey(compact, CELL_TOPOLOGY_QUADBIN_HEADER_HIGH, z, 52u - 2u * z);
}

fn cellTopologyQuadbinGetParent(cell: vec2u, targetResolution: u32) -> vec2u {
  let z = (cell.x >> 20u) & 0x1fu;
  let compact = cellGetCompactKey(cell, 52u - 2u * z, 2u * z);
  return cellGetKey(cellShiftRight(compact, 2u * (z - targetResolution)), CELL_TOPOLOGY_QUADBIN_HEADER_HIGH, targetResolution, 52u - 2u * targetResolution);
}

fn cellTopologyQuadbinGetChild(cell: vec2u, targetResolution: u32, depth: u32, slot: u32) -> vec2u {
  let z = (cell.x >> 20u) & 0x1fu;
  let compact = cellGetCompactKey(cell, 52u - 2u * z, 2u * z);
  let child = cellShiftLeft(compact, 2u * depth) | vec2u(0u, slot);
  return cellGetKey(child, CELL_TOPOLOGY_QUADBIN_HEADER_HIGH, targetResolution, 52u - 2u * targetResolution);
}

var<private> cellTopologyCells: array<vec2u, CELL_TOPOLOGY_CAPACITY>;
var<private> cellTopologyDistances: array<u32, CELL_TOPOLOGY_CAPACITY>;

/** Insertion sort of the first \`count\` private entries by (distance, key). */
fn cellTopologySortNeighborhood(count: u32) {
  for (var i = 1u; i < count; i++) {
    let cell = cellTopologyCells[i];
    let distance = cellTopologyDistances[i];
    var j = i;
    loop {
      if (j == 0u) {
        break;
      }
      let otherCell = cellTopologyCells[j - 1u];
      let otherDistance = cellTopologyDistances[j - 1u];
      let isLess = distance < otherDistance ||
        (distance == otherDistance && (cell.x < otherCell.x || (cell.x == otherCell.x && cell.y < otherCell.y)));
      if (!isLess) {
        break;
      }
      cellTopologyCells[j] = otherCell;
      cellTopologyDistances[j] = otherDistance;
      j = j - 1u;
    }
    cellTopologyCells[j] = cell;
    cellTopologyDistances[j] = distance;
  }
}

fn cellTopologyQuadbinNeighborhood(cell: vec2u, ringOnly: bool) -> u32 {
  let z = (cell.x >> 20u) & 0x1fu;
  let tile = cellTopologyQuadbinGetTile(cell);
  let n = i32(1u << z);
  let k = i32(CELL_TOPOLOGY_K);
  var count = 0u;
  for (var dy = -k; dy <= k; dy++) {
    let y = i32(tile.y) + dy;
    if (y < 0 || y >= n) {
      continue;
    }
    for (var dx = -k; dx <= k; dx++) {
      // Keep the representative of each wrapped column with the smallest |dx|.
      let r = ((dx % n) + n) % n;
      let representative = select(r - n, r, 2 * r <= n);
      if (representative != dx) {
        continue;
      }
      let distance = u32(max(abs(dx), abs(dy)));
      if (ringOnly && distance != CELL_TOPOLOGY_K) {
        continue;
      }
      let x = ((i32(tile.x) + dx) % n + n) % n;
      cellTopologyCells[count] = cellTopologyQuadbinGetKey(u32(x), u32(y), z);
      cellTopologyDistances[count] = distance;
      count++;
    }
  }
  cellTopologySortNeighborhood(count);
  return count;
}
`;

/**
 * H3 neighborhoods by in-thread breadth-first search, plus parent and children helpers.
 *
 * Concatenate after `dggs.source`, `CELL_KEY_WGSL`, `H3_NEIGHBOR_WGSL` and
 * {@link QUADBIN_TOPOLOGY_WGSL}, which declares the shared private arrays and sort. Defines:
 * - `cellTopologyH3Neighborhood(cell, ringOnly) -> u32`: exact `gridDiskDistances` (ring: distance
 *   exactly `K`) including pentagons, sorted by (distance, key).
 * - `cellTopologyH3GetChild(cell, targetResolution, depth, slot) -> vec2u`: the `slot`-th child at resolution
 *   `targetResolution = resolution(cell) + depth` in ascending key order, zero past the child count. Pentagon
 *   center chains skip digit 1, so a pentagon has `(5 * 7^depth + 1) / 6` children.
 * - `cellTopologyH3GetChildCount(cell, depth) -> u32`.
 *
 * @internal
 */
export const H3_TOPOLOGY_WGSL = /* wgsl */ `
fn cellTopologyPow7(exponent: u32) -> u32 {
  var result = 1u;
  for (var i = 0u; i < exponent; i++) {
    result = result * 7u;
  }
  return result;
}

/** Children of a pentagon over \`depth\` levels: (5 * 7^depth + 1) / 6. */
fn cellTopologyH3PentagonChildCount(depth: u32) -> u32 {
  return (5u * cellTopologyPow7(depth) + 1u) / 6u;
}

fn cellTopologyH3GetChildCount(cell: vec2u, depth: u32) -> u32 {
  if (cellTopologyH3IsPentagon(cell)) {
    return cellTopologyH3PentagonChildCount(depth);
  }
  return cellTopologyPow7(depth);
}

fn cellTopologyH3GetChild(cell: vec2u, targetResolution: u32, depth: u32, slot: u32) -> vec2u {
  if (slot >= cellTopologyH3GetChildCount(cell, depth)) {
    return vec2u(0u);
  }
  let resolution = targetResolution - depth;
  var result = dggs_u64_set_bits(cell, 52u, 4u, targetResolution);
  var rank = slot;
  var onPentagonChain = cellTopologyH3IsPentagon(cell);
  for (var level = 1u; level <= depth; level++) {
    let remaining = depth - level + 1u;
    let subtreeSize = cellTopologyPow7(remaining - 1u);
    var digit = 0u;
    if (onPentagonChain) {
      let chainSize = cellTopologyH3PentagonChildCount(remaining - 1u);
      if (rank >= chainSize) {
        rank = rank - chainSize;
        digit = 2u + rank / subtreeSize;
        rank = rank % subtreeSize;
        onPentagonChain = false;
      }
    } else {
      digit = rank / subtreeSize;
      rank = rank % subtreeSize;
    }
    result = cellTopologyH3SetDigit(result, resolution + level, digit);
  }
  return result;
}

fn cellTopologyH3Neighborhood(cell: vec2u, ringOnly: bool) -> u32 {
  cellTopologyCells[0] = cell;
  cellTopologyDistances[0] = 0u;
  var count = 1u;
  var head = 0u;
  loop {
    if (head >= count) {
      break;
    }
    let distance = cellTopologyDistances[head];
    if (distance >= CELL_TOPOLOGY_K) {
      break;
    }
    let current = cellTopologyCells[head];
    head++;
    for (var direction = 1u; direction <= 6u; direction++) {
      let neighbor = cellTopologyH3Neighbor(current, direction);
      if (neighbor.x == 0u && neighbor.y == 0u) {
        continue;
      }
      var seen = false;
      for (var i = 0u; i < count; i++) {
        let other = cellTopologyCells[i];
        if (other.x == neighbor.x && other.y == neighbor.y) {
          seen = true;
          break;
        }
      }
      if (!seen && count < CELL_TOPOLOGY_CAPACITY) {
        cellTopologyCells[count] = neighbor;
        cellTopologyDistances[count] = distance + 1u;
        count++;
      }
    }
  }
  if (ringOnly) {
    var kept = 0u;
    for (var i = 0u; i < count; i++) {
      if (cellTopologyDistances[i] == CELL_TOPOLOGY_K) {
        cellTopologyCells[kept] = cellTopologyCells[i];
        cellTopologyDistances[kept] = CELL_TOPOLOGY_K;
        kept++;
      }
    }
    count = kept;
  }
  cellTopologySortNeighborhood(count);
  return count;
}
`;
