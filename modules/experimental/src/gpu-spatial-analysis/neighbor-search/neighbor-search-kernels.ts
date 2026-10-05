// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';

/**
 * WGSL shared by the neighbor-search kernels that read `parameters`: the per-frame lattice,
 * point validity and cell lookup. Kernels that include it must bind `parameters` as `array<f32>`.
 *
 * Radius mode derives cells at least `radius` wide (with a 2^-10 relative margin) so the 3x3 cell
 * neighborhood always holds the whole band. kNN mode spreads `gridSize` cells over the bounds and
 * searches expanding rings.
 *
 * @internal
 */
export function getNeighborSearchLatticeWGSL(
  mode: 'knn' | 'radius',
  gridSize: readonly [number, number]
): string {
  const cellSizeWGSL =
    mode === 'radius'
      ? `let cellRadius = radius * CELL_MARGIN;
  lattice.valid = lattice.valid && isFiniteFloat(radius) && isFiniteFloat(radiusSquared) &&
    isFiniteFloat(cellRadius) && radius > 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.cellWidth = max(width / f32(COLUMNS), cellRadius);
  lattice.cellHeight = max(height / f32(ROWS), cellRadius);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.bounded = true;`
      : `lattice.cellWidth = select(1.0, width / f32(COLUMNS), width > 0.0);
  lattice.cellHeight = select(1.0, height / f32(ROWS), height > 0.0);
  lattice.columns = COLUMNS;
  lattice.rows = ROWS;
  lattice.bounded = isFiniteFloat(radius) && isFiniteFloat(radiusSquared) && radius > 0.0;`;
  return /* wgsl */ `
const COLUMNS: u32 = ${gridSize[0]}u;
const ROWS: u32 = ${gridSize[1]}u;
const CELL_COUNT: u32 = ${gridSize[0] * gridSize[1]}u;
const CELL_MARGIN: f32 = 1.0009765625;

struct Lattice {
  valid: bool,
  bounded: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  radiusSquared: f32
}

fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.bounded = false;
  let minimumX = readParameter(0u);
  let minimumY = readParameter(1u);
  let maximumX = readParameter(2u);
  let maximumY = readParameter(3u);
  let radius = readParameter(4u);
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  let radiusSquared = radius * radius;
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.radiusSquared = radiusSquared;
  lattice.valid =
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(width) && isFiniteFloat(height) &&
    width >= 0.0 && height >= 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  ${cellSizeWGSL}
  return lattice;
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY;
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}

${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
`;
}

/**
 * WGSL statements that visit every valid target within the 3x3 cell neighborhood of `(x, y)` and
 * inside the distance band, running `action` with `neighbor` (target row) and `distanceSquared`
 * in scope. Requires bindings `positions`, `sortedRows`, `cellOffsets` and locals `lattice`, `x`,
 * `y`. The visiting order is fixed but not by ID; callers sort.
 *
 * @internal
 */
export function getRadiusNeighborLoopWGSL(action: string): string {
  return /* wgsl */ `
  let column = getCellColumn(lattice, x);
  let row = getCellRow(lattice, y);
  let firstColumn = max(column, 1u) - 1u;
  let lastColumn = min(column + 1u, lattice.columns - 1u);
  let firstRow = max(row, 1u) - 1u;
  let lastRow = min(row + 1u, lattice.rows - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    let rowBase = cellRow * lattice.columns;
    let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
    let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
    for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
      let neighbor = sortedRows[sortedRowsOffset + cellSlot];
      let deltaX = positions[positionsOffset + neighbor * 2u] - x;
      let deltaY = positions[positionsOffset + neighbor * 2u + 1u] - y;
      let distanceSquared = deltaX * deltaX + deltaY * deltaY;
      if (distanceSquared <= lattice.radiusSquared) {
        ${action}
      }
    }
  }`;
}

/**
 * WGSL statements of the kNN expanding-ring search around `(x, y)`.
 *
 * Keeps the `K` best `(distanceSquared, id)` pairs in ascending lexicographic order in the private
 * arrays `bestDistances` / `bestIds` with `found` entries. Ring `r` visits the cells at Chebyshev
 * distance `r` from the query cell. After each ring the search stops when no unvisited target can
 * beat the current k-th best: every unvisited target lies outside the visited box, so its distance
 * is at least the box's clearance `g` (an infinite side where the box reaches the lattice edge).
 * `g` is shrunk by an absolute slack covering f32 rounding of cell assignment, and the test is
 * strict, so ties at the k-th distance are always fully resolved by the lowest ID.
 *
 * Requires bindings `positions`, `sortedRows`, `cellOffsets`, locals `lattice`, `x`, `y`, `index`,
 * and a WGSL constant `K`. `selfCondition` filters candidates (for example `neighbor != index`).
 *
 * @internal
 */
export function getNearestNeighborSearchWGSL(selfCondition: string): string {
  const visit = (
    begin: string,
    end: string
  ) => `for (var cellSlot = ${begin}; cellSlot < ${end}; cellSlot++) {
          let neighbor = sortedRows[sortedRowsOffset + cellSlot];
          if (${selfCondition}) {
            let deltaX = positions[positionsOffset + neighbor * 2u] - x;
            let deltaY = positions[positionsOffset + neighbor * 2u + 1u] - y;
            let distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if (!lattice.bounded || distanceSquared <= lattice.radiusSquared) {
              insertNeighbor(&bestDistances, &bestIds, &found, distanceSquared, neighbor);
            }
          }
        }`;
  return /* wgsl */ `
  let queryColumn = i32(getCellColumn(lattice, x));
  let queryRow = i32(getCellRow(lattice, y));
  let lastColumn = i32(lattice.columns) - 1;
  let lastRow = i32(lattice.rows) - 1;
  let maximumRing = max(lattice.columns, lattice.rows);
  let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
    (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
  for (var ringIndex = 0u; ringIndex < maximumRing; ringIndex++) {
    let ring = i32(ringIndex);
    let firstColumn = max(queryColumn - ring, 0);
    let endColumn = min(queryColumn + ring, lastColumn);
    let firstRow = max(queryRow - ring, 0);
    let endRow = min(queryRow + ring, lastRow);
    for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
      let rowBase = u32(cellRow) * lattice.columns;
      if (abs(cellRow - queryRow) == ring) {
        let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + u32(firstColumn)];
        let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + u32(endColumn) + 1u];
        ${visit('cellBegin', 'cellEnd')}
      } else {
        if (queryColumn - ring >= 0) {
          let cell = rowBase + u32(queryColumn - ring);
          let cellBegin = cellOffsets[cellOffsetsOffset + cell];
          let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
          ${visit('cellBegin', 'cellEnd')}
        }
        if (queryColumn + ring <= lastColumn) {
          let cell = rowBase + u32(queryColumn + ring);
          let cellBegin = cellOffsets[cellOffsetsOffset + cell];
          let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
          ${visit('cellBegin', 'cellEnd')}
        }
      }
    }
    // Clearance of the visited box on each side; a side at the lattice edge has nothing beyond it.
    var clearance = 3.0e38;
    var open = false;
    if (queryColumn - ring > 0) {
      clearance = min(clearance, x - (lattice.minimumX + f32(queryColumn - ring) * lattice.cellWidth));
      open = true;
    }
    if (queryColumn + ring < lastColumn) {
      clearance = min(clearance, lattice.minimumX + f32(queryColumn + ring + 1) * lattice.cellWidth - x);
      open = true;
    }
    if (queryRow - ring > 0) {
      clearance = min(clearance, y - (lattice.minimumY + f32(queryRow - ring) * lattice.cellHeight));
      open = true;
    }
    if (queryRow + ring < lastRow) {
      clearance = min(clearance, lattice.minimumY + f32(queryRow + ring + 1) * lattice.cellHeight - y);
      open = true;
    }
    if (!open) {
      break;
    }
    let safeClearance = max(clearance - slack, 0.0);
    let safeSquared = safeClearance * safeClearance;
    if (found == K && bestDistances[K - 1u] < safeSquared) {
      break;
    }
    if (lattice.bounded && lattice.radiusSquared < safeSquared) {
      break;
    }
  }`;
}

/**
 * WGSL helper `insertNeighbor` for {@link getNearestNeighborSearchWGSL}: inserts a candidate into
 * the private top-`K` arrays ordered by `(distanceSquared, id)`.
 *
 * @internal
 */
export function getInsertNeighborWGSL(k: number): string {
  return /* wgsl */ `
const K: u32 = ${k}u;

fn isNeighborBefore(leftDistance: f32, leftId: u32, rightDistance: f32, rightId: u32) -> bool {
  return leftDistance < rightDistance || (leftDistance == rightDistance && leftId < rightId);
}

fn insertNeighbor(
  bestDistances: ptr<function, array<f32, ${k}>>,
  bestIds: ptr<function, array<u32, ${k}>>,
  found: ptr<function, u32>,
  distanceSquared: f32,
  neighbor: u32
) {
  if (*found == K &&
      !isNeighborBefore(distanceSquared, neighbor, (*bestDistances)[K - 1u], (*bestIds)[K - 1u])) {
    return;
  }
  var slot = min(*found, K - 1u);
  loop {
    if (slot == 0u) {
      break;
    }
    let previous = slot - 1u;
    if (isNeighborBefore((*bestDistances)[previous], (*bestIds)[previous], distanceSquared, neighbor)) {
      break;
    }
    (*bestDistances)[slot] = (*bestDistances)[previous];
    (*bestIds)[slot] = (*bestIds)[previous];
    slot = previous;
  }
  (*bestDistances)[slot] = distanceSquared;
  (*bestIds)[slot] = neighbor;
  *found = min(*found + 1u, K);
}
`;
}

/**
 * WGSL helper `getNeighborWeight(distance, bandwidth)` for the per-frame weight kind, inverse
 * distance exponent, floor and kernel profile in parameter slots 5 to 8. Non-finite weights are
 * written as 0. Requires `readParameter`.
 *
 * @internal
 */
export const NEIGHBOR_WEIGHT_WGSL = /* wgsl */ `
fn getNeighborWeight(distance: f32, bandwidth: f32) -> f32 {
  let weightKind = u32(readParameter(5u));
  var weight = 1.0;
  if (weightKind == 1u) {
    weight = pow(max(distance, readParameter(7u)), -readParameter(6u));
  } else if (weightKind == 2u) {
    let z = select(0.0, distance / bandwidth, bandwidth > 0.0);
    let kernel = u32(readParameter(8u));
    if (kernel == 0u) {
      weight = exp(-0.5 * z * z) * 0.3989422804014327;
    } else if (kernel == 1u) {
      weight = max(1.0 - z, 0.0);
    } else if (kernel == 2u) {
      weight = 0.75 * max(1.0 - z * z, 0.0);
    } else if (kernel == 3u) {
      let base = max(1.0 - z * z, 0.0);
      weight = 0.9375 * base * base;
    } else {
      weight = 0.5;
    }
  }
  return select(0.0, weight, isFiniteFloat(weight) && weight >= 0.0);
}
`;
