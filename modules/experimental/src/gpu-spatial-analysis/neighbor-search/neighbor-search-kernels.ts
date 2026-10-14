// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {NEIGHBOR_SEARCH_FLOAT_WGSL} from './neighbor-search-wgsl';

/**
 * WGSL shared by the neighbor-search kernels that read `parameters`: the per-frame lattice,
 * point validity and cell lookup. Kernels that include it must bind `parameters` as `array<f32>`.
 *
 * Radius mode derives cells at least `radius` wide (with a 2^-10 relative margin) so the 3x3 cell
 * neighborhood always holds the whole band. kNN mode spreads `gridSize` cells over the bounds and
 * searches expanding rings.
 *
 * With `sphereRadius` (great-circle search) the lattice is in degrees of longitude and latitude, the
 * radius is a great-circle distance in the units of `sphereRadius`, `valid` additionally needs
 * a longitude span of at most 360 degrees, and `periodic` marks a lattice that covers the whole
 * circle of longitudes. Radius-mode cells are at least the angular radius (in degrees) wide.
 *
 * @internal
 */
export function getNeighborSearchLatticeWGSL(
  mode: 'knn' | 'radius',
  gridSize: readonly [number, number],
  sphereRadius?: number
): string {
  const geodesic = sphereRadius !== undefined;
  const geodesicCellSizeWGSL = `let radiusDegrees = min(degrees(radius / SPHERE_RADIUS) * CELL_MARGIN + 0.0001, 360.0);
  lattice.valid = lattice.valid && isFiniteFloat(radius) && isFiniteFloat(radiusDegrees) && radius > 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.radiusDegrees = min(radiusDegrees, 180.0);
  lattice.cellWidth = max(width / f32(COLUMNS), radiusDegrees);
  lattice.cellHeight = max(height / f32(ROWS), radiusDegrees);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.bounded = true;`;
  const geodesicKnnWGSL = `lattice.cellWidth = select(1.0, width / f32(COLUMNS), width > 0.0);
  lattice.cellHeight = select(1.0, height / f32(ROWS), height > 0.0);
  lattice.columns = COLUMNS;
  lattice.rows = ROWS;
  lattice.bounded = isFiniteFloat(radius) && radius > 0.0;`;
  const cellSizeWGSL = geodesic
    ? mode === 'radius'
      ? geodesicCellSizeWGSL
      : geodesicKnnWGSL
    : mode === 'radius'
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
${geodesic ? `const SPHERE_RADIUS: f32 = ${Number(sphereRadius).toPrecision(17)};` : ''}

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
  radius: f32,
  radiusSquared: f32,
  radiusDegrees: f32,
  periodic: bool
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
  lattice.radiusDegrees = 0.0;
  lattice.periodic = false;
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
  lattice.radius = radius;
  lattice.radiusSquared = radiusSquared;
  lattice.valid =
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(width) && isFiniteFloat(height) &&
    width >= 0.0 && height >= 0.0;
  ${
    geodesic
      ? `lattice.valid = lattice.valid && width <= 360.001;
  lattice.periodic = lattice.valid && abs(width - 360.0) <= 0.001;`
      : ''
  }
  if (!lattice.valid) {
    return lattice;
  }
  ${cellSizeWGSL}
  return lattice;
}

// Finite and inside the bounds, independent of whether the radius or bounds form a usable lattice.
fn isPointInBounds(lattice: Lattice, x: f32, y: f32) -> bool {
  return isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY${geodesic ? ' && abs(y) <= 90.0' : ''};
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isPointInBounds(lattice, x, y);
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}

// Distance along one axis from position (in cell queryCell) to the near edge of cell; zero
// for the query cell itself. A lower bound of the distance to any point of that cell, up to the
// rounding slack the callers subtract.
fn getAxisGap(position: f32, minimum: f32, cellSize: f32, queryCell: i32, cell: i32) -> f32 {
  if (cell < queryCell) {
    return position - (minimum + f32(cell + 1) * cellSize);
  }
  if (cell > queryCell) {
    return (minimum + f32(cell) * cellSize) - position;
  }
  return 0.0;
}

${NEIGHBOR_SEARCH_FLOAT_WGSL}
${geodesic ? GEODESIC_WGSL + GEODESIC_WRAPPED_LONGITUDE_BOUND_WGSL : ''}
`;
}

/**
 * WGSL statements that visit every valid target within the 3x3 cell neighborhood of `(x, y)` and
 * inside the distance band, running `action` with `neighbor` (target row) and `distanceSquared`
 * in scope. On coarse lattices, cells whose bounding boxes cannot intersect the distance band are
 * skipped before any targets are loaded. The conservative rounding slack keeps points on cell and
 * radius boundaries.
 * Requires bindings `sortedTargets` (`[x bits, y bits, row]` triples in cell order, `u32`),
 * `cellOffsets` and locals `lattice`, `x`, `y`. The visiting order is fixed but not by ID; callers
 * sort.
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
  if (lattice.cellWidth > lattice.radius * 2.0 || lattice.cellHeight > lattice.radius * 2.0) {
    // Coarse grids can put many irrelevant targets in the neighboring cells. Test each cell's
    // distance lower bound before touching its target range.
    let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
      (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowGap = max(getAxisGap(y, lattice.minimumY, lattice.cellHeight, i32(row), i32(cellRow)) - slack, 0.0);
      let rowGapSquared = rowGap * rowGap;
      let rowBase = cellRow * COLUMNS;
      for (var cellColumn = firstColumn; cellColumn <= lastColumn; cellColumn++) {
        let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, i32(column), i32(cellColumn)) - slack, 0.0);
        if (columnGap * columnGap + rowGapSquared <= lattice.radiusSquared) {
          let cell = rowBase + cellColumn;
          let cellBegin = cellOffsets[cellOffsetsOffset + cell];
          let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
          for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
            let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * 3u + 2u];
            let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u]) - x;
            let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u + 1u]) - y;
            let distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if (distanceSquared <= lattice.radiusSquared) {
              ${action}
            }
          }
        }
      }
    }
  } else {
    // Fine grids already have radius-sized cells. Preserve contiguous reads across each row.
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * COLUMNS;
      let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
        let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * 3u + 2u];
        let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u]) - x;
        let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u + 1u]) - y;
        let distanceSquared = deltaX * deltaX + deltaY * deltaY;
        if (distanceSquared <= lattice.radiusSquared) {
          ${action}
        }
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
 * Requires bindings `sortedTargets` (`[x bits, y bits, row]` triples in cell order, `u32`), `cellOffsets`, locals `lattice`, `x`, `y`, `index`,
 * and a WGSL constant `K`. `selfCondition` filters candidates (for example `neighbor != index`).
 *
 * @internal
 */
export function getNearestNeighborSearchWGSL(selfCondition: string): string {
  const visit = (
    begin: string,
    end: string
  ) => `for (var cellSlot = ${begin}; cellSlot < ${end}; cellSlot++) {
          let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * 3u + 2u];
          if (${selfCondition}) {
            let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u]) - x;
            let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u + 1u]) - y;
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
      let rowBase = u32(cellRow) * COLUMNS;
      // Cell-level pruning once K candidates are held: a cell whose near edges are farther than
      // the k-th best cannot contribute (strict test, so k-th distance ties are still visited).
      let rowGap = max(getAxisGap(y, lattice.minimumY, lattice.cellHeight, queryRow, cellRow) - slack, 0.0);
      let rowGapSquared = rowGap * rowGap;
      if (found == K && bestDistances[K - 1u] < rowGapSquared) {
        continue;
      }
      if (abs(cellRow - queryRow) == ring) {
        // A full row of the ring: only the columns within reach of the k-th best are visited.
        var rangeFirst = firstColumn;
        var rangeEnd = endColumn;
        if (found == K) {
          let reach = sqrt(max(bestDistances[K - 1u] - rowGapSquared, 0.0)) + slack;
          let lowColumn = floor((x - reach - lattice.minimumX) / lattice.cellWidth);
          let highColumn = floor((x + reach - lattice.minimumX) / lattice.cellWidth);
          rangeFirst = max(firstColumn, i32(clamp(lowColumn, -1.0, f32(lastColumn))));
          rangeEnd = min(endColumn, i32(clamp(highColumn, -1.0, f32(lastColumn))));
        }
        if (rangeFirst <= rangeEnd) {
          let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + u32(rangeFirst)];
          let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + u32(rangeEnd) + 1u];
          ${visit('cellBegin', 'cellEnd')}
        }
      } else {
        if (queryColumn - ring >= 0) {
          let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, queryColumn, queryColumn - ring) - slack, 0.0);
          if (!(found == K && bestDistances[K - 1u] < columnGap * columnGap + rowGapSquared)) {
            let cell = rowBase + u32(queryColumn - ring);
            let cellBegin = cellOffsets[cellOffsetsOffset + cell];
            let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
            ${visit('cellBegin', 'cellEnd')}
          }
        }
        if (queryColumn + ring <= lastColumn) {
          let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, queryColumn, queryColumn + ring) - slack, 0.0);
          if (!(found == K && bestDistances[K - 1u] < columnGap * columnGap + rowGapSquared)) {
            let cell = rowBase + u32(queryColumn + ring);
            let cellBegin = cellOffsets[cellOffsetsOffset + cell];
            let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
            ${visit('cellBegin', 'cellEnd')}
          }
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
    // 1 - z from the difference of the distances: for the adaptive k-th neighbor the bandwidth is
    // d_k * 1.0000001, and 1 - d / bandwidth rounds to 0 in f32 for some rows, which would drop the
    // slot (zero weights are not neighbors). The difference stays positive and accurate.
    let oneMinusZ = select(1.0, (bandwidth - distance) / bandwidth, bandwidth > 0.0);
    let kernel = u32(readParameter(8u));
    if (kernel == 0u) {
      weight = exp(-0.5 * z * z) * 0.3989422804014327;
    } else if (kernel == 1u) {
      weight = max(oneMinusZ, 0.0);
    } else if (kernel == 2u) {
      weight = 0.75 * max(oneMinusZ * (1.0 + z), 0.0);
    } else if (kernel == 3u) {
      let base = max(oneMinusZ * (1.0 + z), 0.0);
      weight = 0.9375 * base * base;
    } else {
      weight = 0.5;
    }
  }
  return select(0.0, weight, isFiniteFloat(weight) && weight >= 0.0);
}
`;

/**
 * Great-circle counterpart of {@link getRadiusNeighborLoopWGSL}: visits every valid target whose
 * great-circle distance from `(x, y)` (longitude, latitude in degrees) is at most `lattice.radius`,
 * running `action` with `neighbor` and `distance` (in the units of `SPHERE_RADIUS`) in scope.
 *
 * The band is a spherical cap of angular radius `a`. Its latitude span is `y +- a`. Its longitude
 * half-span is `asin(sin a / cos y)`, the exact extent of a cap, and the full circle when the cap
 * reaches a pole (`|y| + a >= 90`). Both are inflated (relative 2^-10 plus 1e-4 degree) to cover f32
 * cell assignment. The longitude interval is tested against the bounds shifted by -360, 0 and +360
 * degrees, so a band wrapping across +-180 visits the cells on both sides; overlapping column
 * ranges are merged, so no cell is visited twice. Every candidate is tested with the exact
 * haversine distance, so the cell ranges only need to be a superset.
 *
 * Requires the same bindings and locals as {@link getRadiusNeighborLoopWGSL}.
 *
 * @internal
 */
export function getGeodesicRadiusNeighborLoopWGSL(action: string): string {
  return /* wgsl */ `
  let reach = lattice.radiusDegrees;
  let firstRow = getCellRow(lattice, max(y - reach, lattice.minimumY));
  let lastRow = getCellRow(lattice, min(y + reach, lattice.maximumY));
  var reachLongitude = 180.0;
  if (abs(y) + reach < 90.0) {
    let ratio = min(sin(radians(reach)) / geodesicCosLatitude(y), 1.0);
    reachLongitude = min(degrees(asin(ratio)) * CELL_MARGIN + 0.0001, 180.0);
  }
  let queryPoint = vec2<f32>(x, y);
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    let rowBase = cellRow * COLUMNS;
    var nextColumn = 0u;
    for (var shiftIndex = 0u; shiftIndex < 3u; shiftIndex++) {
      var lowLongitude = lattice.minimumX;
      var highLongitude = lattice.maximumX;
      if (reachLongitude < 180.0) {
        let columnOffset = 360.0 * (f32(shiftIndex) - 1.0);
        lowLongitude = max(x - reachLongitude + columnOffset, lattice.minimumX);
        highLongitude = min(x + reachLongitude + columnOffset, lattice.maximumX);
      } else if (shiftIndex != 1u) {
        continue;
      }
      if (lowLongitude > highLongitude) {
        continue;
      }
      let firstColumn = max(getCellColumn(lattice, lowLongitude), nextColumn);
      let lastColumn = getCellColumn(lattice, highLongitude);
      if (firstColumn > lastColumn) {
        continue;
      }
      nextColumn = lastColumn + 1u;
      let cellBegin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let cellEnd = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
        let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * 3u + 2u];
        let targetPoint = vec2<f32>(
          bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u]),
          bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u + 1u]));
        let distance = geodesicHaversineDistance(queryPoint, targetPoint, SPHERE_RADIUS);
        if (distance <= lattice.radius) {
          ${action}
        }
      }
    }
  }`;
}

/**
 * Great-circle counterpart of {@link getNearestNeighborSearchWGSL}: the same expanding rings and
 * `(distance, id)` ordering, but the keys are great-circle distances (not squared) and the stop
 * rule uses a spherical lower bound.
 *
 * Rings run over the degree lattice. A lattice that spans the full 360 degrees (`lattice.periodic`)
 * wraps its columns, so a query next to +-180 meets the other side in ring 1 rather than at the far
 * end; other lattices do not wrap.
 *
 * Stop rule (conservative; a looser bound only costs speed). After ring `r` the visited box has
 * rows `[r0, r1]` and a virtual column arc. Any unvisited target is either
 * (a) outside the box in latitude, so its latitude differs from the query by at least the clearance
 *     `c` to the nearest unvisited row side and its distance is at least `R * c`
 *     (haversine term at least `sin^2(c / 2)`), or
 * (b) inside the box rows but outside the column arc, so its wrapped longitude difference is at least
 *     `L`, the minimum over the unvisited longitude intervals of the sawtooth `min(d, 360 - d)`
 *     (taken at the interval ends, or 0 when an end-to-end interval contains a multiple of 360), and
 *     its latitude has `|latitude| <= m`, the largest absolute latitude of the box rows. The haversine
 *     term is at least `cos(y) cos(m) sin^2(L / 2)` because the dropped latitude term is nonnegative.
 * The smaller of the two terms gives a distance lower bound `2 R asin(sqrt(h))`, shrunk by 2^-13 and
 * with every clearance reduced by an absolute degree slack for f32 cell assignment. The search stops
 * when K candidates are held and the K-th distance is strictly below the bound, so ties at the K-th
 * distance are always resolved by the lowest ID, or when the bound exceeds a finite radius cap.
 *
 * Requires bindings, locals and the constant `K` as {@link getNearestNeighborSearchWGSL}.
 *
 * @internal
 */
export function getGeodesicNearestNeighborSearchWGSL(selfCondition: string): string {
  return /* wgsl */ `
  let queryColumn = i32(getCellColumn(lattice, x));
  let queryRow = i32(getCellRow(lattice, y));
  let columnCount = i32(lattice.columns);
  let lastColumn = columnCount - 1;
  let lastRow = i32(lattice.rows) - 1;
  let periodic = lattice.periodic;
  let lowReach = select(queryColumn, columnCount / 2, periodic);
  let highReach = select(lastColumn - queryColumn, columnCount - 1 - columnCount / 2, periodic);
  let maximumRing = max(lattice.columns, lattice.rows);
  let slack = (lattice.cellWidth + lattice.cellHeight) * 0.0009765625 +
    (abs(lattice.minimumX) + abs(lattice.maximumX) + abs(lattice.minimumY) + abs(lattice.maximumY)) * 0.00000095367431640625;
  let queryPoint = vec2<f32>(x, y);
  let queryCosine = geodesicCosLatitude(y);
  for (var ringIndex = 0u; ringIndex < maximumRing; ringIndex++) {
    let ring = i32(ringIndex);
    let lowExtent = min(ring, lowReach);
    let highExtent = min(ring, highReach);
    let firstRow = max(queryRow - ring, 0);
    let endRow = min(queryRow + ring, lastRow);
    for (var cellRow = firstRow; cellRow <= endRow; cellRow++) {
      let rowBase = u32(cellRow) * COLUMNS;
      let fullRow = abs(cellRow - queryRow) == ring;
      // Full rows of the ring visit every column of the box; other rows only its two sides
      // (a non-full row has ring > 0), each only while it still holds a column not seen before.
      let columnStep = select(2 * ring, 1, fullRow);
      let startOffset = select(-ring, -lowExtent, fullRow);
      let stopOffset = select(ring, highExtent, fullRow);
      for (var columnOffset = startOffset; columnOffset <= stopOffset; columnOffset += columnStep) {
        if (!fullRow && ((columnOffset < 0 && ring > lowReach) || (columnOffset > 0 && ring > highReach))) {
          continue;
        }
        var column = queryColumn + columnOffset;
        if (periodic) {
          column = ((column % columnCount) + columnCount) % columnCount;
        }
        let cell = rowBase + u32(column);
        let cellBegin = cellOffsets[cellOffsetsOffset + cell];
        let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
        for (var cellSlot = cellBegin; cellSlot < cellEnd; cellSlot++) {
          let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * 3u + 2u];
          if (${selfCondition}) {
            let targetPoint = vec2<f32>(
              bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u]),
              bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u + 1u]));
            let distance = geodesicHaversineDistance(queryPoint, targetPoint, SPHERE_RADIUS);
            if (!lattice.bounded || distance <= lattice.radius) {
              insertNeighbor(&bestDistances, &bestIds, &found, distance, neighbor);
            }
          }
        }
      }
    }
    // The visited box: rows [firstRow, endRow], virtual columns [queryColumn - lowExtent, queryColumn + highExtent].
    let hasSouth = firstRow > 0;
    let hasNorth = endRow < lastRow;
    let arcFull = lowExtent + highExtent + 1 >= columnCount;
    let hasWest = select(queryColumn - lowExtent > 0, !arcFull, periodic);
    let hasEast = select(queryColumn + highExtent < lastColumn, !arcFull, periodic);
    if (!(hasSouth || hasNorth || hasWest || hasEast)) {
      break;
    }
    var hLatitude = 1.0;
    if (hasSouth || hasNorth) {
      var clearance = 360.0;
      if (hasSouth) {
        clearance = min(clearance, y - (lattice.minimumY + f32(firstRow) * lattice.cellHeight));
      }
      if (hasNorth) {
        clearance = min(clearance, lattice.minimumY + f32(endRow + 1) * lattice.cellHeight - y);
      }
      let halfSine = sin(0.5 * radians(clamp(clearance - slack, 0.0, 180.0)));
      hLatitude = halfSine * halfSine;
    }
    var hLongitude = 1.0;
    if (hasWest || hasEast) {
      let westEdge = lattice.minimumX + f32(queryColumn - lowExtent) * lattice.cellWidth;
      let eastEdge = lattice.minimumX + f32(queryColumn + highExtent + 1) * lattice.cellWidth;
      var longitudeBound = 360.0;
      if (hasEast) {
        let farEdge = select(lattice.maximumX, westEdge + 360.0, periodic);
        longitudeBound = min(longitudeBound, getWrappedLongitudeBound(eastEdge - x - slack, farEdge - x + slack));
      }
      if (hasWest) {
        let farEdge = select(lattice.minimumX, eastEdge - 360.0, periodic);
        longitudeBound = min(longitudeBound, getWrappedLongitudeBound(x - westEdge - slack, x - farEdge + slack));
      }
      let southLatitude = lattice.minimumY + f32(firstRow) * lattice.cellHeight;
      let northLatitude = min(lattice.minimumY + f32(endRow + 1) * lattice.cellHeight, lattice.maximumY);
      let maximumLatitude = clamp(max(abs(southLatitude), abs(northLatitude)) + slack, 0.0, 90.0);
      let halfSine = sin(0.5 * radians(clamp(longitudeBound, 0.0, 180.0)));
      hLongitude = queryCosine * geodesicCosLatitude(maximumLatitude) * halfSine * halfSine;
    }
    let boundTerm = clamp(min(hLatitude, hLongitude), 0.0, 1.0);
    let safeDistance = 2.0 * SPHERE_RADIUS * asin(sqrt(boundTerm)) * 0.9998779296875;
    if (found == K && bestDistances[K - 1u] < safeDistance) {
      break;
    }
    if (lattice.bounded && lattice.radius < safeDistance) {
      break;
    }
  }`;
}

/**
 * WGSL helper `getWrappedLongitudeBound(lowValue, highValue)` of {@link getGeodesicNearestNeighborSearchWGSL}:
 * the smallest value of the wrapped longitude difference `min(d, 360 - d)` for `d` in `[lowValue, highValue]`
 * degrees (`lowValue` is clamped at 0).
 *
 * @internal
 */
export const GEODESIC_WRAPPED_LONGITUDE_BOUND_WGSL = /* wgsl */ `
fn getWrappedLongitudeBound(lowValue: f32, highValue: f32) -> f32 {
  let lowEnd = max(lowValue, 0.0);
  if (ceil(lowEnd / 360.0) * 360.0 <= highValue) {
    return 0.0;
  }
  return min(abs(lowEnd - 360.0 * round(lowEnd / 360.0)), abs(highValue - 360.0 * round(highValue / 360.0)));
}
`;

/** `u32` words per BVH node in the packed `nodeData` table: minimum xy, maximum xy, leaf row. @internal */
export const NEIGHBOR_SEARCH_BVH_NODE_WORDS = 5;

/**
 * Former experimental auto-selection threshold. Retained for source compatibility only;
 * `searchStructure: 'auto'` now uses the grid because neither measured distribution favored BVH.
 * @deprecated No longer used for structure selection.
 */
export const GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD = 32;

/** Compile-time shape of the BVH the traversal WGSL walks. @internal */
export type NeighborSearchBVHShape = {
  /** Number of target rows (leaf rows at or above it are padding). */
  targetRows: number;
  /** Number of internal nodes, `leafCapacity - 1`; node `n` has children `2n + 1` and `2n + 2`. */
  internalNodeCount: number;
  /** Levels of the complete tree; sizes the traversal stack. */
  levelCount: number;
};

/**
 * Declarations of the BVH traversal WGSL: `getBVHBoxBound` (squared distance from `(x, y)` to a
 * node box, or -1 for an empty box) and the pruning slack. Requires a `nodeData` binding
 * (`u32` words, see {@link NEIGHBOR_SEARCH_BVH_NODE_WORDS}), so a kernel needs one storage buffer
 * for the whole tree.
 *
 * A point leaf has a degenerate box, so its bound is exactly the `dx^2 + dy^2` the grid path
 * computes. Float rounding is monotone, so an ancestor's bound never exceeds the computed
 * distance of a point inside its box; the slack only absorbs a compiler fusing the multiply-add.
 *
 * @internal
 */
export function getBVHNeighborSearchDeclarationsWGSL(shape: NeighborSearchBVHShape): string {
  return /* wgsl */ `
const BVH_NODE_WORDS: u32 = ${NEIGHBOR_SEARCH_BVH_NODE_WORDS}u;
const BVH_TARGET_ROWS: u32 = ${shape.targetRows}u;
const BVH_INTERNAL_NODE_COUNT: u32 = ${shape.internalNodeCount}u;
const BVH_STACK_SIZE: u32 = ${shape.levelCount + 2}u;
const BVH_BOUND_SLACK: f32 = 1.000001;

fn getBVHBoxBound(node: u32, x: f32, y: f32) -> f32 {
  let base = nodeDataOffset + node * BVH_NODE_WORDS;
  let minimumX = bitcast<f32>(nodeData[base]);
  let minimumY = bitcast<f32>(nodeData[base + 1u]);
  let maximumX = bitcast<f32>(nodeData[base + 2u]);
  let maximumY = bitcast<f32>(nodeData[base + 3u]);
  if (minimumX > maximumX || minimumY > maximumY) {
    return -1.0;
  }
  let gapX = max(max(minimumX - x, x - maximumX), 0.0);
  let gapY = max(max(minimumY - y, y - maximumY), 0.0);
  return gapX * gapX + gapY * gapY;
}
`;
}

/**
 * WGSL statements that map the invocation to its query row `queryIndex`. A self join walks the
 * queries in BVH leaf order, which is the Hilbert order of the points, so neighboring threads
 * search neighboring subtrees; a cross join uses the invocation index. Returns for padding slots.
 *
 * @internal
 */
export function getBVHThreadToRowWGSL(crossJoin: boolean): string {
  return crossJoin
    ? 'let queryIndex = index;'
    : `let queryIndex = nodeData[nodeDataOffset + (BVH_INTERNAL_NODE_COUNT + index) * BVH_NODE_WORDS + 4u];
  if (queryIndex >= BVH_TARGET_ROWS) {
    return;
  }`;
}

/**
 * WGSL statements of the BVH k-nearest search around `(x, y)`: near-first depth-first traversal that pops the
 * nearer child first and prunes every node whose box bound exceeds the k-th best squared distance
 * (widened by `BVH_BOUND_SLACK`, with a strict test, so ties at the k-th distance are visited and
 * resolved by the lowest ID) or the radius limit. Candidates enter the private top-`K` list with
 * `insertNeighbor`, the `(d^2, id)` order of the grid path, so the selected set is the same.
 *
 * Same contract as {@link getNearestNeighborSearchWGSL}: arrays `bestDistances` / `bestIds`, `found`,
 * `lattice`, `x`, `y`, `queryIndex`, `K`; `selfCondition` filters `neighbor`.
 *
 * @internal
 */
export function getBVHNearestNeighborSearchWGSL(selfCondition: string): string {
  return /* wgsl */ `
  var stackNode: array<u32, BVH_STACK_SIZE>;
  var stackBound: array<f32, BVH_STACK_SIZE>;
  var top = 0u;
  let rootBound = getBVHBoxBound(0u, x, y);
  if (rootBound >= 0.0) {
    stackNode[0] = 0u;
    stackBound[0] = rootBound;
    top = 1u;
  }
  loop {
    if (top == 0u) {
      break;
    }
    top = top - 1u;
    let node = stackNode[top];
    var limited = lattice.bounded;
    var limit = lattice.radiusSquared * BVH_BOUND_SLACK;
    if (found == K) {
      limit = select(bestDistances[K - 1u] * BVH_BOUND_SLACK,
        min(limit, bestDistances[K - 1u] * BVH_BOUND_SLACK), limited);
      limited = true;
    }
    if (limited && stackBound[top] > limit) {
      continue;
    }
    if (node < BVH_INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstBound = getBVHBoxBound(first, x, y);
      let secondBound = getBVHBoxBound(second, x, y);
      let firstUsable = firstBound >= 0.0 && !(limited && firstBound > limit);
      let secondUsable = secondBound >= 0.0 && !(limited && secondBound > limit);
      // Push the farther child first so the nearer one is popped first.
      if (firstUsable && secondUsable) {
        if (firstBound <= secondBound) {
          stackNode[top] = second;
          stackBound[top] = secondBound;
          stackNode[top + 1u] = first;
          stackBound[top + 1u] = firstBound;
        } else {
          stackNode[top] = first;
          stackBound[top] = firstBound;
          stackNode[top + 1u] = second;
          stackBound[top + 1u] = secondBound;
        }
        top = top + 2u;
      } else if (firstUsable) {
        stackNode[top] = first;
        stackBound[top] = firstBound;
        top = top + 1u;
      } else if (secondUsable) {
        stackNode[top] = second;
        stackBound[top] = secondBound;
        top = top + 1u;
      }
      continue;
    }
    let base = nodeDataOffset + node * BVH_NODE_WORDS;
    let neighbor = nodeData[base + 4u];
    if (neighbor >= BVH_TARGET_ROWS || !(${selfCondition})) {
      continue;
    }
    let deltaX = bitcast<f32>(nodeData[base]) - x;
    let deltaY = bitcast<f32>(nodeData[base + 1u]) - y;
    let distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (!lattice.bounded || distanceSquared <= lattice.radiusSquared) {
      insertNeighbor(&bestDistances, &bestIds, &found, distanceSquared, neighbor);
    }
  }`;
}

/**
 * WGSL statements that visit every valid target within the radius of `(x, y)` through the BVH,
 * running `action` with `neighbor` and `distanceSquared` in scope, like
 * {@link getRadiusNeighborLoopWGSL}. The visiting order is fixed but not by ID; callers sort.
 * Requires locals `lattice`, `x`, `y`, `queryIndex`.
 *
 * @internal
 */
export function getBVHRadiusNeighborLoopWGSL(action: string): string {
  return /* wgsl */ `
  var stackNode: array<u32, BVH_STACK_SIZE>;
  var top = 0u;
  let limit = lattice.radiusSquared * BVH_BOUND_SLACK;
  let rootBound = getBVHBoxBound(0u, x, y);
  if (rootBound >= 0.0 && rootBound <= limit) {
    stackNode[0] = 0u;
    top = 1u;
  }
  loop {
    if (top == 0u) {
      break;
    }
    top = top - 1u;
    let node = stackNode[top];
    if (node < BVH_INTERNAL_NODE_COUNT) {
      let first = node * 2u + 1u;
      let second = first + 1u;
      let firstBound = getBVHBoxBound(first, x, y);
      let secondBound = getBVHBoxBound(second, x, y);
      if (firstBound >= 0.0 && firstBound <= limit) {
        stackNode[top] = first;
        top = top + 1u;
      }
      if (secondBound >= 0.0 && secondBound <= limit) {
        stackNode[top] = second;
        top = top + 1u;
      }
      continue;
    }
    let base = nodeDataOffset + node * BVH_NODE_WORDS;
    let neighbor = nodeData[base + 4u];
    if (neighbor >= BVH_TARGET_ROWS) {
      continue;
    }
    let deltaX = bitcast<f32>(nodeData[base]) - x;
    let deltaY = bitcast<f32>(nodeData[base + 1u]) - y;
    let distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (distanceSquared <= lattice.radiusSquared) {
      ${action}
    }
  }`;
}
