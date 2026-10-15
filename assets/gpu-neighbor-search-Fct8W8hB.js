import{D as e,E as t,S as n}from"./analysis-frame-Blu0tYdw.js";import{t as r}from"./parameter-schema-DB2zG6AH.js";import{_ as i,c as a,d as o,h as s,o as c,v as ee,w as l,y as u}from"./resources-CwvhEOBe.js";import{t as d}from"./gpu-scan-BuGhTLDQ.js";import{t as f}from"./gpu-grid-index-D1JNO067.js";import{l as p,s as m}from"./wgsl-kernel-nodes-Bm5c4bxp.js";import{c as h}from"./wgsl-common-BNRVzgd8.js";import{l as g,t as _}from"./spatial-weights-D4bd9AQ6.js";import{n as te}from"./gpu-spatial-weights-transform-B3cihvVH.js";import{t as v}from"./geodesic-wgsl-DS0CJZ42.js";import{t as ne}from"./nearest-nodes-DBRmVX1E.js";var y=h;function re(e,t,n){let r=n!==void 0,i=r?e===`radius`?`let radiusDegrees = min(degrees(radius / SPHERE_RADIUS) * CELL_MARGIN + 0.0001, 360.0);
  lattice.valid = lattice.valid && isFiniteFloat(radius) && isFiniteFloat(radiusDegrees) && radius > 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.radiusDegrees = min(radiusDegrees, 180.0);
  lattice.cellWidth = max(width / f32(COLUMNS), radiusDegrees);
  lattice.cellHeight = max(height / f32(ROWS), radiusDegrees);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.bounded = true;`:`lattice.cellWidth = select(1.0, width / f32(COLUMNS), width > 0.0);
  lattice.cellHeight = select(1.0, height / f32(ROWS), height > 0.0);
  lattice.columns = COLUMNS;
  lattice.rows = ROWS;
  lattice.bounded = isFiniteFloat(radius) && radius > 0.0;`:e===`radius`?`let cellRadius = radius * CELL_MARGIN;
  lattice.valid = lattice.valid && isFiniteFloat(radius) && isFiniteFloat(radiusSquared) &&
    isFiniteFloat(cellRadius) && radius > 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.cellWidth = max(width / f32(COLUMNS), cellRadius);
  lattice.cellHeight = max(height / f32(ROWS), cellRadius);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.bounded = true;`:`lattice.cellWidth = select(1.0, width / f32(COLUMNS), width > 0.0);
  lattice.cellHeight = select(1.0, height / f32(ROWS), height > 0.0);
  lattice.columns = COLUMNS;
  lattice.rows = ROWS;
  lattice.bounded = isFiniteFloat(radius) && isFiniteFloat(radiusSquared) && radius > 0.0;`;return`
const COLUMNS: u32 = ${t[0]}u;
const ROWS: u32 = ${t[1]}u;
const CELL_COUNT: u32 = ${t[0]*t[1]}u;
const CELL_MARGIN: f32 = 1.0009765625;
${r?`const SPHERE_RADIUS: f32 = ${Number(n).toPrecision(17)};`:``}

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
  ${r?`lattice.valid = lattice.valid && width <= 360.001;
  lattice.periodic = lattice.valid && abs(width - 360.0) <= 0.001;`:``}
  if (!lattice.valid) {
    return lattice;
  }
  ${i}
  return lattice;
}

// Finite and inside the bounds, independent of whether the radius or bounds form a usable lattice.
fn isPointInBounds(lattice: Lattice, x: f32, y: f32) -> bool {
  return isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY${r?` && abs(y) <= 90.0`:``};
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

${y}
${r?v+b:``}
`}function ie(e){return`
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
              ${e}
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
          ${e}
        }
      }
    }
  }`}function ae(e){let t=(t,n)=>`for (var cellSlot = ${t}; cellSlot < ${n}; cellSlot++) {
          let neighbor = sortedTargets[sortedTargetsOffset + cellSlot * 3u + 2u];
          if (${e}) {
            let deltaX = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u]) - x;
            let deltaY = bitcast<f32>(sortedTargets[sortedTargetsOffset + cellSlot * 3u + 1u]) - y;
            let distanceSquared = deltaX * deltaX + deltaY * deltaY;
            if (!lattice.bounded || distanceSquared <= lattice.radiusSquared) {
              insertNeighbor(&bestDistances, &bestIds, &found, distanceSquared, neighbor);
            }
          }
        }`;return`
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
          ${t(`cellBegin`,`cellEnd`)}
        }
      } else {
        if (queryColumn - ring >= 0) {
          let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, queryColumn, queryColumn - ring) - slack, 0.0);
          if (!(found == K && bestDistances[K - 1u] < columnGap * columnGap + rowGapSquared)) {
            let cell = rowBase + u32(queryColumn - ring);
            let cellBegin = cellOffsets[cellOffsetsOffset + cell];
            let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
            ${t(`cellBegin`,`cellEnd`)}
          }
        }
        if (queryColumn + ring <= lastColumn) {
          let columnGap = max(getAxisGap(x, lattice.minimumX, lattice.cellWidth, queryColumn, queryColumn + ring) - slack, 0.0);
          if (!(found == K && bestDistances[K - 1u] < columnGap * columnGap + rowGapSquared)) {
            let cell = rowBase + u32(queryColumn + ring);
            let cellBegin = cellOffsets[cellOffsetsOffset + cell];
            let cellEnd = cellOffsets[cellOffsetsOffset + cell + 1u];
            ${t(`cellBegin`,`cellEnd`)}
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
  }`}function oe(e){return`
const K: u32 = ${e}u;

fn isNeighborBefore(leftDistance: f32, leftId: u32, rightDistance: f32, rightId: u32) -> bool {
  return leftDistance < rightDistance || (leftDistance == rightDistance && leftId < rightId);
}

fn insertNeighbor(
  bestDistances: ptr<function, array<f32, ${e}>>,
  bestIds: ptr<function, array<u32, ${e}>>,
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
`}var se=`
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
`;function ce(e){return`
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
          ${e}
        }
      }
    }
  }`}function le(e){return`
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
          if (${e}) {
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
  }`}var b=`
fn getWrappedLongitudeBound(lowValue: f32, highValue: f32) -> f32 {
  let lowEnd = max(lowValue, 0.0);
  if (ceil(lowEnd / 360.0) * 360.0 <= highValue) {
    return 0.0;
  }
  return min(abs(lowEnd - 360.0 * round(lowEnd / 360.0)), abs(highValue - 360.0 * round(highValue / 360.0)));
}
`;function ue(e){return`
const BVH_NODE_WORDS: u32 = 5u;
const BVH_TARGET_ROWS: u32 = ${e.targetRows}u;
const BVH_INTERNAL_NODE_COUNT: u32 = ${e.internalNodeCount}u;
const BVH_STACK_SIZE: u32 = ${e.levelCount+2}u;
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
`}function de(e){return e?`let queryIndex = index;`:`let queryIndex = nodeData[nodeDataOffset + (BVH_INTERNAL_NODE_COUNT + index) * BVH_NODE_WORDS + 4u];
  if (queryIndex >= BVH_TARGET_ROWS) {
    return;
  }`}function fe(e){return`
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
    if (neighbor >= BVH_TARGET_ROWS || !(${e})) {
      continue;
    }
    let deltaX = bitcast<f32>(nodeData[base]) - x;
    let deltaY = bitcast<f32>(nodeData[base + 1u]) - y;
    let distanceSquared = deltaX * deltaX + deltaY * deltaY;
    if (!lattice.bounded || distanceSquared <= lattice.radiusSquared) {
      insertNeighbor(&bestDistances, &bestIds, &found, distanceSquared, neighbor);
    }
  }`}function pe(e){return`
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
      ${e}
    }
  }`}r({id:`neighbor-search`,format:`float32`,wordLength:12,fields:[{name:`minimumX`,format:`float32`,wordOffset:0,defaultValue:0,dynamic:!0},{name:`minimumY`,format:`float32`,wordOffset:1,defaultValue:0,dynamic:!0},{name:`maximumX`,format:`float32`,wordOffset:2,defaultValue:0,dynamic:!0},{name:`maximumY`,format:`float32`,wordOffset:3,defaultValue:0,dynamic:!0},{name:`radius`,format:`float32`,wordOffset:4,defaultValue:1/0,allowNonFinite:!0,units:`position-units`,dynamic:!0},{name:`weightKind`,format:`float32`,wordOffset:5,defaultValue:0,dynamic:!0},{name:`power`,format:`float32`,wordOffset:6,defaultValue:1,dynamic:!0},{name:`distanceFloor`,format:`float32`,wordOffset:7,defaultValue:0,minimum:0,units:`position-units`,dynamic:!0},{name:`kernel`,format:`float32`,wordOffset:8,defaultValue:1,dynamic:!0},{name:`rowStandardize`,format:`float32`,wordOffset:9,defaultValue:0,minimum:0,maximum:1,dynamic:!0}]});var x={binary:0,inverseDistance:1,kernel:2},S={gaussian:0,triangular:1,epanechnikov:2,bisquare:3,uniform:4};function C(t){let n=new Float32Array(12),[r,i,a,o]=t.bounds,s=t.weightKind??`binary`,c=t.kernel??`triangular`;if(!(s in x))throw Error(`GPUNeighborSearch unknown weightKind ${s}`);if(!(c in S))throw Error(`GPUNeighborSearch unknown kernel ${c}`);return e(`GPUNeighborSearch`,t.spatialContext,t.radius),n[0]=r,n[1]=i,n[2]=a,n[3]=o,n[4]=t.radius??1/0,n[5]=x[s],n[6]=t.power??1,n[7]=t.distanceFloor??0,n[8]=S[c],n[9]=t.rowStandardize?1:0,n}var w=`GPUNeighborSearch`,T=Object.freeze({coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metricsByCoordinateSpace:Object.freeze({planar:Object.freeze([`none`,`native`]),"longitude-latitude":Object.freeze([`great-circle`])}),measure:`distance`}),E=class{id;props;spatialContext;outputWeights;searchStructure;constructor(e){let n=e.id??`neighbor-search`;if(this.id=n,this.props=e,this.spatialContext=t(n,e.spatialContext,T),e.mode!==`knn`&&e.mode!==`radius`)throw Error(`${n} mode must be 'knn' or 'radius'`);let r=e.searchStructure??`auto`;if(![`auto`,`grid`,`bvh`].includes(r))throw Error(`${n} searchStructure must be 'auto', 'grid' or 'bvh'`);if(r===`bvh`&&this.spatialContext.coordinateSpace===`longitude-latitude`)throw Error(`${n} searchStructure 'bvh' supports planar positions only`);this.searchStructure=this.spatialContext.coordinateSpace===`longitude-latitude`?`grid`:r,u(e.positions,[`float32x2`],`${n} positions`);let i=e.positions.length;o(n,`positions length`,i,1,2**31-1),e.queryPositions&&(u(e.queryPositions,[`float32x2`],`${n} queryPositions`),o(n,`queryPositions length`,e.queryPositions.length,1,2**31-1));let a=this.getQueryRows();if(u(e.parameters,[`float32`],`${n} parameters`),e.parameters.length<12)throw Error(`${n} parameters must hold 12 float32 values`);let[c,ee]=e.gridSize;if(o(n,`gridSize columns`,c,1,2**53-1),o(n,`gridSize rows`,ee,1,2**53-1),e.mode===`knn`&&o(n,`k`,e.k??NaN,1,32),e.mask&&(u(e.mask,[`uint32`],`${n} mask`),e.mask.length!==i))throw Error(`${n} mask length must equal positions length`);if(e.queryMask&&(u(e.queryMask,[`uint32`],`${n} queryMask`),e.queryMask.length!==a))throw Error(`${n} queryMask length must equal the query row count`);let{output:l}=e;if(g(n,l)!==a)throw Error(`${n} output.offsets length must equal the query row count + 1`);let{requiredCount:d,invalidCount:f}=l.status;for(let[e,t]of[[`output.status.overflow`,l.status.overflow],[`output.status.requiredCount`,d],[`output.status.invalidCount`,f]])if(t&&(u(t,[`uint32`],`${n} ${e}`),t.length<1))throw Error(`${n} ${e} must hold one uint32`);if(l.neighborCounts&&(u(l.neighborCounts,[`uint32`],`${n} output.neighborCounts`),l.neighborCounts.length!==a))throw Error(`${n} output.neighborCounts length must equal the query row count`);if(e.weightTransform!==void 0&&![`binary`,`raw`,`kernel`,`row`].includes(e.weightTransform))throw Error(`${n} weightTransform must be 'binary', 'raw', 'kernel' or 'row'`);if(e.rowStandardize!==void 0&&e.weightTransform!==void 0&&e.rowStandardize!==(e.weightTransform===`row`))throw Error(`${n} rowStandardize ${e.rowStandardize} conflicts with weightTransform '${e.weightTransform}'`);let p=e.rowStandardize?`row`:e.weightTransform??`unknown`,m=this.spatialContext.coordinateSpace===`planar`&&e.mode===`radius`&&!e.queryPositions&&!e.queryMask;this.outputWeights=_(l,{spatialContext:this.spatialContext,information:{rowCount:a,transform:p,symmetric:m&&p!==`unknown`&&p!==`row`?!0:`unknown`,hasSelfWeights:!1,provenance:{source:w,criterion:e.mode===`knn`?`knn`:`distance-band`,k:e.mode===`knn`?e.k:void 0}}}),s(n,[f,l.offsets,l.neighbors,l.weights,l.distances,l.status.overflow,d,l.neighborCounts],[e.positions,e.queryPositions,e.mask,e.queryMask,e.parameters])}getQueryRows(){return(this.props.queryPositions??this.props.positions).length}getCommandNodes(e){let{id:t,props:r}=this,{positions:o,queryPositions:s,parameters:u,gridSize:h,output:g,mode:_}=r;i(t,e,[o,s,r.mask,r.queryMask,u,g.offsets,g.neighbors,g.weights,g.distances,g.status.overflow,g.status.requiredCount,g.status.invalidCount,g.neighborCounts]),ee(t,`gridSize cells`,h[0]*h[1],c(e.device));let v=o.length,y=this.getQueryRows(),b=g.neighbors.length,x=h[0]*h[1],S=r.k??1,{spatialContext:C}=this,T=C.coordinateSpace===`longitude-latitude`,E=re(_,h,n(C)),me=T?ce:ie,D=!!s,O=s?[{name:`queryPositions`,view:s,type:`f32`,access:`read`}]:[],k=D?`queryPositions`:`positions`,A=e=>`let x = ${k}[${k}Offset + ${e} * 2u];
  let y = ${k}[${k}Offset + ${e} * 2u + 1u];`,j=D?`true`:`neighbor != queryIndex`,M=r.queryMask??(D?void 0:r.mask),N=!D&&!r.queryMask,P=this.searchStructure===`bvh`,F=!P,I=e=>F?e:1,L=l(e,`${t}-grid-bounds`,`float32`,I(4)),R=l(e,`${t}-grid-positions`,`float32x2`,v),he=l(e,`${t}-cell-offsets`,`uint32`,I(x+1)),z=l(e,`${t}-sorted-rows`,`uint32`,I(v)),B=l(e,`${t}-sorted-targets`,`uint32`,I(v*3)),V=l(e,`${t}-grid-count`,`uint32`,1),ge=l(e,`${t}-grid-overflow`,`uint32`,1),H=g.neighborCounts??l(e,`${t}-counts`,`uint32`,y),_e=l(e,`${t}-starts`,`uint32`,y),U=g.distances??l(e,`${t}-distances`,`float32`,b),ve=N?z:l(e,`${t}-query-order`,`uint32`,I(y)),ye=N?B:l(e,`${t}-sorted-queries`,`uint32`,I(y*3)),W=N?V:l(e,`${t}-query-count`,`uint32`,1),G=[...F?[m(e,{id:`${t}-grid-bounds`,operation:w,variant:`grid-bounds`,bindings:[{name:`parameters`,view:u,type:`f32`,access:`read`},{name:`gridBounds`,view:L,type:`f32`,access:`read_write`}],invocationCount:1,declarations:E,body:`let lattice = readLattice();
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridBounds[gridBoundsOffset] = select(invalid, lattice.minimumX, lattice.valid);
  gridBounds[gridBoundsOffset + 1u] = select(invalid, lattice.minimumY, lattice.valid);
  gridBounds[gridBoundsOffset + 2u] = select(invalid,
    max(lattice.maximumX, lattice.minimumX + lattice.cellWidth * f32(COLUMNS)), lattice.valid);
  gridBounds[gridBoundsOffset + 3u] = select(invalid,
    max(lattice.maximumY, lattice.minimumY + lattice.cellHeight * f32(ROWS)), lattice.valid);`})]:[],m(e,{id:`${t}-grid-positions`,operation:w,variant:`grid-positions`,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},{name:`parameters`,view:u,type:`f32`,access:`read`},...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],{name:`gridPositions`,view:R,type:`f32`,access:`read_write`}],invocationCount:v,declarations:E,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${r.mask?`mask[maskOffset + index] != 0u`:`true`} && isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] = select(invalid, x, included);
  gridPositions[gridPositionsOffset + index * 2u + 1u] = select(invalid, y, included);`}),...F?new f({id:`${t}-grid-index`,positions:R,gridSize:[h[0],h[1]],bounds:[0,0,1,1],boundsBuffer:L,cellOffsets:he,objectIds:z,count:V,overflow:ge}).getCommandNodes(e):[],m(e,{id:`${t}-query-valid`,operation:w,variant:`query-valid`,bindings:[...D?O:[{name:`positions`,view:o,type:`f32`,access:`read`}],{name:`parameters`,view:u,type:`f32`,access:`read`},...M?[{name:`queryMask`,view:M,type:`u32`,access:`read`}]:[],{name:`counts`,view:H,type:`u32`,access:`read_write`}],invocationCount:y,declarations:E,body:`let lattice = readLattice();
  ${A(`index`)}
  let included = ${M?`queryMask[queryMaskOffset + index] != 0u`:`true`};
  counts[countsOffset + index] = select(0u, 1u, included && isPointValid(lattice, x, y));`})];if(F&&!N){let n=l(e,`${t}-query-grid-positions`,`float32x2`,y);G.push(m(e,{id:`${t}-query-grid-positions`,operation:w,variant:`query-grid-positions`,bindings:[...D?O:[{name:`positions`,view:o,type:`f32`,access:`read`}],{name:`parameters`,view:u,type:`f32`,access:`read`},...M?[{name:`queryMask`,view:M,type:`u32`,access:`read`}]:[],{name:`queryGridPositions`,view:n,type:`f32`,access:`read_write`}],invocationCount:y,declarations:E,body:`let lattice = readLattice();
  ${A(`index`)}
  let included = ${M?`queryMask[queryMaskOffset + index] != 0u`:`true`} && isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  queryGridPositions[queryGridPositionsOffset + index * 2u] = select(invalid, x, included);
  queryGridPositions[queryGridPositionsOffset + index * 2u + 1u] = select(invalid, y, included);`}),...new f({id:`${t}-query-grid-index`,positions:n,gridSize:[h[0],h[1]],bounds:[0,0,1,1],boundsBuffer:L,cellOffsets:l(e,`${t}-query-cell-offsets`,`uint32`,x+1),objectIds:ve,count:W,overflow:l(e,`${t}-query-grid-overflow`,`uint32`,1)}).getCommandNodes(e))}let K=N?`sortedTargets`:`sortedQueries`,q=N?`gridCount`:`queryCount`,J=[{name:`sortedTargets`,view:B,type:`u32`,access:`read`},...N?[]:[{name:`sortedQueries`,view:ye,type:`u32`,access:`read`}],{name:`parameters`,view:u,type:`f32`,access:`read`},{name:`cellOffsets`,view:he,type:`u32`,access:`read`},{name:q,view:W,type:`u32`,access:`read`}],be=`if (index >= ${q}[${q}Offset]) {
    return;
  }
  let queryIndex = ${K}[${K}Offset + index * 3u + 2u];
  let x = bitcast<f32>(${K}[${K}Offset + index * 3u]);
  let y = bitcast<f32>(${K}[${K}Offset + index * 3u + 1u]);`;F&&G.push(m(e,{id:`${t}-sorted-targets`,operation:w,variant:`sorted-targets`,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},{name:`sortedRows`,view:z,type:`u32`,access:`read`},{name:`gridCount`,view:V,type:`u32`,access:`read`},{name:`sortedTargets`,view:B,type:`u32`,access:`read_write`}],invocationCount:v,body:`if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * 3u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedTargets[sortedTargetsOffset + index * 3u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedTargets[sortedTargetsOffset + index * 3u + 2u] = row;`})),F&&!N&&G.push(m(e,{id:`${t}-sorted-queries`,operation:w,variant:`sorted-queries`,bindings:[{name:`queryPositions`,view:s??o,type:`f32`,access:`read`},{name:`queryOrder`,view:ve,type:`u32`,access:`read`},{name:`queryCount`,view:W,type:`u32`,access:`read`},{name:`sortedQueries`,view:ye,type:`u32`,access:`read_write`}],invocationCount:y,body:`if (index >= queryCount[queryCountOffset]) {
    return;
  }
  let row = queryOrder[queryOrderOffset + index];
  sortedQueries[sortedQueriesOffset + index * 3u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u]);
  sortedQueries[sortedQueriesOffset + index * 3u + 1u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u + 1u]);
  sortedQueries[sortedQueriesOffset + index * 3u + 2u] = row;`}));let Y,X=``;if(P){let{bvh:n,nodes:r}=ne(e,{id:`${t}-bvh`,operation:w,features:{kind:`points`,positions:R},featureCount:v,leafCapacity:Math.max(2,a(v)),spatialSort:!0});G.push(...r),Y=l(e,`${t}-bvh-node-data`,`uint32`,n.nodeCount*5),X=ue({targetRows:v,internalNodeCount:n.internalNodeCount,levelCount:n.levelCount}),G.push(m(e,{id:`${t}-bvh-node-data`,operation:w,variant:`bvh-node-data`,bindings:[{name:`nodeMinima`,view:n.nodeMinima,type:`f32`,access:`read`},{name:`nodeMaxima`,view:n.nodeMaxima,type:`f32`,access:`read`},{name:`leafIds`,view:n.leafIds,type:`u32`,access:`read`},{name:`nodeData`,view:Y,type:`u32`,access:`read_write`}],invocationCount:n.nodeCount,declarations:`const INTERNAL_NODE_COUNT: u32 = ${n.internalNodeCount}u;
const NO_ROW: u32 = 0xffffffffu;`,body:`let base = nodeDataOffset + index * 5u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_ROW, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`}))}let xe=Y?[{name:`nodeData`,view:Y,type:`u32`,access:`read`},{name:k,view:s??o,type:`f32`,access:`read`},{name:`parameters`,view:u,type:`f32`,access:`read`}]:[],Z=de(D),Q,$;if(_===`knn`){Q=l(e,`${t}-knn-ids`,`uint32`,y*S),$=l(e,`${t}-knn-distances`,`float32`,y*S);let n=`// Re-sort the selected neighbors by ID (insertion sort over at most K entries).
  for (var slot = 1u; slot < found; slot++) {
    let neighborId = bestIds[slot];
    let distanceSquared = bestDistances[slot];
    var destination = slot;
    while (destination > 0u && bestIds[destination - 1u] > neighborId) {
      bestIds[destination] = bestIds[destination - 1u];
      bestDistances[destination] = bestDistances[destination - 1u];
      destination--;
    }
    bestIds[destination] = neighborId;
    bestDistances[destination] = distanceSquared;
  }
  for (var slot = 0u; slot < found; slot++) {
    knnIds[knnIdsOffset + queryIndex * K + slot] = bestIds[slot];
    knnDistances[knnDistancesOffset + queryIndex * K + slot] = ${T?`bestDistances[slot]`:`sqrt(bestDistances[slot])`};
  }
  counts[countsOffset + queryIndex] = found;`,r=[{name:`counts`,view:H,type:`u32`,access:`read_write`},{name:`knnIds`,view:Q,type:`u32`,access:`read_write`},{name:`knnDistances`,view:$,type:`f32`,access:`read_write`}];F&&G.push(m(e,{id:`${t}-knn`,operation:w,variant:`knn`,bindings:[...J,...r],invocationCount:y,declarations:`${E}
${oe(S)}`,body:`${be}
  var found = 0u;
  var bestDistances: array<f32, ${S}>;
  var bestIds: array<u32, ${S}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${T?le(j):ae(j)}
  }
  ${n}`})),P&&G.push(m(e,{id:`${t}-bvh-knn`,operation:w,variant:`bvh-knn`,bindings:[...xe,...r],invocationCount:y,declarations:`${E}
${oe(S)}
${X}`,body:`${Z}
  var found = 0u;
  var bestDistances: array<f32, ${S}>;
  var bestIds: array<u32, ${S}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${A(`queryIndex`)}
    ${fe(j)}
  }
  ${n}`}))}else F&&G.push(m(e,{id:`${t}-count`,operation:w,variant:`radius-count`,bindings:[...J,{name:`counts`,view:H,type:`u32`,access:`read_write`}],invocationCount:y,declarations:E,body:`${be}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${me(`if (${j}) {
          count++;
        }`)}
  }
  counts[countsOffset + queryIndex] = count;`})),P&&G.push(m(e,{id:`${t}-bvh-count`,operation:w,variant:`bvh-radius-count`,bindings:[...xe,{name:`counts`,view:H,type:`u32`,access:`read_write`}],invocationCount:y,declarations:`${E}
${X}`,body:`${Z}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${A(`queryIndex`)}
    ${pe(`if (${j}) {
        count++;
      }`)}
  }
  counts[countsOffset + queryIndex] = count;`}));if(G.push(...new d({id:`${t}-scan`,input:H,output:_e,mode:`exclusive`}).getCommandNodes(e),m(e,{id:`${t}-offsets`,operation:w,variant:`offsets`,bindings:[{name:`counts`,view:H,type:`u32`,access:`read`},{name:`starts`,view:_e,type:`u32`,access:`read`},{name:`offsets`,view:g.offsets,type:`u32`,access:`read_write`},{name:`overflow`,view:g.status.overflow,type:`u32`,access:`read_write`},...g.status.requiredCount?[{name:`totalNeighbors`,view:g.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:y+1,declarations:`const QUERY_ROWS: u32 = ${y}u;
const CAPACITY: u32 = ${b}u;`,body:`if (index < QUERY_ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = starts[startsOffset + QUERY_ROWS - 1u] + counts[countsOffset + QUERY_ROWS - 1u];
    offsets[offsetsOffset + QUERY_ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${g.status.requiredCount?`totalNeighbors[totalNeighborsOffset] = total;`:``}
  }`})),Q&&$)G.push(m(e,{id:`${t}-emit`,operation:w,variant:`knn-emit`,bindings:[{name:`knnIds`,view:Q,type:`u32`,access:`read`},{name:`knnDistances`,view:$,type:`f32`,access:`read`},{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read_write`},{name:`distances`,view:U,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`const K: u32 = ${S}u;`,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin; slot < end; slot++) {
    let source = index * K + slot - begin;
    neighbors[neighborsOffset + slot] = knnIds[knnIdsOffset + source];
    distances[distancesOffset + slot] = knnDistances[knnDistancesOffset + source];
  }`}));else{let n=[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read_write`},{name:`distances`,view:U,type:`f32`,access:`read_write`}];P&&G.push(m(e,{id:`${t}-bvh-emit`,operation:w,variant:`bvh-radius-emit`,bindings:[...xe,...n],invocationCount:y,declarations:`${E}
${X}`,body:`${Z}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    ${A(`queryIndex`)}
    var next = begin;
    ${pe(`if (${j} && next < end) {
        neighbors[neighborsOffset + next] = neighbor;
        distances[distancesOffset + next] = sqrt(distanceSquared);
        next++;
      }`)}
  }`})),F&&G.push(m(e,{id:`${t}-emit`,operation:w,variant:`radius-emit`,bindings:[...J,...n],invocationCount:y,declarations:E,body:`${be}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    var next = begin;
    ${me(`if (${j} && next < end) {
          neighbors[neighborsOffset + next] = neighbor;
          distances[distancesOffset + next] = ${T?`distance`:`sqrt(distanceSquared)`};
          next++;
        }`)}
  }`})),G.push(m(e,{id:`${t}-sort-rows`,operation:w,variant:`radius-sort-rows`,bindings:[{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:g.neighbors,type:`u32`,access:`read_write`},{name:`distances`,view:U,type:`f32`,access:`read_write`}],invocationCount:y,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let neighborId = neighbors[neighborsOffset + slot];
    let distance = distances[distancesOffset + slot];
    var destination = slot;
    while (destination > begin && neighbors[neighborsOffset + destination - 1u] > neighborId) {
      neighbors[neighborsOffset + destination] = neighbors[neighborsOffset + destination - 1u];
      distances[distancesOffset + destination] = distances[distancesOffset + destination - 1u];
      destination--;
    }
    neighbors[neighborsOffset + destination] = neighborId;
    distances[distancesOffset + destination] = distance;
  }`}))}if(g.status.invalidCount){let n=v+(D?y:0),i=l(e,`${t}-invalid-flags`,`uint32`,n),a=l(e,`${t}-invalid-starts`,`uint32`,n),s=(e,t)=>`let x = ${e}[${e}Offset + ${t} * 2u];
    let y = ${e}[${e}Offset + ${t} * 2u + 1u];`;G.push(m(e,{id:`${t}-invalid-flags`,operation:w,variant:`invalid-flags`,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},...O,{name:`parameters`,view:u,type:`f32`,access:`read`},...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],...D&&r.queryMask?[{name:`queryMask`,view:r.queryMask,type:`u32`,access:`read`}]:[],{name:`invalidFlags`,view:i,type:`u32`,access:`read_write`}],invocationCount:n,declarations:`${E}
const TARGET_ROWS: u32 = ${v}u;`,body:`let lattice = readLattice();
  var included = true;
  var inside = false;
  if (index < TARGET_ROWS) {
    ${s(`positions`,`index`)}
    included = ${r.mask?`mask[maskOffset + index] != 0u`:`true`};
    inside = isPointInBounds(lattice, x, y);
  } else {
    ${D?`let queryRow = index - TARGET_ROWS;
    ${s(`queryPositions`,`queryRow`)}
    included = ${D&&r.queryMask?`queryMask[queryMaskOffset + queryRow] != 0u`:`true`};
    inside = isPointInBounds(lattice, x, y);`:``}
  }
  invalidFlags[invalidFlagsOffset + index] = select(0u, 1u, included && !inside);`}),...new d({id:`${t}-invalid-scan`,input:i,output:a,mode:`exclusive`}).getCommandNodes(e),m(e,{id:`${t}-invalid-count`,operation:w,variant:`invalid-count`,bindings:[{name:`invalidFlags`,view:i,type:`u32`,access:`read`},{name:`invalidStarts`,view:a,type:`u32`,access:`read`},{name:`invalidCount`,view:g.status.invalidCount,type:`u32`,access:`read_write`}],invocationCount:1,body:`invalidCount[invalidCountOffset] =
    invalidStarts[invalidStartsOffset + ${n-1}u] + invalidFlags[invalidFlagsOffset + ${n-1}u];`}))}let Se=_===`radius`?`let bandwidth = readParameter(4u);`:`var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  bandwidth = bandwidth * ${p(te)};`;return G.push(m(e,{id:`${t}-weights`,operation:w,variant:`weights`,bindings:[{name:`parameters`,view:u,type:`f32`,access:`read`},{name:`offsets`,view:g.offsets,type:`u32`,access:`read`},{name:`distances`,view:U,type:`f32`,access:`read`},{name:`weights`,view:g.weights,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`${E}
${se}`,body:`let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${Se}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let weight = getNeighborWeight(distances[distancesOffset + slot], bandwidth);
    weights[weightsOffset + slot] = weight;
    sum += weight;
  }
  if (${r.rowStandardize===void 0?`readParameter(9u) != 0.0`:String(r.rowStandardize)} && sum > 0.0 && isFiniteFloat(sum)) {
    for (var slot = begin; slot < end; slot++) {
      weights[weightsOffset + slot] = weights[weightsOffset + slot] / sum;
    }
  }`})),G}};export{C as n,E as t};