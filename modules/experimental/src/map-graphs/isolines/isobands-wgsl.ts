// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared WGSL for the {@link GPUIsobands} count and scatter kernels. It implements the "Band
 * regions" construction of the marching-squares definition: per cell and band, a combinatorial
 * boundary walk with crossing points, joined through segment partners, fan-triangulated.
 *
 * The including kernel must declare `values`, optional `validity`, `breaks` and `params` bindings,
 * the constants `WIDTH`, `HEIGHT`, `MAXIMUM_BREAK_COUNT`, the helper functions from
 * `getRasterAlgebraValueWGSL` and `getBreakSearchWGSL('countBreaksBelow', 'breaks')`, and a
 * `writeTriangle(index, a, b, c, band)` function (a no-op in the count kernel).
 *
 * @internal
 */
export function getIsobandsCellWGSL(hasValidity: boolean): string {
  return /* wgsl */ `
const CELL_COLUMNS: u32 = WIDTH - 1u;
const NO_EVENT: u32 = 0xffffffffu;

// Event kinds in bits 0..1: 0 corner, 1 entry crossing, 2 exit crossing. Bit 2 is the level
// (0 low, 1 high), bits 3..5 the corner index (corners) or canonical edge index (crossings).
var<private> cornerValues: array<f32, 4>;
var<private> bandStates: array<u32, 4>;
var<private> bandMasks: array<u32, 2>;
var<private> bandJoined: array<bool, 2>;
var<private> cellEvents: array<u32, 12>;
var<private> cellEventCount: u32;
var<private> cellColumn: u32;
var<private> cellRow: u32;
var<private> bandLow: f32;
var<private> bandHigh: f32;
var<private> pieceVertexCount: u32;
var<private> pieceFirst: vec2<f32>;
var<private> piecePrevious: vec2<f32>;
var<private> emittedTriangles: u32;
var<private> EDGE_START: array<u32, 4> = array<u32, 4>(0u, 1u, 3u, 0u);
var<private> EDGE_END: array<u32, 4> = array<u32, 4>(1u, 2u, 2u, 3u);

fn getParameterCount(value: f32) -> u32 {
  return min(u32(clamp(select(0.0, value, isFiniteValue(value)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
}

fn getWorldPosition(gridX: f32, gridY: f32) -> vec2<f32> {
  let x = params[paramsOffset + 4u] + (gridX + 0.5) * params[paramsOffset + 8u];
  let y = params[paramsOffset + 5u] + (gridY + 0.5) * params[paramsOffset + 9u];
  return vec2<f32>(x, y);
}

fn getCornerPosition(corner: u32) -> vec2<f32> {
  let dx = select(0u, 1u, corner == 1u || corner == 2u);
  let dy = select(0u, 1u, corner >= 2u);
  return getWorldPosition(f32(cellColumn + dx), f32(cellRow + dy));
}

// Crossing of the band's low (level 0) or high (level 1) break on a canonical edge: horizontal
// edges run +x, vertical edges +y, from the lower-index sample to the higher-index sample.
fn getCrossingPosition(level: u32, edge: u32) -> vec2<f32> {
  let boundary = select(bandLow, bandHigh, level == 1u);
  let startValue = cornerValues[EDGE_START[edge]];
  let endValue = cornerValues[EDGE_END[edge]];
  let t = (boundary - startValue) / (endValue - startValue);
  let baseX = f32(cellColumn);
  let baseY = f32(cellRow);
  if (edge == 0u) {
    return getWorldPosition(baseX + t, baseY);
  }
  if (edge == 1u) {
    return getWorldPosition(f32(cellColumn + 1u), baseY + t);
  }
  if (edge == 2u) {
    return getWorldPosition(baseX + t, f32(cellRow + 1u));
  }
  return getWorldPosition(baseX, baseY + t);
}

fn getEventPosition(cellEvent: u32) -> vec2<f32> {
  let index = (cellEvent >> 3u) & 7u;
  if ((cellEvent & 3u) == 0u) {
    return getCornerPosition(index);
  }
  return getCrossingPosition((cellEvent >> 2u) & 1u, index);
}

fn pushEvent(cellEvent: u32) {
  cellEvents[cellEventCount] = cellEvent;
  cellEventCount++;
}

fn buildEvents() {
  cellEventCount = 0u;
  for (var walk = 0u; walk < 4u; walk++) {
    let startState = bandStates[walk];
    let endState = bandStates[(walk + 1u) % 4u];
    if (startState == 1u) {
      pushEvent(walk << 3u);
    }
    if (startState < endState) {
      if (startState == 0u) {
        pushEvent(1u | (walk << 3u));
      }
      if (endState == 2u) {
        pushEvent(2u | 4u | (walk << 3u));
      }
    } else if (startState > endState) {
      if (startState == 2u) {
        pushEvent(1u | 4u | (walk << 3u));
      }
      if (endState == 0u) {
        pushEvent(2u | (walk << 3u));
      }
    }
  }
}

fn isEdgeCrossed(mask: u32, edge: u32) -> bool {
  let startHigh = ((mask >> EDGE_START[edge]) & 1u) != 0u;
  let endHigh = ((mask >> EDGE_END[edge]) & 1u) != 0u;
  return startHigh != endHigh;
}

// Other endpoint of the level segment through the crossing on 'edge'. Saddles (high corners on one
// diagonal) pair edges by the centre rule: joined cuts off the low corners, separated the high.
fn getPartnerEdge(level: u32, edge: u32) -> u32 {
  let mask = bandMasks[level];
  if (mask == 5u || mask == 10u) {
    let pairsAdjacent = (mask == 5u) == bandJoined[level];
    return select(3u - edge, edge ^ 1u, pairsAdjacent);
  }
  for (var other = 0u; other < 4u; other++) {
    if (other != edge && isEdgeCrossed(mask, other)) {
      return other;
    }
  }
  return edge;
}

fn findCrossingEvent(level: u32, edge: u32) -> u32 {
  for (var i = 0u; i < cellEventCount; i++) {
    let cellEvent = cellEvents[i];
    if ((cellEvent & 3u) != 0u && ((cellEvent >> 2u) & 1u) == level && ((cellEvent >> 3u) & 7u) == edge) {
      return i;
    }
  }
  return NO_EVENT;
}

fn pushPieceVertex(cellEvent: u32, shouldWrite: bool, baseTriangle: u32, band: u32) {
  var position = vec2<f32>(0.0, 0.0);
  if (shouldWrite) {
    position = getEventPosition(cellEvent);
  }
  if (pieceVertexCount == 0u) {
    pieceFirst = position;
  } else if (pieceVertexCount >= 2u) {
    if (shouldWrite) {
      writeTriangle(baseTriangle + emittedTriangles, pieceFirst, piecePrevious, position, band);
    }
    emittedTriangles++;
  }
  piecePrevious = position;
  pieceVertexCount++;
}

fn setupBand(band: u32, breakCount: u32) -> bool {
  let hasLow = band > 0u;
  let hasHigh = band < breakCount;
  bandLow = select(0.0, breaks[breaksOffset + max(band, 1u) - 1u], hasLow);
  bandHigh = select(0.0, breaks[breaksOffset + min(band, MAXIMUM_BREAK_COUNT - 1u)], hasHigh);
  if (hasLow && hasHigh && bandLow >= bandHigh) {
    return false;
  }
  var lowMask = 0u;
  var highMask = 0u;
  for (var corner = 0u; corner < 4u; corner++) {
    let value = cornerValues[corner];
    var state = 1u;
    if (hasLow && value < bandLow) {
      state = 0u;
    } else if (hasHigh && value >= bandHigh) {
      state = 2u;
    }
    bandStates[corner] = state;
    lowMask |= select(0u, 1u << corner, state >= 1u);
    highMask |= select(0u, 1u << corner, state == 2u);
  }
  bandMasks[0] = lowMask;
  bandMasks[1] = highMask;
  let centre = ((cornerValues[0] + cornerValues[1]) + (cornerValues[2] + cornerValues[3])) * 0.25;
  bandJoined[0] = centre >= bandLow;
  bandJoined[1] = centre >= bandHigh;
  return true;
}

// Triangles of one band in the loaded cell; writes them from 'baseTriangle' when 'shouldWrite' is set.
fn processBand(shouldWrite: bool, baseTriangle: u32, band: u32, breakCount: u32) -> u32 {
  emittedTriangles = 0u;
  if (!setupBand(band, breakCount)) {
    return 0u;
  }
  buildEvents();
  var hasCrossing = false;
  for (var i = 0u; i < cellEventCount; i++) {
    if ((cellEvents[i] & 3u) != 0u) {
      hasCrossing = true;
    }
  }
  if (!hasCrossing) {
    if (cellEventCount == 4u) {
      pieceVertexCount = 0u;
      for (var i = 0u; i < 4u; i++) {
        pushPieceVertex(cellEvents[i], shouldWrite, baseTriangle, band);
      }
    }
    return emittedTriangles;
  }
  var visited = 0u;
  for (var i = 0u; i < cellEventCount; i++) {
    if ((cellEvents[i] & 3u) != 1u || ((visited >> i) & 1u) != 0u) {
      continue;
    }
    pieceVertexCount = 0u;
    var current = i;
    for (var arc = 0u; arc < 8u; arc++) {
      visited |= 1u << current;
      var exitIndex = current;
      for (var stepIndex = 0u; stepIndex < 12u; stepIndex++) {
        pushPieceVertex(cellEvents[exitIndex], shouldWrite, baseTriangle, band);
        if ((cellEvents[exitIndex] & 3u) == 2u) {
          break;
        }
        exitIndex = (exitIndex + 1u) % cellEventCount;
      }
      let exitEvent = cellEvents[exitIndex];
      let level = (exitEvent >> 2u) & 1u;
      current = findCrossingEvent(level, getPartnerEdge(level, (exitEvent >> 3u) & 7u));
      if (current == i || current == NO_EVENT) {
        break;
      }
    }
  }
  return emittedTriangles;
}

fn getSampleIndex(cell: u32, corner: u32) -> u32 {
  let column = cell % CELL_COLUMNS + select(0u, 1u, corner == 1u || corner == 2u);
  let row = cell / CELL_COLUMNS + select(0u, 1u, corner >= 2u);
  return row * WIDTH + column;
}

fn isSampleValid(sample: u32) -> bool {
  var isValid = !isNoDataValue(values[valuesOffset + sample]);
  ${hasValidity ? 'isValid = isValid && validity[validityOffset + sample] != 0u;' : ''}
  return isValid;
}

// Loads the four corners of a marching-squares cell; false when any corner is nodata.
fn loadCell(cell: u32) -> bool {
  cellColumn = cell % CELL_COLUMNS;
  cellRow = cell / CELL_COLUMNS;
  for (var corner = 0u; corner < 4u; corner++) {
    let sample = getSampleIndex(cell, corner);
    if (!isSampleValid(sample)) {
      return false;
    }
    cornerValues[corner] = values[valuesOffset + sample];
  }
  return true;
}

// Triangles of every emitted band in a cell, in band order.
fn processCell(cell: u32, shouldWrite: bool, baseTriangle: u32) -> u32 {
  if (!loadCell(cell)) {
    return 0u;
  }
  let breakCount = getParameterCount(params[paramsOffset]);
  var minimumClass = 0xffffffffu;
  var maximumClass = 0u;
  for (var corner = 0u; corner < 4u; corner++) {
    let bandClass = countBreaksBelow(0u, breakCount, cornerValues[corner], false);
    minimumClass = min(minimumClass, bandClass);
    maximumClass = max(maximumClass, bandClass);
  }
  let firstBand = max(minimumClass, getParameterCount(params[paramsOffset + 1u]));
  let lastBand = min(maximumClass, min(getParameterCount(params[paramsOffset + 2u]), breakCount));
  var total = 0u;
  for (var band = firstBand; band <= lastBand; band++) {
    total += processBand(shouldWrite, baseTriangle + total, band, breakCount);
  }
  return total;
}`;
}
