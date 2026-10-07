// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Lanes of the cooperative per-feature workgroup. @internal */
export const COOPERATIVE_LANES = 64;

/**
 * Rings with more rows than this make a feature cooperative: it is measured by one workgroup
 * instead of one invocation. Features whose rings are all at most this long keep the serial path
 * (bit-identical to earlier versions). @internal
 */
export const COOPERATIVE_RING_ROWS = 512;

/**
 * WGSL helpers shared by the serial and cooperative kernels to classify features: the ring range
 * of a feature, the rows of a ring, and `isCooperativeFeature`. Needs the kernel's `ringOffsets`
 * (and `featureRingOffsets` when given) bindings and the `ROW_COUNT` and `RING_COUNT` constants.
 *
 * @internal
 */
export function getRingHelpersSource(
  hasFeatureRingOffsets: boolean,
  cooperativeRingRows: number
): string {
  return /* wgsl */ `
const COOPERATIVE_RING_ROWS: u32 = ${cooperativeRingRows}u;

// Ring range (begin, end) of a feature.
fn getFeatureRingRange(feature: u32) -> vec2<u32> {
  ${
    hasFeatureRingOffsets
      ? `let ringBegin = min(featureRingOffsets[featureRingOffsetsOffset + feature], RING_COUNT);
  let ringEnd = min(max(featureRingOffsets[featureRingOffsetsOffset + feature + 1u], ringBegin), RING_COUNT);
  return vec2<u32>(ringBegin, ringEnd);`
      : 'return vec2<u32>(feature, feature + 1u);'
  }
}

// First row and row count of a ring.
fn getRingRows(ring: u32) -> vec2<u32> {
  let rowBegin = min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
  let rowEnd = min(max(ringOffsets[ringOffsetsOffset + ring + 1u], rowBegin), ROW_COUNT);
  return vec2<u32>(rowBegin, rowEnd - rowBegin);
}

// A feature with a ring above the threshold is measured by the cooperative kernel.
fn isCooperativeFeature(feature: u32) -> bool {
  let range = getFeatureRingRange(feature);
  for (var ring = range.x; ring < range.y; ring++) {
    if (getRingRows(ring).y > COOPERATIVE_RING_ROWS) {
      return true;
    }
  }
  return false;
}
`;
}

/** WGSL body of the cooperative kernel (one 64-lane workgroup per batch of features). @internal */
export const COOPERATIVE_BODY_SOURCE = /* wgsl */ `
  let lane = localInvocationIndex;
  if (lane == 0u) {
    sharedGroup = index / COOPERATIVE_LANES;
  }
  let group = workgroupUniformLoad(&sharedGroup);
  // Workgroups stride over batches of 64 features; each batch flags its cooperative features, and
  // the whole workgroup measures them one after another.
  for (var base = group * COOPERATIVE_LANES; base < FEATURE_COUNT; base += COOPERATIVE_GROUPS * COOPERATIVE_LANES) {
    let candidate = base + lane;
    sharedFlags[lane] = select(0u, 1u, candidate < FEATURE_COUNT && isCooperativeFeature(candidate));
    workgroupBarrier();
    for (var slot = 0u; slot < COOPERATIVE_LANES; slot++) {
      if (workgroupUniformLoad(&sharedFlags[slot]) == 1u) {
        measureFeatureCooperatively(base + slot, lane);
      }
    }
    workgroupBarrier();
  }`;

/**
 * WGSL of the cooperative feature kernel: one workgroup measures a feature that owns a ring of
 * more than `COOPERATIVE_RING_ROWS` rows. Lane 0 walks the rings in order and measures small
 * rings itself; for a large ring the 64 lanes each take a contiguous chunk of vertices and run the
 * same per-vertex loop as the serial kernel, started from the chunk's unwrapped longitude (an
 * exclusive scan of the chunks' wrapped longitude steps). Lane 0 merges the lane partials in lane
 * order with Neumaier addition, so results are deterministic. The longest dependent chain is a
 * chunk of `rows / 64` vertices instead of the whole ring.
 *
 * @internal
 */
export function getCooperativeSource(props: {
  groupCount: number;
  featureCount: number;
  edgeCrossSource: string;
  /** Statements that write the final statistics from `acc` (a `FeatureAccumulators`). */
  tail: string;
}): string {
  return /* wgsl */ `
const COOPERATIVE_LANES: u32 = ${COOPERATIVE_LANES}u;
const COOPERATIVE_GROUPS: u32 = ${props.groupCount}u;
const FEATURE_COUNT: u32 = ${props.featureCount}u;

var<workgroup> sharedGroup: u32;
var<workgroup> sharedFlags: array<u32, ${COOPERATIVE_LANES}>;
var<workgroup> sharedRing: u32;
var<workgroup> sharedRingEnd: u32;
var<workgroup> sharedOrigin: vec2<f32>;
var<workgroup> laneScan: array<f32, ${COOPERATIVE_LANES}>;
var<workgroup> laneLengthSum: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneLineMomentX: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneLineMomentY: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneArea: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneStraightArea: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneMomentX: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneMomentY: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneVertexSum: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneBoundsMin: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneBoundsMax: array<vec2<f32>, ${COOPERATIVE_LANES}>;
var<workgroup> laneExtremeRows: array<vec4<u32>, ${COOPERATIVE_LANES}>;

// Sums of one ring (or one lane's chunk of it): each is a Neumaier (sum, compensation) pair.
struct RingAccumulators {
  lengthSum: vec2<f32>,
  lineMomentX: vec2<f32>,
  lineMomentY: vec2<f32>,
  area: vec2<f32>,
  straightArea: vec2<f32>,
  momentX: vec2<f32>,
  momentY: vec2<f32>,
  vertexSum: vec2<f32>,
  boundsMin: vec2<f32>,
  boundsMax: vec2<f32>,
  extremeRows: vec4<u32>
}

struct FeatureAccumulators {
  lengthSum: vec2<f32>,
  areaSum: vec2<f32>,
  momentX: vec2<f32>,
  momentY: vec2<f32>,
  lineMomentX: vec2<f32>,
  lineMomentY: vec2<f32>,
  straightSum: vec2<f32>,
  vertexSum: vec2<f32>,
  boundsMin: vec2<f32>,
  boundsMax: vec2<f32>,
  extremeRows: vec4<u32>,
  vertexCount: u32,
  isFirstRing: bool
}

fn initRing() -> RingAccumulators {
  return RingAccumulators(
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0),
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(3.4e38), vec2<f32>(-3.4e38),
    vec4<u32>(0xffffffffu)
  );
}

fn initFeature() -> FeatureAccumulators {
  return FeatureAccumulators(
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0),
    vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(3.4e38), vec2<f32>(-3.4e38),
    vec4<u32>(0xffffffffu), 0u, true
  );
}

// The serial kernel's per-vertex loop over vertices [vertexStart, vertexEnd) of a ring: bounds and extremes,
// vertex sum, and per edge the length, line moments, ring area and area moments. startLocalX is
// the unwrapped local longitude of vertex vertexStart (geographic coordinates, not point sets).
fn accumulateRange(
  rowBegin: u32, rowCount: u32, vertexStart: u32, vertexEnd: u32, startLocalX: f32, origin: vec2<f32>,
  acc: ptr<function, RingAccumulators>
) {
  if (vertexStart >= vertexEnd) {
    return;
  }
  let firstPosition = getPosition(rowBegin);
  var firstLocal = firstPosition - origin;
  if (IS_GEOGRAPHIC) {
    firstLocal.x = geodesicWrapLongitudeDelta(firstPosition.x - origin.x);
  }
  var position = getPosition(rowBegin + vertexStart);
  var local = position - origin;
  if (IS_GEOGRAPHIC) {
    local.x = select(startLocalX, geodesicWrapLongitudeDelta(position.x - origin.x), IS_POINT);
  }
  var areaX = local.x;
  if (IS_GEOGRAPHIC) {
    areaX = local.x * GEODESIC_DEGREES_TO_RADIANS;
  }
  var areaY = getAreaY(position, origin);
  let edgeCount = select(rowCount - 1u, rowCount, IS_POLYGON);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    // Strict comparisons keep the lowest vertex row on ties.
    if (local.x < (*acc).boundsMin.x) {
      (*acc).extremeRows.x = rowBegin + vertex;
    }
    if (local.y < (*acc).boundsMin.y) {
      (*acc).extremeRows.y = rowBegin + vertex;
    }
    if (local.x > (*acc).boundsMax.x) {
      (*acc).extremeRows.z = rowBegin + vertex;
    }
    if (local.y > (*acc).boundsMax.y) {
      (*acc).extremeRows.w = rowBegin + vertex;
    }
    (*acc).boundsMin = min((*acc).boundsMin, local);
    (*acc).boundsMax = max((*acc).boundsMax, local);
    (*acc).vertexSum += local;
    if (IS_POINT) {
      if (vertex + 1u < vertexEnd) {
        let nextPoint = getPosition(rowBegin + vertex + 1u);
        local = nextPoint - origin;
        if (IS_GEOGRAPHIC) {
          local.x = geodesicWrapLongitudeDelta(nextPoint.x - origin.x);
        }
      }
      continue;
    }
    if (vertex >= edgeCount) {
      break;
    }
    let nextRow = select(rowBegin + vertex + 1u, rowBegin, vertex + 1u == rowCount);
    let nextPosition = getPosition(nextRow);
    var nextLocal = nextPosition - origin;
    if (IS_GEOGRAPHIC) {
      nextLocal = vec2<f32>(
        local.x + geodesicWrapLongitudeDelta(nextPosition.x - position.x),
        nextPosition.y - origin.y
      );
    }
    if (vertex + 1u == rowCount) {
      nextLocal = firstLocal;
    }
    var nextAreaX = nextLocal.x;
    if (IS_GEOGRAPHIC) {
      nextAreaX = nextLocal.x * GEODESIC_DEGREES_TO_RADIANS;
    }
    let nextAreaY = getAreaY(nextPosition, origin);
    let edgeLength = getEdgeLength(position, nextPosition);
    addCompensated(&(*acc).lengthSum, edgeLength);
    addCompensated(&(*acc).lineMomentX, edgeLength * 0.5 * (local.x + nextLocal.x));
    addCompensated(&(*acc).lineMomentY, edgeLength * 0.5 * (local.y + nextLocal.y));
    let cross = areaX * nextAreaY - nextAreaX * areaY;
    ${props.edgeCrossSource}
    addCompensated(&(*acc).area, edgeCross);
    addCompensated(&(*acc).straightArea, cross);
    addCompensated(&(*acc).momentX, (areaX + nextAreaX) * cross);
    addCompensated(&(*acc).momentY, (areaY + nextAreaY) * cross);
    position = nextPosition;
    local = nextLocal;
    areaX = nextAreaX;
    areaY = nextAreaY;
  }
}

// Folds a finished ring into the feature (the serial kernel's end-of-ring step).
fn mergeRing(feature: ptr<function, FeatureAccumulators>, ring: ptr<function, RingAccumulators>, rowCount: u32) {
  (*feature).vertexCount += rowCount;
  addCompensated(&(*feature).lengthSum, (*ring).lengthSum.x);
  addCompensated(&(*feature).lengthSum, (*ring).lengthSum.y);
  addCompensated(&(*feature).lineMomentX, (*ring).lineMomentX.x);
  addCompensated(&(*feature).lineMomentX, (*ring).lineMomentX.y);
  addCompensated(&(*feature).lineMomentY, (*ring).lineMomentY.x);
  addCompensated(&(*feature).lineMomentY, (*ring).lineMomentY.y);
  (*feature).vertexSum += (*ring).vertexSum;
  if ((*ring).boundsMin.x < (*feature).boundsMin.x) {
    (*feature).extremeRows.x = (*ring).extremeRows.x;
  }
  if ((*ring).boundsMin.y < (*feature).boundsMin.y) {
    (*feature).extremeRows.y = (*ring).extremeRows.y;
  }
  if ((*ring).boundsMax.x > (*feature).boundsMax.x) {
    (*feature).extremeRows.z = (*ring).extremeRows.z;
  }
  if ((*ring).boundsMax.y > (*feature).boundsMax.y) {
    (*feature).extremeRows.w = (*ring).extremeRows.w;
  }
  (*feature).boundsMin = min((*feature).boundsMin, (*ring).boundsMin);
  (*feature).boundsMax = max((*feature).boundsMax, (*ring).boundsMax);
  var ringFactor = 1.0;
  if (FIRST_RING_EXTERIOR) {
    let ringSign = select(-1.0, 1.0, getCompensated((*ring).area) >= 0.0);
    ringFactor = select(-1.0, 1.0, (*feature).isFirstRing) * ringSign;
  }
  addCompensated(&(*feature).areaSum, ringFactor * getCompensated((*ring).area));
  addCompensated(&(*feature).straightSum, ringFactor * getCompensated((*ring).straightArea));
  addCompensated(&(*feature).momentX, ringFactor * getCompensated((*ring).momentX));
  addCompensated(&(*feature).momentY, ringFactor * getCompensated((*ring).momentY));
  (*feature).isFirstRing = false;
}

// Adds lane laneIndex's published partial to a ring accumulator.
fn addLanePartial(ring: ptr<function, RingAccumulators>, laneIndex: u32) {
  addCompensated(&(*ring).lengthSum, laneLengthSum[laneIndex].x);
  addCompensated(&(*ring).lengthSum, laneLengthSum[laneIndex].y);
  addCompensated(&(*ring).lineMomentX, laneLineMomentX[laneIndex].x);
  addCompensated(&(*ring).lineMomentX, laneLineMomentX[laneIndex].y);
  addCompensated(&(*ring).lineMomentY, laneLineMomentY[laneIndex].x);
  addCompensated(&(*ring).lineMomentY, laneLineMomentY[laneIndex].y);
  addCompensated(&(*ring).area, laneArea[laneIndex].x);
  addCompensated(&(*ring).area, laneArea[laneIndex].y);
  addCompensated(&(*ring).straightArea, laneStraightArea[laneIndex].x);
  addCompensated(&(*ring).straightArea, laneStraightArea[laneIndex].y);
  addCompensated(&(*ring).momentX, laneMomentX[laneIndex].x);
  addCompensated(&(*ring).momentX, laneMomentX[laneIndex].y);
  addCompensated(&(*ring).momentY, laneMomentY[laneIndex].x);
  addCompensated(&(*ring).momentY, laneMomentY[laneIndex].y);
  (*ring).vertexSum += laneVertexSum[laneIndex];
  // Lanes own ascending chunks, so a strict comparison keeps the lowest row on ties.
  if (laneBoundsMin[laneIndex].x < (*ring).boundsMin.x) {
    (*ring).extremeRows.x = laneExtremeRows[laneIndex].x;
  }
  if (laneBoundsMin[laneIndex].y < (*ring).boundsMin.y) {
    (*ring).extremeRows.y = laneExtremeRows[laneIndex].y;
  }
  if (laneBoundsMax[laneIndex].x > (*ring).boundsMax.x) {
    (*ring).extremeRows.z = laneExtremeRows[laneIndex].z;
  }
  if (laneBoundsMax[laneIndex].y > (*ring).boundsMax.y) {
    (*ring).extremeRows.w = laneExtremeRows[laneIndex].w;
  }
  (*ring).boundsMin = min((*ring).boundsMin, laneBoundsMin[laneIndex]);
  (*ring).boundsMax = max((*ring).boundsMax, laneBoundsMax[laneIndex]);
}

fn writeFeatureStats(index: u32, origin: vec2<f32>, acc: FeatureAccumulators) {
${props.tail}
}

// Wrapped longitude step into vertex k of a ring (vertex 0 steps from the feature origin).
fn getLongitudeStep(rowBegin: u32, vertex: u32, origin: vec2<f32>) -> f32 {
  let current = getPosition(rowBegin + vertex);
  if (vertex == 0u) {
    return geodesicWrapLongitudeDelta(current.x - origin.x);
  }
  return geodesicWrapLongitudeDelta(current.x - getPosition(rowBegin + vertex - 1u).x);
}

// Measures one cooperative feature with the whole workgroup. Called from uniform control flow.
fn measureFeatureCooperatively(feature: u32, lane: u32) {
  let range = getFeatureRingRange(feature);
  if (lane == 0u) {
    for (var ring = range.x; ring < range.y; ring++) {
      let rows = getRingRows(ring);
      if (rows.y > 0u) {
        sharedOrigin = getPosition(rows.x);
        break;
      }
    }
    sharedRingEnd = range.y;
  }
  let ringEnd = workgroupUniformLoad(&sharedRingEnd);
  let origin = workgroupUniformLoad(&sharedOrigin);
  var accumulated = initFeature();
  var ring = range.x;
  loop {
    if (lane == 0u) {
      // Small rings are lane 0's: no barriers until the next large ring.
      while (ring < ringEnd) {
        let rows = getRingRows(ring);
        if (rows.y > COOPERATIVE_RING_ROWS) {
          break;
        }
        if (rows.y > 0u) {
          var smallRing = initRing();
          let firstPosition = getPosition(rows.x);
          accumulateRange(
            rows.x, rows.y, 0u, rows.y,
            geodesicWrapLongitudeDelta(firstPosition.x - origin.x), origin, &smallRing
          );
          mergeRing(&accumulated, &smallRing, rows.y);
        }
        ring++;
      }
      sharedRing = ring;
    }
    let current = workgroupUniformLoad(&sharedRing);
    if (current >= ringEnd) {
      break;
    }
    let rows = getRingRows(current);
    let chunk = (rows.y + COOPERATIVE_LANES - 1u) / COOPERATIVE_LANES;
    let vertexStart = min(lane * chunk, rows.y);
    let vertexEnd = min(vertexStart + chunk, rows.y);
    var startLocalX = 0.0;
    if (IS_GEOGRAPHIC && !IS_POINT) {
      // Unwrapped longitude of the chunk's first vertex: exclusive scan of the chunks' sums of
      // wrapped longitude steps (Hillis-Steele in workgroup memory).
      var stepSum = vec2<f32>(0.0);
      for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
        addCompensated(&stepSum, getLongitudeStep(rows.x, vertex, origin));
      }
      laneScan[lane] = stepSum.x + stepSum.y;
      workgroupBarrier();
      for (var offset = 1u; offset < COOPERATIVE_LANES; offset = offset * 2u) {
        var value = laneScan[lane];
        if (lane >= offset) {
          value = value + laneScan[lane - offset];
        }
        workgroupBarrier();
        laneScan[lane] = value;
        workgroupBarrier();
      }
      var prefix = 0.0;
      if (lane > 0u) {
        prefix = laneScan[lane - 1u];
      }
      if (vertexStart < vertexEnd) {
        startLocalX = prefix + getLongitudeStep(rows.x, vertexStart, origin);
      }
    }
    var partial = initRing();
    accumulateRange(rows.x, rows.y, vertexStart, vertexEnd, startLocalX, origin, &partial);
    laneLengthSum[lane] = partial.lengthSum;
    laneLineMomentX[lane] = partial.lineMomentX;
    laneLineMomentY[lane] = partial.lineMomentY;
    laneArea[lane] = partial.area;
    laneStraightArea[lane] = partial.straightArea;
    laneMomentX[lane] = partial.momentX;
    laneMomentY[lane] = partial.momentY;
    laneVertexSum[lane] = partial.vertexSum;
    laneBoundsMin[lane] = partial.boundsMin;
    laneBoundsMax[lane] = partial.boundsMax;
    laneExtremeRows[lane] = partial.extremeRows;
    workgroupBarrier();
    if (lane == 0u) {
      var combined = initRing();
      for (var laneIndex = 0u; laneIndex < COOPERATIVE_LANES; laneIndex++) {
        addLanePartial(&combined, laneIndex);
      }
      mergeRing(&accumulated, &combined, rows.y);
      ring = current + 1u;
    }
  }
  if (lane == 0u) {
    writeFeatureStats(feature, origin, accumulated);
  }
}
`;
}
