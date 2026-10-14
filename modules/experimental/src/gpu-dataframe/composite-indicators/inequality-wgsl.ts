// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_NAN_WGSL, GET_ORDERED_KEY_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';

/** Floats per zone in the interleaved statistics buffer. */
export const INEQUALITY_STATS_STRIDE = 10;

/** Slots of one zone's interleaved statistics. */
export const INEQUALITY_STAT = {
  MEAN: 0,
  GINI: 1,
  THEIL_T: 2,
  THEIL_L: 3,
  ATKINSON: 4,
  HOOVER: 5,
  PALMA: 6,
  TOTAL_WEIGHT: 7,
  TOTAL_INCOME: 8,
  WEIGHTED_X_LOG_X: 9
} as const;

/** Shared WGSL helpers. */
export const INEQUALITY_HELPERS_WGSL = /* wgsl */ `
${GET_NAN_WGSL}

${IS_FINITE_BITS_WGSL}

${GET_ORDERED_KEY_WGSL}

fn interpolateLorenz(cut: f32, previousP: f32, nextP: f32, previousL: f32, nextL: f32) -> f32 {
  if (cut >= nextP) {
    return nextL;
  }
  return previousL + (cut - previousP) / (nextP - previousP) * (nextL - previousL);
}
`;

/** Body of the row validation and key generation kernel. `weightCheck` is empty without weights. */
export function getInequalityKeysBody(options: {hasMask: boolean; hasWeights: boolean}): string {
  return /* wgsl */ `let value = values[valuesOffset + index];
  let zone = zoneIds[zoneIdsOffset + index];
  var isValid = isFiniteBits(value) && value >= 0.0 && zone < ZONE_COUNT;
  ${options.hasMask ? 'isValid = isValid && rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${
    options.hasWeights
      ? `let weight = weights[weightsOffset + index];
  isValid = isValid && isFiniteBits(weight) && weight > 0.0;`
      : ''
  }
  // Valid keys are below 0xff800000, so invalid rows sort after every valid row; -0 sorts as +0.
  valueKeys[valueKeysOffset + index] = select(0xffffffffu, getOrderedKey(select(value, 0.0, value == 0.0)), isValid);
  rowIndices[rowIndicesOffset + index] = index;`;
}

/** Body of the kernel that keys the value-sorted rows by zone. */
export const INEQUALITY_ZONE_KEYS_BODY = /* wgsl */ `let row = sortedRows[sortedRowsOffset + index];
  let isValid = sortedValueKeys[sortedValueKeysOffset + index] != 0xffffffffu;
  zoneKeys[zoneKeysOffset + index] = select(ZONE_COUNT, zoneIds[zoneIdsOffset + row], isValid);`;

/** Body of the per-zone statistics kernel (one thread per zone). */
export function getInequalityZoneStatsBody(weightExpression: string): string {
  return /* wgsl */ `let zone = index;
  var searchLow = 0u;
  var searchHigh = ROW_COUNT;
  while (searchLow < searchHigh) {
    let middle = (searchLow + searchHigh) / 2u;
    if (sortedZones[sortedZonesOffset + middle] < zone) {
      searchLow = middle + 1u;
    } else {
      searchHigh = middle;
    }
  }
  let rangeStart = searchLow;
  searchHigh = ROW_COUNT;
  while (searchLow < searchHigh) {
    let middle = (searchLow + searchHigh) / 2u;
    if (sortedZones[sortedZonesOffset + middle] <= zone) {
      searchLow = middle + 1u;
    } else {
      searchHigh = middle;
    }
  }
  let rangeEnd = searchLow;
  let count = rangeEnd - rangeStart;
  counts[countsOffset + zone] = count;
  let nan = getNaN();
  let base = statsOffset + zone * STATS_STRIDE;
  let knotBase = lorenzOffset + zone * KNOT_COUNT;
  for (var slot = 0u; slot < STATS_STRIDE; slot++) {
    stats[base + slot] = nan;
  }
  for (var knot = 0u; knot < KNOT_COUNT; knot++) {
    lorenz[knotBase + knot] = nan;
  }
  stats[base + 7u] = 0.0;
  stats[base + 8u] = 0.0;
  stats[base + 9u] = 0.0;
  if (count == 0u) {
    return;
  }

  // Pass 1: totals in sorted order.
  var totalWeight = 0.0;
  var totalIncome = 0.0;
  var weightedXLogX = 0.0;
  var hasZero = false;
  for (var position = rangeStart; position < rangeEnd; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    let x = values[valuesOffset + row];
    let w = ${weightExpression};
    totalWeight = totalWeight + w;
    totalIncome = totalIncome + w * x;
    if (x > 0.0) {
      weightedXLogX = weightedXLogX + w * x * log(x);
    } else {
      hasZero = true;
    }
  }
  let mean = totalIncome / totalWeight;
  stats[base] = mean;
  stats[base + 7u] = totalWeight;
  stats[base + 8u] = totalIncome;
  stats[base + 9u] = weightedXLogX;
  if (!(totalIncome > 0.0)) {
    return;
  }

  // Pass 2: Lorenz walk and the mean-relative sums, in the same order.
  let epsilon = params[paramsOffset];
  let palmaTop = params[paramsOffset + 1u];
  let palmaBottom = params[paramsOffset + 2u];
  let power = 1.0 - epsilon;
  let usePower = epsilon != 1.0;
  let palmaBottomTarget = palmaBottom;
  let palmaTopTarget = 1.0 - palmaTop;
  var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  var previousP = 0.0;
  var previousL = 0.0;
  var trapezoidArea = 0.0;
  var sumTheilT = 0.0;
  var sumLog = 0.0;
  var sumPower = 0.0;
  var sumAbsolute = 0.0;
  var nextKnot = 1u;
  var bottomL = nan;
  var topCutL = nan;
  var isBottomDone = false;
  var isTopDone = false;
  lorenz[knotBase] = 0.0;
  for (var position = rangeStart; position < rangeEnd; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    let x = values[valuesOffset + row];
    let w = ${weightExpression};
    cumulativeWeight = cumulativeWeight + w;
    cumulativeIncome = cumulativeIncome + w * x;
    let nextP = cumulativeWeight / totalWeight;
    let nextL = cumulativeIncome / totalIncome;
    trapezoidArea = trapezoidArea + (nextP - previousP) * (previousL + nextL);
    sumAbsolute = sumAbsolute + w * abs(x - mean);
    if (x > 0.0) {
      let logRatio = log(x / mean);
      sumTheilT = sumTheilT + w * (x / mean) * logRatio;
      sumLog = sumLog + w * logRatio;
      if (usePower) {
        sumPower = sumPower + w * exp(power * logRatio);
      }
    }
    while (nextKnot < KNOT_COUNT && f32(nextKnot) / f32(KNOT_COUNT - 1u) <= nextP) {
      lorenz[knotBase + nextKnot] = interpolateLorenz(f32(nextKnot) / f32(KNOT_COUNT - 1u), previousP, nextP, previousL, nextL);
      nextKnot = nextKnot + 1u;
    }
    if (!isBottomDone && palmaBottomTarget <= nextP) {
      bottomL = interpolateLorenz(palmaBottomTarget, previousP, nextP, previousL, nextL);
      isBottomDone = true;
    }
    if (!isTopDone && palmaTopTarget <= nextP) {
      topCutL = interpolateLorenz(palmaTopTarget, previousP, nextP, previousL, nextL);
      isTopDone = true;
    }
    previousP = nextP;
    previousL = nextL;
  }
  // Rounding can leave the last cumulative share a hair below a cut of exactly 1.
  while (nextKnot < KNOT_COUNT) {
    lorenz[knotBase + nextKnot] = previousL;
    nextKnot = nextKnot + 1u;
  }
  if (!isBottomDone) {
    bottomL = previousL;
  }
  if (!isTopDone) {
    topCutL = previousL;
  }

  stats[base + 1u] = 1.0 - trapezoidArea;
  stats[base + 2u] = sumTheilT / totalWeight;
  stats[base + 3u] = select(-sumLog / totalWeight, nan, hasZero);
  var atkinson = nan;
  if (!usePower) {
    if (!hasZero) {
      atkinson = 1.0 - exp(sumLog / totalWeight);
    }
  } else if (power > 0.0 || !hasZero) {
    let powerMean = sumPower / totalWeight;
    atkinson = 1.0 - pow(powerMean, 1.0 / power);
  }
  stats[base + 4u] = select(nan, atkinson, isFiniteBits(atkinson));
  stats[base + 5u] = sumAbsolute / (2.0 * totalIncome);
  stats[base + 6u] = select(nan, (1.0 - topCutL) / bottomL, bottomL > 0.0);`;
}

/** Lanes of the cooperative workgroup used by the global summary and pooled-Gini finish kernels. */
export const INEQUALITY_REDUCTION_LANES = 256;

/** Workgroup memory the global summary kernel needs; append to the kernel's declarations. */
export const INEQUALITY_GLOBAL_SUMS_DECLARATIONS = /* wgsl */ `
var<workgroup> laneFirst: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> laneSecond: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> laneThird: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> laneIncluded: array<u32, ${INEQUALITY_REDUCTION_LANES}>;`;

/**
 * Body of the global summary kernel that combines zone totals. One 256-lane workgroup: pass A
 * strides the zones (lane l takes zones l, l + 256, ...) and a fixed binary tree combines the lane
 * partials into the totals, which every lane then reads from workgroup memory; pass B does the same
 * for the between and within terms. Lane 0 writes the summary rows. Depth is
 * O(zones / 256 + log2 256) instead of two serial passes over all zones.
 */
export const INEQUALITY_GLOBAL_SUMS_BODY = /* wgsl */ `let lane = localInvocationIndex;
  var laneWeight = 0.0;
  var laneIncome = 0.0;
  var laneXLogX = 0.0;
  var laneCount = 0u;
  for (var zone = lane; zone < ZONE_COUNT; zone += ${INEQUALITY_REDUCTION_LANES}u) {
    let zoneCount = counts[countsOffset + zone];
    if (zoneCount == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    laneWeight = laneWeight + stats[base + 7u];
    laneIncome = laneIncome + stats[base + 8u];
    laneXLogX = laneXLogX + stats[base + 9u];
    laneCount = laneCount + zoneCount;
  }
  laneFirst[lane] = laneWeight;
  laneSecond[lane] = laneIncome;
  laneThird[lane] = laneXLogX;
  laneIncluded[lane] = laneCount;
  workgroupBarrier();
  for (var stride = ${INEQUALITY_REDUCTION_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneFirst[lane] = laneFirst[lane] + laneFirst[lane + stride];
      laneSecond[lane] = laneSecond[lane] + laneSecond[lane + stride];
      laneThird[lane] = laneThird[lane] + laneThird[lane + stride];
      laneIncluded[lane] = laneIncluded[lane] + laneIncluded[lane + stride];
    }
    workgroupBarrier();
  }
  // Every lane reads the totals, then all lanes pass a barrier before the arrays are reused.
  let totalWeight = laneFirst[0];
  let totalIncome = laneSecond[0];
  let weightedXLogX = laneThird[0];
  let includedCount = laneIncluded[0];
  workgroupBarrier();
  // Lanes contribute nothing to pass B when the pooled income is not positive.
  let hasIncome = totalIncome > 0.0;
  var laneBetween = 0.0;
  var laneWithin = 0.0;
  for (var zone = lane; zone < ZONE_COUNT; zone += ${INEQUALITY_REDUCTION_LANES}u) {
    if (!hasIncome || counts[countsOffset + zone] == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    let zoneIncome = stats[base + 8u];
    if (!(zoneIncome > 0.0)) {
      continue;
    }
    let incomeShare = zoneIncome / totalIncome;
    let populationShare = stats[base + 7u] / totalWeight;
    laneBetween = laneBetween + incomeShare * log(incomeShare / populationShare);
    laneWithin = laneWithin + incomeShare * stats[base + 2u];
  }
  laneFirst[lane] = laneBetween;
  laneSecond[lane] = laneWithin;
  workgroupBarrier();
  for (var stride = ${INEQUALITY_REDUCTION_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneFirst[lane] = laneFirst[lane] + laneFirst[lane + stride];
      laneSecond[lane] = laneSecond[lane] + laneSecond[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let nan = getNaN();
    summary[summaryOffset + 4u] = f32(includedCount);
    summary[summaryOffset + 5u] = select(nan, totalIncome / totalWeight, includedCount != 0u);
    summary[summaryOffset + 6u] = totalWeight;
    summary[summaryOffset + 7u] = totalIncome;
    if (!hasIncome) {
      summary[summaryOffset] = nan;
      summary[summaryOffset + 1u] = nan;
      summary[summaryOffset + 2u] = nan;
    } else {
      summary[summaryOffset] = weightedXLogX / totalIncome - log(totalIncome / totalWeight);
      summary[summaryOffset + 1u] = laneFirst[0];
      summary[summaryOffset + 2u] = laneSecond[0];
    }
  }`;

/** Rows per tile of the tiled pooled-Gini passes. */
export const INEQUALITY_GINI_TILE_ROWS = 256;

/**
 * Body of the pooled-Gini tile totals kernel (one thread per tile of the globally value-sorted
 * rows): sums weight and weighted income of the tile's valid rows. Valid rows are a prefix of the
 * sorted order, so a tile past the prefix is empty.
 */
export function getInequalityGiniTileTotalsBody(weightExpression: string): string {
  return /* wgsl */ `let firstPosition = index * GINI_TILE_ROWS;
  let endPosition = min(firstPosition + GINI_TILE_ROWS, ROW_COUNT);
  var tileWeight = 0.0;
  var tileIncome = 0.0;
  for (var position = firstPosition; position < endPosition; position++) {
    if (sortedValueKeys[sortedValueKeysOffset + position] == 0xffffffffu) {
      break;
    }
    let row = sortedRows[sortedRowsOffset + position];
    let w = ${weightExpression};
    tileWeight = tileWeight + w;
    tileIncome = tileIncome + w * values[valuesOffset + row];
  }
  tileTotals[tileTotalsOffset + 2u * index] = tileWeight;
  tileTotals[tileTotalsOffset + 2u * index + 1u] = tileIncome;`;
}

/** Workgroup memory the tile prefix kernel needs; append to the kernel's declarations. */
export const INEQUALITY_GINI_TILE_PREFIX_DECLARATIONS = /* wgsl */ `
var<workgroup> scanWeight: array<f32, ${INEQUALITY_REDUCTION_LANES}>;
var<workgroup> scanIncome: array<f32, ${INEQUALITY_REDUCTION_LANES}>;`;

/**
 * Body of the 256-lane kernel that turns the tile totals into exclusive tile prefixes and the
 * pooled totals. Lane l owns the contiguous chunk of tiles [l * C, min((l + 1) * C, T)) with
 * C = ceil(T / 256), sums it in order, and a fixed-order Hillis-Steele scan of the 256 chunk sums
 * gives each lane its starting offset. Only the owning lane touches its chunk, so the rewrite of
 * `tileTotals` in place is race free. Depth is O(T / 256 + log2 256), not O(T).
 */
export const INEQUALITY_GINI_TILE_PREFIX_BODY = /* wgsl */ `let lane = localInvocationIndex;
  const chunkSize: u32 = (GINI_TILE_COUNT + ${INEQUALITY_REDUCTION_LANES - 1}u) / ${INEQUALITY_REDUCTION_LANES}u;
  let firstTile = min(lane * chunkSize, GINI_TILE_COUNT);
  let endTile = min(firstTile + chunkSize, GINI_TILE_COUNT);
  var chunkWeight = 0.0;
  var chunkIncome = 0.0;
  for (var tile = firstTile; tile < endTile; tile++) {
    chunkWeight = chunkWeight + tileTotals[tileTotalsOffset + 2u * tile];
    chunkIncome = chunkIncome + tileTotals[tileTotalsOffset + 2u * tile + 1u];
  }
  scanWeight[lane] = chunkWeight;
  scanIncome[lane] = chunkIncome;
  workgroupBarrier();
  for (var offset = 1u; offset < ${INEQUALITY_REDUCTION_LANES}u; offset = offset << 1u) {
    var addWeight = 0.0;
    var addIncome = 0.0;
    if (lane >= offset) {
      addWeight = scanWeight[lane - offset];
      addIncome = scanIncome[lane - offset];
    }
    workgroupBarrier();
    scanWeight[lane] = scanWeight[lane] + addWeight;
    scanIncome[lane] = scanIncome[lane] + addIncome;
    workgroupBarrier();
  }
  // The scan is inclusive; this lane starts after the chunks before it.
  var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  if (lane > 0u) {
    cumulativeWeight = scanWeight[lane - 1u];
    cumulativeIncome = scanIncome[lane - 1u];
  }
  for (var tile = firstTile; tile < endTile; tile++) {
    let base = tileTotalsOffset + 2u * tile;
    let tileWeight = tileTotals[base];
    let tileIncome = tileTotals[base + 1u];
    tileTotals[base] = cumulativeWeight;
    tileTotals[base + 1u] = cumulativeIncome;
    cumulativeWeight = cumulativeWeight + tileWeight;
    cumulativeIncome = cumulativeIncome + tileIncome;
  }
  if (lane == ${INEQUALITY_REDUCTION_LANES - 1}u) {
    pooledTotals[pooledTotalsOffset] = scanWeight[lane];
    pooledTotals[pooledTotalsOffset + 1u] = scanIncome[lane];
  }`;

/**
 * Body of the pooled-Gini tile walk (one thread per tile): the trapezoid area of the Lorenz curve
 * over the tile's rows, starting from the exclusive prefix of the earlier tiles.
 */
export function getInequalityGiniTileAreaBody(weightExpression: string): string {
  return /* wgsl */ `let totalWeight = pooledTotals[pooledTotalsOffset];
  let totalIncome = pooledTotals[pooledTotalsOffset + 1u];
  var area = 0.0;
  if (totalIncome > 0.0) {
    let firstPosition = index * GINI_TILE_ROWS;
    let endPosition = min(firstPosition + GINI_TILE_ROWS, ROW_COUNT);
    var cumulativeWeight = tileTotals[tileTotalsOffset + 2u * index];
    var cumulativeIncome = tileTotals[tileTotalsOffset + 2u * index + 1u];
    var previousP = cumulativeWeight / totalWeight;
    var previousL = cumulativeIncome / totalIncome;
    for (var position = firstPosition; position < endPosition; position++) {
      if (sortedValueKeys[sortedValueKeysOffset + position] == 0xffffffffu) {
        break;
      }
      let row = sortedRows[sortedRowsOffset + position];
      let w = ${weightExpression};
      cumulativeWeight = cumulativeWeight + w;
      cumulativeIncome = cumulativeIncome + w * values[valuesOffset + row];
      let nextP = cumulativeWeight / totalWeight;
      let nextL = cumulativeIncome / totalIncome;
      area = area + (nextP - previousP) * (previousL + nextL);
      previousP = nextP;
      previousL = nextL;
    }
  }
  tileAreas[tileAreasOffset + index] = area;`;
}

/** Workgroup memory the pooled-Gini finish kernel needs; append to the kernel's declarations. */
export const INEQUALITY_GINI_FINISH_DECLARATIONS = /* wgsl */ `
var<workgroup> laneAreas: array<f32, ${INEQUALITY_REDUCTION_LANES}>;`;

/**
 * Body of the 256-lane kernel that sums the tile areas into the pooled Gini: lanes stride the
 * tiles, a fixed binary tree combines the partials, and lane 0 writes `1 - area` (or NaN when the
 * pooled income is not positive; every lane still passes every barrier).
 */
export const INEQUALITY_GINI_FINISH_BODY = /* wgsl */ `let lane = localInvocationIndex;
  var laneArea = 0.0;
  for (var tile = lane; tile < GINI_TILE_COUNT; tile += ${INEQUALITY_REDUCTION_LANES}u) {
    laneArea = laneArea + tileAreas[tileAreasOffset + tile];
  }
  laneAreas[lane] = laneArea;
  workgroupBarrier();
  for (var stride = ${INEQUALITY_REDUCTION_LANES / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      laneAreas[lane] = laneAreas[lane] + laneAreas[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    if (!(pooledTotals[pooledTotalsOffset + 1u] > 0.0)) {
      summary[summaryOffset + 3u] = getNaN();
    } else {
      summary[summaryOffset + 3u] = 1.0 - laneAreas[0];
    }
  }`;
