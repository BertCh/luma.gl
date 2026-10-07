// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

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

/** Body of the single-thread global summary kernel that combines zone totals. */
export const INEQUALITY_GLOBAL_SUMS_BODY = /* wgsl */ `let nan = getNaN();
  var totalWeight = 0.0;
  var totalIncome = 0.0;
  var weightedXLogX = 0.0;
  var includedCount = 0u;
  for (var zone = 0u; zone < ZONE_COUNT; zone++) {
    let zoneCount = counts[countsOffset + zone];
    if (zoneCount == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    totalWeight = totalWeight + stats[base + 7u];
    totalIncome = totalIncome + stats[base + 8u];
    weightedXLogX = weightedXLogX + stats[base + 9u];
    includedCount = includedCount + zoneCount;
  }
  summary[summaryOffset + 4u] = f32(includedCount);
  summary[summaryOffset + 5u] = select(nan, totalIncome / totalWeight, includedCount != 0u);
  summary[summaryOffset + 6u] = totalWeight;
  summary[summaryOffset + 7u] = totalIncome;
  if (!(totalIncome > 0.0)) {
    summary[summaryOffset] = nan;
    summary[summaryOffset + 1u] = nan;
    summary[summaryOffset + 2u] = nan;
    return;
  }
  var between = 0.0;
  var within = 0.0;
  for (var zone = 0u; zone < ZONE_COUNT; zone++) {
    if (counts[countsOffset + zone] == 0u) {
      continue;
    }
    let base = statsOffset + zone * STATS_STRIDE;
    let zoneIncome = stats[base + 8u];
    if (!(zoneIncome > 0.0)) {
      continue;
    }
    let incomeShare = zoneIncome / totalIncome;
    let populationShare = stats[base + 7u] / totalWeight;
    between = between + incomeShare * log(incomeShare / populationShare);
    within = within + incomeShare * stats[base + 2u];
  }
  summary[summaryOffset] = weightedXLogX / totalIncome - log(totalIncome / totalWeight);
  summary[summaryOffset + 1u] = between;
  summary[summaryOffset + 2u] = within;`;

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

/**
 * Body of the single-thread kernel that turns the tile totals into exclusive tile prefixes and the
 * pooled totals. It walks `ROW_COUNT / GINI_TILE_ROWS` tiles, not rows.
 */
export const INEQUALITY_GINI_TILE_PREFIX_BODY = /* wgsl */ `var cumulativeWeight = 0.0;
  var cumulativeIncome = 0.0;
  for (var tile = 0u; tile < GINI_TILE_COUNT; tile++) {
    let base = tileTotalsOffset + 2u * tile;
    let tileWeight = tileTotals[base];
    let tileIncome = tileTotals[base + 1u];
    tileTotals[base] = cumulativeWeight;
    tileTotals[base + 1u] = cumulativeIncome;
    cumulativeWeight = cumulativeWeight + tileWeight;
    cumulativeIncome = cumulativeIncome + tileIncome;
  }
  pooledTotals[pooledTotalsOffset] = cumulativeWeight;
  pooledTotals[pooledTotalsOffset + 1u] = cumulativeIncome;`;

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

/** Body of the single-thread kernel that sums the tile areas into the pooled Gini. */
export const INEQUALITY_GINI_FINISH_BODY = /* wgsl */ `if (!(pooledTotals[pooledTotalsOffset + 1u] > 0.0)) {
    summary[summaryOffset + 3u] = getNaN();
    return;
  }
  var area = 0.0;
  for (var tile = 0u; tile < GINI_TILE_COUNT; tile++) {
    area = area + tileAreas[tileAreasOffset + tile];
  }
  summary[summaryOffset + 3u] = 1.0 - area;`;
