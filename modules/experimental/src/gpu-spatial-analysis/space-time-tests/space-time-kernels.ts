// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PERMUTATION_RANDOM_WGSL} from '../permutation-inference/permutation-random';
import {GPU_SPACE_TIME_SUMMARY} from './space-time-parameters';

/** Largest number of row blocks one (permutation, block) kernel splits the rows into. @internal */
export const SPACE_TIME_MAXIMUM_BLOCKS = 256;

/** Largest compile-time `maximumPermutations`. @internal */
export const SPACE_TIME_MAXIMUM_PERMUTATIONS = 2 ** 16;

/** Returns `[blocks, rowsPerBlock]` for `rows` rows. @internal */
export function getSpaceTimeBlocks(rows: number): [number, number] {
  const blocks = Math.min(SPACE_TIME_MAXIMUM_BLOCKS, rows);
  return [blocks, Math.ceil(rows / blocks)];
}

/**
 * WGSL common to the space-time kernels. Requires a `parameters` `u32` binding and defines
 * `ROWS`, `BLOCKS`, `ROWS_PER_BLOCK`, `MAXIMUM_PERMUTATIONS`; `getPermutedRow(row, slot, keys,
 * halfBits)` maps a row through permutation `slot - 1` (slot 0 is the identity).
 *
 * @internal
 */
export function getSpaceTimeCommonWGSL(
  rows: number,
  maximumPermutations: number,
  blocks: number,
  rowsPerBlock: number
): string {
  return /* wgsl */ `
${PERMUTATION_RANDOM_WGSL}
const ROWS: u32 = ${rows}u;
const BLOCKS: u32 = ${blocks}u;
const ROWS_PER_BLOCK: u32 = ${rowsPerBlock}u;
const MAXIMUM_PERMUTATIONS: u32 = ${maximumPermutations}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn readTimeThreshold() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
fn getPermutedRow(row: u32, slot: u32, keys: vec4<u32>, halfBits: u32) -> u32 {
  if (slot == 0u) {
    return row;
  }
  return getFeistelPermutationIndex(row, ROWS, halfBits, keys);
}
`;
}

/**
 * WGSL of the summary kernel body. Bindings: `statistics` (`array<STAT>`), `parameters`, and
 * writes `summary` (`array<f32>`, `GPU_SPACE_TIME_SUMMARY_LENGTH` words). `STAT` is `u32` or
 * `f32`. The caller defines `observedPairs`, `timeClosePairs` and `expected` as f32 lets before
 * this body is spliced, which then writes the whole summary.
 *
 * @internal
 */
export function getSpaceTimeSummaryBodyWGSL(statisticType: 'u32' | 'f32'): string {
  const index = GPU_SPACE_TIME_SUMMARY;
  const toFloat = (expression: string) =>
    statisticType === 'u32' ? `f32(${expression})` : expression;
  return /* wgsl */ `
  let permutations = readPermutationCount();
  let observed = statistics[statisticsOffset];
  var greater = 0u;
  var lesser = 0u;
  var sumDeviation = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = statistics[statisticsOffset + slot];
    if (value >= observed) { greater++; }
    if (value <= observed) { lesser++; }
    sumDeviation += ${statisticType === 'u32' ? 'f32(i32(value) - i32(observed))' : 'value - observed'};
  }
  let permutationCount = f32(permutations);
  let meanDeviation = sumDeviation / permutationCount;
  var sumSquares = 0.0;
  for (var slot = 1u; slot <= permutations; slot++) {
    let value = statistics[statisticsOffset + slot];
    let deviation = ${statisticType === 'u32' ? 'f32(i32(value) - i32(observed))' : 'value - observed'} - meanDeviation;
    sumSquares += deviation * deviation;
  }
  let variance = select(0.0, sumSquares / (permutationCount - 1.0), permutations > 1u);
  let mean = ${toFloat('observed')} + meanDeviation;
  let pGreater = f32(greater + 1u) / (permutationCount + 1.0);
  let pLesser = f32(lesser + 1u) / (permutationCount + 1.0);
  summary[summaryOffset + ${index.observed}u] = ${toFloat('observed')};
  summary[summaryOffset + ${index.pairCount}u] = observedPairs;
  summary[summaryOffset + ${index.timeClosePairs}u] = timeClosePairs;
  summary[summaryOffset + ${index.expected}u] = expected;
  summary[summaryOffset + ${index.permutationMean}u] = mean;
  summary[summaryOffset + ${index.permutationVariance}u] = variance;
  summary[summaryOffset + ${index.greaterCount}u] = f32(greater);
  summary[summaryOffset + ${index.lesserCount}u] = f32(lesser);
  summary[summaryOffset + ${index.pseudoPGreater}u] = pGreater;
  summary[summaryOffset + ${index.pseudoPLesser}u] = pLesser;
  summary[summaryOffset + ${index.pseudoPTwoSided}u] = min(1.0, 2.0 * min(pGreater, pLesser));
  let zScore = (${toFloat('observed')} - mean) / sqrt(variance);
  summary[summaryOffset + ${index.zScore}u] = select(bitcast<f32>(parameters[parametersOffset + 3u] | 0x7fc00000u), zScore, variance > 0.0);
`;
}
