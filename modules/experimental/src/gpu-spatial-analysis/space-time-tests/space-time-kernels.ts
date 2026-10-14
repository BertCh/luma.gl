// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {PERMUTATION_RANDOM_WGSL} from '../permutation-inference/permutation-random';
import {PERMUTATION_SHUFFLE_WGSL} from '../permutation-inference/permutation-shuffle';
import {GPU_SPACE_TIME_SUMMARY} from './space-time-parameters';

/** Largest number of row blocks one (permutation, block) kernel splits the rows into. @internal */
export const SPACE_TIME_MAXIMUM_BLOCKS = 256;

/** Largest compile-time `maximumPermutations`. @internal */
export const SPACE_TIME_MAXIMUM_PERMUTATIONS = 2 ** 16;

/** Workgroup size of the (permutation, block) kernels; one workgroup shares one permutation. @internal */
export const SPACE_TIME_WORKGROUP_SIZE = 64;

/** Domains of 2^24 rows or more exceed the swap-or-not round capacity. @internal */
export const SPACE_TIME_MAXIMUM_ROWS = 2 ** 24 - 1;

/** Returns the number of workgroups per permutation for `blocks` row blocks. @internal */
export function getSpaceTimeGroupsPerSlot(blocks: number): number {
  return Math.ceil(blocks / SPACE_TIME_WORKGROUP_SIZE);
}

/** Returns `[blocks, rowsPerBlock]` for `rows` rows. @internal */
export function getSpaceTimeBlocks(rows: number): [number, number] {
  const blocks = Math.min(SPACE_TIME_MAXIMUM_BLOCKS, rows);
  return [blocks, Math.ceil(rows / blocks)];
}

/**
 * Creates the `pair-mask` node: writes 1 into `pairMask[entry]` for every CSR entry that is the
 * representative of its unordered pair `{row, other}` and 0 otherwise, so a statistic that sums
 * over entries with `pairMask != 0` visits each unordered pair exactly once.
 *
 * An entry `(row, other)` is the representative when `other` is a different valid row, it is the
 * first occurrence of `other` in `row`'s list (duplicates are ignored), and either `other > row`
 * or `row` is absent from `other`'s list. A symmetric list therefore takes each pair from its
 * lower row (as before), and a pair listed only from the higher row, as with a k-nearest-neighbor
 * list, is counted from that row instead of being dropped. The mask depends only on the pair list,
 * so it is computed once per evaluation and shared by every permutation. Lists need not be sorted;
 * the cost is a bounded scan of two rows per entry.
 *
 * @internal
 */
export function createSpaceTimePairMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rows: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    pairMask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-pair-mask`,
    operation: props.operation,
    variant: 'pair-mask',
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'pairMask', view: props.pairMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.rows,
    declarations: `const ROWS: u32 = ${props.rows}u;`,
    body: `let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var entry = start; entry < end; entry++) {
    let other = neighbors[neighborsOffset + entry];
    var counted = other < ROWS && other != index;
    if (counted) {
      for (var earlier = start; earlier < entry; earlier++) {
        if (neighbors[neighborsOffset + earlier] == other) {
          counted = false;
          break;
        }
      }
    }
    if (counted && other < index) {
      for (var mirror = offsets[offsetsOffset + other]; mirror < offsets[offsetsOffset + other + 1u]; mirror++) {
        if (neighbors[neighborsOffset + mirror] == index) {
          counted = false;
          break;
        }
      }
    }
    pairMask[pairMaskOffset + entry] = select(0u, 1u, counted);
  }`
  });
}

/**
 * WGSL common to the space-time kernels. Requires a `parameters` `u32` binding and defines
 * `ROWS`, `BLOCKS`, `ROWS_PER_BLOCK`, `MAXIMUM_PERMUTATIONS`.
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
`;
}

/**
 * WGSL of the swap-or-not shuffle for the (permutation, block) kernels (append after
 * {@link getSpaceTimeCommonWGSL}). Defines `SPACE_TIME_WORKGROUP_SIZE`, `GROUPS_PER_SLOT` and
 * `getPermutedRow(row, slot)`, which maps a row through permutation number `slot` of the
 * swap-or-not shuffle (slot 0 is the identity). The round material of the workgroup's slot must
 * have been loaded with `loadSwapOrNotRounds` followed by a `workgroupBarrier()`.
 *
 * @internal
 */
export function getSpaceTimeShuffleWGSL(blocks: number): string {
  return /* wgsl */ `
${PERMUTATION_SHUFFLE_WGSL}
const SPACE_TIME_WORKGROUP_SIZE: u32 = ${SPACE_TIME_WORKGROUP_SIZE}u;
const GROUPS_PER_SLOT: u32 = ${getSpaceTimeGroupsPerSlot(blocks)}u;
fn getPermutedRow(row: u32, slot: u32) -> u32 {
  if (slot == 0u) {
    return row;
  }
  return getSwapOrNotPermutationIndex(row, ROWS);
}
`;
}

/**
 * Body prologue of the (permutation, block) kernels: declares `slot`, `block`, `permutations` and
 * `runsSlot` (the workgroup runs a permutation and this invocation owns a block), loads the shuffle
 * rounds and reaches the barrier in uniform control flow. Use with `guardIndex: false` and
 * `invocationCount = slotCount * groupsPerSlot * SPACE_TIME_WORKGROUP_SIZE`.
 *
 * @internal
 */
export const SPACE_TIME_SLOT_PROLOGUE_WGSL = /* wgsl */ `let workgroup = index / SPACE_TIME_WORKGROUP_SIZE;
  let slot = workgroup / GROUPS_PER_SLOT;
  let block = (workgroup - slot * GROUPS_PER_SLOT) * SPACE_TIME_WORKGROUP_SIZE + localInvocationIndex;
  let permutations = readPermutationCount();
  if (slot > 0u && slot <= permutations) {
    loadSwapOrNotRounds(readSeedKey(), slot, ROWS, localInvocationIndex, SPACE_TIME_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  let runsSlot = slot <= permutations && block < BLOCKS;`;

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
