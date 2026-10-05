// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {createPairStatisticsClearNode, getPairStatisticsSharedWGSL} from './pair-statistics-grid';

/** Focus rows per pair-histogram workgroup. */
export const PAIR_HISTOGRAM_WORKGROUP_SIZE = 64;

/**
 * Largest `slotCount * channelCount`: two `atomic<u32>` words per accumulator must fit the 16 KiB
 * WebGPU default `maxComputeWorkgroupStorageSize`.
 *
 * @internal
 */
export const PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS = 2048;

/** Largest quantized amount that `quantizePairAmount` produces, `2^32 - 256` (exact in f32). */
const MAXIMUM_QUANTIZED_AMOUNT = '4294967040.0';

/** Properties of {@link getPairHistogramNodes}. @internal */
export type PairHistogramProps = {
  /** Node ID prefix. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Packed planar points. */
  positions: GraphDataView<'float32x2'>;
  /** Per-frame parameters starting with the shared pair-statistics slots. */
  parameters: GraphDataView<'float32'>;
  /** Maximum `[columns, rows]` of the cell lattice, as passed to the input nodes. */
  gridSize: readonly [number, number];
  /** Included rows grouped by cell, from `getPairStatisticsInputNodes`. */
  sortedRows: GraphDataView<'uint32'>;
  /** Cell offsets, from `getPairStatisticsInputNodes`. */
  cellOffsets: GraphDataView<'uint32'>;
  /** Number of histogram slots, for example lag bins times direction sectors. */
  slotCount: number;
  /** Number of accumulator channels per slot, for example count, sum and distance. */
  channelCount: number;
  /**
   * `'unordered'` visits each pair once (`neighbor > focus`); `'ordered'` visits both `(i, j)` and
   * `(j, i)` (`neighbor != focus`), for asymmetric pair weights or per-focus tallies.
   */
  pairOrder: 'unordered' | 'ordered';
  /** Up to three additional read-only bindings, for example values and per-frame scales. */
  extraBindings?: readonly WGSLKernelBinding[];
  /** Module-scope WGSL: helper functions used by the actions below. */
  declarations?: string;
  /**
   * WGSL run once per included focus row before its pair loop, with `focus` (row index), `x`, `y`
   * and `lattice` in scope. Must not `return` (the kernel uses workgroup barriers).
   */
  focusPrologue?: string;
  /**
   * WGSL run once per visited pair within `maximumDistance`, with `focus`, `neighbor`, `x`, `y`,
   * `deltaX`, `deltaY` (neighbor minus focus), `distanceSquared` and `pairDistance` in scope. It
   * calls `accumulate(slot, channel, amount)` with a `u32` amount (use `quantizePairAmount` for
   * non-negative floats). Must not `return`; `continue` skips the pair.
   */
  pairAction: string;
  /** WGSL run once per included focus row after its pair loop. Must not `return`. */
  focusEpilogue?: string;
};

/** Views produced by {@link getPairHistogramNodes}. @internal */
export type PairHistogram<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `slotCount * channelCount * 2` words: per accumulator `(slot * channelCount + channel)` the
   * low and high `u32` words of an exact unsigned 64-bit sum.
   */
  accumulators: GraphDataView<'uint32'>;
};

/**
 * Validates the slot and channel counts of a pair histogram.
 *
 * @internal
 */
export function validatePairHistogramShape(
  id: string,
  slotCount: number,
  channelCount: number
): void {
  if (!Number.isInteger(slotCount) || slotCount < 1) {
    throw new Error(`${id} slot count must be a positive integer`);
  }
  if (!Number.isInteger(channelCount) || channelCount < 1) {
    throw new Error(`${id} channel count must be a positive integer`);
  }
  if (slotCount * channelCount > PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS) {
    throw new Error(
      `${id} needs at most ${PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS} histogram accumulators (bins x channels)`
    );
  }
}

/**
 * Builds the shared pair-histogram kernel: one thread per included focus row (in cell order, so a
 * workgroup's foci are spatially coherent) visits every included row of the 3x3 cell
 * neighborhood within the per-frame `maximumDistance` and lets the recipe accumulate integer
 * amounts into `slotCount x channelCount` histogram accumulators.
 *
 * Determinism without float atomics: each workgroup accumulates into shared-memory 64-bit
 * accumulators (two `atomic<u32>` words with carry detection from the `atomicAdd` return), then
 * adds its partials into global 64-bit accumulators the same way. Integer addition is associative,
 * so the result is exact and independent of scheduling. Float quantities are accumulated in fixed
 * point: the recipe scales a non-negative term to at most about `2^24` before
 * `quantizePairAmount`, which leaves 40 bits of headroom (about 10^12 pairs per accumulator).
 *
 * Cost is the number of candidate pairs in the 3x3 neighborhoods. When `maximumDistance` covers the
 * extent the lattice has one cell and this is a plain all-pairs loop, `n^2 / 2` (unordered) or
 * `n^2` (ordered) pair visits.
 *
 * WGSL helpers for finishing kernels are in {@link getPairHistogramReadWGSL}.
 *
 * @internal
 */
export function getPairHistogramNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PairHistogramProps
): PairHistogram<Parameters> {
  const {id, operation, positions, parameters, gridSize, sortedRows, cellOffsets} = props;
  const {slotCount, channelCount} = props;
  validatePairHistogramShape(id, slotCount, channelCount);
  const extraBindings = props.extraBindings ?? [];
  if (extraBindings.length > 3) {
    throw new Error(`${id} pair histogram accepts at most 3 extra bindings`);
  }
  const accumulatorCount = slotCount * channelCount;
  const accumulators = createTransientView(
    graph,
    `${id}-accumulators`,
    'uint32',
    accumulatorCount * 2
  );
  const rows = positions.length;
  const neighborTest = props.pairOrder === 'unordered' ? 'neighbor > focus' : 'neighbor != focus';
  const workgroupSize = PAIR_HISTOGRAM_WORKGROUP_SIZE;
  const nodes: GPUCommandNode<Parameters>[] = [
    createPairStatisticsClearNode<Parameters>(graph, `${id}-clear`, operation, accumulators),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-pairs`,
      operation,
      variant: `pair-histogram-${props.pairOrder}`,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
        ...extraBindings,
        {
          name: 'accumulators',
          view: accumulators,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: rows,
      workgroupSize,
      guardIndex: false,
      declarations: `${getPairStatisticsSharedWGSL(gridSize)}
const SLOT_COUNT: u32 = ${slotCount}u;
const CHANNEL_COUNT: u32 = ${channelCount}u;
const ACCUMULATOR_COUNT: u32 = ${accumulatorCount}u;
var<workgroup> localLow: array<atomic<u32>, ${accumulatorCount}>;
var<workgroup> localHigh: array<atomic<u32>, ${accumulatorCount}>;

fn accumulate(slot: u32, channel: u32, amount: u32) {
  if (amount == 0u || slot >= SLOT_COUNT) {
    return;
  }
  let word = slot * CHANNEL_COUNT + channel;
  let previous = atomicAdd(&localLow[word], amount);
  if (previous > 0xffffffffu - amount) {
    atomicAdd(&localHigh[word], 1u);
  }
}

// Rounds a non-negative scaled term to a u32 amount; negative or NaN terms give 0.
fn quantizePairAmount(scaledValue: f32) -> u32 {
  return select(0u, u32(min(scaledValue + 0.5, ${MAXIMUM_QUANTIZED_AMOUNT})), scaledValue > 0.0);
}

${props.declarations ?? ''}`,
      // No early return: every invocation of a workgroup must reach the barriers.
      body: `for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += ${workgroupSize}u) {
    atomicStore(&localLow[word], 0u);
    atomicStore(&localHigh[word], 0u);
  }
  workgroupBarrier();
  let lattice = readLattice();
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  if (index < INVOCATION_COUNT && index < includedCount && lattice.valid) {
    let focus = sortedRows[sortedRowsOffset + index];
    let x = positions[positionsOffset + focus * 2u];
    let y = positions[positionsOffset + focus * 2u + 1u];
    ${props.focusPrologue ?? ''}
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var candidateSlot = begin; candidateSlot < end; candidateSlot++) {
        let neighbor = sortedRows[sortedRowsOffset + candidateSlot];
        if (${neighborTest}) {
          let deltaX = positions[positionsOffset + neighbor * 2u] - x;
          let deltaY = positions[positionsOffset + neighbor * 2u + 1u] - y;
          let distanceSquared = deltaX * deltaX + deltaY * deltaY;
          if (distanceSquared <= lattice.radiusSquared) {
            let pairDistance = sqrt(distanceSquared);
            ${props.pairAction}
          }
        }
      }
    }
    ${props.focusEpilogue ?? ''}
  }
  workgroupBarrier();
  for (var word = localInvocationIndex; word < ACCUMULATOR_COUNT; word += ${workgroupSize}u) {
    let low = atomicLoad(&localLow[word]);
    let high = atomicLoad(&localHigh[word]);
    if (low != 0u || high != 0u) {
      let previous = atomicAdd(&accumulators[accumulatorsOffset + word * 2u], low);
      let carry = select(0u, 1u, previous > 0xffffffffu - low);
      if (high + carry != 0u) {
        atomicAdd(&accumulators[accumulatorsOffset + word * 2u + 1u], high + carry);
      }
    }
  }`
    })
  ];
  return {nodes, accumulators};
}

/**
 * WGSL helpers for kernels that read pair-histogram accumulators bound as `array<u32>` under
 * `bindingName`: `readPairAccumulator(slot, channel) -> f32` converts the exact 64-bit sum to f32
 * (one rounding), and `readPairCount(slot, channel) -> u32` returns the low word (exact while the
 * sum is below 2^32).
 *
 * @internal
 */
export function getPairHistogramReadWGSL(
  bindingName: string,
  channelCount: number,
  functionSuffix = ''
): string {
  return /* wgsl */ `
fn readPairAccumulator${functionSuffix}(slot: u32, channel: u32) -> f32 {
  let word = (slot * ${channelCount}u + channel) * 2u;
  return f32(${bindingName}[${bindingName}Offset + word + 1u]) * 4294967296.0 +
    f32(${bindingName}[${bindingName}Offset + word]);
}

fn readPairCount${functionSuffix}(slot: u32, channel: u32) -> u32 {
  return ${bindingName}[${bindingName}Offset + (slot * ${channelCount}u + channel) * 2u];
}
`;
}

/**
 * CPU mirror of `quantizePairAmount` for oracles: rounds `scaledValue + 0.5` down in f32 terms,
 * returning 0 for non-positive or NaN input.
 *
 * @internal
 */
export function quantizePairAmountOnCPU(scaledValue: number): number {
  if (!(scaledValue > 0)) {
    return 0;
  }
  return Math.floor(Math.min(Math.fround(Math.fround(scaledValue) + 0.5), 4294967040));
}
