// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createPairStatisticsClearNode,
  getPairStatisticsSharedWGSL,
  getPairStatisticsSortedPointsNodes
} from './pair-statistics-grid';

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

/**
 * Terms above `2^32 - 256` units go to a coarser "large" channel in units of
 * `2^LARGE_TERM_SHIFT` (see `accumulateScaled`): a term that large is rounded to a relative
 * precision of at most `2^-(32 - LARGE_TERM_SHIFT - 1)`.
 *
 * @internal
 */
export const PAIR_HISTOGRAM_LARGE_TERM_SHIFT = 12;

/** Largest large-channel amount in units of `2^LARGE_TERM_SHIFT`: `2^63`, so `high <= 2^31`. */
const MAXIMUM_LARGE_AMOUNT = '9223372036854775808.0';

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
   * `'unordered'` visits each pair exactly once, from whichever row comes first in cell order, so
   * `focus` may be the lower or the higher row: the action must be symmetric in the two rows, and
   * `deltaX`/`deltaY` are always oriented from the lower to the higher row ID (so a direction
   * folded modulo a half turn does not depend on cell order). `'ordered'` visits both `(i, j)` and
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
   * `deltaX`, `deltaY` (neighbor minus focus; see `pairOrder` for unordered pairs),
   * `distanceSquared` and `pairDistance` in scope. It
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
   * `slotCount * channelCount * 2 + 1` words: per accumulator `(slot * channelCount + channel)`
   * the low and high `u32` words of an exact unsigned 64-bit sum, then one overflow word that is
   * 1 when any 64-bit sum (workgroup partial or global) carried out of bit 63 or a single term
   * was clamped, in which case the sums are incomplete (see `readPairOverflow`).
   */
  accumulators: GraphDataView<'uint32'>;
  /**
   * Included rows in cell order as `[x bits, y bits, row]` triples (`3 * rows` words, only the
   * first `3 * n` written), for follow-up kernels that scan the same neighborhoods.
   */
  sortedPoints: GraphDataView<'uint32'>;
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
  // Contributors validate their own counts before building a histogram.
  assert(Number.isInteger(slotCount) && slotCount >= 1);
  assert(Number.isInteger(channelCount) && channelCount >= 1);
  if (slotCount * channelCount > PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS) {
    throw new Error(
      `${id} needs at most ${PAIR_HISTOGRAM_MAXIMUM_ACCUMULATORS} histogram accumulators (bins x channels)`
    );
  }
}

/**
 * Builds the shared pair-histogram kernel: one thread per included focus row (in cell order, so a
 * workgroup's foci are spatially coherent) visits every included row of the 3x3 cell
 * neighborhood within the per-frame `maximumDistance` and lets the contributor accumulate integer
 * amounts into `slotCount x channelCount` histogram accumulators.
 *
 * Determinism without float atomics: each workgroup accumulates into shared-memory 64-bit
 * accumulators (two `atomic<u32>` words with carry detection from the `atomicAdd` return), then
 * adds its partials into global 64-bit accumulators the same way. Integer addition is associative,
 * so the result is exact and independent of scheduling. Float quantities are accumulated in fixed
 * point: the contributor scales a non-negative term to at most about `2^24` before
 * `quantizePairAmount`, which leaves 40 bits of headroom (about 10^12 pairs per accumulator).
 *
 * Cost is the number of candidate pairs in the 3x3 neighborhoods. Candidates are read from a
 * cell-ordered copy of the points (sequential loads, no dependent gather). An unordered histogram
 * walks only the forward half of the stencil (the rest of the focus cell, then the cell to its
 * right and the three cells below): in cell order that is two contiguous slot ranges, each pair is
 * met once, and no ID test is needed, which halves the candidate reads and distance tests of the
 * naive 3x3 loop. When `maximumDistance` covers the extent the lattice has one cell and this is a
 * plain all-pairs loop, `n^2 / 2` (unordered) or `n^2` (ordered) pair visits.
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
  // The kernel declares three extra binding slots.
  assert(extraBindings.length <= 3);
  const accumulatorCount = slotCount * channelCount;
  const accumulators = createTransientView(
    graph,
    `${id}-accumulators`,
    'uint32',
    accumulatorCount * 2 + 1
  );
  const rows = positions.length;
  const isOrdered = props.pairOrder === 'ordered';
  const sorted = getPairStatisticsSortedPointsNodes<Parameters>(graph, {
    id,
    operation,
    positions,
    sortedRows,
    cellOffsets,
    gridSize
  });
  const workgroupSize = PAIR_HISTOGRAM_WORKGROUP_SIZE;
  // Visits the candidate slots [begin, end) of the cell-ordered points.
  const visitCandidates = (
    begin: string,
    end: string
  ) => `for (var candidateSlot = ${begin}; candidateSlot < ${end}; candidateSlot++) {
        let neighbor = sortedPoints[sortedPointsOffset + candidateSlot * 3u + 2u];
        if (${isOrdered ? 'neighbor != focus' : 'true'}) {
          let neighborX = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u]);
          let neighborY = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u + 1u]);
          ${
            isOrdered
              ? `let deltaX = neighborX - x;
          let deltaY = neighborY - y;`
              : `// Oriented from the lower to the higher row ID, whichever row is the focus.
          let focusIsLower = focus < neighbor;
          let deltaX = select(x - neighborX, neighborX - x, focusIsLower);
          let deltaY = select(y - neighborY, neighborY - y, focusIsLower);`
          }
          let distanceSquared = deltaX * deltaX + deltaY * deltaY;
          if (distanceSquared <= lattice.radiusSquared) {
            let pairDistance = sqrt(distanceSquared);
            ${props.pairAction}
          }
        }
      }`;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...sorted.nodes,
    createPairStatisticsClearNode<Parameters>(graph, `${id}-clear`, operation, accumulators),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-pairs`,
      operation,
      variant: `pair-histogram-${props.pairOrder}`,
      bindings: [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'sortedPoints', view: sorted.sortedPoints, type: 'u32', access: 'read'},
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

// Adds the 64-bit amount (highAmount * 2^32 + amount) to an accumulator. highAmount is at most
// 2^31. A carry out of bit 63 of the workgroup partial sets the global overflow flag.
fn accumulateWide(slot: u32, channel: u32, amount: u32, highAmount: u32) {
  if ((amount == 0u && highAmount == 0u) || slot >= SLOT_COUNT) {
    return;
  }
  let word = slot * CHANNEL_COUNT + channel;
  var highAddend = highAmount;
  if (amount != 0u) {
    let previous = atomicAdd(&localLow[word], amount);
    if (previous > 0xffffffffu - amount) {
      highAddend += 1u;
    }
  }
  if (highAddend != 0u) {
    let previousHigh = atomicAdd(&localHigh[word], highAddend);
    if (previousHigh > 0xffffffffu - highAddend) {
      atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
    }
  }
}

fn accumulate(slot: u32, channel: u32, amount: u32) {
  accumulateWide(slot, channel, amount, 0u);
}

// Rounds a non-negative scaled term to a u32 amount; negative or NaN terms give 0.
fn quantizePairAmount(scaledValue: f32) -> u32 {
  return select(0u, u32(min(scaledValue + 0.5, ${MAXIMUM_QUANTIZED_AMOUNT})), scaledValue > 0.0);
}

// Accumulates a non-negative scaled term without letting its magnitude set the rounding unit:
// a term up to 2^32 - 256 units is rounded to the nearest unit into smallChannel; a larger
// term (rare: an outlier or sentinel pair) is rounded to a multiple of 2^${PAIR_HISTOGRAM_LARGE_TERM_SHIFT} units (relative
// error below 2^-20, plus the f32 ulp) and added as an exact 64-bit amount to largeChannel,
// which readers scale back by 2^${PAIR_HISTOGRAM_LARGE_TERM_SHIFT}. A term beyond 2^${63 + PAIR_HISTOGRAM_LARGE_TERM_SHIFT} units is clamped and
// raises the overflow flag, as does any accumulator sum that carries out of 64 bits.
fn accumulateScaled(slot: u32, smallChannel: u32, largeChannel: u32, scaledValue: f32) {
  if (!(scaledValue > 0.0)) {
    return;
  }
  if (scaledValue <= ${MAXIMUM_QUANTIZED_AMOUNT}) {
    accumulateWide(slot, smallChannel, u32(scaledValue + 0.5), 0u);
    return;
  }
  var coarse = scaledValue * ${1 / 2 ** PAIR_HISTOGRAM_LARGE_TERM_SHIFT};
  if (!(coarse <= ${MAXIMUM_LARGE_AMOUNT})) {
    atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
    coarse = ${MAXIMUM_LARGE_AMOUNT};
  }
  // Exact in f32: scaling by powers of two, and coarse - high * 2^32 has no more bits than coarse.
  let high = floor(coarse * 2.3283064365386963e-10);
  let low = coarse - high * 4294967296.0;
  accumulateWide(slot, largeChannel, u32(min(low + 0.5, ${MAXIMUM_QUANTIZED_AMOUNT})), u32(high));
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
    let focus = sortedPoints[sortedPointsOffset + index * 3u + 2u];
    let x = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u]);
    let y = bitcast<f32>(sortedPoints[sortedPointsOffset + index * 3u + 1u]);
    ${props.focusPrologue ?? ''}
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    ${
      isOrdered
        ? `let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      ${visitCandidates('begin', 'end')}
    }`
        : `// Forward half stencil: the cells after the focus cell in row-major cell order. The rest
    // of the focus row up to the right neighbor cell is one slot range, the row below another.
    {
      let rowEnd = cellOffsets[cellOffsetsOffset + row * lattice.columns + lastColumn + 1u];
      ${visitCandidates('index + 1u', 'rowEnd')}
    }
    if (row + 1u < lattice.rows) {
      let rowBase = (row + 1u) * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      ${visitCandidates('begin', 'end')}
    }`
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
      let highAddend = high + carry;
      if (highAddend < high) {
        atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
      }
      if (highAddend != 0u) {
        let previousHigh = atomicAdd(&accumulators[accumulatorsOffset + word * 2u + 1u], highAddend);
        if (previousHigh > 0xffffffffu - highAddend) {
          atomicStore(&accumulators[accumulatorsOffset + ACCUMULATOR_COUNT * 2u], 1u);
        }
      }
    }
  }
`
    })
  ];
  return {nodes, accumulators, sortedPoints: sorted.sortedPoints};
}

/**
 * WGSL helpers for kernels that read pair-histogram accumulators bound as `array<u32>` under
 * `bindingName`: `readPairAccumulator(slot, channel) -> f32` converts the exact 64-bit sum to f32
 * (one rounding), and `readPairCount(slot, channel) -> u32` returns the sum saturated to
 * `0xffffffff` (exact below 2^32; a nonzero high word saturates, matching the Ripley and
 * correlogram counts). With `slotCount`, `readPairOverflow() -> bool` also reads the overflow word
 * of the accumulators (see {@link PairHistogram.accumulators}).
 *
 * @internal
 */
export function getPairHistogramReadWGSL(
  bindingName: string,
  channelCount: number,
  functionSuffix = '',
  slotCount?: number
): string {
  const overflowReader =
    slotCount === undefined
      ? ''
      : `
fn readPairOverflow${functionSuffix}() -> bool {
  return ${bindingName}[${bindingName}Offset + ${slotCount * channelCount * 2}u] != 0u;
}
`;
  return /* wgsl */ `
fn readPairAccumulator${functionSuffix}(slot: u32, channel: u32) -> f32 {
  let word = (slot * ${channelCount}u + channel) * 2u;
  return f32(${bindingName}[${bindingName}Offset + word + 1u]) * 4294967296.0 +
    f32(${bindingName}[${bindingName}Offset + word]);
}

fn readPairCount${functionSuffix}(slot: u32, channel: u32) -> u32 {
  let word = (slot * ${channelCount}u + channel) * 2u;
  return select(
    ${bindingName}[${bindingName}Offset + word],
    0xffffffffu,
    ${bindingName}[${bindingName}Offset + word + 1u] != 0u
  );
}
${overflowReader}`;
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

/**
 * CPU mirror of `accumulateScaled`'s routing for oracles: returns which channel a scaled term
 * lands in and its integer amount in that channel's unit (`2^PAIR_HISTOGRAM_LARGE_TERM_SHIFT`
 * units for `'large'`), with f32 arithmetic where the kernel uses it.
 *
 * @internal
 */
export function quantizeScaledTermOnCPU(scaledValue: number): {
  channel: 'none' | 'small' | 'large';
  amount: bigint;
  overflow: boolean;
} {
  if (!(scaledValue > 0)) {
    return {channel: 'none', amount: 0n, overflow: false};
  }
  const scaled = Math.fround(scaledValue);
  if (scaled <= 4294967040) {
    return {
      channel: 'small',
      amount: BigInt(Math.floor(Math.fround(scaled + 0.5))),
      overflow: false
    };
  }
  let coarse = Math.fround(scaled * 2 ** -PAIR_HISTOGRAM_LARGE_TERM_SHIFT);
  let overflow = false;
  if (!(coarse <= 2 ** 63)) {
    overflow = true;
    coarse = 2 ** 63;
  }
  const high = Math.floor(Math.fround(coarse * 2 ** -32));
  const low = Math.fround(coarse - high * 2 ** 32);
  const amount =
    BigInt(high) * 4294967296n + BigInt(Math.floor(Math.min(Math.fround(low + 0.5), 4294967040)));
  return {channel: 'large', amount, overflow};
}
