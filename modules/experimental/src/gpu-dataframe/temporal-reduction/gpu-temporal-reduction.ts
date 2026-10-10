// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {TIME_WORDS_WGSL} from '../time-window-filter/time-words';
import {
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH
} from './temporal-reduction-parameters';

const OPERATION = 'GPUTemporalReduction';
/** Largest slot count reduced in workgroup-private tables (5 words per slot). */
const PRIVATIZED_SLOT_LIMIT = 256;
const NO_SLOT = '0xffffffffu';
/** Largest slot count; `0xffffffff` is the internal "no slot" sentinel. */
const MAXIMUM_SLOT_COUNT = 0xfffffffe;
/**
 * Largest bucket count in float32 mode. Below 2^20 the 2.5 ULP error of an f32 division moves the
 * quotient by well under one bucket and `fround(b * width)` is strictly increasing in `b`.
 */
const MAXIMUM_FLOAT_BUCKET_COUNT = 2 ** 20;

/** Caller-owned dense columns of {@link GPUTemporalReduction}, indexed `cell * bucketCount + bucket`. */
export type GPUTemporalReductionOutput = {
  /** Rows reduced into each slot. Zero for an empty slot. */
  counts: GraphDataView<'uint32'>;
  /** Minimum value per slot; NaN for an empty slot. */
  min: GraphDataView<'float32'>;
  /** Maximum value per slot; NaN for an empty slot. */
  max: GraphDataView<'float32'>;
  /** Value at the earliest timestamp per slot (lowest row on ties); NaN for an empty slot. */
  first: GraphDataView<'float32'>;
  /** Value at the latest timestamp per slot (lowest row on ties); NaN for an empty slot. */
  last: GraphDataView<'float32'>;
  /**
   * Compact list of occupied slot IDs in ascending order, `ids[i] = cell * bucketCount + bucket`.
   * `count` is clamped to `ids.length`; `overflow` is 1 when more slots are occupied than fit.
   */
  occupiedSlots: GPUCompactOutput;
};

/**
 * Properties for {@link GPUTemporalReduction}.
 *
 * Per-frame (no recompile): the contents of `parameters` (bucket origin and width) and of every
 * input buffer. Topology (needs a new graph): `cellCount`, `bucketCount`, view lengths, whether
 * `mask` is present, the time format, and `occupiedSlots.ids.length`.
 */
export type GPUTemporalReductionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'temporal-reduction'`. */
  id?: string;
  /** Packed `uint32` cell index per row. `0xffffffff` and values `>= cellCount` skip the row. */
  cellIds: GraphDataView<'uint32'>;
  /**
   * Row times: packed float32 relative times, or exact Int64 words (`uint32x2` `(low, high)` rows,
   * for example from `getInt64TimeWords`). The format selects the mode and the `parameters` layout.
   */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Packed float32 value per row. Rows with a NaN value are skipped. */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame bucketing parameters. Float32 times: float32 view of at least 2 elements written with
   * `getGPUTemporalReductionParameterValues`. Word times: uint32 view of at least 4 elements
   * written with `getGPUTemporalReductionWordParameterValues`.
   */
  parameters: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Number of cells, compile-time. Row cell IDs must be below it. */
  cellCount: number;
  /** Number of coarse time buckets per cell, compile-time. */
  bucketCount: number;
  /** Caller-owned dense outputs, each at least `cellCount * bucketCount` rows. */
  output: GPUTemporalReductionOutput;
};

/**
 * Reduces timestamped rows into per-(cell, coarse time bucket) minimum, maximum, first, last, and
 * count, so a scrubbing view draws one feature per occupied slot instead of one per row.
 *
 * Bucket `b` of a row is `floor((time - origin) / width)`; rows outside `[0, bucketCount)` are
 * dropped, not clamped, so a window that covers part of the data does not pile rows into its edge
 * buckets. Skipped rows (cell `0xffffffff` or `>= cellCount`, zero mask, NaN value or time, bad
 * bucket) contribute nothing.
 *
 * Determinism: every output is exact and independent of GPU thread order. Min and max use
 * `atomicMin` and `atomicMax` on an order-preserving u32 encoding of the f32 value (`-0` sorts
 * below `+0`). First and last use two atomic passes: the earliest or latest time key per slot,
 * then the lowest row index among rows with that key, followed by a gather. In float32 mode the
 * time key is the f32 order key (`+0` and `-0` are the same time); in word mode it is the exact
 * Int64 offset inside the bucket, so equality of times is exact. No float atomics are used.
 * Float32 bucket assignment is adapter independent: with `d = fround(t - origin)`, the bucket is
 * the unique integer `b` with `fround(b * width) <= d < fround((b + 1) * width)`. The kernel
 * estimates `floor(d / width)` (WGSL division may be off by 2.5 ULP) and applies one correction
 * step down and one up using those correctly rounded products, which the CPU oracle mirrors with
 * `Math.fround`. Float mode is limited to 2^20 buckets, where the estimate is within one bucket of
 * the answer and the products are strictly increasing, so the bucket is unique; estimates beyond
 * `[-1, bucketCount + 1]` are dropped. Word mode uses exact integer division.
 *
 * The compact list of occupied slots comes from `GPUVisibilityWorkflow`, so it is ascending and
 * stable. Inputs must be single packed views (chunked vectors are not supported).
 */
export class GPUTemporalReduction implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTemporalReductionProps;
  /** `cellCount * bucketCount`. */
  readonly slotCount: number;

  constructor(props: GPUTemporalReductionProps) {
    this.id = props.id ?? 'temporal-reduction';
    this.props = props;
    const id = this.id;
    const {cellCount, bucketCount, output} = props;
    for (const [name, count] of [
      ['cellCount', cellCount],
      ['bucketCount', bucketCount]
    ] as const) {
      if (!Number.isInteger(count) || count < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    this.slotCount = cellCount * bucketCount;
    if (this.slotCount > MAXIMUM_SLOT_COUNT) {
      throw new Error(`${id} cellCount * bucketCount must not exceed ${MAXIMUM_SLOT_COUNT}`);
    }
    for (const [name, view] of [
      ['cellIds', props.cellIds],
      ['timestamps', props.timestamps],
      ['values', props.values],
      ['mask', props.mask],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rows = props.values.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    const isWordMode = props.timestamps.format === 'uint32x2';
    validatePackedUint32View(props.cellIds, `${id} cellIds`);
    validatePackedView(props.timestamps, [isWordMode ? 'uint32x2' : 'float32'], `${id} timestamps`);
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
    }
    for (const [name, view] of [
      ['cellIds', props.cellIds],
      ['timestamps', props.timestamps],
      ['mask', props.mask]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal values length`);
      }
    }
    if (isWordMode) {
      validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
      if (props.parameters.length < GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH} uint32 values`
        );
      }
    } else {
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH} float32 values`
        );
      }
      if (bucketCount > MAXIMUM_FLOAT_BUCKET_COUNT) {
        throw new Error(
          `${id} float32 times support at most ${MAXIMUM_FLOAT_BUCKET_COUNT} buckets`
        );
      }
    }
    validatePackedUint32View(output.counts, `${id} output.counts`);
    for (const name of ['min', 'max', 'first', 'last'] as const) {
      validatePackedView(output[name], ['float32'], `${id} output.${name}`);
    }
    for (const name of ['counts', 'min', 'max', 'first', 'last'] as const) {
      if (output[name].length < this.slotCount) {
        throw new Error(`${id} output.${name} must hold cellCount * bucketCount rows`);
      }
    }
    validateCompactOutput(id, output.occupiedSlots);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.counts,
        output.min,
        output.max,
        output.first,
        output.last,
        output.occupiedSlots.ids,
        output.occupiedSlots.count,
        output.occupiedSlots.overflow,
        output.occupiedSlots.requiredCount
      ],
      [props.cellIds, props.timestamps, props.values, props.mask, props.parameters]
    );
  }

  /** Returns init, classify, extreme, row-tie, finish, compaction, and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, slotCount} = this;
    const {output, cellCount, bucketCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.cellIds,
      props.timestamps,
      props.values,
      props.mask,
      props.parameters,
      output.counts,
      output.min,
      output.max,
      output.first,
      output.last,
      output.occupiedSlots.ids,
      output.occupiedSlots.count,
      output.occupiedSlots.overflow,
      output.occupiedSlots.requiredCount
    ]);
    const rows = props.values.length;
    const isWordMode = props.timestamps.format === 'uint32x2';
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const rowSlots = u32('row-slots', rows);
    const rowTimeKeys = u32('row-time-keys', rows);
    const minKeys = u32('min-keys', slotCount);
    const maxKeys = u32('max-keys', slotCount);
    const firstTimeKeys = u32('first-time-keys', slotCount);
    const lastTimeKeys = u32('last-time-keys', slotCount);
    const firstRows = u32('first-rows', slotCount);
    const lastRows = u32('last-rows', slotCount);
    const occupied = u32('occupied', slotCount);
    const {counts} = output;
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
    ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init`,
        operation: OPERATION,
        variant: 'init',
        bindings: [
          write('counts', counts),
          write('minKeys', minKeys),
          write('maxKeys', maxKeys),
          write('firstTimeKeys', firstTimeKeys),
          write('lastTimeKeys', lastTimeKeys),
          write('firstRows', firstRows),
          write('lastRows', lastRows)
        ],
        invocationCount: slotCount,
        body: `counts[countsOffset + index] = 0u;
  minKeys[minKeysOffset + index] = 0xffffffffu;
  maxKeys[maxKeysOffset + index] = 0u;
  firstTimeKeys[firstTimeKeysOffset + index] = 0xffffffffu;
  lastTimeKeys[lastTimeKeysOffset + index] = 0u;
  firstRows[firstRowsOffset + index] = 0xffffffffu;
  lastRows[lastRowsOffset + index] = 0xffffffffu;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: isWordMode ? 'classify-words' : 'classify',
        bindings: [
          read('cellIds', props.cellIds, 'u32'),
          read('timestamps', props.timestamps, isWordMode ? 'u32' : 'f32'),
          read('values', props.values, 'f32'),
          ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
          read('params', props.parameters, isWordMode ? 'u32' : 'f32'),
          write('rowSlots', rowSlots),
          write('rowTimeKeys', rowTimeKeys)
        ],
        invocationCount: rows,
        declarations: `const CELL_COUNT: u32 = ${cellCount}u;
const BUCKET_COUNT: u32 = ${bucketCount}u;
${ORDERED_KEY_WGSL}
${isWordMode ? TIME_WORDS_WGSL : ''}`,
        body: `rowSlots[rowSlotsOffset + index] = ${NO_SLOT};
  let cell = cellIds[cellIdsOffset + index];
  let value = values[valuesOffset + index];
  if (cell >= CELL_COUNT || isNanBits(value)) {
    return;
  }
  ${props.mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  var bucket: u32;
  var timeKey: u32;
  ${isWordMode ? WORD_BUCKET_BODY : FLOAT_BUCKET_BODY}
  rowSlots[rowSlotsOffset + index] = cell * BUCKET_COUNT + bucket;
  rowTimeKeys[rowTimeKeysOffset + index] = timeKey;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-extremes`,
        operation: OPERATION,
        variant: 'extremes',
        bindings: [
          read('rowSlots', rowSlots, 'u32'),
          read('rowTimeKeys', rowTimeKeys, 'u32'),
          read('values', props.values, 'f32'),
          write('counts', counts, 'atomic<u32>'),
          write('minKeys', minKeys, 'atomic<u32>'),
          write('maxKeys', maxKeys, 'atomic<u32>'),
          write('firstTimeKeys', firstTimeKeys, 'atomic<u32>'),
          write('lastTimeKeys', lastTimeKeys, 'atomic<u32>')
        ],
        invocationCount: rows,
        // With few slots every row contends on the same handful of counters, so each workgroup
        // reduces into workgroup-private tables and flushes once per occupied slot. Integer add,
        // min and max are associative, so the result is unchanged.
        ...(slotCount <= PRIVATIZED_SLOT_LIMIT
          ? {
              guardIndex: false,
              declarations: `${ORDERED_KEY_WGSL}
const SLOT_COUNT: u32 = ${slotCount}u;
var<workgroup> localCounts: array<atomic<u32>, ${slotCount}>;
var<workgroup> localMinimums: array<atomic<u32>, ${slotCount}>;
var<workgroup> localMaximums: array<atomic<u32>, ${slotCount}>;
var<workgroup> localFirstTimes: array<atomic<u32>, ${slotCount}>;
var<workgroup> localLastTimes: array<atomic<u32>, ${slotCount}>;`,
              // No early return: every invocation of a workgroup must reach the barriers.
              body: `for (var slot = localInvocationIndex; slot < SLOT_COUNT; slot += 256u) {
    atomicStore(&localCounts[slot], 0u);
    atomicStore(&localMinimums[slot], 0xffffffffu);
    atomicStore(&localMaximums[slot], 0u);
    atomicStore(&localFirstTimes[slot], 0xffffffffu);
    atomicStore(&localLastTimes[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = rowSlots[rowSlotsOffset + index];
    if (slot != ${NO_SLOT}) {
      let valueKey = getOrderedKey(values[valuesOffset + index]);
      let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
      atomicAdd(&localCounts[slot], 1u);
      atomicMin(&localMinimums[slot], valueKey);
      atomicMax(&localMaximums[slot], valueKey);
      atomicMin(&localFirstTimes[slot], timeKey);
      atomicMax(&localLastTimes[slot], timeKey);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < SLOT_COUNT; slot += 256u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&counts[countsOffset + slot], count);
      atomicMin(&minKeys[minKeysOffset + slot], atomicLoad(&localMinimums[slot]));
      atomicMax(&maxKeys[maxKeysOffset + slot], atomicLoad(&localMaximums[slot]));
      atomicMin(&firstTimeKeys[firstTimeKeysOffset + slot], atomicLoad(&localFirstTimes[slot]));
      atomicMax(&lastTimeKeys[lastTimeKeysOffset + slot], atomicLoad(&localLastTimes[slot]));
    }
  }`
            }
          : {
              declarations: ORDERED_KEY_WGSL,
              body: `let slot = rowSlots[rowSlotsOffset + index];
  if (slot == ${NO_SLOT}) {
    return;
  }
  let valueKey = getOrderedKey(values[valuesOffset + index]);
  let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
  atomicAdd(&counts[countsOffset + slot], 1u);
  atomicMin(&minKeys[minKeysOffset + slot], valueKey);
  atomicMax(&maxKeys[maxKeysOffset + slot], valueKey);
  atomicMin(&firstTimeKeys[firstTimeKeysOffset + slot], timeKey);
  atomicMax(&lastTimeKeys[lastTimeKeysOffset + slot], timeKey);`
            })
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tie-rows`,
        operation: OPERATION,
        variant: 'tie-rows',
        bindings: [
          read('rowSlots', rowSlots, 'u32'),
          read('rowTimeKeys', rowTimeKeys, 'u32'),
          read('firstTimeKeys', firstTimeKeys, 'u32'),
          read('lastTimeKeys', lastTimeKeys, 'u32'),
          write('firstRows', firstRows, 'atomic<u32>'),
          write('lastRows', lastRows, 'atomic<u32>')
        ],
        invocationCount: rows,
        body: `let slot = rowSlots[rowSlotsOffset + index];
  if (slot == ${NO_SLOT}) {
    return;
  }
  let timeKey = rowTimeKeys[rowTimeKeysOffset + index];
  if (timeKey == firstTimeKeys[firstTimeKeysOffset + slot]) {
    atomicMin(&firstRows[firstRowsOffset + slot], index);
  }
  if (timeKey == lastTimeKeys[lastTimeKeysOffset + slot]) {
    atomicMin(&lastRows[lastRowsOffset + slot], index);
  }`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-values`,
        operation: OPERATION,
        variant: 'finish-values',
        bindings: [
          read('values', props.values, 'f32'),
          read('firstRows', firstRows, 'u32'),
          read('lastRows', lastRows, 'u32'),
          read('counts', counts, 'u32'),
          write('firstOut', output.first, 'f32'),
          write('lastOut', output.last, 'f32'),
          write('occupied', occupied)
        ],
        invocationCount: slotCount,
        declarations: ORDERED_KEY_WGSL,
        body: `let count = counts[countsOffset + index];
  let isOccupied = count != 0u;
  occupied[occupiedOffset + index] = select(0u, 1u, isOccupied);
  let nan = getNaN();
  firstOut[firstOutOffset + index] = select(nan, values[valuesOffset + firstRows[firstRowsOffset + index]], isOccupied);
  lastOut[lastOutOffset + index] = select(nan, values[valuesOffset + lastRows[lastRowsOffset + index]], isOccupied);`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-extrema`,
        operation: OPERATION,
        variant: 'finish-extrema',
        bindings: [
          read('minKeys', minKeys, 'u32'),
          read('maxKeys', maxKeys, 'u32'),
          read('counts', counts, 'u32'),
          write('minOut', output.min, 'f32'),
          write('maxOut', output.max, 'f32')
        ],
        invocationCount: slotCount,
        declarations: ORDERED_KEY_WGSL,
        body: `let isOccupied = counts[countsOffset + index] != 0u;
  let nan = getNaN();
  minOut[minOutOffset + index] = select(nan, decodeOrderedKey(minKeys[minKeysOffset + index]), isOccupied);
  maxOut[maxOutOffset + index] = select(nan, decodeOrderedKey(maxKeys[maxKeysOffset + index]), isOccupied);`
      })
    );

    // Compact straight into the caller's IDs when they can hold every slot; otherwise compact into
    // full-size scratch and let the publish kernel copy the bounded prefix.
    const direct = output.occupiedSlots.ids.length >= slotCount;
    const total = u32('total', 1);
    const compactIds = direct ? output.occupiedSlots.ids : u32('compact-ids', slotCount);
    nodes.push(
      ...new GPUVisibilityWorkflow({
        id: `${id}-compact`,
        predicates: [{kind: 'selection', mask: occupied}],
        output: compactIds,
        count: total
      }).getCommandNodes(graph)
    );
    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        compactIds: direct ? undefined : compactIds,
        output: output.occupiedSlots
      })
    );
    return nodes;
  }
}

/** Order-preserving u32 encoding of f32 and NaN test by bits, shared by several kernels. */
const ORDERED_KEY_WGSL = /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`;

const FLOAT_BUCKET_BODY = /* wgsl */ `let time = timestamps[timestampsOffset + index];
  let origin = params[paramsOffset];
  let width = params[paramsOffset + 1u];
  if (isNanBits(time) || isNanBits(origin) || isNanBits(width)) {
    return;
  }
  if (!(width > 0.0) || (bitcast<u32>(width) & 0x7fffffffu) == 0x7f800000u) {
    return;
  }
  let difference = time - origin;
  var bucketFloat = floor(difference / width);
  if (isNanBits(bucketFloat) || bucketFloat < -1.0 || bucketFloat > f32(BUCKET_COUNT) + 1.0) {
    return;
  }
  // Correct the estimate with correctly rounded products: one step down, then one step up.
  if (difference < bucketFloat * width) {
    bucketFloat = bucketFloat - 1.0;
  } else if (difference >= (bucketFloat + 1.0) * width) {
    bucketFloat = bucketFloat + 1.0;
  }
  if (bucketFloat < 0.0 || bucketFloat >= f32(BUCKET_COUNT)) {
    return;
  }
  bucket = u32(bucketFloat);
  timeKey = getOrderedKey(select(time, 0.0, time == 0.0));`;

const WORD_BUCKET_BODY = /* wgsl */ `let origin = vec2<u32>(params[paramsOffset], params[paramsOffset + 1u]);
  let width = params[paramsOffset + 2u];
  if (width == 0u) {
    return;
  }
  let time = vec2<u32>(timestamps[timestampsOffset + 2u * index], timestamps[timestampsOffset + 2u * index + 1u]);
  let difference = timeWordsSubtract(time, origin);
  if (timeWordsIsNegative(difference)) {
    return;
  }
  // Exact 64-bit by 32-bit division: high word first, then the low word bit by bit.
  let quotientHigh = difference.y / width;
  var remainder = difference.y % width;
  var quotientLow = 0u;
  for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
    let carry = remainder >> 31u;
    remainder = (remainder << 1u) | ((difference.x >> u32(bit)) & 1u);
    if (carry != 0u || remainder >= width) {
      remainder = remainder - width;
      quotientLow = quotientLow | (1u << u32(bit));
    }
  }
  if (quotientHigh != 0u || quotientLow >= BUCKET_COUNT) {
    return;
  }
  bucket = quotientLow;
  timeKey = remainder;`;
