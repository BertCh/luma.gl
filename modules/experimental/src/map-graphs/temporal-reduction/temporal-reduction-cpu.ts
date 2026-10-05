// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_TEMPORAL_REDUCTION_NO_CELL} from './temporal-reduction-parameters';

/** Inputs of {@link reduceTemporalBucketsOnCPU}. */
export type GPUTemporalReductionCPUInput = {
  /** Cell index per row; `0xffffffff` or any value `>= cellCount` skips the row. */
  cellIds: ArrayLike<number>;
  /** Float32 relative times, or Int64 times as `BigInt64Array` (word mode). */
  timestamps: Float32Array | BigInt64Array;
  /** Value per row. Rows with a NaN value are skipped. */
  values: Float32Array;
  /** Optional per-row mask; zero skips the row. */
  mask?: ArrayLike<number>;
  /** Compile-time cell count. */
  cellCount: number;
  /** Compile-time bucket count. */
  bucketCount: number;
  /** Start of bucket 0. Integer in word mode. */
  origin: number | bigint;
  /** Bucket width. Integer in `[1, 2^32 - 1]` in word mode. */
  width: number | bigint;
};

/** Dense result of {@link reduceTemporalBucketsOnCPU}, indexed by `cell * bucketCount + bucket`. */
export type GPUTemporalReductionCPUResult = {
  /** Rows per slot. */
  count: Uint32Array;
  /** Minimum value, NaN for an empty slot. */
  min: Float32Array;
  /** Maximum value, NaN for an empty slot. */
  max: Float32Array;
  /** Value at the earliest time, lowest row on ties; NaN for an empty slot. */
  first: Float32Array;
  /** Value at the latest time, lowest row on ties; NaN for an empty slot. */
  last: Float32Array;
  /** Occupied slot IDs in ascending order. */
  occupiedSlots: number[];
};

const scratchFloat = new Float32Array(1);
const scratchBits = new Uint32Array(scratchFloat.buffer);

/** Order-preserving u32 encoding of a non-NaN f32, the same one the GPU kernels use. */
export function getOrderedFloatKey(value: number): number {
  scratchFloat[0] = value;
  const bits = scratchBits[0];
  return ((bits & 0x80000000) !== 0 ? ~bits : bits ^ 0x80000000) >>> 0;
}

/**
 * CPU oracle for `GPUTemporalReduction`, bit-identical to the GPU for the same bucket assignment.
 *
 * Float32 mode assigns the unique `b` with `fround(b * width) <= d < fround((b + 1) * width)` where
 * `d = fround(t - origin)`, found as `floor(d / width)` plus one correction step each way, exactly as
 * the kernel does; rows whose bucket is NaN, below 0 or `>= bucketCount` are dropped (not clamped). Word mode uses exact BigInt division. Min and
 * max compare by the order-preserving key, so `-0` sorts below `+0`. First and last choose the
 * earliest and latest time (`+0` and `-0` times are equal), ties broken by the lower row index.
 */
export function reduceTemporalBucketsOnCPU(
  input: GPUTemporalReductionCPUInput
): GPUTemporalReductionCPUResult {
  const {cellCount, bucketCount, values} = input;
  const slotCount = cellCount * bucketCount;
  const count = new Uint32Array(slotCount);
  const min = new Float32Array(slotCount).fill(NaN);
  const max = new Float32Array(slotCount).fill(NaN);
  const first = new Float32Array(slotCount).fill(NaN);
  const last = new Float32Array(slotCount).fill(NaN);
  const minKeys = new Array<number>(slotCount).fill(0xffffffff);
  const maxKeys = new Array<number>(slotCount).fill(0);
  const firstTimes = new Array<number>(slotCount).fill(0xffffffff);
  const lastTimes = new Array<number>(slotCount).fill(0);
  const firstRows = new Array<number>(slotCount).fill(-1);
  const lastRows = new Array<number>(slotCount).fill(-1);
  const isWordMode = input.timestamps instanceof BigInt64Array;
  const width = isWordMode ? BigInt(input.width) : Math.fround(Number(input.width));
  const origin = isWordMode ? BigInt(input.origin) : Math.fround(Number(input.origin));

  for (let row = 0; row < values.length; row++) {
    const cell = input.cellIds[row];
    const value = values[row];
    if (
      cell === GPU_TEMPORAL_REDUCTION_NO_CELL ||
      cell >= cellCount ||
      Number.isNaN(value) ||
      (input.mask && input.mask[row] === 0)
    ) {
      continue;
    }
    let bucket: number;
    let timeKey: number;
    if (isWordMode) {
      const difference = (input.timestamps as BigInt64Array)[row] - (origin as bigint);
      if (difference < 0n || difference / (width as bigint) >= BigInt(bucketCount)) {
        continue;
      }
      bucket = Number(difference / (width as bigint));
      timeKey = Number(difference % (width as bigint));
    } else {
      const time = (input.timestamps as Float32Array)[row];
      if (!(width > 0) || !Number.isFinite(width)) {
        continue;
      }
      const difference = Math.fround(time - (origin as number));
      let bucketFloat = Math.floor(Math.fround(difference / (width as number)));
      if (Number.isNaN(bucketFloat) || bucketFloat < -1 || bucketFloat > bucketCount + 1) {
        continue;
      }
      if (difference < Math.fround(bucketFloat * (width as number))) {
        bucketFloat -= 1;
      } else if (difference >= Math.fround((bucketFloat + 1) * (width as number))) {
        bucketFloat += 1;
      }
      if (bucketFloat < 0 || bucketFloat >= bucketCount) {
        continue;
      }
      bucket = bucketFloat;
      timeKey = getOrderedFloatKey(time === 0 ? 0 : time);
    }
    const slot = cell * bucketCount + bucket;
    const valueKey = getOrderedFloatKey(value);
    count[slot]++;
    minKeys[slot] = Math.min(minKeys[slot], valueKey);
    maxKeys[slot] = Math.max(maxKeys[slot], valueKey);
    // Strict comparisons keep the lowest row among equal times because rows are visited ascending.
    if (firstRows[slot] < 0 || timeKey < firstTimes[slot]) {
      firstTimes[slot] = timeKey;
      firstRows[slot] = row;
    }
    if (lastRows[slot] < 0 || timeKey > lastTimes[slot]) {
      lastTimes[slot] = timeKey;
      lastRows[slot] = row;
    }
  }

  const occupiedSlots: number[] = [];
  for (let slot = 0; slot < slotCount; slot++) {
    if (count[slot] === 0) {
      continue;
    }
    occupiedSlots.push(slot);
    min[slot] = decodeOrderedFloatKey(minKeys[slot]);
    max[slot] = decodeOrderedFloatKey(maxKeys[slot]);
    first[slot] = values[firstRows[slot]];
    last[slot] = values[lastRows[slot]];
  }
  return {count, min, max, first, last, occupiedSlots};
}

function decodeOrderedFloatKey(key: number): number {
  scratchBits[0] = ((key & 0x80000000) !== 0 ? key ^ 0x80000000 : ~key) >>> 0;
  return scratchFloat[0];
}
