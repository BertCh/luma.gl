// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CELL_MAXIMUM_SCALED_VALUE} from '../../../src/gpu-spatial-analysis/cell-aggregation/cell-table';

const f = Math.fround;

/** Reserved "no key" values. */
export const NO_KEY_32 = 0xffffffffn;
export const NO_KEY_64 = 0xffffffffffffffffn;

/** Deterministic xorshift generator in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** `roundHalfEven(fround(value * sumScale))`, saturated like the kernel. */
export function getScaledValue(value: number, sumScale: number): bigint {
  const product = f(value * sumScale);
  const clamped = Math.min(
    Math.max(product, -CELL_MAXIMUM_SCALED_VALUE),
    CELL_MAXIMUM_SCALED_VALUE
  );
  const floor = Math.floor(clamped);
  const difference = clamped - floor;
  let rounded =
    difference > 0.5 ? floor + 1 : difference < 0.5 ? floor : floor % 2 === 0 ? floor : floor + 1;
  if (Object.is(rounded, -0)) {
    rounded = 0;
  }
  return BigInt(rounded);
}

/** The kernel's `cellI64ToF32(total) / SUM_SCALE` decode with an f32 rounding per step. */
export function decodeFixedPointSum(total: bigint, sumScale: number): number {
  const negative = total < 0n;
  const magnitude = BigInt.asUintN(64, negative ? -total : total);
  const high = Number(magnitude >> 32n);
  const low = Number(magnitude & 0xffffffffn);
  const result = f(f(high) * 4294967296 + f(low));
  return f((negative ? -result : result) / sumScale);
}

/** Order-preserving comparison treating `-0` below `+0`. */
function isLess(left: number, right: number): boolean {
  return (
    left < right || (left === 0 && right === 0 && Object.is(left, -0) && !Object.is(right, -0))
  );
}

/** Operation of an oracle aggregate. */
export type OracleAggregateOperation = 'count' | 'sum' | 'mean' | 'minimum' | 'maximum';

/** Inputs of {@link joinOnCPU}. */
export type OracleJoinInput = {
  keyBits: 32 | 64;
  leftKeys: readonly bigint[];
  rightKeys: readonly bigint[];
  leftMask?: ArrayLike<number>;
  rightMask?: ArrayLike<number>;
  kind: 'left' | 'inner';
  /** Capacity of the inner-join row output. */
  capacity?: number;
  sumScale: number;
  /** Right-aligned columns: raw 32-bit words of gathers, f32 values of aggregates. */
  gatherColumns?: readonly {words: Uint32Array; isFloat: boolean}[];
  aggregates?: readonly {
    operation: OracleAggregateOperation;
    values?: Float32Array;
  }[];
};

/** Result of {@link joinOnCPU}. */
export type OracleJoinResult = {
  /** 0xffffffff when unmatched. */
  rightRows: number[];
  matchCounts: number[];
  matched: number[];
  rightMatched: number[];
  /** Gathered words per column (NaN bits or all-ones when unmatched). */
  gathered: number[][];
  /** Aggregate values per aggregate (f32), and exact sums (null when unmatched or no sums). */
  aggregates: {values: number[]; sums: bigint[]}[];
  innerRows: number[];
  innerTotal: number;
  innerOverflow: number;
};

const NAN_BITS = 0x7fc00000;

/** Reference attribute join with the kernel's semantics. */
export function joinOnCPU(input: OracleJoinInput): OracleJoinResult {
  const noKey = input.keyBits === 32 ? NO_KEY_32 : NO_KEY_64;
  const leftCount = input.leftKeys.length;
  const rightCount = input.rightKeys.length;
  // Valid right rows grouped by key, ascending row order.
  const groups = new Map<bigint, number[]>();
  for (let row = 0; row < rightCount; row++) {
    const key = input.rightKeys[row];
    if (key === noKey || (input.rightMask && input.rightMask[row] === 0)) {
      continue;
    }
    const group = groups.get(key);
    if (group) {
      group.push(row);
    } else {
      groups.set(key, [row]);
    }
  }
  const result: OracleJoinResult = {
    rightRows: [],
    matchCounts: [],
    matched: [],
    rightMatched: new Array(rightCount).fill(0),
    gathered: (input.gatherColumns ?? []).map(() => []),
    aggregates: (input.aggregates ?? []).map(() => ({values: [], sums: []})),
    innerRows: [],
    innerTotal: 0,
    innerOverflow: 0
  };
  for (let row = 0; row < leftCount; row++) {
    const key = input.leftKeys[row];
    const rows =
      key === noKey || (input.leftMask && input.leftMask[row] === 0) ? undefined : groups.get(key);
    result.matched.push(rows ? 1 : 0);
    result.rightRows.push(rows ? rows[0] : 0xffffffff);
    result.matchCounts.push(rows ? rows.length : 0);
    if (rows) {
      for (const rightRow of rows) {
        result.rightMatched[rightRow] = 1;
      }
      result.innerTotal++;
      if (result.innerRows.length < (input.capacity ?? 0)) {
        result.innerRows.push(row);
      }
    }
    for (const [index, column] of (input.gatherColumns ?? []).entries()) {
      result.gathered[index].push(
        rows ? column.words[rows[0]] : column.isFloat ? NAN_BITS : 0xffffffff
      );
    }
    for (const [index, aggregate] of (input.aggregates ?? []).entries()) {
      const out = result.aggregates[index];
      if (!rows) {
        out.values.push(aggregate.operation === 'count' ? 0 : NaN);
        out.sums.push(0n);
        continue;
      }
      const finite = aggregate.values
        ? rows.map(rightRow => aggregate.values![rightRow]).filter(Number.isFinite)
        : [];
      let sum = 0n;
      let minimum = NaN;
      let maximum = NaN;
      for (const [position, value] of finite.entries()) {
        sum = BigInt.asIntN(64, sum + getScaledValue(value, input.sumScale));
        if (position === 0 || isLess(value, minimum)) {
          minimum = value;
        }
        if (position === 0 || isLess(maximum, value)) {
          maximum = value;
        }
      }
      out.sums.push(sum);
      const decoded = decodeFixedPointSum(sum, input.sumScale);
      switch (aggregate.operation) {
        case 'count':
          out.values.push(aggregate.values ? finite.length : rows.length);
          break;
        case 'sum':
          out.values.push(decoded);
          break;
        case 'mean':
          out.values.push(finite.length ? f(decoded / finite.length) : NaN);
          break;
        case 'minimum':
          out.values.push(minimum);
          break;
        case 'maximum':
          out.values.push(maximum);
          break;
      }
    }
  }
  result.innerOverflow = result.innerTotal > (input.capacity ?? 0) ? 1 : 0;
  return result;
}

/** Splits 64-bit keys into little-endian `(low, high)` words, or 32-bit keys into single words. */
export function packKeys(keys: readonly bigint[], keyBits: 32 | 64): Uint32Array {
  const words = new Uint32Array(keys.length * (keyBits === 64 ? 2 : 1));
  for (const [index, key] of keys.entries()) {
    if (keyBits === 64) {
      words[2 * index] = Number(key & 0xffffffffn);
      words[2 * index + 1] = Number(key >> 32n);
    } else {
      words[index] = Number(key);
    }
  }
  return words;
}
