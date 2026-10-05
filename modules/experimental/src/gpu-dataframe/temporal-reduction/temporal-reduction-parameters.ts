// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {splitTimeWords} from '../time-window-filter/time-words';

/** Number of float32 elements in a relative-time bucketing parameter view. */
export const GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH = 2;

/** Number of uint32 elements in an Int64 word-time bucketing parameter view. */
export const GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH = 4;

/** `cellIds` value that makes a row skip the reduction. */
export const GPU_TEMPORAL_REDUCTION_NO_CELL = 0xffffffff;

/**
 * Packs relative-time bucketing parameters for f32 timestamps.
 *
 * Layout: `[origin, width]` as float32. Bucket `b` covers `[origin + b * width, origin + (b + 1) *
 * width)`. A `width` that is not finite and positive makes every row skip the reduction.
 *
 * @param origin Start of bucket 0, in the unit of the timestamps.
 * @param width Bucket width, in the unit of the timestamps.
 * @param target Optional destination of at least 2 elements.
 */
export function getGPUTemporalReductionParameterValues(
  origin: number,
  width: number,
  target: Float32Array = new Float32Array(GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH) {
    throw new Error(
      `Temporal reduction parameter target must hold ${GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH} elements`
    );
  }
  target[0] = origin;
  target[1] = width;
  return target;
}

/**
 * Packs bucketing parameters for exact Int64 word timestamps.
 *
 * Layout: `[originLow, originHigh, width, 0]` as uint32. Both values are integers in the unit of
 * the timestamps (for example epoch milliseconds); the width is at most `2^32 - 1` and bucket
 * assignment is exact integer division.
 *
 * @param origin Integer start of bucket 0, any signed 64-bit value.
 * @param width Integer bucket width in `[1, 2^32 - 1]`.
 * @param target Optional destination of at least 4 elements.
 * @throws If `origin` is not an integer, `width` is not an integer in range, or `target` is short.
 */
export function getGPUTemporalReductionWordParameterValues(
  origin: number | bigint,
  width: number | bigint,
  target: Uint32Array = new Uint32Array(GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH) {
    throw new Error(
      `Temporal reduction word parameter target must hold ${GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH} elements`
    );
  }
  if (typeof origin === 'number' && !Number.isInteger(origin)) {
    throw new Error('Temporal reduction word origin must be an integer');
  }
  const widthValue =
    typeof width === 'bigint' ? width : Number.isInteger(width) ? BigInt(width) : -1n;
  if (widthValue < 1n || widthValue > 0xffffffffn) {
    throw new Error('Temporal reduction word width must be an integer in [1, 2^32 - 1]');
  }
  const words = splitTimeWords(origin);
  target[0] = words.low;
  target[1] = words.high;
  target[2] = Number(widthValue);
  target[3] = 0;
  return target;
}
