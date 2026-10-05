// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Float32 widths whose multiples are not exact in f32. */
export const ADVERSARIAL_WIDTHS = [0.1, 0.3, 1 / 3].map(Math.fround);

/** Buckets covered by the adversarial scene. */
export const ADVERSARIAL_BUCKET_COUNT = 301;

/** The adjacent f32 above (`step = 1`) or below (`step = -1`) a positive finite f32. */
export function getAdjacentFloat32(value: number, step: 1 | -1): number {
  const floats = new Float32Array([value]);
  const bits = new Uint32Array(floats.buffer);
  bits[0] += step;
  return floats[0];
}

/**
 * One row per (edge k, neighbor): the time `fround(k * width)` and its two f32 neighbors, with
 * `values` equal to the row index. Each row gets its own cell so slots identify rows.
 */
export function createAdversarialEdgeScene(width: number) {
  const times: number[] = [];
  const edgeBuckets: (number | undefined)[] = [];
  for (let edge = 1; edge < ADVERSARIAL_BUCKET_COUNT; edge++) {
    const edgeTime = Math.fround(edge * width);
    times.push(getAdjacentFloat32(edgeTime, -1), edgeTime, getAdjacentFloat32(edgeTime, 1));
    // The edge time itself is the first time of bucket `edge` by definition.
    edgeBuckets.push(undefined, edge, undefined);
  }
  const rows = times.length;
  return {
    cellIds: Uint32Array.from({length: rows}, (_, row) => row),
    timestamps: Float32Array.from(times),
    values: Float32Array.from({length: rows}, (_, row) => row),
    cellCount: rows,
    bucketCount: ADVERSARIAL_BUCKET_COUNT,
    edgeBuckets
  };
}
