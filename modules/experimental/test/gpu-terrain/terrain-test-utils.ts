// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect} from 'vitest';

/** Declares a transient float32 buffer-backed raster band on `graph`. */
export function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: createTransientView(graph, id, 'float32', length)
    }
  };
}

/** Raw float32 bit patterns of `values`, with every NaN canonicalised, for exact comparisons. */
export function toBits(values: Float32Array): number[] {
  const bits = new Uint32Array(values.length);
  const floats = new Float32Array(bits.buffer);
  for (const [index, value] of values.entries()) {
    floats[index] = Number.isNaN(value) ? NaN : value;
  }
  return Array.from(bits);
}

/** Asserts `actual` equals `expected` within `tolerance`, with identical NaN patterns. Returns the worst error. */
export function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label = ''
): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(Number.isNaN(actual[index]), `${label} index ${index} should be NaN`).toBe(true);
      continue;
    }
    expect(Number.isNaN(actual[index]), `${label} index ${index} expected ${expected[index]}`).toBe(
      false
    );
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  expect(worst, `${label} worst error`).toBeLessThan(tolerance);
  return worst;
}

/** Deterministic integer noise in `[0, levels)` so equal heights (ties) are common. */
export function createNoise(
  width: number,
  height: number,
  levels: number,
  seed: number
): Float32Array {
  let state = seed >>> 0;
  return Float32Array.from({length: width * height}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % levels;
  });
}

export type GaussianHill = {x: number; y: number; amplitude: number; sigma: number};

/** Sum of isotropic Gaussian hills on a `width` by `height` grid. */
export function createHills(
  width: number,
  height: number,
  hills: readonly GaussianHill[]
): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return hills.reduce(
      (sum, hill) =>
        sum +
        hill.amplitude *
          Math.exp(-((column - hill.x) ** 2 + (row - hill.y) ** 2) / (2 * hill.sigma ** 2)),
      0
    );
  });
}
