// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';

const f = Math.fround;

/** f32 twin of the DEV variance, old form: d2 / n - (d1 / n)^2 from exact integer sums. */
function getCentredVariance(d1: bigint, d2: bigint, n: number): number {
  const mean = f(f(Number(d1)) / n);
  return f(f(f(Number(d2)) / n) - f(mean * mean));
}

/** f32 twin of the re-centred form: integer shift s, exact (d - s) sums, residual mean r. */
function getShiftedVariance(d1: bigint, d2: bigint, n: number): number {
  const mean = f(f(Number(d1)) / n);
  const shift = BigInt(Math.round(mean));
  const shiftedSquare = d2 - 2n * shift * d1 + BigInt(n) * shift * shift;
  const residual = f(f(Number(d1 - BigInt(n) * shift)) / n);
  return f(f(f(Number(shiftedSquare)) / n) - f(residual * residual));
}

it('review C-16: the re-centred DEV variance does not cancel far from the window mean', () => {
  // R = 500 window (1001^2 cells), 1/256 m quanta: the centre is a 100 m peak above a rough plain
  // (relative heights -25600 quanta, spread about 1 m = 256 quanta), so mean^2 / variance is
  // about 1e4 and the old difference d2 / n - mean^2 cancels four decimal digits.
  const side = 1001;
  const n = side * side;
  let d1 = 0n;
  let d2 = 0n;
  let state = 12345;
  for (let cell = 0; cell < n; cell++) {
    state = (state * 1103515245 + 12345) % 2147483648;
    const d = BigInt(Math.round(-25600 + ((state / 2147483648) * 2 - 1) * 443));
    d1 += d;
    d2 += d * d;
  }
  const exact = Number(d2 * BigInt(n) - d1 * d1) / (n * n);
  const oldError = Math.abs(getCentredVariance(d1, d2, n) - exact) / exact;
  const newError = Math.abs(getShiftedVariance(d1, d2, n) - exact) / exact;
  // Old: each f32 rounding of d2 / n (about 6e-8 relative) is amplified by mean^2 / var = 1e4.
  // New: the shifted sum is about n * var, so a handful of f32 roundings, < 1e-6 relative.
  expect(oldError).toBeGreaterThan(1e-5);
  expect(newError).toBeLessThan(1e-6);
});
