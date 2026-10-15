// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** S profile, per-element output scale; exact integer counts and non-finite masks. */
export function compareParity(
  className: string,
  caseName: string,
  reference: string,
  rowCount: number,
  metric: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  exact = false
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < Math.min(actual.length, expected.length); index++) {
    const observed = actual[index];
    const target = expected[index];
    if (!Number.isFinite(observed) || !Number.isFinite(target)) {
      if (!Object.is(observed, target)) {
        mismatches++;
      }
      continue;
    }
    const absoluteError = Math.abs(observed - target);
    const relativeError =
      target === 0 ? (absoluteError === 0 ? 0 : Infinity) : absoluteError / Math.abs(target);
    maxAbs = Math.max(maxAbs, absoluteError);
    maxRel = Math.max(maxRel, relativeError);
    const bound = exact ? 0 : 1e-6 * Math.max(1, Math.abs(target)) + 1e-4 * Math.abs(target);
    if (absoluteError > bound) {
      mismatches++;
    }
  }
  const tolerance = exact
    ? {profile: 'E', absolute: 0, relative: 0}
    : {profile: 'S', absolute: '1e-6 * max(1, abs(reference element))', relative: 1e-4};
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference,
      n: rowCount,
      metric,
      maxAbs,
      maxRel: Number.isFinite(maxRel) ? maxRel : 'Infinity',
      mismatches,
      tolerance
    })
  );
  // Soft assertions keep the remaining full-array measurements visible on a failing run.
  expect.soft(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}
