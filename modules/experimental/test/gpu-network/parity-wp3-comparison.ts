// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export type Tolerance = {absolute: number; relative: number};
export const EXACT: Tolerance = {absolute: 0, relative: 0};

/** Compare all values and masks, printing measured errors even when an assertion will fail. */
export function compareParity(
  className: string,
  fixture: {case: string; reference: string},
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: Tolerance,
  caseName = fixture.case
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < Math.min(actual.length, expected.length); index++) {
    const observed = actual[index];
    const reference = expected[index];
    if (Object.is(observed, reference)) {
      continue;
    }
    if (!Number.isFinite(observed) || !Number.isFinite(reference)) {
      mismatches++;
      maxAbs = Infinity;
      maxRel = Infinity;
      continue;
    }
    const absolute = Math.abs(observed - reference);
    const relative = absolute / Math.max(Math.abs(reference), Number.MIN_VALUE);
    maxAbs = Math.max(maxAbs, absolute);
    maxRel = Math.max(maxRel, relative);
    mismatches += Number(absolute > tolerance.absolute + tolerance.relative * Math.abs(reference));
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference: fixture.reference,
      n: expected.length,
      metric,
      maxAbs: Number.isFinite(maxAbs) ? maxAbs : 'Infinity',
      maxRel: Number.isFinite(maxRel) ? maxRel : 'Infinity',
      mismatches,
      tolerance
    })
  );
  expect.soft(mismatches, `${caseName}: ${metric}`).toBe(0);
}
