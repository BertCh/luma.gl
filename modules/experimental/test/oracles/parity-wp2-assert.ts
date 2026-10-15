// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Log every comparison before asserting, including lengths and non-finite masks. */
export function compareParity(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  absolute = 0,
  relative = 0
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < Math.min(actual.length, expected.length); index++) {
    const measured = actual[index];
    const wanted = expected[index];
    if (!Number.isFinite(measured) || !Number.isFinite(wanted)) {
      mismatches += Number(!Object.is(measured, wanted));
      continue;
    }
    const difference = Math.abs(measured - wanted);
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(wanted), 1e-30));
    mismatches += Number(difference > absolute + relative * Math.abs(wanted));
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference,
      n: expected.length,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {absolute, relative}
    })
  );
  expect(mismatches, `${className}: ${caseName}: ${metric}`).toBe(0);
}
