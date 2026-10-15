// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** R per-cell budget; exact special-value masks. A supplied G bound is derived by the caller. */
export function compareRoundFour(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  profile: 'R' | 'G' | 'E',
  absoluteBound?: number
) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of Array.from(expected)) {
    if (Number.isFinite(value)) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  const absolute = absoluteBound ?? (profile === 'R' ? 1e-5 * Math.max(1, maximum - minimum) : 0);
  const relative = profile === 'R' ? 1e-4 : 0;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(expected.length - actual.length);
  const firstMismatches: {index: number; actual: number; expected: number}[] = [];
  for (let index = 0; index < expected.length; index++) {
    const wanted = expected[index];
    const received = actual[index];
    let mismatch: boolean;
    if (!Number.isFinite(wanted) || !Number.isFinite(received)) {
      mismatch = !(Object.is(wanted, received) || (Number.isNaN(wanted) && Number.isNaN(received)));
    } else {
      const difference = Math.abs(received - wanted);
      maxAbs = Math.max(maxAbs, difference);
      maxRel = Math.max(maxRel, difference / Math.max(Math.abs(wanted), 1e-300));
      mismatch = difference > absolute + relative * Math.abs(wanted);
    }
    if (mismatch) {
      mismatches++;
      if (firstMismatches.length < 8)
        firstMismatches.push({index, actual: received, expected: wanted});
    }
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
      tolerance: {profile, absolute, relative}
    })
  );
  expect(mismatches, `${className}/${caseName}/${metric}: ${JSON.stringify(firstMismatches)}`).toBe(
    0
  );
}
