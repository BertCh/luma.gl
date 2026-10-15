// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export function compareParity(
  className: string,
  testCase: {name: string; reference: string; n: number},
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  profile = 'S',
  absoluteOverride?: number,
  relativeOverride?: number
): void {
  const scale = expected.reduce(
    (maximum, value) => (Number.isFinite(value) ? Math.max(maximum, Math.abs(value)) : maximum),
    1
  );
  const absolute = absoluteOverride ?? (profile === 'E' ? 0 : 1e-6 * scale);
  const relative = relativeOverride ?? (profile === 'E' ? 0 : 1e-4);
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  expected.forEach((value, index) => {
    const measured = actual[index];
    if (!Number.isFinite(value) || !Number.isFinite(measured)) {
      if (!Object.is(value, measured)) mismatches++;
      return;
    }
    const error = Math.abs(measured - value);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(value), Number.MIN_VALUE));
    if (error > absolute + relative * Math.abs(value)) mismatches++;
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.n,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {profile, absolute, relative}
    })
  );
  expect.soft(mismatches, `${className} ${testCase.name}: ${metric}`).toBe(0);
}
