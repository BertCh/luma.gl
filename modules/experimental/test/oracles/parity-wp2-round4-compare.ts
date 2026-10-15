// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Per-element analytic bounds; report native units and failing rows before assertion. */
export function compareBoundedParity(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: number[],
  expected: number[],
  bounds: number[]
) {
  let maxAbs = 0;
  let maxRel = 0;
  const failures: {index: number; actual: string; expected: number; bound: number}[] = [];
  for (let index = 0; index < expected.length; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    if (!Number.isFinite(difference) || difference > bounds[index]) {
      failures.push({
        index,
        actual: String(actual[index]),
        expected: expected[index],
        bound: bounds[index]
      });
    }
    if (Number.isFinite(difference)) {
      maxAbs = Math.max(maxAbs, difference);
      maxRel = Math.max(maxRel, difference / Math.max(Math.abs(expected[index]), 1e-30));
    }
  }
  const mismatches = failures.length + Math.abs(actual.length - expected.length);
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
      tolerance: {
        absolute: {minimum: Math.min(...bounds), maximum: Math.max(...bounds)},
        relative: 0,
        derivation: 'per-element analytic bound in generator',
        failures
      }
    })
  );
  expect(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}
