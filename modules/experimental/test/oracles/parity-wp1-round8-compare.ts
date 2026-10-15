// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export function compareReference(
  className: string,
  scene: {name: string; reference: string},
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  exact = false
): void {
  const scale = Math.max(1, ...expected.filter(Number.isFinite).map(Math.abs));
  const absoluteTolerance = exact ? 0 : 1e-6 * scale;
  const relativeTolerance = exact ? 0 : 1e-4;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const reference = expected[index];
    const observed = actual[index];
    if (!Number.isFinite(reference) || !Number.isFinite(observed)) {
      if (
        !(Object.is(reference, observed) || (Number.isNaN(reference) && Number.isNaN(observed)))
      ) {
        mismatches++;
      }
      continue;
    }
    const difference = Math.abs(reference - observed);
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(reference), 1e-30));
    if (difference > absoluteTolerance + relativeTolerance * Math.abs(reference)) mismatches++;
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: scene.name,
      reference: scene.reference,
      n: expected.length,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {absolute: absoluteTolerance, relative: relativeTolerance}
    })
  );
  expect(mismatches, metric).toBe(0);
}
