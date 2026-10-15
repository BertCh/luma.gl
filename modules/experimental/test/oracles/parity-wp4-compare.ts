// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Full-array measurement; NaN/Infinity masks and lengths are always exact. */
export function compareRasterParity(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  profile: 'L' | 'S' | 'E'
): void {
  let scale = 1;
  for (const value of Array.from(expected)) {
    if (Number.isFinite(value)) scale = Math.max(scale, Math.abs(value));
  }
  const absolute = profile === 'E' ? 0 : 1e-6 * scale;
  const relative = profile === 'E' ? 0 : 1e-4;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const wanted = expected[index];
    const received = actual[index];
    if (!Number.isFinite(wanted) || !Number.isFinite(received)) {
      if (!(Object.is(received, wanted) || (Number.isNaN(received) && Number.isNaN(wanted)))) {
        mismatches++;
      }
      continue;
    }
    const error = Math.abs(received - wanted);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(wanted), 1e-300));
    if (error > absolute + relative * Math.abs(wanted)) mismatches++;
  }
  const scaledResidual = maxAbs / scale;
  const tolerance = {profile, absolute, relative, scaledResidual: profile === 'L' ? 1e-4 : 0};
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
      tolerance
    })
  );
  expect(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
  if (profile === 'L')
    expect(scaledResidual, `${metric} scaled residual`).toBeLessThanOrEqual(1e-4);
}
