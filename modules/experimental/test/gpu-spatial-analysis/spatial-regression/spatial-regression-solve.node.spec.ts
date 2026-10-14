// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  factorCholeskyOnCPU,
  getBelsleyConditionNumberOnCPU,
  getSymmetricEigenvalueRangeOnCPU,
  getSymmetricEigenvalueRangeWGSL
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve';

// Oracle: <scratchpad>/fix/work/C/coordinator/eig.py (numpy 2.5.4 eigvalsh, spreg 1.9.1 mulColli).
const MATRIX_EIGENVALUES = [1.8816802101698977, 2.3867603743778885, 4.731559415452213];
const DESIGN = [
  [1, 0.2, 3.1],
  [1, 1.5, 2.9],
  [1, 2.1, 4.4],
  [1, 3.7, 5.0],
  [1, 4.2, 7.3],
  [1, 5.9, 8.1]
];
const SPREG_CONDITION_INDEX = 17.048096533339308;

it('getSymmetricEigenvalueRangeOnCPU matches numpy extreme eigenvalues', () => {
  const matrix = [4, 1, 0.5, 1, 3, 0.25, 0.5, 0.25, 2];
  const [smallest, largest] = getSymmetricEigenvalueRangeOnCPU(matrix, 3);
  expect(smallest).toBeCloseTo(MATRIX_EIGENVALUES[0], 12);
  expect(largest).toBeCloseTo(MATRIX_EIGENVALUES[2], 12);
});

it('unit-diagonal Gram eigenvalues give the spreg condition index', () => {
  const p = 3;
  const gram = new Float64Array(p * p);
  for (const row of DESIGN) {
    for (let first = 0; first < p; first++) {
      for (let second = 0; second < p; second++) {
        gram[first * p + second] += row[first] * row[second];
      }
    }
  }
  const scale = Array.from({length: p}, (_, index) => Math.sqrt(gram[index * p + index]));
  for (let first = 0; first < p; first++) {
    for (let second = 0; second < p; second++) {
      gram[first * p + second] /= scale[first] * scale[second];
    }
  }
  const [smallest, largest] = getSymmetricEigenvalueRangeOnCPU(gram, p);
  expect(Math.sqrt(largest / smallest)).toBeCloseTo(SPREG_CONDITION_INDEX, 8);
});

it('getSymmetricEigenvalueRangeWGSL emits one suffixed function and validates p', () => {
  const wgsl = getSymmetricEigenvalueRangeWGSL(4);
  expect(wgsl).toContain('fn getSymmetricEigenvalueRange_4(a: ptr<function, array<f32, 16>>)');
  expect(() => getSymmetricEigenvalueRangeWGSL(17)).toThrow(/predictor count/);
});

it('getBelsleyConditionNumberOnCPU reproduces the spreg condition index from the factored design', () => {
  const n = DESIGN.length;
  const k = 2;
  const means = [0, 0];
  for (const row of DESIGN) {
    means[0] += row[1] / n;
    means[1] += row[2] / n;
  }
  const centered = new Float64Array(k * k);
  for (const row of DESIGN) {
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        centered[i * k + j] += (row[1 + i] - means[i]) * (row[1 + j] - means[j]);
      }
    }
  }
  const correlation = new Float32Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      correlation[i * k + j] =
        centered[i * k + j] / Math.sqrt(centered[i * k + i] * centered[j * k + j]);
    }
  }
  expect(factorCholeskyOnCPU(correlation, k)).toBe(true);
  const alpha = means.map(
    (mean, i) => (Math.sqrt(n) * mean) / Math.sqrt(n * mean * mean + centered[i * k + i])
  );
  const beta = means.map((mean, i) =>
    Math.sqrt(centered[i * k + i] / (n * mean * mean + centered[i * k + i]))
  );
  // The factor is float32-rounded, so agreement is to float32 precision times the condition number.
  expect(
    Math.abs(
      getBelsleyConditionNumberOnCPU(correlation, alpha, beta, k) / SPREG_CONDITION_INDEX - 1
    )
  ).toBeLessThan(1e-5);
});
