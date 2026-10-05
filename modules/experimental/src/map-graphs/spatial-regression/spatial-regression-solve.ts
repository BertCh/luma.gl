// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest predictor count (including the intercept) the small dense solvers support. */
export const SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT = 16;

/**
 * Returns WGSL for an in-place Cholesky factorization and solve of a symmetric positive definite
 * `p x p` system held in function-local arrays, with `p` fixed at compile time.
 *
 * Generated functions (suffix `_${p}`):
 * - `choleskyFactor_p(a: ptr<function, array<f32, p*p>>) -> bool` overwrites the lower triangle
 *   (row-major `a[row * p + column]`) with `L` such that `A = L L^T`. Returns false when a pivot is
 *   not strictly positive or not finite (singular, indefinite, or ill-conditioned system).
 * - `choleskySolve_p(a, b: ptr<function, array<f32, p>>)` solves `L L^T x = b` in place in `b`
 *   using a factor from `choleskyFactor_p`.
 * - `choleskyInverseDiagonal_p(a, column: u32) -> f32` returns `(A^-1)[column][column]`, used for
 *   coefficient standard errors and hat-matrix terms.
 *
 * The loop order is fixed, so results are deterministic for a given adapter. The CPU mirror
 * {@link factorCholeskyOnCPU} uses the same operation order with `Math.fround` rounding.
 */
export function getCholeskyWGSL(predictorCount: number): string {
  validatePredictorCount(predictorCount);
  const p = predictorCount;
  const size = p * p;
  return /* wgsl */ `
fn choleskyFactor_${p}(a: ptr<function, array<f32, ${size}>>) -> bool {
  for (var column = 0u; column < ${p}u; column++) {
    var diagonal = (*a)[column * ${p}u + column];
    for (var k = 0u; k < column; k++) {
      let value = (*a)[column * ${p}u + k];
      diagonal = diagonal - value * value;
    }
    if (!(diagonal > 0.0) || diagonal > 3.0e38) {
      return false;
    }
    let pivot = sqrt(diagonal);
    (*a)[column * ${p}u + column] = pivot;
    for (var row = column + 1u; row < ${p}u; row++) {
      var value = (*a)[row * ${p}u + column];
      for (var k = 0u; k < column; k++) {
        value = value - (*a)[row * ${p}u + k] * (*a)[column * ${p}u + k];
      }
      (*a)[row * ${p}u + column] = value / pivot;
    }
  }
  return true;
}

fn choleskySolve_${p}(a: ptr<function, array<f32, ${size}>>, b: ptr<function, array<f32, ${p}>>) {
  for (var row = 0u; row < ${p}u; row++) {
    var value = (*b)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*a)[row * ${p}u + k] * (*b)[k];
    }
    (*b)[row] = value / (*a)[row * ${p}u + row];
  }
  for (var offset = 0u; offset < ${p}u; offset++) {
    let row = ${p - 1}u - offset;
    var value = (*b)[row];
    for (var k = row + 1u; k < ${p}u; k++) {
      value = value - (*a)[k * ${p}u + row] * (*b)[k];
    }
    (*b)[row] = value / (*a)[row * ${p}u + row];
  }
}

fn choleskyInverseDiagonal_${p}(a: ptr<function, array<f32, ${size}>>, column: u32) -> f32 {
  var unit: array<f32, ${p}>;
  for (var row = 0u; row < ${p}u; row++) {
    unit[row] = select(0.0, 1.0, row == column);
  }
  choleskySolve_${p}(a, &unit);
  return unit[column];
}
`;
}

/**
 * CPU mirror of `choleskyFactor_p`: factors row-major `matrix` (length `p * p`) in place with f32
 * rounding after every operation. Returns false for a non-positive or non-finite pivot.
 */
export function factorCholeskyOnCPU(matrix: Float32Array, predictorCount: number): boolean {
  const p = predictorCount;
  const round = Math.fround;
  for (let column = 0; column < p; column++) {
    let diagonal = matrix[column * p + column];
    for (let k = 0; k < column; k++) {
      const value = matrix[column * p + k];
      diagonal = round(diagonal - round(value * value));
    }
    if (!(diagonal > 0) || diagonal > 3.0e38) {
      return false;
    }
    const pivot = round(Math.sqrt(diagonal));
    matrix[column * p + column] = pivot;
    for (let row = column + 1; row < p; row++) {
      let value = matrix[row * p + column];
      for (let k = 0; k < column; k++) {
        value = round(value - round(matrix[row * p + k] * matrix[column * p + k]));
      }
      matrix[row * p + column] = round(value / pivot);
    }
  }
  return true;
}

/** CPU mirror of `choleskySolve_p`: solves in place in `rightHandSide` with f32 rounding. */
export function solveCholeskyOnCPU(
  factor: Float32Array,
  rightHandSide: Float32Array,
  predictorCount: number
): void {
  const p = predictorCount;
  const round = Math.fround;
  for (let row = 0; row < p; row++) {
    let value = rightHandSide[row];
    for (let k = 0; k < row; k++) {
      value = round(value - round(factor[row * p + k] * rightHandSide[k]));
    }
    rightHandSide[row] = round(value / factor[row * p + row]);
  }
  for (let row = p - 1; row >= 0; row--) {
    let value = rightHandSide[row];
    for (let k = row + 1; k < p; k++) {
      value = round(value - round(factor[k * p + row] * rightHandSide[k]));
    }
    rightHandSide[row] = round(value / factor[row * p + row]);
  }
}

function validatePredictorCount(predictorCount: number): void {
  if (
    !Number.isInteger(predictorCount) ||
    predictorCount < 1 ||
    predictorCount > SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
  ) {
    throw new Error(
      `Spatial regression predictor count must be an integer in [1, ${SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT}]`
    );
  }
}
