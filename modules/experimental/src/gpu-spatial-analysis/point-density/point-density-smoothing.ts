// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Returns a normalized square Gaussian kernel of `(2 * radius + 1)^2` row-major weights that sum
 * to one, for `GPUPointDensitySmoothing.kernel`.
 *
 * @param radius Integer kernel radius in cells, from 0 to 32. Radius 0 returns `[1]`.
 * @param sigma Standard deviation in cells. Defaults to `max(radius / 2, 0.5)`.
 */
export function createGPUPointDensityGaussianKernel(radius: number, sigma?: number): Float32Array {
  if (!Number.isInteger(radius) || radius < 0 || radius > 32) {
    throw new Error('Gaussian kernel radius must be an integer in [0, 32]');
  }
  const resolvedSigma = sigma ?? Math.max(radius / 2, 0.5);
  if (!Number.isFinite(resolvedSigma) || resolvedSigma <= 0) {
    throw new Error('Gaussian kernel sigma must be positive and finite');
  }
  const size = 2 * radius + 1;
  const weights = new Float64Array(size * size);
  let total = 0;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const dx = column - radius;
      const dy = row - radius;
      const weight = Math.exp(-(dx * dx + dy * dy) / (2 * resolvedSigma * resolvedSigma));
      weights[row * size + column] = weight;
      total += weight;
    }
  }
  return Float32Array.from(weights, weight => weight / total);
}

/**
 * Returns a normalized 1D Gaussian kernel of `2 * radius + 1` weights that sum to one, for
 * `GPUPointDensitySmoothing.separableKernel` (use it for both `horizontal` and `vertical`). The
 * outer product of two such kernels equals `createGPUPointDensityGaussianKernel(radius, sigma)`.
 *
 * @param radius Integer kernel radius in cells, from 0 to 32.
 * @param sigma Standard deviation in cells. Defaults to `max(radius / 2, 0.5)`.
 */
export function createGPUPointDensityGaussianKernel1D(
  radius: number,
  sigma?: number
): Float32Array {
  if (!Number.isInteger(radius) || radius < 0 || radius > 32) {
    throw new Error('Gaussian kernel radius must be an integer in [0, 32]');
  }
  const resolvedSigma = sigma ?? Math.max(radius / 2, 0.5);
  if (!Number.isFinite(resolvedSigma) || resolvedSigma <= 0) {
    throw new Error('Gaussian kernel sigma must be positive and finite');
  }
  const weights = new Float64Array(2 * radius + 1);
  let total = 0;
  for (let index = 0; index < weights.length; index++) {
    const offset = index - radius;
    weights[index] = Math.exp(-(offset * offset) / (2 * resolvedSigma * resolvedSigma));
    total += weights[index];
  }
  return Float32Array.from(weights, weight => weight / total);
}
