// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUTextureShadingCascadeSigmas,
  getGPUTextureShadingKernel
} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';

/**
 * Float64 texture shading by normalized convolution: blur `value * valid` and `valid` with the
 * cascade kernels (edge-replicating borders), divide, and sum weighted band differences.
 */
export function computeTextureShading(options: {
  width: number;
  height: number;
  elevation: ArrayLike<number>;
  levelCount: number;
  baseSigma: number;
  gain: number;
  weights: readonly number[];
}): number[] {
  const {width, height, elevation, levelCount, baseSigma} = options;
  const pixelCount = width * height;
  let numerator = Array.from({length: pixelCount}, (_, pixel) =>
    Number.isFinite(elevation[pixel]) ? elevation[pixel] : 0
  );
  let denominator: number[] = Array.from({length: pixelCount}, (_, pixel) =>
    Number.isFinite(elevation[pixel]) ? 1 : 0
  );
  const sum = new Array(pixelCount).fill(0);
  const blurred = (n: number[], d: number[], pixel: number) =>
    d[pixel] > 0 ? n[pixel] / d[pixel] : 0;
  const sigmas = getGPUTextureShadingCascadeSigmas(levelCount, baseSigma);
  for (let level = 0; level < levelCount; level++) {
    const kernel = getGPUTextureShadingKernel(sigmas[level]);
    const radius = (kernel.length - 1) / 2;
    const convolve = (values: number[], horizontal: boolean) =>
      values.map((_, pixel) => {
        const column = pixel % width;
        const row = Math.floor(pixel / width);
        let total = 0;
        for (let offset = -radius; offset <= radius; offset++) {
          const sample = horizontal
            ? row * width + Math.min(Math.max(column + offset, 0), width - 1)
            : Math.min(Math.max(row + offset, 0), height - 1) * width + column;
          total += kernel[offset + radius] * values[sample];
        }
        return total;
      });
    const nextNumerator = convolve(convolve(numerator, true), false);
    const nextDenominator = convolve(convolve(denominator, true), false);
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      sum[pixel] +=
        options.weights[level] *
        (blurred(numerator, denominator, pixel) - blurred(nextNumerator, nextDenominator, pixel));
    }
    numerator = nextNumerator;
    denominator = nextDenominator;
  }
  return sum.map((value, pixel) =>
    Number.isFinite(elevation[pixel]) ? options.gain * value : NaN
  );
}
