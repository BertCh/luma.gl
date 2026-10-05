// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleFieldOnCPU, type FieldRaster} from './flow-texture-field';
import {getPhilox4x32, getPhiloxUnitFloat} from './flow-texture-random';

/** Philox key word 1 of the LIC white-noise stream. */
export const LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE = 1;

/** Result of {@link convolveLineIntegralOnCPU}. */
export type LineIntegralConvolutionCPUResult = {
  /** Row-major LIC values in `[0, 1]`, NaN where the pixel centre has no field data. */
  values: Float32Array;
  /** Field speed at each pixel centre, NaN where it has no field data. */
  speeds: Float32Array;
  /** Row-major white noise in `[0, 1)`. */
  noise: Float32Array;
};

/** White noise of output pixel `(column, row)`: Philox `((column, row, 0, 0), (seed, 1)).x`. */
export function getLineIntegralConvolutionNoise(column: number, row: number, seed: number): number {
  return getPhiloxUnitFloat(
    getPhilox4x32([column, row, 0, 0], [seed, LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE])[0]
  );
}

/** Kernel weight of signed step `step`: a Hann window times the optional animated ripple. */
export function getLineIntegralConvolutionWeight(
  step: number,
  stepCount: number,
  phase: number,
  period: number
): number {
  const window = 0.5 * (1 + Math.cos((Math.PI * step) / (stepCount + 1)));
  const ripple = period > 0 ? 0.5 * (1 + Math.cos(2 * Math.PI * (step / period - phase))) : 1;
  return window * ripple;
}

/**
 * CPU oracle of `GPULineIntegralConvolution`.
 *
 * Mirrors the WGSL control flow exactly; values differ from the GPU by transcendental and
 * normalisation rounding, which can occasionally move a streamline sample into a neighbouring
 * pixel near a pixel edge.
 *
 * @param field Packed vector field.
 * @param width Output width in pixels.
 * @param height Output height in pixels.
 * @param stepCount Steps per direction `L`.
 * @param parameters Float parameters from `getGPULineIntegralConvolutionParameterValues`.
 * @param seed Noise seed.
 */
export function convolveLineIntegralOnCPU(
  field: FieldRaster,
  width: number,
  height: number,
  stepCount: number,
  parameters: ArrayLike<number>,
  seed: number
): LineIntegralConvolutionCPUResult {
  const fround = Math.fround;
  const fieldExtent = [parameters[0], parameters[1], parameters[2], parameters[3]].map(fround);
  const originX = fround(parameters[4]);
  const originY = fround(parameters[5]);
  const cellWidth = fround(parameters[6]);
  const cellHeight = fround(parameters[7]);
  const stepX = fround(fround(parameters[8]) * cellWidth);
  const stepY = fround(fround(parameters[8]) * cellHeight);
  const minimumSpeed = fround(parameters[9]);
  const phase = fround(parameters[10]);
  const period = fround(parameters[11]);
  const noise = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      noise[row * width + column] = getLineIntegralConvolutionNoise(column, row, seed);
    }
  }
  const values = new Float32Array(width * height);
  const speeds = new Float32Array(width * height);
  const getDirection = (x: number, y: number, sign: number): [number, number] | undefined => {
    const sample = sampleFieldOnCPU(field, x, y, fieldExtent);
    if (!sample) {
      return undefined;
    }
    const speed = Math.hypot(sample[0], sample[1]);
    if (!(speed > 0) || speed < minimumSpeed) {
      return undefined;
    }
    return [(sign * sample[0]) / speed, (sign * sample[1]) / speed];
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const pixel = row * width + column;
      const centreX = fround(originX + fround((column + 0.5) * cellWidth));
      const centreY = fround(originY + fround((row + 0.5) * cellHeight));
      const centre = sampleFieldOnCPU(field, centreX, centreY, fieldExtent);
      if (!centre) {
        values[pixel] = NaN;
        speeds[pixel] = NaN;
        continue;
      }
      speeds[pixel] = Math.hypot(centre[0], centre[1]);
      let weightSum = getLineIntegralConvolutionWeight(0, stepCount, phase, period);
      let sum = weightSum * noise[pixel];
      for (const sign of [1, -1]) {
        let x = centreX;
        let y = centreY;
        for (let step = 1; step <= stepCount; step++) {
          const first = getDirection(x, y, sign);
          if (!first) {
            break;
          }
          const middle = getDirection(
            fround(x + 0.5 * stepX * first[0]),
            fround(y + 0.5 * stepY * first[1]),
            sign
          );
          if (!middle) {
            break;
          }
          x = fround(x + stepX * middle[0]);
          y = fround(y + stepY * middle[1]);
          const sampleColumn = Math.floor((x - originX) / cellWidth);
          const sampleRow = Math.floor((y - originY) / cellHeight);
          if (sampleColumn < 0 || sampleRow < 0 || sampleColumn >= width || sampleRow >= height) {
            break;
          }
          const weight = getLineIntegralConvolutionWeight(sign * step, stepCount, phase, period);
          sum += weight * noise[sampleRow * width + sampleColumn];
          weightSum += weight;
        }
      }
      values[pixel] = weightSum > 0 ? sum / weightSum : noise[pixel];
    }
  }
  return {values, speeds, noise};
}
