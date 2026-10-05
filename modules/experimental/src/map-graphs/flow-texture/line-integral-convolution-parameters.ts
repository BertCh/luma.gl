// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPULineIntegralConvolution} parameter view. */
export const GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH = 12;

/** Number of uint32 elements in a {@link GPULineIntegralConvolution} word parameter view. */
export const GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPULineIntegralConvolution}. */
export type GPULineIntegralConvolutionSettings = {
  /** Field placement `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. */
  fieldExtent: readonly [number, number, number, number];
  /** Output raster placement `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. */
  outputExtent: readonly [number, number, number, number];
  /** Streamline step in output pixels. Defaults to 0.5. */
  stepLength?: number;
  /** A streamline stops where the field is slower than this. Defaults to 0. */
  minimumSpeed?: number;
  /** Animation phase in kernel periods; advance it every frame to make the texture flow. Defaults to 0. */
  phase?: number;
  /** Ripple period in steps. `0` disables the animated ripple (static LIC). Defaults to 0. */
  period?: number;
  /** Seed of the white noise. Defaults to 0. */
  seed?: number;
};

/**
 * Packs float parameters of {@link GPULineIntegralConvolution}.
 *
 * Layout: `[fieldOriginX, fieldOriginY, fieldCellWidth, fieldCellHeight, outputOriginX,
 * outputOriginY, outputCellWidth, outputCellHeight, stepLength, minimumSpeed, phase, period]`.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 12 elements.
 */
export function getGPULineIntegralConvolutionParameterValues(
  settings: GPULineIntegralConvolutionSettings,
  target: Float32Array = new Float32Array(GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH) {
    throw new Error(
      `Line integral convolution parameter target must hold ${GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH} elements`
    );
  }
  target.set([
    ...settings.fieldExtent,
    ...settings.outputExtent,
    settings.stepLength ?? 0.5,
    settings.minimumSpeed ?? 0,
    settings.phase ?? 0,
    settings.period ?? 0
  ]);
  return target;
}

/**
 * Packs integer parameters of {@link GPULineIntegralConvolution}. Layout: `[seed, 0, 0, 0]`.
 *
 * @param settings Per-frame settings; only `seed` is read.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPULineIntegralConvolutionWordParameterValues(
  settings: Pick<GPULineIntegralConvolutionSettings, 'seed'>,
  target: Uint32Array = new Uint32Array(GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH) {
    throw new Error(
      `Line integral convolution word parameter target must hold ${GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH} elements`
    );
  }
  const seed = settings.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('Line integral convolution seed must be a uint32 integer');
  }
  target.set([seed, 0, 0, 0]);
  return target;
}
