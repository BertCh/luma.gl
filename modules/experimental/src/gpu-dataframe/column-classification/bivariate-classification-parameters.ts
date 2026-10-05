// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUBivariateClassification} parameter view. */
export const GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH = 8;

/** Class id written for a row that is masked or has a NaN value. */
export const GPU_BIVARIATE_CLASSIFICATION_NO_CLASS = 0xffffffff;

/** Value-by-alpha options of {@link getGPUBivariateClassificationParameterValues}. */
export type GPUBivariateValueByAlpha = {
  /** `[lo, hi]` range of `alphaValues` that maps to the alpha ramp. */
  domain: readonly [number, number];
  /** Alpha factor in `[0, 1]` at `alphaValues <= lo` (and for NaN alpha values). */
  minimumAlpha: number;
};

/** Options of {@link getGPUBivariateClassificationParameterValues}. */
export type GPUBivariateClassificationParameterOptions = {
  /** Active classes along X, `1..maximumClassCount`; `breaksX` holds `classCountX + 1` edges. */
  classCountX: number;
  /** Active classes along Y, `1..maximumClassCount`; `breaksY` holds `classCountY + 1` edges. */
  classCountY: number;
  /** Packed `rgba8` colour (`r | g << 8 | b << 16 | a << 24`) for no-data rows. Default `0`. */
  noDataColor?: number;
  /** Enables value-by-alpha. Omit to leave the palette alpha unchanged. */
  valueByAlpha?: GPUBivariateValueByAlpha;
};

/**
 * Packs per-frame {@link GPUBivariateClassification} parameters.
 *
 * Layout (float32): `[classCountX, classCountY, noDataColorLow16, noDataColorHigh16,
 * alphaEnabled (0 or 1), alphaLow, alphaHigh, minimumAlpha]`. The no-data colour is split into
 * 16-bit halves that are exact f32 integers.
 *
 * @param options Per-frame options.
 * @param target Optional destination of at least {@link GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH} elements.
 * @throws If a class count is not a non-negative integer or `target` is short.
 */
export function getGPUBivariateClassificationParameterValues(
  options: GPUBivariateClassificationParameterOptions,
  target: Float32Array = new Float32Array(GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH) {
    throw new Error(
      `Bivariate classification parameter target must hold ${GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH} elements`
    );
  }
  for (const [name, count] of [
    ['classCountX', options.classCountX],
    ['classCountY', options.classCountY]
  ] as const) {
    if (!Number.isInteger(count) || count < 0) {
      throw new Error(`Bivariate classification ${name} must be a non-negative integer`);
    }
  }
  const noDataColor = (options.noDataColor ?? 0) >>> 0;
  const alpha = options.valueByAlpha;
  target[0] = options.classCountX;
  target[1] = options.classCountY;
  target[2] = noDataColor & 0xffff;
  target[3] = noDataColor >>> 16;
  target[4] = alpha ? 1 : 0;
  target[5] = alpha ? alpha.domain[0] : 0;
  target[6] = alpha ? alpha.domain[1] : 1;
  target[7] = alpha ? alpha.minimumAlpha : 1;
  return target;
}
