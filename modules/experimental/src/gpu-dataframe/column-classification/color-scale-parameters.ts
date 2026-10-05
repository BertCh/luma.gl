// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUColorScale} parameter view. */
export const GPU_COLOR_SCALE_PARAMETER_LENGTH = 9;

/** Class index written for a row that is masked, NaN, or outside an unclamped domain. */
export const GPU_COLOR_SCALE_NO_CLASS = 0xffffffff;

/**
 * Scale codes of the `scale` parameter. They follow d3 semantics.
 *
 * - `linear`, `sqrt`, `pow`, `log`, `symlog`: continuous scales over `[domain[0], domain[last]]`.
 * - `quantize`: equal intervals between `domain[0]` and `domain[last]`, one per palette entry.
 * - `threshold` and `quantile`: class by the shared edge convention (count of inner edges `<= v`).
 *   They share one code path because quantile breaks are just a threshold domain.
 * - `ordinal`: category code to palette entry; it is the only scale of a `uint32` values view.
 */
export const GPU_COLOR_SCALE_CODES = {
  linear: 0,
  sqrt: 1,
  pow: 2,
  log: 3,
  symlog: 4,
  quantize: 5,
  threshold: 6,
  quantile: 7,
  ordinal: 8
} as const;

/** Scale name accepted by {@link getGPUColorScaleParameterValues}. */
export type GPUColorScaleType = keyof typeof GPU_COLOR_SCALE_CODES;

/** Interpolation between palette entries of a continuous scale. */
export type GPUColorScaleInterpolation = 'step' | 'linear';

/** Options of {@link getGPUColorScaleParameterValues}. */
export type GPUColorScaleParameterOptions = {
  /** Scale type. */
  scale: GPUColorScaleType;
  /**
   * Number of active domain entries, `1..maximumDomainCount`. Ignored when the contributor was given a
   * `domainCount` view, which then supplies a CLASS count `k` and the domain has `k + 1` edges.
   */
  domainCount: number;
  /** Number of active palette entries, `1..maximumPaletteCount`. */
  paletteCount: number;
  /** `'step'` picks `palette[class]` exactly; `'linear'` blends adjacent entries per channel. Default `'step'`. */
  interpolation?: GPUColorScaleInterpolation;
  /** Clamp out-of-domain values to the first or last colour instead of no-data. Default `false`. */
  clamp?: boolean;
  /** Packed `rgba8` colour (`r | g << 8 | b << 16 | a << 24`) for no-data rows. Default `0`. */
  noDataColor?: number;
  /**
   * Value that replaces `v <= 0` (and non-positive domain entries) on the `log` scale, as kepler
   * does with a small positive floor. `NaN` treats such rows as no-data. Default `1e-5`.
   */
  logFloor?: number;
  /** Exponent of the `pow` scale. Default `1`. */
  exponent?: number;
};

/**
 * Packs per-frame {@link GPUColorScale} parameters.
 *
 * Layout (float32): `[scaleCode, domainCount, paletteCount, interpolation (0 step, 1 linear),
 * clamp (0 or 1), noDataColorLow16, noDataColorHigh16, logFloor, exponent]`. The no-data colour
 * is split into its low and high 16 bits, each exactly representable as an f32 integer, so no NaN
 * bit pattern is ever stored in a float slot.
 *
 * @param options Scale options.
 * @param target Optional destination of at least {@link GPU_COLOR_SCALE_PARAMETER_LENGTH} elements.
 * @throws If a count is not a non-negative integer or `target` is short.
 */
export function getGPUColorScaleParameterValues(
  options: GPUColorScaleParameterOptions,
  target: Float32Array = new Float32Array(GPU_COLOR_SCALE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_COLOR_SCALE_PARAMETER_LENGTH) {
    throw new Error(
      `Color scale parameter target must hold ${GPU_COLOR_SCALE_PARAMETER_LENGTH} elements`
    );
  }
  if (!(options.scale in GPU_COLOR_SCALE_CODES)) {
    throw new Error(`Unknown color scale '${String(options.scale)}'`);
  }
  for (const [name, count] of [
    ['domainCount', options.domainCount],
    ['paletteCount', options.paletteCount]
  ] as const) {
    if (!Number.isInteger(count) || count < 0) {
      throw new Error(`Color scale ${name} must be a non-negative integer`);
    }
  }
  const noDataColor = (options.noDataColor ?? 0) >>> 0;
  target[0] = GPU_COLOR_SCALE_CODES[options.scale];
  target[1] = options.domainCount;
  target[2] = options.paletteCount;
  target[3] = options.interpolation === 'linear' ? 1 : 0;
  target[4] = options.clamp ? 1 : 0;
  target[5] = noDataColor & 0xffff;
  target[6] = noDataColor >>> 16;
  target[7] = options.logFloor ?? 1e-5;
  target[8] = options.exponent ?? 1;
  return target;
}

/** Packs `r`, `g`, `b`, `a` bytes into the shared `rgba8` `u32` layout. */
export function packGPUColor(r: number, g: number, b: number, a: number = 255): number {
  return ((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)) >>> 0;
}
