// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a `GPUInequality` parameter view. */
export const GPU_INEQUALITY_PARAMETER_LENGTH = 4;

/** Default Palma numerator: the richest 10% of the population. */
export const GPU_INEQUALITY_DEFAULT_PALMA_TOP_SHARE = 0.1;

/** Default Palma denominator: the poorest 40% of the population. */
export const GPU_INEQUALITY_DEFAULT_PALMA_BOTTOM_SHARE = 0.4;

/** Largest Atkinson epsilon accepted; larger values overflow `(x / mean) ^ (1 - epsilon)` in f32. */
export const GPU_INEQUALITY_MAXIMUM_EPSILON = 32;

/** Float32 slots of the `globalSummary` output of `GPUInequality`. */
export const GPU_INEQUALITY_GLOBAL_SUMMARY = {
  /** Theil T of all included rows pooled, computed directly from the row values. */
  TOTAL_THEIL_T: 0,
  /** Between-zone term `sum_g s_g ln(s_g / p_g)` of the Theil T decomposition. */
  BETWEEN_THEIL_T: 1,
  /** Within-zone term `sum_g s_g T_g` of the Theil T decomposition. */
  WITHIN_THEIL_T: 2,
  /** Gini coefficient of all included rows pooled. */
  GINI: 3,
  /** Number of included rows, as a float. */
  COUNT: 4,
  /** Weighted mean of all included rows. */
  MEAN: 5,
  /** Total weight (the row count when there are no weights). */
  TOTAL_WEIGHT: 6,
  /** Total weighted income `sum w x`. */
  TOTAL_INCOME: 7
} as const;

/** Number of float32 elements in the `globalSummary` output. */
export const GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH = 8;

/** CPU description of the per-frame parameters of `GPUInequality`. */
export type GPUInequalitySettings = {
  /**
   * Atkinson inequality aversion, `0` to {@link GPU_INEQUALITY_MAXIMUM_EPSILON}. `1` selects the
   * geometric-mean form. Defaults to 1.
   */
  epsilon?: number;
  /** Population share of the Palma numerator, in `(0, 1]`. Defaults to 0.1. */
  palmaTopShare?: number;
  /** Population share of the Palma denominator, in `(0, 1]`. Defaults to 0.4. */
  palmaBottomShare?: number;
};

/**
 * Packs `settings` into the float32 parameter layout `[epsilon, palmaTopShare, palmaBottomShare, 0]`.
 *
 * @param target Optional destination, at least {@link GPU_INEQUALITY_PARAMETER_LENGTH} long.
 * @throws If a value is outside its range.
 */
export function getGPUInequalityParameterValues(
  settings: GPUInequalitySettings = {},
  target: Float32Array = new Float32Array(GPU_INEQUALITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_INEQUALITY_PARAMETER_LENGTH) {
    throw new Error(`target must hold ${GPU_INEQUALITY_PARAMETER_LENGTH} values`);
  }
  const epsilon = settings.epsilon ?? 1;
  const top = settings.palmaTopShare ?? GPU_INEQUALITY_DEFAULT_PALMA_TOP_SHARE;
  const bottom = settings.palmaBottomShare ?? GPU_INEQUALITY_DEFAULT_PALMA_BOTTOM_SHARE;
  if (!Number.isFinite(epsilon) || epsilon < 0 || epsilon > GPU_INEQUALITY_MAXIMUM_EPSILON) {
    throw new Error(`epsilon must be in [0, ${GPU_INEQUALITY_MAXIMUM_EPSILON}]`);
  }
  for (const [name, share] of [
    ['palmaTopShare', top],
    ['palmaBottomShare', bottom]
  ] as const) {
    if (!(share > 0 && share <= 1)) {
      throw new Error(`${name} must be in (0, 1]`);
    }
  }
  target[0] = epsilon;
  target[1] = top;
  target[2] = bottom;
  target[3] = 0;
  return target;
}
