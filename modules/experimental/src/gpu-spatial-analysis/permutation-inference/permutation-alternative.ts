// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Alternative hypothesis (tail) of a permutation pseudo p-value, following esda 2.10's
 * `esda.significance.calculate_significance` with `R` permutations, `g = #{sim >= observed}` and
 * `l = #{sim <= observed}` (ties count in both):
 * - `'directed'`: `p = (min(g, R - g) + 1) / (R + 1)`. The tail is chosen from the data, so `p` is
 *   uniformly too small; this is esda's legacy default (`alternative=None`) and the default here.
 * - `'greater'`: `p = (g + 1) / (R + 1)`; the observed statistic is large.
 * - `'lesser'` (alias `'less'`): `p = (l + 1) / (R + 1)`; the observed statistic is small.
 * - `'two-sided'`: `p = min(2 (min(g, l) + 1) / (R + 1), 1)`; either tail. This is the doubled
 *   smaller tail, not esda's definition: esda 2.10 places the observed value at a percentile of the
 *   simulated statistics and counts the simulations beyond the interpolated percentiles at both
 *   ends, which needs every simulated value of a row stored and sorted. The two can differ by a few
 *   counts (for one 20-permutation fixture esda gives 3/21 and this gives 2/21). Use `'folded'` for
 *   an undirected test that matches esda exactly.
 * - `'folded'`: `p = (f + 1) / (R + 1)` with `f = #{|sim - mean| >= |observed - mean|}` and `mean` the
 *   mean of the `R` simulated statistics (the observed value is excluded from the mean, as esda
 *   computes it from the reference distribution). Ties count; the test is symmetric about the
 *   simulated mean.
 */
export type GPUPermutationAlternative =
  | 'directed'
  | 'two-sided'
  | 'greater'
  | 'lesser'
  | 'less'
  | 'folded';

/** Validates an alternative and returns its canonical name (`'less'` becomes `'lesser'`). @internal */
export function resolvePermutationAlternative(
  id: string,
  alternative: GPUPermutationAlternative | undefined
): 'directed' | 'two-sided' | 'greater' | 'lesser' | 'folded' {
  const value = alternative ?? 'directed';
  if (value === 'less') {
    return 'lesser';
  }
  if (!['directed', 'two-sided', 'greater', 'lesser', 'folded'].includes(value)) {
    throw new Error(
      `${id} alternative must be 'directed', 'two-sided', 'greater', 'lesser' or 'folded', got ${String(alternative)}`
    );
  }
  return value;
}

/**
 * WGSL expression for the exceedance count `M` given u32 variables `greater`, `lesser` and (folded only) `folded` and the
 * permutation count `permutations`. @internal
 */
export function getExceedanceCountWGSL(
  alternative: ReturnType<typeof resolvePermutationAlternative>
): string {
  switch (alternative) {
    case 'greater':
      return 'greater';
    case 'lesser':
      return 'lesser';
    case 'two-sided':
      return 'min(greater, lesser)';
    case 'folded':
      return 'folded';
    default:
      return 'min(greater, permutations - greater)';
  }
}

/** Numerator multiplier of the pseudo p-value (2 for two-sided, else 1). @internal */
export function getAlternativeMultiplier(
  alternative: ReturnType<typeof resolvePermutationAlternative>
): number {
  return alternative === 'two-sided' ? 2 : 1;
}
