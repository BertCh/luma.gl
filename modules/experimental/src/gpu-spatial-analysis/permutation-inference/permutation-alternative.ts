// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_DEFAULT_INFERENCE_ALTERNATIVE, type GPUInferenceAlternative} from '../contracts/index';

/**
 * Alternative hypothesis (tail) of a permutation pseudo p-value. This is the shared
 * {@link GPUInferenceAlternative} union; the default is `'two-sided'`.
 *
 * With `R` permutations, `g = #{sim >= observed}` and `l = #{sim <= observed}` (ties count in both):
 * - `'two-sided'` (default): `p = min(2 (min(g, l) + 1) / (R + 1), 1)`; either tail. This is the
 *   doubled smaller tail, not esda's definition: esda 2.10 places the observed value at a percentile
 *   of the simulated statistics and counts the simulations beyond the interpolated percentiles at
 *   both ends, which needs every simulated value of a row stored and sorted. The two can differ by
 *   a few counts (for one 20-permutation fixture esda gives 3/21 and this gives 2/21). Use
 *   `'folded'` for an undirected test that matches esda exactly. Ties never make a row look more
 *   significant: when every simulation equals the observed value, `g = l = R` and `p = 1`.
 * - `'greater'`: `p = (g + 1) / (R + 1)`; the observed statistic is large.
 * - `'lesser'` (alias `'less'`): `p = (l + 1) / (R + 1)`; the observed statistic is small.
 * - `'folded'`: `p = (f + 1) / (R + 1)` with `f = #{|sim - mean| >= |observed - mean|}` and `mean` the
 *   mean of the `R` simulated statistics (the observed value is excluded from the mean, as esda
 *   computes it from the reference distribution). Ties count; the test is symmetric about the
 *   simulated mean.
 * - `'directed'` (legacy, anti-conservative): `p = (min(g, l) + 1) / (R + 1)`. The tail is chosen
 *   from the data and not doubled, so under the null `p` is about twice too small and `p <= 0.05`
 *   is a roughly 10 percent test; do not combine it with multiple-testing control. It is esda's
 *   legacy default (`alternative=None`, `min(g, R - g)`), with one deliberate deviation: esda's
 *   `R - g` counts simulations strictly below the observed value only, so when simulations tie the
 *   observed value (a degenerate or discrete reference distribution) esda reports the smallest
 *   attainable `p = 1/(R+1)`, while `min(g, l)` counts the ties on both sides and reports `p = 1`.
 *   The two agree whenever no simulation equals the observed value.
 *
 * @deprecated Use {@link GPUInferenceAlternative}; this name remains as an alias.
 */
export type GPUPermutationAlternative = GPUInferenceAlternative;

/** Validates an alternative and returns its canonical name (`'less'` becomes `'lesser'`). @internal */
export function resolvePermutationAlternative(
  id: string,
  alternative: GPUInferenceAlternative | undefined
): 'directed' | 'two-sided' | 'greater' | 'lesser' | 'folded' {
  const value = alternative ?? GPU_DEFAULT_INFERENCE_ALTERNATIVE;
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
 * WGSL expression for the exceedance count `M` given u32 variables `greater`, `lesser` and (folded only) `folded`. @internal
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
      // 'directed': the smaller tail, counting ties on both sides (see GPUPermutationAlternative).
      return 'min(greater, lesser)';
  }
}

/** Numerator multiplier of the pseudo p-value (2 for two-sided, else 1). @internal */
export function getAlternativeMultiplier(
  alternative: ReturnType<typeof resolvePermutationAlternative>
): number {
  return alternative === 'two-sided' ? 2 : 1;
}

/**
 * CPU twin of the kernels' exceedance count and pseudo p-value for one tested row or statistic.
 *
 * @param alternative Tail; `undefined` is `'two-sided'`.
 * @param counts `greater = #{sim >= observed}`, `lesser = #{sim <= observed}` and, for `'folded'`
 * only, `folded = #{|sim - mean| >= |observed - mean|}`.
 * @param permutations Number of simulations `R`.
 * @returns The exceedance count `M` and `p = min(m (M + 1), R + 1) / (R + 1)`.
 */
export function getPermutationPValue(
  alternative: GPUInferenceAlternative | undefined,
  counts: {greater: number; lesser: number; folded?: number},
  permutations: number
): {exceedances: number; pValue: number} {
  const resolved = resolvePermutationAlternative('getPermutationPValue', alternative);
  const exceedances = {
    greater: counts.greater,
    lesser: counts.lesser,
    'two-sided': Math.min(counts.greater, counts.lesser),
    folded: counts.folded ?? 0,
    directed: Math.min(counts.greater, counts.lesser)
  }[resolved];
  const multiplier = getAlternativeMultiplier(resolved);
  return {
    exceedances,
    pValue: Math.min(multiplier * (exceedances + 1), permutations + 1) / (permutations + 1)
  };
}
