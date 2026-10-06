// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Column offsets of one scale's row in the `indices` output of `GPUSegregation`, for `K` groups.
 * Columns are float32; each scale occupies `stride` consecutive values.
 */
export type GPUSegregationLayout = {
  /** Multigroup information-theory index H (Theil, Reardon and Firebaugh). */
  entropy: number;
  /** Multigroup dissimilarity D (Reardon and Firebaugh). */
  multiGroupDissimilarity: number;
  /** Population diversity `E = sum_m P_m ln(1 / P_m)` (the denominator of H, in nats). */
  diversity: number;
  /** First of `K` per-group dissimilarities `D_g` (group `g` against everyone else). */
  dissimilarity: number;
  /** First of `K` per-group isolation indices `xPx_g`. */
  isolation: number;
  /** First of `K` per-group Atkinson indices `A_g`. */
  atkinson: number;
  /** First of `K * K` interaction indices `xPy`, row-major `[g * K + h]` (exposure of g to h). */
  interaction: number;
  /** Number of columns per scale. */
  stride: number;
};

/** Returns the {@link GPUSegregationLayout} of a `groupCount`-group result. */
export function getGPUSegregationLayout(groupCount: number): GPUSegregationLayout {
  return {
    entropy: 0,
    multiGroupDissimilarity: 1,
    diversity: 2,
    dissimilarity: 3,
    isolation: 3 + groupCount,
    atkinson: 3 + 2 * groupCount,
    interaction: 3 + 3 * groupCount,
    stride: 3 + 3 * groupCount + groupCount * groupCount
  };
}
