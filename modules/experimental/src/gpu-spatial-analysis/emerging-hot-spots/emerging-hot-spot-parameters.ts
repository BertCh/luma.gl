// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in an emerging-hot-spot parameter buffer. */
export const GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH = 8;

/** Number of float32 rows written to an optional `globalStatistics` output. */
export const GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH = 4;

/** Largest supported compile-time `sliceCount`; long Mann-Kendall series use 64 GPU lanes. */
export const GPU_EMERGING_HOT_SPOT_MAXIMUM_SLICE_COUNT = 256;

/** Largest supported compile-time `maximumRadius`, in cells. */
export const GPU_EMERGING_HOT_SPOT_MAXIMUM_RADIUS = 32;

/**
 * Stable `uint32` codes of the 17 ArcGIS emerging hot spot categories written to `category`.
 *
 * Hot codes are `1..8` and the cold mirror `9..16` in the same order, so `code + 8` converts a
 * hot category to its cold variant. `0` means no pattern was detected.
 */
export const GPU_EMERGING_HOT_SPOT_CATEGORIES = {
  /** No pattern detected. */
  NO_PATTERN: 0,
  /** Significant hot spot in the final slice and never before. */
  NEW_HOT: 1,
  /** Uninterrupted run of at least two hot slices at the end, none before, under 90% hot. */
  CONSECUTIVE_HOT: 2,
  /** Hot for at least 90% of slices including the final one, with a significant upward trend. */
  INTENSIFYING_HOT: 3,
  /** Hot for at least 90% of slices including the final one, with no significant trend. */
  PERSISTENT_HOT: 4,
  /** Hot for at least 90% of slices including the final one, with a significant downward trend. */
  DIMINISHING_HOT: 5,
  /** On-again off-again hot spot that was never a cold spot, under 90% hot. */
  SPORADIC_HOT: 6,
  /** Hot in the final slice with some earlier cold slice, under 90% hot. */
  OSCILLATING_HOT: 7,
  /** Not hot in the final slice but hot for at least 90% of slices. */
  HISTORICAL_HOT: 8,
  /** Significant cold spot in the final slice and never before. */
  NEW_COLD: 9,
  /** Uninterrupted run of at least two cold slices at the end, none before, under 90% cold. */
  CONSECUTIVE_COLD: 10,
  /** Cold for at least 90% of slices including the final one, with a significant downward trend. */
  INTENSIFYING_COLD: 11,
  /** Cold for at least 90% of slices including the final one, with no significant trend. */
  PERSISTENT_COLD: 12,
  /** Cold for at least 90% of slices including the final one, with a significant upward trend. */
  DIMINISHING_COLD: 13,
  /** On-again off-again cold spot that was never a hot spot, under 90% cold. */
  SPORADIC_COLD: 14,
  /** Cold in the final slice with some earlier hot slice, under 90% cold. */
  OSCILLATING_COLD: 15,
  /** Not cold in the final slice but cold for at least 90% of slices. */
  HISTORICAL_COLD: 16
} as const;

/** Two-sided normal critical z of the 90%, 95% and 99% confidence levels. */
export const GPU_EMERGING_HOT_SPOT_CRITICAL_Z_SCORES: Readonly<Record<string, number>> = {
  '0.9': 1.6448536269514722,
  '0.95': 1.959963984540054,
  '0.99': 2.5758293035489004
};

/**
 * CPU description of the per-frame parameters of `GPUEmergingHotSpots`. Every field can change
 * between encodings without rebuilding or recompiling the graph.
 */
export type GPUEmergingHotSpotParameters = {
  /**
   * Lattice mode: spatial neighborhood radius in lattice cells. Cell `(dx, dy)` offsets with
   * `dx * dx + dy * dy <= radius * radius` are neighbors (the focal cell included, so `0` is the
   * focal cell alone). Clamped to the compile-time `maximumRadius`. Ignored in weights mode.
   * Defaults to `0`.
   */
  radius?: number;
  /** Temporal window `k >= 0`: neighbors span the current slice and the `k` previous slices. */
  temporalWindow: number;
  /** Confidence level of hot and cold bins: `0.9`, `0.95` or `0.99`. Defaults to `0.9`. */
  confidenceLevel?: 0.9 | 0.95 | 0.99;
  /** Explicit critical z that overrides `confidenceLevel` for hot and cold bins. */
  criticalZ?: number;
  /**
   * Two-sided p-value at or below which the Mann-Kendall trend is significant, in `(0, 1)`.
   * Defaults to `1 - confidenceLevel`.
   */
  trendSignificanceLevel?: number;
  /** Fraction of slices that makes a spot persistent or historical, in `(0.5, 1]`. Default `0.9`. */
  persistentFraction?: number;
};

/**
 * Packs emerging-hot-spot parameters into the 8-element float32 layout read by
 * `GPUEmergingHotSpots`.
 *
 * Layout: `[radius, temporalWindow, criticalZ, trendSignificanceLevel, persistentFraction, 0, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite or out of range, `confidenceLevel` is not 0.9, 0.95 or 0.99
 * without a `criticalZ`, or `target` is too short.
 */
export function getGPUEmergingHotSpotParameterValues(
  parameters: GPUEmergingHotSpotParameters,
  target: Float32Array = new Float32Array(GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH) {
    throw new Error(
      `Emerging hot spot target must hold ${GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH} elements`
    );
  }
  const {temporalWindow} = parameters;
  const radius = parameters.radius ?? 0;
  if (!Number.isFinite(radius) || radius < 0) {
    throw new Error('Emerging hot spot radius must be a finite number >= 0');
  }
  if (!Number.isInteger(temporalWindow) || temporalWindow < 0) {
    throw new Error('Emerging hot spot temporalWindow must be an integer >= 0');
  }
  const confidenceLevel = parameters.confidenceLevel ?? 0.9;
  let criticalZ = parameters.criticalZ;
  if (criticalZ === undefined) {
    criticalZ = GPU_EMERGING_HOT_SPOT_CRITICAL_Z_SCORES[String(confidenceLevel)];
    if (criticalZ === undefined) {
      throw new Error('Emerging hot spot confidenceLevel must be 0.9, 0.95 or 0.99');
    }
  } else if (!Number.isFinite(criticalZ) || criticalZ <= 0) {
    throw new Error('Emerging hot spot criticalZ must be a finite number > 0');
  }
  const trendSignificanceLevel = parameters.trendSignificanceLevel ?? 1 - confidenceLevel;
  if (!(trendSignificanceLevel > 0 && trendSignificanceLevel < 1)) {
    throw new Error('Emerging hot spot trendSignificanceLevel must be in (0, 1)');
  }
  const persistentFraction = parameters.persistentFraction ?? 0.9;
  if (!(persistentFraction > 0.5 && persistentFraction <= 1)) {
    throw new Error('Emerging hot spot persistentFraction must be in (0.5, 1]');
  }
  target.set([
    radius,
    temporalWindow,
    criticalZ,
    trendSignificanceLevel,
    persistentFraction,
    0,
    0,
    0
  ]);
  return target;
}
