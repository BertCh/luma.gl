// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUMultipleTestingMethod} from '../contracts/inference';

/** Number of float32 elements in an emerging-hot-spot parameter buffer. */
export const GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH = 8;

/** Number of float32 rows written to an optional `globalStatistics` output. */
export const GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH = 4;

/** Largest supported compile-time `sliceCount`; long Mann-Kendall series use 64 GPU lanes. */
export const GPU_EMERGING_HOT_SPOT_MAXIMUM_SLICE_COUNT = 256;

/** Largest supported compile-time `maximumRadius`, in cells. */
export const GPU_EMERGING_HOT_SPOT_MAXIMUM_RADIUS = 32;

/** Sentinel written to an optional `lastValidSlice` output for a cell with no finite slice. */
export const GPU_EMERGING_HOT_SPOT_NO_SLICE = 0xffffffff;

/** Default `minimumTrendSliceCount`: ESRI advises at least 10 time steps for Mann-Kendall. */
export const GPU_EMERGING_HOT_SPOT_DEFAULT_MINIMUM_TREND_SLICE_COUNT = 10;

/**
 * Stable `uint32` codes of the 17 ArcGIS emerging hot spot categories and the extra
 * `UNDETERMINED` code written to `category`.
 *
 * Hot codes are `1..8` and the cold mirror `9..16` in the same order, so `code + 8` converts a
 * hot category to its cold variant. `0` means no pattern was detected and `17` means the pattern
 * cannot be decided (see `UNDETERMINED`).
 *
 * Legend contract: a code never changes meaning with `multipleTesting`, `confidenceLevel`,
 * `finalSlice` or any other option. The options only change which bins count as hot or cold, so a
 * legend built once stays valid.
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
  HISTORICAL_COLD: 16,
  /**
   * The category cannot be decided. Either the final slice is missing and `finalSlice` is
   * `'require'` (the default), or the cell is persistent in one direction (at least
   * `persistentFraction` of its slices) but has fewer finite slices than `minimumTrendSliceCount`,
   * so its Intensifying, Persistent or Diminishing class would need a Mann-Kendall trend decision
   * that is not supported by enough data. Cells without any finite slice stay `NO_PATTERN`.
   */
  UNDETERMINED: 17
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
  /**
   * Confidence level of hot and cold bins, in `(0, 1)`. A bin is hot or cold when its adjusted
   * two-sided p-value is at most `1 - confidenceLevel`. With `multipleTesting: 'none'` only `0.9`,
   * `0.95` and `0.99` (the tabulated critical z values) or an explicit `criticalZ` are accepted.
   * Defaults to `0.95` (it was `0.9`).
   */
  confidenceLevel?: number;
  /**
   * Explicit critical z that overrides `confidenceLevel` for hot and cold bins. Only meaningful
   * without a correction: it requires `multipleTesting: 'none'` and throws otherwise, because a
   * fixed critical z cannot express a family-wise or false discovery rate threshold.
   */
  criticalZ?: number;
  /**
   * The compile-time `multipleTesting` of the node these parameters feed. It does not change the
   * packed values; it makes this function reject a `criticalZ` combined with a correction and
   * decide whether `confidenceLevel` must be tabulated. Defaults to `'benjamini-hochberg'`, like
   * the node.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /**
   * Two-sided p-value at or below which the Mann-Kendall trend is significant, in `(0, 1)`.
   * Defaults to `1 - confidenceLevel`. Consecutive Gi* z values share `temporalWindow` slices, so
   * the series is serially correlated and this p-value is anti-conservative (ESRI has the same
   * property); treat it as a screening threshold.
   */
  trendSignificanceLevel?: number;
  /**
   * Cells with fewer finite slices than this get NaN `trendZ` and `trendP`, and a persistent cell
   * that would need the trend decision gets `UNDETERMINED`. A non-negative integer; `0` disables
   * the guard. Defaults to 10, ESRI's guidance for Mann-Kendall.
   */
  minimumTrendSliceCount?: number;
  /** Fraction of slices that makes a spot persistent or historical, in `(0.5, 1]`. Default `0.9`. */
  persistentFraction?: number;
};

/**
 * Packs emerging-hot-spot parameters into the 8-element float32 layout read by
 * `GPUEmergingHotSpots`.
 *
 * Layout: `[radius, temporalWindow, criticalZ, trendSignificanceLevel, persistentFraction,
 * minimumTrendSliceCount, significanceLevel, 0]`, where `significanceLevel = 1 - confidenceLevel`
 * and `criticalZ` is only read with `multipleTesting: 'none'`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite or out of range, `criticalZ` is combined with a correction,
 * `confidenceLevel` is not 0.9, 0.95 or 0.99 without a `criticalZ` and `multipleTesting: 'none'`,
 * or `target` is too short.
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
  const confidenceLevel = parameters.confidenceLevel ?? 0.95;
  if (!(confidenceLevel > 0 && confidenceLevel < 1)) {
    throw new Error('Emerging hot spot confidenceLevel must be in (0, 1)');
  }
  const multipleTesting = parameters.multipleTesting ?? 'benjamini-hochberg';
  let criticalZ = parameters.criticalZ;
  if (criticalZ !== undefined) {
    if (multipleTesting !== 'none') {
      throw new Error("Emerging hot spot criticalZ requires multipleTesting 'none'");
    }
    if (!Number.isFinite(criticalZ) || criticalZ <= 0) {
      throw new Error('Emerging hot spot criticalZ must be a finite number > 0');
    }
  } else {
    criticalZ = GPU_EMERGING_HOT_SPOT_CRITICAL_Z_SCORES[String(confidenceLevel)];
    if (criticalZ === undefined) {
      if (multipleTesting === 'none') {
        throw new Error('Emerging hot spot confidenceLevel must be 0.9, 0.95 or 0.99');
      }
      // Unused with a correction: the adjusted p-value decides.
      criticalZ = 0;
    }
  }
  const minimumTrendSliceCount =
    parameters.minimumTrendSliceCount ?? GPU_EMERGING_HOT_SPOT_DEFAULT_MINIMUM_TREND_SLICE_COUNT;
  if (!Number.isInteger(minimumTrendSliceCount) || minimumTrendSliceCount < 0) {
    throw new Error('Emerging hot spot minimumTrendSliceCount must be an integer >= 0');
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
    minimumTrendSliceCount,
    1 - confidenceLevel,
    0
  ]);
  return target;
}
