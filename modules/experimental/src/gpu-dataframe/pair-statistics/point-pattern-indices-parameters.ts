// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUInferenceAlternative} from '../../analysis-core/contracts/inference';
import {validatePairStatisticsBounds} from './pair-statistics-parameters';

/** Number of float32 elements in a `GPUPointPatternIndices` parameter buffer. */
export const GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH = 8;

/**
 * Number of float32 rows of the `clarkEvans` summary of `GPUPointPatternIndices`:
 * `[n, observed, expected, ratio, standardError, zScore, pValue]`.
 */
export const GPU_CLARK_EVANS_LENGTH = 7;

/**
 * Number of float32 rows of the `quadratStatistics` summary of `GPUPointPatternIndices`:
 * `[quadratCount, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]`.
 */
export const GPU_QUADRAT_STATISTICS_LENGTH = 7;

/**
 * Clark-Evans edge correction codes stored in parameter slot 5. `'none'` is the plain
 * infinite-plane expectation; `'donnelly'` is Donnelly (1978) for a rectangular window.
 */
export const GPU_CLARK_EVANS_CORRECTION = {none: 0, donnelly: 1} as const;

/** Clark-Evans edge correction name. */
export type GPUClarkEvansCorrection = keyof typeof GPU_CLARK_EVANS_CORRECTION;

/** Tail codes stored in parameter slot 6. */
export const GPU_POINT_PATTERN_ALTERNATIVE = {'two-sided': 0, greater: 1, lesser: 2} as const;

/** Largest quadrat count (`columns * rows`) of `GPUPointPatternIndices`. */
export const GPU_QUADRAT_MAXIMUM_COUNT = 65536;

/**
 * CPU description of the per-frame parameters of `GPUPointPatternIndices`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPUPointPatternIndicesSettings = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` window. Rows outside it, or with a non-finite coordinate,
   * are excluded. Its area is the study area of the Clark-Evans index and the quadrats tile it.
   */
  bounds: readonly [number, number, number, number];
  /**
   * Size hint of the neighbor-search cells (cells are at least this wide). Results never depend on
   * it, only speed: a value near the expected nearest-neighbor distance is fastest.
   */
  maximumDistance: number;
  /**
   * Clark-Evans edge correction. Defaults to `'donnelly'` (Donnelly 1978: expected mean distance
   * `0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance
   * `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the window perimeter), because the plain
   * expectation is biased toward apparent clustering for windows with a large perimeter.
   * `'none'` gives the classic `0.5 sqrt(A / n)` with standard error `0.26136 sqrt(A) / n`
   * (pointpats, ArcGIS).
   */
  clarkEvansCorrection?: GPUClarkEvansCorrection;
  /**
   * Tail of the Clark-Evans z-test and the quadrat chi-square test. Defaults to `'two-sided'`.
   * `'greater'` tests for a statistic above its complete-spatial-randomness expectation
   * (Clark-Evans: regular; quadrat chi-square: clustered), `'lesser'` for one below it. The
   * permutation-only tails `'folded'` and `'directed'` throw.
   */
  alternative?: GPUInferenceAlternative;
};

/**
 * Packs `GPUPointPatternIndices` parameters into the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, clarkEvansCorrectionCode, alternativeCode, 0]`.
 *
 * @param@param settings Settings to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, the bounds are inverted, `maximumDistance <= 0`, the correction
 * or alternative is unknown, or `target` is too short.
 */
export function getGPUPointPatternIndicesParameterValues(
  settings: GPUPointPatternIndicesSettings,
  target: Float32Array = new Float32Array(GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH) {
    throw new Error(
      `GPUPointPatternIndices target must hold ${GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH} elements`
    );
  }
  validatePairStatisticsBounds('GPUPointPatternIndices', settings.bounds, settings.maximumDistance);
  const correction = settings.clarkEvansCorrection ?? 'donnelly';
  if (!Object.prototype.hasOwnProperty.call(GPU_CLARK_EVANS_CORRECTION, correction)) {
    throw new Error(`GPUPointPatternIndices clarkEvansCorrection must be 'none' or 'donnelly'`);
  }
  const alternative = settings.alternative ?? 'two-sided';
  if (!Object.prototype.hasOwnProperty.call(GPU_POINT_PATTERN_ALTERNATIVE, alternative)) {
    throw new Error(
      `GPUPointPatternIndices alternative must be 'two-sided', 'greater' or 'lesser'`
    );
  }
  target.set([
    ...settings.bounds,
    settings.maximumDistance,
    GPU_CLARK_EVANS_CORRECTION[correction],
    GPU_POINT_PATTERN_ALTERNATIVE[alternative as keyof typeof GPU_POINT_PATTERN_ALTERNATIVE],
    0
  ]);
  return target;
}
