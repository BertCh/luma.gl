// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of leading control float32 elements of a {@link GPUSimilarLocations} parameter view. */
export const GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH = 4;

/** Rank written for rows that cannot be ranked (invalid, masked out, or excluded references). */
export const GPU_SIMILAR_LOCATIONS_NO_RANK = 0xffffffff;

/** Returns the parameter view length for `attributeCount` attributes: 4 controls plus weights. */
export function getGPUSimilarLocationsParameterLength(attributeCount: number): number {
  return GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH + attributeCount;
}

/** Per-frame settings of {@link GPUSimilarLocations}. */
export type GPUSimilarLocationsSettings = {
  /** Number of top-ranked rows reported in `output.topIds`, clamped to the compile-time capacity. */
  resultCount: number;
  /** `'most'` ranks the nearest rows first; `'least'` ranks the farthest first. Defaults to `'most'`. */
  direction?: 'most' | 'least';
  /** Excludes the reference (selected) rows from the ranking. Defaults to true. */
  excludeReference?: boolean;
  /** Non-negative weight per attribute. Defaults to 1 for every attribute. */
  weights?: ArrayLike<number>;
};

/**
 * Packs per-frame {@link GPUSimilarLocations} parameters.
 *
 * Layout (float32): `[resultCount, direction (0 most, 1 least), excludeReference, reserved,
 * weights...]`.
 *
 * @param settings Result count, direction, reference exclusion and attribute weights.
 * @param attributeCount Number of attributes of the contributor.
 * @param target Optional destination of at least `4 + attributeCount` elements.
 * @throws If the target is too short, a count is invalid, or a weight is negative or not finite.
 */
export function getGPUSimilarLocationsParameterValues(
  settings: GPUSimilarLocationsSettings,
  attributeCount: number,
  target: Float32Array = new Float32Array(getGPUSimilarLocationsParameterLength(attributeCount))
): Float32Array {
  const length = getGPUSimilarLocationsParameterLength(attributeCount);
  if (target.length < length) {
    throw new Error(`Similar locations parameter target must hold ${length} elements`);
  }
  if (!Number.isSafeInteger(settings.resultCount) || settings.resultCount < 0) {
    throw new Error('Similar locations resultCount must be a non-negative integer');
  }
  if (settings.weights && settings.weights.length !== attributeCount) {
    throw new Error('Similar locations weights must have one entry per attribute');
  }
  target[0] = settings.resultCount;
  target[1] = settings.direction === 'least' ? 1 : 0;
  target[2] = settings.excludeReference === false ? 0 : 1;
  target[3] = 0;
  for (let attribute = 0; attribute < attributeCount; attribute++) {
    const weight = settings.weights ? settings.weights[attribute] : 1;
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error('Similar locations weights must be finite and non-negative');
    }
    target[GPU_SIMILAR_LOCATIONS_CONTROL_LENGTH + attribute] = weight;
  }
  return target;
}
