// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Discrete global grid families that {@link GPUPointToCell} can index points into.
 *
 * Every family stores a cell as one 64-bit key (two `uint32` words, little-endian `(low, high)`):
 * - `'quadbin'`: CARTO Quadbin.
 * - `'h3'`: H3 index, mode 1.
 * - `'quadkey'`: Web Mercator tile packed as length in bits 58..63 and base-4 digits
 *   (`digit = xBit | yBit << 1`, first digit most significant) right-aligned in the low 58 bits.
 * - `'geohash'`: length in bits 60..63 and base-32 character codes right-aligned in the low 60
 *   bits, first character most significant.
 * - `'s2'`: standard S2CellId (face in bits 61..63, Hilbert position, trailing 1 marker bit).
 */
export type GPUCellIndexFamily = 'quadbin' | 'h3' | 'quadkey' | 'geohash' | 's2';

/** Inclusive resolution range of one {@link GPUCellIndexFamily}. */
export type GPUCellIndexResolutionRange = {
  /** Coarsest valid resolution. */
  minimum: number;
  /** Finest valid resolution. */
  maximum: number;
};

/**
 * Valid resolutions per family: Quadbin 0-26, H3 0-15, quadkey 1-29 (zoom), geohash 1-12
 * (characters), S2 0-30 (level). The packed quadkey and geohash keys are never zero for these
 * ranges, so the zero key can mean "no cell".
 */
export const GPU_CELL_INDEX_RESOLUTION_RANGES: Readonly<
  Record<GPUCellIndexFamily, GPUCellIndexResolutionRange>
> = {
  quadbin: {minimum: 0, maximum: 26},
  h3: {minimum: 0, maximum: 15},
  quadkey: {minimum: 1, maximum: 29},
  geohash: {minimum: 1, maximum: 12},
  s2: {minimum: 0, maximum: 30}
};

/** All supported family names, in documentation order. */
export const GPU_CELL_INDEX_FAMILIES: readonly GPUCellIndexFamily[] = [
  'quadbin',
  'h3',
  'quadkey',
  'geohash',
  's2'
];

/** Returns true when `value` names a {@link GPUCellIndexFamily}. */
export function isGPUCellIndexFamily(value: unknown): value is GPUCellIndexFamily {
  return typeof value === 'string' && value in GPU_CELL_INDEX_RESOLUTION_RANGES;
}

/**
 * Throws unless `family` is a known family and `resolution` an integer in its range.
 *
 * @param id Recipe ID used as the error prefix.
 * @param family Family name to check.
 * @param resolution Resolution to check against the family's range.
 */
export function validateCellIndexResolution(
  id: string,
  family: GPUCellIndexFamily,
  resolution: number
): void {
  if (!isGPUCellIndexFamily(family)) {
    throw new Error(`${id} family must be one of ${GPU_CELL_INDEX_FAMILIES.join(', ')}`);
  }
  const {minimum, maximum} = GPU_CELL_INDEX_RESOLUTION_RANGES[family];
  if (!Number.isInteger(resolution) || resolution < minimum || resolution > maximum) {
    throw new Error(`${id} ${family} resolution must be an integer in [${minimum}, ${maximum}]`);
  }
}
