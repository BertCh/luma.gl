// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Smallest H3 cell inradius per resolution, in degrees of latitude, with a 0.85 safety factor.
 *
 * Measured with h3-js 4.4.0: for every cell of resolutions 0 to 3 and for 120,000 uniformly random
 * sphere points plus the three-ring neighborhood of every pentagon at resolutions 4 to 10, the
 * minimum distance from the cell center to its boundary edges, measured in the local equirectangular
 * metric `(dLng * cos(centerLat), dLat)` and restricted to cells with a center within 89 degrees of
 * the equator (the cover lattice never goes beyond 89 degrees). The minimum is always reached next
 * to a pentagon. Resolutions 11 to 15 are extrapolated by dividing by sqrt(7) per level, which the
 * measured ratios (2.59 to 2.70) support. Resolutions 0 to 2 are further reduced to keep the table
 * monotone.
 *
 * @internal
 */
export const CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES: readonly number[] = [
  3.7,
  0.56,
  0.5,
  0.32,
  0.12,
  0.046,
  0.017,
  0.0066,
  0.00245,
  0.00094,
  0.00035,
  ...[1, 2, 3, 4, 5].map(level => 0.00035 / Math.sqrt(7) ** level)
];

/**
 * Lattice spacing in degrees of latitude per H3 resolution: half the minimum inradius, so that the
 * lattice point nearest a cell center is at most `0.354 * inradius` away and lies inside the cell.
 *
 * @internal
 */
export const CELL_COVER_H3_LATTICE_SPACING_DEGREES: readonly number[] =
  CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES.map(inradius => Math.fround(0.5 * inradius));

/** Latitude (degrees) beyond which the H3 cover lattice is clamped. @internal */
export const CELL_COVER_H3_MAXIMUM_LATITUDE = 89;
