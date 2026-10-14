// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Whether observer heights are above the ground under them or absolute heights in the elevation datum. */
export type GPUTerrainHeightReference = 'ground' | 'absolute';

/** Reserved vertical datum: heights above the reference ellipsoid (GNSS, drone, ICESat-2). */
export const GPU_ELLIPSOIDAL_VERTICAL_DATUM = 'ellipsoidal';

/**
 * Throws when both vertical datums are declared and differ (keys: trim + upper-case, no aliases).
 * Other strings name a geoid/national datum: 'EPSG:5773' (EGM96), 'EPSG:3855' (EGM2008),
 * 'EPSG:5703' (NAVD88). The library checks declarations and never converts (no geoid-undulation input).
 */
export function assertCompatibleGPUVerticalDatums(
  id: string,
  heights?: string,
  elevation?: string,
  heightsName = 'observer'
): void {
  if (
    heights !== undefined &&
    elevation !== undefined &&
    heights.trim().toUpperCase() !== elevation.trim().toUpperCase()
  ) {
    throw new Error(
      `${id} ${heightsName} heights are ${heights} but elevation is ${elevation}; convert to one vertical datum first`
    );
  }
}
