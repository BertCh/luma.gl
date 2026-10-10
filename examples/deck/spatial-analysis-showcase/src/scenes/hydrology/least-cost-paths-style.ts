// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleRamp} from '../../engine/ramps';

/** Cost limit slider value that means "no limit" (minutes). */
export const NO_COST_LIMIT_MINUTES = 720;
/** Distance cap slider value that means "no cap" (kilometers). */
export const NO_DISTANCE_CAP_KILOMETERS = 40;
export const BAND_COUNT = 8;

/** Colors of the eight land-cover classes, in {@link LAND_COVER_CLASSES} order. */
export const LAND_COVER_PALETTE = [
  [34, 102, 51, 255],
  [143, 170, 70, 255],
  [200, 210, 110, 255],
  [230, 190, 90, 255],
  [205, 60, 60, 255],
  [190, 180, 160, 255],
  [60, 120, 220, 255],
  [90, 170, 170, 255]
] as const;

export const LAND_COVER_CLASSES = [
  'Tree cover',
  'Shrubland',
  'Grassland',
  'Cropland',
  'Built-up',
  'Bare / sparse',
  'Water',
  'Wetland'
] as const;

/** Colors of start points in the nearest-start allocation display. */
export const START_PALETTE = [
  [255, 190, 60, 235],
  [90, 190, 255, 235],
  [190, 130, 255, 235],
  [90, 225, 160, 235],
  [255, 120, 160, 235],
  [240, 235, 90, 235],
  [140, 160, 255, 235],
  [255, 140, 90, 235]
] as const;

/** Travel-time band colors, nearest first. */
export const BAND_PALETTE = Array.from({length: BAND_COUNT}, (_, band) => {
  const [red, green, blue] = sampleRamp('lajolla', 1 - (band + 0.5) / BAND_COUNT);
  return [red, green, blue, 235] as const;
});
