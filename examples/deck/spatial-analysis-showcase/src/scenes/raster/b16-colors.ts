// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Light-weight color constants of the raster chapter (safe to import from `*.scene.ts`). */

/** One 8-bit class color per USGS burn severity class, regrowth to high severity. */
export const SEVERITY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [0, 104, 55, 230],
  [102, 189, 99, 230],
  [217, 239, 139, 200],
  [254, 224, 139, 235],
  [253, 174, 97, 240],
  [244, 109, 67, 245],
  [165, 0, 38, 250]
];

/** Class names matching {@link SEVERITY_COLORS}. */
export const SEVERITY_NAMES = [
  'Enhanced regrowth, high',
  'Enhanced regrowth, low',
  'Unburned',
  'Low severity',
  'Moderate-low severity',
  'Moderate-high severity',
  'High severity'
] as const;

/** Categorical colors cycled over patch labels. */
export const PATCH_COLORS: readonly (readonly [number, number, number, number])[] = [
  [78, 201, 255, 235],
  [255, 148, 72, 235],
  [189, 122, 255, 235],
  [87, 235, 168, 235],
  [255, 105, 168, 235],
  [245, 220, 87, 235],
  [107, 158, 255, 235],
  [255, 92, 92, 235]
];

/** What the suitability map can show. */
export type SuitabilityLayer =
  | 'score'
  | 'meanCriteria'
  | 'spread'
  | 'weakest'
  | 'strongest'
  | 'range'
  | 'validLayers'
  | 'c-severity'
  | 'c-slope'
  | 'c-cover'
  | 'c-proximity'
  | 'c-greenness'
  | 'cover';

/** ESA WorldCover class names by zone index (`code / 10 - 1`). */
export const COVER_NAMES = [
  'Tree cover',
  'Shrubland',
  'Grassland',
  'Cropland',
  'Built-up',
  'Bare / sparse vegetation',
  'Snow and ice',
  'Permanent water',
  'Herbaceous wetland'
] as const;

/** ESA WorldCover legend colors by zone index. */
export const COVER_COLORS: readonly (readonly [number, number, number, number])[] = [
  [0, 100, 0, 235],
  [255, 187, 34, 235],
  [255, 255, 76, 235],
  [240, 150, 255, 235],
  [250, 0, 0, 235],
  [180, 180, 180, 235],
  [240, 240, 240, 235],
  [0, 100, 200, 235]
];

/** Four ramp stops, slow to fast. */
export type WindRamp = readonly (readonly [number, number, number])[];

/** Speed ramp used on light and dark basemaps: blue, teal, amber, red-orange. */
export const WIND_RAMP: WindRamp = [
  [45, 90, 225],
  [0, 175, 195],
  [240, 175, 0],
  [235, 70, 40]
];
