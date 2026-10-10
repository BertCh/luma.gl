// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Emerging-hot-spot category colors, indexed by category code; code 0 is transparent. */
export const EMERGING_CATEGORY_COLORS: readonly (readonly [number, number, number])[] = [
  [0, 0, 0],
  [255, 224, 102],
  [255, 160, 60],
  [165, 0, 38],
  [215, 48, 39],
  [244, 160, 150],
  [253, 208, 162],
  [200, 100, 200],
  [170, 140, 140],
  [160, 230, 255],
  [90, 170, 255],
  [8, 48, 107],
  [33, 102, 172],
  [150, 190, 230],
  [190, 230, 215],
  [120, 100, 220],
  [130, 150, 170]
];

/** Emerging-hot-spot category names in code order. */
export const EMERGING_CATEGORY_NAMES = [
  'No pattern',
  'New hot spot',
  'Consecutive hot spot',
  'Intensifying hot spot',
  'Persistent hot spot',
  'Diminishing hot spot',
  'Sporadic hot spot',
  'Oscillating hot spot',
  'Historical hot spot',
  'New cold spot',
  'Consecutive cold spot',
  'Intensifying cold spot',
  'Persistent cold spot',
  'Diminishing cold spot',
  'Sporadic cold spot',
  'Oscillating cold spot',
  'Historical cold spot'
] as const;
