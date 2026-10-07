// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialAnalysisColor} from '../../engine/layers';

/** Color of places that are not significant. */
export const NEUTRAL: SpatialAnalysisColor = [150, 160, 175, 150];

/** Color of a place with no data. */
export const NO_DATA: SpatialAnalysisColor = [140, 148, 160, 60];

/**
 * Gi* bins are `sint32` in `-3..3`. Reinterpreted as uint32 and reduced modulo the palette size 8
 * they select entries 0 (not significant), 1..3 (hot 90/95/99%) and 7, 6, 5 (cold 90/95/99%).
 */
export const GI_COLORS: Record<number, SpatialAnalysisColor> = {
  1: [253, 174, 97, 235],
  2: [244, 109, 67, 245],
  3: [165, 0, 38, 255],
  7: [171, 217, 233, 235],
  6: [116, 173, 209, 245],
  5: [49, 54, 149, 255]
};

/** Local Moran quadrant codes 1 HH, 2 LH, 3 LL, 4 HL. */
export const MORAN_COLORS: Record<number, SpatialAnalysisColor> = {
  1: [165, 0, 38, 255],
  2: [171, 217, 233, 245],
  3: [49, 54, 149, 255],
  4: [253, 174, 97, 245]
};

/** Focus-row classes of the weights scene: 0 other, 1 neighbour, 2 focus. */
export const FOCUS_COLORS: readonly SpatialAnalysisColor[] = [
  [175, 184, 196, 70],
  [245, 140, 30, 235],
  [190, 25, 70, 255]
];

/** Colour of an island (a place without neighbours). */
export const ISLAND_COLOR: SpatialAnalysisColor = [225, 20, 60, 255];

/** Colors of the global-statistics views. */
export const BLACK_JOIN_COLOR: SpatialAnalysisColor = [196, 70, 40, 235];
export const WHITE_JOIN_COLOR: SpatialAnalysisColor = [120, 150, 190, 150];

/** Link color per theme. */
export function getLinkColor(theme: 'light' | 'dark', dense = false): SpatialAnalysisColor {
  if (dense) return theme === 'dark' ? [235, 240, 255, 55] : [30, 45, 80, 42];
  return theme === 'dark' ? [235, 240, 255, 95] : [30, 45, 80, 80];
}

/** Outline color per theme. */
export function getOutlineColor(theme: 'light' | 'dark'): SpatialAnalysisColor {
  return theme === 'dark' ? [235, 240, 255, 60] : [20, 30, 50, 70];
}
