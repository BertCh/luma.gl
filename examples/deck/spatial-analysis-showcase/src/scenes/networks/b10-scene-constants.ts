// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Light constants shared by B10 scene files and their compute modules (no luma or engine imports). */

/** Largest values of the range sliders; the top end means "no upper limit". */
export const SPEED_MAXIMUM = 100;
export const LENGTH_MAXIMUM = 1000;

/** Category colors for communities and for the component highlight. */
export const COMMUNITY_COLORS = [
  [66, 133, 230, 255],
  [235, 122, 40, 255],
  [48, 170, 100, 255],
  [170, 90, 200, 255],
  [220, 190, 40, 255],
  [220, 70, 110, 255],
  [40, 175, 190, 255],
  [140, 140, 150, 255]
] as const;
export const COMPONENT_COLORS = [
  [60, 130, 220, 255],
  [235, 64, 52, 255]
] as const;

/** Category palette of the accuracy coloring: wrong, same edge, same street (other direction). */
export const ACCURACY_COLORS = [
  [235, 64, 52, 255],
  [30, 190, 120, 255],
  [245, 180, 40, 255]
] as const;
