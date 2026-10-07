// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Categorical palette of the Geometry chapter. The layers (WGSL) and the legends read this one
 * list. This module has no imports so scene files can use it without loading the GPU code.
 */
export const B3_PALETTE: readonly (readonly [number, number, number])[] = [
  [78, 168, 222],
  [240, 150, 60],
  [150, 110, 220],
  [70, 196, 140],
  [232, 90, 140],
  [226, 196, 70],
  [120, 140, 232],
  [226, 96, 80]
];
