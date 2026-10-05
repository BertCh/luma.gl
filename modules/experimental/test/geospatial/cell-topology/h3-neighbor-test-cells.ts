// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellToChildren, getPentagons, gridDisk, latLngToCell} from 'h3-js';

/**
 * H3 test cells at resolutions 4-15: `perResolution` area-uniform random cells per resolution,
 * every pentagon with its two-ring, and the children of resolution-3 pentagons at resolution 5.
 */
export function getH3TestCells(random: () => number, perResolution: number): string[] {
  const cells = new Set<string>();
  for (let resolution = 4; resolution <= 15; resolution++) {
    for (let index = 0; index < perResolution; index++) {
      const latitude = (Math.asin(2 * random() - 1) * 180) / Math.PI;
      const longitude = 360 * random() - 180;
      cells.add(latLngToCell(latitude, longitude, resolution));
    }
    for (const pentagon of getPentagons(resolution)) {
      for (const cell of gridDisk(pentagon, 2)) {
        cells.add(cell);
      }
    }
  }
  for (const pentagon of getPentagons(3)) {
    for (const cell of cellToChildren(pentagon, 5)) {
      cells.add(cell);
    }
  }
  return [...cells];
}
