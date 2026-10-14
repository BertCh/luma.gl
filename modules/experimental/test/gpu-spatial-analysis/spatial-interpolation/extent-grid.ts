// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import type {GPURasterMetadata} from '../../../src/gpu-raster';

/**
 * Returns the grid whose rows advance north from `[minX, minY]` and that covers `extent`.
 *
 * The packed corners `[startX, startY, endX, endY]` of such a grid equal `extent`, which keeps
 * the cell centers (and therefore every expected value) the same as when the contributors took
 * `width`, `height` and an `extent` directly. Pass `rowDirection: 'south'` for the north-up grid
 * of the same rectangle (row 0 at the top).
 */
export function makeExtentGrid(
  width: number,
  height: number,
  extent: readonly [number, number, number, number],
  rowDirection: 'north' | 'south' = 'north'
): GPURasterMetadata {
  const [minX, minY, maxX, maxY] = extent;
  return makeGPURasterMetadata({
    width,
    height,
    cellSize: [(maxX - minX) / width, (maxY - minY) / height],
    origin: [minX, rowDirection === 'north' ? minY : maxY],
    rowDirection
  });
}
