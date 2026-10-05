// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM as EMPTY_MAXIMUM,
  GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM as EMPTY_MINIMUM,
  type GPURasterExtremaPyramidLayout
} from '../../../src/gpu-raster/raster-pyramid';

type Extrema = {maximum: Float32Array; minimum: Float32Array};

/**
 * CPU mirror of the GPU build, in the exact same order: level 0 scans footprints rows then
 * columns; higher levels reduce the existing children (2cx,2cy), (2cx+1,2cy), (2cx,2cy+1),
 * (2cx+1,2cy+1) of the level below. Compare-from-sentinel with strict `>` / `<`.
 */
export function computeRasterExtremaPyramid(
  values: Float32Array,
  validity: ArrayLike<number> | undefined,
  layout: GPURasterExtremaPyramidLayout
): Extrema {
  const maximum = new Float32Array(layout.length);
  const minimum = new Float32Array(layout.length);
  const extra = layout.footprint === 'bilinear' ? 1 : 0;
  for (const level of layout.levels) {
    for (let cy = 0; cy < level.height; cy++) {
      for (let cx = 0; cx < level.width; cx++) {
        let mx = EMPTY_MAXIMUM;
        let mn = EMPTY_MINIMUM;
        if (level.level === 0) {
          const x1 = Math.min(cx * level.blockSize + level.blockSize + extra, layout.width);
          const y1 = Math.min(cy * level.blockSize + level.blockSize + extra, layout.height);
          for (let y = cy * level.blockSize; y < y1; y++) {
            for (let x = cx * level.blockSize; x < x1; x++) {
              const pixel = y * layout.width + x;
              if (validity === undefined || validity[pixel] !== 0) {
                const v = values[pixel];
                if (v > mx) mx = v;
                if (v < mn) mn = v;
              }
            }
          }
        } else {
          const below = layout.levels[level.level - 1];
          for (let child = 0; child < 4; child++) {
            const column = 2 * cx + (child & 1);
            const row = 2 * cy + (child >> 1);
            if (column < below.width && row < below.height) {
              const index = below.offset + row * below.width + column;
              if (maximum[index] > mx) mx = maximum[index];
              if (minimum[index] < mn) mn = minimum[index];
            }
          }
        }
        maximum[level.offset + cy * level.width + cx] = mx;
        minimum[level.offset + cy * level.width + cx] = mn;
      }
    }
  }
  return {maximum, minimum};
}

/** Direct definition: every level straight from the pixels of each cell footprint. */
export function computeRasterExtremaPyramidDirect(
  values: Float32Array,
  validity: ArrayLike<number> | undefined,
  layout: GPURasterExtremaPyramidLayout
): Extrema {
  const maximum = new Float32Array(layout.length);
  const minimum = new Float32Array(layout.length);
  const extra = layout.footprint === 'bilinear' ? 1 : 0;
  for (const level of layout.levels) {
    for (let cy = 0; cy < level.height; cy++) {
      for (let cx = 0; cx < level.width; cx++) {
        let mx = EMPTY_MAXIMUM;
        let mn = EMPTY_MINIMUM;
        const x1 = Math.min(cx * level.blockSize + level.blockSize + extra, layout.width);
        const y1 = Math.min(cy * level.blockSize + level.blockSize + extra, layout.height);
        for (let y = cy * level.blockSize; y < y1; y++) {
          for (let x = cx * level.blockSize; x < x1; x++) {
            const pixel = y * layout.width + x;
            if (validity === undefined || validity[pixel] !== 0) {
              mx = Math.max(mx, values[pixel]);
              mn = Math.min(mn, values[pixel]);
            }
          }
        }
        maximum[level.offset + cy * level.width + cx] = mx;
        minimum[level.offset + cy * level.width + cx] = mn;
      }
    }
  }
  return {maximum, minimum};
}
