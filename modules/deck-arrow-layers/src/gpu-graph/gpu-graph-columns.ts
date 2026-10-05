// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';

/** Element format of a GPU-resident, row-aligned graph column. */
export type GPUGraphColumnFormat = 'uint32' | 'float32';

/**
 * One GPU-resident column with exactly one packed 4-byte row per graph node (or edge).
 *
 * Columns are bound as read-only storage buffers and indexed by `instance_index`, never as
 * vertex attributes, so any number of analysis outputs can be swapped in without touching the
 * WebGL2-era 16-attribute ceiling or rebuilding a pipeline. `float32` rows are read with
 * `bitcast<f32>` from the same `array<u32>` binding, so switching formats is a uniform change.
 */
export type GPUGraphNodeColumn = {
  /** Caller-owned buffer with `Buffer.STORAGE` usage. The layer never destroys it. */
  buffer: Buffer;
  /** Format of every row. */
  format: GPUGraphColumnFormat;
};

/** An sRGB color with 0-255 channels. */
export type GPUGraphColor = readonly [number, number, number];

/** Maximum number of palette stops a {@link GPUGraphColorScale} can hold. */
export const GPU_GRAPH_MAXIMUM_PALETTE_LENGTH = 8;

/** Row value treated as "no data", for example unreached reachability bands. */
export const GPU_GRAPH_NULL_UINT32 = 0xffffffff;

/**
 * Maps a column value to a color entirely on the GPU. Uploaded as uniforms, so changing the
 * domain or palette rewrites one small uniform block and never rebuilds a pipeline.
 */
export type GPUGraphColorScale = {
  /**
   * `'linear'` maps `domain` to `[0, 1]` and interpolates evenly spaced palette stops.
   * `'categorical'` picks `palette[value % palette.length]` and ignores `domain`.
   */
  type: 'linear' | 'categorical';
  /** Linear input domain `[min, max]`. Defaults to `[0, 1]`. */
  domain?: readonly [number, number];
  /** One to {@link GPU_GRAPH_MAXIMUM_PALETTE_LENGTH} colors. */
  palette: readonly GPUGraphColor[];
  /** Color for `uint32` rows equal to {@link GPU_GRAPH_NULL_UINT32} or non-finite `float32` rows. */
  nullColor?: GPUGraphColor;
};

/** Maps a column value to a node radius in pixels, linearly and clamped. */
export type GPUGraphSizeScale = {
  /** Input domain `[min, max]`. Defaults to `[0, 1]`. */
  domain?: readonly [number, number];
  /** Output radius range `[minPixels, maxPixels]`. */
  range: readonly [number, number];
};
