// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata, GPURasterScalarFormat} from '../types';

/** Options for {@link makeGPURasterMetadata}. */
export type GPURasterMetadataOptions = {
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /**
   * Cell size in world units, `[x, y]` or one value for square cells. Defaults to 1. Meters for
   * `cellSizeMode: 'uniform'`, EPSG:3857 meters for `'web-mercator'`, degrees for `'geographic'`.
   */
  cellSize?: number | readonly [number, number];
  /** World `[x, y]` of the top-left corner of pixel `(0, 0)`. Defaults to `[0, 0]`. */
  origin?: readonly [number, number];
  /**
   * Direction in which rows advance in world y. Defaults to `'south'`: row 0 is the top row and
   * world y decreases down the grid (a north-up raster).
   */
  rowDirection?: 'south' | 'north';
};

/**
 * Returns the metadata of an axis-aligned area grid: `affine = [cellX, 0, originX, 0, -cellY,
 * originY]` (`+cellY` when rows advance north).
 *
 * Every raster and terrain analysis contributor takes its shape from `grid`, and every parameter
 * packer that needs a cell size or a latitude band reads them from the same object.
 */
export function makeGPURasterMetadata(options: GPURasterMetadataOptions): GPURasterMetadata {
  const cellSize = options.cellSize ?? 1;
  const [cellX, cellY] = typeof cellSize === 'number' ? [cellSize, cellSize] : cellSize;
  const [originX, originY] = options.origin ?? [0, 0];
  const rowStep = (options.rowDirection ?? 'south') === 'south' ? -cellY : cellY;
  return {
    width: options.width,
    height: options.height,
    affine: [cellX, 0, originX, 0, rowStep, originY],
    pixelInterpretation: 'area'
  };
}

/** Optional band metadata for {@link makeGPURasterBand}. */
export type GPURasterBandOptions = {
  /** Band identifier. Defaults to `'band'`. */
  id?: string;
  /** Optional validity flags, one per pixel: zero rejects a sample. */
  validity?: GraphDataView<'uint32'>;
  /** Raw nodata value, compared before calibration. */
  noDataValue?: number;
  /** Calibration multiplier. Defaults to one. */
  scale?: number;
  /** Calibration offset. Defaults to zero. */
  offset?: number;
  /**
   * Vertical datum of height values, such as `'ellipsoidal'` or `'EPSG:5773'` (EGM96).
   * Descriptive only: operators compare declared datums and never convert. Absent means
   * undeclared; derived bands do not inherit it.
   */
  verticalDatum?: string;
};

/**
 * Wraps a packed per-pixel view, typically the output of one contributor, as a buffer raster band
 * on `grid`, so it can feed the next contributor (for example `GPUTerrainFlow` directions into
 * `GPUTerrainHeightAboveDrainage`) without a hand-written band literal.
 *
 * @throws If the view does not hold exactly one value per grid pixel.
 */
export function makeGPURasterBand<Format extends GPURasterScalarFormat>(
  values: GraphDataView<Format>,
  grid: GPURasterMetadata,
  options: GPURasterBandOptions = {}
): GPURasterBufferBand<Format> {
  const pixelCount = grid.width * grid.height;
  if (
    values.length !== pixelCount ||
    (options.validity && options.validity.length !== pixelCount)
  ) {
    throw new Error(`${options.id ?? 'band'} must hold one value per grid pixel`);
  }
  return {
    id: options.id ?? 'band',
    format: values.format,
    storage: {kind: 'buffer', values},
    ...(options.validity ? {validity: options.validity} : {}),
    ...(options.noDataValue !== undefined ? {noDataValue: options.noDataValue} : {}),
    ...(options.scale !== undefined ? {scale: options.scale} : {}),
    ...(options.offset !== undefined ? {offset: options.offset} : {}),
    ...(options.verticalDatum !== undefined ? {verticalDatum: options.verticalDatum} : {})
  } as GPURasterBufferBand<Format>;
}
