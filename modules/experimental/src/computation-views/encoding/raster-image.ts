// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {h, type ViewElementNode} from '../view-node';
import {encodeBase64, encodePNG} from './png';

/** Display-only resampling of captured values; never recomputes the captured analysis. */
export type RasterImageOptions = {
  columns: number;
  rows: number;
  limit: number;
  valueAt(index: number): number;
  colorFor(value: number): string;
  categorical?: boolean;
  flipRows?: boolean;
  attributes?: ViewElementNode['attributes'];
};

/** Describes display resampling without implying additional GPU analysis. */
export function getRasterSamplingNote(
  columns: number,
  rows: number,
  limit: number,
  categorical = false
): string {
  return Math.max(columns, rows) > limit
    ? `downsampled from ${columns}×${rows} (${categorical ? 'nearest' : 'mean'})`
    : '';
}

/** One indexed PNG per panel, classed after nearest-category or finite-mean resampling. */
export function renderRasterImage(options: RasterImageOptions): ViewElementNode {
  const {columns, rows, limit} = options;
  const ratio = Math.min(1, limit / Math.max(columns, rows));
  const width = Math.max(1, Math.floor(columns * ratio));
  const height = Math.max(1, Math.floor(rows * ratio));
  const pixels = new Uint8Array(width * height);
  const palette: string[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const firstColumn = Math.floor((column * columns) / width);
      const firstRow = Math.floor((row * rows) / height);
      let value = options.valueAt(
        Math.floor(((row + 0.5) * rows) / height) * columns +
          Math.floor(((column + 0.5) * columns) / width)
      );
      if (!options.categorical) {
        let sum = 0;
        let count = 0;
        for (
          let sourceRow = firstRow;
          sourceRow < Math.floor(((row + 1) * rows) / height);
          sourceRow++
        ) {
          for (
            let sourceColumn = firstColumn;
            sourceColumn < Math.floor(((column + 1) * columns) / width);
            sourceColumn++
          ) {
            const sample = options.valueAt(sourceRow * columns + sourceColumn);
            if (Number.isFinite(sample)) {
              sum += sample;
              count++;
            }
          }
        }
        value = count ? sum / count : NaN;
      }
      const color = options.colorFor(value);
      let colorIndex = palette.indexOf(color);
      if (colorIndex < 0) {
        colorIndex = palette.length;
        palette.push(color);
      }
      pixels[(options.flipRows ? height - row - 1 : row) * width + column] = colorIndex;
    }
  }
  const note = getRasterSamplingNote(columns, rows, limit, options.categorical);
  return h(
    'image',
    {
      width: columns,
      height: rows,
      preserveAspectRatio: 'none',
      'image-rendering': 'pixelated',
      ...options.attributes,
      'data-raster-width': width,
      'data-raster-height': height,
      'data-sampling': options.categorical ? 'nearest' : 'mean',
      href: `data:image/png;base64,${encodeBase64(encodePNG(width, height, pixels, palette))}`
    },
    note ? h('desc', null, note) : null
  );
}
