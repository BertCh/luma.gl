// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {encodePNG, encodeBase64} from '../../../src/computation-views/encoding/png';
import {renderRasterImage} from '../../../src/computation-views/encoding/raster-image';

import {decodePNG, decodeRasterImage} from '../fixtures/png';

describe('DOM-free indexed PNG', () => {
  test.each([
    [1, 1],
    [3, 2],
    [257, 260]
  ])('round trips %i×%i including multiple stored blocks', (width, height) => {
    const pixels = Uint8Array.from({length: width * height}, (_, index) => index % 2);
    const bytes = encodePNG(width, height, pixels, ['#123456', '#abcdef']);
    expect(encodeBase64(bytes)).toBe(Buffer.from(bytes).toString('base64'));
    expect(decodePNG(bytes)).toEqual({
      width,
      height,
      pixels: [...pixels],
      palette: [18, 52, 86, 171, 205, 239]
    });
  });
  test('averages ordered samples before classification and samples categories without averaging', () => {
    const values = [0, 10, NaN, NaN, 0, 10, NaN, NaN];
    const make = (categorical: boolean) =>
      renderRasterImage({
        columns: 4,
        rows: 2,
        limit: 2,
        categorical,
        valueAt: index => values[index],
        colorFor: value =>
          !Number.isFinite(value) ? '#dddddd' : value === 10 ? '#000000' : '#ffffff'
      });
    expect(decodeRasterImage(make(false))).toMatchObject({
      width: 2,
      height: 1,
      pixels: [0, 1],
      palette: [255, 255, 255, 221, 221, 221]
    });
    expect(decodeRasterImage(make(true)).palette).toEqual([0, 0, 0, 221, 221, 221]);
    expect(make(false).children).toContainEqual(
      expect.objectContaining({tag: 'desc', children: ['downsampled from 4×2 (mean)']})
    );
  });
  test('preserves row orientation and the longest-side cap on non-square grids', () => {
    const image = renderRasterImage({
      columns: 320,
      rows: 80,
      limit: 64,
      flipRows: true,
      valueAt: index => (index < 320 * 40 ? 0 : 1),
      colorFor: value => (value === 0 ? '#000000' : '#ffffff')
    });
    const decoded = decodeRasterImage(image);
    expect([decoded.width, decoded.height]).toEqual([64, 16]);
    expect(decoded.pixels.slice(0, 64)).toEqual(new Array(64).fill(1));
    expect(decoded.pixels.slice(-64)).toEqual(new Array(64).fill(0));
  });
});
