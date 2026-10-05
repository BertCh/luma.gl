// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength
} from '../../../src/geospatial/region-statistics';

it('getGPURegionStatisticsSummaryLength adds the header', () => {
  expect(getGPURegionStatisticsSummaryLength(0)).toBe(8);
  expect(getGPURegionStatisticsSummaryLength(16)).toBe(24);
});

it('decodeGPURegionStatistics round-trips every field', () => {
  const words = new Uint32Array(12);
  const floats = new Float32Array(words.buffer);
  words[0] = 5;
  words[1] = 4;
  floats[2] = 55;
  floats[3] = 13.75;
  floats[4] = -5;
  floats[5] = 30;
  words[6] = 1;
  words[7] = 3;
  words.set([1, 2, 3, 4], 8);
  const result = decodeGPURegionStatistics(words);
  expect(result).toMatchObject({
    selectedCount: 5,
    valueCount: 4,
    sum: 55,
    mean: 13.75,
    minimum: -5,
    maximum: 30,
    histogramOutsideCount: 1,
    selectionTruncated: true,
    regionTruncated: true
  });
  expect(Array.from(result.histogram)).toEqual([1, 2, 3, 4]);
  expect(() => decodeGPURegionStatistics(new Uint8Array(28))).toThrow();
  expect(() => decodeGPURegionStatistics(new Uint8Array(34))).toThrow();
});
