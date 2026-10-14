// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphBufferHandle, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../src/gpu-raster';

it('makeGPURasterBand preserves only explicitly declared vertical datums', () => {
  const grid = makeGPURasterMetadata({width: 3, height: 2});
  const buffer = new GraphBufferHandle(
    {id: 'band-datum'},
    {id: 'elevation', byteLength: 24, usage: 0},
    false
  );
  const values = new GraphDataView(buffer, {
    format: 'float32',
    length: 6,
    byteOffset: 0,
    byteStride: 4,
    rowByteLength: 4
  });
  expect(makeGPURasterBand(values, grid, {verticalDatum: 'EPSG:5773'}).verticalDatum).toBe(
    'EPSG:5773'
  );
  expect(makeGPURasterBand(values, grid)).not.toHaveProperty('verticalDatum');
  expect(makeGPURasterBand(values, grid, {verticalDatum: undefined})).not.toHaveProperty(
    'verticalDatum'
  );
});
