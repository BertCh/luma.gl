// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {inflateSync} from 'node:zlib';
import {expect} from 'vitest';
import type {renderRasterImage} from '../../../src/computation-views/encoding/raster-image';

/** Independent chunk/scanline decoder used only as a test oracle. */
export function decodePNG(bytes: Uint8Array) {
  const buffer = Buffer.from(bytes);
  expect([...buffer.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  const chunks = new Map<string, Buffer>();
  for (let offset = 8; offset < buffer.length; ) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString('ascii', offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    // Polynomial long division, independent of the encoder's reflected implementation.
    let checksum = 0xffffffff;
    for (const byte of buffer.subarray(offset + 4, offset + 8 + length)) {
      let reversed = 0;
      for (let bit = 0; bit < 8; bit++) reversed = (reversed << 1) | ((byte >>> bit) & 1);
      checksum ^= reversed << 24;
      for (let bit = 0; bit < 8; bit++)
        checksum = (checksum << 1) ^ (checksum < 0 ? 0x04c11db7 : 0);
    }
    let reversed = 0;
    for (let bit = 0; bit < 32; bit++) reversed = (reversed << 1) | ((checksum >>> bit) & 1);
    expect((reversed ^ 0xffffffff) >>> 0, `${type} CRC32`).toBe(
      buffer.readUInt32BE(offset + 8 + length)
    );
    chunks.set(type, data);
    offset += length + 12;
  }
  expect([...chunks.keys()]).toEqual(['IHDR', 'PLTE', 'IDAT', 'IEND']);
  const header = chunks.get('IHDR')!;
  const width = header.readUInt32BE(0);
  const height = header.readUInt32BE(4);
  expect([...header.subarray(8)]).toEqual([8, 3, 0, 0, 0]);
  // Node's inflater also validates Adler32 and the stored-block complements.
  const scanlines = inflateSync(chunks.get('IDAT')!);
  const pixels: number[] = [];
  for (let row = 0; row < height; row++) {
    expect(scanlines[row * (width + 1)]).toBe(0);
    pixels.push(...scanlines.subarray(row * (width + 1) + 1, (row + 1) * (width + 1)));
  }
  return {width, height, pixels, palette: [...chunks.get('PLTE')!]};
}

/** Decodes the single image emitted by the view helper. */
export function decodeRasterImage(image: ReturnType<typeof renderRasterImage>) {
  return decodePNG(Buffer.from(String(image.attributes['href']).split(',')[1], 'base64'));
}
