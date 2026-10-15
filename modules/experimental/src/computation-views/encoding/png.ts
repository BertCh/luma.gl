// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Encodes row-major palette indices as a DOM-free PNG with stored DEFLATE blocks. */
export function encodePNG(
  width: number,
  height: number,
  pixels: Uint8Array,
  palette: readonly string[]
): Uint8Array {
  const integer = (value: number) => [
    value >>> 24,
    (value >>> 16) & 255,
    (value >>> 8) & 255,
    value & 255
  ];
  const chunk = (name: string, data: number[] | Uint8Array): number[] => {
    const payload = [...Array.from(name, character => character.charCodeAt(0)), ...data];
    let checksum = 0xffffffff;
    for (const byte of payload) {
      checksum ^= byte;
      for (let bit = 0; bit < 8; bit++)
        checksum = (checksum >>> 1) ^ (checksum & 1 ? 0xedb88320 : 0);
    }
    return [...integer(data.length), ...payload, ...integer((checksum ^ 0xffffffff) >>> 0)];
  };
  const scanlines = new Uint8Array((width + 1) * height);
  for (let row = 0; row < height; row++)
    scanlines.set(pixels.subarray(row * width, (row + 1) * width), row * (width + 1) + 1);
  const deflate: number[] = [0x78, 0x01];
  for (let offset = 0; offset < scanlines.length; offset += 65535) {
    const length = Math.min(65535, scanlines.length - offset);
    deflate.push(
      Number(offset + length === scanlines.length),
      length & 255,
      length >>> 8,
      ~length & 255,
      (~length >>> 8) & 255
    );
    for (const byte of scanlines.subarray(offset, offset + length)) deflate.push(byte);
  }
  let sum = 1;
  let cumulative = 0;
  for (const byte of scanlines) {
    sum = (sum + byte) % 65521;
    cumulative = (cumulative + sum) % 65521;
  }
  deflate.push(...integer(((cumulative << 16) | sum) >>> 0));
  const colors = palette.flatMap(color => {
    const value = Number.parseInt(color.slice(1), 16);
    return [(value >>> 16) & 255, (value >>> 8) & 255, value & 255];
  });
  return Uint8Array.from([
    137,
    80,
    78,
    71,
    13,
    10,
    26,
    10,
    ...chunk('IHDR', [...integer(width), ...integer(height), 8, 3, 0, 0, 0]),
    ...chunk('PLTE', colors),
    ...chunk('IDAT', Uint8Array.from(deflate)),
    ...chunk('IEND', [])
  ]);
}

/** Converts bytes to base64 without Node Buffer or browser globals. */
export function encodeBase64(bytes: Uint8Array): string {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let result = '';
  for (let index = 0; index < bytes.length; index += 3) {
    const value = (bytes[index] << 16) | ((bytes[index + 1] ?? 0) << 8) | (bytes[index + 2] ?? 0);
    result +=
      alphabet[(value >>> 18) & 63] +
      alphabet[(value >>> 12) & 63] +
      (index + 1 < bytes.length ? alphabet[(value >>> 6) & 63] : '=') +
      (index + 2 < bytes.length ? alphabet[value & 63] : '=');
  }
  return result;
}
