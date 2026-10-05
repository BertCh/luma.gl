// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphTextureView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainRGBDecode,
  type GPUTerrainRGBDecodeProps,
  type GPUTerrainRGBEncoding
} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-rgb-decode';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildExpectedBits,
  decodeWithPolicy,
  getFloat32Bits,
  INVALID_FLOAT_BITS
} from './terrain-rgb-decode-oracle';

const EXHAUSTIVE_SIDE = 4096;
const EXHAUSTIVE_PIXEL_COUNT = EXHAUSTIVE_SIDE * EXHAUSTIVE_SIDE;
const ALL_VALID_RANGE = [-Infinity, Infinity] as const;

type DecodeOptions = Partial<
  Pick<
    GPUTerrainRGBDecodeProps,
    'alphaNoData' | 'noDataRGB' | 'validRange' | 'clampBathymetry' | 'encoding'
  >
>;

/** Runs the contributor on packed RGBA8 words through the buffer or the texture path. */
async function runDecode(
  device: Device,
  words: Uint32Array,
  width: number,
  height: number,
  path: 'buffer' | 'texture',
  options: DecodeOptions & {inputValidity?: Uint32Array}
): Promise<{values: Uint32Array; validity: Uint32Array}> {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: `rgb-decode-${path}`});
  const outputValues = createOutputBuffer(device, pixelCount);
  const outputValidity = createOutputBuffer(device, pixelCount);
  const disposables: {destroy(): void}[] = [outputValues, outputValidity];
  let input: GPUTerrainRGBDecodeProps['input'];
  let inputValidityView: GPUTerrainRGBDecodeProps['inputValidity'];
  if (path === 'texture') {
    const texture = device.createTexture({
      id: 'encoded-rgb',
      format: 'rgba8unorm',
      width,
      height,
      usage: Texture.SAMPLE | Texture.COPY_DST
    });
    texture.writeData(new Uint8Array(words.buffer, words.byteOffset, words.byteLength));
    disposables.push(texture);
    const handle = graph.importTexture(
      {
        id: 'encoded-rgb',
        format: 'rgba8unorm',
        width,
        height,
        usage: texture.props.usage
      },
      texture
    );
    input = {
      texture: graph.createTextureView(handle, {mipLevelCount: 1}) as GraphTextureView<'rgba8unorm'>
    };
  } else {
    const wordBuffer = createInputBuffer(device, words);
    disposables.push(wordBuffer);
    input = {buffer: importGraphBuffer(graph, 'words', wordBuffer, 'uint32', pixelCount)};
    if (options.inputValidity) {
      const validityBuffer = createInputBuffer(device, options.inputValidity);
      disposables.push(validityBuffer);
      inputValidityView = importGraphBuffer(
        graph,
        'input-validity',
        validityBuffer,
        'uint32',
        pixelCount
      );
    }
  }
  const {inputValidity: _ignored, encoding, ...policy} = options;
  graph.add(
    new GPUTerrainRGBDecode({
      width,
      height,
      encoding: encoding ?? 'terrarium',
      input,
      inputValidity: inputValidityView,
      values: importGraphBuffer(graph, 'values', outputValues, 'float32', pixelCount),
      validity: importGraphBuffer(graph, 'validity', outputValidity, 'uint32', pixelCount),
      ...policy
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const valueBytes = await outputValues.readAsync();
  const validityBytes = await outputValidity.readAsync();
  const result = {
    values: new Uint32Array(valueBytes.slice(0, pixelCount * 4).buffer),
    validity: new Uint32Array(validityBytes.slice(0, pixelCount * 4).buffer)
  };
  compiled.destroy();
  for (const disposable of disposables) {
    disposable.destroy();
  }
  return result;
}

function packWord(red: number, green: number, blue: number, alpha: number): number {
  return ((alpha << 24) | (blue << 16) | (green << 8) | red) >>> 0;
}

/** Row-major pixel `N` holds bytes (N >> 16, (N >> 8) & 255, N & 255) with alpha 255. */
function buildAllCodeWords(): Uint32Array {
  const words = new Uint32Array(EXHAUSTIVE_PIXEL_COUNT);
  for (let code = 0; code < words.length; code++) {
    words[code] = packWord(code >> 16, (code >> 8) & 255, code & 255, 255);
  }
  return words;
}

for (const encoding of ['terrarium', 'mapbox'] as GPUTerrainRGBEncoding[]) {
  for (const path of ['texture', 'buffer'] as const) {
    it(`GPUTerrainRGBDecode ${encoding} ${path} path is bit-exact for all 2^24 RGB triples`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) {
        return;
      }
      const result = await runDecode(
        device,
        buildAllCodeWords(),
        EXHAUSTIVE_SIDE,
        EXHAUSTIVE_SIDE,
        path,
        {
          encoding,
          validRange: ALL_VALID_RANGE
        }
      );
      const expected = buildExpectedBits(encoding);
      let mismatchCount = 0;
      let firstMismatch = -1;
      let invalidCount = 0;
      let distinctSample = 0;
      for (let code = 0; code < expected.length; code++) {
        if (result.values[code] !== expected[code]) {
          mismatchCount++;
          if (firstMismatch < 0) {
            firstMismatch = code;
          }
        }
        invalidCount += result.validity[code] === 1 ? 0 : 1;
        distinctSample += result.values[code] !== result.values[0] ? 1 : 0;
      }
      expect(mismatchCount, `first mismatch at code ${firstMismatch}`).toBe(0);
      expect(invalidCount).toBe(0);
      // A failed compile yields zeros: assert the output is non-trivial.
      expect(distinctSample).toBeGreaterThan(16_000_000);
    }, 300000);
  }
}

const SMALL_WORDS = Uint32Array.from([
  packWord(128, 0, 0, 255), // Terrarium 0 m
  packWord(0, 0, 0, 255), // blank canvas, -32768
  packWord(255, 255, 255, 255), // 32767.996
  packWord(130, 0, 0, 255), // 512 m
  packWord(127, 128, 0, 255), // -128 m (bathymetry)
  packWord(127, 252, 0, 255), // -4 m
  packWord(129, 0, 0, 0), // alpha 0 transparent
  packWord(10, 20, 30, 255),
  packWord(135, 0, 0, 255), // 7 * 256 = 1792 m
  packWord(1, 134, 160, 255), // Mapbox 0 m
  packWord(200, 100, 50, 128),
  packWord(250, 250, 250, 255)
]);

for (const path of ['buffer', 'texture'] as const) {
  for (const encoding of ['terrarium', 'mapbox'] as GPUTerrainRGBEncoding[]) {
    it(`GPUTerrainRGBDecode ${encoding} ${path} applies alpha, noDataRGB, validRange and clamp`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) {
        return;
      }
      const cases: DecodeOptions[] = [
        {encoding},
        {encoding, alphaNoData: false, validRange: ALL_VALID_RANGE},
        {encoding, noDataRGB: [10, 20, 30], validRange: ALL_VALID_RANGE},
        {encoding, validRange: [-100, 1000]},
        {encoding, validRange: [-Infinity, 1000]},
        {encoding, validRange: [-100, Infinity]},
        {encoding, clampBathymetry: true, validRange: ALL_VALID_RANGE}
      ];
      for (const options of cases) {
        const result = await runDecode(device, SMALL_WORDS, 4, 3, path, options);
        const expected = decodeWithPolicy(SMALL_WORDS, encoding, options);
        expect(Array.from(result.values), JSON.stringify(options)).toEqual(
          Array.from(expected.values)
        );
        expect(Array.from(result.validity), JSON.stringify(options)).toEqual(
          Array.from(expected.validity)
        );
      }
      // Non-trivial and nodata convention: default policy invalidates alpha 0 and the blank canvas.
      const defaults = await runDecode(device, SMALL_WORDS, 4, 3, path, {encoding});
      expect(defaults.validity.some(flag => flag === 1)).toBe(true);
      expect(defaults.validity.some(flag => flag === 0)).toBe(true);
      expect(defaults.validity[6]).toBe(0);
      expect(defaults.values[6]).toBe(INVALID_FLOAT_BITS);
      if (encoding === 'terrarium') {
        expect(defaults.validity[1]).toBe(0);
        expect(defaults.validity[2]).toBe(0);
        expect(defaults.values[0]).toBe(getFloat32Bits(0));
        expect(defaults.values[3]).toBe(getFloat32Bits(512));
        const clamped = await runDecode(device, SMALL_WORDS, 4, 3, path, {
          encoding,
          clampBathymetry: true
        });
        expect(clamped.values[4]).toBe(getFloat32Bits(0));
        expect(clamped.values[5]).toBe(getFloat32Bits(0));
        expect(defaults.values[4]).toBe(getFloat32Bits(-128));
        expect(defaults.values[5]).toBe(getFloat32Bits(-4));
      }
    }, 60000);
  }
}

it('GPUTerrainRGBDecode honours inputValidity and never lets a nodata value through', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Pixel 3 holds the blank-canvas -32768 colour but is flagged nodata: no output may carry it.
  const inputValidity = Uint32Array.from([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1]);
  const result = await runDecode(device, SMALL_WORDS, 4, 3, 'buffer', {
    encoding: 'terrarium',
    inputValidity
  });
  const expected = decodeWithPolicy(SMALL_WORDS, 'terrarium', {inputValidity});
  expect(Array.from(result.values)).toEqual(Array.from(expected.values));
  expect(Array.from(result.validity)).toEqual(Array.from(expected.validity));
  expect(result.validity[2]).toBe(0);
  expect(result.values[2]).toBe(INVALID_FLOAT_BITS);
  expect(result.validity[0]).toBe(1);
});
