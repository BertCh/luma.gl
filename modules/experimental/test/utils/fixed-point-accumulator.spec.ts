// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createFixedPointClearNode,
  createFixedPointDecodeNode,
  createFixedPointScaleNodes,
  FIXED_POINT_WGSL,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../src/utils/fixed-point-accumulator';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {sumFixedPointOracle} from './fixed-point-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Adds `values[i]` into cell `keys[i]` with one quantize and one atomic add per row. */
async function sumOnGpu(
  device: Device,
  values: Float32Array,
  keys: Uint32Array,
  cellCount: number
): Promise<Uint32Array> {
  const graph = new GPUCommandGraph<undefined>(device, {id: 'fixed-point'});
  const valuesBuffer = createInputBuffer(device, values);
  const keysBuffer = createInputBuffer(device, keys);
  const outputBuffer = createOutputBuffer(device, cellCount);
  const valuesView = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', values.length);
  const keysView = importGraphBuffer(graph, 'keys', keysBuffer, 'uint32', keys.length);
  const scale = createTransientView(graph, 'scale', 'uint32', 1);
  const words = createTransientView(graph, 'words', 'uint32', cellCount * 3);
  const nodes = [
    ...createFixedPointScaleNodes(graph, {
      id: 'scale',
      operation: 'FixedPointSpec',
      values: valuesView,
      scale
    }),
    createFixedPointClearNode(graph, {id: 'clear', operation: 'FixedPointSpec', words}),
    createWGSLKernelNode(graph, {
      id: 'accumulate',
      operation: 'FixedPointSpec',
      bindings: [
        {name: 'values', view: valuesView, type: 'f32', access: 'read'},
        {name: 'keys', view: keysView, type: 'u32', access: 'read'},
        {name: 'scale', view: scale, type: 'u32', access: 'read'},
        {name: 'words', view: words, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: values.length,
      declarations: `override HEADROOM_BITS: u32;
override CELL_COUNT: u32;
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('words')}`,
      constants: {HEADROOM_BITS: getFixedPointHeadroomBits(values.length), CELL_COUNT: cellCount},
      body: `let key = keys[keysOffset + index];
  if (key < CELL_COUNT) {
    let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], HEADROOM_BITS);
    addFixedPointToWords(key, quantizeFixedPoint(values[valuesOffset + index], unitExponent));
  }`
    }),
    createFixedPointDecodeNode(graph, {
      id: 'decode',
      operation: 'FixedPointSpec',
      words,
      scale,
      maximumTermCount: values.length,
      output: importGraphBuffer(graph, 'output', outputBuffer, 'float32', cellCount)
    })
  ];
  for (const node of nodes) graph.add(node);
  const compiled = await graph.compileAsync();
  submitGraph(device, compiled, undefined);
  const bits = Uint32Array.from(await readUint32(outputBuffer, cellCount));
  compiled.destroy();
  for (const buffer of [valuesBuffer, keysBuffer, outputBuffer]) buffer.destroy();
  return bits;
}

function getBits(values: number[]): Uint32Array {
  return new Uint32Array(Float32Array.from(values).buffer);
}

it('fixed-point sums equal the exact BigInt oracle bit for bit, in any row order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(29);
  const rowCount = 200_000;
  const cellCount = 37;
  const values = new Float32Array(rowCount);
  const keys = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    // Half the rows share cell 0 (a hotspot); magnitudes span 2^-12 to 2^8 with both signs.
    keys[row] = random() < 0.5 ? 0 : Math.floor(random() * cellCount);
    values[row] = (random() - 0.4) * 2 ** Math.floor(random() * 20 - 12);
  }
  // A 1e7 outlier and its cancellation, non-finite rows (skipped) and out-of-range keys.
  values[10] = 1e7;
  keys[10] = 5;
  values[11] = -1e7;
  keys[11] = 5;
  values[12] = Number.NaN;
  values[13] = Number.POSITIVE_INFINITY;
  keys[14] = cellCount;
  keys[15] = 0xffffffff;
  const expected = sumFixedPointOracle(values, keys, cellCount);
  const first = await sumOnGpu(device, values, keys, cellCount);
  expect(Array.from(first)).toEqual(Array.from(getBits(expected.sums)));
  // Two runs and a reversed and a shuffled row order (different workgroups, different atomic
  // interleavings) give the same bits.
  expect(Array.from(await sumOnGpu(device, values, keys, cellCount))).toEqual(Array.from(first));
  const order = Array.from({length: rowCount}, (_, row) => rowCount - 1 - row);
  for (let row = order.length - 1; row > 0; row -= 3) {
    const other = Math.floor(random() * (row + 1));
    [order[row], order[other]] = [order[other], order[row]];
  }
  const shuffled = await sumOnGpu(
    device,
    Float32Array.from(order, row => values[row]),
    Uint32Array.from(order, row => keys[row]),
    cellCount
  );
  expect(Array.from(shuffled)).toEqual(Array.from(first));
  // The outlier pair cancels exactly: cell 5 keeps its small terms (rule 7).
  const smallTermsOfCell5 = sumFixedPointOracle(
    values.map((value, row) => (row === 10 || row === 11 ? 0 : value)),
    keys,
    cellCount
  );
  expect(Math.abs(new Float32Array(first.buffer)[5] - smallTermsOfCell5.sums[5])).toBeLessThan(
    1e-6 * Math.max(1, Math.abs(smallTermsOfCell5.sums[5]))
  );
});

it('fixed-point sums decode empty, negative, tiny and huge cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The unit follows the largest term, so the huge cells run separately: next to 3e38 the
  // dynamic range 2^(71 - h) ends near 2^61, and -3.75 would round to zero units.
  const values = Float32Array.from([-3.5, -0.25, 2 ** -100, 2 ** -100, 0, -0]);
  const keys = Uint32Array.from([0, 0, 1, 1, 2, 2]);
  const bits = await sumOnGpu(device, values, keys, 4);
  expect(Array.from(bits)).toEqual(Array.from(getBits(sumFixedPointOracle(values, keys, 4).sums)));
  // 2^-100 is 2^-102 of the largest term, past the 2^(71 - h) = 2^68 range: it rounds to 0.
  expect(Array.from(new Float32Array(bits.buffer))).toEqual([-3.75, 0, 0, 0]);
  const tiny = Float32Array.from([2 ** -100, 2 ** -100, -(2 ** -120)]);
  const tinyBits = await sumOnGpu(device, tiny, Uint32Array.from([0, 0, 1]), 2);
  expect(Array.from(new Float32Array(tinyBits.buffer))).toEqual([2 ** -99, -(2 ** -120)]);
  const huge = Float32Array.from([3e38, 3e38, -3e38, 1]);
  const hugeKeys = Uint32Array.from([0, 0, 1, 1]);
  const hugeBits = await sumOnGpu(device, huge, hugeKeys, 2);
  expect(Array.from(hugeBits)).toEqual(
    Array.from(getBits(sumFixedPointOracle(huge, hugeKeys, 2).sums))
  );
  expect(Array.from(new Float32Array(hugeBits.buffer))).toEqual([
    Number.POSITIVE_INFINITY,
    Math.fround(-3e38)
  ]);
});
