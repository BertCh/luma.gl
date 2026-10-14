// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  GET_NAN_WGSL,
  GET_QUIET_NAN_SEEDED_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_NAN_BITS_WGSL
} from '../../src/utils/wgsl-common';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

const WORDS_PER_ROW = 8;
const ROWS = 4;

function isNaNBits(bits: number): boolean {
  return (bits & 0x7fffffff) >>> 0 > 0x7f800000;
}

// Report B section 5: `getQuietNaN(seed)` builds its NaN from `0x7fc00000u | (seed & 0u)` and
// relies on the driver not folding it away. Write every NaN helper to storage, plus NaN that went
// through a select, arithmetic and a storage round trip, and check the bits on the CPU and with
// `isNanBits` on the GPU.
it('WGSL NaN helpers survive compilation and storage writes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'wgsl-nan'});
  const inputBuffer = createInputBuffer(device, Float32Array.from([NaN, 1, Infinity, -0]));
  const outputBuffer = createOutputBuffer(device, ROWS * WORDS_PER_ROW);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'write-nan',
      operation: 'WGSLNaNTest',
      bindings: [
        {
          name: 'inputs',
          view: importGraphBuffer(graph, 'inputs', inputBuffer, 'float32', 4),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'words',
          view: importGraphBuffer(graph, 'words', outputBuffer, 'uint32', ROWS * WORDS_PER_ROW),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: ROWS,
      declarations: `${GET_NAN_WGSL}${GET_QUIET_NAN_SEEDED_WGSL}${IS_NAN_BITS_WGSL}${IS_FINITE_BITS_WGSL}`,
      body: `let seeded = getQuietNaN(index);
  let unseeded = getNaN();
  let selected = select(seeded, 1.0, index > 100u);
  let propagated = seeded * 2.0 + 1.0;
  let stored = inputs[inputsOffset];
  let base = wordsOffset + index * ${WORDS_PER_ROW}u;
  words[base] = bitcast<u32>(seeded);
  words[base + 1u] = bitcast<u32>(unseeded);
  words[base + 2u] = bitcast<u32>(selected);
  words[base + 3u] = bitcast<u32>(propagated);
  words[base + 4u] = bitcast<u32>(stored);
  words[base + 5u] = select(0u, 1u, isNanBits(seeded) && isNanBits(propagated) && isNanBits(stored));
  words[base + 6u] = select(0u, 1u, isFiniteBits(seeded) || isFiniteBits(inputs[inputsOffset + 2u]));
  words[base + 7u] = select(0u, 1u, isNanBits(inputs[inputsOffset + 1u]) || isNanBits(inputs[inputsOffset + 3u]));`
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const words = await readUint32(outputBuffer, ROWS * WORDS_PER_ROW);
  compiled.destroy();
  inputBuffer.destroy();
  outputBuffer.destroy();
  for (let row = 0; row < ROWS; row++) {
    const values = words.slice(row * WORDS_PER_ROW, (row + 1) * WORDS_PER_ROW);
    for (let word = 0; word < 5; word++) {
      expect(isNaNBits(values[word]), `row ${row} word ${word} bits ${values[word]}`).toBe(true);
    }
    expect(values[0]).toBe(0x7fc00000);
    expect(values.slice(5), `row ${row} GPU bit tests`).toEqual([1, 0, 0]);
  }
});
