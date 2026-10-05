// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {createMapGraphKernelNode} from '../../../src/map-graphs/map-graph-kernels';
import {
  getFeistelHalfBits,
  getFeistelPermutationIndex,
  getFeistelRoundKeys,
  getPhilox4x32,
  PERMUTATION_RANDOM_WGSL,
  PhiloxStream
} from '../../../src/map-graphs/permutation-inference/permutation-random';
import {createOutputBuffer, readUint32} from '../map-graph-test-utils';

const BOUNDS = [1, 2, 3, 6, 7, 1000, 65537, 0x80000001, 0xfffffffe, 0xffffffff];
const DRAWS = 16;
const FEISTEL_COUNTS = [1, 2, 5, 1000, 4097];

it('the WGSL Philox, bounded draws and Feistel bijection are bit-identical to TypeScript', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const invocations = 64;
  const blockWords = invocations * 4;
  const drawWords = invocations * BOUNDS.length * DRAWS;
  const feistelWords = FEISTEL_COUNTS.length * 4097 + FEISTEL_COUNTS.length;
  const blocks = createOutputBuffer(device, blockWords);
  const draws = createOutputBuffer(device, drawWords);
  const feistel = createOutputBuffer(device, feistelWords);
  const graph = new GPUCommandGraph(device, {id: 'philox-test'});
  const blocksView = importGraphBuffer(graph, 'blocks', blocks, 'uint32', blockWords);
  const drawsView = importGraphBuffer(graph, 'draws', draws, 'uint32', drawWords);
  const feistelView = importGraphBuffer(graph, 'feistel', feistel, 'uint32', feistelWords);
  graph.add({
    getCommandNodes: target => [
      createMapGraphKernelNode(target, {
        id: 'philox-test-kernel',
        operation: 'PhiloxTest',
        bindings: [
          {name: 'blocks', view: blocksView, type: 'u32', access: 'read_write'},
          {name: 'draws', view: drawsView, type: 'u32', access: 'read_write'},
          {name: 'feistel', view: feistelView, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 4097,
        declarations: `${PERMUTATION_RANDOM_WGSL}
const BOUNDS = array<u32, ${BOUNDS.length}>(${BOUNDS.map(bound => `${bound}u`).join(', ')});
const FEISTEL_COUNTS = array<u32, ${FEISTEL_COUNTS.length}>(${FEISTEL_COUNTS.map(count => `${count}u`).join(', ')});`,
        body: `if (index < ${invocations}u) {
    let block = getPhilox4x32(vec4<u32>(index, index * 7u, 0xdeadbeefu, index ^ 0x5555u), vec2<u32>(index * 13u, 99u));
    for (var lane = 0u; lane < 4u; lane++) {
      blocks[blocksOffset + index * 4u + lane] = block[lane];
    }
    for (var bound = 0u; bound < ${BOUNDS.length}u; bound++) {
      var stream = createPhiloxStream(vec2<u32>(17u, index), index, bound, 3u);
      for (var draw = 0u; draw < ${DRAWS}u; draw++) {
        draws[drawsOffset + (index * ${BOUNDS.length}u + bound) * ${DRAWS}u + draw] = nextPhiloxBelow(&stream, BOUNDS[bound]);
      }
    }
  }
  for (var countIndex = 0u; countIndex < ${FEISTEL_COUNTS.length}u; countIndex++) {
    let count = FEISTEL_COUNTS[countIndex];
    let halfBits = getFeistelHalfBits(count);
    if (index == 0u) {
      feistel[feistelOffset + ${FEISTEL_COUNTS.length * 4097}u + countIndex] = halfBits;
    }
    if (index < count) {
      let keys = getFeistelRoundKeys(vec2<u32>(123u, 4u), countIndex + 1u);
      feistel[feistelOffset + countIndex * 4097u + index] = getFeistelPermutationIndex(index, count, halfBits, keys);
    }
  }`
      })
    ]
  });
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const gpuBlocks = await readUint32(blocks, blockWords);
  const gpuDraws = await readUint32(draws, drawWords);
  const gpuFeistel = await readUint32(feistel, feistelWords);
  for (let index = 0; index < invocations; index++) {
    const expected = getPhilox4x32(
      [index, index * 7, 0xdeadbeef, (index ^ 0x5555) >>> 0],
      [(index * 13) >>> 0, 99]
    );
    expect(gpuBlocks.slice(index * 4, index * 4 + 4), `block ${index}`).toEqual(expected);
    for (const [boundIndex, bound] of BOUNDS.entries()) {
      const stream = new PhiloxStream([17, index], index, boundIndex, 3);
      const expectedDraws = Array.from({length: DRAWS}, () => stream.nextBelow(bound));
      const offset = (index * BOUNDS.length + boundIndex) * DRAWS;
      expect(gpuDraws.slice(offset, offset + DRAWS), `draws ${index} ${bound}`).toEqual(
        expectedDraws
      );
    }
  }
  for (const [countIndex, count] of FEISTEL_COUNTS.entries()) {
    expect(gpuFeistel[FEISTEL_COUNTS.length * 4097 + countIndex]).toBe(getFeistelHalfBits(count));
    const keys = getFeistelRoundKeys([123, 4], countIndex + 1);
    const expected = Array.from({length: count}, (_, index) =>
      getFeistelPermutationIndex(index, count, keys)
    );
    expect(
      gpuFeistel.slice(countIndex * 4097, countIndex * 4097 + count),
      `feistel ${count}`
    ).toEqual(expected);
  }
  compiled.destroy();
  blocks.destroy();
  draws.destroy();
  feistel.destroy();
});
