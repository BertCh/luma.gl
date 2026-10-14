// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createSegmentSumNode} from '../../src/utils/sorted-segment-sums';
import {createInputBuffer, createOutputBuffer, submitGraph} from './gpu-contributor-test-utils';

const WORKGROUP_SIZE = 256;

/** Small deterministic generator so failures reproduce. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Reference with the documented order: lane-strided partials, then the stride 128..1 tree. */
function sumSegmentsOnCpu(values: Float32Array, offsets: Uint32Array): Float32Array {
  const output = new Float32Array(offsets.length - 1);
  const scratch = new Float32Array(WORKGROUP_SIZE);
  for (let segment = 0; segment < output.length; segment++) {
    const end = Math.min(offsets[segment + 1], values.length);
    for (let lane = 0; lane < WORKGROUP_SIZE; lane++) {
      let partial = 0;
      for (let row = offsets[segment] + lane; row < end; row += WORKGROUP_SIZE) {
        partial = Math.fround(partial + values[row]);
      }
      scratch[lane] = partial;
    }
    for (let stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
      for (let lane = 0; lane < stride; lane++) {
        scratch[lane] = Math.fround(scratch[lane] + scratch[lane + stride]);
      }
    }
    output[segment] = scratch[0];
  }
  return output;
}

function createCase(segmentCount: number, seed: number) {
  const random = createRandom(seed);
  const offsets = new Uint32Array(segmentCount + 1);
  const lengths = new Uint32Array(segmentCount);
  for (let segment = 0; segment < segmentCount; segment++) {
    const draw = random();
    // About 25% empty segments, a few mid-sized, one large segment in the middle.
    lengths[segment] = draw < 0.25 ? 0 : draw < 0.95 ? Math.floor(random() * 12) + 1 : 700;
  }
  lengths[Math.floor(segmentCount / 2)] = 50000;
  for (let segment = 0; segment < segmentCount; segment++) {
    offsets[segment + 1] = offsets[segment] + lengths[segment];
  }
  const values = new Float32Array(offsets[segmentCount]);
  for (let row = 0; row < values.length; row++) {
    values[row] = (random() - 0.5) * 2 ** Math.floor(random() * 20 - 8);
  }
  // NaN in a populated segment propagates to its sum.
  const nanSegment = Math.floor(segmentCount / 3);
  if (lengths[nanSegment] > 0) {
    values[offsets[nanSegment]] = Number.NaN;
  } else {
    lengths[nanSegment] = 0;
  }
  return {offsets, values, nanSegment, nanPopulated: lengths[nanSegment] > 0};
}

async function runImplementation(
  device: Device,
  testCase: ReturnType<typeof createCase>,
  implementation: 'auto' | 'kernel',
  offsetsWithinInput: boolean
): Promise<{bits: Uint32Array; nodeCount: number; milliseconds: number}> {
  const {offsets, values} = testCase;
  const segmentCount = offsets.length - 1;
  const graph = new GPUCommandGraph(device, {id: `segment-sum-${implementation}`});
  const inputBuffer = createInputBuffer(device, values.length ? values : new Float32Array(1));
  const offsetBuffer = createInputBuffer(device, offsets);
  const outputBuffer = createOutputBuffer(device, segmentCount);
  const node = createSegmentSumNode(graph, {
    id: 'sum',
    operation: 'SegmentSumParity',
    segmentCount,
    input: importGraphBuffer(graph, 'input', inputBuffer, 'float32', values.length),
    segmentOffsets: importGraphBuffer(graph, 'offsets', offsetBuffer, 'uint32', offsets.length),
    output: importGraphBuffer(graph, 'output', outputBuffer, 'float32', segmentCount),
    offsetsWithinInput,
    implementation
  });
  graph.add(node);
  const compiled = graph.compile();
  const start = performance.now();
  submitGraph(device, compiled, undefined);
  const bytes = await outputBuffer.readAsync();
  const milliseconds = performance.now() - start;
  const bits = new Uint32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + segmentCount * 4)
  );
  compiled.destroy();
  inputBuffer.destroy();
  offsetBuffer.destroy();
  outputBuffer.destroy();
  return {bits, nodeCount: 1, milliseconds};
}

function expectSameBits(actual: Uint32Array, expected: Float32Array): void {
  const expectedBits = new Uint32Array(expected.buffer);
  const actualFloats = new Float32Array(actual.buffer);
  let mismatches = 0;
  for (let index = 0; index < expected.length; index++) {
    const bothNaN = Number.isNaN(actualFloats[index]) && Number.isNaN(expected[index]);
    if (!bothNaN && actual[index] !== expectedBits[index]) mismatches++;
  }
  expect(mismatches, 'segments whose f32 bits differ').toBe(0);
}

it('createSegmentSumNode: upstream path is bit-identical to the local kernel and the CPU order', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const testCase = createCase(3000, 7);
  const upstream = await runImplementation(device, testCase, 'auto', true);
  const kernel = await runImplementation(device, testCase, 'kernel', true);
  expectSameBits(upstream.bits, new Float32Array(kernel.bits.buffer));
  expectSameBits(kernel.bits, sumSegmentsOnCpu(testCase.values, testCase.offsets));
  const sums = new Float32Array(upstream.bits.buffer);
  if (testCase.nanPopulated) expect(Number.isNaN(sums[testCase.nanSegment])).toBe(true);
  // Empty segments are exactly +0 on both paths.
  for (let segment = 0; segment < sums.length; segment++) {
    if (testCase.offsets[segment] === testCase.offsets[segment + 1]) {
      expect(upstream.bits[segment]).toBe(0);
      expect(kernel.bits[segment]).toBe(0);
    }
  }
  console.log(
    `segment-sum 3000 segments: upstream ${upstream.milliseconds.toFixed(1)} ms, kernel ${kernel.milliseconds.toFixed(1)} ms`
  );
});

it('createSegmentSumNode: above 65535 segments falls back to the local kernel and stays correct', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const testCase = createCase(70000, 11);
  // 'auto' with offsetsWithinInput would exceed upstream's x-dimension limit, so it must fall back.
  const automatic = await runImplementation(device, testCase, 'auto', true);
  const kernel = await runImplementation(device, testCase, 'kernel', true);
  expectSameBits(automatic.bits, new Float32Array(kernel.bits.buffer));
  expectSameBits(kernel.bits, sumSegmentsOnCpu(testCase.values, testCase.offsets));
  console.log(
    `segment-sum 70000 segments: auto ${automatic.milliseconds.toFixed(1)} ms, kernel ${kernel.milliseconds.toFixed(1)} ms`
  );
});

it('createSegmentSumNode: offsets beyond the input stay clamped when not declared within the input', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const base = createCase(50, 3);
  // Last segment end runs past the input, as an excluded-key offset can.
  const offsets = Uint32Array.from(base.offsets);
  offsets[offsets.length - 1] += 1000;
  const testCase = {...base, offsets};
  const automatic = await runImplementation(device, testCase, 'auto', false);
  expectSameBits(automatic.bits, sumSegmentsOnCpu(base.values, offsets));
});
