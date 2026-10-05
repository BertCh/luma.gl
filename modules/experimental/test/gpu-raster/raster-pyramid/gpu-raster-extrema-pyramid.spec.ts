// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createRasterExtremaPyramidNodes,
  getGPURasterExtremaPyramidLayout,
  getRasterExtremaPyramidWGSL,
  GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MAXIMUM as EMPTY_MAXIMUM,
  GPU_RASTER_EXTREMA_PYRAMID_EMPTY_MINIMUM as EMPTY_MINIMUM,
  GPURasterExtremaPyramid,
  type GPURasterExtremaPyramidFootprint
} from '../../../src/gpu-raster/raster-pyramid';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeRasterExtremaPyramid} from './raster-pyramid-oracle';

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** Values with negatives, small fractions and large magnitudes, plus exact zeros. */
function createValues(width: number, height: number, seed: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: width * height}, () => {
    const kind = random();
    if (kind < 0.05) return 0;
    if (kind < 0.15) return (random() - 0.5) * 3e7;
    return (random() - 0.5) * 2000;
  });
}

/** Marks the top-left `size x size` block (an all-invalid block for every tested block size) plus noise. */
function createValidity(width: number, height: number, size: number, seed: number): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length: width * height}, (_, pixel) => {
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    return x < size && y < size ? 0 : random() < 0.1 ? 0 : 1;
  });
}

function bits(values: Float32Array): number[] {
  return Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length));
}

async function runPyramid(
  device: Device,
  options: {
    width: number;
    height: number;
    firstBlockSize: number;
    footprint: GPURasterExtremaPyramidFootprint;
    values: Float32Array | Uint32Array;
    validity?: Uint32Array;
    scale?: number;
    offset?: number;
    maximum?: boolean;
    minimum?: boolean;
  }
) {
  const {width, height} = options;
  const pixelCount = width * height;
  const layout = getGPURasterExtremaPyramidLayout(width, height, options);
  const inputBuffer = createInputBuffer(device, options.values);
  const validityBuffer = options.validity ? createInputBuffer(device, options.validity) : undefined;
  const maximumBuffer = createOutputBuffer(device, layout.length);
  const minimumBuffer = createOutputBuffer(device, layout.length);
  const graph = new GPUCommandGraph(device, {id: 'raster-pyramid-test'});
  graph.add(
    new GPURasterExtremaPyramid({
      width,
      height,
      firstBlockSize: options.firstBlockSize,
      footprint: options.footprint,
      input: {
        id: 'input',
        format: options.values instanceof Uint32Array ? 'uint32' : 'float32',
        scale: options.scale,
        offset: options.offset,
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(
            graph,
            'input',
            inputBuffer,
            options.values instanceof Uint32Array ? 'uint32' : 'float32',
            pixelCount
          )
        },
        validity: validityBuffer
          ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
          : undefined
      } as never,
      maximum:
        options.maximum === false
          ? undefined
          : importGraphBuffer(graph, 'maximum', maximumBuffer, 'float32', layout.length),
      minimum:
        options.minimum === false
          ? undefined
          : importGraphBuffer(graph, 'minimum', minimumBuffer, 'float32', layout.length)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    layout,
    maximum: await readUint32(maximumBuffer, layout.length),
    minimum: await readUint32(minimumBuffer, layout.length)
  };
  compiled.destroy();
  for (const buffer of [inputBuffer, validityBuffer, maximumBuffer, minimumBuffer]) {
    buffer?.destroy();
  }
  return result;
}

const CASES: {width: number; height: number}[] = [
  {width: 37, height: 23},
  {width: 64, height: 64}
];

for (const {width, height} of CASES) {
  for (const firstBlockSize of [1, 2, 4]) {
    for (const footprint of ['cell', 'bilinear'] as const) {
      it(`GPURasterExtremaPyramid is bit-identical to the oracle (${width}x${height}, block ${firstBlockSize}, ${footprint})`, async () => {
        const device = await getWebGPUTestDevice();
        if (!device) {
          return;
        }
        const values = createValues(width, height, 7 + firstBlockSize);
        const validity = createValidity(width, height, 10, 99);
        const result = await runPyramid(device, {
          width,
          height,
          firstBlockSize,
          footprint,
          values,
          validity
        });
        const expected = computeRasterExtremaPyramid(values, validity, result.layout);
        // A WGSL compile failure leaves zeros: assert structure the oracle also has.
        const emptyMaximumBits = bits(Float32Array.of(EMPTY_MAXIMUM))[0];
        expect(result.maximum.filter(bit => bit === emptyMaximumBits).length).toBeGreaterThan(0);
        expect(result.maximum.filter(bit => bit !== emptyMaximumBits).length).toBeGreaterThan(0);
        expect(new Set(result.maximum).size).toBeGreaterThan(10);
        expect(result.maximum).toEqual(bits(expected.maximum));
        expect(result.minimum).toEqual(bits(expected.minimum));
        // The top cell holds the global extremum of valid pixels.
        const last = result.layout.length - 1;
        const valid = Array.from(values).filter((_, pixel) => validity[pixel] !== 0);
        expect(new Float32Array(Uint32Array.of(result.maximum[last]).buffer)[0]).toBe(
          Math.max(...valid)
        );
        expect(new Float32Array(Uint32Array.of(result.minimum[last]).buffer)[0]).toBe(
          Math.min(...valid)
        );
      });
    }
  }
}

it('GPURasterExtremaPyramid canonicalizes uint32 calibration and works without validity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 23;
  const raw = Uint32Array.from({length: width * height}, (_, index) => (index * 7919) % 4001);
  const calibrated = Float32Array.from(raw, value => value * 0.5 - 1000);
  const result = await runPyramid(device, {
    width,
    height,
    firstBlockSize: 2,
    footprint: 'bilinear',
    values: raw,
    scale: 0.5,
    offset: -1000
  });
  const expected = computeRasterExtremaPyramid(calibrated, undefined, result.layout);
  expect(new Set(result.minimum).size).toBeGreaterThan(10);
  expect(result.maximum).toEqual(bits(expected.maximum));
  expect(result.minimum).toEqual(bits(expected.minimum));
});

it('GPURasterExtremaPyramid builds only the requested extremum and respects levelCount', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 23;
  const values = createValues(width, height, 3);
  const onlyMaximum = await runPyramid(device, {
    width,
    height,
    firstBlockSize: 4,
    footprint: 'bilinear',
    values,
    minimum: false
  });
  const expected = computeRasterExtremaPyramid(values, undefined, onlyMaximum.layout);
  expect(onlyMaximum.maximum).toEqual(bits(expected.maximum));
  expect(new Set(onlyMaximum.minimum)).toEqual(new Set([0]));
  const onlyMinimum = await runPyramid(device, {
    width,
    height,
    firstBlockSize: 4,
    footprint: 'bilinear',
    values,
    maximum: false
  });
  expect(onlyMinimum.minimum).toEqual(bits(expected.minimum));
  expect(new Set(onlyMinimum.maximum)).toEqual(new Set([0]));
});

it('combined output plus getRasterExtremaPyramidWGSL index the same cells', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 23;
  const values = createValues(width, height, 21);
  const validity = createValidity(width, height, 10, 5);
  const layout = getGPURasterExtremaPyramidLayout(width, height, {firstBlockSize: 2});
  const probeLevel = 2;
  const valuesBuffer = createInputBuffer(device, values);
  const validityBuffer = createInputBuffer(device, validity);
  const probeBuffer = createOutputBuffer(device, 3 * width * height);
  const graph = new GPUCommandGraph(device, {id: 'raster-pyramid-combined'});
  const combined = createTransientView(graph, 'combined', 'float32', 2 * layout.length);
  const probe = importGraphBuffer(graph, 'probe', probeBuffer, 'float32', 3 * width * height);
  for (const node of createRasterExtremaPyramidNodes(graph, {
    id: 'combined-pyramid',
    layout,
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', width * height),
    validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', width * height),
    combined
  })) {
    graph.add(node);
  }
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'probe',
      operation: 'RasterPyramidProbe',
      bindings: [
        {name: 'pyramid', view: combined, type: 'f32', access: 'read'},
        {name: 'probe', view: probe, type: 'f32', access: 'read_write'}
      ],
      invocationCount: width * height,
      declarations: `${getRasterExtremaPyramidWGSL(layout, 'testPyramid')}`,
      body: `let column = index % ${width}u;
  let row = index / ${width}u;
  let cell = testPyramidLevelIndex(${probeLevel}u, column, row);
  probe[probeOffset + 3u * index] = pyramid[pyramidOffset + cell];
  probe[probeOffset + 3u * index + 1u] = pyramid[pyramidOffset + TEST_PYRAMID_MINIMUM_OFFSET + cell];
  probe[probeOffset + 3u * index + 2u] = f32(testPyramidLevelBlockSize(${probeLevel}u));`
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const output = await readUint32(probeBuffer, 3 * width * height);
  const expected = computeRasterExtremaPyramid(values, validity, layout);
  const level = layout.levels[probeLevel];
  const expectedProbe = new Float32Array(3 * width * height);
  for (let index = 0; index < width * height; index++) {
    const cell =
      level.offset +
      Math.floor(Math.floor(index / width) / level.blockSize) * level.width +
      Math.floor((index % width) / level.blockSize);
    expectedProbe[3 * index] = expected.maximum[cell];
    expectedProbe[3 * index + 1] = expected.minimum[cell];
    expectedProbe[3 * index + 2] = level.blockSize;
  }
  expect(output.filter((_, i) => i % 3 === 2)[0]).toBe(bits(Float32Array.of(8))[0]);
  expect(output).toEqual(bits(expectedProbe));
  expect(EMPTY_MINIMUM).toBeGreaterThan(0);
  compiled.destroy();
  for (const buffer of [valuesBuffer, validityBuffer, probeBuffer]) {
    buffer.destroy();
  }
});
