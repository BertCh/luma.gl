// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {SEGMENT_PREDICATES_WGSL} from '../../../src/gpu-spatial-analysis/segment-intersection/segment-intersection-wgsl';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, orientExact, type OraclePoint} from './segment-intersection-oracle';

it('orientSign matches exact arithmetic on nearly degenerate triples', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(3);
  const triples: [OraclePoint, OraclePoint, OraclePoint][] = [];
  const round = (x: number, y: number): OraclePoint => [Math.fround(x), Math.fround(y)];
  for (let row = 0; row < 3000; row++) {
    const a = round(random() * 100, random() * 100);
    const b = round(a[0] + (random() - 0.5) * 60, a[1] + (random() - 0.5) * 60);
    const mode = row % 4;
    let c: OraclePoint;
    if (mode === 0) {
      const t = random();
      c = round(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
    } else if (mode === 1) {
      // Exactly collinear dyadic points.
      const t = Math.floor(random() * 16) / 8;
      c = round(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t);
    } else if (mode === 2) {
      // One ulp off the midpoint along a coordinate.
      const mid = round((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      c = [mid[0] + (random() < 0.5 ? 1 : -1) * Math.abs(mid[0]) * 2 ** -23, mid[1]];
      c = round(c[0], c[1]);
    } else {
      c = round(random() * 100, random() * 100);
    }
    triples.push([a, b, c]);
  }
  const data = Float32Array.from(triples.flatMap(([a, b, c]) => [...a, ...b, ...c]));
  const graph = new GPUCommandGraph(device, {id: 'orient'});
  const buffers: Buffer[] = [];
  const input = createInputBuffer(device, data);
  const output = createOutputBuffer(device, triples.length);
  buffers.push(input, output);
  const inputView = importGraphBuffer(graph, 'input', input, 'float32', data.length);
  const outputView = importGraphBuffer(graph, 'output', output, 'uint32', triples.length);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'orient',
      operation: 'orient-test',
      bindings: [
        {name: 'input', view: inputView, type: 'f32', access: 'read'},
        {name: 'output', view: outputView, type: 'u32', access: 'read_write'}
      ],
      invocationCount: triples.length,
      declarations: SEGMENT_PREDICATES_WGSL,
      body: `let base = inputOffset + index * 6u;
  let sign = orientSign(vec2f(input[base], input[base + 1u]), vec2f(input[base + 2u], input[base + 3u]), vec2f(input[base + 4u], input[base + 5u]));
  output[outputOffset + index] = u32(sign + 1);`
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const signs = (await readUint32(output, triples.length)).map(value => value - 1);
  let zeros = 0;
  let wrong = 0;
  triples.forEach(([a, b, c], row) => {
    const exact = orientExact(a, b, c);
    zeros += exact === 0 ? 1 : 0;
    if (signs[row] !== exact) {
      wrong++;
      if (wrong < 5) {
        console.log(
          'orient mismatch',
          JSON.stringify([a, b, c]),
          'gpu',
          signs[row],
          'exact',
          exact
        );
      }
    }
  });
  expect(zeros, 'scene contains exact zeros').toBeGreaterThan(100);
  expect(signs.filter(sign => sign !== 0).length, 'scene contains nonzero signs').toBeGreaterThan(
    1000
  );
  expect(wrong).toBe(0);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
