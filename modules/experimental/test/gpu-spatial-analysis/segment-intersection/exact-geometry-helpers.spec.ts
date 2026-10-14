// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {EXACT_GEOMETRY_WGSL} from '../../../src/utils/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../../src/utils/exact-orientation-wgsl';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createBufferGraph} from '../polygon-buffer/polygon-buffer-test-utils';
import {
  createPointProbe,
  createSideProbe,
  getExactProbeCrossing,
  type ExactProbe,
  type Point
} from './exact-geometry-reference';
import {requireTestDevice} from '../../utils/require-test-device';

// The WGSL helpers of utils/exact-geometry-wgsl.ts against their CPU reference: every case is an
// edge c -> d and a probe (point, or left/right of a segment); integer data hits every tie.

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

it('exactProbeCrossing matches the CPU reference on integer ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(5);
  const integer = () => Math.floor(random() * 7);
  const cases: {c: Point; d: Point; probe: ExactProbe}[] = [];
  for (let index = 0; index < 4096; index++) {
    const c: Point = [integer(), integer()];
    const d: Point = [integer(), integer()];
    const start: Point = [integer(), integer()];
    const end: Point = [integer(), integer()];
    const kind = index % 3;
    const probe =
      kind === 0 ? createPointProbe(start) : createSideProbe(start, end, kind === 1 ? 1 : -1);
    cases.push({c, d, probe});
  }
  for (const value of [NaN, Infinity, -Infinity]) {
    cases.push({c: [2, value], d: [2, 2], probe: createPointProbe([0, 1])});
  }
  // Per case: c, d, point, start, end (10 floats) and the side.
  const geometry = new Float32Array(cases.length * 10);
  const sides = new Uint32Array(cases.length);
  cases.forEach(({c, d, probe}, index) => {
    geometry.set([...c, ...d, ...probe.point, ...probe.start, ...probe.end], index * 10);
    sides[index] = probe.side + 1;
  });
  const graph = new GPUCommandGraph(device, {id: 'exact-geometry-helpers'});
  const buffers = createBufferGraph(device, graph);
  const geometryView = buffers.input('geometry', geometry, 'float32', geometry.length);
  const sidesView = buffers.input('sides', sides, 'uint32', sides.length);
  const result = buffers.output('result', 'uint32', cases.length);
  graph.add({
    getCommandNodes: commandGraph => [
      createWGSLKernelNode(commandGraph, {
        id: 'exact-probe-crossing',
        operation: 'ExactGeometryHelpers',
        variant: 'probe-crossing',
        bindings: [
          {name: 'geometry', view: geometryView, type: 'f32', access: 'read'},
          {name: 'sides', view: sidesView, type: 'u32', access: 'read'},
          {name: 'result', view: result.view, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cases.length,
        declarations: `${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
const CASE_COUNT: u32 = ${cases.length}u;
fn readPoint(base: u32) -> vec2f { return vec2f(geometry[geometryOffset + base], geometry[geometryOffset + base + 1u]); }`,
        body: `if (index >= CASE_COUNT) { return; }
  let base = index * 10u;
  let probe = ExactProbe(readPoint(base + 4u), readPoint(base + 6u), readPoint(base + 8u), i32(sides[sidesOffset + index]) - 1);
  let crossing = exactProbeCrossing(probe, readPoint(base), readPoint(base + 2u));
  result[resultOffset + index] = u32(crossing.x + 1) | (u32(crossing.y) << 4u);`
      })
    ]
  });
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const values = await readUint32(result.buffer, cases.length);
  compiled.destroy();
  for (const buffer of buffers.buffers) {
    buffer.destroy();
  }
  const mismatches: string[] = [];
  cases.forEach(({c, d, probe}, index) => {
    const expected = getExactProbeCrossing(probe, c, d);
    const crossing = (values[index] & 15) - 1;
    const boundary = ((values[index] >> 4) & 1) === 1;
    if (![...c, ...d].every(Number.isFinite)) {
      expect((values[index] >> 4) & 2, 'non-finite uncertainty flag').toBe(2);
    }
    if (crossing !== expected.crossing || boundary !== expected.boundary) {
      mismatches.push(
        `${JSON.stringify({c, d, probe})} gpu ${crossing}/${boundary} cpu ${expected.crossing}/${expected.boundary}`
      );
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
});
