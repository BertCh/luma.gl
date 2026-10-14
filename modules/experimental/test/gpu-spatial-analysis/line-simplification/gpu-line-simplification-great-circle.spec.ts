// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GREAT_CIRCLE_LINE_CASES} from './line-simplification-great-circle-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPULineSimplification great-circle importance matches numpy/pyproj in meters at high latitude and across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = Object.values(GREAT_CIRCLE_LINE_CASES);
  const positions = Float32Array.from(cases.flatMap(testCase => testCase.line.flat()));
  const lineOffsets = Uint32Array.of(
    0,
    ...cases.map((_, index) => cases.slice(0, index + 1).reduce((sum, c) => sum + c.line.length, 0))
  );
  const rowCount = positions.length / 2;
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, positions);
  const offsetsBuffer = createInputBuffer(device, lineOffsets);
  const importanceBuffer = createOutputBuffer(device, rowCount);
  buffers.push(positionsBuffer, offsetsBuffer, importanceBuffer);
  const graph = new GPUCommandGraph(device, {id: 'line-great-circle'});
  graph.add(
    new GPULineSimplification({
      id: 'simplify',
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
      lineOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', lineOffsets.length),
      importance: importGraphBuffer(graph, 'importance', importanceBuffer, 'float32', rowCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const importance = await readFloat32(importanceBuffer, rowCount);
  let row = 0;
  for (const [name, testCase] of Object.entries(GREAT_CIRCLE_LINE_CASES)) {
    for (const expected of testCase.importance) {
      const actual = importance[row++];
      if (expected === Infinity) {
        expect(actual, `${name} endpoint`).toBe(Infinity);
      } else {
        // f32 haversine and the first-order segment fraction: a few parts in a thousand.
        expect(Math.abs(actual - expected), `${name} row ${row - 1}`).toBeLessThanOrEqual(
          2e-3 * expected + 1
        );
      }
    }
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  device.destroy?.();
});
