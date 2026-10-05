// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUIsobandsParameterValues,
  getGPUIsolinesParameterValues,
  GPU_ISOBANDS_PARAMETER_LENGTH,
  GPU_ISOLINES_PARAMETER_LENGTH,
  GPUIsobands,
  GPUIsolines
} from '../../../src/map-graphs/isolines';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {createRandom} from '../raster-algebra/raster-algebra-test-utils';

const WIDTH = 23;
const HEIGHT = 19;
const LEVELS = [-0.5, 0.25, 1, 1.75];
const SEGMENT_CAPACITY = 8192;
const TRIANGLE_CAPACITY = 16384;

/** Bit pattern key of an f32 point, so the comparison is exact. */
function getPointKey(x: number, y: number): string {
  const bits = new Uint32Array(new Float32Array([x, y]).buffer);
  return `${bits[0]}:${bits[1]}`;
}

it('GPUIsobands boundaries coincide bit for bit with GPUIsolines segment endpoints', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(77);
  const sampleCount = WIDTH * HEIGHT;
  const values = Float32Array.from({length: sampleCount}, (_, sample) => {
    const column = sample % WIDTH;
    const row = Math.floor(sample / WIDTH);
    if (random() < 0.03) {
      return NaN;
    }
    // Smooth waves plus noise: many saddles, some samples exactly on levels.
    const wave = Math.sin(column * 0.7) + Math.cos(row * 0.9) + (random() - 0.5) * 0.6;
    return random() < 0.1 ? LEVELS[Math.floor(random() * LEVELS.length)] : Math.fround(wave);
  });
  const extent = [100.5, -20.25, 146.5, 17.75] as const;
  const valuesBuffer = createInputBuffer(device, values);
  const levelsBuffer = createInputBuffer(device, Float32Array.from(LEVELS));
  const isolineParameters = new GPUMapGraphParameterBuffer(device, {
    id: 'consistency-isolines',
    format: 'float32',
    length: GPU_ISOLINES_PARAMETER_LENGTH,
    values: getGPUIsolinesParameterValues({
      width: WIDTH,
      height: HEIGHT,
      levelCount: LEVELS.length,
      extent
    })
  });
  const isobandParameters = new GPUMapGraphParameterBuffer(device, {
    id: 'consistency-isobands',
    format: 'float32',
    length: GPU_ISOBANDS_PARAMETER_LENGTH,
    values: getGPUIsobandsParameterValues({
      width: WIDTH,
      height: HEIGHT,
      breakCount: LEVELS.length,
      extent
    })
  });
  const segmentsBuffer = createOutputBuffer(device, SEGMENT_CAPACITY * 4);
  const segmentLevelsBuffer = createOutputBuffer(device, SEGMENT_CAPACITY);
  const segmentCountBuffer = createOutputBuffer(device, 1);
  const segmentOverflowBuffer = createOutputBuffer(device, 1);
  const trianglesBuffer = createOutputBuffer(device, TRIANGLE_CAPACITY * 6);
  const triangleBandsBuffer = createOutputBuffer(device, TRIANGLE_CAPACITY);
  const triangleCountBuffer = createOutputBuffer(device, 1);
  const triangleOverflowBuffer = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device, {id: 'consistency-graph'});
  graph.add(
    new GPUIsolines({
      id: 'lines',
      width: WIDTH,
      height: HEIGHT,
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', sampleCount),
      levels: importGraphBuffer(graph, 'levels', levelsBuffer, 'float32', LEVELS.length),
      parameters: isolineParameters.importToGraph(graph),
      output: {
        segments: importGraphBuffer(
          graph,
          'segments',
          segmentsBuffer,
          'float32x4',
          SEGMENT_CAPACITY
        ),
        segmentLevels: importGraphBuffer(
          graph,
          'segment-levels',
          segmentLevelsBuffer,
          'uint32',
          SEGMENT_CAPACITY
        ),
        count: importGraphBuffer(graph, 'segment-count', segmentCountBuffer, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'segment-overflow', segmentOverflowBuffer, 'uint32', 1)
      }
    })
  );
  graph.add(
    new GPUIsobands({
      id: 'bands',
      width: WIDTH,
      height: HEIGHT,
      values: importGraphBuffer(graph, 'values-again', valuesBuffer, 'float32', sampleCount),
      breaks: importGraphBuffer(graph, 'breaks', levelsBuffer, 'float32', LEVELS.length),
      parameters: isobandParameters.importToGraph(graph),
      output: {
        triangles: importGraphBuffer(
          graph,
          'triangles',
          trianglesBuffer,
          'float32x2',
          TRIANGLE_CAPACITY * 3
        ),
        triangleBands: importGraphBuffer(
          graph,
          'triangle-bands',
          triangleBandsBuffer,
          'uint32',
          TRIANGLE_CAPACITY
        ),
        count: importGraphBuffer(graph, 'triangle-count', triangleCountBuffer, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'triangle-overflow', triangleOverflowBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [segmentCount] = await readUint32(segmentCountBuffer, 1);
  const [triangleCount] = await readUint32(triangleCountBuffer, 1);
  expect(await readUint32(segmentOverflowBuffer, 1)).toEqual([0]);
  expect(await readUint32(triangleOverflowBuffer, 1)).toEqual([0]);
  expect(segmentCount).toBeGreaterThan(100);
  const segments = await readFloat32(segmentsBuffer, segmentCount * 4);
  const segmentLevels = await readUint32(segmentLevelsBuffer, segmentCount);
  const triangles = await readFloat32(trianglesBuffer, triangleCount * 6);
  const triangleBands = await readUint32(triangleBandsBuffer, triangleCount);

  // Every vertex of a band fragment adjacent to level k (band k or k + 1) is either a sample
  // corner or a level crossing; every level-k segment endpoint must be such a vertex, bit for bit.
  const bandVertices = new Map<number, Set<string>>();
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const band = triangleBands[triangle];
    const keys = bandVertices.get(band) ?? new Set<string>();
    for (let corner = 0; corner < 3; corner++) {
      const base = triangle * 6 + corner * 2;
      keys.add(getPointKey(triangles[base], triangles[base + 1]));
    }
    bandVertices.set(band, keys);
  }
  let checked = 0;
  for (let segment = 0; segment < segmentCount; segment++) {
    const level = segmentLevels[segment];
    for (const endpoint of [0, 2]) {
      const key = getPointKey(
        segments[segment * 4 + endpoint],
        segments[segment * 4 + endpoint + 1]
      );
      // Level k separates band k (below) from band k + 1 (above): both share the vertex.
      expect(bandVertices.get(level)?.has(key), `segment ${segment} level ${level} below`).toBe(
        true
      );
      expect(bandVertices.get(level + 1)?.has(key), `segment ${segment} level ${level} above`).toBe(
        true
      );
      checked++;
    }
  }
  expect(checked).toBe(segmentCount * 2);
  compiled.destroy();
  isolineParameters.destroy();
  isobandParameters.destroy();
  for (const buffer of [
    valuesBuffer,
    levelsBuffer,
    segmentsBuffer,
    segmentLevelsBuffer,
    segmentCountBuffer,
    segmentOverflowBuffer,
    trianglesBuffer,
    triangleBandsBuffer,
    triangleCountBuffer,
    triangleOverflowBuffer
  ]) {
    buffer.destroy();
  }
});
