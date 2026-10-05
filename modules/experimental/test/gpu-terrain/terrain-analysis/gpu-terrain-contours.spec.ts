// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTerrainContours} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {countContourSegments, sampleBilinear} from './terrain-analysis-oracle';

const SIZE = 5;
const PIXEL_COUNT = SIZE * SIZE;
const RAMP = Float32Array.from(
  {length: PIXEL_COUNT},
  (_, index) => (index % SIZE) + Math.floor(index / SIZE)
);

async function expectVerticesOnLevel(
  vertices: Buffer,
  count: number,
  level: number
): Promise<void> {
  const values = await readFloat32(vertices, Math.max(count * 4, 1));
  for (let vertex = 0; vertex < count * 2; vertex++) {
    const x = values[vertex * 2] - 0.5;
    const y = values[vertex * 2 + 1] - 0.5;
    expect(Math.abs(sampleBilinear(RAMP, SIZE, SIZE, x, y) - level)).toBeLessThan(1e-4);
  }
}

it('GPUTerrainContours extracts numeric and per-frame GPU levels with one overflow flag', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevationBuffer = createInputBuffer(device, RAMP);
  const levelsParameter = new GPUParameterBuffer(device, {
    id: 'levels',
    format: 'float32',
    length: 2,
    values: Float32Array.from([0, 5.5])
  });
  const buffers = {
    verticesA: createOutputBuffer(device, 128),
    countA: createOutputBuffer(device, 1),
    requiredA: createOutputBuffer(device, 1),
    verticesB: createOutputBuffer(device, 128),
    countB: createOutputBuffer(device, 1),
    requiredB: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'terrain-contours-test'});
  const levelsView = levelsParameter.importToGraph(graph);
  graph.add(
    new GPUTerrainContours({
      width: SIZE,
      height: SIZE,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      levels: [
        {
          level: 3.5,
          vertices: importGraphBuffer(graph, 'vertices-a', buffers.verticesA, 'float32x2', 64),
          segmentCount: importGraphBuffer(graph, 'count-a', buffers.countA, 'uint32', 1),
          requiredSegmentCount: importGraphBuffer(
            graph,
            'required-a',
            buffers.requiredA,
            'uint32',
            1
          )
        },
        {
          level: graph.createDataView(levelsView.buffer, {
            format: 'float32',
            length: 1,
            byteOffset: 4
          }),
          vertices: importGraphBuffer(graph, 'vertices-b', buffers.verticesB, 'float32x2', 64),
          segmentCount: importGraphBuffer(graph, 'count-b', buffers.countB, 'uint32', 1),
          requiredSegmentCount: importGraphBuffer(
            graph,
            'required-b',
            buffers.requiredB,
            'uint32',
            1
          )
        }
      ],
      overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
    })
  );
  const compiled = graph.compile();

  submitGraph(device, compiled, undefined);
  const [countA] = await readUint32(buffers.countA, 1);
  const [countB] = await readUint32(buffers.countB, 1);
  expect(countA).toBe(countContourSegments(RAMP, SIZE, SIZE, 3.5));
  expect(countB).toBe(countContourSegments(RAMP, SIZE, SIZE, 5.5));
  await expectVerticesOnLevel(buffers.verticesA, countA, 3.5);
  await expectVerticesOnLevel(buffers.verticesB, countB, 5.5);
  expect(await readUint32(buffers.overflow, 1)).toEqual([0]);

  levelsParameter.write(Float32Array.from([1.5]), 1);
  submitGraph(device, compiled, undefined);
  const [updatedCount] = await readUint32(buffers.countB, 1);
  expect(updatedCount).toBe(countContourSegments(RAMP, SIZE, SIZE, 1.5));
  await expectVerticesOnLevel(buffers.verticesB, updatedCount, 1.5);

  compiled.destroy();
  levelsParameter.destroy();
  for (const buffer of [elevationBuffer, ...Object.values(buffers)]) buffer.destroy();
});

it('GPUTerrainContours reports and resets capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevationBuffer = createInputBuffer(device, RAMP);
  const level = new GPUParameterBuffer(device, {
    id: 'level',
    format: 'float32',
    length: 1,
    values: Float32Array.from([3.5])
  });
  const verticesBuffer = createOutputBuffer(device, 4);
  const countBuffer = createOutputBuffer(device, 1);
  const requiredBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device, {id: 'terrain-contours-overflow'});
  graph.add(
    new GPUTerrainContours({
      width: SIZE,
      height: SIZE,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      levels: [
        {
          level: level.importToGraph(graph),
          vertices: importGraphBuffer(graph, 'vertices', verticesBuffer, 'float32x2', 2),
          segmentCount: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          requiredSegmentCount: importGraphBuffer(graph, 'required', requiredBuffer, 'uint32', 1)
        }
      ],
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(countBuffer, 1)).toEqual([1]);
  expect(await readUint32(requiredBuffer, 1)).toEqual([
    countContourSegments(RAMP, SIZE, SIZE, 3.5)
  ]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([1]);

  level.write(Float32Array.from([100]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(countBuffer, 1)).toEqual([0]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([0]);

  compiled.destroy();
  level.destroy();
  for (const buffer of [
    elevationBuffer,
    verticesBuffer,
    countBuffer,
    requiredBuffer,
    overflowBuffer
  ]) {
    buffer.destroy();
  }
});

const SENTINEL_COMMAND = {vertexCount: 77, instanceCount: 78, firstVertex: 79, firstInstance: 0};

type DrawRun = {record: number[]; neighbors: number[]; count: number};

/** Runs one contour level with an indirect draw record at `COMMAND_INDEX` of a 3-record buffer. */
async function runContourDraw(options: {
  id: string;
  level: number;
  segmentCapacity: number;
  drawLayout?: 'instanced' | 'line-list';
  verticesPerInstance?: number;
}): Promise<DrawRun | undefined> {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return undefined;
  }
  const COMMAND_INDEX = 1;
  const elevationBuffer = createInputBuffer(device, RAMP);
  const verticesBuffer = createOutputBuffer(device, options.segmentCapacity * 4);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const contributorOverflowBuffer = createOutputBuffer(device, 1);
  const commands = new DrawCommandBuffer(device, {
    id: `${options.id}-commands`,
    type: 'draw',
    commands: [SENTINEL_COMMAND, SENTINEL_COMMAND, SENTINEL_COMMAND]
  });
  const graph = new GPUCommandGraph(device, {id: options.id});
  graph.add(
    new GPUTerrainContours({
      width: SIZE,
      height: SIZE,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        }
      },
      levels: [
        {
          level: options.level,
          vertices: importGraphBuffer(
            graph,
            'vertices',
            verticesBuffer,
            'float32x2',
            options.segmentCapacity * 2
          ),
          segmentCount: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'level-overflow', overflowBuffer, 'uint32', 1),
          draw: commands.importToGraph(graph),
          drawCommandIndex: COMMAND_INDEX,
          drawLayout: options.drawLayout,
          verticesPerInstance: options.verticesPerInstance
        }
      ],
      overflow: importGraphBuffer(graph, 'overflow', contributorOverflowBuffer, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const words = new Uint32Array(await readUint32(commands.buffer, 12));
  const [count] = await readUint32(countBuffer, 1);
  compiled.destroy();
  commands.destroy();
  for (const buffer of [
    elevationBuffer,
    verticesBuffer,
    countBuffer,
    overflowBuffer,
    contributorOverflowBuffer
  ]) {
    buffer.destroy();
  }
  return {
    record: Array.from(words.slice(4, 8)),
    neighbors: [...Array.from(words.slice(0, 4)), ...Array.from(words.slice(8, 12))],
    count
  };
}

const SENTINEL_WORDS = [77, 78, 79, 0];
const FULL_COUNT = countContourSegments(RAMP, SIZE, SIZE, 3.5);

it('GPUTerrainContours default draw record stays [2, segmentCount, 0, 0]', async () => {
  const run = await runContourDraw({id: 'contour-draw-default', level: 3.5, segmentCapacity: 32});
  if (!run) return;
  expect(run.count).toBe(FULL_COUNT);
  expect(run.record).toEqual([2, FULL_COUNT, 0, 0]);
  expect(run.neighbors).toEqual([...SENTINEL_WORDS, ...SENTINEL_WORDS]);
});

it('GPUTerrainContours instanced draw uses the caller vertex count per segment', async () => {
  const run = await runContourDraw({
    id: 'contour-draw-quad',
    level: 3.5,
    segmentCapacity: 32,
    verticesPerInstance: 6
  });
  if (!run) return;
  expect(run.record).toEqual([6, FULL_COUNT, 0, 0]);
  expect(run.neighbors).toEqual([...SENTINEL_WORDS, ...SENTINEL_WORDS]);
});

it('GPUTerrainContours line-list draw is one non-instanced draw of 2 * segmentCount vertices', async () => {
  const run = await runContourDraw({
    id: 'contour-draw-line-list',
    level: 3.5,
    segmentCapacity: 32,
    drawLayout: 'line-list'
  });
  if (!run) return;
  expect(run.record).toEqual([2 * FULL_COUNT, 1, 0, 0]);
  expect(run.neighbors).toEqual([...SENTINEL_WORDS, ...SENTINEL_WORDS]);
});

it('GPUTerrainContours draw records clamp to capacity and reset on zero segments', async () => {
  expect(FULL_COUNT).toBeGreaterThan(2);
  for (const [name, layout, verticesPerInstance, expectedFor] of [
    ['default', undefined, undefined, (count: number) => [2, count, 0, 0]],
    ['quad', 'instanced', 4, (count: number) => [4, count, 0, 0]],
    ['line-list', 'line-list', undefined, (count: number) => [2 * count, 1, 0, 0]]
  ] as const) {
    const overflowing = await runContourDraw({
      id: `contour-draw-overflow-${name}`,
      level: 3.5,
      segmentCapacity: 2,
      drawLayout: layout,
      verticesPerInstance
    });
    if (!overflowing) return;
    expect(overflowing.count).toBe(2);
    expect(overflowing.record).toEqual(expectedFor(2));

    const empty = await runContourDraw({
      id: `contour-draw-empty-${name}`,
      level: 100,
      segmentCapacity: 2,
      drawLayout: layout,
      verticesPerInstance
    });
    if (!empty) return;
    expect(empty.count).toBe(0);
    expect(empty.record).toEqual(expectedFor(0));
  }
});
