// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUPickRegionMask, GPURegionMask} from '../../../src/geospatial/region-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  isSoftwareDevice,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  isInsidePolygon,
  isInsideRectangle,
  LASSO,
  POSITIONS,
  RECTANGLE,
  SCREEN_TRANSFORM,
  selectRows
} from './region-statistics-fixtures';

const ROW_COUNT = POSITIONS.length / 2;

function toMask(rows: number[]): number[] {
  return Array.from({length: ROW_COUNT}, (_, row) => (rows.includes(row) ? 1 : 0));
}

it('GPURegionMask selects a world lasso with a per-frame vertex count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positionsBuffer = createInputBuffer(device, POSITIONS);
  const maskBuffer = createOutputBuffer(device, ROW_COUNT);
  const overflowBuffer = createOutputBuffer(device, 1);
  const vertices = new GPUParameterBuffer(device, {
    id: 'lasso',
    format: 'float32',
    length: LASSO.length,
    values: LASSO
  });
  const vertexCount = new GPUParameterBuffer(device, {
    id: 'lasso-count',
    format: 'uint32',
    length: 1,
    values: Uint32Array.of(5)
  });
  const graph = new GPUCommandGraph(device, {id: 'region-mask-lasso'});
  graph.add(
    new GPURegionMask({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', ROW_COUNT),
      region: {
        kind: 'polygon',
        vertices: importGraphBuffer(graph, 'lasso', vertices.buffer, 'float32x2', 5),
        vertexCount: vertexCount.importToGraph(graph)
      },
      outputMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', ROW_COUNT),
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  const expectPolygon = async (count: number, overflow: number) => {
    submitGraph(device, compiled, undefined);
    const rows = selectRows((x, y) => isInsidePolygon(x, y, LASSO, Math.min(count, 5)));
    expect(await readUint32(maskBuffer, ROW_COUNT)).toEqual(toMask(rows));
    expect(await readUint32(overflowBuffer, 1)).toEqual([overflow]);
    return rows;
  };

  expect(await expectPolygon(5, 0)).toEqual([0, 1, 2, 3, 5, 8, 10]);
  vertexCount.write(Uint32Array.of(4));
  expect(await expectPolygon(4, 0)).toEqual([0, 1, 2, 3, 5, 8]);
  vertexCount.write(Uint32Array.of(9));
  await expectPolygon(9, 1);
  vertexCount.write(Uint32Array.of(2));
  expect(await expectPolygon(2, 0)).toEqual([]);

  compiled.destroy();
  vertices.destroy();
  vertexCount.destroy();
  for (const buffer of [positionsBuffer, maskBuffer, overflowBuffer]) buffer.destroy();
});

it('GPURegionMask matches world results in screen space and rejects points behind the camera', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const positionsBuffer = createInputBuffer(device, POSITIONS);
  const rectangleMaskBuffer = createOutputBuffer(device, ROW_COUNT);
  const polygonMaskBuffer = createOutputBuffer(device, ROW_COUNT);
  const overflowBuffers = [createOutputBuffer(device, 1), createOutputBuffer(device, 1)];
  const bounds = createInputBuffer(device, Float32Array.from([0, 0, 42.5, 42.5]));
  const lasso = createInputBuffer(
    device,
    LASSO.map(value => value * 10)
  );
  const lassoCount = createInputBuffer(device, Uint32Array.of(5));
  const transform = new GPUParameterBuffer(device, {
    id: 'transform',
    format: 'float32',
    length: 20,
    values: SCREEN_TRANSFORM
  });
  const graph = new GPUCommandGraph(device, {id: 'region-mask-screen'});
  const positions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', ROW_COUNT);
  const screenTransform = transform.importToGraph(graph);
  graph.add(
    new GPURegionMask({
      id: 'screen-rectangle',
      positions,
      region: {
        kind: 'rectangle',
        bounds: importGraphBuffer(graph, 'bounds', bounds, 'float32', 4),
        screenTransform
      },
      outputMask: importGraphBuffer(
        graph,
        'rectangle-mask',
        rectangleMaskBuffer,
        'uint32',
        ROW_COUNT
      ),
      overflow: importGraphBuffer(graph, 'rectangle-overflow', overflowBuffers[0], 'uint32', 1)
    })
  );
  graph.add(
    new GPURegionMask({
      id: 'screen-polygon',
      positions,
      region: {
        kind: 'polygon',
        vertices: importGraphBuffer(graph, 'lasso', lasso, 'float32x2', 5),
        vertexCount: importGraphBuffer(graph, 'lasso-count', lassoCount, 'uint32', 1),
        screenTransform
      },
      outputMask: importGraphBuffer(graph, 'polygon-mask', polygonMaskBuffer, 'uint32', ROW_COUNT),
      overflow: importGraphBuffer(graph, 'polygon-overflow', overflowBuffers[1], 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(rectangleMaskBuffer, ROW_COUNT)).toEqual(
    toMask(selectRows((x, y) => isInsideRectangle(x, y, RECTANGLE)))
  );
  expect(await readUint32(polygonMaskBuffer, ROW_COUNT)).toEqual(
    toMask(selectRows((x, y) => isInsidePolygon(x, y, LASSO, 5)))
  );

  const behindCamera = Float32Array.from(SCREEN_TRANSFORM);
  behindCamera[15] = -1;
  transform.write(behindCamera);
  submitGraph(device, compiled, undefined);
  expect(await readUint32(rectangleMaskBuffer, ROW_COUNT)).toEqual(toMask([]));

  compiled.destroy();
  transform.destroy();
  for (const buffer of [
    positionsBuffer,
    rectangleMaskBuffer,
    polygonMaskBuffer,
    ...overflowBuffers,
    bounds,
    lasso,
    lassoCount
  ]) {
    buffer.destroy();
  }
});

it('GPURegionMask preserves vector position chunks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const firstBuffer = createInputBuffer(device, POSITIONS.slice(0, 10));
  const secondBuffer = createInputBuffer(device, POSITIONS.slice(10));
  const bounds = createInputBuffer(device, RECTANGLE);
  const maskBuffer = createOutputBuffer(device, ROW_COUNT);
  const overflowBuffer = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device, {id: 'region-mask-vector'});
  graph.add(
    new GPURegionMask({
      positions: createVectorView('positions', 'float32x2', [
        importGraphBuffer(graph, 'p0', firstBuffer, 'float32x2', 5),
        importGraphBuffer(graph, 'p1', secondBuffer, 'float32x2', 7)
      ]),
      region: {kind: 'rectangle', bounds: importGraphBuffer(graph, 'bounds', bounds, 'float32', 4)},
      outputMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', ROW_COUNT),
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(maskBuffer, ROW_COUNT)).toEqual(
    toMask(selectRows((x, y) => isInsideRectangle(x, y, RECTANGLE)))
  );
  compiled.destroy();
  for (const buffer of [firstBuffer, secondBuffer, bounds, maskBuffer, overflowBuffer]) {
    buffer.destroy();
  }
});

it('GPUPickRegionMask deduplicates picked rows, filters batches, and reports overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const resultBuffer = createInputBuffer(
    device,
    Uint32Array.from([5, 0, 0, 0, 2, 0, 2, 0, 7, 1, 99, 0])
  );
  const batchMaskBuffer = createOutputBuffer(device, ROW_COUNT);
  const allMaskBuffer = createOutputBuffer(device, ROW_COUNT);
  const overflowBuffers = [createOutputBuffer(device, 1), createOutputBuffer(device, 1)];
  const graph = new GPUCommandGraph(device, {id: 'pick-region-mask'});
  const result = importGraphBuffer(graph, 'result', resultBuffer, 'uint32', 12);
  graph.add(
    new GPUPickRegionMask({
      id: 'batch-zero',
      result,
      batchIndex: 0,
      outputMask: importGraphBuffer(graph, 'batch-mask', batchMaskBuffer, 'uint32', ROW_COUNT),
      overflow: importGraphBuffer(graph, 'batch-overflow', overflowBuffers[0], 'uint32', 1)
    })
  );
  graph.add(
    new GPUPickRegionMask({
      id: 'all-batches',
      result,
      outputMask: importGraphBuffer(graph, 'all-mask', allMaskBuffer, 'uint32', ROW_COUNT),
      overflow: importGraphBuffer(graph, 'all-overflow', overflowBuffers[1], 'uint32', 1)
    })
  );
  const compiled = graph.compile();

  submitGraph(device, compiled, undefined);
  expect(await readUint32(batchMaskBuffer, ROW_COUNT)).toEqual(toMask([0, 2]));
  expect(await readUint32(allMaskBuffer, ROW_COUNT)).toEqual(toMask([0, 2, 7]));
  expect(await readUint32(overflowBuffers[0], 1)).toEqual([0]);

  resultBuffer.write(Uint32Array.from([7, 1, 0xffffffff, 0, 4, 0, 4, 0, 11, 0, 3, 0]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(batchMaskBuffer, ROW_COUNT)).toEqual(toMask([3, 4, 11]));
  expect(await readUint32(overflowBuffers[0], 1)).toEqual([1]);

  compiled.destroy();
  for (const buffer of [resultBuffer, batchMaskBuffer, allMaskBuffer, ...overflowBuffers]) {
    buffer.destroy();
  }
});
