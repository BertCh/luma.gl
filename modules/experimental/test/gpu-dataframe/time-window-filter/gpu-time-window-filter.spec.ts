// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTimeWindowFilterParameterValues,
  GPUTimeWindowFilter,
  splitTimestamps,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  type GPUTimeWindowFilterSettings
} from '../../../src/gpu-dataframe/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  isSoftwareDevice,
  readCompactIds,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {TIME_WINDOW_EDGE_POLICY} from '../dataframe-edge-policy';
import {requireTestDevice} from '../../utils/require-test-device';

type OracleResult = {
  mask: number[];
  ids: number[];
  count: number;
  total: number;
  overflow: number;
  weights: number[];
  clip: number[];
};

/** CPU reference for the time-window classify and bounded publish semantics. */
function filterTimeWindowOracle(
  rows: {start: number[]; end?: number[]},
  window: GPUTimeWindowFilterSettings,
  capacity: number,
  sourceIds?: number[]
): OracleResult {
  const startFade = window.startFadeDuration ?? 0;
  const endFade = window.endFadeDuration ?? 0;
  const result: OracleResult = {
    mask: [],
    ids: [],
    count: 0,
    total: 0,
    overflow: 0,
    weights: [],
    clip: []
  };
  const accepted: number[] = [];
  for (let row = 0; row < rows.start.length; row++) {
    const start = rows.start[row];
    const end = rows.end?.[row] ?? start;
    const endAfterStart = end - window.start;
    const startBeforeEnd = window.end - start;
    // A reversed window accepts nothing (B-13), whatever the rows' extent.
    const isAccepted = window.end >= window.start && endAfterStart >= 0 && startBeforeEnd >= 0;
    result.mask.push(isAccepted ? 1 : 0);
    const startRamp = startFade > 0 ? Math.min(Math.max(endAfterStart / startFade, 0), 1) : 1;
    const endRamp = endFade > 0 ? Math.min(Math.max(startBeforeEnd / endFade, 0), 1) : 1;
    result.weights.push(isAccepted ? Math.min(startRamp, endRamp) : 0);
    const duration = end - start;
    const clipStart =
      duration > 0 ? Math.min(Math.max((window.start - start) / duration, 0), 1) : 0;
    const clipEnd = duration > 0 ? Math.min(Math.max(startBeforeEnd / duration, 0), 1) : 1;
    result.clip.push(isAccepted ? clipStart : 0, isAccepted ? clipEnd : 0);
    if (isAccepted) {
      accepted.push(sourceIds?.[row] ?? row);
    }
  }
  result.total = accepted.length;
  result.count = Math.min(accepted.length, capacity);
  result.overflow = accepted.length > capacity ? 1 : 0;
  result.ids = accepted.slice(0, result.count);
  return result;
}

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(actual[index]).toBeCloseTo(value, 5);
  }
}

function destroyAll(buffers: Buffer[]): void {
  for (const buffer of buffers) {
    buffer.destroy();
  }
}

it('GPUTimeWindowFilter clamps instant rows, fades, and updates without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const timestamps = [0, 1, 2, 3, 4, 5, 6, 7];
  const sourceIds = timestamps.map(row => 100 + row);
  const timestampsBuffer = createInputBuffer(device, Float32Array.from(timestamps));
  const sourceIdsBuffer = createInputBuffer(device, Uint32Array.from(sourceIds));
  const idsBuffer = createOutputBuffer(device, 3);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const totalBuffer = createOutputBuffer(device, 1);
  const weightsBuffer = createOutputBuffer(device, 8);
  const drawCommands = new DrawCommandBuffer(device, {
    id: 'time-window-draw',
    type: 'draw',
    commands: [{vertexCount: 6, instanceCount: 0}]
  });
  const window = new GPUParameterBuffer(device, {
    id: 'window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues({
      start: 2,
      end: 5,
      startFadeDuration: 2
    })
  });

  const graph = new GPUCommandGraph(device, {id: 'time-window-instant'});
  graph.add(
    new GPUTimeWindowFilter({
      timestamps: importGraphBuffer(graph, 'timestamps', timestampsBuffer, 'float32', 8),
      sourceIds: importGraphBuffer(graph, 'source-ids', sourceIdsBuffer, 'uint32', 8),
      window: window.importToGraph(graph),
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', 3),
        status: {
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'total', totalBuffer, 'uint32', 1)
        },
        fadeWeights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 8),
        drawInstanceCount: graph.importGPUData('draw-count', drawCommands.getInstanceCountData(0))
      }
    })
  );
  const compiled = graph.compile();
  const readDrawCount = async () => {
    const bytes = await drawCommands.buffer.readAsync(
      drawCommands.getInstanceCountByteOffset(0),
      4
    );
    return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
  };

  submitGraph(device, compiled, undefined);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([102, 103, 104]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([1]);
  expect(await readUint32(totalBuffer, 1)).toEqual([4]);
  expect(await readDrawCount()).toBe(3);
  expectClose(await readFloat32(weightsBuffer, 8), [0, 0, 0, 0.5, 1, 1, 0, 0]);

  window.write(getGPUTimeWindowFilterParameterValues({start: 6, end: 6}));
  submitGraph(device, compiled, undefined);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([106]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([0]);
  expect(await readUint32(totalBuffer, 1)).toEqual([1]);
  expect(await readDrawCount()).toBe(1);
  expectClose(await readFloat32(weightsBuffer, 8), [0, 0, 0, 0, 0, 0, 1, 0]);

  window.write(getGPUTimeWindowFilterParameterValues({start: 5, end: 2}));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(countBuffer, 1)).toEqual([0]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([0]);

  compiled.destroy();
  window.destroy();
  drawCommands.destroy();
  destroyAll([
    timestampsBuffer,
    sourceIdsBuffer,
    idsBuffer,
    countBuffer,
    overflowBuffer,
    totalBuffer,
    weightsBuffer
  ]);
});

it('GPUTimeWindowFilter clips trail segments and counts visible rows per track', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const starts = [0, 10, 20, 5, 15, 25];
  const ends = [10, 20, 30, 15, 25, 35];
  const tracks = [0, 0, 0, 1, 1, 1];
  const startsBuffer = createInputBuffer(device, Float32Array.from(starts));
  const endsBuffer = createInputBuffer(device, Float32Array.from(ends));
  const tracksBuffer = createInputBuffer(device, Uint32Array.from(tracks));
  const idsBuffer = createOutputBuffer(device, 6);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const maskBuffer = createOutputBuffer(device, 6);
  const weightsBuffer = createOutputBuffer(device, 6);
  const clipBuffer = createOutputBuffer(device, 12);
  const trackCountsBuffer = createOutputBuffer(device, 2);
  const firstWindow = {start: 12, end: 22, startFadeDuration: 10};
  const window = new GPUParameterBuffer(device, {
    id: 'trail-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues(firstWindow)
  });

  const graph = new GPUCommandGraph(device, {id: 'time-window-trail'});
  graph.add(
    new GPUTimeWindowFilter({
      id: 'trail',
      timestamps: importGraphBuffer(graph, 'starts', startsBuffer, 'float32', 6),
      endTimestamps: importGraphBuffer(graph, 'ends', endsBuffer, 'float32', 6),
      window: window.importToGraph(graph),
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', 6),
        status: {
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
        },
        outputMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 6),
        fadeWeights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 6),
        clipFractions: importGraphBuffer(graph, 'clip', clipBuffer, 'float32x2', 6),
        trackVisibleCounts: importGraphBuffer(graph, 'track-counts', trackCountsBuffer, 'uint32', 2)
      },
      trackIds: importGraphBuffer(graph, 'tracks', tracksBuffer, 'uint32', 6)
    })
  );
  const compiled = graph.compile();

  submitGraph(device, compiled, undefined);
  let oracle = filterTimeWindowOracle({start: starts, end: ends}, firstWindow, 6);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([1, 2, 3, 4]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([0]);
  expect(await readUint32(maskBuffer, 6)).toEqual([0, 1, 1, 1, 1, 0]);
  expect(await readUint32(trackCountsBuffer, 2)).toEqual([2, 2]);
  expectClose(await readFloat32(clipBuffer, 12), [0, 0, 0.2, 1, 0, 0.2, 0.7, 1, 0, 0.7, 0, 0]);
  expectClose(await readFloat32(weightsBuffer, 6), [0, 0.8, 1, 0.3, 1, 0]);
  expectClose(await readFloat32(clipBuffer, 12), oracle.clip);

  const secondWindow = {start: 22, end: 32, startFadeDuration: 10};
  window.write(getGPUTimeWindowFilterParameterValues(secondWindow));
  submitGraph(device, compiled, undefined);
  oracle = filterTimeWindowOracle({start: starts, end: ends}, secondWindow, 6);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([2, 4, 5]);
  expect(oracle.ids).toEqual([2, 4, 5]);
  expect(await readUint32(trackCountsBuffer, 2)).toEqual([1, 2]);
  expect(await readUint32(maskBuffer, 6)).toEqual(oracle.mask);
  expectClose(await readFloat32(weightsBuffer, 6), oracle.weights);
  expectClose(await readFloat32(clipBuffer, 12), oracle.clip);

  compiled.destroy();
  window.destroy();
  destroyAll([
    startsBuffer,
    endsBuffer,
    tracksBuffer,
    idsBuffer,
    countBuffer,
    overflowBuffer,
    maskBuffer,
    weightsBuffer,
    clipBuffer,
    trackCountsBuffer
  ]);
});

it('GPUTimeWindowFilter reversed windows reject interval rows too (B-13)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Row 1 spans both window edges, so the bare overlap test accepted it before B-13.
  const starts = [0, 5, 12, 30];
  const ends = [10, 25, 18, 40];
  const startsBuffer = createInputBuffer(device, Float32Array.from(starts));
  const endsBuffer = createInputBuffer(device, Float32Array.from(ends));
  const idsBuffer = createOutputBuffer(device, 4);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const maskBuffer = createOutputBuffer(device, 4);
  const reversedWindow = {start: 20, end: 10};
  const window = new GPUParameterBuffer(device, {
    id: 'reversed-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues(reversedWindow)
  });
  const graph = new GPUCommandGraph(device, {id: 'time-window-reversed'});
  graph.add(
    new GPUTimeWindowFilter({
      id: 'reversed',
      timestamps: importGraphBuffer(graph, 'starts', startsBuffer, 'float32', 4),
      endTimestamps: importGraphBuffer(graph, 'ends', endsBuffer, 'float32', 4),
      window: window.importToGraph(graph),
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', 4),
        status: {
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
        },
        outputMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 4)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = filterTimeWindowOracle({start: starts, end: ends}, reversedWindow, 4);
  expect(oracle.mask).toEqual([0, 0, 0, 0]);
  expect(await readUint32(maskBuffer, 4)).toEqual(oracle.mask);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([]);
  const orderedWindow = {start: 10, end: 20};
  window.write(getGPUTimeWindowFilterParameterValues(orderedWindow));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(maskBuffer, 4)).toEqual(
    filterTimeWindowOracle({start: starts, end: ends}, orderedWindow, 4).mask
  );
  expect(await readUint32(maskBuffer, 4)).toEqual([1, 1, 1, 0]);
  compiled.destroy();
  window.destroy();
  destroyAll([startsBuffer, endsBuffer, idsBuffer, countBuffer, overflowBuffer, maskBuffer]);
});

it('GPUTimeWindowFilter separates epoch timestamps with double-single parts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const base = 1_700_000_000;
  expect(Math.fround(base + 10)).toBe(Math.fround(base + 20));
  const {high, low} = splitTimestamps([0, 10, 20, 30, 40].map(offset => base + offset));
  const highBuffer = createInputBuffer(device, high);
  const lowBuffer = createInputBuffer(device, low);
  const idsBuffer = createOutputBuffer(device, 5);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const window = new GPUParameterBuffer(device, {
    id: 'epoch-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues({
      start: base + 15,
      end: base + 35
    })
  });
  const graph = new GPUCommandGraph(device, {id: 'time-window-epoch'});
  graph.add(
    new GPUTimeWindowFilter({
      timestamps: importGraphBuffer(graph, 'high', highBuffer, 'float32', 5),
      timestampsLow: importGraphBuffer(graph, 'low', lowBuffer, 'float32', 5),
      window: window.importToGraph(graph),
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', 5),
        status: {
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([2, 3]);

  compiled.destroy();
  window.destroy();
  destroyAll([highBuffer, lowBuffer, idsBuffer, countBuffer, overflowBuffer]);
});

it('GPUTimeWindowFilter ANDs extra predicates across vector chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const firstBuffer = createInputBuffer(device, Float32Array.from([0, 1, 2]));
  const secondBuffer = createInputBuffer(device, Float32Array.from([3, 4]));
  const selectionBuffer = createInputBuffer(device, Uint32Array.from([1, 0, 1, 1, 1]));
  const maskFirstBuffer = createOutputBuffer(device, 3);
  const maskSecondBuffer = createOutputBuffer(device, 2);
  const idsBuffer = createOutputBuffer(device, 5);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const window = new GPUParameterBuffer(device, {
    id: 'vector-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues({start: 1, end: 3})
  });
  const graph = new GPUCommandGraph(device, {id: 'time-window-vector'});
  graph.add(
    new GPUTimeWindowFilter({
      timestamps: createVectorView('timestamps', 'float32', [
        importGraphBuffer(graph, 't0', firstBuffer, 'float32', 3),
        importGraphBuffer(graph, 't1', secondBuffer, 'float32', 2)
      ]),
      window: window.importToGraph(graph),
      additionalPredicates: [
        {
          kind: 'selection',
          mask: importGraphBuffer(graph, 'selection', selectionBuffer, 'uint32', 5)
        }
      ],
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', 5),
        status: {
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
        },
        outputMask: createVectorView('mask', 'uint32', [
          importGraphBuffer(graph, 'm0', maskFirstBuffer, 'uint32', 3),
          importGraphBuffer(graph, 'm1', maskSecondBuffer, 'uint32', 2)
        ])
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([2, 3]);
  expect([
    ...(await readUint32(maskFirstBuffer, 3)),
    ...(await readUint32(maskSecondBuffer, 2))
  ]).toEqual([0, 0, 1, 1, 0]);

  selectionBuffer.write(Uint32Array.from([1, 1, 1, 1, 1]));
  submitGraph(device, compiled, undefined);
  expect(await readCompactIds(idsBuffer, countBuffer)).toEqual([1, 2, 3]);

  compiled.destroy();
  window.destroy();
  destroyAll([
    firstBuffer,
    secondBuffer,
    selectionBuffer,
    maskFirstBuffer,
    maskSecondBuffer,
    idsBuffer,
    countBuffer,
    overflowBuffer
  ]);
});

it('GPUTimeWindowFilter follows the time window edge policy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const row of TIME_WINDOW_EDGE_POLICY) {
    const rows = row.starts.length;
    const buffers = [
      createInputBuffer(device, Float32Array.from(row.starts)),
      createInputBuffer(device, Float32Array.from(row.ends)),
      createOutputBuffer(device, rows),
      createOutputBuffer(device, 1),
      createOutputBuffer(device, 1),
      createOutputBuffer(device, rows)
    ];
    const [startsBuffer, endsBuffer, idsBuffer, countBuffer, overflowBuffer, maskBuffer] = buffers;
    const window = new GPUParameterBuffer(device, {
      id: 'policy-window',
      format: 'float32',
      length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
      values: getGPUTimeWindowFilterParameterValues(row.window)
    });
    const graph = new GPUCommandGraph(device, {id: 'time-window-policy'});
    graph.add(
      new GPUTimeWindowFilter({
        id: 'policy',
        timestamps: importGraphBuffer(graph, 'starts', startsBuffer, 'float32', rows),
        endTimestamps: importGraphBuffer(graph, 'ends', endsBuffer, 'float32', rows),
        window: window.importToGraph(graph),
        output: {
          ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', rows),
          status: {
            count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
          },
          outputMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(maskBuffer, rows), row.name).toEqual(row.expectedMask);
    } finally {
      compiled.destroy();
      window.destroy();
      destroyAll(buffers);
    }
  }
});
