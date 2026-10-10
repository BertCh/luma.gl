// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  GPURegionStatistics,
  GPURegionStatisticsReadback,
  type GPURegionStatisticsProps,
  type GPURegionStatisticsResult
} from '../../../src/gpu-spatial-analysis/region-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readCompactIds,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  CIRCLE,
  computeRegionStatistics,
  isInsidePolygon,
  isInsideRectangle,
  LASSO,
  POSITIONS,
  RECTANGLE,
  RECTANGLE_2,
  SOURCE_IDS,
  selectRows,
  VALUES
} from './region-statistics-fixtures';

const ROW_COUNT = POSITIONS.length / 2;

async function readSummary(buffer: Buffer, binCount: number): Promise<GPURegionStatisticsResult> {
  const bytes = await buffer.readAsync(0, getGPURegionStatisticsSummaryLength(binCount) * 4);
  return decodeGPURegionStatistics(bytes);
}

function expectSummary(
  actual: GPURegionStatisticsResult,
  expected: ReturnType<typeof computeRegionStatistics>
): void {
  expect(actual.selectedCount).toBe(expected.selectedCount);
  expect(actual.valueCount).toBe(expected.valueCount);
  expect(actual.sum).toBeCloseTo(expected.sum, 4);
  expect(actual.mean).toBeCloseTo(expected.mean, 4);
  expect(actual.minimum).toBeCloseTo(expected.minimum, 4);
  expect(actual.maximum).toBeCloseTo(expected.maximum, 4);
  expect(Array.from(actual.histogram)).toEqual(expected.histogram);
  expect(actual.histogramOutsideCount).toBe(expected.histogramOutsideCount);
}

type Fixture = {
  device: Device;
  graph: GPUCommandGraph;
  buffers: Buffer[];
  positions: ReturnType<typeof importGraphBuffer<'float32x2', void>>;
  values: ReturnType<typeof importGraphBuffer<'float32', void>>;
  sourceIds: ReturnType<typeof importGraphBuffer<'uint32', void>>;
  output: GPURegionStatisticsProps['output'] & {};
  idsBuffer: Buffer;
  countBuffer: Buffer;
  overflowBuffer: Buffer;
  totalBuffer: Buffer;
  summaryBuffer: Buffer;
  summary: ReturnType<typeof importGraphBuffer<'uint32', void>>;
};

function createFixture(device: Device, id: string, capacity: number, binCount: number): Fixture {
  const graph = new GPUCommandGraph(device, {id});
  const positionsBuffer = createInputBuffer(device, POSITIONS);
  const valuesBuffer = createInputBuffer(device, VALUES);
  const sourceIdsBuffer = createInputBuffer(device, SOURCE_IDS);
  const idsBuffer = createOutputBuffer(device, capacity);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const totalBuffer = createOutputBuffer(device, 1);
  const summaryBuffer = createOutputBuffer(device, getGPURegionStatisticsSummaryLength(binCount));
  return {
    device,
    graph,
    buffers: [
      positionsBuffer,
      valuesBuffer,
      sourceIdsBuffer,
      idsBuffer,
      countBuffer,
      overflowBuffer,
      totalBuffer,
      summaryBuffer
    ],
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', ROW_COUNT),
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', ROW_COUNT),
    sourceIds: importGraphBuffer(graph, 'source-ids', sourceIdsBuffer, 'uint32', ROW_COUNT),
    output: {
      ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', capacity),
      count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
      requiredCount: importGraphBuffer(graph, 'total', totalBuffer, 'uint32', 1)
    },
    idsBuffer,
    countBuffer,
    overflowBuffer,
    totalBuffer,
    summaryBuffer,
    summary: importGraphBuffer(
      graph,
      'summary',
      summaryBuffer,
      'uint32',
      getGPURegionStatisticsSummaryLength(binCount)
    )
  };
}

const RECTANGLE_ROWS = selectRows((x, y) => isInsideRectangle(x, y, RECTANGLE));

it('GPURegionStatistics summarizes a world rectangle and updates per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'region-rectangle', 12, 4);
  const bounds = new GPUParameterBuffer(device, {
    id: 'bounds',
    format: 'float32',
    length: 4,
    values: RECTANGLE
  });
  fixture.graph.add(
    new GPURegionStatistics({
      selection: {kind: 'rectangle', bounds: bounds.importToGraph(fixture.graph)},
      positions: fixture.positions,
      values: fixture.values,
      sourceIds: fixture.sourceIds,
      histogram: {binCount: 4},
      output: fixture.output,
      summary: fixture.summary
    })
  );
  const compiled = fixture.graph.compile();

  submitGraph(device, compiled, undefined);
  expect(RECTANGLE_ROWS).toEqual([0, 1, 2, 5, 8]);
  let summary = await readSummary(fixture.summaryBuffer, 4);
  expectSummary(summary, computeRegionStatistics(RECTANGLE_ROWS, VALUES, 4));
  expect(summary.sum).toBe(55);
  expect(Array.from(summary.histogram)).toEqual([1, 1, 1, 1]);
  expect(await readCompactIds(fixture.idsBuffer, fixture.countBuffer)).toEqual([
    100, 101, 102, 105, 108
  ]);
  expect(await readUint32(fixture.overflowBuffer, 1)).toEqual([0]);
  expect(await readUint32(fixture.totalBuffer, 1)).toEqual([5]);

  bounds.write(RECTANGLE_2);
  submitGraph(device, compiled, undefined);
  const rows = selectRows((x, y) => isInsideRectangle(x, y, RECTANGLE_2));
  summary = await readSummary(fixture.summaryBuffer, 4);
  expectSummary(summary, computeRegionStatistics(rows, VALUES, 4));
  expect(Array.from(summary.histogram)).toEqual([1, 0, 0, 2]);
  expect(await readCompactIds(fixture.idsBuffer, fixture.countBuffer)).toEqual([103, 106, 107]);

  bounds.write(Float32Array.from([20, 20, 30, 30]));
  submitGraph(device, compiled, undefined);
  summary = await readSummary(fixture.summaryBuffer, 4);
  expectSummary(summary, computeRegionStatistics([], VALUES, 4));
  expect(await readUint32(fixture.countBuffer, 1)).toEqual([0]);

  compiled.destroy();
  bounds.destroy();
  for (const buffer of fixture.buffers) buffer.destroy();
});

it('GPURegionStatistics clamps selected IDs without truncating statistics', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'region-capacity', 2, 0);
  const boundsBuffer = createInputBuffer(device, RECTANGLE);
  fixture.graph.add(
    new GPURegionStatistics({
      selection: {
        kind: 'rectangle',
        bounds: importGraphBuffer(fixture.graph, 'bounds', boundsBuffer, 'float32', 4)
      },
      positions: fixture.positions,
      values: fixture.values,
      sourceIds: fixture.sourceIds,
      output: fixture.output,
      summary: fixture.summary
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readCompactIds(fixture.idsBuffer, fixture.countBuffer)).toEqual([100, 101]);
  expect(await readUint32(fixture.overflowBuffer, 1)).toEqual([1]);
  expect(await readUint32(fixture.totalBuffer, 1)).toEqual([5]);
  const summary = await readSummary(fixture.summaryBuffer, 0);
  expect(summary.selectionTruncated).toBe(true);
  expect(summary.selectedCount).toBe(5);
  compiled.destroy();
  for (const buffer of [...fixture.buffers, boundsBuffer]) buffer.destroy();
});

it('GPURegionStatistics writes the clamped selected count to a draw record instanceCount', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const capacity of [12, 2]) {
    const fixture = createFixture(device, `region-draw-${capacity}`, capacity, 0);
    const bounds = new GPUParameterBuffer(device, {
      id: `bounds-${capacity}`,
      format: 'float32',
      length: 4,
      values: RECTANGLE
    });
    // A 4-word draw record [vertexCount, instanceCount, firstVertex, firstInstance] seeded with
    // sentinels; the contributor must write only the 1-row slice at element offset 1.
    const recordBuffer = device.createBuffer({
      data: Uint32Array.from([6, 999, 0, 0]),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    const recordHandle = importGraphBuffer(fixture.graph, 'record', recordBuffer, 'uint32', 4);
    const drawInstanceCount = fixture.graph.createDataView(recordHandle.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    fixture.graph.add(
      new GPURegionStatistics({
        selection: {kind: 'rectangle', bounds: bounds.importToGraph(fixture.graph)},
        positions: fixture.positions,
        values: fixture.values,
        sourceIds: fixture.sourceIds,
        output: fixture.output,
        drawInstanceCount,
        summary: fixture.summary
      })
    );
    const compiled = fixture.graph.compile();

    submitGraph(device, compiled, undefined);
    const expectedCount = Math.min(RECTANGLE_ROWS.length, capacity);
    expect(await readUint32(recordBuffer, 4)).toEqual([6, expectedCount, 0, 0]);
    expect(await readUint32(fixture.countBuffer, 1)).toEqual([expectedCount]);
    expect(await readUint32(fixture.overflowBuffer, 1)).toEqual([
      RECTANGLE_ROWS.length > capacity ? 1 : 0
    ]);

    bounds.write(Float32Array.from([20, 20, 30, 30]));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(recordBuffer, 4)).toEqual([6, 0, 0, 0]);

    compiled.destroy();
    bounds.destroy();
    recordBuffer.destroy();
    for (const buffer of fixture.buffers) buffer.destroy();
  }
});

it('GPURegionStatistics uses literal and per-frame histogram domains', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'region-domain', 12, 4);
  const boundsBuffer = createInputBuffer(device, RECTANGLE);
  const literalSummaryBuffer = createOutputBuffer(device, 12);
  const domain = new GPUParameterBuffer(device, {
    id: 'domain',
    format: 'float32',
    length: 2,
    values: Float32Array.from([0, 40])
  });
  const bounds = importGraphBuffer(fixture.graph, 'bounds', boundsBuffer, 'float32', 4);
  fixture.graph.add(
    new GPURegionStatistics({
      id: 'literal',
      selection: {kind: 'rectangle', bounds},
      positions: fixture.positions,
      values: fixture.values,
      histogram: {binCount: 4, domain: [0, 40]},
      summary: importGraphBuffer(
        fixture.graph,
        'literal-summary',
        literalSummaryBuffer,
        'uint32',
        12
      )
    })
  );
  fixture.graph.add(
    new GPURegionStatistics({
      id: 'view',
      selection: {kind: 'rectangle', bounds},
      positions: fixture.positions,
      values: fixture.values,
      histogram: {binCount: 4, domain: domain.importToGraph(fixture.graph)},
      summary: fixture.summary
    })
  );
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const expected = computeRegionStatistics(RECTANGLE_ROWS, VALUES, 4, [0, 40]);
  expect(expected.histogram).toEqual([0, 1, 1, 1]);
  expect(expected.histogramOutsideCount).toBe(1);
  expectSummary(await readSummary(literalSummaryBuffer, 4), expected);
  expectSummary(await readSummary(fixture.summaryBuffer, 4), expected);

  domain.write(Float32Array.from([-10, 30]));
  submitGraph(device, compiled, undefined);
  expectSummary(
    await readSummary(fixture.summaryBuffer, 4),
    computeRegionStatistics(RECTANGLE_ROWS, VALUES, 4, [-10, 30])
  );

  compiled.destroy();
  domain.destroy();
  for (const buffer of [...fixture.buffers, boundsBuffer, literalSummaryBuffer]) buffer.destroy();
});

it('GPURegionStatistics summarizes a lasso and flags vertex overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'region-lasso', 12, 4);
  const lasso = createInputBuffer(device, LASSO);
  const vertexCount = new GPUParameterBuffer(device, {
    id: 'vertex-count',
    format: 'uint32',
    length: 1,
    values: Uint32Array.of(5)
  });
  fixture.graph.add(
    new GPURegionStatistics({
      selection: {
        kind: 'polygon',
        vertices: importGraphBuffer(fixture.graph, 'lasso', lasso, 'float32x2', 5),
        vertexCount: vertexCount.importToGraph(fixture.graph)
      },
      positions: fixture.positions,
      values: fixture.values,
      histogram: {binCount: 4},
      output: fixture.output,
      summary: fixture.summary
    })
  );
  const compiled = fixture.graph.compile();
  const rows = selectRows((x, y) => isInsidePolygon(x, y, LASSO, 5));

  submitGraph(device, compiled, undefined);
  let summary = await readSummary(fixture.summaryBuffer, 4);
  expectSummary(summary, computeRegionStatistics(rows, VALUES, 4));
  expect(Array.from(summary.histogram)).toEqual([2, 1, 2, 1]);
  expect(await readCompactIds(fixture.idsBuffer, fixture.countBuffer)).toEqual(rows);
  expect(summary.regionTruncated).toBe(false);

  vertexCount.write(Uint32Array.of(99));
  submitGraph(device, compiled, undefined);
  summary = await readSummary(fixture.summaryBuffer, 4);
  expect(summary.regionTruncated).toBe(true);
  expect(summary.selectedCount).toBe(rows.length);
  expect(await readUint32(fixture.overflowBuffer, 1)).toEqual([1]);

  compiled.destroy();
  vertexCount.destroy();
  for (const buffer of [...fixture.buffers, lasso]) buffer.destroy();
});

it('GPURegionStatistics handles radius, pick-region, mask, and count-only selections', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'region-kinds', 12, 4);
  const circleBuffer = createInputBuffer(device, CIRCLE);
  const pickBuffer = createInputBuffer(
    device,
    Uint32Array.from([5, 0, 0, 0, 2, 0, 2, 0, 7, 1, 99, 0])
  );
  const maskBuffer = createInputBuffer(
    device,
    Uint32Array.from([0, 0, 3, 0, 0, 7, 0, 0, 0, 0, 0, 0])
  );
  const boundsBuffer = createInputBuffer(device, RECTANGLE);
  const pickSummaryBuffer = createOutputBuffer(device, 8);
  const maskSummaryBuffer = createOutputBuffer(device, 8);
  const countSummaryBuffer = createOutputBuffer(device, 8);
  const outputMaskBuffer = createOutputBuffer(device, ROW_COUNT);
  const {graph} = fixture;
  graph.add(
    new GPURegionStatistics({
      id: 'radius',
      selection: {
        kind: 'radius',
        circle: importGraphBuffer(graph, 'circle', circleBuffer, 'float32', 3)
      },
      positions: fixture.positions,
      values: fixture.values,
      histogram: {binCount: 4},
      summary: fixture.summary
    })
  );
  graph.add(
    new GPURegionStatistics({
      id: 'pick',
      selection: {
        kind: 'pick-region',
        result: importGraphBuffer(graph, 'pick-result', pickBuffer, 'uint32', 12),
        batchIndex: 0
      },
      values: fixture.values,
      summary: importGraphBuffer(graph, 'pick-summary', pickSummaryBuffer, 'uint32', 8)
    })
  );
  graph.add(
    new GPURegionStatistics({
      id: 'mask',
      selection: {kind: 'mask', mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 12)},
      values: fixture.values,
      summary: importGraphBuffer(graph, 'mask-summary', maskSummaryBuffer, 'uint32', 8)
    })
  );
  graph.add(
    new GPURegionStatistics({
      id: 'count-only',
      selection: {
        kind: 'rectangle',
        bounds: importGraphBuffer(graph, 'bounds', boundsBuffer, 'float32', 4)
      },
      positions: fixture.positions,
      outputMask: importGraphBuffer(graph, 'output-mask', outputMaskBuffer, 'uint32', ROW_COUNT),
      summary: importGraphBuffer(graph, 'count-summary', countSummaryBuffer, 'uint32', 8)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);

  const circleRows = selectRows((x, y) => Math.hypot(x - 2, y - 2) <= 1.6);
  expect(circleRows).toEqual([0, 1, 2, 5]);
  expectSummary(
    await readSummary(fixture.summaryBuffer, 4),
    computeRegionStatistics(circleRows, VALUES, 4)
  );
  expectSummary(
    await readSummary(pickSummaryBuffer, 0),
    computeRegionStatistics([0, 2], VALUES, 0)
  );
  const maskSummary = await readSummary(maskSummaryBuffer, 0);
  expect(maskSummary).toMatchObject({selectedCount: 2, valueCount: 1, sum: 30});
  const countSummary = await readSummary(countSummaryBuffer, 0);
  expect(countSummary).toMatchObject({selectedCount: 5, valueCount: 0, sum: 0, mean: 0});
  expect(await readUint32(outputMaskBuffer, ROW_COUNT)).toEqual(
    Array.from({length: ROW_COUNT}, (_, row) => (RECTANGLE_ROWS.includes(row) ? 1 : 0))
  );

  compiled.destroy();
  for (const buffer of [
    ...fixture.buffers,
    circleBuffer,
    pickBuffer,
    maskBuffer,
    boundsBuffer,
    pickSummaryBuffer,
    maskSummaryBuffer,
    countSummaryBuffer,
    outputMaskBuffer
  ]) {
    buffer.destroy();
  }
});

it('GPURegionStatisticsReadback reads summaries and drops frames under backpressure', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, 'region-readback', 12, 4);
  const boundsBuffer = createInputBuffer(device, RECTANGLE);
  fixture.graph.add(
    new GPURegionStatistics({
      selection: {
        kind: 'rectangle',
        bounds: importGraphBuffer(fixture.graph, 'bounds', boundsBuffer, 'float32', 4)
      },
      positions: fixture.positions,
      values: fixture.values,
      histogram: {binCount: 4},
      summary: fixture.summary
    })
  );
  const compiled = fixture.graph.compile();
  const readback = new GPURegionStatisticsReadback(device, {binCount: 4, slotCount: 2});

  const encoder = device.createCommandEncoder();
  compiled.encode(encoder, {parameters: undefined});
  const ticket = readback.encodeRead(encoder, fixture.summaryBuffer);
  device.submit(encoder.finish());
  expect(ticket).not.toBeNull();
  expectSummary(await readback.read(ticket!), computeRegionStatistics(RECTANGLE_ROWS, VALUES, 4));

  const pressureEncoder = device.createCommandEncoder();
  const tickets = [
    readback.encodeRead(pressureEncoder, fixture.summaryBuffer),
    readback.encodeRead(pressureEncoder, fixture.summaryBuffer)
  ];
  expect(readback.encodeRead(pressureEncoder, fixture.summaryBuffer)).toBeNull();
  device.submit(pressureEncoder.finish());
  for (const pending of tickets) {
    await readback.read(pending!);
  }

  readback.destroy();
  compiled.destroy();
  for (const buffer of [...fixture.buffers, boundsBuffer]) buffer.destroy();
});
