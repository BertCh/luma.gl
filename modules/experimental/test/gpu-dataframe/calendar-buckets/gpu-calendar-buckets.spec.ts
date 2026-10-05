// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUCalendarBucketsParameterValues,
  GPUCalendarBuckets,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  type GPUCalendarBucketsOutput,
  type GPUCalendarBucketsProps
} from '../../../src/gpu-dataframe/calendar-buckets';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  countHourWeekdayOnCPU,
  createRandomTimestamps,
  createRandomWords,
  decodeCalendarOnCPU,
  getInvalidCalendarFields,
  type CalendarFields
} from './calendar-buckets-oracle';

const COLUMNS = [
  ['year', 'year', 'sint32'],
  ['month', 'month', 'uint32'],
  ['dayOfMonth', 'day', 'uint32'],
  ['hour', 'hour', 'uint32'],
  ['minute', 'minute', 'uint32'],
  ['weekday', 'weekday', 'uint32'],
  ['dayOfYear', 'dayOfYear', 'uint32'],
  ['isoWeek', 'isoWeek', 'uint32'],
  ['isoWeekYear', 'isoWeekYear', 'sint32'],
  ['quarter', 'quarter', 'uint32']
] as const;

type Scene = {
  timestamps: BigInt64Array;
  mask?: Uint32Array;
  utcOffsets?: Int32Array;
};

type CalendarResult = Record<string, number[]> & {
  hourWeekdayCounts: number[];
};

/** Counts how often the contributor emits its nodes, which a parameter change must never trigger. */
class CountingCalendarBuckets extends GPUCalendarBuckets {
  static nodeBuilds = 0;
  override getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    CountingCalendarBuckets.nodeBuilds++;
    return super.getCommandNodes(graph);
  }
}

function createFixture(device: Device, scene: Scene, columns: readonly string[] = []) {
  const rows = scene.timestamps.length;
  const graph = new GPUCommandGraph(device, {id: 'calendar-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'calendar-parameters',
    format: 'sint32',
    length: 2
  });
  const output: Record<string, GraphDataView> = {};
  const outputBuffers: Record<string, Buffer> = {};
  for (const [name, , format] of COLUMNS) {
    if (columns.includes(name)) {
      outputBuffers[name] = track(createOutputBuffer(device, rows));
      output[name] = importGraphBuffer(graph, `o-${name}`, outputBuffers[name], format, rows);
    }
  }
  if (columns.includes('hourWeekdayCounts')) {
    outputBuffers['hourWeekdayCounts'] = track(
      createOutputBuffer(device, GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)
    );
    output['hourWeekdayCounts'] = importGraphBuffer(
      graph,
      'o-matrix',
      outputBuffers['hourWeekdayCounts'],
      'uint32',
      GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
    );
  }
  const props: GPUCalendarBucketsProps = {
    id: 'calendar',
    timestamps: importGraphBuffer(
      graph,
      'times',
      track(createInputBuffer(device, getInt64TimeWords(scene.timestamps))),
      'uint32x2',
      rows
    ),
    mask: scene.mask
      ? importGraphBuffer(
          graph,
          'mask',
          track(createInputBuffer(device, scene.mask)),
          'uint32',
          rows
        )
      : undefined,
    utcOffsets: scene.utcOffsets
      ? importGraphBuffer(
          graph,
          'offsets',
          track(createInputBuffer(device, scene.utcOffsets)),
          'sint32',
          rows
        )
      : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    output: output as GPUCalendarBucketsOutput
  };
  CountingCalendarBuckets.nodeBuilds = 0;
  graph.add(new CountingCalendarBuckets(props));
  const compiled = graph.compile();
  return {
    async run(offsetMinutes: number, firstDayOfWeek: number) {
      parameterBuffer.write(getGPUCalendarBucketsParameterValues(offsetMinutes, firstDayOfWeek));
      submitGraph(device, compiled, undefined);
      const result = {} as CalendarResult;
      for (const [name, , format] of COLUMNS) {
        if (outputBuffers[name]) {
          const values = await readUint32(outputBuffers[name], rows);
          result[name] = format === 'sint32' ? values.map(value => value | 0) : values;
        }
      }
      if (outputBuffers['hourWeekdayCounts']) {
        result.hourWeekdayCounts = await readUint32(
          outputBuffers['hourWeekdayCounts'],
          GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
        );
      }
      return result;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function getExpectedRows(
  scene: Scene,
  offsetMinutes: number,
  firstDayOfWeek: number
): CalendarFields[] {
  return Array.from(scene.timestamps, (milliseconds, row) =>
    scene.mask && scene.mask[row] === 0
      ? getInvalidCalendarFields()
      : decodeCalendarOnCPU(
          milliseconds,
          scene.utcOffsets ? scene.utcOffsets[row] : offsetMinutes,
          firstDayOfWeek
        )
  );
}

function expectParity(
  actual: CalendarResult,
  scene: Scene,
  offsetMinutes: number,
  firstDayOfWeek: number
): void {
  const expected = getExpectedRows(scene, offsetMinutes, firstDayOfWeek);
  for (const [name, field] of COLUMNS) {
    if (actual[name]) {
      const expectedColumn = expected.map(row => row[field]);
      // Exact equality; report the first mismatching row instead of a 10k-element diff.
      const mismatch = expectedColumn.findIndex((value, row) => actual[name][row] !== value);
      expect(
        mismatch === -1
          ? 'equal'
          : `${name} row ${mismatch} time ${scene.timestamps[mismatch]} offset ${offsetMinutes} got ${actual[name][mismatch]} want ${expectedColumn[mismatch]}`
      ).toBe('equal');
    }
  }
  if (actual.hourWeekdayCounts) {
    expect(actual.hourWeekdayCounts).toEqual(Array.from(countHourWeekdayOnCPU(expected)));
  }
}

const ALL_OUTPUTS = [...COLUMNS.map(([name]) => name), 'hourWeekdayCounts'];
const DAY = 86_400_000;

function createEdgeScene(): Scene {
  const edges = [
    0,
    -1,
    1,
    -DAY,
    -DAY - 1,
    Date.UTC(2000, 1, 29),
    Date.UTC(1900, 1, 28, 23, 59),
    Date.UTC(1900, 2, 1),
    Date.UTC(2100, 2, 1),
    Date.UTC(2020, 11, 31),
    Date.UTC(2021, 0, 3),
    Date.UTC(2024, 11, 30),
    Date.UTC(2020, 11, 31, 23, 30),
    Date.UTC(9999, 11, 31, 23, 59, 59, 999),
    new Date(0).setUTCFullYear(-1000, 0, 1),
    -62_167_219_200_000
  ].map(BigInt);
  edges.push(2n ** 63n - 1n, -(2n ** 63n), 2n ** 62n, -(2n ** 62n), 1_000_000_000n * 86_400_000n);
  return {timestamps: BigInt64Array.from(edges)};
}

it('GPUCalendarBuckets matches the BigInt oracle on adversarial edges for every output', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createEdgeScene();
  const fixture = createFixture(device, scene, ALL_OUTPUTS);
  for (const [offset, firstDay] of [
    [0, 0],
    [840, 6],
    [-720, 3],
    [330, 0],
    [-1440, 1],
    [1440, 5],
    [1, 0]
  ]) {
    expectParity(await fixture.run(offset, firstDay), scene, offset, firstDay);
  }
  // The compiled graph built its nodes once while the parameters changed seven times.
  expect(CountingCalendarBuckets.nodeBuilds).toBe(1);
  fixture.destroy();
});

it('GPUCalendarBuckets matches the oracle on 10k fuzzed rows with mask and per-row offsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 10_000;
  const next = createRandomWords(7);
  const scene: Scene = {
    timestamps: createRandomTimestamps(11, rows),
    mask: Uint32Array.from({length: rows}, (_, row) => (row % 9 === 4 ? 0 : 1)),
    // Mostly valid offsets with a few out-of-range ones that must come back invalid.
    utcOffsets: Int32Array.from({length: rows}, () => {
      const roll = next();
      return roll % 50 === 0 ? 1441 + (roll % 5) : (roll % 2881) - 1440;
    })
  };
  const fixture = createFixture(device, scene, ALL_OUTPUTS);
  for (const firstDay of [0, 6, 2]) {
    expectParity(await fixture.run(0, firstDay), scene, 0, firstDay);
  }
  fixture.destroy();
});

it('GPUCalendarBuckets matches the oracle on 10k fuzzed rows with a fixed offset', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: Scene = {timestamps: createRandomTimestamps(99, 10_000)};
  // A subset of outputs exercises the single-kernel path.
  const fixture = createFixture(device, scene, [
    'year',
    'isoWeek',
    'isoWeekYear',
    'hourWeekdayCounts'
  ]);
  for (const [offset, firstDay] of [
    [0, 0],
    [-300, 6],
    [545, 4]
  ]) {
    const actual = await fixture.run(offset, firstDay);
    expectParity(actual, scene, offset, firstDay);
  }
  fixture.destroy();
});

it('GPUCalendarBuckets matrix is cleared on every encoding and counts only valid rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: Scene = {
    // 2021-01-04 Monday 10:15 twice, one masked row, one out-of-range row.
    timestamps: BigInt64Array.from([
      BigInt(Date.UTC(2021, 0, 4, 10, 15)),
      BigInt(Date.UTC(2021, 0, 4, 10, 45)),
      BigInt(Date.UTC(2021, 0, 4, 10, 50)),
      2n ** 63n - 1n
    ]),
    mask: Uint32Array.from([1, 1, 0, 1])
  };
  const fixture = createFixture(device, scene, ['hourWeekdayCounts']);
  for (let encoding = 0; encoding < 3; encoding++) {
    const {hourWeekdayCounts} = await fixture.run(0, 0);
    expect(hourWeekdayCounts[10]).toBe(2);
    expect(hourWeekdayCounts.reduce((sum, count) => sum + count, 0)).toBe(2);
  }
  // Moving the zone to +14:00 shifts the rows to Tuesday 00:15.
  const shifted = (await fixture.run(840, 0)).hourWeekdayCounts;
  expect(shifted[1 * 24 + 0]).toBe(2);
  // Week starting on Sunday: Monday is weekday 1.
  expect((await fixture.run(0, 6)).hourWeekdayCounts[1 * 24 + 10]).toBe(2);
  fixture.destroy();
});
