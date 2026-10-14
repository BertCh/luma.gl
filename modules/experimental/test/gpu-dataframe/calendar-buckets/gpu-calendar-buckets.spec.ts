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
  decodeCalendarWithTableOnCPU,
  getInvalidCalendarFields,
  type CalendarFields
} from './calendar-buckets-oracle';
import {TIME_ZONE_REFERENCE} from './time-zone-reference';
import {requireTestDevice} from '../../utils/require-test-device';

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
  /** Transition words (`uint32x2` rows) and offsets in seconds, one more than transitions. */
  timeZone?: {transitions: Uint32Array; offsets: Int32Array};
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
    timeZone: scene.timeZone
      ? {
          transitions: importGraphBuffer(
            graph,
            'zone-transitions',
            track(createInputBuffer(device, scene.timeZone.transitions)),
            'uint32x2',
            scene.timeZone.transitions.length / 2
          ),
          offsets: importGraphBuffer(
            graph,
            'zone-offsets',
            track(createInputBuffer(device, scene.timeZone.offsets)),
            'sint32',
            scene.timeZone.offsets.length
          )
        }
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
  const zoneTransitions = scene.timeZone
    ? Array.from(
        new BigInt64Array(
          scene.timeZone.transitions.buffer,
          scene.timeZone.transitions.byteOffset,
          scene.timeZone.transitions.length / 2
        )
      )
    : [];
  return Array.from(scene.timestamps, (milliseconds, row) =>
    scene.mask && scene.mask[row] === 0
      ? getInvalidCalendarFields()
      : scene.timeZone
        ? decodeCalendarWithTableOnCPU(
            milliseconds,
            zoneTransitions,
            scene.timeZone.offsets,
            firstDayOfWeek
          )
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
  requireTestDevice(device);
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
  requireTestDevice(device);
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
  requireTestDevice(device);
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
  requireTestDevice(device);
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

/** Table of a fixture zone; a zone without transitions gets the Int64 minimum sentinel. */
function getReferenceTable(zone: string): NonNullable<Scene['timeZone']> {
  const reference = TIME_ZONE_REFERENCE.zones[zone];
  if (reference.transitions.length === 0) {
    return {
      transitions: getInt64TimeWords(BigInt64Array.of(-(2n ** 63n))),
      offsets: Int32Array.of(reference.offsets[0], reference.offsets[0])
    };
  }
  return {
    transitions: getInt64TimeWords(BigInt64Array.from(reference.transitions, BigInt)),
    offsets: Int32Array.from(reference.offsets)
  };
}

it('GPUCalendarBuckets time zone table equals the zoneinfo fields around every transition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fieldNames = COLUMNS.map(([name, field]) => [name, field] as const);
  for (const zone of Object.keys(TIME_ZONE_REFERENCE.zones)) {
    const samples = TIME_ZONE_REFERENCE.zones[zone].samples;
    const scene: Scene = {
      timestamps: BigInt64Array.from(samples, sample => BigInt(sample.time)),
      timeZone: getReferenceTable(zone)
    };
    const fixture = createFixture(
      device,
      scene,
      COLUMNS.map(([name]) => name)
    );
    // First day Monday: the fixture weekday is Python's Monday = 0.
    const actual = await fixture.run(0, 0);
    for (const [name, field] of fieldNames) {
      const reference = field === 'day' ? 'dayOfMonth' : field;
      const mismatch = samples.findIndex(
        (sample, row) => actual[name][row] !== (sample as Record<string, number>)[reference]
      );
      expect(
        mismatch === -1
          ? 'equal'
          : `${zone} ${name} time ${samples[mismatch].time} got ${actual[name][mismatch]} want ${
              (samples[mismatch] as Record<string, number>)[reference]
            }`
      ).toBe('equal');
    }
    fixture.destroy();
  }
});

it('GPUCalendarBuckets time zone table matches the oracle on fuzzed rows, mask and bad offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 10_000;
  const base = getReferenceTable('America/New_York');
  // One out-of-bound offset (86401 s) in the table makes the rows after that transition invalid.
  const badOffsets = Int32Array.from(base.offsets);
  badOffsets[badOffsets.length - 1] = 86_401;
  const scenes: Scene[] = [
    {timestamps: createRandomTimestamps(21, rows), timeZone: base},
    {
      timestamps: createRandomTimestamps(22, rows),
      mask: Uint32Array.from({length: rows}, (_, row) => (row % 7 === 3 ? 0 : 1)),
      timeZone: base
    },
    {timestamps: createRandomTimestamps(23, rows), timeZone: {...base, offsets: badOffsets}}
  ];
  for (const scene of scenes) {
    const fixture = createFixture(device, scene, ALL_OUTPUTS);
    for (const firstDay of [0, 5]) {
      expectParity(await fixture.run(0, firstDay), scene, 0, firstDay);
    }
    fixture.destroy();
  }
});

it('GPUCalendarBuckets time zone table handles the Int64 extremes and one transition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const instants = [
    -(2n ** 63n),
    2n ** 63n - 1n,
    0n,
    -1n,
    1_000_000_000n * 86_400_000n,
    -(2n ** 62n)
  ];
  // A single transition at the epoch: before it +01:00:30, from it -05:00:00.
  const scene: Scene = {
    timestamps: BigInt64Array.from(instants),
    timeZone: {
      transitions: getInt64TimeWords(BigInt64Array.of(0n)),
      offsets: Int32Array.of(3630, -18_000)
    }
  };
  const fixture = createFixture(device, scene, ALL_OUTPUTS);
  expectParity(await fixture.run(0, 0), scene, 0, 0);
  fixture.destroy();
});

// A/B (same run, same GPU): per-row offsets column (the old DST path, offsets precomputed on the
// CPU) against the transition table, over 1e7 rows. The offsets column costs a CPU pass plus an
// 4 B per row upload; the table costs a binary search of ~8 steps per row. Results must be equal.
it('GPUCalendarBuckets A/B per-row offsets column vs time zone table at 1e7 rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 10_000_000;
  const table = getReferenceTable('America/New_York');
  const transitions = Array.from(
    new BigInt64Array(
      table.transitions.buffer,
      table.transitions.byteOffset,
      table.transitions.length / 2
    )
  );
  const next = createRandomWords(5);
  const start = BigInt(Date.UTC(1970, 0, 1));
  const span = BigInt(Date.UTC(2040, 0, 1)) - start;
  const timestamps = new BigInt64Array(rows);
  for (let row = 0; row < rows; row++) {
    timestamps[row] = start + (((BigInt(next()) << 16n) ^ BigInt(next())) % span);
  }
  const columns = ['hour', 'weekday'];
  const cpuStart = performance.now();
  const utcOffsets = new Int32Array(rows);
  for (let row = 0; row < rows; row++) {
    let low = 0;
    let high = transitions.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (transitions[middle] <= timestamps[row]) low = middle + 1;
      else high = middle;
    }
    utcOffsets[row] = table.offsets[low] / 60;
  }
  const cpuMilliseconds = performance.now() - cpuStart;
  const column = createFixture(device, {timestamps, utcOffsets}, columns);
  const zone = createFixture(device, {timestamps, timeZone: table}, columns);
  // Warm up both pipelines, then time several submits each including the readback.
  const expected = await column.run(0, 0);
  const actual = await zone.run(0, 0);
  expect(actual['hour'].every((value, row) => value === expected['hour'][row])).toBe(true);
  expect(actual['weekday'].every((value, row) => value === expected['weekday'][row])).toBe(true);
  const time = async (fixture: typeof column) => {
    const begin = performance.now();
    for (let repeat = 0; repeat < 5; repeat++) {
      await fixture.run(0, 0);
    }
    return (performance.now() - begin) / 5;
  };
  const columnMilliseconds = await time(column);
  const zoneMilliseconds = await time(zone);
  console.log(
    `calendar A/B 1e7 rows: offsets column ${columnMilliseconds.toFixed(1)} ms (+ ${cpuMilliseconds.toFixed(0)} ms CPU offsets), time zone table ${zoneMilliseconds.toFixed(1)} ms`
  );
  column.destroy();
  zone.destroy();
});
