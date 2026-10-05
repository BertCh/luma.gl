// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUCalendarBucketsParameterValues,
  GPUCalendarBuckets,
  GPU_CALENDAR_BUCKETS_INVALID_UINT32,
  GPU_CALENDAR_BUCKETS_INVALID_YEAR,
  type GPUCalendarBucketsProps
} from '../../../src/gpu-dataframe/calendar-buckets';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createRandomTimestamps,
  decodeCalendarOnCPU,
  decodeCalendarWithDate,
  type CalendarFields
} from './calendar-buckets-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCalendarBucketsProps> = {}
): GPUCalendarBucketsProps {
  const view = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    timestamps: view('uint32x2', 6),
    parameters: view('sint32', 2),
    output: {hour: view('uint32', 6), year: view('sint32', 6)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUCalendarBucketsProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUCalendarBuckets(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const view = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `v-${serial++}`, format, length);

const DAY = 86_400_000;

it('GPUCalendarBuckets parameter helper packs and validates', () => {
  expect(Array.from(getGPUCalendarBucketsParameterValues(-300, 6))).toEqual([-300, 6]);
  expect(Array.from(getGPUCalendarBucketsParameterValues(840))).toEqual([840, 0]);
  expect(() => getGPUCalendarBucketsParameterValues(1441)).toThrow(/offset/);
  expect(() => getGPUCalendarBucketsParameterValues(1.5)).toThrow(/offset/);
  expect(() => getGPUCalendarBucketsParameterValues(0, 7)).toThrow(/first day/);
  expect(() => getGPUCalendarBucketsParameterValues(0, 0, new Int32Array(1))).toThrow(/hold/);
});

it('GPUCalendarBuckets validates its inputs', () => {
  expectThrows(() => ({output: {}}), /at least one output/);
  expectThrows(graph => ({timestamps: view(graph, 'uint32x2', 0)}), /at least one row/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 5)}), /mask length/);
  expectThrows(graph => ({utcOffsets: view(graph, 'sint32', 5)}), /utcOffsets length/);
  expectThrows(graph => ({parameters: view(graph, 'sint32', 1)}), /parameters must hold/);
  expectThrows(graph => ({output: {hour: view(graph, 'uint32', 5)}}), /output.hour must hold/);
  expectThrows(
    graph => ({output: {year: view(graph, 'uint32', 6) as never}}),
    /output.year must be packed/
  );
  expectThrows(
    graph => ({output: {hourWeekdayCounts: view(graph, 'uint32', 100)}}),
    /hourWeekdayCounts must hold 168/
  );
  expectThrows(graph => {
    const shared = view(graph, 'uint32', 6);
    return {mask: shared, output: {hour: shared}};
  }, /./);
});

it('GPUCalendarBuckets splits outputs over kernels within the binding limit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const u32 = (length = 6) => view(graph, 'uint32', length);
  const output = {
    year: view(graph, 'sint32', 6),
    month: u32(),
    dayOfMonth: u32(),
    hour: u32(),
    minute: u32(),
    weekday: u32(),
    dayOfYear: u32(),
    isoWeek: u32(),
    isoWeekYear: view(graph, 'sint32', 6),
    quarter: u32(),
    hourWeekdayCounts: u32(168)
  };
  const contributor = new GPUCalendarBuckets({
    id: 'calendar',
    timestamps: view(graph, 'uint32x2', 6),
    mask: u32(),
    utcOffsets: view(graph, 'sint32', 6),
    parameters: view(graph, 'sint32', 2),
    output
  });
  const nodes = contributor.getCommandNodes(graph);
  // 4 inputs + 11 outputs at 4 outputs per kernel = 3 kernels, after the matrix fill.
  expect(nodes.map(node => node.id)).toEqual([
    'calendar-fill-matrix',
    'calendar-decode-0',
    'calendar-decode-1',
    'calendar-decode-2'
  ]);

  const minimal = new GPUCalendarBuckets(createProps(graph, {id: 'small'}));
  expect(minimal.getCommandNodes(graph).map(node => node.id)).toEqual(['small-decode-0']);
  device.destroy();
});

const EXPECTED = (
  fields: Partial<CalendarFields> & Pick<CalendarFields, 'year' | 'month' | 'day'>
): Partial<CalendarFields> => fields;

it('decodeCalendarOnCPU handles documented edge cases', () => {
  const cases: [string, bigint, number, Partial<CalendarFields>][] = [
    [
      'epoch',
      0n,
      0,
      EXPECTED({
        year: 1970,
        month: 1,
        day: 1,
        hour: 0,
        weekday: 3,
        dayOfYear: 1,
        isoWeek: 1
      })
    ],
    [
      '-1 ms',
      -1n,
      0,
      EXPECTED({
        year: 1969,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        weekday: 2,
        dayOfYear: 365,
        isoWeek: 1,
        isoWeekYear: 1970
      })
    ],
    [
      'leap day 2000',
      BigInt(Date.UTC(2000, 1, 29)),
      0,
      EXPECTED({
        year: 2000,
        month: 2,
        day: 29,
        dayOfYear: 60,
        weekday: 1,
        quarter: 1
      })
    ],
    [
      '1900 is not leap',
      BigInt(Date.UTC(1900, 2, 1)),
      0,
      EXPECTED({year: 1900, month: 3, day: 1, dayOfYear: 60})
    ],
    [
      '2100 is not leap',
      BigInt(Date.UTC(2100, 2, 1)),
      0,
      EXPECTED({year: 2100, month: 3, day: 1, dayOfYear: 60})
    ],
    [
      '2020-12-31 is week 53',
      BigInt(Date.UTC(2020, 11, 31)),
      0,
      EXPECTED({
        year: 2020,
        month: 12,
        day: 31,
        isoWeek: 53,
        isoWeekYear: 2020,
        quarter: 4
      })
    ],
    [
      '2021-01-03 is week 53 of 2020',
      BigInt(Date.UTC(2021, 0, 3)),
      0,
      EXPECTED({
        year: 2021,
        month: 1,
        day: 3,
        isoWeek: 53,
        isoWeekYear: 2020,
        weekday: 6
      })
    ],
    [
      '2024-12-30 is week 1 of 2025',
      BigInt(Date.UTC(2024, 11, 30)),
      0,
      EXPECTED({
        year: 2024,
        month: 12,
        day: 30,
        isoWeek: 1,
        isoWeekYear: 2025
      })
    ],
    [
      '+14:00 crosses the year boundary',
      BigInt(Date.UTC(2020, 11, 31, 23, 30)),
      840,
      EXPECTED({
        year: 2021,
        month: 1,
        day: 1,
        hour: 13,
        minute: 30,
        isoWeek: 53
      })
    ],
    [
      '-12:00 crosses back',
      BigInt(Date.UTC(2021, 0, 1, 0, 30)),
      -720,
      EXPECTED({year: 2020, month: 12, day: 31, hour: 12, minute: 30})
    ],
    [
      'year 9999 end',
      BigInt(Date.UTC(9999, 11, 31, 23, 59, 59, 999)),
      0,
      EXPECTED({
        year: 9999,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        dayOfYear: 365
      })
    ]
  ];
  for (const [name, milliseconds, offset, expected] of cases) {
    expect(decodeCalendarOnCPU(milliseconds, offset), name).toMatchObject(expected);
  }
  // Sunday first: Sunday is weekday 0 and Monday is 1.
  expect(decodeCalendarOnCPU(BigInt(Date.UTC(2021, 0, 3)), 0, 6).weekday).toBe(0);
  expect(decodeCalendarOnCPU(BigInt(Date.UTC(2021, 0, 4)), 0, 6).weekday).toBe(1);
});

it('decodeCalendarOnCPU marks out-of-range rows invalid', () => {
  const maximum = 2n ** 63n - 1n;
  for (const [milliseconds, offset] of [
    [maximum, 1],
    [-(2n ** 63n), -1],
    [2n ** 62n, 0],
    [0n, 1441]
  ] as const) {
    const fields = decodeCalendarOnCPU(milliseconds, offset);
    expect(fields.valid).toBe(false);
    expect(fields.hour).toBe(GPU_CALENDAR_BUCKETS_INVALID_UINT32);
    expect(fields.year).toBe(GPU_CALENDAR_BUCKETS_INVALID_YEAR);
  }
  // The extremes of the supported day range stay valid at the edge.
  expect(decodeCalendarOnCPU(1_000_000_000n * 86_400_000n, 0).valid).toBe(true);
  expect(decodeCalendarOnCPU(-1_000_000_000n * 86_400_000n, 0).valid).toBe(true);
});

it('decodeCalendarOnCPU agrees with JS Date on edges and random fixed offsets', () => {
  const edges = [
    0,
    -1,
    1,
    -DAY,
    -DAY - 1,
    Date.UTC(2000, 1, 29, 12),
    Date.UTC(1900, 1, 28, 23, 59, 59, 999),
    Date.UTC(2100, 11, 31, 23, 59, 59, 999),
    Date.UTC(9999, 11, 31, 23, 59, 59, 999),
    new Date(0).setUTCFullYear(-1000, 0, 1),
    new Date(0).setUTCFullYear(0, 0, 1),
    -62_167_219_200_000,
    8_600_000_000_000_000,
    -8_600_000_000_000_000
  ];
  const offsets = [0, 840, -720, 330, -210, 1440, -1440, 1];
  for (const milliseconds of edges) {
    for (const offset of offsets) {
      for (const firstDay of [0, 6, 3]) {
        expect(
          decodeCalendarOnCPU(BigInt(milliseconds), offset, firstDay),
          `${milliseconds} ${offset} ${firstDay}`
        ).toEqual(decodeCalendarWithDate(milliseconds, offset, firstDay));
      }
    }
  }
  const random = createRandomTimestamps(42, 10_000);
  let compared = 0;
  for (let row = 0; row < random.length; row++) {
    const milliseconds = Number(random[row]);
    const offset = ((row * 37) % 2881) - 1440;
    if (Math.abs(milliseconds) > 8.6e15) {
      continue;
    }
    compared++;
    expect(decodeCalendarOnCPU(random[row], offset, row % 7)).toEqual(
      decodeCalendarWithDate(milliseconds, offset, row % 7)
    );
  }
  expect(compared).toBeGreaterThan(5000);
});
