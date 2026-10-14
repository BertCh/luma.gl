// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTimeZoneTransitionTable,
  GPUCalendarBuckets
} from '../../../src/gpu-dataframe/calendar-buckets';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {decodeCalendarWithTableOnCPU} from './calendar-buckets-oracle';
import {TIME_ZONE_REFERENCE, type TimeZoneReference} from './time-zone-reference';

// Reference: Python zoneinfo (generator test/oracles/dataframe-time-zones.py). The helper reads
// Node's bundled ICU, which can be another tz database release than the fixture's. The windows
// below are asserted exactly; a difference is reported with the zone and instant.
const JANUARY_1_1970 = 0;
const JANUARY_1_2040 = Date.UTC(2040, 0, 1);
// Africa/Casablanca lists its Ramadan transitions explicitly and the fixture's tz database
// (2026d) ends 2026 on permanent +00:00 from 2026-09-20 while Node's ICU keeps predicting
// Ramadan shifts, so that zone is compared up to 2026-06-01. Every other zone matches over the
// whole fixture range 1900 to 2040, transition for transition.
const EXACT_END: Record<string, number> = {'Africa/Casablanca': Date.UTC(2026, 5, 1)};
const getExactEnd = (zone: string) => EXACT_END[zone] ?? JANUARY_1_2040;

function toBigInts(words: Uint32Array): bigint[] {
  const signed = new BigInt64Array(words.buffer, words.byteOffset, words.length / 2);
  return Array.from(signed);
}

function getReferenceWindow(reference: TimeZoneReference, start: number, end: number) {
  const transitions: number[] = [];
  const offsets: number[] = [];
  reference.transitions.forEach((transition, index) => {
    if (transition > start && transition <= end) {
      if (transitions.length === 0) {
        offsets.push(reference.offsets[index]);
      }
      transitions.push(transition);
      offsets.push(reference.offsets[index + 1]);
    }
  });
  return {transitions, offsets};
}

const ZONES = Object.keys(TIME_ZONE_REFERENCE.zones);

it('time zone fixture records its generator and database', () => {
  expect(TIME_ZONE_REFERENCE.generator).toBe('test/oracles/dataframe-time-zones.py');
  expect(TIME_ZONE_REFERENCE.tzDatabase.length).toBeGreaterThan(0);
  expect(ZONES).toContain('America/New_York');
});

it('getGPUTimeZoneTransitionTable matches zoneinfo exactly for 1970 to 2040 (Casablanca to mid 2026)', () => {
  for (const zone of ZONES) {
    if (zone === 'UTC') {
      continue;
    }
    const expected = getReferenceWindow(
      TIME_ZONE_REFERENCE.zones[zone],
      JANUARY_1_1970,
      getExactEnd(zone)
    );
    const table = getGPUTimeZoneTransitionTable(zone, {
      start: JANUARY_1_1970,
      end: getExactEnd(zone)
    });
    // Leading offset: the reference offset in force at the window start.
    const reference = TIME_ZONE_REFERENCE.zones[zone];
    const startIndex = reference.transitions.filter(time => time <= JANUARY_1_1970).length;
    const expectedOffsets =
      expected.transitions.length > 0
        ? [reference.offsets[startIndex], ...expected.offsets.slice(1)]
        : [reference.offsets[startIndex], reference.offsets[startIndex]];
    expect({zone, transitions: toBigInts(table.transitions).map(Number)}).toEqual({
      zone,
      transitions: expected.transitions.length > 0 ? expected.transitions : [-(2 ** 63)]
    });
    expect({zone, offsets: Array.from(table.offsets)}).toEqual({zone, offsets: expectedOffsets});
  }
});

it('getGPUTimeZoneTransitionTable matches zoneinfo for 1900 to 1970 including mean time seconds', () => {
  // Offsets with seconds (Asia/Kathmandu +05:41:16, Europe/Dublin -00:25:21) are asserted
  // exactly: V8 formats longOffset with seconds, so no rounding to minutes occurs.
  for (const zone of ['Asia/Kathmandu', 'Europe/Dublin']) {
    const reference = TIME_ZONE_REFERENCE.zones[zone];
    const table = getGPUTimeZoneTransitionTable(zone, {
      start: TIME_ZONE_REFERENCE.rangeStart,
      end: JANUARY_1_1970
    });
    const expected = getReferenceWindow(reference, TIME_ZONE_REFERENCE.rangeStart, JANUARY_1_1970);
    expect(toBigInts(table.transitions).map(Number)).toEqual(expected.transitions);
    expect(Array.from(table.offsets).slice(1)).toEqual(expected.offsets.slice(1));
    expect(table.offsets[0]).toBe(reference.offsets[0]);
  }
  expect(
    getGPUTimeZoneTransitionTable('Asia/Kathmandu', {
      start: TIME_ZONE_REFERENCE.rangeStart,
      end: Date.UTC(1930, 0, 1)
    }).offsets[0]
  ).toBe(5 * 3600 + 41 * 60 + 16);
});

it('getGPUTimeZoneTransitionTable matches zoneinfo over 1900 to 2040 for rule-based zones', () => {
  for (const zone of ZONES.filter(name => name !== 'UTC' && name !== 'Africa/Casablanca')) {
    const reference = TIME_ZONE_REFERENCE.zones[zone];
    const table = getGPUTimeZoneTransitionTable(zone, {
      start: TIME_ZONE_REFERENCE.rangeStart,
      end: TIME_ZONE_REFERENCE.rangeEnd
    });
    expect({zone, transitions: toBigInts(table.transitions).map(Number)}).toEqual({
      zone,
      transitions: reference.transitions
    });
    expect({zone, offsets: Array.from(table.offsets)}).toEqual({zone, offsets: reference.offsets});
  }
});

it('getGPUTimeZoneTransitionTable gives a sentinel table for zones without transitions', () => {
  const table = getGPUTimeZoneTransitionTable('UTC', {start: 0, end: JANUARY_1_2040});
  expect(toBigInts(table.transitions)).toEqual([-(2n ** 63n)]);
  expect(Array.from(table.offsets)).toEqual([0, 0]);
});

it('getGPUTimeZoneTransitionTable validates its inputs', () => {
  expect(() => getGPUTimeZoneTransitionTable('Mars/Olympus', {start: 0, end: 1})).toThrow(
    /Unknown time zone/
  );
  expect(() => getGPUTimeZoneTransitionTable('UTC', {start: 5, end: 1})).toThrow(/range/);
  expect(() => getGPUTimeZoneTransitionTable('UTC', {start: NaN, end: 1})).toThrow(/range/);
});

it('the table CPU decode oracle equals the zoneinfo local fields around every transition', () => {
  const mismatches: string[] = [];
  let compared = 0;
  for (const [zone, reference] of Object.entries(TIME_ZONE_REFERENCE.zones)) {
    const transitions = reference.transitions.length
      ? reference.transitions.map(BigInt)
      : [-(2n ** 63n)];
    const offsets = reference.transitions.length
      ? reference.offsets
      : [reference.offsets[0], reference.offsets[0]];
    for (const sample of reference.samples) {
      const fields = decodeCalendarWithTableOnCPU(BigInt(sample.time), transitions, offsets, 0);
      compared++;
      const actual = [
        fields.year,
        fields.month,
        fields.day,
        fields.hour,
        fields.minute,
        fields.weekday,
        fields.dayOfYear,
        fields.isoWeek,
        fields.isoWeekYear,
        fields.quarter
      ];
      const want = [
        sample.year,
        sample.month,
        sample.dayOfMonth,
        sample.hour,
        sample.minute,
        sample.weekday,
        sample.dayOfYear,
        sample.isoWeek,
        sample.isoWeekYear,
        sample.quarter
      ];
      if (actual.join() !== want.join()) {
        mismatches.push(`${zone} ${sample.time}: got ${actual.join()} want ${want.join()}`);
      }
    }
  }
  expect(compared).toBeGreaterThan(3000);
  expect(mismatches).toEqual([]);
});

it('a helper table drives the CPU decode to the zoneinfo fields for 1970 to 2040', () => {
  for (const zone of ['America/New_York', 'Australia/Lord_Howe', 'Africa/Casablanca']) {
    const table = getGPUTimeZoneTransitionTable(zone, {
      start: JANUARY_1_1970,
      end: getExactEnd(zone)
    });
    const transitions = toBigInts(table.transitions);
    for (const sample of TIME_ZONE_REFERENCE.zones[zone].samples) {
      if (sample.time < JANUARY_1_1970 || sample.time > getExactEnd(zone)) {
        continue;
      }
      const fields = decodeCalendarWithTableOnCPU(BigInt(sample.time), transitions, table.offsets);
      expect({zone, time: sample.time, hour: fields.hour, minute: fields.minute}).toEqual({
        zone,
        time: sample.time,
        hour: sample.hour,
        minute: sample.minute
      });
    }
  }
});

it('GPUCalendarBuckets validates the timeZone input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `tz-${serial++}`, format, length);
  const create = (overrides: object) =>
    new GPUCalendarBuckets({
      timestamps: view('uint32x2', 6),
      parameters: view('sint32', 2),
      output: {hour: view('uint32', 6)},
      timeZone: {transitions: view('uint32x2', 3), offsets: view('sint32', 4)},
      ...overrides
    });
  expect(() => create({})).not.toThrow();
  expect(() => create({utcOffsets: view('sint32', 6)})).toThrow(/mutually exclusive/);
  expect(() =>
    create({timeZone: {transitions: view('uint32x2', 3), offsets: view('sint32', 3)}})
  ).toThrow(/transitions length \+ 1/);
  expect(() =>
    create({timeZone: {transitions: view('uint32', 3), offsets: view('sint32', 4)}})
  ).toThrow(/timeZone.transitions/);
  device.destroy();
});

it('GPUCalendarBuckets table length changes only the declarations, not the node count', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'sint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `tzn-${serial++}`, format, length);
  const build = (count: number) =>
    new GPUCalendarBuckets({
      id: `tz${count}`,
      timestamps: view('uint32x2', 6),
      parameters: view('sint32', 2),
      output: {hour: view('uint32', 6), year: view('sint32', 6)},
      timeZone: {transitions: view('uint32x2', count), offsets: view('sint32', count + 1)}
    }).getCommandNodes(graph);
  // The binary search is a data-driven loop: the table size is a documented topology constant
  // and no loop index reaches the WGSL text, so both builds make the same single kernel.
  expect(build(200).map(node => node.id.replace(/^tz\d+/, ''))).toEqual(
    build(201).map(node => node.id.replace(/^tz\d+/, ''))
  );
  device.destroy();
});
