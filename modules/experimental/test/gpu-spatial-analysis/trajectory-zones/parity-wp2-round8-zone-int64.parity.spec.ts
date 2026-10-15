// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runZoneEvents, type ZoneGPUResult} from './zone-events-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp2-assert';
import fixtureJson from '../../oracles/fixtures/parity-wp2-zone-events.json';

type EventRow = [
  track: number,
  zone: number,
  type: number,
  time: number,
  x: number,
  y: number,
  row: number
];
type ZoneCase = {
  name: string;
  tracks: FixtureRow[][];
  zoneCount: number;
  polygons: {
    positions: number[];
    featureOffsets: number[];
    polygonOffsets: number[];
    ringOffsets: number[];
  };
  positionAbsolute: number;
  timeAbsolute: number;
  expected: {
    events: EventRow[];
    intervals: number[][];
    table: number[][];
    dwellTimes: number[];
    visitCounts: number[];
    trackEventCounts: number[];
  };
};
const fixture = decodeOracleFixture<{reference: string; cases: ZoneCase[]}>(fixtureJson);

function getEvents(actual: ZoneGPUResult, entry: ZoneCase, canonical = true): EventRow[] {
  const offsets = [0];
  for (const track of entry.tracks) offsets.push(offsets[offsets.length - 1] + track.length);
  const getParameter = (event: EventRow) => {
    const row = event[6] - offsets[event[0]];
    const samples = entry.tracks[event[0]];
    const previous = samples[row - 1];
    const current = samples[row];
    if (!previous || !current) return 0;
    const differenceX = current[0] - previous[0];
    const differenceY = current[1] - previous[1];
    if (Math.abs(differenceX) >= Math.abs(differenceY)) {
      return differenceX === 0 ? 0 : (event[4] - previous[0]) / differenceX;
    }
    return (event[5] - previous[1]) / differenceY;
  };
  // Canonical path order remains defined when several crossings share a timestamp.
  const events = Array.from(
    {length: actual.count},
    (_, index): EventRow => [
      actual.tracks[index],
      actual.zones[index],
      actual.types[index],
      actual.times[index],
      actual.positions[index * 2],
      actual.positions[index * 2 + 1],
      actual.rows[index]
    ]
  );
  if (!canonical) return events;
  return events.sort(
    (first, second) =>
      first[0] - second[0] ||
      first[1] - second[1] ||
      first[6] - second[6] ||
      getParameter(first) - getParameter(second) ||
      first[2] - second[2]
  );
}

for (const entry of fixture.cases.filter(value => value.tracks.length < 100)) {
  it(`GPUZoneEvents Int64 Shapely parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const compare = (
      metric: string,
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
      absolute = 0,
      relative = 0
    ) =>
      compareParity(
        'GPUZoneEvents',
        entry.name,
        fixture.reference,
        metric,
        actual,
        expected,
        absolute,
        relative
      );
    const tracks = packTracks(entry.tracks);
    // Shift by an epoch above f32 integer precision; preserve exact source time differences.
    tracks.times = {
      kind: 'words',
      values: entry.tracks.flatMap(track => track.map(row => 1700000000000n + BigInt(row[2])))
    };
    // Only the polygon layout is consumed by this existing GPU execution harness.
    const edges = {
      starts: new Float32Array(0),
      ends: new Float32Array(0),
      zones: new Uint32Array(0),
      polygons: {
        positions: Float32Array.from(entry.polygons.positions),
        featureOffsets: Uint32Array.from(entry.polygons.featureOffsets),
        polygonOffsets: Uint32Array.from(entry.polygons.polygonOffsets),
        ringOffsets: Uint32Array.from(entry.polygons.ringOffsets)
      }
    };
    const options = {
      zoneCount: entry.zoneCount,
      candidateCapacity: entry.tracks.length > 100 ? 1 << 18 : 1 << 12,
      trackEventCapacity: Math.max(...entry.expected.trackEventCounts) + 1,
      eventCapacity: entry.expected.events.length + 1,
      tableCapacity: entry.expected.table.length + 1
    };
    try {
      for (const spatialSort of [false, true]) {
        const actual = await runZoneEvents(device, tracks, edges, {...options, spatialSort});
        compare(
          `status sorted=${spatialSort}`,
          [
            actual.count,
            actual.requiredCount,
            actual.overflow,
            actual.candidateOverflow,
            actual.trackOverflow,
            actual.eventOverflow,
            actual.table.count,
            actual.table.requiredCount,
            actual.table.overflow
          ],
          [
            entry.expected.events.length,
            entry.expected.events.length,
            0,
            0,
            0,
            0,
            entry.expected.table.length,
            entry.expected.table.length,
            0
          ]
        );
        const events = getEvents(actual, entry);
        const publicEvents = getEvents(actual, entry, false);
        compare(
          'event track/zone/type/segment',
          events.flatMap(event => [event[0], event[1], event[2], event[6]]),
          entry.expected.events.flatMap(event => [event[0], event[1], event[2], event[6]])
        );
        compare(
          'every crossing time',
          events.map(event => event[3]),
          entry.expected.events.map(event => event[3]),
          entry.timeAbsolute,
          2e-6
        );
        compare(
          'every crossing position',
          events.flatMap(event => event.slice(4, 6)),
          entry.expected.events.flatMap(event => event.slice(4, 6)),
          entry.positionAbsolute,
          2e-6
        );
        compare('dense visit counts', actual.visitCounts, entry.expected.visitCounts);
        compare('per-track event counts', actual.trackEventCounts, entry.expected.trackEventCounts);
        compare(
          'dense dwell seconds',
          actual.dwellTimes,
          entry.expected.dwellTimes,
          entry.timeAbsolute,
          2e-6
        );
        const table = actual.table;
        compare(
          'sparse visit identities/counts',
          Array.from({length: table.count}, (_, index) => [
            table.tracks[index],
            table.zones[index],
            table.visits[index]
          ]).flat(),
          entry.expected.table.flatMap(row => row.slice(0, 3))
        );
        compare(
          'sparse dwell/first/last',
          Array.from({length: table.count}, (_, index) => [
            table.dwellTimes[index],
            table.firstEnterTimes[index],
            table.lastExitTimes[index]
          ]).flat(),
          entry.expected.table.flatMap(row => row.slice(3)),
          entry.timeAbsolute,
          2e-6
        );
        // Reconstruct every maximal interval from readback, including open start/end visits.
        const intervals: number[][] = [];
        for (let track = 0; track < entry.tracks.length; track++) {
          const samples = entry.tracks[track];
          for (let zone = 0; zone < entry.zoneCount; zone++) {
            if (actual.visitCounts[track * entry.zoneCount + zone] === 0) continue;
            // Types alternate in public (track,time,edge) order, including tied times.
            const cellEvents = publicEvents.filter(
              event => event[0] === track && event[1] === zone
            );
            let entered: number | undefined =
              cellEvents.length === 0 || cellEvents[0][2] === 1 ? 0 : undefined;
            for (const event of cellEvents) {
              if (event[2] === 0) entered = event[3];
              else {
                intervals.push([track, zone, entered ?? NaN, event[3]]);
                entered = undefined;
              }
            }
            if (entered !== undefined)
              intervals.push([
                track,
                zone,
                entered,
                samples[samples.length - 1][2] - samples[0][2]
              ]);
          }
        }
        compare(
          'interval identities',
          intervals.flatMap(interval => interval.slice(0, 2)),
          entry.expected.intervals.flatMap(interval => interval.slice(0, 2))
        );
        compare(
          'every entry/exit interval',
          intervals.flatMap(interval => interval.slice(2)),
          entry.expected.intervals.flatMap(interval => interval.slice(2)),
          entry.timeAbsolute,
          2e-6
        );
        compare(
          'event/table tail sentinels',
          [
            actual.tracks[actual.count],
            actual.zones[actual.count],
            actual.rows[actual.count],
            table.tracks[table.count],
            table.zones[table.count],
            table.visits[table.count]
          ],
          [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]
        );
      }
      const bounded = await runZoneEvents(device, tracks, edges, {
        ...options,
        eventCapacity: 1,
        tableCapacity: 1
      });
      compare(
        'bounded event/table status',
        [
          bounded.count,
          bounded.requiredCount,
          bounded.overflow,
          bounded.eventOverflow,
          bounded.candidateOverflow,
          bounded.trackOverflow,
          bounded.table.count,
          bounded.table.requiredCount,
          bounded.table.overflow
        ],
        [1, entry.expected.events.length, 1, 1, 0, 0, 1, entry.expected.table.length, 1]
      );
      compare('bounded dense visits', bounded.visitCounts, entry.expected.visitCounts);
      compare(
        'bounded dense dwell',
        bounded.dwellTimes,
        entry.expected.dwellTimes,
        entry.timeAbsolute,
        2e-6
      );
      if (entry.tracks.length < 100) {
        const trackLimited = await runZoneEvents(device, tracks, edges, {
          ...options,
          trackEventCapacity: 1
        });
        const keptCount = entry.expected.trackEventCounts.filter(count => count > 0).length;
        compare(
          'per-track capacity status',
          [
            trackLimited.count,
            trackLimited.requiredCount,
            trackLimited.overflow,
            trackLimited.trackOverflow,
            trackLimited.eventOverflow,
            trackLimited.candidateOverflow
          ],
          [keptCount, keptCount, 1, 1, 0, 0]
        );
        compare(
          'per-track bounded visit counts',
          trackLimited.visitCounts,
          entry.expected.visitCounts
        );
        compare(
          'per-track bounded dwell',
          trackLimited.dwellTimes,
          entry.expected.dwellTimes,
          entry.timeAbsolute,
          2e-6
        );
      }
    } finally {
      device.destroy?.();
    }
  }, 120_000);
}
