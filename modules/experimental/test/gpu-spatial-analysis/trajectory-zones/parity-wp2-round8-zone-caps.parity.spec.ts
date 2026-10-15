// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runZoneEvents} from './zone-events-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-round8-zone-caps.json';
const fixture = decodeOracleFixture<
  Omit<typeof fixtureJson, 'cases'> & {
    cases: (Omit<(typeof fixtureJson.cases)[number], 'tracks'> & {tracks: FixtureRow[][]})[];
  }
>(fixtureJson);
for (const entry of fixture.cases)
  for (const wordMode of [false, true])
    for (const spatialSort of [false, true]) {
      it(`showcase zone capacities GEOS words=${wordMode}/sort=${spatialSort}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const tracks = packTracks(entry.tracks, wordMode ? 1700000000000n : undefined);
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
        try {
          for (const trackEventCapacity of [4, 8, 16, 32, 64, 128]) {
            const expected = entry.tracks.flatMap((_, track) =>
              entry.expected.events.filter(event => event[0] === track).slice(0, trackEventCapacity)
            );
            const actual = await runZoneEvents(device, tracks, edges, {
              zoneCount: entry.zoneCount,
              candidateCapacity: 4096,
              trackEventCapacity,
              eventCapacity: entry.tracks.length * trackEventCapacity + 1,
              spatialSort
            });
            const label = `${entry.name}/words=${wordMode}/sort=${spatialSort}/cap=${trackEventCapacity}`;
            const compare = (
              metric: string,
              measured: number[],
              reference: number[],
              absolute = 0
            ) =>
              compareParity(
                'GPUZoneEvents',
                label,
                fixture.reference,
                metric,
                measured,
                reference,
                absolute
              );
            compare(
              'count/required/event-overflow/track-overflow',
              [actual.count, actual.requiredCount, actual.eventOverflow, actual.trackOverflow],
              [expected.length, expected.length, 0, 1]
            );
            compare(
              'every retained event identity',
              Array.from({length: actual.count}, (_, index) => [
                actual.tracks[index],
                actual.zones[index],
                actual.types[index],
                actual.rows[index]
              ]).flat(),
              expected.flatMap(event => [event[0], event[1], event[2], event[6]])
            );
            compare(
              'every retained event time',
              actual.times.slice(0, actual.count),
              expected.map(event => event[3]),
              entry.timeAbsolute
            );
            compare(
              'every retained position',
              actual.positions.slice(0, actual.count * 2),
              expected.flatMap(event => event.slice(4, 6)),
              entry.positionAbsolute
            );
            compare(
              'complete visits despite event cap',
              actual.visitCounts,
              entry.expected.visitCounts
            );
            compare(
              'complete dwell despite event cap',
              actual.dwellTimes,
              entry.expected.dwellTimes,
              entry.timeAbsolute
            );
            compare(
              'unclamped per-track counts',
              actual.trackEventCounts,
              entry.expected.trackEventCounts
            );
          }
        } finally {
          device.destroy?.();
        }
      }, 120000);
    }
