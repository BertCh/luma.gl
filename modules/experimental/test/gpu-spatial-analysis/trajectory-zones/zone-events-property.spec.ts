// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runZoneEvents} from './zone-events-harness';
import {createRingEdges} from './zone-events-oracle';
import fixtureJson from './fixtures/zone-events-grid.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/zone-events-grid.json, written by test/oracles/geometry-properties.py. Twenty
// random tracks on an integer grid [0, 12]^2 and three integer-vertex zones, so touches, vertex hits
// and zero-length steps occur. Expected enter and exit counts and the final inside state come from an
// exact rational evaluation of the documented rule (sample (x, y) read as (x - e^2, y - e)) in python
// fractions, cross-checked against shapely for samples off the boundary; the rule is not taken from the
// TypeScript replay in zone-crossing-rule.ts. Time of sample k is k, so enter/exit counts are all that
// depends on the rule.

type Fixture = {
  zones: [number, number][][];
  tracks: {
    positions: [number, number][];
    zones: {initialInside: boolean; enters: number; exits: number; finalInside: boolean}[];
  }[];
};

const fixture: Fixture = fixtureJson as unknown as Fixture;

it('GPUZoneEvents enter/exit counts and final state equal the exact perturbed-point rule on a grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const zoneCount = fixture.zones.length;
  const trackRows: FixtureRow[][] = fixture.tracks.map(track =>
    track.positions.map(([x, y], row): FixtureRow => [x, y, row])
  );
  const tracks = packTracks(trackRows);
  const edges = createRingEdges(
    fixture.zones.map((ring, zone) => [zone, ring.flat()] as [number, number[]])
  );
  const totalEvents = fixture.tracks.reduce(
    (total, track) => total + track.zones.reduce((sum, zone) => sum + zone.enters + zone.exits, 0),
    0
  );
  expect(totalEvents).toBeGreaterThan(30);
  for (const spatialSort of [false, true]) {
    const label = spatialSort ? 'spatialSort' : 'default';
    const actual = await runZoneEvents(device, tracks, edges, {
      zoneCount,
      candidateCapacity: 16384,
      trackEventCapacity: 256,
      eventCapacity: 4096,
      spatialSort
    });
    expect(actual.candidateOverflow, `${label} candidate overflow`).toBe(0);
    expect(actual.trackOverflow, `${label} track overflow`).toBe(0);
    expect(actual.eventOverflow, `${label} event overflow`).toBe(0);
    expect(actual.overflow, `${label} overflow`).toBe(0);
    expect(actual.count, `${label} event count`).toBe(totalEvents);
    for (const [track, expectedTrack] of fixture.tracks.entries()) {
      let trackEvents = 0;
      for (const [zone, expected] of expectedTrack.zones.entries()) {
        const cell = `${label} track ${track} zone ${zone}`;
        // Events of this cell in output order (sorted by track, then time).
        const types: number[] = [];
        for (let row = 0; row < actual.count; row++) {
          if (actual.tracks[row] === track && actual.zones[row] === zone) {
            types.push(actual.types[row]);
          }
        }
        trackEvents += types.length;
        const enters = types.filter(type => type === 0).length;
        const exits = types.filter(type => type === 1).length;
        expect({enters, exits}, `${cell} enters/exits`).toEqual({
          enters: expected.enters,
          exits: expected.exits
        });
        // A zone alternates: the first event leaves when the track starts inside.
        types.forEach((type, index) => {
          expect(type, `${cell} alternation ${index}`).toBe(
            (index + (expected.initialInside ? 1 : 0)) % 2
          );
        });
        // Starting inside counts one visit in addition to the enters.
        expect(actual.visitCounts[track * zoneCount + zone], `${cell} visits`).toBe(
          (expected.initialInside ? 1 : 0) + expected.enters
        );
        const finalInside =
          types.length > 0 ? types[types.length - 1] === 0 : expected.initialInside;
        expect(finalInside, `${cell} final inside`).toBe(expected.finalInside);
      }
      expect(actual.trackEventCounts[track], `${label} track ${track} events`).toBe(trackEvents);
    }
  }
  device.destroy?.();
});
