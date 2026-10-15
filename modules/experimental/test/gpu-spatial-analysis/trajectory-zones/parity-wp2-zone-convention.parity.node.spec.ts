// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/parity-wp2-zone-events.json';
import {computeZoneEventsOracle, createRingEdges} from './zone-events-oracle';
import {
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';

it('Shapely crossings use the existing equal-time edge-order event convention', () => {
  const entry = fixture.cases[1];
  const rings: [number, number[]][] = [];
  for (let zone = 0; zone < entry.zoneCount; zone++) {
    for (
      let polygon = entry.polygons.featureOffsets[zone];
      polygon < entry.polygons.featureOffsets[zone + 1];
      polygon++
    ) {
      for (
        let ring = entry.polygons.polygonOffsets[polygon];
        ring < entry.polygons.polygonOffsets[polygon + 1];
        ring++
      ) {
        rings.push([
          zone,
          entry.polygons.positions.slice(
            2 * entry.polygons.ringOffsets[ring],
            2 * entry.polygons.ringOffsets[ring + 1]
          )
        ]);
      }
    }
  }
  const tracks = packTracks(entry.tracks as FixtureRow[][]);
  const actual = computeZoneEventsOracle(tracks, createRingEdges(rings), entry.zoneCount);
  expect(actual.events.length).toBe(entry.expected.events.length);
  const matched = new Set<number>();
  for (const event of entry.expected.events) {
    const index = actual.events.findIndex(
      (candidate, candidateIndex) =>
        !matched.has(candidateIndex) &&
        candidate.track === event[0] &&
        candidate.zone === event[1] &&
        candidate.row === event[6] &&
        Math.hypot(candidate.x - event[4], candidate.y - event[5]) < entry.positionAbsolute
    );
    expect(index).toBeGreaterThanOrEqual(0);
    expect(matched.has(index)).toBe(false);
    matched.add(index);
    expect(
      actual.events[index].type,
      JSON.stringify({expected: event, actual: actual.events[index]})
    ).toBe(event[2]);
    expect(Math.abs(actual.events[index].time - event[3])).toBeLessThan(entry.timeAbsolute);
  }
  expect(actual.visitCounts).toEqual(entry.expected.visitCounts);
  expect(actual.trackEventCounts).toEqual(entry.expected.trackEventCounts);
  actual.dwellTimes.forEach((value, index) =>
    expect(Math.abs(value - entry.expected.dwellTimes[index])).toBeLessThan(entry.timeAbsolute)
  );
  // Hand-worked zero-duration crossing: east edge row 1 sorts before west edge row 3.
  const zeroDuration = actual.events.filter(event => event.track === 12 && event.zone === 0);
  expect(zeroDuration.map(event => [event.time, event.edge, event.type, event.x])).toEqual([
    [0, 1, 0, 10],
    [0, 3, 1, 0]
  ]);
});
