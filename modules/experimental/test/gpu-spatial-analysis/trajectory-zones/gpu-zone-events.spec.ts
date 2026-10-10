// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createRandom,
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runZoneEvents, type ZoneGPUResult} from './zone-events-harness';
import {
  computeZoneEventsOracle,
  createRingEdges,
  type ZoneOracleResult
} from './zone-events-oracle';

/** Zone 0 square with a hole, zone 1 beside it sharing an edge, zone 2 a triangle. Fractional coordinates avoid integer-track degeneracies. */
const EDGES = createRingEdges([
  [0, [100.5, 100.5, 300.5, 100.5, 300.5, 300.5, 100.5, 300.5]],
  [0, [150.5, 150.5, 150.5, 200.5, 200.5, 200.5, 200.5, 150.5]],
  [1, [300.5, 100.5, 500.5, 100.5, 500.5, 300.5, 300.5, 300.5]],
  [2, [700.3, 100.3, 900.7, 150.2, 800.1, 400.9]]
]);
const ZONE_COUNT = 3;

/** Long-stride random walks so most tracks cross several boundaries. */
function createWalkingTracks(seed: number, trackCount: number): FixtureRow[][] {
  const random = createRandom(seed);
  const tracks: FixtureRow[][] = [];
  for (let track = 0; track < trackCount; track++) {
    const rowCount = track % 17 === 0 ? 0 : track % 13 === 0 ? 1 : 2 + Math.floor(random() * 40);
    let [x, y, time] = [
      Math.round(random() * 1000),
      Math.round(random() * 500),
      Math.floor(random() * 100)
    ];
    const rows: FixtureRow[] = [];
    for (let row = 0; row < rowCount; row++) {
      rows.push([x, y, time]);
      time += random() < 0.15 ? 0 : 1 + Math.floor(random() * 20);
      x += Math.round((random() - 0.5) * 160);
      y += Math.round((random() - 0.5) * 160);
    }
    tracks.push(rows);
  }
  return tracks;
}

type EventKey = [track: number, zone: number, time: number, type: number, x: number, y: number];

function getKeys(
  events: {
    track: number;
    zone: number;
    time: number;
    type: number;
    x: number;
    y: number;
  }[]
): EventKey[] {
  return events
    .map((event): EventKey => [event.track, event.zone, event.time, event.type, event.x, event.y])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
}

/** Sparse table: one row per visited `(track, zone)` cell, ordered by cell. */
function expectTableParity(actual: ZoneGPUResult, expected: ZoneOracleResult, label: string): void {
  const zoneCount = ZONE_COUNT;
  const cells = expected.visitCounts.flatMap((visits, cell) => (visits > 0 ? [cell] : []));
  const {table} = actual;
  expect(table.count, `${label} table count`).toBe(cells.length);
  expect(table.requiredCount, `${label} table total`).toBe(cells.length);
  expect(table.overflow, `${label} table overflow`).toBe(0);
  for (const [row, cell] of cells.entries()) {
    const track = Math.floor(cell / zoneCount);
    expect([table.tracks[row], table.zones[row]], `${label} table key ${row}`).toEqual([
      track,
      cell % zoneCount
    ]);
    expect(table.visits[row], `${label} table visits ${row}`).toBe(expected.visitCounts[cell]);
    const tolerance = 2e-3 + 1e-4 * Math.abs(expected.dwellTimes[cell]);
    expect(Math.abs(table.dwellTimes[row] - expected.dwellTimes[cell])).toBeLessThanOrEqual(
      tolerance
    );
    expect(
      Math.abs(table.firstEnterTimes[row] - expected.firstEnterTimes[cell])
    ).toBeLessThanOrEqual(2e-3 + 1e-5 * expected.firstEnterTimes[cell]);
    expect(Math.abs(table.lastExitTimes[row] - expected.lastExitTimes[cell])).toBeLessThanOrEqual(
      2e-3 + 1e-5 * expected.lastExitTimes[cell]
    );
  }
  // Rows past the count hold sentinels.
  if (cells.length < table.tracks.length) {
    expect(table.tracks[cells.length]).toBe(0xffffffff);
  }
}

function expectParity(actual: ZoneGPUResult, expected: ZoneOracleResult, label: string): void {
  const actualEvents = Array.from({length: actual.count}, (_, index) => ({
    track: actual.tracks[index],
    zone: actual.zones[index],
    time: actual.times[index],
    type: actual.types[index],
    x: actual.positions[2 * index],
    y: actual.positions[2 * index + 1]
  }));
  expect(actual.count, `${label} count`).toBe(expected.events.length);
  expect(actual.requiredCount, `${label} total`).toBe(expected.events.length);
  expect(actual.overflow, `${label} overflow`).toBe(0);
  // Global order: track, then time.
  for (let index = 1; index < actualEvents.length; index++) {
    const [previous, current] = [actualEvents[index - 1], actualEvents[index]];
    expect(current.track >= previous.track, `${label} track order ${index}`).toBe(true);
    if (current.track === previous.track) {
      expect(current.time >= previous.time, `${label} time order ${index}`).toBe(true);
    }
  }
  const actualKeys = getKeys(actualEvents);
  const expectedKeys = getKeys(expected.events);
  for (const [index, key] of expectedKeys.entries()) {
    expect(actualKeys[index].slice(0, 2), `${label} event ${index}`).toEqual(key.slice(0, 2));
    expect(actualKeys[index][3], `${label} type ${index}`).toBe(key[3]);
    expect(Math.abs(actualKeys[index][2] - key[2])).toBeLessThanOrEqual(1e-3 + 1e-5 * key[2]);
    // Crossing positions interpolate the same segment parameter. Events of one track and zone
    // with equal times (zero-duration segments) have no defined order, so skip those.
    const isTied = [index - 1, index + 1].some(
      other =>
        other >= 0 &&
        other < expectedKeys.length &&
        expectedKeys[other][0] === key[0] &&
        expectedKeys[other][1] === key[1] &&
        expectedKeys[other][2] === key[2]
    );
    if (isTied) {
      continue;
    }
    expect(Math.abs(actualKeys[index][4] - key[4]), `${label} x ${index}`).toBeLessThanOrEqual(
      0.05
    );
    expect(Math.abs(actualKeys[index][5] - key[5]), `${label} y ${index}`).toBeLessThanOrEqual(
      0.05
    );
  }
  expectTableParity(actual, expected, label);
  expect(actual.visitCounts, `${label} visits`).toEqual(expected.visitCounts);
  expect(actual.trackEventCounts, `${label} track counts`).toEqual(expected.trackEventCounts);
  for (const [index, dwell] of expected.dwellTimes.entries()) {
    expect(
      Math.abs(actual.dwellTimes[index] - dwell),
      `${label} dwell ${index}`
    ).toBeLessThanOrEqual(2e-3 + 1e-4 * dwell);
  }
}

it('GPUZoneEvents matches the oracle on random walks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seed of [11, 12]) {
    const tracks = packTracks(createWalkingTracks(seed, 300));
    const expected = computeZoneEventsOracle(tracks, EDGES, ZONE_COUNT);
    // Teeth: events of every type and zone, tracks starting inside, multi-visit dwell.
    expect(expected.events.length).toBeGreaterThan(100);
    expect(new Set(expected.events.map(event => event.zone)).size).toBe(3);
    expect(expected.events.some(event => event.type === 0)).toBe(true);
    expect(expected.events.some(event => event.type === 1)).toBe(true);
    expect(expected.visitCounts.some(visits => visits > 1)).toBe(true);
    expect(Math.max(...expected.dwellTimes)).toBeGreaterThan(10);
    const options = {
      zoneCount: ZONE_COUNT,
      candidateCapacity: 4096,
      maxEventsPerTrack: 1000,
      eventCapacity: 2048
    };
    expectParity(await runZoneEvents(device, tracks, EDGES, options), expected, `seed ${seed}`);
    expectParity(
      await runZoneEvents(device, tracks, EDGES, {...options, spatialSort: true}),
      expected,
      `seed ${seed} spatialSort`
    );
  }
  device.destroy?.();
});

it('GPUZoneEvents accepts Int64 word timestamps and matches float32 results', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = createWalkingTracks(21, 120);
  const words = packTracks(rows, 397n * 2n ** 32n - 20n);
  const expected = computeZoneEventsOracle(words, EDGES, ZONE_COUNT);
  expect(expected.events.length).toBeGreaterThan(30);
  expectParity(
    await runZoneEvents(device, words, EDGES, {
      zoneCount: ZONE_COUNT,
      candidateCapacity: 2048,
      maxEventsPerTrack: 1000,
      eventCapacity: 1024
    }),
    expected,
    'words'
  );
  device.destroy?.();
});

it('GPUZoneEvents handles tracks that start inside a zone and shared borders', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Track 0 starts inside zone 0, leaves into zone 1 across the shared border, then ends inside.
  // Track 1 starts inside the hole (outside zone 0) and never moves. Track 2 never touches a zone.
  const tracks = packTracks([
    [
      [120, 120, 0],
      [320, 120, 10],
      [420, 220, 30]
    ],
    [
      [170, 170, 5],
      [170, 170, 25]
    ],
    [
      [0, 0, 0],
      [10, 0, 10]
    ]
  ]);
  const expected = computeZoneEventsOracle(tracks, EDGES, ZONE_COUNT);
  expect(expected.events.map(event => [event.zone, event.type])).toEqual([
    [0, 1],
    [1, 0]
  ]);
  const actual = await runZoneEvents(device, tracks, EDGES, {
    zoneCount: ZONE_COUNT,
    candidateCapacity: 64,
    maxEventsPerTrack: 8,
    eventCapacity: 16
  });
  expectParity(actual, expected, 'inside start');
  // Track 0: visit zone 0 for 10 * (300.5 - 120) / 200 = 9.025, zone 1 from there to the end.
  expect(actual.visitCounts.slice(0, 3)).toEqual([1, 1, 0]);
  expect(actual.visitCounts.slice(3, 6)).toEqual([0, 0, 0]);
  expect(actual.dwellTimes[0]).toBeCloseTo(9.025, 2);
  expect(actual.dwellTimes[1]).toBeCloseTo(30 - 9.025, 2);
  device.destroy?.();
});

it('GPUZoneEvents bounds events per track and flags overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = packTracks(createWalkingTracks(11, 300));
  const expected = computeZoneEventsOracle(tracks, EDGES, ZONE_COUNT);
  const base = {zoneCount: ZONE_COUNT, candidateCapacity: 4096, eventCapacity: 2048};
  const truncated = await runZoneEvents(device, tracks, EDGES, {...base, maxEventsPerTrack: 2});
  const keptExpected = expected.events.filter(event => {
    const index = expected.events.filter(other => other.track === event.track).indexOf(event);
    return index < 2;
  });
  expect(expected.trackEventCounts.some(count => count > 2)).toBe(true);
  expect(truncated.overflow).toBe(1);
  expect([truncated.candidateOverflow, truncated.trackOverflow, truncated.eventOverflow]).toEqual([
    0, 1, 0
  ]);
  expect(truncated.candidateCount).toBeGreaterThan(0);
  expect(truncated.candidateCount).toBeLessThanOrEqual(base.candidateCapacity);
  expect(truncated.count).toBe(keptExpected.length);
  expect(truncated.trackEventCounts).toEqual(expected.trackEventCounts);
  // Dwell and visits still use every event.
  expect(truncated.visitCounts).toEqual(expected.visitCounts);
  // A too-small output capacity clamps the count and reports the unclamped total.
  const clamped = await runZoneEvents(device, tracks, EDGES, {
    ...base,
    maxEventsPerTrack: 1000,
    eventCapacity: 10
  });
  expect(clamped.count).toBe(10);
  expect(clamped.requiredCount).toBe(expected.events.length);
  expect(clamped.overflow).toBe(1);
  expect([clamped.candidateOverflow, clamped.trackOverflow, clamped.eventOverflow]).toEqual([
    0, 0, 1
  ]);
  // Candidate scratch overflow is reported.
  const starved = await runZoneEvents(device, tracks, EDGES, {
    ...base,
    maxEventsPerTrack: 1000,
    candidateCapacity: 16
  });
  expect(starved.overflow).toBe(1);
  expect([starved.candidateOverflow, starved.trackOverflow, starved.eventOverflow]).toEqual([
    1, 0, 0
  ]);
  // The unclamped count is the capacity needed: rerunning with it is exact.
  expect(starved.candidateCount).toBeGreaterThan(16);
  expect(starved.candidateCount).toBe(truncated.candidateCount);
  const sized = await runZoneEvents(device, tracks, EDGES, {
    ...base,
    maxEventsPerTrack: 1000,
    candidateCapacity: starved.candidateCount
  });
  expect(sized.candidateOverflow).toBe(0);
  expect(sized.overflow).toBe(0);
  expect(sized.count).toBe(expected.events.length);
  device.destroy?.();
});

it('GPUZoneEvents bounds the sparse table and reports its total', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = packTracks(createWalkingTracks(11, 300));
  const expected = computeZoneEventsOracle(tracks, EDGES, ZONE_COUNT);
  const visited = expected.visitCounts.filter(visits => visits > 0).length;
  expect(visited).toBeGreaterThan(20);
  const result = await runZoneEvents(device, tracks, EDGES, {
    zoneCount: ZONE_COUNT,
    candidateCapacity: 4096,
    maxEventsPerTrack: 1000,
    eventCapacity: 2048,
    tableCapacity: 10
  });
  expect(result.table.requiredCount).toBe(visited);
  expect(result.table.count).toBe(10);
  expect(result.table.overflow).toBe(1);
  // The kept rows are the first ten visited cells in track, zone order.
  const firstCells = expected.visitCounts.flatMap((v, cell) => (v > 0 ? [cell] : [])).slice(0, 10);
  expect(result.table.tracks.slice(0, 10)).toEqual(firstCells.map(cell => Math.floor(cell / 3)));
  device.destroy?.();
});
