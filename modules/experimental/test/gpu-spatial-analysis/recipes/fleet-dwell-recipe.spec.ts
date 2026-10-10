// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {getGPUTrajectoryMetricsParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {
  computeTrajectoryOracle,
  generateTrajectoryTracks
} from '../trajectory-analysis/trajectory-metrics-oracle';
import {computeZoneEventsOracle, createRingEdges} from '../trajectory-zones/zone-events-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

/** Three non-overlapping rectangles with fractional borders: `[zone id, x0, y0, x1, y1]`. */
const RECTANGLES = [
  [10, -300.37, -300.41, -0.37, -0.41],
  [20, -0.37, -300.41, 300.37, -0.41],
  [30, -300.37, -0.41, 300.37, 300.41]
] as const;

function rectangleRing(x0: number, y0: number, x1: number, y1: number): number[] {
  return [x0, y0, x1, y0, x1, y1, x0, y1];
}

function isInside(x: number, y: number, [, x0, y0, x1, y1]: (typeof RECTANGLES)[number]) {
  return x > x0 && x < x1 && y > y0 && y < y1;
}

it('addFleetDwellRecipe joins stop centroids to zones and summarises dwell per zone', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const speedThreshold = 1;
  const minimumDuration = 3;
  const tracks = generateTrajectoryTracks(7, 40, 15, 30, speedThreshold);
  const rowCount = tracks.positions.length / 2;
  const trackCount = tracks.trackOffsets.length - 1;
  const oracle = computeTrajectoryOracle(tracks, speedThreshold, minimumDuration);
  expect(oracle.stops.length).toBeGreaterThan(10);
  const stopCapacity = oracle.stops.length + 16;

  const polygonPositions = Float32Array.from(
    RECTANGLES.flatMap(([, x0, y0, x1, y1]) => rectangleRing(x0, y0, x1, y1))
  );
  const fixture = new RecipeTestFixture(device, 'fleet-dwell-test');
  try {
    const outputs = {
      ids: fixture.output('stop-ids', 'uint32', stopCapacity),
      count: fixture.output('stop-count', 'uint32', 1),
      overflow: fixture.output('stop-overflow', 'uint32', 1),
      centroids: fixture.output('stop-centroids', 'float32x2', stopCapacity),
      durations: fixture.output('stop-durations', 'float32', stopCapacity),
      zones: fixture.output('stop-zones', 'uint32', stopCapacity),
      joinOverflow: fixture.output('join-overflow', 'uint32', 1),
      keys: fixture.output('zone-keys', 'uint32', 32),
      counts: fixture.output('zone-counts', 'uint32', 32),
      tableCount: fixture.output('zone-count', 'uint32', 1),
      tableOverflow: fixture.output('zone-overflow', 'uint32', 1),
      sums: fixture.output('zone-sums', 'float32', 32),
      means: fixture.output('zone-means', 'float32', 32),
      maximums: fixture.output('zone-maximums', 'float32', 32)
    };
    const recipe = addFleetDwellRecipe(fixture.graph, {
      positions: fixture.input('positions', tracks.positions, 'float32x2', rowCount),
      timestamps: fixture.input('timestamps', tracks.timestamps, 'float32', rowCount),
      trackOffsets: fixture.input(
        'track-offsets',
        Uint32Array.from(tracks.trackOffsets),
        'uint32',
        trackCount + 1
      ),
      parameters: fixture.parameters(
        'metrics-parameters',
        'float32',
        getGPUTrajectoryMetricsParameterValues({
          stopSpeedThreshold: speedThreshold,
          stopMinimumDuration: minimumDuration
        })
      ),
      stopCapacity,
      zones: {
        polygonPositions: fixture.input('zone-positions', polygonPositions, 'float32x2', 12),
        featureOffsets: fixture.input('feature-offsets', Uint32Array.of(0, 1, 2, 3), 'uint32', 4),
        polygonOffsets: fixture.input('polygon-offsets', Uint32Array.of(0, 1, 2, 3), 'uint32', 4),
        ringOffsets: fixture.input('ring-offsets', Uint32Array.of(0, 4, 8, 12), 'uint32', 4),
        featureIds: fixture.input('zone-ids', Uint32Array.of(10, 20, 30), 'uint32', 3),
        candidateCapacity: 4096
      },
      zoneCapacity: 32,
      outputs: {
        stops: {
          ids: outputs.ids.view,
          count: outputs.count.view,
          overflow: outputs.overflow.view,
          centroids: outputs.centroids.view,
          durations: outputs.durations.view
        },
        joinOverflow: outputs.joinOverflow.view,
        table: {
          keys: outputs.keys.view,
          counts: outputs.counts.view,
          count: outputs.tableCount.view,
          overflow: outputs.tableOverflow.view,
          sumValues: outputs.sums.view,
          means: outputs.means.view,
          maximums: outputs.maximums.view
        }
      },
      scratch: {stopZones: outputs.zones.view}
    });
    expect(recipe.contributors.length).toBe(3);
    expect(recipe.outputs?.stops).toBe(recipe.stops);
    expect(recipe.intermediates?.stopZones).toBe(recipe.stopZones);
    expect(recipe.status?.stages.map(({stage}) => stage)).toEqual([
      'metrics',
      'zone-join',
      'zone-statistics'
    ]);
    fixture.run();

    // Stops match the trajectory oracle.
    const [stopCount] = await fixture.readUint32(outputs.count, 1);
    expect(stopCount).toBe(oracle.stops.length);
    const ids = await fixture.readUint32(outputs.ids, stopCount);
    const durations = await fixture.readFloat32(outputs.durations, stopCount);
    const centroids = await fixture.readFloat32(outputs.centroids, 2 * stopCount);
    expect(ids).toEqual(oracle.stops.map(stop => stop.track));
    for (const [index, stop] of oracle.stops.entries()) {
      expect(durations[index]).toBeCloseTo(stop.duration, 3);
    }
    expect((await fixture.readUint32(outputs.overflow, 1))[0]).toBe(0);
    expect((await fixture.readUint32(outputs.joinOverflow, 1))[0]).toBe(0);

    // Zone table matches a CPU join of the GPU centroids and a per-zone reduction of the durations.
    const expected = new Map<number, {count: number; sum: number; max: number}>();
    for (let stop = 0; stop < stopCount; stop++) {
      const zone = RECTANGLES.find(rectangle =>
        isInside(centroids[2 * stop], centroids[2 * stop + 1], rectangle)
      );
      if (zone) {
        const entry = expected.get(zone[0]) ?? {count: 0, sum: 0, max: -Infinity};
        entry.count++;
        entry.sum += durations[stop];
        entry.max = Math.max(entry.max, durations[stop]);
        expected.set(zone[0], entry);
      }
    }
    expect(expected.size).toBeGreaterThanOrEqual(2);
    // Dense table: one row per zone id in [0, 32); zones nothing stopped in keep an empty row.
    const [rows] = await fixture.readUint32(outputs.tableCount, 1);
    expect(rows).toBe(32);
    const keys = await fixture.readUint32(outputs.keys, rows);
    const counts = await fixture.readUint32(outputs.counts, rows);
    const sums = await fixture.readFloat32(outputs.sums, rows);
    const means = await fixture.readFloat32(outputs.means, rows);
    const maximums = await fixture.readFloat32(outputs.maximums, rows);
    expect((await fixture.readUint32(outputs.tableOverflow, 1))[0]).toBe(0);
    for (let key = 0; key < rows; key++) {
      const want = expected.get(key);
      expect(keys[key]).toBe(key);
      if (!want) {
        expect(counts[key], `zone ${key} empty`).toBe(0);
        expect(sums[key]).toBe(0);
        expect(means[key]).toBeNaN();
        continue;
      }
      expect(counts[key], `zone ${key} stops`).toBe(want.count);
      expect(isClose(sums[key], want.sum, 1e-3, 1e-4), `zone ${key} sum`).toBe(true);
      expect(isClose(means[key], want.sum / want.count, 1e-3, 1e-4), `zone ${key} mean`).toBe(true);
      expect(maximums[key]).toBeCloseTo(want.max, 3);
    }
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addFleetDwellZoneEventsRecipe rolls dwell per (track, zone) up per zone', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tracks = generateTrajectoryTracks(11, 36, 20, 40, 8);
  const rowCount = tracks.positions.length / 2;
  const trackCount = tracks.trackOffsets.length - 1;
  const zoneCount = RECTANGLES.length;
  const edges = createRingEdges(
    RECTANGLES.map(([, x0, y0, x1, y1], zone) => [zone, rectangleRing(x0, y0, x1, y1)] as const)
  );
  const oracle = computeZoneEventsOracle(
    {
      positions: tracks.positions,
      times: {kind: 'float32', values: tracks.timestamps},
      trackOffsets: tracks.trackOffsets
    },
    edges,
    zoneCount
  );
  expect(oracle.events.length).toBeGreaterThan(5);
  const maxEventsPerTrack = 64;
  expect(Math.max(...oracle.trackEventCounts)).toBeLessThanOrEqual(maxEventsPerTrack);

  const fixture = new RecipeTestFixture(device, 'fleet-dwell-zone-events-test');
  try {
    const cellCount = trackCount * zoneCount;
    const outputs = {
      eventCount: fixture.output('event-count', 'uint32', 1),
      eventOverflow: fixture.output('event-overflow', 'uint32', 1),
      eventTypes: fixture.output('event-types', 'uint32', trackCount * maxEventsPerTrack),
      candidateCount: fixture.output('candidate-count', 'uint32', 1),
      dwell: fixture.output('dwell', 'float32', cellCount),
      visits: fixture.output('visits', 'uint32', cellCount),
      keys: fixture.output('zone-keys', 'uint32', zoneCount),
      counts: fixture.output('zone-counts', 'uint32', zoneCount),
      count: fixture.output('zone-count', 'uint32', 1),
      sums: fixture.output('zone-sums', 'float32', zoneCount),
      means: fixture.output('zone-means', 'float32', zoneCount),
      maximums: fixture.output('zone-maximums', 'float32', zoneCount)
    };
    const recipe = addFleetDwellZoneEventsRecipe(fixture.graph, {
      positions: fixture.input('positions', tracks.positions, 'float32x2', rowCount),
      timestamps: fixture.input('timestamps', tracks.timestamps, 'float32', rowCount),
      trackOffsets: fixture.input(
        'track-offsets',
        Uint32Array.from(tracks.trackOffsets),
        'uint32',
        trackCount + 1
      ),
      edgeStarts: fixture.input('edge-starts', edges.starts, 'float32x2', edges.zones.length),
      edgeEnds: fixture.input('edge-ends', edges.ends, 'float32x2', edges.zones.length),
      edgeZones: fixture.input('edge-zones', edges.zones, 'uint32', edges.zones.length),
      zoneCount,
      candidateCapacity: 1 << 16,
      maxEventsPerTrack,
      outputs: {
        events: {
          output: {count: outputs.eventCount.view, overflow: outputs.eventOverflow.view},
          eventTypes: outputs.eventTypes.view
        },
        diagnostics: {candidateCount: outputs.candidateCount.view},
        dwellTimes: outputs.dwell.view,
        visitCounts: outputs.visits.view,
        table: {
          keys: outputs.keys.view,
          counts: outputs.counts.view,
          count: outputs.count.view,
          sumValues: outputs.sums.view,
          means: outputs.means.view,
          maximums: outputs.maximums.view
        }
      }
    });
    expect(recipe.contributors.length).toBe(2);
    fixture.run();

    expect((await fixture.readUint32(outputs.eventOverflow, 1))[0]).toBe(0);
    const [candidateCount] = await fixture.readUint32(outputs.candidateCount, 1);
    expect(candidateCount).toBeGreaterThan(0);
    expect(candidateCount).toBeLessThanOrEqual(1 << 16);
    expect((await fixture.readUint32(outputs.eventCount, 1))[0]).toBe(oracle.events.length);
    const dwell = await fixture.readFloat32(outputs.dwell, cellCount);
    const visits = await fixture.readUint32(outputs.visits, cellCount);
    expect(visits).toEqual(oracle.visitCounts);
    for (let cell = 0; cell < cellCount; cell++) {
      expect(isClose(dwell[cell], oracle.dwellTimes[cell], 1e-2, 1e-3), `dwell ${cell}`).toBe(true);
    }

    // Per-zone roll-up over visiting tracks, from the oracle matrix.
    const expected = new Map<number, {count: number; sum: number; max: number}>();
    for (let cell = 0; cell < cellCount; cell++) {
      if (oracle.visitCounts[cell] > 0) {
        const zone = cell % zoneCount;
        const entry = expected.get(zone) ?? {count: 0, sum: 0, max: -Infinity};
        entry.count++;
        entry.sum += oracle.dwellTimes[cell];
        entry.max = Math.max(entry.max, oracle.dwellTimes[cell]);
        expected.set(zone, entry);
      }
    }
    expect(expected.size).toBeGreaterThanOrEqual(2);
    // Dense table: row `zone` for every zone, empty zones keep a row.
    const [rows] = await fixture.readUint32(outputs.count, 1);
    expect(rows).toBe(zoneCount);
    const keys = await fixture.readUint32(outputs.keys, rows);
    const counts = await fixture.readUint32(outputs.counts, rows);
    const sums = await fixture.readFloat32(outputs.sums, rows);
    const means = await fixture.readFloat32(outputs.means, rows);
    const maximums = await fixture.readFloat32(outputs.maximums, rows);
    for (let zone = 0; zone < zoneCount; zone++) {
      const want = expected.get(zone);
      expect(keys[zone]).toBe(zone);
      if (!want) {
        expect(counts[zone]).toBe(0);
        expect(means[zone]).toBeNaN();
        continue;
      }
      expect(counts[zone], `zone ${zone} tracks`).toBe(want.count);
      expect(isClose(sums[zone], want.sum, 1e-1, 1e-3), `zone ${zone} sum`).toBe(true);
      expect(isClose(means[zone], want.sum / want.count, 1e-1, 1e-3)).toBe(true);
      expect(isClose(maximums[zone], want.max, 1e-2, 1e-3)).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
}, 120000);
