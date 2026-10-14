// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  CLOSEST_APPROACH_FIXTURE,
  computeClosestApproachOracle,
  type ClosestApproachPair,
  type ClosestApproachTrack
} from './closest-approach-oracle';
import {runClosestApproach, type ClosestApproachGPUResult} from './closest-approach-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: test/oracles/trajectory-closest-approach.py (brute force over every segment pair in
// float64 with sampling and shapely self-checks); tolerances are in the fixture.
const {tolerances} = CLOSEST_APPROACH_FIXTURE;

function expectPairs(
  actual: ClosestApproachGPUResult,
  expected: readonly ClosestApproachPair[],
  label: string
) {
  expect(actual.entryOverflow, `${label} entry overflow`).toBe(0);
  expect(actual.hitOverflow, `${label} hit overflow`).toBe(0);
  expect(actual.overflow, `${label} pair overflow`).toBe(0);
  const found = new Map<string, number>();
  for (let index = 0; index < actual.count; index++) {
    found.set(`${actual.ids[index]}-${actual.partners[index]}`, index);
  }
  // Pairs are ordered by (track, partner).
  for (let index = 1; index < actual.count; index++) {
    const order =
      actual.ids[index - 1] - actual.ids[index] ||
      actual.partners[index - 1] - actual.partners[index];
    expect(order, `${label} order ${index}`).toBeLessThan(0);
  }
  const expectedKeys = new Set(expected.map(pair => `${pair.track}-${pair.partner}`));
  for (const key of found.keys()) {
    const pair = expected.find(entry => `${entry.track}-${entry.partner}` === key);
    // A GPU pair the oracle does not list is only allowed for a near-tangent miss in f64 terms.
    expect(expectedKeys.has(key) || pair?.isNearTangent, `${label} extra pair ${key}`).toBe(true);
  }
  for (const pair of expected) {
    const key = `${pair.track}-${pair.partner}`;
    const index = found.get(key);
    if (pair.isNearTangent) {
      continue;
    }
    expect(index, `${label} missing pair ${key}`).toBeDefined();
    if (index === undefined) {
      continue;
    }
    const distanceTolerance =
      tolerances.distanceAbsolute + tolerances.distanceRelative * Math.abs(pair.minimumDistance);
    expect(
      Math.abs(actual.minimumDistances[index] - pair.minimumDistance),
      `${label} ${key} distance`
    ).toBeLessThanOrEqual(distanceTolerance);
    const timeTolerance = (time: number) =>
      tolerances.timeAbsolute + tolerances.timeRelative * Math.abs(time);
    expect(
      Math.abs(actual.firstTimes[index] - pair.firstTime),
      `${label} ${key} first time`
    ).toBeLessThanOrEqual(timeTolerance(pair.firstTime));
    expect(
      Math.abs(actual.durations[index] - pair.duration),
      `${label} ${key} duration`
    ).toBeLessThanOrEqual(timeTolerance(pair.duration) + timeTolerance(pair.firstTime));
    expect(
      Math.abs(actual.closestTimes[index] - pair.closestTime),
      `${label} ${key} closest time`
    ).toBeLessThanOrEqual(timeTolerance(pair.closestTime) * 10);
  }
}

it('GPUTrajectoryClosestApproach matches the exact closest-approach oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of CLOSEST_APPROACH_FIXTURE.scenes) {
    for (const [cellSize, timeCellSize] of [
      [scene.distance * 2, 20],
      [scene.distance, 5]
    ]) {
      const actual = await runClosestApproach(device, scene.tracks, {
        distance: scene.distance,
        cellSize,
        timeCellSize,
        entryCapacity: 1 << 18,
        hitCapacity: 1 << 16,
        pairCapacity: 512
      });
      expectPairs(actual, scene.pairs, `${scene.name} cell ${cellSize} time ${timeCellSize}`);
    }
  }
  device.destroy?.();
}, 120000);

it('GPUTrajectoryClosestApproach keeps word timestamps exact at large epochs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Integer seconds 1.7e9 apart from zero: float32 seconds would quantize to 128 s steps.
  const tracks: ClosestApproachTrack[] = [
    {
      positions: [
        [0, 0],
        [100, 0]
      ],
      times: [0, 100]
    },
    {
      positions: [
        [60, 5],
        [60, 5]
      ],
      times: [50, 200]
    }
  ];
  const expected = computeClosestApproachOracle(tracks, 10);
  const actual = await runClosestApproach(device, tracks, {
    distance: 10,
    cellSize: 20,
    timeCellSize: 30,
    entryCapacity: 4096,
    hitCapacity: 256,
    pairCapacity: 8,
    wordBase: 1_700_000_000n
  });
  expectPairs(actual, expected, 'words');
  device.destroy?.();
}, 60000);

it('GPUTrajectoryClosestApproach finds a pass between samples that clock mode misses', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = CLOSEST_APPROACH_FIXTURE.scenes.find(
    entry => entry.name === 'cross-between-samples'
  );
  expect(scene).toBeDefined();
  if (!scene) {
    return;
  }
  // Clock mode on the two shared instants t = 0 and t = 10: the tracks are 100 apart at both.
  const graph = new GPUCommandGraph(device, {id: 'clock-preview'});
  const samples = new Float32Array([0, 0, 100, 0, 50, -50, 50, 50]);
  const buffers = [
    createInputBuffer(device, samples),
    createInputBuffer(device, Float32Array.of(scene.distance)),
    createOutputBuffer(device, 4),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1),
    createOutputBuffer(device, 1)
  ];
  graph.add(
    new GPUTrajectoryEncounters({
      samples: importGraphBuffer(graph, 'samples', buffers[0], 'float32x2', 4),
      trackCount: 2,
      bucketCount: 2,
      distance: importGraphBuffer(graph, 'distance', buffers[1], 'float32', 1),
      cellSize: 4,
      bounds: [-200, -200, 200, 200],
      hitCapacity: 64,
      output: {
        pairs: {
          ids: importGraphBuffer(graph, 'ids', buffers[2], 'uint32', 4),
          status: {
            count: importGraphBuffer(graph, 'count', buffers[3], 'uint32', 1),
            overflow: importGraphBuffer(graph, 'overflow', buffers[4], 'uint32', 1)
          }
        },
        status: {candidateOverflow: importGraphBuffer(graph, 'candidate', buffers[5], 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect((await readUint32(buffers[3], 1))[0], 'clock mode pairs').toBe(0);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  const exact = await runClosestApproach(device, scene.tracks, {
    distance: scene.distance,
    cellSize: 4,
    timeCellSize: 2,
    entryCapacity: 4096,
    hitCapacity: 64,
    pairCapacity: 4
  });
  expect(exact.count).toBe(1);
  expect(exact.minimumDistances[0]).toBeLessThan(1e-3);
  expect(Math.abs(exact.closestTimes[0] - 5)).toBeLessThan(1e-3);
  device.destroy?.();
}, 60000);

it('GPUTrajectoryClosestApproach reports entry, hit and pair overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = CLOSEST_APPROACH_FIXTURE.scenes.find(entry => entry.name === 'random');
  if (!scene) {
    throw new Error('random scene missing');
  }
  const base = {
    distance: scene.distance,
    cellSize: scene.distance * 2,
    timeCellSize: 20,
    entryCapacity: 1 << 18,
    hitCapacity: 1 << 16,
    pairCapacity: 512
  };
  expect(
    (await runClosestApproach(device, scene.tracks, {...base, entryCapacity: 16})).entryOverflow
  ).toBe(1);
  expect(
    (await runClosestApproach(device, scene.tracks, {...base, hitCapacity: 2})).hitOverflow
  ).toBe(1);
  const pairs = await runClosestApproach(device, scene.tracks, {...base, pairCapacity: 2});
  expect(pairs.overflow).toBe(1);
  expect(pairs.count).toBe(2);
  expect(pairs.requiredCount).toBeGreaterThan(2);
  // A tiny cell makes long segments cover more than 4096 cells.
  expect(
    (await runClosestApproach(device, scene.tracks, {...base, cellSize: 0.01, timeCellSize: 0.01}))
      .entryOverflow
  ).toBe(1);
  device.destroy?.();
}, 120000);
