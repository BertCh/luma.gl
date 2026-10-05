// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TERRAIN_CRITICAL_POINT as CLASS,
  GPUTerrainCriticalPoints
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-critical-points';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';
import {
  classifyCriticalPoints,
  createEggCrate,
  createRandomElevation,
  createTwoBumps,
  getEulerSum
} from './terrain-critical-points-oracle';

type Run = {
  classes: number[];
  signChanges: number[];
  counts: number[];
  countsAfterSecondEncoding: number[];
};

async function runCriticalPoints(
  device: Device,
  elevation: Float32Array,
  validity: Uint32Array | undefined,
  width: number,
  height: number,
  connectivity: 8 | 6
): Promise<Run> {
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'critical-points-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const validityBuffer = validity ? createInputBuffer(device, validity) : undefined;
  const classesBuffer = createOutputBuffer(device, pixelCount);
  const signChangesBuffer = createOutputBuffer(device, pixelCount);
  const countsBuffer = createOutputBuffer(device, 6);
  graph.add(
    new GPUTerrainCriticalPoints({
      width,
      height,
      connectivity,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: validityBuffer
          ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
          : undefined
      },
      classes: importGraphBuffer(graph, 'classes', classesBuffer, 'uint32', pixelCount),
      signChanges: importGraphBuffer(
        graph,
        'sign-changes',
        signChangesBuffer,
        'uint32',
        pixelCount
      ),
      counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 6)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const run: Run = {
    classes: await readUint32(classesBuffer, pixelCount),
    signChanges: await readUint32(signChangesBuffer, pixelCount),
    counts: await readUint32(countsBuffer, 6),
    countsAfterSecondEncoding: []
  };
  // Counts are cleared by a fill node on every encoding, so they must not accumulate.
  submitGraph(device, compiled, undefined);
  run.countsAfterSecondEncoding = await readUint32(countsBuffer, 6);
  compiled.destroy();
  for (const buffer of [
    elevationBuffer,
    validityBuffer,
    classesBuffer,
    signChangesBuffer,
    countsBuffer
  ]) {
    buffer?.destroy();
  }
  return run;
}

function expectMatchesOracle(
  run: Run,
  elevation: Float32Array,
  validity: Uint32Array | undefined,
  width: number,
  height: number,
  connectivity: 8 | 6
): ReturnType<typeof classifyCriticalPoints> {
  const expected = classifyCriticalPoints(elevation, validity, width, height, connectivity);
  expect(run.classes).toEqual(Array.from(expected.classes));
  expect(run.signChanges).toEqual(Array.from(expected.signChanges));
  expect(run.counts).toEqual(Array.from(expected.counts));
  expect(run.countsAfterSecondEncoding).toEqual(Array.from(expected.counts));
  return expected;
}

for (const connectivity of [8, 6] as const) {
  it(`GPUTerrainCriticalPoints ${connectivity}-ring matches the oracle on random DEMs, plateaus and nodata holes`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const width = 37;
    const height = 29;
    for (const [seed, levels] of [
      [1, 0],
      [2, 0],
      [3, 3],
      [4, 2]
    ] as const) {
      const elevation = createRandomElevation(width, height, seed, levels);
      const validity = new Uint32Array(width * height).fill(1);
      // Nodata holes holding a huge sentinel in the value slot: it must never bleed.
      for (const [column, row, radius] of [
        [10, 8, 1],
        [25, 18, 2],
        [0, 14, 1]
      ] as const) {
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const x = column + dx;
            const y = row + dy;
            if (x >= 0 && y >= 0 && x < width && y < height) {
              validity[y * width + x] = 0;
              elevation[y * width + x] = 1e30;
            }
          }
        }
      }
      const run = await runCriticalPoints(device, elevation, validity, width, height, connectivity);
      const expected = expectMatchesOracle(run, elevation, validity, width, height, connectivity);
      // Non-trivial structure: every class that must exist does exist.
      expect(expected.counts[CLASS.noData]).toBeGreaterThan(10);
      expect(expected.counts[CLASS.peak]).toBeGreaterThan(0);
      expect(expected.counts[CLASS.pit]).toBeGreaterThan(0);
      expect(expected.counts[CLASS.saddle]).toBeGreaterThan(0);
      expect(expected.counts[CLASS.regular]).toBeGreaterThan(100);
      expect(run.counts.reduce((sum, count) => sum + count, 0)).toBe(width * height);

      const flipped = Float32Array.from(elevation, (value, index) =>
        validity[index] === 0 ? -1e30 : value
      );
      const flippedRun = await runCriticalPoints(
        device,
        flipped,
        validity,
        width,
        height,
        connectivity
      );
      expect(flippedRun.classes, 'nodata sentinel value must not matter').toEqual(run.classes);
    }
  }, 60000);
}

it('GPUTerrainCriticalPoints classifies an egg-crate: 6-ring obeys the torus Euler relation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const period = 7;
  const periodCount = 3;
  const size = period * periodCount + 2;
  const elevation = createEggCrate(size, size, period, 0.3);
  const window = {
    columnStart: 1,
    columnEnd: 1 + period * periodCount,
    rowStart: 1,
    rowEnd: 1 + period * periodCount
  };

  const six = await runCriticalPoints(device, elevation, undefined, size, size, 6);
  const sixExpected = expectMatchesOracle(six, elevation, undefined, size, size, 6);
  // A full period window of a periodic function: its ring neighbours are the periodic values, so the
  // counts are those of the torus. Per period: 2 peaks, 2 pits, 4 saddles.
  expect(getEulerSum(sixExpected, size, window)).toEqual({
    peaks: 2 * periodCount ** 2,
    pits: 2 * periodCount ** 2,
    saddleMultiplicity: 4 * periodCount ** 2,
    sum: 0
  });

  const eight = await runCriticalPoints(device, elevation, undefined, size, size, 8);
  const eightExpected = expectMatchesOracle(eight, elevation, undefined, size, size, 8);
  const eightSum = getEulerSum(eightExpected, size, window);
  // Same extrema, but the 8-ring counts spurious saddles at diagonal ambiguities: the relation breaks.
  expect(eightSum.peaks).toBe(2 * periodCount ** 2);
  expect(eightSum.pits).toBe(2 * periodCount ** 2);
  expect(eightSum.saddleMultiplicity).toBeGreaterThan(4 * periodCount ** 2);
  expect(eightSum.sum).toBeLessThan(0);
}, 60000);

it('GPUTerrainCriticalPoints 6-ring satisfies the disk Euler relation for Gaussian bumps', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const size = 41;
  const wholeGrid = {columnStart: 0, columnEnd: size, rowStart: 0, rowEnd: size};

  // One bump (slightly tilted so no two heights are symmetric): 1 peak, no saddle, no interior pit.
  const single = Float32Array.from({length: size * size}, (_, index) => {
    const column = index % size;
    const row = Math.floor(index / size);
    const distance = (column - 20.13) ** 2 + (row - 19.71) ** 2;
    return 100 * Math.exp(-distance / (2 * 6 * 6));
  });
  const singleRun = await runCriticalPoints(device, single, undefined, size, size, 6);
  const singleExpected = expectMatchesOracle(singleRun, single, undefined, size, size, 6);
  expect(getEulerSum(singleExpected, size, wholeGrid)).toEqual({
    peaks: 1,
    pits: 0,
    saddleMultiplicity: 0,
    sum: 1
  });

  // Two bumps: the height decreases towards the rim, so critical points on the boundary pixels do
  // not matter and the interior sum equals the Euler characteristic of a disk, 1.
  const two = createTwoBumps(size);
  const twoRun = await runCriticalPoints(device, two, undefined, size, size, 6);
  const twoExpected = expectMatchesOracle(twoRun, two, undefined, size, size, 6);
  const twoSum = getEulerSum(twoExpected, size, wholeGrid);
  expect(twoSum.peaks).toBe(2);
  expect(twoSum.sum).toBe(1);
}, 60000);
