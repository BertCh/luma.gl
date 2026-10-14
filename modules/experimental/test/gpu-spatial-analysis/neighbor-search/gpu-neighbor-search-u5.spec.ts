// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/gpu-spatial-analysis/contracts';
import {createNeighborSearchHarness, type NeighborSearchReadback} from './neighbor-search-harness';
import {NEIGHBOR_SEARCH_U5_ORACLE, type GreatCircleFixture} from './neighbor-search-u5-oracle';

// Oracle: fix/work/A/oracle-u5-neighbor-search.py (numpy float64 haversine with R = 6371008.8 m on
// float32-exact [lon, lat] fixtures; libpysal 4.15 Kernel(fixed=False) for the planar kernels).
// The fixtures guarantee a 50 m gap at every kNN cut and radius edge, far above f32 haversine error,
// so neighbor IDs are compared exactly. Distances: f32 trigonometry budget of 2e-5 relative + 1 m.
const WORLD = [-180, -90, 180, 90] as const;
const GREAT_CIRCLE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
const KILOMETERS: GPUSpatialContext = {...GREAT_CIRCLE, units: 'kilometers'};

function expectRowsMatch(
  result: NeighborSearchReadback,
  rows: [number, number][][],
  scale: number,
  label: string
): void {
  const expectedOffsets = [0];
  for (const row of rows) {
    expectedOffsets.push(expectedOffsets[expectedOffsets.length - 1] + row.length);
  }
  expect(result.overflow, `${label} overflow`).toBe(0);
  expect(result.offsets, `${label} offsets`).toEqual(expectedOffsets);
  expect(result.neighbors, `${label} neighbors`).toEqual(rows.flatMap(row => row.map(e => e[0])));
  const expectedDistances = rows.flatMap(row => row.map(e => e[1] * scale));
  for (let slot = 0; slot < expectedDistances.length; slot++) {
    const tolerance = 1 * scale + 2e-5 * expectedDistances[slot];
    if (Math.abs(result.distances[slot] - expectedDistances[slot]) > tolerance) {
      throw new Error(
        `${label}: slot ${slot} distance ${result.distances[slot]} != ${expectedDistances[slot]}`
      );
    }
  }
}

async function runFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  fixture: GreatCircleFixture,
  context: GPUSpatialContext,
  scale: number,
  gridSize: readonly [number, number]
): Promise<void> {
  const positions = Float32Array.from(fixture.positions);
  const label = `${fixture.name} ${context.units} grid ${gridSize}`;
  const knn = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: fixture.k,
    positions,
    capacity: fixture.knn.length * fixture.k,
    parameters: {bounds: WORLD},
    spatialContext: context,
    gridSize
  });
  expectRowsMatch(await knn.run(), fixture.knn, scale, `${label} knn`);
  knn.destroy();
  const total = fixture.radiusRows.reduce((sum, row) => sum + row.length, 0);
  const radius = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    capacity: total + 16,
    parameters: {bounds: WORLD, radius: fixture.radius * scale},
    spatialContext: context,
    gridSize
  });
  expectRowsMatch(await radius.run(), fixture.radiusRows, scale, `${label} radius`);
  radius.destroy();
}

it('GPUNeighborSearch great-circle kNN and radius match numpy haversine globally, over the antimeridian and at the poles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const fixture of NEIGHBOR_SEARCH_U5_ORACLE.fixtures) {
    // Results must not depend on the lattice: single cell, coarse and fine.
    for (const gridSize of [
      [1, 1],
      [8, 4],
      [36, 18]
    ] as const) {
      await runFixture(device, fixture, GREAT_CIRCLE, 1, gridSize);
    }
    await runFixture(device, fixture, KILOMETERS, 1e-3, [8, 4]);
  }
});

it('GPUNeighborSearch great-circle search with bounds narrower than 360 degrees does not wrap beyond them', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = NEIGHBOR_SEARCH_U5_ORACLE.fixtures.find(f => f.name === 'antimeridian')!;
  const positions = Float32Array.from(fixture.positions);
  // Bounds hugging the antimeridian: [170, 190] is expressed as the full circle because positions
  // use [-180, 180]; here the western bound excludes nothing, so results equal the full-circle run.
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    capacity: fixture.radiusRows.reduce((sum, row) => sum + row.length, 0) + 16,
    parameters: {bounds: [-180, -20, 180, 20], radius: fixture.radius},
    spatialContext: GREAT_CIRCLE,
    gridSize: [12, 4]
  });
  expectRowsMatch(await harness.run(), fixture.radiusRows, 1, 'narrow latitude bounds');
  expect((await harness.run()).invalidCount).toBe(0);
  harness.destroy();
});

it('GPUNeighborSearch kNN adaptive kernels match libpysal Kernel(fixed=False) on planar points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {planar} = NEIGHBOR_SEARCH_U5_ORACLE;
  const positions = Float32Array.from(planar.positions);
  for (const [kernel, oracle] of Object.entries(planar.kernels)) {
    const harness = createNeighborSearchHarness(device, {
      mode: 'knn',
      k: planar.k,
      positions,
      capacity: (positions.length / 2) * planar.k,
      parameters: {
        bounds: [0, 0, 100, 100],
        weightKind: 'kernel',
        kernel: kernel as 'triangular'
      },
      gridSize: [8, 8]
    });
    const result = await harness.run();
    const expectedIds = oracle.rows.flatMap(row => row.map(e => e[0]));
    expect(result.neighbors, `${kernel} neighbors`).toEqual(expectedIds);
    const expectedWeights = oracle.rows.flatMap(row => row.map(e => e[2]));
    for (let slot = 0; slot < expectedWeights.length; slot++) {
      const tolerance = 1e-6 + 2e-5 * Math.abs(expectedWeights[slot]);
      if (Math.abs(result.weights[slot] - expectedWeights[slot]) > tolerance) {
        throw new Error(
          `${kernel} slot ${slot}: weight ${result.weights[slot]} != libpysal ${expectedWeights[slot]}`
        );
      }
    }
    harness.destroy();
  }
});

it('GPUNeighborSearch invalidCount counts masked rows dropped for non-finite coordinates or bounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Rows: 0 ok, 1 NaN, 2 outside bounds, 3 outside but masked out, 4 ok, 5 infinite.
  const positions = new Float32Array([1, 1, NaN, 2, 50, 1, 60, 60, 2, 2, Infinity, 3]);
  const mask = new Uint32Array([1, 1, 1, 0, 1, 1]);
  const harness = createNeighborSearchHarness(device, {
    mode: 'radius',
    positions,
    mask,
    capacity: 16,
    parameters: {bounds: [0, 0, 10, 10], radius: 5}
  });
  const result = await harness.run();
  expect(result.invalidCount).toBe(3);
  expect(result.neighborCounts).toEqual([1, 0, 0, 0, 1, 0]);
  harness.destroy();

  // Cross join: target rows and query rows are both counted.
  const cross = createNeighborSearchHarness(device, {
    mode: 'knn',
    k: 1,
    positions: new Float32Array([1, 1, 20, 20]),
    queryPositions: new Float32Array([2, 2, NaN, NaN, 3, 3]),
    capacity: 8,
    parameters: {bounds: [0, 0, 10, 10]}
  });
  expect((await cross.run()).invalidCount).toBe(2);
  cross.destroy();
});
