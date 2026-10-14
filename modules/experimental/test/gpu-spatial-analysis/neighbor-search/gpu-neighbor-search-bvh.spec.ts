// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * `searchStructure` equivalence: the BVH path and the GPU-gated `'auto'` path must give exactly the
 * grid path's IDs, offsets, distances and weights (bitwise), and the brute-force oracle's and
 * scipy cKDTree's neighbors, on clustered, duplicate, collinear, masked and invalid points.
 *
 * Fixture: `test/oracles/geometry-neighbor-search-clustered.py` (integer coordinates, so float32
 * squared distances are exact and ties are real; ties resolved by lowest id).
 */

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {
  GPUNeighborSearchSettings,
  GPUNeighborSearchStructure
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {
  createNeighborSearchHarness,
  type NeighborSearchHarnessOptions,
  type NeighborSearchReadback
} from './neighbor-search-harness';
import {computeNeighborSearchOracle, createSeededRandom} from './neighbor-search-oracle';
import {
  NEIGHBOR_SEARCH_CLUSTERED_ORACLE,
  type ClusteredCSR
} from './neighbor-search-clustered-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const STRUCTURES: GPUNeighborSearchStructure[] = ['grid', 'bvh', 'auto'];
const BOUNDS = NEIGHBOR_SEARCH_CLUSTERED_ORACLE.bounds;
const f32 = Math.fround;

/** Weights of a kernel make any difference in a neighbor's distance visible as a different bit pattern. */
const KERNEL_PARAMETERS = {weightKind: 'kernel', kernel: 'gaussian'} as const;

async function runStructure(
  device: Parameters<typeof createNeighborSearchHarness>[0],
  options: Omit<NeighborSearchHarnessOptions, 'searchStructure'>,
  structure: GPUNeighborSearchStructure
): Promise<NeighborSearchReadback> {
  const harness = createNeighborSearchHarness(device, {...options, searchStructure: structure});
  try {
    return await harness.run();
  } finally {
    harness.destroy();
  }
}

/** Exact equality of everything a consumer reads, including the weight bit patterns. */
function expectIdentical(
  actual: NeighborSearchReadback,
  expected: NeighborSearchReadback,
  label: string
): void {
  expect(actual.offsets, `${label} offsets`).toEqual(expected.offsets);
  expect(actual.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
  expect(actual.distances, `${label} distances`).toEqual(expected.distances);
  expect(actual.weightBits, `${label} weights`).toEqual(expected.weightBits);
  expect(actual.neighborCounts, `${label} counts`).toEqual(expected.neighborCounts);
  expect(actual.overflow, `${label} overflow`).toBe(expected.overflow);
  expect(actual.totalNeighbors, `${label} total`).toBe(expected.totalNeighbors);
  expect(actual.invalidCount, `${label} invalid`).toBe(expected.invalidCount);
}

function expectMatchesScipy(result: NeighborSearchReadback, csr: ClusteredCSR, label: string) {
  expect(result.offsets, `${label} offsets`).toEqual(csr.offsets);
  expect(result.neighbors, `${label} neighbors`).toEqual(csr.neighbors);
  for (let slot = 0; slot < csr.neighbors.length; slot++) {
    const expected = f32(Math.sqrt(csr.distancesSquared[slot]));
    // Two ulp: the GPU sqrt is correctly rounded or within one ulp of it.
    expect(
      Math.abs(result.distances[slot] - expected),
      `${label} slot ${slot} distance ${result.distances[slot]} vs ${expected}`
    ).toBeLessThanOrEqual(2.4e-7 * expected);
  }
}

it('GPUNeighborSearch grid, bvh and auto equal each other, scipy and the oracle on clustered scenes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of NEIGHBOR_SEARCH_CLUSTERED_ORACLE.scenes) {
    const positions = Float32Array.from(scene.positions);
    const mask = Array.isArray(scene.mask) ? Uint32Array.from(scene.mask) : undefined;
    const queryPositions = Array.isArray(scene.queryPositions)
      ? Float32Array.from(scene.queryPositions)
      : undefined;
    const base = {positions, mask, queryPositions, gridSize: [32, 32] as [number, number]};
    const cases = [
      ...Object.entries(scene.knn).map(([k, csr]) => ({
        label: `${scene.name} k=${k}`,
        csr,
        mode: 'knn' as const,
        k: Number(k),
        parameters: {bounds: BOUNDS, ...KERNEL_PARAMETERS} as GPUNeighborSearchSettings
      })),
      ...Object.entries(scene.radius).map(([radius, csr]) => ({
        label: `${scene.name} radius=${radius}`,
        csr,
        mode: 'radius' as const,
        k: undefined,
        parameters: {
          bounds: BOUNDS,
          radius: Number(radius),
          ...KERNEL_PARAMETERS
        } as GPUNeighborSearchSettings
      }))
    ];
    for (const {label, csr, mode, k, parameters} of cases) {
      const options = {
        ...base,
        mode,
        k,
        parameters,
        capacity: Math.max(csr.neighbors.length, 1)
      };
      const results: Partial<Record<GPUNeighborSearchStructure, NeighborSearchReadback>> = {};
      for (const structure of STRUCTURES) {
        results[structure] = await runStructure(device, options, structure);
      }
      expectMatchesScipy(results.grid!, csr, `${label} grid`);
      expectIdentical(results.bvh!, results.grid!, `${label} bvh vs grid`);
      expectIdentical(results.auto!, results.grid!, `${label} auto vs grid`);
      // The existing brute-force oracle agrees too (it shares the f32 distance expression).
      const oracle = computeNeighborSearchOracle({
        mode,
        k,
        positions,
        mask,
        queryPositions,
        parameters
      });
      expect(results.bvh!.neighbors, `${label} oracle`).toEqual(oracle.neighbors);
      expect(results.bvh!.offsets, `${label} oracle offsets`).toEqual(oracle.offsets);
    }
  }
}, 600_000);

it('GPUNeighborSearch bvh and auto equal the grid with a query mask and an exceeded capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(31);
  const count = 900;
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const clustered = row % 5 !== 0;
    positions[row * 2] = f32(clustered ? 500 + random() * 20 : random() * 1000);
    positions[row * 2 + 1] = f32(clustered ? 500 + random() * 20 : random() * 1000);
  }
  const queryMask = new Uint32Array(count).fill(1);
  for (let row = 0; row < count; row += 4) queryMask[row] = 0;
  const parameters: GPUNeighborSearchSettings = {bounds: [0, 0, 1000, 1000], radius: 6};
  for (const mode of ['knn', 'radius'] as const) {
    const options = {
      mode,
      k: mode === 'knn' ? 8 : undefined,
      positions,
      queryMask,
      parameters: mode === 'knn' ? {bounds: parameters.bounds} : parameters,
      gridSize: [16, 16] as [number, number]
    };
    // Radius output can contain every unmasked query's other count - 1 points.
    const fullCapacity = count * (count - 1);
    const full = await runStructure(device, {...options, capacity: fullCapacity}, 'grid');
    expect(full.overflow).toBe(0);
    const total = full.offsets[count];
    expect(total).toBeGreaterThan(0);
    for (const structure of ['bvh', 'auto'] as const) {
      expectIdentical(
        await runStructure(device, {...options, capacity: fullCapacity}, structure),
        full,
        `${mode} ${structure} query mask`
      );
    }
    // Capacity cut in the middle of a row: offsets, flags and totals are still identical; only
    // the one cut row may keep different neighbors, every complete row is identical.
    const capacity = Math.floor(total / 2);
    const grid = await runStructure(device, {...options, capacity}, 'grid');
    expect(grid.overflow).toBe(1);
    for (const structure of ['bvh', 'auto'] as const) {
      const other = await runStructure(device, {...options, capacity}, structure);
      expect(other.offsets, `${mode} ${structure} capped offsets`).toEqual(grid.offsets);
      expect(other.overflow).toBe(1);
      expect(other.totalNeighbors).toBe(grid.totalNeighbors);
      expect(other.neighborCounts).toEqual(grid.neighborCounts);
      const cutRow = grid.offsets.findIndex((offset, row) => row > 0 && offset === capacity);
      const completeEnd = grid.offsets[Math.max(cutRow - 1, 0)];
      expect(other.neighbors.slice(0, completeEnd)).toEqual(grid.neighbors.slice(0, completeEnd));
      expect(other.distances.slice(0, completeEnd)).toEqual(grid.distances.slice(0, completeEnd));
    }
  }
}, 600_000);

it('GPUNeighborSearch bvh and auto equal the grid when per-frame inputs change without a rebuild', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(77);
  const count = 700;
  const positions = new Float32Array(count * 2);
  for (let index = 0; index < positions.length; index++) positions[index] = f32(random() * 100);
  const mask = new Uint32Array(count).fill(1);
  const harnesses = Object.fromEntries(
    STRUCTURES.map(structure => [
      structure,
      createNeighborSearchHarness(device, {
        mode: 'knn',
        k: 6,
        positions,
        mask,
        capacity: count * 6,
        parameters: {bounds: [0, 0, 100, 100], ...KERNEL_PARAMETERS},
        gridSize: [16, 16],
        searchStructure: structure
      })
    ])
  ) as Record<GPUNeighborSearchStructure, ReturnType<typeof createNeighborSearchHarness>>;
  try {
    // Bounds shrink (points outside become invalid), then a mask change, then clustered positions.
    const frames: (() => void)[] = [
      () => {},
      () => {},
      () => {
        for (const structure of STRUCTURES) {
          harnesses[structure].writeMask(
            Uint32Array.from(mask, (_, row) => (row % 3 === 0 ? 0 : 1))
          );
        }
      },
      () => {
        const clustered = positions.slice();
        for (let row = 0; row < count; row++) {
          if (row % 10 !== 0) {
            clustered[row * 2] = f32(50 + random());
            clustered[row * 2 + 1] = f32(50 + random());
          }
        }
        for (const structure of STRUCTURES) harnesses[structure].writePositions(clustered);
      }
    ];
    const parameterFrames: GPUNeighborSearchSettings[] = [
      {bounds: [0, 0, 100, 100], ...KERNEL_PARAMETERS},
      {bounds: [10, 10, 60, 90], ...KERNEL_PARAMETERS},
      {bounds: [0, 0, 100, 100], ...KERNEL_PARAMETERS},
      {bounds: [0, 0, 100, 100], ...KERNEL_PARAMETERS}
    ];
    for (const [frame, update] of frames.entries()) {
      update();
      const grid = await harnesses.grid.run(parameterFrames[frame]);
      for (const structure of ['bvh', 'auto'] as const) {
        expectIdentical(
          await harnesses[structure].run(parameterFrames[frame]),
          grid,
          `frame ${frame} ${structure}`
        );
      }
    }
    for (const structure of STRUCTURES) {
      expect(harnesses[structure].buildCount).toBe(1);
    }
  } finally {
    for (const structure of STRUCTURES) harnesses[structure].destroy();
  }
}, 600_000);
