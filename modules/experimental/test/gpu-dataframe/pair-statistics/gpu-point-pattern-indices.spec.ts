// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUPointPatternIndices} from '../../../src/gpu-dataframe/pair-statistics/gpu-point-pattern-indices';
import {
  getGPUPointPatternIndicesParameterValues,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_STATISTICS_LENGTH,
  type GPUPointPatternIndicesParameters
} from '../../../src/gpu-dataframe/pair-statistics/point-pattern-indices-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {
  computePointPatternIndicesOnCPU,
  type PointPatternIndicesOracleResult
} from './point-pattern-indices-oracle';

const BOUNDS = [0, 0, 100, 100] as const;

type Scene = {positions: Float32Array; mask: Uint32Array};

type PointPatternReadback = Record<
  | 'nearestNeighborDistances'
  | 'nearestNeighborIds'
  | 'clarkEvans'
  | 'quadratCounts'
  | 'quadratStatistics',
  number[]
>;

/** Narrows a harness so its readback has the five named outputs. */
function typeHarness(harness: ReturnType<typeof createPairStatisticsHarness>) {
  return {
    run: async (parameters: Float32Array) =>
      (await harness.run(parameters)) as PointPatternReadback,
    get rebuildCount() {
      return harness.rebuildCount;
    },
    destroy: () => harness.destroy()
  };
}

/** Adds outside-the-window rows and masked rows to a pattern of in-window points. */
function finishScene(points: number[], random: () => number): Scene {
  const rows = points.length / 2;
  const positions = new Float32Array(points);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    mask[row] = random() < 0.93 ? 1 : 0;
    if (random() < 0.03) {
      positions[row * 2 + 1] = random() < 0.5 ? -2 - random() : 102 + random();
    }
  }
  return {positions, mask};
}

function createRandomScene(seed: number, rows: number): Scene {
  const random = createRandom(seed);
  return finishScene(
    Array.from({length: rows * 2}, () => random() * 100),
    random
  );
}

function createClusteredScene(seed: number, rows: number): Scene {
  const random = createRandom(seed);
  const centers = Array.from({length: 10}, () => [8 + random() * 84, 8 + random() * 84]);
  const points: number[] = [];
  for (let row = 0; row < rows; row++) {
    const center = centers[row % centers.length];
    const spread = () => (random() + random() + random() - 1.5) * 3;
    points.push(
      Math.min(Math.max(center[0] + spread(), 0), 100),
      Math.min(Math.max(center[1] + spread(), 0), 100)
    );
  }
  return finishScene(points, random);
}

/** 20 x 20 lattice, spacing 5, every nearest-neighbor distance a tie among up to four rows. */
function createLatticeScene(seed: number): Scene {
  const points: number[] = [];
  for (let index = 0; index < 400; index++) {
    points.push(((index % 20) + 0.5) * 5, (Math.floor(index / 20) + 0.5) * 5);
  }
  return finishScene(points, createRandom(seed));
}

/** A dense blob in one corner and three isolated points far away from everything. */
function createIsolatedScene(seed: number): Scene {
  const random = createRandom(seed);
  const points: number[] = [];
  for (let index = 0; index < 300; index++) {
    points.push(random() * 20, random() * 20);
  }
  points.push(90, 90, 95, 10, 50, 97);
  const rows = points.length / 2;
  return {positions: new Float32Array(points), mask: new Uint32Array(rows).fill(1)};
}

function createHarness(device: Device, scene: Scene, quadratGrid: readonly [number, number]) {
  const rows = scene.positions.length / 2;
  const quadratCount = quadratGrid[0] * quadratGrid[1];
  const harness = createPairStatisticsHarness(device, {
    ...scene,
    parameterLength: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
    outputs: {
      nearestNeighborDistances: {format: 'float32', length: rows},
      nearestNeighborIds: {format: 'uint32', length: rows},
      clarkEvans: {format: 'float32', length: GPU_CLARK_EVANS_LENGTH},
      quadratCounts: {format: 'uint32', length: quadratCount},
      quadratStatistics: {format: 'float32', length: GPU_QUADRAT_STATISTICS_LENGTH}
    },
    createRecipe: views =>
      new GPUPointPatternIndices({
        positions: views.positions,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [32, 32],
        quadratGrid,
        nearestNeighborDistances: views.outputs.nearestNeighborDistances as never,
        nearestNeighborIds: views.outputs.nearestNeighborIds as never,
        clarkEvans: views.outputs.clarkEvans as never,
        quadratCounts: views.outputs.quadratCounts as never,
        quadratStatistics: views.outputs.quadratStatistics as never
      })
  });
  return typeHarness(harness);
}

function expectClose(label: string, actual: number, expected: number, tolerance: number): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  if (!(Math.abs(actual - expected) <= tolerance)) {
    throw new Error(`${label}: ${actual} != ${expected} (tolerance ${tolerance})`);
  }
}

function expectMatchesOracle(
  scene: Scene,
  result: PointPatternReadback,
  oracle: PointPatternIndicesOracleResult,
  label: string
): void {
  const rows = scene.positions.length / 2;
  for (let row = 0; row < rows; row++) {
    expectClose(
      `${label} distance ${row}`,
      result.nearestNeighborDistances[row],
      oracle.nearestNeighborDistances[row],
      1e-5 * oracle.nearestNeighborDistances[row]
    );
    if (result.nearestNeighborIds[row] !== oracle.nearestNeighborIds[row]) {
      // Only a near-exact distance tie may resolve differently (GPU contraction of dx^2 + dy^2).
      const distanceTo = (other: number) =>
        Math.hypot(
          scene.positions[other * 2] - scene.positions[row * 2],
          scene.positions[other * 2 + 1] - scene.positions[row * 2 + 1]
        );
      const gpuDistance = distanceTo(result.nearestNeighborIds[row]);
      expect(
        Math.abs(gpuDistance - oracle.nearestNeighborDistances[row]),
        `${label} id ${row}: ${result.nearestNeighborIds[row]} vs ${oracle.nearestNeighborIds[row]}`
      ).toBeLessThan(1e-5 * gpuDistance);
    }
  }
  const [n, observed, expected, ratio, standardError, zScore] = oracle.clarkEvans;
  expect(result.clarkEvans[0], `${label} n`).toBe(n);
  expectClose(`${label} observed`, result.clarkEvans[1], observed, 1e-5 * Math.abs(observed));
  expectClose(`${label} expected`, result.clarkEvans[2], expected, 1e-6 * Math.abs(expected));
  expectClose(`${label} ratio`, result.clarkEvans[3], ratio, 1e-5 * Math.abs(ratio));
  expectClose(`${label} se`, result.clarkEvans[4], standardError, 1e-6 * Math.abs(standardError));
  expectClose(`${label} z`, result.clarkEvans[5], zScore, 5e-3 + 1e-4 * Math.abs(zScore));
  expect(result.quadratCounts, `${label} quadrat counts`).toEqual(oracle.quadratCounts);
  const stats = oracle.quadratStatistics;
  expect(result.quadratStatistics[0]).toBe(stats[0]);
  expectClose(`${label} mean`, result.quadratStatistics[1], stats[1], 1e-6 * Math.abs(stats[1]));
  expectClose(
    `${label} variance`,
    result.quadratStatistics[2],
    stats[2],
    1e-4 * Math.abs(stats[2]) + 1e-6
  );
  expectClose(
    `${label} vmr`,
    result.quadratStatistics[3],
    stats[3],
    2e-4 * Math.abs(stats[3]) + 1e-6
  );
  expectClose(
    `${label} chi`,
    result.quadratStatistics[4],
    stats[4],
    2e-4 * Math.abs(stats[4]) + 1e-5
  );
  expect(result.quadratStatistics[5]).toBe(stats[5]);
}

for (const [name, createScene, quadratGrid] of [
  ['random', () => createRandomScene(5, 1500), [5, 4]],
  ['clustered', () => createClusteredScene(9, 1500), [6, 6]],
  ['lattice', () => createLatticeScene(3), [4, 4]],
  ['isolated', () => createIsolatedScene(2), [3, 2]]
] as const) {
  it(`GPUPointPatternIndices matches the f64 oracle on a ${name} pattern across per-frame changes without rebuilding`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createScene();
    const harness = createHarness(device, scene, quadratGrid);
    try {
      const frames: GPUPointPatternIndicesParameters[] = [
        {bounds: BOUNDS, maximumDistance: 3},
        {bounds: BOUNDS, maximumDistance: 15},
        // Beyond the extent: the lattice collapses to one cell.
        {bounds: BOUNDS, maximumDistance: 400},
        {bounds: [10, 10, 90, 90], maximumDistance: 4}
      ];
      let firstDistances: number[] | undefined;
      for (const frame of frames) {
        const label = `${name} D=${frame.maximumDistance} [${frame.bounds}]`;
        const result = await harness.run(getGPUPointPatternIndicesParameterValues(frame));
        const oracle = computePointPatternIndicesOnCPU(scene, frame, quadratGrid);
        expect(oracle.clarkEvans[0]).toBeGreaterThan(20);
        expectMatchesOracle(scene, result, oracle, label);
        if (frame.bounds === BOUNDS) {
          // The search-cell hint changes only speed, never the result.
          firstDistances ??= result.nearestNeighborDistances;
          expect(getFloatBits(result.nearestNeighborDistances), `${label} hint`).toEqual(
            getFloatBits(firstDistances)
          );
        }
        const again = await harness.run(getGPUPointPatternIndicesParameterValues(frame));
        expect(getFloatBits(again.clarkEvans), `${label} determinism`).toEqual(
          getFloatBits(result.clarkEvans)
        );
        expect(getFloatBits(again.quadratStatistics)).toEqual(
          getFloatBits(result.quadratStatistics)
        );
        expect(again.nearestNeighborIds).toEqual(result.nearestNeighborIds);
        expect(again.quadratCounts).toEqual(result.quadratCounts);
      }
      expect(harness.rebuildCount).toBe(0);
    } finally {
      harness.destroy();
    }
  }, 240000);
}

it('GPUPointPatternIndices separates clustered, random and regular patterns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const frame = {bounds: BOUNDS, maximumDistance: 6};
  const run = async (scene: Scene) => {
    const harness = createHarness(
      device,
      {...scene, mask: new Uint32Array(scene.mask.length).fill(1)},
      [10, 10]
    );
    try {
      return await harness.run(getGPUPointPatternIndicesParameterValues(frame));
    } finally {
      harness.destroy();
    }
  };
  const clustered = await run(createClusteredScene(2, 1000));
  const random = await run(createRandomScene(4, 1000));
  const lattice = await run(createLatticeScene(1));
  // Clark-Evans R: below 1 clustered, near 1 random (ArcGIS: R ~ 1.0 +- 0.1), above 1 regular.
  expect(clustered.clarkEvans[3]).toBeLessThan(0.7);
  expect(clustered.clarkEvans[5]).toBeLessThan(-5);
  expect(Math.abs(random.clarkEvans[3] - 1)).toBeLessThan(0.1);
  expect(lattice.clarkEvans[3]).toBeGreaterThan(1.5);
  expect(lattice.clarkEvans[5]).toBeGreaterThan(5);
  // Variance-to-mean ratio: well above 1 clustered, about 1 random, near 0 regular.
  expect(clustered.quadratStatistics[3]).toBeGreaterThan(3);
  expect(Math.abs(random.quadratStatistics[3] - 1)).toBeLessThan(0.7);
  expect(lattice.quadratStatistics[3]).toBeLessThan(0.5);
});

it('GPUPointPatternIndices reproduces a hand-checked four-point pattern and its ties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Window area 100; two close pairs; quadrats [2, 2] hold counts [2, 0, 0, 2].
  const scene = {
    positions: new Float32Array([1, 1, 2, 1, 8, 8, 9, 8]),
    mask: new Uint32Array(4).fill(1)
  };
  const harness = createHarness(device, scene, [2, 2]);
  try {
    const result = await harness.run(
      getGPUPointPatternIndicesParameterValues({bounds: [0, 0, 10, 10], maximumDistance: 3})
    );
    expect(result.nearestNeighborDistances).toEqual([1, 1, 1, 1]);
    expect(result.nearestNeighborIds).toEqual([1, 0, 3, 2]);
    const expected = 0.5 / Math.sqrt(4 / 100);
    expect(result.clarkEvans[0]).toBe(4);
    expect(result.clarkEvans[1]).toBe(1);
    expect(result.clarkEvans[2]).toBeCloseTo(expected, 5);
    expect(result.clarkEvans[3]).toBeCloseTo(1 / expected, 5);
    expect(result.quadratCounts).toEqual([2, 0, 0, 2]);
    expect(result.quadratStatistics[0]).toBe(4);
    expect(result.quadratStatistics[1]).toBe(1);
    expect(result.quadratStatistics[2]).toBeCloseTo(4 / 3, 5);
    expect(result.quadratStatistics[3]).toBeCloseTo(4 / 3, 5);
    expect(result.quadratStatistics[4]).toBeCloseTo(4, 5);
    expect(result.quadratStatistics[5]).toBe(3);
  } finally {
    harness.destroy();
  }
  // Equidistant neighbors resolve to the smallest row index, in both orders of the pair.
  const tieScene = {
    positions: new Float32Array([2, 0, 0, 0, 1, 0, 4, 0, 3, 3]),
    mask: new Uint32Array(5).fill(1)
  };
  const tieHarness = createHarness(device, tieScene, [1, 1]);
  try {
    const result = await tieHarness.run(
      getGPUPointPatternIndicesParameterValues({bounds: [0, 0, 5, 5], maximumDistance: 1})
    );
    // Row 2 (x=1) is 1 from row 0 (x=2) and row 1 (x=0): the smaller row 0 wins. Row 0 is 1 from row 2.
    expect(result.nearestNeighborIds.slice(0, 3)).toEqual([2, 2, 0]);
    expect(result.nearestNeighborDistances.slice(0, 3)).toEqual([1, 1, 1]);
  } finally {
    tieHarness.destroy();
  }
});

it('GPUPointPatternIndices finds isolated points exactly beyond the search cells', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createIsolatedScene(8);
  const rows = scene.positions.length / 2;
  const harness = createHarness(device, scene, [2, 2]);
  try {
    // Cells are about 3 wide, so the isolated points (tens of units from anything) need long rings.
    const result = await harness.run(
      getGPUPointPatternIndicesParameterValues({bounds: BOUNDS, maximumDistance: 3})
    );
    const oracle = computePointPatternIndicesOnCPU(
      scene,
      {bounds: BOUNDS, maximumDistance: 3},
      [2, 2]
    );
    for (const row of [rows - 3, rows - 2, rows - 1]) {
      expect(oracle.nearestNeighborDistances[row]).toBeGreaterThan(30);
      expect(result.nearestNeighborIds[row]).toBe(oracle.nearestNeighborIds[row]);
      expect(result.nearestNeighborDistances[row]).toBeCloseTo(
        oracle.nearestNeighborDistances[row],
        3
      );
    }
  } finally {
    harness.destroy();
  }
});

it('GPUPointPatternIndices reports NaN and no neighbor with fewer than two included points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = {
    positions: new Float32Array([1, 1, 3, 1, 50, 50]),
    mask: new Uint32Array([1, 0, 1])
  };
  const harness = createHarness(device, scene, [2, 2]);
  try {
    const result = await harness.run(
      getGPUPointPatternIndicesParameterValues({bounds: [0, 0, 4, 4], maximumDistance: 3})
    );
    expect(result.nearestNeighborDistances.every(Number.isNaN)).toBe(true);
    expect(result.nearestNeighborIds).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
    expect(result.clarkEvans[0]).toBe(1);
    expect(result.clarkEvans.slice(1).every(Number.isNaN)).toBe(true);
    expect(result.quadratCounts).toEqual([1, 0, 0, 0]);
    expect(result.quadratStatistics[0]).toBe(4);
    expect(result.quadratStatistics[1]).toBe(0.25);
    // Invalid parameters exclude every row: no quadrat points, undefined variance-to-mean ratio.
    const invalid = await harness.run(new Float32Array([0, 0, 4, 4, -1, 0, 0, 0]));
    expect(invalid.quadratCounts).toEqual([0, 0, 0, 0]);
    expect(invalid.quadratStatistics[1]).toBe(0);
    expect(invalid.quadratStatistics[3]).toBeNaN();
    expect(invalid.clarkEvans[0]).toBe(0);
  } finally {
    harness.destroy();
  }
});

it('GPUPointPatternIndices builds each part on its own with the same results', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(21, 800);
  const rows = scene.positions.length / 2;
  const frame = {bounds: BOUNDS, maximumDistance: 8};
  const full = createHarness(device, scene, [4, 4]);
  const summaryOnly = typeHarness(
    createPairStatisticsHarness(device, {
      ...scene,
      parameterLength: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
      outputs: {
        clarkEvans: {format: 'float32', length: GPU_CLARK_EVANS_LENGTH},
        quadratStatistics: {format: 'float32', length: GPU_QUADRAT_STATISTICS_LENGTH}
      },
      createRecipe: views =>
        new GPUPointPatternIndices({
          positions: views.positions,
          mask: views.mask,
          parameters: views.parameters,
          gridSize: [32, 32],
          quadratGrid: [4, 4],
          clarkEvans: views.outputs.clarkEvans as never,
          quadratStatistics: views.outputs.quadratStatistics as never
        })
    })
  );
  const idsOnly = typeHarness(
    createPairStatisticsHarness(device, {
      ...scene,
      parameterLength: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
      outputs: {nearestNeighborIds: {format: 'uint32', length: rows}},
      createRecipe: views =>
        new GPUPointPatternIndices({
          positions: views.positions,
          mask: views.mask,
          parameters: views.parameters,
          gridSize: [32, 32],
          nearestNeighborIds: views.outputs.nearestNeighborIds as never
        })
    })
  );
  try {
    const parameters = getGPUPointPatternIndicesParameterValues(frame);
    const expected = await full.run(parameters);
    const summary = await summaryOnly.run(parameters);
    expect(getFloatBits(summary.clarkEvans)).toEqual(getFloatBits(expected.clarkEvans));
    expect(getFloatBits(summary.quadratStatistics)).toEqual(
      getFloatBits(expected.quadratStatistics)
    );
    expect((await idsOnly.run(parameters)).nearestNeighborIds).toEqual(expected.nearestNeighborIds);
  } finally {
    full.destroy();
    summaryOnly.destroy();
    idsOnly.destroy();
  }
});
