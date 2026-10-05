// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURipley} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley';
import {
  getGPURipleyParameterValues,
  GPU_RIPLEY_PARAMETER_LENGTH,
  type GPURipleyParameters
} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyOnCPU, type RipleyOracleResult} from './ripley-oracle';

const BOUNDS = [0, 0, 100, 100] as const;
const RADIUS_COUNT = 12;

type Scene = {positions: Float32Array; mask: Uint32Array};

type RipleyReadback = Record<
  'k' | 'l' | 'lMinusR' | 'pairCorrelation' | 'pairCounts' | 'radii',
  number[]
>;

/** Adds outside-the-window rows and masked rows to a pattern of in-window points. */
function finishScene(points: number[], random: () => number): Scene {
  const rows = points.length / 2;
  const positions = new Float32Array(points);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    mask[row] = random() < 0.92 ? 1 : 0;
    if (random() < 0.04) {
      positions[row * 2] = random() < 0.5 ? -3 - random() : 103 + random();
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
  const centers = Array.from({length: 14}, () => [8 + random() * 84, 8 + random() * 84]);
  const points: number[] = [];
  for (let row = 0; row < rows; row++) {
    const center = centers[row % centers.length];
    const spread = () => (random() + random() + random() - 1.5) * 4;
    points.push(
      Math.min(Math.max(center[0] + spread(), 0), 100),
      Math.min(Math.max(center[1] + spread(), 0), 100)
    );
  }
  return finishScene(points, random);
}

/** 20 x 20 lattice, spacing 5: radii steps and border distances are exact binary fractions. */
function createLatticeScene(seed: number): Scene {
  const points: number[] = [];
  for (let index = 0; index < 400; index++) {
    points.push(((index % 20) + 0.5) * 5, (Math.floor(index / 20) + 0.5) * 5);
  }
  return finishScene(points, createRandom(seed));
}

function createHarness(device: Device, scene: Scene, radiusCount = RADIUS_COUNT) {
  const harness = createPairStatisticsHarness(device, {
    ...scene,
    parameterLength: GPU_RIPLEY_PARAMETER_LENGTH,
    outputs: {
      k: {format: 'float32', length: radiusCount},
      l: {format: 'float32', length: radiusCount},
      lMinusR: {format: 'float32', length: radiusCount},
      pairCorrelation: {format: 'float32', length: radiusCount},
      pairCounts: {format: 'uint32', length: radiusCount},
      radii: {format: 'float32', length: radiusCount}
    },
    createRecipe: views =>
      new GPURipley({
        positions: views.positions,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [32, 32],
        radiusCount,
        k: views.outputs.k as never,
        l: views.outputs.l as never,
        lMinusR: views.outputs.lMinusR as never,
        pairCorrelation: views.outputs.pairCorrelation as never,
        pairCounts: views.outputs.pairCounts as never,
        radii: views.outputs.radii as never
      })
  });
  return {
    run: async (parameters: Float32Array) => (await harness.run(parameters)) as RipleyReadback,
    get rebuildCount() {
      return harness.rebuildCount;
    },
    destroy: () => harness.destroy()
  };
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
  result: RipleyReadback,
  oracle: RipleyOracleResult,
  label: string
): void {
  expect(
    result.pairCounts.reduce((sum, count) => sum + count, 0),
    `${label} pair total`
  ).toBe(oracle.pairCounts.reduce((sum, count) => sum + count, 0));
  for (let b = 0; b < oracle.radii.length; b++) {
    expect(
      Math.abs(result.pairCounts[b] - oracle.pairCounts[b]),
      `${label} pairs ${b}`
    ).toBeLessThanOrEqual(oracle.pairCountTolerance[b]);
    expectClose(`${label} radius ${b}`, result.radii[b], oracle.radii[b], 1e-6 * oracle.radii[b]);
    expectClose(`${label} K ${b}`, result.k[b], oracle.k[b], oracle.kTolerance[b]);
    expectClose(`${label} L ${b}`, result.l[b], oracle.l[b], oracle.lTolerance[b]);
    expectClose(
      `${label} L-r ${b}`,
      result.lMinusR[b],
      oracle.lMinusR[b],
      oracle.lTolerance[b] + 2e-6 * oracle.radii[b]
    );
    expectClose(
      `${label} g ${b}`,
      result.pairCorrelation[b],
      oracle.pairCorrelation[b],
      oracle.pairCorrelationTolerance[b] + 1e-5 * Math.abs(oracle.pairCorrelation[b])
    );
  }
}

const MODES = ['none', 'border', 'isotropic'] as const;

for (const [name, createScene] of [
  ['random', () => createRandomScene(5, 1500)],
  ['clustered', () => createClusteredScene(9, 1500)],
  ['lattice', () => createLatticeScene(3)]
] as const) {
  it(`GPURipley matches the f64 oracle on a ${name} pattern for every edge correction without rebuilding`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createScene();
    const harness = createHarness(device, scene);
    try {
      const frames: GPURipleyParameters[] = [];
      for (const edgeCorrection of MODES) {
        // Small, large and all-pairs (distance beyond the extent) radii, and a sub-window.
        frames.push({bounds: BOUNDS, maximumDistance: 20, edgeCorrection});
        frames.push({bounds: BOUNDS, maximumDistance: 150, edgeCorrection});
        frames.push({bounds: [20, 10, 80, 70], maximumDistance: 12, edgeCorrection});
      }
      for (const frame of frames) {
        const label = `${name} ${frame.edgeCorrection} D=${frame.maximumDistance} [${frame.bounds}]`;
        const result = await harness.run(getGPURipleyParameterValues(frame));
        const oracle = computeRipleyOnCPU(scene, frame, RADIUS_COUNT);
        expect(oracle.pairCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(100);
        expectMatchesOracle(result, oracle, label);
        const again = await harness.run(getGPURipleyParameterValues(frame));
        expect(getFloatBits(again.k), `${label} determinism`).toEqual(getFloatBits(result.k));
        expect(getFloatBits(again.pairCorrelation)).toEqual(getFloatBits(result.pairCorrelation));
        expect(again.pairCounts).toEqual(result.pairCounts);
      }
      expect(harness.rebuildCount).toBe(0);
    } finally {
      harness.destroy();
    }
  }, 240000);
}

it('GPURipley shows clustering, randomness and regularity in L(r) - r', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const frame = {bounds: BOUNDS, maximumDistance: 12, edgeCorrection: 'isotropic'} as const;
  const lMinusR = async (scene: Scene) => {
    const harness = createHarness(
      device,
      {...scene, mask: new Uint32Array(scene.mask.length).fill(1)},
      6
    );
    try {
      return (await harness.run(getGPURipleyParameterValues(frame))).lMinusR;
    } finally {
      harness.destroy();
    }
  };
  const clustered = await lMinusR(createClusteredScene(2, 1200));
  const random = await lMinusR(createRandomScene(4, 1200));
  const lattice = await lMinusR(createLatticeScene(1));
  expect(clustered[2], 'clustered L - r at r = 6').toBeGreaterThan(3);
  expect(Math.abs(random[2]), 'CSR L - r at r = 6').toBeLessThan(1);
  expect(lattice[1], 'regular L - r at r = 4').toBeLessThan(-3);
});

it('GPURipley reproduces a hand-checked corner pair and the three corrections', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Two points at (1, 1) and (3, 1) in [0, 10]^2, d = 2. Weight for the first point: 12 / 5 (corner
  // inside the circle), for the second 3 / 2 (only the bottom side cuts the circle).
  const scene = {positions: new Float32Array([1, 1, 3, 1]), mask: new Uint32Array([1, 1])};
  const harness = createHarness(device, scene, 4);
  try {
    const run = (edgeCorrection: 'none' | 'border' | 'isotropic') =>
      harness.run(
        getGPURipleyParameterValues({bounds: [0, 0, 10, 10], maximumDistance: 4, edgeCorrection})
      );
    const none = await run('none');
    expect(none.pairCounts).toEqual([0, 2, 0, 0]);
    expect(none.k).toEqual([0, 100, 100, 100]);
    expect(none.radii).toEqual([1, 2, 3, 4]);
    const isotropic = await run('isotropic');
    expect(isotropic.k[0]).toBe(0);
    for (let b = 1; b < 4; b++) {
      // A / (n (n - 1)) * (12 / 5 + 3 / 2) = 50 * 3.9
      expect(isotropic.k[b]).toBeCloseTo(195, 2);
    }
    expect(isotropic.l[1]).toBeCloseTo(Math.sqrt(195 / Math.PI), 4);
    // Both points are 1 from the boundary: only r = 1 has a border set, and it sees no pair.
    const border = await run('border');
    expect(border.k[0]).toBe(0);
    expect(border.k.slice(1).every(Number.isNaN)).toBe(true);
    expect(harness.rebuildCount).toBe(0);
  } finally {
    harness.destroy();
  }
});

it('GPURipley reports NaN with fewer than two included points and for an invalid window', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = {positions: new Float32Array([1, 1, 3, 1, 5, 5]), mask: new Uint32Array([1, 0, 1])};
  const harness = createHarness(device, scene, 4);
  try {
    const single = await harness.run(
      getGPURipleyParameterValues({
        bounds: [0, 0, 4, 4],
        maximumDistance: 3,
        edgeCorrection: 'none'
      })
    );
    expect(single.k.every(Number.isNaN)).toBe(true);
    expect(single.l.every(Number.isNaN)).toBe(true);
    expect(single.pairCorrelation.every(Number.isNaN)).toBe(true);
    expect(single.pairCounts).toEqual([0, 0, 0, 0]);
    // A degenerate (zero-area) window has no defined K.
    const flat = await harness.run(
      getGPURipleyParameterValues({
        bounds: [1, 1, 5, 1],
        maximumDistance: 3,
        edgeCorrection: 'none'
      })
    );
    expect(flat.k.every(Number.isNaN)).toBe(true);
    // Raw invalid parameters (bypassing the packer) exclude every row.
    const invalid = await harness.run(new Float32Array([0, 0, 10, 10, -1, 0, 0, 0]));
    expect(invalid.k.every(Number.isNaN)).toBe(true);
  } finally {
    harness.destroy();
  }
});
