// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURipleyDistanceFunctions} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {
  getGPURipleyDistanceParameterValues,
  GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH,
  type GPURipleyDistanceParameters
} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyDistanceFunctionsOnCPU} from './ripley-distance-functions-oracle';

const BOUNDS = [0, 0, 100, 100] as const;
const RADIUS_COUNT = 10;
// Coprime with the window so no reference location sits exactly on a radius or border threshold.
const REFERENCE_GRID = [37, 31] as const;

type Scene = {positions: Float32Array; mask: Uint32Array};

function finishScene(points: number[], random: () => number): Scene {
  const rows = points.length / 2;
  const positions = new Float32Array(points);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    mask[row] = random() < 0.93 ? 1 : 0;
    if (random() < 0.03) positions[row * 2] = 104 + random();
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

function createHarness(device: Parameters<typeof createPairStatisticsHarness>[0], scene: Scene) {
  return createPairStatisticsHarness(device, {
    ...scene,
    parameterLength: GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH,
    outputs: {
      g: {format: 'float32', length: RADIUS_COUNT},
      f: {format: 'float32', length: RADIUS_COUNT},
      j: {format: 'float32', length: RADIUS_COUNT},
      radii: {format: 'float32', length: RADIUS_COUNT}
    },
    createContributor: views =>
      new GPURipleyDistanceFunctions({
        positions: views.positions,
        mask: views.mask,
        parameters: views.parameters,
        gridSize: [24, 24],
        referenceGrid: REFERENCE_GRID,
        radiusCount: RADIUS_COUNT,
        g: views.outputs.g as never,
        f: views.outputs.f as never,
        j: views.outputs.j as never,
        radii: views.outputs.radii as never
      })
  });
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

for (const [name, createScene] of [
  ['random', () => createRandomScene(5, 700)],
  ['clustered', () => createClusteredScene(9, 700)]
] as const) {
  it(`GPURipleyDistanceFunctions matches the f64 oracle on a ${name} pattern without rebuilding`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const scene = createScene();
    const harness = createHarness(device, scene);
    try {
      const frames: GPURipleyDistanceParameters[] = [];
      for (const edgeCorrection of ['none', 'border'] as const) {
        frames.push({bounds: BOUNDS, maximumDistance: 10, edgeCorrection});
        frames.push({bounds: BOUNDS, maximumDistance: 150, edgeCorrection});
        frames.push({bounds: [20, 10, 80, 70], maximumDistance: 8, edgeCorrection});
      }
      for (const frame of frames) {
        const label = `${name} ${frame.edgeCorrection} D=${frame.maximumDistance} [${frame.bounds}]`;
        const result = await harness.run(getGPURipleyDistanceParameterValues(frame));
        const oracle = computeRipleyDistanceFunctionsOnCPU(
          scene,
          frame,
          RADIUS_COUNT,
          REFERENCE_GRID
        );
        // Nonzero guards: a failed WGSL compile gives zeros.
        expect(oracle.g.some(value => value > 0.05)).toBe(true);
        expect(result.g.some(value => value > 0.05)).toBe(true);
        expect(result.f.some(value => value > 0.01)).toBe(true);
        for (let b = 0; b < RADIUS_COUNT; b++) {
          // Allow a couple of distances that land within f32 rounding of a radius.
          const gTolerance = 3 / Math.max(oracle.gDenominators[b], 1) + 1e-5;
          const fTolerance = 3 / Math.max(oracle.fDenominators[b], 1) + 1e-5;
          expectClose(
            `${label} radius ${b}`,
            result.radii[b],
            oracle.radii[b],
            1e-5 * oracle.radii[b]
          );
          expectClose(`${label} G ${b}`, result.g[b], oracle.g[b], gTolerance);
          expectClose(`${label} F ${b}`, result.f[b], oracle.f[b], fTolerance);
          if (Number.isFinite(oracle.j[b]) && oracle.f[b] < 0.9) {
            expectClose(
              `${label} J ${b}`,
              result.j[b],
              oracle.j[b],
              (gTolerance + fTolerance) * 4 * Math.max(1, oracle.j[b])
            );
          }
        }
        const again = await harness.run(getGPURipleyDistanceParameterValues(frame));
        expect(getFloatBits(again.g), `${label} determinism`).toEqual(getFloatBits(result.g));
        expect(getFloatBits(again.f)).toEqual(getFloatBits(result.f));
        expect(getFloatBits(again.j)).toEqual(getFloatBits(result.j));
      }
      expect(harness.rebuildCount).toBe(0);
    } finally {
      harness.destroy();
    }
  }, 240000);
}

it('GPURipleyDistanceFunctions separates clustering from complete spatial randomness with J', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const frame = {bounds: BOUNDS, maximumDistance: 8, edgeCorrection: 'border'} as const;
  const run = async (scene: Scene) => {
    const harness = createHarness(device, {
      ...scene,
      mask: new Uint32Array(scene.mask.length).fill(1)
    });
    try {
      return await harness.run(getGPURipleyDistanceParameterValues(frame));
    } finally {
      harness.destroy();
    }
  };
  const random = await run(createRandomScene(21, 400));
  // Under CSR, G(r) and F(r) both approach 1 - exp(-lambda pi r^2); J stays near 1.
  const intensity = 400 / 10000;
  for (const b of [1, 3, 5]) {
    const radius = random.radii[b];
    const expected = 1 - Math.exp(-intensity * Math.PI * radius * radius);
    expect(Math.abs(random.g[b] - expected)).toBeLessThan(0.12);
    expect(Math.abs(random.f[b] - expected)).toBeLessThan(0.12);
  }
  expect(Math.abs(random.j[2] - 1)).toBeLessThan(0.4);
  const clustered = await run(createClusteredScene(22, 400));
  // Clustering: events find neighbors sooner than empty space finds events, so J < 1.
  expect(clustered.j[3]).toBeLessThan(0.5);
  expect(clustered.g[3]).toBeGreaterThan(clustered.f[3]);
});
