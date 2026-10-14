// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURipleyDistanceFunctions} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {
  getGPURipleyDistanceFunctionsParameterValues,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
  type GPURipleyDistanceFunctionsSettings
} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {createPairStatisticsHarness, getFloatBits} from './pair-statistics-harness';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyDistanceFunctionsOnCPU} from './ripley-distance-functions-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

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
    parameterLength: GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
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
        output: {
          g: views.outputs.g as never,
          f: views.outputs.f as never,
          j: views.outputs.j as never,
          radii: views.outputs.radii as never
        }
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
    requireTestDevice(device);
    const scene = createScene();
    const harness = createHarness(device, scene);
    try {
      const frames: GPURipleyDistanceFunctionsSettings[] = [];
      for (const edgeCorrection of ['none', 'border', 'kaplan-meier', 'hanisch'] as const) {
        frames.push({bounds: BOUNDS, maximumDistance: 10, edgeCorrection});
        frames.push({bounds: BOUNDS, maximumDistance: 150, edgeCorrection});
        frames.push({bounds: [20, 10, 80, 70], maximumDistance: 8, edgeCorrection});
      }
      for (const frame of frames) {
        const label = `${name} ${frame.edgeCorrection} D=${frame.maximumDistance} [${frame.bounds}]`;
        const result = await harness.run(getGPURipleyDistanceFunctionsParameterValues(frame));
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
          const weighted =
            frame.edgeCorrection === 'kaplan-meier' || frame.edgeCorrection === 'hanisch';
          const slack = weighted ? 4 : 1;
          const gTolerance =
            (3 * slack) / Math.max(oracle.gDenominators[b], 1) + (weighted ? 1e-3 : 1e-5);
          const fTolerance =
            (3 * slack) / Math.max(oracle.fDenominators[b], 1) + (weighted ? 1e-3 : 1e-5);
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
        const again = await harness.run(getGPURipleyDistanceFunctionsParameterValues(frame));
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
  requireTestDevice(device);
  const frame = {bounds: BOUNDS, maximumDistance: 8, edgeCorrection: 'border'} as const;
  const run = async (scene: Scene) => {
    const harness = createHarness(device, {
      ...scene,
      mask: new Uint32Array(scene.mask.length).fill(1)
    });
    try {
      return await harness.run(getGPURipleyDistanceFunctionsParameterValues(frame));
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

import {createPairStatisticsWindowHarness} from './pair-statistics-harness';
import {
  ORACLE_RECTANGLE,
  ORACLE_L_SHAPE,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';

it('polygon G/F/J matches rectangle border/KM and the same-lattice Hanisch oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = new Float32Array(240);
  const random = createRandom(178);
  for (let row = 0; row < positions.length / 2; row++) {
    positions[row * 2] = 20 + random() * 1960;
    positions[row * 2 + 1] = 20 + random() * 960;
  }
  const radiusCount = 8;
  const referenceGrid = [53, 47] as const;
  const makeHarness = (window?: OracleStudyWindow) => {
    const common = {
      positions,
      parameterLength: GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
      outputs: {
        g: {format: 'float32', length: radiusCount},
        f: {format: 'float32', length: radiusCount},
        j: {format: 'float32', length: radiusCount},
        radii: {format: 'float32', length: radiusCount}
      }
    } as const;
    if (window)
      return createPairStatisticsWindowHarness(device, {
        ...common,
        window,
        createContributor: (views, studyWindow) =>
          new GPURipleyDistanceFunctions({
            positions: views.positions,
            parameters: views.parameters,
            studyWindow,
            gridSize: [16, 16],
            referenceGrid,
            radiusCount,
            output: {
              g: views.outputs.g as never,
              f: views.outputs.f as never,
              j: views.outputs.j as never,
              radii: views.outputs.radii as never
            }
          })
      });
    return createPairStatisticsHarness(device, {
      ...common,
      createContributor: views =>
        new GPURipleyDistanceFunctions({
          positions: views.positions,
          parameters: views.parameters,
          gridSize: [16, 16],
          referenceGrid,
          radiusCount,
          output: {
            g: views.outputs.g as never,
            f: views.outputs.f as never,
            j: views.outputs.j as never,
            radii: views.outputs.radii as never
          }
        })
    });
  };
  const rectangle = makeHarness();
  const polygonRectangle = makeHarness(ORACLE_RECTANGLE);
  const polygon = makeHarness(ORACLE_L_SHAPE);
  try {
    for (const edgeCorrection of ['border', 'kaplan-meier'] as const) {
      const frame = {bounds: [0, 0, 2000, 1000] as const, maximumDistance: 160, edgeCorrection};
      const expected = await rectangle.run(getGPURipleyDistanceFunctionsParameterValues(frame));
      const actual = await polygonRectangle.run(
        getGPURipleyDistanceFunctionsParameterValues({...frame, bounds: 'study-window'})
      );
      for (const name of ['g', 'f', 'j'] as const) {
        actual[name].forEach((value, index) =>
          expectClose(`${edgeCorrection} ${name} ${index}`, value, expected[name][index], 1e-5)
        );
      }
    }
    for (const edgeCorrection of ['none', 'hanisch'] as const) {
      const frame = {bounds: 'study-window', maximumDistance: 160, edgeCorrection} as const;
      const actual = await polygon.run(getGPURipleyDistanceFunctionsParameterValues(frame));
      const expected = computeRipleyDistanceFunctionsOnCPU(
        {positions},
        frame,
        radiusCount,
        referenceGrid,
        ORACLE_L_SHAPE
      );
      expect(expected.fDenominators[0]).toBeLessThan(referenceGrid[0] * referenceGrid[1]);
      expect(actual.f.some(value => value > 0)).toBe(true);
      for (const name of ['g', 'f', 'j'] as const) {
        actual[name].forEach((value, index) =>
          expectClose(`${edgeCorrection} ${name} ${index}`, value, expected[name][index], 1e-4)
        );
      }
      const again = await polygon.run(getGPURipleyDistanceFunctionsParameterValues(frame));
      expect(getFloatBits(actual.g)).toEqual(getFloatBits(again.g));
      expect(getFloatBits(actual.f)).toEqual(getFloatBits(again.f));
    }
  } finally {
    rectangle.destroy();
    polygonRectangle.destroy();
    polygon.destroy();
  }
}, 240000);
