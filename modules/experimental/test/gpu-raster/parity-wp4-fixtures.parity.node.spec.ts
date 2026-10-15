// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {
  advanceParticlesOnCPU,
  createParticleAdvectionCPUState
} from '../../src/gpu-raster/particle-advection/particle-advection-cpu';
import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues
} from '../../src/gpu-raster/particle-advection/particle-advection-parameters';
import advectionJSON from '../oracles/fixtures/parity-wp4-advection.json';
import changeJSON from '../oracles/fixtures/parity-wp4-change-detection.json';
import costJSON from '../oracles/fixtures/parity-wp4-cost-distance.json';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import {compareRasterParity} from '../oracles/parity-wp4-compare';

it('WP4 independent fixtures retain source hashes, versions and size limits', () => {
  for (const [name, fixture] of [
    ['advection', advectionJSON],
    ['change-detection', changeJSON],
    ['cost-distance', costJSON]
  ] as const) {
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.versions.numpy).toBeTruthy();
    expect(
      createHash('sha256')
        .update(readFileSync(resolve(fixture.source.path)))
        .digest('hex')
    ).toBe(fixture.source.sha256);
    expect(
      readFileSync(resolve(`modules/experimental/test/oracles/fixtures/parity-wp4-${name}.json`))
        .byteLength
    ).toBeLessThanOrEqual(1_500_000);
  }
});

it('SciPy advection fixture agrees with the analytic rigid rotation at every time', () => {
  const scene = advectionJSON.cases.find(
    entry => entry.name === 'rigid-rotation-anisotropic-cells'
  )!;
  const expected: number[] = [];
  for (const time of scene.times) {
    const angle = time / 8;
    for (let index = 0; index < scene.seeds.length; index += 2) {
      const horizontal = scene.seeds[index] - 8;
      const vertical = scene.seeds[index + 1] - 16;
      expected.push(
        8 + horizontal * Math.cos(angle) - vertical * Math.sin(angle),
        16 + horizontal * Math.sin(angle) + vertical * Math.cos(angle)
      );
    }
  }
  compareRasterParity(
    'GPUParticleAdvection',
    'fixture-analytic-rotation',
    'analytic rigid rotation (fixture validation only)',
    'allReferencePositions',
    scene.positions.flat(),
    expected,
    'L'
  );
  expect(
    Math.max(...scene.positions.flat().map((value, index) => Math.abs(value - expected[index])))
  ).toBeLessThan(1e-10);
});

it('SciPy change fixture preserves missing-time gaps and invalid output masks', () => {
  const scene = decodeOracleFixture<{cases: {name: string; expected: Record<string, number[]>}[]}>(
    changeJSON
  ).cases.find(entry => entry.name === 'ties-missing-original-times-zero-negative-mask')!;
  compareRasterParity(
    'GPUChangeDetection',
    'fixture-original-time-axis',
    'analytic slope between ordinal times 0 and 15 (fixture validation only)',
    'slopeAcrossMissingDates',
    [scene.expected['senSlope'][6]],
    [0.2],
    'S'
  );
  compareRasterParity(
    'GPUChangeDetection',
    'fixture-all-ties',
    'constant-series Mann-Kendall definition (fixture validation only)',
    'score-z-p',
    [
      scene.expected['mannKendallS'][2],
      Number(scene.expected['mannKendallZ'][2]),
      Number(scene.expected['mannKendallP'][2])
    ],
    [0, 0, 1],
    'E'
  );
  for (const name of [
    'senSlope',
    'mannKendallZ',
    'mannKendallP',
    'tStatistic',
    'tDegreesOfFreedom',
    'tPValue'
  ] as const) {
    expect(Number.isNaN(scene.expected[name][4]), `${name} all missing`).toBe(true);
    expect(Number.isNaN(scene.expected[name][11]), `${name} masked`).toBe(true);
  }
});

it('skimage cost fixture agrees with all analytic unit-grid distances and full paths', () => {
  const fixture = decodeOracleFixture<{
    cases: {
      name: string;
      width: number;
      height: number;
      costs: number[];
      paths: {indices: number[]; cost: number}[];
      friction: number[];
      spacing: number[];
    }[];
  }>(costJSON);
  const scene = fixture.cases.find(entry => entry.name === 'unit-diagonals')!;
  const expected = scene.costs.map((_, index) => {
    const row = Math.floor(index / scene.width);
    const column = index % scene.width;
    return Math.min(row, column) * Math.SQRT2 + Math.abs(row - column);
  });
  compareRasterParity(
    'GPUCostDistance',
    'fixture-unit-grid',
    'analytic octile distance (fixture validation only)',
    'everyCost',
    scene.costs,
    expected,
    'L'
  );
  for (const entry of fixture.cases) {
    for (const path of entry.paths) {
      if (!path.indices.length) continue;
      const cost = path.indices.slice(1).reduce((sum, next, index) => {
        const current = path.indices[index];
        const horizontal = ((current % entry.width) - (next % entry.width)) * entry.spacing[0];
        const vertical =
          (Math.floor(current / entry.width) - Math.floor(next / entry.width)) * entry.spacing[1];
        return (
          sum +
          (Math.hypot(horizontal, vertical) * (entry.friction[current] + entry.friction[next])) / 2
        );
      }, 0);
      compareRasterParity(
        'GPUCostDistancePath',
        `fixture-${entry.name}/${path.indices[0]}`,
        'skimage.graph.MCP_Geometric.traceback (fixture validation only)',
        'completeReferencePathCost',
        [cost],
        [path.cost],
        'L'
      );
    }
  }
});

it('particle CPU mirror diagnostic stays within the independent ODE budget (not a GPU run)', () => {
  const fixture = decodeOracleFixture<{
    cases: {
      name: string;
      width: number;
      height: number;
      field: number[];
      seeds: number[];
      extent: [number, number, number, number];
      spawnBounds: [number, number, number, number];
      timeStep: number;
      times: number[];
      positions: number[][];
      speeds: number[][];
    }[];
  }>(advectionJSON);
  for (const scene of fixture.cases) {
    const state = createParticleAdvectionCPUState(scene.seeds.length / 2);
    state.positions.set(scene.seeds);
    const field = {
      width: scene.width,
      height: scene.height,
      velocities: new Float32Array(scene.field)
    };
    const parameters = getGPUParticleAdvectionParameterValues(
      {fieldExtent: scene.extent, spawnBounds: scene.spawnBounds, timeStep: scene.timeStep},
      [scene.width, scene.height]
    );
    const positions: number[] = [];
    const speeds: number[] = [];
    for (let frame = 0; frame < scene.times.length; frame++) {
      advanceParticlesOnCPU(
        state,
        field,
        parameters,
        getGPUParticleAdvectionUint32ParameterValues({seed: 7, frame, maximumAge: 0})
      );
      positions.push(...state.positions);
      speeds.push(...state.speeds);
    }
    compareRasterParity(
      'GPUParticleAdvection',
      `CPU-mirror-diagnostic/${scene.name}`,
      'scipy.integrate.solve_ivp + scipy.ndimage.map_coordinates (not GPU evidence)',
      'allPositions',
      positions,
      scene.positions.flat(),
      'L'
    );
    compareRasterParity(
      'GPUParticleAdvection',
      `CPU-mirror-diagnostic/${scene.name}`,
      'scipy.integrate.solve_ivp + scipy.ndimage.map_coordinates (not GPU evidence)',
      'allSpeeds',
      speeds,
      scene.speeds.flat(),
      'L'
    );
  }
});
