// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  GPUTerrainHorizon,
  type GPUTerrainHorizonSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {getTerrainHorizonSweepNode} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainHorizon, createRandom, createSmoothTerrain} from './terrain-horizon-oracle';
import {
  computeTerrainHorizonSweepBruteForce,
  computeTerrainHorizonSweepHull
} from './terrain-horizon-sweep-oracle';

// Opt in with `VITE_TERRAIN_SWEEP_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_TERRAIN_SWEEP_BENCHMARK'
  ] === 'true';

type SweepFixtureOptions = {
  width: number;
  height: number;
  directionCount: number;
  maximumRadius: number;
  cellSizeMode?: GPUTerrainCellSizeMode;
  rowDirection?: 'south' | 'north';
  zFactorSign?: 1 | -1;
  /** `horizon` stores every angle pixel-major, `sine` accumulates `sin(max(h, 0))` per pixel. */
  output: 'horizon' | 'sine';
};

type SweepFixture = {
  run(settings: GPUTerrainHorizonSettings): Promise<number[]>;
  submit(settings: GPUTerrainHorizonSettings): void;
  compiled: {destroy(): void};
  destroy(): void;
};

/** Test-only harness: one sweep node per sector, outputs written through `outputWGSL`. */
function createSweepFixture(
  device: Device,
  elevation: Float32Array,
  options: SweepFixtureOptions
): SweepFixture {
  const {width, height, directionCount} = options;
  const pixelCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'terrain-sweep-test'});
  const values = createInputBuffer(
    device,
    Float32Array.from(elevation, value => (Number.isFinite(value) ? value : 0))
  );
  const validity = createInputBuffer(
    device,
    Uint32Array.from(elevation, value => (Number.isFinite(value) ? 1 : 0))
  );
  const outputLength = options.output === 'horizon' ? pixelCount * directionCount : pixelCount;
  const output = createOutputBuffer(device, outputLength);
  const settings = new GPUParameterBuffer(device, {
    id: 'sweep-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainHorizonParameterValues({cellSize: [10, 10]})
  });
  graph.add({
    id: 'sweep-harness',
    getCommandNodes: nodeGraph => {
      const hull = createTransientView(nodeGraph, 'sweep-hull', 'uint32', pixelCount);
      const valuesView = importGraphBuffer(nodeGraph, 'values', values, 'float32', pixelCount);
      const validityView = importGraphBuffer(nodeGraph, 'validity', validity, 'uint32', pixelCount);
      const settingsView = settings.importToGraph(nodeGraph);
      const outputView = importGraphBuffer(nodeGraph, 'out', output, 'float32', outputLength);
      return Array.from({length: directionCount}, (_, sector) =>
        getTerrainHorizonSweepNode(nodeGraph, {
          id: `sweep-${sector}`,
          width,
          height,
          direction: getGPUTerrainHorizonDirection(
            sector,
            directionCount,
            options.rowDirection ?? 'south'
          ),
          maximumRadius: options.maximumRadius,
          cellSizeMode: options.cellSizeMode ?? 'uniform',
          zFactorSign: options.zFactorSign ?? 1,
          elevationValues: valuesView,
          elevationValidity: validityView,
          settings: settingsView,
          hull,
          outputBindings: [{name: 'out', view: outputView, type: 'f32', access: 'read_write'}],
          outputDeclarations: `const DIRECTION_COUNT: u32 = ${directionCount}u;
const SECTOR: u32 = ${sector}u;`,
          outputWGSL:
            options.output === 'horizon'
              ? 'out[outOffset + pixel * DIRECTION_COUNT + SECTOR] = horizonAngle;'
              : `let sine = sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);
      ${sector === 0 ? 'out[outOffset + pixel] = sine;' : 'out[outOffset + pixel] += sine;'}`
        })
      );
    }
  });
  const compiled = graph.compile();
  return {
    compiled,
    submit(values) {
      settings.write(getGPUTerrainHorizonParameterValues(values));
      submitGraph(device, compiled, undefined);
    },
    async run(values) {
      this.submit(values);
      return readFloat32(output, outputLength);
    },
    destroy() {
      compiled.destroy();
      settings.destroy();
      values.destroy();
      validity.destroy();
      output.destroy();
    }
  };
}

function expectCloseToOracle(actual: number[], expected: number[], tolerance: number): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `NaN pattern at ${index}`).toBe(true);
      continue;
    }
    expect(Number.isFinite(actual[index]), `finite at ${index}`).toBe(true);
    worst = Math.max(worst, Math.abs(actual[index] - value));
  }
  expect(worst).toBeLessThan(tolerance);
  return worst;
}

function createNoisyTerrain(width: number, height: number, seed: number): Float32Array {
  const random = createRandom(seed);
  const terrain = createSmoothTerrain(width, height, seed);
  for (let index = 0; index < terrain.length; index++) {
    terrain[index] += (random() - 0.5) * 12;
  }
  for (const hole of [5 * width + 7, 5 * width + 8, 20 * width + 30, (height - 1) * width + 3]) {
    terrain[hole] = NaN;
  }
  for (let row = 10; row < 13; row++) {
    for (let column = 14; column < 19; column++) {
      terrain[row * width + column] = NaN;
    }
  }
  return terrain;
}

it('sweep kernel matches the float64 brute force on random terrain with nodata', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 48;
  const height = 40;
  const elevation = createNoisyTerrain(width, height, 21);
  const settingsList: GPUTerrainHorizonSettings[] = [
    {cellSize: [10, 10]},
    {cellSize: [10, 12], zFactor: 2},
    {cellSize: [10, 10], curvatureCoefficient: 1e-3, maximumDistance: 95}
  ];
  for (const [directionCount, maximumRadius] of [
    [8, 12],
    [16, 30],
    [7, 100],
    [16, 3]
  ] as const) {
    const fixture = createSweepFixture(device, elevation, {
      width,
      height,
      directionCount,
      maximumRadius,
      output: 'horizon'
    });
    for (const settings of settingsList) {
      const actual = await fixture.run(settings);
      const expected = computeTerrainHorizonSweepBruteForce({
        width,
        height,
        elevation,
        directionCount,
        maximumRadius,
        cellSize: [settings.cellSize[0], settings.cellSize[1]],
        zFactor: settings.zFactor,
        curvatureCoefficient: settings.curvatureCoefficient,
        maximumDistance: settings.maximumDistance
      });
      // A failed WGSL compile reads back as zeros: require real structure first.
      expect(actual.filter(angle => angle > 5).length).toBeGreaterThan(50);
      expect(actual.filter(angle => angle < -5).length).toBeGreaterThan(50);
      expectCloseToOracle(actual, expected.horizon, 2e-3);
    }
    fixture.destroy();
  }
});

it('sweep kernel supports north rows, Web Mercator cells, and the nadir pass', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 32;
  const height = 24;
  const elevation = createNoisyTerrain(width, height, 5);
  const settings: GPUTerrainHorizonSettings = {
    cellSize: [20, 20],
    northEdge: 0.3,
    southEdge: 0.32
  };
  for (const zFactorSign of [1, -1] as const) {
    const fixture = createSweepFixture(device, elevation, {
      width,
      height,
      directionCount: 12,
      maximumRadius: 10,
      rowDirection: 'north',
      cellSizeMode: 'web-mercator',
      zFactorSign,
      output: 'horizon'
    });
    const actual = await fixture.run(settings);
    const expected = computeTerrainHorizonSweepBruteForce({
      width,
      height,
      elevation,
      directionCount: 12,
      maximumRadius: 10,
      rowDirection: 'north',
      cellSizeMode: 'web-mercator',
      cellSize: [20, 20],
      northEdge: 0.3,
      southEdge: 0.32,
      zFactor: zFactorSign
    });
    expect(actual.filter(angle => angle > 5).length).toBeGreaterThan(30);
    expectCloseToOracle(actual, expected.horizon, 2e-3);
    fixture.destroy();
  }
});

it('sweep kernel reports exact zero on planes and analytic angles on ramps and walls', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 41;
  const height = 21;
  const flat = new Float32Array(width * height).fill(4321.123);
  const flatFixture = createSweepFixture(device, flat, {
    width,
    height,
    directionCount: 8,
    maximumRadius: 40,
    output: 'horizon'
  });
  for (const zFactor of [1, 0.3, 1.37]) {
    const result = await flatFixture.run({cellSize: [10, 10], zFactor});
    expect(result.every(angle => angle === 0)).toBe(true);
  }
  flatFixture.destroy();

  // z = 3.7 * column: 0.37 rise per 10 m cell along the east axis.
  const ramp = Float32Array.from({length: width * height}, (_, index) => 3.7 * (index % width));
  const rampFixture = createSweepFixture(device, ramp, {
    width,
    height,
    directionCount: 8,
    maximumRadius: 8,
    output: 'horizon'
  });
  const rampResult = await rampFixture.run({cellSize: [10, 10]});
  const center = 10 * width + 20;
  const expected = (Math.atan(0.37) * 180) / Math.PI;
  expect(rampResult[center * 8 + 2]).toBeCloseTo(expected, 4); // east
  expect(rampResult[center * 8 + 6]).toBeCloseTo(-expected, 4); // west
  expect(rampResult[center * 8 + 0]).toBeCloseTo(0, 4); // north: constant elevation
  rampFixture.destroy();

  // A wall 20 m high in column 30: the east horizon from column 20 is atan(20 / 100).
  const wall = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    wall[row * width + 30] = 20;
  }
  const wallAngle = (Math.atan(20 / 100) * 180) / Math.PI;
  for (const [maximumRadius, expectedAngle] of [
    [9, 0],
    [10, wallAngle],
    [40, wallAngle]
  ] as const) {
    const wallFixture = createSweepFixture(device, wall, {
      width,
      height,
      directionCount: 8,
      maximumRadius,
      output: 'horizon'
    });
    const wallResult = await wallFixture.run({cellSize: [10, 10]});
    expect(wallResult[(10 * width + 20) * 8 + 2]).toBeCloseTo(expectedAngle, 4);
    // Maximum distance 95 m excludes the wall at 100 m regardless of the pixel radius.
    const limited = await wallFixture.run({cellSize: [10, 10], maximumDistance: 95});
    expect(limited[(10 * width + 20) * 8 + 2]).toBeCloseTo(0, 4);
    wallFixture.destroy();
  }
});

/** Gentle hills (gradients below 0.2): lateral digital-line offsets of half a pixel matter little. */
function createGentleTerrain(width: number, height: number): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return (
      100 +
      30 * Math.sin(column * 0.07) * Math.cos(row * 0.05) +
      20 * Math.sin((column + 2 * row) * 0.04)
    );
  });
}

it('sweep kernel agrees statistically with the ray march', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 64;
  const height = 64;
  const rough = createSmoothTerrain(width, height, 13);
  // createSmoothTerrain has gradients up to ~0.6 and a 4-pixel peak, where nearest-pixel digital
  // lines and bilinear rays legitimately sample different points (up to half a pixel laterally);
  // gentle hills still differ by half-pixel lateral offsets on the 22.5 degree sectors (the axis
  // sectors agree exactly, see the node spec). The nearest-pixel max is also slightly noisier than the bilinear one.
  for (const [label, elevation, bound] of [
    ['peaked', rough, 0.02],
    ['gentle', createGentleTerrain(width, height), 0.012]
  ] as const) {
    for (const maximumRadius of [20, 63]) {
      const fixture = createSweepFixture(device, elevation, {
        width,
        height,
        directionCount: 16,
        maximumRadius,
        output: 'sine'
      });
      const sine = await fixture.run({cellSize: [10, 10]});
      const march = computeTerrainHorizon({
        width,
        height,
        elevation,
        directionCount: 16,
        stepDistances: getGPUTerrainHorizonStepDistances(maximumRadius),
        cellSize: [10, 10]
      });
      let sum = 0;
      let worst = 0;
      for (let pixel = 0; pixel < width * height; pixel++) {
        const difference = Math.abs(1 - sine[pixel] / 16 - march.skyViewFactor[pixel]);
        sum += difference;
        worst = Math.max(worst, difference);
      }
      const mean = sum / (width * height);
      expect(mean, `sweep vs march ${label} radius ${maximumRadius}: mean |dSVF|`).toBeLessThan(
        bound
      );
      // Not all-flat: terrain has real occlusion.
      expect(march.skyViewFactor.some(value => value < 0.97)).toBe(true);
      expect(sine.some(value => value > 1)).toBe(true);
      fixture.destroy();
    }
  }
});

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measure(device: Device, submit: () => void): Promise<number> {
  submit();
  await waitForQueue(device);
  const samples: number[] = [];
  for (let sample = 0; sample < 3; sample++) {
    const start = performance.now();
    submit();
    await waitForQueue(device);
    samples.push(performance.now() - start);
  }
  samples.sort((left, right) => left - right);
  return samples[1];
}

it.skipIf(!RUN_BENCHMARK)('sweep vs march timing on 1024^2', {timeout: 600000}, async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const size = 1024;
  const pixelCount = size * size;
  const elevation = createSmoothTerrain(size, size, 3);
  const cpu = computeTerrainHorizonSweepHull({
    width: 256,
    height: 256,
    elevation: createSmoothTerrain(256, 256, 3),
    directionCount: 16,
    cellSize: [10, 10]
  });
  console.log(
    `hull walk steps / pixel (256^2, 16 sectors, full radius): ${(cpu.statistics.walkSteps / cpu.statistics.pixelCount).toFixed(3)}`
  );
  const windowed = computeTerrainHorizonSweepHull({
    width: 256,
    height: 256,
    elevation: createSmoothTerrain(256, 256, 3),
    directionCount: 16,
    maximumRadius: 64,
    cellSize: [10, 10]
  });
  console.log(
    `radius 64: walk ${(windowed.statistics.walkSteps / windowed.statistics.pixelCount).toFixed(3)}, window queries ${(windowed.statistics.windowQueries / windowed.statistics.pixelCount).toFixed(3)} / px, window steps ${(windowed.statistics.windowSteps / windowed.statistics.pixelCount).toFixed(3)} / px`
  );
  for (const maximumRadius of [256, 1023]) {
    const sweep = createSweepFixture(device, elevation, {
      width: size,
      height: size,
      directionCount: 16,
      maximumRadius,
      output: 'sine'
    });
    const sweepMilliseconds = await measure(device, () => sweep.submit({cellSize: [10, 10]}));
    const sine = await sweep.run({cellSize: [10, 10]});
    sweep.destroy();

    const graph = new GPUCommandGraph(device, {id: 'march'});
    const values = createInputBuffer(device, elevation);
    const svf: Buffer = createOutputBuffer(device, pixelCount);
    const settings = new GPUParameterBuffer(device, {
      id: 'march-settings',
      format: 'float32',
      length: 8,
      values: getGPUTerrainHorizonParameterValues({cellSize: [10, 10]})
    });
    graph.add(
      new GPUTerrainHorizon({
        width: size,
        height: size,
        maximumRadius,
        directionCount: 16,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', values, 'float32', pixelCount)
          }
        },
        settings: settings.importToGraph(graph),
        skyViewFactor: importGraphBuffer(graph, 'svf', svf, 'float32', pixelCount)
      })
    );
    const compiled = graph.compile();
    const marchMilliseconds = await measure(device, () => submitGraph(device, compiled, undefined));
    const march = await readFloat32(svf, pixelCount);
    let sum = 0;
    let worst = 0;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      const difference = Math.abs(1 - sine[pixel] / 16 - march[pixel]);
      sum += difference;
      worst = Math.max(worst, difference);
    }
    console.log(
      `1024^2 D=16 radius ${maximumRadius}: sweep ${sweepMilliseconds.toFixed(1)} ms, march ${marchMilliseconds.toFixed(1)} ms, mean |dSVF| ${(sum / pixelCount).toExponential(2)}, max ${worst.toExponential(2)}`
    );
    compiled.destroy();
    settings.destroy();
    values.destroy();
    svf.destroy();
  }
});
