// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUFocalStatisticsParameterValues,
  GPUFocalStatistics,
  type GPUFocalStatisticsSettings
} from '../../../src/geospatial/spatial-interpolation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeFocalStatisticsOnCPU,
  type FocalStatisticsOracleResult
} from './spatial-interpolation-oracle';

/**
 * Tolerances against the oracle. `count`, `min`, `max`, `range`, and `sum` (f32 sums in the same
 * window order) must be bit-exact. `mean` allows 3 f32 ULP (WGSL division is within 2.5 ULP); `standardDeviation`
 * allows `1e-5 * valueScale` for the f32 centered second pass.
 */
const STANDARD_DEVIATION_TOLERANCE = 1e-5;

type Scene = {
  values: Float32Array;
  validity?: Uint32Array;
  noDataValue?: number;
  width: number;
  height: number;
  maximumRadius: number;
};

const FLOAT_STATISTICS = ['mean', 'sum', 'min', 'max', 'range', 'standardDeviation'] as const;

type Fixture = {
  run(settings: GPUFocalStatisticsSettings): Promise<FocalStatisticsOracleResult>;
  rebuildCount: number;
  destroy(): void;
};

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

function createFixture(device: Device, scene: Scene): Fixture {
  const cellCount = scene.width * scene.height;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'focal-parameters',
    format: 'float32',
    length: 4
  });
  const outputBuffers = Object.fromEntries(
    [...FLOAT_STATISTICS, 'count'].map(name => [name, track(createOutputBuffer(device, cellCount))])
  ) as Record<(typeof FLOAT_STATISTICS)[number] | 'count', Buffer>;
  const graph = new GPUCommandGraph(device, {id: 'focal-graph'});
  const float = (name: (typeof FLOAT_STATISTICS)[number]) =>
    importGraphBuffer(graph, `out-${name}`, outputBuffers[name], 'float32', cellCount);
  graph.add(
    new GPUFocalStatistics({
      id: 'focal',
      values: importGraphBuffer(
        graph,
        'raster',
        track(createInputBuffer(device, scene.values)),
        'float32',
        cellCount
      ),
      validity: scene.validity
        ? importGraphBuffer(
            graph,
            'validity',
            track(createInputBuffer(device, scene.validity)),
            'uint32',
            cellCount
          )
        : undefined,
      noDataValue: scene.noDataValue,
      width: scene.width,
      height: scene.height,
      maximumRadius: scene.maximumRadius,
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        mean: float('mean'),
        sum: float('sum'),
        min: float('min'),
        max: float('max'),
        range: float('range'),
        standardDeviation: float('standardDeviation'),
        count: importGraphBuffer(graph, 'out-count', outputBuffers.count, 'uint32', cellCount)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      parameterBuffer.write(getGPUFocalStatisticsParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const result: Partial<FocalStatisticsOracleResult> = {
        count: Uint32Array.from(await readUint32(outputBuffers.count, cellCount))
      };
      for (const name of FLOAT_STATISTICS) {
        result[name] = Float32Array.from(await readFloat32(outputBuffers[name], cellCount));
      }
      return result as FocalStatisticsOracleResult;
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function getUlpDistance(left: number, right: number): number {
  const bits = new Int32Array(new Float32Array([left, right]).buffer);
  return Math.abs(bits[0] - bits[1]);
}

function expectParity(
  actual: FocalStatisticsOracleResult,
  scene: Scene,
  settings: GPUFocalStatisticsSettings
): FocalStatisticsOracleResult {
  const expected = computeFocalStatisticsOnCPU({
    ...scene,
    radius: settings.radius,
    shape: settings.shape ?? 'square',
    minimumCount: settings.minimumCount ?? 1,
    propagateCenterNoData: settings.propagateCenterNoData ?? false
  });
  expect(Array.from(actual.count)).toEqual(Array.from(expected.count));
  let valueScale = 1;
  for (const value of scene.values) {
    if (Number.isFinite(value)) {
      valueScale = Math.max(valueScale, Math.abs(value));
    }
  }
  for (const name of FLOAT_STATISTICS) {
    for (let cell = 0; cell < expected.count.length; cell++) {
      const gpu = actual[name][cell];
      const cpu = expected[name][cell];
      if (Number.isNaN(cpu) || Number.isNaN(gpu)) {
        expect(Number.isNaN(gpu), `${name} cell ${cell} nodata`).toBe(Number.isNaN(cpu));
      } else if (name === 'mean') {
        expect(getUlpDistance(gpu, cpu), `mean cell ${cell}`).toBeLessThanOrEqual(3);
      } else if (name === 'standardDeviation') {
        expect(Math.abs(gpu - cpu), `std cell ${cell}`).toBeLessThanOrEqual(
          STANDARD_DEVIATION_TOLERANCE * valueScale
        );
      } else {
        expect(gpu, `${name} cell ${cell}`).toBe(cpu);
      }
    }
  }
  return expected;
}

function createRandomScene(seed: number, width: number, height: number): Scene {
  const random = createRandom(seed);
  const values = new Float32Array(width * height);
  const validity = new Uint32Array(width * height);
  for (let cell = 0; cell < values.length; cell++) {
    const roll = random();
    // Values with a large common offset stress the centered standard deviation.
    values[cell] = roll < 0.04 ? NaN : roll < 0.07 ? -9999 : 1000 + (random() - 0.5) * 50;
    validity[cell] = random() < 0.05 ? 0 : 1;
  }
  return {
    values,
    validity,
    noDataValue: -9999,
    width,
    height,
    maximumRadius: 6
  };
}

it('GPUFocalStatistics matches the CPU oracle as radius, shape, and policy change without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(1, 37, 23);
  const fixture = createFixture(device, scene);
  const frames: GPUFocalStatisticsSettings[] = [
    {radius: 1},
    {radius: 0},
    {radius: 2.7, shape: 'circle'},
    {radius: 3},
    {radius: 3, shape: 'circle', minimumCount: 20},
    {radius: 1, propagateCenterNoData: true},
    // Above maximumRadius (6): the square clamps to 6.
    {radius: 10},
    {radius: 6.5, shape: 'circle'},
    // Negative and NaN radii give empty windows (all nodata, count 0).
    {radius: -1},
    {radius: NaN},
    {radius: 1}
  ];
  for (const settings of frames) {
    const actual = await fixture.run(settings);
    expectParity(actual, scene, settings);
  }
  const empty = await fixture.run({radius: -1});
  expect(Array.from(empty.count).every(count => count === 0)).toBe(true);
  expect(Array.from(empty.mean).every(Number.isNaN)).toBe(true);
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUFocalStatistics clips windows at raster edges and skips nodata', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: Scene = {
    values: Float32Array.from([1, 2, 3, 4, NaN, 6, 7, 8, -9]),
    noDataValue: -9,
    width: 3,
    height: 3,
    maximumRadius: 2
  };
  const fixture = createFixture(device, scene);
  let actual = await fixture.run({radius: 1});
  expectParity(actual, scene, {radius: 1});
  // Corner window is 2 x 2: values 1, 2, 4 (NaN skipped).
  expect([actual.count[0], actual.sum[0], actual.min[0], actual.max[0], actual.range[0]]).toEqual([
    3, 7, 1, 4, 3
  ]);
  expect(actual.count[4]).toBe(7);
  // A NaN center still gets statistics unless the center policy is set.
  expect(actual.mean[4]).toBeCloseTo(31 / 7, 5);
  actual = await fixture.run({radius: 1, propagateCenterNoData: true});
  expect(Number.isNaN(actual.mean[4])).toBe(true);
  expect(Number.isNaN(actual.mean[8])).toBe(true);
  expect(actual.count[4]).toBe(7);
  // Radius 0 reproduces the raster, with nodata cells as NaN.
  actual = await fixture.run({radius: 0});
  expect(Array.from(actual.min)).toEqual([1, 2, 3, 4, NaN, 6, 7, 8, NaN]);
  expect(Array.from(actual.standardDeviation).filter(value => !Number.isNaN(value))).toEqual(
    Array(7).fill(0)
  );
  fixture.destroy();
});
