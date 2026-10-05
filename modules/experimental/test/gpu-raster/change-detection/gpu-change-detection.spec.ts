// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection,
  type GPUChangeDetectionParameters
} from '../../../src/gpu-raster/change-detection';
import {createInputBuffer, createOutputBuffer} from '../../utils/gpu-contributor-test-utils';
import {detectChangeOnCPU, type ChangeDetectionCPUResult} from './change-detection-cpu';

type Scene = {
  slices: Float32Array;
  mask?: Uint32Array;
  cellCount: number;
  sliceCount: number;
  bandCount?: number;
  significanceSource?: 't-test' | 'mann-kendall';
};

type Mode = 'two-slice' | 'welch' | 'trend' | 'multiband';

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

/** Cells with trends, noise, ties (coarse quantization) and missing slices. */
function createRandomScene(
  seed: number,
  cellCount: number,
  sliceCount: number,
  bandCount = 1
): Scene {
  const random = createRandom(seed);
  const slices = new Float32Array(cellCount * sliceCount * bandCount);
  for (let cell = 0; cell < cellCount; cell++) {
    const trend = (random() - 0.5) * 2;
    for (let slice = 0; slice < sliceCount; slice++) {
      for (let band = 0; band < bandCount; band++) {
        const value = Math.round((trend * slice + (random() - 0.5) * 6) * 4) / 4;
        slices[(cell * sliceCount + slice) * bandCount + band] = random() < 0.05 ? NaN : value + 10;
      }
    }
  }
  return {slices, cellCount, sliceCount, bandCount};
}

const OUTPUT_FORMATS = {
  difference: 'float32',
  logRatio: 'float32',
  percentChange: 'float32',
  tStatistic: 'float32',
  tDegreesOfFreedom: 'float32',
  tPValue: 'float32',
  senSlope: 'float32',
  mannKendallS: 'sint32',
  mannKendallZ: 'float32',
  mannKendallP: 'float32',
  changeMagnitude: 'float32',
  changeDirection: 'float32',
  significance: 'uint32'
} as const;

type OutputName = keyof typeof OUTPUT_FORMATS;
type Columns = Record<OutputName, number[]>;

type Fixture = {
  run(parameters: GPUChangeDetectionParameters): Promise<Columns>;
  rebuildCount: number;
  destroy(): void;
};

function createFixture(device: Device, scene: Scene, outputNames: OutputName[]): Fixture {
  const bandCount = scene.bandCount ?? 1;
  const graph = new GPUCommandGraph(device, {id: 'change-detection-graph'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'change-parameters',
    format: 'float32',
    length: 5
  });
  const lengths: Record<string, number> = {};
  const outputBuffers: Record<string, Buffer> = {};
  const output: Record<string, ReturnType<typeof importGraphBuffer>> = {};
  for (const name of outputNames) {
    const isBandRow = name === 'difference' || name === 'logRatio' || name === 'percentChange';
    const length = isBandRow ? scene.cellCount * bandCount : scene.cellCount;
    lengths[name] = length;
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputBuffers[name] = buffer;
    const format = name === 'changeDirection' && bandCount > 2 ? 'uint32' : OUTPUT_FORMATS[name];
    output[name] = importGraphBuffer(graph, `o-${name}`, buffer, format, length);
  }
  graph.add(
    new GPUChangeDetection({
      slices: importGraphBuffer(graph, 'slices', input(scene.slices), 'float32'),
      mask: scene.mask
        ? importGraphBuffer(graph, 'mask', input(scene.mask), 'uint32', scene.cellCount)
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      cellCount: scene.cellCount,
      sliceCount: scene.sliceCount,
      bandCount,
      significanceSource: scene.significanceSource,
      output
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(parameters) {
      parameterBuffer.write(getGPUChangeDetectionParameterValues(parameters));
      submitGraph(device, compiled, undefined);
      const result = {} as Columns;
      for (const name of outputNames) {
        const bytes = await outputBuffers[name].readAsync();
        const length = lengths[name];
        const isSigned = OUTPUT_FORMATS[name] === 'sint32';
        const isUnsigned =
          OUTPUT_FORMATS[name] === 'uint32' || (name === 'changeDirection' && bandCount > 2);
        const ArrayType = isSigned ? Int32Array : isUnsigned ? Uint32Array : Float32Array;
        result[name] = Array.from(
          new ArrayType(bytes.buffer as ArrayBuffer, bytes.byteOffset, length)
        );
      }
      return result;
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

function expectClose(
  actual: number[],
  expected: ArrayLike<number>,
  tolerance: number,
  name: string
) {
  expect(actual.length, name).toBeGreaterThanOrEqual(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const want = expected[index];
    const got = actual[index];
    if (Number.isNaN(want)) {
      expect(got, `${name}[${index}]`).toBeNaN();
    } else {
      const bound = tolerance * Math.max(1, Math.abs(want));
      expect(Math.abs(got - want), `${name}[${index}] ${got} vs ${want}`).toBeLessThanOrEqual(
        bound
      );
    }
  }
}

function expectParity(
  actual: Columns,
  expected: ChangeDetectionCPUResult,
  names: OutputName[]
): void {
  for (const name of names) {
    const want = expected[name] as ArrayLike<number>;
    if (name === 'mannKendallS' || name === 'significance') {
      expect(actual[name], name).toEqual(Array.from(want));
    } else if (name === 'changeDirection' && !Number.isNaN(want[0]) && want[0] > 4) {
      expect(actual[name], name).toEqual(Array.from(want));
    } else if (name === 'tPValue') {
      expectClose(actual[name], want, 2e-3, name);
    } else if (name === 'tDegreesOfFreedom' || name === 'tStatistic') {
      expectClose(actual[name], want, 5e-4, name);
    } else {
      expectClose(actual[name], want, 2e-4, name);
    }
  }
}

const MODES: Record<Mode, OutputName[]> = {
  'two-slice': ['difference', 'logRatio', 'percentChange'],
  welch: ['tStatistic', 'tDegreesOfFreedom', 'tPValue', 'significance'],
  trend: ['senSlope', 'mannKendallS', 'mannKendallZ', 'mannKendallP'],
  multiband: ['changeMagnitude', 'changeDirection']
};

it('GPUChangeDetection two-slice outputs match the CPU oracle without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(1, 300, 8);
  scene.slices[3] = 0; // exact zero before-value for percentChange
  const fixture = createFixture(device, scene, MODES['two-slice']);
  for (const parameters of [
    {beforeSlice: 0, afterSlice: 7, epsilon: 0.5},
    {beforeSlice: 3, afterSlice: 2, epsilon: 1e-3},
    {beforeSlice: 7, afterSlice: 0, epsilon: 20},
    {beforeSlice: -1, afterSlice: 2}
  ]) {
    const actual = await fixture.run(parameters);
    expectParity(actual, detectChangeOnCPU({...scene, ...parameters}), MODES['two-slice']);
  }
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUChangeDetection Welch t-test and significance match the CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(2, 400, 12);
  scene.mask = Uint32Array.from({length: 400}, (_, cell) => (cell % 7 === 3 ? 0 : 1));
  const fixture = createFixture(device, scene, MODES.welch);
  for (const parameters of [
    {beforeSlice: 0, afterSlice: 1, splitSlice: 6, alpha: 0.05},
    {beforeSlice: 0, afterSlice: 1, splitSlice: 3, alpha: 0.2},
    {beforeSlice: 0, afterSlice: 1, splitSlice: 10, alpha: 0.01},
    {beforeSlice: 0, afterSlice: 1, splitSlice: 0}
  ]) {
    const actual = await fixture.run(parameters);
    const expected = detectChangeOnCPU({...scene, ...parameters});
    expectParity(actual, expected, ['tStatistic', 'tDegreesOfFreedom', 'tPValue']);
    // The class is exact unless a p-value sits within float32 error of alpha.
    const alpha = parameters.alpha ?? 0.05;
    let compared = 0;
    for (let cell = 0; cell < scene.cellCount; cell++) {
      const p = expected.tPValue[cell];
      if (Number.isNaN(p) || Math.abs(p - alpha) < 5e-3 * alpha) {
        continue;
      }
      expect(actual.significance[cell], `cell ${cell}`).toBe(expected.significance[cell]);
      compared++;
    }
    if (parameters.splitSlice === 0) {
      expect(compared).toBe(0);
    } else {
      expect(compared).toBeGreaterThan(100);
    }
  }
  // The mask suppresses cells.
  const masked = await fixture.run({
    beforeSlice: 0,
    afterSlice: 1,
    splitSlice: 6
  });
  expect(masked.tStatistic[3]).toBeNaN();
  expect(masked.significance[3]).toBe(0);
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUChangeDetection Sen slope and Mann-Kendall match the CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [seed, sliceCount] of [
    [3, 9],
    [4, 16],
    [5, 64]
  ]) {
    const scene = createRandomScene(seed, 200, sliceCount);
    const fixture = createFixture(device, scene, MODES.trend);
    const expected = detectChangeOnCPU({
      ...scene,
      beforeSlice: 0,
      afterSlice: 1
    });
    const actual = await fixture.run({beforeSlice: 0, afterSlice: 1});
    // S is integer exact, including tie handling from the quantized data.
    expect(actual.mannKendallS).toEqual(Array.from(expected.mannKendallS));
    expectParity(actual, expected, ['senSlope', 'mannKendallZ', 'mannKendallP']);
    fixture.destroy();
  }
});

it('GPUChangeDetection reproduces hand-checked fixtures', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Cell 0: [0, 1, 2, 3, 100] -> Sen 1. Cell 1: [1, 2, 2, 3, 3] -> S 8.
  const scene: Scene = {
    slices: Float32Array.from([0, 1, 2, 3, 100, 1, 2, 2, 3, 3]),
    cellCount: 2,
    sliceCount: 5
  };
  const fixture = createFixture(device, scene, MODES.trend);
  const actual = await fixture.run({beforeSlice: 0, afterSlice: 1});
  expect(actual.senSlope[0]).toBe(1);
  // [1, 2, 2, 3, 3]: S = 4 + 2 + 2 + 0 = 8 and two tie pairs: 2 * (2 * 1 * 9) = 36,
  // var = (5 * 4 * 15 - 36) / 18, Z = 7 / sqrt(var).
  expect(actual.mannKendallS[1]).toBe(8);
  expect(actual.mannKendallZ[1]).toBeCloseTo(7 / Math.sqrt(264 / 18), 4);
  fixture.destroy();

  // Welch fixture with known t = 3 / sqrt(2 / 3), df 4, p = 0.0213.
  const welch = createFixture(
    device,
    {
      slices: Float32Array.from([1, 2, 3, 4, 5, 6]),
      cellCount: 1,
      sliceCount: 6
    },
    MODES.welch
  );
  const result = await welch.run({
    beforeSlice: 0,
    afterSlice: 1,
    splitSlice: 3
  });
  expect(result.tStatistic[0]).toBeCloseTo(3 / Math.sqrt(2 / 3), 4);
  expect(result.tDegreesOfFreedom[0]).toBeCloseTo(4, 4);
  expect(result.tPValue[0]).toBeCloseTo(0.0213, 3);
  expect(result.significance[0]).toBe(1);
  welch.destroy();
});

it('GPUChangeDetection derives significance from Mann-Kendall when requested', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(6, 300, 14);
  scene.significanceSource = 'mann-kendall';
  const names: OutputName[] = ['mannKendallP', 'significance'];
  const fixture = createFixture(device, scene, names);
  const parameters = {beforeSlice: 0, afterSlice: 1, alpha: 0.1};
  const actual = await fixture.run(parameters);
  const expected = detectChangeOnCPU({...scene, ...parameters});
  let compared = 0;
  for (let cell = 0; cell < scene.cellCount; cell++) {
    const p = expected.mannKendallP[cell];
    if (Number.isNaN(p) || Math.abs(p - 0.1) < 1e-3) {
      continue;
    }
    expect(actual.significance[cell]).toBe(expected.significance[cell]);
    compared++;
  }
  expect(compared).toBeGreaterThan(100);
  expect(new Set(expected.significance).size).toBe(3);
  fixture.destroy();
});

it('GPUChangeDetection multiband magnitude and direction match the CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const bandCount of [2, 3, 5]) {
    const scene = createRandomScene(7 + bandCount, 250, 4, bandCount);
    scene.mask = Uint32Array.from({length: 250}, (_, cell) => (cell % 11 === 0 ? 0 : 1));
    const fixture = createFixture(device, scene, [...MODES.multiband, 'difference']);
    for (const parameters of [
      {beforeSlice: 0, afterSlice: 3},
      {beforeSlice: 2, afterSlice: 1}
    ]) {
      const actual = await fixture.run(parameters);
      expectParity(actual, detectChangeOnCPU({...scene, ...parameters}), [
        ...MODES.multiband,
        'difference'
      ]);
    }
    expect(fixture.rebuildCount).toBe(0);
    fixture.destroy();
  }
});
