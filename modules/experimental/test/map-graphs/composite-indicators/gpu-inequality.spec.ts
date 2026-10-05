// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {GPUInequality} from '../../../src/map-graphs/composite-indicators/gpu-inequality';
import {
  getGPUInequalityParameterValues,
  GPU_INEQUALITY_GLOBAL_SUMMARY,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  GPU_INEQUALITY_PARAMETER_LENGTH,
  type GPUInequalitySettings
} from '../../../src/map-graphs/composite-indicators/inequality-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {computeInequalityOnCPU, type InequalityOracleResult} from './inequality-oracle';

type Scene = {
  values: Float32Array;
  zoneIds: Uint32Array;
  zoneCount: number;
  weights?: Float32Array;
  mask?: Uint32Array;
};

type GPUResult = Omit<InequalityOracleResult, 'count'> & {count: number[]};

type Fixture = {
  run(settings: GPUInequalitySettings): Promise<GPUResult>;
  readonly rebuildCount: number;
  destroy(): void;
};

const KNOT_COUNT = 11;

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

/**
 * Six populated zones of different spread (zone 3 has zeros, zone 4 is perfectly equal, zone 5 is
 * a single holder), an empty zone 6, plus negative, NaN, masked and unassigned rows.
 */
function createScene(rows: number, seed: number, withWeights: boolean): Scene {
  const random = createRandom(seed);
  const values = new Float32Array(rows);
  const zoneIds = new Uint32Array(rows);
  const mask = new Uint32Array(rows).fill(1);
  const weights = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const zone = row % 6;
    zoneIds[row] = zone;
    const spread = 0.3 + zone * 0.4;
    values[row] = Math.exp((random() - 0.5) * 2 * spread) * 1000;
    if (zone === 3 && row % 12 === 3) {
      values[row] = 0;
    }
    if (zone === 4) {
      values[row] = 2500;
    }
    if (zone === 5) {
      values[row] = row === 5 ? 80000 : 0;
    }
    weights[row] = 0.5 + Math.floor(random() * 8) / 2;
    if (row % 53 === 7) {
      values[row] = -3;
    }
    if (row % 61 === 11) {
      values[row] = NaN;
    }
    if (row % 71 === 13) {
      zoneIds[row] = 0xffffffff;
    }
    if (row % 83 === 17) {
      zoneIds[row] = 99;
    }
    if (row % 47 === 19) {
      mask[row] = 0;
    }
  }
  return {values, zoneIds, zoneCount: 7, mask, weights: withWeights ? weights : undefined};
}

function createFixture(device: Device, scene: Scene): Fixture {
  const {zoneCount} = scene;
  const rows = scene.values.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'inequality-parameters',
    format: 'float32',
    length: GPU_INEQUALITY_PARAMETER_LENGTH
  });
  const lengths = {
    gini: zoneCount,
    theilT: zoneCount,
    theilL: zoneCount,
    atkinson: zoneCount,
    hoover: zoneCount,
    palma: zoneCount,
    mean: zoneCount,
    count: zoneCount,
    lorenzKnots: zoneCount * KNOT_COUNT,
    globalSummary: GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH
  };
  const outputs = Object.fromEntries(
    Object.entries(lengths).map(([name, length]) => [
      name,
      track(createOutputBuffer(device, length))
    ])
  ) as Record<keyof typeof lengths, Buffer>;
  const graph = new GPUCommandGraph(device, {id: 'inequality-graph'});
  const float = (name: keyof typeof lengths) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', lengths[name]);
  graph.add(
    new GPUInequality({
      id: 'inequality',
      values: importGraphBuffer(
        graph,
        'values',
        track(createInputBuffer(device, scene.values)),
        'float32',
        rows
      ),
      zoneIds: importGraphBuffer(
        graph,
        'zones',
        track(createInputBuffer(device, scene.zoneIds)),
        'uint32',
        rows
      ),
      zoneCount,
      weights: scene.weights
        ? importGraphBuffer(
            graph,
            'weights',
            track(createInputBuffer(device, scene.weights)),
            'float32',
            rows
          )
        : undefined,
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            rows
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      lorenzKnotCount: KNOT_COUNT,
      output: {
        gini: float('gini'),
        theilT: float('theilT'),
        theilL: float('theilL'),
        atkinson: float('atkinson'),
        hoover: float('hoover'),
        palma: float('palma'),
        mean: float('mean'),
        count: importGraphBuffer(graph, 'out-count', outputs.count, 'uint32', zoneCount),
        lorenzKnots: float('lorenzKnots'),
        globalSummary: float('globalSummary')
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      parameterBuffer.write(getGPUInequalityParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        gini: await readFloat32(outputs.gini, lengths.gini),
        theilT: await readFloat32(outputs.theilT, lengths.theilT),
        theilL: await readFloat32(outputs.theilL, lengths.theilL),
        atkinson: await readFloat32(outputs.atkinson, lengths.atkinson),
        hoover: await readFloat32(outputs.hoover, lengths.hoover),
        palma: await readFloat32(outputs.palma, lengths.palma),
        mean: await readFloat32(outputs.mean, lengths.mean),
        count: await readUint32(outputs.count, lengths.count),
        lorenzKnots: await readFloat32(outputs.lorenzKnots, lengths.lorenzKnots),
        globalSummary: await readFloat32(outputs.globalSummary, lengths.globalSummary)
      };
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
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const want = expected[index];
    const got = actual[index];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label}[${index}] should be NaN, got ${got}`).toBe(true);
    } else {
      const scale = Math.max(1, Math.abs(want));
      expect(Math.abs(got - want), `${label}[${index}] ${got} vs ${want}`).toBeLessThanOrEqual(
        tolerance * scale
      );
    }
  }
}

function compare(result: GPUResult, oracle: InequalityOracleResult, label: string): void {
  expect(result.count, `${label} count`).toEqual(oracle.count);
  expectClose(result.mean, oracle.mean, 1e-4, `${label} mean`);
  expectClose(result.gini, oracle.gini, 2e-4, `${label} gini`);
  expectClose(result.theilT, oracle.theilT, 5e-4, `${label} theilT`);
  expectClose(result.theilL, oracle.theilL, 5e-4, `${label} theilL`);
  expectClose(result.atkinson, oracle.atkinson, 5e-4, `${label} atkinson`);
  expectClose(result.hoover, oracle.hoover, 2e-4, `${label} hoover`);
  expectClose(result.palma, oracle.palma, 5e-4, `${label} palma`);
  expectClose(result.lorenzKnots, oracle.lorenzKnots, 2e-4, `${label} lorenz`);
  expectClose(result.globalSummary, oracle.globalSummary, 1e-3, `${label} summary`);
}

const SETTINGS: GPUInequalitySettings[] = [
  {epsilon: 1},
  {epsilon: 0.5},
  {epsilon: 2, palmaTopShare: 0.2, palmaBottomShare: 0.5},
  {epsilon: 0, palmaTopShare: 0.05, palmaBottomShare: 0.3}
];

for (const withWeights of [false, true]) {
  it(`GPUInequality matches the CPU oracle ${withWeights ? 'with' : 'without'} weights and never rebuilds`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createScene(2400, 5, withWeights);
    const fixture = createFixture(device, scene);
    try {
      for (const settings of SETTINGS) {
        const result = await fixture.run(settings);
        const oracle = computeInequalityOnCPU({...scene, settings, lorenzKnotCount: KNOT_COUNT});
        compare(result, oracle, `epsilon ${settings.epsilon}`);
        // Zero-value zone: Theil L and epsilon >= 1 are NaN, empty zone is NaN, equal zone is 0.
        if (settings.epsilon !== 0.5 && settings.epsilon !== 0) {
          expect(result.atkinson[3]).toBeNaN();
        }
        expect(result.theilL[3]).toBeNaN();
        expect(result.gini[6]).toBeNaN();
        expect(result.count[6]).toBe(0);
        expect(Math.abs(result.gini[4])).toBeLessThan(1e-5);
        expect(Math.abs(result.theilT[4])).toBeLessThan(1e-5);
        expect(Math.abs(result.hoover[4])).toBeLessThan(1e-5);
        // Decomposition identity on the GPU result itself.
        const summary = result.globalSummary;
        expect(
          Math.abs(
            summary[GPU_INEQUALITY_GLOBAL_SUMMARY.TOTAL_THEIL_T] -
              summary[GPU_INEQUALITY_GLOBAL_SUMMARY.BETWEEN_THEIL_T] -
              summary[GPU_INEQUALITY_GLOBAL_SUMMARY.WITHIN_THEIL_T]
          )
        ).toBeLessThan(2e-3);
      }
      expect(fixture.rebuildCount).toBe(0);
    } finally {
      fixture.destroy();
    }
  });
}

it('GPUInequality gives the known answers for equality, one holder and a textbook zone', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Zone 0 equal, zone 1 one holder of 4, zone 2 holds 1, 2, 3, 4; zones 3 to 6 are empty.
  const scene: Scene = {
    values: Float32Array.from([5, 5, 5, 5, 0, 0, 0, 12, 1, 2, 3, 4]),
    zoneIds: Uint32Array.from([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    zoneCount: 7
  };
  const fixture = createFixture(device, scene);
  try {
    const result = await fixture.run({epsilon: 1});
    expect(result.count.slice(0, 3)).toEqual([4, 4, 4]);
    for (const index of [
      result.gini,
      result.theilT,
      result.theilL,
      result.atkinson,
      result.hoover
    ]) {
      expect(Math.abs(index[0])).toBeLessThan(1e-5);
    }
    expect(result.palma[0]).toBeCloseTo(0.25, 5);
    expect(result.gini[1]).toBeCloseTo(0.75, 5);
    expect(result.theilT[1]).toBeCloseTo(Math.log(4), 5);
    expect(result.hoover[1]).toBeCloseTo(0.75, 5);
    expect(result.theilL[1]).toBeNaN();
    expect(result.gini[2]).toBeCloseTo(0.25, 5);
    expect(result.mean[2]).toBeCloseTo(2.5, 5);
    expect(result.lorenzKnots[2 * KNOT_COUNT + 5]).toBeCloseTo(0.3, 5);
    expect(result.lorenzKnots[2 * KNOT_COUNT]).toBe(0);
    expect(result.lorenzKnots[2 * KNOT_COUNT + 10]).toBeCloseTo(1, 6);
    expect(result.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.COUNT]).toBe(12);
  } finally {
    fixture.destroy();
  }
});

it('GPUInequality is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, createScene(1500, 9, true));
  try {
    const first = await fixture.run({epsilon: 2});
    const second = await fixture.run({epsilon: 2});
    for (const name of Object.keys(first) as (keyof GPUResult)[]) {
      const left = Float32Array.from(first[name]);
      const right = Float32Array.from(second[name]);
      expect(Array.from(new Uint32Array(left.buffer)), name).toEqual(
        Array.from(new Uint32Array(right.buffer))
      );
    }
  } finally {
    fixture.destroy();
  }
});

it('GPUInequality reports NaN when no row is included', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: Scene = {
    values: Float32Array.from([1, 2]),
    zoneIds: Uint32Array.from([0, 1]),
    zoneCount: 2,
    mask: Uint32Array.from([0, 0])
  };
  const fixture = createFixture(device, scene);
  try {
    const result = await fixture.run({});
    expect(result.count).toEqual([0, 0]);
    expect(result.gini.every(Number.isNaN)).toBe(true);
    expect(result.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.TOTAL_THEIL_T]).toBeNaN();
    expect(result.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.GINI]).toBeNaN();
    expect(result.globalSummary[GPU_INEQUALITY_GLOBAL_SUMMARY.COUNT]).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('GPUInequality stays accurate for one large zone and a large radix sort', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(21);
  const rows = 30000;
  const values = new Float32Array(rows);
  const zoneIds = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    values[row] = Math.exp((random() - 0.3) * 4) * 500;
    zoneIds[row] = random() < 0.9 ? 0 : 1;
  }
  const scene: Scene = {values, zoneIds, zoneCount: 2};
  const fixture = createFixture(device, scene);
  try {
    const result = await fixture.run({epsilon: 1});
    const oracle = computeInequalityOnCPU({
      ...scene,
      settings: {epsilon: 1},
      lorenzKnotCount: KNOT_COUNT
    });
    expect(result.count).toEqual(oracle.count);
    expectClose(result.gini, oracle.gini, 1e-3, 'gini');
    expectClose(result.theilT, oracle.theilT, 2e-3, 'theilT');
    expectClose(result.lorenzKnots, oracle.lorenzKnots, 1e-3, 'lorenz');
    expectClose(result.globalSummary, oracle.globalSummary, 2e-3, 'summary');
  } finally {
    fixture.destroy();
  }
});
