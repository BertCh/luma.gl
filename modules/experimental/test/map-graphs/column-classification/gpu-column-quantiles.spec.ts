// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {getColumnQuantileNodes} from '../../../src/map-graphs/column-classification/column-quantiles-nodes';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues,
  type GPUColumnQuantileInterpolation
} from '../../../src/map-graphs/column-classification/column-quantiles-parameters';
import {GPUColumnQuantiles} from '../../../src/map-graphs/column-classification/gpu-column-quantiles';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeColumnQuantilesOracle,
  getFloat32BitsCanonical,
  getUlpDistance,
  type ColumnQuantilesOracleResult
} from './column-quantiles-oracle';

const INTERPOLATIONS: GPUColumnQuantileInterpolation[] = [
  'lower',
  'higher',
  'nearest',
  'linear',
  'midpoint'
];

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

type ColumnKind = 'wide' | 'ties' | 'special' | 'heavy-duplicates' | 'all-equal' | 'negative';

function createColumn(kind: ColumnKind, seed: number, rows: number): Float32Array {
  const random = createRandom(seed);
  const values = new Float32Array(rows);
  const specials = [NaN, Infinity, -Infinity, 0, -0, 1e-40, -1e-40, 3.4e38, -3.4e38];
  for (let row = 0; row < rows; row++) {
    switch (kind) {
      case 'wide':
        values[row] = (random() - 0.4) * 10 ** Math.floor(random() * 8 - 3);
        break;
      case 'ties':
        values[row] = Math.round((random() - 0.5) * 80) / 4;
        break;
      case 'special':
        values[row] =
          random() < 0.3 ? specials[Math.floor(random() * specials.length)] : random() * 20 - 10;
        break;
      case 'heavy-duplicates':
        values[row] = [1, 2, 3, 5, 8][Math.floor(random() ** 3 * 5)];
        break;
      case 'all-equal':
        values[row] = 42.5;
        break;
      case 'negative':
        values[row] = -random() * 1000;
        break;
    }
  }
  return values;
}

function createMask(seed: number, rows: number, keepProbability: number): Uint32Array {
  const random = createRandom(seed);
  return Uint32Array.from({length: rows}, () => (random() < keepProbability ? 1 : 0));
}

type Frame = {
  quantiles: number[];
  interpolation?: GPUColumnQuantileInterpolation;
  filterRange?: [number, number];
};

type Result = {
  quantiles: Float32Array;
  validCount: number;
  filterMask: Uint32Array;
  filterBounds: Float32Array;
};

type Fixture = {
  readonly valuesBuffer: Buffer;
  readonly counters: {getCommandNodes: number; compile: number};
  encode(frame: Frame): void;
  read(): Promise<Result>;
  /** Resolves when all submitted frames finished (reads the one-row validCount). */
  waitForGPU(): Promise<void>;
  run(frame: Frame): Promise<Result>;
  destroy(): void;
};

function createFixture(
  device: Device,
  scene: {
    values: Float32Array;
    mask?: Uint32Array;
    quantileCount: number;
    withFilter?: boolean;
  }
): Fixture {
  const rows = scene.values.length;
  const {quantileCount} = scene;
  const withFilter = scene.withFilter ?? true;
  const graph = new GPUCommandGraph(device, {id: 'column-quantiles-graph'});
  const counters = {getCommandNodes: 0, compile: 0};
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const valuesBuffer = track(createInputBuffer(device, scene.values));
  const maskBuffer = scene.mask ? track(createInputBuffer(device, scene.mask)) : undefined;
  const outputs = {
    quantiles: track(createOutputBuffer(device, quantileCount)),
    validCount: track(createOutputBuffer(device, 1)),
    filterMask: track(createOutputBuffer(device, rows)),
    filterBounds: track(createOutputBuffer(device, 2))
  };
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'quantile-parameters',
    format: 'float32',
    length: getGPUColumnQuantilesParameterLength(quantileCount)
  });
  const recipe = new GPUColumnQuantiles({
    id: 'quantiles',
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
    mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    quantileCount,
    output: {
      quantiles: importGraphBuffer(
        graph,
        'o-quantiles',
        outputs.quantiles,
        'float32',
        quantileCount
      ),
      validCount: importGraphBuffer(graph, 'o-valid', outputs.validCount, 'uint32', 1),
      ...(withFilter
        ? {
            filterMask: importGraphBuffer(graph, 'o-mask', outputs.filterMask, 'uint32', rows),
            filterBounds: importGraphBuffer(graph, 'o-bounds', outputs.filterBounds, 'float32', 2)
          }
        : {})
    }
  });
  const originalGetCommandNodes = recipe.getCommandNodes.bind(recipe);
  recipe.getCommandNodes = graphArgument => {
    counters.getCommandNodes++;
    return originalGetCommandNodes(graphArgument);
  };
  graph.add(recipe);
  counters.compile++;
  const compiled = graph.compile();
  const encode = (frame: Frame) => {
    parameterBuffer.write(getGPUColumnQuantilesParameterValues({...frame, quantileCount}));
    submitGraph(device, compiled, undefined);
  };
  const read = async (): Promise<Result> => ({
    quantiles: Float32Array.from(await readFloat32(outputs.quantiles, quantileCount)),
    validCount: (await readUint32(outputs.validCount, 1))[0],
    filterMask: Uint32Array.from(await readUint32(outputs.filterMask, rows)),
    filterBounds: Float32Array.from(await readFloat32(outputs.filterBounds, 2))
  });
  return {
    valuesBuffer,
    counters,
    encode,
    read,
    async waitForGPU() {
      await readUint32(outputs.validCount, 1);
    },
    async run(frame) {
      encode(frame);
      return read();
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

/** Bitwise (NaN-canonical) equality of float arrays. */
function toBits(values: ArrayLike<number>): number[] {
  return Array.from(values, getFloat32BitsCanonical);
}

function expectParity(
  actual: Result,
  expected: ColumnQuantilesOracleResult,
  interpolation: GPUColumnQuantileInterpolation,
  label: string,
  checkFilter = true
) {
  expect(actual.validCount, `${label} validCount`).toBe(expected.validCount);
  const isExact =
    interpolation === 'lower' || interpolation === 'higher' || interpolation === 'nearest';
  const maximumUlps = isExact ? 0 : 1;
  actual.quantiles.forEach((value, index) => {
    const distance = getUlpDistance(value, expected.quantiles[index]);
    expect(
      distance,
      `${label} ${interpolation} q${index}: got ${value}, expected ${expected.quantiles[index]}`
    ).toBeLessThanOrEqual(maximumUlps);
  });
  if (checkFilter) {
    expect(toBits(actual.filterBounds), `${label} bounds`).toEqual(toBits(expected.filterBounds));
    expect(Array.from(actual.filterMask), `${label} mask`).toEqual(Array.from(expected.filterMask));
  }
}

const PROBABILITIES = [0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1];

it('GPUColumnQuantiles matches the sort oracle on random columns and every interpolation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const kinds: ColumnKind[] = ['wide', 'ties', 'special', 'heavy-duplicates', 'negative'];
  for (const [kindIndex, kind] of kinds.entries()) {
    for (const rows of [3001, 20000]) {
      const values = createColumn(kind, 11 + kindIndex, rows);
      const mask = createMask(5 + kindIndex, rows, 0.8);
      for (const useMask of [false, true]) {
        const fixture = createFixture(device, {
          values,
          mask: useMask ? mask : undefined,
          quantileCount: PROBABILITIES.length
        });
        for (const interpolation of INTERPOLATIONS) {
          const frame: Frame = {
            quantiles: PROBABILITIES,
            interpolation,
            filterRange: [0.1, 0.85]
          };
          const expected = computeColumnQuantilesOracle({
            values,
            mask: useMask ? mask : undefined,
            ...frame
          });
          expectParity(
            await fixture.run(frame),
            expected,
            interpolation,
            `${kind}/${rows}/${useMask ? 'mask' : 'nomask'}`
          );
        }
        fixture.destroy();
      }
    }
  }
}, 120000);

it('GPUColumnQuantiles handles degenerate columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cases: {name: string; values: Float32Array; mask?: Uint32Array}[] = [
    {name: 'all-equal', values: createColumn('all-equal', 1, 5000)},
    {name: 'n=1', values: Float32Array.from([-3.5])},
    {name: 'n=2', values: Float32Array.from([7, -7])},
    {
      name: 'n=1 among masked',
      values: Float32Array.from([1, 2, 3]),
      mask: Uint32Array.from([0, 1, 0])
    },
    {
      name: 'n=0 full mask',
      values: createColumn('wide', 3, 100),
      mask: new Uint32Array(100)
    },
    {name: 'n=0 all NaN', values: new Float32Array(10).fill(NaN)},
    {
      name: 'signed zeros',
      values: Float32Array.from([0, -0, 0, -0, 0, -0, 1, -1])
    },
    {
      name: 'infinities',
      values: Float32Array.from([Infinity, -Infinity, Infinity, 5, -Infinity, 0])
    },
    {
      name: 'only infinities',
      values: Float32Array.from([Infinity, Infinity, -Infinity])
    }
  ];
  for (const scene of cases) {
    const fixture = createFixture(device, {...scene, quantileCount: 9});
    for (const interpolation of INTERPOLATIONS) {
      // Probabilities beyond the list stay NaN, and invalid ones (-0.1, 1.5, NaN) give NaN.
      const frame: Frame = {
        quantiles: [0, 0.5, 1, 0.25, 0.75, -0.1, 1.5, NaN],
        interpolation,
        filterRange: [0.3, 0.6]
      };
      const expected = computeColumnQuantilesOracle({
        values: scene.values,
        mask: scene.mask,
        quantileCount: 9,
        ...frame
      });
      expectParity(await fixture.run(frame), expected, interpolation, scene.name);
    }
    fixture.destroy();
  }
});

it('GPUColumnQuantiles percentile filter matches the oracle for edge fractions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createColumn('ties', 21, 4097);
  const fixture = createFixture(device, {values, quantileCount: 1});
  const ranges: [number, number][] = [
    [0, 1],
    [0, 0],
    [1, 1],
    [0.5, 0.5],
    [0.9, 0.1],
    [0.333, 0.667],
    [NaN, NaN],
    [-1, 2],
    [0.0001, 0.9999]
  ];
  for (const filterRange of ranges) {
    const frame: Frame = {
      quantiles: [0.5],
      interpolation: 'lower',
      filterRange
    };
    expectParity(
      await fixture.run(frame),
      computeColumnQuantilesOracle({values, ...frame}),
      'lower',
      `filter ${filterRange}`
    );
  }
  fixture.destroy();
});

it('GPUColumnQuantiles runs several frames on one compiled graph without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 8000;
  let values = createColumn('wide', 77, rows);
  const fixture = createFixture(device, {values, quantileCount: 5});
  const frames: Frame[] = [
    {
      quantiles: [0.1, 0.2, 0.3, 0.4, 0.5],
      interpolation: 'linear',
      filterRange: [0, 1]
    },
    {
      quantiles: [0.9, 0.5],
      interpolation: 'nearest',
      filterRange: [0.2, 0.4]
    },
    {
      quantiles: [1, 0, 0.5, 0.5, 0.5],
      interpolation: 'midpoint',
      filterRange: [0.05, 0.95]
    },
    {quantiles: [0.33], interpolation: 'higher', filterRange: [0.7, 0.8]},
    {
      quantiles: [0.2, 0.4, 0.6, 0.8, 1],
      interpolation: 'lower',
      filterRange: [0, 0.5]
    }
  ];
  for (const [frameIndex, frame] of frames.entries()) {
    if (frameIndex === 3) {
      // Contents of the input change between frames too.
      values = createColumn('ties', 78, rows);
      fixture.valuesBuffer.write(values);
    }
    expectParity(
      await fixture.run(frame),
      computeColumnQuantilesOracle({values, quantileCount: 5, ...frame}),
      frame.interpolation ?? 'linear',
      `frame ${frameIndex}`
    );
  }
  expect(fixture.counters).toEqual({getCommandNodes: 1, compile: 1});
  fixture.destroy();
});

it('GPUColumnQuantiles is bitwise identical across repeated runs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createColumn('wide', 99, 60000);
  const fixture = createFixture(device, {values, quantileCount: 9});
  const frame: Frame = {
    quantiles: PROBABILITIES,
    interpolation: 'linear',
    filterRange: [0.2, 0.9]
  };
  const first = await fixture.run(frame);
  for (let repeat = 0; repeat < 3; repeat++) {
    const next = await fixture.run(frame);
    expect(toBits(next.quantiles)).toEqual(toBits(first.quantiles));
    expect(toBits(next.filterBounds)).toEqual(toBits(first.filterBounds));
    expect(Array.from(next.filterMask)).toEqual(Array.from(first.filterMask));
    expect(next.validCount).toBe(first.validCount);
  }
  fixture.destroy();
});

it('GPUColumnQuantiles supports outputs without a filter', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createColumn('ties', 5, 2500);
  const fixture = createFixture(device, {
    values,
    quantileCount: 3,
    withFilter: false
  });
  const frame: Frame = {
    quantiles: [0.25, 0.5, 0.75],
    interpolation: 'nearest'
  };
  expectParity(
    await fixture.run(frame),
    computeColumnQuantilesOracle({values, ...frame}),
    'nearest',
    'no filter',
    false
  );
  fixture.destroy();
});

it('GPUColumnQuantiles handles 1M rows and reports timings for K=9', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const timings: Record<string, number> = {};
  for (const rows of [1_000_000, 4_000_000]) {
    const values = createColumn('wide', 123, rows);
    const mask = createMask(124, rows, 0.9);
    const fixture = createFixture(device, {values, mask, quantileCount: 9});
    const frame: Frame = {
      quantiles: PROBABILITIES,
      interpolation: 'linear',
      filterRange: [0.1, 0.9]
    };
    const result = await fixture.run(frame);
    if (rows === 1_000_000) {
      expectParity(
        result,
        computeColumnQuantilesOracle({values, mask, ...frame}),
        'linear',
        '1M rows'
      );
    }
    const iterations = 5;
    const start = performance.now();
    for (let iteration = 0; iteration < iterations; iteration++) {
      fixture.encode(frame);
    }
    await fixture.waitForGPU();
    timings[`${rows} rows (ms per frame)`] = (performance.now() - start) / iterations;
    fixture.destroy();
  }
  // eslint-disable-next-line no-console
  console.log('GPUColumnQuantiles timings K=9, with mask and filter', JSON.stringify(timings));
}, 240000);

it('getColumnQuantileNodes leaves outputs untouched while its gate is off', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 5000;
  const quantileCount = 3;
  const values = createColumn('wide', 31, rows);
  const graph = new GPUCommandGraph(device, {id: 'gated-quantiles'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const gateBuffer = track(createInputBuffer(device, Uint32Array.from([0])));
  const valuesBuffer = track(createInputBuffer(device, values));
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'gated-parameters',
    format: 'float32',
    length: getGPUColumnQuantilesParameterLength(quantileCount)
  });
  const outputs = {
    quantiles: track(createOutputBuffer(device, quantileCount)),
    validCount: track(createOutputBuffer(device, 1)),
    filterMask: track(createOutputBuffer(device, rows)),
    filterBounds: track(createOutputBuffer(device, 2))
  };
  const sentinel = {
    quantiles: Float32Array.from([-111, -222, -333]),
    validCount: Uint32Array.from([4242]),
    filterMask: new Uint32Array(rows).fill(7),
    filterBounds: Float32Array.from([-5, -6])
  };
  const resetOutputs = () => {
    outputs.quantiles.write(sentinel.quantiles);
    outputs.validCount.write(sentinel.validCount);
    outputs.filterMask.write(sentinel.filterMask);
    outputs.filterBounds.write(sentinel.filterBounds);
  };
  const nodes = getColumnQuantileNodes(graph, {
    id: 'gated',
    operation: 'GatedQuantilesTest',
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
    parameters: parameterBuffer.importToGraph(graph),
    quantileCount,
    output: {
      quantiles: importGraphBuffer(graph, 'o-q', outputs.quantiles, 'float32', quantileCount),
      validCount: importGraphBuffer(graph, 'o-v', outputs.validCount, 'uint32', 1),
      filterMask: importGraphBuffer(graph, 'o-m', outputs.filterMask, 'uint32', rows),
      filterBounds: importGraphBuffer(graph, 'o-b', outputs.filterBounds, 'float32', 2)
    },
    gate: {
      view: importGraphBuffer(graph, 'gate', gateBuffer, 'uint32', 1),
      value: 7
    }
  });
  graph.add(nodes);
  const compiled = graph.compile();
  const frame: Frame = {
    quantiles: [0.2, 0.5, 0.8],
    interpolation: 'linear',
    filterRange: [0.25, 0.75]
  };
  parameterBuffer.write(getGPUColumnQuantilesParameterValues(frame));
  const readAll = async (): Promise<Result> => ({
    quantiles: Float32Array.from(await readFloat32(outputs.quantiles, quantileCount)),
    validCount: (await readUint32(outputs.validCount, 1))[0],
    filterMask: Uint32Array.from(await readUint32(outputs.filterMask, rows)),
    filterBounds: Float32Array.from(await readFloat32(outputs.filterBounds, 2))
  });

  resetOutputs();
  gateBuffer.write(Uint32Array.from([0]));
  submitGraph(device, compiled, undefined);
  const gatedOff = await readAll();
  expect(Array.from(gatedOff.quantiles)).toEqual(Array.from(sentinel.quantiles));
  expect(gatedOff.validCount).toBe(4242);
  expect(Array.from(gatedOff.filterMask)).toEqual(Array.from(sentinel.filterMask));
  expect(Array.from(gatedOff.filterBounds)).toEqual(Array.from(sentinel.filterBounds));

  const expected = computeColumnQuantilesOracle({values, ...frame});
  for (const gateValue of [7, 0, 7]) {
    gateBuffer.write(Uint32Array.from([gateValue]));
    resetOutputs();
    submitGraph(device, compiled, undefined);
    const result = await readAll();
    if (gateValue === 7) {
      expectParity(result, expected, 'linear', 'gate on');
    } else {
      expect(result.validCount).toBe(4242);
    }
  }
  compiled.destroy();
  parameterBuffer.destroy();
  buffers.forEach(buffer => buffer.destroy());
});
