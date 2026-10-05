// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  type GPUClassBreaksMethod,
  type GPUClassBreaksParameters
} from '../../../src/map-graphs/column-classification/class-breaks-parameters';
import {GPUClassBreaks} from '../../../src/map-graphs/column-classification/gpu-class-breaks';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeClassBreaksOracle,
  countClassesOracle,
  getNaturalBreaksCost,
  type ClassBreaksOracleResult
} from './class-breaks-oracle';

const MAXIMUM_CLASS_COUNT = 10;

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

/** Heavy-tailed (Pareto-like) values with a few clusters, NaNs, infinities, and ties. */
function createColumn(
  seed: number,
  rows: number,
  options: {specials?: boolean} = {}
): Float32Array {
  const random = createRandom(seed);
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const choice = random();
    if (options.specials && choice < 0.01) {
      values[row] = [NaN, Infinity, -Infinity, -0, 0][Math.floor(random() * 5)];
    } else if (choice < 0.3) {
      values[row] = 10 + Math.round(random() * 4);
    } else if (choice < 0.6) {
      values[row] = 100 / (1 - random() * 0.999) ** 0.8;
    } else {
      values[row] = (random() - 0.5) * 50;
    }
  }
  return values;
}

/** Three tight, well separated clusters, so natural breaks have one clear optimum. */
function createClusteredColumn(seed: number, rows: number): Float32Array {
  const random = createRandom(seed);
  const centers = [5, 40, 90];
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    values[row] = centers[row % 3] + (random() - 0.5) * 4;
  }
  values[0] = 0;
  values[1] = 100;
  return values;
}

type Result = {breaks: number[]; breakBits: number[]; classCount: number; classCounts: number[]};

type Harness = {
  run(
    values: Float32Array,
    mask: Uint32Array | undefined,
    parameters: GPUClassBreaksParameters
  ): Promise<Result>;
  /** Submits the compiled graph `frameCount` times and waits for one small readback. */
  submit(frameCount: number): Promise<void>;
  buildCount: number;
  destroy(): void;
};

function createHarness(
  device: Device,
  props: {
    rows: number;
    withMask: boolean;
    naturalBreaksBinCount: number;
    methods?: GPUClassBreaksMethod[];
  }
): Harness {
  const {rows, withMask, naturalBreaksBinCount} = props;
  const buffers: Buffer[] = [];
  const keep = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const valuesBuffer = keep(createInputBuffer(device, new Float32Array(rows)));
  const maskBuffer = withMask ? keep(createInputBuffer(device, new Uint32Array(rows))) : undefined;
  const breaksBuffer = keep(createOutputBuffer(device, MAXIMUM_CLASS_COUNT + 1));
  const classCountBuffer = keep(createOutputBuffer(device, 1));
  const classCountsBuffer = keep(createOutputBuffer(device, MAXIMUM_CLASS_COUNT));
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'class-breaks-parameters',
    format: 'float32',
    length: getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
  });
  const graph = new GPUCommandGraph(device, {id: 'class-breaks-graph'});
  const recipe = new GPUClassBreaks({
    id: 'breaks',
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
    mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    maximumClassCount: MAXIMUM_CLASS_COUNT,
    methods: props.methods,
    naturalBreaksBinCount,
    output: {
      breaks: importGraphBuffer(
        graph,
        'o-breaks',
        breaksBuffer,
        'float32',
        MAXIMUM_CLASS_COUNT + 1
      ),
      classCount: importGraphBuffer(graph, 'o-class-count', classCountBuffer, 'uint32', 1),
      classCounts: importGraphBuffer(
        graph,
        'o-class-counts',
        classCountsBuffer,
        'uint32',
        MAXIMUM_CLASS_COUNT
      )
    }
  });
  const harness: Harness = {
    buildCount: 0,
    async run(values, mask, parameters) {
      valuesBuffer.write(values);
      if (maskBuffer) {
        maskBuffer.write(mask ?? new Uint32Array(rows).fill(1));
      }
      parameterBuffer.write(getGPUClassBreaksParameterValues(parameters, MAXIMUM_CLASS_COUNT));
      submitGraph(device, compiled, undefined);
      const breaks = await readFloat32(breaksBuffer, MAXIMUM_CLASS_COUNT + 1);
      const breakBits = await readUint32(breaksBuffer, MAXIMUM_CLASS_COUNT + 1);
      const [classCount] = await readUint32(classCountBuffer, 1);
      const classCounts = await readUint32(classCountsBuffer, MAXIMUM_CLASS_COUNT);
      return {breaks, breakBits, classCount, classCounts};
    },
    async submit(frameCount) {
      for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        submitGraph(device, compiled, undefined);
      }
      await readUint32(classCountBuffer, 1);
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
  graph.add(recipe);
  harness.buildCount++;
  const compiled = graph.compile();
  return harness;
}

/** Relative closeness for edges that come out of f32 arithmetic. */
function expectEdgesClose(
  actual: number[],
  expected: ClassBreaksOracleResult,
  tolerance: number,
  label: string
): void {
  const scale = Math.max(
    Math.abs(expected.maximum - expected.minimum),
    Math.abs(expected.maximum),
    1e-30
  );
  for (let edge = 0; edge <= MAXIMUM_CLASS_COUNT; edge++) {
    const want = expected.breaks[edge];
    const got = actual[edge];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label} edge ${edge} should be NaN, got ${got}`).toBe(true);
    } else {
      expect(Math.abs(got - want), `${label} edge ${edge}: ${got} vs ${want}`).toBeLessThanOrEqual(
        tolerance * scale
      );
    }
  }
}

function expectEdgesExact(
  actual: number[],
  expected: ClassBreaksOracleResult,
  label: string
): void {
  for (let edge = 0; edge <= MAXIMUM_CLASS_COUNT; edge++) {
    const want = expected.breaks[edge];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(actual[edge]), `${label} edge ${edge}`).toBe(true);
    } else {
      expect(
        Object.is(actual[edge], Math.fround(want)) || actual[edge] === Math.fround(want),
        `${label} edge ${edge}: ${actual[edge]} vs ${want}`
      ).toBe(true);
    }
  }
}

function checkFrame(
  result: Result,
  values: Float32Array,
  mask: Uint32Array | undefined,
  parameters: GPUClassBreaksParameters,
  naturalBreaksBinCount: number
): void {
  const label = `${parameters.method} k=${parameters.classCount ?? ''}`;
  const oracle = computeClassBreaksOracle({
    values,
    mask,
    parameters,
    maximumClassCount: MAXIMUM_CLASS_COUNT,
    naturalBreaksBinCount
  });
  if (parameters.method === 'natural-breaks' && oracle.naturalHistogram && result.classCount > 1) {
    // Near-tied partitions may resolve differently in f32; the GPU partition must be optimal.
    const width = Math.fround(
      Math.fround(oracle.maximum - oracle.minimum) * Math.fround(1 / naturalBreaksBinCount)
    );
    const startBins = result.breaks
      .slice(1, result.classCount)
      .map(edge => Math.round((edge - oracle.minimum) / width));
    const optimum = getNaturalBreaksCost(oracle.naturalHistogram, oracle.naturalStartBins!);
    const gpuCost = getNaturalBreaksCost(oracle.naturalHistogram, startBins);
    expect(result.classCount, label).toBe(oracle.classCount);
    expect(gpuCost, label).toBeLessThanOrEqual(optimum * (1 + 1e-5) + 1e-6);
  } else {
    expect(result.classCount, label).toBe(oracle.classCount);
    switch (parameters.method) {
      case 'quantile':
      case 'maximum-breaks':
      case 'custom':
        if (parameters.method === 'quantile') {
          // Linear interpolation may differ by one ulp from a fused multiply-add.
          expectEdgesClose(result.breaks, oracle, 1e-6, label);
        } else {
          expectEdgesExact(result.breaks, oracle, label);
        }
        break;
      case 'head-tail':
        // f32 fixed-order means versus f64 means.
        expectEdgesClose(result.breaks, oracle, 1e-5, label);
        break;
      default:
        expectEdgesClose(result.breaks, oracle, 1e-5, label);
    }
  }
  // Class counts are exact for the edges the GPU produced.
  expect(result.classCounts, `${label} class counts`).toEqual(
    Array.from(
      countClassesOracle(values, mask, result.breaks, result.classCount, MAXIMUM_CLASS_COUNT)
    )
  );
}

const FRAMES: GPUClassBreaksParameters[] = [
  {method: 'equal-interval', classCount: 5},
  {method: 'quantile', classCount: 7},
  {method: 'standard-deviation', classCount: 6, standardDeviationInterval: 0.5},
  {method: 'head-tail', classCount: 10},
  {method: 'head-tail', classCount: 10, headTailRatio: 1},
  {method: 'box-plot'},
  {method: 'maximum-breaks', classCount: 6},
  {method: 'natural-breaks', classCount: 5},
  {method: 'custom', customEdges: [-100, 0, 10, 50, 1000]},
  {method: 'quantile', classCount: 3},
  {method: 'equal-interval', classCount: 1},
  {method: 'natural-breaks', classCount: 10}
];

it('GPUClassBreaks matches the oracle for every method across per-frame changes without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 6000;
  const binCount = 256;
  const harness = createHarness(device, {rows, withMask: true, naturalBreaksBinCount: binCount});
  try {
    const random = createRandom(99);
    for (const [frameIndex, frame] of FRAMES.entries()) {
      const values = createColumn(frameIndex + 1, rows, {specials: frameIndex % 2 === 0});
      const mask = new Uint32Array(rows).map(() => (random() < 0.8 ? 1 : 0));
      const result = await harness.run(values, mask, frame);
      checkFrame(result, values, mask, frame, binCount);
      const again = await harness.run(values, mask, frame);
      expect(again.breakBits, `${frame.method} run-to-run`).toEqual(result.breakBits);
      expect(again.classCounts).toEqual(result.classCounts);
    }
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUClassBreaks natural breaks find the clear optimum on clustered data', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 3000;
  const binCount = 1024;
  const harness = createHarness(device, {
    rows,
    withMask: false,
    naturalBreaksBinCount: binCount,
    methods: ['natural-breaks', 'head-tail']
  });
  try {
    const values = createClusteredColumn(5, rows);
    const parameters: GPUClassBreaksParameters = {method: 'natural-breaks', classCount: 3};
    const result = await harness.run(values, undefined, parameters);
    const oracle = computeClassBreaksOracle({
      values,
      parameters,
      maximumClassCount: MAXIMUM_CLASS_COUNT,
      naturalBreaksBinCount: binCount
    });
    expect(result.classCount).toBe(3);
    expectEdgesClose(result.breaks, oracle, 1e-6, 'clustered natural');
    // The breaks fall in the gaps between the clusters.
    expect(result.breaks[1]).toBeGreaterThan(7);
    expect(result.breaks[1]).toBeLessThan(38);
    expect(result.breaks[2]).toBeGreaterThan(42);
    expect(result.breaks[2]).toBeLessThan(88);
    expect(result.classCounts.slice(0, 3)).toEqual([1000, 999, 1001]);
    // A method that is not compiled in yields no classes.
    const missing = await harness.run(values, undefined, {method: 'quantile', classCount: 4});
    expect(missing.classCount).toBe(0);
    expect(missing.breaks.every(Number.isNaN)).toBe(true);
  } finally {
    harness.destroy();
  }
}, 60000);

it('GPUClassBreaks handles empty, constant, and tiny columns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 64;
  const harness = createHarness(device, {rows, withMask: true, naturalBreaksBinCount: 64});
  try {
    const methods: GPUClassBreaksParameters[] = FRAMES.filter(frame => frame.method !== 'custom');
    const constant = new Float32Array(rows).fill(3.5);
    const allNaN = new Float32Array(rows).fill(NaN);
    const twoValues = new Float32Array(rows).map((_, row) => (row % 2 ? 1 : 2));
    const oneRowMask = new Uint32Array(rows);
    oneRowMask[7] = 1;
    for (const frame of methods) {
      for (const [label, values, mask] of [
        ['constant', constant, undefined],
        ['all NaN', allNaN, undefined],
        ['fully masked', twoValues, new Uint32Array(rows)],
        ['one row', twoValues, oneRowMask],
        ['two values', twoValues, undefined]
      ] as const) {
        const result = await harness.run(values, mask, frame);
        const oracle = computeClassBreaksOracle({
          values,
          mask,
          parameters: frame,
          maximumClassCount: MAXIMUM_CLASS_COUNT,
          naturalBreaksBinCount: 64
        });
        expect(result.classCount, `${frame.method} ${label}`).toBe(oracle.classCount);
        if (oracle.classCount <= 1) {
          expectEdgesExact(result.breaks, oracle, `${frame.method} ${label}`);
        } else {
          checkFrame(result, values, mask, frame, 64);
        }
      }
    }
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUClassBreaks scales to 1M rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 1 << 20;
  const harness = createHarness(device, {
    rows,
    withMask: false,
    naturalBreaksBinCount: 1024,
    methods: [
      'quantile',
      'natural-breaks',
      'standard-deviation',
      'head-tail',
      'equal-interval',
      'box-plot'
    ]
  });
  try {
    const values = createColumn(77, rows);
    for (const frame of [
      {method: 'quantile', classCount: 9},
      {method: 'natural-breaks', classCount: 7},
      {method: 'head-tail', classCount: 10},
      {method: 'standard-deviation', classCount: 6}
    ] as GPUClassBreaksParameters[]) {
      const result = await harness.run(values, undefined, frame);
      checkFrame(result, values, undefined, frame, 1024);
      await harness.submit(2);
      const frameCount = 20;
      const start = performance.now();
      await harness.submit(frameCount);
      // Graph encode, submit, and GPU time per frame; the column is already resident.
      // biome-ignore lint/suspicious/noConsole: benchmark output for the handoff
      console.log(
        `GPUClassBreaks ${frame.method} 1M rows: ${((performance.now() - start) / frameCount).toFixed(2)} ms/frame`
      );
    }
  } finally {
    harness.destroy();
  }
}, 180000);
