// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
  type GPUClassBreaksMethod,
  type GPUClassBreaksSettings
} from '../../../src/gpu-dataframe/column-classification/class-breaks-parameters';
import {GPUClassBreaks} from '../../../src/gpu-dataframe/column-classification/gpu-class-breaks';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeClassBreaksOracle,
  computeExactNaturalBreaks,
  countClassesOracle,
  getWithinClassCost,
  type ClassBreaksOracleResult
} from './class-breaks-oracle';
import {
  CLASS_BREAKS_REFERENCE,
  getGoodnessOfVarianceFit,
  getReferenceEdgeMismatches,
  getReferenceParameters,
  NATURAL_BREAKS_LOCAL_OPTIMA,
  NATURAL_BREAKS_LOCAL_OPTIMUM_FIT_TOLERANCE
} from './class-breaks-reference';
import {LegacyNaturalBreaks} from './legacy-natural-breaks';
import {CLASS_BREAKS_EDGE_POLICY} from '../dataframe-edge-policy';
import {requireTestDevice} from '../../utils/require-test-device';

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
  /** `parameters` may be a raw packed view, to write values the CPU packer refuses. */
  run(
    values: Float32Array,
    mask: Uint32Array | undefined,
    parameters: GPUClassBreaksSettings | Float32Array
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
    valueFormat?: 'float32' | 'uint32' | 'sint32';
    /** Defaults to MAXIMUM_CLASS_COUNT. */
    maximumClassCount?: number;
  }
): Harness {
  const {rows, withMask, naturalBreaksBinCount} = props;
  const maximumClassCount = props.maximumClassCount ?? MAXIMUM_CLASS_COUNT;
  const buffers: Buffer[] = [];
  const keep = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const valuesBuffer = keep(createInputBuffer(device, new Float32Array(rows)));
  const maskBuffer = withMask ? keep(createInputBuffer(device, new Uint32Array(rows))) : undefined;
  const breaksBuffer = keep(createOutputBuffer(device, maximumClassCount + 1));
  const classCountBuffer = keep(createOutputBuffer(device, 1));
  const classCountsBuffer = keep(createOutputBuffer(device, maximumClassCount));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'class-breaks-parameters',
    format: 'float32',
    length: getGPUClassBreaksParameterLength(maximumClassCount)
  });
  const graph = new GPUCommandGraph(device, {id: 'class-breaks-graph'});
  const contributor = new GPUClassBreaks({
    id: 'breaks',
    values: importGraphBuffer(graph, 'values', valuesBuffer, props.valueFormat ?? 'float32', rows),
    mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
    parameters: parameterBuffer.importToGraph(graph),
    maximumClassCount,
    methods: props.methods,
    naturalBreaksBinCount,
    output: {
      breaks: importGraphBuffer(graph, 'o-breaks', breaksBuffer, 'float32', maximumClassCount + 1),
      classCount: importGraphBuffer(graph, 'o-class-count', classCountBuffer, 'uint32', 1),
      classCounts: importGraphBuffer(
        graph,
        'o-class-counts',
        classCountsBuffer,
        'uint32',
        maximumClassCount
      )
    }
  });
  const harness: Harness = {
    buildCount: 0,
    async run(values, mask, parameters) {
      valuesBuffer.write(
        props.valueFormat === 'uint32'
          ? Uint32Array.from(values)
          : props.valueFormat === 'sint32'
            ? Int32Array.from(values)
            : values
      );
      if (maskBuffer) {
        maskBuffer.write(mask ?? new Uint32Array(rows).fill(1));
      }
      parameterBuffer.write(
        parameters instanceof Float32Array
          ? parameters
          : getGPUClassBreaksParameterValues(parameters, maximumClassCount)
      );
      submitGraph(device, compiled, undefined);
      const breaks = await readFloat32(breaksBuffer, maximumClassCount + 1);
      const breakBits = await readUint32(breaksBuffer, maximumClassCount + 1);
      const [classCount] = await readUint32(classCountBuffer, 1);
      const classCounts = await readUint32(classCountsBuffer, maximumClassCount);
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
  graph.add(contributor);
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
  parameters: GPUClassBreaksSettings,
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
    // The CPU mirror runs the same algorithm; f32 rounding on the GPU may resolve a near tie
    // differently, in which case the partition must cost as little (relative 1e-6: the f32 class
    // cost has a relative error below 1e-6, see natural-breaks.node.spec.ts).
    expect(result.classCount, label).toBe(oracle.classCount);
    const gpuEdges = result.breaks.slice(1, result.classCount);
    const mirrorEdges = Array.from(oracle.breaks.slice(1, oracle.classCount));
    if (gpuEdges.some((edge, index) => edge !== mirrorEdges[index])) {
      const finite = Array.from(values).filter((value, row) => !mask || mask[row] !== 0);
      expect(getWithinClassCost(finite, gpuEdges), label).toBeLessThanOrEqual(
        getWithinClassCost(finite, mirrorEdges) * (1 + 1e-6)
      );
    }
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

const FRAMES: GPUClassBreaksSettings[] = [
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
  requireTestDevice(device);
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
  requireTestDevice(device);
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
    const parameters: GPUClassBreaksSettings = {method: 'natural-breaks', classCount: 3};
    const result = await harness.run(values, undefined, parameters);
    const oracle = computeClassBreaksOracle({
      values,
      parameters,
      maximumClassCount: MAXIMUM_CLASS_COUNT,
      naturalBreaksBinCount: binCount
    });
    expect(result.classCount).toBe(3);
    expectEdgesClose(result.breaks, oracle, 1e-6, 'clustered natural');
    // The breaks are the smallest values of the second and third cluster (centres 40 and 90, half
    // width 2): data values, not bin edges.
    expect(result.breaks[1]).toBeGreaterThanOrEqual(38);
    expect(result.breaks[1]).toBeLessThan(38.1);
    expect(result.breaks[2]).toBeGreaterThanOrEqual(88);
    expect(result.breaks[2]).toBeLessThan(88.1);
    expect(Array.from(values)).toContain(result.breaks[1]);
    expect(Array.from(values)).toContain(result.breaks[2]);
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
  requireTestDevice(device);
  const rows = 64;
  const harness = createHarness(device, {rows, withMask: true, naturalBreaksBinCount: 64});
  try {
    const methods: GPUClassBreaksSettings[] = FRAMES.filter(frame => frame.method !== 'custom');
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
  requireTestDevice(device);
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
    ] as GPUClassBreaksSettings[]) {
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

it('GPUClassBreaks classifies uint32 and sint32 columns like the converted float column', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 4000;
  const binCount = 256;
  for (const valueFormat of ['uint32', 'sint32'] as const) {
    const harness = createHarness(device, {
      rows,
      withMask: true,
      naturalBreaksBinCount: binCount,
      valueFormat
    });
    try {
      const random = createRandom(valueFormat === 'uint32' ? 5 : 6);
      const mask = new Uint32Array(rows).map(() => (random() < 0.85 ? 1 : 0));
      for (const [frameIndex, frame] of FRAMES.entries()) {
        // Count-like integers: many ties, a heavy tail, and negatives for sint32.
        const values = new Float32Array(rows).map(() => {
          const count = Math.floor(3 / (1 - random() * 0.999) ** 1.5);
          return valueFormat === 'sint32' ? count - 4 : count;
        });
        const result = await harness.run(values, mask, frame);
        checkFrame(result, values, mask, frame, binCount);
        expect(result.classCount, `${valueFormat} frame ${frameIndex}`).toBeGreaterThan(0);
      }
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUClassBreaks matches the mapclassify and jenkspy references', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fixture: test/oracles/dataframe-class-breaks.py (mapclassify 2.11, jenkspy 0.4.1); each case
  // carries its derived tolerance. The CPU mirror passes the same cases in
  // class-breaks-reference.node.spec.ts.
  const mismatches: string[] = [];
  const localOptima: string[] = [];
  const unexpectedLocalOptima: string[] = [];
  for (const [name, scene] of Object.entries(CLASS_BREAKS_REFERENCE.scenes)) {
    const values = Float32Array.from(scene.values);
    const harness = createHarness(device, {
      rows: values.length,
      withMask: false,
      naturalBreaksBinCount: 2048
    });
    try {
      for (const referenceCase of scene.cases) {
        const result = await harness.run(values, undefined, getReferenceParameters(referenceCase));
        mismatches.push(
          ...getReferenceEdgeMismatches(
            `${name} ${referenceCase.method} k=${referenceCase.classCount}`,
            result.breaks,
            result.classCount,
            referenceCase
          )
        );
      }
      for (const reference of scene.naturalBreaks) {
        const result = await harness.run(values, undefined, {
          method: 'natural-breaks',
          classCount: reference.classCount
        });
        const label = `${name} ${reference.classCount}`;
        expect(result.classCount, label).toBe(reference.classCount);
        const inner = result.breaks.slice(1, reference.classCount);
        // Exact lower edges of the exact Fisher-Jenks partition: equal data values. The JSON of the
        // fixture is f32 data, so compare after Math.fround.
        const expected = reference.exactLowerEdges.map(Math.fround);
        if (inner.every((edge, index) => edge === expected[index])) {
          expect(result.classCounts.slice(0, reference.classCount), label).toEqual(
            reference.exactClassSizes
          );
        } else {
          // A neighbouring local optimum (see NATURAL_BREAKS_LOCAL_OPTIMA), also where f32 rounding
          // on the GPU resolved a near tie differently from the CPU mirror: allowed only within the
          // documented goodness of variance fit.
          const shortfall =
            reference.exactGoodnessOfVarianceFit - getGoodnessOfVarianceFit(values, inner);
          localOptima.push(`${label} (${shortfall.toExponential(1)})`);
          expect(shortfall, label).toBeLessThanOrEqual(NATURAL_BREAKS_LOCAL_OPTIMUM_FIT_TOLERANCE);
          if (!NATURAL_BREAKS_LOCAL_OPTIMA[label]) {
            unexpectedLocalOptima.push(label);
          }
        }
      }
    } finally {
      harness.destroy();
    }
  }
  expect(mismatches).toEqual([]);
  // Beyond the documented local optima the GPU must find the exact partition; log any other.
  console.log(`natural breaks local optima: ${localOptima.join(', ') || 'none'}`);
  expect(unexpectedLocalOptima).toEqual([]);
}, 120000);

it('GPUClassBreaks natural breaks handle ties, outliers, constant columns and missing rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const run = async (
    values: Float32Array,
    classCount: number,
    mask?: Uint32Array,
    maximumClassCount = MAXIMUM_CLASS_COUNT
  ) => {
    const harness = createHarness(device, {
      rows: values.length,
      withMask: Boolean(mask),
      naturalBreaksBinCount: 2048,
      methods: ['natural-breaks'],
      maximumClassCount
    });
    try {
      return await harness.run(values, mask, {method: 'natural-breaks', classCount});
    } finally {
      harness.destroy();
    }
  };
  // A constant column is one class, whatever was requested.
  const constant = await run(new Float32Array(200).fill(4.5), 5);
  expect(constant.classCount).toBe(1);
  expect(constant.breaks.slice(0, 2)).toEqual([4.5, 4.5]);
  // A class count above the distinct values gives one class per distinct value.
  for (const name of ['plateaus', 'two-values']) {
    const scene = CLASS_BREAKS_REFERENCE.scenes[name];
    const result = await run(Float32Array.from(scene.values), MAXIMUM_CLASS_COUNT);
    const distinct = scene.distinctValueEdges!.map(Math.fround);
    expect(result.classCount, name).toBe(distinct.length + 1);
    expect(result.breaks.slice(1, result.classCount), name).toEqual(distinct);
  }
  // Heavy ties, k = 2 and a single extreme outlier against the exact dynamic program.
  const random = createRandom(31);
  const ties = new Float32Array(900).map(() => Math.floor(random() * 11));
  const outlier = new Float32Array(500).map(() => 10 + (random() + random() + random() - 1.5) * 2);
  outlier[123] = 1e7;
  for (const [label, values, classCounts] of [
    ['ties', ties, [2, 3, 5, 7]],
    ['outlier', outlier, [2, 3, 4, 5, 6, 7]]
  ] as const) {
    for (const classCount of classCounts) {
      const exact = computeExactNaturalBreaks(values, classCount);
      const result = await run(values, classCount);
      expect(result.classCount, `${label} ${classCount}`).toBe(classCount);
      const inner = Array.from(result.breaks.slice(1, classCount));
      if (exact.unique) {
        expect(inner, `${label} ${classCount}`).toEqual(exact.edges.map(Math.fround));
      }
    }
  }
  // NaN, infinities and masked rows are missing: the result is that of the remaining column.
  const clean = Float32Array.from(CLASS_BREAKS_REFERENCE.scenes['clusters'].values);
  const noisy = new Float32Array(clean.length * 2);
  const mask = new Uint32Array(noisy.length);
  clean.forEach((value, row) => {
    noisy[2 * row] = value;
    mask[2 * row] = 1;
    // Between every two rows: a decoy (masked), and every few rows NaN or an infinity.
    noisy[2 * row + 1] = [1e6, NaN, Infinity, -Infinity][row % 4];
    mask[2 * row + 1] = row % 4 === 0 ? 0 : 1;
  });
  const cleanResult = await run(clean, 4);
  const noisyResult = await run(noisy, 4, mask);
  expect(noisyResult.breakBits).toEqual(cleanResult.breakBits);
  expect(noisyResult.classCount).toBe(4);
}, 120000);

it('GPUClassBreaks natural breaks A/B: O(k B^2) binned path against divide and conquer with refinement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Same device, same column, same run: the previous pipeline (test-only legacy-natural-breaks.ts)
  // against the current one, at 1024 and 2048 bins, k = 8, 2^20 rows. Logs the time per frame and
  // the within-class cost of both partitions; the new cost must not be above the old one, whose
  // breaks snap to bin edges.
  const rows = 1 << 20;
  const classCount = 8;
  const values = createColumn(77, rows);
  const frameCount = 20;
  for (const binCount of [1024, 2048]) {
    const harness = createHarness(device, {
      rows,
      withMask: false,
      naturalBreaksBinCount: binCount,
      methods: ['natural-breaks'],
      maximumClassCount: classCount
    });
    const valuesBuffer = createInputBuffer(device, values);
    const breaksBuffer = createOutputBuffer(device, classCount + 1);
    const legacyGraph = new GPUCommandGraph(device, {id: `legacy-${binCount}`});
    legacyGraph.add(
      new LegacyNaturalBreaks({
        id: 'legacy',
        values: importGraphBuffer(legacyGraph, 'legacy-values', valuesBuffer, 'float32', rows),
        classCount,
        binCount,
        breaks: importGraphBuffer(
          legacyGraph,
          'legacy-breaks',
          breaksBuffer,
          'float32',
          classCount + 1
        )
      })
    );
    const legacy = legacyGraph.compile();
    try {
      const frame: GPUClassBreaksSettings = {method: 'natural-breaks', classCount};
      const current = await harness.run(values, undefined, frame);
      submitGraph(device, legacy, undefined);
      const legacyBreaks = Array.from(await readFloat32(breaksBuffer, classCount + 1));
      // Warm up, then time whole frames (encode, submit, GPU) with one small readback each.
      await harness.submit(2);
      const start = performance.now();
      await harness.submit(frameCount);
      const currentTime = (performance.now() - start) / frameCount;
      for (let index = 0; index < 2; index++) {
        submitGraph(device, legacy, undefined);
      }
      await readFloat32(breaksBuffer, 1);
      const legacyStart = performance.now();
      for (let index = 0; index < frameCount; index++) {
        submitGraph(device, legacy, undefined);
      }
      await readFloat32(breaksBuffer, 1);
      const legacyTime = (performance.now() - legacyStart) / frameCount;
      const currentCost = getWithinClassCost(values, current.breaks.slice(1, classCount));
      const legacyCost = getWithinClassCost(values, legacyBreaks.slice(1, classCount));
      console.log(
        `natural breaks A/B ${binCount} bins, k=${classCount}, ${rows} rows: ` +
          `old ${legacyTime.toFixed(2)} ms/frame, new ${currentTime.toFixed(2)} ms/frame, ` +
          `within-class cost old ${legacyCost.toExponential(6)} new ${currentCost.toExponential(6)}`
      );
      expect(current.classCount).toBe(classCount);
      expect(currentCost).toBeLessThanOrEqual(legacyCost * (1 + 1e-9));
    } finally {
      legacy.destroy();
      valuesBuffer.destroy();
      breaksBuffer.destroy();
      harness.destroy();
    }
  }
}, 240000);

it('GPUClassBreaks follows the class breaks edge policy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const maximumClassCount = 4;
  for (const row of CLASS_BREAKS_EDGE_POLICY) {
    const values = Float32Array.from(row.values);
    const harness = createHarness(device, {
      rows: values.length,
      withMask: false,
      naturalBreaksBinCount: 16,
      maximumClassCount,
      // Only the methods the policy rows use: the constructor rightly refuses 'box-plot' (6
      // classes) at maximumClassCount 4, and the default compiles every method.
      methods: ['equal-interval', 'quantile', 'standard-deviation', 'custom']
    });
    try {
      // Pack a valid frame (in-range counts as given, so quantile probabilities are packed), then
      // write the raw class count: out-of-range rows test the GPU clamp.
      const isInRange = row.classCount >= 1 && row.classCount <= maximumClassCount;
      const parameters = getGPUClassBreaksParameterValues(
        {
          method: row.method,
          classCount: isInRange ? row.classCount : 1,
          // NaN edges are written raw below: the packer refuses them.
          customEdges: row.customEdges?.map(edge => (Number.isNaN(edge) ? 0 : edge))
        },
        maximumClassCount
      );
      parameters[1] = row.classCount;
      row.customEdges?.forEach((edge, index) => {
        parameters[GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH + index] = edge;
      });
      if (row.standardDeviationInterval !== undefined) {
        parameters[2] = row.standardDeviationInterval;
      }
      const result = await harness.run(values, undefined, parameters);
      expect(result.classCount, row.name).toBe(row.expectedClassCount);
      for (const [index, expected] of row.expectedBreaks.entries()) {
        expect(
          Object.is(result.breaks[index], expected),
          `${row.name} e[${index}]: ${result.breaks[index]}`
        ).toBe(true);
      }
    } finally {
      harness.destroy();
    }
  }
});
