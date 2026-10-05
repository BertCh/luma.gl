// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {GPUColorScale} from '../../../src/gpu-dataframe/column-classification/gpu-color-scale';
import {
  getGPUColorScaleParameterValues,
  packGPUColor,
  type GPUColorScaleParameterOptions
} from '../../../src/gpu-dataframe/column-classification/color-scale-parameters';
import {computeColorScaleOnCPU, type ColorScaleOracleResult} from './color-scale-oracle';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';

const NONE = 0xffffffff;

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

type Frame = {
  domain: number[];
  palette: number[];
  options: Omit<GPUColorScaleParameterOptions, 'domainCount' | 'paletteCount'> & {
    domainCount?: number;
    paletteCount?: number;
  };
  /** Class count written to the GPU-side domainCount view (view path only). */
  classCount?: number;
};

type FixtureConfig = {
  values: Float32Array | Uint32Array;
  mask?: Uint32Array;
  maximumDomainCount: number;
  maximumPaletteCount: number;
  useDomainCountView?: boolean;
  outputs?: {colors?: boolean; classIndices?: boolean; classCounts?: boolean};
};

type FrameResult = {colors: Uint32Array; classIndices: Uint32Array; classCounts: Uint32Array};

type Fixture = {
  compileCount: number;
  frames: number;
  run(frame: Frame): Promise<FrameResult>;
  destroy(): void;
};

function createFixture(device: Device, config: FixtureConfig): Fixture {
  const rows = config.values.length;
  const isOrdinal = config.values instanceof Uint32Array;
  const wanted = config.outputs ?? {colors: true, classIndices: true, classCounts: true};
  const graph = new GPUCommandGraph(device, {id: 'color-scale-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'scale-parameters',
    format: 'float32',
    length: 9
  });
  const domainBuffer = track(
    createInputBuffer(device, new Float32Array(config.maximumDomainCount))
  );
  const paletteBuffer = track(
    createInputBuffer(device, new Uint32Array(config.maximumPaletteCount))
  );
  const outputBuffers = {
    colors: track(createOutputBuffer(device, rows)),
    classIndices: track(createOutputBuffer(device, rows)),
    classCounts: track(createOutputBuffer(device, config.maximumPaletteCount))
  };
  let domainCountView;
  let classCountSource: Buffer | undefined;
  const extraNodes = [];
  if (config.useDomainCountView) {
    // A GPU kernel writes the class count into a graph transient, like GPUClassBreaks would.
    classCountSource = track(createInputBuffer(device, new Uint32Array(1)));
    const sourceView = importGraphBuffer(
      graph,
      'class-count-source',
      classCountSource,
      'uint32',
      1
    );
    domainCountView = createTransientView(graph, 'domain-count', 'uint32', 1);
    extraNodes.push(
      createWGSLKernelNode(graph, {
        id: 'write-class-count',
        operation: 'test',
        bindings: [
          {name: 'classCountIn', view: sourceView, type: 'u32', access: 'read'},
          {name: 'classCountOut', view: domainCountView, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: 'classCountOut[classCountOutOffset] = classCountIn[classCountInOffset];'
      })
    );
  }
  for (const node of extraNodes) {
    graph.add(node);
  }
  graph.add(
    new GPUColorScale({
      id: 'scale',
      values: isOrdinal
        ? importGraphBuffer(
            graph,
            'values',
            track(createInputBuffer(device, config.values)),
            'uint32',
            rows
          )
        : importGraphBuffer(
            graph,
            'values',
            track(createInputBuffer(device, config.values)),
            'float32',
            rows
          ),
      mask: config.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, config.mask)),
            'uint32',
            rows
          )
        : undefined,
      domain: importGraphBuffer(
        graph,
        'domain',
        domainBuffer,
        'float32',
        config.maximumDomainCount
      ),
      domainCount: domainCountView,
      palette: importGraphBuffer(
        graph,
        'palette',
        paletteBuffer,
        'uint32',
        config.maximumPaletteCount
      ),
      parameters: parameterBuffer.importToGraph(graph),
      maximumDomainCount: config.maximumDomainCount,
      maximumPaletteCount: config.maximumPaletteCount,
      output: {
        colors: wanted.colors
          ? importGraphBuffer(graph, 'o-colors', outputBuffers.colors, 'uint32', rows)
          : undefined,
        classIndices: wanted.classIndices
          ? importGraphBuffer(graph, 'o-classes', outputBuffers.classIndices, 'uint32', rows)
          : undefined,
        classCounts: wanted.classCounts
          ? importGraphBuffer(
              graph,
              'o-counts',
              outputBuffers.classCounts,
              'uint32',
              config.maximumPaletteCount
            )
          : undefined
      }
    })
  );
  const fixture: Fixture = {
    compileCount: 0,
    frames: 0,
    async run(frame) {
      const domain = new Float32Array(config.maximumDomainCount);
      domain.set(frame.domain);
      const palette = new Uint32Array(config.maximumPaletteCount);
      palette.set(frame.palette);
      domainBuffer.write(domain);
      paletteBuffer.write(palette);
      classCountSource?.write(Uint32Array.of(frame.classCount ?? 0));
      parameterBuffer.write(
        getGPUColorScaleParameterValues({
          ...frame.options,
          domainCount: frame.options.domainCount ?? frame.domain.length,
          paletteCount: frame.options.paletteCount ?? frame.palette.length
        })
      );
      submitGraph(device, compiled, undefined);
      fixture.frames++;
      return {
        colors: wanted.colors
          ? Uint32Array.from(await readUint32(outputBuffers.colors, rows))
          : new Uint32Array(0),
        classIndices: wanted.classIndices
          ? Uint32Array.from(await readUint32(outputBuffers.classIndices, rows))
          : new Uint32Array(0),
        classCounts: wanted.classCounts
          ? Uint32Array.from(
              await readUint32(outputBuffers.classCounts, config.maximumPaletteCount)
            )
          : new Uint32Array(0)
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
  const compiled = graph.compile();
  fixture.compileCount++;
  return fixture;
}

function getOracle(config: FixtureConfig, frame: Frame): ColorScaleOracleResult {
  const domain = new Float32Array(config.maximumDomainCount);
  domain.set(frame.domain);
  const palette = new Uint32Array(config.maximumPaletteCount);
  palette.set(frame.palette);
  const activeDomainCount =
    frame.classCount !== undefined
      ? frame.classCount > 0
        ? frame.classCount + 1
        : 0
      : (frame.options.domainCount ?? frame.domain.length);
  return computeColorScaleOnCPU({
    values: config.values,
    mask: config.mask,
    domain,
    activeDomainCount,
    palette,
    scale: frame.options.scale,
    paletteCount: frame.options.paletteCount ?? frame.palette.length,
    interpolation: frame.options.interpolation,
    clamp: frame.options.clamp,
    noDataColor: frame.options.noDataColor,
    logFloor: frame.options.logFloor,
    exponent: frame.options.exponent,
    maximumDomainCount: config.maximumDomainCount,
    maximumPaletteCount: config.maximumPaletteCount
  });
}

/**
 * Compares a frame with the oracle. `exact` demands bitwise classes and colours (step scales and
 * dyadic inputs); otherwise rows near a class boundary are skipped and linear colours may differ
 * by 1 per channel.
 */
function expectParity(
  actual: FrameResult,
  config: FixtureConfig,
  frame: Frame,
  exact: boolean,
  label: string
): ColorScaleOracleResult {
  const expected = getOracle(config, frame);
  const rows = config.values.length;
  const isLinear =
    frame.options.interpolation === 'linear' &&
    ['linear', 'sqrt', 'pow', 'log', 'symlog'].includes(frame.options.scale);
  let ambiguousCount = 0;
  const countDelta = new Array(config.maximumPaletteCount).fill(0);
  for (let row = 0; row < rows; row++) {
    if (!exact && expected.ambiguous[row]) {
      ambiguousCount++;
      continue;
    }
    if (actual.classIndices.length) {
      expect(actual.classIndices[row], `${label} class row ${row}`).toBe(
        expected.classIndices[row]
      );
    }
    if (actual.colors.length) {
      if (isLinear && !exact) {
        for (let channel = 0; channel < 4; channel++) {
          const difference = Math.abs(
            ((actual.colors[row] >>> (channel * 8)) & 255) -
              ((expected.colors[row] >>> (channel * 8)) & 255)
          );
          expect(difference, `${label} color row ${row} channel ${channel}`).toBeLessThanOrEqual(1);
        }
      } else {
        expect(actual.colors[row], `${label} color row ${row}`).toBe(expected.colors[row]);
      }
    }
  }
  if (actual.classCounts.length) {
    for (let index = 0; index < countDelta.length; index++) {
      countDelta[index] = Math.abs(actual.classCounts[index] - expected.classCounts[index]);
      expect(countDelta[index], `${label} count ${index}`).toBeLessThanOrEqual(
        exact ? 0 : ambiguousCount
      );
    }
  }
  return expected;
}

const GRADIENT = [
  packGPUColor(0, 0, 0, 255),
  packGPUColor(255, 0, 0, 255),
  packGPUColor(255, 255, 0, 128),
  packGPUColor(10, 200, 255, 0)
];
const NO_DATA = 0x11223344;

it('GPUColorScale linear step is exact on dyadic values, with clamp on and off', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const config: FixtureConfig = {
    values: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, NaN, 2, Infinity, -Infinity, -0]),
    mask: Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]),
    maximumDomainCount: 4,
    maximumPaletteCount: 4
  };
  const fixture = createFixture(device, config);
  const base = {scale: 'linear', noDataColor: NO_DATA} as const;
  const unclamped: Frame = {domain: [0, 8], palette: GRADIENT, options: base};
  const result = await fixture.run(unclamped);
  expect(Array.from(result.classIndices)).toEqual([
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    3,
    NONE,
    NONE,
    NONE,
    NONE,
    NONE,
    NONE,
    0
  ]);
  expect(result.colors[1]).toBe(GRADIENT[0]);
  expect(result.colors[4]).toBe(GRADIENT[2]);
  expect(result.colors[9]).toBe(NO_DATA);
  expect(result.colors[12]).toBe(NO_DATA);
  expect(Array.from(result.classCounts)).toEqual([3, 2, 2, 3]);
  expectParity(result, config, unclamped, true, 'unclamped');
  const clamped: Frame = {...unclamped, options: {...base, clamp: true}};
  const clampedResult = await fixture.run(clamped);
  expect(Array.from(clampedResult.classIndices)).toEqual([
    0,
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    3,
    3,
    0,
    NONE,
    NONE,
    3,
    0,
    0
  ]);
  expectParity(clampedResult, config, clamped, true, 'clamped');
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUColorScale matches the oracle for every scale across frames on one compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(11);
  const rows = 4000;
  const values = new Float32Array(rows);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    const roll = random();
    values[row] =
      roll < 0.02
        ? NaN
        : roll < 0.06
          ? -random() * 50
          : roll < 0.08
            ? 0
            : random() * random() * 1200 - 20;
    mask[row] = random() < 0.05 ? 0 : 1;
  }
  const config: FixtureConfig = {values, mask, maximumDomainCount: 6, maximumPaletteCount: 6};
  const palette6 = [0, 1, 2, 3, 4, 5].map(index =>
    packGPUColor(index * 50, 255 - index * 40, (index * 97) & 255, 200 + index * 10)
  );
  const frames: Frame[] = [];
  for (const interpolation of ['step', 'linear'] as const) {
    for (const clamp of [false, true]) {
      const common = {interpolation, clamp, noDataColor: NO_DATA};
      frames.push(
        {domain: [0, 1000], palette: palette6, options: {...common, scale: 'linear'}},
        {domain: [0, 900], palette: palette6.slice(0, 4), options: {...common, scale: 'sqrt'}},
        {
          domain: [-50, 500],
          palette: palette6,
          options: {...common, scale: 'pow', exponent: 2, paletteCount: 5}
        },
        {domain: [1, 1000], palette: palette6, options: {...common, scale: 'log'}},
        {domain: [1, 1000], palette: palette6, options: {...common, scale: 'log', logFloor: 0.01}},
        {domain: [1, 1000], palette: palette6, options: {...common, scale: 'log', logFloor: NaN}},
        {domain: [-20, 1000], palette: palette6, options: {...common, scale: 'symlog'}},
        // Multi-stop (piecewise) because paletteCount equals the domain length.
        {
          domain: [0, 10, 100, 1000],
          palette: palette6.slice(0, 4),
          options: {...common, scale: 'linear'}
        },
        {
          domain: [1, 10, 100, 1000, 2000],
          palette: palette6.slice(0, 5),
          options: {...common, scale: 'log'}
        },
        {domain: [0, 800], palette: palette6, options: {...common, scale: 'quantize'}}
      );
    }
  }
  const fixture = createFixture(device, config);
  for (const [index, frame] of frames.entries()) {
    const actual = await fixture.run(frame);
    expectParity(actual, config, frame, false, `frame ${index} ${frame.options.scale}`);
  }
  expect(fixture.frames).toBe(frames.length);
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUColorScale quantize is exact on dyadic values and clamps or drops out-of-domain rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const config: FixtureConfig = {
    values: Float32Array.from([0, 1.5, 2, 3.5, 4, 6, 7.5, 8, 9, -2]),
    maximumDomainCount: 2,
    maximumPaletteCount: 4
  };
  const fixture = createFixture(device, config);
  const frame: Frame = {
    domain: [0, 8],
    palette: GRADIENT,
    options: {scale: 'quantize', noDataColor: NO_DATA}
  };
  const result = await fixture.run(frame);
  expect(Array.from(result.classIndices)).toEqual([0, 0, 1, 1, 2, 3, 3, 3, NONE, NONE]);
  expectParity(result, config, frame, true, 'quantize');
  const clamped = {...frame, options: {...frame.options, clamp: true}};
  expect(Array.from((await fixture.run(clamped)).classIndices)).toEqual([
    0, 0, 1, 1, 2, 3, 3, 3, 3, 0
  ]);
  fixture.destroy();
});

it('GPUColorScale threshold and quantile use bisectRight on exact edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = Float32Array.from([
    -Infinity,
    -1e30,
    -0.5,
    -0,
    0,
    0.5,
    1,
    1,
    1.5,
    2,
    5,
    5,
    6,
    1e30,
    Infinity,
    NaN,
    3,
    4.999999
  ]);
  const config: FixtureConfig = {values, maximumDomainCount: 8, maximumPaletteCount: 8};
  const palette = Array.from({length: 8}, (_, index) =>
    packGPUColor(index * 30, index, 255 - index, 255)
  );
  const fixture = createFixture(device, config);
  const frames: Frame[] = [
    // Infinite end edges and a duplicated inner edge: class 1 is empty.
    {
      domain: [-Infinity, 1, 1, 5, Infinity],
      palette: palette.slice(0, 4),
      options: {scale: 'threshold', noDataColor: NO_DATA}
    },
    // Edges on -0 / +0 compare equal to the value 0.
    {
      domain: [-1, -0, 0, 2, 5, 10],
      palette,
      options: {scale: 'quantile', noDataColor: NO_DATA}
    },
    // Fewer palette entries than classes: excess classes clamp to the last entry.
    {
      domain: [-Infinity, 0, 1, 2, 5, Infinity],
      palette: palette.slice(0, 3),
      options: {scale: 'threshold', noDataColor: NO_DATA}
    },
    // Single class (two edges): every valid row is class 0.
    {domain: [0, 1], palette, options: {scale: 'threshold', noDataColor: NO_DATA}},
    // A single edge means zero classes: everything is no-data.
    {domain: [0], palette, options: {scale: 'threshold', noDataColor: NO_DATA}},
    // All eight edges used.
    {
      domain: [-Infinity, -10, -1, 0, 1, 5, 6, Infinity],
      palette,
      options: {scale: 'threshold', noDataColor: NO_DATA, clamp: true}
    }
  ];
  for (const [index, frame] of frames.entries()) {
    const actual = await fixture.run(frame);
    const expected = expectParity(actual, config, frame, true, `threshold frame ${index}`);
    if (index === 0) {
      expect(Array.from(actual.classIndices.slice(0, 15))).toEqual([
        0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3
      ]);
      expect(actual.classCounts[1]).toBe(0);
    }
    if (index === 4) {
      expect(Array.from(actual.classCounts).every(count => count === 0)).toBe(true);
    }
    expect(Array.from(actual.classCounts)).toEqual(Array.from(expected.classCounts));
  }
  fixture.destroy();
});

it('GPUColorScale ordinal maps category codes and sends out-of-range codes to no-data', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const config: FixtureConfig = {
    values: Uint32Array.from([0, 1, 2, 3, 4, 2, 1, 0xffffffff, 0, 2]),
    mask: Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 0, 1]),
    maximumDomainCount: 1,
    maximumPaletteCount: 5
  };
  const fixture = createFixture(device, config);
  const frame: Frame = {
    domain: [0],
    palette: GRADIENT.concat([0xaabbccdd]),
    options: {scale: 'ordinal', paletteCount: 3, noDataColor: NO_DATA, domainCount: 0}
  };
  const result = await fixture.run(frame);
  expect(Array.from(result.classIndices)).toEqual([0, 1, 2, NONE, NONE, 2, 1, NONE, NONE, 2]);
  expect(result.colors[3]).toBe(NO_DATA);
  expect(Array.from(result.classCounts)).toEqual([1, 2, 3, 0, 0]);
  expectParity(result, config, frame, true, 'ordinal');
  // More palette entries per frame, same graph.
  const wider = {...frame, options: {...frame.options, paletteCount: 5}};
  const widerResult = await fixture.run(wider);
  expectParity(widerResult, config, wider, true, 'ordinal wider');
  expect(widerResult.classIndices[3]).toBe(3);
  fixture.destroy();
});

it('GPUColorScale reads the active domain length from a GPU-written class count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(5);
  const values = Float32Array.from({length: 1500}, () => random() * 14 - 2);
  const config: FixtureConfig = {
    values,
    maximumDomainCount: 6,
    maximumPaletteCount: 5,
    useDomainCountView: true
  };
  const fixture = createFixture(device, config);
  const domain = [0, 2, 4, 8, 12, 100];
  const palette = [0, 1, 2, 3, 4].map(index => packGPUColor(index * 60, 0, 255 - index * 60, 255));
  // k classes use k + 1 edges; k = 0 makes every row no-data; k = 9 clamps to the 6 edges.
  for (const classCount of [3, 2, 4, 1, 0, 5, 9]) {
    for (const scale of ['threshold', 'linear', 'quantize'] as const) {
      const frame: Frame = {
        domain,
        palette,
        classCount,
        options: {scale, noDataColor: NO_DATA, paletteCount: 5, clamp: scale !== 'threshold'}
      };
      const actual = await fixture.run(frame);
      // The oracle caps the active edges at the domain length like the kernel does.
      expectParity(
        actual,
        config,
        {...frame, classCount: Math.min(classCount, 5)},
        scale === 'threshold',
        `classCount ${classCount} ${scale}`
      );
    }
  }
  const none = await fixture.run({
    domain,
    palette,
    classCount: 0,
    options: {scale: 'threshold', noDataColor: NO_DATA, paletteCount: 5}
  });
  expect(Array.from(none.classCounts)).toEqual([0, 0, 0, 0, 0]);
  expect(none.colors.every(color => color === NO_DATA)).toBe(true);
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUColorScale linear interpolation blends adjacent palette entries within 1 per channel', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const config: FixtureConfig = {
    values: Float32Array.from([0, 2, 4, 6, 8, 1, 3, 5, 7]),
    maximumDomainCount: 2,
    maximumPaletteCount: 3
  };
  const fixture = createFixture(device, config);
  const palette = [
    packGPUColor(0, 0, 0, 0),
    packGPUColor(200, 100, 50, 255),
    packGPUColor(0, 100, 250, 100)
  ];
  const frame: Frame = {
    domain: [0, 8],
    palette,
    options: {scale: 'linear', interpolation: 'linear', paletteCount: 3}
  };
  const result = await fixture.run(frame);
  // t = v / 8 and position = t * 2, so values 0, 4, 8 land exactly on entries.
  expect(result.colors[0]).toBe(palette[0]);
  expect(result.colors[2]).toBe(palette[1]);
  expect(result.colors[4]).toBe(palette[2]);
  // Value 2 is halfway between entries 0 and 1; value 1 is a quarter of the way (12.5 rounds up).
  expect(result.colors[1]).toBe(packGPUColor(100, 50, 25, 128));
  expect(result.colors[5]).toBe(packGPUColor(50, 25, 13, 64));
  expectParity(result, config, frame, false, 'blend');
  // Step interpolation on the same graph returns palette entries bit exactly.
  const stepResult = await fixture.run({
    ...frame,
    options: {...frame.options, interpolation: 'step'}
  });
  for (const colorValue of stepResult.colors) {
    expect(palette).toContain(colorValue);
  }
  fixture.destroy();
});

it('GPUColorScale writes subsets of outputs and is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(31);
  const values = Float32Array.from({length: 2000}, () => random() * 100 - 10);
  const frame: Frame = {
    domain: [0, 20, 50, 100],
    palette: GRADIENT,
    options: {scale: 'log', interpolation: 'linear', clamp: true, noDataColor: NO_DATA, logFloor: 1}
  };
  const outputSets = [
    {colors: true},
    {classIndices: true},
    {classCounts: true},
    {classIndices: true, classCounts: true},
    {colors: true, classIndices: true, classCounts: true}
  ];
  let reference: FrameResult | undefined;
  for (const outputs of outputSets) {
    const config: FixtureConfig = {
      values,
      maximumDomainCount: 4,
      maximumPaletteCount: 4,
      outputs
    };
    const fixture = createFixture(device, config);
    const first = await fixture.run(frame);
    const second = await fixture.run(frame);
    expect(Array.from(second.colors)).toEqual(Array.from(first.colors));
    expect(Array.from(second.classIndices)).toEqual(Array.from(first.classIndices));
    expect(Array.from(second.classCounts)).toEqual(Array.from(first.classCounts));
    if (outputs.colors && outputs.classIndices && outputs.classCounts) {
      reference = first;
    }
    fixture.destroy();
  }
  expect(reference).toBeDefined();
  const full = createFixture(device, {
    values,
    maximumDomainCount: 4,
    maximumPaletteCount: 4
  });
  const fullResult = await full.run(frame);
  expect(Array.from(fullResult.colors)).toEqual(Array.from(reference!.colors));
  expect(Array.from(fullResult.classCounts).reduce((sum, count) => sum + count, 0)).toBe(2000);
  full.destroy();
});

it('GPUColorScale classifies 1M rows like the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(2024);
  const rows = 1_000_000;
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    values[row] = random() < 0.01 ? NaN : Math.fround(random() * 130 - 15);
  }
  const config: FixtureConfig = {values, maximumDomainCount: 8, maximumPaletteCount: 8};
  const fixture = createFixture(device, config);
  const palette = Array.from({length: 8}, (_, index) =>
    packGPUColor(index * 30, 100, 255 - index * 30, 255)
  );
  const threshold: Frame = {
    domain: [-15, 0, 10, 25, 40, 60, 90, 115],
    palette,
    options: {scale: 'threshold', noDataColor: NO_DATA}
  };
  const linear: Frame = {
    domain: [-10, 100],
    palette,
    options: {scale: 'sqrt', interpolation: 'linear', clamp: true, noDataColor: NO_DATA}
  };
  await fixture.run(threshold);
  const start = performance.now();
  const thresholdResult = await fixture.run(threshold);
  const thresholdMilliseconds = performance.now() - start;
  expectParityQuick(thresholdResult, getOracle(config, threshold), true);
  const linearStart = performance.now();
  const linearResult = await fixture.run(linear);
  const linearMilliseconds = performance.now() - linearStart;
  expectParityQuick(linearResult, getOracle(config, linear), false);
  console.log(
    `GPUColorScale 1M rows: threshold ${thresholdMilliseconds.toFixed(1)} ms, ` +
      `sqrt+linear ${linearMilliseconds.toFixed(1)} ms (submit + 3 readbacks)`
  );
  fixture.destroy();
}, 60000);

/** Whole-array comparison for the large case; boundary rows are tolerated unless `exact`. */
function expectParityQuick(actual: FrameResult, expected: ColorScaleOracleResult, exact: boolean) {
  let mismatches = 0;
  let ambiguous = 0;
  for (let row = 0; row < expected.classIndices.length; row++) {
    if (expected.ambiguous[row] && !exact) {
      ambiguous++;
      continue;
    }
    if (actual.classIndices[row] !== expected.classIndices[row]) {
      mismatches++;
    }
  }
  expect(mismatches).toBe(0);
  const total = Array.from(actual.classCounts).reduce((sum, count) => sum + count, 0);
  const expectedTotal = Array.from(expected.classCounts).reduce((sum, count) => sum + count, 0);
  expect(Math.abs(total - expectedTotal)).toBeLessThanOrEqual(ambiguous);
  if (exact) {
    expect(Array.from(actual.classCounts)).toEqual(Array.from(expected.classCounts));
    expect(Array.from(actual.colors.slice(0, 50000))).toEqual(
      Array.from(expected.colors.slice(0, 50000))
    );
  }
}
