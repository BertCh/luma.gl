// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPURasterStretchParameterValues,
  GPU_RASTER_STRETCH_PARAMETER_LENGTH,
  GPURasterStretch,
  type GPURasterStretchSettings
} from '../../../src/map-graphs/raster-stretch';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {createRandom} from '../raster-algebra/raster-algebra-test-utils';
import {
  computeRasterStretchLutOnCPU,
  computeRasterStretchStatistics,
  getPaletteColorOnCPU,
  packColor,
  stretchRasterOnCPU,
  type RasterStretchScene
} from './raster-stretch-oracle';

type OutputName = 'stretched' | 'colors' | 'lut' | 'lutColors' | 'histogram' | 'statistics';
const ALL_OUTPUTS: OutputName[] = [
  'stretched',
  'colors',
  'lut',
  'lutColors',
  'histogram',
  'statistics'
];

type HarnessOptions = {
  scene: RasterStretchScene;
  palette?: Uint32Array;
  outputs?: OutputName[];
};

/** Compiles one graph; `run` rewrites parameters, mask and palette and returns the readbacks. */
function createHarness(device: Device, options: HarnessOptions) {
  const {scene} = options;
  const {width, height, binCount, lutSize} = scene;
  const cellCount = width * height;
  const outputs =
    options.outputs ??
    ALL_OUTPUTS.filter(name => options.palette || (name !== 'colors' && name !== 'lutColors'));
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'stretch-parameters',
    format: 'float32',
    length: GPU_RASTER_STRETCH_PARAMETER_LENGTH
  });
  const valuesBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'stretch-values',
    format: 'float32',
    length: cellCount,
    values: scene.values
  });
  const maskBuffer = scene.regionMask
    ? new GPUMapGraphParameterBuffer(device, {
        id: 'stretch-mask',
        format: 'uint32',
        length: cellCount,
        values: scene.regionMask
      })
    : undefined;
  const paletteBuffer = options.palette
    ? new GPUMapGraphParameterBuffer(device, {
        id: 'stretch-palette',
        format: 'uint32',
        length: options.palette.length,
        values: options.palette
      })
    : undefined;
  const validityBuffer = scene.validity ? createInputBuffer(device, scene.validity) : undefined;
  const lengths: Record<OutputName, number> = {
    stretched: cellCount,
    colors: cellCount,
    lut: lutSize,
    lutColors: lutSize,
    histogram: binCount,
    statistics: 8
  };
  const formats: Record<OutputName, 'float32' | 'uint32'> = {
    stretched: 'float32',
    colors: 'uint32',
    lut: 'float32',
    lutColors: 'uint32',
    histogram: 'uint32',
    statistics: 'float32'
  };
  const outputBuffers = Object.fromEntries(
    outputs.map(name => [name, createOutputBuffer(device, lengths[name])])
  ) as Record<OutputName, ReturnType<typeof createOutputBuffer>>;
  const graph = new GPUCommandGraph(device, {id: 'stretch-graph'});
  graph.add(
    new GPURasterStretch({
      id: 'stretch',
      values: valuesBuffer.importToGraph(graph),
      width,
      height,
      noDataValue: scene.noDataValue,
      validity: validityBuffer
        ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount)
        : undefined,
      regionMask: maskBuffer?.importToGraph(graph),
      parameters: parameterBuffer.importToGraph(graph),
      palette: paletteBuffer?.importToGraph(graph),
      binCount,
      lutSize,
      output: Object.fromEntries(
        outputs.map(name => [
          name,
          importGraphBuffer(graph, `out-${name}`, outputBuffers[name], formats[name], lengths[name])
        ])
      )
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(
      settings: GPURasterStretchSettings,
      update?: {mask?: Uint32Array; palette?: Uint32Array}
    ) {
      if (update?.mask) {
        maskBuffer!.write(update.mask);
      }
      if (update?.palette) {
        paletteBuffer!.write(update.palette);
      }
      const parameters = getGPURasterStretchParameterValues(settings);
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const result: Partial<Record<OutputName, number[]>> = {};
      for (const name of outputs) {
        result[name] =
          formats[name] === 'float32'
            ? await readFloat32(outputBuffers[name], lengths[name])
            : await readUint32(outputBuffers[name], lengths[name]);
      }
      return {parameters, result};
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      valuesBuffer.destroy();
      maskBuffer?.destroy();
      paletteBuffer?.destroy();
      validityBuffer?.destroy();
      for (const buffer of Object.values(outputBuffers)) {
        buffer.destroy();
      }
    }
  };
}

type Harness = ReturnType<typeof createHarness>;

function expectClose(actual: number, expected: number, tolerance: number, label: string): void {
  if (Number.isNaN(expected) || Number.isNaN(actual)) {
    expect(Number.isNaN(actual), `${label} nodata (gpu ${actual}, cpu ${expected})`).toBe(
      Number.isNaN(expected)
    );
    return;
  }
  expect(Math.abs(actual - expected), `${label} gpu ${actual} cpu ${expected}`).toBeLessThanOrEqual(
    tolerance
  );
}

type CheckOptions = {
  palette?: Uint32Array;
  /** Allowed L1 distance between GPU and CPU histograms (0 when bin arithmetic is exact). */
  histogramTolerance?: number;
  /** Tolerance of lo/hi in bin widths; the GPU divides in f32. */
  boundTolerance?: number;
  /** Absolute tolerance of the stretched values and the LUT. */
  valueTolerance?: number;
};

/** Runs one frame and compares every output with the CPU oracle. */
async function checkFrame(
  harness: Harness,
  scene: RasterStretchScene,
  settings: GPURasterStretchSettings,
  options: CheckOptions = {},
  update?: {mask?: Uint32Array; palette?: Uint32Array}
) {
  const {parameters, result} = await harness.run(settings, update);
  if (update?.mask) {
    // The GPU keeps the written mask for later frames.
    scene.regionMask = update.mask;
  }
  const live = scene;
  const stats = computeRasterStretchStatistics(live, parameters);
  const histogramTolerance = options.histogramTolerance ?? 0;
  const histogram = result.histogram!;
  let distance = 0;
  for (let bin = 0; bin < scene.binCount; bin++) {
    distance += Math.abs(histogram[bin] - stats.histogram[bin]);
  }
  expect(distance, 'histogram L1 distance').toBeLessThanOrEqual(histogramTolerance);
  if (histogramTolerance === 0) {
    expect(histogram).toEqual(Array.from(stats.histogram));
  }
  const statistics = result.statistics!;
  const nan = Number.NaN;
  expect(statistics[4], 'validCount').toBe(stats.validCount);
  expect(statistics[6]).toBe(0);
  expect(statistics[7]).toBe(0);
  expectClose(statistics[0], stats.isEmpty ? nan : stats.domainMin, 0, 'domainMin');
  expectClose(statistics[1], stats.isEmpty ? nan : stats.domainMax, 0, 'domainMax');
  expectClose(
    statistics[5],
    stats.isEmpty ? nan : stats.binWidth,
    1e-6 * stats.binWidth,
    'binWidth'
  );
  const boundTolerance = (options.boundTolerance ?? 1e-4) * Math.max(stats.binWidth, 1e-12);
  expectClose(statistics[2], stats.isEmpty ? nan : stats.lo, boundTolerance, 'lo');
  expectClose(statistics[3], stats.isEmpty ? nan : stats.hi, boundTolerance, 'hi');
  // Compare the per-cell maps against the oracle evaluated at the GPU's own bounds, so that f32
  // division in the bound search does not leak into the per-cell tolerance.
  const gpuStats = {
    ...stats,
    lo: stats.isEmpty ? 0 : statistics[2],
    hi: stats.isEmpty ? 0 : statistics[3]
  };
  const valueTolerance = options.valueTolerance ?? 2e-5;
  const expectedStretched = stretchRasterOnCPU(live, parameters, gpuStats);
  const stretched = result.stretched!;
  for (let cell = 0; cell < expectedStretched.length; cell++) {
    expectClose(stretched[cell], expectedStretched[cell], valueTolerance, `stretched[${cell}]`);
    if (!Number.isNaN(stretched[cell])) {
      expect(stretched[cell]).toBeGreaterThanOrEqual(0);
      expect(stretched[cell]).toBeLessThanOrEqual(1);
    }
  }
  const expectedLut = computeRasterStretchLutOnCPU(live, parameters, gpuStats);
  const lut = result.lut!;
  for (let index = 0; index < expectedLut.length; index++) {
    expectClose(lut[index], expectedLut[index], valueTolerance, `lut[${index}]`);
  }
  if (options.palette) {
    const palette = update?.palette ?? options.palette;
    const linear = settings.paletteInterpolation === 'linear';
    const compareColors = (actual: number[], source: number[], label: string) => {
      for (let index = 0; index < source.length; index++) {
        const expected = getPaletteColorOnCPU(source[index], palette, linear);
        if (linear) {
          for (let channel = 0; channel < 4; channel++) {
            const shift = 8 * channel;
            const delta = Math.abs(
              ((actual[index] >>> shift) & 255) - ((expected >>> shift) & 255)
            );
            expect(delta, `${label}[${index}] channel ${channel}`).toBeLessThanOrEqual(1);
          }
        } else {
          expect(actual[index], `${label}[${index}]`).toBe(expected);
        }
      }
    };
    // Colors derive from the GPU's own t, so nearest lookups are exact.
    compareColors(result.colors!, stretched, 'colors');
    compareColors(result.lutColors!, lut, 'lutColors');
  }
  return {parameters, result, stats, scene: live};
}

/** Integers in [0, 1024] with both endpoints present, NaN, sentinel and zero-validity cells. */
function createIntegerScene(seed: number, width: number, height: number): RasterStretchScene {
  const random = createRandom(seed);
  const cellCount = width * height;
  const values = new Float32Array(cellCount);
  const validity = new Uint32Array(cellCount).fill(1);
  for (let cell = 0; cell < cellCount; cell++) {
    // Skewed so that equalization differs from a linear stretch.
    values[cell] = Math.round(random() ** 2 * 1024);
    const roll = random();
    if (roll < 0.03) {
      values[cell] = NaN;
    } else if (roll < 0.06) {
      values[cell] = -9999;
    } else if (roll < 0.09) {
      validity[cell] = 0;
    } else if (roll < 0.1) {
      values[cell] = roll < 0.095 ? Infinity : -Infinity;
    }
  }
  values[3] = 0;
  values[4] = 1024;
  validity[3] = 1;
  validity[4] = 1;
  return {values, width, height, noDataValue: -9999, validity, binCount: 1024, lutSize: 64};
}

const PALETTE = Uint32Array.from([
  packColor(0, 0, 128, 255),
  packColor(0, 200, 255, 255),
  packColor(255, 255, 0, 200),
  packColor(255, 0, 0, 255),
  packColor(120, 10, 10, 255)
]);

it('GPURasterStretch matches the CPU oracle across modes, domains, windows and masks without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createIntegerScene(5, 37, 29);
  scene.regionMask = new Uint32Array(37 * 29).map((_, cell) => ((cell * 7) % 5 === 0 ? 0 : 1));
  const harness = createHarness(device, {scene, palette: PALETTE});
  const options = {palette: PALETTE};
  const frames: GPURasterStretchSettings[] = [
    {mode: 'linear'},
    {mode: 'linear', domain: [0, 512], paletteInterpolation: 'linear'},
    {mode: 'percentile', percentiles: [2, 98]},
    {mode: 'percentile', percentiles: [0, 100], gamma: 0.5},
    {
      mode: 'percentile',
      percentiles: [10, 90],
      gamma: 2.2,
      sigmoidContrast: 6,
      sigmoidMidpoint: 0.4
    },
    {mode: 'equalize', paletteInterpolation: 'linear'},
    {mode: 'equalize', domain: [0, 512], sigmoidContrast: 4},
    {mode: 'linear', window: [10, 5, 30, 20]},
    {mode: 'equalize', window: [10, 5, 30, 20], domain: [0, 1024]},
    {mode: 'percentile', window: [-5, -5, 1000, 1000], percentiles: [5, 95], gamma: 1.5}
  ];
  for (const [frameIndex, settings] of frames.entries()) {
    // Mask contents change between frames; frame 0 keeps the initial mask.
    const update =
      frameIndex === 4
        ? {mask: new Uint32Array(37 * 29).map((_, cell) => (cell % 3 === 0 ? 1 : 0))}
        : undefined;
    const {stats} = await checkFrame(harness, scene, settings, options, update);
    expect(stats.validCount, `frame ${frameIndex} has data`).toBeGreaterThan(50);
  }
  // Palette contents change per frame too.
  await checkFrame(harness, scene, {mode: 'percentile'}, options, {
    palette: Uint32Array.from([
      packColor(1, 2, 3, 4),
      packColor(250, 251, 252, 253),
      0,
      7,
      0xffffffff
    ])
  });
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

it('GPURasterStretch automatic domain is the exact minimum and maximum, percentile bounds honor the bin error', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(21);
  const width = 64;
  const height = 48;
  const values = new Float32Array(width * height);
  for (let cell = 0; cell < values.length; cell++) {
    values[cell] = (random() - 0.3) * 50 + random() * random() * 7;
  }
  values[11] = NaN;
  const scene: RasterStretchScene = {values, width, height, binCount: 1024, lutSize: 256};
  const harness = createHarness(device, {scene});
  // The f32 bin arithmetic is not exact for this domain: allow cells near bin edges to differ.
  const {stats, result} = await checkFrame(
    harness,
    scene,
    {mode: 'percentile', percentiles: [2, 98]},
    {histogramTolerance: 24, boundTolerance: 3}
  );
  const finite = Array.from(values)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  expect(stats.domainMin).toBe(finite[0]);
  expect(stats.domainMax).toBe(finite[finite.length - 1]);
  // lo/hi are within one bin width of the exact percentile of the data.
  const exactLow = finite[Math.floor(0.02 * finite.length)];
  const exactHigh = finite[Math.min(Math.ceil(0.98 * finite.length) - 1, finite.length - 1)];
  expect(Math.abs(result.statistics![2] - exactLow)).toBeLessThanOrEqual(1.5 * stats.binWidth);
  expect(Math.abs(result.statistics![3] - exactHigh)).toBeLessThanOrEqual(1.5 * stats.binWidth);
  await checkFrame(
    harness,
    scene,
    {mode: 'equalize'},
    {histogramTolerance: 24, valueTolerance: 2e-3}
  );
  harness.destroy();
});

it('GPURasterStretch defines all-nodata, empty-window and constant rasters', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // All nodata: NaN everywhere, zero counts, no crash.
  const nodata: RasterStretchScene = {
    values: new Float32Array(20).fill(NaN),
    width: 5,
    height: 4,
    binCount: 16,
    lutSize: 8
  };
  const nodataHarness = createHarness(device, {scene: nodata, palette: PALETTE});
  for (const mode of ['linear', 'percentile', 'equalize'] as const) {
    const {result} = await nodataHarness.run({mode});
    expect(result.stretched!.every(Number.isNaN)).toBe(true);
    expect(result.colors!.every(color => color === 0)).toBe(true);
    expect(result.histogram!.every(count => count === 0)).toBe(true);
    expect(result.statistics![4]).toBe(0);
    expect([0, 1, 2, 3, 5].every(index => Number.isNaN(result.statistics![index]))).toBe(true);
  }
  // An explicit domain stays defined without data.
  const explicit = await nodataHarness.run({domain: [2, 6]});
  expect(Array.from(explicit.result.statistics!)).toEqual([2, 6, 2, 6, 0, 0.25, 0, 0]);
  expect(nodataHarness.rebuildCount).toBe(0);
  nodataHarness.destroy();

  // Constant raster: lo == hi, every cell maps to 0.5, histogram holds all cells in bin 0.
  const constant: RasterStretchScene = {
    values: new Float32Array(30).fill(7),
    width: 6,
    height: 5,
    binCount: 16,
    lutSize: 8
  };
  const constantHarness = createHarness(device, {scene: constant, palette: PALETTE});
  for (const mode of ['linear', 'percentile', 'equalize'] as const) {
    const {result} = await constantHarness.run({mode, gamma: 2, sigmoidContrast: 5});
    expect(Array.from(result.statistics!)).toEqual([7, 7, 7, 7, 30, 0, 0, 0]);
    expect(result.stretched!.every(value => value === 0.5)).toBe(true);
    expect(result.histogram![0]).toBe(30);
    expect(result.lut!.every(value => value === 0.5)).toBe(true);
    // Nearest lookup of t = 0.5 in a five-color palette is index 2.
    expect(result.colors!.every(color => color === PALETTE[2])).toBe(true);
  }
  // A wider explicit domain makes the constant a normal value.
  const wide = await constantHarness.run({domain: [0, 10]});
  expect(wide.result.stretched!.every(value => Math.abs(value - 0.7) < 1e-6)).toBe(true);
  // Values outside an explicit domain clamp.
  const clamped = await constantHarness.run({domain: [8, 9]});
  expect(clamped.result.stretched!.every(value => value === 0)).toBe(true);
  expect(clamped.result.histogram!.every(count => count === 0)).toBe(true);
  expect(clamped.result.statistics![4]).toBe(30);
  expect(constantHarness.rebuildCount).toBe(0);
  constantHarness.destroy();

  // A window with no cells reports empty statistics but the apply step still covers every cell:
  // valid cells fall back to the degenerate mapping against lo = hi = 0.
  const mixed = createIntegerScene(8, 12, 9);
  const mixedHarness = createHarness(device, {scene: mixed});
  const {result: empty} = await mixedHarness.run({window: [3, 3, 3, 3]});
  expect(empty.statistics![4]).toBe(0);
  expect(Number.isNaN(empty.statistics![2])).toBe(true);
  for (let cell = 0; cell < mixed.values.length; cell++) {
    const value = mixed.values[cell];
    const isNoData = Number.isNaN(value) || value === -9999 || mixed.validity![cell] === 0;
    const expected = isNoData ? NaN : value < 0 ? 0 : value > 0 ? 1 : 0.5;
    expectClose(empty.stretched![cell], expected, 0, `stretched[${cell}]`);
  }
  mixedHarness.destroy();
});

it('GPURasterStretch supports output subsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createIntegerScene(3, 20, 10);
  const histogramOnly = createHarness(device, {scene, outputs: ['histogram', 'statistics']});
  const {parameters, result} = await histogramOnly.run({mode: 'percentile'});
  expect(result.histogram).toEqual(
    Array.from(computeRasterStretchStatistics(scene, parameters).histogram)
  );
  histogramOnly.destroy();
  // Only lutColors (and the palette); then only colors.
  const lutOnly = createHarness(device, {scene, palette: PALETTE, outputs: ['lutColors']});
  const lutResult = await lutOnly.run({mode: 'equalize', paletteInterpolation: 'linear'});
  expect(lutResult.result.lutColors!.length).toBe(scene.lutSize);
  expect(lutResult.result.lutColors!.some(color => color !== 0)).toBe(true);
  lutOnly.destroy();
  const colorsOnly = createHarness(device, {scene, palette: PALETTE, outputs: ['colors']});
  const colorResult = await colorsOnly.run({});
  expect(colorResult.result.colors!.some(color => color !== 0)).toBe(true);
  colorsOnly.destroy();
});
