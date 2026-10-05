// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUEmergingHotSpotParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH,
  type GPUEmergingHotSpotParameters
} from '../../../src/map-graphs/emerging-hot-spots';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeEmergingHotSpotCells,
  computeEmergingHotSpotMoments,
  computeSpaceTimeGiStar,
  createDesignedCube,
  createSeededRandom,
  type EmergingHotSpotCube
} from './emerging-hot-spots-oracle';

const GARBAGE = 0x7f7f7f7f;
const Z_ABSOLUTE_TOLERANCE = 2e-3;
const Z_RELATIVE_TOLERANCE = 1e-3;

type Readback = {
  giZScores: number[];
  giZScoreBits: number[];
  trendZ: number[];
  trendP: number[];
  trendS: number[];
  category: number[];
  hotSliceCount: number[];
  coldSliceCount: number[];
  globalStatistics: number[];
};

type Harness = {
  readonly buildCount: number;
  run(parameters: GPUEmergingHotSpotParameters): Promise<Readback>;
  writeValues(values: Float32Array | Uint32Array): void;
  destroy(): void;
};

function createHarness(
  device: Device,
  cube: EmergingHotSpotCube & {values: Float32Array | Uint32Array},
  options: {maximumRadius?: number} = {}
): Harness {
  const {gridWidth, gridHeight, sliceCount} = cube;
  const cellCount = gridWidth * gridHeight;
  const binCount = cellCount * sliceCount;
  const format = cube.values instanceof Uint32Array ? 'uint32' : 'float32';
  const valuesBuffer = createInputBuffer(device, cube.values);
  const maskBuffer = cube.mask && createInputBuffer(device, Uint32Array.from(cube.mask));
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'emerging-hot-spot-parameters',
    format: 'float32',
    length: GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH
  });
  const outputs: Record<
    | 'giZScores'
    | 'trendZ'
    | 'trendP'
    | 'trendS'
    | 'category'
    | 'hotSliceCount'
    | 'coldSliceCount'
    | 'globalStatistics',
    Buffer
  > = {
    giZScores: createOutputBuffer(device, binCount),
    trendZ: createOutputBuffer(device, cellCount),
    trendP: createOutputBuffer(device, cellCount),
    trendS: createOutputBuffer(device, cellCount),
    category: createOutputBuffer(device, cellCount),
    hotSliceCount: createOutputBuffer(device, cellCount),
    coldSliceCount: createOutputBuffer(device, cellCount),
    globalStatistics: createOutputBuffer(device, 4)
  };
  const graph = new GPUCommandGraph(device, {id: 'emerging-hot-spots-test'});
  const recipe = new GPUEmergingHotSpots({
    values:
      format === 'uint32'
        ? importGraphBuffer(graph, 'values', valuesBuffer, 'uint32', binCount)
        : importGraphBuffer(graph, 'values', valuesBuffer, 'float32', binCount),
    gridWidth,
    gridHeight,
    sliceCount,
    maximumRadius: options.maximumRadius,
    mask: maskBuffer && importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', cellCount),
    parameters: parameterBuffer.importToGraph(graph),
    giZScores: importGraphBuffer(graph, 'gi-z', outputs.giZScores, 'float32', binCount),
    trendZ: importGraphBuffer(graph, 'trend-z', outputs.trendZ, 'float32', cellCount),
    trendP: importGraphBuffer(graph, 'trend-p', outputs.trendP, 'float32', cellCount),
    trendS: importGraphBuffer(graph, 'trend-s', outputs.trendS, 'sint32', cellCount),
    category: importGraphBuffer(graph, 'category', outputs.category, 'uint32', cellCount),
    hotSliceCount: importGraphBuffer(graph, 'hot', outputs.hotSliceCount, 'uint32', cellCount),
    coldSliceCount: importGraphBuffer(graph, 'cold', outputs.coldSliceCount, 'uint32', cellCount),
    globalStatistics: importGraphBuffer(graph, 'stats', outputs.globalStatistics, 'float32', 4)
  });
  let buildCount = 0;
  const getCommandNodes = recipe.getCommandNodes.bind(recipe);
  recipe.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof recipe.getCommandNodes;
  graph.add(recipe);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return buildCount;
    },
    async run(parameters) {
      parameterBuffer.write(getGPUEmergingHotSpotParameterValues(parameters));
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const giZScoreBits = await readUint32(outputs.giZScores, binCount);
      const trendSBits = await readUint32(outputs.trendS, cellCount);
      return {
        giZScores: Array.from(new Float32Array(Uint32Array.from(giZScoreBits).buffer)),
        giZScoreBits,
        trendZ: await readFloat32(outputs.trendZ, cellCount),
        trendP: await readFloat32(outputs.trendP, cellCount),
        trendS: Array.from(new Int32Array(Uint32Array.from(trendSBits).buffer)),
        category: await readUint32(outputs.category, cellCount),
        hotSliceCount: await readUint32(outputs.hotSliceCount, cellCount),
        coldSliceCount: await readUint32(outputs.coldSliceCount, cellCount),
        globalStatistics: await readFloat32(outputs.globalStatistics, 4)
      };
    },
    writeValues(values) {
      valuesBuffer.write(values);
    },
    destroy() {
      compiled.destroy();
      valuesBuffer.destroy();
      maskBuffer?.destroy();
      parameterBuffer.destroy();
      for (const buffer of Object.values(outputs)) {
        buffer.destroy();
      }
    }
  };
}

function expectZScoresClose(actual: number[], expected: ArrayLike<number>, label: string): void {
  for (let bin = 0; bin < actual.length; bin++) {
    if (Number.isNaN(expected[bin])) {
      if (!Number.isNaN(actual[bin])) {
        throw new Error(`${label}: bin ${bin} z ${actual[bin]} should be NaN`);
      }
    } else if (
      !(
        Math.abs(actual[bin] - expected[bin]) <=
        Z_ABSOLUTE_TOLERANCE + Z_RELATIVE_TOLERANCE * Math.abs(expected[bin])
      )
    ) {
      throw new Error(`${label}: bin ${bin} z ${actual[bin]} != oracle ${expected[bin]}`);
    }
  }
}

/** Passes 3 and 4 are exact given the GPU's own z-scores, away from the trend p-value bound. */
function expectCellsMatchOracle(
  result: Readback,
  cube: EmergingHotSpotCube,
  parameters: GPUEmergingHotSpotParameters,
  label: string
): void {
  const packed = getGPUEmergingHotSpotParameterValues(parameters);
  const oracle = computeEmergingHotSpotCells(result.giZScores, cube, packed);
  expect(result.trendS, `${label} trendS`).toEqual(oracle.trendS);
  expect(result.hotSliceCount, `${label} hot`).toEqual(oracle.hotSliceCount);
  expect(result.coldSliceCount, `${label} cold`).toEqual(oracle.coldSliceCount);
  const level = packed[3];
  for (const [cell, expected] of oracle.category.entries()) {
    expect(
      Math.abs(result.trendZ[cell] - oracle.trendZ[cell]),
      `${label} trendZ ${cell}`
    ).toBeLessThan(1e-4 + 1e-4 * Math.abs(oracle.trendZ[cell]));
    expect(
      Math.abs(result.trendP[cell] - oracle.trendP[cell]),
      `${label} trendP ${cell}`
    ).toBeLessThan(1e-5 + 1e-3 * oracle.trendP[cell]);
    if (Math.abs(oracle.trendP[cell] - level) > 0.02 * level) {
      expect(result.category[cell], `${label} category ${cell}`).toBe(expected);
    }
  }
}

it('GPUEmergingHotSpots classifies all 17 categories on the designed cube', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {cube, designed, expectedCategories} = createDesignedCube();
  const harness = createHarness(device, cube);
  try {
    const parameters = {radius: 0, temporalWindow: 0};
    const result = await harness.run(parameters);
    const packed = getGPUEmergingHotSpotParameterValues(parameters);
    expectZScoresClose(result.giZScores, computeSpaceTimeGiStar(cube, packed), 'designed');
    const moments = computeEmergingHotSpotMoments(cube);
    expect(result.globalStatistics[0]).toBe(moments.count);
    expect(Math.abs(result.globalStatistics[1] - moments.mean)).toBeLessThan(1e-3);
    expect(Math.abs(result.globalStatistics[3] - moments.standardDeviation)).toBeLessThan(1e-3);
    expectCellsMatchOracle(result, cube, parameters, 'designed');
    for (const [cell, expected] of expectedCategories.entries()) {
      expect(result.category[cell], `cell ${cell} ${designed[cell]?.name}`).toBe(expected);
    }
    expect(new Set(result.category.slice(0, designed.length)).size).toBe(17);
    // Masked and all-missing cells carry no pattern, and their z is NaN.
    expect(
      result.giZScores.slice(designed.length * 20, (designed.length + 2) * 20).every(Number.isNaN)
    ).toBe(true);
    expect(result.category[designed.length]).toBe(0);
    expect(result.category[designed.length + 1]).toBe(0);
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

function createSpaceTimeCube(seed: number): EmergingHotSpotCube & {values: Float32Array} {
  const gridWidth = 10;
  const gridHeight = 7;
  const sliceCount = 24;
  const random = createSeededRandom(seed);
  const values = new Float32Array(gridWidth * gridHeight * sliceCount);
  for (let cell = 0; cell < gridWidth * gridHeight; cell++) {
    const column = cell % gridWidth;
    const row = Math.floor(cell / gridWidth);
    for (let slice = 0; slice < sliceCount; slice++) {
      // A hot blob that drifts and grows over time, plus a cold corner, plus noise.
      const hotDistance = Math.hypot(column - (2 + slice / 6), row - 3);
      const coldDistance = Math.hypot(column - 9, row - 6);
      values[cell * sliceCount + slice] =
        random() * 2 + (hotDistance < 2.5 ? 6 + slice / 8 : 0) - (coldDistance < 2.5 ? 8 : 0);
    }
  }
  // Missing bins, and a masked cell.
  for (const bin of [5, 77, 301, 640, 1111]) {
    values[bin] = NaN;
  }
  const mask = new Uint32Array(gridWidth * gridHeight).fill(1);
  mask[13] = 0;
  return {gridWidth, gridHeight, sliceCount, values, mask};
}

it('GPUEmergingHotSpots matches the space-time oracle across per-frame parameters without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cube = createSpaceTimeCube(5);
  const harness = createHarness(device, cube, {maximumRadius: 4});
  try {
    const frames: GPUEmergingHotSpotParameters[] = [
      {radius: 0, temporalWindow: 0},
      {radius: 1, temporalWindow: 0},
      {radius: 1.5, temporalWindow: 2},
      {radius: 2.3, temporalWindow: 5, confidenceLevel: 0.95},
      {
        radius: 3,
        temporalWindow: 23,
        confidenceLevel: 0.99,
        persistentFraction: 0.75
      },
      // Radius above maximumRadius is clamped; a window above sliceCount - 1 is clamped.
      {
        radius: 9,
        temporalWindow: 40,
        criticalZ: 1.2,
        trendSignificanceLevel: 0.2
      }
    ];
    const categoriesSeen = new Set<number>();
    for (const frame of frames) {
      const packed = getGPUEmergingHotSpotParameterValues(frame);
      const result = await harness.run(frame);
      const oracleZ = computeSpaceTimeGiStar(cube, packed, 4);
      expectZScoresClose(result.giZScores, oracleZ, JSON.stringify(frame));
      expectCellsMatchOracle(result, cube, frame, JSON.stringify(frame));
      for (const category of result.category) {
        categoriesSeen.add(category);
      }
      expect(result.globalStatistics[0]).toBe(computeEmergingHotSpotMoments(cube).count);
      // Repeated encodings are bitwise identical.
      expect((await harness.run(frame)).giZScoreBits).toEqual(result.giZScoreBits);
    }
    expect(categoriesSeen.size).toBeGreaterThan(3);
    // New values in the same compiled graph.
    const shifted = Float32Array.from(cube.values, (value, bin) => value * 0.5 + (bin % 5));
    harness.writeValues(shifted);
    const shiftedCube = {...cube, values: shifted};
    const frame = {radius: 2, temporalWindow: 3};
    const result = await harness.run(frame);
    expectZScoresClose(
      result.giZScores,
      computeSpaceTimeGiStar(shiftedCube, getGPUEmergingHotSpotParameterValues(frame)),
      'shifted'
    );
    expectCellsMatchOracle(result, shiftedCube, frame, 'shifted');
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUEmergingHotSpots accepts uint32 counts like GPUTemporalReduction output', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createSeededRandom(3);
  const gridWidth = 6;
  const gridHeight = 5;
  const sliceCount = 12;
  const counts = Uint32Array.from({length: gridWidth * gridHeight * sliceCount}, (_, bin) => {
    const slice = bin % sliceCount;
    const cell = Math.floor(bin / sliceCount);
    return Math.floor(random() * 3) + (cell < 8 ? slice * 2 : 0);
  });
  const cube = {gridWidth, gridHeight, sliceCount, values: counts};
  const harness = createHarness(device, cube);
  try {
    const frame = {radius: 1, temporalWindow: 1};
    const result = await harness.run(frame);
    expectZScoresClose(
      result.giZScores,
      computeSpaceTimeGiStar(cube, getGPUEmergingHotSpotParameterValues(frame)),
      'counts'
    );
    expectCellsMatchOracle(result, cube, frame, 'counts');
    expect(result.globalStatistics[0]).toBe(counts.length);
    expect(result.category.some(category => category !== 0)).toBe(true);
  } finally {
    harness.destroy();
  }
}, 60000);

it('GPUEmergingHotSpots writes NaN z for degenerate cubes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = new Float32Array(2 * 2 * 4).fill(3);
  const harness = createHarness(device, {
    gridWidth: 2,
    gridHeight: 2,
    sliceCount: 4,
    values
  });
  try {
    // A constant cube has zero deviation: no z, no trend, no pattern.
    const constant = await harness.run({radius: 1, temporalWindow: 1});
    expect(constant.giZScores.every(Number.isNaN)).toBe(true);
    expect(constant.category.every(category => category === 0)).toBe(true);
    expect(constant.trendS.every(statistic => statistic === 0)).toBe(true);
    expect(constant.trendP.every(p => p === 1)).toBe(true);
    // Every bin inside one neighborhood (k = n) is undefined.
    harness.writeValues(Float32Array.from(values, (_, bin) => bin));
    const everything = await harness.run({radius: 4, temporalWindow: 3});
    expect(everything.giZScores.filter(Number.isFinite).length).toBeLessThan(values.length);
  } finally {
    harness.destroy();
  }
}, 60000);
