// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights';
import type {GPUMultipleTestingMethod} from '../../../src/gpu-spatial-analysis/contracts/inference';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUEmergingHotSpotParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOT_CATEGORIES,
  GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH,
  type GPUEmergingHotSpotParameters
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchParameters
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {createDistanceBandWeights} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeAdjustedBinPValues,
  computeBinPValues,
  computeEmergingHotSpotCells,
  computeEmergingHotSpotMoments,
  computeSpaceTimeGiStar,
  computeSpaceTimeGiStarWeighted,
  createDesignedCube,
  createSeededRandom,
  MULTIPLE_TESTING_ADJUSTED,
  MULTIPLE_TESTING_SIGNIFICANT,
  MULTIPLE_TESTING_TOP_BINS,
  MULTIPLE_TESTING_VALUES,
  type EmergingHotSpotCube
} from './emerging-hot-spots-oracle';

const GARBAGE = 0x7f7f7f7f;
const Z_ABSOLUTE_TOLERANCE = 2e-3;
const Z_RELATIVE_TOLERANCE = 1e-3;

type Readback = {
  giZScores: number[];
  giZScoreBits: number[];
  pValues: number[];
  adjustedPValues: number[];
  lastValidSlice: number[];
  trendZ: number[];
  trendP: number[];
  trendS: number[];
  category: number[];
  hotSliceCount: number[];
  coldSliceCount: number[];
  globalStatistics: number[];
  /** Weights mode only: the CSR the contributor read (neighbors/weights untrimmed). */
  csr: {offsets: number[]; neighbors: number[]; weights: number[]};
};

/** Weights mode: a hand-built CSR, or a `GPUNeighborSearch` radius search in the same graph. */
type WeightsSource =
  | {kind: 'csr'; offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array}
  | {
      kind: 'neighbor-search';
      positions: Float32Array;
      parameters: GPUNeighborSearchParameters;
      capacity: number;
    };

type Harness = {
  readonly buildCount: number;
  run(
    parameters: Omit<GPUEmergingHotSpotParameters, 'multipleTesting'>,
    neighborParameters?: GPUNeighborSearchParameters
  ): Promise<Readback>;
  writeValues(values: Float32Array | Uint32Array): void;
  destroy(): void;
};

function createHarness(
  device: Device,
  cube: EmergingHotSpotCube & {values: Float32Array | Uint32Array},
  options: {
    maximumRadius?: number;
    weights?: WeightsSource;
    selfWeight?: number;
    multipleTesting?: GPUMultipleTestingMethod;
    finalSlice?: 'require' | 'last-valid';
  } = {}
): Harness {
  // The node default, and the method the parameters are validated against.
  const multipleTesting = options.multipleTesting ?? 'benjamini-hochberg';
  const {gridWidth, gridHeight, sliceCount} = cube;
  const cellCount = gridWidth * gridHeight;
  const binCount = cellCount * sliceCount;
  const source = options.weights;
  const capacity = source ? (source.kind === 'csr' ? source.neighbors.length : source.capacity) : 1;
  const format = cube.values instanceof Uint32Array ? 'uint32' : 'float32';
  const valuesBuffer = createInputBuffer(device, cube.values);
  const maskBuffer = cube.mask && createInputBuffer(device, Uint32Array.from(cube.mask));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'emerging-hot-spot-parameters',
    format: 'float32',
    length: GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH
  });
  const outputs: Record<
    | 'giZScores'
    | 'offsets'
    | 'neighbors'
    | 'spatialWeights'
    | 'overflow'
    | 'trendZ'
    | 'trendP'
    | 'trendS'
    | 'category'
    | 'hotSliceCount'
    | 'coldSliceCount'
    | 'globalStatistics'
    | 'pValues'
    | 'adjustedPValues'
    | 'lastValidSlice',
    Buffer
  > = {
    giZScores: createOutputBuffer(device, binCount),
    offsets: createOutputBuffer(device, cellCount + 1),
    neighbors: createOutputBuffer(device, capacity),
    spatialWeights: createOutputBuffer(device, capacity),
    overflow: createOutputBuffer(device, 1),
    trendZ: createOutputBuffer(device, cellCount),
    trendP: createOutputBuffer(device, cellCount),
    trendS: createOutputBuffer(device, cellCount),
    category: createOutputBuffer(device, cellCount),
    hotSliceCount: createOutputBuffer(device, cellCount),
    coldSliceCount: createOutputBuffer(device, cellCount),
    globalStatistics: createOutputBuffer(device, 4),
    pValues: createOutputBuffer(device, binCount),
    adjustedPValues: createOutputBuffer(device, binCount),
    lastValidSlice: createOutputBuffer(device, cellCount)
  };
  const graph = new GPUCommandGraph(device, {id: 'emerging-hot-spots-test'});
  const extraBuffers: Buffer[] = [];
  let neighborParameterBuffer: GPUParameterBuffer<'float32'> | undefined;
  let weights: GPUSpatialWeights | undefined;
  if (source) {
    if (source.kind === 'csr') {
      outputs.offsets.write(source.offsets);
      outputs.neighbors.write(source.neighbors);
      outputs.spatialWeights.write(source.weights);
    }
    weights = {
      offsets: importGraphBuffer(graph, 'offsets', outputs.offsets, 'uint32', cellCount + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', outputs.neighbors, 'uint32', capacity),
      weights: importGraphBuffer(
        graph,
        'spatial-weights',
        outputs.spatialWeights,
        'float32',
        capacity
      )
    };
    if (source.kind === 'neighbor-search') {
      const positionsBuffer = createInputBuffer(device, source.positions);
      extraBuffers.push(positionsBuffer);
      neighborParameterBuffer = new GPUParameterBuffer(device, {
        id: 'neighbor-search-parameters',
        format: 'float32',
        length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
        values: getGPUNeighborSearchParameterValues(source.parameters)
      });
      graph.add(
        new GPUNeighborSearch({
          mode: 'radius',
          gridSize: [16, 16],
          positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', cellCount),
          parameters: neighborParameterBuffer.importToGraph(graph),
          weights,
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
        })
      );
    }
  }
  const contributor = new GPUEmergingHotSpots({
    values:
      format === 'uint32'
        ? importGraphBuffer(graph, 'values', valuesBuffer, 'uint32', binCount)
        : importGraphBuffer(graph, 'values', valuesBuffer, 'float32', binCount),
    ...(weights ? {weights, selfWeight: options.selfWeight} : {gridWidth, gridHeight}),
    sliceCount,
    maximumRadius: options.maximumRadius,
    multipleTesting: options.multipleTesting,
    finalSlice: options.finalSlice,
    pValues: importGraphBuffer(graph, 'p-values', outputs.pValues, 'float32', binCount),
    adjustedPValues: importGraphBuffer(
      graph,
      'adjusted-p-values',
      outputs.adjustedPValues,
      'float32',
      binCount
    ),
    lastValidSlice: importGraphBuffer(
      graph,
      'last-valid-slice',
      outputs.lastValidSlice,
      'uint32',
      cellCount
    ),
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
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return buildCount;
    },
    async run(parameters, neighborParameters) {
      parameterBuffer.write(getGPUEmergingHotSpotParameterValues({...parameters, multipleTesting}));
      if (neighborParameters && neighborParameterBuffer) {
        neighborParameterBuffer.write(getGPUNeighborSearchParameterValues(neighborParameters));
      }
      for (const [name, buffer] of Object.entries(outputs)) {
        if (source?.kind === 'csr' && ['offsets', 'neighbors', 'spatialWeights'].includes(name)) {
          continue;
        }
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const giZScoreBits = await readUint32(outputs.giZScores, binCount);
      const trendSBits = await readUint32(outputs.trendS, cellCount);
      return {
        giZScores: Array.from(new Float32Array(Uint32Array.from(giZScoreBits).buffer)),
        giZScoreBits,
        pValues: await readFloat32(outputs.pValues, binCount),
        adjustedPValues: await readFloat32(outputs.adjustedPValues, binCount),
        lastValidSlice: await readUint32(outputs.lastValidSlice, cellCount),
        trendZ: await readFloat32(outputs.trendZ, cellCount),
        trendP: await readFloat32(outputs.trendP, cellCount),
        trendS: Array.from(new Int32Array(Uint32Array.from(trendSBits).buffer)),
        category: await readUint32(outputs.category, cellCount),
        hotSliceCount: await readUint32(outputs.hotSliceCount, cellCount),
        coldSliceCount: await readUint32(outputs.coldSliceCount, cellCount),
        globalStatistics: await readFloat32(outputs.globalStatistics, 4),
        csr: {
          offsets: await readUint32(outputs.offsets, cellCount + 1),
          neighbors: await readUint32(outputs.neighbors, capacity),
          weights: await readFloat32(outputs.spatialWeights, capacity)
        }
      };
    },
    writeValues(values) {
      valuesBuffer.write(values);
    },
    destroy() {
      compiled.destroy();
      valuesBuffer.destroy();
      maskBuffer?.destroy();
      for (const buffer of extraBuffers) {
        buffer.destroy();
      }
      neighborParameterBuffer?.destroy();
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

/** Relative-or-absolute closeness that treats NaN as equal to NaN only. */
function isCloseOrBothNaN(actual: number, expected: number, absolute: number, relative: number) {
  if (Number.isNaN(expected) || Number.isNaN(actual)) {
    return Number.isNaN(expected) && Number.isNaN(actual);
  }
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}

/**
 * Passes 3 and 4 are exact given the GPU's own z-scores and adjusted p-values, away from the trend
 * p-value bound. The p-values and adjusted p-values are checked against the double precision oracle.
 */
function expectCellsMatchOracle(
  result: Readback,
  cube: EmergingHotSpotCube,
  parameters: Omit<GPUEmergingHotSpotParameters, 'multipleTesting'>,
  label: string,
  options: {multipleTesting?: GPUMultipleTestingMethod; finalSlice?: 'require' | 'last-valid'} = {}
): void {
  const multipleTesting = options.multipleTesting ?? 'benjamini-hochberg';
  const packed = getGPUEmergingHotSpotParameterValues({...parameters, multipleTesting});
  // p-values come from the GPU's own z; the family is every finite z.
  const oraclePValues = computeBinPValues(result.giZScores);
  const oracleAdjusted = computeAdjustedBinPValues(result.giZScores, multipleTesting);
  for (let bin = 0; bin < result.giZScores.length; bin++) {
    expect(
      isCloseOrBothNaN(result.pValues[bin], oraclePValues[bin], 1e-7, 2e-3),
      `${label} p ${bin}: ${result.pValues[bin]} vs ${oraclePValues[bin]}`
    ).toBe(true);
    expect(
      isCloseOrBothNaN(result.adjustedPValues[bin], oracleAdjusted[bin], 1e-7, 5e-3),
      `${label} adjusted p ${bin}: ${result.adjustedPValues[bin]} vs ${oracleAdjusted[bin]}`
    ).toBe(true);
  }
  // Classification is checked from the GPU's adjusted p-values so a bin within float32 error of the
  // significance level cannot flip a comparison.
  const oracle = computeEmergingHotSpotCells(result.giZScores, cube, packed, {
    multipleTesting,
    finalSlice: options.finalSlice,
    adjustedPValues: result.adjustedPValues
  });
  expect(result.trendS, `${label} trendS`).toEqual(oracle.trendS);
  expect(result.hotSliceCount, `${label} hot`).toEqual(oracle.hotSliceCount);
  expect(result.coldSliceCount, `${label} cold`).toEqual(oracle.coldSliceCount);
  expect(result.lastValidSlice, `${label} lastValidSlice`).toEqual(oracle.lastValidSlice);
  const level = packed[3];
  for (const [cell, expected] of oracle.category.entries()) {
    expect(
      isCloseOrBothNaN(result.trendZ[cell], oracle.trendZ[cell], 1e-4, 1e-4),
      `${label} trendZ ${cell}`
    ).toBe(true);
    expect(
      isCloseOrBothNaN(result.trendP[cell], oracle.trendP[cell], 1e-5, 1e-3),
      `${label} trendP ${cell}`
    ).toBe(true);
    // Skip cells whose trend p-value sits on the level (float32 versus double precision).
    if (!(Math.abs(oracle.trendP[cell] - level) <= 0.02 * level)) {
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
        confidenceLevel: 0.9,
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

/** Series lengths around the Gi* workgroup tiling: whole cells per 256 threads, leftover threads, one cell per group. */
it('GPUEmergingHotSpots matches the oracle for slice counts that tile the Gi* workgroup unevenly', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const sliceCount of [1, 2, 100, 129, 256]) {
    const gridWidth = 6;
    const gridHeight = 5;
    const random = createSeededRandom(30 + sliceCount);
    const values = new Float32Array(gridWidth * gridHeight * sliceCount);
    for (let cell = 0; cell < gridWidth * gridHeight; cell++) {
      const hot = cell % gridWidth < 2 && Math.floor(cell / gridWidth) < 3;
      for (let slice = 0; slice < sliceCount; slice++) {
        values[cell * sliceCount + slice] = random() * 2 + (hot ? 3 + (4 * slice) / sliceCount : 0);
      }
    }
    values[3] = NaN;
    const mask = new Uint32Array(gridWidth * gridHeight).fill(1);
    mask[8] = 0;
    const cube = {gridWidth, gridHeight, sliceCount, values, mask};
    const harness = createHarness(device, cube, {maximumRadius: 2});
    try {
      for (const frame of [
        {radius: 1, temporalWindow: 0},
        {radius: 2, temporalWindow: 7},
        // Windows at or above the series length clamp to a prefix sum of the whole series.
        {radius: 1.5, temporalWindow: 400}
      ]) {
        const label = `slices ${sliceCount} ${JSON.stringify(frame)}`;
        const result = await harness.run(frame);
        const packed = getGPUEmergingHotSpotParameterValues(frame);
        expectZScoresClose(result.giZScores, computeSpaceTimeGiStar(cube, packed, 2), label);
        expectCellsMatchOracle(result, cube, frame, label);
      }
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUEmergingHotSpots guards padded cooperative workgroups under bounded dispatch', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const descriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  const limits = device.limits;
  Object.defineProperty(device, 'limits', {
    configurable: true,
    value: new Proxy(limits, {
      get: (target, property) =>
        property === 'maxComputeWorkgroupsPerDimension' ? 2 : Reflect.get(target, property, target)
    })
  });
  try {
    // Five cells require a padded 2x2x2 dispatch when each dimension is limited to two.
    const cube = {
      gridWidth: 5,
      gridHeight: 1,
      sliceCount: 7,
      values: Float32Array.from({length: 35}, (_, index) =>
        index === 4 ? NaN : ((index * 17) % 23) / 3
      )
    };
    const harness = createHarness(device, cube, {maximumRadius: 0});
    try {
      const frame = {radius: 0, temporalWindow: 0};
      const result = await harness.run(frame);
      expectCellsMatchOracle(result, cube, frame, 'bounded dispatch');
    } finally {
      harness.destroy();
    }
  } finally {
    if (descriptor) {
      Object.defineProperty(device, 'limits', descriptor);
    }
  }
}, 60000);

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
    const constant = await harness.run({radius: 1, temporalWindow: 1, minimumTrendSliceCount: 0});
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

/** CSR of the lattice disc neighborhood (focal cell excluded): the weights-mode twin of a radius. */
function createLatticeDiscWeights(
  gridWidth: number,
  gridHeight: number,
  radius: number
): {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array} {
  const offsets = [0];
  const neighbors: number[] = [];
  const reach = Math.floor(radius);
  for (let cell = 0; cell < gridWidth * gridHeight; cell++) {
    const column = cell % gridWidth;
    const row = Math.floor(cell / gridWidth);
    for (
      let neighborRow = Math.max(row - reach, 0);
      neighborRow <= Math.min(row + reach, gridHeight - 1);
      neighborRow++
    ) {
      for (
        let neighborColumn = Math.max(column - reach, 0);
        neighborColumn <= Math.min(column + reach, gridWidth - 1);
        neighborColumn++
      ) {
        const squared = (neighborRow - row) ** 2 + (neighborColumn - column) ** 2;
        if (squared <= Math.fround(radius * radius) && squared > 0) {
          neighbors.push(neighborRow * gridWidth + neighborColumn);
        }
      }
    }
    offsets.push(neighbors.length);
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: new Float32Array(neighbors.length).fill(1)
  };
}

it('GPUEmergingHotSpots weights mode reproduces lattice mode on lattice-disc weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cube = createSpaceTimeCube(5);
  for (const radius of [1, 1.5, 2.3]) {
    const weights = createLatticeDiscWeights(cube.gridWidth, cube.gridHeight, radius);
    const harness = createHarness(device, cube, {weights: {kind: 'csr', ...weights}});
    try {
      for (const frame of [
        {temporalWindow: 0},
        {temporalWindow: 3, confidenceLevel: 0.95 as const},
        {temporalWindow: 23}
      ]) {
        const latticeFrame = {...frame, radius};
        const result = await harness.run(latticeFrame);
        const label = `radius ${radius} ${JSON.stringify(frame)}`;
        expectZScoresClose(
          result.giZScores,
          computeSpaceTimeGiStar(cube, getGPUEmergingHotSpotParameterValues(latticeFrame)),
          label
        );
        expectCellsMatchOracle(result, cube, latticeFrame, label);
        expect((await harness.run(latticeFrame)).giZScoreBits).toEqual(result.giZScoreBits);
      }
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  }
}, 120000);

/** Cells at random points (a stand-in for H3 cell centers): irregular neighborhoods. */
function createScatteredCube(seed: number, cellCount: number, sliceCount: number) {
  const random = createSeededRandom(seed);
  const positions = Float32Array.from({length: cellCount * 2}, () => random() * 100);
  const values = new Float32Array(cellCount * sliceCount);
  for (let cell = 0; cell < cellCount; cell++) {
    const hot = Math.hypot(positions[cell * 2] - 30, positions[cell * 2 + 1] - 60) < 25;
    for (let slice = 0; slice < sliceCount; slice++) {
      values[cell * sliceCount + slice] = random() * 2 + (hot ? 4 + slice / 3 : 0);
    }
  }
  values[7] = NaN;
  const mask = new Uint32Array(cellCount).fill(1);
  mask[11] = 0;
  return {
    positions,
    cube: {gridWidth: cellCount, gridHeight: 1, sliceCount, values, mask} as EmergingHotSpotCube & {
      values: Float32Array;
    }
  };
}

it('GPUEmergingHotSpots weights mode matches the weighted oracle on irregular weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, cube} = createScatteredCube(9, 150, 10);
  for (const [selfWeight, rowStandardize] of [
    [1, false],
    [0, false],
    [0.5, true]
  ] as const) {
    const weights = createDistanceBandWeights(positions, 14, {rowStandardize});
    const harness = createHarness(device, cube, {
      weights: {kind: 'csr', ...weights},
      selfWeight
    });
    try {
      for (const frame of [
        {temporalWindow: 0},
        {temporalWindow: 2},
        {temporalWindow: 9, confidenceLevel: 0.9}
      ]) {
        const result = await harness.run(frame);
        const label = `self ${selfWeight} standardized ${rowStandardize} ${JSON.stringify(frame)}`;
        const packed = getGPUEmergingHotSpotParameterValues(frame);
        const expectedZ = computeSpaceTimeGiStarWeighted(cube, weights, selfWeight, packed);
        expectZScoresClose(result.giZScores, expectedZ, label);
        expect(expectedZ.some(z => z >= 1.96)).toBe(true);
        expectCellsMatchOracle(result, cube, frame, label);
      }
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUEmergingHotSpots weights mode consumes weights written by GPUNeighborSearch', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, cube} = createScatteredCube(21, 150, 10);
  const harness = createHarness(device, cube, {
    weights: {
      kind: 'neighbor-search',
      positions,
      parameters: {bounds: [0, 0, 100, 100], radius: 10},
      capacity: 150 * 150
    }
  });
  try {
    for (const radius of [10, 18]) {
      const frame = {temporalWindow: 3};
      const result = await harness.run(frame, {bounds: [0, 0, 100, 100], radius});
      const expected = createDistanceBandWeights(positions, radius);
      const used = result.csr.offsets[150];
      expect(result.csr.neighbors.slice(0, used), `radius ${radius}`).toEqual(
        Array.from(expected.neighbors)
      );
      const expectedZ = computeSpaceTimeGiStarWeighted(
        cube,
        expected,
        1,
        getGPUEmergingHotSpotParameterValues(frame)
      );
      expectZScoresClose(result.giZScores, expectedZ, `radius ${radius}`);
      expectCellsMatchOracle(result, cube, frame, `radius ${radius}`);
      expect(result.category.some(category => category !== 0)).toBe(true);
    }
    expect(harness.buildCount).toBe(1);
  } finally {
    harness.destroy();
  }
}, 120000);

/** 100 bins at radius 0 and window 0: Gi* z is the population standard score of the bin. */
function createMultipleTestingCube() {
  return {
    gridWidth: MULTIPLE_TESTING_VALUES.length,
    gridHeight: 1,
    sliceCount: 1,
    values: Float32Array.from(MULTIPLE_TESTING_VALUES)
  };
}

it('GPUEmergingHotSpots multiple testing drops weakly significant bins (statsmodels reference)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cube = createMultipleTestingCube();
  const frame = {radius: 0, temporalWindow: 0};
  for (const method of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const harness = createHarness(device, cube, {multipleTesting: method});
    try {
      const result = await harness.run(frame);
      const significant = MULTIPLE_TESTING_SIGNIFICANT[method];
      // A single-slice bin that is hot is New hot, cold is New cold; the rest have no pattern.
      const flagged = result.category
        .map((category, cell) => (category === 0 ? -1 : cell))
        .filter(cell => cell >= 0);
      expect(flagged, method).toEqual(significant);
      if (method !== 'none') {
        // Adjusted p-values against statsmodels multipletests (float32 erfcc fit, 2e-3 relative).
        MULTIPLE_TESTING_TOP_BINS.forEach((bin, index) => {
          const expected = MULTIPLE_TESTING_ADJUSTED[method][index];
          expect(
            Math.abs(result.adjustedPValues[bin] - expected),
            `${method} bin ${bin}`
          ).toBeLessThan(2e-3 * expected + 1e-7);
        });
      }
      // The legend does not change with the correction: signs map to the same category codes.
      expect(result.category[92]).toBe(GPU_EMERGING_HOT_SPOT_CATEGORIES.NEW_HOT);
      expect(result.category[94]).toBe(GPU_EMERGING_HOT_SPOT_CATEGORIES.NEW_COLD);
      expectCellsMatchOracle(result, cube, frame, method, {multipleTesting: method});
    } finally {
      harness.destroy();
    }
  }
}, 120000);

it('GPUEmergingHotSpots criticalZ override reproduces the uncorrected rule with multipleTesting none', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cube = createSpaceTimeCube(5);
  const harness = createHarness(device, cube, {maximumRadius: 4, multipleTesting: 'none'});
  try {
    for (const frame of [
      {radius: 1.5, temporalWindow: 2, confidenceLevel: 0.9 as const},
      {radius: 2, temporalWindow: 3, criticalZ: 1.2, trendSignificanceLevel: 0.2}
    ]) {
      const result = await harness.run(frame);
      const label = JSON.stringify(frame);
      // |z| >= criticalZ exactly, on the GPU's own z.
      const criticalZ = Math.fround(
        getGPUEmergingHotSpotParameterValues({...frame, multipleTesting: 'none'})[2]
      );
      const hot = Array.from({length: 70}, (_, cell) =>
        result.giZScores
          .slice(cell * cube.sliceCount, (cell + 1) * cube.sliceCount)
          .filter(z => z >= criticalZ)
      );
      expect(result.hotSliceCount, label).toEqual(hot.map(series => series.length));
      expectCellsMatchOracle(result, cube, frame, label, {multipleTesting: 'none'});
    }
  } finally {
    harness.destroy();
  }
}, 120000);

it('GPUEmergingHotSpots handles a missing final slice under both finalSlice modes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cube = createSpaceTimeCube(5);
  const {sliceCount} = cube;
  // Stale cells: final slice missing. One cell is entirely missing.
  const staleCells = [3, 17, 40];
  const values = Float32Array.from(cube.values);
  for (const cell of staleCells) {
    values[cell * sliceCount + sliceCount - 1] = NaN;
  }
  values.fill(NaN, 50 * sliceCount, 51 * sliceCount);
  const staleCube = {...cube, values};
  const frame = {radius: 1, temporalWindow: 2};
  const results = new Map<string, Readback>();
  for (const finalSlice of ['require', 'last-valid'] as const) {
    const harness = createHarness(device, staleCube, {finalSlice});
    try {
      const result = await harness.run(frame);
      results.set(finalSlice, result);
      expectCellsMatchOracle(result, staleCube, frame, finalSlice, {finalSlice});
      for (const cell of staleCells) {
        expect(result.lastValidSlice[cell], `${finalSlice} last valid ${cell}`).toBe(
          sliceCount - 2
        );
      }
      // A cell with no data at all has no pattern under either mode.
      expect(result.lastValidSlice[50]).toBe(0xffffffff);
      expect(result.category[50]).toBe(GPU_EMERGING_HOT_SPOT_CATEGORIES.NO_PATTERN);
    } finally {
      harness.destroy();
    }
  }
  const required = results.get('require')!;
  const lastValid = results.get('last-valid')!;
  for (const cell of staleCells) {
    expect(required.category[cell], `require ${cell}`).toBe(
      GPU_EMERGING_HOT_SPOT_CATEGORIES.UNDETERMINED
    );
    expect(lastValid.category[cell], `last-valid ${cell}`).not.toBe(
      GPU_EMERGING_HOT_SPOT_CATEGORIES.UNDETERMINED
    );
  }
  // Cells with a final slice are identical under both modes.
  for (let cell = 0; cell < required.category.length; cell++) {
    if (!staleCells.includes(cell) && cell !== 50) {
      expect(required.category[cell], `cell ${cell}`).toBe(lastValid.category[cell]);
    }
  }
}, 120000);

it('GPUEmergingHotSpots reports undetermined categories for series below the Mann-Kendall minimum', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 5 x 4 cells, six slices: cell 0 constantly hot, cell 1 constantly cold, the rest noise.
  const sliceCount = 6;
  const cellCount = 20;
  const random = createSeededRandom(2);
  const values = new Float32Array(cellCount * sliceCount);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let slice = 0; slice < sliceCount; slice++) {
      values[cell * sliceCount + slice] = cell === 0 ? 30 : cell === 1 ? -30 : random() * 2 - 1;
    }
  }
  const cube = {gridWidth: 5, gridHeight: 4, sliceCount, values};
  const frame = {radius: 0, temporalWindow: 0};
  const categories = GPU_EMERGING_HOT_SPOT_CATEGORIES;
  const guarded = createHarness(device, cube);
  try {
    // Default minimumTrendSliceCount 10 > 6: no trend z or p, and the persistent classes are
    // undetermined instead of silently Persistent.
    const result = await guarded.run(frame);
    expect(result.category[0]).toBe(categories.UNDETERMINED);
    expect(result.category[1]).toBe(categories.UNDETERMINED);
    expect(result.trendZ.every(Number.isNaN)).toBe(true);
    expect(result.trendP.every(Number.isNaN)).toBe(true);
    expect(result.trendS[0]).toBe(0);
    expectCellsMatchOracle(result, cube, frame, 'guarded');
  } finally {
    guarded.destroy();
  }
  const unguarded = createHarness(device, cube);
  try {
    const parameters = {...frame, minimumTrendSliceCount: 0};
    const result = await unguarded.run(parameters);
    expect(result.category[0]).toBe(categories.PERSISTENT_HOT);
    expect(result.category[1]).toBe(categories.PERSISTENT_COLD);
    expect(result.trendP[0]).toBe(1);
    expectCellsMatchOracle(result, cube, parameters, 'unguarded');
  } finally {
    unguarded.destroy();
  }
}, 120000);
