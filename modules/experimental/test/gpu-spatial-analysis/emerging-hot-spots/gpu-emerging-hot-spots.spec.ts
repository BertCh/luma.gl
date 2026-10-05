// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUEmergingHotSpotParameterValues,
  GPUEmergingHotSpots,
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
  computeEmergingHotSpotCells,
  computeEmergingHotSpotMoments,
  computeSpaceTimeGiStar,
  computeSpaceTimeGiStarWeighted,
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
    parameters: GPUEmergingHotSpotParameters,
    neighborParameters?: GPUNeighborSearchParameters
  ): Promise<Readback>;
  writeValues(values: Float32Array | Uint32Array): void;
  destroy(): void;
};

function createHarness(
  device: Device,
  cube: EmergingHotSpotCube & {values: Float32Array | Uint32Array},
  options: {maximumRadius?: number; weights?: WeightsSource; selfWeight?: number} = {}
): Harness {
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
    | 'globalStatistics',
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
    globalStatistics: createOutputBuffer(device, 4)
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
      parameterBuffer.write(getGPUEmergingHotSpotParameterValues(parameters));
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
        {temporalWindow: 9, criticalZ: 1.2}
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
