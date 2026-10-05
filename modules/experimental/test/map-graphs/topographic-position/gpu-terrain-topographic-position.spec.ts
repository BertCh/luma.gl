// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPUTerrainTopographicPosition,
  type GPUTerrainTopographicPositionScale
} from '../../../src/map-graphs/topographic-position/gpu-terrain-topographic-position';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeTopographicPosition,
  createTopographicTerrain,
  type TopographicPositionOracleResult
} from './topographic-position-oracle';

type TopographicPositionRun = {
  topographicPositionIndex: number[];
  deviationFromMean: number[];
  maximumDeviation: number[];
  maximumDeviationRadius: number[];
  validity: number[];
};

async function runTopographicPosition(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  scales: readonly GPUTerrainTopographicPositionScale[],
  options: {mask?: Uint32Array; quantum?: number} = {}
): Promise<TopographicPositionRun> {
  const pixelCount = width * height;
  const planeLength = pixelCount * scales.length;
  const graph = new GPUCommandGraph(device, {id: 'topographic-position-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = options.mask ? createInputBuffer(device, options.mask) : undefined;
  const buffers = {
    topographicPositionIndex: createOutputBuffer(device, planeLength),
    deviationFromMean: createOutputBuffer(device, planeLength),
    maximumDeviation: createOutputBuffer(device, pixelCount),
    maximumDeviationRadius: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  graph.add(
    new GPUTerrainTopographicPosition({
      width,
      height,
      scales,
      quantum: options.quantum,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pixelCount)
          : undefined
      },
      topographicPositionIndex: importGraphBuffer(
        graph,
        'tpi',
        buffers.topographicPositionIndex,
        'float32',
        planeLength
      ),
      deviationFromMean: importGraphBuffer(
        graph,
        'dev',
        buffers.deviationFromMean,
        'float32',
        planeLength
      ),
      maximumDeviation: importGraphBuffer(
        graph,
        'dev-max',
        buffers.maximumDeviation,
        'float32',
        pixelCount
      ),
      maximumDeviationRadius: importGraphBuffer(
        graph,
        'dev-radius',
        buffers.maximumDeviationRadius,
        'uint32',
        pixelCount
      ),
      validity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const run = {
    topographicPositionIndex: await readFloat32(buffers.topographicPositionIndex, planeLength),
    deviationFromMean: await readFloat32(buffers.deviationFromMean, planeLength),
    maximumDeviation: await readFloat32(buffers.maximumDeviation, pixelCount),
    maximumDeviationRadius: await readUint32(buffers.maximumDeviationRadius, pixelCount),
    validity: await readUint32(buffers.validity, pixelCount)
  };
  compiled.destroy();
  const owned: Buffer[] = [elevationBuffer, ...(maskBuffer ? [maskBuffer] : [])];
  for (const buffer of [...owned, ...Object.values(buffers)]) buffer.destroy();
  return run;
}

/** Returns the largest |actual - expected| / (absolute + relative * |expected|) ratio. */
function getErrorRatio(
  actual: readonly number[],
  expected: readonly number[],
  absolute: number,
  relative: number
): {ratio: number; maximumError: number} {
  let ratio = 0;
  let maximumError = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `index ${index} should be NaN`).toBe(true);
      continue;
    }
    const error = Math.abs(actual[index] - value);
    maximumError = Math.max(maximumError, error);
    ratio = Math.max(ratio, error / (absolute + relative * Math.abs(value)));
  }
  return {ratio, maximumError};
}

function expectMatchesOracle(
  run: TopographicPositionRun,
  oracle: TopographicPositionOracleResult,
  tpiTolerance: number,
  deviationTolerance: number
): {tpiError: number; deviationError: number} {
  const tpi = getErrorRatio(
    run.topographicPositionIndex,
    oracle.topographicPositionIndex,
    tpiTolerance,
    0
  );
  const deviation = getErrorRatio(
    run.deviationFromMean,
    oracle.deviationFromMean,
    deviationTolerance,
    deviationTolerance
  );
  expect(tpi.ratio).toBeLessThanOrEqual(1);
  expect(deviation.ratio).toBeLessThanOrEqual(1);
  expect(run.validity).toEqual(oracle.validity);
  return {tpiError: tpi.maximumError, deviationError: deviation.maximumError};
}

const SCALES: GPUTerrainTopographicPositionScale[] = [
  {radius: 1},
  {radius: 3, innerRadius: 1},
  {radius: 6}
];

it('GPUTerrainTopographicPosition matches the quantized oracle exactly up to float32 rounding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 41;
  const height = 33;
  const elevation = createTopographicTerrain(width, height, 1500, 120);
  const mask = new Uint32Array(width * height).fill(1);
  for (const index of [0, 7 * width + 9, 20 * width + 30, 32 * width + 40]) {
    mask[index] = 0;
  }
  const run = await runTopographicPosition(device, elevation, width, height, SCALES, {mask});
  // Structure checks guard against silent-zero shader failures.
  expect(run.deviationFromMean.some(value => Math.abs(value) > 0.5)).toBe(true);
  expect(run.topographicPositionIndex.some(value => Math.abs(value) > 1)).toBe(true);
  expect(run.validity.filter(value => value === 0)).toHaveLength(4);

  const quantized = computeTopographicPosition(elevation, mask, width, height, SCALES, 1 / 256);
  const errors = expectMatchesOracle(run, quantized, 1e-4, 2e-5);
  for (const [index, value] of quantized.maximumDeviation.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(run.maximumDeviation[index])).toBe(true);
      continue;
    }
    expect(Math.abs(run.maximumDeviation[index] - value)).toBeLessThan(
      2e-5 * (1 + Math.abs(value))
    );
  }
  // Radii agree wherever the best two scales are not within float32 noise of each other.
  let radiusMismatches = 0;
  for (const [index, radius] of quantized.maximumDeviationRadius.entries()) {
    if (run.maximumDeviationRadius[index] !== radius) radiusMismatches++;
  }
  expect(radiusMismatches).toBeLessThanOrEqual(2);

  // Against the unquantized definition, TPI moves by at most one quantum.
  const definition = computeTopographicPosition(elevation, mask, width, height, SCALES);
  const definitionErrors = expectMatchesOracle(run, definition, 1 / 256, 1e-3);
  console.info(
    `topographic position errors: quantized TPI ${errors.tpiError}, DEV ${errors.deviationError}; ` +
      `definition TPI ${definitionErrors.tpiError}, DEV ${definitionErrors.deviationError}`
  );
});

it('GPUTerrainTopographicPosition radius-1 TPI equals the 8-neighbour gdaldem TPI', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 9;
  const height = 7;
  // Integer elevations at a quarter-unit step are exact at the default quantum.
  const elevation = Float32Array.from(
    {length: width * height},
    (_, index) => 300 + ((index * 37) % 23) * 0.25
  );
  const run = await runTopographicPosition(device, elevation, width, height, [{radius: 1}]);
  for (let row = 1; row < height - 1; row++) {
    for (let column = 1; column < width - 1; column++) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx !== 0 || dy !== 0) sum += elevation[(row + dy) * width + column + dx];
        }
      }
      const index = row * width + column;
      expect(run.topographicPositionIndex[index]).toBeCloseTo(elevation[index] - sum / 8, 5);
    }
  }
});

it('GPUTerrainTopographicPosition keeps large high tiles exact where a float32 table fails', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 512;
  const height = 384;
  const elevation = createTopographicTerrain(width, height, 3800, 40);
  const scales = [{radius: 2}, {radius: 25}];
  const run = await runTopographicPosition(device, elevation, width, height, scales);
  // Sample pixels near the far corner, where table entries are largest.
  const samples = [
    [width - 3, height - 3],
    [width - 40, height - 30],
    [width >> 1, height >> 1],
    [5, 5]
  ];
  const pixelCount = width * height;
  let maximumTpiError = 0;
  let maximumDeviationError = 0;
  let float32TableTpiError = 0;
  const table = getFloat32SummedAreaTable(elevation, width, height);
  for (const [column, row] of samples) {
    const index = row * width + column;
    for (const [scaleIndex, scale] of scales.entries()) {
      const {mean, deviation, tpi} = getWindowStatistics(
        elevation,
        width,
        height,
        column,
        row,
        scale.radius
      );
      maximumTpiError = Math.max(
        maximumTpiError,
        Math.abs(run.topographicPositionIndex[scaleIndex * pixelCount + index] - tpi)
      );
      maximumDeviationError = Math.max(
        maximumDeviationError,
        Math.abs(run.deviationFromMean[scaleIndex * pixelCount + index] - deviation)
      );
      const float32Mean = getFloat32TableMean(table, width, height, column, row, scale.radius);
      float32TableTpiError = Math.max(float32TableTpiError, Math.abs(float32Mean - mean));
    }
  }
  console.info(
    `512x384 @ 3800 m: TPI error ${maximumTpiError} m, DEV error ${maximumDeviationError}; ` +
      `a float32 summed-area table mean errs by ${float32TableTpiError} m`
  );
  expect(maximumTpiError).toBeLessThan(1 / 256);
  expect(maximumDeviationError).toBeLessThan(1e-3);
  expect(float32TableTpiError).toBeGreaterThan(100 * maximumTpiError);
});

function getWindowStatistics(
  elevation: Float32Array,
  width: number,
  height: number,
  column: number,
  row: number,
  radius: number
): {mean: number; deviation: number; tpi: number} {
  let count = 0;
  let sum = 0;
  let squareSum = 0;
  const centre = elevation[row * width + column];
  for (
    let sampleRow = Math.max(row - radius, 0);
    sampleRow <= Math.min(row + radius, height - 1);
    sampleRow++
  ) {
    for (
      let sampleColumn = Math.max(column - radius, 0);
      sampleColumn <= Math.min(column + radius, width - 1);
      sampleColumn++
    ) {
      const difference = elevation[sampleRow * width + sampleColumn] - centre;
      count++;
      sum += difference;
      squareSum += difference * difference;
    }
  }
  const meanOffset = sum / count;
  const variance = squareSum / count - meanOffset * meanOffset;
  return {
    mean: centre + meanOffset,
    deviation: variance > 0 ? -meanOffset / Math.sqrt(variance) : 0,
    tpi: -sum / (count - 1)
  };
}

/** A naive float32 summed-area table, accumulated in float32 as a GPU float table would be. */
function getFloat32SummedAreaTable(
  elevation: Float32Array,
  width: number,
  height: number
): Float32Array {
  const table = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    let rowSum = 0;
    for (let column = 0; column < width; column++) {
      rowSum = Math.fround(rowSum + elevation[row * width + column]);
      const above = row > 0 ? table[(row - 1) * width + column] : 0;
      table[row * width + column] = Math.fround(rowSum + above);
    }
  }
  return table;
}

function getFloat32TableMean(
  table: Float32Array,
  width: number,
  height: number,
  column: number,
  row: number,
  radius: number
): number {
  const firstColumn = Math.max(column - radius, 0);
  const lastColumn = Math.min(column + radius, width - 1);
  const firstRow = Math.max(row - radius, 0);
  const lastRow = Math.min(row + radius, height - 1);
  const read = (sampleColumn: number, sampleRow: number) =>
    sampleColumn < 0 || sampleRow < 0 ? 0 : table[sampleRow * width + sampleColumn];
  const sum = Math.fround(
    Math.fround(
      Math.fround(read(lastColumn, lastRow) - read(firstColumn - 1, lastRow)) -
        read(lastColumn, firstRow - 1)
    ) + read(firstColumn - 1, firstRow - 1)
  );
  return sum / ((lastColumn - firstColumn + 1) * (lastRow - firstRow + 1));
}
