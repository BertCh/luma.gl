// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getGPUClassBreaksParameterValues} from '../../../src/gpu-dataframe/column-classification/class-breaks-parameters';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPUClassAssignment,
  GPULISAMarkov,
  GPUSpatialMarkov,
  GPUTransitionMatrix,
  GPU_LISA_MARKOV_STATE_COUNT
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/index';
import {
  computeLocalMoranOracle,
  createAutocorrelatedScene,
  createDistanceBandWeights
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {computeTransitionOracle, getOracleClass, getOracleQuantile} from './distribution-oracle';

const NO_CLASS = 0xffffffff;

function expectClose(actual: number[], expected: number[], label: string, relative = 1e-5): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Math.abs(actual[index] - expected[index]) > 1e-6 + relative * Math.abs(expected[index])) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

it('GPUClassAssignment follows the shared class rule with ties, NaN, infinities and mask', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const edges = [0, 10, 20, 30, 40];
  const values = [
    -5,
    0,
    9.99,
    10,
    10.01,
    20,
    29,
    30,
    39.9,
    40,
    100,
    NaN,
    Infinity,
    -Infinity,
    15,
    25,
    35,
    5
  ];
  const mask = values.map((_, index) => (index === 14 ? 0 : 1));
  const rig = new AnalysisRig(device);
  const output = rig.output('uint32', values.length);
  rig.run(
    new GPUClassAssignment({
      values: rig.input(Float32Array.from(values), 'float32'),
      breaks: rig.input(Float32Array.from(edges), 'float32'),
      classCount: rig.input(Uint32Array.of(4), 'uint32'),
      mask: rig.input(Uint32Array.from(mask), 'uint32'),
      output
    })
  );
  const actual = await rig.readUint(output);
  const expected = values.map((value, index) =>
    mask[index] === 0 ? NO_CLASS : getOracleClass(value, edges, 4)
  );
  expect(actual).toEqual(expected);
  expect(new Set(actual).size).toBeGreaterThan(3);
  rig.destroy();
});

it('GPUTransitionMatrix matches the oracle for conditions, period lag and invalid classes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const {rows, periods, classCount, conditionCount, periodLag, seed} of [
    {rows: 700, periods: 6, classCount: 5, conditionCount: 1, periodLag: 1, seed: 1},
    {rows: 333, periods: 7, classCount: 4, conditionCount: 3, periodLag: 2, seed: 2},
    // Several workgroup tiles with a partial last tile (workgroup-private histogram).
    {rows: 5003, periods: 5, classCount: 5, conditionCount: 1, periodLag: 1, seed: 3},
    {rows: 3001, periods: 4, classCount: 6, conditionCount: 7, periodLag: 1, seed: 5},
    // 64 * 64 cells exceed the private histogram: global atomics.
    {rows: 2100, periods: 4, classCount: 64, conditionCount: 1, periodLag: 1, seed: 4}
  ]) {
    const random = createSeededRandom(seed);
    const classes = Uint32Array.from({length: rows * periods}, () =>
      random() < 0.05 ? NO_CLASS : Math.floor(random() * classCount)
    );
    const conditions = Uint32Array.from({length: rows * periods}, () =>
      random() < 0.03 ? NO_CLASS : Math.floor(random() * conditionCount)
    );
    const mask = Uint32Array.from({length: rows}, () => (random() < 0.9 ? 1 : 0));
    const cells = conditionCount * classCount * classCount;
    const rig = new AnalysisRig(device);
    const counts = rig.output('uint32', cells + 2);
    const probabilities = rig.output('float32', cells);
    const rowTotals = rig.output('uint32', conditionCount * classCount);
    const ignored = rig.output('uint32', 1);
    rig.run(
      new GPUTransitionMatrix({
        classes: rig.input(classes, 'uint32'),
        rows,
        periods,
        classCount,
        periodLag,
        conditionClasses: conditionCount > 1 ? rig.input(conditions, 'uint32') : undefined,
        conditionCount,
        mask: rig.input(mask, 'uint32'),
        output: {counts, probabilities, rowTotals, ignored}
      })
    );
    const expected = computeTransitionOracle({
      classes,
      conditions: conditionCount > 1 ? conditions : undefined,
      mask,
      rows,
      periods,
      classCount,
      conditionCount,
      periodLag
    });
    const actualCounts = (await rig.readUint(counts)).slice(0, cells);
    expect(actualCounts.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(rows);
    expect(actualCounts).toEqual(expected.counts);
    expect(await rig.readUint(rowTotals)).toEqual(expected.rowTotals);
    expect((await rig.readUint(ignored))[0]).toBe(expected.ignored);
    expectClose(await rig.readFloat(probabilities), expected.probabilities, 'probabilities');
    rig.destroy();
  }
});

it('GPUSpatialMarkov conditions transitions on pooled spatial-lag classes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rows = 90;
  const periods = 5;
  const classCount = 4;
  const lagClassCount = 3;
  const scene = createAutocorrelatedScene(7, rows, [14]);
  const weights = createDistanceBandWeights(scene.positions, 14, {rowStandardize: true});
  const random = createSeededRandom(5);
  const values = new Float32Array(rows * periods);
  for (let period = 0; period < periods; period++) {
    for (let row = 0; row < rows; row++) {
      values[period * rows + row] = Math.fround(
        scene.values[row] * (1 + 0.15 * period) + random() * 3
      );
    }
  }
  const classes = Uint32Array.from({length: rows * periods}, () =>
    Math.floor(random() * classCount)
  );
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 === 0 ? 0 : 1));
  const rig = new AnalysisRig(device);
  const csr = {
    offsets: Array.from(weights.offsets),
    neighbors: Array.from(weights.neighbors),
    weights: Array.from(weights.weights)
  };
  const cells = lagClassCount * classCount * classCount;
  const counts = rig.output('uint32', cells);
  const lagBreaks = rig.output('float32', lagClassCount + 1);
  const lagClassCountView = rig.output('uint32', 1);
  const lagValues = rig.output('float32', rows * periods);
  const lagClasses = rig.output('uint32', rows * periods);
  rig.run(
    new GPUSpatialMarkov({
      values: rig.input(values, 'float32'),
      classes: rig.input(classes, 'uint32'),
      classCount,
      weights: rig.weights(csr),
      periods,
      lagMaximumClassCount: lagClassCount,
      lagParameters: rig.input(
        getGPUClassBreaksParameterValues(
          {method: 'quantile', classCount: lagClassCount},
          lagClassCount
        ),
        'float32'
      ),
      mask: rig.input(mask, 'uint32'),
      output: {
        counts,
        lagBreaks,
        lagClassCount: lagClassCountView,
        lagValues,
        lagClasses
      }
    })
  );
  // Lag against a CPU row-standardized lag over unmasked neighbors.
  const actualLag = await rig.readFloat(lagValues);
  const expectedLag: number[] = [];
  for (let period = 0; period < periods; period++) {
    for (let row = 0; row < rows; row++) {
      let sum = 0;
      let weightSum = 0;
      if (mask[row]) {
        for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
          const neighbor = csr.neighbors[slot];
          if (mask[neighbor]) {
            sum += csr.weights[slot] * values[period * rows + neighbor];
            weightSum += csr.weights[slot];
          }
        }
      }
      expectedLag.push(weightSum > 0 ? sum / weightSum : 0);
    }
  }
  expectClose(actualLag, expectedLag, 'lag values', 1e-4);
  // Pooled quantile edges of the unmasked lag column.
  const pooled: number[] = [];
  for (let index = 0; index < rows * periods; index++) {
    if (mask[index % rows]) pooled.push(actualLag[index]);
  }
  pooled.sort((a, b) => a - b);
  const actualBreaks = await rig.readFloat(lagBreaks);
  const expectedEdges = [0, 1, 2, 3].map(index =>
    index === 0
      ? pooled[0]
      : index === lagClassCount
        ? pooled[pooled.length - 1]
        : getOracleQuantile(pooled, Math.fround(index / lagClassCount))
  );
  expectClose(actualBreaks, expectedEdges, 'pooled lag breaks', 1e-5);
  expect((await rig.readUint(lagClassCountView))[0]).toBe(lagClassCount);
  // Exact counts from the GPU lag values and edges.
  const lagClassOracle = actualLag.map((value, index) =>
    mask[index % rows] ? getOracleClass(value, actualBreaks, lagClassCount) : NO_CLASS
  );
  expect(await rig.readUint(lagClasses)).toEqual(lagClassOracle);
  const expected = computeTransitionOracle({
    classes,
    conditions: lagClassOracle,
    mask,
    rows,
    periods,
    classCount,
    conditionCount: lagClassCount
  });
  const actualCounts = await rig.readUint(counts);
  expect(actualCounts).toEqual(expected.counts);
  for (let condition = 0; condition < lagClassCount; condition++) {
    const slice = actualCounts.slice(
      condition * classCount * classCount,
      (condition + 1) * classCount * classCount
    );
    expect(slice.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(0);
  }
  rig.destroy();
});

it('GPULISAMarkov counts transitions between per-period LISA quadrants', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rows = 120;
  const periods = 4;
  const scene = createAutocorrelatedScene(3, rows, [16]);
  const weights = createDistanceBandWeights(scene.positions, 16, {rowStandardize: true});
  const random = createSeededRandom(9);
  const values = new Float32Array(rows * periods);
  for (let period = 0; period < periods; period++) {
    for (let row = 0; row < rows; row++) {
      values[period * rows + row] = Math.fround(
        scene.values[row] * (period % 2 === 0 ? 1 : -0.6) + random() * 2 + period
      );
    }
  }
  const parameters = getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.1});
  expect(parameters.length).toBe(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH);
  const rig = new AnalysisRig(device);
  const cells = GPU_LISA_MARKOV_STATE_COUNT ** 2;
  const counts = rig.output('uint32', cells);
  const rowTotals = rig.output('uint32', GPU_LISA_MARKOV_STATE_COUNT);
  const quadrants = rig.output('uint32', rows * periods);
  rig.run(
    new GPULISAMarkov({
      values: rig.input(values, 'float32'),
      weights: rig.weights({
        offsets: Array.from(weights.offsets),
        neighbors: Array.from(weights.neighbors),
        weights: Array.from(weights.weights)
      }),
      periods,
      parameters: rig.input(parameters, 'float32'),
      output: {counts, rowTotals, quadrants}
    })
  );
  const expectedQuadrants: number[] = [];
  for (let period = 0; period < periods; period++) {
    const result = computeLocalMoranOracle({
      weights,
      values: values.subarray(period * rows, (period + 1) * rows),
      parameters: {significanceLevel: 0.1}
    });
    expectedQuadrants.push(...result.quadrants);
  }
  const actualQuadrants = await rig.readUint(quadrants);
  expect(actualQuadrants).toEqual(expectedQuadrants);
  expect(actualQuadrants.filter(value => value > 0).length).toBeGreaterThan(10);
  const expected = computeTransitionOracle({
    classes: expectedQuadrants,
    rows,
    periods,
    classCount: GPU_LISA_MARKOV_STATE_COUNT
  });
  expect(await rig.readUint(counts)).toEqual(expected.counts);
  expect(await rig.readUint(rowTotals)).toEqual(expected.rowTotals);
  rig.destroy();
});

// Reference from giddy 'Markov' (PySAL venv): 40 locations x 5 periods, 4 classes (seeded numpy).
const GIDDY_CLASSES = [
  2, 3, 1, 0, 1, 1, 3, 3, 1, 2, 1, 3, 0, 3, 3, 2, 3, 0, 3, 3, 2, 3, 3, 1, 2, 3, 0, 1, 1, 3, 1, 2, 2,
  1, 2, 2, 3, 1, 3, 2, 3, 3, 1, 2, 3, 0, 1, 2, 2, 0, 1, 3, 2, 2, 3, 0, 2, 2, 0, 0, 2, 1, 1, 0, 1, 0,
  3, 3, 0, 0, 2, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0, 2, 3, 0, 3, 2, 0, 1, 0, 3, 1, 2, 2, 0, 1, 0, 1, 3,
  3, 0, 2, 3, 3, 0, 1, 2, 3, 3, 2, 0, 2, 0, 1, 2, 0, 2, 0, 2, 3, 2, 1, 3, 0, 1, 3, 3, 1, 3, 2, 2, 0,
  2, 0, 1, 1, 3, 1, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0, 0, 1, 0, 3, 2, 3, 2, 1, 1, 3, 2, 2, 3, 2, 2, 2, 3,
  3, 2, 2, 0, 1, 1, 0, 3, 1, 1, 0, 2, 1, 3, 1, 1, 3, 1, 1, 3, 1, 0, 3, 0, 0, 3, 2, 2, 1, 2, 0, 3, 2,
  2, 2
];
const GIDDY_COUNTS = [9, 11, 8, 14, 9, 9, 10, 8, 12, 9, 12, 7, 15, 7, 12, 8];
const GIDDY_P = [
  0.214286, 0.261905, 0.190476, 0.333333, 0.25, 0.25, 0.277778, 0.222222, 0.3, 0.225, 0.3, 0.175,
  0.357143, 0.166667, 0.285714, 0.190476
];

it('GPUTransitionMatrix reproduces giddy Markov counts and probabilities', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rig = new AnalysisRig(device);
  const counts = rig.output('uint32', 16);
  const probabilities = rig.output('float32', 16);
  rig.run(
    new GPUTransitionMatrix({
      classes: rig.input(Uint32Array.from(GIDDY_CLASSES), 'uint32'),
      rows: 40,
      periods: 5,
      classCount: 4,
      output: {counts, probabilities}
    })
  );
  expect(await rig.readUint(counts)).toEqual(GIDDY_COUNTS);
  expectClose(await rig.readFloat(probabilities), GIDDY_P, 'giddy p', 1e-5);
  rig.destroy();
});
