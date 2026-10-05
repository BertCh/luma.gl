// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUGeographicallyWeightedRegression} from '../../../src/geospatial/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/geospatial/spatial-regression/geographically-weighted-regression-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computeGeographicallyWeightedRegressionOnCPU,
  type GeographicallyWeightedRegressionOracleResult
} from './geographically-weighted-regression-oracle';

const LADDER_CAPACITY = 8;
const MAXIMUM_NEIGHBOR_COUNT = 64;

type Scene = {
  positions: Float32Array;
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
  mask?: Uint32Array;
};

type GPUResult = {
  coefficients: number[];
  localR2: number[];
  fitted: number[];
  residuals: number[];
  hatDiagonal: number[];
  localStatus: number[];
  bandwidthScores: number[];
  selectedBandwidth: number[];
  summary: number[];
};

type Fixture = {
  run(settings: GPUGeographicallyWeightedRegressionSettings): Promise<GPUResult>;
  readonly rebuildCount: number;
  destroy(): void;
};

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

/**
 * Jittered `side x side` grid on `[0, 10]^2`, two predictors, and a response
 * `intercept(x, y) + slope1(x, y) * x1 + slope2(x, y) * x2 + noise`.
 */
function createScene(
  side: number,
  seed: number,
  coefficients: (x: number, y: number) => [number, number, number],
  noise: number
): Scene {
  const random = createRandom(seed);
  const rows = side * side;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const x = ((row % side) + 0.2 + 0.6 * random()) * (10 / side);
    const y = (Math.floor(row / side) + 0.2 + 0.6 * random()) * (10 / side);
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    const x1 = random() * 4 - 2;
    const x2 = random() * 4 - 2;
    predictors[2 * row] = x1;
    predictors[2 * row + 1] = x2;
    const [b0, b1, b2] = coefficients(Math.fround(x), Math.fround(y));
    response[row] = b0 + b1 * Math.fround(x1) + b2 * Math.fround(x2) + noise * (random() - 0.5);
  }
  return {positions, predictors, predictorCount: 2, response};
}

function createFixture(device: Device, scene: Scene): Fixture {
  const {predictorCount} = scene;
  const rows = scene.response.length;
  const p = predictorCount + 1;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'gwr-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
  });
  const outputs = {
    coefficients: track(createOutputBuffer(device, rows * p)),
    localR2: track(createOutputBuffer(device, rows)),
    fitted: track(createOutputBuffer(device, rows)),
    residuals: track(createOutputBuffer(device, rows)),
    hatDiagonal: track(createOutputBuffer(device, rows)),
    localStatus: track(createOutputBuffer(device, rows)),
    bandwidthScores: track(createOutputBuffer(device, LADDER_CAPACITY)),
    selectedBandwidth: track(createOutputBuffer(device, 2)),
    summary: track(
      createOutputBuffer(device, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
    )
  };
  const graph = new GPUCommandGraph(device, {id: 'gwr-graph'});
  const float = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', length);
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, scene.positions)),
        'float32x2',
        rows
      ),
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device, scene.predictors)),
        'float32',
        rows * predictorCount
      ),
      predictorCount,
      response: importGraphBuffer(
        graph,
        'response',
        track(createInputBuffer(device, scene.response)),
        'float32',
        rows
      ),
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            rows
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      maximumBandwidthCount: LADDER_CAPACITY,
      maximumNeighborCount: MAXIMUM_NEIGHBOR_COUNT,
      output: {
        coefficients: float('coefficients', rows * p),
        localR2: float('localR2', rows),
        fitted: float('fitted', rows),
        residuals: float('residuals', rows),
        hatDiagonal: float('hatDiagonal', rows),
        localStatus: importGraphBuffer(graph, 'out-status', outputs.localStatus, 'uint32', rows),
        bandwidthScores: float('bandwidthScores', LADDER_CAPACITY),
        selectedBandwidth: float('selectedBandwidth', 2),
        summary: float('summary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      parameterBuffer.write(
        getGPUGeographicallyWeightedRegressionParameterValues(settings, LADDER_CAPACITY)
      );
      submitGraph(device, compiled, undefined);
      return {
        coefficients: await readFloat32(outputs.coefficients, rows * p),
        localR2: await readFloat32(outputs.localR2, rows),
        fitted: await readFloat32(outputs.fitted, rows),
        residuals: await readFloat32(outputs.residuals, rows),
        hatDiagonal: await readFloat32(outputs.hatDiagonal, rows),
        localStatus: await readUint32(outputs.localStatus, rows),
        bandwidthScores: await readFloat32(outputs.bandwidthScores, LADDER_CAPACITY),
        selectedBandwidth: await readFloat32(outputs.selectedBandwidth, 2),
        summary: await readFloat32(
          outputs.summary,
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
        )
      };
    },
    get rebuildCount() {
      return compileCount - 1;
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

function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const want = expected[index];
    const got = actual[index];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label}[${index}] should be NaN, got ${got}`).toBe(true);
    } else {
      const scale = Math.max(1, Math.abs(want));
      expect(Math.abs(got - want), `${label}[${index}] ${got} vs ${want}`).toBeLessThanOrEqual(
        tolerance * scale
      );
    }
  }
}

function compare(
  result: GPUResult,
  oracle: GeographicallyWeightedRegressionOracleResult,
  label: string,
  tolerance = 2e-3
): void {
  const rows = oracle.localStatus.length;
  expect(result.localStatus, `${label} status`).toEqual(oracle.localStatus);
  expectClose(
    result.bandwidthScores.slice(0, oracle.bandwidthScores.length),
    oracle.bandwidthScores,
    tolerance,
    `${label} scores`
  );
  expect(result.selectedBandwidth[0], `${label} selected index`).toBe(oracle.selectedIndex);
  expectClose(result.coefficients, oracle.coefficients, tolerance, `${label} coefficients`);
  expectClose(result.fitted, oracle.fitted, tolerance, `${label} fitted`);
  expectClose(result.residuals, oracle.residuals, tolerance, `${label} residuals`);
  expectClose(result.hatDiagonal, oracle.hatDiagonal, tolerance, `${label} hat`);
  expectClose(result.localR2, oracle.localR2, tolerance, `${label} local R2`);
  expect(result.coefficients.length).toBe(rows * 3);
  const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
  expectClose(
    [
      result.summary[SUMMARY.RESIDUAL_SUM_OF_SQUARES],
      result.summary[SUMMARY.TRACE_OF_HAT],
      result.summary[SUMMARY.AICC],
      result.summary[SUMMARY.R_SQUARED],
      result.summary[SUMMARY.OBSERVATION_COUNT],
      result.summary[SUMMARY.HAS_VALID_CANDIDATE]
    ],
    [
      oracle.residualSumOfSquares,
      oracle.traceOfHat,
      oracle.aicc,
      oracle.rSquared,
      oracle.observationCount,
      oracle.hasValidCandidate ? 1 : 0
    ],
    tolerance,
    `${label} summary`
  );
}

function computeOracle(
  scene: Scene,
  settings: GPUGeographicallyWeightedRegressionSettings
): GeographicallyWeightedRegressionOracleResult {
  return computeGeographicallyWeightedRegressionOnCPU({
    ...scene,
    settings,
    maximumNeighborCount: MAXIMUM_NEIGHBOR_COUNT
  });
}

function getCorrelation(left: number[], right: number[]): number {
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const leftMean = mean(left);
  const rightMean = mean(right);
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < left.length; index++) {
    covariance += (left[index] - leftMean) * (right[index] - rightMean);
    leftVariance += (left[index] - leftMean) ** 2;
    rightVariance += (right[index] - rightMean) ** 2;
  }
  return covariance / Math.sqrt(leftVariance * rightVariance);
}

const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;

it('GPUGeographicallyWeightedRegression recovers spatially constant coefficients', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const truth: [number, number, number] = [1, 2, -3];
  const scene = createScene(16, 5, () => truth, 0.02);
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUGeographicallyWeightedRegressionSettings = {
      kernel: 'bisquare',
      bandwidths: [3, 4, 6]
    };
    const result = await fixture.run(settings);
    expect(result.localStatus.every(status => status === STATUS.OK)).toBe(true);
    for (let row = 0; row < 256; row++) {
      for (let column = 0; column < 3; column++) {
        expect(
          Math.abs(result.coefficients[row * 3 + column] - truth[column]),
          `row ${row} column ${column}`
        ).toBeLessThan(0.05);
      }
    }
    compare(result, computeOracle(scene, settings), 'constant');
    // A near-perfect global model needs no more flexibility than the widest bandwidth.
    expect(result.selectedBandwidth[0]).toBe(2);
    expect(
      result.summary[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.R_SQUARED]
    ).toBeGreaterThan(0.99);
  } finally {
    fixture.destroy();
  }
});

it('GPUGeographicallyWeightedRegression recovers a coefficient that varies with x', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(20, 9, (x, y) => [1, x, 0.5], 0.1);
  const fixture = createFixture(device, scene);
  try {
    for (const settings of [
      {
        kernel: 'gaussian',
        bandwidthMode: 'adaptive',
        bandwidths: [20, 40, 60]
      },
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [1.5, 2.5, 4]}
    ] as GPUGeographicallyWeightedRegressionSettings[]) {
      const result = await fixture.run(settings);
      const oracle = computeOracle(scene, settings);
      compare(result, oracle, `${settings.kernel}/${settings.bandwidthMode}`);
      const slopes: number[] = [];
      const xs: number[] = [];
      let error = 0;
      for (let row = 0; row < 400; row++) {
        slopes.push(result.coefficients[row * 3 + 1]);
        xs.push(scene.positions[2 * row]);
        error += Math.abs(slopes[row] - xs[row]);
        expect(Math.abs(result.coefficients[row * 3 + 2] - 0.5)).toBeLessThan(0.6);
      }
      // The recovered slope follows the x coordinate: strong correlation and small mean error.
      expect(getCorrelation(slopes, xs)).toBeGreaterThan(0.97);
      expect(error / 400).toBeLessThan(0.4);
    }
    expect(fixture.rebuildCount).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('GPUGeographicallyWeightedRegression picks the oracle AICc argmin on a designed ladder', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // The slope changes quickly with x: a very small bandwidth overfits and a large one is biased.
  const scene = createScene(24, 21, (x, y) => [0.5, 1 + 0.8 * Math.sin(x), 0.3 * y], 0.3);
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUGeographicallyWeightedRegressionSettings = {
      kernel: 'gaussian',
      bandwidths: [0.3, 0.4, 0.5, 0.6, 1.2, 3]
    };
    const oracle = computeOracle(scene, settings);
    const sorted = oracle.bandwidthScores.filter(Number.isFinite).sort((a, b) => a - b);
    // The design must leave a clear margin so f32 rounding cannot change the argmin.
    expect(sorted[1] - sorted[0]).toBeGreaterThan(2);
    expect(oracle.selectedIndex).toBeGreaterThan(0);
    expect(oracle.selectedIndex).toBe(2);
    expect(oracle.selectedIndex).toBeLessThan(5);
    const result = await fixture.run(settings);
    expect(result.selectedBandwidth[0]).toBe(oracle.selectedIndex);
    expect(result.selectedBandwidth[1]).toBeCloseTo(oracle.selectedValue, 5);
    compare(result, oracle, 'ladder');
    // Unused ladder slots are NaN.
    expect(result.bandwidthScores.slice(6).every(Number.isNaN)).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('GPUGeographicallyWeightedRegression changes kernel, mode and ladder without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(14, 33, (x, y) => [1, 0.5 * x, 2 - 0.2 * y], 0.1);
  const fixture = createFixture(device, scene);
  try {
    const sequence: GPUGeographicallyWeightedRegressionSettings[] = [
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [3]},
      {kernel: 'gaussian', bandwidthMode: 'fixed', bandwidths: [1, 2, 3]},
      {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [25, 50]},
      {
        kernel: 'gaussian',
        bandwidthMode: 'adaptive',
        bandwidths: [20, 40, 60, 64, 2, 1000]
      },
      {
        kernel: 'bisquare',
        bandwidthMode: 'fixed',
        bandwidths: [2, 5, 9, 12, 1.5, 3, 4, 6]
      }
    ];
    for (const [index, settings] of sequence.entries()) {
      const result = await fixture.run(settings);
      compare(result, computeOracle(scene, settings), `sequence ${index}`);
    }
    expect(fixture.rebuildCount).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('GPUGeographicallyWeightedRegression is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(16, 41, (x, y) => [x, 1, y - 5], 0.2);
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUGeographicallyWeightedRegressionSettings = {
      kernel: 'gaussian',
      bandwidthMode: 'adaptive',
      bandwidths: [15, 30, 60]
    };
    const first = await fixture.run(settings);
    const second = await fixture.run(settings);
    for (const key of Object.keys(first) as (keyof GPUResult)[]) {
      const left = new Float32Array(first[key] as number[]);
      const right = new Float32Array(second[key] as number[]);
      expect(new Uint32Array(left.buffer), key).toEqual(new Uint32Array(right.buffer));
    }
  } finally {
    fixture.destroy();
  }
});

it('GPUGeographicallyWeightedRegression excludes masked and non-finite rows and flags singular fits', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(12, 51, (x, y) => [1, 1, x * 0.2], 0.1);
  scene.mask = new Uint32Array(144).fill(1);
  scene.mask[7] = 0;
  scene.response[30] = Number.NaN;
  scene.positions[2 * 90] = Number.POSITIVE_INFINITY;
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUGeographicallyWeightedRegressionSettings = {
      kernel: 'gaussian',
      bandwidths: [2, 3]
    };
    const result = await fixture.run(settings);
    for (const row of [7, 30, 90]) {
      expect(result.localStatus[row]).toBe(STATUS.EXCLUDED);
      expect(Number.isNaN(result.fitted[row])).toBe(true);
    }
    expect(result.summary[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.OBSERVATION_COUNT]).toBe(
      141
    );
    compare(result, computeOracle(scene, settings), 'masked');

    // A tiny fixed bandwidth leaves no neighbours: every candidate is singular.
    const tiny = await fixture.run({
      kernel: 'bisquare',
      bandwidths: [1e-4, 2e-4]
    });
    expect(tiny.summary[GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.HAS_VALID_CANDIDATE]).toBe(
      0
    );
    expect(tiny.bandwidthScores.every(Number.isNaN)).toBe(true);
    expect(tiny.localStatus[0]).toBe(STATUS.SINGULAR);
    expect(tiny.localStatus[7]).toBe(STATUS.EXCLUDED);
    // An invalid candidate is skipped while a valid one still wins.
    const mixed = await fixture.run({
      kernel: 'bisquare',
      bandwidths: [1e-4, 3]
    });
    expect(mixed.bandwidthScores[0]).toBeNaN();
    expect(mixed.selectedBandwidth[0]).toBe(1);
  } finally {
    fixture.destroy();
  }
});

it('GPUGeographicallyWeightedRegression handles 4096 locations with an adaptive ladder', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(64, 61, (x, y) => [1, 0.3 * x, 0.2 * y], 0.2);
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUGeographicallyWeightedRegressionSettings = {
      kernel: 'bisquare',
      bandwidthMode: 'adaptive',
      bandwidths: [8, 16, 24, 32, 40, 48, 56, 64]
    };
    const start = performance.now();
    const result = await fixture.run(settings);
    const elapsed = performance.now() - start;
    // eslint-disable-next-line no-console
    console.log(`GWR 4096 rows x 8 candidates: ${elapsed.toFixed(0)} ms`);
    expect(result.localStatus.every(status => status === STATUS.OK)).toBe(true);
    expect(Number.isFinite(result.selectedBandwidth[1])).toBe(true);
  } finally {
    fixture.destroy();
  }
}, 60000);
