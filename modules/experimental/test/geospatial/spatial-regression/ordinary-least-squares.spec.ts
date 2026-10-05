// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUOrdinaryLeastSquares} from '../../../src/geospatial/spatial-regression/gpu-ordinary-least-squares';
import {
  getGPUOrdinaryLeastSquaresParameterValues,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from '../../../src/geospatial/spatial-regression/ordinary-least-squares-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  createOrdinaryLeastSquaresRandom,
  createOrdinaryLeastSquaresScene,
  fitOrdinaryLeastSquaresOnCPU,
  type OrdinaryLeastSquaresOracleResult,
  type OrdinaryLeastSquaresScene
} from './ordinary-least-squares-oracle';

type Result = {
  status: number;
  coefficients: Float32Array;
  standardErrors: Float32Array;
  tStatistics: Float32Array;
  summary: Float32Array;
  residuals: Float32Array;
  fitted: Float32Array;
};

type Fixture = {
  run(ridgeLambda?: number): Promise<Result>;
  /** Graph compilations after the first; per-frame changes must keep it at zero. */
  rebuildCount: number;
  destroy(): void;
};

function createFixture(
  device: Device,
  scene: OrdinaryLeastSquaresScene,
  tileRowCount?: number
): Fixture {
  const k = scene.predictorCount;
  const rows = scene.response.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'ols-parameters',
    format: 'float32',
    length: 1
  });
  const outputs = {
    coefficients: track(createOutputBuffer(device, k + 1)),
    standardErrors: track(createOutputBuffer(device, k + 1)),
    tStatistics: track(createOutputBuffer(device, k + 1)),
    summary: track(createOutputBuffer(device, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH)),
    status: track(createOutputBuffer(device, 1)),
    residuals: track(createOutputBuffer(device, rows)),
    fitted: track(createOutputBuffer(device, rows))
  };
  const graph = new GPUCommandGraph(device, {id: 'ols-graph'});
  graph.add(
    new GPUOrdinaryLeastSquares({
      id: 'ols',
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device, scene.predictors)),
        'float32',
        rows * k
      ),
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
      predictorCount: k,
      tileRowCount,
      output: {
        coefficients: importGraphBuffer(graph, 'o-coef', outputs.coefficients, 'float32', k + 1),
        standardErrors: importGraphBuffer(graph, 'o-se', outputs.standardErrors, 'float32', k + 1),
        tStatistics: importGraphBuffer(graph, 'o-t', outputs.tStatistics, 'float32', k + 1),
        summary: importGraphBuffer(
          graph,
          'o-summary',
          outputs.summary,
          'float32',
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
        ),
        status: importGraphBuffer(graph, 'o-status', outputs.status, 'uint32', 1),
        residuals: importGraphBuffer(graph, 'o-residuals', outputs.residuals, 'float32', rows),
        fitted: importGraphBuffer(graph, 'o-fitted', outputs.fitted, 'float32', rows)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(ridgeLambda = 0) {
      parameterBuffer.write(getGPUOrdinaryLeastSquaresParameterValues(ridgeLambda));
      submitGraph(device, compiled, undefined);
      const [status] = await readUint32(outputs.status, 1);
      return {
        status,
        coefficients: Float32Array.from(await readFloat32(outputs.coefficients, k + 1)),
        standardErrors: Float32Array.from(await readFloat32(outputs.standardErrors, k + 1)),
        tStatistics: Float32Array.from(await readFloat32(outputs.tStatistics, k + 1)),
        summary: Float32Array.from(
          await readFloat32(outputs.summary, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH)
        ),
        residuals: Float32Array.from(await readFloat32(outputs.residuals, rows)),
        fitted: Float32Array.from(await readFloat32(outputs.fitted, rows))
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

/**
 * Asserts `|actual - expected| <= relative * max(|expected|, floor)`, or both NaN. Values are
 * compared to a float64 oracle, so the tolerances cover f32 accumulation and solve error.
 */
function expectClose(
  label: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  relative: number,
  floor: number
): void {
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(actual[index], `${label}[${index}]`).toBeNaN();
      continue;
    }
    const tolerance = relative * Math.max(Math.abs(expected[index]), floor);
    expect(
      Math.abs(actual[index] - expected[index]),
      `${label}[${index}] ${actual[index]} vs ${expected[index]}`
    ).toBeLessThanOrEqual(tolerance);
  }
}

/** Compares a GPU result with the oracle: coefficients, errors, summary, diagnostics, columns. */
function expectParity(
  actual: Result,
  expected: OrdinaryLeastSquaresOracleResult,
  scale: {coefficients: number} = {coefficients: 1}
): void {
  expect(actual.status).toBe(expected.status);
  expect(actual.summary[0]).toBe(expected.rowCount);
  if (expected.status !== 0) {
    expect(Array.from(actual.coefficients).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual.residuals).every(Number.isNaN)).toBe(true);
    return;
  }
  expectClose('coefficients', actual.coefficients, expected.coefficients, 2e-3, scale.coefficients);
  expectClose('standardErrors', actual.standardErrors, expected.standardErrors, 5e-3, 1e-3);
  expectClose('tStatistics', actual.tStatistics, expected.tStatistics, 1e-2, 1);
  const s = actual.summary;
  const e = expected.summary;
  expectClose('rSquared', [s[1], s[2]], [e[1], e[2]], 1e-4, 1);
  expectClose('sigma2', [s[3]], [e[3]], 5e-3, 1e-6);
  expectClose('information', [s[4], s[5], s[6]], [e[4], e[5], e[6]], 1e-3, 1);
  expectClose('residualSums', [s[11], s[12]], [e[11], e[12]], 5e-3, 1e-6);
  expectClose('moments', [s[13], s[14]], [e[13], e[14]], 2e-2, 1);
  expectClose('jarqueBera', [s[7]], [e[7]], 3e-2, 1);
  expect(Math.abs(s[8] - e[8])).toBeLessThan(1e-2);
  expectClose('breuschPagan', [s[9]], [e[9]], 3e-2, 1);
  expect(Math.abs(s[10] - e[10])).toBeLessThan(1e-2);
  expect(s[15]).toBe(e[15]);
  for (let row = 0; row < expected.residuals.length; row++) {
    if (Number.isNaN(expected.residuals[row])) {
      expect(actual.residuals[row]).toBeNaN();
      expect(actual.fitted[row]).toBeNaN();
    }
  }
  const residualScale = Math.sqrt(expected.summary[3]) + 1e-6;
  for (let row = 0; row < expected.residuals.length; row++) {
    if (!Number.isNaN(expected.residuals[row])) {
      expect(Math.abs(actual.residuals[row] - expected.residuals[row])).toBeLessThan(
        0.02 * residualScale + 1e-3
      );
    }
  }
}

function bitsOf(result: Result): number[] {
  return [
    result.status,
    ...[
      result.coefficients,
      result.standardErrors,
      result.tStatistics,
      result.summary,
      result.residuals,
      result.fitted
    ].flatMap(values =>
      Array.from(new Uint32Array(values.buffer, values.byteOffset, values.length))
    )
  ];
}

it('GPUOrdinaryLeastSquares reproduces a textbook regression', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: OrdinaryLeastSquaresScene = {
    predictors: Float32Array.from([1, 2, 3, 4, 5]),
    response: Float32Array.from([2, 4, 5, 4, 5]),
    predictorCount: 1
  };
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  expect(result.status).toBe(0);
  expect(result.coefficients[0]).toBeCloseTo(2.2, 4);
  expect(result.coefficients[1]).toBeCloseTo(0.6, 5);
  expect(result.summary[1]).toBeCloseTo(0.6, 5);
  expect(result.summary[3]).toBeCloseTo(0.8, 4);
  expect(result.standardErrors[1]).toBeCloseTo(Math.sqrt(0.08), 4);
  expect(result.standardErrors[0]).toBeCloseTo(Math.sqrt(0.88), 4);
  expectParity(result, fitOrdinaryLeastSquaresOnCPU(scene));
  fixture.destroy();
});

it('GPUOrdinaryLeastSquares matches the float64 oracle on seeded noisy scenes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cases = [
    {
      seed: 11,
      rows: 3000,
      intercept: 2,
      slopes: [3, -1],
      noise: 0.5,
      tile: 50
    },
    {
      seed: 12,
      rows: 5000,
      intercept: -7,
      slopes: [0.25, 1.5, -2, 0.5],
      noise: 2,
      tile: undefined
    },
    {seed: 13, rows: 777, intercept: 100, slopes: [-4], noise: 3, tile: 7}
  ];
  for (const testCase of cases) {
    const scene = createOrdinaryLeastSquaresScene(
      testCase.seed,
      testCase.rows,
      testCase.intercept,
      testCase.slopes,
      testCase.noise,
      {withMask: true, withNonFinite: true}
    );
    const fixture = createFixture(device, scene, testCase.tile);
    const result = await fixture.run();
    const expected = fitOrdinaryLeastSquaresOnCPU(scene);
    expect(expected.status).toBe(0);
    expectParity(result, expected, {coefficients: 1});
    fixture.destroy();
  }
});

it('GPUOrdinaryLeastSquares recovers y = 2 + 3 x1 - x2 from a seeded LCG with small noise', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createOrdinaryLeastSquaresScene(21, 20000, 2, [3, -1], 0.05);
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  expect(result.status).toBe(0);
  expect(result.coefficients[0]).toBeCloseTo(2, 0);
  expect(Math.abs(result.coefficients[1] - 3)).toBeLessThan(1e-3);
  expect(Math.abs(result.coefficients[2] + 1)).toBeLessThan(1e-3);
  expect(result.summary[1]).toBeGreaterThan(0.9999);
  fixture.destroy();
});

it('GPUOrdinaryLeastSquares fits exactly linear data with R2 close to 1', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createOrdinaryLeastSquaresScene(31, 1000, 2, [3, -1], 0);
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  expect(result.status).toBe(0);
  expect(result.summary[1]).toBeGreaterThan(1 - 1e-5);
  expect(Math.abs(result.coefficients[1] - 3)).toBeLessThan(1e-3);
  expect(Math.abs(result.coefficients[2] + 1)).toBeLessThan(1e-3);
  expect(Number.isFinite(result.summary[4])).toBe(true);
  fixture.destroy();
});

it('GPUOrdinaryLeastSquares detects heteroskedastic residuals with Breusch-Pagan', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createOrdinaryLeastSquaresScene(41, 4000, 1, [2, 0.5], 0);
  const random = createOrdinaryLeastSquaresRandom(99);
  for (let row = 0; row < 4000; row++) {
    const spread = (scene.predictors[row * 2] - 40) * 0.5;
    scene.response[row] += (random() - 0.5) * 2 * spread;
  }
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  const expected = fitOrdinaryLeastSquaresOnCPU(scene);
  expectParity(result, expected);
  expect(result.summary[10]).toBeLessThan(1e-6);
  expect(expected.summary[10]).toBeLessThan(1e-6);
  // Homoskedastic uniform noise: a large p-value.
  const calm = createOrdinaryLeastSquaresScene(42, 4000, 1, [2, 0.5], 1);
  const calmFixture = createFixture(device, calm);
  const calmResult = await calmFixture.run();
  expectParity(calmResult, fitOrdinaryLeastSquaresOnCPU(calm));
  fixture.destroy();
  calmFixture.destroy();
});

it('GPUOrdinaryLeastSquares reports singular and short designs through status', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const duplicate = createOrdinaryLeastSquaresScene(51, 200, 1, [2, 0], 0.1);
  for (let row = 0; row < 200; row++) {
    duplicate.predictors[row * 2 + 1] = duplicate.predictors[row * 2];
  }
  let fixture = createFixture(device, duplicate);
  let result = await fixture.run();
  expect(result.status).toBe(1);
  expectParity(result, fitOrdinaryLeastSquaresOnCPU(duplicate));
  expect(Array.from(result.summary).slice(1).every(Number.isNaN)).toBe(true);
  fixture.destroy();

  const constant = createOrdinaryLeastSquaresScene(52, 100, 1, [2, 1], 0.1);
  for (let row = 0; row < 100; row++) {
    constant.predictors[row * 2 + 1] = 7;
  }
  fixture = createFixture(device, constant);
  expect((await fixture.run()).status).toBe(1);
  fixture.destroy();

  const short = createOrdinaryLeastSquaresScene(53, 3, 1, [2, 1], 0.1);
  fixture = createFixture(device, short);
  result = await fixture.run();
  expect(result.status).toBe(2);
  expect(result.summary[0]).toBe(3);
  fixture.destroy();

  // Masking down to two rows leaves n <= p even with many input rows.
  const masked = createOrdinaryLeastSquaresScene(54, 40, 1, [2], 0.1);
  masked.mask = Uint32Array.from({length: 40}, (_, row) => (row < 2 ? 1 : 0));
  fixture = createFixture(device, masked);
  result = await fixture.run();
  expect(result.status).toBe(2);
  expect(result.summary[0]).toBe(2);
  fixture.destroy();
});

it('GPUOrdinaryLeastSquares is bitwise deterministic across runs and graphs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createOrdinaryLeastSquaresScene(61, 6000, 3, [1, -2, 0.5], 1, {
    withMask: true,
    withNonFinite: true
  });
  const fixture = createFixture(device, scene, 37);
  const first = bitsOf(await fixture.run());
  const second = bitsOf(await fixture.run());
  expect(second).toEqual(first);
  const otherFixture = createFixture(device, scene, 37);
  expect(bitsOf(await otherFixture.run())).toEqual(first);
  fixture.destroy();
  otherFixture.destroy();
});

it('GPUOrdinaryLeastSquares changes the ridge penalty between encodings without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createOrdinaryLeastSquaresScene(71, 2000, 4, [3, -1, 2], 1);
  const fixture = createFixture(device, scene);
  let previousSlope = Infinity;
  for (const ridgeLambda of [0, 200, 5000, 100000, 0]) {
    const result = await fixture.run(ridgeLambda);
    const expected = fitOrdinaryLeastSquaresOnCPU({...scene, ridgeLambda});
    expect(result.status).toBe(0);
    expect(result.summary[15]).toBe(ridgeLambda);
    expectClose('ridge coefficients', result.coefficients, expected.coefficients, 5e-3, 0.05);
    expectClose('ridge rss', [result.summary[11]], [expected.summary[11]], 5e-3, 1e-3);
    if (ridgeLambda > 0 && ridgeLambda !== 5000) {
      expect(Math.abs(result.coefficients[1])).toBeLessThan(previousSlope + 1e-3);
    }
    previousSlope = ridgeLambda === 0 ? Infinity : Math.abs(result.coefficients[1]);
  }
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUOrdinaryLeastSquares supports the largest predictor count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const slopes = Array.from({length: 15}, (_, index) => ((index % 4) - 1.5) * 0.5);
  const scene = createOrdinaryLeastSquaresScene(81, 3000, 5, slopes, 1);
  const fixture = createFixture(device, scene);
  const result = await fixture.run();
  const expected = fitOrdinaryLeastSquaresOnCPU(scene);
  expect(result.status).toBe(expected.status);
  expect(expected.status).toBe(0);
  expectClose(
    'coefficients',
    result.coefficients.slice(1),
    expected.coefficients.slice(1),
    3e-2,
    0.1
  );
  expectClose('rSquared', [result.summary[1]], [expected.summary[1]], 1e-3, 1);
  fixture.destroy();
});
