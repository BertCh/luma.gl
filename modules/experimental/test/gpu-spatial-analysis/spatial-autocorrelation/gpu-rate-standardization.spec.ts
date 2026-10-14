// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPUGlobalSpatialStatistics
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  GPUAssuncaoReisStandardization,
  GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-rate-standardization';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getReferenceWeights} from './spatial-autocorrelation-fixture';
import {
  computeAssuncaoReisOracle,
  getRateInputs,
  getRateReferenceZ,
  RATE_ROW_COUNT,
  toFloat64
} from './gpu-rate-standardization-fixture';
import {RATE_REFERENCE} from './gpu-rate-standardization-reference';
import {isClose} from './spatial-autocorrelation-harness';
import {requireTestDevice} from '../../utils/require-test-device';

const LAYOUT = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
const FIELD = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
const GARBAGE = 0x7f7f7f7f;

/** z agrees with esda within `1e-3 + 1e-3 |z|` (f32 sums, cancellation in `p - b`). */
const Z_ABSOLUTE = 1e-3;
const Z_RELATIVE = 1e-3;

/**
 * One graph: `GPUAssuncaoReisStandardization` -> (`GPULocalMoran`, `GPUGlobalSpatialStatistics`)
 * over row-standardized weights with `islands: 'keep'` (esda's convention), the composition recipe
 * for `Moran_Rate` and `Moran_Local_Rate`.
 */
function createPipeline(device: Device, rows: number, withMask: boolean) {
  const csr = getReferenceWeights({rowStandardize: true});
  const capacity = csr.neighbors.length;
  const buffers = {
    events: createOutputBuffer(device, rows),
    populations: createOutputBuffer(device, rows),
    mask: withMask ? createOutputBuffer(device, rows) : undefined,
    offsets: createInputBuffer(device, csr.offsets),
    neighbors: createInputBuffer(device, csr.neighbors),
    weights: createInputBuffer(device, csr.weights),
    statistics: createOutputBuffer(device, GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH),
    invalidCount: createOutputBuffer(device, 1),
    zScores: createOutputBuffer(device, rows),
    localI: createOutputBuffer(device, rows),
    results: createOutputBuffer(device, LAYOUT.length)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'rate-moran-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    values: getGPUSpatialAutocorrelationParameterValues()
  });
  const graph = new GPUCommandGraph(device, {id: 'rate-moran-test'});
  const weights = {
    offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', rows + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity),
    information: {rowCount: rows, transform: 'row' as const, symmetric: 'unknown' as const}
  };
  const mask = buffers.mask && importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', rows);
  const standardizedRates = createTransientView(graph, 'standardized-rates', 'float32', rows);
  graph.add(
    new GPUAssuncaoReisStandardization({
      events: importGraphBuffer(graph, 'events', buffers.events, 'float32', rows),
      populations: importGraphBuffer(graph, 'populations', buffers.populations, 'float32', rows),
      mask,
      output: {
        standardizedRates,
        statistics: importGraphBuffer(
          graph,
          'rate-statistics',
          buffers.statistics,
          'float32',
          GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
        ),
        status: {
          invalidCount: importGraphBuffer(graph, 'invalid-count', buffers.invalidCount, 'uint32', 1)
        }
      }
    })
  );
  graph.add(
    new GPUGlobalSpatialStatistics({
      weights,
      values: standardizedRates,
      statistics: ['moran'],
      islands: 'keep',
      output: {
        summary: importGraphBuffer(graph, 'results', buffers.results, 'float32', LAYOUT.length)
      }
    })
  );
  graph.add(
    new GPULocalMoran({
      weights,
      values: standardizedRates,
      parameters: parameterBuffer.importToGraph(graph),
      islands: 'keep',
      output: {
        zScores: importGraphBuffer(graph, 'z-scores', buffers.zScores, 'float32', rows),
        localI: importGraphBuffer(graph, 'local-i', buffers.localI, 'float32', rows)
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(events: Float32Array, populations: Float32Array, rowMask?: Uint32Array) {
      buffers.events.write(events);
      buffers.populations.write(populations);
      if (rowMask && buffers.mask) {
        buffers.mask.write(rowMask);
      }
      for (const buffer of [
        buffers.statistics,
        buffers.invalidCount,
        buffers.localI,
        buffers.results
      ]) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const results = await readFloat32(buffers.results, LAYOUT.length);
      return {
        statistics: await readFloat32(
          buffers.statistics,
          GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
        ),
        invalidCount: (await readUint32(buffers.invalidCount, 1))[0],
        localI: await readFloat32(buffers.localI, rows),
        globalMoran: {
          statistic: results[LAYOUT.moran + FIELD.statistic],
          expected: results[LAYOUT.moran + FIELD.expected],
          zNormality: results[LAYOUT.moran + FIELD.zNormality],
          zRandomization: results[LAYOUT.moran + FIELD.zRandomization]
        }
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of Object.values(buffers)) {
        buffer?.destroy();
      }
    }
  };
}

/** Standardization alone, with the rates written to a caller buffer. */
function createStandardizer(device: Device, rows: number, withMask: boolean) {
  const buffers = {
    events: createOutputBuffer(device, rows),
    populations: createOutputBuffer(device, rows),
    mask: withMask ? createOutputBuffer(device, rows) : undefined,
    rates: createOutputBuffer(device, rows),
    statistics: createOutputBuffer(device, GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH),
    invalidCount: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'rate-standardization-test'});
  graph.add(
    new GPUAssuncaoReisStandardization({
      events: importGraphBuffer(graph, 'events', buffers.events, 'float32', rows),
      populations: importGraphBuffer(graph, 'populations', buffers.populations, 'float32', rows),
      mask: buffers.mask && importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', rows),
      output: {
        standardizedRates: importGraphBuffer(graph, 'rates', buffers.rates, 'float32', rows),
        statistics: importGraphBuffer(
          graph,
          'statistics',
          buffers.statistics,
          'float32',
          GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
        ),
        status: {
          invalidCount: importGraphBuffer(graph, 'invalid-count', buffers.invalidCount, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(events: Float32Array, populations: Float32Array, rowMask?: Uint32Array) {
      buffers.events.write(events);
      buffers.populations.write(populations);
      if (rowMask && buffers.mask) {
        buffers.mask.write(rowMask);
      }
      for (const buffer of [buffers.rates, buffers.statistics, buffers.invalidCount]) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const zBits = await readUint32(buffers.rates, rows);
      return {
        zBits,
        z: await readFloat32(buffers.rates, rows),
        statistics: await readFloat32(
          buffers.statistics,
          GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
        ),
        invalidCount: (await readUint32(buffers.invalidCount, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of Object.values(buffers)) {
        buffer?.destroy();
      }
    }
  };
}

function expectRates(actual: number[], expected: number[], label: string): void {
  expect(actual.length, label).toBe(expected.length);
  for (const [row, value] of expected.entries()) {
    if (!isClose(actual[row], value, Z_ABSOLUTE, Z_RELATIVE)) {
      throw new Error(`${label}: row ${row} z ${actual[row]} != esda ${value}`);
    }
  }
}

function expectStatistics(actual: number[], expected: number[], label: string): void {
  expect(actual[0], `${label} n`).toBe(expected[0]);
  expect(
    isClose(actual[1], expected[1], 1e-9, 1e-5),
    `${label} b ${actual[1]} vs ${expected[1]}`
  ).toBe(true);
  expect(
    isClose(actual[2], expected[2], 1e-9, 1e-3),
    `${label} s2 ${actual[2]} vs ${expected[2]}`
  ).toBe(true);
  // a is a difference of two similar numbers; allow it a fraction of s2.
  expect(
    Math.abs(actual[3] - expected[3]),
    `${label} a ${actual[3]} vs ${expected[3]}`
  ).toBeLessThanOrEqual(1e-3 * expected[2] + 1e-3 * Math.abs(expected[3]));
  expect(isClose(actual[4], expected[4], 0, 1e-6), `${label} population`).toBe(true);
}

// Oracle: test/oracles/statistics-rate-standardization.py (esda 2.10.0 / libpysal 4.15.0)
it('GPUAssuncaoReisStandardization matches esda assuncao_rate, including a negative prior variance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const standardizer = createStandardizer(device, RATE_ROW_COUNT, false);
  for (const dataset of ['full', 'missing', 'under'] as const) {
    const {events, populations} = getRateInputs(dataset);
    const result = await standardizer.run(events, populations);
    expectRates(result.z, getRateReferenceZ(dataset), dataset);
    const reference =
      dataset === 'full'
        ? RATE_REFERENCE.fullStatistics
        : dataset === 'missing'
          ? RATE_REFERENCE.missingStatistics
          : RATE_REFERENCE.underStatistics;
    expectStatistics(result.statistics, [...reference], dataset);
    expect(result.invalidCount, `${dataset} invalidCount`).toBe(
      dataset === 'missing' ? Object.keys(RATE_REFERENCE.invalidRows).length : 0
    );
  }
  // Missing rows write NaN, and a second run is bitwise identical (fixed-order sums).
  const {events, populations} = getRateInputs('missing');
  const first = await standardizer.run(events, populations);
  const second = await standardizer.run(events, populations);
  expect(second.zBits).toEqual(first.zBits);
  for (const row of Object.keys(RATE_REFERENCE.invalidRows).map(Number)) {
    expect(first.z[row]).toBeNaN();
  }
  standardizer.destroy();
});

it('GPUAssuncaoReisStandardization treats masked rows as outside the analysis', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const standardizer = createStandardizer(device, RATE_ROW_COUNT, true);
  const {events, populations} = getRateInputs('full');
  const mask = new Uint32Array(RATE_ROW_COUNT).fill(1);
  const maskedRows = [1, 2, 17, 58];
  for (const row of maskedRows) {
    mask[row] = 0;
  }
  const reduced = Float32Array.from(events);
  for (const row of maskedRows) {
    reduced[row] = NaN;
  }
  // The CPU oracle is pinned to esda in the node spec; masked rows equal missing rows.
  const oracle = computeAssuncaoReisOracle(reduced, populations);
  const result = await standardizer.run(events, populations, mask);
  expectRates(result.z, oracle.z, 'masked');
  expectStatistics(result.statistics, oracle.statistics, 'masked');
  expect(result.invalidCount).toBe(0);
  standardizer.destroy();
});

it('EB Moran for rates: standardization -> GPUGlobalSpatialStatistics and GPULocalMoran match esda Moran_Rate and Moran_Local_Rate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pipeline = createPipeline(device, RATE_ROW_COUNT, false);
  for (const dataset of ['full', 'missing'] as const) {
    const {events, populations} = getRateInputs(dataset);
    const result = await pipeline.run(events, populations);
    const moran = dataset === 'full' ? RATE_REFERENCE.fullMoran : RATE_REFERENCE.missingMoran;
    const localI = toFloat64(
      dataset === 'full' ? RATE_REFERENCE.fullLocalI : RATE_REFERENCE.missingLocalI
    );
    const global = result.globalMoran;
    expect(Math.abs(global.statistic - moran.I), `${dataset} I`).toBeLessThanOrEqual(
      1e-3 + 1e-3 * Math.abs(moran.I)
    );
    expect(Math.abs(global.expected - moran.EI), `${dataset} EI`).toBeLessThanOrEqual(1e-6);
    expect(
      Math.abs(global.zNormality - moran.zNorm),
      `${dataset} z_norm ${global.zNormality} vs ${moran.zNorm}`
    ).toBeLessThanOrEqual(2e-2 + 5e-3 * Math.abs(moran.zNorm));
    expect(
      Math.abs(global.zRandomization - moran.zRand),
      `${dataset} z_rand ${global.zRandomization} vs ${moran.zRand}`
    ).toBeLessThanOrEqual(2e-2 + 5e-3 * Math.abs(moran.zRand));
    for (const [row, expected] of localI.entries()) {
      if (!isClose(result.localI[row], expected, 2e-3, 2e-3)) {
        throw new Error(`${dataset}: row ${row} local I ${result.localI[row]} != esda ${expected}`);
      }
    }
  }
  pipeline.destroy();
});
