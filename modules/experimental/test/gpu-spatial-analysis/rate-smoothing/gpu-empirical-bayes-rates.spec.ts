// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUEmpiricalBayesRates,
  GPU_EMPIRICAL_BAYES_SUMMARY
} from '../../../src/gpu-spatial-analysis/rate-smoothing';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeEmpiricalBayesOracle} from './empirical-bayes-oracle';

const GARBAGE = 0x7f7f7f7f;
const S = GPU_EMPIRICAL_BAYES_SUMMARY;

function createRates(rows: number) {
  let state = 99;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const populations = Float32Array.from({length: rows}, () =>
    Math.floor(50 + random() ** 3 * 20000)
  );
  const events = Float32Array.from({length: rows}, (_, row) =>
    Math.floor(populations[row] * (0.01 + 0.02 * random()) * (row % 7 === 0 ? 3 : 1))
  );
  return {events, populations, random};
}

async function runRates(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  input: {events: Float32Array; populations: Float32Array; mask?: Uint32Array}
) {
  const rows = input.events.length;
  const graph = new GPUCommandGraph(device, {id: 'empirical-bayes'});
  const buffers = [
    createInputBuffer(device, input.events),
    createInputBuffer(device, input.populations),
    input.mask && createInputBuffer(device, input.mask)
  ];
  const outputs = {
    standardized: createOutputBuffer(device, rows),
    smoothed: createOutputBuffer(device, rows),
    raw: createOutputBuffer(device, rows),
    summary: createOutputBuffer(device, S.length)
  };
  for (const buffer of Object.values(outputs)) {
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
  }
  graph.add(
    new GPUEmpiricalBayesRates({
      events: importGraphBuffer(graph, 'events', buffers[0]!, 'float32', rows),
      populations: importGraphBuffer(graph, 'populations', buffers[1]!, 'float32', rows),
      mask: buffers[2] && importGraphBuffer(graph, 'mask', buffers[2], 'uint32', rows),
      standardizedRates: importGraphBuffer(graph, 'z', outputs.standardized, 'float32', rows),
      smoothedRates: importGraphBuffer(graph, 'smoothed', outputs.smoothed, 'float32', rows),
      rawRates: importGraphBuffer(graph, 'raw', outputs.raw, 'float32', rows),
      summary: importGraphBuffer(graph, 'summary', outputs.summary, 'float32', S.length)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    standardized: await readFloat32(outputs.standardized, rows),
    smoothed: await readFloat32(outputs.smoothed, rows),
    raw: await readFloat32(outputs.raw, rows),
    summary: await readFloat32(outputs.summary, S.length)
  };
  compiled.destroy();
  for (const buffer of [...buffers, ...Object.values(outputs)]) {
    buffer?.destroy();
  }
  return result;
}

function expectClose(actual: number, expected: number, label: string, relative = 2e-3): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    1e-5 + relative * Math.abs(expected)
  );
}

it('GPUEmpiricalBayesRates reproduces the esda assuncao_rate docstring example', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const events = Float32Array.from([30, 25, 25, 15, 33, 21, 30, 20]);
  const populations = Float32Array.from([100, 100, 110, 90, 100, 90, 110, 90]);
  const result = await runRates(device, {events, populations});
  const expected = [1.03843594, -0.04099089, -0.56250375, -1.73061861];
  for (const [row, value] of expected.entries()) {
    expect(Math.abs(result.standardized[row] - value), `row ${row}`).toBeLessThan(1e-4);
  }
  expect(result.summary[S.count]).toBe(8);
});

it('GPUEmpiricalBayesRates matches the CPU oracle with masks and excluded rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {events, populations} = createRates(5000);
  populations[3] = 0;
  events[10] = NaN;
  populations[20] = -5;
  const mask = Uint32Array.from({length: 5000}, (_, row) => (row % 11 === 4 ? 0 : 1));
  const result = await runRates(device, {events, populations, mask});
  const oracle = computeEmpiricalBayesOracle({events, populations, mask});
  expect(result.summary[S.count]).toBe(oracle.count);
  expectClose(result.summary[S.pooledRate], oracle.pooledRate, 'pooled rate', 1e-4);
  expectClose(
    result.summary[S.weightedRateVariance],
    oracle.weightedRateVariance,
    'variance',
    1e-3
  );
  expectClose(result.summary[S.priorVariance], oracle.priorVariance, 'prior variance', 5e-3);
  expect(oracle.priorVariance).toBeGreaterThan(0);
  let nonzero = 0;
  for (let row = 0; row < events.length; row++) {
    expectClose(result.raw[row], oracle.raw[row], `raw ${row}`, 1e-5);
    expectClose(result.standardized[row], oracle.standardized[row], `z ${row}`, 5e-3);
    expectClose(result.smoothed[row], oracle.smoothed[row], `smoothed ${row}`, 5e-3);
    nonzero += Math.abs(result.standardized[row]) > 0.5 ? 1 : 0;
  }
  expect(nonzero).toBeGreaterThan(500);
  // Smoothing shrinks small-population rates toward the pooled rate.
  expect(Number.isNaN(result.standardized[3])).toBe(true);
  expect(Number.isNaN(result.smoothed[4])).toBe(true);
});

it('GPUEmpiricalBayesRates feeds GPUGlobalSpatialStatistics Moran in one graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 600;
  const {events, populations} = createRates(rows);
  // Ring-like lists with spatially varying risk so Moran's I is clearly positive.
  const offsets = new Uint32Array(rows + 1);
  const neighborList: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (const delta of [-2, -1, 1, 2]) {
      neighborList.push((row + delta + rows) % rows);
    }
    offsets[row + 1] = neighborList.length;
  }
  const neighbors = new Uint32Array(rows * 4);
  const weights = new Float32Array(rows * 4).fill(0.25);
  for (let row = 0; row < rows; row++) {
    const sorted = neighborList.slice(row * 4, row * 4 + 4).sort((a, b) => a - b);
    neighbors.set(sorted, row * 4);
  }
  for (let row = 0; row < rows; row++) {
    events[row] = Math.floor(populations[row] * (0.01 + 0.02 * (0.5 + 0.5 * Math.sin(row / 20))));
  }
  const oracle = computeEmpiricalBayesOracle({events, populations});
  const z = oracle.standardized;
  const mean = z.reduce((a, b) => a + b, 0) / rows;
  let cross = 0;
  let squares = 0;
  for (let row = 0; row < rows; row++) {
    squares += (z[row] - mean) ** 2;
    for (let slot = row * 4; slot < row * 4 + 4; slot++) {
      cross += 0.25 * (z[row] - mean) * (z[neighbors[slot]] - mean);
    }
  }
  const expectedMoran = (rows / rows) * (cross / squares);

  const graph = new GPUCommandGraph(device, {id: 'eb-moran'});
  const inputs = {
    events: createInputBuffer(device, events),
    populations: createInputBuffer(device, populations),
    offsets: createInputBuffer(device, offsets),
    neighbors: createInputBuffer(device, neighbors),
    weights: createInputBuffer(device, weights)
  };
  const results = createOutputBuffer(device, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
  results.write(new Uint32Array(results.byteLength / 4).fill(GARBAGE));
  const standardized = createTransientView(graph, 'standardized', 'float32', rows);
  graph.add(
    new GPUEmpiricalBayesRates({
      events: importGraphBuffer(graph, 'events', inputs.events, 'float32', rows),
      populations: importGraphBuffer(graph, 'populations', inputs.populations, 'float32', rows),
      standardizedRates: standardized
    })
  );
  graph.add(
    new GPUGlobalSpatialStatistics({
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', inputs.offsets, 'uint32', rows + 1),
        neighbors: importGraphBuffer(graph, 'neighbors', inputs.neighbors, 'uint32', rows * 4),
        weights: importGraphBuffer(graph, 'weights', inputs.weights, 'float32', rows * 4)
      },
      values: standardized,
      statistics: ['moran'],
      results: importGraphBuffer(
        graph,
        'results',
        results,
        'float32',
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
      )
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const values = await readFloat32(results, GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
  const moran =
    values[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
    ];
  expect(expectedMoran).toBeGreaterThan(0.3);
  expect(Math.abs(moran - expectedMoran)).toBeLessThan(5e-3);
  compiled.destroy();
  for (const buffer of [...Object.values(inputs), results]) {
    buffer.destroy();
  }
});
