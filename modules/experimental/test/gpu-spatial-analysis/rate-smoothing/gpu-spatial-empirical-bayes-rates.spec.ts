// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialEmpiricalBayesRates} from '../../../src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandomWeights} from '../spatial-weights/spatial-weights-algebra-oracle';
import {ESDA_SPATIAL_EB_LATTICE} from './esda-spatial-eb-fixture';
import {computeSpatialEmpiricalBayesOracle} from './spatial-empirical-bayes-oracle';

const GARBAGE = 0x7f7f7f7f;

async function run(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  input: {
    events: Float32Array;
    populations: Float32Array;
    offsets: number[];
    neighbors: number[];
    mask?: Uint32Array;
  }
) {
  const rows = input.events.length;
  const graph = new GPUCommandGraph(device, {id: 'spatial-eb'});
  const slots = Math.max(input.neighbors.length, 1);
  const neighbors = Uint32Array.from(input.neighbors.length ? input.neighbors : [0]);
  const buffers = [
    createInputBuffer(device, input.events),
    createInputBuffer(device, input.populations),
    createInputBuffer(device, Uint32Array.from(input.offsets)),
    createInputBuffer(device, neighbors),
    createInputBuffer(device, new Float32Array(slots).fill(1)),
    input.mask && createInputBuffer(device, input.mask)
  ];
  const outputs = {
    spatialRate: createOutputBuffer(device, rows),
    smoothed: createOutputBuffer(device, rows),
    priorMean: createOutputBuffer(device, rows),
    priorVariance: createOutputBuffer(device, rows)
  };
  for (const buffer of Object.values(outputs)) {
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
  }
  graph.add(
    new GPUSpatialEmpiricalBayesRates({
      events: importGraphBuffer(graph, 'events', buffers[0]!, 'float32', rows),
      populations: importGraphBuffer(graph, 'populations', buffers[1]!, 'float32', rows),
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', buffers[2]!, 'uint32', rows + 1),
        neighbors: importGraphBuffer(graph, 'neighbors', buffers[3]!, 'uint32', slots),
        weights: importGraphBuffer(graph, 'weights', buffers[4]!, 'float32', slots)
      },
      mask: buffers[5] && importGraphBuffer(graph, 'mask', buffers[5], 'uint32', rows),
      spatialRates: importGraphBuffer(graph, 'sr', outputs.spatialRate, 'float32', rows),
      smoothedRates: importGraphBuffer(graph, 'sm', outputs.smoothed, 'float32', rows),
      priorMeans: importGraphBuffer(graph, 'pm', outputs.priorMean, 'float32', rows),
      priorVariances: importGraphBuffer(graph, 'pv', outputs.priorVariance, 'float32', rows)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    spatialRate: await readFloat32(outputs.spatialRate, rows),
    smoothed: await readFloat32(outputs.smoothed, rows),
    priorMean: await readFloat32(outputs.priorMean, rows),
    priorVariance: await readFloat32(outputs.priorVariance, rows)
  };
  compiled.destroy();
  for (const buffer of [...buffers, ...Object.values(outputs)]) buffer?.destroy();
  return result;
}

function expectClose(actual: number[], expected: number[], label: string, relative = 2e-3): void {
  for (const [row, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(actual[row], `${label} row ${row}`).toBeNaN();
    } else {
      expect(
        Math.abs(actual[row] - value),
        `${label} row ${row}: ${actual[row]} vs ${value}`
      ).toBeLessThanOrEqual(1e-6 + relative * Math.abs(value));
    }
  }
}

it('GPUSpatialEmpiricalBayesRates matches the oracle on a hand-checked path graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Path 0-1-2-3 plus isolate 4.
  const events = Float32Array.from([10, 30, 20, 50, 7]);
  const populations = Float32Array.from([100, 200, 100, 400, 50]);
  const offsets = [0, 1, 3, 5, 6, 6];
  const neighbors = [1, 0, 2, 1, 3, 2];
  const result = await run(device, {events, populations, offsets, neighbors});
  // Row 1: members {1,0,2}: E = 60, B = 400, rate 0.15.
  expect(result.spatialRate[1]).toBeCloseTo(0.15, 6);
  // Isolate: own rate, variance clamps to zero, smoothed equals own rate.
  expect(result.spatialRate[4]).toBeCloseTo(0.14, 6);
  expect(result.smoothed[4]).toBeCloseTo(0.14, 6);
  expect(result.priorVariance[4]).toBe(0);
  const oracle = computeSpatialEmpiricalBayesOracle({events, populations, offsets, neighbors});
  expectClose(result.spatialRate, oracle.spatialRate, 'spatialRate');
  expectClose(result.smoothed, oracle.smoothed, 'smoothed');
  expectClose(result.priorMean, oracle.priorMean, 'priorMean');
  expectClose(result.priorVariance, oracle.priorVariance, 'priorVariance', 1e-2);
  expect(result.smoothed.every(value => Number.isFinite(value))).toBe(true);
});

it('GPUSpatialEmpiricalBayesRates matches the oracle with random weights, masks and bad rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 600;
  let state = 7;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const populations = Float32Array.from({length: rows}, () =>
    Math.floor(50 + random() ** 3 * 20000)
  );
  const events = Float32Array.from({length: rows}, (_, row) =>
    Math.floor(populations[row] * (0.01 + 0.03 * random()) * (row % 7 === 0 ? 3 : 1))
  );
  populations[5] = 0;
  events[9] = NaN;
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 === 3 ? 0 : 1));
  const csr = createRandomWeights(rows, 0.02, 21, true);
  const input = {events, populations, offsets: csr.offsets, neighbors: csr.neighbors, mask};
  const result = await run(device, input);
  const oracle = computeSpatialEmpiricalBayesOracle(input);
  expectClose(result.spatialRate, oracle.spatialRate, 'spatialRate');
  expectClose(result.smoothed, oracle.smoothed, 'smoothed');
  expectClose(result.priorMean, oracle.priorMean, 'priorMean');
  expectClose(result.priorVariance, oracle.priorVariance, 'priorVariance', 1e-2);
  expect(result.spatialRate[5]).toBeNaN();
  expect(result.smoothed[3]).toBeNaN();
  expect(result.smoothed.filter(value => Number.isFinite(value)).length).toBeGreaterThan(400);
});

it('GPUSpatialEmpiricalBayesRates reproduces esda 2.10 Spatial_Rate and Spatial_Empirical_Bayes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = ESDA_SPATIAL_EB_LATTICE;
  const result = await run(device, {
    events: Float32Array.from(fixture.events),
    populations: Float32Array.from(fixture.populations),
    offsets: [...fixture.offsets],
    neighbors: [...fixture.neighbors]
  });
  expectClose(result.spatialRate, [...fixture.spatialRate], 'esda spatialRate', 1e-5);
  expectClose(result.smoothed, [...fixture.smoothed], 'esda smoothed', 2e-4);
  expect(result.priorVariance.some(value => value > 0)).toBe(true);
});
