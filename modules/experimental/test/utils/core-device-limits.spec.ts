// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Guards contributor kernels against the WebGPU core limits. Most contributor specs run on a
 * `'max'` feature-level device, so a kernel that binds more than 8 storage buffers per stage
 * passes there and fails on devices that grant only the core default. These specs run the
 * kernels that came closest to the limit on a `'core'` device and compare with the `'max'` run.
 */

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialEmpiricalBayesRates} from '../../src/gpu-spatial-analysis/rate-smoothing/gpu-spatial-empirical-bayes-rates';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues
} from '../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {GPUGeographicallyWeightedRegression} from '../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from './gpu-contributor-test-utils';

const LADDER_CAPACITY = 4;

function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/** Grid-indexed GWR: its final fit kernel binds the most storage buffers of any contributor. */
async function runGeographicallyWeightedRegression(device: Device): Promise<number[]> {
  const rowCount = 2048;
  const random = createRandom(987654321);
  const positions = new Float32Array(rowCount * 2);
  const predictors = new Float32Array(rowCount * 2);
  const response = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    positions[2 * row] = random() * 100;
    positions[2 * row + 1] = random() * 100;
    predictors[2 * row] = random() * 4 - 2;
    predictors[2 * row + 1] = random() * 4 - 2;
    response[row] = 1 + 0.5 * predictors[2 * row] + 0.2 * random();
  }
  const mask = Uint32Array.from({length: rowCount}, (_, row) => (row % 13 === 5 ? 0 : 1));
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => (buffers.push(buffer), buffer);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'core-limits-gwr-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
  });
  const coefficients = track(createOutputBuffer(device, rowCount * 3));
  const summary = track(createOutputBuffer(device, 6));
  const graph = new GPUCommandGraph(device, {id: 'core-limits-gwr'});
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, positions)),
        'float32x2',
        rowCount
      ),
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device, predictors)),
        'float32',
        rowCount * 2
      ),
      predictorCount: 2,
      response: importGraphBuffer(
        graph,
        'response',
        track(createInputBuffer(device, response)),
        'float32',
        rowCount
      ),
      mask: importGraphBuffer(
        graph,
        'mask',
        track(createInputBuffer(device, mask)),
        'uint32',
        rowCount
      ),
      parameters: parameterBuffer.importToGraph(graph),
      maximumBandwidthCount: LADDER_CAPACITY,
      output: {
        coefficients: importGraphBuffer(
          graph,
          'coefficients',
          coefficients,
          'float32',
          rowCount * 3
        ),
        summary: importGraphBuffer(graph, 'summary', summary, 'float32', 6)
      }
    })
  );
  const compiled = graph.compile();
  parameterBuffer.write(
    getGPUGeographicallyWeightedRegressionParameterValues(
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [12, 20]},
      LADDER_CAPACITY
    )
  );
  submitGraph(device, compiled, undefined);
  const result = [
    ...(await readFloat32(coefficients, rowCount * 3)),
    ...(await readFloat32(summary, 6))
  ];
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

/** Spatial empirical Bayes with a mask and all four outputs: the widest kernel of its family. */
async function runSpatialEmpiricalBayesRates(device: Device): Promise<number[]> {
  const rows = 256;
  const random = createRandom(7);
  const populations = Float32Array.from({length: rows}, () => Math.floor(50 + random() * 5000));
  const events = Float32Array.from({length: rows}, (_, row) =>
    Math.floor(populations[row]! * (0.01 + 0.03 * random()))
  );
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 === 3 ? 0 : 1));
  const offsets = Uint32Array.from({length: rows + 1}, (_, row) => 2 * row);
  const neighbors = Uint32Array.from({length: 2 * rows}, (_, slot) =>
    slot % 2 === 0 ? (slot / 2 + 1) % rows : (slot / 2 + rows - 1) % rows
  );
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => (buffers.push(buffer), buffer);
  const graph = new GPUCommandGraph(device, {id: 'core-limits-eb'});
  const output = {
    spatialRates: track(createOutputBuffer(device, rows)),
    smoothedRates: track(createOutputBuffer(device, rows)),
    priorMeans: track(createOutputBuffer(device, rows)),
    priorVariances: track(createOutputBuffer(device, rows))
  };
  graph.add(
    new GPUSpatialEmpiricalBayesRates({
      events: importGraphBuffer(
        graph,
        'events',
        track(createInputBuffer(device, events)),
        'float32',
        rows
      ),
      populations: importGraphBuffer(
        graph,
        'populations',
        track(createInputBuffer(device, populations)),
        'float32',
        rows
      ),
      weights: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          track(createInputBuffer(device, offsets)),
          'uint32',
          rows + 1
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          track(createInputBuffer(device, neighbors)),
          'uint32',
          neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'weights',
          track(createInputBuffer(device, new Float32Array(neighbors.length).fill(1))),
          'float32',
          neighbors.length
        )
      },
      mask: importGraphBuffer(
        graph,
        'mask',
        track(createInputBuffer(device, mask)),
        'uint32',
        rows
      ),
      spatialRates: importGraphBuffer(graph, 'spatialRates', output.spatialRates, 'float32', rows),
      smoothedRates: importGraphBuffer(
        graph,
        'smoothedRates',
        output.smoothedRates,
        'float32',
        rows
      ),
      priorMeans: importGraphBuffer(graph, 'priorMeans', output.priorMeans, 'float32', rows),
      priorVariances: importGraphBuffer(
        graph,
        'priorVariances',
        output.priorVariances,
        'float32',
        rows
      )
    } as ConstructorParameters<typeof GPUSpatialEmpiricalBayesRates>[0])
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: number[] = [];
  for (const buffer of Object.values(output)) {
    result.push(...(await readFloat32(buffer, rows)));
  }
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

const KERNELS: [string, (device: Device) => Promise<number[]>][] = [
  ['grid-indexed GPUGeographicallyWeightedRegression', runGeographicallyWeightedRegression],
  ['masked GPUSpatialEmpiricalBayesRates with all outputs', runSpatialEmpiricalBayesRates]
];

for (const [name, run] of KERNELS) {
  it(`${name} runs on a core device and matches the max device bitwise`, async () => {
    const coreDevice = await getWebGPUTestDevice('core');
    const maxDevice = await getWebGPUTestDevice('max');
    if (!coreDevice || !maxDevice) {
      return;
    }
    const coreResult = await run(coreDevice);
    const maxResult = await run(maxDevice);
    expect(coreResult.some(value => Number.isFinite(value))).toBe(true);
    expect(coreResult.map(value => (Number.isNaN(value) ? 'nan' : value))).toEqual(
      maxResult.map(value => (Number.isNaN(value) ? 'nan' : value))
    );
  });
}
