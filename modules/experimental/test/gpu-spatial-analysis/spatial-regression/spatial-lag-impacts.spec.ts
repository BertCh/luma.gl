// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUSpatialLagImpacts} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-parameters';
import {
  getSpatialLagImpactsFromTraces,
  getSpatialWeightsTracesOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createNearestNeighborScene} from './spatial-regression-diagnostics-oracle';
import {createLagScene} from './spatial-two-stage-least-squares-oracle';
import {SPATIAL_WEIGHTS_TRACES_REFERENCE} from './spatial-weights-traces-reference';
import {requireTestDevice} from '../../utils/require-test-device';

const MAXIMUM_POWER = 40;

type Scene = ReturnType<typeof createLagScene>;
type Result = {status: number; table: Float32Array; impacts: Float32Array};

/** Runs 2SLS, the exact traces of the same W and the impacts node in one graph. */
async function runLagImpacts(device: Device, scene: Scene): Promise<Result> {
  const rows = scene.response.length;
  const k = scene.predictorCount;
  const tableLength = (k + 2) * 4;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'lag-impacts-graph'});
  const weights = {
    offsets: importGraphBuffer(
      graph,
      'offsets',
      track(createInputBuffer(device, scene.weights.offsets)),
      'uint32',
      scene.weights.offsets.length
    ),
    neighbors: importGraphBuffer(
      graph,
      'neighbors',
      track(createInputBuffer(device, scene.weights.neighbors)),
      'uint32',
      scene.weights.neighbors.length
    ),
    weights: importGraphBuffer(
      graph,
      'weights',
      track(createInputBuffer(device, scene.weights.weights)),
      'float32',
      scene.weights.weights.length
    )
  };
  const tableBuffer = track(createOutputBuffer(device, tableLength));
  const statusBuffer = track(createOutputBuffer(device, 1));
  const table = importGraphBuffer(graph, 'table', tableBuffer, 'float32', tableLength);
  const status = importGraphBuffer(graph, 'status', statusBuffer, 'uint32', 1);
  const powerTraces = importGraphBuffer(
    graph,
    'power-traces',
    track(createOutputBuffer(device, MAXIMUM_POWER)),
    'float32',
    MAXIMUM_POWER
  );
  const powerSums = importGraphBuffer(
    graph,
    'power-sums',
    track(createOutputBuffer(device, MAXIMUM_POWER)),
    'float32',
    MAXIMUM_POWER
  );
  const infinityNorm = importGraphBuffer(
    graph,
    'infinity-norm',
    track(createOutputBuffer(device, 1)),
    'float32',
    1
  );
  const impactsBuffer = track(createOutputBuffer(device, k * 3));
  graph.add(
    new GPUSpatialTwoStageLeastSquares({
      id: 'two-stage',
      weights,
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
      predictorCount: k,
      output: {
        table,
        summary: importGraphBuffer(
          graph,
          'summary',
          track(createOutputBuffer(device, GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH)),
          'float32',
          GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
        ),
        status: {code: status}
      }
    })
  );
  graph.add(
    new GPUSpatialWeightsTraces({
      id: 'traces',
      weights,
      maximumPower: MAXIMUM_POWER,
      mode: 'exact',
      output: {powerTraces, powerSums, infinityNorm}
    })
  );
  graph.add(
    new GPUSpatialLagImpacts({
      id: 'impacts',
      coefficients: table,
      coefficientStride: 4,
      rhoIndex: k + 1,
      predictorCount: k,
      rowCount: rows,
      maximumPower: MAXIMUM_POWER,
      traces: {powerTraces, powerSums, infinityNorm},
      status,
      output: {impacts: importGraphBuffer(graph, 'impacts-out', impactsBuffer, 'float32', k * 3)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    status: (await readUint32(statusBuffer, 1))[0],
    table: Float32Array.from(await readFloat32(tableBuffer, tableLength)),
    impacts: Float32Array.from(await readFloat32(impactsBuffer, k * 3))
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectClose(
  label: string,
  actual: number,
  expected: number,
  relative: number,
  floor: number
) {
  expect(Math.abs(actual - expected), `${label} ${actual} vs ${expected}`).toBeLessThanOrEqual(
    relative * Math.max(Math.abs(expected), floor)
  );
}

const referenceScenes = [
  ['lattice', createLagScene(12, 21, 0.5, true), SPATIAL_WEIGHTS_TRACES_REFERENCE.lattice],
  [
    'knn60',
    createNearestNeighborScene(60, 3, 13, 0.6, false),
    SPATIAL_WEIGHTS_TRACES_REFERENCE.knn60
  ],
  [
    'knn90',
    createNearestNeighborScene(90, 3, 29, 0.6, false),
    SPATIAL_WEIGHTS_TRACES_REFERENCE.knn90
  ]
] as const;

it('GPUSpatialLagImpacts turns a 2SLS fit into impacts that match spreg full multipliers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [name, scene, reference] of referenceScenes) {
    const actual = await runLagImpacts(device, scene);
    expect(actual.status).toBe(0);
    const rows = scene.response.length;
    const rho = actual.table[(scene.predictorCount + 1) * 4];
    // 1. Against the float64 mirror evaluated at the GPU's own rho and betas: only float32
    // arithmetic of the traces and the Horner sum separates them.
    const cpu = getSpatialWeightsTracesOnCPU(scene.weights, MAXIMUM_POWER);
    for (let predictor = 0; predictor < scene.predictorCount; predictor++) {
      const beta = actual.table[(1 + predictor) * 4];
      const expected = getSpatialLagImpactsFromTraces(
        rho,
        beta,
        cpu.powerTraces,
        cpu.powerSums,
        rows,
        cpu.infinityNorm
      );
      expectClose(
        `${name} direct ${predictor}`,
        actual.impacts[predictor * 3],
        expected.direct,
        1e-4,
        1e-2
      );
      expectClose(
        `${name} indirect ${predictor}`,
        actual.impacts[predictor * 3 + 1],
        expected.indirect,
        1e-3,
        1e-2
      );
      expectClose(
        `${name} total ${predictor}`,
        actual.impacts[predictor * 3 + 2],
        expected.total,
        1e-4,
        1e-2
      );
      // 2. Against spreg GM_Lag(spat_impacts='full'): the 2SLS rho and betas themselves differ from
      // spreg's by up to 5e-3 relative (see the 2SLS spec), and the total multiplier 1 / (1 - rho)
      // amplifies a rho error by at most (1 + |rho|) / (1 - |rho|)^2 < 6 here.
      const [direct, indirect, total] = reference.spreg.full.multipliers;
      const spregBeta = reference.spreg.betas[1 + predictor];
      expectClose(
        `${name} spreg direct ${predictor}`,
        actual.impacts[predictor * 3],
        direct * spregBeta,
        2e-2,
        0.02
      );
      expectClose(
        `${name} spreg indirect ${predictor}`,
        actual.impacts[predictor * 3 + 1],
        indirect * spregBeta,
        5e-2,
        0.02
      );
      expectClose(
        `${name} spreg total ${predictor}`,
        actual.impacts[predictor * 3 + 2],
        total * spregBeta,
        2e-2,
        0.02
      );
      // indirect = total - direct.
      expect(
        Math.abs(
          actual.impacts[predictor * 3 + 1] -
            (actual.impacts[predictor * 3 + 2] - actual.impacts[predictor * 3])
        )
      ).toBeLessThan(1e-3 * Math.abs(actual.impacts[predictor * 3 + 2]) + 1e-5);
    }
  }
});

it('GPUSpatialLagImpacts writes NaN for a failed fit, non-finite rho or rho outside the series range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const traces = SPATIAL_WEIGHTS_TRACES_REFERENCE.knn60;
  const run = async (rho: number, fitStatus: number, rhoBound?: number) => {
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const graph = new GPUCommandGraph(device, {id: 'impacts-nan-graph'});
    const maximumPower = traces.powerTraces.length;
    const coefficients = Float32Array.of(1, 0, 0, 0, 2, 0, 0, 0, -1, 0, 0, 0, rho, 0, 0, 0);
    const impactsBuffer = track(createOutputBuffer(device, 6));
    graph.add(
      new GPUSpatialLagImpacts({
        coefficients: importGraphBuffer(
          graph,
          'coefficients',
          track(createInputBuffer(device, coefficients)),
          'float32',
          16
        ),
        coefficientStride: 4,
        rhoIndex: 3,
        predictorCount: 2,
        rowCount: 60,
        maximumPower,
        rhoBound,
        traces: {
          powerTraces: importGraphBuffer(
            graph,
            'traces',
            track(createInputBuffer(device, Float32Array.from(traces.powerTraces))),
            'float32',
            maximumPower
          ),
          powerSums: importGraphBuffer(
            graph,
            'sums',
            track(createInputBuffer(device, Float32Array.from(traces.powerSums))),
            'float32',
            maximumPower
          ),
          // The reference traces are of row-standardized weights: ||W||_inf = 1.
          infinityNorm: importGraphBuffer(
            graph,
            'infinity-norm',
            track(createInputBuffer(device, Float32Array.of(1))),
            'float32',
            1
          )
        },
        status: importGraphBuffer(
          graph,
          'status',
          track(createInputBuffer(device, Uint32Array.of(fitStatus))),
          'uint32',
          1
        ),
        output: {impacts: importGraphBuffer(graph, 'impacts', impactsBuffer, 'float32', 6)}
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const impacts = Float32Array.from(await readFloat32(impactsBuffer, 6));
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
    return impacts;
  };
  const good = await run(0.3, 0);
  expect(Array.from(good).every(Number.isFinite)).toBe(true);
  // The 12-term series at rho = 0.3 agrees with the float64 mirror on the same truncated traces.
  const expected = getSpatialLagImpactsFromTraces(0.3, 2, traces.powerTraces, traces.powerSums, 60);
  expectClose('direct', good[0], expected.direct, 1e-4, 1e-3);
  expectClose('total', good[2], expected.total, 1e-4, 1e-3);
  const beta = -1;
  expectClose('second predictor total', good[5], beta * (expected.total / 2), 1e-4, 1e-3);
  for (const impacts of [
    await run(0.3, 3),
    await run(Number.NaN, 0),
    await run(1.2, 0),
    await run(0.3, 0, 0.2)
  ]) {
    expect(Array.from(impacts).every(Number.isNaN)).toBe(true);
  }
});
