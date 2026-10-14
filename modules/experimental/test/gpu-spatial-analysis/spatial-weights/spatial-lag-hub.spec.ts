// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {computeLagOracle, createSeededRandom, type OracleCSR} from './spatial-weights-oracle';

const HUB_DEGREE = 100_000;
const ROWS = 120_000;

/** One hub row linked to 1e5 leaves, every leaf linked back to the hub, a few leaf-leaf links. */
function createHubCSR(): OracleCSR {
  const random = createSeededRandom(5);
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < ROWS; row++) {
    if (row === 0) {
      for (let leaf = 1; leaf <= HUB_DEGREE; leaf++) neighbors.push(leaf);
    } else {
      neighbors.push(0);
      if (random() < 0.3) neighbors.push(1 + Math.floor(random() * (ROWS - 1)));
    }
    while (weights.length < neighbors.length) weights.push(Math.fround(0.5 + random()));
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors, weights, distances: []};
}

type Views = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  weights: GraphDataView<'float32'>;
  values: GraphDataView<'float32'>;
  output: GraphDataView<'float32'>;
};

/** The pre-migration thread-per-row lag kernel, kept here only as the A/B baseline. */
function getRowLoopProducer(views: Views): GPUCommandNodeProducer {
  return {
    getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) =>
      [
        createWGSLKernelNode<Parameters>(graph, {
          id: 'row-loop-lag',
          operation: 'GPUSpatialLagBaseline',
          bindings: [
            {name: 'offsets', view: views.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: views.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: views.weights, type: 'f32', access: 'read'},
            {name: 'values', view: views.values, type: 'f32', access: 'read'},
            {name: 'output', view: views.output, type: 'f32', access: 'read_write'}
          ],
          invocationCount: ROWS,
          declarations: `const ROWS: u32 = ${ROWS}u;`,
          body: `var lag = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS) {
      lag += weights[weightsOffset + slot] * values[valuesOffset + neighbor];
    }
  }
  output[outputOffset + index] = lag;`
        }) as GPUCommandNode<Parameters>
      ] as const
  } as GPUCommandNodeProducer;
}

async function runLag(
  device: Device,
  csr: OracleCSR,
  values: Float32Array,
  implementation: 'load-balanced' | 'row-loop'
): Promise<{lag: number[]; milliseconds: number}> {
  const graph = new GPUCommandGraph<undefined>(device, {id: `lag-${implementation}`});
  const buffers = [
    createInputBuffer(device, Uint32Array.from(csr.offsets)),
    createInputBuffer(device, Uint32Array.from(csr.neighbors)),
    createInputBuffer(device, Float32Array.from(csr.weights)),
    createInputBuffer(device, values),
    createOutputBuffer(device, ROWS)
  ];
  const views: Views = {
    offsets: importGraphBuffer(graph, 'offsets', buffers[0], 'uint32', csr.offsets.length),
    neighbors: importGraphBuffer(graph, 'neighbors', buffers[1], 'uint32', csr.neighbors.length),
    weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', csr.weights.length),
    values: importGraphBuffer(graph, 'values', buffers[3], 'float32', ROWS),
    output: importGraphBuffer(graph, 'output', buffers[4], 'float32', ROWS)
  };
  graph.add(
    implementation === 'row-loop'
      ? getRowLoopProducer(views)
      : new GPUSpatialLag({
          values: views.values,
          weights: {offsets: views.offsets, neighbors: views.neighbors, weights: views.weights},
          output: views.output
        })
  );
  const compiled = await graph.compileAsync();
  // Warm-up, then one timed submission through the readback.
  submitGraph(device, compiled, undefined);
  await readFloat32(buffers[4], 1);
  const start = performance.now();
  submitGraph(device, compiled, undefined);
  const lag = await readFloat32(buffers[4], ROWS);
  const milliseconds = performance.now() - start;
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return {lag, milliseconds};
}

it('GPUSpatialLag matches the oracle on a 1e5-neighbor hub (same-run A/B against the row loop)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = createHubCSR();
  const random = createSeededRandom(9);
  const values = new Float32Array(ROWS).map(() => Math.fround(random() * 2 - 1));
  const expected = computeLagOracle(csr, values);
  const balanced = await runLag(device, csr, values, 'load-balanced');
  const rowLoop = await runLag(device, csr, values, 'row-loop');
  // Tolerance: as the utility spec, 1e-5 * sum |w y| per row (about 120 rounding steps of 2^-24).
  for (let row = 0; row < ROWS; row++) {
    let magnitude = 0;
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      magnitude += Math.abs(csr.weights[slot] * values[csr.neighbors[slot]]);
    }
    if (Math.abs(balanced.lag[row] - expected[row]) > 1e-5 * magnitude) {
      expect(balanced.lag[row], `row ${row}`).toBeCloseTo(expected[row], 5);
    }
  }
  // A/B report (decision 3): the hub row serializes one thread of the row loop.
  console.info(
    `spatial lag hub A/B: load-balanced ${balanced.milliseconds.toFixed(2)} ms, ` +
      `row loop ${rowLoop.milliseconds.toFixed(2)} ms`
  );
  expect(rowLoop.lag[0]).toBeCloseTo(expected[0], 1);
});

it('normalized spatial lag shares one masked weight sum across all value columns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const output = rig.output('float32', 12);
    rig.run(
      new GPUSpatialLag({
        weights: rig.weights({
          offsets: [0, 3, 5, 5, 7],
          neighbors: [1, 2, 3, 0, 3, 1, 2],
          weights: [1, 9, 3, 1, 1, 2, 7]
        }),
        values: rig.input(
          Float32Array.from([1, 10, -1, 2, 20, -2, 999, 999, 999, 4, 40, -4]),
          'float32'
        ),
        columnCount: 3,
        mask: rig.input(Uint32Array.from([1, 1, 0, 1]), 'uint32'),
        normalize: true,
        output
      })
    );
    // All sums and quotients are exactly representable, including the masked empty row.
    expect(await rig.readFloat(output)).toEqual([3.5, 35, -3.5, 2.5, 25, -2.5, 0, 0, 0, 2, 20, -2]);
  } finally {
    rig.destroy();
  }
});
