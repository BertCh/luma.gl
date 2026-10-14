// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  GPUTerrainFlow,
  GPUTerrainStreamOrder,
  getGPUTerrainFlowParameterValues
} from '../../../src/gpu-terrain/hydrology';
import {getTerrainFlowTourRoundCount} from '../../../src/gpu-terrain/hydrology/terrain-flow-euler-tour';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {toBits} from '../terrain-test-utils';
import {TERRAIN_FLOW_DEMS as DEMS} from './terrain-flow-oracle';
import {
  computeStrahlerOrderRecursive,
  getCodes,
  makeCaterpillar,
  makeRandomForest,
  makeStem,
  makeStrahlerTree,
  modelStrahlerOrder,
  TOUR_NONE
} from './terrain-flow-euler-tour-model';

/**
 * GPU specs of the Euler-tour D8 accumulation and Strahler order. The CPU model in
 * `terrain-flow-euler-tour-model.ts` repeats the same algorithm and is checked by the node specs;
 * these specs check the kernels against the model and the oracles.
 */

async function runAccumulation(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: {runoff?: Float32Array; maximumAccumulationIterations?: number} = {}
) {
  const cellCount = width * height;
  const grid = makeGPURasterMetadata({width, height});
  const elevationBuffer = createInputBuffer(device, elevation);
  const runoffBuffer = options.runoff ? createInputBuffer(device, options.runoff) : undefined;
  const outputs = {
    accumulation: createOutputBuffer(device, cellCount),
    streams: createOutputBuffer(device, cellCount),
    converged: createOutputBuffer(device, 1),
    rounds: createOutputBuffer(device, 1)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'tour-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowParameterValues({streamThreshold: 3}, grid)
  });
  const graph = new GPUCommandGraph(device, {id: 'tour-accumulation-test'});
  graph.add(
    new GPUTerrainFlow({
      grid,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount),
        grid,
        {id: 'elevation'}
      ),
      parameters: settings.importToGraph(graph),
      maximumAccumulationIterations: options.maximumAccumulationIterations,
      runoff: runoffBuffer
        ? importGraphBuffer(graph, 'runoff', runoffBuffer, 'float32', cellCount)
        : undefined,
      output: {
        accumulation: importGraphBuffer(graph, 'acc', outputs.accumulation, 'float32', cellCount),
        streams: importGraphBuffer(graph, 'streams', outputs.streams, 'uint32', cellCount),
        status: {
          accumulationConverged: importGraphBuffer(graph, 'conv', outputs.converged, 'uint32', 1),
          accumulationIterationCount: importGraphBuffer(
            graph,
            'rounds',
            outputs.rounds,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    accumulation: Float32Array.from(await readFloat32(outputs.accumulation, cellCount)),
    streams: Uint32Array.from(await readUint32(outputs.streams, cellCount)),
    converged: (await readUint32(outputs.converged, 1))[0],
    rounds: (await readUint32(outputs.rounds, 1))[0]
  };
  compiled.destroy();
  settings.destroy();
  elevationBuffer.destroy();
  runoffBuffer?.destroy();
  for (const buffer of Object.values(outputs)) buffer.destroy();
  return result;
}

it('D8 cells accumulation is exact on a 1 x 100000 stem that the 64-round pull left unresolved', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const length = 100000;
  const strip = Float32Array.from({length}, (_, row) => length - row);
  const result = await runAccumulation(device, strip, 1, length);
  expect(result.converged).toBe(1);
  expect(result.rounds).toBeLessThanOrEqual(getTerrainFlowTourRoundCount(length));
  for (let row = 0; row < length; row++) {
    if (result.accumulation[row] !== row + 1) {
      throw new Error(`row ${row}: ${result.accumulation[row]}`);
    }
  }
  // streamThreshold 3: only the first two cells are below it.
  expect(result.streams[0]).toBe(0);
  expect(result.streams[1]).toBe(0);
  expect(result.streams[2]).toBe(1);
  expect(result.streams.every(word => word !== GPU_TERRAIN_FLOW_STREAM_UNRESOLVED)).toBe(true);
  device.destroy();
});

it('the tour and the pull path write the same bits, and the iteration limit no longer matters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [width, height] of [
    [37, 29],
    [64, 64]
  ]) {
    const elevation = DEMS.noise(width, height);
    const tour = await runAccumulation(device, elevation, width, height, {
      maximumAccumulationIterations: 1
    });
    // Unit runoff selects the pull path with the same weights.
    const pull = await runAccumulation(device, elevation, width, height, {
      runoff: new Float32Array(width * height).fill(1),
      maximumAccumulationIterations: 256
    });
    expect(tour.converged).toBe(1);
    expect(pull.converged).toBe(1);
    expect(toBits(tour.accumulation)).toEqual(toBits(pull.accumulation));
    expect(Array.from(tour.streams)).toEqual(Array.from(pull.streams));
  }
  device.destroy();
});

async function runStreamOrder(
  device: Device,
  width: number,
  height: number,
  flowDirections: Uint32Array,
  streams: Uint32Array
) {
  const cellCount = width * height;
  const inputs = [createInputBuffer(device, flowDirections), createInputBuffer(device, streams)];
  const outputs = {
    order: createOutputBuffer(device, cellCount),
    converged: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'tour-order-test'});
  graph.add(
    new GPUTerrainStreamOrder({
      grid: makeGPURasterMetadata({width, height}),
      flowDirections: importGraphBuffer(graph, 'directions', inputs[0], 'uint32', cellCount),
      streams: importGraphBuffer(graph, 'streams', inputs[1], 'uint32', cellCount),
      output: {
        streamOrder: importGraphBuffer(graph, 'order', outputs.order, 'uint32', cellCount),
        status: {converged: importGraphBuffer(graph, 'conv', outputs.converged, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    order: Uint32Array.from(await readUint32(outputs.order, cellCount)),
    converged: (await readUint32(outputs.converged, 1))[0]
  };
  compiled.destroy();
  for (const buffer of [...inputs, ...Object.values(outputs)]) buffer.destroy();
  return result;
}

it('GPUTerrainStreamOrder gives Strahler order 5 on a complete binary tree', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (let order = 1; order <= 5; order++) {
    const tree = makeStrahlerTree(order);
    const streams = new Uint32Array(tree.parents.length);
    for (let cell = 0; cell < streams.length; cell++) {
      streams[cell] = tree.parents[cell] !== TOUR_NONE || cell === tree.root ? 1 : 0;
    }
    const codes = getCodes(tree);
    const result = await runStreamOrder(device, tree.width, tree.height, codes, streams);
    expect(result.converged).toBe(1);
    expect(result.order[tree.root]).toBe(order);
    expect(Array.from(result.order)).toEqual(
      Array.from(
        computeStrahlerOrderRecursive({
          width: tree.width,
          height: tree.height,
          flowDirections: codes,
          streams,
          unresolvedStream: GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
        })
      )
    );
  }
  device.destroy();
});

it('GPUTerrainStreamOrder matches the model on a caterpillar, a long stem and random forests', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = [
    ['caterpillar', makeCaterpillar(300)],
    ['stem', makeStem(20000)],
    ...[1, 2, 3].map(
      seed => [`random ${seed}`, makeRandomForest(40, 30, seed, 0.1).forest] as const
    )
  ] as const;
  for (const [name, forest] of cases) {
    const codes = getCodes(forest);
    const streams = new Uint32Array(codes.length).fill(1);
    const result = await runStreamOrder(device, forest.width, forest.height, codes, streams);
    const model = modelStrahlerOrder({
      width: forest.width,
      height: forest.height,
      flowDirections: codes,
      streams,
      unresolvedStream: GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
    });
    expect(result.converged, name).toBe(1);
    expect(Array.from(result.order), name).toEqual(Array.from(model.order));
  }
  device.destroy();
});
