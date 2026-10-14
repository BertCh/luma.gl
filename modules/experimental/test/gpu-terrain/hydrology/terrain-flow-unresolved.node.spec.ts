// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_DRAINAGE_NONE,
  GPU_TERRAIN_DRAINAGE_UNRESOLVED,
  GPU_TERRAIN_FLOW_NONE,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  GPUTerrainFlow,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainStreamOrder
} from '../../../src/gpu-terrain/hydrology';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {computeTerrainFlow, TERRAIN_FLOW_DEMS} from './terrain-flow-oracle';

const WIDTH = 31;
const HEIGHT = 23;
const THRESHOLD = 20;

/** Twin of the `GPUTerrainFlow` streams kernel; `'previous'` is the code before review C-03. */
function getStreamsTwin(
  elevation: ArrayLike<number>,
  accumulation: ArrayLike<number>,
  rule: 'current' | 'previous'
): Uint32Array {
  return Uint32Array.from(accumulation, (value, cell) => {
    if (Number.isFinite(value)) return value >= THRESHOLD ? 1 : 0;
    const isValid = Number.isFinite(elevation[cell]);
    return rule === 'current' && isValid ? GPU_TERRAIN_FLOW_STREAM_UNRESOLVED : 0;
  });
}

/** Twin of the HAND drainage cell: first non-zero stream word on the D8 receiver path. */
function getDrainageTwin(receivers: ArrayLike<number>, streams: ArrayLike<number>): Uint32Array {
  return Uint32Array.from(receivers, (_, start) => {
    let cell = start;
    for (let step = 0; step < receivers.length; step++) {
      if (streams[cell] !== 0) {
        return streams[cell] === GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
          ? GPU_TERRAIN_DRAINAGE_UNRESOLVED
          : cell;
      }
      if (receivers[cell] === GPU_TERRAIN_FLOW_NONE) break;
      cell = receivers[cell];
    }
    return GPU_TERRAIN_DRAINAGE_NONE;
  });
}

it('review C-03: unresolved accumulation no longer erases the main river from streams', () => {
  const elevation = TERRAIN_FLOW_DEMS.valley(WIDTH, HEIGHT);
  const oracle = computeTerrainFlow({
    elevation,
    grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
    settings: {streamThreshold: THRESHOLD}
  });
  // Emulate an accumulation round limit: cells whose donor tree is taller than 8 stay NaN. Each
  // cell waits for all of its donors, so the unresolved set is closed downstream: the trunk.
  const treeHeight = new Int32Array(WIDTH * HEIGHT).fill(-1);
  const getTreeHeight = (cell: number): number => {
    if (treeHeight[cell] >= 0) return treeHeight[cell];
    let result = 0;
    for (let donor = 0; donor < WIDTH * HEIGHT; donor++) {
      if (oracle.receivers[donor] === cell) result = Math.max(result, 1 + getTreeHeight(donor));
    }
    treeHeight[cell] = result;
    return result;
  };
  const truncated = Float32Array.from(oracle.accumulation, (value, cell) =>
    getTreeHeight(cell) > 8 ? NaN : value
  );
  const trunk = Array.from({length: HEIGHT}, (_, row) => row * WIDTH + (WIDTH >> 1)).filter(cell =>
    Number.isNaN(truncated[cell])
  );
  expect(trunk.length).toBeGreaterThan(10);

  const converged = getStreamsTwin(elevation, oracle.accumulation, 'current');
  const previous = getStreamsTwin(elevation, truncated, 'previous');
  const current = getStreamsTwin(elevation, truncated, 'current');
  for (const cell of trunk) {
    // The converged run has the trunk as a stream; the previous code silently read "not a stream".
    expect(converged[cell]).toBe(1);
    expect(previous[cell]).toBe(0);
    expect(current[cell]).toBe(GPU_TERRAIN_FLOW_STREAM_UNRESOLVED);
  }
  // Resolved cells are unchanged.
  for (let cell = 0; cell < WIDTH * HEIGHT; cell++) {
    if (Number.isFinite(truncated[cell])) expect(current[cell]).toBe(converged[cell]);
  }
  // HAND: with the previous mask, hillslopes that drain into the trunk report "no stream
  // downstream"; now they report an unknown drainage cell.
  const previousDrainage = getDrainageTwin(oracle.receivers, previous);
  const currentDrainage = getDrainageTwin(oracle.receivers, current);
  const hillslope = (HEIGHT - 3) * WIDTH + 3;
  expect(getDrainageTwin(oracle.receivers, converged)[hillslope]).not.toBe(
    GPU_TERRAIN_DRAINAGE_NONE
  );
  expect(previousDrainage[hillslope]).toBe(GPU_TERRAIN_DRAINAGE_NONE);
  expect(currentDrainage[hillslope]).toBe(GPU_TERRAIN_DRAINAGE_UNRESOLVED);
});

it('streams, HAND and stream order kernels carry the unresolved stream word', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const cells = 30;
  let instance = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length = cells) =>
    createTransientView(graph, `view-${instance++}`, format, length);
  const grid = makeGPURasterMetadata({width: 6, height: 5});
  const band = () => makeGPURasterBand(view('float32'), grid, {id: `band-${instance++}`});
  expect(GPU_TERRAIN_FLOW_STREAM_UNRESOLVED).toBe(0xffffffff);
  expect(GPU_TERRAIN_DRAINAGE_UNRESOLVED).toBe(0xfffffffe);

  const flowSources = getGPUCommandNodeKernelSources(
    new GPUTerrainFlow({
      id: 'flow',
      grid,
      elevation: band(),
      parameters: view('float32', 8),
      output: {streams: view('uint32')},
      maximumAccumulationIterations: 1
    }).getCommandNodes(graph)
  );
  const streams = flowSources.find(source => source.id.includes('flow-streams'));
  expect(streams?.source).toContain('if (isValid && !isResolved) { stream = 4294967295u; }');

  const handSources = getGPUCommandNodeKernelSources(
    new GPUTerrainHeightAboveDrainage({
      id: 'hand',
      grid,
      elevation: band(),
      flowDirections: view('uint32'),
      streams: view('uint32'),
      output: {heightAboveDrainage: view('float32'), drainageCells: view('uint32')}
    }).getCommandNodes(graph)
  );
  const height = handSources.find(source => source.id.includes('hand-height'));
  expect(height?.source).toContain('streams[streamsOffset + reached] == 4294967295u');
  expect(height?.source).toContain('if (drainage < 4294967294u)');

  const orderSources = getGPUCommandNodeKernelSources(
    new GPUTerrainStreamOrder({
      id: 'order',
      grid,
      flowDirections: view('uint32'),
      streams: view('uint32'),
      output: {streamOrder: view('uint32')}
    }).getCommandNodes(graph)
  );
  // Euler-tour path: unknown stream words and directions seed the poison pass.
  const seeds = orderSources.find(source => source.id.includes('order-unknown-scatter'));
  expect(seeds?.source).toContain('streams[streamsOffset + cell] == 4294967295u ||');
  expect(seeds?.source).toContain('== UNRESOLVED_DIRECTION');

  // Pull path (tour arrays above one storage binding): the same contract in the round kernel.
  const narrowDevice = createNullWebGPUDevice();
  Object.defineProperty(narrowDevice, 'limits', {
    value: {...narrowDevice.limits, maxStorageBufferBindingSize: 1300}
  });
  const narrowGraph = new GPUCommandGraph(narrowDevice);
  // 20 x 10 cells: the 1600 byte tour arrays exceed the limit, one cell word array does not.
  const narrowGrid = makeGPURasterMetadata({width: 20, height: 10});
  const narrowView = (name: string) => createTransientView(narrowGraph, name, 'uint32', 200);
  const pullSources = getGPUCommandNodeKernelSources(
    new GPUTerrainStreamOrder({
      id: 'order',
      grid: narrowGrid,
      flowDirections: narrowView('pull-directions'),
      streams: narrowView('pull-streams'),
      output: {streamOrder: narrowView('pull-order')}
    }).getCommandNodes(narrowGraph)
  );
  const round = pullSources.find(source => source.id.includes('order-order-round-0'));
  expect(round?.source).toContain('return streams[streamsOffset + cell] == 4294967295u ||');
  expect(round?.source).toContain('if (isUnresolvedStream(index)) { return; }');
  narrowDevice.destroy();
  device.destroy();
});
