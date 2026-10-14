// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {
  GPU_TERRAIN_WATERSHED_NONE,
  GPUTerrainWatersheds
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const WIDTH = 6;
const HEIGHT = 5;
const CELLS = WIDTH * HEIGHT;
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT});

/** Overrides whose `output` members are merged over the default outputs. */
type Overrides = Record<string, unknown> & {output?: Record<string, unknown>};

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length = CELLS) =>
    createTransientView(graph, `view-${instance++}`, format, length);
  const hand = ({output, ...overrides}: Overrides = {}) =>
    new GPUTerrainHeightAboveDrainage({
      grid: GRID,
      elevation: makeGPURasterBand(view('float32'), GRID, {id: `elevation-${instance}`}),
      flowDirections: view('uint32'),
      streams: view('uint32'),
      ...overrides,
      output: {heightAboveDrainage: view('float32'), ...output}
    });
  const watersheds = ({output, ...overrides}: Overrides = {}) =>
    new GPUTerrainWatersheds({
      grid: GRID,
      flowDirections: view('uint32'),
      ...overrides,
      output: {labels: view('uint32'), ...output}
    });
  const order = ({output, ...overrides}: Overrides = {}) =>
    new GPUTerrainStreamOrder({
      grid: GRID,
      flowDirections: view('uint32'),
      streams: view('uint32'),
      ...overrides,
      output: {streamOrder: view('uint32'), ...output}
    });
  return {device, graph, view, hand, watersheds, order};
}

it('exports stable contributor names and constants', () => {
  const {device, hand, watersheds, order} = createFixture();
  expect(hand().id).toBe('terrain-height-above-drainage');
  expect(watersheds().id).toBe('terrain-watersheds');
  expect(order().id).toBe('terrain-stream-order');
  expect(GPU_TERRAIN_WATERSHED_NONE).toBe(0xffffffff);
  device.destroy();
});

it('node ids carry the contributor prefix and grow with the iteration limit', () => {
  const {device, graph, view, hand, watersheds, order} = createFixture();
  const handNodes = hand({
    id: 'h',
    maximumIterations: 3,
    output: {status: {converged: view('uint32', 1)}}
  }).getCommandNodes(graph);
  const handIds = handNodes.map(node => node.id);
  expect(handIds.every(id => id.startsWith('h-'))).toBe(true);
  expect(handIds).toContain('h-pointer-init');
  expect(handIds).toContain('h-pointer-finalize');
  expect(handIds).toContain('h-height');
  expect(handIds.filter(id => id.startsWith('h-pointer-round-'))).toHaveLength(3);
  expect(new Set(handIds).size).toBe(handIds.length);

  const basinIds = watersheds({id: 'w', maximumIterations: 2})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(basinIds).toEqual([
    'w-pointer-init',
    'w-pointer-reset',
    'w-pointer-gate-slots-write',
    'w-pointer-round-0',
    'w-pointer-round-1',
    'w-pointer-gate-0',
    'w-labels'
  ]);
  const pourIds = watersheds({id: 'p', maximumIterations: 1, pourPoints: view('uint32', 4)})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(pourIds.slice(0, 2)).toEqual(['p-markers-init', 'p-markers-mark']);
  expect(pourIds).toContain('p-labels');

  const orderIds = order({
    id: 'o',
    maximumIterations: 4,
    output: {status: {converged: view('uint32', 1)}}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  // The Euler-tour path: its round and level counts follow the grid, not maximumIterations.
  expect(orderIds.slice(0, 2)).toEqual(['o-order-init', 'o-tour-link']);
  expect(orderIds.filter(id => id.startsWith('o-order-round-'))).toHaveLength(0);
  expect(orderIds.filter(id => /^o-order-level-\d+-gather$/.test(id))).toHaveLength(4);
  expect(orderIds.at(-1)).toBe('o-order-finalize');
  expect(new Set(orderIds).size).toBe(orderIds.length);
  const longerIds = order({id: 'o2', maximumIterations: 9}).getCommandNodes(graph);
  expect(longerIds.length).toBe(orderIds.length - 1);

  // Tour arrays above one storage binding fall back to the pull rounds, which follow the limit.
  const narrowDevice = createNullWebGPUDevice();
  Object.defineProperty(narrowDevice, 'limits', {
    value: {...narrowDevice.limits, maxStorageBufferBindingSize: 1300}
  });
  const narrowGraph = new GPUCommandGraph(narrowDevice);
  // 20 x 10 cells: 2 n tour words are 1600 bytes, above the 1300 byte limit; a cell word array fits.
  const narrowGrid = makeGPURasterMetadata({width: 20, height: 10});
  const narrowView = (name: string) => createTransientView(narrowGraph, name, 'uint32', 200);
  const pullIds = new GPUTerrainStreamOrder({
    id: 'q',
    grid: narrowGrid,
    flowDirections: narrowView('pull-directions'),
    streams: narrowView('pull-streams'),
    maximumIterations: 4,
    output: {streamOrder: narrowView('pull-order')}
  })
    .getCommandNodes(narrowGraph)
    .map(node => node.id);
  expect(pullIds.slice(0, 2)).toEqual(['q-order-init', 'q-order-reset']);
  expect(pullIds.filter(id => id.startsWith('q-order-round-'))).toHaveLength(4);
  narrowDevice.destroy();
  device.destroy();
});

it('GPUTerrainHeightAboveDrainage validates its props', () => {
  const {device, graph, view, hand} = createFixture();
  expect(() => hand({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  expect(() => hand({output: {heightAboveDrainage: undefined}})).toThrow(/at least one output/);
  expect(() => hand({flowDirections: view('uint32', CELLS - 1)})).toThrow(/one value per cell/);
  expect(() => hand({streams: view('float32') as never})).toThrow(/uint32/);
  expect(() => hand({output: {heightAboveDrainage: view('float32', CELLS + 1)}})).toThrow(
    /one value per cell/
  );
  expect(() => hand({output: {status: {converged: view('uint32', 2)}}})).toThrow(/one row/);
  expect(() => hand({output: {status: {iterationCount: view('uint32', 2)}}})).toThrow(/one row/);
  expect(() => hand({maximumIterations: 0})).toThrow(/maximumIterations/);
  expect(() => hand({maximumIterations: 2000})).toThrow(/maximumIterations/);
  expect(() =>
    hand({
      elevation: {
        id: 'short',
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32', CELLS - 1)}
      }
    })
  ).toThrow(/one value per cell/);
  const shared = view('uint32');
  expect(() => hand({flowDirections: shared, output: {drainageCells: shared}})).toThrow(
    /share buffers/
  );
  const output = view('uint32');
  const aliased = graph.createDataView(output.buffer, {format: 'float32', length: CELLS});
  expect(() => hand({output: {drainageCells: output, heightAboveDrainage: aliased}})).toThrow(
    /share buffers/
  );
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    hand({
      flowDirections: createTransientView(foreign, 'foreign', 'uint32', CELLS)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  // A drainage-only request is valid and skips canonicalization.
  const drainageOnly = hand({
    output: {heightAboveDrainage: undefined, drainageCells: view('uint32')}
  });
  expect(drainageOnly.getCommandNodes(graph).map(node => node.id)).not.toContain(
    'terrain-height-above-drainage-elevation'
  );
  device.destroy();
});

it('GPUTerrainWatersheds validates its props', () => {
  const {device, graph, view, watersheds} = createFixture();
  expect(() => watersheds({grid: {...GRID, height: 0}})).toThrow(/dimensions/);
  expect(() => watersheds({output: {labels: view('uint32', CELLS - 1)}})).toThrow(
    /one value per cell/
  );
  expect(() => watersheds({flowDirections: view('float32') as never})).toThrow(/uint32/);
  expect(() => watersheds({pourPoints: view('float32', 3) as never})).toThrow(/uint32/);
  expect(() => watersheds({output: {status: {converged: view('uint32', 0)}}})).toThrow(/one row/);
  expect(() => watersheds({maximumIterations: 0.5})).toThrow(/maximumIterations/);
  const shared = view('uint32');
  expect(() => watersheds({flowDirections: shared, output: {labels: shared}})).toThrow(
    /share buffers/
  );
  expect(() => watersheds({pourPoints: shared, output: {labels: shared}})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    watersheds({pourPoints: createTransientView(foreign, 'foreign', 'uint32', 3)}).getCommandNodes(
      graph
    )
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainStreamOrder validates its props', () => {
  const {device, graph, view, order} = createFixture();
  expect(() => order({grid: {...GRID, width: 1.5}})).toThrow(/dimensions/);
  expect(() => order({output: {streamOrder: view('uint32', CELLS - 1)}})).toThrow(
    /one value per cell/
  );
  expect(() => order({streams: view('uint32', CELLS + 1)})).toThrow(/one value per cell/);
  expect(() => order({output: {status: {converged: view('uint32', 3)}}})).toThrow(/one row/);
  expect(() => order({output: {status: {iterationCount: view('uint32', 3)}}})).toThrow(/one row/);
  expect(() => order({maximumIterations: 1025})).toThrow(/maximumIterations/);
  const shared = view('uint32');
  expect(() => order({streams: shared, output: {streamOrder: shared}})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    order({
      output: {streamOrder: createTransientView(foreign, 'foreign', 'uint32', CELLS)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
