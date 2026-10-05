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
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const WIDTH = 6;
const HEIGHT = 5;
const CELLS = WIDTH * HEIGHT;

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length = CELLS) =>
    createTransientView(graph, `view-${instance++}`, format, length);
  const hand = (overrides: Record<string, unknown> = {}) =>
    new GPUTerrainHeightAboveDrainage({
      width: WIDTH,
      height: HEIGHT,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32')}
      },
      flowDirections: view('uint32'),
      streams: view('uint32'),
      heightAboveDrainage: view('float32'),
      ...overrides
    });
  const watersheds = (overrides: Record<string, unknown> = {}) =>
    new GPUTerrainWatersheds({
      width: WIDTH,
      height: HEIGHT,
      flowDirections: view('uint32'),
      labels: view('uint32'),
      ...overrides
    });
  const order = (overrides: Record<string, unknown> = {}) =>
    new GPUTerrainStreamOrder({
      width: WIDTH,
      height: HEIGHT,
      flowDirections: view('uint32'),
      streams: view('uint32'),
      streamOrder: view('uint32'),
      ...overrides
    });
  return {device, graph, view, hand, watersheds, order};
}

it('exports stable recipe names and constants', () => {
  const {device, hand, watersheds, order} = createFixture();
  expect(hand().recipe).toBe('terrain-height-above-drainage');
  expect(watersheds().recipe).toBe('terrain-watersheds');
  expect(order().recipe).toBe('terrain-stream-order');
  expect(hand().id).toBe('terrain-height-above-drainage');
  expect(GPU_TERRAIN_WATERSHED_NONE).toBe(0xffffffff);
  device.destroy();
});

it('node ids carry the recipe prefix and grow with the iteration limit', () => {
  const {device, graph, view, hand, watersheds, order} = createFixture();
  const handNodes = hand({id: 'h', maxIterations: 3, converged: view('uint32', 1)}).getCommandNodes(
    graph
  );
  const handIds = handNodes.map(node => node.id);
  expect(handIds.every(id => id.startsWith('h-'))).toBe(true);
  expect(handIds).toContain('h-pointer-init');
  expect(handIds).toContain('h-pointer-finalize');
  expect(handIds).toContain('h-height');
  expect(handIds.filter(id => id.startsWith('h-pointer-round-'))).toHaveLength(3);
  expect(new Set(handIds).size).toBe(handIds.length);

  const basinIds = watersheds({id: 'w', maxIterations: 2})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(basinIds).toEqual([
    'w-pointer-init',
    'w-pointer-reset',
    'w-pointer-round-0',
    'w-pointer-gate-0',
    'w-pointer-round-1',
    'w-pointer-gate-1',
    'w-labels'
  ]);
  const pourIds = watersheds({id: 'p', maxIterations: 1, pourPoints: view('uint32', 4)})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(pourIds.slice(0, 2)).toEqual(['p-markers-init', 'p-markers-mark']);
  expect(pourIds).toContain('p-labels');

  const orderIds = order({id: 'o', maxIterations: 4, converged: view('uint32', 1)})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(orderIds.slice(0, 2)).toEqual(['o-order-init', 'o-order-reset']);
  expect(orderIds.filter(id => id.startsWith('o-order-round-'))).toHaveLength(4);
  expect(orderIds.at(-1)).toBe('o-order-finalize');
  device.destroy();
});

it('GPUTerrainHeightAboveDrainage validates its props', () => {
  const {device, graph, view, hand} = createFixture();
  expect(() => hand({width: 0})).toThrow(/dimensions/);
  expect(() => hand({heightAboveDrainage: undefined})).toThrow(/at least one output/);
  expect(() => hand({flowDirections: view('uint32', CELLS - 1)})).toThrow(/one value per cell/);
  expect(() => hand({streams: view('float32') as never})).toThrow(/uint32/);
  expect(() => hand({heightAboveDrainage: view('float32', CELLS + 1)})).toThrow(
    /one value per cell/
  );
  expect(() => hand({converged: view('uint32', 2)})).toThrow(/one row/);
  expect(() => hand({maxIterations: 0})).toThrow(/maxIterations/);
  expect(() => hand({maxIterations: 2000})).toThrow(/maxIterations/);
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
  expect(() => hand({flowDirections: shared, drainageCells: shared})).toThrow(/share buffers/);
  const output = view('uint32');
  const aliased = graph.createDataView(output.buffer, {format: 'float32', length: CELLS});
  expect(() => hand({drainageCells: output, heightAboveDrainage: aliased})).toThrow(
    /share buffers/
  );
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    hand({
      flowDirections: createTransientView(foreign, 'foreign', 'uint32', CELLS)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  // A drainage-only request is valid and skips canonicalization.
  const drainageOnly = hand({heightAboveDrainage: undefined, drainageCells: view('uint32')});
  expect(drainageOnly.getCommandNodes(graph).map(node => node.id)).not.toContain(
    'terrain-height-above-drainage-elevation'
  );
  device.destroy();
});

it('GPUTerrainWatersheds validates its props', () => {
  const {device, graph, view, watersheds} = createFixture();
  expect(() => watersheds({height: 0})).toThrow(/dimensions/);
  expect(() => watersheds({labels: view('uint32', CELLS - 1)})).toThrow(/one value per cell/);
  expect(() => watersheds({flowDirections: view('float32') as never})).toThrow(/uint32/);
  expect(() => watersheds({pourPoints: view('float32', 3) as never})).toThrow(/uint32/);
  expect(() => watersheds({converged: view('uint32', 0)})).toThrow(/one row/);
  expect(() => watersheds({maxIterations: 0.5})).toThrow(/maxIterations/);
  const shared = view('uint32');
  expect(() => watersheds({flowDirections: shared, labels: shared})).toThrow(/share buffers/);
  expect(() => watersheds({pourPoints: shared, labels: shared})).toThrow(/share buffers/);
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
  expect(() => order({width: 1.5})).toThrow(/dimensions/);
  expect(() => order({streamOrder: view('uint32', CELLS - 1)})).toThrow(/one value per cell/);
  expect(() => order({streams: view('uint32', CELLS + 1)})).toThrow(/one value per cell/);
  expect(() => order({converged: view('uint32', 3)})).toThrow(/one row/);
  expect(() => order({maxIterations: 1025})).toThrow(/maxIterations/);
  const shared = view('uint32');
  expect(() => order({streams: shared, streamOrder: shared})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    order({streamOrder: createTransientView(foreign, 'foreign', 'uint32', CELLS)}).getCommandNodes(
      graph
    )
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
