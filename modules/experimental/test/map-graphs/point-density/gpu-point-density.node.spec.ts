// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUPointDensity, type GPUPointDensityProps} from '../../../src/map-graphs/point-density';
import {createNullWebGPUDevice, createVectorView} from '../map-graph-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUPointDensityProps> = {}
): GPUPointDensityProps {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    bounds: [0, 0, 4, 2],
    gridSize: [4, 2],
    output: {
      values: createTransientView(graph, 'values', 'float32', 8),
      counts: createTransientView(graph, 'counts', 'uint32', 8)
    },
    ...overrides
  };
}

it('GPUPointDensity schedules grid counts and finalize', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUPointDensity(createProps(graph)).getCommandNodes(graph).map(node => node.id);
  expect(ids[0].startsWith('point-density-counts')).toBe(true);
  expect(ids.at(-1)).toBe('point-density-finalize');
  expect(ids.some(id => id.startsWith('point-density-sums'))).toBe(false);
  expect(ids.some(id => id.startsWith('point-density-hexagon'))).toBe(false);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.every(id => id.startsWith('point-density-'))).toBe(true);
  device.destroy();
});

it('GPUPointDensity orders weighted grid steps', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUPointDensity({
    ...createProps(graph),
    id: 'density',
    weights: createTransientView(graph, 'weights', 'float32', 8),
    statistic: 'mean',
    output: {
      values: createTransientView(graph, 'mean-values', 'float32', 8),
      means: createTransientView(graph, 'means', 'float32', 8),
      extent: createTransientView(graph, 'extent', 'float32', 2),
      histogram: createTransientView(graph, 'histogram', 'uint32', 4),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  const prefixes = [
    'density-clear-overflow',
    'density-counts',
    'density-sums',
    'density-finalize',
    'density-extent',
    'density-histogram'
  ];
  const steps = ids
    .map(id => prefixes.find(prefix => id.startsWith(prefix)))
    .filter((step, index, all) => step !== all[index - 1]);
  expect(steps).toEqual(prefixes);
  device.destroy();
});

it('GPUPointDensity emits one hexagon keys node per nonempty chunk', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createVectorView('chunked-positions', 'float32x2', [
    createTransientView(graph, 'p0', 'float32x2', 3),
    createTransientView(graph, 'p1', 'float32x2', 0),
    createTransientView(graph, 'p2', 'float32x2', 2)
  ]);
  const ids = new GPUPointDensity({
    ...createProps(new GPUCommandGraph(device)),
    positions,
    binning: 'hexagon',
    hexagonRadius: 1,
    gridSize: [5, 5],
    bounds: [0, 0, 6, 6],
    output: {values: createTransientView(graph, 'hex-values', 'float32', 25)}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('point-density-hexagon-keys-0');
  expect(ids).toContain('point-density-hexagon-keys-2');
  expect(ids).not.toContain('point-density-hexagon-keys-1');
  expect(ids.indexOf('point-density-hexagon-keys-2')).toBeLessThan(
    ids.findIndex(id => id.startsWith('point-density-counts'))
  );

  const sharedGraph = new GPUCommandGraph(device);
  const sharedPositions = createTransientView(sharedGraph, 'shared-positions', 'float32x2', 8);
  const createInstance = (id: string) =>
    new GPUPointDensity({
      id,
      positions: sharedPositions,
      bounds: [0, 0, 4, 2],
      gridSize: [4, 2],
      output: {values: createTransientView(sharedGraph, `${id}-values`, 'float32', 8)}
    });
  const aIds = createInstance('a')
    .getCommandNodes(sharedGraph)
    .map(node => node.id);
  const bIds = createInstance('b')
    .getCommandNodes(sharedGraph)
    .map(node => node.id);
  expect(aIds.some(id => bIds.includes(id))).toBe(false);
  device.destroy();
});

it('GPUPointDensity validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUPointDensityProps>) =>
    new GPUPointDensity({...base, ...overrides});
  const weights = createTransientView(graph, 'weights', 'float32', 8);

  expect(() => create({gridSize: [0, 4]})).toThrow(/gridSize/);
  expect(() => create({gridSize: [2.5, 4]})).toThrow(/gridSize/);
  expect(() =>
    create({output: {values: createTransientView(graph, 'short', 'float32', 7)}})
  ).toThrow(/one row per cell/);
  expect(() => create({statistic: 'sum'})).toThrow(/requires weights/);
  expect(() =>
    create({
      output: {...base.output, means: createTransientView(graph, 'means', 'float32', 8)}
    })
  ).toThrow(/requires weights/);
  expect(() =>
    create({weights: createTransientView(graph, 'short-weights', 'float32', 7)})
  ).toThrow(/weights length/);
  expect(() => create({binning: 'hexagon'})).toThrow(/hexagonRadius/);
  expect(() => create({binning: 'hexagon', hexagonRadius: 0})).toThrow(/hexagonRadius/);
  expect(() => create({hexagonRadius: 1})).toThrow(/hexagonRadius/);
  const kernel = createTransientView(graph, 'kernel', 'float32', 9);
  expect(() =>
    create({
      binning: 'hexagon',
      hexagonRadius: 1,
      smoothing: {kernel, kernelWidth: 3, kernelHeight: 3}
    })
  ).toThrow(/smoothing requires grid/);
  expect(() => create({smoothing: {kernel, kernelWidth: 2, kernelHeight: 3}})).toThrow(/odd/);
  expect(() => create({smoothing: {kernel, kernelWidth: 5, kernelHeight: 3}})).toThrow(/shorter/);
  expect(() => create({bounds: [1, 0, 0, 1]})).toThrow(/bounds/);
  expect(() => create({bounds: createTransientView(graph, 'bounds3', 'float32', 3)})).toThrow(
    /bounds/
  );
  expect(() =>
    create({
      weights,
      output: {...base.output, extent: createTransientView(graph, 'extent3', 'float32', 3)}
    })
  ).toThrow(/extent/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      output: {
        values: graph.createDataView(shared, {format: 'float32', length: 8}),
        counts: graph.createDataView(shared, {format: 'uint32', length: 8})
      }
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUPointDensity(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});
