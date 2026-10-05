// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUEdgeBundling,
  createGPUEdgeBundlingParameterValues,
  getGPUEdgeBundlingFixedPointExponent,
  type GPUEdgeBundlingProps
} from '../../../src/gpu-network/edge-bundling';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUEdgeBundlingProps> = {}
): GPUEdgeBundlingProps {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 10),
    sourceVertices: createTransientView(graph, 'sources', 'uint32', 6),
    targetVertices: createTransientView(graph, 'targets', 'uint32', 6),
    paths: createTransientView(graph, 'paths', 'float32x2', 6 * 16),
    ...overrides
  };
}

it('GPUEdgeBundling schedules box, initialize, per-iteration, finalize, and indices nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUEdgeBundling({
    ...createProps(graph, {
      paths: createTransientView(graph, 'paths-4', 'float32x2', 6 * 4),
      edgeMask: createTransientView(graph, 'mask', 'uint32', 6),
      parameters: createTransientView(graph, 'parameters', 'uint32', 5),
      startIndices: createTransientView(graph, 'start-indices', 'uint32', 7),
      drawRecord: createTransientView(graph, 'draw-record', 'uint32', 4)
    }),
    id: 'bundle',
    pointsPerEdge: 4,
    iterations: 2,
    densityResolution: 32
  }).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'bundle-box-reset',
    'bundle-box',
    'bundle-initialize',
    'bundle-clear-0',
    'bundle-splat-0',
    'bundle-update-0',
    'bundle-clear-1',
    'bundle-splat-1',
    'bundle-update-1',
    'bundle-finalize',
    'bundle-indices'
  ]);

  const minimalGraph = new GPUCommandGraph(device);
  const minimal = new GPUEdgeBundling(createProps(minimalGraph));
  expect(minimal.id).toBe('edge-bundling');
  const ids = minimal.getCommandNodes(minimalGraph).map(node => node.id);
  expect(ids.length).toBe(3 + 3 * 15 + 1);
  expect(ids[0]).toBe('edge-bundling-box-reset');
  expect(ids).not.toContain('edge-bundling-indices');
  device.destroy();
});

it('GPUEdgeBundling validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUEdgeBundlingProps>) =>
    new GPUEdgeBundling({...base, ...overrides});
  expect(() =>
    create({
      targetVertices: createTransientView(graph, 'targets5', 'uint32', 5)
    })
  ).toThrow(/targetVertices length/);
  expect(() => create({edgeMask: createTransientView(graph, 'mask5', 'uint32', 5)})).toThrow(
    /edgeMask length/
  );
  for (const pointsPerEdge of [1, 65, 2.5]) {
    expect(() => create({pointsPerEdge})).toThrow(/pointsPerEdge/);
  }
  for (const iterations of [0, 65, 1.5]) {
    expect(() => create({iterations})).toThrow(/iterations/);
  }
  expect(() => create({densityResolution: 4})).toThrow(/densityResolution/);
  expect(() =>
    create({
      paths: createTransientView(graph, 'paths-short', 'float32x2', 10)
    })
  ).toThrow(/paths length/);
  expect(() =>
    create({
      startIndices: createTransientView(graph, 'start-6', 'uint32', 6)
    })
  ).toThrow(/startIndices length/);
  expect(() => create({drawRecord: createTransientView(graph, 'draw-3', 'uint32', 3)})).toThrow(
    /drawRecord/
  );
  expect(() =>
    create({
      parameters: createTransientView(graph, 'parameters-3', 'uint32', 3)
    })
  ).toThrow(/parameters/);
  expect(() =>
    create({
      parameters: createTransientView(graph, 'parameters-sint', 'sint32', 5) as never
    })
  ).toThrow(/parameters/);
  expect(() =>
    create({
      positions: createTransientView(graph, 'positions-flat', 'float32', 10) as never
    })
  ).toThrow(/positions/);
  expect(() =>
    create({
      positions: createTransientView(graph, 'positions-empty', 'float32x2', 0)
    })
  ).toThrow(/at least one vertex/);
  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 1024,
    usage: 128
  });
  expect(() =>
    create({
      positions: graph.createDataView(shared, {
        format: 'float32x2',
        length: 10
      }),
      paths: graph.createDataView(shared, {format: 'float32x2', length: 96})
    })
  ).toThrow(/separate buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUEdgeBundling(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  const largeGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUEdgeBundling(createProps(largeGraph, {densityResolution: 8192})).getCommandNodes(
      largeGraph
    )
  ).toThrow(/maxStorageBufferBindingSize/);
  device.destroy();
});

it('fixed-point exponent never allows density overflow', () => {
  for (const pointCount of [1, 1000, 160000, 1 << 20, 1 << 24, 1 << 30]) {
    const exponent = getGPUEdgeBundlingFixedPointExponent(pointCount);
    expect(pointCount * 2 ** exponent).toBeLessThanOrEqual(0xffffffff);
    expect(exponent).toBeLessThanOrEqual(16);
  }
  expect(getGPUEdgeBundlingFixedPointExponent(160000)).toBe(14);
  expect(getGPUEdgeBundlingFixedPointExponent((1 << 24) - 1)).toBe(8);
});

it('parameter values pack in both layouts', () => {
  const words = createGPUEdgeBundlingParameterValues({activeIterations: 3, lambda: 0.7}, 'uint32');
  expect(words[0]).toBe(3);
  expect(new Float32Array(words.buffer)[1]).toBeCloseTo(0.03, 6);
  expect(new Float32Array(words.buffer)[2]).toBeCloseTo(0.7, 6);
  const floats = createGPUEdgeBundlingParameterValues({stepScale: 2}, 'float32');
  expect(Array.from(floats)).toEqual([64, Math.fround(0.03), Math.fround(0.85), 0.5, 2]);
});
