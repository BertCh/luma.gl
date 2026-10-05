// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS,
  GPUTerrainSpikeRepair,
  type GPUTerrainSpikeRepairProps
} from '../../../src/map-graphs/terrain-decode/gpu-terrain-spike-repair';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

it('GPUTerrainSpikeRepair schedules canonicalisation, graph, components and repair kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainSpikeRepairProps> = {}) => {
    instance++;
    return new GPUTerrainSpikeRepair({
      width: 6,
      height: 5,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `e-${instance}`, 'float32', 30)
        }
      },
      values: createTransientView(graph, `values-${instance}`, 'float32', 30),
      validity: createTransientView(graph, `validity-${instance}`, 'uint32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('terrain-spike-repair');
  expect(recipe.componentIterations).toBe(32);
  const nodeIds = recipe.getCommandNodes(graph).map(node => node.id);
  for (const step of [
    'clear',
    'graph',
    'sizes',
    'main-size',
    'main-label',
    'votes',
    'agree',
    'decide',
    'shift',
    'recount'
  ]) {
    expect(nodeIds).toContain(`terrain-spike-repair-${step}`);
  }
  expect(nodeIds.some(id => id.startsWith('terrain-spike-repair-components'))).toBe(true);
  expect(nodeIds).not.toContain('terrain-spike-repair-publish-labels');
  expect(
    create({id: 'repair', labels: createTransientView(graph, 'labels', 'uint32', 30)})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toContain('repair-publish-labels');
  expect(Object.values(GPU_TERRAIN_SPIKE_REPAIR_STATISTICS)).toEqual([0, 1, 2, 3, 4]);
  recipe.destroy();
  device.destroy();
});

it('GPUTerrainSpikeRepair validates its properties', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainSpikeRepairProps> = {}) => {
    instance++;
    return new GPUTerrainSpikeRepair({
      width: 6,
      height: 5,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `e-${instance}`, 'float32', 30)
        }
      },
      values: createTransientView(graph, `values-${instance}`, 'float32', 30),
      validity: createTransientView(graph, `validity-${instance}`, 'uint32', 30),
      ...overrides
    });
  };
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({step: 0})).toThrow(/step/);
  expect(() => create({jump: -1})).toThrow(/jump/);
  expect(() => create({tolerance: 128})).toThrow(/less than step/);
  expect(() => create({tolerance: 0})).toThrow(/tolerance/);
  expect(() => create({step: 100, tolerance: 50})).toThrow(/less than step/);
  expect(() => create({agreement: 0.5})).toThrow(/agreement/);
  expect(() => create({agreement: 0.805})).toThrow(/agreement/);
  expect(() => create({agreement: 1.01})).toThrow(/agreement/);
  expect(() => create({maximumComponentFraction: 0})).toThrow(/maximumComponentFraction/);
  expect(() => create({maximumComponentFraction: 0.255})).toThrow(/maximumComponentFraction/);
  expect(() => create({componentIterations: 0})).toThrow(/componentIterations/);
  expect(() => create({componentIterations: 2000})).toThrow(/componentIterations/);
  expect(() => create({values: createTransientView(graph, 'short', 'float32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() => create({statistics: createTransientView(graph, 'stats4', 'uint32', 4)})).toThrow(
    /at least 5/
  );
  expect(() =>
    create({agreement: 0.51, maximumComponentFraction: 1, step: 25.6, tolerance: 5})
  ).not.toThrow();
  const shared = createTransientView(graph, 'shared', 'float32', 30);
  expect(() =>
    create({
      elevation: {
        id: 'shared-elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: shared}
      },
      values: shared
    })
  ).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainSpikeRepair({
      width: 2,
      height: 2,
      elevation: {
        id: 'foreign',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(otherGraph, 'foreign-e', 'float32', 4)
        }
      },
      values: createTransientView(otherGraph, 'foreign-values', 'float32', 4),
      validity: createTransientView(otherGraph, 'foreign-validity', 'uint32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
