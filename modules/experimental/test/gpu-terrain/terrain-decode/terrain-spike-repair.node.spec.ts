// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS,
  GPU_TERRAIN_SPIKE_REPAIR_UNRESOLVED,
  GPUTerrainSpikeRepair,
  type GPUTerrainSpikeRepairOutput,
  type GPUTerrainSpikeRepairProps
} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

it('GPUTerrainSpikeRepair schedules canonicalisation, graph, components and repair kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<GPUTerrainSpikeRepairProps> = {},
    outputOverrides: Partial<GPUTerrainSpikeRepairOutput> = {}
  ) => {
    instance++;
    return new GPUTerrainSpikeRepair({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `e-${instance}`, 'float32', 30)
        }
      },
      output: {
        values: createTransientView(graph, `values-${instance}`, 'float32', 30),
        outputValidity: createTransientView(graph, `validity-${instance}`, 'uint32', 30),
        ...outputOverrides
      },
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.maximumComponentIterations).toBe(32);
  const nodeIds = contributor.getCommandNodes(graph).map(node => node.id);
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
    create({id: 'repair'}, {labels: createTransientView(graph, 'labels', 'uint32', 30)})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toContain('repair-publish-labels');
  expect(Object.values(GPU_TERRAIN_SPIKE_REPAIR_STATISTICS)).toEqual([0, 1, 2, 3, 4]);
  contributor.destroy();
  device.destroy();
});

it('GPUTerrainSpikeRepair validates its properties', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<GPUTerrainSpikeRepairProps> = {},
    outputOverrides: Partial<GPUTerrainSpikeRepairOutput> = {}
  ) => {
    instance++;
    return new GPUTerrainSpikeRepair({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `e-${instance}`, 'float32', 30)
        }
      },
      output: {
        values: createTransientView(graph, `values-${instance}`, 'float32', 30),
        outputValidity: createTransientView(graph, `validity-${instance}`, 'uint32', 30),
        ...outputOverrides
      },
      ...overrides
    });
  };
  expect(() => create({grid: makeGPURasterMetadata({width: 0, height: 5})})).toThrow(/dimensions/);
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
  expect(() => create({maximumComponentIterations: 0})).toThrow(/maximumComponentIterations/);
  expect(() => create({maximumComponentIterations: 2000})).toThrow(/maximumComponentIterations/);
  expect(() => create({}, {values: createTransientView(graph, 'short', 'float32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() => create({}, {statistics: createTransientView(graph, 'stats4', 'uint32', 4)})).toThrow(
    /at least 5/
  );
  expect(() =>
    create({agreement: 0.51, maximumComponentFraction: 1, step: 25.6, tolerance: 5})
  ).not.toThrow();
  const shared = createTransientView(graph, 'shared', 'float32', 30);
  expect(() =>
    create(
      {
        elevation: {
          id: 'shared-elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: shared}
        }
      },
      {values: shared}
    )
  ).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainSpikeRepair({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: {
        id: 'foreign',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(otherGraph, 'foreign-e', 'float32', 4)
        }
      },
      output: {
        values: createTransientView(otherGraph, 'foreign-values', 'float32', 4),
        outputValidity: createTransientView(otherGraph, 'foreign-validity', 'uint32', 4)
      }
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('review C-09: unconverged labelling marks the repaired outputs instead of copying them', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUTerrainSpikeRepair({
    grid: makeGPURasterMetadata({width: 6, height: 5}),
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'e', 'float32', 30)}
    },
    output: {
      values: createTransientView(graph, 'values', 'float32', 30),
      outputValidity: createTransientView(graph, 'validity', 'uint32', 30)
    },
    maximumComponentIterations: 1
  }).getCommandNodes(graph);
  const sources = getRasterPipelines(nodes).map(pipeline => pipeline.source);
  // Before: an unconverged run copied the input, which reads as "nothing to repair".
  expect(
    sources.some(source =>
      source.includes(
        `validityOut[validityOutOffset + index] = ${GPU_TERRAIN_SPIKE_REPAIR_UNRESOLVED}u;`
      )
    )
  ).toBe(true);
});

it('review A §7.4: a second graph is rejected under the contributor id', () => {
  const device = createNullWebGPUDevice();
  const first = new GPUCommandGraph(device, {id: 'first'});
  const second = new GPUCommandGraph(device, {id: 'second'});
  const contributor = new GPUTerrainSpikeRepair({
    id: 'repair',
    grid: makeGPURasterMetadata({width: 6, height: 5}),
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(first, 'e', 'float32', 30)}
    },
    output: {
      values: createTransientView(first, 'values', 'float32', 30),
      outputValidity: createTransientView(first, 'validity', 'uint32', 30)
    }
  });
  // The props' views belong to the first graph, so owned buffers can never be shared by two.
  expect(() => contributor.getCommandNodes(second)).toThrow(/^repair /);
  expect(contributor.getCommandNodes(first).length).toBeGreaterThan(0);
  contributor.destroy();
  device.destroy();
});
