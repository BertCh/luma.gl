// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainPeakSnap,
  GPU_TERRAIN_PEAK_SNAP_STATUS,
  type GPUTerrainPeakSnapProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';
import {
  GPUTerrainSummits,
  type GPUTerrainSummitsProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: createTransientView(graph, id, 'float32', length)}
  };
}

function createOutput(graph: GPUCommandGraph, prefix: string, capacity: number) {
  return {
    ids: createTransientView(graph, `${prefix}-ids`, 'uint32', capacity),
    count: createTransientView(graph, `${prefix}-count`, 'uint32', 1),
    overflow: createTransientView(graph, `${prefix}-overflow`, 'uint32', 1)
  };
}

it('GPUTerrainSummits schedules canonicalisation, search, compaction and publish', () => {
  const device = createNullWebGPUDevice();
  // Every recipe gets its own graph: node and transient IDs must be unique within one graph.
  const getNodeIds = (
    build: (graph: GPUCommandGraph) => Partial<GPUTerrainSummitsProps>
  ): string[] => {
    const graph = new GPUCommandGraph(device);
    return new GPUTerrainSummits({
      width: 6,
      height: 5,
      elevation: createBand(graph, 'elevation', 30),
      settings: createTransientView(graph, 'settings', 'float32', 8),
      ...build(graph)
    })
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const maskOnly = getNodeIds(graph => ({
    summitMask: createTransientView(graph, 'mask', 'uint32', 30)
  }));
  expect(maskOnly.at(-1)).toBe('terrain-summits-summits');
  expect(maskOnly.some(nodeId => nodeId.includes('publish'))).toBe(false);

  const ids = getNodeIds(graph => ({
    id: 'hill',
    output: createOutput(graph, 'listing', 8),
    outputDrop: createTransientView(graph, 'listing-drop', 'float32', 8),
    overflow: createTransientView(graph, 'radius-overflow', 'uint32', 1),
    drop: createTransientView(graph, 'drop', 'float32', 30)
  }));
  expect(ids.every(nodeId => nodeId.startsWith('hill-'))).toBe(true);
  for (const step of ['overflow-clear', 'summits', 'pixel-ids', 'publish', 'output-drop']) {
    expect(ids).toContain(`hill-${step}`);
  }
  expect(ids.indexOf('hill-overflow-clear')).toBeLessThan(ids.indexOf('hill-summits'));
  expect(ids.indexOf('hill-publish')).toBeLessThan(ids.indexOf('hill-output-drop'));

  // A list-only recipe keeps the mask in a transient.
  expect(getNodeIds(graph => ({output: createOutput(graph, 'list-only', 4)}))).toContain(
    'terrain-summits-publish'
  );
  for (const cellSizeMode of ['uniform', 'web-mercator', 'geographic'] as const) {
    expect(() =>
      getNodeIds(graph => ({
        cellSizeMode,
        summitMask: createTransientView(graph, 'mask', 'uint32', 30)
      }))
    ).not.toThrow();
  }
  device.destroy();
});

it('GPUTerrainSummits validates its props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainSummitsProps> = {}) => {
    instance++;
    return new GPUTerrainSummits({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      summitMask: createTransientView(graph, `mask-${instance}`, 'uint32', 30),
      ...overrides
    });
  };
  expect(() => create({summitMask: undefined})).toThrow(/at least one output/);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({maximumRadiusPixels: 0})).toThrow(/maximumRadiusPixels/);
  expect(() => create({maximumRadiusPixels: 65})).toThrow(/maximumRadiusPixels/);
  expect(() => create({maximumRadiusPixels: 2.5})).toThrow(/maximumRadiusPixels/);
  expect(() => create({incompleteNeighborhood: 'drop' as never})).toThrow(/incompleteNeighborhood/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 7)})
  ).toThrow(/settings/);
  expect(() =>
    create({summitMask: createTransientView(graph, 'short-mask', 'uint32', 29)})
  ).toThrow(/one value per pixel/);
  expect(() => create({drop: createTransientView(graph, 'short-drop', 'float32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() =>
    create({outputDrop: createTransientView(graph, 'orphan-drop', 'float32', 4)})
  ).toThrow(/requires output/);
  expect(() =>
    create({
      output: createOutput(graph, 'o', 8),
      outputDrop: createTransientView(graph, 'short-column', 'float32', 7)
    })
  ).toThrow(/output.ids.length/);
  expect(() =>
    create({overflow: createTransientView(graph, 'empty-overflow', 'uint32', 0)})
  ).toThrow(/overflow/);
  const shared = createBand(graph, 'shared', 30);
  expect(() => create({elevation: shared, drop: shared.storage.values})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainSummits({
      width: 2,
      height: 2,
      elevation: createBand(foreign, 'foreign', 4),
      settings: createTransientView(foreign, 'foreign-settings', 'float32', 8),
      summitMask: createTransientView(foreign, 'foreign-mask', 'uint32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainPeakSnap schedules search and unpack nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainPeakSnapProps> = {}) => {
    instance++;
    return new GPUTerrainPeakSnap({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      candidates: createTransientView(graph, `candidates-${instance}`, 'float32x2', 3),
      positions: createTransientView(graph, `positions-${instance}`, 'float32x2', 3),
      heights: createTransientView(graph, `heights-${instance}`, 'float32', 3),
      status: createTransientView(graph, `status-${instance}`, 'uint32', 3),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('terrain-peak-snap');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids.slice(-2)).toEqual(['terrain-peak-snap-search', 'terrain-peak-snap-unpack']);
  expect(ids).not.toContain('terrain-peak-snap-overflow-clear');
  expect(
    create({
      id: 'peaks',
      candidateHeights: createTransientView(graph, 'candidate-heights', 'float32', 3),
      candidateRadii: createTransientView(graph, 'candidate-radii', 'float32', 3),
      snapDistance: createTransientView(graph, 'distance', 'float32', 3),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1),
      cellSizeMode: 'geographic'
    })
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toContain('peaks-overflow-clear');
  expect(new Set(Object.values(GPU_TERRAIN_PEAK_SNAP_STATUS)).size).toBe(7);
  device.destroy();
});

it('GPUTerrainPeakSnap validates its props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainPeakSnapProps> = {}) => {
    instance++;
    return new GPUTerrainPeakSnap({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      candidates: createTransientView(graph, `candidates-${instance}`, 'float32x2', 3),
      positions: createTransientView(graph, `positions-${instance}`, 'float32x2', 3),
      heights: createTransientView(graph, `heights-${instance}`, 'float32', 3),
      status: createTransientView(graph, `status-${instance}`, 'uint32', 3),
      ...overrides
    });
  };
  expect(() => create({height: 0})).toThrow(/dimensions/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({maximumRadiusPixels: 100})).toThrow(/maximumRadiusPixels/);
  expect(() => create({settings: createTransientView(graph, 'short', 'float32', 7)})).toThrow(
    /settings/
  );
  expect(() =>
    create({candidates: createTransientView(graph, 'wrong-format', 'float32', 6) as never})
  ).toThrow(/candidates/);
  expect(() =>
    create({candidates: createTransientView(graph, 'no-candidates', 'float32x2', 0)})
  ).toThrow(/at least one row/);
  for (const [name, view] of [
    ['candidateHeights', createTransientView(graph, 'h2', 'float32', 2)],
    ['candidateRadii', createTransientView(graph, 'r2', 'float32', 2)],
    ['heights', createTransientView(graph, 'o2', 'float32', 2)],
    ['snapDistance', createTransientView(graph, 'd2', 'float32', 2)]
  ] as const) {
    expect(() => create({[name]: view})).toThrow(/one value per candidate/);
  }
  expect(() => create({positions: createTransientView(graph, 'p2', 'float32x2', 2)})).toThrow(
    /one value per candidate/
  );
  expect(() => create({status: createTransientView(graph, 's2', 'uint32', 2)})).toThrow(
    /one value per candidate/
  );
  expect(() =>
    create({overflow: createTransientView(graph, 'empty-overflow', 'uint32', 0)})
  ).toThrow(/overflow/);
  const candidates = createTransientView(graph, 'aliased', 'float32x2', 3);
  expect(() => create({candidates, positions: candidates})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainPeakSnap({
      width: 2,
      height: 2,
      elevation: createBand(foreign, 'foreign', 4),
      settings: createTransientView(foreign, 'foreign-settings', 'float32', 8),
      candidates: createTransientView(foreign, 'foreign-candidates', 'float32x2', 1),
      positions: createTransientView(foreign, 'foreign-positions', 'float32x2', 1),
      heights: createTransientView(foreign, 'foreign-heights', 'float32', 1),
      status: createTransientView(foreign, 'foreign-status', 'uint32', 1)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
