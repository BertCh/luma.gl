// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainPeakSnap,
  GPU_TERRAIN_PEAK_SNAP_STATUS,
  getGPUTerrainPeakSnapParameterValues,
  type GPUTerrainPeakSnapOutput,
  type GPUTerrainPeakSnapProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';
import {
  GPUTerrainSummits,
  getGPUTerrainSummitsParameterValues,
  type GPUTerrainSummitsOutput,
  type GPUTerrainSummitsProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

function createOutput(graph: GPUCommandGraph, prefix: string, capacity: number) {
  return {
    ids: createTransientView(graph, `${prefix}-ids`, 'uint32', capacity),
    status: {
      count: createTransientView(graph, `${prefix}-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${prefix}-overflow`, 'uint32', 1)
    }
  };
}

it('GPUTerrainSummits schedules canonicalisation, search, compaction and publish', () => {
  const device = createNullWebGPUDevice();
  // Every contributor gets its own graph: node and transient IDs must be unique within one graph.
  const getNodeIds = (
    build: (graph: GPUCommandGraph) => GPUTerrainSummitsOutput,
    overrides: Partial<GPUTerrainSummitsProps> = {}
  ): string[] => {
    const graph = new GPUCommandGraph(device);
    return new GPUTerrainSummits({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, 'elevation', 30),
      parameters: createTransientView(graph, 'settings', 'float32', 8),
      output: build(graph),
      ...overrides
    })
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const maskOnly = getNodeIds(graph => ({
    summitMask: createTransientView(graph, 'mask', 'uint32', 30)
  }));
  expect(maskOnly.at(-1)).toBe('terrain-summits-summits');
  expect(maskOnly.some(nodeId => nodeId.includes('publish'))).toBe(false);

  const ids = getNodeIds(
    graph => ({
      summits: createOutput(graph, 'listing', 8),
      summitDrop: createTransientView(graph, 'listing-drop', 'float32', 8),
      status: {radiusClamped: createTransientView(graph, 'radius-overflow', 'uint32', 1)},
      drop: createTransientView(graph, 'drop', 'float32', 30)
    }),
    {id: 'hill'}
  );
  expect(ids.every(nodeId => nodeId.startsWith('hill-'))).toBe(true);
  for (const step of ['overflow-clear', 'summits', 'pixel-ids', 'publish', 'output-drop']) {
    expect(ids).toContain(`hill-${step}`);
  }
  expect(ids.indexOf('hill-overflow-clear')).toBeLessThan(ids.indexOf('hill-summits'));
  expect(ids.indexOf('hill-publish')).toBeLessThan(ids.indexOf('hill-output-drop'));

  // A list-only contributor keeps the mask in a transient.
  expect(getNodeIds(graph => ({summits: createOutput(graph, 'list-only', 4)}))).toContain(
    'terrain-summits-publish'
  );
  for (const cellSizeMode of ['uniform', 'web-mercator', 'geographic'] as const) {
    expect(() =>
      getNodeIds(graph => ({summitMask: createTransientView(graph, 'mask', 'uint32', 30)}), {
        cellSizeMode
      })
    ).not.toThrow();
  }
  device.destroy();
});

it('GPUTerrainSummits validates its props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<GPUTerrainSummitsProps> = {},
    outputOverrides: Partial<GPUTerrainSummitsOutput> = {}
  ) => {
    instance++;
    return new GPUTerrainSummits({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {
        summitMask: createTransientView(graph, `mask-${instance}`, 'uint32', 30),
        ...outputOverrides
      },
      ...overrides
    });
  };
  expect(() => create({}, {summitMask: undefined})).toThrow(/at least one output/);
  expect(() => create({grid: makeGPURasterMetadata({width: 0, height: 5})})).toThrow(/dimensions/);
  expect(() =>
    create({
      cellSizeMode: 'geographic',
      grid: {...makeGPURasterMetadata({width: 6, height: 5}), affine: [1, 0.5, 0, 0, -1, 0]}
    })
  ).toThrow(/axis-aligned/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({maximumRadiusPixels: 0})).toThrow(/maximumRadiusPixels/);
  expect(() => create({maximumRadiusPixels: 65})).toThrow(/maximumRadiusPixels/);
  expect(() => create({maximumRadiusPixels: 2.5})).toThrow(/maximumRadiusPixels/);
  expect(() => create({incompleteNeighborhood: 'drop' as never})).toThrow(/incompleteNeighborhood/);
  expect(() =>
    create({parameters: createTransientView(graph, 'short-settings', 'float32', 7)})
  ).toThrow(/parameters/);
  expect(() =>
    create({}, {summitMask: createTransientView(graph, 'short-mask', 'uint32', 29)})
  ).toThrow(/one value per pixel/);
  expect(() => create({}, {drop: createTransientView(graph, 'short-drop', 'float32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() =>
    create({}, {summitDrop: createTransientView(graph, 'orphan-drop', 'float32', 4)})
  ).toThrow(/requires output.summits/);
  expect(() =>
    create(
      {},
      {
        summits: createOutput(graph, 'o', 8),
        summitDrop: createTransientView(graph, 'short-column', 'float32', 7)
      }
    )
  ).toThrow(/summits.ids.length/);
  expect(() =>
    create({}, {status: {radiusClamped: createTransientView(graph, 'empty-overflow', 'uint32', 0)}})
  ).toThrow(/radiusClamped/);
  const shared = createBand(graph, 'shared', 30);
  expect(() => create({elevation: shared}, {drop: shared.storage.values})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainSummits({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: createBand(foreign, 'foreign', 4),
      parameters: createTransientView(foreign, 'foreign-settings', 'float32', 8),
      output: {summitMask: createTransientView(foreign, 'foreign-mask', 'uint32', 4)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainPeakSnap schedules search and unpack nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<GPUTerrainPeakSnapProps> = {},
    outputOverrides: Partial<GPUTerrainPeakSnapOutput> = {}
  ) => {
    instance++;
    return new GPUTerrainPeakSnap({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      candidates: createTransientView(graph, `candidates-${instance}`, 'float32x2', 3),
      output: {
        positions: createTransientView(graph, `positions-${instance}`, 'float32x2', 3),
        heights: createTransientView(graph, `heights-${instance}`, 'float32', 3),
        snapStatus: createTransientView(graph, `status-${instance}`, 'uint32', 3),
        ...outputOverrides
      },
      ...overrides
    });
  };
  const contributor = create();
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.slice(-2)).toEqual(['terrain-peak-snap-search', 'terrain-peak-snap-unpack']);
  expect(ids).not.toContain('terrain-peak-snap-overflow-clear');
  expect(
    create(
      {
        id: 'peaks',
        candidateHeights: createTransientView(graph, 'candidate-heights', 'float32', 3),
        candidateRadii: createTransientView(graph, 'candidate-radii', 'float32', 3),
        cellSizeMode: 'geographic'
      },
      {
        snapDistance: createTransientView(graph, 'distance', 'float32', 3),
        status: {radiusClamped: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    )
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
  const create = (
    overrides: Partial<GPUTerrainPeakSnapProps> = {},
    outputOverrides: Partial<GPUTerrainPeakSnapOutput> = {}
  ) => {
    instance++;
    return new GPUTerrainPeakSnap({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      candidates: createTransientView(graph, `candidates-${instance}`, 'float32x2', 3),
      output: {
        positions: createTransientView(graph, `positions-${instance}`, 'float32x2', 3),
        heights: createTransientView(graph, `heights-${instance}`, 'float32', 3),
        snapStatus: createTransientView(graph, `status-${instance}`, 'uint32', 3),
        ...outputOverrides
      },
      ...overrides
    });
  };
  expect(() => create({grid: makeGPURasterMetadata({width: 6, height: 0})})).toThrow(/dimensions/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({maximumRadiusPixels: 100})).toThrow(/maximumRadiusPixels/);
  expect(() => create({parameters: createTransientView(graph, 'short', 'float32', 7)})).toThrow(
    /parameters/
  );
  expect(() =>
    create({candidates: createTransientView(graph, 'wrong-format', 'float32', 6) as never})
  ).toThrow(/candidates/);
  expect(() =>
    create({candidates: createTransientView(graph, 'no-candidates', 'float32x2', 0)})
  ).toThrow(/at least one row/);
  for (const [name, view] of [
    ['candidateHeights', createTransientView(graph, 'h2', 'float32', 2)],
    ['candidateRadii', createTransientView(graph, 'r2', 'float32', 2)]
  ] as const) {
    expect(() => create({[name]: view})).toThrow(/one value per candidate/);
  }
  for (const [name, view] of [
    ['heights', createTransientView(graph, 'o2', 'float32', 2)],
    ['snapDistance', createTransientView(graph, 'd2', 'float32', 2)],
    ['positions', createTransientView(graph, 'p2', 'float32x2', 2)],
    ['snapStatus', createTransientView(graph, 's2', 'uint32', 2)]
  ] as const) {
    expect(() => create({}, {[name]: view})).toThrow(/one value per candidate/);
  }
  expect(() =>
    create({}, {status: {radiusClamped: createTransientView(graph, 'empty-overflow', 'uint32', 0)}})
  ).toThrow(/radiusClamped/);
  const candidates = createTransientView(graph, 'aliased', 'float32x2', 3);
  expect(() => create({candidates}, {positions: candidates})).toThrow(/share buffers/);
  const foreign = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainPeakSnap({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: createBand(foreign, 'foreign', 4),
      parameters: createTransientView(foreign, 'foreign-settings', 'float32', 8),
      candidates: createTransientView(foreign, 'foreign-candidates', 'float32x2', 1),
      output: {
        positions: createTransientView(foreign, 'foreign-positions', 'float32x2', 1),
        heights: createTransientView(foreign, 'foreign-heights', 'float32', 1),
        snapStatus: createTransientView(foreign, 'foreign-status', 'uint32', 1)
      }
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('review C-17: one pixel per row reports a clamped summit radius', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUTerrainSummits({
    grid: makeGPURasterMetadata({width: 6, height: 5}),
    elevation: createBand(graph, 'elevation', 30),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    output: {
      summitMask: createTransientView(graph, 'mask', 'uint32', 30),
      status: {radiusClamped: createTransientView(graph, 'radius-overflow', 'uint32', 1)}
    }
  }).getCommandNodes(graph);
  const summits = getRasterPipelines(nodes).find(pipeline => pipeline.id.endsWith('-summits'))!;
  const atomics = summits.source.match(/atomicMax\(&overflowValues/g) ?? [];
  expect(atomics).toHaveLength(1);
  expect(summits.source).toMatch(/if \(column == 0 && isRadiusClamped\(requested, cell\)\)/);
  device.destroy();
});

it('summit and peak snap packers write the grid cell size and row edges', () => {
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [0.5, 0.25], origin: [3, 46]});
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  expect([cellSize[0], cellSize[1], northEdge, southEdge]).toEqual([0.5, 0.25, 46, 44.75]);
  const summits = getGPUTerrainSummitsParameterValues({radius: 2, minimumDrop: 1}, grid);
  expect(Array.from(summits)).toEqual([2, 1, 0.5, 0.25, 46, 44.75, 0, 0]);
  const snap = getGPUTerrainPeakSnapParameterValues({interior: false}, grid);
  expect(Array.from(snap)).toEqual([150, 300, 80, 0.5, 0.25, 46, 44.75, 0]);
  expect(() =>
    getGPUTerrainSummitsParameterValues(
      {radius: 2},
      makeGPURasterMetadata({width: 1, height: 1, cellSize: 0})
    )
  ).toThrow(/cell size/);
});
