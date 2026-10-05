// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTileLODFrustumPlanes,
  getGPUTileLODQuadtreeTile,
  GPU_TILE_LOD_VIEW_OFFSETS,
  GPUTileLODSelection,
  makeGPUTileLODQuadtree,
  getGPUTileLODViewParameterValues,
  type GPUTileLODSelectionProps
} from '../../../src/map-graphs/tile-lod-selection';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {FIXTURE_A, makeFixtureView, selectTileLODOnCPU, UNLIMITED} from './tile-lod-oracle';

function createProps(
  graph: GPUCommandGraph,
  prefix: string = '',
  overrides: Partial<GPUTileLODSelectionProps> = {}
): GPUTileLODSelectionProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x4' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${prefix}${name}`, format, length);
  return {
    hierarchy: {
      sphereBounds: view('bounds', 'float32x4', 7),
      geometricErrors: view('errors', 'float32', 7),
      children: view('children', 'uint32x2', 7),
      levelOffsets: [0, 1, 3, 7],
      tileIds: view('tile-ids', 'uint32', 7),
      parents: view('parents', 'uint32', 7),
      nodeCosts: view('costs', 'uint32', 7)
    },
    view: view('view', 'float32', 56),
    output: {
      ids: view('ids', 'uint32', 7),
      count: view('count', 'uint32', 1),
      overflow: view('overflow', 'uint32', 1)
    },
    ...overrides
  };
}

it('GPUTileLODSelection schedules level passes and optional outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUTileLODSelection(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id)
    .filter(id => !id.startsWith('tile-lod-selection-draw-visibility'));
  expect(ids).toEqual(
    ['prepare', 'level-0', 'level-1', 'level-2', 'emit', 'draw-publish'].map(
      step => `tile-lod-selection-${step}`
    )
  );

  const budgetGraph = new GPUCommandGraph(device);
  const props = createProps(budgetGraph, '', {
    id: 'tiles',
    budget: createTransientView(budgetGraph, 'budget', 'uint32', 2),
    statistics: createTransientView(budgetGraph, 'statistics', 'uint32', 8),
    drawnAncestors: createTransientView(budgetGraph, 'ancestors', 'uint32', 7)
  });
  props.requests = {
    ids: createTransientView(budgetGraph, 'request-ids', 'uint32', 3),
    count: createTransientView(budgetGraph, 'request-count', 'uint32', 1),
    overflow: createTransientView(budgetGraph, 'request-overflow', 'uint32', 1),
    priorities: createTransientView(budgetGraph, 'request-priorities', 'float32', 3)
  };
  const budgetIds = new GPUTileLODSelection(props)
    .getCommandNodes(budgetGraph)
    .map(node => node.id);
  for (const id of [
    'tiles-level-0-evaluate',
    'tiles-level-0-budget',
    'tiles-level-0-decide',
    'tiles-level-2-decide',
    'tiles-indirect',
    'tiles-request-publish',
    'tiles-drawn-ancestors'
  ]) {
    expect(budgetIds).toContain(id);
  }
  expect(budgetIds.some(id => id.startsWith('tiles-request-visibility'))).toBe(true);
  expect(budgetIds.some(id => id.startsWith('tiles-request-priorities'))).toBe(true);
  expect(budgetIds.every(id => id.startsWith('tiles-'))).toBe(true);

  const sharedGraph = new GPUCommandGraph(device);
  const first = new GPUTileLODSelection({...createProps(sharedGraph, 'a-'), id: 'a'});
  const second = new GPUTileLODSelection({...createProps(sharedGraph, 'b-'), id: 'b'});
  const firstIds = first.getCommandNodes(sharedGraph).map(node => node.id);
  expect(second.getCommandNodes(sharedGraph).some(node => firstIds.includes(node.id))).toBe(false);
  device.destroy();
});

it('GPUTileLODSelection validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUTileLODSelectionProps>) =>
    new GPUTileLODSelection({...base, ...overrides});
  expect(() => create({hierarchy: {...base.hierarchy, levelOffsets: [0, 3, 2, 7]}})).toThrow(
    /levelOffsets/
  );
  expect(() =>
    create({
      hierarchy: {
        ...base.hierarchy,
        geometricErrors: createTransientView(graph, 'e6', 'float32', 6)
      }
    })
  ).toThrow(/geometricErrors length/);
  expect(() => create({view: createTransientView(graph, 'v55', 'float32', 55)})).toThrow(/view/);
  expect(() => create({budget: createTransientView(graph, 'b1', 'uint32', 1)})).toThrow(/budget/);
  expect(() => create({statistics: createTransientView(graph, 's7', 'uint32', 7)})).toThrow(
    /statistics/
  );
  expect(() =>
    create({
      requests: {
        ids: createTransientView(graph, 'r-ids', 'uint32', 3),
        count: createTransientView(graph, 'r-count', 'uint32', 1),
        overflow: createTransientView(graph, 'r-overflow', 'uint32', 1),
        priorities: createTransientView(graph, 'r-priorities', 'float32', 2)
      }
    })
  ).toThrow(/priorities/);
  expect(() =>
    create({
      hierarchy: {...base.hierarchy, parents: undefined},
      drawnAncestors: createTransientView(graph, 'anc', 'uint32', 7)
    })
  ).toThrow(/parents/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      output: {...base.output, ids: graph.createDataView(shared, {format: 'uint32', length: 7})},
      drawMask: graph.createDataView(shared, {format: 'uint32', length: 7})
    })
  ).toThrow(/overlap/);
  const commands = graph.createTransientBuffer({id: 'commands', byteLength: 16, usage: 128 | 256});
  const words = graph.createDataView(commands, {format: 'uint32', length: 4});
  expect(() =>
    create({
      indirectDraw: {
        commands: {
          type: 'draw',
          capacity: 1,
          recordByteLength: 16,
          buffer: commands,
          words,
          instanceCounts: words,
          firstInstances: words
        },
        commandIndex: 1
      }
    })
  ).toThrow(/commandIndex/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUTileLODSelection(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});

it('getGPUTileLODViewParameterValues and the quadtree helpers pack the documented layout', () => {
  const values = getGPUTileLODViewParameterValues({
    viewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    cameraPosition: [1, 2, 3],
    viewportSize: [200, 100],
    maximumScreenSpaceError: 4
  });
  expect(values[GPU_TILE_LOD_VIEW_OFFSETS.cameraPosition + 2]).toBe(3);
  expect(values[GPU_TILE_LOD_VIEW_OFFSETS.maximumScreenSpaceError]).toBe(4);
  expect(values[GPU_TILE_LOD_VIEW_OFFSETS.pixelProjectionScale]).toBeCloseTo(
    100 / (2 * Math.tan(Math.PI / 6)),
    4
  );
  expect(values[GPU_TILE_LOD_VIEW_OFFSETS.foveationRadius]).toBeCloseTo(0.15, 6);
  expect(
    Array.from(
      getGPUTileLODFrustumPlanes([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).slice(0, 8)
    )
  ).toEqual([1, 0, 0, 1, -1, 0, 0, 1]);
  expect(() =>
    getGPUTileLODViewParameterValues({
      viewProjectionMatrix: [1],
      cameraPosition: [0, 0, 0],
      viewportSize: [1, 1],
      maximumScreenSpaceError: 1
    })
  ).toThrow();

  const quadtree = makeGPUTileLODQuadtree({bounds: [0, 0, 4, 4], maximumLevel: 2});
  expect(quadtree.nodeCount).toBe(21);
  expect(quadtree.levelOffsets).toEqual([0, 1, 5, 21]);
  expect(Array.from(quadtree.children.slice(4, 6))).toEqual([9, 4]);
  expect(getGPUTileLODQuadtreeTile(9, 2)).toEqual({level: 2, x: 2, y: 0});
  expect(getGPUTileLODQuadtreeTile(8, 2)).toEqual({level: 2, x: 1, y: 1});
  expect(Array.from(quadtree.parents.slice(9, 13))).toEqual([2, 2, 2, 2]);
  expect(quadtree.sphereBounds[3]).toBeCloseTo(0.5 * Math.hypot(4, 4), 5);
  expect(quadtree.geometricErrors[0]).toBeCloseTo(4 / 256, 7);
  expect(() => makeGPUTileLODQuadtree({bounds: [0, 0, 0, 1], maximumLevel: 1})).toThrow();
  expect(() => makeGPUTileLODQuadtree({bounds: [0, 0, 1, 1], maximumLevel: 13})).toThrow();
});

it('selectTileLODOnCPU pins the fixture expectations', () => {
  const at = (threshold: number) => makeFixtureView({maximumScreenSpaceError: threshold});
  expect(selectTileLODOnCPU(FIXTURE_A, at(100)).drawnIds).toEqual([1000]);
  expect(selectTileLODOnCPU(FIXTURE_A, at(50)).drawnIds).toEqual([1100, 1101]);
  expect(selectTileLODOnCPU(FIXTURE_A, at(10)).drawnIds).toEqual([1101, 1200, 1201]);
  expect(selectTileLODOnCPU(FIXTURE_A, at(5)).statistics).toEqual([4, 160, 4, 160, 0, 0, 7, 0]);
  const capped = selectTileLODOnCPU(FIXTURE_A, at(5), {budget: [150, UNLIMITED]});
  expect(capped.drawnIds).toEqual([1101, 1200, 1201]);
  expect(capped.statistics[1]).toBe(140);
  expect(capped.statistics[5]).toBe(1);
  const standIn = selectTileLODOnCPU(
    {...FIXTURE_A, residency: Uint32Array.from([1, 1, 1, 1, 0, 1, 1])},
    at(5)
  );
  expect(standIn.drawnIds).toEqual([1100, 1202, 1203]);
  expect(standIn.requestedIds).toEqual([1201]);
});
