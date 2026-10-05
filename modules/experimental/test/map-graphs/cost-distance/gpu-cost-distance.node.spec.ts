// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUCostDistanceParameterValues,
  GPUCostDistance,
  GPUCostDistancePath,
  GPU_COST_DISTANCE_NONE,
  GPU_COST_DISTANCE_PARAMETER_LENGTH,
  GPU_RASTER_D8_DIRECTIONS,
  GPU_RASTER_MAXIMUM_ITERATIONS,
  type GPUCostDistanceProps
} from '../../../src/map-graphs/cost-distance';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

const WIDTH = 20;
const HEIGHT = 10;
const CELLS = WIDTH * HEIGHT;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCostDistanceProps> = {}
): GPUCostDistanceProps {
  return {
    width: WIDTH,
    height: HEIGHT,
    friction: {
      id: 'friction',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: createTransientView(graph, 'friction', 'float32', CELLS)
      }
    },
    settings: createTransientView(graph, 'settings', 'float32', 8),
    sources: createTransientView(graph, 'sources', 'uint32', 3),
    costs: createTransientView(graph, 'costs', 'float32', CELLS),
    ...overrides
  };
}

it('exports constants and packs settings', () => {
  expect(GPU_COST_DISTANCE_PARAMETER_LENGTH).toBe(8);
  expect(GPU_COST_DISTANCE_NONE).toBe(0xffffffff);
  expect(GPU_RASTER_MAXIMUM_ITERATIONS).toBe(1024);
  expect(GPU_RASTER_D8_DIRECTIONS.map(direction => direction.code)).toEqual([
    1, 2, 4, 8, 16, 32, 64, 128
  ]);
  expect(
    Array.from(
      getGPUCostDistanceParameterValues({
        cellSize: [3, 4],
        northEdge: 0.25,
        southEdge: 0.5
      })
    )
  ).toEqual([3, 4, 0.25, 0.5, Infinity, 0, 0, 0]);
  expect(Array.from(getGPUCostDistanceParameterValues({cellSize: [1, 1], costLimit: 7}))[4]).toBe(
    7
  );
  expect(() => getGPUCostDistanceParameterValues({cellSize: [1, 1]}, new Float32Array(4))).toThrow(
    /8 values/
  );
});

it('GPUCostDistance prefixes every node ID and grows two nodes per iteration', () => {
  const device = createNullWebGPUDevice();
  const countNodes = (maxIterations: number) => {
    const graph = new GPUCommandGraph(device);
    const recipe = new GPUCostDistance({
      ...createProps(graph),
      id: 'cd',
      maxIterations,
      sourceMask: createTransientView(graph, 'mask', 'uint32', CELLS),
      backLinks: createTransientView(graph, 'back-links', 'uint32', CELLS),
      bandThresholds: createTransientView(graph, 'thresholds', 'float32', 2),
      bands: createTransientView(graph, 'bands', 'uint32', CELLS),
      bandCounts: createTransientView(graph, 'band-counts', 'uint32', 2),
      converged: createTransientView(graph, 'converged', 'uint32', 1),
      iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
    });
    expect(recipe.recipe).toBe('cost-distance');
    expect(recipe.maxIterations).toBe(maxIterations);
    return recipe.getCommandNodes(graph);
  };
  const small = countNodes(2);
  const large = countNodes(5);
  expect(large.length - small.length).toBe(6);
  for (const node of large) {
    expect(node.id.startsWith('cd-')).toBe(true);
  }
  const ids = large.map(node => node.id);
  expect(ids).toContain('cd-seed');
  expect(ids).toContain('cd-finalize');
  expect(ids).toContain('cd-back-links');
  // Back-links add the cycle-safe tie phase: prepare, roots, and a gated tile relaxation.
  expect(ids).toContain('cd-tie-prepare');
  expect(ids).toContain('cd-tie-roots');
  expect(ids).toContain('cd-tie-relax-7');
  expect(ids).not.toContain('cd-tie-relax-8');
  expect(ids).toContain('cd-bands');
  expect(ids).toContain('cd-relax-4');
  expect(ids).toContain('cd-relax-gate-4');
  expect(new Set(ids).size).toBe(ids.length);
  const relax = large.find(node => node.id === 'cd-relax-0');
  expect(relax?.condition).toMatchObject({source: 'gpu', mode: 'indirect'});

  const minimalGraph = new GPUCommandGraph(device);
  const minimal = new GPUCostDistance(createProps(minimalGraph)).getCommandNodes(minimalGraph);
  expect(minimal.every(node => node.id.startsWith('cost-distance-'))).toBe(true);
  expect(minimal.map(node => node.id)).not.toContain('cost-distance-finalize');
  device.destroy();
});

it('GPUCostDistance validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUCostDistanceProps>) =>
    new GPUCostDistance({...base, ...overrides});
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({width: 1.5})).toThrow(/dimensions/);
  expect(() => create({costs: createTransientView(graph, 'costs19', 'float32', 19)})).toThrow(
    /costs must contain one value per cell/
  );
  expect(() =>
    create({
      costs: createTransientView(graph, 'uint-costs', 'uint32', CELLS) as never
    })
  ).toThrow(/costs/);
  expect(() => create({settings: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /settings/
  );
  expect(() => create({sources: undefined})).toThrow(/sources or sourceMask/);
  expect(() =>
    create({
      sourceCosts: createTransientView(graph, 'source-costs', 'float32', 2)
    })
  ).toThrow(/sourceCosts length/);
  expect(() =>
    create({
      sourceCount: createTransientView(graph, 'source-count', 'uint32', 2)
    })
  ).toThrow(/sourceCount must contain exactly one row/);
  expect(() => create({sourceMask: createTransientView(graph, 'mask5', 'uint32', 5)})).toThrow(
    /sourceMask/
  );
  expect(() => create({backLinks: createTransientView(graph, 'links5', 'uint32', 5)})).toThrow(
    /backLinks/
  );
  for (const maxTieIterations of [0, 1025, 1.5]) {
    expect(() => create({maxTieIterations})).toThrow(/maxTieIterations/);
  }
  for (const maxIterations of [0, 1025, 1.5]) {
    expect(() => create({maxIterations})).toThrow(/maxIterations/);
  }
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({bands: createTransientView(graph, 'bands', 'uint32', CELLS)})).toThrow(
    /bandThresholds/
  );
  const thresholds = createTransientView(graph, 'thresholds', 'float32', 2);
  expect(() => create({bandThresholds: thresholds})).toThrow(/bandThresholds/);
  expect(() =>
    create({
      bandThresholds: thresholds,
      bandCounts: createTransientView(graph, 'band-counts', 'uint32', 3)
    })
  ).toThrow(/bandCounts length/);
  expect(() =>
    create({
      converged: createTransientView(graph, 'converged2', 'uint32', 2)
    })
  ).toThrow(/converged must contain exactly one row/);
  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 1024,
    usage: 128
  });
  expect(() =>
    create({
      sources: graph.createDataView(shared, {format: 'uint32', length: 3}),
      costs: graph.createDataView(shared, {format: 'float32', length: CELLS})
    })
  ).toThrow(/must not share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUCostDistance(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});

it('GPUCostDistancePath schedules a walk and publish node and validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    id: 'path',
    width: WIDTH,
    height: HEIGHT,
    backLinks: createTransientView(graph, 'back-links', 'uint32', CELLS),
    target: createTransientView(graph, 'target', 'uint32', 1),
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 8),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1),
      totalCount: createTransientView(graph, 'total-count', 'uint32', 1)
    }
  };
  const recipe = new GPUCostDistancePath(props);
  expect(recipe.recipe).toBe('cost-distance-path');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['path-walk', 'path-publish']);
  expect(new GPUCostDistancePath({...props, id: undefined}).id).toBe('cost-distance-path');
  expect(
    () =>
      new GPUCostDistancePath({
        ...props,
        backLinks: createTransientView(graph, 'links5', 'uint32', 5)
      })
  ).toThrow(/backLinks/);
  expect(
    () =>
      new GPUCostDistancePath({
        ...props,
        target: createTransientView(graph, 'target2', 'uint32', 2)
      })
  ).toThrow(/target must contain exactly one row/);
  expect(
    () =>
      new GPUCostDistancePath({
        ...props,
        output: {
          ...props.output,
          ids: createTransientView(graph, 'ids0', 'uint32', 0)
        }
      })
  ).toThrow(/capacity/);
  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 1024,
    usage: 128
  });
  expect(
    () =>
      new GPUCostDistancePath({
        ...props,
        output: {
          ...props.output,
          ids: graph.createDataView(shared, {format: 'uint32', length: 4}),
          count: graph.createDataView(shared, {format: 'uint32', length: 1})
        }
      })
  ).toThrow(/must not share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUCostDistancePath({
      ...props,
      backLinks: createTransientView(otherGraph, 'back-links', 'uint32', CELLS)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('attributes internal transient ID collisions to the recipe that generated them', () => {
  const device = createNullWebGPUDevice();
  // GPUCostDistancePath creates `${id}-total`.
  let graph = new GPUCommandGraph(device);
  graph.createTransientBuffer({id: 'path-total', byteLength: 64, usage: 128});
  const path = new GPUCostDistancePath({
    id: 'path',
    width: WIDTH,
    height: HEIGHT,
    backLinks: createTransientView(graph, 'back-links', 'uint32', CELLS),
    target: createTransientView(graph, 'target', 'uint32', 1),
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 16),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    }
  });
  expect(() => path.getCommandNodes(graph)).toThrow(/GPUCostDistancePath "path".*"path-total"/);
  try {
    path.getCommandNodes(graph);
  } catch (error) {
    expect((error as Error).cause).toBeInstanceOf(Error);
    expect(((error as Error).cause as Error).message).toMatch(/already in use/);
  }

  // GPUCostDistance creates `${id}-friction-auxiliary` (and the iteration state transients).
  graph = new GPUCommandGraph(device);
  graph.createTransientBuffer({
    id: 'cd-friction-auxiliary',
    byteLength: 1024,
    usage: 128
  });
  const costDistance = new GPUCostDistance({...createProps(graph), id: 'cd'});
  expect(() => costDistance.getCommandNodes(graph)).toThrow(
    /GPUCostDistance "cd".*"cd-friction-auxiliary"/
  );

  // The shared iteration state of the relaxation also reports its owner.
  graph = new GPUCommandGraph(device);
  graph.createTransientBuffer({
    id: 'cd-relax-status',
    byteLength: 16,
    usage: 128
  });
  expect(() =>
    new GPUCostDistance({...createProps(graph), id: 'cd'}).getCommandNodes(graph)
  ).toThrow(/GPUCostDistance "cd".*"cd-relax-status"/);
  device.destroy();
});
