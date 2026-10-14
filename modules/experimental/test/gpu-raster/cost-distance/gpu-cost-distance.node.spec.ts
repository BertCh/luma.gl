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
  GPU_COST_DISTANCE_UNRESOLVED,
  GPU_RASTER_D8_DIRECTIONS,
  GPU_RASTER_MAXIMUM_ITERATIONS,
  type GPUCostDistanceOutput,
  type GPUCostDistanceProps
} from '../../../src/gpu-raster/cost-distance';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeCostDistance,
  getEdgeCost,
  isPassable,
  type CostDistanceOracleOptions
} from './cost-distance-oracle';
import {
  computeBackLinksTwin,
  emulateTiledRelaxation,
  getNeighbor,
  walkBackLinksPreviousTwin,
  walkBackLinksTwin
} from './raster-relaxation-emulator';

const WIDTH = 20;
const HEIGHT = 10;
const CELLS = WIDTH * HEIGHT;
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT});

type PropsOverrides = Partial<Omit<GPUCostDistanceProps, 'output'>> & {
  /** Merged over the default `costs` output. */
  output?: Partial<GPUCostDistanceOutput>;
};

function createProps(graph: GPUCommandGraph, overrides: PropsOverrides = {}): GPUCostDistanceProps {
  const {output, ...rest} = overrides;
  return {
    grid: GRID,
    friction: {
      id: 'friction',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: createTransientView(graph, 'friction', 'float32', CELLS)
      }
    },
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    sources: createTransientView(graph, 'sources', 'uint32', 3),
    ...rest,
    output: {costs: createTransientView(graph, 'costs', 'float32', CELLS), ...output}
  };
}

it('exports constants and packs settings', () => {
  expect(GPU_COST_DISTANCE_PARAMETER_LENGTH).toBe(8);
  expect(GPU_COST_DISTANCE_NONE).toBe(0xffffffff);
  expect(GPU_COST_DISTANCE_UNRESOLVED).toBe(0xfffffffe);
  expect(GPU_RASTER_MAXIMUM_ITERATIONS).toBe(1024);
  expect(GPU_RASTER_D8_DIRECTIONS.map(direction => direction.code)).toEqual([
    1, 2, 4, 8, 16, 32, 64, 128
  ]);
  // Cell size and row edges come from the grid.
  const grid = makeGPURasterMetadata({
    width: WIDTH,
    height: HEIGHT,
    cellSize: [3, 4],
    origin: [10, 100]
  });
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  expect(Array.from(getGPUCostDistanceParameterValues({}, grid))).toEqual([
    cellSize[0],
    cellSize[1],
    northEdge,
    southEdge,
    Infinity,
    0,
    0,
    0
  ]);
  expect([cellSize, northEdge, southEdge]).toEqual([[3, 4], 100, 60]);
  expect(Array.from(getGPUCostDistanceParameterValues({costLimit: 7}, GRID))[4]).toBe(7);
  expect(() => getGPUCostDistanceParameterValues({}, GRID, new Float32Array(4))).toThrow(
    /8 values/
  );
});

it('GPUCostDistance prefixes every node ID and grows one node per iteration and one gate per four', () => {
  const device = createNullWebGPUDevice();
  const countNodes = (maximumIterations: number) => {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUCostDistance({
      ...createProps(graph, {
        output: {
          backLinks: createTransientView(graph, 'back-links', 'uint32', CELLS),
          bands: createTransientView(graph, 'bands', 'uint32', CELLS),
          bandCounts: createTransientView(graph, 'band-counts', 'uint32', 2),
          status: {
            converged: createTransientView(graph, 'converged', 'uint32', 1),
            iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
          }
        }
      }),
      id: 'cd',
      maximumIterations,
      maximumTieIterations: 8,
      sourceMask: createTransientView(graph, 'mask', 'uint32', CELLS),
      costBreaks: createTransientView(graph, 'thresholds', 'float32', 2)
    });
    expect(contributor.maximumIterations).toBe(maximumIterations);
    return contributor.getCommandNodes(graph);
  };
  const small = countNodes(2);
  const large = countNodes(5);
  expect(large.length - small.length).toBe(4);
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

  // The tie budget defaults to maximumIterations and follows the cost phase's sweep schedule.
  const tieGraph = new GPUCommandGraph(device);
  const tieContributor = new GPUCostDistance({
    ...createProps(tieGraph, {
      output: {backLinks: createTransientView(tieGraph, 'tie-back-links', 'uint32', CELLS)}
    }),
    id: 'tie',
    maximumIterations: 100,
    sweepAfterIteration: 96
  });
  expect(tieContributor.maximumTieIterations).toBe(100);
  const tieNodes = tieContributor.getCommandNodes(tieGraph);
  const variantOf = (id: string) =>
    (tieNodes.find(node => node.id === id)?.workload as {variant?: string} | undefined)?.variant;
  expect(variantOf('tie-tie-relax-99')).toBe('sweep-down');
  expect(variantOf('tie-tie-relax-95')).toBe('tiled-relax');
  expect(variantOf('tie-tie-relax-97')).toBe('sweep-right');
  expect(variantOf('tie-relax-97')).toBe('sweep-right');
  expect(tieNodes.map(node => node.id)).not.toContain('tie-tie-relax-100');

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
  const create = (overrides: PropsOverrides) =>
    new GPUCostDistance({...base, ...overrides, output: {...base.output, ...overrides.output}});
  const createWithGrid = (width: number) => create({grid: {...GRID, width}});
  expect(() => createWithGrid(0)).toThrow(/dimensions/);
  expect(() => createWithGrid(1.5)).toThrow(/dimensions/);
  expect(() =>
    create({output: {costs: createTransientView(graph, 'costs19', 'float32', 19)}})
  ).toThrow(/output.costs must contain one value per cell/);
  expect(() =>
    create({
      output: {costs: createTransientView(graph, 'uint-costs', 'uint32', CELLS) as never}
    })
  ).toThrow(/output.costs/);
  expect(() => create({parameters: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /parameters/
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
  expect(() =>
    create({output: {backLinks: createTransientView(graph, 'links5', 'uint32', 5)}})
  ).toThrow(/output.backLinks/);
  for (const maximumTieIterations of [0, 1025, 1.5]) {
    expect(() => create({maximumTieIterations})).toThrow(/maximumTieIterations/);
  }
  for (const maximumIterations of [0, 1025, 1.5]) {
    expect(() => create({maximumIterations})).toThrow(/maximumIterations/);
  }
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() =>
    create({output: {bands: createTransientView(graph, 'bands', 'uint32', CELLS)}})
  ).toThrow(/costBreaks/);
  const costBreaks = createTransientView(graph, 'thresholds', 'float32', 2);
  expect(() => create({costBreaks})).toThrow(/costBreaks/);
  expect(() =>
    create({
      costBreaks,
      output: {bandCounts: createTransientView(graph, 'band-counts', 'uint32', 3)}
    })
  ).toThrow(/bandCounts length/);
  expect(() =>
    create({
      output: {status: {converged: createTransientView(graph, 'converged2', 'uint32', 2)}}
    })
  ).toThrow(/output.status.converged must contain exactly one row/);
  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 1024,
    usage: 128
  });
  expect(() =>
    create({
      sources: graph.createDataView(shared, {format: 'uint32', length: 3}),
      output: {costs: graph.createDataView(shared, {format: 'float32', length: CELLS})}
    })
  ).toThrow(/must not share buffers/);
  // Latitude-dependent cell sizes need an axis-aligned grid.
  expect(() =>
    create({cellSizeMode: 'geographic', grid: {...GRID, affine: [1, 0.5, 0, 0, -1, 0]}})
  ).toThrow(/axis-aligned/);
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
    grid: GRID,
    backLinks: createTransientView(graph, 'back-links', 'uint32', CELLS),
    target: createTransientView(graph, 'target', 'uint32', 1),
    output: {
      path: {
        ids: createTransientView(graph, 'ids', 'uint32', 8),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          requiredCount: createTransientView(graph, 'total-count', 'uint32', 1)
        }
      }
    }
  };
  const contributor = new GPUCostDistancePath(props);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['path-walk', 'path-publish']);
  const flagged = new GPUCostDistancePath({
    ...props,
    id: 'flagged-path',
    output: {
      ...props.output,
      status: {iterationLimitReached: createTransientView(graph, 'limit', 'uint32', 1)}
    }
  });
  expect(flagged.getCommandNodes(graph).map(node => node.id)).toEqual([
    'flagged-path-walk',
    'flagged-path-publish'
  ]);
  expect(
    () =>
      new GPUCostDistancePath({
        ...props,
        output: {
          ...props.output,
          status: {iterationLimitReached: createTransientView(graph, 'limit2', 'uint32', 2)}
        }
      })
  ).toThrow(/output.status.iterationLimitReached must contain exactly one row/);
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
          path: {
            ...props.output.path,
            ids: createTransientView(graph, 'ids0', 'uint32', 0)
          }
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
          path: {
            ...props.output.path,
            ids: graph.createDataView(shared, {format: 'uint32', length: 4}),
            status: {
              ...props.output.path.status,
              count: graph.createDataView(shared, {format: 'uint32', length: 1})
            }
          }
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

it('attributes internal transient ID collisions to the contributor that generated them', () => {
  const device = createNullWebGPUDevice();
  // GPUCostDistancePath creates `${id}-total`.
  let graph = new GPUCommandGraph(device);
  graph.createTransientBuffer({id: 'path-total', byteLength: 64, usage: 128});
  const path = new GPUCostDistancePath({
    id: 'path',
    grid: GRID,
    backLinks: createTransientView(graph, 'back-links', 'uint32', CELLS),
    target: createTransientView(graph, 'target', 'uint32', 1),
    output: {
      path: {
        ids: createTransientView(graph, 'ids', 'uint32', 16),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
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

/** Cost-distance inputs for the tie twins: zero-friction plateaus inside friction 3. */
function getTieScenario(width: number, height: number, plateau: (cell: number) => boolean) {
  const friction = Float32Array.from({length: width * height}, (_, cell) =>
    plateau(cell) ? 0 : 3
  );
  const options: CostDistanceOracleOptions = {
    width,
    height,
    friction,
    georeference: getRasterGridGeoreference(makeGPURasterMetadata({width, height})),
    sources: [{cell: 0, cost: 0}]
  };
  const costs = computeCostDistance(options);
  const passable = (cell: number) => isPassable(friction[cell]);
  // Twin of tie-prepare: strict tight predecessor (smallest candidate, 4 f32 ulp = 2^-21 relative), tie masks.
  const strictDirection = (cell: number) => {
    let best = Infinity;
    let bestDirection = -1;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(width, height, cell, direction);
      if (neighbor < 0 || !passable(neighbor) || !(costs[neighbor] < costs[cell])) continue;
      const candidate = costs[neighbor] + getEdgeCost(options, cell, direction);
      if (candidate < best) {
        best = candidate;
        bestDirection = direction;
      }
    }
    return bestDirection >= 0 && best <= costs[cell] * (1 + 2 ** -21) ? bestDirection : -1;
  };
  const masks = new Int32Array(width * height);
  const levels = new Float64Array(width * height).fill(Infinity);
  for (let cell = 0; cell < width * height; cell++) {
    if (!passable(cell) || !Number.isFinite(costs[cell])) continue;
    if (strictDirection(cell) >= 0 || cell === 0) levels[cell] = 0;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(width, height, cell, direction);
      if (neighbor < 0 || !passable(neighbor) || costs[neighbor] !== costs[cell]) continue;
      if (costs[neighbor] + getEdgeCost(options, cell, direction) <= costs[cell]) {
        masks[cell] |= 1 << direction;
      }
    }
  }
  const relax = (maximumIterations: number, sweepAfterIteration?: number) => {
    const values = Float64Array.from(levels);
    const status = emulateTiledRelaxation({
      width,
      height,
      values,
      participates: cell => passable(cell) && Number.isFinite(costs[cell]),
      getEdgeCost: (cell, direction) => (((masks[cell] >> direction) & 1) === 0 ? Infinity : 1),
      maxIterations: maximumIterations,
      sweepAfterIteration,
      activateAllTiles: true
    });
    return {...status, levels: values};
  };
  const backLinks = (relaxed: Float64Array, rule: 'current' | 'previous' = 'current') =>
    computeBackLinksTwin(
      {
        width,
        height,
        costs,
        passable,
        strictDirection,
        masks,
        levels: relaxed,
        costConverged: true
      },
      rule
    );
  return {costs, relax, backLinks};
}

it('review C-02: a zero-cost plateau deeper than the old 8-iteration tie budget gave a fake source', () => {
  // A 320-cell zero-friction strip on row 0 reached from source cell 0: every strip cell costs 0,
  // so only tie levels (0 at the source, +1 per cell) can link it back.
  const width = 320;
  const height = 4;
  const scenario = getTieScenario(width, height, cell => cell < width);
  const target = 300;
  expect(scenario.costs[target]).toBe(0);

  // Previous code: 8 plain tiled tie iterations (no sweeps) cover 8 tiles = 128 cells, and an
  // unreached level kept link 0, so the walk stopped at the target as if it were a source.
  const old = scenario.relax(8);
  expect(old.converged).toBe(false);
  expect(old.levels[127]).toBe(127);
  expect(old.levels[128]).toBe(Infinity);
  const oldLinks = scenario.backLinks(old.levels, 'previous');
  expect(oldLinks[target]).toBe(0);
  expect(walkBackLinksPreviousTwin(width, height, oldLinks, target)).toEqual([target]);

  // Same truncated levels with the current rule: unresolved, and the path reports it.
  const truncatedLinks = scenario.backLinks(old.levels);
  expect(truncatedLinks[target]).toBe(GPU_COST_DISTANCE_UNRESOLVED);
  expect(truncatedLinks[100]).toBe(16);
  expect(walkBackLinksTwin(width, height, truncatedLinks, target)).toEqual({
    path: [],
    iterationLimitReached: true
  });
  expect(walkBackLinksTwin(width, height, truncatedLinks, 100).path).toHaveLength(101);

  // Current default: the tie budget is maximumIterations (512) with sweeps after 96.
  const fixed = scenario.relax(512, 96);
  expect(fixed.converged).toBe(true);
  const links = scenario.backLinks(fixed.levels);
  expect(links.includes(GPU_COST_DISTANCE_UNRESOLVED)).toBe(false);
  const walk = walkBackLinksTwin(width, height, links, target);
  expect(walk.iterationLimitReached).toBe(false);
  expect(walk.path).toEqual(Array.from({length: target + 1}, (_, index) => target - index));
});

it('review C-02: the tie sweeps resolve a winding plateau that plain tiled passes cannot', () => {
  // 256^2 serpentine zero-friction corridor (rows with (row & 7) < 4, alternating gaps), about
  // 8000 cells deep. Emulated: plain tiled passes need 482 iterations, the sweep schedule 159.
  // With a budget of 256 only the sweeps (the cost phase's schedule, now shared) converge.
  const size = 256;
  const plateau = (cell: number) => {
    const row = Math.floor(cell / size);
    const column = cell % size;
    if ((row & 7) < 4) return true;
    return (row >> 3) % 2 === 0 ? column < 8 : column >= size - 8;
  };
  const scenario = getTieScenario(size, size, plateau);
  expect(scenario.relax(512).iterationCount).toBe(482);
  const plain = scenario.relax(256);
  const swept = scenario.relax(256, 96);
  expect(plain.converged).toBe(false);
  expect(swept.converged).toBe(true);
  expect(swept.iterationCount).toBe(159);
  const links = scenario.backLinks(swept.levels);
  expect(links.includes(GPU_COST_DISTANCE_UNRESOLVED)).toBe(false);
  const target = size * size - 1;
  const walk = walkBackLinksTwin(size, size, links, target);
  expect(walk.path[walk.path.length - 1]).toBe(0);
  expect(new Set(walk.path).size).toBe(walk.path.length);
  // The plain budget leaves the far end unresolved instead of linking it to a fake source.
  const plainLinks = scenario.backLinks(plain.levels);
  expect(plainLinks[target]).toBe(GPU_COST_DISTANCE_UNRESOLVED);
  expect(walkBackLinksTwin(size, size, plainLinks, target).iterationLimitReached).toBe(true);
}, 120000);

it('back-links of a cost relaxation that hit maximumIterations are all unresolved', () => {
  const scenario = getTieScenario(20, 10, () => false);
  const links = computeBackLinksTwin({
    width: 20,
    height: 10,
    costs: scenario.costs,
    passable: () => true,
    strictDirection: () => 0,
    masks: new Int32Array(200),
    levels: new Float64Array(200),
    costConverged: false
  });
  expect(links.every(link => link === GPU_COST_DISTANCE_UNRESOLVED)).toBe(true);
  expect(walkBackLinksTwin(20, 10, links, 57)).toEqual({path: [], iterationLimitReached: true});
  // Corrupt forests give no path rather than a truncated one: a cycle and an invalid code.
  const cycle = new Uint32Array(200).fill(0);
  cycle[1] = 16;
  cycle[0] = 1;
  expect(walkBackLinksTwin(20, 10, cycle, 1)).toEqual({path: [], iterationLimitReached: false});
  const invalid = new Uint32Array(200).fill(0);
  invalid[5] = 3;
  expect(walkBackLinksTwin(20, 10, invalid, 5).path).toEqual([]);
});
