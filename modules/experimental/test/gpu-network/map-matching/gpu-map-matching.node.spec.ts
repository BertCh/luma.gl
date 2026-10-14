// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  GPUMapMatching,
  type GPUMapMatchingProps
} from '../../../src/gpu-network/map-matching/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createManhattanGrid,
  createRouteQueries,
  getExactRouteDistance,
  searchRouteMirror,
  type RouteTablePolicy
} from './route-search-mirror';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  output: Partial<GPUMapMatchingProps['output']> = {}
): GPUMapMatchingProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `map-matching-view-${serial++}`, format, length);
  return {
    id: 'map-matching',
    points: view('float32x2', 6),
    trackOffsets: view('uint32', 2),
    nodePositions: view('float32x2', 4),
    offsets: view('uint32', 5),
    edgeTargets: view('uint32', 8),
    parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
    cellSize: 50,
    bounds: {minimum: [0, 0], maximum: [200, 200]},
    output: {matchedEdges: view('uint32', 6), ...output}
  };
}

type BudgetTally = {missed: number; wrong: number; silent: number; droppingSearches: number};

/** Runs the mirror over the B-02 queries and tallies misses against exact Dijkstra. */
function tallyRouteBudget(
  sampleSpacing: number,
  budget: number,
  policy: RouteTablePolicy
): BudgetTally {
  const grid = createManhattanGrid(41, 100);
  const tally: BudgetTally = {missed: 0, wrong: 0, silent: 0, droppingSearches: 0};
  for (const {start, target, limit} of createRouteQueries(41, 100, sampleSpacing, 227)) {
    const exact = getExactRouteDistance(grid, start, target, limit);
    const {targetDistances, droppedCount} = searchRouteMirror(
      grid,
      start,
      limit,
      [target],
      budget,
      policy
    );
    const found = targetDistances[0];
    const isWrong = exact < 0 ? found >= 0 : found < 0 || Math.abs(found - exact) > 1e-6;
    tally.missed += exact >= 0 && found < 0 ? 1 : 0;
    tally.wrong += isWrong ? 1 : 0;
    tally.silent += isWrong && droppedCount === 0 ? 1 : 0;
    tally.droppingSearches += droppedCount > 0 ? 1 : 0;
  }
  return tally;
}

// Review B-02: the route search's node table silently dropped nodes in discovery order. The mirror
// follows the WGSL search line for line (A*, table of `routeNodeBudget` nodes) on a 100 m
// Manhattan grid with the default bound 3 * straight + 50; the GPU spec runs the real kernel.
it('route search budget: the old drop policy misses most 1 km routes silently', () => {
  const old = tallyRouteBudget(1000, 64, 'reject');
  // 171 of 227 routes missed at 1 km spacing (the review's own mimic, with plain Dijkstra and
  // its own sampling, found 186), and nothing reported them before the fix.
  expect(old).toEqual({missed: 171, wrong: 171, silent: 0, droppingSearches: 173});
  expect(tallyRouteBudget(500, 64, 'reject').wrong).toBe(0);
});

it('route search budget: eviction keeps more routes and every wrong one is counted', () => {
  // Evicting the worst open key keeps the search heading to the target: the same budget covers
  // 800 m samples (64 nodes) and 1.1 km (128 nodes) exactly, where dropping missed 92 and 25.
  expect(tallyRouteBudget(800, 64, 'reject').missed).toBe(92);
  expect(tallyRouteBudget(800, 64, 'evict-worst')).toEqual({
    missed: 0,
    wrong: 0,
    silent: 0,
    droppingSearches: 92
  });
  expect(tallyRouteBudget(1100, 128, 'reject').missed).toBe(25);
  expect(tallyRouteBudget(1100, 128, 'evict-worst').wrong).toBe(0);
  // Past the budget's reach it still misses routes, but never silently: each wrong route comes
  // from a search that dropped nodes, which raises overflow and routeOverflow.
  const beyond = tallyRouteBudget(1000, 64, 'evict-worst');
  expect(beyond).toEqual({missed: 80, wrong: 80, silent: 0, droppingSearches: 173});
});

it('GPUMapMatching reports route-search drops through overflow and routeOverflow', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-status'});
  const overflow = createTransientView(graph, 'mm-overflow', 'uint32', 1);
  const routeOverflow = createTransientView(graph, 'mm-route-overflow', 'uint32', 1);
  const nodes = new GPUMapMatching(createProps(graph, {overflow, routeOverflow})).getCommandNodes(
    graph
  );
  const ids = nodes.map(node => node.id);
  const routes = ids.indexOf('map-matching-routes');
  const status = ids.indexOf('map-matching-route-status');
  expect(routes).toBeGreaterThanOrEqual(0);
  expect(status).toBeGreaterThan(routes);
  expect(ids.indexOf('map-matching-clear-route-overflow')).toBeGreaterThanOrEqual(0);
  const statusBuffers = nodes[status].resources?.map(resource => resource.buffer) ?? [];
  expect(statusBuffers).toContain(overflow);
  expect(statusBuffers).toContain(routeOverflow);
  // The route kernel stays within the default 8 storage buffers.
  expect(nodes[routes].resources?.length).toBe(8);

  expect(
    () =>
      new GPUMapMatching(
        createProps(graph, {routeOverflow: createTransientView(graph, 'mm-bad', 'uint32', 2)})
      )
  ).toThrow(/routeOverflow must contain 1 rows/);
});

it('GPUMapMatching clears the whole edge table before the per-node edge kernel', () => {
  // Review B-04: the per-node kernel writes only rows offsets[i]..offsets[i + 1], so the rows
  // past offsets[NODE_COUNT] kept stale words and became zero-length or phantom edges.
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-edge-table'});
  const ids = new GPUMapMatching(createProps(graph)).getCommandNodes(graph).map(node => node.id);
  const clear = ids.indexOf('map-matching-clear-edge-table');
  expect(clear).toBeGreaterThanOrEqual(0);
  expect(clear).toBeLessThan(ids.indexOf('map-matching-edge-table'));
});
