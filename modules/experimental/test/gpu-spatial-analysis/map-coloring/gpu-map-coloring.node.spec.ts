// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUMapColoringParameterValues,
  GPUMapColoring,
  GPU_MAP_COLORING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/map-coloring/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import fixture from '../../oracles/fixtures/geometry-map-coloring.json';
import {createSeededRandom} from '../spatial-weights/spatial-weights-oracle';
import {
  computeMapColoringOracle,
  countColoringConflicts,
  createCliqueCSR,
  createOracleCSRFromEdges,
  createRandomPlanarCSR,
  createRookLatticeCSR,
  createStarCSR,
  getMapColoringOrder,
  simulateMapColoringRounds
} from './map-coloring-oracle';

function setup(id: string = 'map-coloring-nodes') {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const props = {
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    parameters: view('uint32', GPU_MAP_COLORING_PARAMETER_LENGTH),
    output: {colors: view('uint32', 4)}
  };
  return {graph, view, props};
}

it('GPUMapColoring validates props and declares gated rounds', () => {
  const {graph, view, props: base} = setup();
  expect(() => new GPUMapColoring({...base, output: {colors: view('uint32', 3)}})).toThrow(
    /row count/
  );
  expect(() => new GPUMapColoring({...base, maximumIterations: 0})).toThrow(/maximumIterations/);
  expect(() => new GPUMapColoring({...base, parameters: view('uint32', 0)})).toThrow(/parameters/);
  const ids = new GPUMapColoring({
    ...base,
    maximumIterations: 3,
    output: {...base.output, conflictCount: view('uint32', 1)}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('map-coloring-round-2-select');
  expect(ids).not.toContain('map-coloring-round-3-select');
  expect(ids.at(-1)).toBe('map-coloring-conflicts');
});

it('GPUMapColoring packs the seed per frame', () => {
  expect(Array.from(getGPUMapColoringParameterValues())).toEqual([0]);
  expect(Array.from(getGPUMapColoringParameterValues({seed: 0xffffffff}))).toEqual([0xffffffff]);
  expect(() => getGPUMapColoringParameterValues({seed: -1})).toThrow(/seed/);
  expect(() => getGPUMapColoringParameterValues({seed: 1.5})).toThrow(/seed/);
});

it('GPUMapColoring keeps one pipeline per kernel across rounds (rule 12)', () => {
  const build = (maximumIterations: number) => {
    const {graph, view, props} = setup();
    return new GPUMapColoring({
      ...props,
      maximumIterations,
      output: {...props.output, conflictCount: view('uint32', 1)}
    }).getCommandNodes(graph);
  };
  expect(getGPUCommandNodePipelineKeys(build(4)).size).toBe(
    getGPUCommandNodePipelineKeys(build(3)).size
  );
});

it('GPUMapColoring validates and compiles the priority option', () => {
  const {props} = setup();
  expect(() => new GPUMapColoring({...props, priority: 'smallest-last' as 'random'})).toThrow(
    /priority/
  );
  expect(new GPUMapColoring(props).priority).toBe('largest-degree-first');
  const kernelKeys = (priority?: 'random' | 'largest-degree-first') => {
    const built = setup();
    return getGPUCommandNodePipelineKeys(
      new GPUMapColoring({...built.props, priority}).getCommandNodes(built.graph)
    );
  };
  expect(kernelKeys().size).toBe(kernelKeys('random').size);
  expect([...kernelKeys()]).not.toEqual([...kernelKeys('random')]);
  expect([...kernelKeys('largest-degree-first')]).toEqual([...kernelKeys()]);
});

const SCENES: [string, () => ReturnType<typeof createRookLatticeCSR>][] = [
  ['lattice', () => createRookLatticeCSR(12, 9)],
  ['planar', () => createRandomPlanarCSR(30, 30, createSeededRandom(5))],
  ['planar-2', () => createRandomPlanarCSR(40, 25, createSeededRandom(11))],
  ['star', () => createStarCSR(40)],
  ['clique', () => createCliqueCSR(9)],
  ['empty', () => createOracleCSRFromEdges(0, [])],
  ['isolated', () => createOracleCSRFromEdges(7, [[1, 2]])]
];

it('the rounds model equals sequential greedy in priority order for both priorities', () => {
  for (const [name, create] of SCENES) {
    const csr = create();
    for (const priority of ['largest-degree-first', 'random'] as const) {
      for (const seed of [0, 7, 0xffffffff]) {
        const {colors, rounds} = simulateMapColoringRounds(csr, seed, priority);
        const expected = computeMapColoringOracle(csr, seed, priority);
        expect(colors, `${name} ${priority} ${seed}`).toEqual(expected);
        expect(countColoringConflicts(csr, colors)).toBe(0);
        expect(rounds).toBeLessThanOrEqual(64);
      }
    }
  }
});

it('largest-degree-first order is degree descending, then hash, then ID', () => {
  const csr = createStarCSR(5);
  const order = getMapColoringOrder(csr, 3);
  expect(order[0]).toBe(0);
  expect(order.slice().sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5]);
  // Equal degrees (isolated rows) fall back to the hash and then the ID.
  expect(getMapColoringOrder(createOracleCSRFromEdges(4, []), 0, 'largest-degree-first')).toEqual(
    getMapColoringOrder(createOracleCSRFromEdges(4, []), 0, 'random')
  );
  expect(computeMapColoringOracle(createStarCSR(40), 0)).toEqual([
    0,
    ...new Array<number>(40).fill(1)
  ]);
  expect(
    computeMapColoringOracle(createCliqueCSR(9), 0)
      .slice()
      .sort((a, b) => a - b)
  ).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
});

it('largest-degree-first needs no more colors than random on planar maps (aggregate) and few rounds', () => {
  let degreeFirstTotal = 0;
  let randomTotal = 0;
  let maximumRounds = 0;
  for (const seed of [0, 1, 2, 3, 4, 5, 6, 7]) {
    const csr = createRandomPlanarCSR(60, 60, createSeededRandom(100 + seed));
    const degreeFirst = simulateMapColoringRounds(csr, seed, 'largest-degree-first');
    const random = simulateMapColoringRounds(csr, seed, 'random');
    degreeFirstTotal += Math.max(...degreeFirst.colors) + 1;
    randomTotal += Math.max(...random.colors) + 1;
    maximumRounds = Math.max(maximumRounds, degreeFirst.rounds);
  }
  expect(degreeFirstTotal).toBeLessThanOrEqual(randomTotal);
  // Chains of strictly decreasing degree are short on planar maps: the default cap of 64 is ample.
  expect(maximumRounds).toBeLessThanOrEqual(32);
});

it('largest-degree-first stays within one color of networkx largest_first on Delaunay maps', () => {
  for (const graph of fixture.graphs) {
    const csr = {offsets: graph.offsets, neighbors: graph.neighbors, weights: [], distances: []};
    const degreeFirst = simulateMapColoringRounds(csr, 0, 'largest-degree-first');
    const random = simulateMapColoringRounds(csr, 0, 'random');
    expect(countColoringConflicts(csr, degreeFirst.colors), graph.name).toBe(0);
    const colorCount = Math.max(...degreeFirst.colors) + 1;
    // Same heuristic, different tie-breaking (seeded hash instead of networkx's stable sort).
    expect(colorCount, graph.name).toBeLessThanOrEqual(graph.networkxColorCounts.largest_first + 1);
    expect(colorCount, graph.name).toBeLessThanOrEqual(Math.max(...random.colors) + 1);
    expect(degreeFirst.rounds, graph.name).toBeLessThanOrEqual(64);
    console.log(
      `${graph.name}: ldf colors=${colorCount} rounds=${degreeFirst.rounds}; random colors=${Math.max(...random.colors) + 1} rounds=${random.rounds}; networkx`,
      graph.networkxColorCounts
    );
  }
});
