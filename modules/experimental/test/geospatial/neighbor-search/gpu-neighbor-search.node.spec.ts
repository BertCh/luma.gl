// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchProps
} from '../../../src/geospatial/neighbor-search';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeNeighborSearchOracle, getNeighborWeight} from './neighbor-search-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNeighborSearchProps> = {}
): GPUNeighborSearchProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    mode: 'knn',
    k: 4,
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH),
    gridSize: [4, 4],
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 40),
      weights: view('float32', 40),
      distances: view('float32', 40)
    },
    overflow: view('uint32', 1),
    ...overrides
  };
}

it('GPUNeighborSearch validates modes, k, grid, weights and lengths', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-validation'});
  expect(() => new GPUNeighborSearch(createProps(graph))).not.toThrow();
  expect(
    () => new GPUNeighborSearch(createProps(graph, {mode: 'radius', k: undefined}))
  ).not.toThrow();
  for (const k of [0, 33, 2.5, undefined]) {
    expect(() => new GPUNeighborSearch(createProps(graph, {k})), `k ${k}`).toThrow(/k must/);
  }
  expect(() => new GPUNeighborSearch(createProps(graph, {gridSize: [0, 4]}))).toThrow(/gridSize/);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  // Offsets must have queryRows + 1 entries.
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          weights: {
            offsets: view('uint32', 10),
            neighbors: view('uint32', 4),
            weights: view('float32', 4)
          }
        })
      )
  ).toThrow(/offsets length/);
  // Cross joins size the offsets by the query rows.
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          queryPositions: view('float32x2', 3),
          weights: {
            offsets: view('uint32', 4),
            neighbors: view('uint32', 4),
            weights: view('float32', 4)
          }
        })
      )
  ).not.toThrow();
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          weights: {
            offsets: view('uint32', 11),
            neighbors: view('uint32', 4),
            weights: view('float32', 5)
          }
        })
      )
  ).toThrow(/weights length/);
  expect(() => new GPUNeighborSearch(createProps(graph, {mask: view('uint32', 9)}))).toThrow(
    /mask length/
  );
  expect(() => new GPUNeighborSearch(createProps(graph, {queryMask: view('uint32', 9)}))).toThrow(
    /queryMask length/
  );
  expect(() => new GPUNeighborSearch(createProps(graph, {parameters: view('float32', 3)}))).toThrow(
    /parameters/
  );
  const counts = view('uint32', 10);
  expect(
    () => new GPUNeighborSearch(createProps(graph, {mask: counts, neighborCounts: counts}))
  ).toThrow(/must not share/);
});

it('GPUNeighborSearch emits deterministic node IDs in both modes and every join shape', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-nodes'});
  const knn = new GPUNeighborSearch({...createProps(graph), id: 'knn'});
  const first = knn.getCommandNodes(graph).map(node => node.id);
  expect(first).toContain('knn-knn');
  expect(first).toContain('knn-emit');
  expect(first.at(-1)).toBe('knn-weights');
  const radius = new GPUNeighborSearch({
    ...createProps(graph, {
      mode: 'radius',
      queryPositions: createTransientView(graph, 'q', 'float32x2', 10)
    }),
    id: 'radius'
  });
  const ids = radius.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('radius-count');
  expect(ids).toContain('radius-sort-rows');
  expect(new Set(ids).size).toBe(ids.length);
});

it('getGPUNeighborSearchParameterValues packs the layout and rejects unknown enums', () => {
  const values = getGPUNeighborSearchParameterValues({
    bounds: [1, 2, 3, 4],
    radius: 5,
    weightKind: 'kernel',
    power: 2,
    distanceFloor: 0.25,
    kernel: 'bisquare',
    rowStandardize: true
  });
  expect(Array.from(values)).toEqual([1, 2, 3, 4, 5, 2, 2, 0.25, 3, 1, 0, 0]);
  expect(getGPUNeighborSearchParameterValues({bounds: [0, 0, 1, 1]})[4]).toBe(Infinity);
  expect(() =>
    getGPUNeighborSearchParameterValues({bounds: [0, 0, 1, 1], weightKind: 'nope' as 'binary'})
  ).toThrow(/weightKind/);
});

it('the neighbor-search oracle selects lowest IDs on ties and sorts rows by ID', () => {
  // Four points at distance 1 from the origin point 0, plus a far point.
  const positions = new Float32Array([5, 5, 6, 5, 4, 5, 5, 6, 5, 4, 9, 9]);
  const parameters = {bounds: [0, 0, 10, 10]} as const;
  const result = computeNeighborSearchOracle({mode: 'knn', k: 2, positions, parameters});
  expect(result.neighbors.slice(0, 2)).toEqual([1, 2]);
  const radius = computeNeighborSearchOracle({
    mode: 'radius',
    positions,
    parameters: {...parameters, radius: 1}
  });
  expect(radius.neighbors.slice(radius.offsets[0], radius.offsets[1])).toEqual([1, 2, 3, 4]);
  expect(radius.counts[5]).toBe(0);
  expect(
    getNeighborWeight(0.5, 1, {...parameters, weightKind: 'kernel', kernel: 'triangular'})
  ).toBe(0.5);
  expect(getNeighborWeight(0, 1, {...parameters, weightKind: 'inverseDistance'})).toBe(0);
});
