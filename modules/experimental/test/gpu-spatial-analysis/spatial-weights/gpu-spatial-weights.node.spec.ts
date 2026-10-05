// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUContiguityWeights,
  GPULatticeWeights,
  GPUSpatialLag,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeContiguityOracle,
  computeLagOracle,
  computeLatticeOracle,
  computeTransformOracle,
  flattenPolygons,
  type OraclePolygons
} from './spatial-weights-oracle';

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'spatial-weights-nodes'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const weights = (rows: number, capacity: number, distances = true): GPUSpatialWeights => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined
  });
  return {graph, view, weights};
}

it('GPUContiguityWeights validates props and declares nodes', () => {
  const {graph, view, weights} = setup();
  const base = {
    criterion: 'queen' as const,
    positions: view('float32x2', 8),
    ringOffsets: view('uint32', 3),
    polygonOffsets: view('uint32', 3),
    weights: weights(2, 8, false),
    overflow: view('uint32', 1)
  };
  expect(() => new GPUContiguityWeights({...base, criterion: 'bishop' as 'queen'})).toThrow(
    /criterion/
  );
  expect(() => new GPUContiguityWeights({...base, weights: weights(3, 8)})).toThrow(
    /polygonOffsets length/
  );
  expect(() => new GPUContiguityWeights({...base, snapTolerance: -1})).toThrow(/snapTolerance/);
  expect(() => new GPUContiguityWeights({...base, pairCapacity: 0})).toThrow(/pairCapacity/);
  const queen = new GPUContiguityWeights(base).getCommandNodes(graph).map(node => node.id);
  expect(queen[0]).toBe('contiguity-weights-vertex-topology');
  expect(queen.at(-1)).toBe('contiguity-weights-emit');
  const {graph: rookGraph, view: rookView, weights: rookWeights} = setup();
  const rook = new GPUContiguityWeights({
    ...base,
    criterion: 'rook',
    positions: rookView('float32x2', 8),
    ringOffsets: rookView('uint32', 3),
    polygonOffsets: rookView('uint32', 3),
    weights: rookWeights(2, 8, false),
    overflow: rookView('uint32', 1)
  })
    .getCommandNodes(rookGraph)
    .map(node => node.id);
  expect(rook).toContain('contiguity-weights-edge-keys');
  expect(rook.length).toBeGreaterThan(queen.length);
});

it('GPULatticeWeights validates props', () => {
  const {graph, view, weights} = setup();
  const base = {
    width: 3,
    height: 2,
    criterion: 'rook' as const,
    weights: weights(6, 20),
    overflow: view('uint32', 1)
  };
  expect(() => new GPULatticeWeights({...base, width: 0})).toThrow(/width/);
  expect(() => new GPULatticeWeights({...base, radius: 0})).toThrow(/radius/);
  expect(() => new GPULatticeWeights({...base, radius: 99})).toThrow(/radius/);
  expect(() => new GPULatticeWeights({...base, weights: weights(5, 20)})).toThrow(/offsets length/);
  expect(() => new GPULatticeWeights({...base, mask: view('uint32', 5)})).toThrow(/mask length/);
  expect(new GPULatticeWeights(base).getCommandNodes(graph).at(-1)?.id).toBe(
    'lattice-weights-emit'
  );
});

it('GPUSpatialWeightsTransform and GPUSpatialLag validate props', () => {
  const {graph, view, weights} = setup();
  const input = weights(4, 10);
  expect(
    () => new GPUSpatialWeightsTransform({operation: 'nope' as 'row', weights: input})
  ).toThrow(/operation/);
  expect(
    () => new GPUSpatialWeightsTransform({operation: 'kernel', weights: weights(4, 10, false)})
  ).toThrow(/distances/);
  expect(
    () => new GPUSpatialWeightsTransform({operation: 'kernel', bandwidth: -1, weights: input})
  ).toThrow(/bandwidth/);
  expect(() => new GPUSpatialWeightsTransform({operation: 'symmetrize', weights: input})).toThrow(
    /separate/
  );
  expect(
    new GPUSpatialWeightsTransform({
      operation: 'symmetrize',
      weights: input,
      output: view('float32', 10)
    })
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['spatial-weights-transform-symmetrize']);
  expect(
    () =>
      new GPUSpatialLag({values: view('float32', 3), weights: input, output: view('float32', 4)})
  ).toThrow(/values length/);
  expect(
    () =>
      new GPUSpatialLag({
        values: view('float32', 4),
        weights: input,
        output: view('float32', 4),
        mask: view('uint32', 3)
      })
  ).toThrow(/mask length/);
});

it('spatial-weights oracles satisfy the CSR contract', () => {
  const polygons: OraclePolygons = [
    [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ]
    ],
    [
      [
        [1, 0],
        [2, 0],
        [2, 1],
        [1, 1]
      ]
    ]
  ];
  expect(flattenPolygons(polygons).polygonOffsets).toEqual(new Uint32Array([0, 1, 2]));
  const csr = computeContiguityOracle(polygons, 'rook');
  expect(csr.neighbors).toEqual([1, 0]);
  const lattice = computeLatticeOracle({width: 3, height: 3, criterion: 'rook'});
  expect(lattice.offsets[9]).toBe(24);
  const row = computeTransformOracle(lattice, 'row');
  expect(row.slice(lattice.offsets[4], lattice.offsets[5])).toEqual([0.25, 0.25, 0.25, 0.25]);
  expect(computeLagOracle(lattice, [1, 1, 1, 1, 1, 1, 1, 1, 1])[4]).toBe(4);
});
