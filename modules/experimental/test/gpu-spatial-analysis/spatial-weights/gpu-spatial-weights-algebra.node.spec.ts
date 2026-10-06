// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeBinaryOracle,
  computeBlockOracle,
  computeHigherOrderOracle,
  computeSelfWeightOracle,
  computeSubgraphOracle,
  computeSummaryOracle,
  createRandomWeights
} from './spatial-weights-algebra-oracle';
import {computeLatticeOracle, computeTransformOracle} from './spatial-weights-oracle';

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'spatial-weights-algebra-nodes'
  });
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `algebra-view-${serial++}`, format, length);
  const weights = (rows: number, capacity: number, distances = true): GPUSpatialWeights => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined
  });
  return {graph, view, weights};
}

it('GPUSpatialWeightsAlgebra validates props and declares nodes', () => {
  const {graph, view, weights} = setup();
  const left = weights(4, 12);
  const right = weights(4, 12);
  const output = () => ({output: weights(4, 20), overflow: view('uint32', 1)});
  expect(
    () => new GPUSpatialWeightsAlgebra({operation: 'nope' as 'union', ...output()} as never)
  ).toThrow(/operation/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({operation: 'union', left, right: weights(5, 12), ...output()})
  ).toThrow(/right.offsets length/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'union',
        left,
        right,
        ...output(),
        overflow: view('uint32', 0)
      })
  ).toThrow(/overflow/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'union',
        left,
        right,
        weightRule: 'median' as 'sum',
        ...output()
      })
  ).toThrow(/weightRule/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'union',
        left,
        right,
        output: left,
        overflow: view('uint32', 1)
      })
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'union',
        left: weights(4, 12, false),
        right,
        output: weights(4, 20),
        overflow: view('uint32', 1)
      })
  ).toThrow(/distances/);
  for (const order of [0, 1.5, 33]) {
    expect(
      () =>
        new GPUSpatialWeightsAlgebra({operation: 'higherOrder', weights: left, order, ...output()})
    ).toThrow(/order/);
  }
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'higherOrder',
        weights: left,
        order: 2,
        output: weights(4, 20),
        overflow: view('uint32', 1)
      })
  ).toThrow(/no distances/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'selfWeight',
        weights: left,
        selfWeight: -1,
        ...output()
      })
  ).toThrow(/selfWeight/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'selfWeight',
        weights: left,
        selfWeight: view('float32', 3),
        ...output()
      })
  ).toThrow(/selfWeight view length/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'subgraph',
        weights: left,
        mask: view('uint32', 3),
        ...output()
      })
  ).toThrow(/mask length/);
  expect(
    () =>
      new GPUSpatialWeightsAlgebra({
        operation: 'block',
        groupIds: view('uint32', 4),
        groupCount: 0,
        output: {...weights(4, 20), distances: undefined},
        overflow: view('uint32', 1)
      })
  ).toThrow(/groupCount/);

  const ids = (producer: GPUSpatialWeightsAlgebra) =>
    producer.getCommandNodes(graph).map(node => node.id);
  const binaryIds = ids(
    new GPUSpatialWeightsAlgebra({operation: 'union', left, right, ...output()})
  );
  expect(binaryIds[0]).toBe('spatial-weights-algebra-counts');
  expect(binaryIds.at(-1)).toBe('spatial-weights-algebra-emit-distances');
  const higher = ids(
    new GPUSpatialWeightsAlgebra({
      operation: 'higherOrder',
      weights: left,
      order: 3,
      cumulative: true,
      output: {...weights(4, 20), distances: undefined},
      overflow: view('uint32', 1)
    })
  );
  expect(higher).toContain('spatial-weights-algebra-visited-3-offsets');
  expect(higher.at(-1)).toBe('spatial-weights-algebra-weights');
});

it('GPUSpatialWeightsSummary and the D/V transforms validate props', () => {
  const {graph, view, weights} = setup();
  const input = weights(4, 10);
  expect(
    () =>
      new GPUSpatialWeightsSummary({
        weights: input,
        statistics: view('float32', 2),
        counts: view('uint32', 5)
      })
  ).toThrow(/statistics/);
  expect(
    () =>
      new GPUSpatialWeightsSummary({
        weights: input,
        statistics: view('float32', 3),
        counts: view('uint32', 4)
      })
  ).toThrow(/counts/);
  expect(
    () =>
      new GPUSpatialWeightsSummary({
        weights: input,
        statistics: view('float32', 3),
        counts: view('uint32', 5),
        cardinality: view('uint32', 3)
      })
  ).toThrow(/cardinality/);
  expect(
    new GPUSpatialWeightsSummary({
      weights: input,
      statistics: view('float32', 3),
      counts: view('uint32', 5)
    })
      .getCommandNodes(graph)
      .map(node => node.id)
      .at(-1)
  ).toBe('spatial-weights-summary-counts');
  expect(
    () =>
      new GPUSpatialWeightsTransform({
        operation: 'double',
        doubleSum: 'all' as 'one',
        weights: input
      })
  ).toThrow(/doubleSum/);
  for (const operation of ['double', 'variance'] as const) {
    const ids = new GPUSpatialWeightsTransform({id: operation, operation, weights: input})
      .getCommandNodes(graph)
      .map(node => node.id);
    expect(ids[0]).toBe(`${operation}-partials`);
    expect(ids.at(-1)).toBe(`${operation}-apply`);
  }
});

it('CPU algebra oracles agree with libpysal reference values', () => {
  // libpysal lat2W(3, 3) (rook): s0 = 24, s1 = 48, s2 = 272, cardinalities 2,3,2,3,4,3,2,3,2.
  const lattice = computeLatticeOracle({width: 3, height: 3, criterion: 'rook'});
  const summary = computeSummaryOracle(lattice);
  expect([summary.s0, summary.s1, summary.s2]).toEqual([24, 48, 272]);
  expect(summary.cardinality).toEqual([2, 3, 2, 3, 4, 3, 2, 3, 2]);
  expect(summary.asymmetricSlots).toBe(0);
  expect(summary.isolates).toBe(0);

  // Higher order on the 3x3 rook lattice: row 0 has 1,3 at order 1; 2,4,6 at 2; 5,7 at 3; 8 at 4.
  const rowOf = (order: number, cumulative = false, row = 0) => {
    const csr = computeHigherOrderOracle(lattice, order, cumulative);
    return csr.neighbors.slice(csr.offsets[row], csr.offsets[row + 1]);
  };
  expect(rowOf(1)).toEqual([1, 3]);
  expect(rowOf(2)).toEqual([2, 4, 6]);
  expect(rowOf(3)).toEqual([5, 7]);
  expect(rowOf(4)).toEqual([8]);
  expect(rowOf(2, true)).toEqual([1, 2, 3, 4, 6]);
  expect(rowOf(2, false, 4)).toEqual([0, 2, 6, 8]); // the center reaches the corners in 2 steps

  // Row standardization sums to n = 9; V sums to n; D sums to 1.
  const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);
  expect(sum(computeTransformOracle(lattice, 'row'))).toBeCloseTo(9, 10);
  expect(sum(computeTransformOracle(lattice, 'variance'))).toBeCloseTo(9, 10);
  expect(sum(computeTransformOracle(lattice, 'double'))).toBeCloseTo(1, 12);
  expect(sum(computeTransformOracle(lattice, 'double', {doubleSum: 'rows'}))).toBeCloseTo(9, 10);

  // Set algebra identities.
  const left = createRandomWeights(30, 0.15, 1);
  const right = createRandomWeights(30, 0.15, 2);
  const union = computeBinaryOracle('union', left, right);
  const intersection = computeBinaryOracle('intersection', left, right);
  expect(union.neighbors.length + intersection.neighbors.length).toBe(
    left.neighbors.length + right.neighbors.length
  );
  expect(computeBinaryOracle('union', left, left).neighbors).toEqual(left.neighbors);
  expect(computeBinaryOracle('difference', left, left).neighbors).toEqual([]);
  expect(computeBinaryOracle('intersection', left, right, 'sum').weights.length).toBe(
    intersection.neighbors.length
  );

  // Self weight, subgraph, block.
  expect(computeSelfWeightOracle(lattice, 1).neighbors.length).toBe(24 + 9);
  const mask = [1, 1, 0, 1, 1, 0, 0, 0, 0];
  expect(computeSubgraphOracle(lattice, mask).neighbors).toEqual([1, 3, 0, 4, 0, 4, 1, 3]);
  expect(computeBlockOracle([0, 0, 1, 1, 1, 2], 2).neighbors).toEqual([1, 0, 3, 4, 2, 4, 2, 3]);
});

/** CSR from `[focal, neighbor, weight]` triples sorted by focal then neighbor. */
function fromTriples(rows: number, triples: [number, number, number][]) {
  const csr = {offsets: [0], neighbors: [] as number[], weights: [] as number[], distances: []};
  for (let row = 0; row < rows; row++) {
    for (const [focal, neighbor, weight] of triples) {
      if (focal === row) {
        csr.neighbors.push(neighbor);
        csr.weights.push(weight);
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

const pairs = (csr: {offsets: number[]; neighbors: number[]}) =>
  csr.offsets
    .slice(0, -1)
    .flatMap((begin, row) =>
      csr.neighbors.slice(begin, csr.offsets[row + 1]).map(neighbor => [row, neighbor])
    );

it('CPU oracles match libpysal 4.15 Graph reference values (generated with real libpysal)', () => {
  // A and B: two 6-node directed graphs with distinct weights, so a weight rule is observable.
  const A = fromTriples(6, [
    [0, 1, 1],
    [0, 2, 2],
    [1, 2, 3],
    [2, 0, 4],
    [3, 4, 5],
    [4, 3, 1.5],
    [4, 5, 2.5]
  ]);
  const B = fromTriples(6, [
    [0, 1, 10],
    [0, 3, 20],
    [1, 2, 30],
    [2, 1, 40],
    [3, 4, 50],
    [4, 5, 7],
    [5, 4, 8]
  ]);
  // libpysal Graph.union/intersection/difference/symmetric_difference, all weights 1.0.
  const expected = {
    union: [
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [2, 0],
      [2, 1],
      [3, 4],
      [4, 3],
      [4, 5],
      [5, 4]
    ],
    intersection: [
      [0, 1],
      [1, 2],
      [3, 4],
      [4, 5]
    ],
    difference: [
      [0, 2],
      [2, 0],
      [4, 3]
    ],
    symmetricDifference: [
      [0, 2],
      [0, 3],
      [2, 0],
      [2, 1],
      [4, 3],
      [5, 4]
    ]
  } as const;
  for (const operation of Object.keys(expected) as (keyof typeof expected)[]) {
    const result = computeBinaryOracle(operation, A, B, 'binary', false);
    expect(pairs(result), operation).toEqual(expected[operation].map(pair => [...pair]));
    expect(result.weights.every(weight => weight === 1)).toBe(true);
  }

  // Transform D and V of directed A (libpysal Graph.transform('D'|'V'), n = 6 with isolate 5).
  const sixDecimals = (values: number[]) => values.map(value => Math.round(value * 1e6) / 1e6);
  expect(sixDecimals(computeTransformOracle(A, 'double'))).toEqual(
    [1, 2, 3, 4, 5, 1.5, 2.5].map(weight => Math.round((weight / 19) * 1e6) / 1e6)
  );
  expect(sixDecimals(computeTransformOracle(A, 'variance'))).toEqual([
    0.469628, 0.939256, 1.050121, 1.050121, 1.050121, 0.540283, 0.900471
  ]);

  // 3x3 rook lattice: D = 1/24 everywhere; V is 0.436334 (corner), 0.356265 (edge), 0.308535 (center) rows.
  const lattice = computeLatticeOracle({width: 3, height: 3, criterion: 'rook'});
  const double = sixDecimals(computeTransformOracle(lattice, 'double'));
  expect(new Set(double)).toEqual(new Set([0.041667]));
  const variance = sixDecimals(computeTransformOracle(lattice, 'variance'));
  const rowValues = (row: number) => variance.slice(lattice.offsets[row], lattice.offsets[row + 1]);
  expect(new Set(rowValues(0))).toEqual(new Set([0.436334]));
  expect(new Set(rowValues(1))).toEqual(new Set([0.356265]));
  expect(new Set(rowValues(4))).toEqual(new Set([0.308535]));
});
