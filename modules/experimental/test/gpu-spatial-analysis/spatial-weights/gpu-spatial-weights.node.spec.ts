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
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  computeContiguityOracle,
  computeLagOracle,
  computeLatticeOracle,
  computeTransformOracle,
  flattenFeatures,
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
  const polygons = (vertexCount: number, polygonCount: number, featureCount = polygonCount) => ({
    kind: 'polygons' as const,
    positions: view('float32x2', vertexCount),
    ringOffsets: view('uint32', polygonCount + 1),
    polygonOffsets: view('uint32', polygonCount + 1),
    featureOffsets: view('uint32', featureCount + 1)
  });
  const output = (rows: number, capacity: number, distances = true) => ({
    ...weights(rows, capacity, distances),
    status: {overflow: view('uint32', 1)}
  });
  return {graph, view, weights, polygons, output};
}

it('GPUContiguityWeights validates props and declares nodes', () => {
  const {graph, polygons, output} = setup();
  const base = {
    criterion: 'queen' as const,
    polygons: polygons(8, 2),
    output: output(2, 8, false)
  };
  expect(() => new GPUContiguityWeights({...base, criterion: 'bishop' as 'queen'})).toThrow(
    /criterion/
  );
  expect(() => new GPUContiguityWeights({...base, output: output(3, 8)})).toThrow(/feature count/);
  expect(
    () => new GPUContiguityWeights({...base, output: {...output(2, 8), status: {}} as never})
  ).toThrow(/overflow/);
  expect(() => new GPUContiguityWeights({...base, snapTolerance: -1})).toThrow(/snapTolerance/);
  expect(() => new GPUContiguityWeights({...base, pairCapacity: 0})).toThrow(/pairCapacity/);
  const queen = new GPUContiguityWeights(base).getCommandNodes(graph).map(node => node.id);
  expect(queen[0]).toBe('contiguity-weights-vertex-topology');
  expect(queen.at(-1)).toBe('contiguity-weights-emit');
  const {graph: rookGraph, polygons: rookPolygons, output: rookOutput} = setup();
  const rook = new GPUContiguityWeights({
    ...base,
    criterion: 'rook',
    polygons: rookPolygons(8, 2),
    output: rookOutput(2, 8, false)
  })
    .getCommandNodes(rookGraph)
    .map(node => node.id);
  expect(rook).toContain('contiguity-weights-edge-keys');
  expect(rook.length).toBeGreaterThan(queen.length);
});

it('GPUContiguityWeights rows are features: multipart layout and status wiring', () => {
  const square = (x: number, y: number): [number, number][][] => [
    [
      [x, y],
      [x + 1, y],
      [x + 1, y + 1],
      [x, y + 1]
    ]
  ];
  // Feature 0 = two parts on either side of feature 1; feature 2 sits above part 0.
  const features: OraclePolygons[] = [[square(0, 0), square(2, 0)], [square(1, 0)], [square(0, 1)]];
  const layout = flattenFeatures(features);
  expect(layout.featureOffsets).toEqual(new Uint32Array([0, 2, 3, 4]));
  expect(layout.polygonOffsets).toEqual(new Uint32Array([0, 1, 2, 3, 4]));
  const featureOf = [0, 0, 1, 2];
  // Both parts of feature 0 touch feature 1: one neighbor slot, never a self slot.
  expect(computeContiguityOracle(features.flat(), 'rook', 0, featureOf)).toMatchObject({
    offsets: [0, 2, 3, 4],
    neighbors: [1, 2, 0, 0]
  });
  expect(computeContiguityOracle(features.flat(), 'queen', 0, featureOf)).toMatchObject({
    offsets: [0, 2, 4, 6],
    neighbors: [1, 2, 0, 2, 0, 1]
  });

  const {graph, view, polygons, weights} = setup();
  const statusOutput = (candidateOverflow: boolean) => ({
    ...weights(3, 16, false),
    status: {
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      candidateOverflow: candidateOverflow ? view('uint32', 1) : undefined
    }
  });
  for (const criterion of ['queen', 'rook'] as const) {
    for (const candidateOverflow of [false, true]) {
      const contiguity = new GPUContiguityWeights({
        id: `multipart-${criterion}-${candidateOverflow}`,
        criterion,
        polygons: polygons(16, 4, 3),
        output: statusOutput(candidateOverflow)
      });
      expect(contiguity.getFeatureCount()).toBe(3);
      expect(contiguity.outputWeights.information?.rowCount).toBe(3);
      expect(contiguity.outputWeights.status?.candidateOverflow).toBe(
        contiguity.props.output.status.candidateOverflow
      );
      const sources = getGPUCommandNodeKernelSources(contiguity.getCommandNodes(graph));
      const source = (suffix: string) => sources.find(({id}) => id.endsWith(suffix))?.source ?? '';
      expect(source('-vertex-topology')).toContain('featureOffsets');
      expect(source('-totals').includes('candidateOverflow['), `${criterion}`).toBe(
        candidateOverflow
      );
      expect(source('-totals')).toContain('requiredCount[');
    }
  }
  expect(
    () =>
      new GPUContiguityWeights({
        criterion: 'queen',
        polygons: polygons(16, 4, 3),
        output: {...weights(4, 16, false), status: {overflow: view('uint32', 1)}}
      })
  ).toThrow(/feature count/);
});

it('GPULatticeWeights validates props', () => {
  const {graph, view, output} = setup();
  const base = {
    width: 3,
    height: 2,
    criterion: 'rook' as const,
    output: output(6, 20)
  };
  expect(() => new GPULatticeWeights({...base, width: 0})).toThrow(/width/);
  expect(() => new GPULatticeWeights({...base, radius: 0})).toThrow(/radius/);
  expect(() => new GPULatticeWeights({...base, radius: 99})).toThrow(/radius/);
  expect(() => new GPULatticeWeights({...base, output: output(5, 20)})).toThrow(/output rows/);
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
  // The value row count is derived (values.length / columnCount): cross weights are allowed.
  expect(
    () =>
      new GPUSpatialLag({
        values: view('float32', 3),
        columnCount: 2,
        weights: input,
        output: view('float32', 8)
      })
  ).toThrow(/multiple of columnCount/);
  expect(
    () =>
      new GPUSpatialLag({
        values: view('float32', 3),
        weights: input,
        output: view('float32', 4),
        mask: view('uint32', 4)
      })
  ).toThrow(/square/);
  expect(
    () =>
      new GPUSpatialLag({values: view('float32', 4), weights: input, output: view('float32', 3)})
  ).toThrow(/output length/);
  const cross = new GPUSpatialLag({
    values: view('float32', 3),
    weights: input,
    output: view('float32', 4)
  });
  expect(cross.getValueRowCount()).toBe(3);
  // uint32 values (counts) add exactly one conversion node before the lag.
  const counts = new GPUSpatialLag({
    values: view('uint32', 4),
    weights: input,
    output: view('float32', 4)
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(counts).toEqual([
    'spatial-lag-values-to-float32',
    'spatial-lag-lag',
    'spatial-lag-lag-fixup'
  ]);
  expect(
    new GPUSpatialLag({
      id: 'float-lag',
      values: view('float32', 4),
      weights: input,
      output: view('float32', 4)
    })
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['float-lag-lag', 'float-lag-lag-fixup']);
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
