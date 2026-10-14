// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSpatialWeightsCapacity,
  getGPUSpatialWeightsKernelValue,
  GPU_SPATIAL_WEIGHTS_VIOLATION,
  GPUContiguityWeights,
  GPULatticeWeights,
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsTransform,
  GPUSpatialWeightsTranspose,
  GPUSpatialWeightsValidation,
  validateGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {WEIGHTS_ORACLE} from './spatial-weights-u4-oracle';

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights-first-class'});
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

const KERNELS: GPUSpatialWeightsKernel[] = [
  'triangular',
  'epanechnikov',
  'bisquare',
  'uniform',
  'gaussian'
];

it('validateGPUSpatialWeights checks status formats and declared metadata', () => {
  const {view, weights} = setup();
  const base = weights(4, 10);
  const info = {rowCount: 4, transform: 'raw' as const, symmetric: true as const};
  expect(validateGPUSpatialWeights('test', {...base, info})).toBe(4);
  expect(() => validateGPUSpatialWeights('test', {...base, info: {...info, rowCount: 3}})).toThrow(
    /rowCount/
  );
  expect(() =>
    validateGPUSpatialWeights('test', {...base, info: {...info, transform: 'nope' as 'raw'}})
  ).toThrow(/transform/);
  expect(() =>
    validateGPUSpatialWeights('test', {...base, info: {...info, symmetric: 'maybe' as 'unknown'}})
  ).toThrow(/symmetric/);
  expect(() =>
    validateGPUSpatialWeights('test', {
      ...base,
      status: {overflow: view('float32', 1) as unknown as ReturnType<typeof view<'uint32'>>}
    })
  ).toThrow(/overflow/);
  expect(() => validateGPUSpatialWeights('test', {...base, distances: view('float32', 9)})).toThrow(
    /distances length/
  );
});

it('getGPUSpatialWeightsCapacity grows with headroom only when needed', () => {
  expect(getGPUSpatialWeightsCapacity(100, 128)).toBe(128);
  expect(getGPUSpatialWeightsCapacity(128, 128)).toBe(128);
  expect(getGPUSpatialWeightsCapacity(200, 128)).toBe(250);
  expect(getGPUSpatialWeightsCapacity(200, 128, 0)).toBe(200);
  expect(() => getGPUSpatialWeightsCapacity(Number.NaN, 8)).toThrow(/requiredCount/);
});

it('getGPUSpatialWeightsKernelValue reproduces libpysal Kernel(fixed=False) weights and K(0)', () => {
  // Oracle: fix/work/A/oracle-u4-weights.py (libpysal 4.15), adaptive bandwidth h_i = d_k * 1.0000001.
  for (const kernel of KERNELS) {
    const oracle = WEIGHTS_ORACLE.kernels[kernel];
    expect(getGPUSpatialWeightsKernelValue(kernel, 0), `${kernel} K(0)`).toBeCloseTo(
      oracle.selfWeight,
      12
    );
    oracle.rows.forEach((row, index) => {
      for (const [, distance, weight] of row) {
        const actual = getGPUSpatialWeightsKernelValue(kernel, distance / oracle.bandwidth[index]);
        expect(Math.abs(actual - weight), `${kernel} row ${index}`).toBeLessThan(1e-12);
      }
    });
  }
  expect(getGPUSpatialWeightsKernelValue('triangular', -1)).toBe(0);
  expect(getGPUSpatialWeightsKernelValue('gaussian', Number.NaN)).toBe(0);
  expect(getGPUSpatialWeightsKernelValue('uniform', 1.5)).toBe(0);
});

it("GPUSpatialWeightsTransform 'kernel' requires a bandwidth and kNN weights for 'adaptive'", () => {
  const {weights} = setup();
  const input = weights(4, 10);
  const knn: GPUSpatialWeights = {
    ...input,
    info: {
      rowCount: 4,
      transform: 'raw',
      symmetric: false,
      provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: 3}
    }
  };
  expect(() => new GPUSpatialWeightsTransform({operation: 'kernel', weights: input})).toThrow(
    /requires a bandwidth/
  );
  expect(
    () =>
      new GPUSpatialWeightsTransform({operation: 'kernel', bandwidth: 'adaptive', weights: input})
  ).toThrow(/kNN weights/);
  expect(
    () =>
      new GPUSpatialWeightsTransform({
        operation: 'kernel',
        bandwidth: 'adaptive',
        weights: {
          ...input,
          info: {...knn.info!, provenance: {source: 'x', criterion: 'distance-band'}}
        }
      })
  ).toThrow(/numeric bandwidth/);
  expect(
    new GPUSpatialWeightsTransform({operation: 'kernel', bandwidth: 'adaptive', weights: knn})
      .outputWeights.info
  ).toMatchObject({transform: 'kernel', symmetric: 'unknown'});
  expect(
    () =>
      new GPUSpatialWeightsTransform({
        operation: 'kernel',
        bandwidth: 2,
        weights: input,
        output: setup().weights(4, 9).weights
      })
  ).toThrow(/output length/);
});

it('GPUSpatialWeightsTransform.outputWeights describes the transform honestly', () => {
  const {weights, view} = setup();
  const symmetricInput: GPUSpatialWeights = {
    ...weights(4, 10),
    info: {
      rowCount: 4,
      transform: 'raw',
      symmetric: true,
      provenance: {source: 'GPUNeighborSearch', criterion: 'distance-band', radius: 3}
    },
    status: {overflow: view('uint32', 1)}
  };
  const describe = (props: Partial<ConstructorParameters<typeof GPUSpatialWeightsTransform>[0]>) =>
    new GPUSpatialWeightsTransform({
      operation: 'binary',
      weights: symmetricInput,
      ...props
    } as ConstructorParameters<typeof GPUSpatialWeightsTransform>[0]).outputWeights;

  for (const [operation, transform, symmetric] of [
    ['binary', 'binary', true],
    ['double', 'double', true],
    ['row', 'row', 'unknown'],
    ['variance', 'variance', 'unknown']
  ] as const) {
    const output = describe({operation});
    expect(output.info, operation).toMatchObject({rowCount: 4, transform, symmetric});
    expect(output.status?.overflow).toBe(symmetricInput.status?.overflow);
    expect(output.info?.provenance).toMatchObject({
      source: 'GPUSpatialWeightsTransform',
      criterion: 'distance-band',
      radius: 3
    });
  }
  const fixed = describe({operation: 'kernel', bandwidth: 2.5});
  expect(fixed.info).toMatchObject({transform: 'kernel', symmetric: true});
  expect(fixed.info?.provenance?.radius).toBe(2.5);
  const symmetrize = describe({operation: 'symmetrize', output: view('float32', 10)});
  expect(symmetrize.info).toMatchObject({transform: 'symmetric', symmetric: true});
  expect(symmetrize.weights).not.toBe(symmetricInput.weights);
  // Unknown symmetry on the input is never promoted.
  const unknown = new GPUSpatialWeightsTransform({
    operation: 'binary',
    weights: weights(4, 10)
  }).outputWeights;
  expect(unknown.info?.symmetric).toBe('unknown');
});

it('producers expose outputWeights with info and status', () => {
  const {graph, view, weights} = setup();
  const contiguity = new GPUContiguityWeights({
    criterion: 'rook',
    positions: view('float32x2', 8),
    ringOffsets: view('uint32', 3),
    polygonOffsets: view('uint32', 3),
    weights: weights(2, 8, false),
    overflow: view('uint32', 1),
    totalNeighbors: view('uint32', 1)
  });
  expect(contiguity.outputWeights.info).toMatchObject({
    rowCount: 2,
    transform: 'binary',
    symmetric: true,
    hasSelfWeights: false,
    provenance: {source: 'GPUContiguityWeights', criterion: 'rook'}
  });
  expect(contiguity.outputWeights.status?.overflow).toBe(contiguity.props.overflow);
  expect(contiguity.outputWeights.status?.requiredCount).toBe(contiguity.props.totalNeighbors);
  expect(contiguity.outputWeights.offsets).toBe(contiguity.props.weights.offsets);

  const lattice = new GPULatticeWeights({
    width: 3,
    height: 2,
    criterion: 'queen',
    weights: weights(6, 20),
    overflow: view('uint32', 1)
  });
  expect(lattice.outputWeights.info).toMatchObject({
    rowCount: 6,
    symmetric: true,
    provenance: {criterion: 'queen', order: 1}
  });
  expect(lattice.outputWeights.status?.requiredCount).toBeUndefined();

  const left: GPUSpatialWeights = {
    ...weights(4, 10),
    info: {rowCount: 4, transform: 'raw', symmetric: true}
  };
  const right: GPUSpatialWeights = {...weights(4, 10), info: {...left.info!, symmetric: 'unknown'}};
  const union = new GPUSpatialWeightsAlgebra({
    operation: 'union',
    left,
    right,
    output: weights(4, 16),
    overflow: view('uint32', 1),
    totalNeighbors: view('uint32', 1)
  });
  expect(union.outputWeights.info).toMatchObject({transform: 'raw', symmetric: 'unknown'});
  expect(union.outputWeights.status?.requiredCount).toBe(union.props.totalNeighbors);
  const second = new GPUSpatialWeightsAlgebra({
    operation: 'higherOrder',
    order: 2,
    weights: left,
    output: weights(4, 16, false),
    overflow: view('uint32', 1),
    totalNeighbors: view('uint32', 1)
  });
  expect(second.outputWeights.info).toMatchObject({
    transform: 'binary',
    symmetric: true,
    provenance: {order: 2}
  });
  // An intermediate-set overflow would make this a lower bound, so it is not published as exact.
  expect(second.outputWeights.status?.requiredCount).toBeUndefined();
  const first = new GPUSpatialWeightsAlgebra({
    operation: 'higherOrder',
    order: 1,
    weights: left,
    output: weights(4, 16, false),
    overflow: view('uint32', 1),
    totalNeighbors: view('uint32', 1)
  });
  expect(first.outputWeights.status?.requiredCount).toBe(first.props.totalNeighbors);
  const self = new GPUSpatialWeightsAlgebra({
    operation: 'selfWeight',
    selfWeight: 1,
    weights: {...left, info: {...left.info!, transform: 'row'}},
    output: weights(4, 16),
    overflow: view('uint32', 1)
  });
  expect(self.outputWeights.info).toMatchObject({transform: 'unknown', hasSelfWeights: true});

  const transpose = new GPUSpatialWeightsTranspose({
    weights: {...left, info: {...left.info!, transform: 'row', symmetric: false}},
    output: weights(4, 10)
  });
  expect(transpose.outputWeights.info).toMatchObject({
    rowCount: 4,
    transform: 'unknown',
    symmetric: false
  });
  expect(graph).toBeDefined();
});

it('GPUSpatialWeightsValidation validates props and declares nodes', () => {
  const {graph, view, weights} = setup();
  const input = weights(4, 10);
  expect(
    () => new GPUSpatialWeightsValidation({weights: input, invalidCount: view('uint32', 0)})
  ).toThrow(/invalidCount/);
  expect(
    () =>
      new GPUSpatialWeightsValidation({
        weights: input,
        invalidCount: view('uint32', 1),
        flags: view('uint32', 3)
      })
  ).toThrow(/flags length/);
  const node = new GPUSpatialWeightsValidation({
    weights: input,
    invalidCount: view('uint32', 1),
    flags: view('uint32', 4)
  });
  expect(node.getCommandNodes(graph)[0].id).toBe('spatial-weights-validation-rows');
  expect(node.outputWeights.status?.invalidCount).toBe(node.props.invalidCount);
  const bits = Object.values(GPU_SPATIAL_WEIGHTS_VIOLATION);
  expect(new Set(bits).size).toBe(bits.length);
  expect(bits.every(bit => (bit & (bit - 1)) === 0)).toBe(true);
});
