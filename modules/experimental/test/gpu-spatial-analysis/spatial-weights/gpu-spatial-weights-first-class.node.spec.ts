// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSpatialWeightsCapacityPlan,
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
  const information = {rowCount: 4, transform: 'raw' as const, symmetric: true as const};
  expect(validateGPUSpatialWeights('test', {...base, information})).toBe(4);
  expect(() =>
    validateGPUSpatialWeights('test', {...base, information: {...information, rowCount: 3}})
  ).toThrow(/rowCount/);
  expect(() =>
    validateGPUSpatialWeights('test', {
      ...base,
      information: {...information, transform: 'nope' as 'raw'}
    })
  ).toThrow(/transform/);
  expect(() =>
    validateGPUSpatialWeights('test', {
      ...base,
      information: {...information, symmetric: 'maybe' as 'unknown'}
    })
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

it('getGPUSpatialWeightsCapacityPlan grows with headroom only when needed', () => {
  const plan = (requiredCount: number, capacity: number, headroom?: number) =>
    getGPUSpatialWeightsCapacityPlan({rowCount: 10, capacity, requiredCount, headroom}).capacity;
  expect(plan(100, 128)).toBe(128);
  expect(plan(128, 128)).toBe(128);
  expect(plan(200, 128)).toBe(250);
  expect(plan(200, 128, 0)).toBe(200);
  expect(() => plan(Number.NaN, 8)).toThrow(/requiredCount/);
  // Output bytes: neighbor + weight per slot (+ distance), and rowCount + 1 offsets.
  const grown = getGPUSpatialWeightsCapacityPlan({
    rowCount: 10,
    capacity: 128,
    requiredCount: 200,
    distances: true
  });
  expect(grown).toMatchObject({capacity: 250, estimatedRequiredCount: 200, limited: false});
  expect(grown.offsetsByteLength).toBe(44);
  expect(grown.estimatedOutputBytes).toBe(250 * 12 + 44);
  expect(getGPUSpatialWeightsCapacityPlan({rowCount: 10, capacity: 8}).estimatedOutputBytes).toBe(
    8 * 8 + 44
  );
});

it('getGPUSpatialWeightsKernelValue reproduces libpysal Kernel(fixed=False) weights and K(0)', () => {
  // Oracle: test/oracles/statistics-spatial-weights.py (libpysal 4.15), adaptive bandwidth h_i = d_k * 1.0000001.
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
    information: {
      rowCount: 4,
      transform: 'raw',
      symmetric: false,
      provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: 3}
    }
  };
  expect(() => new GPUSpatialWeightsTransform({operation: 'kernel', weights: input})).toThrow(
    /needs a bandwidth/
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
          information: {...knn.information!, provenance: {source: 'x', criterion: 'distance-band'}}
        }
      })
  ).toThrow(/kNN weights/);
  expect(
    new GPUSpatialWeightsTransform({operation: 'kernel', bandwidth: 'adaptive', weights: knn})
      .outputWeights.information
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
    information: {
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
    expect(output.information, operation).toMatchObject({rowCount: 4, transform, symmetric});
    expect(output.status?.overflow).toBe(symmetricInput.status?.overflow);
    expect(output.information?.provenance).toMatchObject({
      source: 'GPUSpatialWeightsTransform',
      criterion: 'distance-band',
      radius: 3
    });
  }
  const fixed = describe({operation: 'kernel', bandwidth: 2.5});
  expect(fixed.information).toMatchObject({transform: 'kernel', symmetric: true});
  expect(fixed.information?.provenance?.radius).toBe(2.5);
  const symmetrize = describe({operation: 'symmetrize', output: view('float32', 10)});
  expect(symmetrize.information).toMatchObject({transform: 'symmetric', symmetric: true});
  expect(symmetrize.weights).not.toBe(symmetricInput.weights);
  // Unknown symmetry on the input is never promoted.
  const unknown = new GPUSpatialWeightsTransform({
    operation: 'binary',
    weights: weights(4, 10)
  }).outputWeights;
  expect(unknown.information?.symmetric).toBe('unknown');
});

it('producers expose outputWeights with information and status', () => {
  const {graph, view, weights} = setup();
  const output = (rows: number, capacity: number, distances = true, requiredCount = true) => ({
    ...weights(rows, capacity, distances),
    status: {
      overflow: view('uint32', 1),
      requiredCount: requiredCount ? view('uint32', 1) : undefined
    }
  });
  const contiguity = new GPUContiguityWeights({
    criterion: 'rook',
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 8),
      ringOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      featureOffsets: view('uint32', 3)
    },
    output: output(2, 8, false)
  });
  expect(contiguity.outputWeights.information).toMatchObject({
    rowCount: 2,
    transform: 'binary',
    symmetric: true,
    hasSelfWeights: false,
    provenance: {source: 'GPUContiguityWeights', criterion: 'rook'}
  });
  expect(contiguity.outputWeights.status?.overflow).toBe(contiguity.props.output.status.overflow);
  expect(contiguity.outputWeights.status?.requiredCount).toBe(
    contiguity.props.output.status.requiredCount
  );
  expect(contiguity.outputWeights.offsets).toBe(contiguity.props.output.offsets);

  const lattice = new GPULatticeWeights({
    width: 3,
    height: 2,
    criterion: 'queen',
    output: output(6, 20, true, false)
  });
  expect(lattice.outputWeights.information).toMatchObject({
    rowCount: 6,
    symmetric: true,
    provenance: {criterion: 'queen', order: 1}
  });
  expect(lattice.outputWeights.status?.requiredCount).toBeUndefined();

  const left: GPUSpatialWeights = {
    ...weights(4, 10),
    information: {rowCount: 4, transform: 'raw', symmetric: true}
  };
  const right: GPUSpatialWeights = {
    ...weights(4, 10),
    information: {...left.information!, symmetric: 'unknown'}
  };
  const union = new GPUSpatialWeightsAlgebra({
    operation: 'union',
    left,
    right,
    output: output(4, 16)
  });
  expect(union.outputWeights.information).toMatchObject({transform: 'raw', symmetric: 'unknown'});
  expect(union.outputWeights.status?.requiredCount).toBe(union.props.output.status.requiredCount);
  const second = new GPUSpatialWeightsAlgebra({
    operation: 'higherOrder',
    order: 2,
    weights: left,
    output: output(4, 16, false)
  });
  expect(second.outputWeights.information).toMatchObject({
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
    output: output(4, 16, false)
  });
  expect(first.outputWeights.status?.requiredCount).toBe(first.props.output.status.requiredCount);
  const self = new GPUSpatialWeightsAlgebra({
    operation: 'selfWeight',
    selfWeight: 1,
    weights: {...left, information: {...left.information!, transform: 'row'}},
    output: output(4, 16, true, false)
  });
  expect(self.outputWeights.information).toMatchObject({
    transform: 'unknown',
    hasSelfWeights: true
  });

  const transpose = new GPUSpatialWeightsTranspose({
    weights: {...left, information: {...left.information!, transform: 'row', symmetric: false}},
    output: weights(4, 10)
  });
  expect(transpose.outputWeights.information).toMatchObject({
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
    () =>
      new GPUSpatialWeightsValidation({
        weights: input,
        output: {status: {invalidCount: view('uint32', 0)}}
      })
  ).toThrow(/invalidCount/);
  expect(
    () =>
      new GPUSpatialWeightsValidation({
        weights: input,
        output: {flags: view('uint32', 3), status: {invalidCount: view('uint32', 1)}}
      })
  ).toThrow(/flags length/);
  const node = new GPUSpatialWeightsValidation({
    weights: input,
    output: {flags: view('uint32', 4), status: {invalidCount: view('uint32', 1)}}
  });
  expect(node.getCommandNodes(graph)[0].id).toBe('spatial-weights-validation-rows');
  expect(node.outputWeights.status?.invalidCount).toBe(node.props.output.status.invalidCount);
  const bits = Object.values(GPU_SPATIAL_WEIGHTS_VIOLATION);
  expect(new Set(bits).size).toBe(bits.length);
  expect(bits.every(bit => (bit & (bit - 1)) === 0)).toBe(true);
});
