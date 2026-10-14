// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
  GPU_PLANAR_SPATIAL_CONTEXT,
  type GPUSpatialContext
} from '../../../src/analysis-core/contracts/spatial-context';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeightsOutput
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `context-${serial++}`, format, length);
  const weights = (spatialContext?: GPUSpatialContext): GPUSpatialWeightsOutput => ({
    spatialContext,
    offsets: view('uint32', 4),
    neighbors: view('uint32', 6),
    weights: view('float32', 6),
    distances: view('float32', 6),
    status: {overflow: view('uint32', 1)}
  });
  return {device, view, weights};
}

it('weights descriptions, transforms, transpose and validation preserve the input context', () => {
  const {device, view, weights} = createFixture();
  const spatialContext = GPU_GREAT_CIRCLE_SPATIAL_CONTEXT;
  const input = weights(spatialContext);
  const information = {rowCount: 3, transform: 'raw', symmetric: 'unknown'} as const;
  expect(describeGPUSpatialWeights(input, {information}).spatialContext).toBe(spatialContext);
  expect(describeGPUSpatialWeights(weights(), {information, spatialContext}).spatialContext).toBe(
    spatialContext
  );
  for (const operation of [
    'row',
    'binary',
    'double',
    'variance',
    'symmetrize',
    'kernel'
  ] as const) {
    const transform = new GPUSpatialWeightsTransform({
      weights: input,
      operation,
      output: view('float32', 6),
      bandwidth: 10
    });
    expect(transform.outputWeights.spatialContext, operation).toBe(spatialContext);
  }
  expect(
    new GPUSpatialWeightsTranspose({weights: input, output: weights()}).outputWeights.spatialContext
  ).toBe(spatialContext);
  expect(
    new GPUSpatialWeightsValidation({
      weights: input,
      output: {status: {invalidCount: view('uint32', 1)}}
    }).outputWeights.spatialContext
  ).toBe(spatialContext);
  device.destroy();
});

it('kernel bandwidth checks contexts only when both are declared', () => {
  const {device, weights} = createFixture();
  const input = weights(GPU_GREAT_CIRCLE_SPATIAL_CONTEXT);
  const create = (spatialContext?: GPUSpatialContext, carryContext = true) =>
    new GPUSpatialWeightsTransform({
      weights: carryContext ? input : {...input, spatialContext: undefined},
      operation: 'kernel',
      bandwidth: 10,
      spatialContext
    });
  expect(() => create(GPU_PLANAR_SPATIAL_CONTEXT)).toThrow(/incompatible spatial contexts/);
  expect(() => create({...GPU_GREAT_CIRCLE_SPATIAL_CONTEXT})).not.toThrow();
  expect(() => create()).not.toThrow();
  expect(() => create(GPU_PLANAR_SPATIAL_CONTEXT, false)).not.toThrow();
  expect(() => create({...GPU_GREAT_CIRCLE_SPATIAL_CONTEXT, units: 'kilometers'})).toThrow(
    /incompatible spatial contexts/
  );
  device.destroy();
});

it('weights algebra checks two declared contexts and preserves either declared input', () => {
  const {device, view, weights} = createFixture();
  const spatialContext = GPU_GREAT_CIRCLE_SPATIAL_CONTEXT;
  for (const operation of ['union', 'intersection', 'difference', 'symmetricDifference'] as const) {
    for (const [leftContext, rightContext] of [
      [spatialContext, spatialContext],
      [spatialContext, undefined],
      [undefined, spatialContext]
    ]) {
      const result = new GPUSpatialWeightsAlgebra({
        operation,
        left: weights(leftContext),
        right: weights(rightContext),
        output: weights()
      });
      expect(result.outputWeights.spatialContext, operation).toBe(spatialContext);
    }
    expect(
      () =>
        new GPUSpatialWeightsAlgebra({
          operation,
          left: weights(spatialContext),
          right: weights(GPU_PLANAR_SPATIAL_CONTEXT),
          output: weights()
        })
    ).toThrow(/incompatible spatial contexts/);
  }
  const input = weights(spatialContext);
  expect(
    new GPUSpatialWeightsAlgebra({
      operation: 'selfWeight',
      weights: input,
      selfWeight: 1,
      output: weights()
    }).outputWeights.spatialContext
  ).toBe(spatialContext);
  expect(
    new GPUSpatialWeightsAlgebra({
      operation: 'subgraph',
      weights: input,
      mask: view('uint32', 3),
      output: weights()
    }).outputWeights.spatialContext
  ).toBe(spatialContext);
  expect(
    new GPUSpatialWeightsAlgebra({
      operation: 'higherOrder',
      weights: input,
      order: 2,
      output: {...weights(), distances: undefined}
    }).outputWeights.spatialContext
  ).toBe(spatialContext);
  device.destroy();
});
