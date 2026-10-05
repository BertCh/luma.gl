// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  convolveLineIntegralOnCPU,
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionWordParameterValues,
  GPULineIntegralConvolution,
  type GPULineIntegralConvolutionProps
} from '../../../src/gpu-raster/flow-texture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createUniformField} from './flow-texture-scenes';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULineIntegralConvolutionProps> = {}
): GPULineIntegralConvolutionProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    velocities: view('float32x2', 16),
    fieldWidth: 4,
    fieldHeight: 4,
    width: 8,
    height: 6,
    stepCount: 5,
    parameters: view('float32', 12),
    wordParameters: view('uint32', 4),
    output: {values: view('float32', 48), speeds: view('float32', 48)},
    ...overrides
  };
}

it('GPULineIntegralConvolution validates props and builds deterministic nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'lic-validation'
  });
  const contributor = new GPULineIntegralConvolution(createProps(graph, {id: 'flow'}));
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'flow-noise',
    'flow-convolve'
  ]);
  expect(() => new GPULineIntegralConvolution(createProps(graph, {stepCount: 300}))).toThrow(
    /stepCount/
  );
  expect(() => new GPULineIntegralConvolution(createProps(graph, {width: 7}))).toThrow(
    /output.values/
  );
  expect(() => new GPULineIntegralConvolution(createProps(graph, {fieldWidth: 3}))).toThrow(
    /velocities/
  );
  const props = createProps(graph);
  expect(
    () =>
      new GPULineIntegralConvolution({
        ...props,
        parameters: createTransientView(graph, 'short', 'float32', 4)
      })
  ).toThrow(/parameters must contain 12/);
});

it('LIC parameter helpers pack the documented layout', () => {
  expect(
    Array.from(
      getGPULineIntegralConvolutionParameterValues({
        fieldExtent: [1, 2, 3, 4],
        outputExtent: [5, 6, 7, 8],
        stepLength: 0.25,
        phase: 0.5,
        period: 8
      })
    )
  ).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 0.25, 0, 0.5, 8]);
  expect(Array.from(getGPULineIntegralConvolutionWordParameterValues({seed: 9}))).toEqual([
    9, 0, 0, 0
  ]);
  expect(() => getGPULineIntegralConvolutionWordParameterValues({seed: -2})).toThrow(/seed/);
});

it('convolveLineIntegralOnCPU smooths along the flow and not across it', () => {
  // Horizontal flow: values along a row are strongly correlated, values down a column are not.
  const field = createUniformField(16, 16, 1, 0);
  const parameters = getGPULineIntegralConvolutionParameterValues({
    fieldExtent: [0, 0, 1, 1],
    outputExtent: [0, 0, 0.25, 0.25],
    stepLength: 1
  });
  const {values, noise} = convolveLineIntegralOnCPU(field, 64, 64, 10, parameters, 4);
  let alongFlow = 0;
  let acrossFlow = 0;
  for (let row = 10; row < 54; row++) {
    for (let column = 10; column < 54; column++) {
      const value = values[row * 64 + column];
      alongFlow += Math.abs(value - values[row * 64 + column + 1]);
      acrossFlow += Math.abs(value - values[(row + 1) * 64 + column]);
    }
  }
  expect(alongFlow * 4).toBeLessThan(acrossFlow);
  // Smoothing reduces the variance of the noise.
  const variance = (data: Float32Array) => {
    const mean = data.reduce((sum, value) => sum + value, 0) / data.length;
    return data.reduce((sum, value) => sum + (value - mean) ** 2, 0) / data.length;
  };
  expect(variance(values)).toBeLessThan(variance(noise) / 3);
  // Zero steps return the noise itself.
  const unsmoothed = convolveLineIntegralOnCPU(field, 64, 64, 0, parameters, 4);
  expect(Array.from(unsmoothed.values)).toEqual(Array.from(unsmoothed.noise));
});
