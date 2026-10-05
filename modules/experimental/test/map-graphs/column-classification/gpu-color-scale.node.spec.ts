// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUColorScale} from '../../../src/map-graphs/column-classification/gpu-color-scale';
import type {GPUColorScaleProps} from '../../../src/map-graphs/column-classification/gpu-color-scale';
import {
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_CODES,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  packGPUColor
} from '../../../src/map-graphs/column-classification/color-scale-parameters';
import {computeColorScaleOnCPU} from './color-scale-oracle';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUColorScaleProps> = {}
): GPUColorScaleProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    values: view('float32', 10),
    domain: view('float32', 8),
    palette: view('uint32', 6),
    parameters: view('float32', GPU_COLOR_SCALE_PARAMETER_LENGTH),
    maximumDomainCount: 8,
    maximumPaletteCount: 6,
    output: {
      colors: view('uint32', 10),
      classIndices: view('uint32', 10),
      classCounts: view('uint32', 6)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUColorScaleProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUColorScale(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

const view = <Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `extra-${serial++}`, format, length);

it('GPUColorScale parameter helper packs the documented layout', () => {
  const values = getGPUColorScaleParameterValues({
    scale: 'log',
    domainCount: 5,
    paletteCount: 4,
    interpolation: 'linear',
    clamp: true,
    noDataColor: 0xdeadbeef,
    logFloor: 0.5,
    exponent: 3
  });
  expect(values.length).toBe(GPU_COLOR_SCALE_PARAMETER_LENGTH);
  expect(Array.from(values)).toEqual([
    GPU_COLOR_SCALE_CODES.log,
    5,
    4,
    1,
    1,
    0xbeef,
    0xdead,
    0.5,
    3
  ]);
  const defaults = getGPUColorScaleParameterValues({
    scale: 'ordinal',
    domainCount: 0,
    paletteCount: 1
  });
  expect(Array.from(defaults)).toEqual([8, 0, 1, 0, 0, 0, 0, Math.fround(1e-5), 1]);
  expect(() =>
    getGPUColorScaleParameterValues({
      scale: 'linear',
      domainCount: 1.5,
      paletteCount: 1
    })
  ).toThrow(/domainCount/);
  expect(() =>
    getGPUColorScaleParameterValues({
      scale: 'bogus' as 'linear',
      domainCount: 1,
      paletteCount: 1
    })
  ).toThrow(/Unknown/);
  expect(() =>
    getGPUColorScaleParameterValues(
      {scale: 'linear', domainCount: 1, paletteCount: 1},
      new Float32Array(2)
    )
  ).toThrow(/hold/);
  expect(packGPUColor(1, 2, 3, 4)).toBe(0x04030201);
  expect(packGPUColor(255, 255, 255)).toBe(0xffffffff);
});

it('GPUColorScale accepts valid props and returns deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUColorScale(createProps(graph, {id: 'scale'}));
  expect(recipe.recipe).toBe('color-scale');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['scale-init', 'scale-classify', 'scale-colorize']);
  const secondGraph = new GPUCommandGraph(device);
  const secondRecipe = new GPUColorScale(createProps(secondGraph, {id: 'scale'}));
  expect(secondRecipe.getCommandNodes(secondGraph).map(node => node.id)).toEqual(ids);
  // Classes only need no colorize node and no init.
  const classOnlyGraph = new GPUCommandGraph(device);
  const classOnly = new GPUColorScale(
    createProps(classOnlyGraph, {
      id: 'classes-only',
      output: {classIndices: view(classOnlyGraph, 'uint32', 10)}
    })
  );
  expect(classOnly.getCommandNodes(classOnlyGraph).map(node => node.id)).toEqual([
    'classes-only-classify'
  ]);
  device.destroy();
});

it('GPUColorScale accepts uint32 ordinal values, domainCount views and masks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUColorScale(
    createProps(graph, {
      values: view(graph, 'uint32', 10),
      mask: view(graph, 'uint32', 10),
      domainCount: view(graph, 'uint32', 1)
    })
  );
  expect(recipe.getCommandNodes(graph).length).toBe(3);
  device.destroy();
});

it('GPUColorScale validates props', () => {
  expectThrows(() => ({maximumDomainCount: 0}), /maximumDomainCount/);
  expectThrows(() => ({maximumPaletteCount: 1.5}), /maximumPaletteCount/);
  expectThrows(() => ({output: {}}), /at least one output/);
  expectThrows(graph => ({values: view(graph, 'float32', 0)}), /at least one row/);
  expectThrows(graph => ({values: view(graph, 'float32x2' as 'float32', 10)}), /values/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(graph => ({mask: view(graph, 'float32' as 'uint32', 10)}), /mask/);
  expectThrows(graph => ({domain: view(graph, 'float32', 7)}), /domain must hold/);
  expectThrows(graph => ({domain: view(graph, 'uint32' as 'float32', 8)}), /domain/);
  expectThrows(graph => ({palette: view(graph, 'uint32', 5)}), /palette must hold/);
  expectThrows(graph => ({palette: view(graph, 'float32' as 'uint32', 6)}), /palette/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 8)}), /parameters must hold/);
  expectThrows(graph => ({parameters: view(graph, 'uint32' as 'float32', 9)}), /parameters/);
  expectThrows(graph => ({domainCount: view(graph, 'uint32', 0)}), /domainCount/);
  expectThrows(graph => ({domainCount: view(graph, 'float32' as 'uint32', 1)}), /domainCount/);
  expectThrows(graph => ({output: {colors: view(graph, 'uint32', 9)}}), /output.colors/);
  expectThrows(
    graph => ({output: {classIndices: view(graph, 'uint32', 9)}}),
    /output.classIndices/
  );
  expectThrows(graph => ({output: {classCounts: view(graph, 'uint32', 5)}}), /output.classCounts/);
  expectThrows(
    graph => ({output: {colors: view(graph, 'float32' as 'uint32', 10)}}),
    /output.colors/
  );
});

it('GPUColorScale rejects aliased buffers and foreign graphs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, {mask: view(graph, 'uint32', 10)});
  const aliased: GPUColorScaleProps = {
    ...props,
    output: {...props.output, colors: props.mask as GraphDataView<'uint32'>}
  };
  expect(() => new GPUColorScale(aliased)).toThrow(/share buffers/);
  const sharedOutputs: GPUColorScaleProps = {
    ...props,
    output: {colors: props.output.colors, classIndices: props.output.colors}
  };
  expect(() => new GPUColorScale(sharedOutputs)).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  const recipe = new GPUColorScale(createProps(graph));
  expect(() => recipe.getCommandNodes(otherGraph)).toThrow(/target graph/);
  device.destroy();
});

it('computeColorScaleOnCPU follows d3 semantics on a hand-checked case', () => {
  const palette = [10, 20, 30, 40].map(value => packGPUColor(value, 0, 0, 255));
  const base = {
    values: Float32Array.from([-1, 0, 2, 3, 8, 9, NaN, 4]),
    mask: [1, 1, 1, 1, 1, 1, 1, 0],
    domain: Float32Array.from([0, 8]),
    activeDomainCount: 2,
    palette,
    paletteCount: 4,
    noDataColor: 0x01020304,
    maximumDomainCount: 2,
    maximumPaletteCount: 4
  };
  const unclamped = computeColorScaleOnCPU({...base, scale: 'linear'});
  const none = 0xffffffff;
  expect(Array.from(unclamped.classIndices)).toEqual([none, 0, 1, 1, 3, none, none, none]);
  expect(unclamped.colors[0]).toBe(0x01020304);
  expect(Array.from(unclamped.classCounts)).toEqual([1, 2, 0, 1]);
  const clamped = computeColorScaleOnCPU({
    ...base,
    scale: 'linear',
    clamp: true
  });
  expect(Array.from(clamped.classIndices)).toEqual([0, 0, 1, 1, 3, 3, none, none]);
  const threshold = computeColorScaleOnCPU({
    ...base,
    scale: 'threshold',
    domain: Float32Array.from([-Infinity, 1, 1, 5, Infinity]),
    activeDomainCount: 5,
    maximumDomainCount: 5
  });
  // Inner edges 1, 1, 5: a value on edge 1 counts both, so class 1 is empty.
  expect(Array.from(threshold.classIndices)).toEqual([0, 0, 2, 2, 3, 3, none, none]);
  const logScale = computeColorScaleOnCPU({
    ...base,
    scale: 'log',
    values: Float32Array.from([0, 1, 10, 100]),
    mask: undefined,
    domain: Float32Array.from([1, 100]),
    clamp: true
  });
  // log floor sends 0 below the domain, which clamps to class 0.
  expect(Array.from(logScale.classIndices)).toEqual([0, 0, 2, 3]);
  const ordinal = computeColorScaleOnCPU({
    ...base,
    scale: 'ordinal',
    values: Uint32Array.from([0, 3, 4, 100])
  });
  expect(Array.from(ordinal.classIndices)).toEqual([0, 3, none, none]);
});
