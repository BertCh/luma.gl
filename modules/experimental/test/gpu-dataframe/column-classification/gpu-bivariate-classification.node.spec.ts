// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUBivariateClassification} from '../../../src/gpu-dataframe/column-classification/gpu-bivariate-classification';
import type {GPUBivariateClassificationProps} from '../../../src/gpu-dataframe/column-classification/gpu-bivariate-classification';
import {
  getGPUBivariateClassificationParameterValues,
  GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/column-classification/bivariate-classification-parameters';
import {computeBivariateClassificationOnCPU} from './bivariate-classification-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

const view = <Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `view-${serial++}`, format, length);

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUBivariateClassificationProps> = {}
): GPUBivariateClassificationProps {
  return {
    valuesX: view(graph, 'float32', 10),
    valuesY: view(graph, 'float32', 10),
    breaksX: view(graph, 'float32', 4),
    breaksY: view(graph, 'float32', 4),
    palette: view(graph, 'uint32', 9),
    parameters: view(graph, 'float32', GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH),
    maximumClassCount: 3,
    output: {
      classIds: view(graph, 'uint32', 10),
      colors: view(graph, 'uint32', 10),
      classCounts: view(graph, 'uint32', 9)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUBivariateClassificationProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUBivariateClassification(createProps(graph, overrides(graph)))).toThrow(
    message
  );
  device.destroy();
}

it('GPUBivariateClassification parameter helper packs the documented layout', () => {
  expect(
    Array.from(
      getGPUBivariateClassificationParameterValues({
        classCountX: 3,
        classCountY: 4,
        noDataColor: 0xdeadbeef,
        valueByAlpha: {domain: [0.5, 2], minimumAlpha: 0.25}
      })
    )
  ).toEqual([3, 4, 0xbeef, 0xdead, 1, 0.5, 2, 0.25]);
  expect(
    Array.from(
      getGPUBivariateClassificationParameterValues({
        classCountX: 2,
        classCountY: 2
      })
    )
  ).toEqual([2, 2, 0, 0, 0, 0, 1, 1]);
  expect(() =>
    getGPUBivariateClassificationParameterValues({
      classCountX: -1,
      classCountY: 2
    })
  ).toThrow(/classCountX/);
  expect(() =>
    getGPUBivariateClassificationParameterValues(
      {classCountX: 1, classCountY: 1},
      new Float32Array(3)
    )
  ).toThrow(/hold/);
});

it('GPUBivariateClassification returns deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUBivariateClassification(createProps(graph, {id: 'bi'}));
  expect(recipe.recipe).toBe('bivariate-classification');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['bi-init', 'bi-classify', 'bi-colorize']);
  const otherGraph = new GPUCommandGraph(device);
  const other = new GPUBivariateClassification(createProps(otherGraph, {id: 'bi'}));
  expect(other.getCommandNodes(otherGraph).map(node => node.id)).toEqual(ids);
  const withAlpha = new GPUBivariateClassification(
    createProps(otherGraph, {
      id: 'va',
      alphaValues: view(otherGraph, 'float32', 10),
      mask: view(otherGraph, 'uint32', 10)
    })
  );
  expect(withAlpha.getCommandNodes(otherGraph).length).toBe(3);
  device.destroy();
});

it('GPUBivariateClassification validates props', () => {
  expectThrows(() => ({maximumClassCount: 0}), /maximumClassCount/);
  expectThrows(() => ({maximumClassCount: 17}), /maximumClassCount/);
  expectThrows(() => ({maximumClassCount: 2.5}), /maximumClassCount/);
  expectThrows(() => ({output: {}}), /at least one output/);
  expectThrows(
    graph => ({
      valuesX: view(graph, 'float32', 0),
      valuesY: view(graph, 'float32', 0)
    }),
    /at least one row/
  );
  expectThrows(graph => ({valuesX: view(graph, 'uint32' as 'float32', 10)}), /valuesX/);
  expectThrows(graph => ({valuesY: view(graph, 'float32', 9)}), /valuesY length/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(graph => ({alphaValues: view(graph, 'float32', 9)}), /alphaValues length/);
  expectThrows(graph => ({breaksX: view(graph, 'float32', 3)}), /breaksX must hold/);
  expectThrows(graph => ({breaksY: view(graph, 'float32', 3)}), /breaksY must hold/);
  expectThrows(graph => ({palette: view(graph, 'uint32', 8)}), /palette must hold/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 7)}), /parameters must hold/);
  expectThrows(graph => ({output: {classIds: view(graph, 'uint32', 9)}}), /output.classIds/);
  expectThrows(graph => ({output: {colors: view(graph, 'uint32', 9)}}), /output.colors/);
  expectThrows(graph => ({output: {classCounts: view(graph, 'uint32', 8)}}), /output.classCounts/);
});

it('GPUBivariateClassification rejects aliased buffers and foreign graphs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, {mask: view(graph, 'uint32', 10)});
  expect(
    () =>
      new GPUBivariateClassification({
        ...props,
        output: {...props.output, colors: props.mask as GraphDataView<'uint32'>}
      })
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUBivariateClassification({
        ...props,
        output: {classIds: props.output.colors, colors: props.output.colors}
      })
  ).toThrow(/share buffers/);
  const recipe = new GPUBivariateClassification(createProps(graph));
  expect(() => recipe.getCommandNodes(new GPUCommandGraph(device))).toThrow(/target graph/);
  device.destroy();
});

it('computeBivariateClassificationOnCPU matches a hand-checked 2x2 grid', () => {
  const none = 0xffffffff;
  const result = computeBivariateClassificationOnCPU({
    valuesX: Float32Array.from([0, 5, 10, 5, NaN, 2, 100]),
    valuesY: Float32Array.from([0, 5, 10, 0, 1, NaN, -100]),
    mask: [1, 1, 1, 1, 1, 1, 0],
    breaksX: Float32Array.from([0, 5, 10]),
    breaksY: Float32Array.from([0, 5, 10]),
    classCountX: 2,
    classCountY: 2,
    maximumClassCount: 2,
    palette: [0xff000001, 0xff000002, 0xff000003, 0xff000004],
    noDataColor: 7
  });
  // Row 1 sits on both inner edges, row 2 clamps to the last class on both axes.
  expect(Array.from(result.classIds)).toEqual([0, 3, 3, 1, none, none, none]);
  expect(Array.from(result.classCounts)).toEqual([1, 1, 0, 2]);
  expect(result.colors[4]).toBe(7);
  const faded = computeBivariateClassificationOnCPU({
    valuesX: Float32Array.from([0, 0, 0]),
    valuesY: Float32Array.from([0, 0, 0]),
    breaksX: Float32Array.from([0, 5, 10]),
    breaksY: Float32Array.from([0, 5, 10]),
    classCountX: 2,
    classCountY: 2,
    maximumClassCount: 2,
    palette: [0xff0000aa, 0, 0, 0],
    alphaValues: Float32Array.from([0, 1, NaN]),
    valueByAlpha: {domain: [0, 1], minimumAlpha: 0.5}
  });
  expect(Array.from(faded.colors, color => color >>> 24)).toEqual([128, 255, 128]);
});
