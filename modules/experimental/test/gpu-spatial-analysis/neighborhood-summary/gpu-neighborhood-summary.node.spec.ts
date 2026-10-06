// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUNeighborhoodSummary} from '../../../src/gpu-spatial-analysis/neighborhood-summary';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUNeighborhoodSummary validates its props and declares one node per kernel', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighborhood-validation'});
  const view = <Format extends 'uint32' | 'float32'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  const weights = {
    offsets: view('uint32', 11, 'offsets'),
    neighbors: view('uint32', 40, 'neighbors'),
    weights: view('float32', 40, 'weights')
  };
  const base = {weights, values: view('float32', 10, 'values')};
  expect(
    () =>
      new GPUNeighborhoodSummary({...base, statistics: ['mean'], output: view('float32', 9, 'o1')})
  ).toThrow(/output length/);
  expect(
    () =>
      new GPUNeighborhoodSummary({
        ...base,
        statistics: ['median'],
        output: view('float32', 10, 'o2')
      })
  ).toThrow(/median requires overflow/);
  expect(
    () =>
      new GPUNeighborhoodSummary({
        ...base,
        statistics: ['mean', 'mean'],
        output: view('float32', 20, 'o3')
      })
  ).toThrow(/must not repeat/);
  expect(() => new GPUNeighborhoodSummary({weights, modes: view('uint32', 10, 'm1')})).toThrow(
    /require categories/
  );
  expect(() => new GPUNeighborhoodSummary({weights})).toThrow(/no statistic/);
  const contributor = new GPUNeighborhoodSummary({
    ...base,
    categories: view('uint32', 10, 'categories'),
    statistics: ['mean', 'median'],
    output: view('float32', 20, 'output'),
    overflow: view('uint32', 1, 'overflow'),
    modes: view('uint32', 10, 'modes'),
    entropy: view('float32', 10, 'entropy')
  });
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'neighborhood-summary-overflow-clear',
    'neighborhood-summary-numeric',
    'neighborhood-summary-categorical'
  ]);
});
