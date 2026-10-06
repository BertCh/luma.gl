// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSegregationLayout,
  GPUSegregation
} from '../../../src/gpu-spatial-analysis/segregation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

it('GPUSegregation declares nodes and validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'segregation-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const layout = getGPUSegregationLayout(3);
  expect(layout.stride).toBe(3 + 9 + 9);
  const weights = {
    offsets: view('uint32', 11),
    neighbors: view('uint32', 20),
    weights: view('float32', 20)
  };
  const props = {
    unitCount: 10,
    groupCount: 3,
    groupCounts: view('float32', 30),
    scales: [null, weights],
    indices: view('float32', 2 * layout.stride)
  };
  expect(new GPUSegregation(props).getCommandNodes(graph).length).toBeGreaterThan(8);
  expect(() => new GPUSegregation({...props, groupCount: 1})).toThrow(/groupCount/);
  expect(() => new GPUSegregation({...props, atkinsonB: 1})).toThrow(/atkinsonB/);
  expect(() => new GPUSegregation({...props, indices: view('float32', layout.stride)})).toThrow(
    /indices/
  );
  expect(
    () => new GPUSegregation({...props, scales: [{...weights, offsets: view('uint32', 5)}]})
  ).toThrow(/one row per unit/);
});
