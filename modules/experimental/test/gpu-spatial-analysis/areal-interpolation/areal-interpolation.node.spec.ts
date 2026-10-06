// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUArealInterpolation,
  GPUPycnophylactic
} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'areal-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {graph, view};
}

it('GPUArealInterpolation declares nodes and validates its props', () => {
  const {graph, view} = setup();
  const props = {
    sourceZones: view('uint32', 64),
    targetZones: view('uint32', 64),
    sourceCount: 4,
    targetCount: 3,
    weights: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    overflow: view('uint32', 1)
  };
  const nodes = new GPUArealInterpolation(props).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(10);
  expect(() => new GPUArealInterpolation({...props, targetCount: 4})).toThrow(/targetCount \+ 1/);
  expect(() => new GPUArealInterpolation({...props, targetZones: view('uint32', 63)})).toThrow(
    /same nonzero length/
  );
  expect(() => new GPUArealInterpolation({...props, alternateWeights: view('float32', 7)})).toThrow(
    /alternateWeights length/
  );
});

it('GPUPycnophylactic declares nodes per iteration and validates its props', () => {
  const {graph, view} = setup();
  const props = {
    width: 8,
    height: 4,
    zones: view('uint32', 32),
    zoneCount: 3,
    totals: view('float32', 3),
    iterations: 2,
    output: view('float32', 32)
  };
  const few = new GPUPycnophylactic(props).getCommandNodes(graph).length;
  const {graph: other, view: otherView} = setup();
  const many = new GPUPycnophylactic({
    ...props,
    zones: otherView('uint32', 32),
    totals: otherView('float32', 3),
    output: otherView('float32', 32),
    iterations: 4
  }).getCommandNodes(other).length;
  expect(many).toBeGreaterThan(few);
  expect(() => new GPUPycnophylactic({...props, iterations: -1})).toThrow(/iterations/);
  expect(() => new GPUPycnophylactic({...props, totals: view('float32', 2)})).toThrow(/totals/);
});

it('GPUSpatialLag accepts cross weights with sourceCount and columnCount', () => {
  const {graph, view} = setup();
  const weights = {
    offsets: view('uint32', 4),
    neighbors: view('uint32', 8),
    weights: view('float32', 8)
  };
  const props = {weights, sourceCount: 5, values: view('float32', 10), output: view('float32', 6)};
  expect(new GPUSpatialLag({...props, columnCount: 2}).getCommandNodes(graph)).toHaveLength(1);
  expect(() => new GPUSpatialLag(props)).toThrow(/sourceCount \* columnCount/);
  expect(() => new GPUSpatialLag({...props, columnCount: 2, mask: view('uint32', 3)})).toThrow(
    /square/
  );
});
