// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights-transpose-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const weights = (rows: number, capacity: number, distances = false): GPUSpatialWeights => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined
  });
  return {graph, view, weights};
}

it('GPUSpatialWeightsTranspose validates props and declares nodes', () => {
  const {graph, view, weights} = setup();
  const input = weights(4, 10);
  expect(() => new GPUSpatialWeightsTranspose({weights: input, output: weights(5, 10)})).toThrow(
    /columnCount \+ 1/
  );
  expect(() => new GPUSpatialWeightsTranspose({weights: input, output: weights(4, 9)})).toThrow(
    /capacity/
  );
  expect(
    () => new GPUSpatialWeightsTranspose({weights: weights(4, 10, true), output: weights(4, 10)})
  ).toThrow(/distances/);
  expect(
    () => new GPUSpatialWeightsTranspose({weights: input, columnCount: 0, output: weights(4, 10)})
  ).toThrow(/columnCount/);
  expect(
    () =>
      new GPUSpatialWeightsTranspose({
        weights: input,
        columnCount: 6,
        output: weights(6, 10),
        asymmetricSlots: view('uint32', 1)
      })
  ).toThrow(/square/);
  expect(
    () =>
      new GPUSpatialWeightsTranspose({
        weights: input,
        output: weights(4, 10),
        asymmetricSlots: view('uint32', 2)
      })
  ).toThrow(/one uint32/);
  expect(
    () =>
      new GPUSpatialWeightsTranspose({
        weights: input,
        output: weights(4, 10),
        symmetryTolerance: -1
      })
  ).toThrow(/symmetryTolerance/);
  const nodes = new GPUSpatialWeightsTranspose({
    weights: input,
    columnCount: 6,
    output: weights(6, 10)
  }).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(3);
  expect(nodes[0].id).toBe('spatial-weights-transpose-slot-keys');
});
