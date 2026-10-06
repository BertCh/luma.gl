// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUMapColoring} from '../../../src/gpu-spatial-analysis/map-coloring/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUMapColoring validates props and declares gated rounds', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-coloring-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const make = () => ({
    weights: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8)
    },
    colors: view('uint32', 4)
  });
  const base = make();
  expect(() => new GPUMapColoring({...base, colors: view('uint32', 3)})).toThrow(/row count/);
  expect(() => new GPUMapColoring({...base, maximumRounds: 0})).toThrow(/maximumRounds/);
  expect(() => new GPUMapColoring({...base, seed: -1})).toThrow(/seed/);
  const ids = new GPUMapColoring({...base, maximumRounds: 3, conflictCount: view('uint32', 1)})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('map-coloring-round-2-select');
  expect(ids).not.toContain('map-coloring-round-3-select');
  expect(ids.at(-1)).toBe('map-coloring-conflicts');
});
