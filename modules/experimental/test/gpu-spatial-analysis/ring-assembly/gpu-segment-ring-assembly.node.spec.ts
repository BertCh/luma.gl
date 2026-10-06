// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSegmentRingAssembly,
  type GPUSegmentRingAssemblyProps
} from '../../../src/gpu-spatial-analysis/ring-assembly';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSegmentRingAssemblyProps> = {}
): GPUSegmentRingAssemblyProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `ring-view-${serial++}`, format, length);
  return {
    endpoints: view('float32x4', 16),
    output: {
      ringOffsets: view('uint32', 5),
      positions: view('float32x2', 32),
      ringShells: view('uint32', 4),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    ...overrides
  };
}

it('GPUSegmentRingAssembly declares hash, sort, tracing, scan, placement and ring nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-node'});
  const nodes = new GPUSegmentRingAssembly(createProps(graph)).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(20);
  const single = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-node-single'});
  const fewer = new GPUSegmentRingAssembly(
    createProps(single, {splitTouchingRings: false})
  ).getCommandNodes(single);
  expect(fewer.length).toBeLessThan(nodes.length);
});

it('GPUSegmentRingAssembly validates its properties', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-validate'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `validate-${serial++}`, format, length);
  const props = createProps(graph);
  expect(() => new GPUSegmentRingAssembly({...props, vertexTolerance: 0})).toThrow(
    /vertexTolerance/
  );
  expect(() => new GPUSegmentRingAssembly({...props, groups: view('uint32', 3)})).toThrow(
    /groups length/
  );
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        output: {...props.output, ringShells: view('uint32', 7)}
      })
  ).toThrow(/ring capacity/);
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        output: {...props.output, ringGroups: view('uint32', 4)}
      })
  ).toThrow(/groups input/);
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        interiorSide: 'up' as 'left'
      })
  ).toThrow(/interiorSide/);
});
