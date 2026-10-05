// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUBufferSelection,
  type GPUBufferSelectionProps
} from '../../../src/map-graphs/spatial-join/gpu-buffer-selection';
import {createNullWebGPUDevice, createVectorView} from '../map-graph-test-utils';

let propsCount = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUBufferSelectionProps> = {}
): GPUBufferSelectionProps {
  const prefix = `props-${propsCount++}-`;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${prefix}${name}`, format, length);
  return {
    points: view('points', 'float32x2', 10),
    features: {
      kind: 'segments',
      starts: view('starts', 'float32x2', 3),
      ends: view('ends', 'float32x2', 3)
    },
    distance: view('distance', 'float32', 1),
    candidateCapacity: 16,
    outputMask: view('mask', 'uint32', 10),
    ...overrides
  };
}

it('GPUBufferSelection builds nodes for a valid configuration', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-node'});
  const selection = new GPUBufferSelection(createProps(graph));
  expect(selection.recipe).toBe('buffer-selection');
  expect(selection.id).toBe('buffer-selection');
  const ids = selection.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('buffer-selection-mask');
  expect(ids.some(id => id.startsWith('buffer-selection-join'))).toBe(true);
  expect(ids.some(id => id.startsWith('buffer-selection-publish'))).toBe(false);
});

it('GPUBufferSelection builds one mask node per chunk and a publish node with output', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-chunks'});
  const view = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const selection = new GPUBufferSelection(
    createProps(graph, {
      points: createVectorView('points', 'float32x2', [
        view('points-0', 'float32x2', 6),
        view('points-1', 'float32x2', 4)
      ]),
      outputMask: createVectorView('mask', 'uint32', [
        view('mask-0', 'uint32', 6),
        view('mask-1', 'uint32', 4)
      ]),
      output: {
        ids: view('out-ids', 'uint32', 4),
        count: view('out-count', 'uint32', 1),
        overflow: view('out-overflow', 'uint32', 1)
      }
    })
  );
  const ids = selection.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('buffer-selection-mask-chunk-0');
  expect(ids).toContain('buffer-selection-mask-chunk-1');
  expect(ids).toContain('buffer-selection-publish');
});

it('GPUBufferSelection validates outputs and the distance view', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-selection-validation'});
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  expect(
    () => new GPUBufferSelection(createProps(graph, {outputMask: undefined, output: undefined}))
  ).toThrow(/requires outputMask or output/);
  expect(
    () => new GPUBufferSelection(createProps(graph, {distance: view('bad-distance', 'float32', 2)}))
  ).toThrow(/distance must contain one float32 row/);
  expect(
    () => new GPUBufferSelection(createProps(graph, {outputMask: view('short-mask', 'uint32', 9)}))
  ).toThrow(/outputMask must match the source topology/);
  expect(
    () =>
      new GPUBufferSelection(
        createProps(graph, {
          outputMask: undefined,
          output: {
            ids: view('empty-ids', 'uint32', 0),
            count: view('count', 'uint32', 1),
            overflow: view('overflow', 'uint32', 1)
          }
        })
      )
  ).toThrow(/output.ids must hold at least one row/);
  expect(() => new GPUBufferSelection(createProps(graph, {candidateCapacity: 0}))).toThrow(
    /candidateCapacity/
  );
});
