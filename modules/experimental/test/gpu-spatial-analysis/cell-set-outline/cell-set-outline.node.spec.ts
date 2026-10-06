// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCellSetOutline,
  type GPUCellSetOutlineProps
} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellSetOutlineProps> = {}
): GPUCellSetOutlineProps {
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    family: 'h3',
    cells: view('uint32x2', 8),
    output: {
      rows: view('uint32', 16),
      cells: view('uint32x2', 16),
      edgeIndices: view('uint32', 16),
      endpoints: view('float32x4', 16),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    ...overrides
  };
}

it('GPUCellSetOutline declares classify, scan, write, total and publish nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'outline-node'});
  const nodes = new GPUCellSetOutline(createProps(graph)).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThanOrEqual(5);
  const quadbin = new GPUCellSetOutline(createProps(graph, {family: 'quadbin'}));
  expect(quadbin.edgeSlots).toBe(4);
  expect(new GPUCellSetOutline(createProps(graph)).edgeSlots).toBe(10);
});

it('GPUCellSetOutline validates its properties', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'outline-validate'});
  const view = <Format extends 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `validate-${serial++}`, format, length);
  expect(() => new GPUCellSetOutline(createProps(graph, {family: 's2' as 'h3'}))).toThrow(/family/);
  expect(() => new GPUCellSetOutline(createProps(graph, {groups: view('uint32', 3)}))).toThrow(
    /groups length/
  );
  const props = createProps(graph);
  expect(
    () =>
      new GPUCellSetOutline({
        ...props,
        output: {...props.output, edgeIndices: view('uint32', 3)}
      })
  ).toThrow(/same length/);
  expect(
    () =>
      new GPUCellSetOutline({
        ...props,
        output: {...props.output, groups: view('uint32', 16)}
      })
  ).toThrow(/groups input/);
  expect(() => new GPUCellSetOutline({...props, cells: props.output.cells})).toThrow(
    /disjoint|alias|share/i
  );
});
