// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUCoverageSimplification validates props and declares nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'coverage-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const make = () => ({
    positions: view('float32x2', 12),
    ringOffsets: view('uint32', 4),
    polygonOffsets: view('uint32', 4),
    parameters: view('float32', 4),
    output: {
      positions: view('float32x2', 12),
      ringOffsets: view('uint32', 4),
      overflow: view('uint32', 1)
    }
  });
  const base = make();
  expect(() => new GPUCoverageSimplification({...base, snapTolerance: -1})).toThrow(
    /snapTolerance/
  );
  expect(() => new GPUCoverageSimplification({...base, parameters: view('float32', 2)})).toThrow(
    /parameters/
  );
  expect(
    () =>
      new GPUCoverageSimplification({
        ...base,
        output: {...base.output, ringOffsets: view('uint32', 5)}
      })
  ).toThrow(/ringOffsets length/);
  expect(
    () =>
      new GPUCoverageSimplification({
        ...base,
        output: {...base.output, keepMask: view('uint32', 3)}
      })
  ).toThrow(/keepMask length/);
  const ids = new GPUCoverageSimplification(base).getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('coverage-simplification-vertex-topology');
  expect(ids.at(-1)).toBe('coverage-simplification-emit-offsets');
  expect(ids).toContain('coverage-simplification-arcs-init');
});
