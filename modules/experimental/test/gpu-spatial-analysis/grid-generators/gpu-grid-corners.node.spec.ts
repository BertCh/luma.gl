// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGridGenerator,
  type GPUGridGeneratorProps
} from '../../../src/gpu-spatial-analysis/grid-generators';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPUGridGenerator validates corner output capacities, types, dependencies and graph ownership', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `corner-view-${serial++}`, format, length);
  const base: GPUGridGeneratorProps = {
    gridType: 'square',
    columns: 2,
    rows: 3,
    parameters: view('float32', 4),
    output: {corners: view('float32x2', 12)}
  };
  const extent = {positions: view('float32x2', 4), ringOffsets: view('uint32', 2)};
  expect(new GPUGridGenerator(base).cornerCount).toBe(12);
  expect(new GPUGridGenerator(base).getCommandNodes(graph).map(node => node.id)).toEqual([
    'grid-generator-corners'
  ]);
  for (const gridType of ['triangle', 'point'] as const) {
    expect(() => new GPUGridGenerator({...base, gridType})).toThrow(/output.corners/);
  }
  expect(() => new GPUGridGenerator({...base, output: {corners: view('float32x2', 11)}})).toThrow(
    /output.corners/
  );
  expect(
    () => new GPUGridGenerator({...base, output: {corners: view('uint32', 12) as never}})
  ).toThrow();
  expect(() => new GPUGridGenerator({...base, extent})).toThrow(/intersection/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        output: {...base.output, cornerIntersects: view('uint32', 12)}
      })
  ).toThrow(/intersection/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        extent,
        output: {...base.output, cornerIntersects: view('uint32', 11)}
      })
  ).toThrow(/one row per/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        extent,
        output: {positions: view('float32x2', 24), cornerIntersects: view('uint32', 12)}
      })
  ).toThrow(/one row per/);
  expect(
    () =>
      new GPUGridGenerator({
        ...base,
        extent,
        output: {...base.output, intersects: view('uint32', 6)}
      })
  ).toThrow(/cell geometry/);
  expect(
    new GPUGridGenerator({
      ...base,
      extent,
      output: {...base.output, cornerIntersects: view('uint32', 12)}
    })
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['grid-generator-corners', 'grid-generator-corner-intersects']);
  const otherGraph = new GPUCommandGraph(device);
  const foreignCorners = createTransientView(otherGraph, 'foreign-corners', 'float32x2', 12);
  expect(() =>
    new GPUGridGenerator({...base, output: {corners: foreignCorners}}).getCommandNodes(graph)
  ).toThrow();
  device.destroy();
});
