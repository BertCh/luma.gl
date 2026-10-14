// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUGridCellCount,
  getGPUGridCornerCount,
  getGPUGridVerticesPerCell,
  GPUGridGenerator,
  type GPUGridGeneratorProps
} from '../../../src/gpu-spatial-analysis/grid-generators';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

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
  const extent = {
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2)
  };
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

it('GPUGridGenerator polygon and corner kernels parse for every grid type', () => {
  const device = createNullWebGPUDevice();
  for (const [gridType, hexOrientation] of [
    ['square', undefined],
    ['hex', undefined],
    ['hex', 'flat'],
    ['triangle', undefined]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const columns = 3;
    const rows = 2;
    const generator = new GPUGridGenerator({
      gridType,
      hexOrientation,
      columns,
      rows,
      parameters: createTransientView(graph, 'parameters', 'float32', 4),
      output: {
        positions: createTransientView(
          graph,
          'positions',
          'float32x2',
          getGPUGridCellCount(gridType, columns, rows) * getGPUGridVerticesPerCell(gridType)
        ),
        centers: createTransientView(
          graph,
          'centers',
          'float32x2',
          getGPUGridCellCount(gridType, columns, rows)
        ),
        ...(gridType === 'triangle'
          ? {}
          : {
              corners: createTransientView(
                graph,
                'corners',
                'float32x2',
                getGPUGridCornerCount(gridType, columns, rows)
              )
            })
      }
    });
    const kernels = getGPUCommandNodeKernelSources(generator.getCommandNodes(graph));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels) {
      expect(() => new WgslReflect(kernel.source), `${gridType} ${kernel.id}`).not.toThrow();
    }
  }
  device.destroy();
});
