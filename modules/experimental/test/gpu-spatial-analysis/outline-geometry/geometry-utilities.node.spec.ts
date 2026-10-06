// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUGridGenerator} from '../../../src/gpu-spatial-analysis/grid-generators';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density';
import {GPUOutlineGeometry} from '../../../src/gpu-spatial-analysis/outline-geometry';
import {GPURectangleClip} from '../../../src/gpu-spatial-analysis/rectangle-clip';
import {GPUShapeDescriptors} from '../../../src/gpu-spatial-analysis/shape-descriptors';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function view<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

it('GPUOutlineGeometry validates sizes and declares one node', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'outline'});
  const props = {
    positions: view(graph, 'float32x2', 5),
    pathOffsets: view(graph, 'uint32', 3),
    geometryType: 'lines' as const,
    joinSegments: 4,
    parameters: view(graph, 'float32', 4)
  };
  const contributor = new GPUOutlineGeometry({
    ...props,
    output: {positions: view(graph, 'float32x2', 5 * 18)}
  });
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'outline-geometry-generate'
  ]);
  expect(
    () => new GPUOutlineGeometry({...props, output: {positions: view(graph, 'float32x2', 7)}})
  ).toThrow(/rows/);
  expect(
    () =>
      new GPUOutlineGeometry({
        ...props,
        pathOffsets: undefined,
        output: {positions: view(graph, 'float32x2', 5 * 18)}
      })
  ).toThrow(/pathOffsets/);
  expect(
    () =>
      new GPUOutlineGeometry({
        ...props,
        joinSegments: 2,
        output: {positions: view(graph, 'float32x2', 30)}
      })
  ).toThrow(/joinSegments/);
});

it('GPULabelPoint validates outputs and grid sizes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'label'});
  const props = {
    positions: view(graph, 'float32x2', 8),
    ringOffsets: view(graph, 'uint32', 3)
  };
  const contributor = new GPULabelPoint({...props, output: {points: view(graph, 'float32x2', 2)}});
  // One per-thread kernel for small features and one workgroup kernel for large ones.
  expect(contributor.getCommandNodes(graph)).toHaveLength(2);
  expect(
    () => new GPULabelPoint({...props, output: {points: view(graph, 'float32x2', 3)}})
  ).toThrow(/points/);
  expect(
    () =>
      new GPULabelPoint({
        ...props,
        refinementCandidates: 40,
        output: {points: view(graph, 'float32x2', 2)}
      })
  ).toThrow(/refinementCandidates/);
});

it('GPUShapeDescriptors requires an output and composes measures with kernels', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape'});
  const props = {
    positions: view(graph, 'float32x2', 8),
    ringOffsets: view(graph, 'uint32', 3),
    parameters: view(graph, 'float32', 4)
  };
  expect(() => new GPUShapeDescriptors({...props, output: {}})).toThrow(/output column/);
  const contributor = new GPUShapeDescriptors({
    ...props,
    output: {
      polsbyPopper: view(graph, 'float32', 2),
      elongation: view(graph, 'float32', 2),
      convexity: view(graph, 'float32', 2)
    }
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('shape-descriptors-compactness');
  expect(ids).toContain('shape-descriptors-moments');
  expect(ids).toContain('shape-descriptors-convexity');
});

it('GPULineDensity, GPURectangleClip and GPUGridGenerator validate their props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'misc'});
  const density = new GPULineDensity({
    positions: view(graph, 'float32x2', 6),
    pathOffsets: view(graph, 'uint32', 3),
    columns: 4,
    rows: 3,
    parameters: view(graph, 'float32', 4),
    output: {lengths: view(graph, 'float32', 12), overflow: view(graph, 'uint32', 1)}
  });
  const densityIds = density.getCommandNodes(graph).map(node => node.id);
  expect(densityIds).toContain('line-density-emit');
  expect(densityIds).toContain('line-density-publish');
  expect(
    () =>
      new GPULineDensity({
        positions: view(graph, 'float32x2', 6),
        pathOffsets: view(graph, 'uint32', 3),
        columns: 4,
        rows: 3,
        parameters: view(graph, 'float32', 4),
        output: {lengths: view(graph, 'float32', 11), overflow: view(graph, 'uint32', 1)}
      })
  ).toThrow(/lengths/);

  const clipOutput = (capacity: number, pathCapacity: number) => ({
    positions: view(graph, 'float32x2', capacity),
    pathOffsets: view(graph, 'uint32', pathCapacity + 1),
    count: view(graph, 'uint32', 1),
    overflow: view(graph, 'uint32', 1)
  });
  const clip = new GPURectangleClip({
    positions: view(graph, 'float32x2', 6),
    pathOffsets: view(graph, 'uint32', 3),
    geometryType: 'polygons',
    parameters: view(graph, 'float32', 4),
    output: clipOutput(32, 2)
  });
  // Four stages of count, scan, emit and offsets, then publish.
  expect(clip.getCommandNodes(graph).length).toBeGreaterThan(16);
  expect(
    () =>
      new GPURectangleClip({
        positions: view(graph, 'float32x2', 6),
        pathOffsets: view(graph, 'uint32', 3),
        geometryType: 'polygons',
        parameters: view(graph, 'float32', 4),
        output: clipOutput(32, 5)
      })
  ).toThrow(/output.pathOffsets/);

  const grid = new GPUGridGenerator({
    gridType: 'hex',
    columns: 3,
    rows: 2,
    parameters: view(graph, 'float32', 4),
    output: {positions: view(graph, 'float32x2', 36)}
  });
  expect(grid.getCommandNodes(graph)).toHaveLength(1);
  expect(
    () =>
      new GPUGridGenerator({
        gridType: 'point',
        columns: 3,
        rows: 2,
        parameters: view(graph, 'float32', 4),
        output: {positions: view(graph, 'float32x2', 6)}
      })
  ).toThrow(/centers/);
});
