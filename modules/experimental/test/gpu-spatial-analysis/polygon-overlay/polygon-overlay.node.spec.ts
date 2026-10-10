// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUBufferSurface,
  GPUPolygonOverlay,
  type GPUPolygonOverlayOutput
} from '../../../src/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function view<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `overlay-view-${serial++}`, format, length);
}

function polygon(graph: GPUCommandGraph, vertices = 8, rings = 2, polygons = 1) {
  return {
    kind: 'polygons' as const,
    positions: view(graph, 'float32x2', vertices),
    featureOffsets: view(graph, 'uint32', polygons + 1),
    polygonOffsets: view(graph, 'uint32', polygons + 1),
    ringOffsets: view(graph, 'uint32', rings + 1)
  };
}

function output(graph: GPUCommandGraph, vertices = 64, rings = 16): GPUPolygonOverlayOutput {
  const sourceIds = view(graph, 'uint32', rings);
  const status = () => ({
    count: view(graph, 'uint32', 1),
    requiredCount: view(graph, 'uint32', 1),
    overflow: view(graph, 'uint32', 1),
    candidateOverflow: view(graph, 'uint32', 1)
  });
  return {
    geometry: {
      kind: 'polygons',
      positions: view(graph, 'float32x2', vertices),
      featureOffsets: view(graph, 'uint32', rings + 1),
      polygonOffsets: view(graph, 'uint32', rings + 1),
      ringOffsets: view(graph, 'uint32', rings + 1),
      sourceIds
    },
    sourceIds,
    status: status(),
    boundary: {
      endpoints: view(graph, 'float32x4', 64),
      operandIds: view(graph, 'uint32', 64),
      sourceFeatureIds: view(graph, 'uint32', 64),
      status: status()
    }
  };
}

it('GPUPolygonOverlay builds a noded Boolean boundary and compact polygon surface', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'overlay'});
  const contributor = new GPUPolygonOverlay({
    left: polygon(graph),
    right: polygon(graph),
    operation: 'intersection',
    capacity: {intersections: 32, nodedSegments: 64},
    vertexTolerance: 1e-5,
    output: output(graph)
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => id.startsWith('polygon-overlay-noding-'))).toBe(true);
  expect(ids).toContain('polygon-overlay-classify-boundary');
  expect(ids).toContain('polygon-overlay-emit-boundary');
  expect(ids.some(id => id.startsWith('polygon-overlay-assemble-'))).toBe(true);
  expect(ids.at(-1)).toBe('polygon-overlay-publish');
});

it('GPUPolygonOverlay enforces unary dissolve and shared canonical source IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'overlay-validation'});
  expect(
    () =>
      new GPUPolygonOverlay({
        left: polygon(graph),
        operation: 'union',
        capacity: {intersections: 8, nodedSegments: 16},
        vertexTolerance: 1e-5,
        output: output(graph)
      })
  ).toThrow(/requires the right operand/);
  expect(
    () =>
      new GPUPolygonOverlay({
        left: polygon(graph),
        right: polygon(graph),
        operation: 'dissolve',
        capacity: {intersections: 8, nodedSegments: 16},
        vertexTolerance: 1e-5,
        output: output(graph)
      })
  ).toThrow(/accepts only the left operand/);
  const mismatched = output(graph);
  mismatched.sourceIds = view(graph, 'uint32', 16);
  expect(
    () =>
      new GPUPolygonOverlay({
        left: polygon(graph),
        operation: 'dissolve',
        capacity: {intersections: 8, nodedSegments: 16},
        vertexTolerance: 1e-5,
        output: mismatched
      })
  ).toThrow(/must be the same view/);
});

it('GPUBufferSurface composes joins, line caps and topology repair', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer'});
  const lineBuffer = new GPUBufferSurface({
    geometry: {
      kind: 'lines',
      positions: view(graph, 'float32x2', 6),
      lineOffsets: view(graph, 'uint32', 3)
    },
    parameters: view(graph, 'float32', 4),
    joinStyle: 'round',
    capStyle: 'square',
    quadSegments: 4,
    vertexTolerance: 1e-5,
    capacity: {intersections: 64, nodedSegments: 128},
    output: output(graph, 128, 32)
  });
  const lineIds = lineBuffer.getCommandNodes(graph).map(node => node.id);
  expect(lineIds).toContain('buffer-surface-left-offset-generate');
  expect(lineIds).toContain('buffer-surface-right-offset-generate');
  expect(lineIds).toContain('buffer-surface-line-boundaries');
  expect(lineIds).toContain('buffer-surface-repair-classify-boundary');

  const polygonGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'polygon-buffer'});
  const polygonBuffer = new GPUBufferSurface({
    geometry: polygon(polygonGraph),
    parameters: view(polygonGraph, 'float32', 4),
    joinStyle: 'mitre',
    vertexTolerance: 1e-5,
    capacity: {intersections: 64, nodedSegments: 128},
    output: output(polygonGraph, 128, 32)
  });
  const polygonIds = polygonBuffer.getCommandNodes(polygonGraph).map(node => node.id);
  expect(polygonIds).toContain('buffer-surface-orient-orientation');
  expect(polygonIds).toContain('buffer-surface-offset-generate');
  expect(polygonIds.some(id => id.startsWith('buffer-surface-repair-assemble-'))).toBe(true);
});
