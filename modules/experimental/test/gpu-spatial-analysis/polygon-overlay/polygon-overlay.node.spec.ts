// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPUBufferSurface,
  GPUPolygonOverlay,
  type GPUPolygonOverlayOutput
} from '../../../src/gpu-spatial-analysis';
import {
  createPointerJumpNodes,
  getPointerJumpRoundCount
} from '../../../src/gpu-spatial-analysis/polygon-overlay/overlay-face-labelling-wgsl';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  countDistinctPipelines,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getPrivateMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {getUndeclaredCalls} from '../segment-intersection/wgsl-call-sites';

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
  expect(ids).not.toContain('polygon-overlay-assemble-cancel');
  expect(ids).not.toContain('polygon-overlay-publish-required');
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
  // The output has no top-level `sourceIds`: `geometry.sourceIds` is the only source-ID view, so only
  // its length (one entry per ring) is checked.
  const mismatched = output(graph);
  mismatched.geometry.sourceIds = view(graph, 'uint32', 15);
  expect(
    () =>
      new GPUPolygonOverlay({
        left: polygon(graph),
        operation: 'dissolve',
        capacity: {intersections: 8, nodedSegments: 16},
        vertexTolerance: 1e-5,
        output: mismatched
      })
  ).toThrow(/share one ring capacity/);
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
  expect(polygonIds).toContain('buffer-surface-stroke-left-offset-generate');
  expect(polygonIds.some(id => id.startsWith('buffer-surface-repair-assemble-'))).toBe(true);
});

function createOverlay(
  nodedSegments: number,
  operation: 'intersection' | 'dissolve' = 'intersection'
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `overlay-${nodedSegments}`});
  const contributor = new GPUPolygonOverlay({
    left: polygon(graph),
    right: operation === 'dissolve' ? undefined : polygon(graph),
    operation,
    capacity: {intersections: 32, nodedSegments},
    vertexTolerance: 0.25,
    output: output(graph)
  });
  return {graph, nodes: contributor.getCommandNodes(graph)};
}

it('GPUPolygonOverlay labels faces by ray shoots and pointer jumping, without ring membership', () => {
  const {nodes} = createOverlay(64);
  const ids = nodes.map(node => node.id);
  const order = [
    'polygon-overlay-ring-polygons',
    'polygon-overlay-piece-canonical',
    'polygon-overlay-piece-keys',
    'polygon-overlay-sort-gather',
    'polygon-overlay-group-flags',
    'polygon-overlay-group-heads',
    'polygon-overlay-ray-shoot-1',
    'polygon-overlay-group-coverage',
    'polygon-overlay-ray-shoot-2',
    'polygon-overlay-classify-boundary',
    'polygon-overlay-emit-boundary',
    'polygon-overlay-lattice-to-world',
    'polygon-overlay-publish'
  ].map(id => ids.indexOf(id));
  expect(
    order.every(index => index >= 0),
    JSON.stringify(order)
  ).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  // Ray shoots and pointer jumping after the sort; no per-ring membership or source-segment pass.
  expect(ids.some(id => /membership|ring-bounds|source-segments/.test(id))).toBe(false);
  const rounds = getPointerJumpRoundCount(64);
  expect(ids.filter(id => id.startsWith('polygon-overlay-jump-1-round-'))).toHaveLength(rounds);
  expect(ids.filter(id => id.startsWith('polygon-overlay-jump-2-round-'))).toHaveLength(rounds);
});

it('GPUPolygonOverlay kernels parse, call only declared functions and respect core limits', () => {
  for (const operation of ['intersection', 'dissolve'] as const) {
    const {nodes} = createOverlay(64, operation);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      const {source} = kernel;
      expect(() => new WgslReflect(source), kernel.id).not.toThrow();
      // workgroupUniformLoad is a WGSL builtin the helper does not know.
      expect(
        getUndeclaredCalls(source).filter(name => name !== 'workgroupUniformLoad'),
        `${kernel.id} undeclared calls`
      ).toEqual([]);
      expect(
        source.match(/var<storage/g)?.length ?? 0,
        `${kernel.id} bindings`
      ).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), `${kernel.id} private memory`).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
      // Reserved WGSL words that wgsl_reflect accepts (the noding kernels have their own audit).
      if (!kernel.id.includes('-noding-')) {
        expect(source, kernel.id).not.toMatch(/\b(?:target|from|sample|filter)\s*[:=)]/);
      }
    }
  }
});

it('GPUPolygonOverlay pointer-jump rounds share pipelines whatever the round count (library rule 12)', () => {
  // The jump kernels depend on the group capacity and the component count only: every round of a
  // level uses one pipeline, the two levels (one and two components) two, plus their initializers.
  const jumpPipelines = (nodedSegments: number) => {
    const {nodes} = createOverlay(nodedSegments);
    return countDistinctPipelines(
      nodes.filter(node => node.id.startsWith('polygon-overlay-jump-'))
    );
  };
  expect(getPointerJumpRoundCount(64)).not.toBe(getPointerJumpRoundCount(128));
  // Per level: the initializer and the round kernel.
  expect(jumpPipelines(64)).toBe(4);
  expect(jumpPipelines(128)).toBe(4);
  // Direct: the same views, three rounds and four rounds.
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'jump-rounds'});
  const build = (rounds: number) =>
    createPointerJumpNodes(graph, {
      id: `jump-${rounds}`,
      operation: 'GPUPolygonOverlay',
      parents: createTransientView(graph, `parents-${rounds}`, 'uint32', 32),
      values: createTransientView(graph, `values-${rounds}`, 'uint32', 32),
      componentCount: 1,
      rounds
    }).nodes;
  const three = build(3);
  const four = build(4);
  expect(three).toHaveLength(3);
  expect(four).toHaveLength(4);
  expect(countDistinctPipelines(three)).toBe(1);
  expect(countDistinctPipelines(four)).toBe(1);
});

it('GPUPolygonOverlay bounds hot pixels by a quarter of the segment table', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'overlay-limit'});
  expect(() =>
    new GPUPolygonOverlay({
      left: polygon(graph),
      right: polygon(graph),
      operation: 'union',
      capacity: {intersections: 1048576, nodedSegments: 64},
      vertexTolerance: 0.25,
      output: output(graph)
    }).getCommandNodes(graph)
  ).toThrow(/positions rows \+ rings \+ capacity\.intersections must be at most 1048576/);
});

it('GPUPolygonOverlay builds with the smallest capacities', () => {
  for (const nodedSegments of [1, 2, 3]) {
    const {nodes} = createOverlay(nodedSegments);
    expect(nodes.length).toBeGreaterThan(40);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  }
});
