// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation';
import {
  getDelaunayStepCount,
  getDelaunayParallelStepCount
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {getInlinedCallSites, getUndeclaredCalls} from '../segment-intersection/wgsl-call-sites';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {countDistinctPipelines} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function scalar(graph: GPUCommandGraph) {
  return createTransientView(graph, `tessellation-scalar-${serial++}`, 'uint32', 1);
}

it('tessellation contributors stay within minimum WebGPU storage-binding limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'tessellation-binding-limit'});
  const positions = createTransientView(graph, 'tessellation-positions', 'float32x2', 8);
  const triangles = createTransientView(graph, 'tessellation-triangles', 'uint32x3', 16);
  const triangleCount = scalar(graph);

  const delaunay = new GPUDelaunayTessellation({
    positions,
    output: {
      triangles,
      status: {
        count: scalar(graph),
        requiredCount: triangleCount,
        overflow: scalar(graph),
        invalidCount: scalar(graph),
        duplicateCount: scalar(graph),
        converged: scalar(graph),
        iterationCount: scalar(graph),
        iterationLimitReached: scalar(graph)
      }
    }
  });
  const delaunayNodeIds = delaunay.getCommandNodes(graph).map(node => node.id);
  const stepCount = getDelaunayParallelStepCount(8);
  expect(delaunayNodeIds[0]).toBe('delaunay-tessellation-initialize-status');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-classify-sites');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-flag-duplicate-sites');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-initialize-triangles');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-initialize-sites');
  // Parallel insertion and flipping: nine nodes per step, no per-site nodes (P3 #1).
  const stepNodes = delaunayNodeIds.filter(nodeId =>
    nodeId.startsWith('delaunay-tessellation-step-0-')
  );
  expect(stepNodes).toEqual([
    'delaunay-tessellation-step-0-gate',
    'delaunay-tessellation-step-0-insert-vote',
    'delaunay-tessellation-step-0-insert-split',
    'delaunay-tessellation-step-0-insertion-repair-adjacency',
    'delaunay-tessellation-step-0-insertion-relocate',
    'delaunay-tessellation-step-0-flip-evaluate',
    'delaunay-tessellation-step-0-flip-apply',
    'delaunay-tessellation-step-0-flip-repair-adjacency',
    'delaunay-tessellation-step-0-flip-relocate'
  ]);
  expect(delaunayNodeIds.filter(nodeId => nodeId.includes('-step-')).length).toBe(9 * stepCount);
  expect(delaunayNodeIds).toContain(`delaunay-tessellation-step-${stepCount - 1}-flip-relocate`);
  expect(delaunayNodeIds).not.toContain(`delaunay-tessellation-step-${stepCount}-gate`);
  expect(delaunayNodeIds.some(nodeId => nodeId.includes('-site-0-'))).toBe(false);
  expect(delaunayNodeIds).toContain('delaunay-tessellation-finalize');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-classify-output');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-materialize');
  expect(delaunayNodeIds.at(-1)).toBe('delaunay-tessellation-publish-status');

  const voronoi = new GPUVoronoiDiagram({
    positions,
    triangles,
    triangleCount,
    clipBounds: createTransientView(graph, 'voronoi-clip-bounds', 'float32x4', 1),
    output: {
      segments: createTransientView(graph, 'voronoi-segments', 'float32x4', 48),
      siteIds: createTransientView(graph, 'voronoi-site-ids', 'uint32x2', 48),
      status: {
        count: scalar(graph),
        requiredCount: scalar(graph),
        overflow: scalar(graph),
        invalidCount: scalar(graph)
      }
    }
  });
  const voronoiNodeIds = voronoi.getCommandNodes(graph).map(node => node.id);
  expect(voronoiNodeIds[0]).toBe('voronoi-diagram-initialize');
  expect(voronoiNodeIds).toContain('voronoi-diagram-circumcenters');
  expect(voronoiNodeIds).toContain('voronoi-diagram-edge-keys');
  expect(voronoiNodeIds).toContain('voronoi-diagram-classify-edges');
  expect(voronoiNodeIds).toContain('voronoi-diagram-compact-edges-scatter');
  expect(voronoiNodeIds).toContain('voronoi-diagram-materialize');
  expect(voronoiNodeIds.at(-1)).toBe('voronoi-diagram-publish-status');
});

it('GPUDelaunayTessellation kernels parse, keep one call site per exact helper, and ignore the step', () => {
  const buildNodes = (pointCount: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `delaunay-${pointCount}`});
    return new GPUDelaunayTessellation({
      positions: createTransientView(graph, `positions-${pointCount}`, 'float32x2', pointCount),
      output: {
        triangles: createTransientView(
          graph,
          `triangles-${pointCount}`,
          'uint32x3',
          4 * pointCount
        ),
        status: {
          count: scalar(graph),
          requiredCount: scalar(graph),
          overflow: scalar(graph),
          invalidCount: scalar(graph),
          duplicateCount: scalar(graph),
          converged: scalar(graph),
          iterationCount: scalar(graph),
          iterationLimitReached: scalar(graph)
        }
      }
    }).getCommandNodes(graph);
  };
  for (const {id, source} of getGPUCommandNodeKernelSources(buildNodes(40))) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(source, id).not.toContain('isFiniteCoordinate');
    expect(source.split('fn isFiniteBits(').length - 1, id).toBeLessThanOrEqual(1);
    // Core-device limit: at most eight storage buffers per kernel.
    expect(source.match(/var<storage/g)?.length ?? 0, id).toBeLessThanOrEqual(8);
    // wgsl_reflect does not resolve calls: every called helper must be declared.
    expect(getUndeclaredCalls(source), id).toEqual([]);
    // One inlined copy of each big exact helper per kernel (the 2.5x trap).
    for (const target of ['orientSign', 'incircleExact']) {
      for (const [name, count] of getInlinedCallSites(source, target)) {
        expect(count, `${id} call sites of ${name}`).toBeLessThanOrEqual(1);
      }
    }
  }
  const flipSource = getGPUCommandNodeKernelSources(buildNodes(6)).find(kernel =>
    kernel.id.endsWith('step-0-flip-evaluate')
  );
  expect(flipSource?.source).toContain('fn incircleExact(');
  // Library rule 12: the phase id is a GPU state word, so every step shares its pipelines. Two
  // sizes with different step counts build the same pipeline set.
  expect(getDelaunayStepCount(6)).not.toBe(getDelaunayStepCount(300));
  expect(countDistinctPipelines(buildNodes(6))).toBe(countDistinctPipelines(buildNodes(7)));
  const stepPipelines = (pointCount: number) =>
    new Set(
      getGPUCommandNodeKernelSources(buildNodes(pointCount))
        .filter(kernel => kernel.id.includes('-step-'))
        .map(kernel => `${kernel.source}|${JSON.stringify(kernel.constants)}`)
    ).size;
  expect(stepPipelines(6)).toBe(9);
  expect(stepPipelines(300)).toBe(9);
  const nodes = buildNodes(300);
  const iterationNodes = nodes.filter(node => /-step-[01]-/.test(node.id));
  const pipelines = (rounds: number) =>
    new Set(
      getGPUCommandNodeKernelSources(iterationNodes.slice(0, 9 * rounds)).map(
        kernel => `${kernel.source}|${JSON.stringify(kernel.constants)}`
      )
    );
  expect(pipelines(1)).toEqual(pipelines(2));
  for (const node of nodes.filter(
    node => node.id.includes('-step-') && !node.id.endsWith('-gate')
  )) {
    expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
  const smallNodes = buildNodes(1000);
  expect(smallNodes.filter(node => node.id.includes('-step-'))).toHaveLength(9 * 88);
  expect(smallNodes.filter(node => node.id.includes('-continue-'))).toHaveLength(2);
  expect(getDelaunayParallelStepCount(2048)).toBeLessThan(110);
  expect(getDelaunayParallelStepCount(2049)).toBe(4096);
  expect(getDelaunayStepCount(2000)).toBe(4032);
  expect(getDelaunayStepCount(100000)).toBe(4096);
});
