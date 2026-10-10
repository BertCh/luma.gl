// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

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
        invalidCount: scalar(graph)
      },
      duplicateCount: scalar(graph)
    }
  });
  const delaunayNodeIds = delaunay.getCommandNodes(graph).map(node => node.id);
  expect(delaunayNodeIds[0]).toBe('delaunay-tessellation-initialize-status');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-site-0-cavity');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-site-7-rebuild');
  expect(delaunayNodeIds).toContain('delaunay-tessellation-compact-output-scatter');
  expect(delaunayNodeIds.at(-1)).toBe('delaunay-tessellation-publish-status');
  expect(delaunayNodeIds).not.toContain('delaunay-tessellation-bowyer-watson');

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
  expect(voronoiNodeIds).toContain('voronoi-diagram-compact-edges-scatter');
  expect(voronoiNodeIds).toContain('voronoi-diagram-materialize');
  expect(voronoiNodeIds.at(-1)).toBe('voronoi-diagram-publish-status');
});
