// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GPUGridIndexView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  decodeGPURegionStatistics,
  GPU_REGION_STATISTICS_FLAGS,
  GPURegionStatistics,
  type GPURegionStatisticsProps
} from '../../../src/map-graphs/region-statistics';
import {createNullWebGPUDevice, createVectorView} from '../map-graph-test-utils';

function createIndex(graph: GPUCommandGraph, dimension: 2 | 3 = 2, prefix = ''): GPUGridIndexView {
  const gridSize = dimension === 2 ? ([4, 4] as const) : ([4, 4, 4] as const);
  const cellCount = gridSize.reduce((product, size) => product * size, 1);
  return {
    gridSize,
    bounds: dimension === 2 ? [0, 0, 10, 10] : [0, 0, 0, 10, 10, 10],
    cellOffsets: createTransientView(graph, `${prefix}cell-offsets`, 'uint32', cellCount + 1),
    objectIds: createTransientView(graph, `${prefix}object-ids`, 'uint32', 12),
    count: createTransientView(graph, `${prefix}index-count`, 'uint32', 1),
    overflow: createTransientView(graph, `${prefix}index-overflow`, 'uint32', 1)
  };
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURegionStatisticsProps> = {}
): GPURegionStatisticsProps {
  return {
    selection: {kind: 'rectangle', bounds: createTransientView(graph, 'bounds', 'float32', 4)},
    positions: createTransientView(graph, 'positions', 'float32x2', 12),
    values: createTransientView(graph, 'values', 'float32', 12),
    histogram: {binCount: 4},
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 12),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    },
    summary: createTransientView(graph, 'summary', 'uint32', 12),
    spatialIndex: overrides.spatialIndex ?? {
      kind: 'grid',
      index: createIndex(graph),
      candidateCapacity: 8
    },
    ...overrides
  };
}

it('GPURegionStatistics with spatialIndex orders query, gather, predicate, statistics, and output', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const ids = new GPURegionStatistics({...createProps(graph), id: 'stats'})
    .getCommandNodes(graph)
    .map(node => node.id);
  const order = [
    'stats-cell-ranges',
    'stats-cell-gather',
    'stats-gather',
    'stats-filter',
    'stats-mask-clear',
    'stats-mask-scatter',
    'stats-counts-clear',
    'stats-candidate-status',
    'stats-count',
    'stats-sum',
    'stats-histogram',
    'stats-visibility',
    'stats-selection-finalize'
  ].map(prefix => ids.findIndex(id => id.startsWith(prefix)));
  expect(order.every(index => index >= 0)).toBe(true);
  expect(ids.at(-1)).toBe('stats-summary');
  expect(new Set(ids).size).toBe(ids.length);
  // Without output or outputMask the O(rowCount) mask nodes are skipped.
  const noOutputGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const noOutputIds = new GPURegionStatistics({
    ...createProps(noOutputGraph, {output: undefined}),
    id: 'stats'
  })
    .getCommandNodes(noOutputGraph)
    .map(node => node.id);
  expect(noOutputIds.some(id => id.startsWith('stats-mask-'))).toBe(false);
  expect(noOutputIds.some(id => id.startsWith('stats-visibility'))).toBe(false);
});

it('GPURegionStatistics with spatialIndex compiles polygon bounds and radius queries', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const polygonIds = new GPURegionStatistics({
    ...createProps(graph),
    id: 'stats',
    selection: {
      kind: 'polygon',
      vertices: createTransientView(graph, 'vertices', 'float32x2', 8),
      vertexCount: createTransientView(graph, 'vertex-count', 'uint32', 1)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(polygonIds).toContain('stats-polygon-bounds-kernel');
  expect(polygonIds).toContain('stats-region');
  const radiusGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const radiusIds = new GPURegionStatistics({
    ...createProps(radiusGraph),
    id: 'stats',
    selection: {kind: 'radius', circle: createTransientView(radiusGraph, 'circle', 'float32', 3)}
  })
    .getCommandNodes(radiusGraph)
    .map(node => node.id);
  expect(radiusIds.some(id => id.startsWith('stats-filter'))).toBe(true);
});

it('GPURegionStatistics rejects unsupported spatialIndex combinations', () => {
  const device = createNullWebGPUDevice();
  let graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPURegionStatistics(
        createProps(graph, {
          selection: {
            kind: 'rectangle',
            bounds: createTransientView(graph, 'screen-bounds', 'float32', 4),
            screenTransform: createTransientView(graph, 'transform', 'float32', 20)
          }
        })
      )
  ).toThrow(/screenTransform/);

  graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPURegionStatistics(
        createProps(graph, {
          selection: {kind: 'pick-region', result: createTransientView(graph, 'pick', 'uint32', 12)}
        })
      )
  ).toThrow(/spatialIndex supports/);

  graph = new GPUCommandGraph(device);
  const chunks = [
    createTransientView(graph, 'chunk-0', 'float32x2', 6),
    createTransientView(graph, 'chunk-1', 'float32x2', 6)
  ];
  expect(
    () =>
      new GPURegionStatistics(
        createProps(graph, {positions: createVectorView('positions', 'float32x2', chunks)})
      )
  ).toThrow(/packed \(non-vector\) positions/);

  graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPURegionStatistics(
        createProps(graph, {
          spatialIndex: {kind: 'grid', index: createIndex(graph, 3, 'three-'), candidateCapacity: 8}
        })
      )
  ).toThrow(/two-dimensional/);

  graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPURegionStatistics(
        createProps(graph, {
          spatialIndex: {kind: 'grid', index: createIndex(graph), candidateCapacity: 0}
        })
      )
  ).toThrow(/candidateCapacity/);
  device.destroy();
});

it('decodeGPURegionStatistics decodes candidatesTruncated', () => {
  const words = new Uint32Array(8);
  words[7] = GPU_REGION_STATISTICS_FLAGS.candidatesTruncated;
  expect(GPU_REGION_STATISTICS_FLAGS.candidatesTruncated).toBe(4);
  expect(decodeGPURegionStatistics(words)).toMatchObject({
    candidatesTruncated: true,
    selectionTruncated: false,
    regionTruncated: false
  });
  words[7] = 3;
  expect(decodeGPURegionStatistics(words)).toMatchObject({
    candidatesTruncated: false,
    selectionTruncated: true,
    regionTruncated: true
  });
});
