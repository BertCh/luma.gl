// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURegionPartitionEvaluation,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function setup(rows: number, capacity: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'regionalization-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity)
  };
  return {graph, view, weights};
}

it('GPUSpatialWeightsMinimumSpanningTree validates props and declares nodes', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    weights,
    values: view('float32', 12),
    columnCount: 2,
    treeEdgeFlags: view('uint32', 16),
    componentLabels: view('uint32', 6)
  };
  expect(() => new GPUSpatialWeightsMinimumSpanningTree({...base, columnCount: 3})).toThrow(
    /values length/
  );
  expect(
    () => new GPUSpatialWeightsMinimumSpanningTree({...base, treeEdgeFlags: view('uint32', 15)})
  ).toThrow(/treeEdgeFlags length/);
  expect(
    () =>
      new GPUSpatialWeightsMinimumSpanningTree({
        ...base,
        standardize: false,
        standardizedValues: view('float32', 12)
      })
  ).toThrow(/requires standardize/);
  const nodes = new GPUSpatialWeightsMinimumSpanningTree({
    ...base,
    edges: {ids: view('uint32', 5), count: view('uint32', 1), overflow: view('uint32', 1)},
    edgeEndpoints: view('uint32', 10),
    edgeCosts: view('float32', 5)
  }).getCommandNodes(graph);
  expect(nodes[0].id).toBe('minimum-spanning-tree-standardize');
  // ceil(log2(6)) = 3 rounds.
  expect(nodes.filter(node => node.id.endsWith('-hook'))).toHaveLength(3);
  expect(nodes.at(-1)?.id).toBe('minimum-spanning-tree-edge-details');
});

it('GPUSkaterRegions validates props and declares one step per allowed cut', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    weights,
    treeEdgeFlags: view('uint32', 16),
    componentLabels: view('uint32', 6),
    values: view('float32', 12),
    columnCount: 2,
    maximumRegionCount: 4,
    parameters: view('uint32', 2),
    labels: view('uint32', 6)
  };
  expect(() => new GPUSkaterRegions({...base, maximumRegionCount: 1})).toThrow(
    /maximumRegionCount/
  );
  expect(() => new GPUSkaterRegions({...base, parameters: view('uint32', 1)})).toThrow(
    /parameters/
  );
  expect(() => new GPUSkaterRegions({...base, cutEdges: view('uint32', 2)})).toThrow(/cutEdges/);
  // Every kernel stays within the 8 storage bindings the null device enforces.
  const nodes = new GPUSkaterRegions({
    ...base,
    regionCount: view('uint32', 1),
    cutEdges: view('uint32', 3),
    cutGains: view('float32', 3)
  }).getCommandNodes(graph);
  expect(nodes.filter(node => node.id.endsWith('-candidates'))).toHaveLength(3);
});

it('GPURegionPartitionEvaluation validates props and declares nodes', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    values: view('float32', 12),
    columnCount: 2,
    labels: view('uint32', 6),
    summary: view('float32', 8)
  };
  expect(() => new GPURegionPartitionEvaluation({...base, summary: view('float32', 4)})).toThrow(
    /summary/
  );
  expect(() => new GPURegionPartitionEvaluation({...base, labelCapacity: 0})).toThrow(
    /labelCapacity/
  );
  const nodes = new GPURegionPartitionEvaluation({...base, weights}).getCommandNodes(graph);
  // The stable label sort contributes its own nodes between the key and region kernels.
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('region-partition-evaluation-sort-keys');
  expect(ids.filter(id => !id.startsWith('region-partition-evaluation-sort-'))).toEqual([
    'region-partition-evaluation-regions',
    'region-partition-evaluation-links',
    'region-partition-evaluation-row-tiles',
    'region-partition-evaluation-label-tiles',
    'region-partition-evaluation-means',
    'region-partition-evaluation-deviation-tiles',
    'region-partition-evaluation-summary'
  ]);
});
