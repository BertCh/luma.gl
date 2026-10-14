// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GPUFlagOffsets,
  type GPUCommandNode
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../../src/utils/sorted-segment-sums';
import {
  createHierarchicalDensityPointerInitNode,
  createHierarchicalDensityClusterInfoNode,
  createHierarchicalDensityContributionsNode,
  createHierarchicalDensityLabelsNode,
  type HierarchicalDensityShape
} from '../../../src/gpu-spatial-analysis/spatial-clustering/hierarchical-density-clustering-kernels';
import {
  getHierarchicalDensityDendrogramNodes,
  getHierarchicalDensityCondensingNodes,
  getHierarchicalDensitySelectionNodes
} from '../../../src/gpu-spatial-analysis/spatial-clustering/hierarchical-density-tree';
import {
  createHierarchicalDensityDendrogramInitNode,
  createHierarchicalDensityDendrogramNode,
  createHierarchicalDensitySelectionNode
} from './hierarchical-density-tree-legacy';
import type {DensityTreeEdge} from './hierarchical-density-tree-model';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/**
 * Same-run old/new tree-stage harness. The input MST is an oracle fixture, never a production
 * CPU fallback. Everything after that input (including stability and compact labels) runs on GPU.
 * Re-encodes the compiled graph to test that all cursors, gates and transient atomics reset.
 */
export async function runDensityTreeStages(
  device: Device,
  rows: number,
  sortedEdges: readonly DensityTreeEdge[],
  minimumClusterSize: number,
  mode: 'serial' | 'parallel',
  timedEncodings = 0
) {
  const graph = new GPUCommandGraph(device, {id: `tree-${mode}`});
  const buffers: Buffer[] = [];
  const input = (name: string, data: Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', data.length);
  };
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const words = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', Math.max(1, length));
  const floats = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', Math.max(1, length));
  const clusterCapacity = Math.floor((2 * rows) / minimumClusterSize) + 1;
  const shape: HierarchicalDensityShape = {
    rowCount: rows,
    minimumSamples: 1,
    minimumClusterSize,
    clusterCapacity,
    leafCapacity: 1,
    internalNodeCount: 0,
    levelCount: 1
  };
  const edgeWords = new Uint32Array(rows * 3);
  const edgeFloats = new Float32Array(edgeWords.buffer);
  sortedEdges.forEach(([first, second, weight], rank) => {
    edgeWords[rank * 3] = first;
    edgeWords[rank * 3 + 1] = second;
    edgeFloats[rank * 3 + 2] = weight;
  });
  const state = input('state', new Uint32Array([rows, sortedEdges.length]));
  const edges = input('edges', edgeWords);
  const edgeOrder = input(
    'order',
    Uint32Array.from({length: rows}, (_, index) => index)
  );
  const dendrogram = output('dendrogram', rows * 10);
  const labels = output('labels', rows);
  const selectedCount = output('selected-count', 1);
  const nodes: GPUCommandNode[] = [];
  if (mode === 'serial') {
    const kruskal = words('kruskal', rows * 2);
    nodes.push(
      createHierarchicalDensityDendrogramInitNode(graph, {
        id: 'dendrogram-init',
        shape,
        dendrogram: dendrogram.view,
        kruskal
      }),
      createHierarchicalDensityDendrogramNode(graph, {
        id: 'dendrogram',
        shape,
        state,
        edges,
        edgeOrder,
        kruskal,
        dendrogram: dendrogram.view
      })
    );
  } else {
    nodes.push(
      ...getHierarchicalDensityDendrogramNodes(graph, {
        id: 'dendrogram',
        shape,
        state,
        edges,
        edgeOrder,
        dendrogram: dendrogram.view
      })
    );
  }
  const pointers = words('pointers', rows * 4);
  const startFlags = words('start-flags', rows * 2);
  const clusterOffsets = words('cluster-offsets', rows * 2);
  const clusterCount = words('cluster-count', 1);
  const clusterInfo = words('cluster-info', clusterCapacity * 3);
  const contributionKeys = words('contribution-keys', rows * 2);
  const contributions = floats('contributions', rows * 2);
  const lowClusters = words('low-clusters', rows);
  const stability = floats('stability', clusterCapacity);
  const selection = words('selection', clusterCapacity * 4);
  nodes.push(
    createHierarchicalDensityPointerInitNode(graph, {
      id: 'pointer-init',
      shape,
      dendrogram: dendrogram.view,
      pointers,
      startFlags
    }),
    ...getHierarchicalDensityCondensingNodes(graph, {id: 'condensing', rowCount: rows, pointers}),
    ...new GPUFlagOffsets({
      id: 'cluster-offsets',
      flags: startFlags,
      offsets: clusterOffsets,
      count: clusterCount
    }).getCommandNodes(graph),
    createHierarchicalDensityClusterInfoNode(graph, {
      id: 'cluster-info',
      shape,
      dendrogram: dendrogram.view,
      pointers,
      startFlags,
      clusterOffsets,
      clusterInfo
    }),
    createHierarchicalDensityContributionsNode(graph, {
      id: 'contributions',
      shape,
      dendrogram: dendrogram.view,
      pointers,
      startFlags,
      clusterOffsets,
      clusterInfo,
      contributionKeys,
      contributions,
      lowClusters
    }),
    ...getSortedSegmentSumNodes(graph, {
      id: 'stability',
      operation: 'DensityTreeTest',
      segmentCount: clusterCapacity,
      segmentKeys: contributionKeys,
      reductions: [{name: 'stability', contributions, output: stability}]
    })
  );
  const selectionProps = {
    id: 'selection',
    shape,
    clusterCount,
    stability,
    clusterInfo,
    selection,
    selectedCount: selectedCount.view
  };
  nodes.push(
    ...(mode === 'serial'
      ? [createHierarchicalDensitySelectionNode(graph, selectionProps)]
      : getHierarchicalDensitySelectionNodes(graph, selectionProps))
  );
  nodes.push(
    createHierarchicalDensityLabelsNode(graph, {
      id: 'labels',
      shape,
      lowClusters,
      selection,
      labels: labels.view
    })
  );
  for (const node of nodes) graph.add(node);
  const compiled = await graph.compileAsync();
  try {
    submitGraph(device, compiled, undefined);
    const firstLabels = await readUint32(labels.buffer, rows);
    const times: number[] = [];
    for (let encoding = 0; encoding < Math.max(timedEncodings, 1); encoding++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(selectedCount.buffer, 1);
      times.push(performance.now() - start);
    }
    times.sort((first, second) => first - second);
    return {
      dendrogram: await readUint32(dendrogram.buffer, rows * 10),
      labels: await readUint32(labels.buffer, rows),
      firstLabels,
      count: (await readUint32(selectedCount.buffer, 1))[0],
      medianMilliseconds: times[Math.floor(times.length / 2)]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}
