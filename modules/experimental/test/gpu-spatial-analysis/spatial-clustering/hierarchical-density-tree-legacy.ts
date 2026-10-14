// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Frozen pre-parallelization GPU kernels, used only by the same-run parity / A/B spec.
// The cluster record stride follows the current private layout (birth, parent, point count).
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import type {HierarchicalDensityShape} from '../../../src/gpu-spatial-analysis/spatial-clustering/hierarchical-density-clustering-kernels';
const OPERATION = 'HierarchicalDensitySerialReference';
function getConstants(shape: HierarchicalDensityShape): string {
  return `const ROW_COUNT: u32 = ${shape.rowCount}u;
const CLUSTER_CAPACITY: u32 = ${shape.clusterCapacity}u;
const DENDROGRAM_WORDS: u32 = 5u;
const EDGE_WORDS: u32 = 3u;
const CLUSTER_WORDS: u32 = 3u;
const SELECTION_WORDS: u32 = 4u;
const NONE: u32 = 0xffffffffu;`;
}
/** Dendrogram leaves (one per point, size 1) and an empty Kruskal union-find. @internal */
export function createHierarchicalDensityDendrogramInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    kruskal: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'dendrogram-init',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read_write'},
      {name: 'kruskal', view: props.kruskal, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: getConstants(props.shape),
    body: `let base = dendrogramOffset + index * DENDROGRAM_WORDS;
  dendrogram[base] = NONE;
  dendrogram[base + 1u] = select(0u, 1u, index < ROW_COUNT);
  dendrogram[base + 2u] = 0u;
  dendrogram[base + 3u] = NONE;
  dendrogram[base + 4u] = NONE;
  if (index < ROW_COUNT) {
    kruskal[kruskalOffset + index * 2u] = index;
    kruskal[kruskalOffset + index * 2u + 1u] = index;
  }`
  });
}

/**
 * Single-linkage dendrogram (the Kruskal reconstruction tree) by union-find over the sorted
 * spanning-tree edges, in one invocation: edge `k` creates node `ROW_COUNT + k` whose children are
 * the current top nodes of the two components. This is the one sequential stage: `O(n alpha(n))`
 * single-thread work (path halving, union by size), which stays far below the CPU reference's
 * cost; a parallel construction is a follow-up. @internal
 */
export function createHierarchicalDensityDendrogramNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    state: GraphDataView<'uint32'>;
    edges: GraphDataView<'uint32'>;
    edgeOrder: GraphDataView<'uint32'>;
    kruskal: GraphDataView<'uint32'>;
    dendrogram: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'dendrogram',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
      {name: 'edgeOrder', view: props.edgeOrder, type: 'u32', access: 'read'},
      {name: 'kruskal', view: props.kruskal, type: 'u32', access: 'read_write'},
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: `${getConstants(props.shape)}
fn findSet(start: u32) -> u32 {
  var current = start;
  loop {
    let parent = kruskal[kruskalOffset + current * 2u];
    if (parent == current) { return current; }
    let grandparent = kruskal[kruskalOffset + parent * 2u];
    kruskal[kruskalOffset + current * 2u] = grandparent;
    current = grandparent;
  }
}`,
    body: `let edgeCount = min(state[stateOffset + 1u], ROW_COUNT - 1u);
  for (var rank = 0u; rank < edgeCount; rank++) {
    let edge = edgesOffset + edgeOrder[edgeOrderOffset + rank] * EDGE_WORDS;
    let first = findSet(edges[edge]);
    let second = findSet(edges[edge + 1u]);
    let node = ROW_COUNT + rank;
    let firstTop = kruskal[kruskalOffset + first * 2u + 1u];
    let secondTop = kruskal[kruskalOffset + second * 2u + 1u];
    let firstSize = dendrogram[dendrogramOffset + firstTop * DENDROGRAM_WORDS + 1u];
    let secondSize = dendrogram[dendrogramOffset + secondTop * DENDROGRAM_WORDS + 1u];
    let base = dendrogramOffset + node * DENDROGRAM_WORDS;
    dendrogram[base + 1u] = firstSize + secondSize;
    dendrogram[base + 2u] = edges[edge + 2u];
    dendrogram[base + 3u] = firstTop;
    dendrogram[base + 4u] = secondTop;
    dendrogram[dendrogramOffset + firstTop * DENDROGRAM_WORDS] = node;
    dendrogram[dendrogramOffset + secondTop * DENDROGRAM_WORDS] = node;
    // Union by size: the larger set keeps its root.
    var root = first;
    var child = second;
    if (firstSize < secondSize) {
      root = second;
      child = first;
    }
    kruskal[kruskalOffset + child * 2u] = root;
    kruskal[kruskalOffset + root * 2u + 1u] = node;
  }`
  });
}

/**
 * Excess-of-mass selection (scikit-learn `cluster_selection_method='eom'`,
 * `allow_single_cluster=False`) in one invocation over the condensed clusters. Cluster IDs follow
 * dendrogram node order, so children come before parents: ascending, each non-root cluster is
 * selected unless its children's best total stability is strictly larger, and passes its best up;
 * descending, a cluster under a selected one is deselected and every cluster records its selected
 * representative; ascending again, selected clusters get compact labels. Cost is linear in the
 * cluster count (at most `2 rows / minimumClusterSize + 1`). @internal
 */
export function createHierarchicalDensitySelectionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    clusterCount: GraphDataView<'uint32'>;
    stability: GraphDataView<'float32'>;
    clusterInfo: GraphDataView<'uint32'>;
    selection: GraphDataView<'uint32'>;
    selectedCount: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'selection',
    bindings: [
      {name: 'clusterCount', view: props.clusterCount, type: 'u32', access: 'read'},
      {name: 'stability', view: props.stability, type: 'f32', access: 'read'},
      {name: 'clusterInfo', view: props.clusterInfo, type: 'u32', access: 'read'},
      {name: 'selection', view: props.selection, type: 'u32', access: 'read_write'},
      {name: 'selectedCount', view: props.selectedCount, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: `${getConstants(props.shape)}
fn getParentCluster(cluster: u32) -> u32 {
  return clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u];
}`,
    body: `let count = min(clusterCount[clusterCountOffset], CLUSTER_CAPACITY);
  for (var cluster = 0u; cluster < count; cluster++) {
    let base = selectionOffset + cluster * SELECTION_WORDS;
    selection[base] = bitcast<u32>(0.0);
    selection[base + 1u] = 0u;
    selection[base + 2u] = NONE;
    selection[base + 3u] = NONE;
  }
  // Children first: select unless the children's best total is strictly larger.
  for (var cluster = 0u; cluster < count; cluster++) {
    let base = selectionOffset + cluster * SELECTION_WORDS;
    let parent = getParentCluster(cluster);
    if (parent == NONE) { continue; }
    let childBest = bitcast<f32>(selection[base]);
    let own = stability[stabilityOffset + cluster];
    var best = own;
    if (childBest > own) {
      best = childBest;
    } else {
      selection[base + 1u] = 1u;
    }
    let parentBase = selectionOffset + parent * SELECTION_WORDS;
    selection[parentBase] = bitcast<u32>(bitcast<f32>(selection[parentBase]) + best);
  }
  // Parents first: a selected ancestor deselects its descendants and becomes their representative.
  for (var step = 0u; step < count; step++) {
    let cluster = count - 1u - step;
    let base = selectionOffset + cluster * SELECTION_WORDS;
    let parent = getParentCluster(cluster);
    var inherited = NONE;
    if (parent != NONE) { inherited = selection[selectionOffset + parent * SELECTION_WORDS + 2u]; }
    if (inherited != NONE) {
      selection[base + 1u] = 0u;
      selection[base + 2u] = inherited;
    } else if (selection[base + 1u] == 1u) {
      selection[base + 2u] = cluster;
    }
  }
  // Compact labels in cluster order.
  var label = 0u;
  for (var cluster = 0u; cluster < count; cluster++) {
    let base = selectionOffset + cluster * SELECTION_WORDS;
    if (selection[base + 1u] == 1u) {
      selection[base + 3u] = label;
      label = label + 1u;
    }
  }
  selectedCount[selectedCountOffset] = label;`
  });
}
