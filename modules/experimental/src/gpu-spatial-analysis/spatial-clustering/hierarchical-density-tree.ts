// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUFlagOffsets,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLDispatchGate, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createParallelTreeBuilder,
  getParallelMergeTreeNodes
} from '../../utils/parallel-merge-tree';
import type {HierarchicalDensityShape} from './hierarchical-density-clustering-kernels';

const OPERATION = 'GPUHierarchicalDensityClustering';
type WordView = GraphDataView<'uint32'>;

function readBinding(name: string, view: WordView): WGSLKernelBinding {
  return {name, view, type: 'u32', access: 'read'};
}
function writeBinding(name: string, view: WordView, atomic = false): WGSLKernelBinding {
  return {name, view, type: atomic ? 'atomic<u32>' : 'u32', access: 'read_write'};
}

/**
 * Parallel Kruskal reconstruction tree. Each MST edge splices its two endpoint paths into an
 * increasing chain of merge ranks. atomicMin inserts the earlier merge and retries the displaced
 * later parent; every retry increases the lower rank, so there is no lock or cross-workgroup wait.
 * The resulting parent of a node is its first containing merge in the exact sorted-edge order.
 * Euler-tour list ranking computes every subtree size in logarithmic dispatch depth, including
 * a dendrogram that is a chain. Child slot order is immaterial to condensing.
 * @internal
 */
export function getHierarchicalDensityDendrogramNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    state: WordView;
    edges: WordView;
    edgeOrder: WordView;
    dendrogram: WordView;
  }
): GPUCommandNode<Parameters>[] {
  const rows = props.shape.rowCount;
  const nodeCount = rows * 2;
  const {nodes, createWordView, appendKernel, appendJumpNodes, parents, children} =
    getParallelMergeTreeNodes(graph, {
      id: props.id,
      operation: OPERATION,
      nodeCount,
      edgeCount: rows,
      initializeBindings: [writeBinding('dendrogram', props.dendrogram)],
      initializeBody: `let base = dendrogramOffset + index * 5u;
  dendrogram[base] = NONE;
  dendrogram[base + 1u] = 0u;
  dendrogram[base + 2u] = 0u;
  dendrogram[base + 3u] = NONE;
  dendrogram[base + 4u] = NONE;`,
      edgeBindings: [
        readBinding('state', props.state),
        readBinding('edges', props.edges),
        readBinding('edgeOrder', props.edgeOrder),
        writeBinding('dendrogram', props.dendrogram)
      ],
      edgeBody: `if (index >= min(state[stateOffset + 1u], ROW_COUNT - 1u)) { return; }
  let edge = edgesOffset + edgeOrder[edgeOrderOffset + index] * 3u;
  let node = ROW_COUNT + index;
  dendrogram[dendrogramOffset + node * 5u + 2u] = edges[edge + 2u];
  let firstEndpoint = edges[edge];
  let secondEndpoint = edges[edge + 1u];`,
      declarations: `const ROW_COUNT: u32 = ${rows}u;`
    });
  const tour = createWordView('tour', nodeCount * 4);
  appendKernel(
    'tour-init',
    nodeCount * 2,
    [
      readBinding('parents', parents),
      readBinding('children', children),
      writeBinding('tour', tour)
    ],
    `
  let node = index / 2u;
  let first = children[childrenOffset + node * 2u];
  var successor = NONE;
  var contribution = 0u;
  if (index % 2u == 0u) {
    successor = node * 2u + 1u;
    if (first != NONE) { successor = first * 2u; }
    else if (node < ROW_COUNT) { contribution = 1u; }
  } else {
    let parent = parents[parentsOffset + node];
    if (parent != NONE) {
      successor = parent * 2u + 1u;
      if (children[childrenOffset + parent * 2u] == node) {
        successor = children[childrenOffset + parent * 2u + 1u] * 2u;
      }
    }
  }
  tour[tourOffset + index * 2u] = successor;
  tour[tourOffset + index * 2u + 1u] = contribution;`,
    `const ROW_COUNT: u32 = ${rows}u;`
  );
  appendJumpNodes(
    'tour',
    nodeCount * 2,
    2,
    Math.ceil(Math.log2(nodeCount * 2)),
    tour,
    `
  let base = tableOffset + index * 2u;
  var successor = table[base];
  var total = table[base + 1u];
  if (successor != NONE) {
    total += table[tableOffset + successor * 2u + 1u];
    successor = table[tableOffset + successor * 2u];
  }
  scratch[scratchOffset + index * 2u] = successor;
  scratch[scratchOffset + index * 2u + 1u] = total;
  if (successor != NONE) { atomicOr(&pending[pendingOffset], 1u); }`
  );
  appendKernel(
    'finish',
    nodeCount,
    [
      readBinding('parents', parents),
      readBinding('children', children),
      readBinding('tour', tour),
      writeBinding('dendrogram', props.dendrogram)
    ],
    `
  let base = dendrogramOffset + index * 5u;
  let first = children[childrenOffset + index * 2u];
  dendrogram[base] = parents[parentsOffset + index];
  dendrogram[base + 1u] = tour[tourOffset + index * 4u + 1u] - tour[tourOffset + index * 4u + 3u];
  dendrogram[base + 3u] = first;
  dendrogram[base + 4u] = select(children[childrenOffset + index * 2u + 1u], NONE, first == NONE);`
  );
  return nodes;
}

/** Condenses both ancestor maps with convergent, indirect-gated pointer doubling. @internal */
export function getHierarchicalDensityCondensingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; rowCount: number; pointers: WordView}
): GPUCommandNode<Parameters>[] {
  const {nodes, appendJumpNodes} = createParallelTreeBuilder(graph, props.id, OPERATION);
  const nodeCount = props.rowCount * 2;
  appendJumpNodes(
    'ancestors',
    nodeCount,
    2,
    Math.ceil(Math.log2(nodeCount)) + 1,
    props.pointers,
    `
  for (var word = 0u; word < 2u; word++) {
    let next = table[tableOffset + index * 2u + word];
    var successor = next;
    if (next != NONE) { successor = table[tableOffset + next * 2u + word]; }
    scratch[scratchOffset + index * 2u + word] = successor;
    if (successor != NONE && table[tableOffset + successor * 2u + word] != successor) {
      atomicOr(&pending[pendingOffset], 1u);
    }
  }`
  );
  return nodes;
}

/**
 * EOM by heavy-path contraction. Every light edge at least halves the number of points, so there
 * are at most log2(rows) light levels even for a comb-shaped condensed tree. Within a heavy path,
 * the recurrence is f(x) = max(ownStability, lightBest + x). Pointer doubling composes these
 * max-plus functions, giving O(log²(rows)) dispatch depth. Stability contributions retain their
 * existing fixed-order segmented sum. A final ancestor maximum suppresses selected descendants;
 * GPUFlagOffsets preserves compact labels in ascending cluster-ID order. @internal
 */
export function getHierarchicalDensitySelectionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    clusterCount: WordView;
    stability: GraphDataView<'float32'>;
    clusterInfo: WordView;
    selection: WordView;
    selectedCount: WordView;
  }
): GPUCommandNode<Parameters>[] {
  const {nodes, createWordView, appendKernel, appendJumpNodes} = createParallelTreeBuilder(
    graph,
    props.id,
    OPERATION
  );
  const capacity = props.shape.clusterCapacity;
  const rounds = Math.ceil(Math.log2(Math.max(capacity, 2))) + 1;
  const children = createWordView('children', capacity * 2);
  const heavy = createWordView('heavy', capacity);
  const depth = createWordView('depth', capacity * 2);
  // [current light depth + 1, unresolved chain links]. Depth zero is the last active level.
  const cursor = createWordView('cursor', 2);
  const functions = createWordView('functions', capacity * 3);
  const best = createTransientView(graph, `${props.id}-best`, 'float32', capacity);
  appendKernel(
    'init',
    capacity,
    [
      writeBinding('children', children),
      writeBinding('selection', props.selection),
      writeBinding('cursor', cursor)
    ],
    `
  children[childrenOffset + index * 2u] = NONE;
  children[childrenOffset + index * 2u + 1u] = 0u;
  selection[selectionOffset + index * 4u] = 0u;
  selection[selectionOffset + index * 4u + 1u] = 0u;
  selection[selectionOffset + index * 4u + 2u] = NONE;
  selection[selectionOffset + index * 4u + 3u] = NONE;
  if (index == 0u) { cursor[cursorOffset] = 0u; cursor[cursorOffset + 1u] = 0u; }`
  );
  appendKernel(
    'children',
    capacity,
    [
      readBinding('clusterCount', props.clusterCount),
      readBinding('clusterInfo', props.clusterInfo),
      writeBinding('children', children, true)
    ],
    `
  if (index >= clusterCount[clusterCountOffset]) { return; }
  let parent = clusterInfo[clusterInfoOffset + index * 3u + 1u];
  if (parent != NONE) {
    atomicMin(&children[childrenOffset + parent * 2u], index);
    atomicMax(&children[childrenOffset + parent * 2u + 1u], index);
  }`
  );
  appendKernel(
    'heavy',
    capacity,
    [
      readBinding('children', children),
      readBinding('clusterInfo', props.clusterInfo),
      writeBinding('heavy', heavy)
    ],
    `
  let first = children[childrenOffset + index * 2u];
  var chosen = first;
  if (first != NONE) {
    let second = children[childrenOffset + index * 2u + 1u];
    if (clusterInfo[clusterInfoOffset + second * 3u + 2u] > clusterInfo[clusterInfoOffset + first * 3u + 2u]) {
      chosen = second;
    }
  }
  heavy[heavyOffset + index] = chosen;`
  );
  appendKernel(
    'depth-init',
    capacity,
    [
      readBinding('clusterCount', props.clusterCount),
      readBinding('clusterInfo', props.clusterInfo),
      readBinding('heavy', heavy),
      writeBinding('depth', depth)
    ],
    `
  var parent = NONE;
  var light = 0u;
  if (index < clusterCount[clusterCountOffset]) {
    parent = clusterInfo[clusterInfoOffset + index * 3u + 1u];
    if (parent != NONE) { light = select(1u, 0u, heavy[heavyOffset + parent] == index); }
  }
  depth[depthOffset + index * 2u] = parent;
  depth[depthOffset + index * 2u + 1u] = light;`
  );
  appendJumpNodes(
    'depth',
    capacity,
    2,
    rounds,
    depth,
    `
  let base = tableOffset + index * 2u;
  var parent = table[base];
  var total = table[base + 1u];
  if (parent != NONE) {
    total += table[tableOffset + parent * 2u + 1u];
    parent = table[tableOffset + parent * 2u];
  }
  scratch[scratchOffset + index * 2u] = parent;
  scratch[scratchOffset + index * 2u + 1u] = total;
  if (parent != NONE) { atomicOr(&pending[pendingOffset], 1u); }`
  );
  appendKernel(
    'maximum-depth',
    capacity,
    [
      readBinding('clusterCount', props.clusterCount),
      readBinding('depth', depth),
      writeBinding('cursor', cursor, true)
    ],
    `
  if (index < clusterCount[clusterCountOffset]) {
    atomicMax(&cursor[cursorOffset], depth[depthOffset + index * 2u + 1u] + 1u);
  }`
  );
  const levelDispatch = createWGSLDispatchGate<Parameters>(graph, {
    id: `${props.id}-level-dispatch`,
    operation: OPERATION,
    shapes: [[capacity]]
  });
  const chainDispatch = createWGSLDispatchGate<Parameters>(graph, {
    id: `${props.id}-chain-dispatch`,
    operation: OPERATION,
    shapes: [[capacity]]
  });
  const scratch = createWordView('functions-scratch', capacity * 3);
  for (let level = 0; level <= Math.floor(Math.log2(props.shape.rowCount)); level++) {
    appendKernel(
      `level-gate-${level}`,
      1,
      [writeBinding('cursor', cursor), levelDispatch.binding],
      `
  if (cursor[cursorOffset] != 0u) { ${levelDispatch.armWGSL} }
  else { ${levelDispatch.disarmWGSL} }
  cursor[cursorOffset + 1u] = 0u;`
    );
    appendKernel(
      `functions-${level}`,
      capacity,
      [
        readBinding('clusterCount', props.clusterCount),
        readBinding('children', children),
        readBinding('heavy', heavy),
        readBinding('depth', depth),
        {name: 'stability', view: props.stability, type: 'f32', access: 'read'},
        {name: 'best', view: best, type: 'f32', access: 'read'},
        writeBinding('functions', functions),
        writeBinding('cursor', cursor, true)
      ],
      `
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != atomicLoad(&cursor[cursorOffset])) { return; }
  let next = heavy[heavyOffset + index];
  var addition = 0.0;
  if (next != NONE) {
    let first = children[childrenOffset + index * 2u];
    let second = children[childrenOffset + index * 2u + 1u];
    let light = select(first, second, next == first);
    addition = best[bestOffset + light];
    atomicOr(&cursor[cursorOffset + 1u], 1u);
  }
  functions[functionsOffset + index * 3u] = next;
  functions[functionsOffset + index * 3u + 1u] = bitcast<u32>(stability[stabilityOffset + index]);
  functions[functionsOffset + index * 3u + 2u] = bitcast<u32>(addition);`,
      '',
      levelDispatch.getGate(0)
    );
    for (let round = 0; round < rounds; round++) {
      appendKernel(
        `chain-gate-${level}-${round}`,
        1,
        [writeBinding('cursor', cursor), chainDispatch.binding],
        `
  if (cursor[cursorOffset] != 0u && cursor[cursorOffset + 1u] != 0u) { ${chainDispatch.armWGSL} }
  else { ${chainDispatch.disarmWGSL} }
  cursor[cursorOffset + 1u] = 0u;`
      );
      appendKernel(
        `chain-jump-${level}-${round}`,
        capacity,
        [
          readBinding('clusterCount', props.clusterCount),
          readBinding('depth', depth),
          readBinding('functions', functions),
          writeBinding('scratch', scratch),
          writeBinding('cursor', cursor, true)
        ],
        `
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != atomicLoad(&cursor[cursorOffset])) { return; }
  let base = functionsOffset + index * 3u;
  var next = functions[base];
  var floorValue = bitcast<f32>(functions[base + 1u]);
  var addition = bitcast<f32>(functions[base + 2u]);
  if (next != NONE) {
    let child = functionsOffset + next * 3u;
    floorValue = max(floorValue, addition + bitcast<f32>(functions[child + 1u]));
    addition += bitcast<f32>(functions[child + 2u]);
    next = functions[child];
  }
  scratch[scratchOffset + index * 3u] = next;
  scratch[scratchOffset + index * 3u + 1u] = bitcast<u32>(floorValue);
  scratch[scratchOffset + index * 3u + 2u] = bitcast<u32>(addition);
  if (next != NONE) { atomicOr(&cursor[cursorOffset + 1u], 1u); }`,
        '',
        chainDispatch.getGate(0)
      );
      appendKernel(
        `chain-copy-${level}-${round}`,
        capacity,
        [
          readBinding('clusterCount', props.clusterCount),
          readBinding('depth', depth),
          readBinding('cursor', cursor),
          readBinding('scratch', scratch),
          writeBinding('functions', functions)
        ],
        `
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != cursor[cursorOffset]) { return; }
  for (var word = 0u; word < 3u; word++) {
    functions[functionsOffset + index * 3u + word] = scratch[scratchOffset + index * 3u + word];
  }`,
        '',
        chainDispatch.getGate(0)
      );
    }
    appendKernel(
      `best-${level}`,
      capacity,
      [
        readBinding('clusterCount', props.clusterCount),
        readBinding('depth', depth),
        readBinding('cursor', cursor),
        readBinding('functions', functions),
        {name: 'best', view: best, type: 'f32', access: 'read_write'}
      ],
      `
  if (index >= clusterCount[clusterCountOffset] || depth[depthOffset + index * 2u + 1u] + 1u != cursor[cursorOffset]) { return; }
  best[bestOffset + index] = bitcast<f32>(functions[functionsOffset + index * 3u + 1u]);`,
      '',
      levelDispatch.getGate(0)
    );
    appendKernel(
      `advance-${level}`,
      1,
      [writeBinding('cursor', cursor)],
      `
  if (cursor[cursorOffset] != 0u) { cursor[cursorOffset] -= 1u; }`
    );
  }
  const representatives = createWordView('representatives', capacity * 2);
  appendKernel(
    'candidates',
    capacity,
    [
      readBinding('clusterCount', props.clusterCount),
      readBinding('clusterInfo', props.clusterInfo),
      readBinding('children', children),
      {name: 'best', view: best, type: 'f32', access: 'read'},
      {name: 'stability', view: props.stability, type: 'f32', access: 'read'},
      writeBinding('representatives', representatives)
    ],
    `
  var parent = NONE;
  var representative = 0u;
  if (index < clusterCount[clusterCountOffset]) {
    parent = clusterInfo[clusterInfoOffset + index * 3u + 1u];
    let first = children[childrenOffset + index * 2u];
    var childBest = 0.0;
    if (first != NONE) { childBest = best[bestOffset + first] + best[bestOffset + children[childrenOffset + index * 2u + 1u]]; }
    // The root is excluded. Equality selects the parent, as in the serial EOM traversal.
    if (parent != NONE && !(childBest > stability[stabilityOffset + index])) { representative = index + 1u; }
  }
  representatives[representativesOffset + index * 2u] = parent;
  representatives[representativesOffset + index * 2u + 1u] = representative;`
  );
  appendJumpNodes(
    'representatives',
    capacity,
    2,
    rounds,
    representatives,
    `
  let base = tableOffset + index * 2u;
  var parent = table[base];
  var representative = table[base + 1u];
  if (parent != NONE) {
    representative = max(representative, table[tableOffset + parent * 2u + 1u]);
    parent = table[tableOffset + parent * 2u];
  }
  scratch[scratchOffset + index * 2u] = parent;
  scratch[scratchOffset + index * 2u + 1u] = representative;
  if (parent != NONE) { atomicOr(&pending[pendingOffset], 1u); }`
  );
  const selected = createWordView('selected', capacity);
  const offsets = createWordView('offsets', capacity);
  appendKernel(
    'flags',
    capacity,
    [
      readBinding('representatives', representatives),
      writeBinding('selected', selected),
      writeBinding('selection', props.selection)
    ],
    `
  let representative = representatives[representativesOffset + index * 2u + 1u];
  selected[selectedOffset + index] = select(0u, 1u, representative == index + 1u);
  selection[selectionOffset + index * 4u + 1u] = selected[selectedOffset + index];
  selection[selectionOffset + index * 4u + 2u] = select(NONE, representative - 1u, representative != 0u);`
  );
  nodes.push(
    ...new GPUFlagOffsets({
      id: `${props.id}-offsets`,
      flags: selected,
      offsets,
      count: props.selectedCount
    }).getCommandNodes(graph)
  );
  appendKernel(
    'labels',
    capacity,
    [
      readBinding('selected', selected),
      readBinding('offsets', offsets),
      writeBinding('selection', props.selection)
    ],
    `
  selection[selectionOffset + index * 4u + 3u] = select(NONE, offsets[offsetsOffset + index], selected[selectedOffset + index] != 0u);`
  );
  return nodes;
}
