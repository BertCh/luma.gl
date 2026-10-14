// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandGraphNodeCondition, GraphBufferUse} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUHierarchicalDensityClustering';

/**
 * State words: `[valid point count, minimum spanning tree edge count]`. @internal
 */
export const HIERARCHICAL_DENSITY_STATE_WORDS = 2;
/** u32 words per Boruvka candidate and per component minimum: weight bits, low row, high row. @internal */
export const HIERARCHICAL_DENSITY_EDGE_WORDS = 3;
/** u32 words per BVH node: minimum xy, maximum xy (f32 bits), leaf row. @internal */
export const HIERARCHICAL_DENSITY_NODE_WORDS = 5;
/** u32 words per dendrogram node: parent, size, merge weight (f32 bits), left child, right child. @internal */
export const HIERARCHICAL_DENSITY_DENDROGRAM_WORDS = 5;
/** u32 words per condensed cluster: birth lambda (f32 bits), parent cluster, point count. @internal */
export const HIERARCHICAL_DENSITY_CLUSTER_WORDS = 3;
/** u32 words of the excess-of-mass work table per cluster: reserved, selected, representative, label. @internal */
export const HIERARCHICAL_DENSITY_SELECTION_WORDS = 4;

/** Sizes shared by the kernels. @internal */
export type HierarchicalDensityShape = {
  rowCount: number;
  minimumSamples: number;
  minimumClusterSize: number;
  /** Largest number of condensed clusters, `floor(2 * rows / minimumClusterSize) + 1`. */
  clusterCapacity: number;
  leafCapacity: number;
  internalNodeCount: number;
  levelCount: number;
};

/** Optional gate of an unrolled Boruvka kernel (library rule 13). @internal */
export type HierarchicalDensityGate<Parameters> = {
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

function getConstants(shape: HierarchicalDensityShape): string {
  return `const ROW_COUNT: u32 = ${shape.rowCount}u;
const NODE_COUNT: u32 = ${2 * shape.rowCount}u;
const MINIMUM_SAMPLES: u32 = ${shape.minimumSamples}u;
const MINIMUM_CLUSTER_SIZE: u32 = ${shape.minimumClusterSize}u;
const CLUSTER_CAPACITY: u32 = ${shape.clusterCapacity}u;
const LEAF_CAPACITY: u32 = ${shape.leafCapacity}u;
const INTERNAL_NODE_COUNT: u32 = ${shape.internalNodeCount}u;
const LEVEL_COUNT: u32 = ${shape.levelCount}u;
const EDGE_WORDS: u32 = ${HIERARCHICAL_DENSITY_EDGE_WORDS}u;
const NODE_WORDS: u32 = ${HIERARCHICAL_DENSITY_NODE_WORDS}u;
const DENDROGRAM_WORDS: u32 = ${HIERARCHICAL_DENSITY_DENDROGRAM_WORDS}u;
const CLUSTER_WORDS: u32 = ${HIERARCHICAL_DENSITY_CLUSTER_WORDS}u;
const SELECTION_WORDS: u32 = ${HIERARCHICAL_DENSITY_SELECTION_WORDS}u;
const NONE: u32 = 0xffffffffu;
const FLOAT32_MAXIMUM: f32 = 3.402823e38;
fn isFiniteNumber(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`;
}

function getGate<Parameters>(gate?: HierarchicalDensityGate<Parameters>): {
  condition?: GPUCommandGraphNodeCondition<Parameters>;
  extraResources?: readonly GraphBufferUse[];
} {
  return gate ? {condition: gate.condition, extraResources: gate.extraResources} : {};
}

/** Path-halving find over an atomic `parents` binding. */
const FIND_ROOT_WGSL = /* wgsl */ `
fn findRoot(start: u32) -> u32 {
  var current = start;
  loop {
    let parent = atomicLoad(&parents[parentsOffset + current]);
    if (parent == current) {
      return current;
    }
    let grandparent = atomicLoad(&parents[parentsOffset + parent]);
    if (grandparent != parent) {
      atomicStore(&parents[parentsOffset + current], grandparent);
    }
    current = parent;
  }
}`;

/**
 * Per point: the core distance (distance to the `minimumSamples`-th nearest neighbor, the point
 * itself included, as scikit-learn computes it; +infinity when fewer neighbors exist), the BVH box
 * (empty for non-finite points), its own union-find parent, and the valid point count. @internal
 */
export function createHierarchicalDensityInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    positions: GraphDataView<'float32x2'>;
    neighborDistances: GraphDataView<'float32'>;
    neighborCounts: GraphDataView<'uint32'>;
    boxMinima: GraphDataView<'float32x2'>;
    boxMaxima: GraphDataView<'float32x2'>;
    parents: GraphDataView<'uint32'>;
    coreDistances: GraphDataView<'float32'>;
    state: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'init',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'neighborDistances', view: props.neighborDistances, type: 'f32', access: 'read'},
      {name: 'neighborCounts', view: props.neighborCounts, type: 'u32', access: 'read'},
      {name: 'boxMinima', view: props.boxMinima, type: 'f32', access: 'read_write'},
      {name: 'boxMaxima', view: props.boxMaxima, type: 'f32', access: 'read_write'},
      {name: 'parents', view: props.parents, type: 'u32', access: 'read_write'},
      {name: 'coreDistances', view: props.coreDistances, type: 'f32', access: 'read_write'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: getConstants(props.shape),
    body: `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let isValid = isFiniteNumber(x) && isFiniteNumber(y);
  parents[parentsOffset + index] = index;
  var core = bitcast<f32>(0x7f800000u | (index & 0u));
  if (isValid && neighborCounts[neighborCountsOffset + index] >= MINIMUM_SAMPLES) {
    core = neighborDistances[neighborDistancesOffset + index * MINIMUM_SAMPLES + MINIMUM_SAMPLES - 1u];
  }
  coreDistances[coreDistancesOffset + index] = core;
  boxMinima[boxMinimaOffset + index * 2u] = select(FLOAT32_MAXIMUM, x, isValid);
  boxMinima[boxMinimaOffset + index * 2u + 1u] = select(FLOAT32_MAXIMUM, y, isValid);
  boxMaxima[boxMaximaOffset + index * 2u] = select(-FLOAT32_MAXIMUM, x, isValid);
  boxMaxima[boxMaximaOffset + index * 2u + 1u] = select(-FLOAT32_MAXIMUM, y, isValid);
  if (isValid) { atomicAdd(&state[stateOffset], 1u); }`
  });
}

/** Packs the point BVH into one table of node boxes and leaf rows. @internal */
export function createHierarchicalDensityNodeDataNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeMinima: GraphDataView;
    nodeMaxima: GraphDataView;
    leafIds: GraphDataView<'uint32'>;
    nodeData: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'node-data',
    bindings: [
      {name: 'nodeMinima', view: props.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: props.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: props.leafIds, type: 'u32', access: 'read'},
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.leafCapacity - 1,
    declarations: getConstants(props.shape),
    body: `let base = nodeDataOffset + index * NODE_WORDS;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  var row = NONE;
  if (index >= INTERNAL_NODE_COUNT) {
    row = leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT];
    if (row >= ROW_COUNT) { row = NONE; }
  }
  nodeData[base + 4u] = row;`
  });
}

/**
 * One-thread gate of a Boruvka round: arms the round's dispatches while more than one component
 * remains (valid points minus spanning-tree edges), otherwise disarms them. @internal
 */
export function createHierarchicalDensityGateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    state: GraphDataView<'uint32'>;
    dispatchBinding: WGSLKernelBinding;
    armWGSL: string;
    disarmWGSL: string;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gate',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      props.dispatchBinding
    ],
    invocationCount: 1,
    body: `if (state[stateOffset] > state[stateOffset + 1u] + 1u) {
    ${props.armWGSL}
  } else {
    ${props.disarmWGSL}
  }`
  });
}

/** Writes every valid point's component root (NONE for invalid points) and clears the component minima. @internal */
export function createHierarchicalDensityResolveNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    positions: GraphDataView<'float32x2'>;
    parents: GraphDataView<'uint32'>;
    components: GraphDataView<'uint32'>;
    componentBest: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'resolve',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'parents', view: props.parents, type: 'atomic<u32>', access: 'read_write'},
      {name: 'components', view: props.components, type: 'u32', access: 'read_write'},
      {name: 'componentBest', view: props.componentBest, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${FIND_ROOT_WGSL}`,
    body: `let isValid = isFiniteNumber(positions[positionsOffset + index * 2u]) &&
    isFiniteNumber(positions[positionsOffset + index * 2u + 1u]);
  components[componentsOffset + index] = select(NONE, findRoot(index), isValid);
  for (var word = 0u; word < EDGE_WORDS; word++) {
    componentBest[componentBestOffset + index * EDGE_WORDS + word] = NONE;
  }`,
    ...getGate(props.gate)
  });
}

/**
 * Flags the BVH leaf slots whose component differs from the previous slot's, for the inclusive
 * scan that answers "does this node hold a single component" in O(1). @internal
 */
export function createHierarchicalDensityLeafFlagsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeData: GraphDataView<'uint32'>;
    components: GraphDataView<'uint32'>;
    changeFlags: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'leaf-flags',
    bindings: [
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read'},
      {name: 'components', view: props.components, type: 'u32', access: 'read'},
      {name: 'changeFlags', view: props.changeFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.leafCapacity,
    declarations: `${getConstants(props.shape)}
fn getLeafComponent(slot: u32) -> u32 {
  let row = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + slot) * NODE_WORDS + 4u];
  return select(components[componentsOffset + min(row, ROW_COUNT - 1u)], NONE, row == NONE);
}`,
    body: `var isChange = 0u;
  if (index > 0u && getLeafComponent(index) != getLeafComponent(index - 1u)) { isChange = 1u; }
  changeFlags[changeFlagsOffset + index] = isChange;`,
    ...getGate(props.gate)
  });
}

/**
 * Boruvka search: every valid point walks the BVH (nearer child first) for the nearest point of
 * another component under the mutual reachability distance `max(core a, core b, |a - b|)`, with
 * the strict total edge order `(weight, low row, high row)` so that Boruvka never closes a cycle.
 * A node is pruned when `max(core a, box distance)` exceeds the best weight (with a 1e-6 relative
 * slack, so rounding never drops a tie), or when all its leaves lie in the point's component: a
 * node covers a contiguous leaf range, and the inclusive scan of component changes is equal at both
 * ends exactly when the range holds one component. Writes the candidate and the component minimum
 * weight. @internal
 */
export function createHierarchicalDensitySearchNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    nodeData: GraphDataView<'uint32'>;
    changeScan: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x2'>;
    coreDistances: GraphDataView<'float32'>;
    components: GraphDataView<'uint32'>;
    candidates: GraphDataView<'uint32'>;
    componentBest: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'search',
    bindings: [
      {name: 'nodeData', view: props.nodeData, type: 'u32', access: 'read'},
      {name: 'changeScan', view: props.changeScan, type: 'u32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'coreDistances', view: props.coreDistances, type: 'f32', access: 'read'},
      {name: 'components', view: props.components, type: 'u32', access: 'read'},
      {name: 'candidates', view: props.candidates, type: 'u32', access: 'read_write'},
      {name: 'componentBest', view: props.componentBest, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
const STACK_SIZE: u32 = ${2 * props.shape.levelCount}u;
const BOUND_SLACK: f32 = 1.000001;
fn getPoint(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}
fn getBoxDistance(node: u32, point: vec2<f32>) -> f32 {
  let base = nodeDataOffset + node * NODE_WORDS;
  let minimum = vec2<f32>(bitcast<f32>(nodeData[base]), bitcast<f32>(nodeData[base + 1u]));
  let maximum = vec2<f32>(bitcast<f32>(nodeData[base + 2u]), bitcast<f32>(nodeData[base + 3u]));
  if (minimum.x > maximum.x || minimum.y > maximum.y) { return FLOAT32_MAXIMUM; }
  let gap = max(max(minimum - point, point - maximum), vec2<f32>(0.0));
  return sqrt(dot(gap, gap));
}
// True when every leaf under the node belongs to 'component' (the node covers a contiguous range
// of leaf slots in the flat complete tree).
fn isNodeInComponent(node: u32, component: u32) -> bool {
  let depth = firstLeadingBit(node + 1u);
  let span = 1u << (LEVEL_COUNT - 1u - depth);
  let first = (node + 1u - (1u << depth)) * span;
  let last = first + span - 1u;
  if (changeScan[changeScanOffset + last] != changeScan[changeScanOffset + first]) { return false; }
  let row = nodeData[nodeDataOffset + (INTERNAL_NODE_COUNT + first) * NODE_WORDS + 4u];
  return row != NONE && components[componentsOffset + row] == component;
}
// Lexicographic (weight bits, low row, high row) order of candidate edges; weights are
// non-negative, so their f32 bits order like the values.
fn isBetter(weight: u32, low: u32, high: u32, bestWeight: u32, bestLow: u32, bestHigh: u32) -> bool {
  if (weight != bestWeight) { return weight < bestWeight; }
  if (low != bestLow) { return low < bestLow; }
  return high < bestHigh;
}`,
    body: `let component = components[componentsOffset + index];
  var bestWeight = NONE;
  var bestLow = NONE;
  var bestHigh = NONE;
  if (component != NONE) {
    let point = getPoint(index);
    let core = coreDistances[coreDistancesOffset + index];
    var stackNode: array<u32, STACK_SIZE>;
    var stackLower: array<f32, STACK_SIZE>;
    var top = 0u;
    if (!isNodeInComponent(0u, component)) {
      stackNode[0] = 0u;
      stackLower[0] = max(core, getBoxDistance(0u, point));
      top = 1u;
    }
    loop {
      if (top == 0u) { break; }
      top = top - 1u;
      let node = stackNode[top];
      let bound = select(FLOAT32_MAXIMUM, bitcast<f32>(bestWeight) * BOUND_SLACK, bestWeight != NONE);
      if (stackLower[top] > bound) { continue; }
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = max(core, getBoxDistance(first, point));
        let secondLower = max(core, getBoxDistance(second, point));
        let firstUsable = firstLower <= bound && getBoxDistance(first, point) < FLOAT32_MAXIMUM &&
          !isNodeInComponent(first, component);
        let secondUsable = secondLower <= bound && getBoxDistance(second, point) < FLOAT32_MAXIMUM &&
          !isNodeInComponent(second, component);
        if (firstUsable && secondUsable && top + 2u <= STACK_SIZE) {
          if (firstLower <= secondLower) {
            stackNode[top] = second; stackLower[top] = secondLower;
            stackNode[top + 1u] = first; stackLower[top + 1u] = firstLower;
          } else {
            stackNode[top] = first; stackLower[top] = firstLower;
            stackNode[top + 1u] = second; stackLower[top + 1u] = secondLower;
          }
          top = top + 2u;
        } else if (firstUsable && top + 1u <= STACK_SIZE) {
          stackNode[top] = first; stackLower[top] = firstLower;
          top = top + 1u;
        } else if (secondUsable && top + 1u <= STACK_SIZE) {
          stackNode[top] = second; stackLower[top] = secondLower;
          top = top + 1u;
        }
        continue;
      }
      let other = nodeData[nodeDataOffset + node * NODE_WORDS + 4u];
      if (other == NONE || other == index || components[componentsOffset + other] == component) { continue; }
      let otherCore = coreDistances[coreDistancesOffset + other];
      // The same expression from both ends gives the same f32 weight (a - b is -(b - a)).
      let weight = bitcast<u32>(max(max(core, otherCore), length(point - getPoint(other))));
      let low = min(index, other);
      let high = max(index, other);
      if (isBetter(weight, low, high, bestWeight, bestLow, bestHigh)) {
        bestWeight = weight;
        bestLow = low;
        bestHigh = high;
      }
    }
    if (bestWeight != NONE) {
      atomicMin(&componentBest[componentBestOffset + component * EDGE_WORDS], bestWeight);
    }
  }
  candidates[candidatesOffset + index * EDGE_WORDS] = bestWeight;
  candidates[candidatesOffset + index * EDGE_WORDS + 1u] = bestLow;
  candidates[candidatesOffset + index * EDGE_WORDS + 2u] = bestHigh;`,
    ...getGate(props.gate)
  });
}

/**
 * Second and third passes of the component minimum: among candidates with the minimum weight the
 * smallest low row (`word` 1), then among those the smallest high row (`word` 2). @internal
 */
export function createHierarchicalDensityTieNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    word: 1 | 2;
    components: GraphDataView<'uint32'>;
    candidates: GraphDataView<'uint32'>;
    componentBest: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  const matchesEarlier =
    props.word === 1
      ? 'true'
      : 'atomicLoad(&componentBest[best + 1u]) == candidates[candidatesOffset + index * EDGE_WORDS + 1u]';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.word === 1 ? 'tie-low' : 'tie-high',
    bindings: [
      {name: 'components', view: props.components, type: 'u32', access: 'read'},
      {name: 'candidates', view: props.candidates, type: 'u32', access: 'read'},
      {name: 'componentBest', view: props.componentBest, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: getConstants(props.shape),
    body: `let component = components[componentsOffset + index];
  let weight = candidates[candidatesOffset + index * EDGE_WORDS];
  if (component == NONE || weight == NONE) { return; }
  let best = componentBestOffset + component * EDGE_WORDS;
  if (atomicLoad(&componentBest[best]) == weight && ${matchesEarlier}) {
    atomicMin(&componentBest[best + ${props.word}u], candidates[candidatesOffset + index * EDGE_WORDS + ${props.word}u]);
  }`,
    ...getGate(props.gate)
  });
}

/**
 * Adds each component root's minimum edge to the spanning tree and unites its endpoints (lock-free,
 * larger root under smaller). Two components that chose the same edge add it once (the lower
 * root). The strict edge order guarantees that the chosen edges form a forest. @internal
 */
export function createHierarchicalDensityAddNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    components: GraphDataView<'uint32'>;
    componentBest: GraphDataView<'uint32'>;
    parents: GraphDataView<'uint32'>;
    edges: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    gate?: HierarchicalDensityGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'add',
    bindings: [
      {name: 'components', view: props.components, type: 'u32', access: 'read'},
      {name: 'componentBest', view: props.componentBest, type: 'u32', access: 'read'},
      {name: 'parents', view: props.parents, type: 'atomic<u32>', access: 'read_write'},
      {name: 'edges', view: props.edges, type: 'u32', access: 'read_write'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${FIND_ROOT_WGSL}
fn unite(first: u32, second: u32) {
  var a = findRoot(first);
  var b = findRoot(second);
  loop {
    if (a == b) {
      return;
    }
    let low = min(a, b);
    let high = max(a, b);
    let result = atomicCompareExchangeWeak(&parents[parentsOffset + high], high, low);
    if (result.exchanged) {
      return;
    }
    a = findRoot(low);
    b = findRoot(high);
  }
}`,
    body: `if (components[componentsOffset + index] != index) { return; }
  let best = componentBestOffset + index * EDGE_WORDS;
  let weight = componentBest[best];
  if (weight == NONE) { return; }
  let low = componentBest[best + 1u];
  let high = componentBest[best + 2u];
  var other = components[componentsOffset + low];
  if (other == index) { other = components[componentsOffset + high]; }
  let otherBest = componentBestOffset + other * EDGE_WORDS;
  let isMutual = componentBest[otherBest] == weight && componentBest[otherBest + 1u] == low &&
    componentBest[otherBest + 2u] == high;
  if (isMutual && other < index) { return; }
  let slot = atomicAdd(&state[stateOffset + 1u], 1u);
  if (slot < ROW_COUNT) {
    edges[edgesOffset + slot * EDGE_WORDS] = low;
    edges[edgesOffset + slot * EDGE_WORDS + 1u] = high;
    edges[edgesOffset + slot * EDGE_WORDS + 2u] = weight;
  }
  unite(low, high);`,
    ...getGate(props.gate)
  });
}

/**
 * Sort keys of the stable spanning-tree edge chain: high row, then low row, then weight bits,
 * giving `(weight, low, high)` order. Slots at and after the edge count sort last. @internal
 */
export function createHierarchicalDensityEdgeKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    word: 0 | 1 | 2;
    state: GraphDataView<'uint32'>;
    edges: GraphDataView<'uint32'>;
    order?: GraphDataView<'uint32'>;
    identity?: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'state', view: props.state, type: 'u32', access: 'read'},
    {name: 'edges', view: props.edges, type: 'u32', access: 'read'}
  ];
  if (props.order) {
    bindings.push({name: 'order', view: props.order, type: 'u32', access: 'read'});
  }
  if (props.identity) {
    bindings.push({name: 'identity', view: props.identity, type: 'u32', access: 'read_write'});
  }
  bindings.push({name: 'keys', view: props.keys, type: 'u32', access: 'read_write'});
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `edge-keys-${props.word}`,
    bindings,
    invocationCount: props.shape.rowCount,
    declarations: getConstants(props.shape),
    body: `let edgeCount = min(state[stateOffset + 1u], ROW_COUNT);
  ${props.identity ? 'identity[identityOffset + index] = index;' : ''}
  let slot = ${props.order ? 'order[orderOffset + index]' : 'index'};
  keys[keysOffset + index] = select(NONE, edges[edgesOffset + slot * EDGE_WORDS + ${props.word}u], slot < edgeCount);`
  });
}

/** WGSL helpers over a `dendrogram` binding. */
const DENDROGRAM_WGSL = /* wgsl */ `
fn getNodeParent(node: u32) -> u32 {
  return dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS];
}
fn getNodeSize(node: u32) -> u32 {
  return dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS + 1u];
}
// Merge density 1 / weight; a zero weight (duplicate points) gives the largest f32 instead of
// scikit-learn's infinity, so differences stay finite.
fn getNodeLambda(node: u32) -> f32 {
  let weight = bitcast<f32>(dendrogram[dendrogramOffset + node * DENDROGRAM_WORDS + 2u]);
  return select(FLOAT32_MAXIMUM, 1.0 / weight, weight > 0.0);
}
fn isBigNode(node: u32) -> bool {
  return node != NONE && getNodeSize(node) >= MINIMUM_CLUSTER_SIZE;
}
// A node of at least minimumClusterSize points whose parent is a true split (both children big),
// or the root, starts a condensed cluster.
fn isClusterStart(node: u32) -> bool {
  if (!isBigNode(node)) { return false; }
  let parent = getNodeParent(node);
  if (parent == NONE) { return true; }
  let base = dendrogramOffset + parent * DENDROGRAM_WORDS;
  return isBigNode(dendrogram[base + 3u]) && isBigNode(dendrogram[base + 4u]);
}`;

/**
 * Initial condensing pointers per dendrogram node: word 0 points to the node itself when it starts
 * a cluster, else to its parent (pointer jumping then finds each big node's cluster start); word 1
 * points to the node itself when it is big, else to its parent (pointer jumping then finds each
 * point's lowest big ancestor, where it falls out of its cluster). Also writes the cluster-start
 * flags for the compact cluster IDs. @internal
 */
export function createHierarchicalDensityPointerInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    pointers: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pointer-init',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read'},
      {name: 'pointers', view: props.pointers, type: 'u32', access: 'read_write'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${DENDROGRAM_WGSL}`,
    body: `let parent = getNodeParent(index);
  let isStart = isClusterStart(index);
  pointers[pointersOffset + index * 2u] = select(parent, index, isStart);
  pointers[pointersOffset + index * 2u + 1u] = select(parent, index, isBigNode(index));
  startFlags[startFlagsOffset + index] = select(0u, 1u, isStart);`
  });
}

/**
 * Per condensed cluster (from its start node): birth lambda (the parent split's lambda, 0 for the
 * root), parent cluster ID, and point count for heavy-path contraction. @internal
 */
export function createHierarchicalDensityClusterInfoNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
    clusterOffsets: GraphDataView<'uint32'>;
    pointers: GraphDataView<'uint32'>;
    clusterInfo: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'cluster-info',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read'},
      {name: 'clusterOffsets', view: props.clusterOffsets, type: 'u32', access: 'read'},
      {name: 'pointers', view: props.pointers, type: 'u32', access: 'read'},
      {name: 'clusterInfo', view: props.clusterInfo, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${DENDROGRAM_WGSL}`,
    body: `if (startFlags[startFlagsOffset + index] == 0u) { return; }
  let cluster = clusterOffsets[clusterOffsetsOffset + index];
  if (cluster >= CLUSTER_CAPACITY) { return; }
  let parent = getNodeParent(index);
  var birth = 0.0;
  var parentCluster = NONE;
  if (parent != NONE) {
    birth = getNodeLambda(parent);
    let parentStart = pointers[pointersOffset + parent * 2u];
    parentCluster = clusterOffsets[clusterOffsetsOffset + parentStart];
  }
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS] = bitcast<u32>(birth);
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u] = parentCluster;
  clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 2u] = getNodeSize(index);`
  });
}

/**
 * Stability contributions, one row per dendrogram node (scikit-learn's condensed tree rows): a
 * point contributes `lambda_p - birth(C)` to the cluster `C` it falls out of at `lambda_p` (its
 * lowest big ancestor), and a child cluster `D` contributes `size(D) (birth(D) - birth(C))` to its
 * parent `C`. Other rows get the key `CLUSTER_CAPACITY`, which the segment sum skips. Also writes
 * each point's lowest cluster for the labels. @internal
 */
export function createHierarchicalDensityContributionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    dendrogram: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
    clusterOffsets: GraphDataView<'uint32'>;
    pointers: GraphDataView<'uint32'>;
    clusterInfo: GraphDataView<'uint32'>;
    contributionKeys: GraphDataView<'uint32'>;
    contributions: GraphDataView<'float32'>;
    lowClusters: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'contributions',
    bindings: [
      {name: 'dendrogram', view: props.dendrogram, type: 'u32', access: 'read'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read'},
      {name: 'clusterOffsets', view: props.clusterOffsets, type: 'u32', access: 'read'},
      {name: 'pointers', view: props.pointers, type: 'u32', access: 'read'},
      {name: 'clusterInfo', view: props.clusterInfo, type: 'u32', access: 'read'},
      {name: 'contributionKeys', view: props.contributionKeys, type: 'u32', access: 'read_write'},
      {name: 'contributions', view: props.contributions, type: 'f32', access: 'read_write'},
      {name: 'lowClusters', view: props.lowClusters, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 2 * props.shape.rowCount,
    declarations: `${getConstants(props.shape)}
${DENDROGRAM_WGSL}
fn getBirth(cluster: u32) -> f32 {
  return bitcast<f32>(clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS]);
}`,
    body: `var key = CLUSTER_CAPACITY;
  var value = 0.0;
  if (index < ROW_COUNT) {
    var lowCluster = NONE;
    let fallNode = pointers[pointersOffset + index * 2u + 1u];
    if (fallNode != NONE) {
      let start = pointers[pointersOffset + fallNode * 2u];
      if (start != NONE) {
        let cluster = clusterOffsets[clusterOffsetsOffset + start];
        if (cluster < CLUSTER_CAPACITY) {
          lowCluster = cluster;
          key = cluster;
          value = getNodeLambda(fallNode) - getBirth(cluster);
        }
      }
    }
    lowClusters[lowClustersOffset + index] = lowCluster;
  } else if (startFlags[startFlagsOffset + index] != 0u) {
    let cluster = clusterOffsets[clusterOffsetsOffset + index];
    if (cluster < CLUSTER_CAPACITY) {
      let parentCluster = clusterInfo[clusterInfoOffset + cluster * CLUSTER_WORDS + 1u];
      if (parentCluster != NONE) {
        key = parentCluster;
        value = f32(getNodeSize(index)) * (getBirth(cluster) - getBirth(parentCluster));
      }
    }
  }
  contributionKeys[contributionKeysOffset + index] = key;
  contributions[contributionsOffset + index] = value;`
  });
}

/** Writes each point's compact label (its lowest cluster's selected representative) or NONE. @internal */
export function createHierarchicalDensityLabelsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: HierarchicalDensityShape;
    lowClusters: GraphDataView<'uint32'>;
    selection: GraphDataView<'uint32'>;
    labels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'labels',
    bindings: [
      {name: 'lowClusters', view: props.lowClusters, type: 'u32', access: 'read'},
      {name: 'selection', view: props.selection, type: 'u32', access: 'read'},
      {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: getConstants(props.shape),
    body: `let cluster = lowClusters[lowClustersOffset + index];
  var label = NONE;
  if (cluster != NONE) {
    let representative = selection[selectionOffset + cluster * SELECTION_WORDS + 2u];
    if (representative != NONE) {
      label = selection[selectionOffset + representative * SELECTION_WORDS + 3u];
    }
  }
  labels[labelsOffset + index] = label;`
  });
}
