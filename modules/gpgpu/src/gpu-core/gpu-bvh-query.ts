// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode, createGPUComputeCommandNode} from './gpu-command-node';
import {Buffer, type Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/engine';
import {GPUCommandGraph, type GraphBufferUse, type GraphDataView} from './gpu-command-graph';
import type {GPUBVHBoundsView} from './gpu-bvh';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewBinding,
  getViewElementOffset,
  validatePackedUint32View,
  validatePackedView
} from './graph-data-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type GPUBoundedDispatchLayout
} from './gpu-dispatch-utils';

const BVH_QUERY_WORKGROUP_SIZE = 256;
const INVALID_NODE = 0xffffffff;
const MAXIMUM_LINEAR_WORKGROUP_COUNT = Math.floor(INVALID_NODE / BVH_QUERY_WORKGROUP_SIZE) + 1;

/** Flat complete-binary hierarchy consumed by {@link GPUBVHQuery}. */
export type GPUBVHView = {
  leafCapacity: number;
  nodeMinima: GPUBVHBoundsView;
  nodeMaxima: GPUBVHBoundsView;
  nodeChildren: GraphDataView<'uint32x2'>;
  leafIds: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Bounds predicate evaluated during BVH traversal. */
export type GPUBVHQueryKind = 'point' | 'bounds';

/** Properties for one complete-binary BVH traversal. */
export type GPUBVHQueryProps = {
  /** Prefix for generated graph resources and node IDs. */
  id?: string;
  /** Flat hierarchy, commonly a `GPUBVH` instance. */
  bvh: GPUBVHView;
  /** Node intersection rule. */
  kind: GPUBVHQueryKind;
  /** Packed point or minima/maxima query, mutable between graph encodings. */
  query: GraphDataView<'float32'>;
  /** Caller-owned capacity-bounded stable leaf IDs. */
  output: GraphDataView<'uint32'>;
  /** Caller-owned row receiving the full stored-tree match count. */
  count: GraphDataView<'uint32'>;
  /** Caller-owned row receiving source-tree or output-capacity overflow. */
  overflow: GraphDataView<'uint32'>;
  /** Optional source-ID-addressed result mask, cleared on every encoding. */
  outputMask?: GraphDataView<'uint32'>;
  /** Optional row receiving the number of active nodes whose bounds were tested. */
  visitedCount?: GraphDataView<'uint32'>;
};

/**
 * Traverses a complete-binary GPU BVH for point containment or bounds intersection.
 *
 * One compute pass processes each tree depth. Only active nodes test their bounds and activate
 * children, while matched leaves append stable IDs atomically. Output order is unspecified;
 * `visitedCount` exposes traversal work for topology-quality and cost comparisons.
 */
export class GPUBVHQuery {
  readonly id: string;
  readonly bvh: GPUBVHView;
  readonly kind: GPUBVHQueryKind;
  readonly query: GraphDataView<'float32'>;
  readonly output: GraphDataView<'uint32'>;
  readonly count: GraphDataView<'uint32'>;
  readonly overflow: GraphDataView<'uint32'>;
  readonly outputMask?: GraphDataView<'uint32'>;
  readonly visitedCount?: GraphDataView<'uint32'>;
  readonly dimension: 2 | 3;
  readonly nodeCount: number;
  readonly internalNodeCount: number;
  readonly levelCount: number;

  constructor(props: GPUBVHQueryProps) {
    this.id = props.id ?? 'gpu-bvh-query';
    this.bvh = props.bvh;
    this.kind = props.kind;
    this.query = props.query;
    this.output = props.output;
    this.count = props.count;
    this.overflow = props.overflow;
    this.outputMask = props.outputMask;
    this.visitedCount = props.visitedCount;
    this.dimension = this.bvh.nodeMinima.format === 'float32x2' ? 2 : 3;

    if (!Number.isSafeInteger(this.bvh.leafCapacity) || !isPowerOfTwo(this.bvh.leafCapacity)) {
      throw new Error(`${this.id} bvh leafCapacity must be a positive power of two`);
    }
    this.nodeCount = this.bvh.leafCapacity * 2 - 1;
    this.internalNodeCount = this.bvh.leafCapacity - 1;
    this.levelCount = Math.log2(this.bvh.leafCapacity) + 1;
    if (!Number.isSafeInteger(this.nodeCount) || this.nodeCount > INVALID_NODE) {
      throw new Error(`${this.id} bvh node count exceeds uint32 range`);
    }
    validateIndexView(this);
    validatePackedView(this.query, ['float32'], `${this.id} query`);
    validatePackedUint32View(this.output, `${this.id} output`);
    validatePackedUint32View(this.count, `${this.id} count`);
    validatePackedUint32View(this.overflow, `${this.id} overflow`);
    if (this.outputMask) validatePackedUint32View(this.outputMask, `${this.id} outputMask`);
    if (this.visitedCount) validatePackedUint32View(this.visitedCount, `${this.id} visitedCount`);
    if (
      this.count.length < 1 ||
      this.overflow.length < 1 ||
      (this.visitedCount && this.visitedCount.length < 1)
    ) {
      throw new Error(`${this.id} count, overflow, and visitedCount must contain one uint32 row`);
    }
    const expectedQueryLength = this.kind === 'point' ? this.dimension : this.dimension * 2;
    if (this.query.length !== expectedQueryLength) {
      throw new Error(`${this.id} ${this.kind} query must contain ${expectedQueryLength} floats`);
    }
    validateDisjointViews(this);
  }

  /** Adds initialization and level-ordered traversal without submission or readback. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    const views = [
      this.bvh.nodeMinima,
      this.bvh.nodeMaxima,
      this.bvh.nodeChildren,
      this.bvh.leafIds,
      this.bvh.overflow,
      this.query,
      this.output,
      this.count,
      this.overflow,
      ...(this.outputMask ? [this.outputMask] : []),
      ...(this.visitedCount ? [this.visitedCount] : [])
    ];
    if (views.some(view => view.buffer.graph !== graph)) {
      throw new Error(`${this.id} views must belong to the target graph`);
    }
    const frontiers = [0, 1].map(frontierIndex =>
      createTransientView(
        graph,
        `${this.id}-frontier-${frontierIndex}`,
        'uint32',
        getFrontierCapacity(this.levelCount, frontierIndex) + 1
      )
    ) as [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
    const frontierDispatches = [0, 1].map(frontierIndex =>
      createTransientView(
        graph,
        `${this.id}-frontier-${frontierIndex}-dispatch`,
        'uint32',
        3,
        Buffer.STORAGE | Buffer.INDIRECT
      )
    ) as [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
    nodes.push(...addInitializePass(graph, this, frontiers, frontierDispatches));
    if (this.outputMask) nodes.push(...addClearOutputMaskPass(graph, this));
    for (let depth = 0; depth < this.levelCount; depth++) {
      const currentFrontierIndex = depth % 2;
      const nextFrontierIndex = 1 - currentFrontierIndex;
      nodes.push(
        ...addTraversalLevelPass(
          graph,
          this,
          frontiers[currentFrontierIndex],
          frontiers[nextFrontierIndex],
          frontierDispatches[currentFrontierIndex],
          depth
        )
      );
      if (depth < this.levelCount - 1) {
        nodes.push(
          ...addFrontierDispatchPass(graph, this, {
            currentFrontier: frontiers[currentFrontierIndex],
            nextFrontier: frontiers[nextFrontierIndex],
            nextDispatch: frontierDispatches[nextFrontierIndex],
            depth
          })
        );
      }
    }
    if (this.outputMask && this.output.length > 0) {
      const outputDispatch = createTransientView(
        graph,
        `${this.id}-output-mask-dispatch`,
        'uint32',
        3,
        Buffer.STORAGE | Buffer.INDIRECT
      );
      nodes.push(...addOutputDispatchPass(graph, this, outputDispatch));
      nodes.push(...addOutputMaskPass(graph, this, outputDispatch));
    }

    return nodes;
  }
}

function validateIndexView(query: GPUBVHQuery): void {
  validatePackedView(query.bvh.nodeMinima, ['float32x2', 'float32x3'], `${query.id} nodeMinima`);
  validatePackedView(query.bvh.nodeMaxima, ['float32x2', 'float32x3'], `${query.id} nodeMaxima`);
  validatePackedView(query.bvh.nodeChildren, ['uint32x2'], `${query.id} nodeChildren`);
  validatePackedUint32View(query.bvh.leafIds, `${query.id} leafIds`);
  validatePackedUint32View(query.bvh.overflow, `${query.id} bvh overflow`);
  if (
    query.bvh.nodeMinima.format !== query.bvh.nodeMaxima.format ||
    query.bvh.nodeMinima.length !== query.nodeCount ||
    query.bvh.nodeMaxima.length !== query.nodeCount ||
    query.bvh.nodeChildren.length !== query.nodeCount ||
    query.bvh.leafIds.length !== query.bvh.leafCapacity
  ) {
    throw new Error(`${query.id} bvh views must match its complete-binary topology`);
  }
  if (query.bvh.overflow.length < 1) {
    throw new Error(`${query.id} bvh overflow must contain one uint32 row`);
  }
}

function addInitializePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUBVHQuery,
  frontiers: readonly [GraphDataView<'uint32'>, GraphDataView<'uint32'>],
  frontierDispatches: readonly [GraphDataView<'uint32'>, GraphDataView<'uint32'>]
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const visitedBinding = query.visitedCount
    ? '@group(0) @binding(7) var<storage, read_write> visitedCount: array<u32>;'
    : '';
  const source = /* wgsl */ `
const FRONTIER_A_OFFSET: u32 = ${getViewElementOffset(frontiers[0])}u;
const FRONTIER_B_OFFSET: u32 = ${getViewElementOffset(frontiers[1])}u;
const DISPATCH_A_OFFSET: u32 = ${getViewElementOffset(frontierDispatches[0])}u;
const DISPATCH_B_OFFSET: u32 = ${getViewElementOffset(frontierDispatches[1])}u;
const BVH_OVERFLOW_OFFSET: u32 = ${getViewElementOffset(query.bvh.overflow)}u;
const COUNT_OFFSET: u32 = ${getViewElementOffset(query.count)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(query.overflow)}u;
${
  query.visitedCount
    ? `const VISITED_OFFSET: u32 = ${getViewElementOffset(query.visitedCount)}u;`
    : ''
}
@group(0) @binding(0) var<storage, read_write> frontierA: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> frontierB: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> dispatchA: array<u32>;
@group(0) @binding(3) var<storage, read_write> dispatchB: array<u32>;
@group(0) @binding(4) var<storage, read> bvhOverflow: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputCount: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputOverflow: array<u32>;
${visitedBinding}

@compute @workgroup_size(1) fn main() {
  atomicStore(&frontierA[FRONTIER_A_OFFSET], 1u);
  atomicStore(&frontierA[FRONTIER_A_OFFSET + 1u], 0u);
  atomicStore(&frontierB[FRONTIER_B_OFFSET], 0u);
  dispatchA[DISPATCH_A_OFFSET + 0u] = 1u;
  dispatchA[DISPATCH_A_OFFSET + 1u] = 1u;
  dispatchA[DISPATCH_A_OFFSET + 2u] = 1u;
  dispatchB[DISPATCH_B_OFFSET + 0u] = 0u;
  dispatchB[DISPATCH_B_OFFSET + 1u] = 1u;
  dispatchB[DISPATCH_B_OFFSET + 2u] = 1u;
  outputCount[COUNT_OFFSET] = 0u;
  outputOverflow[OVERFLOW_OFFSET] = min(bvhOverflow[BVH_OVERFLOW_OFFSET], 1u);
  ${query.visitedCount ? 'visitedCount[VISITED_OFFSET] = 1u;' : ''}
}`;
  const resources: GraphBufferUse[] = [
    {buffer: frontiers[0], usage: 'storage-write'},
    {buffer: frontiers[1], usage: 'storage-write'},
    {buffer: frontierDispatches[0], usage: 'storage-write'},
    {buffer: frontierDispatches[1], usage: 'storage-write'},
    {buffer: query.bvh.overflow, usage: 'storage-read'},
    {buffer: query.count, usage: 'storage-write'},
    {buffer: query.overflow, usage: 'storage-write'},
    ...(query.visitedCount
      ? ([{buffer: query.visitedCount, usage: 'storage-write'}] as GraphBufferUse[])
      : [])
  ];
  nodes.push(
    ...addKernelPass(graph, {
      id: `${query.id}-initialize`,
      source,
      resources,
      bindings: {
        frontierA: frontiers[0],
        frontierB: frontiers[1],
        dispatchA: frontierDispatches[0],
        dispatchB: frontierDispatches[1],
        bvhOverflow: query.bvh.overflow,
        outputCount: query.count,
        outputOverflow: query.overflow,
        ...(query.visitedCount ? {visitedCount: query.visitedCount} : {})
      },
      dispatch: {x: 1, y: 1, z: 1}
    })
  );

  return nodes;
}

function addTraversalLevelPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUBVHQuery,
  currentFrontier: GraphDataView<'uint32'>,
  nextFrontier: GraphDataView<'uint32'>,
  currentDispatch: GraphDataView<'uint32'>,
  depth: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const leafLevel = depth === query.levelCount - 1;
  const predicate = makeNodePredicate(query);
  if (leafLevel) {
    const source = /* wgsl */ `
const NODE_COUNT: u32 = ${query.nodeCount}u;
const INTERNAL_NODE_COUNT: u32 = ${query.internalNodeCount}u;
const LEAF_CAPACITY: u32 = ${query.bvh.leafCapacity}u;
const DIMENSION: u32 = ${query.dimension}u;
const OUTPUT_CAPACITY: u32 = ${query.output.length}u;
const FRONTIER_OFFSET: u32 = ${getViewElementOffset(currentFrontier)}u;
const NODE_MINIMA_OFFSET: u32 = ${getViewElementOffset(query.bvh.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${getViewElementOffset(query.bvh.nodeMaxima)}u;
const LEAF_IDS_OFFSET: u32 = ${getViewElementOffset(query.bvh.leafIds)}u;
const QUERY_OFFSET: u32 = ${getViewElementOffset(query.query)}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(query.output)}u;
const COUNT_OFFSET: u32 = ${getViewElementOffset(query.count)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(query.overflow)}u;
@group(0) @binding(0) var<storage, read_write> currentFrontier: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> nodeMinima: array<f32>;
@group(0) @binding(2) var<storage, read> nodeMaxima: array<f32>;
@group(0) @binding(3) var<storage, read> leafIds: array<u32>;
@group(0) @binding(4) var<storage, read> queryValues: array<f32>;
@group(0) @binding(5) var<storage, read_write> outputIds: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputCount: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> outputOverflow: array<atomic<u32>>;

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${BVH_QUERY_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getIndirectInvocationIndexSource()}
  let frontierCount = atomicLoad(&currentFrontier[FRONTIER_OFFSET]);
  if (index >= frontierCount) { return; }
  let nodeIndex = atomicLoad(&currentFrontier[FRONTIER_OFFSET + 1u + index]);
  if (nodeIndex < INTERNAL_NODE_COUNT || nodeIndex >= NODE_COUNT) { return; }
  ${predicate}
  if (selected) {
    let leafIndex = nodeIndex - INTERNAL_NODE_COUNT;
    if (leafIndex >= LEAF_CAPACITY) { return; }
    let objectId = leafIds[LEAF_IDS_OFFSET + leafIndex];
    let outputIndex = atomicAdd(&outputCount[COUNT_OFFSET], 1u);
    if (outputIndex < OUTPUT_CAPACITY) {
      outputIds[OUTPUT_OFFSET + outputIndex] = objectId;
    } else {
      atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u);
    }
  }
}`;
    nodes.push(
      ...addKernelPass(graph, {
        id: `${query.id}-depth-${depth}`,
        source,
        resources: [
          {buffer: currentFrontier, usage: 'storage-read-write'},
          {buffer: query.bvh.nodeMinima, usage: 'storage-read'},
          {buffer: query.bvh.nodeMaxima, usage: 'storage-read'},
          {buffer: query.bvh.leafIds, usage: 'storage-read'},
          {buffer: query.query, usage: 'storage-read'},
          {buffer: query.output, usage: 'storage-write'},
          {buffer: query.count, usage: 'storage-read-write'},
          {buffer: query.overflow, usage: 'storage-read-write'}
        ],
        bindings: {
          currentFrontier,
          nodeMinima: query.bvh.nodeMinima,
          nodeMaxima: query.bvh.nodeMaxima,
          leafIds: query.bvh.leafIds,
          queryValues: query.query,
          outputIds: query.output,
          outputCount: query.count,
          outputOverflow: query.overflow
        },
        indirectDispatch: currentDispatch
      })
    );
    return nodes;
  }

  const visitedBinding = query.visitedCount
    ? '@group(0) @binding(7) var<storage, read_write> visitedCount: array<atomic<u32>>;'
    : '';
  const source = /* wgsl */ `
const NODE_COUNT: u32 = ${query.nodeCount}u;
const NEXT_CAPACITY: u32 = ${nextFrontier.length - 1}u;
const DIMENSION: u32 = ${query.dimension}u;
const CURRENT_FRONTIER_OFFSET: u32 = ${getViewElementOffset(currentFrontier)}u;
const NEXT_FRONTIER_OFFSET: u32 = ${getViewElementOffset(nextFrontier)}u;
const NODE_MINIMA_OFFSET: u32 = ${getViewElementOffset(query.bvh.nodeMinima)}u;
const NODE_MAXIMA_OFFSET: u32 = ${getViewElementOffset(query.bvh.nodeMaxima)}u;
const CHILDREN_OFFSET: u32 = ${getViewElementOffset(query.bvh.nodeChildren)}u;
const QUERY_OFFSET: u32 = ${getViewElementOffset(query.query)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(query.overflow)}u;
${
  query.visitedCount
    ? `const VISITED_OFFSET: u32 = ${getViewElementOffset(query.visitedCount)}u;`
    : ''
}
@group(0) @binding(0) var<storage, read_write> currentFrontier: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> nextFrontier: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> nodeMinima: array<f32>;
@group(0) @binding(3) var<storage, read> nodeMaxima: array<f32>;
@group(0) @binding(4) var<storage, read> nodeChildren: array<u32>;
@group(0) @binding(5) var<storage, read> queryValues: array<f32>;
@group(0) @binding(6) var<storage, read_write> outputOverflow: array<atomic<u32>>;
${visitedBinding}

fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

@compute @workgroup_size(${BVH_QUERY_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getIndirectInvocationIndexSource()}
  let frontierCount = atomicLoad(&currentFrontier[CURRENT_FRONTIER_OFFSET]);
  if (index >= frontierCount) { return; }
  let nodeIndex = atomicLoad(&currentFrontier[CURRENT_FRONTIER_OFFSET + 1u + index]);
  if (nodeIndex >= NODE_COUNT) { return; }
  ${predicate}
  if (selected) {
    let childComponent = nodeIndex * 2u;
    let left = nodeChildren[CHILDREN_OFFSET + childComponent];
    let right = nodeChildren[CHILDREN_OFFSET + childComponent + 1u];
    var appendedCount = 0u;
    if (left < NODE_COUNT) {
      let nextIndex = atomicAdd(&nextFrontier[NEXT_FRONTIER_OFFSET], 1u);
      if (nextIndex < NEXT_CAPACITY) {
        atomicStore(&nextFrontier[NEXT_FRONTIER_OFFSET + 1u + nextIndex], left);
        appendedCount++;
      } else {
        atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u);
      }
    }
    if (right < NODE_COUNT) {
      let nextIndex = atomicAdd(&nextFrontier[NEXT_FRONTIER_OFFSET], 1u);
      if (nextIndex < NEXT_CAPACITY) {
        atomicStore(&nextFrontier[NEXT_FRONTIER_OFFSET + 1u + nextIndex], right);
        appendedCount++;
      } else {
        atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u);
      }
    }
    ${query.visitedCount ? 'atomicAdd(&visitedCount[VISITED_OFFSET], appendedCount);' : ''}
  }
}`;
  nodes.push(
    ...addKernelPass(graph, {
      id: `${query.id}-depth-${depth}`,
      source,
      resources: [
        {buffer: currentFrontier, usage: 'storage-read-write'},
        {buffer: nextFrontier, usage: 'storage-read-write'},
        {buffer: query.bvh.nodeMinima, usage: 'storage-read'},
        {buffer: query.bvh.nodeMaxima, usage: 'storage-read'},
        {buffer: query.bvh.nodeChildren, usage: 'storage-read'},
        {buffer: query.query, usage: 'storage-read'},
        {buffer: query.overflow, usage: 'storage-read-write'},
        ...(query.visitedCount
          ? ([{buffer: query.visitedCount, usage: 'storage-read-write'}] as GraphBufferUse[])
          : [])
      ],
      bindings: {
        currentFrontier,
        nextFrontier,
        nodeMinima: query.bvh.nodeMinima,
        nodeMaxima: query.bvh.nodeMaxima,
        nodeChildren: query.bvh.nodeChildren,
        queryValues: query.query,
        outputOverflow: query.overflow,
        ...(query.visitedCount ? {visitedCount: query.visitedCount} : {})
      },
      indirectDispatch: currentDispatch
    })
  );

  return nodes;
}

function addClearOutputMaskPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUBVHQuery
): readonly GPUCommandNode<Parameters>[] {
  const dispatch = getBoundedDispatchLayout(
    `${query.id}-clear-output-mask`,
    query.outputMask!.length,
    BVH_QUERY_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const MASK_LENGTH: u32 = ${query.outputMask!.length}u;
const MASK_OFFSET: u32 = ${getViewElementOffset(query.outputMask!)}u;
@group(0) @binding(0) var<storage, read_write> outputMask: array<u32>;

@compute @workgroup_size(${BVH_QUERY_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatch, BVH_QUERY_WORKGROUP_SIZE)}
  if (index < MASK_LENGTH) { outputMask[MASK_OFFSET + index] = 0u; }
}`;
  return addKernelPass(graph, {
    id: `${query.id}-clear-output-mask`,
    source,
    resources: [{buffer: query.outputMask!, usage: 'storage-write'}],
    bindings: {outputMask: query.outputMask!},
    dispatch
  });
}

function addFrontierDispatchPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUBVHQuery,
  props: {
    currentFrontier: GraphDataView<'uint32'>;
    nextFrontier: GraphDataView<'uint32'>;
    nextDispatch: GraphDataView<'uint32'>;
    depth: number;
  }
): readonly GPUCommandNode<Parameters>[] {
  const source = /* wgsl */ `
const CURRENT_FRONTIER_OFFSET: u32 = ${getViewElementOffset(props.currentFrontier)}u;
const NEXT_FRONTIER_OFFSET: u32 = ${getViewElementOffset(props.nextFrontier)}u;
const NEXT_CAPACITY: u32 = ${props.nextFrontier.length - 1}u;
const NEXT_DISPATCH_OFFSET: u32 = ${getViewElementOffset(props.nextDispatch)}u;
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(query.overflow)}u;
const MAXIMUM_WORKGROUP_DIMENSION: u32 = ${graph.device.limits.maxComputeWorkgroupsPerDimension}u;
@group(0) @binding(0) var<storage, read_write> currentFrontier: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> nextFrontier: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> nextDispatch: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputOverflow: array<atomic<u32>>;

fn divideRoundUp(numerator: u32, denominator: u32) -> u32 {
  return numerator / denominator + select(0u, 1u, numerator % denominator != 0u);
}

@compute @workgroup_size(1) fn main() {
  let requiredCount = atomicLoad(&nextFrontier[NEXT_FRONTIER_OFFSET]);
  let frontierCount = min(requiredCount, NEXT_CAPACITY);
  atomicStore(&nextFrontier[NEXT_FRONTIER_OFFSET], frontierCount);
  if (requiredCount > NEXT_CAPACITY) {
    atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u);
  }
  let workgroupTotal = divideRoundUp(frontierCount, ${BVH_QUERY_WORKGROUP_SIZE}u);
  let x = min(workgroupTotal, MAXIMUM_WORKGROUP_DIMENSION);
  let afterX = select(0u, divideRoundUp(workgroupTotal, max(x, 1u)), x != 0u);
  let y = min(max(afterX, 1u), MAXIMUM_WORKGROUP_DIMENSION);
  let z = select(1u, divideRoundUp(afterX, y), x != 0u);
  let valid = z <= MAXIMUM_WORKGROUP_DIMENSION;
  nextDispatch[NEXT_DISPATCH_OFFSET + 0u] = select(0u, x, valid);
  nextDispatch[NEXT_DISPATCH_OFFSET + 1u] = y;
  nextDispatch[NEXT_DISPATCH_OFFSET + 2u] = min(z, MAXIMUM_WORKGROUP_DIMENSION);
  if (!valid) { atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u); }
  atomicStore(&currentFrontier[CURRENT_FRONTIER_OFFSET], 0u);
}`;
  return addKernelPass(graph, {
    id: `${query.id}-depth-${props.depth}-publish-frontier`,
    source,
    resources: [
      {buffer: props.currentFrontier, usage: 'storage-read-write'},
      {buffer: props.nextFrontier, usage: 'storage-read-write'},
      {buffer: props.nextDispatch, usage: 'storage-write'},
      {buffer: query.overflow, usage: 'storage-read-write'}
    ],
    bindings: {
      currentFrontier: props.currentFrontier,
      nextFrontier: props.nextFrontier,
      nextDispatch: props.nextDispatch,
      outputOverflow: query.overflow
    },
    dispatch: {x: 1, y: 1, z: 1}
  });
}

function validateDisjointViews(query: GPUBVHQuery): void {
  const inputs = [
    query.bvh.nodeMinima,
    query.bvh.nodeMaxima,
    query.bvh.nodeChildren,
    query.bvh.leafIds,
    query.bvh.overflow,
    query.query
  ];
  const outputs = [
    query.output,
    query.count,
    query.overflow,
    ...(query.outputMask ? [query.outputMask] : []),
    ...(query.visitedCount ? [query.visitedCount] : [])
  ];
  for (let outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
    const output = outputs[outputIndex]!;
    if (inputs.some(input => doGraphDataViewsOverlap(input, output))) {
      throw new Error(`${query.id} output views must not overlap query or BVH inputs`);
    }
    if (outputs.slice(outputIndex + 1).some(other => doGraphDataViewsOverlap(output, other))) {
      throw new Error(`${query.id} output views must not overlap one another`);
    }
  }
}

function addOutputDispatchPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUBVHQuery,
  outputDispatch: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const source = /* wgsl */ `
const OUTPUT_CAPACITY: u32 = ${query.output.length}u;
const COUNT_OFFSET: u32 = ${getViewElementOffset(query.count)}u;
const DISPATCH_OFFSET: u32 = ${getViewElementOffset(outputDispatch)}u;
const MAXIMUM_WORKGROUP_DIMENSION: u32 = ${graph.device.limits.maxComputeWorkgroupsPerDimension}u;
@group(0) @binding(0) var<storage, read> outputCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputDispatch: array<u32>;

fn divideRoundUp(numerator: u32, denominator: u32) -> u32 {
  return numerator / denominator + select(0u, 1u, numerator % denominator != 0u);
}

@compute @workgroup_size(1) fn main() {
  let storedCount = min(outputCount[COUNT_OFFSET], OUTPUT_CAPACITY);
  let workgroupTotal = divideRoundUp(storedCount, ${BVH_QUERY_WORKGROUP_SIZE}u);
  let x = min(workgroupTotal, MAXIMUM_WORKGROUP_DIMENSION);
  let afterX = select(0u, divideRoundUp(workgroupTotal, max(x, 1u)), x != 0u);
  let y = min(max(afterX, 1u), MAXIMUM_WORKGROUP_DIMENSION);
  let z = select(1u, divideRoundUp(afterX, y), x != 0u);
  outputDispatch[DISPATCH_OFFSET + 0u] = x;
  outputDispatch[DISPATCH_OFFSET + 1u] = y;
  outputDispatch[DISPATCH_OFFSET + 2u] = z;
}`;
  return addKernelPass(graph, {
    id: `${query.id}-publish-output-mask-dispatch`,
    source,
    resources: [
      {buffer: query.count, usage: 'storage-read'},
      {buffer: outputDispatch, usage: 'storage-write'}
    ],
    bindings: {outputCount: query.count, outputDispatch},
    dispatch: {x: 1, y: 1, z: 1}
  });
}

function addOutputMaskPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUBVHQuery,
  outputDispatch: GraphDataView<'uint32'>
): readonly GPUCommandNode<Parameters>[] {
  const source = /* wgsl */ `
const OUTPUT_CAPACITY: u32 = ${query.output.length}u;
const MASK_LENGTH: u32 = ${query.outputMask!.length}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(query.output)}u;
const COUNT_OFFSET: u32 = ${getViewElementOffset(query.count)}u;
const MASK_OFFSET: u32 = ${getViewElementOffset(query.outputMask!)}u;
@group(0) @binding(0) var<storage, read> outputIds: array<u32>;
@group(0) @binding(1) var<storage, read> outputCount: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputMask: array<atomic<u32>>;

@compute @workgroup_size(${BVH_QUERY_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getIndirectInvocationIndexSource()}
  let storedCount = min(outputCount[COUNT_OFFSET], OUTPUT_CAPACITY);
  if (index >= storedCount) { return; }
  let objectId = outputIds[OUTPUT_OFFSET + index];
  if (objectId < MASK_LENGTH) { atomicStore(&outputMask[MASK_OFFSET + objectId], 1u); }
}`;
  return addKernelPass(graph, {
    id: `${query.id}-output-mask`,
    source,
    resources: [
      {buffer: query.output, usage: 'storage-read'},
      {buffer: query.count, usage: 'storage-read'},
      {buffer: query.outputMask!, usage: 'storage-read-write'}
    ],
    bindings: {outputIds: query.output, outputCount: query.count, outputMask: query.outputMask!},
    indirectDispatch: outputDispatch
  });
}

function makeNodePredicate(query: GPUBVHQuery): string {
  const axes = ['X', 'Y', ...(query.dimension === 3 ? ['Z'] : [])];
  const nodeValues = axes
    .map(
      (axis, axisIndex) =>
        `let nodeMin${axis} = nodeMinima[NODE_MINIMA_OFFSET + nodeIndex * DIMENSION + ${axisIndex}u];
  let nodeMax${axis} = nodeMaxima[NODE_MAXIMA_OFFSET + nodeIndex * DIMENSION + ${axisIndex}u];`
    )
    .join('\n  ');
  const validNode = axes
    .map(
      axis => `finite(nodeMin${axis}) && finite(nodeMax${axis}) && nodeMin${axis} <= nodeMax${axis}`
    )
    .join(' && ');
  if (query.kind === 'point') {
    const queryValues = axes
      .map((axis, axisIndex) => `let query${axis} = queryValues[QUERY_OFFSET + ${axisIndex}u];`)
      .join('\n  ');
    const validQuery = axes.map(axis => `finite(query${axis})`).join(' && ');
    const contains = axes
      .map(axis => `query${axis} >= nodeMin${axis} && query${axis} <= nodeMax${axis}`)
      .join(' && ');
    return `${nodeValues}
  ${queryValues}
  let selected = ${validNode} && ${validQuery} && ${contains};`;
  }
  const queryValues = axes
    .map(
      (axis, axisIndex) =>
        `let queryMin${axis} = queryValues[QUERY_OFFSET + ${axisIndex}u];
  let queryMax${axis} = queryValues[QUERY_OFFSET + ${axisIndex + query.dimension}u];`
    )
    .join('\n  ');
  const validQuery = axes
    .map(
      axis =>
        `finite(queryMin${axis}) && finite(queryMax${axis}) && queryMin${axis} <= queryMax${axis}`
    )
    .join(' && ');
  const intersects = axes
    .map(axis => `nodeMax${axis} >= queryMin${axis} && nodeMin${axis} <= queryMax${axis}`)
    .join(' && ');
  return `${nodeValues}
  ${queryValues}
  let selected = ${validNode} && ${validQuery} && ${intersects};`;
}

function addKernelPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    source: string;
    resources: GraphBufferUse[];
    bindings: Record<string, GraphDataView>;
    dispatch?: GPUBoundedDispatchLayout;
    indirectDispatch?: GraphDataView<'uint32'>;
  }
): readonly GPUCommandNode<Parameters>[] {
  if (Boolean(props.dispatch) === Boolean(props.indirectDispatch)) {
    throw new Error(`${props.id} requires exactly one direct or indirect dispatch`);
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  nodes.push(
    createGPUComputeCommandNode<Parameters>({
      id: props.id,
      resources: props.indirectDispatch
        ? [...props.resources, {buffer: props.indirectDispatch.buffer, usage: 'indirect' as const}]
        : props.resources,
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: props.id,
          source: props.source,
          shaderLayout: {
            bindings: Object.keys(props.bindings).map((name, location) => ({
              name,
              type: 'storage' as const,
              group: 0,
              location
            }))
          }
        });
        return {
          encode: ({computePass, getBuffer}) => {
            const bindings: Record<string, Binding> = {};
            for (const [name, view] of Object.entries(props.bindings)) {
              bindings[name] = getViewBinding(view, getBuffer);
            }

            if (props.indirectDispatch) {
              kernel.dispatchIndirect(computePass, {
                bindings,
                indirectBuffer: getBuffer(props.indirectDispatch),
                indirectOffset: props.indirectDispatch.byteOffset
              });
            } else {
              kernel.dispatch(computePass, {
                bindings,
                x: props.dispatch!.x,
                y: props.dispatch!.y,
                z: props.dispatch!.z
              });
            }
          },
          destroy: () => kernel.destroy()
        };
      }
    })
  );

  return nodes;
}

function getFrontierCapacity(levelCount: number, frontierIndex: number): number {
  const deepestDepth = levelCount - 1;
  const deepestMatchingDepth =
    deepestDepth % 2 === frontierIndex ? deepestDepth : Math.max(0, deepestDepth - 1);
  return 2 ** deepestMatchingDepth;
}

function getIndirectInvocationIndexSource(): string {
  return `let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${BVH_QUERY_WORKGROUP_SIZE}u + localInvocationIndex;`;
}

function isPowerOfTwo(value: number): boolean {
  return value > 0 && Number.isInteger(Math.log2(value));
}
