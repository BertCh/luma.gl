// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/engine';
import type {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '../gpu-core/gpu-command-graph';
import {addGPUGraphLabelPropagationToGraphWithDispatchLimit} from './gpu-graph-label-propagation-internals';
import type {GPUGraphAdjacency, GPUGraphTopology} from './gpu-graph-topology';
import {
  isGPUGraphViewColumn,
  usesGPUGraphViews,
  validateGPUGraphViewColumn,
  type GPUGraphColumn,
  type GPUGraphTopologyLike
} from './gpu-graph-topology-view';

const DEFAULT_LABEL_ITERATIONS = 32;
const MAXIMUM_LABEL_ITERATIONS = 1024;
const SCALAR_BYTE_LENGTH = 4;

/** Existing caller-owned graph adjacency, unsigned community labels, and optional GPU status. */
export type GPUGraphLabelPropagationProps<
  Topology extends GPUGraphTopologyLike = GPUGraphTopology,
  Column extends GPUGraphColumn<'uint32'> = GPUVector<'uint32'>
> = {
  /** Prefix for generated command-graph node and imported-resource identifiers. */
  id?: string;
  /** Existing weak-neighborhood adjacency; directed graphs require reverse CSR. */
  topology: Topology;
  /** One caller-owned packed unsigned community label for every stable graph vertex. */
  output: Column;
  /** Bounded number of synchronous majority-vote iterations. Defaults to 32. */
  iterations?: number;
  /** Optional scalar set only when the final compiled iteration reaches a fixed point. */
  converged?: Column;
};

/**
 * Publishes deterministic majority-vote communities without leaving browser GPU memory.
 *
 * Labels begin at stable vertex identifiers. Every synchronous iteration selects the most
 * frequent weak-neighbor label plus one self vote, resolving equal support by the lowest label.
 * Self-loop edges add no extra self votes; duplicate and reciprocated graph edges vote separately.
 * Directed graphs require reverse CSR, while undirected graphs reuse symmetric forward adjacency.
 *
 * Selected adjacency overflow publishes `0xffffffff` labels and zero convergence. Empty graphs
 * report convergence when a status buffer is provided. This bounded label-propagation heuristic
 * is neither Louvain nor Leiden, does not optimize modularity, and has worst-case
 * `O(sum(degree²))` work per iteration.
 */
export class GPUGraphLabelPropagation<
  Topology extends GPUGraphTopologyLike = GPUGraphTopology,
  Column extends GPUGraphColumn<'uint32'> = GPUVector<'uint32'>
> {
  /** Prefix for generated command-graph node and imported-resource identifiers. */
  readonly id: string;
  /** Existing caller-owned GPU graph topology. */
  readonly topology: Topology;
  /** Caller-owned vertex-aligned community labels. */
  readonly output: Column;
  /** Number of compiled, explicitly synchronized majority-vote iterations. */
  readonly iterations: number;
  /** Optional caller-owned GPU-resident final fixed-point status. */
  readonly converged?: Column;

  /** Validates graph metadata without allocating, submitting, destroying, or reading GPU work. */
  constructor(props: GPUGraphLabelPropagationProps<Topology, Column>) {
    this.id = props.id ?? 'gpu-graph-label-propagation';
    this.topology = props.topology;
    this.output = props.output;
    this.iterations = props.iterations ?? DEFAULT_LABEL_ITERATIONS;
    this.converged = props.converged;

    if (
      !Number.isSafeInteger(this.iterations) ||
      this.iterations < 1 ||
      this.iterations > MAXIMUM_LABEL_ITERATIONS
    ) {
      throw new Error(`${this.id} iterations must be a safe integer between one and 1024`);
    }
    if (this.topology.graph.directed && !this.topology.reverse) {
      throw new Error(`${this.id} directed weak-neighbor votes require reverse adjacency`);
    }

    validateColumn(this.output, this.topology.graph.vertexCount, `${this.id} output`);
    if (this.converged) {
      validateColumn(this.converged, 1, `${this.id} converged`);
    }
    if (!usesGPUGraphViews(this.topology, [this.output, this.converged])) {
      validateDistinctLabelOutputs(this);
    }
  }

  /** Declares bounded majority-vote passes without queue submission or CPU synchronization. */
  addToGraph<Parameters>(commandGraph: GPUCommandGraph<Parameters>): void {
    addGPUGraphLabelPropagationToGraphWithDispatchLimit(
      this,
      commandGraph,
      commandGraph.device.limits.maxComputeWorkgroupsPerDimension
    );
  }
}

/** Validates a physical or view column with its exact logical row count. */
function validateColumn(column: GPUGraphColumn<'uint32'>, length: number, name: string): void {
  if (isGPUGraphViewColumn(column)) {
    validateGPUGraphViewColumn(column, 'uint32', length, name);
  } else {
    validateLabelVector(column, length, name);
  }
}

/** Requires exactly one packed, aligned unsigned output chunk and its precise logical length. */
function validateLabelVector(vector: GPUVector<'uint32'>, length: number, name: string): void {
  if (
    vector.data.length !== 1 ||
    vector.format !== 'uint32' ||
    vector.stride !== 1 ||
    vector.byteStride !== SCALAR_BYTE_LENGTH ||
    vector.rowByteLength !== SCALAR_BYTE_LENGTH ||
    vector.valueLength !== vector.length ||
    vector.bufferLayout
  ) {
    throw new Error(`${name} must contain exactly one packed uint32 chunk`);
  }
  if (vector.length !== length) {
    throw new Error(`${name} must contain exactly ${length} uint32 rows`);
  }

  const chunk = vector.data[0];
  if (
    chunk.format !== 'uint32' ||
    chunk.length !== length ||
    chunk.stride !== 1 ||
    chunk.byteStride !== SCALAR_BYTE_LENGTH ||
    chunk.rowByteLength !== SCALAR_BYTE_LENGTH ||
    chunk.valueLength !== chunk.length ||
    !Number.isSafeInteger(chunk.byteOffset) ||
    chunk.byteOffset < 0 ||
    chunk.byteOffset % SCALAR_BYTE_LENGTH !== 0
  ) {
    throw new Error(`${name} must contain one packed, uint32-aligned chunk`);
  }
}

/** Protects graph inputs, CSR statuses, and caller-owned output allocations from physical alias. */
function validateDistinctLabelOutputs(
  propagation: GPUGraphLabelPropagation<GPUGraphTopologyLike, GPUGraphColumn<'uint32'>>
): void {
  // Only called when every topology input and output is a physical GPUVector.
  const topology = propagation.topology as GPUGraphTopology;
  const inputVectors = [
    topology.graph.sourceVertices,
    topology.graph.targetVertices,
    ...(topology.graph.edgeWeights ? [topology.graph.edgeWeights] : []),
    ...(topology.graph.edgeIds ? [topology.graph.edgeIds] : []),
    ...getAdjacencyVectors(topology.forward),
    ...(topology.reverse ? getAdjacencyVectors(topology.reverse) : []),
    topology.invalidEdgeCount
  ];
  const allocations = new Set<Buffer>();
  for (const vector of inputVectors) {
    for (const chunk of vector.data) {
      allocations.add(getPhysicalBuffer(chunk));
    }
  }

  const outputs = [
    {name: 'output', vector: propagation.output as GPUVector<'uint32'>},
    ...(propagation.converged
      ? [{name: 'converged', vector: propagation.converged as GPUVector<'uint32'>}]
      : [])
  ];
  for (const {name, vector} of outputs) {
    const buffer = getPhysicalBuffer(vector.data[0]);
    if (allocations.has(buffer)) {
      throw new Error(`${propagation.id} ${name} must use a distinct physical buffer allocation`);
    }
    allocations.add(buffer);
  }
}

/** Enumerates existing CSR payloads and statuses without changing their source chunk identity. */
function getAdjacencyVectors(
  adjacency: GPUGraphAdjacency
): (GPUVector<'uint32'> | GPUVector<'float32'>)[] {
  return [
    adjacency.offsets,
    adjacency.neighbors,
    adjacency.edgeIds,
    ...(adjacency.edgeWeights ? [adjacency.edgeWeights] : []),
    adjacency.count,
    adjacency.overflow
  ];
}

/** Resolves DynamicBuffer wrappers so non-overlapping slices cannot disguise physical aliasing. */
function getPhysicalBuffer(chunk: GPUData<'uint32'> | GPUData<'float32'>): Buffer {
  return chunk.buffer instanceof DynamicBuffer ? chunk.buffer.buffer : chunk.buffer;
}
