// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {Computation} from '@luma.gl/engine';
import type {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '../gpu-core/gpu-command-graph';
import {GraphDataView} from '../gpu-core/gpu-command-graph-types';
import {
  createTransientView,
  getViewBinding,
  getViewElementOffset,
  validatePackedView
} from '../gpu-core/graph-data-view-utils';
import type {GPUGraphTopology} from './gpu-graph-topology';

const MAXIMUM_UINT32 = 0xffffffff;

/** One compressed sparse row adjacency direction expressed as command-graph views. */
export type GPUGraphAdjacencyView = {
  /** Exclusive packed `uint32` row offsets with `vertexCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** Packed `uint32` adjacent vertex per edge. */
  neighbors: GraphDataView<'uint32'>;
  /**
   * Optional one-row packed `uint32` overflow flag. Algorithms fail closed when it is nonzero, as
   * with {@link GPUGraphTopology} capacity overflow. Omit it for an exact CSR, such as one built
   * by `GPUCOOToCSR`; each algorithm then clears its own graph-owned zero word.
   */
  overflow?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUGraphTopologyView}. */
export type GPUGraphTopologyViewProps = {
  /** Prefix used in validation messages. Defaults to `'gpu-graph-topology-view'`. */
  id?: string;
  /** Number of graph vertices. */
  vertexCount: number;
  /**
   * Whether edges are directed. Defaults to `Boolean(reverse)`. Without a reverse adjacency the
   * forward adjacency of an undirected graph must already be symmetric.
   */
  directed?: boolean;
  /** Forward adjacency. */
  forward: GPUGraphAdjacencyView;
  /** Optional transposed adjacency, required by incoming-edge algorithms on directed graphs. */
  reverse?: GPUGraphAdjacencyView;
};

/**
 * A compressed sparse row topology given as views of one `GPUCommandGraph`, for adjacency that
 * lives only inside a graph, such as a transient CSR built by `GPUCOOToCSR`.
 *
 * `GPUGraphDegree`, `GPUGraphPageRank`, `GPUGraphCoreNumber`, `GPUGraphConnectedComponents` and
 * `GPUGraphLabelPropagation` accept it in place of a {@link GPUGraphTopology}; their outputs may
 * then also be views. The views must belong to the command graph the algorithm is added to.
 * Nothing is copied: the algorithms bind the views directly and the graph tracks their hazards.
 */
export class GPUGraphTopologyView {
  /** Prefix used in validation messages. */
  readonly id: string;
  /**
   * Vertex count, directedness, and forward adjacency entry count, mirroring the `GPUGraph`
   * fields that view-capable algorithms read.
   */
  readonly graph: {
    readonly vertexCount: number;
    readonly directed: boolean;
    readonly edgeCount: number;
  };
  /** Forward adjacency views. */
  readonly forward: GPUGraphAdjacencyView;
  /** Optional reverse adjacency views. */
  readonly reverse?: GPUGraphAdjacencyView;

  /** Validates view formats and lengths without allocating or submitting work. */
  constructor(props: GPUGraphTopologyViewProps) {
    this.id = props.id ?? 'gpu-graph-topology-view';
    const directed = props.directed ?? Boolean(props.reverse);
    if (
      !Number.isSafeInteger(props.vertexCount) ||
      props.vertexCount < 0 ||
      props.vertexCount >= MAXIMUM_UINT32
    ) {
      throw new Error(`${this.id} vertexCount must be a non-negative uint32`);
    }
    if (props.reverse && !directed) {
      throw new Error(`${this.id} reverse adjacency requires a directed graph`);
    }
    validateAdjacencyView(props.forward, props.vertexCount, `${this.id} forward`);
    if (props.reverse) {
      validateAdjacencyView(props.reverse, props.vertexCount, `${this.id} reverse`);
    }
    this.graph = Object.freeze({
      vertexCount: props.vertexCount,
      directed,
      edgeCount: props.forward.neighbors.length
    });
    this.forward = props.forward;
    this.reverse = props.reverse;
  }
}

/** A gpu-graph algorithm column given as a physical `GPUVector` or a command-graph view. */
export type GPUGraphColumn<Format extends 'uint32' | 'float32'> =
  | GPUVector<Format>
  | GraphDataView<Format>;

/** Topology accepted by view-capable gpu-graph algorithms. */
export type GPUGraphTopologyLike = GPUGraphTopology | GPUGraphTopologyView;

/** Returns whether a column is a command-graph view rather than a physical vector. @internal */
export function isGPUGraphViewColumn<Format extends 'uint32' | 'float32'>(
  column: GPUGraphColumn<Format>
): column is GraphDataView<Format> {
  return column instanceof GraphDataView;
}

/**
 * Imports a physical column exactly as `importGPUVector(id, vector).data[0]`, or returns a view
 * column after checking that it belongs to `commandGraph`.
 *
 * @internal
 */
export function importGPUGraphColumn<Format extends 'uint32' | 'float32', Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  id: string,
  column: GPUGraphColumn<Format>
): GraphDataView<Format> {
  if (isGPUGraphViewColumn(column)) {
    if (column.buffer.graph !== (commandGraph as unknown)) {
      throw new Error(`${id} view must belong to command graph ${commandGraph.id}`);
    }
    return column;
  }
  return commandGraph.importGPUVector(id, column).data[0];
}

/**
 * Imports an adjacency overflow flag, or creates a graph-owned zero word and a one-invocation pass
 * that clears it when a view adjacency omits the flag.
 *
 * @internal
 */
export function importGPUGraphOverflow<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  id: string,
  overflow: GPUGraphColumn<'uint32'> | undefined
): GraphDataView<'uint32'> {
  if (overflow) {
    return importGPUGraphColumn(commandGraph, id, overflow);
  }
  const view = createTransientView(commandGraph, `${id}-exact`, 'uint32', 1);
  const source = /* wgsl */ `
const OVERFLOW_OFFSET: u32 = ${getViewElementOffset(view)}u;
@group(0) @binding(0) var<storage, read_write> overflow: array<u32>;

@compute @workgroup_size(1)
fn main() {
  overflow[OVERFLOW_OFFSET] = 0u;
}`;
  commandGraph.addComputePass({
    id: `${id}-exact-clear`,
    resources: [{buffer: view, usage: 'storage-write'}],
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: `${id}-exact-clear`,
        source,
        shaderLayout: {bindings: [{name: 'overflow', type: 'storage', group: 0, location: 0}]}
      });
      return {
        encode: ({computePass, getBuffer}) => {
          computation.setBindings({overflow: getViewBinding(view, getBuffer)});
          computation.dispatch(computePass, 1, 1, 1);
        },
        destroy: () => computation.destroy()
      };
    }
  });
  return view;
}

/** Requires a packed view column with exactly `length` rows. @internal */
export function validateGPUGraphViewColumn(
  view: GraphDataView,
  format: 'uint32' | 'float32',
  length: number,
  name: string
): void {
  validatePackedView(view, [format], name);
  if (view.length !== length) {
    throw new Error(`${name} must contain exactly ${length} ${format} rows`);
  }
}

/**
 * Rejects outputs that share a command-graph buffer handle with an input or another output.
 *
 * View-based topologies are validated after import, where physical and view columns both resolve
 * to graph handles. Physical-only algorithms keep their constructor-time allocation checks.
 *
 * @internal
 */
export function validateDistinctGPUGraphHandles(
  id: string,
  inputs: readonly GraphDataView[],
  outputs: readonly {name: string; view: GraphDataView}[]
): void {
  const handles = new Set(inputs.map(view => view.buffer));
  for (const {name, view} of outputs) {
    if (handles.has(view.buffer)) {
      throw new Error(`${id} ${name} must use a buffer distinct from topology inputs and outputs`);
    }
    handles.add(view.buffer);
  }
}

/**
 * Returns whether an algorithm runs on any view column and so validates graph handles after import.
 *
 * @internal
 */
export function usesGPUGraphViews(
  topology: GPUGraphTopologyLike,
  columns: readonly (GPUGraphColumn<'uint32'> | GPUGraphColumn<'float32'> | undefined)[]
): boolean {
  return (
    topology instanceof GPUGraphTopologyView ||
    columns.some(column => column !== undefined && isGPUGraphViewColumn(column))
  );
}

/** Validates one adjacency direction's view formats and lengths. */
function validateAdjacencyView(
  adjacency: GPUGraphAdjacencyView,
  vertexCount: number,
  name: string
): void {
  validateGPUGraphViewColumn(adjacency.offsets, 'uint32', vertexCount + 1, `${name} offsets`);
  validatePackedView(adjacency.neighbors, ['uint32'], `${name} neighbors`);
  if (adjacency.overflow) {
    validateGPUGraphViewColumn(adjacency.overflow, 'uint32', 1, `${name} overflow`);
  }
  const handles = [adjacency.offsets.buffer, adjacency.neighbors.buffer];
  if (adjacency.overflow) {
    handles.push(adjacency.overflow.buffer);
  }
  if (handles.some(handle => handle.graph !== handles[0].graph)) {
    throw new Error(`${name} views must belong to one command graph`);
  }
}
