// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/engine';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraph, type GPUGraphAdjacency, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';
import {validatePackedView} from '../utils/gpu-contributor-utils';

/**
 * A network in compressed sparse row (CSR) form, the one network input of every `gpu-network`
 * contributor.
 *
 * Node `u`'s outgoing edges occupy slots `[offsets[u], offsets[u + 1])` of `neighbors` and
 * `weights`, so `offsets` has `nodeCount + 1` rows and `neighbors.length` is the edge capacity.
 * An undirected network lists every edge in both directions.
 *
 * The CSR part (`offsets`, `neighbors`, `weights`) has the names and formats of
 * `GPUSpatialWeights`, so a topology can be passed wherever spatial weights are read. The weights
 * invariants (ascending neighbor IDs, no self pairs) are not required of a network; a consumer that
 * relies on them states it.
 */
export type GPUNetworkTopology = {
  /** `nodeCount + 1` exclusive row offsets into the edge slots. */
  offsets: GraphDataView<'uint32'>;
  /** Target node per edge slot. Targets `>= nodeCount` are skipped by every contributor. */
  neighbors: GraphDataView<'uint32'>;
  /** Travel cost per edge slot, aligned with `neighbors`. Negative or NaN costs are impassable. */
  weights: GraphDataView<'float32'>;
  /** Optional position per node, required by the contributors that measure geometry. */
  nodePositions?: GraphDataView<'float32x2'>;
};

/**
 * A {@link GPUNetworkTopology} read for its connectivity, where edge weights are optional. Every
 * `GPUNetworkTopology` is a `GPUNetworkAdjacency`, so one network object feeds every contributor.
 */
export type GPUNetworkAdjacency = Omit<GPUNetworkTopology, 'weights'> & {
  /** Optional travel cost or weight per edge slot, aligned with `neighbors`. */
  weights?: GraphDataView<'float32'>;
};

/** Which optional topology fields a contributor requires. @internal */
export type GPUNetworkTopologyRequirements = {
  /** Require `weights`. */
  weights?: boolean;
  /** Require `nodePositions`. */
  nodePositions?: boolean;
};

/** Node and edge-slot counts of a validated topology. @internal */
export type GPUNetworkTopologyShape = {
  /** `offsets.length - 1`. */
  nodeCount: number;
  /** `neighbors.length`, the edge-slot capacity. */
  edgeCount: number;
};

/**
 * Validates the formats and lengths of a network topology and returns its node and edge counts.
 *
 * @internal
 * @throws If a view is not packed in its format, the topology has no node, or `weights` or
 * `nodePositions` are misaligned or missing while required.
 */
export function validateGPUNetworkTopology(
  id: string,
  topology: GPUNetworkAdjacency,
  requirements: GPUNetworkTopologyRequirements = {}
): GPUNetworkTopologyShape {
  if (!topology) {
    throw new Error(`${id} topology is required`);
  }
  validatePackedView(topology.offsets, ['uint32'], `${id} topology.offsets`);
  validatePackedView(topology.neighbors, ['uint32'], `${id} topology.neighbors`);
  const nodeCount = topology.offsets.length - 1;
  const edgeCount = topology.neighbors.length;
  if (nodeCount < 1) {
    throw new Error(`${id} topology.offsets must contain at least two rows`);
  }
  if (topology.weights) {
    validatePackedView(topology.weights, ['float32'], `${id} topology.weights`);
    if (topology.weights.length !== edgeCount) {
      throw new Error(`${id} topology.weights length must equal topology.neighbors length`);
    }
  } else if (requirements.weights) {
    throw new Error(`${id} topology.weights is required`);
  }
  if (topology.nodePositions) {
    validatePackedView(topology.nodePositions, ['float32x2'], `${id} topology.nodePositions`);
    if (topology.nodePositions.length !== nodeCount) {
      throw new Error(`${id} topology.nodePositions must contain one row per node`);
    }
  } else if (requirements.nodePositions) {
    throw new Error(`${id} topology.nodePositions is required`);
  }
  return {nodeCount, edgeCount};
}

/** Returns the views of a topology, for graph-membership and aliasing checks. @internal */
export function getGPUNetworkTopologyViews(
  topology: GPUNetworkAdjacency
): (GraphDataView | undefined)[] {
  return [topology.offsets, topology.neighbors, topology.weights, topology.nodePositions];
}

/**
 * Returns the physical buffer behind a graph view's default import.
 *
 * @internal
 * @throws If the view was not imported with a default buffer, as transient views are not.
 */
export function getGraphViewDefaultBuffer(
  id: string,
  name: string,
  view: GraphDataView
): Buffer | DynamicBuffer {
  const defaultBuffer = view.buffer.defaultBuffer;
  if (!defaultBuffer) {
    throw new Error(`${id} ${name} must be imported with a default buffer`);
  }
  return defaultBuffer;
}

/**
 * Resolves a default import to its current core buffer, unwrapping a `DynamicBuffer`.
 *
 * @internal
 */
export function getCoreDefaultBuffer(buffer: Buffer | DynamicBuffer): Buffer {
  return buffer instanceof DynamicBuffer ? buffer.buffer : buffer;
}

/**
 * Wraps a packed, imported graph view in a `GPUVector` over the same physical buffer.
 *
 * `GPUCommandGraph.importGPUVector` de-duplicates on the physical buffer, so gpu-graph algorithms
 * that import this vector resolve to the handle the caller already imported and hazards stay
 * tracked between them and the contributor's own nodes.
 *
 * @internal
 * @throws If the view was not imported with a default buffer.
 */
export function getGraphViewGPUVector<Format extends 'uint32' | 'float32'>(
  id: string,
  name: string,
  view: GraphDataView<Format>
): GPUVector<Format> {
  return new GPUVector<Format>({
    type: 'buffer',
    name,
    buffer: getGraphViewDefaultBuffer(id, name, view),
    format: view.format,
    length: view.length,
    byteOffset: view.byteOffset
  });
}

/** Properties for {@link createNetworkAnalyticsTopology}. @internal */
export type NetworkAnalyticsTopologyProps = {
  /** Contributor ID used in error messages. */
  id: string;
  /** Number of nodes. */
  nodeCount: number;
  /** Forward CSR offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** Forward CSR neighbors. */
  neighbors: GraphDataView<'uint32'>;
  /** Reverse CSR offsets; presence makes the graph directed. */
  reverseOffsets?: GraphDataView<'uint32'>;
  /** Reverse CSR neighbors; required with `reverseOffsets`. */
  reverseNeighbors?: GraphDataView<'uint32'>;
  /** Contributor-owned zero-filled uint32 buffer used as the always-zero count and overflow word. */
  status: Buffer;
};

/**
 * Presents an already-built CSR road network as a `GPUGraphTopology` for gpu-graph algorithms.
 *
 * Every gpu-graph algorithm reads only `graph.vertexCount`, `graph.directed`, and
 * `forward`/`reverse` `offsets`, `neighbors`, and `overflow`. The other fields only feed those
 * classes' allocation-disjointness checks. `overflow` points at a contributor-owned zero word because
 * a caller-built CSR is exact.
 *
 * @internal
 */
export function createNetworkAnalyticsTopology(
  props: NetworkAnalyticsTopologyProps
): GPUGraphTopology {
  const {id, nodeCount, status} = props;
  const forwardNeighbors = getGraphViewGPUVector(id, 'neighbors', props.neighbors);
  const statusVector = new GPUVector<'uint32'>({
    type: 'buffer',
    name: `${id}-status`,
    buffer: status,
    format: 'uint32',
    length: 1
  });
  const createAdjacency = (
    offsets: GraphDataView<'uint32'>,
    neighbors: GPUVector<'uint32'>,
    name: string
  ): GPUGraphAdjacency => ({
    offsets: getGraphViewGPUVector(id, name === 'forward' ? 'offsets' : 'reverseOffsets', offsets),
    neighbors,
    edgeIds: neighbors,
    count: statusVector,
    overflow: statusVector
  });
  const forward = createAdjacency(props.offsets, forwardNeighbors, 'forward');
  const reverse =
    props.reverseOffsets && props.reverseNeighbors
      ? createAdjacency(
          props.reverseOffsets,
          getGraphViewGPUVector(id, 'reverseNeighbors', props.reverseNeighbors),
          'reverse'
        )
      : undefined;
  const graph = new GPUGraph({
    vertexCount: nodeCount,
    sourceVertices: forwardNeighbors,
    targetVertices: forwardNeighbors,
    directed: Boolean(reverse)
  });
  // The real `GPUGraphTopology` constructor validates the outputs of a topology *rebuild*: it
  // demands distinct physical allocations per adjacency field and a rebuild-sized `invalidEdgeCount`.
  // This adapter deliberately aliases fields (edge IDs reuse neighbors, count and overflow share one
  // status word) and never rebuilds, so it supplies the structural shape the algorithms read.
  return {
    id: `${id}-topology`,
    graph,
    forward,
    reverse,
    invalidEdgeCount: statusVector,
    addToGraph: () => {
      throw new Error(`${id} topology adapter is read-only and cannot rebuild adjacency`);
    }
  } as unknown as GPUGraphTopology;
}
