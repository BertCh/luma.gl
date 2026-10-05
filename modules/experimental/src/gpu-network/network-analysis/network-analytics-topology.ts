// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {DynamicBuffer} from '@luma.gl/engine';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraph, type GPUGraphAdjacency, type GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

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
