// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkNoding} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs: centreline vertices in local metres, line offsets and the snap tolerance. */
export type NetworkNodingExampleInput = {
  graph: GPUCommandGraph;
  linePositions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  tolerance: GraphDataView<'float32'>;
};

/** Builds the contributor shown on the card; the lens compiles, executes and reads it back. */
export async function runExample(_device: Device, input: NetworkNodingExampleInput) {
  const {graph, linePositions, lineOffsets, tolerance} = input;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length);
  // Capacities: 512 pieces and nodes, 2,048 piece vertices (the lens fails loudly on overflow).
  const output = {
    pieces: {
      geometry: {
        kind: 'lines' as const,
        positions: view('pieceVertices', 'float32x2', 2048),
        lineOffsets: view('pieceOffsets', 'uint32', 513)
      },
      sourceIds: view('pieceSourceIds', 'uint32', 512),
      status: {count: view('pieceCount', 'uint32', 1), overflow: view('pieceOverflow', 'uint32', 1)}
    },
    nodes: {
      positions: view('nodingPositions', 'float32x2', 512),
      status: {count: view('nodeCount', 'uint32', 1)}
    },
    edges: {
      fromNodes: view('edgeFrom', 'uint32', 512),
      toNodes: view('edgeTo', 'uint32', 512),
      lengths: view('edgeLengths', 'float32', 512)
    },
    topology: {
      offsets: view('nodingOffsets', 'uint32', 513),
      neighbors: view('nodingNeighbors', 'uint32', 1024),
      weights: view('nodingWeights', 'float32', 1024)
    },
    status: {overflow: view('nodingOverflow', 'uint32', 1)}
  };
  // #region example
  const noding = new GPUNetworkNoding({
    lines: {kind: 'lines', positions: linePositions, lineOffsets},
    intersectionCapacity: 256,
    tolerance,
    output
  });
  graph.add(noding);
  const {pieces, nodes, edges, topology} = noding.props.output;
  // #endregion example
  return {noding, output, pieces, nodes, edges, topology};
}
