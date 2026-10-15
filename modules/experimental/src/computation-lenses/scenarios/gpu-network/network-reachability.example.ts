// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkReachability, type GPUNetworkTopology} from '@luma.gl/experimental/gpu-network';

/** Uploaded inputs; structureOnly selects synchronous compilation without execution. */
export type NetworkReachabilityExampleInput = {
  graph: GPUCommandGraph;
  roads: GPUNetworkTopology;
  startNodes: GraphDataView<'uint32'>;
  activeIterations: GraphDataView<'uint32'>;
  costBreaks: GraphDataView<'float32'>;
  localIterations?: number;
  structureOnly?: boolean;
};

/** Builds the same public contributor shown in the card; submission belongs to the lens. */
export async function runExample(_device: Device, input: NetworkReachabilityExampleInput) {
  const {graph, roads, startNodes, activeIterations, costBreaks, localIterations} = input;
  const id = 'reachability';
  const nodeCount = roads.offsets.length - 1;
  // #region example
  const output = {
    costs: createTransientView(graph, 'costs', 'float32', nodeCount),
    bands: createTransientView(graph, 'bands', 'uint32', nodeCount),
    predecessors: createTransientView(graph, 'predecessors', 'uint32', nodeCount),
    bandCounts: createTransientView(graph, 'bandCounts', 'uint32', 5),
    status: {
      converged: createTransientView(graph, 'converged', 'uint32', 1),
      iterationCount: createTransientView(graph, 'iterationCount', 'uint32', 1)
    }
  };
  const reachability = new GPUNetworkReachability({
    id,
    topology: roads,
    startNodes,
    activeIterations,
    costBreaks,
    maximumIterations: 256,
    localIterations,
    output
  });
  const {costs, bands, predecessors} = reachability.props.output;
  // #endregion example
  // Null devices cannot create compute pipelines, so structure-only builds skip compilation callbacks.
  const commandNodes = reachability.getCommandNodes(graph);
  graph.add(input.structureOnly ? commandNodes.map(withoutCompilation) : commandNodes);
  const compiled = input.structureOnly ? graph.compile() : await graph.compileAsync();
  return {compiled, commandNodes, reachability, costs, bands, predecessors, output};
}

// Null devices support scheduling and resource allocation, but cannot create compute pipelines.
// Keep the original annotations and resources: only executable compilation callbacks are omitted.
function withoutCompilation(node: GPUCommandNode): GPUCommandNode {
  const {compile: _compile, compileAsync: _compileAsync, ...structure} = node;
  return {
    ...structure,
    compile: () => ({
      encode: () => {
        throw new Error('Structure-only graph cannot execute');
      }
    })
  };
}
