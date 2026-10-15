import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Network

<ExperimentalDocsTabs active="gpu-network" />

:::caution Experimental
`@luma.gl/experimental/gpu-network` is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::

## Overview

`@luma.gl/experimental/gpu-network` provides analysis contributors for networks stored as CSR
adjacency, passed as one `topology: GPUNetworkTopology` (`offsets`, `neighbors`, `weights`, optional
`nodePositions`): weighted reachability and isochrones (isoband or cell-outline polygons), path
extraction, service areas, neighborhoods, snapping, cost matrices, accessibility scores, subgraph
filters, coarsening, network statistics, node-aligned analytics columns, origin-destination flow
aggregation, edge bundling, adjacency-matrix images, and network construction and analysis from line
data (noding, turn-restricted line graphs, map matching, network K function). Each contributor declares resources and
command nodes into a caller-owned [`GPUCommandGraph`](../gpu-core/gpu-command-graph.md) through
`getCommandNodes(graph)` (`GPUCommandNodeProducer`). It never compiles, submits, encodes, or reads
back.

## When to use it

Use GPU Network for map and road-network tasks that need bounded, renderable results: isochrone
bands that recolor as a cost limit moves, a path that follows a dragged endpoint, a statistics panel
or analytics columns for deck.gl styling, or aggregated flows and bundled edges for drawing.

The generic graph algorithms live in [GPU Graph](../gpu-graph.md), and the network contributors build
on them rather than replace them:

- `GPUNetworkAnalyticsColumns` runs `GPUGraphDegree`, `GPUGraphPageRank`, `GPUGraphCoreNumber`,
  `GPUGraphConnectedComponents`, and `GPUGraphLabelPropagation` directly on the caller's CSR views
  through a `GPUGraphTopologyView`, then normalizes the columns for styling. Call the GPU Graph
  algorithms yourself when you need their raw outputs, convergence status, or an algorithm that has
  no network wrapper, such as `GPUGraphModularityOptimization`.
- `GPUNetworkStatistics` reduces a CSR to one summary row (counts, components, degree histograms,
  optional modularity) for a stats panel; it uses `GPUGraphConnectedComponents` internally.
- `GPUNetworkReachability` is a multi-source, cost-limited shortest-path search with isochrone
  bands and cycle-safe predecessors, all written to caller-owned views. Use
  `GPUGraphSingleSourceShortestPath` when you need plain bounded shortest paths from one source and
  the GPU Graph topology and status contracts.
- `GPUGraphForceLayout` and `GPUGraphSpatialForceLayout` position nodes; `GPUEdgeBundling` then
  turns the straight edges into renderable polylines.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkReachability,
  GPUParameterBuffer,
  type GPUNetworkTopology
} from '@luma.gl/experimental/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'network'});
const startNodes = new GPUParameterBuffer(device, {id: 'starts', format: 'uint32', length: 4});
const costLimit = new GPUParameterBuffer(device, {id: 'cost-limit', format: 'float32', length: 1});
const costBreaks = new GPUParameterBuffer(device, {id: 'cost-breaks', format: 'float32', length: 4});

// `offsets`, `neighbors`, `weights`, `costs`, `bands`, `bandCounts`, and `status.converged` are graph
// views created on `graph`. Build the CSR on the GPU with `GPUCOOToCSR`. One topology object
// feeds every network contributor.
const topology: GPUNetworkTopology = {offsets, neighbors, weights};
graph.add(new GPUNetworkReachability({
  topology,
  startNodes: startNodes.importToGraph(graph),
  costLimit: costLimit.importToGraph(graph),
  costBreaks: costBreaks.importToGraph(graph),
  maximumIterations: 48,
  output: {costs, bands, bandCounts, status: {converged}}
}));
const compiled = await graph.compileAsync(); // once

// every frame: rewrite the start node and cost limit, encode the same compiled graph
startNodes.write(new Uint32Array([startNode, 0, 0, 0]));
costLimit.write(new Float32Array([1800]));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

`encode()` only records; submit with `device.submit()`. [Running contributors](../gpu-spatial-analysis/concepts.md#running-contributors)
covers `compileAsync()`, rebuilding without a stall, submission and why `parameters` is `undefined`.

## Concepts and operators

- [GPU Network concepts](./concepts.md): conventions, units and the GPU execution model.
- [GPU Network operators](./operators.md): every contributor, its inputs, outputs, parameters and how it
  computes.

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-network}

## Limits and compatibility

- GPU Network is experimental and WebGPU-only.
- CSR graphs use packed views only. Edge weights must be non-negative where a contributor takes costs.
- Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
  (`dev-docs/roadmaps/gpugraph-roadmap.md`) tracks what is still open.

## Related modules

- [GPU Graph](../gpu-graph.md) supplies the generic algorithms, topology views, and layouts.
- [GPU Raster](../gpu-raster/README.md) has the raster counterpart of cost distance and isolines.
- [GPU Core](../gpu-core/concepts.md) defines contributors, composition levels, and graph ownership.
