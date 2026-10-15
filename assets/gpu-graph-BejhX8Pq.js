var e=`import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';
import {ClientOnlyLiveExample} from '@site/src/components/docs/client-only-live-example';
import {GPUExampleCard} from '@site/src/components/docs/gpu-example-card';
import {GPUGraphExplorerExample} from '@site/src/examples';

# GPU Graph

<ExperimentalDocsTabs active="gpu-graph" />

## Overview

\`@luma.gl/gpgpu/gpu-graph\` provides graph-data structures and algorithms that compose inside a
caller-owned \`GPUCommandGraph\`. It keeps adjacency, traversal frontiers, scores, partitions, and
layouts GPU-resident so analysis can feed another analysis or a renderer without downloading the
graph through JavaScript.

## When to use it

Use GPU Graph for repeated analytics over a graph already stored on the GPU, especially when results
will drive filtering, styling, layout, or later graph operations. For a single small graph or an
algorithm whose result is immediately needed on the CPU, a CPU graph library may be simpler.

## Quick start

\`\`\`ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUGraph, GPUGraphPageRank, GPUGraphTopology} from '@luma.gl/gpgpu/gpu-graph';

// sourceVertices, targetVertices and ranks are existing packed GPUVector columns.
// forward and reverse are caller-owned GPUGraphAdjacency allocations:
// offsets[V + 1], neighbors[E], edgeIds[E], count[1], overflow[1].
const graph = new GPUCommandGraph(device, {id: 'graph-analysis'});
const railGraph = new GPUGraph({vertexCount, sourceVertices, targetVertices, directed: true});
const railTopology = new GPUGraphTopology({
  graph: railGraph, forward, reverse, invalidEdgeCount // uint32 scalar
});
railTopology.addToGraph(graph);
const pageRank = new GPUGraphPageRank({topology: railTopology, output: ranks});
pageRank.addToGraph(graph);
const compiledGraph = graph.compile();
\`\`\`

The application compiles, encodes, submits, and decides whether a bounded result should be read
back. Operations contribute nodes and graph views only.

## Core concepts and data model

- \`GPUGraph\` describes canonical vertex and edge identity.
- \`GPUGraphTopology\` builds reusable adjacency views for algorithms that need neighbors.
- Algorithms publish ordinary graph buffers and bounded status so their outputs compose.
- Iterative operations expose capacity, convergence, and incomplete-result behavior explicitly.
- Weighted, directed, and undirected semantics are operation-specific and documented in the
  operations reference.

## GPU execution model

Topology and algorithms contribute GPU passes to one caller-owned command graph, which the application compiles and submits.

{/* lens:execution-model */}

::lens{scenario=gpu-graph.graph-page-rank rung=graph caption="Inspect the dependencies between CSR construction and PageRank's bounded iterations."}
::lens{scenario=gpu-graph.graph-page-rank rung=kernel caption="Inspect actual dispatch dimensions, storage bindings, and recorded WGSL for a PageRank pass."}

## Explore a live graph

<GPUExampleCard
  demonstrates={['topology', 'GPU analytics', 'selection', 'progressive layout']}
  input="A canonical vertex/edge graph with GPU-resident adjacency"
  gpuOutput="Scores, selections, layout positions, and renderable graph views"
  cpuReadback="Only compact inspector and status results"
  execution="Reusable graph with progressive layout updates"
  compatibility="WebGPU"
  fullPageHref="/examples/experimental/gpu-graph-explorer"
  sourceHref="https://github.com/visgl/luma.gl/tree/master/examples/experimental/gpu-graph-explorer"
  inspectorHref="/examples/experimental/gpu-graph-explorer?panel=graph"
/>

<ClientOnlyLiveExample>
  <GPUGraphExplorerExample embedded />
</ClientOnlyLiveExample>

## Operations and API index

| Family | Operations |
| --- | --- |
| Graph and topology | \`GPUGraph\`, \`GPUGraphTopology\`, \`GPUGraphDegree\` |
| Traversal and paths | \`GPUGraphBreadthFirstSearch\`, \`GPUGraphSingleSourceShortestPath\` |
| Connectivity and communities | \`GPUGraphConnectedComponents\`, \`GPUGraphCoreNumber\`, \`GPUGraphLabelPropagation\`, \`GPUGraphModularityOptimization\` |
| Metrics and ranking | \`GPUGraphLocalClusteringCoefficient\`, \`GPUGraphModularity\`, \`GPUGraphPageRank\` |
| Layout | \`GPUGraphForceLayout\`, \`GPUGraphSpatialForceLayout\` |

The [operations reference](./gpu-graph-operations) documents usage, buffers, execution, capacity,
validation, and performance for each family.

## Catalog

::lens-catalog{entry=gpu-graph}

## Limits and compatibility

- GPU Graph is experimental and WebGPU-only.
- Algorithms use fixed compiled capacities and report convergence or incomplete work explicitly.
- The caller owns source buffers, submission, readback, and cancellation.
- Algorithm support for weights, directionality, self-edges, and parallel edges is documented per
  operation.

## Related modules

- [GPU Core](/docs/api-reference/experimental/gpu-core) provides scheduling and generic GPU
  operations.
- [GPU Dataframe](./gpu-dataframe) analyzes columnar records rather than graph topology.
- [GPU Trace](./gpu-trace) adds trace-specific dependency and hierarchy semantics.
`;export{e as default};