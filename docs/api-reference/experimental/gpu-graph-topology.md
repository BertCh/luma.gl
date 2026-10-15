---
title: GPU Graph topology
description: Describe GPU-resident edges, build reusable adjacency, and count vertex relationships.
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Graph topology

<ExperimentalDocsTabs active="gpu-graph-topology" />

## Describe existing relationships with GPUGraph

**Question: Which existing GPU columns describe the people, accounts, services, or documents in
this network?**

`GPUGraph` is the ownership-preserving entry point. Construct it when an application already has
aligned `GPUVector<'uint32'>` source and target identifiers and knows how many vertices exist,
including isolated vertices that never appear in an edge.

```ts
import {GPUGraph} from '@luma.gl/gpgpu/gpu-graph';

const graph = new GPUGraph({
  vertexCount,
  sourceVertices,
  targetVertices,
  edgeWeights,
  edgeIds,
  nodeAttributes,
  directed: true
});
```

The graph borrows its vectors; it does not allocate a graph buffer, copy or concatenate chunks,
submit commands, or take ownership of source allocations. Source and target chunks must have the
same ordered lengths. Optional stable edge identifiers and `float32` edge weights follow the same
source partitions, while optional vertex and edge property tables retain their existing metadata.

Use this lightweight representation when existing GPU tables or render inputs already describe a
network. It is a description, not an upload helper: first create or adapt your GPU vectors through
the application or the appropriate data adapter.

## Build reusable adjacency with GPUGraphTopology

Rebuild compressed sparse row adjacency from GPU edge columns, publishing per-vertex offsets, neighbors and explicit overflow status.

::lens{scenario=gpu-graph.graph-topology rung=result caption="Read each station's CSR start offset across 485 stations; 1,411 directed edges become adjacency entries."}

**Question: Given a particular vertex, which other vertices does it connect to?**

An edge list answers “what are all relationships?” but repeatedly scanning every edge to discover
one vertex's neighbors is expensive. `GPUGraphTopology` builds compressed sparse row (CSR) adjacency
once so later operations can find each vertex's neighbor interval from adjacent offsets.

For example, a transaction list might contain millions of transfers while an investigator wants
only the accounts directly connected to account 42. Its CSR offset interval identifies that
account's neighbors without asking each later analysis to rescan the entire edge list.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | `string` | `'gpu-graph-topology'` | Prefix for generated command-graph nodes and graph-owned scratch. |
| `graph` | `GPUGraph` | Required | Existing source and target columns, preserving chunk boundaries and metadata. |
| `forward` | `GPUGraphAdjacency` | Required | Caller-owned forward offsets, neighbors, edge identifiers, count and overflow. |
| `reverse` | `GPUGraphAdjacency` | Omitted | Optional caller-owned transposed adjacency; directed graphs only. |
| `invalidEdgeCount` | `GPUVector<'uint32'>` | Required | One caller-owned row receiving the number of invalid source edge endpoints. |

Every output is an existing, caller-owned, single-chunk `GPUVector<'uint32'>`. Offsets have
`vertexCount + 1` rows; neighbors and edge identifiers have equal explicit capacities; `count`,
`overflow`, and `invalidEdgeCount` each have one row. When the source graph supplies edge weights,
each configured adjacency also requires a matching `float32` edge-weight output.

Build reverse adjacency when a directed graph needs incoming-degree queries, incoming or
bidirectional breadth-first search, weak-neighborhood community, core-number, or local-clustering
analysis, or PageRank. Directed weak components use forward adjacency alone, as do outgoing
weighted shortest paths. Modularity reads original edge columns directly and needs no adjacency.
Undirected graphs use one symmetric forward adjacency and must not provide reverse adjacency.

Invalid endpoints are excluded and counted. `count` reports the complete number of accepted
adjacency entries even if neighbor capacity is insufficient; `overflow` makes truncation explicit.
Neighbor order within each vertex is intentionally unspecified.

#### How it computes

{/* lens:GPUGraphTopology */}

::lens{scenario=gpu-graph.graph-topology rung=kernel caption="Inspect the degree count, scan, finalize and scatter kernels that fill each CSR direction."}
::lens{scenario=gpu-graph.graph-topology rung=graph caption="Follow the degree, exclusive-scan and scatter passes from edge columns to offsets and neighbors."}
::lens{scenario=gpu-graph.graph-topology rung=datum datum=409 caption="Inspect Utrecht Centraal, station 409, with its offset interval and neighbors."}
::lens{scenario=gpu-graph.graph-topology rung=formula caption="Follow the exclusive scan of degrees into offsets to the WGSL buffers that hold it."}

## Analyze in-graph CSR views with GPUGraphTopologyView

**Question: The adjacency already exists inside my command graph. Can I analyze it without
copying it into `GPUVector`s?**

`GPUGraphTopologyView` presents compressed sparse row adjacency that already exists as
`GPUCommandGraph` views, such as transients written by `GPUCOOToCSR`, to `GPUGraphDegree`,
`GPUGraphPageRank`, `GPUGraphCoreNumber`, `GPUGraphConnectedComponents`, and
`GPUGraphLabelPropagation`. Their outputs may then also be graph views. Nothing is copied.

```ts
import {GPUGraphPageRank, GPUGraphTopologyView} from '@luma.gl/gpgpu/gpu-graph';

const topology = new GPUGraphTopologyView({
  vertexCount,
  forward: {offsets: rowOffsets, neighbors: columns}
});
new GPUGraphPageRank({topology, output: rankView}).addToGraph(commandGraph);
```

`directed` defaults to whether `reverse` adjacency is supplied. Omit `overflow` for an exact CSR;
each algorithm then clears its own zero word in a one-invocation pass. View packing and length are
checked at construction; graph membership and output-versus-input buffer distinctness are checked
when the algorithm is added to the graph. Other gpu-graph algorithms (breadth-first search,
shortest paths, modularity, clustering, layouts) still take a `GPUGraphTopology`.

## Count relationships with GPUGraphDegree

Count outgoing or incoming relationships per vertex from CSR offsets, exact even when neighbor storage overflows.

::lens{scenario=gpu-graph.graph-degree rung=result caption="Compare outgoing connections across 485 stations; the busiest stations have 15."}

**Question: How many direct relationships does each vertex have?**

`GPUGraphDegree` answers the simplest structural question: how many outgoing or incoming
relationships does each vertex have? Use it to identify network hubs, size junction markers,
detect isolated accounts, or find unusually connected infrastructure and dependency nodes.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | `string` | `'gpu-graph-degree'` | Prefix for generated command-graph node and imported-resource identifiers. |
| `topology` | `GPUGraphTopology` or `GPUGraphTopologyView` | Required | Existing GPU topology with exact, untruncated CSR offsets. |
| `output` | `GPUVector<'uint32'>` or `GraphDataView<'uint32'>` | Required | Caller-owned packed unsigned degree row for every vertex. |
| `direction` | `'outgoing'` or `'incoming'` | `'outgoing'` | Edge orientation counted; directed incoming degree requires reverse adjacency. |

Its caller-owned output has one packed `uint32` row per vertex. Outgoing degree is the default;
incoming degree on a directed graph requires reverse adjacency. Duplicate edges count individually,
and an undirected self-loop counts once.

Degrees come from complete CSR offsets rather than the capacity-bounded neighbor list, so they
remain exact even when the corresponding adjacency reports neighbor overflow. Degree is useful
when raw connectivity is the question; it does not account for whether a vertex's neighbors are
themselves important.

#### How it computes

{/* lens:GPUGraphDegree */}

::lens{scenario=gpu-graph.graph-degree rung=kernel caption="Inspect the single kernel that subtracts adjacent offsets for every vertex."}
::lens{scenario=gpu-graph.graph-degree rung=graph caption="Follow topology construction into the one degree pass that reads its offsets."}
::lens{scenario=gpu-graph.graph-degree rung=datum datum=409 caption="Inspect Utrecht Centraal, station 409, with its offset interval and degree of 15."}
::lens{scenario=gpu-graph.graph-degree rung=formula caption="Follow the offset difference to the WGSL buffers that hold it."}

## Related pages

- [GPU Graph overview](/docs/api-reference/experimental/gpu-graph)
- [GPU Graph operations index](/docs/api-reference/experimental/gpu-graph-operations)
- [GPU Core](/docs/api-reference/experimental/gpu-core)
