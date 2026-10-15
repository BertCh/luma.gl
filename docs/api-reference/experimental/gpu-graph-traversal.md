---
title: GPU Graph traversal and pathfinding
description: Breadth-first neighborhoods and nonnegative weighted shortest paths on GPU-resident graphs.
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Graph traversal and pathfinding

<ExperimentalDocsTabs active="gpu-graph-traversal" />

## Follow unweighted paths with GPUGraphBreadthFirstSearch

Expand outward from seed vertices, publishing shortest hop counts and lowest-identifier parents for every reachable vertex.

::lens{scenario=gpu-graph.graph-breadth-first-search rung=result caption="Follow hop distance from Utrecht Centraal; all 485 stations are reachable within 15 hops."}

**Question: Which entities can I reach within a chosen number of hops, and what shortest path gets
me there?**

`GPUGraphBreadthFirstSearch` expands outward from one or more selected vertices and records the
shortest unweighted hop count to every reachable vertex. Use it to highlight a selected account's
neighborhood, follow a service's dependencies, or explain how two entities connect.

For example, searching two hops from an account finds both its direct counterparties and the
counterparties of those counterparties. Choose breadth-first search over degree when the question
depends on indirect relationships; choose it over connected components when distance, direction,
or a particular starting vertex matters.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | `string` | `'gpu-graph-breadth-first-search'` | Prefix for generated command-graph nodes and imported resources. |
| `topology` | `GPUGraphTopology` | Required | Existing GPU adjacency and its explicit overflow status. |
| `seeds` | `GPUVector<'uint32'>` | Required | Stable source vertex identifiers, preserving every seed chunk. |
| `distances` | `GPUVector<'uint32'>` | Required | Caller-owned unsigned hop-distance row for every vertex. |
| `predecessors` | `GPUVector<'uint32'>` | Required | Caller-owned unsigned predecessor row for every vertex. |
| `mask` | `GPUVector<'uint32'>` | Omitted | Optional vertex-aligned zero or one reachability mask. |
| `seedCount` | `GPUVector<'uint32'>` | Omitted | Optional scalar limiting the number of active seed rows. |
| `maxDepth` | `number` | `1` | Compiled maximum hop count, from zero through 1,024. |
| `activeDepth` | `GPUVector<'uint32'>` | Omitted | Optional scalar dynamically limiting the compiled hop count. |
| `direction` | `'outgoing'`, `'incoming'` or `'both'` | `'outgoing'` | Edge orientation followed; incoming and both need reverse adjacency on directed graphs. |

`outgoing` follows source-to-target relationships; `incoming` follows their reverse; `both` combines
them. Directed incoming and bidirectional searches require reverse adjacency. `maxDepth` bounds the
number of compiled passes, while an optional one-row GPU `activeDepth` can lower the active search
depth between encodings without rebuilding the command graph. An optional GPU `seedCount` similarly
limits which existing seed rows are active.

Reached roots have distance zero. Unreachable vertices and root predecessors contain `0xffffffff`;
equal-length parent ties select the numerically lowest stable vertex identifier. Invalid seeds are
ignored, duplicate seeds are harmless, and the optional mask publishes zero or one per vertex.
This is an unweighted shortest-path operation, not a weighted route or travel-time solver.

#### How it computes

{/* lens:GPUGraphBreadthFirstSearch */}

::lens{scenario=gpu-graph.graph-breadth-first-search rung=kernel caption="Inspect the seed, frontier-expansion and frontier-dispatch kernels run at each depth."}
::lens{scenario=gpu-graph.graph-breadth-first-search rung=graph caption="Follow topology construction into 16 gated depths of expansion and compact frontier dispatch."}
::lens{scenario=gpu-graph.graph-breadth-first-search rung=datum datum=26 caption="Inspect Amsterdam Centraal, station 26, one hop from the source, with its CSR neighborhood."}
::lens{scenario=gpu-graph.graph-breadth-first-search rung=formula caption="Follow the min-distance update and lowest-parent rule to the WGSL buffers that hold them."}
::lens{scenario=gpu-graph.graph-breadth-first-search rung=iterations caption="Compare six recorded distance frames through depth 15; unreached stations are shown as no data."}

## Find least-cost routes with GPUGraphSingleSourceShortestPath

Relax nonnegative edge weights from one source in bounded synchronized rounds, publishing least-cost distances and parents.

::lens{scenario=gpu-graph.graph-single-source-shortest-path rung=result caption="Follow travel time in seconds from Utrecht Centraal; the farthest station takes 9,300 seconds."}

**Question: Which route from my starting vertex costs the least when each connection has its own
travel time, distance, latency, or other nonnegative price?**

The route with the fewest steps is not necessarily the cheapest. Suppose a direct train takes
40 minutes, while two connecting trains take 10 minutes each: breadth-first search chooses the
one-hop direct train, but the least-cost route takes the two connections and arrives in 20 minutes.
`GPUGraphSingleSourceShortestPath` computes that minimum accumulated nonnegative edge weight from one
selected starting vertex to every reachable destination without moving graph columns to the CPU.

Use it for travel-time maps, delivery networks, communication latency, transaction fees, or
service-dependency recovery costs. Choose breadth-first search when every edge has the same cost
and only the number of hops matters; choose weighted shortest paths when the actual sum of edge
weights can change which route wins. This operation computes routes from **one** selected source;
it does not implement all-pairs routing, negative-weight paths, or A* geographic search.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | `string` | `'gpu-graph-single-source-shortest-path'` | Prefix for generated command-graph nodes and imported resources. |
| `topology` | `GPUGraphTopology` | Required | Existing GPU adjacency and its explicit overflow status. |
| `sourceVertex` | `number` | Required | Stable source vertex identifier; zero is also accepted for an empty graph. |
| `distances` | `GPUVector<'float32'>` | Required | Caller-owned floating-point distance row for every vertex. |
| `predecessors` | `GPUVector<'uint32'>` | Required | Caller-owned unsigned predecessor row for every vertex. |
| `maxIterations` | `number` | `min(max(vertexCount - 1, 0), 1024)` | Maximum compiled relaxation rounds, from zero through 1,024. |
| `direction` | `'outgoing'`, `'incoming'` or `'both'` | `'outgoing'` | Edge orientation followed; incoming and both need reverse adjacency on directed graphs. |
| `converged` | `GPUVector<'uint32'>` | Omitted | Optional scalar reporting whether all shortest distances are final. |
| `invalidWeightCount` | `GPUVector<'uint32'>` | Omitted | Optional scalar receiving invalid, valid-endpoint source-edge weights. |

`distances` is a caller-owned, packed `GPUVector<'float32'>` with exactly one row per vertex;
`predecessors` is a separate, caller-owned, packed `GPUVector<'uint32'>` with the same row count.
An unreachable vertex has distance positive infinity (`+Infinity`), not a large finite placeholder,
and predecessor `0xffffffff`. The starting vertex has distance zero and predecessor `0xffffffff`.
When two routes have the same total cost, the one with fewer hops wins; if their costs and hop
counts also tie, the numerically lowest stable predecessor identifier wins. Zero-weight edges are
valid. Distances are ordinary single-precision values, not `float64` or floating-point atomics.

Weighted graphs must preserve their aligned source `GPUVector<'float32'>` edge-weight chunks and
matching `float32` weights in every configured CSR direction. A graph without an edge-weight
column is valid: every edge then costs one, allowing the same operation to run without fabricating
a new weight column. Negative weights, `NaN`, positive infinity, and negative infinity are invalid.
If any original edge with valid endpoints has an invalid weight, routing fails closed: all
distances become `+Infinity`, all predecessors become `0xffffffff`, and an optional one-row
`invalidWeightCount` records the number of invalid **source edges**, not duplicate entries in
forward and reverse adjacency. Edges whose endpoints are already invalid are excluded by topology
and do not contribute to this weight count.

`outgoing` follows source-to-target relationships; `incoming` follows reverse relationships;
`both` admits either direction. Directed `incoming` or `both` routing requires reverse CSR;
directed `outgoing` routing requires forward CSR only. Undirected graphs reuse symmetric forward
adjacency. If any selected adjacency overflowed, all distances again become `+Infinity`, all
predecessors become `0xffffffff`, and an optional convergence status is zero.

The contributor uses bounded Bellman–Ford-style synchronized relaxation, not a CPU Dijkstra queue.
`maxIterations` can range from `0` through `1024`; its default is
`min(max(vertexCount - 1, 0), 1024)`. The optional caller-owned one-row
`GPUVector<'uint32'>` `converged` indicates whether the final compiled relaxation reached a fixed
point. A zero-round budget initializes only the selected root. `sourceVertex` must identify an
existing vertex; an empty graph permits the value zero and has no distance rows. A limited budget
may publish partially relaxed routes without proving shortest-path completion; there is no
automatic early termination or implicit CPU synchronization. With `K` compiled relaxation rounds,
work is bounded by `O(K × (V + E))`, with `O(V)` graph-owned scratch.

#### How it computes

{/* lens:GPUGraphSingleSourceShortestPath */}

::lens{scenario=gpu-graph.graph-single-source-shortest-path rung=kernel caption="Inspect the snapshot, relaxation, predecessor and convergence kernels run each round."}
::lens{scenario=gpu-graph.graph-single-source-shortest-path rung=graph caption="Follow topology construction into 24 relaxation rounds, each snapshotting before it relaxes."}
::lens{scenario=gpu-graph.graph-single-source-shortest-path rung=datum datum=26 caption="Inspect Amsterdam Centraal, station 26, 1,560 seconds from the source, with its neighborhood."}
::lens{scenario=gpu-graph.graph-single-source-shortest-path rung=formula caption="Follow the synchronized relaxation and parent rule to the WGSL buffers that hold them."}
::lens{scenario=gpu-graph.graph-single-source-shortest-path rung=iterations caption="Compare seven recorded distance frames through the final round; no intermediate states are interpolated."}

## Related pages

- [GPU Graph overview](/docs/api-reference/experimental/gpu-graph)
- [GPU Graph operations index](/docs/api-reference/experimental/gpu-graph-operations)
- [GPU Core](/docs/api-reference/experimental/gpu-core)
