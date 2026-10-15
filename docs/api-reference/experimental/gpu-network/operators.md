---
title: "GPU Network operators"
description: "Every GPU Network contributor: inputs, outputs, parameters and how it computes."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Network operators

<ExperimentalDocsTabs active="gpu-network-operators" />

### `GPUNetworkReachability`

Compute travel-time costs, bands and predecessors from one start intersection over a directed road network.

::lens{scenario=gpu-network.network-reachability rung=result caption="Follow travel times across 1,199 intersections; the farthest is about 262.7 seconds from the start."}

Maximum: `output.costs` has at most `getGPUNetworkReachabilityMaximumNodeCount(device)` (16777216 on a default device) rows, `laneCount * nodeCount`, because the frontier queues take 8 bytes per row.

Single- or multi-source shortest-path costs over a directed CSR road network with isochrone bands,
band counts, predecessors, and a convergence flag. Relaxation is chaotic Bellman-Ford over a compact
GPU frontier queue with a round-stamped visited set: each of the `maximumIterations` rounds is one
indirect dispatch sized by the previous round's pushes, so converged frames dispatch nothing and
nothing is read back. Within a round each workgroup chains up to `localIterations` (default 32)
hops through workgroup memory, so a path of `h` hops needs about `h / localIterations` rounds; costs
are identical for every setting. Predecessors are cycle-safe: the smallest strictly cheaper tight
in-neighbor, or across equal-cost (zero-weight) edges the smallest tight in-neighbor one BFS level
closer to a strict entry or a source, found in up to `maximumTieIterations` extra rounds (default 4).
`maximumIterations` counts rounds, one graph node each, not hops. With the default `localIterations`,
a budget of about `h / 22` plus a few rounds suffices; 48 covers the demo networks. The optional
`output.unresolvedCount` (one `uint32` row, requires `output.predecessors`) counts reached nodes that still have no
predecessor because the tie phase stopped at `maximumTieIterations`; 0 means every reached node has its
tie-rule predecessor, and `status.converged` is unchanged.
`recommendReachabilityIterations()` suggests a first-guess `maximumIterations` from the network size, and
`adaptReachabilityIterations()` grows or shrinks it from the last frame's `output.status.iterationCount` and
`output.status.converged` readback. `startCount` is a number (compiled in) or a one-row view (per frame).
Build CSR on the GPU with `GPUCOOToCSR`; a reverse CSR gives "cost to reach" isochrones.
Each round reads its index from its own 256-byte dispatch slot (`[x, y, z, round]`), never from the
WGSL text, so every round of a phase shares one pipeline: one reachability compiles one relax and one
tie-level pipeline at any `maximumIterations`, and so do the label rounds of `GPUNetworkServiceAreas`.

The optional `output.status.relaxationCount` and `output.status.reachedEdgeCount` (two `uint32` rows each, a 64-bit
count as low and high words) are an inflation probe. `relaxationCount` is every edge scanned by every relax visit,
including the chained hops and all lanes; `reachedEdgeCount` is the out-degree sum of every node with a finite final
cost, the edges a label-setting Dijkstra scans once. `getGPUNetworkReachabilityInflation(relaxationCount, reachedEdgeCount)`
returns their ratio (1 is work-efficient, NaN when nothing was reached). Delta-stepping or near-far is only worth
building when the measured inflation exceeds 3. The counters add one atomic per workgroup per round, so leave them off in production.

```ts
graph.add(new GPUNetworkReachability({
  topology, startNodes: starts.importToGraph(graph), costLimit: costLimit.importToGraph(graph),
  costBreaks: costBreaks.importToGraph(graph), maximumIterations: 48,
  output: {costs, bands, bandCounts, status: {converged}}
}));
```

#### How it computes

{/* lens:GPUNetworkReachability */}

::lens{scenario=gpu-network.network-reachability rung=datum bind=page datum=78 caption="Inspect the start neighbour: its final travel time is about 5.1 seconds."}

::lens{scenario=gpu-network.network-reachability rung=formula}

::lens{scenario=gpu-network.network-reachability rung=iterations caption="Watch key frames through round 37; the hatched replay uses a smaller round limit."}

::lens{scenario=gpu-network.network-reachability rung=kernel}

::lens{scenario=gpu-network.network-reachability rung=graph}

### `GPUNetworkCentrality`

Measure outward closeness of sampled pivot intersections from weighted road searches over the network and its transpose.

::lens{scenario=gpu-network.network-centrality rung=result caption="Compare 8 sampled pivots; each reaches 1,199 intersections, and the highlight has the highest closeness."}

Radius-bounded betweenness and closeness centrality of a directed weighted CSR network, exact from every
node as a pivot or estimated from a sample. It needs the network and its transpose (`topology` and
`reverseTopology`, the same edges reversed with bitwise-identical weights, for example from `GPUCOOToCSR`;
an undirected network that lists both directions with equal weights may pass one topology twice) and the
`pivotNodes` to measure from. Outputs (at least one): `betweenness` (per node), `closeness`,
`harmonicCloseness` and `reachedCounts` (per pivot), and `status.converged`.

Pivots run `laneCount` at a time (default `recommendLaneCount()`, about 1M expanded nodes per batch; closeness
outputs are bit-identical for every lane count). Each batch runs one lane-expanded `GPUNetworkReachability`
(`costLimit` is the radius, `maximumIterations` and `localIterations` as there; the defaults are 64 and 16).
Closeness comes straight from those costs: one workgroup per lane reduces the reached count, cost sum and
harmonic sum in a fixed order. Betweenness adds Brandes' (2001) accumulation on the tight edges, those with
`cost[u] < cost[w]` and `cost[u] + weight == cost[w]` in f32, exactly as the relaxation computes it. The
sigma phase (path counts) is a push-ready, pull-sum frontier schedule: a node is queued when its last tight
predecessor releases it through an `atomicSub` on a counter that starts at its tight in-degree, then sums
its predecessors' sigma in reverse-CSR order. The delta phase mirrors it from the nodes without tight
out-edges, summing `sigma(v) / sigma(w) * (1 + delta(w))` in forward-CSR order. Each batch adds the
lane-ordered sum of its deltas (skipping each lane's own pivot) times `betweennessScale` to the output,
zeroed once, so results are deterministic for a given `laneCount`. Both phases use `maximumIterations`
rounds each, one pipeline for every round and batch. Scratch memory is the frontier state, three
`laneCount * nodeCount` word arrays, and a packed copy of both CSRs made once per encoding.

Conventions (oracle `test/oracles/network-centrality.py`, networkx 3.7):

- Betweenness equals `networkx.betweenness_centrality(G, weight='weight', normalized=False)` on a `DiGraph` when
  every node is a pivot, and `betweenness_centrality_subset(G, sources=pivots, targets=list(G), ...)` for a
  sample, times `betweennessScale`. Endpoints are not counted and no 1/2 factor is applied (halve the result for
  an undirected graph that lists both directions). For estimates from a sample use
  `betweennessScale: nodeCount / pivotCount`.
- Distances run outward from the pivots. networkx's `closeness_centrality` and `harmonic_centrality` on a
  `DiGraph` use incoming distances, so the oracle compares with `closeness_centrality(G.reverse(), u=pivot,
  wf_improved=False)` and `harmonic_centrality(G.reverse(), [pivot])`. Swap `topology` and `reverseTopology`
  to get networkx's incoming convention. `closeness` is `(r - 1) / S` for `r` reached nodes (the pivot
  included) and cost sum `S`, 0 when `S` is 0; the Wasserman-Faust factor is not applied.
- `costLimit` (a one-row per-frame view) is an inclusive radius: nodes beyond it and paths longer than it
  are not counted. networkx has no radius option; the oracle uses an explicit Brandes over a cutoff search
  that equals networkx when the radius is infinite.
- Zero-weight edges (and weights below half an ulp of the cost) are never tight, so they are excluded from
  path counts; a node reached only across them is a root with one path. Closeness counts such nodes as reached
  at their predecessor's cost, harmonic closeness skips nodes of cost 0, and parallel edges count as separate
  paths. Parity with networkx holds for positive weights.
- Out-of-range pivots are ignored and their per-pivot rows are 0. `status.converged` is 1 when every batch's
  reachability and, with `betweenness`, both phases finished within `maximumIterations`.
- Sigma is exact below 2^24 paths per pair. Delta rounds once per division (within 2.5 ULP in WGSL), addition
  and multiplication and once per added term, all terms positive, so a node's relative error is at most
  `h (7 + K) 2^-24` for a longest tight chain of `h` hops and largest tight out-degree `K`, plus
  `(pivotCount + 2) 2^-24` for the sum over pivots.
- The sigma and delta phases run one hop per round (`localIterations` applies to the search only), so
  a node's sum reads its predecessors' values only after a dispatch boundary; each phase needs about
  `h + 1` of the `maximumIterations` rounds for a longest tight chain of `h` hops.

```ts
const centrality = new GPUNetworkCentrality({
  topology,
  reverseTopology,
  pivotNodes: pivots.importToGraph(graph),
  costLimit: radius.importToGraph(graph),
  betweennessScale: nodeCount / pivotCount,
  output: {
    betweenness: betweenness.importToGraph(graph),
    closeness: closeness.importToGraph(graph),
    status: {converged: converged.importToGraph(graph)}
  }
});
graph.add(centrality);
```

#### How it computes

{/* lens:GPUNetworkCentrality */}

::lens{scenario=gpu-network.network-centrality rung=datum bind=page datum=3 caption="Inspect this pivot: it reaches 1,199 intersections, with closeness about 0.00615 per second."}

::lens{scenario=gpu-network.network-centrality rung=formula}

::lens{scenario=gpu-network.network-centrality rung=kernel}

::lens{scenario=gpu-network.network-centrality rung=graph}

### `GPUNetworkPathExtraction`, `GPUNetworkServiceAreas`, `GPUNetworkNeighborhood`, and `GPUNetworkAnalyticsColumns`

Network analysis over the same directed CSR road network as `GPUNetworkReachability`.

- `GPUNetworkPathExtraction` walks a reachability predecessor array from per-frame targets. It publishes ordered (source to target) node lists and resolved CSR edge lists as `GPUCompactOutput`s, together with per-target offsets, costs, and found flags. Each walk is bounded by a compile-time `maximumPathLength`, so cyclic or garbage predecessors terminate and raise `output.status.overflow`. The per-target `lineOffsets`, `pathCosts` and `pathFound` are in `output`, and `targetCount` is a number or a one-row view. Pass the search's `output.status.converged` flag as `predecessorsComplete`: when it is 0 no target is reported found, because a chain may end at a node the truncated search never linked rather than at a source.
- `GPUNetworkServiceAreas` assigns every node to its nearest per-frame facility. Ties go to the smallest facility row; optional `predecessors` follow only tight edges between nodes of the same assignment, so a path always ends at the facility that `assignments` names, also on cost ties. It also reports per-facility node counts and the total cost each facility serves, all in `output`. It composes `GPUNetworkReachability` with a frontier-based min-label pass over the tight shortest-path edges. `maximumLabelIterations` bounds the label rounds (default `min(maximumIterations, 32)`) and `localIterations` sets the hops chained per round in both phases (default 16), so large-diameter grids converge at the defaults.
- `GPUNetworkNeighborhood` publishes k-hop ego networks around per-frame `startNodes` (`startCount` a number or a view), with per-frame `k` up to a compile-time `maximumHops`. `output` holds hop distances, masks, and compact node and induced-edge IDs.
- `GPUNetworkAnalyticsColumns` runs `gpu-graph` degree, PageRank (`iterationCount`), connected components, core number, and label-propagation communities (`maximumIterations`) directly on the caller's `topology` (and optional `reverseTopology`) through a `GPUGraphTopologyView`, so a transient CSR built in the same graph by `GPUCOOToCSR` works as well as imported buffers. It publishes node-aligned columns, optionally normalized to `[0, 1]` with a GPU extent.

```ts
graph.add(new GPUNetworkReachability({topology, startNodes: origin.importToGraph(graph),
  maximumIterations: 48, output: {costs, predecessors}}));
graph.add(new GPUNetworkPathExtraction({predecessors, costs,
  targets: destination.importToGraph(graph), output: route,
  edges: {topology, output: routeEdges}}));
graph.add(new GPUNetworkServiceAreas({topology, facilities: stations.importToGraph(graph),
  costLimit: limit.importToGraph(graph), output: {assignments, facilityNodeCounts}}));
graph.add(new GPUNetworkNeighborhood({topology, startNodes: picked.importToGraph(graph),
  hops: hops.importToGraph(graph), maximumHops: 8, output: {hopDistances, nodes, edges}}));
graph.add(new GPUNetworkAnalyticsColumns({topology,
  pageRank: {output: rank, normalized: rankSize}, coreNumber: {output: core}}));
```

#### `GPUNetworkPathExtraction`

Convert reachability predecessors and target intersections into ordered start-to-target node paths with offsets, costs and found flags.

::lens{scenario=gpu-network.network-path-extraction rung=result caption="Follow the highlighted 25-intersection path from the start to the West Loop library target."}

#### How it computes

{/* lens:GPUNetworkPathExtraction */}

::lens{scenario=gpu-network.network-path-extraction rung=datum bind=page datum=648 caption="Inspect the target: travel time is about 162.7 seconds, and its predecessor is intersection 660."}

::lens{scenario=gpu-network.network-path-extraction rung=kernel}

::lens{scenario=gpu-network.network-path-extraction rung=graph}

#### `GPUNetworkServiceAreas`

Assign every road intersection to its nearest facility by travel time, with facility labels, minimum costs and optional service totals.

::lens{scenario=gpu-network.network-service-areas rung=result caption="Compare 4 service areas; each colour marks the facility nearest by travel time."}

#### How it computes

{/* lens:GPUNetworkServiceAreas */}

::lens{scenario=gpu-network.network-service-areas rung=datum bind=page datum=318 caption="Inspect this facility intersection: its assigned facility has travel cost 0 seconds."}

::lens{scenario=gpu-network.network-service-areas rung=formula}

::lens{scenario=gpu-network.network-service-areas rung=kernel}

::lens{scenario=gpu-network.network-service-areas rung=graph}

#### `GPUNetworkNeighborhood`

Find intersections within a hop radius of road-network starts and return hop distances, membership masks and induced edges.

::lens{scenario=gpu-network.network-neighborhood rung=result caption="See the 49 intersections within 6 hops of the start, shaded by hop count."}

#### How it computes

{/* lens:GPUNetworkNeighborhood */}

::lens{scenario=gpu-network.network-neighborhood rung=datum bind=page datum=1088 caption="Inspect the start intersection: its hop distance is 0."}

::lens{scenario=gpu-network.network-neighborhood rung=formula}

::lens{scenario=gpu-network.network-neighborhood rung=kernel}

::lens{scenario=gpu-network.network-neighborhood rung=graph}

#### `GPUNetworkAnalyticsColumns`

Compute node-aligned degree and min-max normalized columns from road connectivity.

::lens{scenario=gpu-network.network-analytics-columns rung=result caption="Find high-degree junctions; the highlighted intersection has 6 outgoing roads."}

#### How it computes

{/* lens:GPUNetworkAnalyticsColumns */}

::lens{scenario=gpu-network.network-analytics-columns rung=datum bind=page datum=1088 caption="Inspect this intersection: degree 2 becomes a normalized score of about 0.2."}

::lens{scenario=gpu-network.network-analytics-columns rung=formula}

::lens{scenario=gpu-network.network-analytics-columns rung=kernel}

::lens{scenario=gpu-network.network-analytics-columns rung=graph}

### `GPUNetworkStatistics`

Summarize road connectivity into component counts, degree histograms and maxima in one GPU-written table of words.

::lens{scenario=gpu-network.network-statistics rung=result caption="Read the GPU summary: 1,199 intersections, 2,810 directed edges, and one connected component."}

Statistics for a graph panel over the same CSR as `GPUNetworkAnalyticsColumns`, reduced on the GPU
into one caller-owned `uint32` summary. It reports live node, edge and slot counts, weak component
count and largest component size, isolated nodes, maximum degrees, out-, in- and total-degree
histograms (`'linear'` or `'log2'` bins), and the modularity of a caller-supplied community
labeling. Optional `nodeMask` and `edgeMask` restrict every statistic to live rows. Components run
`GPUGraphConnectedComponents` on a masked copy of the adjacency, so masking a bridge node splits
its component. Modularity is accumulated with exact integer atomics and a fixed-order f32 finish,
so it is deterministic. On undirected graphs a self-loop adds 2 to the degree, the histograms, the
maxima and the modularity volumes, as in networkx and `GPUGraphModularity`; slot and edge counts count
the stored slot once. The resolution
and linear bin width are per-frame parameters (`getGPUNetworkStatisticsParameterValues`). Read the
summary back and decode it with `decodeGPUNetworkStatistics`. Scratch (the masked adjacency, labels
and convergence flag) is graph transients; the contributor owns no buffers. Components run at most
`maximumComponentIterations` rounds.
The slot loops are load-balanced, so a hub row spreads over many workgroups; `topology.offsets` must be non-decreasing.

```ts
const summary = graph.importGPUData<'uint32'>('stats', statsData);
graph.add(new GPUNetworkStatistics({topology, nodeMask, communities,
  parameters: statsParameters.importToGraph(graph), degreeBinning: 'log2', output: summary}));
// after the application submits the encoding
const stats = decodeGPUNetworkStatistics(new Uint32Array(await statsBuffer.readAsync()));
```

{/* parameters: GPUNetworkStatistics generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkStatistics` parameters

- Packer: `getGPUNetworkStatisticsParameterValues(settings?)`, writes `GPU_NETWORK_STATISTICS_PARAMETER_LENGTH` (2) `uint32` values.

Settings of `getGPUNetworkStatisticsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `resolution` (optional) | `number` | Modularity resolution `gamma`. Defaults to 1. |
| `degreeBinWidth` (optional) | `number` | Linear degree bin width, floored and clamped to at least 1. Defaults to 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUNetworkStatistics */}

::lens{scenario=gpu-network.network-statistics rung=datum bind=page datum=2 caption="Inspect the out-degree histogram: 527 intersections have degree 2."}

::lens{scenario=gpu-network.network-statistics rung=formula}

::lens{scenario=gpu-network.network-statistics rung=kernel}

::lens{scenario=gpu-network.network-statistics rung=graph}

### `GPUNetworkSubgraphFilter`

Keep road edges and intersections that pass range predicates, returning consistent node and edge masks with live counts.

::lens{scenario=gpu-network.network-subgraph-filter rung=result caption="Compare 908 retained directed road edges with the faint full graph; 805 intersections remain."}

Turns attribute and time predicates on node and edge rows into a consistent induced subgraph of a forward CSR network.

- Node `v` is live when its caller `nodeMask` is nonzero and every enabled node range accepts its f32 column value. Ranges are half-open (`min <= value < max`); NaN never passes.
- Slot `u -> v` is live when its caller mask is nonzero, every enabled edge range and time gate accepts it, and both endpoints are live. The f32 `edgeTimes` window and the exact Int64 `edgeTimeWords` window are closed (`start <= t <= end`).
- Undirected graphs list each edge in both slots. By default an edge is live only if both slots pass, so `edgeMask` is always symmetric; set `pairUndirectedSlots: false` to skip the O(degree) pairing when columns are already symmetric.
- `dropIsolated` also kills nodes with no live incident slot.
- Ranges live in one f32 `parameters` view (`getGPUNetworkSubgraphFilterParameterValues`) and the time-word window in `uint32Parameters` (`getGPUTimeWindowFilterWordParameterValues`); rewriting them never recompiles.
- Outputs: `output.nodeMask` and `output.edgeMask` (1 live, 0 dead; feed `GPUNetworkStatistics` and `filterMask`), `counts` (decode with `decodeGPUNetworkSubgraphFilterCounts`), compact `liveNodes` and `liveEdgeSlots` (stable ascending, overflow-reporting), and an `inducedTopology` (`offsets`, `neighbors` over the original node ids, optional `sourceSlots`, `status.overflow`) built with a prefix sum over live degree. Rows past the live totals (`edgeMask` slots past `offsets[nodeCount]`, induced CSR rows past the live slot count) are written every frame: 0 and `0xffffffff`.
- Results are exact: integer arithmetic and comparisons only.

```ts
const filter = new GPUNetworkSubgraphFilter({
  topology,
  nodeColumns: [population], edgeColumns: [speed], parameters,
  dropIsolated: true,
  output: {nodeMask, edgeMask, counts}
});
graph.add(filter);
parameterBuffer.write(
  getGPUNetworkSubgraphFilterParameterValues(layout, {nodeRanges: [[1000, Infinity]], edgeRanges: [[30, 90]]})
);
```

{/* parameters: GPUNetworkSubgraphFilter generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkSubgraphFilter` parameters

- Packer: `getGPUNetworkSubgraphFilterParameterValues(layout, state?)`, writes `getGPUNetworkSubgraphFilterParameterLength(...)` `float32` values.
  - `state` (`GPUNetworkSubgraphFilterState`)

Settings of `getGPUNetworkSubgraphFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `nodeColumnCount` | `number` | Number of node columns. |
| `edgeColumnCount` | `number` | Number of edge columns. |
| `hasEdgeTimes` | `boolean` | Whether an f32 edge time column exists. |

{/* /parameters */}

#### How it computes

{/* lens:GPUNetworkSubgraphFilter */}

::lens{scenario=gpu-network.network-subgraph-filter rung=datum bind=page datum=100 caption="Inspect the rejected edge: its travel time is about 12.4 seconds."}

::lens{scenario=gpu-network.network-subgraph-filter rung=formula}

::lens{scenario=gpu-network.network-subgraph-filter rung=kernel}

::lens{scenario=gpu-network.network-subgraph-filter rung=graph}

### `GPUNetworkCoarsening`

Aggregate road intersections by zone label into supernode counts, centroids and sorted weighted superedges.

::lens{scenario=gpu-network.network-coarsening rung=result caption="Compare 4 zones joined by 9 superedges; line colour counts directed road edges between zones."}

Maximum: with `output.edgeWeights`, `output.edges.ids` has at most `getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(device)` (16777216 on a default device) rows, because the 64-bit weight accumulators take 8 bytes per superedge.

Summarizes a network `topology` (weights and `nodePositions` optional) by a node group label (a community, component or spatial cell) for compound nodes when zoomed out. Every result is in `output`. Supernode outputs: `groupNodeCount` plus optional `groupIntraEdgeCount`, `groupIntraWeight`, `groupCentroid` (`float32x2`), `groupBounds` (`float32x4`) and `groupValueSum`, each with exactly `groupCapacity` rows. Superedges are a bounded `edges` output (`ids` is the source group) with `edgeTargetGroups`, `edgeCounts` and optional `edgeWeights`, sorted by `(source, target)`; undirected superedges have `source < target`. An optional 8-word `summary` reports live and overflowed nodes, counted/dropped/intra/inter edges, group count and the unclamped superedge count.

```ts
const coarsening = new GPUNetworkCoarsening({
  topology: {...topology, nodePositions}, labels, groupCapacity: 1024,
  output: {
    groupNodeCount, groupCentroid, groupBounds,
    edges: {ids: sourceGroups, status: {count, overflow}}, edgeTargetGroups, edgeCounts, edgeWeights
  }
});
graph.add(coarsening);
```

Labels are dense group IDs below the compile-time `groupCapacity` (at most 65535); a live node above it is excluded and sets `edges.status.overflow`. Undirected CSRs list each edge in both directions and each edge counts once. Counts are integer exact. Weight, position and value sums use 64-bit fixed-point atomics (`fixedPointScale`, default 65536), so they are bitwise reproducible. Superedges use a stable sort and segment, so order and the kept prefix on overflow are deterministic. CSR slots past `offsets[nodeCount]` take the sentinel key and never become superedges. Inputs must be finite.

#### How it computes

{/* lens:GPUNetworkCoarsening */}

::lens{scenario=gpu-network.network-coarsening rung=datum bind=page datum=1 caption="Inspect this zone: it contains 216 intersections, summarized by its centroid."}

::lens{scenario=gpu-network.network-coarsening rung=kernel}

::lens{scenario=gpu-network.network-coarsening rung=graph}

### `GPUAdjacencyMatrix` and `GPUAdjacencyMatrixOrder`

Bins a forward `topology` into an `R x R` matrix image for a matrix view linked to a node-link view. Row-major `uint32` counts (`row * R + column`), optional fixed-point weight sums, one-row maxima for color normalization, and an optional `r32float` storage texture. The optional `order` view maps node to matrix position (a permutation computed by the caller, for example with `GPUAdjacencyMatrixOrder` or `computeAdjacencyMatrixOrder`; identity by default). Per-frame zoom is a four-word window `[rowStart, rowEnd, colStart, colEnd)` in matrix positions, the `parameters` view (`getGPUAdjacencyMatrixParameterValues`); position `p` lands in bin `floor((p - start) * R / (end - start))` and positions outside the window are dropped, so panning and zooming never recompile.

```ts
const matrix = new GPUAdjacencyMatrix({
  topology, order, parameters: windowParameters.importToGraph(graph),
  resolution: 64,
  output: {counts, weightSums, maximumCount, maximumWeightSum}
});
graph.add(matrix);
windowParameters.write(
  getGPUAdjacencyMatrixParameterValues({rowStart: 0, rowEnd: 500, colStart: 0, colEnd: 500, resolution: 64})
);
```

Undirected graphs follow the convention that the CSR lists both directions, so each slot is written once and the matrix is symmetric; pass `mirrorSlots: true` for an edge-list CSR that lists each edge once and each slot is also written transposed (self-loops once). Directed: row = source. Masks follow the `GPUNetworkStatistics` liveness rule. Counts are exact `u32` atomics. Weight sums are fixed-point `u32` atomics of `round(weight * weightScale)` (default scale 1024), so results are bit-identical regardless of thread order; the cost is half a step of quantization per edge and wraparound at 2^32. `GPUAdjacencyMatrixOrder` builds `order` from a `uint32` group label and an optional tie key with two stable `GPUSort` passes and matches `computeAdjacencyMatrixOrder` exactly.

{/* parameters: GPUAdjacencyMatrix generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUAdjacencyMatrix` parameters

- Packer: `getGPUAdjacencyMatrixParameterValues(settings)`, writes `GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUAdjacencyMatrixParameterValues`:

| Field | Type | Description |
|---|---|---|
| `rowStart` | `number` | First row position. |
| `rowEnd` | `number` | One past the last row position. |
| `colStart` | `number` | First column position. |
| `colEnd` | `number` | One past the last column position. |
| `resolution` | `number` | Matrix resolution `R` of the `GPUAdjacencyMatrix`, used to bound the window extents. |

{/* /parameters */}

#### `GPUAdjacencyMatrix`

Bin directed road connections into a small adjacency matrix of edge counts, in a chosen node order.

::lens{scenario=gpu-network.adjacency-matrix rung=result caption="Look along the diagonal of the 8-by-8 matrix; the busiest cell contains 306 directed edges."}

#### How it computes

{/* lens:GPUAdjacencyMatrix */}

::lens{scenario=gpu-network.adjacency-matrix rung=datum bind=page datum=9 caption="Inspect this diagonal matrix cell: it contains 254 directed edges."}

::lens{scenario=gpu-network.adjacency-matrix rung=formula}

::lens{scenario=gpu-network.adjacency-matrix rung=kernel}

::lens{scenario=gpu-network.adjacency-matrix rung=graph}

#### `GPUAdjacencyMatrixOrder`

Sort intersections by group label into a stable node-to-matrix-position permutation.

::lens{scenario=gpu-network.adjacency-matrix-order rung=result caption="Compare 12 shown intersections with their matrix positions, sorted by group then node id."}

#### How it computes

{/* lens:GPUAdjacencyMatrixOrder */}

::lens{scenario=gpu-network.adjacency-matrix-order rung=datum bind=page datum=648 caption="Inspect this intersection: group 4 places it at matrix position 590."}

::lens{scenario=gpu-network.adjacency-matrix-order rung=kernel}

::lens{scenario=gpu-network.adjacency-matrix-order rung=graph}

### `GPUNetworkSnapping`, `GPUNetworkCostMatrix`, and `GPUNetworkAccessibility`

For `GPUNetworkSnapping.maximumSnapDistance`, no warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](../gpu-spatial-analysis/concepts.md#precision-of-longitudelatitude-positions).

Accessibility over the same directed CSR road network as `GPUNetworkReachability`, in three steps.

- `GPUNetworkSnapping` snaps planar points (origins, facilities, opportunities) to the nearest
  network edge of a `topology` with `nodePositions` (edge costs are `topology.weights`, or the
  edge length when the topology has no weights). For each point it
  outputs the edge row (ties go to the smallest row, so a point on a node takes the smallest
  incident edge), the fraction along the edge, the snap distance, the snapped position, and the
  two start costs: `fraction * edgeCost` to the edge source and `(1 - fraction) * edgeCost` to the
  target, all in `output`. `output.startNodes` / `output.startCosts` (two rows per point) feed a
  search directly. `startDirection` picks which endpoints count: `'both'` for undirected
  networks, `'forward'` for a point that leaves along a one-way edge, `'reverse'` for a point that
  a one-way edge arrives at. Without `candidateCapacity`, every point scans every edge, which is
  exact. With it, candidates come from `GPUNearestFeatureJoin` over the edge segments within a
  per-frame `maximumSnapDistance`, and the required `output.status.overflow` reports when capacity runs out. With `candidateCapacity` the network has at most `getGPUNetworkSnappingMaximumCandidateEdgeCount(device)` (8388608 on a default device) edges. With
  `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` points and nodes
  are degrees: the closest point of an edge is found in the point's local equirectangular frame
  (longitudes wrapped at +-180), and snap distances, `maximumSnapDistance` and default edge costs are
  in the context's length unit (meters or kilometers). Edges should be road-length. With `candidateCapacity` the context is passed to
  `GPUNearestFeatureJoin`, whose great-circle search ignores the capacity. Edge rows past
  `offsets[nodeCount]` (a CSR shorter than `neighbors`) are not part of the network:
  every edge-source row is cleared before the expansion, so no stale row becomes a phantom edge.
- `GPUNetworkCostMatrix` computes a bounded many-to-all cost matrix: one shortest-path search per
  row, from that row's starts (`startRows`, or `startsPerRow: 2` for snapped points) to every node.
  It batches `laneCount` rows (default `recommendLaneCount()`: about 1M expanded nodes per batch,
  at least 32 lanes, within 128 MB of scratch, at most `rowCount`, rounded down to make
  `laneCount * nodeCount` a multiple of 64 when there are several batches) into one `GPUNetworkReachability` over a lane-expanded
  copy of the CSR. Lane `l`, node `u` becomes node `l * nodeCount + u`, so the frontier rounds,
  hop chaining, per-frame `costLimit`, and convergence flag serve all lanes at once. The matrix is
  bit-identical for every `laneCount`. Every batch runs at the full lane count (a short last batch
  masks its lanes) on 64-element-aligned sub-views, so all batches share one set of pipelines. A
  batch whose block of `costs` is not 256-byte aligned searches into a scratch block and is copied.
  `targetNodes` (a compile-time length `targetCount`, per-frame node IDs) makes `output.costs` the
  compact `[rowCount x targetCount]` matrix: column `t` is the cost to `targetNodes[t]`, `+Infinity`
  when unreached or out of range, bit-identical to that column of the dense matrix. Each batch
  searches into one `laneCount * nodeCount` scratch block and a gather kernel writes its rows, so
  memory is that block plus `rowCount * targetCount` instead of `rowCount * nodeCount`; lower
  `laneCount` to shrink the scratch block. `columnCount` reports the column count.
  Rows set the number of searches, so put the smaller side on
  rows. To score every node, put opportunities or facilities on rows and search the reverse CSR
  (an undirected network can reuse its CSR).
- `GPUNetworkAccessibility` scores a retained matrix with per-frame parameters
  (`getGPUNetworkAccessibilityParameterValues`: threshold, decay `'none' | 'exponential' | 'power'`,
  beta, minimum cost for power decay). It outputs cumulative opportunities within the threshold,
  gravity accessibility, and the two- or three-step floating catchment area (`catchment`: `method`
  `'2sfca'` (default) or `'3sfca'`, supply on rows, demand per node, per-facility ratios, and
  per-node accessibility, with the decay applied inside each catchment). The catchment composes
  `GPUNetworkCostWeights` (the matrix as facility weights and their transpose, `rowCount x nodeCount`
  transient slots each) with `GPUCatchmentAccessibility`, the one floating catchment kernel; for a
  large matrix build sparser weights and run the two directly. `orientation: 'origin-rows'` scores a
  forward matrix whose rows are origins.
  Per-node sums gather rows in ascending order. Per-row sums use one workgroup and a fixed tree.
  Neither uses float atomics, so results are bitwise reproducible. Non-finite opportunity, supply or
  demand weights are missing values and are skipped; the encoder rejects a negative or non-finite
  `beta` and a `minimumCost` that is not positive and finite.

Encode the matrix graph when the network or the opportunity set changes. Encode the scoring graph
every frame: threshold, decay, and beta changes never re-run a search. The threshold must not
exceed the `costLimit` the matrix was built with.

```ts
matrixGraph.add(new GPUNetworkSnapping({positions: jobs, topology: {...topology, nodePositions},
  output: {snappedEdges, startNodes, startCosts}}));
matrixGraph.add(new GPUNetworkCostMatrix({topology: reverseTopology, startNodes, startCosts,
  startsPerRow: 2, costLimit: limit.importToGraph(matrixGraph), maximumIterations: 48,
  output: {costs: matrix}}));
scoreGraph.add(new GPUNetworkAccessibility({costs: matrixView, opportunityWeights: jobCounts,
  parameters: scoring.importToGraph(scoreGraph), cumulative, gravity}));
scoring.write(getGPUNetworkAccessibilityParameterValues({threshold: 1800, decay: 'exponential', beta: 0.002}));
```

{/* parameters: GPUNetworkAccessibility generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkAccessibility` parameters

- Packer: `getGPUNetworkAccessibilityParameterValues(settings)`, writes `GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUNetworkAccessibilityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `threshold` | `number` | Cost cutoff: only matrix entries with a finite cost `<= threshold` contribute. `Infinity` keeps every reached entry. Must not exceed the `costLimit` the matrix was built with. |
| `decay` (optional) | `GPUNetworkAccessibilityDecay` | Decay applied to the `gravity` and two-step floating catchment outputs. Defaults to `'none'`. |
| `beta` (optional) | `number` | Decay rate, finite and non-negative. Defaults to 0. |
| `minimumCost` (optional) | `number` | Power-decay floor so zero costs stay finite. Must be positive and finite. Defaults to 1. |

{/* /parameters */}

#### `GPUNetworkSnapping`

Project point locations onto the nearest road edges and return edge rows, fractions, distances and search start costs.

::lens{scenario=gpu-network.network-snapping rung=result caption="Compare 19 facility snaps; the longest moves about 95.8 metres onto its nearest road."}

#### How it computes

{/* lens:GPUNetworkSnapping */}

::lens{scenario=gpu-network.network-snapping rung=datum bind=page datum=3 caption="Inspect this facility: it snaps about 8.47 metres to edge 49."}

::lens{scenario=gpu-network.network-snapping rung=formula}

::lens{scenario=gpu-network.network-snapping rung=kernel}

::lens{scenario=gpu-network.network-snapping rung=graph}

#### `GPUNetworkCostMatrix`

Compute row-major travel costs from snapped facility starts to every road intersection or to selected target nodes.

::lens{scenario=gpu-network.network-cost-matrix rung=result caption="Read the first facility row: travel times to 1,199 intersections, reaching about 381.3 seconds."}

#### How it computes

{/* lens:GPUNetworkCostMatrix */}

::lens{scenario=gpu-network.network-cost-matrix rung=datum bind=page datum=648 caption="Inspect this intersection: travel time from the first facility is about 277.8 seconds."}

::lens{scenario=gpu-network.network-cost-matrix rung=kernel}

::lens{scenario=gpu-network.network-cost-matrix rung=graph}

#### `GPUNetworkAccessibility`

Reduce a facility-by-node cost matrix and opportunity weights into cumulative, gravity or floating-catchment accessibility scores.

::lens{scenario=gpu-network.network-accessibility rung=result caption="Find the highest gravity access score, about 2.373, among intersections near the facilities."}

#### How it computes

{/* lens:GPUNetworkAccessibility */}

::lens{scenario=gpu-network.network-accessibility rung=datum bind=page datum=318 caption="Inspect this intersection: its gravity score is about 2.110."}

::lens{scenario=gpu-network.network-accessibility rung=formula}

::lens{scenario=gpu-network.network-accessibility rung=kernel}

::lens{scenario=gpu-network.network-accessibility rung=graph}

### `GPUNetworkCostWeights`

Convert a facility-by-node travel-cost matrix into sparse catchment weights with a cost cutoff and distance decay.

::lens{scenario=gpu-network.network-cost-weights rung=result caption="See the 823 intersections within the first facility's cost limit; shade shows the decay weight."}

Turns a network cost matrix (the `output.costs` of `GPUNetworkCostMatrix`, one row per start row)
or a reachability result (one row per lane) into `GPUSpatialWeights` with a distance decay, so every
weights consumer runs on network costs. The main use is `GPUCatchmentAccessibility` (2SFCA and 3SFCA)
on travel times: with facilities on the matrix rows, `orientation: 'rows'` gives the facility weights
and `orientation: 'columns'` their exact transpose, the demand weights.

- An entry becomes a slot when its cost is finite, non-negative and `<= costLimit`. Its weight is the
  decay `f(c)` and its distance is the cost. Slots are laid out row by row with ascending neighbor
  IDs, so the weights are deterministic; a zero cost (a facility at its own demand location) is kept.
- Decays (per frame, `getGPUNetworkCostWeightsParameterValues`): `'none'` (1, the binary catchment),
  `'exponential'` `exp(-beta c)`, `'power'` `max(c, minimumCost)^-beta`, and `'gaussian'`, the Dai
  (2010) kernel `(exp(-c^2 / 2L^2) - exp(-1/2)) / (1 - exp(-1/2))` with `L = costLimit`, which is 1 at
  0 and 0 at the limit (weights consumers skip zero-weight slots).
- `columns` maps locations to matrix columns, for example the node each demand location is snapped
  to, so neighbor IDs index locations rather than nodes. Out-of-range columns are unreached.
- Capacity is `output.neighbors.length`; `rowCount x locations` always suffices. Slots past it are
  dropped, offsets are clamped to stay a consistent CSR, and `output.status.overflow` / `output.status.requiredCount`
  feed the `getGPUSpatialWeightsCapacity` grow loop.
- Passes: a per-row count (one workgroup per row), a `GPUScan` into `offsets`, an ordered per-row fill
  and a finalize pass. No atomics.

```ts
// `createWeightsViews(rows, capacity)` returns fresh `offsets`, `neighbors` and `weights` views.
const decayView = decay.importToGraph(graph);
const weights = (orientation: 'rows' | 'columns', rows: number) => {
  const contributor = new GPUNetworkCostWeights({
    id: `${orientation}-weights`, costs: matrix, rowCount: facilityCount, orientation,
    parameters: decayView, output: createWeightsViews(rows, facilityCount * nodeCount)
  });
  graph.add(contributor);
  return contributor.outputWeights;
};
graph.add(new GPUCatchmentAccessibility({
  method: '2sfca', supply, demand,
  facilityWeights: weights('rows', facilityCount), demandWeights: weights('columns', nodeCount),
  output: {accessibility}
}));
decay.write(getGPUNetworkCostWeightsParameterValues({costLimit: 1800, decay: 'gaussian'}));
```

{/* parameters: GPUNetworkCostWeights generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkCostWeights` parameters

- Packer: `getGPUNetworkCostWeightsParameterValues(settings)`, writes `GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUNetworkCostWeightsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `costLimit` | `number` | Catchment cost: an entry becomes a neighbor when its cost is finite, non-negative and `<= costLimit`. `Infinity` keeps every reached entry (not allowed with `'gaussian'`). |
| `decay` (optional) | `GPUNetworkCostWeightsDecay` | Decay that turns a cost into a weight. Defaults to `'none'`. |
| `beta` (optional) | `number` | Rate of the `'exponential'` and `'power'` decays, finite and non-negative. Defaults to 0. |
| `minimumCost` (optional) | `number` | Floor of the `'power'` decay so a zero cost stays finite, positive and finite. Defaults to 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUNetworkCostWeights */}

::lens{scenario=gpu-network.network-cost-weights rung=datum bind=page datum=0 caption="Inspect the first facility row: it retains 823 entries beginning at offset 0."}

::lens{scenario=gpu-network.network-cost-weights rung=formula}

::lens{scenario=gpu-network.network-cost-weights rung=kernel}

::lens{scenario=gpu-network.network-cost-weights rung=graph}

### `GPUNetworkIsochrones`

Turn road-network travel times into cost-band triangles or cell outlines around the reachable streets.

::lens{scenario=gpu-network.network-isochrones rung=result caption="Read the travel-time bands across 4,622 output triangles radiating from the start."}

Isochrone polygons from a network `topology` with `nodePositions` (pgRouting `pgr_drivingDistance`
with a buffer, QGIS service area polygons). Costs come from the caller (`costs`, for example
`GPUNetworkServiceAreas` `output.costs`) or from `startNodes`, which runs `GPUNetworkReachability`
and writes `costs`; `costs` and `assignments` therefore stay top-level props, inputs or
outputs depending on `startNodes`. With `startNodes`, `output.status.converged`
(one row) reports whether that search reached a fixpoint within `maximumIterations`, and
`localIterations` sets its hops per round. A 0 means the costs, and so every polygon, are truncated
too small. As on `GPUNetworkReachability` and `GPUNetworkServiceAreas` the flag is optional, and
omitting it accepts that truncation unchecked.

- **Raster path.** Every edge is sampled at up to `raster.maximumSamplesPerEdge` points. The cost is
  interpolated linearly from the source node along the edge weight, and the minimum (default) or
  maximum cost is written to the pixels within the walking buffer (`bufferRadius`, with cost
  `cost + walkCostPerUnit * distance`) using integer atomics on order-preserving float bits, so the
  result is independent of thread order. The raster feeds `GPUIsobands`: band `k` is cost in
  `[costBreaks[k - 1], costBreaks[k])`, and the last band is beyond the last break, which includes unreached
  space. The default `lastBand` is `breakCount - 1`, so unreached space gets no triangles.
- **Cell path.** Nodes with cost at most `cellCostLimit` are keyed to H3 or Quadbin cells
  (`GPUCellAggregation`, or `GPUPointToCell` then `GPUCellAggregation`) and outlined with
  [`GPUCellSetOutline`](/docs/api-reference/experimental/gpu-spatial-analysis/operators-cells#gpucellsetoutline).
  Positions must be longitude and latitude. `cellOutline.rings` (same options as
  `GPUCellSetOutline` `rings`: `vertexTolerance`, `normalizeWinding`, `output`) chains the outline into
  closed isochrone rings with shells, holes, shell assignment and an optional polygon layout
  ([`GPUSegmentRingAssembly`](/docs/api-reference/experimental/gpu-spatial-analysis/operators-cells#gpusegmentringassembly)).
  One run is one cost limit, so bands need one run each. The `polygons` output plugs into
  `GPUPointInPolygonJoin` for "is this demand point inside the isochrone", which
  `addDriveTimeCatchmentRecipe` wires as `isochrones.joinDemand`. The recipe is exported from this
  entry (`@luma.gl/experimental/gpu-network`) and documented with the other recipes in
  [spatial analysis](/docs/api-reference/experimental/gpu-spatial-analysis).
  For the recipe's `maximumSnapDistance`, no warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](../gpu-spatial-analysis/concepts.md#precision-of-longitudelatitude-positions).
- **Parameters.** Per-frame values are packed with
  `getGPUNetworkIsochronesParameterValues({breakCount, extent, bufferRadius, walkCostPerUnit,
  firstBand, lastBand, cellCostLimit})` into a 12-element float32 view. `costBreaks` is a float32 view
  whose length is the maximum break count. The raster size, `maximumBufferPixels` (default 6, at most
  16), `maximumSamplesPerEdge` (default 64, at most 1024), `unreachedCost` (default 1e30) and the
  mode are compile-time.
- **Accuracy.** Pixel accurate. Edges are straight node-to-node segments, the outer contour is inset
  by up to one pixel (interpolation toward `unreachedCost`), and the effective buffer is at least half
  a pixel diagonal, truncated at `maximumBufferPixels`. The raster is planar in the units of
  `topology.nodePositions`, so use projected meters for an isotropic buffer. An edge longer than
  `maximumSamplesPerEdge * bufferRadius / 2` leaves gaps.
- **Facilities.** `assignments` (a `uint32` view of the facility per node) is an input, or, when `startNodes`
  is given, an output: the nearest-facility allocation of `GPUNetworkServiceAreas` (ties to the lowest
  row; `maximumLabelIterations` bounds its rounds). Raster and outline capacity status is in
  `raster.output.status` and `cellOutline.output.status`. `raster.output.pixelFacilities` (minimum mode, at most 255
  sources; the pixel word keeps the top 24 cost bits, relative precision 2^-15) and
  `raster.output.triangleFacilities` label the raster path, and `raster.rings` writes band rings
  ([`GPUIsobandRings`](/docs/api-reference/experimental/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands)).
  On the cell path `cellOutline.byFacility` with `cellFacilities` gives each cell the facility of its
  cheapest node and outlines with groups, so rings never mix facilities (Quadbin is keyed through
  `GPUPointToCell`).
- **Coordinates.** The public `spatialContext` always holds a resolved context. Omitted means
  native planar for `raster`, or `{coordinateSpace: 'longitude-latitude', metric: 'none'}` for
  `cellOutline` alone. With both outputs and no context, the raster's planar default applies and
  cell-outline positions remain unchecked for compatibility. An explicit context checks both
  paths: longitude/latitude with `raster` throws (project into an AEQD frame first), and planar
  with `cellOutline` throws. Lambert equal-area frames are rejected because buffers measure distance.
- **Limits.** The cell path samples reached nodes only, so long edges on sparse networks drop cells. The
  rings follow the cell outline, so they have cell resolution. Three mid-latitude H3 pentagons (resolutions
  2, 3 and 5; alone, in disks, as holes and open rings) match h3-js `cellsToMultiPolygon`; the two polar
  pentagons enclose a pole and have no planar lng/lat orientation.

{/* parameters: GPUNetworkIsochrones generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkIsochrones` parameters

- Packer: `getGPUIsobandsParameterValues(settings, grid)`, writes `GPU_ISOBANDS_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): Frame raster grid; its width and height give the cell sizes and must match the contributor's.
- Packer: `getGPUNetworkIsochronesParameterValues(settings)`, writes `GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH` (12) `float32` values.

Settings of `getGPUIsobandsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `breakCount` | `number` | Number of active ascending breaks, at most the contributor's `maximumBreakCount` (larger values clamp on the GPU). `breakCount + 1` bands exist: band `k` covers `[b[k - 1], b[k])`. |
| `extent` | `readonly [number, number, number, number]` | Raster extent `[minX, minY, maxX, maxY]` in world units. Row 0 is at `minY`. |
| `firstBand` (optional) | `number` | First band emitted as geometry. Defaults to 0. Does not affect `bandClasses`. |
| `lastBand` (optional) | `number` | Last band (inclusive) emitted as geometry. Defaults to `breakCount` (every band). |
| `localOrigin` (optional) | `readonly [number, number]` | Float64 point that triangle vertices (and `GPUIsobandRings` ring vertices) are written relative to; add it back on the CPU or as the layer's coordinate origin. Defaults to `[0, 0]` (absolute vertices). Pass the raster or tile corner `[minX, minY]` for projected rasters: absolute float32 vertices are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about `1.2e-7` of their distance from the origin (1.2 cm at 100 km). |

Settings of `getGPUNetworkIsochronesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `breakCount` | `number` | Number of active ascending cost breaks, at most `costBreaks.length`. `breakCount + 1` bands exist: band `k` holds cost in `[costBreaks[k - 1], costBreaks[k])`, band 0 everything below the first break and the last band everything at or above the last active break (including unreached pixels). |
| `extent` | `readonly [number, number, number, number]` | Raster extent `[minX, minY, maxX, maxY]` in the context's length unit. Row 0 is at `minY`. |
| `bufferRadius` (optional) | `number` | Walking buffer radius in the context's length unit. Pixels within it of an edge sample get that sample's cost plus `walkCostPerUnit` times the distance. The effective radius is at least half a pixel diagonal so every sample marks its own pixel. Defaults to 0. |
| `walkCostPerUnit` (optional) | `number` | Cost per context length unit walked off the network inside the buffer. Defaults to 0. |
| `firstBand` (optional) | `number` | First band emitted as geometry. Defaults to 0. |
| `lastBand` (optional) | `number` | Last band (inclusive) emitted as geometry. Defaults to `breakCount - 1`, so unreached space has no triangles. |
| `cellCostLimit` (optional) | `number` | Cost limit of the cell-outline path: nodes with cost `<=` it contribute their cell. Defaults to `Infinity`, every reached node. |

{/* /parameters */}

#### How it computes

{/* lens:GPUNetworkIsochrones */}

::lens{scenario=gpu-network.network-isochrones rung=datum bind=page datum=1088 caption="Inspect the start intersection: its travel cost is 0 seconds before the raster splat."}

::lens{scenario=gpu-network.network-isochrones rung=formula}

::lens{scenario=gpu-network.network-isochrones rung=kernel}

::lens{scenario=gpu-network.network-isochrones rung=graph}

#### `addDriveTimeCatchmentRecipe`

Snap facilities and demand points to roads, assign drive-time catchments and count demand per cost band.

::lens{scenario=gpu-network.drive-time-catchment-recipe rung=result caption="Compare drive times from 100 demand points to their nearest of 4 facilities."}

#### How it computes

{/* lens:addDriveTimeCatchmentRecipe */}

::lens{scenario=gpu-network.drive-time-catchment-recipe rung=datum bind=page datum=5 caption="Inspect this demand point: its drive time is about 44.8 seconds, in band 0."}

::lens{scenario=gpu-network.drive-time-catchment-recipe rung=formula}

::lens{scenario=gpu-network.drive-time-catchment-recipe rung=kernel}

::lens{scenario=gpu-network.drive-time-catchment-recipe rung=graph}

### `GPUNetworkNoding`

Split raw centrelines at their intersections, merge end points within a tolerance, and return nodes, edge pieces and a network.

::lens{scenario=gpu-network.network-noding rung=result caption="Compare 16 raw centrelines with 134 noded pieces and 87 nodes; bridges are not distinguished."}

Maximum: `intersectionCapacity` is at most `getGPUNetworkNodingMaximumIntersectionCapacity(device)` (1048576 on a default device), a quarter of `getGPULineSplitMaximumPieceBound(device)`.

Builds a routable network from raw linestrings. `GPULineSplit` pieces
(see [GPU Spatial Analysis](/docs/api-reference/experimental/gpu-spatial-analysis/operators-lines#gpulinesplit-and-gpulinemerge))
are the edges. Their end points are snapped by a per-frame `tolerance` (a one-row `float32` view; integer
grid cells `floor((x - origin) / tolerance)` counted from the minimum corner of the live end points, so UTM
northings keep exact cells; 0 means exact), deduplicated by a stable two-pass `GPUSort`, and the node ID
is the group rank. Cells are exact up to 2^24 per axis; a wider extent sets `output.status.overflow`. Props: `lines`, `intersectionCapacity`, `tolerance`, and `output` with `pieces`, `nodes`
(`positions` and `status.count`, optional `status.requiredCount`), `edges` (`fromNodes`, `toNodes`, `lengths`) and `topology`
(`offsets`, `neighbors`, `weights`, optional `edgeIds`), with optional `output.status.overflow` and `output.status.uncertainCount`.
The topology is undirected and feeds `GPUNetworkReachability`, `GPUNetworkServiceAreas` and the other
contributors directly.

```ts
graph.add(new GPUNetworkNoding({
  lines: {kind: 'lines', positions, lineOffsets}, intersectionCapacity: 1 << 16,
  tolerance: tolerance.importToGraph(graph),
  output: {
    pieces, nodes: {positions: nodePositions, status: {count: nodeCount}},
    edges: {fromNodes, toNodes, lengths}, topology: {offsets, neighbors, weights}
  }
}));
```

Planar only: the snapping cells, `tolerance` and edge lengths are Euclidean, so a longitude/latitude
`spatialContext` throws. Project the lines to a planar frame first (for example `createGPULocalFrame`).

Limits: the tolerance merges end points only (end points that straddle a grid cell boundary stay separate),
and a line that stops near another line's interior is not connected to it. Crossings that should not
connect (bridges) cannot be excluded. Capacities are fixed, and `output.status.overflow` reports a truncated result.

#### How it computes

{/* lens:GPUNetworkNoding */}

::lens{scenario=gpu-network.network-noding rung=datum bind=page datum=12 caption="Inspect this noded piece: its length is about 969.4 metres."}

::lens{scenario=gpu-network.network-noding rung=kernel}

::lens{scenario=gpu-network.network-noding rung=graph}

### `GPUNetworkLineGraph`

Convert directed road edges into line-graph nodes and legal turns into arcs costed by travel and turn time.

::lens{scenario=gpu-network.network-line-graph rung=result caption="Compare road-edge midpoints linked by 5,635 legal turns; U-turns are excluded."}

Maximum: `topology` has at most `getGPUNetworkLineGraphMaximumEdgeCount(device)` (16777216 on a default device) edges, or `getGPUNetworkLineGraphMaximumGeodesicEdgeCount(device)` (8388608) with a great-circle `spatialContext`, because the edge direction vectors take 8 or 16 bytes per edge.

Turn-restricted routing as a line graph in CSR form: one node per directed edge of the road network and
one arc per allowed turn. The arc cost is `weights[next]` plus a turn cost from the turn angle: a base
`angleCost`, extra left and right costs, a U-turn cost or ban, and optional `bannedTurns` pairs. Props:
`topology` (with `nodePositions`), per-frame `parameters`
(`getGPUNetworkLineGraphParameterValues`), and `output`: a line-graph `topology` (`offsets`,
`neighbors`, `weights`, sized to the compile-time arc capacity, rows past the arc total hold
`0xffffffff` and -1), `arcCount` and the required `status.overflow`. Edge rows past `offsets[nodeCount]` get no arcs. `output.topology` is the `topology` of
`GPUNetworkReachability` and `GPUNetworkCostMatrix` unchanged; to route from an origin, start at the
out-edges of the origin node at their own weight. With a great-circle `spatialContext` positions are degrees and the turn
angle is measured between the great-circle arrival bearing of the incoming edge and the departure
bearing of the outgoing edge at the turn node, so turns are not skewed by the east-west stretch of
degrees. The active `bannedTurns` pairs are sorted by `fromEdge` on the GPU every frame and each turn
binary-searches them, so city-scale restriction sets cost `O(log bans)` per arc. The turn angle special-cases a
zero dot product because Metal returns the wrong sign for `atan2(y, -0.0)`. Start helpers and node
costs are not provided.

{/* parameters: GPUNetworkLineGraph generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkLineGraph` parameters

- Packer: `getGPUNetworkLineGraphParameterValues(settings?)`, writes `GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUNetworkLineGraphParameterValues`:

| Field | Type | Description |
|---|---|---|
| `angleCost` (optional) | `number` | Cost per radian of absolute turn angle, so a right angle costs `angleCost * pi / 2`. Defaults to 0. |
| `leftTurnCost` (optional) | `number` | Extra cost of a left turn (counter-clockwise, `angle > straightAngle`). Defaults to 0. |
| `rightTurnCost` (optional) | `number` | Extra cost of a right turn (clockwise, `angle < -straightAngle`). Defaults to 0. |
| `uTurnCost` (optional) | `number` | Cost of a U-turn (absolute angle at or above `uTurnAngle`), replacing the angle cost. A negative value bans U-turns. Defaults to -1 (banned). |
| `straightAngle` (optional) | `number` | Absolute angle in radians below which a turn counts as straight for left and right extras. Defaults to 0.5. |
| `uTurnAngle` (optional) | `number` | Absolute angle in radians at or above which a turn is a U-turn. Defaults to 2.9 (about 166 degrees). |
| `bannedTurnCount` (optional) | `number` | Number of active banned turns, at most half the length of `bannedTurns`. Defaults to 0. |

{/* /parameters */}

#### How it computes

{/* lens:GPUNetworkLineGraph */}

::lens{scenario=gpu-network.network-line-graph rung=datum bind=page datum=5 caption="Inspect this turn: its next edge is 1836, with cost about 44.8 seconds."}

::lens{scenario=gpu-network.network-line-graph rung=formula}

::lens{scenario=gpu-network.network-line-graph rung=kernel}

::lens{scenario=gpu-network.network-line-graph rung=graph}

### `GPUNetworkCostTable`

Search every node within a cost limit and store each reachable target and its cost in one sparse CSR table.

::lens{scenario=gpu-network.network-cost-table rung=result caption="Follow the start intersection's row of the 30-second table, which stores 23,449 entries network-wide."}

Precomputes a bounded all-pairs cost table of a static directed network, in the spirit of the UBODT
of fast map matching (Yang and Gidofalvi 2018). Row `u` lists every node `v` whose shortest-path cost
from `u` is finite and at most `costLimit`, `u` itself included at cost 0, as a CSR (`offsets`,
`targets`, `costs`) with ascending `targets`, so a consumer finds a cost by binary search of the row.
The table is exact up to its limit: a target missing from a row is farther than `costLimit`.
`GPUMapMatching.routeTable` is the first consumer.

- Props: `topology` (with `weights` as the travel cost), `costLimit` (one row, per frame), `laneCount`,
  `maximumIterations`, `localIterations` (default 16) and `output`: `offsets` (`nodeCount + 1` rows),
  `targets` (its length is the entry capacity), `costs` (same length) and optional
  `status.overflow`, `status.requiredCount` and `status.converged`.
- Method: source nodes are searched `laneCount` at a time with one lane-expanded
  `GPUNetworkReachability` per batch into one scratch block, then a per-lane count, a single-workgroup
  scan of the batch's lane counts on top of a GPU running entry cursor (which writes the batch's
  `offsets` rows), and a per-lane ordered fill. The batch index never reaches the shader, so every
  batch shares pipelines. Costs are the f32 fixpoint of reachability; the table is identical for every
  `laneCount`. No float atomics, deterministic.
- Capacity: entries past `output.targets.length` are dropped and `offsets` are clamped, so a reader
  always sees a consistent, possibly truncated CSR. `status.overflow` is 1 and `status.requiredCount`
  is the exact entry count (saturating at `0xffffffff`) for the grow loop. Do not use a table that
  overflowed or did not converge (`GPUMapMatching` ignores an overflowed table when you pass its flag as `routeTable.status.overflow`).
- Scaling: each batch sweeps its `laneCount x nodeCount` scratch block, so work and memory traffic are
  `O(nodeCount^2 / laneCount)` words per batch and `O(nodeCount^2)` words per build, plus the searches.
  That targets static networks up to roughly 1e5 nodes, built once and reused across many encodings; the
  table holds only the entries within `costLimit`. Rebuild only when weights change. The scratch block is
  `laneCount x nodeCount` floats and must fit one storage binding, so lower `laneCount` for large networks.

```ts
// Build once for a static network. The weights must equal the edge lengths GPUMapMatching measures
// from the node positions (here 100 m blocks, so a 700 m limit covers 7 blocks).
const table = new GPUNetworkCostTable({
  id: 'street-table',
  topology: {offsets, neighbors, weights: edgeLengths},
  costLimit: tableLimit,
  output: {
    offsets: tableOffsets, targets: tableTargets, costs: tableCosts,
    status: {overflow: tableOverflow, requiredCount: tableRequired, converged: tableConverged}
  }
});
graph.add(table);
graph.add(new GPUMapMatching({
  positions, trackOffsets, topology: {offsets, neighbors, nodePositions}, parameters,
  cellSize: 60, bounds, output: {matchedEdges},
  routeTable: {
    offsets: tableOffsets, targets: tableTargets, costs: tableCosts,
    costLimit: tableLimit, status: {overflow: tableOverflow}
  }
}));
```

#### How it computes

{/* lens:GPUNetworkCostTable */}

::lens{scenario=gpu-network.network-cost-table rung=datum bind=page datum=1088 caption="Inspect the start row: offsets 21,357 to 21,373 delimit 16 entries."}

::lens{scenario=gpu-network.network-cost-table rung=formula}

::lens{scenario=gpu-network.network-cost-table rung=kernel}

::lens{scenario=gpu-network.network-cost-table rung=graph}

### `GPUMapMatching`

Match noisy GPS tracks to directed road edges with candidate edges, bounded route searches and a Viterbi decode.

::lens{scenario=gpu-network.map-matching rung=result caption="Compare 3 simulated GPS tracks with their road-snapped runs; 137 of 139 fixes matched."}

Pass optional `spatialContext` in the settings of `getGPUMapMatchingParameterValues` to warn when finite positive `sigma` or `searchRadius` is below `10·Q` (16.97 m with the default sphere); see [Precision of longitude/latitude positions](../gpu-spatial-analysis/concepts.md#precision-of-longitudelatitude-positions).

Maximums: `topology` has at most `getGPUMapMatchingMaximumEdgeCount(device)` (11184810 on a default device) edges and `entryCapacity` is at most `getGPUMapMatchingMaximumEntryCapacity(device)` (33554432); the default `entryCapacity` is capped there and `status.overflow` reports lost entries.

Hidden Markov model map matching of GPS tracks to a directed network (Newson and Krumm 2009). Per point,
up to `candidateCount` (at most 8, default 8: a directed edge row is one candidate, so the default keeps the
4 nearest two-way roads) distinct directed edges within `searchRadius` come from an edge grid
rebuilt every encoding (`cellSize` and `bounds` are compile-time). The emission is Gaussian on the
perpendicular distance (`sigma`) and the transition exponential on `|straight - route|` (`beta`). The route
is the rest of the current edge, a bounded A* search (it stops at `routeFactor * straight + routeSlack`,
exits early once every target is settled, and uses a `routeNodeBudget` node table) and the start of the
next edge; a route longer than `routeFactor * straight + routeSlack`, partial end edges included, is
infeasible. Viterbi runs one thread per track.

- Props: `positions`, `trackOffsets`, the network `topology` (`offsets`, `neighbors`, `nodePositions`;
  weights unused), `parameters` (`getGPUMapMatchingParameterValues`: `sigma`, `beta`, `searchRadius`, `routeFactor`, `routeSlack`, per
  frame), `candidateCount`, `cellSize`, `bounds`, `entryCapacity`, `routeNodeBudget` and
  `spatialContext`.
- Geodesic: with `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` positions
  and nodes are degrees and every distance (emission, straight line, edge and route lengths, and the
  distance parameters) is in the context's length unit (meters or kilometers); the A* heuristic stays consistent. `cellSize` and `bounds` stay in
  degrees, the candidate window widens in longitude by `1 / cos` of its highest latitude, and longitudes
  are unwrapped around the grid center, so give bounds across the antimeridian (for example 179.9 to
  180.1) for a network that crosses it.
- Outputs (`output`): `matchedEdges` (a CSR row per point; `GPU_MAP_MATCHING_NONE` when unmatched),
  `matchedFractions`, `matchedOffsets`, `snapDistances`, `snappedPositions`, `breaks`,
  `trackLogLikelihoods`, `matchedCount`, `breakCount`, `status.overflow` and `status.routeOverflow`. A break restarts
  the model where no transition is feasible or a point has no candidate (`breaks` value 1). A break
  where a route search into the point dropped nodes is a budget break (value 2): the route may exist
  but did not fit the table. `status.overflow` is 1 when the edge grid or any route table dropped entries, and
  `status.routeOverflow` counts the dropped route nodes (saturating at `0xffffffff`).
- `routeTable` (optional): a `GPUNetworkCostTable` of the same network, built with `weights` equal to the
  edge lengths this contributor measures from the node positions (in the context's length unit), passed as `{offsets, targets, costs, costLimit, status?: {overflow}}`. A route search whose
  allowance (`routeFactor * straight + routeSlack` minus the tail of the source edge) is within the table's
  `costLimit` reads its network costs by binary search of the source node's row, with no search and no node
  budget, so it is exact and never counted in `status.routeOverflow`. Searches above the limit, or all of
  them while the table's `overflow` flag is 1, run the bounded A* search as before, so there is no cliff and
  no extra flag. Matched edges equal the A*-only results where A* is exact; route lengths differ only by f32
  rounding of the summed costs (about `2^-24` relative per hop). The table is built once and reused, which
  pays off for large track batches. Without `routeTable` nothing changes.
- Route budget: a full node table keeps the open nodes with the best A* keys and drops the rest. A
  search needs about as many nodes as the street blocks between two samples. On a 100 m street grid the
  default 64 covers samples up to about 800 m apart and 128 up to about 1.1 km. At 1 km and 64 it still
  misses routes (80 of 227 in a mimic, 171 under the old drop-in-discovery-order policy, which reported
  nothing). Each miss is reported, so sample more densely or raise `routeNodeBudget` when `status.overflow` is
  set. CSR rows past `offsets[nodeCount]` are not part of the network.
- The route passes run per `(point, previous candidate)` work item, so Viterbi is a cheap recursion: on a
  New York scene the matching graph went from 923 to 30 ms at a budget of 64 (20 ms at 40), with identical
  results.
- Limits: distances are planar unless a great-circle `spatialContext` is given, and edges are straight
  segments in the coordinates, so on longitude/latitude they should be road-length. The bounded search is
  an approximation that can overestimate or miss routes when samples are far apart for the budget
  (always reported, see above), and the route table is `pointCount * k * (k + 1)` floats. A two-way road needs both directions, and each takes a candidate slot.
  Exact routes and time-aware transitions are not provided.

{/* parameters: GPUMapMatching generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMapMatching` parameters

- Packer: `getGPUMapMatchingParameterValues(settings)`, writes `GPU_MAP_MATCHING_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUMapMatchingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `sigma` | `number` | GPS noise standard deviation in the context's length unit. Must be positive. |
| `beta` | `number` | Scale in the context's length unit of `exp(-\|straight - route\| / beta)`. Must be positive. |
| `searchRadius` | `number` | Candidate search radius in the context's length unit. Edges farther away are not candidates. Values above `2 * cellSize` are clamped to `2 * cellSize` (on longitude/latitude, to the context's length unit equivalent of `2 * cellSize` degrees of latitude). |
| `routeFactor` (optional) | `number` | A route, including the partial edges at both ends, may be at most `routeFactor * straightDistance + routeSlack` long; longer routes are infeasible. Defaults to 3. |
| `routeSlack` (optional) | `number` | Additive route allowance in the context's length unit, see `routeFactor`. Defaults to 50 (50 m in meters; pass for example 0.05 with a kilometer context). |

{/* /parameters */}

#### How it computes

{/* lens:GPUMapMatching */}

::lens{scenario=gpu-network.map-matching rung=datum bind=page datum=40 caption="Inspect this GPS fix: it snaps about 25.05 metres to edge 418 without a break."}

::lens{scenario=gpu-network.map-matching rung=kernel}

::lens{scenario=gpu-network.map-matching rung=graph}

### `GPUNetworkKFunction`

Count crash pairs by network distance band and return the network K value at each distance threshold.

::lens{scenario=gpu-network.network-k-function rung=result caption="Compare K for 58 snapped crashes against the seeded simulation minimum, mean and maximum."}

For `maximumSnapDistance` and `maximumDistance`, no warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](../gpu-spatial-analysis/concepts.md#precision-of-longitudelatitude-positions).

Maximum: `rowsPerBlock * events` is at most 4294967295 (one 1-D dispatch), checked under the contributor's own ID.

Network-constrained Ripley K function (Okabe and Yamada; spaghetti `GlobalAutoK`). Events are snapped with
`GPUNetworkSnapping`, then multi-source reachability searches limited to `maximumDistance` run in blocks of
`rowsPerBlock` rows (which bounds scratch memory). The lane-expanded CSR is built once for all rows and
shared by every block, so `rowsPerBlock` trades only the node count of the relax and initialize passes
against scratch memory. Unordered pairs below each of `bandCount`
thresholds are counted with integer atomics and `K = 2 * pairs * L / n^2` (spaghetti's normalization;
multiply by `n / (n - 1)` for the Okabe-Yamada ratio form), with `L` the per-frame `networkLength`.
Optional envelope: `simulationCount` random patterns (Philox, an edge drawn proportional to its length
through an integer prefix sum quantized against the GPU total of the live weights, so a wrong
`networkLength` changes K but never the draw). Only the live CSR rows `0..offsets[nodeCount]` are snapped
to or drawn; `networkLength` is half the sum of their weights. Blocks and batches run at a fixed shape on
64-element-aligned sub-views and read their first row from a per-block info word, so every block
shares one set of pipelines.

- Props: `positions`, the network `topology` (with `nodePositions`), `maximumDistance`,
  `networkLength`, `maximumSnapDistance`, `parameters` (`getGPUNetworkKFunctionParameterValues`: seed and
  active simulations), `bandCount` (at most `GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT`), `simulationCount`,
  `rowsPerBlock`, `candidateCapacity`. For many events pass `candidateCapacity` (with `maximumSnapDistance`) so
  snapping uses the BVH join instead of scanning every edge per event; `output.status.overflow` is then required and
  is set when the snap candidates overflow (K is then inexact). `spatialContext` is passed to
  `GPUNetworkSnapping`: a great-circle context snaps longitude/latitude events in the context's length unit (give
  `weights`, `maximumDistance`, `maximumSnapDistance` and `networkLength` in that unit too).
  `kValues` and `envelope` are also in the context's length unit.
- Outputs (`output`): `kValues`, optional `pairCounts`, `envelope` (min, mean and max per band), `snappedEventCount`,
  `status.overflow` and `status.converged`.
- Counts and K are pinned to spaghetti 1.7.6. Parallel edges between the same two nodes count as one road,
  and the envelope is the plain min, mean and max (spaghetti scales the extremes by the threshold).
  192 events with 20 patterns on 97,000 edges took about 61 ms (181 ms before the shared CSR expansion
  and BVH snapping), of which about 41 ms is relaxation. Cross-K between two patterns is not provided.
  Open: a block of 192 rows produced no result in one explorer capture (not investigated, so keep
  `rowsPerBlock` near the 128 MB lane budget of `recommendLaneCount`), and snap-candidate overflow is
  reported only through `status.overflow`.

{/* parameters: GPUNetworkKFunction generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNetworkKFunction` parameters

- Packer: `getGPUNetworkKFunctionParameterValues(settings)`, writes `GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUNetworkKFunctionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed of the simulated patterns, an integer in `[0, 2^53)`. |
| `activeSimulations` | `number` | Number of simulated patterns evaluated, at most the compile-time `simulationCount`. The remaining patterns are skipped (their K values and counts are 0). |

{/* /parameters */}

#### How it computes

{/* lens:GPUNetworkKFunction */}

::lens{scenario=gpu-network.network-k-function rung=datum bind=page datum=7 caption="Inspect the 1,000-metre threshold: 278 unordered crash pairs give K of about 35,210.9 metres."}

::lens{scenario=gpu-network.network-k-function rung=formula}

::lens{scenario=gpu-network.network-k-function rung=kernel}

::lens{scenario=gpu-network.network-k-function rung=graph}

### Rendering network contributor outputs with deck.gl

`@deck.gl-community/arrow-layers` renders the network contributors' node-aligned outputs without
copying or reading them back. `GPUGraphNodeLayer` keeps node positions as its only vertex
attribute and binds every other input as a read-only storage buffer indexed by instance:

| Layer prop | Typical contributor output |
| --- | --- |
| `colorColumn` / `sizeColumn` (`{buffer, format: 'uint32' \| 'float32'}`) | `GPUNetworkAnalyticsColumns` normalized degree, PageRank, core number; component or community labels; `GPUNetworkReachability` `output.bands` |
| `highlightMask` | `GPUNetworkNeighborhood` `output.nodeMask` |
| `pathRanks` (0 = off path, k = 1-based order) | `GPUNetworkPathExtraction` compact `ids`, scattered by rank |
| `filterMask` (0 hides the row from drawing **and** picking) | any `uint32` node mask |

`colorScale` (`linear` or `categorical`, domain, up to 8 palette stops, `nullColor` for
`0xffffffff`) and `sizeScale` live in a uniform block. Swapping a column, changing a domain or a
mask rebinds or rewrites that block and never rebuilds a pipeline; the block is uploaded only when
its contents change, and `layer.getRenderStats()` reports draws, style uploads and rebindings.
`GPUGraphEdgeLayer` draws an edge only when both endpoints pass `filterMask`, highlights it when both
endpoints are in `highlightMask`, and marks it on the path when both endpoints have consecutive
`pathRanks`.

`GPUGraphAnalysisColumns` (owned by `GPUGraphDeckEffect`) wires these contributors onto a symmetrized
topology of the original edge batches and encodes them inside deck.gl's frame encoder: analytics
once, and the neighborhood, reachability and path contributors only when `setHoverVertex`,
`setNeighborhoodHops` or `setPathEndpoints` change their imported input buffers.

The layers require WebGPU and throw on WebGL2: the contributors are WebGPU compute, and WebGL2 cannot
read storage buffers in the vertex stage. Render CPU-side columns with standard deck.gl layers.

### `GPUEdgeBundling`

Bend origin-destination lines toward shared corridors by iterating a density field, keeping both endpoints fixed.

::lens{scenario=gpu-network.edge-bundling rung=result caption="Compare 150 straight taxi flows with their bundled paths after 8 density iterations; endpoints stay fixed."}

Kernel-density edge bundling (KDEEB) as WebGPU compute, producing renderable polylines. Each edge
starts as a straight subdivision. Every iteration does four things:

1. Splats Epanechnikov weights of all control points into a fixed-point `atomic<u32>` density
   buffer.
2. Advects interior points one kernel radius along the bilinearly sampled normalized gradient.
3. Resamples each edge to uniform arc length and applies one Laplacian smoothing pass.
4. Anneals the radius by `lambda`.

Endpoints are pinned. Density is a storage buffer, so there is no float-renderable texture
requirement and no texture size cap.

Lengths are relative to a square work box computed on the GPU every encoding from live-edge
endpoints, so results are scale invariant. Dead edges (masked, out-of-range or non-finite) do not
bundle, and their path collapses onto the source position. Outputs (`output`):

- `paths`: `edgeCount * pointsPerEdge` `float32x2` rows, edge-major, in caller coordinates.
- `lineOffsets` (optional): for PathLayer binary attributes.
- `drawRecord` (optional): a four-word `drawIndirect` record `[pointsPerEdge, edgeCount, 0, 0]`
  for instanced line strips.

`pointsPerEdge` (2 to 64), `maximumIterations` (1 to 64) and `densityResolution` are
compile-time. When `parameters` is given, one gate node writes the indirect dispatches of every
iteration, so iterations beyond `activeIterations` dispatch nothing (the graph has `5 + 3 * maximumIterations`
nodes, plus one gate with `parameters` and one for `indices`). Each iteration reads its index from a
256-byte aligned iteration-info slot, so all iterations share one splat and one update pipeline. With a longitude/latitude `spatialContext`
`nodePositions` are longitude and latitude in degrees and longitude is
scaled by the cosine of the mid-latitude of the live endpoints (computed on the GPU, clamped at 0.01); the
output stays in degrees and endpoints are exact. Bundling measures nothing in physical units, so the
context's metric is not used, and data spanning the antimeridian is not supported. The per-frame
`parameters` view holds `[activeIterations, kernelRadius, lambda,
smoothing, stepScale]`; pack it with `getGPUEdgeBundlingParameterValues`. The edge list is
`sourceNodes` and `targetNodes` into `nodePositions`.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'bundling', format: 'uint32', length: 5,
  values: getGPUEdgeBundlingParameterValues({kernelRadius: 0.03}, 'uint32')});
graph.add(new GPUEdgeBundling({nodePositions, sourceNodes, targetNodes, edgeMask,
  output: {paths, lineOffsets, drawRecord}, pointsPerEdge: 16, maximumIterations: 15,
  densityResolution: 256, parameters: parameters.importToGraph(graph)}));
parameters.write(getGPUEdgeBundlingParameterValues({activeIterations: 6, kernelRadius: 0.05}, 'uint32'));
```

The normalized-gradient schedule is chaotic in f32. The f32 CPU oracle matches below 1e-5 of the
box after one iteration, and only statistically after that. Gradients below four fixed-point
quanta are treated as zero so a lone edge does not drift.

{/* parameters: GPUEdgeBundling generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUEdgeBundling` parameters

- Packer: `getGPUEdgeBundlingParameterValues(settings, format)`, writes `GPU_EDGE_BUNDLING_PARAMETER_LENGTH` (5) `uint32` values.
  - `format` (`'uint32'`)

Settings of `getGPUEdgeBundlingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `activeIterations` (optional) | `number` | Iterations to run, clamped to the compiled `maximumIterations`. Default: the compiled maximum. |
| `kernelRadius` (optional) | `number` | Initial kernel radius as a fraction of the work box side. Default 0.03. |
| `lambda` (optional) | `number` | Radius decay per iteration, clamped to `[0.5, 0.9]`. Default 0.85. |
| `smoothing` (optional) | `number` | Laplacian smoothing strength, clamped to `[0, 1]`. Default 0.5. |
| `stepScale` (optional) | `number` | Advection step multiplier. Default 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUEdgeBundling */}

::lens{scenario=gpu-network.edge-bundling rung=datum bind=page datum=10 caption="Inspect this flow: origin zone 5 connects to destination zone 2; its path starts at vertex 160."}

::lens{scenario=gpu-network.edge-bundling rung=iterations caption="Step through 5 key frames: the straight start, rounds 1, 2, 4, and the final output."}

::lens{scenario=gpu-network.edge-bundling rung=kernel}

::lens{scenario=gpu-network.edge-bundling rung=graph}

### `GPUFlowAggregation`

Group origin-destination records by zone pair and rank the pairs by summed trips, with record counts per pair.

::lens{scenario=gpu-network.flow-aggregation rung=result caption="Compare 450 taxi records with the top 32 of 150 zone pairs; the highlight is the busiest pair."}

Maximum: `rows * maximumProbeCount` is at most 4294967295, so inputs above 2^24 rows need a `maximumProbeCount` below the default.

Origin–destination aggregation for flow maps. Each row's origin and destination are assigned to
square-grid cells, hexagons, or caller zone IDs. Rows are then grouped by zone pair through a
compile-time-capacity `GPUHashIndex`, counted and weighted with `GPUGroupAggregation`, and ranked by
two stable `GPUSort` passes. The result is a deterministic top-K list (weight descending, ties by
origin then destination zone) with zone columns, counts, and weights for an `ArcLayer`, plus
per-zone outgoing and incoming totals. An optional caller mask and per-frame time window
(`getGPUTimeWindowFilterParameterValues`) gate rows without recompiling. All results are in
`output` (a `GPUCompactOutput` plus the flow and zone columns). `output.status.pairOverflow` reports a full
pair table, while `status.requiredCount > ids.length` only means the list was truncated to the top K.

For grid and hexagon zones, `zones.activeGridSize` is a per-frame packed `uint32` `[columns, rows]`
clamped to `1..gridSize`, so `gridSize` becomes the compile-time capacity: zone IDs are row-major in the
active size, zone outputs keep capacity length with zeros past `columns * rows`, and pair keys use the
capacity zone count (`GPUFlowAggregationActiveGridSize`). Pair it with the per-frame hexagon `radius`.
`zoneOutCountExtent` and `zoneInCountExtent` (two `float32` rows `[min, max]`) hold the extent of the
nonzero per-zone counts (both 0 when none), usable directly as a layer extent buffer.

The time gate accepts the same three timestamp forms as `GPUTimeWindowFilter`, including exact
`Int64` words with a `uint32` word window. `sumOrder` defaults to `'sorted'`: rows are sorted by pair and by zone and each group is summed in a
fixed tree, so `output.flowWeights`, `output.zoneOutWeights` and `output.zoneInWeights` are bitwise reproducible, at the
cost of three sorts and scans per encoding (38 instead of 22 nodes). `'atomic'` uses float
compare-exchange addition and serializes when rows share a pair (1M rows on one pair: 12-17 s against
22 ms sorted); it is faster only when nearly every row has its own pair. The sorted pair sum runs one
256-lane workgroup per stored pair, dispatched indirectly from the stored-pair count, so with weights and
`'sorted'` `pairCapacity` is at most 2^24.

```ts
graph.add(new GPUFlowAggregation({
  zones: {kind: 'hexagon', bounds: viewport.importToGraph(graph), gridSize: [64, 48], radius: 250},
  origins, destinations, weights: tripCounts,
  timeWindow: {timestamps: departures, window: window.importToGraph(graph)},
  pairCapacity: 8192, excludeSelfFlows: true,
  output: {
    ids: flowKeys, status: {count, overflow, requiredCount},
    flowOriginZoneIds, flowDestinationZoneIds, flowWeights, zoneOutCounts, zoneInCounts,
    drawInstanceCount: graph.importGPUData('arcs', arcDraws.getInstanceCountData(0))
  }
}));
```

{/* parameters: GPUFlowAggregation generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUFlowAggregation` parameters

- Packer: `getGPUTimeWindowFilterParameterValues(settings)`, writes `GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH` (8) `float32` values.
- Packer: `getGPUTimeWindowFilterWordParameterValues(settings)`, writes `GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH` (8) `uint32` values.

Settings of `getGPUTimeWindowFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number` | Inclusive window start. |
| `end` | `number` | Inclusive window end. A window with `end < start` accepts nothing. |
| `startFadeDuration` (optional) | `number` | Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. |
| `endFadeDuration` (optional) | `number` | Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. |

Settings of `getGPUTimeWindowFilterWordParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number \| bigint` | Inclusive window start. |
| `end` | `number \| bigint` | Inclusive window end. A window with `end < start` accepts nothing. |
| `startFadeDuration` (optional) | `number` | Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. |
| `endFadeDuration` (optional) | `number` | Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. |

{/* /parameters */}

#### How it computes

{/* lens:GPUFlowAggregation */}

::lens{scenario=gpu-network.flow-aggregation rung=datum bind=page datum=0 caption="Inspect the busiest pair: 3 records sum to 54,331 trips."}

::lens{scenario=gpu-network.flow-aggregation rung=formula}

::lens{scenario=gpu-network.flow-aggregation rung=kernel}

::lens{scenario=gpu-network.flow-aggregation rung=graph}
