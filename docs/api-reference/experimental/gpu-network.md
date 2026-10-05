import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Network

<ExperimentalDocsTabs active="gpu-network" />

:::caution Experimental
`@luma.gl/experimental/gpu-network` is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::

## Overview

`@luma.gl/experimental/gpu-network` provides analysis contributors for networks stored as CSR
adjacency (`offsets`, `neighbors`, optional `weights`): weighted reachability and isochrones, path
extraction, service areas, neighborhoods, snapping, cost matrices, accessibility scores, subgraph
filters, coarsening, network statistics, node-aligned analytics columns, origin-destination flow
aggregation, edge bundling, and adjacency-matrix images. Each contributor declares resources and
command nodes into a caller-owned [`GPUCommandGraph`](./gpu-core/gpu-command-graph.md) through
`getCommandNodes(graph)` (`GPUCommandNodeProducer`). It never compiles, submits, encodes, or reads
back.

## When to use it

Use GPU Network for map and road-network tasks that need bounded, renderable results: isochrone
bands that recolor as a cost limit moves, a path that follows a dragged endpoint, a statistics panel
or analytics columns for deck.gl styling, or aggregated flows and bundled edges for drawing.

The generic graph algorithms live in [GPU Graph](./gpu-graph.md), and the network contributors build
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
  GPUParameterBuffer
} from '@luma.gl/experimental/gpu-network';

const graph = new GPUCommandGraph(device, {id: 'network'});
const sources = new GPUParameterBuffer(device, {id: 'sources', format: 'uint32', length: 4});
const costLimit = new GPUParameterBuffer(device, {id: 'cost-limit', format: 'float32', length: 1});
const thresholds = new GPUParameterBuffer(device, {id: 'thresholds', format: 'float32', length: 4});

// `offsets`, `neighbors`, `weights`, `costs`, `bands`, `bandCounts`, and `converged` are graph
// views created on `graph`. Build the CSR on the GPU with `GPUCOOToCSR`.
graph.add(new GPUNetworkReachability({
  offsets, neighbors, weights, costs, bands, bandCounts, converged,
  sources: sources.importToGraph(graph), costLimit: costLimit.importToGraph(graph),
  bandThresholds: thresholds.importToGraph(graph), maxIterations: 48
}));
const compiled = graph.compile(); // once

// every frame: rewrite the source vertex and cost limit, encode the same compiled graph
sources.write(new Uint32Array([startVertex, 0, 0, 0]));
costLimit.write(new Float32Array([1800]));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids`, `count`, `overflow`, optional `totalCount`). `count` is clamped to `ids.length` and can be
  an indirect draw instance count; `overflow` is rewritten every encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **CSR convention.** Undirected graphs list both directions in the CSR. A reverse CSR, built with
  `GPUCOOToCSR`, gives "cost to reach" semantics.

## API

### `GPUNetworkReachability`

Single- or multi-source shortest-path costs over a directed CSR road network with isochrone bands,
band counts, predecessors, and a convergence flag. Relaxation is chaotic Bellman-Ford over a compact
GPU frontier queue with a round-stamped visited set: each of the `maxIterations` rounds is one
indirect dispatch sized by the previous round's pushes, so converged frames dispatch nothing and
nothing is read back. Within a round each workgroup chains up to `localIterations` (default 16)
hops through workgroup memory, so a path of `h` hops needs about `h / localIterations` rounds; costs
are identical for every setting. Predecessors are cycle-safe: the smallest strictly cheaper tight
in-neighbor, or across equal-cost (zero-weight) edges the smallest tight in-neighbor one BFS level
closer to a strict entry or a source, found in up to `maxTieIterations` extra rounds (default 4).
`maxIterations` counts rounds, one graph node each, not hops. With the default `localIterations`,
a budget of about `h / 16` plus a few rounds suffices; 48 covers the demo networks.
Build CSR on the GPU with `GPUCOOToCSR`; a reverse CSR gives "cost to reach" isochrones.

```ts
graph.add(new GPUNetworkReachability({
  offsets, neighbors, weights, costs, bands, bandCounts, converged,
  sources: sources.importToGraph(graph), costLimit: costLimit.importToGraph(graph),
  bandThresholds: thresholds.importToGraph(graph), maxIterations: 48
}));
```

### `GPUNetworkPathExtraction`, `GPUNetworkServiceAreas`, `GPUNetworkNeighborhood`, and `GPUNetworkAnalyticsColumns`

Network analysis over the same directed CSR road network as `GPUNetworkReachability`.

- `GPUNetworkPathExtraction` walks a reachability predecessor array from per-frame targets. It publishes ordered (source to target) node lists and resolved CSR edge lists as `GPUCompactOutput`s, together with per-target offsets, costs, and found flags. Each walk is bounded by a compile-time `maxPathLength`, so cyclic or garbage predecessors terminate and raise `overflow`.
- `GPUNetworkServiceAreas` assigns every node to its nearest per-frame facility. Ties go to the smallest facility row. It also reports per-facility node counts and the total cost each facility serves. It composes `GPUNetworkReachability` with a GPU-gated min-label pass over the tight shortest-path edges.
- `GPUNetworkNeighborhood` publishes k-hop ego networks around per-frame seeds, with per-frame `k` up to a compile-time `maxHops`. Outputs are hop distances, masks, and compact node and induced-edge IDs.
- `GPUNetworkAnalyticsColumns` runs `gpu-graph` degree, PageRank, connected components, core number, and label-propagation communities directly on the caller's CSR views through a `GPUGraphTopologyView`, so a transient CSR built in the same graph by `GPUCOOToCSR` works as well as imported buffers. It publishes node-aligned columns, optionally normalized to `[0, 1]` with a GPU extent.

```ts
graph.add(new GPUNetworkReachability({offsets, neighbors, weights, costs, predecessors,
  sources: origin.importToGraph(graph), maxIterations: 48}));
graph.add(new GPUNetworkPathExtraction({predecessors, costs,
  targets: destination.importToGraph(graph), output: route,
  edges: {offsets, neighbors, weights, output: routeEdges}}));
graph.add(new GPUNetworkServiceAreas({offsets, neighbors, weights, assignments, facilityNodeCounts,
  facilities: stations.importToGraph(graph), costLimit: limit.importToGraph(graph)}));
graph.add(new GPUNetworkNeighborhood({offsets, neighbors, hopDistances, nodes, edges,
  seeds: picked.importToGraph(graph), hops: hops.importToGraph(graph), maxHops: 8}));
graph.add(new GPUNetworkAnalyticsColumns({offsets, neighbors,
  pageRank: {output: rank, normalized: rankSize}, coreNumber: {output: core}}));
```

### `GPUNetworkStatistics`

Statistics for a graph panel over the same CSR as `GPUNetworkAnalyticsColumns`, reduced on the GPU
into one caller-owned `uint32` summary. It reports live vertex, edge and slot counts, weak component
count and largest component size, isolated vertices, maximum degrees, out-, in- and total-degree
histograms (`'linear'` or `'log2'` bins), and the modularity of a caller-supplied community
labeling. Optional vertex and edge masks restrict every statistic to live rows. Components run
`GPUGraphConnectedComponents` on a masked copy of the adjacency, so masking a bridge vertex splits
its component. Modularity is accumulated with exact integer atomics and a fixed-order f32 finish,
so it is deterministic; a self-loop is counted once, unlike `GPUGraphModularity`. The resolution
and linear bin width are per-frame parameters (`encodeGPUNetworkStatisticsParameters`). Read the
summary back and decode it with `decodeGPUNetworkStatistics`. The contributor owns small scratch
buffers imported under fixed IDs, so one instance belongs to one graph.

```ts
const summary = graph.importGPUData('stats', statsData);
graph.add(new GPUNetworkStatistics({offsets, neighbors, vertexMask, communities,
  parameters: statsParameters.importToGraph(graph), degreeBinning: 'log2', output: summary}));
// after the application submits the encoding
const stats = decodeGPUNetworkStatistics(new Uint32Array(await statsBuffer.readAsync()));
```

### `GPUNetworkSubgraphFilter`

Turns attribute and time predicates on vertex and edge rows into a consistent induced subgraph of a forward CSR network.

- Vertex `v` is live when its caller mask is nonzero and every enabled vertex range accepts its f32 column value. Ranges are half-open (`min <= value < max`); NaN never passes.
- Slot `u -> v` is live when its caller mask is nonzero, every enabled edge range and time gate accepts it, and both endpoints are live. The f32 `edgeTimes` window and the exact Int64 `edgeTimeWords` window are closed (`start <= t <= end`).
- Undirected graphs list each edge in both slots. By default an edge is live only if both slots pass, so `edgeMask` is always symmetric; set `pairUndirectedSlots: false` to skip the O(degree) pairing when columns are already symmetric.
- `dropIsolated` also kills vertices with no live incident slot.
- Ranges live in one f32 `parameters` view (`getGPUNetworkSubgraphFilterParameterValues`) and the time-word window in `timeWordParameters` (`getGPUTimeWindowWordParameterValues`); rewriting them never recompiles.
- Outputs: `vertexMask` and `edgeMask` (1 live, 0 dead; feed `GPUNetworkStatistics` and `filterMask`), `counts` (decode with `decodeGPUNetworkSubgraphFilterCounts`), compact `liveVertices` and `liveEdgeSlots` (stable ascending, overflow-reporting), and an `inducedCSR` (`offsets`, `neighbors` over the original vertex ids, optional `sourceSlots`, `overflow`) built with a prefix sum over live degree.
- Results are exact: integer arithmetic and comparisons only.

```ts
const filter = new GPUNetworkSubgraphFilter({
  offsets, neighbors,
  vertexColumns: [population], edgeColumns: [speed], parameters,
  dropIsolated: true,
  output: {vertexMask, edgeMask, counts}
});
graph.add(filter);
parameterBuffer.write(
  getGPUNetworkSubgraphFilterParameterValues(layout, {vertexRanges: [[1000, Infinity]], edgeRanges: [[30, 90]]})
);
```

### `GPUNetworkCoarsening`

Summarizes a CSR network by a vertex group label (a community, component or spatial cell) for compound nodes when zoomed out. Supernode outputs: `groupVertexCount` plus optional `groupIntraEdgeCount`, `groupIntraWeight`, `groupCentroid` (`float32x2`), `groupBounds` (`float32x4`) and `groupValueSum`, each with exactly `groupCapacity` rows. Superedges are a bounded `edges` output (`ids` is the source group) with `edgeTargets`, `edgeCounts` and optional `edgeWeights`, sorted by `(source, target)`; undirected superedges have `source < target`. An optional 8-word `summary` reports live and overflowed vertices, counted/dropped/intra/inter edges, group count and the unclamped superedge count.

```ts
const coarsening = new GPUNetworkCoarsening({
  offsets, neighbors, labels, positions, groupCapacity: 1024,
  groupVertexCount, groupCentroid, groupBounds,
  edges: {ids: sourceGroups, count, overflow}, edgeTargets, edgeCounts, edgeWeights
});
graph.add(coarsening);
```

Labels are dense group IDs below the compile-time `groupCapacity` (at most 65535); a live vertex above it is excluded and sets `edges.overflow`. Undirected CSRs list each edge in both directions and each edge counts once. Counts are integer exact. Weight, position and value sums use 64-bit fixed-point atomics (`fixedPointScale`, default 65536), so they are bitwise reproducible. Superedges use a stable sort and segment, so order and the kept prefix on overflow are deterministic. Inputs must be finite.

### `GPUAdjacencyMatrix` and `GPUAdjacencyMatrixOrder`

Bins a forward CSR into an `R x R` matrix image for a matrix view linked to a node-link view. Row-major `uint32` counts (`row * R + column`), optional fixed-point weight sums, one-row maxima for color normalization, and an optional `r32float` storage texture. The optional `order` view maps vertex to matrix position (a permutation computed by the caller, for example with `GPUAdjacencyMatrixOrder` or `computeAdjacencyMatrixOrder`; identity by default). Per-frame zoom is a four-word window `[rowStart, rowEnd, colStart, colEnd)` in matrix positions (`encodeGPUAdjacencyMatrixWindow`); position `p` lands in bin `floor((p - start) * R / (end - start))` and positions outside the window are dropped, so panning and zooming never recompile.

```ts
const matrix = new GPUAdjacencyMatrix({
  offsets, neighbors, weights, order, window: windowParameters.importToGraph(graph),
  resolution: 64,
  output: {counts, weightSums, maxCount, maxWeightSum}
});
graph.add(matrix);
windowParameters.write(encodeGPUAdjacencyMatrixWindow({rowStart: 0, rowEnd: 500, colStart: 0, colEnd: 500}, 64));
```

Undirected graphs follow the convention that the CSR lists both directions, so each slot is written once and the matrix is symmetric; pass `mirrorSlots: true` for an edge-list CSR that lists each edge once and each slot is also written transposed (self-loops once). Directed: row = source. Masks follow the `GPUNetworkStatistics` liveness rule. Counts are exact `u32` atomics. Weight sums are fixed-point `u32` atomics of `round(weight * weightScale)` (default scale 1024), so results are bit-identical regardless of thread order; the cost is half a step of quantization per edge and wraparound at 2^32. `GPUAdjacencyMatrixOrder` builds `order` from a `uint32` group label and an optional tie key with two stable `GPUSort` passes and matches `computeAdjacencyMatrixOrder` exactly.

### `GPUNetworkSnapping`, `GPUNetworkCostMatrix`, and `GPUNetworkAccessibility`

Accessibility over the same directed CSR road network as `GPUNetworkReachability`, in three steps.

- `GPUNetworkSnapping` snaps planar points (origins, facilities, opportunities) to the nearest
  network edge, given node positions and a COO edge list or CSR `offsets`. For each point it
  outputs the edge row (ties go to the smallest row, so a point on a node takes the smallest
  incident edge), the fraction along the edge, the snap distance, the snapped position, and the
  two seed costs: `fraction * edgeCost` to the edge source and `(1 - fraction) * edgeCost` to the
  target. The edge cost defaults to the planar length. `seedNodes` / `seedCosts` (two rows per point)
  feed a search directly. `seedDirection` picks which endpoints count: `'both'` for undirected
  networks, `'forward'` for a point that leaves along a one-way edge, `'reverse'` for a point that
  a one-way edge arrives at. Without `candidateCapacity`, every point scans every edge, which is
  exact. With it, candidates come from `GPUNearestFeatureJoin` over the edge segments within a
  per-frame `maxSnapDistance`, and `overflow` reports when capacity runs out.
- `GPUNetworkCostMatrix` computes a bounded many-to-all cost matrix: one shortest-path search per
  row, from that row's seeds (`seedRows`, or `seedsPerRow: 2` for snapped points) to every node.
  It batches `laneCount` rows (default 32) into one `GPUNetworkReachability` over a lane-expanded
  copy of the CSR. Lane `l`, node `u` becomes node `l * nodeCount + u`, so the frontier rounds,
  hop chaining, per-frame `costLimit`, and convergence flag serve all lanes at once. The matrix is
  bit-identical for every `laneCount`. Rows set the number of searches, so put the smaller side on
  rows. To score every node, put opportunities or facilities on rows and search the reverse CSR
  (an undirected network can reuse its CSR).
- `GPUNetworkAccessibility` scores a retained matrix with per-frame parameters
  (`encodeGPUNetworkAccessibilityParameters`: threshold, decay `'none' | 'exponential' | 'power'`,
  beta, minimum cost for power decay). It outputs cumulative opportunities within the threshold,
  gravity accessibility, and the two-step floating catchment area (`catchment`: supply on rows,
  demand per node, per-facility ratios, and per-node accessibility, with the decay applied inside
  each catchment). `orientation: 'origin-rows'` scores a forward matrix whose rows are origins.
  Per-node sums gather rows in ascending order. Per-row sums use one workgroup and a fixed tree.
  Neither uses float atomics, so results are bitwise reproducible.

Encode the matrix graph when the network or the opportunity set changes. Encode the scoring graph
every frame: threshold, decay, and beta changes never re-run a search. The threshold must not
exceed the `costLimit` the matrix was built with.

```ts
matrixGraph.add(new GPUNetworkSnapping({points: jobs, nodePositions, offsets, edgeTargets: neighbors,
  edgeCosts: weights, snappedEdges, seedNodes, seedCosts}));
matrixGraph.add(new GPUNetworkCostMatrix({offsets: reverseOffsets, neighbors: reverseNeighbors,
  weights: reverseWeights, seedNodes, seedCosts, seedsPerRow: 2, costs: matrix,
  costLimit: limit.importToGraph(matrixGraph), maxIterations: 48}));
scoreGraph.add(new GPUNetworkAccessibility({costs: matrixView, opportunityWeights: jobCounts,
  parameters: scoring.importToGraph(scoreGraph), cumulative, gravity}));
scoring.write(encodeGPUNetworkAccessibilityParameters({threshold: 1800, decay: 'exponential', beta: 0.002}));
```

### Rendering network contributor outputs with deck.gl

`@deck.gl-community/arrow-layers` renders the network contributors' node-aligned outputs without
copying or reading them back. `GPUGraphNodeLayer` keeps node positions as its only vertex
attribute and binds every other input as a read-only storage buffer indexed by instance:

| Layer prop | Typical contributor output |
| --- | --- |
| `colorColumn` / `sizeColumn` (`{buffer, format: 'uint32' \| 'float32'}`) | `GPUNetworkAnalyticsColumns` normalized degree, PageRank, core number; component or community labels; `GPUNetworkReachability` `bands` |
| `highlightMask` | `GPUNetworkNeighborhood` `nodeMask` |
| `pathRanks` (0 = off path, k = 1-based order) | `GPUNetworkPathExtraction` compact `ids`, scattered by rank |
| `filterMask` (0 hides the row from drawing **and** picking) | any `uint32` node mask |

`colorScale` (`linear` or `categorical`, domain, up to 8 palette stops, `nullColor` for
`0xffffffff`) and `sizeScale` live in a uniform block. Swapping a column, changing a domain or a
mask rebinds or rewrites that block and never rebuilds a pipeline; the block is uploaded only when
its contents change, and `layer.getRenderStats()` reports draws, style uploads and rebindings.
`GPUGraphEdgeLayer` draws an edge only when both endpoints pass `filterMask`, highlights it when both
endpoints are in `highlightMask`, and marks it on the path when both endpoints have consecutive
`pathRanks`.

`GPUGraphRecipeColumns` (owned by `GPUGraphDeckEffect`) wires these contributors onto a symmetrized
topology of the original edge batches and encodes them inside deck.gl's frame encoder: analytics
once, and the neighborhood, reachability and path contributors only when `setHoverVertex`,
`setNeighborhoodHops` or `setPathEndpoints` change their imported input buffers.

The layers require WebGPU and throw on WebGL2: the contributors are WebGPU compute, and WebGL2 cannot
read storage buffers in the vertex stage. Render CPU-side columns with standard deck.gl layers.

### `GPUEdgeBundling`

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
bundle, and their path collapses onto the source position. Outputs:

- `paths`: `edgeCount * pointsPerEdge` `float32x2` rows, edge-major, in caller coordinates.
- `startIndices` (optional): for PathLayer binary attributes.
- `drawRecord` (optional): a four-word `drawIndirect` record `[pointsPerEdge, edgeCount, 0, 0]`
  for instanced line strips.

`pointsPerEdge` (2 to 64), the maximum `iterations` (1 to 64) and `densityResolution` are
compile-time. The per-frame `parameters` view holds `[activeIterations, kernelRadius, lambda,
smoothing, stepScale]`; pack it with `createGPUEdgeBundlingParameterValues`.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'bundling', format: 'uint32', length: 5,
  values: createGPUEdgeBundlingParameterValues({kernelRadius: 0.03}, 'uint32')});
graph.add(new GPUEdgeBundling({positions, sourceVertices, targetVertices, edgeMask,
  paths, startIndices, drawRecord, pointsPerEdge: 16, iterations: 15, densityResolution: 256,
  parameters: parameters.importToGraph(graph)}));
parameters.write(createGPUEdgeBundlingParameterValues({activeIterations: 6, kernelRadius: 0.05}, 'uint32'));
```

The normalized-gradient schedule is chaotic in f32. The f32 CPU oracle matches below 1e-5 of the
box after one iteration, and only statistically after that. Gradients below four fixed-point
quanta are treated as zero so a lone edge does not drift.

### `GPUFlowAggregation`

Origin–destination aggregation for flow maps. Each row's origin and destination are assigned to
square-grid cells, hexagons, or caller zone IDs. Rows are then grouped by zone pair through a
compile-time-capacity `GPUHashIndex`, counted and weighted with `GPUGroupAggregation`, and ranked by
two stable `GPUSort` passes. The result is a deterministic top-K list (weight descending, ties by
origin then destination zone) with zone columns, counts, and weights for an `ArcLayer`, plus
per-zone outgoing and incoming totals. An optional caller mask and per-frame time window
(`getGPUTimeWindowParameterValues`) gate rows without recompiling. `pairOverflow` reports a full
pair table, while `totalCount > ids.length` only means the list was truncated to the top K.

The time gate accepts the same three timestamp forms as `GPUTimeWindowFilter`, including exact
`Int64` words with a `uint32` word window. Weight sums use atomic float addition by default;
`sumOrder: 'sorted'` sorts rows by pair and by zone and sums each group in a fixed tree, so
`flowWeights`, `zoneOutWeights` and `zoneInWeights` are bitwise reproducible, at the cost of
three sorts and scans per encoding (38 instead of 22 nodes).

```ts
graph.add(new GPUFlowAggregation({
  zones: {kind: 'hexagon', bounds: viewport.importToGraph(graph), gridSize: [64, 48], radius: 250},
  origins, destinations, weights: tripCounts,
  timeWindow: {timestamps: departures, window: window.importToGraph(graph)},
  pairCapacity: 8192, excludeSelfFlows: true,
  output: {ids: flowKeys, count, overflow, totalCount},
  flowOriginZoneIds, flowDestinationZoneIds, flowWeights, zoneOutCounts, zoneInCounts,
  drawInstanceCount: graph.importGPUData('arcs', arcDraws.getInstanceCountData(0))
}));
```

## Limits and compatibility

- GPU Network is experimental and WebGPU-only.
- CSR graphs use packed views only. Edge weights must be non-negative where a contributor takes costs.
- Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
  (`dev-docs/roadmaps/gpugraph-roadmap.md`) tracks what is still open.

## Related modules

- [GPU Graph](./gpu-graph.md) supplies the generic algorithms, topology views, and layouts.
- [GPU Raster](./gpu-raster/README.md) has the raster counterpart of cost distance and isolines.
- [GPU Core](./gpu-core/concepts.md) defines contributors, composition levels, and graph ownership.
