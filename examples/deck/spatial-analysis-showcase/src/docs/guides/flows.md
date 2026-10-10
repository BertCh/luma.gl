---
title: Flows and graphs
summary: Rank origin-destination flows, bundle edges into corridors, and read a network as a matrix or a coarsened summary.
order: 12
---

A flow dataset says that something moved from A to B: taxi trips, commuters, airline routes. The map problem is always the same. There are too many edges to draw, and the ones that matter are lost in the crowd. The flows chapter has three families of tools, each a different way of making a large edge set readable.

## Which tool when

| Question | Tool | Scene |
| --- | --- | --- |
| Which zone pairs carry the most movement, and when? | `GPUFlowAggregation` | [#/story/taxi-flows](#/story/taxi-flows) |
| Where do many edges share a corridor? | `GPUEdgeBundling` | [#/story/flight-bundling](#/story/flight-bundling) |
| Who connects to whom, and is the network organised in groups? | `GPUAdjacencyMatrix`, `GPUAdjacencyMatrixOrder` | [#/story/flight-matrix](#/story/flight-matrix) |
| What does the network look like zoomed out? | `GPUNetworkCoarsening` | [#/story/flight-matrix](#/story/flight-matrix) |

## Aggregating flows: GPUFlowAggregation

[`GPUFlowAggregation`](#/reference/GPUFlowAggregation) takes one row per movement (an origin, a destination, a weight and optionally a timestamp), assigns each end to a **zone**, sums the weight of every zone pair in a GPU hash table and ranks the pairs. The output is a weight-sorted top-K list, per-zone departure and arrival totals, and an instance count for an indirect draw.

Key options:

- **Zones.** Caller IDs (`ids`) when you already have areas, or a `hexagon` or `grid` lattice built from positions. The lattice dimensions and hexagon radius are per-frame values under a compile-time capacity, so a zone-size slider never recompiles. The same trips aggregated into different zones give different pictures: the modifiable areal unit problem.
- **Time window.** Rows outside a half-open window are gated on the GPU. The window is eight floats in a parameter buffer, so scrubbing or playing it re-runs the same graph.
- **Weights and mask.** A weight buffer (trips, fares, jobs by income) and a per-row mask (weekday or weekend) are rewritten without a rebuild.
- **`excludeSelfFlows`.** Drops rows whose two ends fall in one zone. Compile-time.
- **`sumOrder`.** `sorted` (default) accumulates in a fixed tree: bitwise identical on every run and device. `atomic` skips the sorts and only wins when nearly every row has its own pair; its rounding depends on scheduling.
- **Top-K and overflow.** `requiredCount` larger than K means the list is a truncation but every aggregate is exact. `pairOverflow` means the hash table filled and aggregates are incomplete.

Pitfall: zone totals count every accepted record, not every weight unit, so colour zones by the weight totals (`zoneOutWeights`, `zoneInWeights`) when rows are not single trips.

## Bundling edges: GPUEdgeBundling

[`GPUEdgeBundling`](#/reference/GPUEdgeBundling) implements kernel-density edge bundling. Each edge is a polyline with its endpoints pinned. Every iteration splats the density of all control points, moves interior points up the density gradient, resamples them evenly and smooths them. The radius shrinks by a decay factor each round, so lines merge into bundles and then tighten.

- **Per-frame:** active iterations, kernel radius, radius decay, smoothing (stiffness), step scale, and the edge mask. Moving a slider re-encodes the same graph.
- **Compile-time:** control points per edge, the maximum iteration count, the density grid resolution, and `geographic` (longitude is scaled by the cosine of the latitude so the kernel is round on the ground).

Pitfalls: bundled lines are not routes, and curvature has no meaning beyond grouping. The kernel radius is relative to the box around the live edges, so a few far-away edges make every bundle looser; mask them out rather than lowering the radius. Straight lines in longitude and latitude cross the whole map for pairs that span the antimeridian; split such edges (the scenes do) or filter them.

## Matrix view and ordering: GPUAdjacencyMatrix

[`GPUAdjacencyMatrix`](#/reference/GPUAdjacencyMatrix) bins a CSR graph into an R by R image with exact integer counts and fixed-point weight sums. A **window** of four words picks the visible range, so zooming and panning rewrite four numbers. [`GPUAdjacencyMatrixOrder`](#/reference/GPUAdjacencyMatrixOrder) produces the permutation the matrix reads: it sorts vertices by a group label and then by a tie key (for example degree). The result is bit-identical to the CPU helper `computeAdjacencyMatrixOrder`.

A matrix is only as informative as its ordering. With arbitrary order it is speckle; with community order it shows blocks, bright on the diagonal for within-group links. Weight sums are `round(weight * 1024)` per edge, so they are order independent but wrap at 2^32 in total; scale weights down if one cell can hold a lot of them.

## Coarsening: GPUNetworkCoarsening

[`GPUNetworkCoarsening`](#/reference/GPUNetworkCoarsening) collapses a graph by a vertex label into **supernodes** (vertex count, centroid, bounds, summed values, intra-group edges and weight) and a sorted list of **superedges** (group pair, edge count, summed weight). It is the zoomed-out summary: show supernodes at low zoom and the real nodes at high zoom. Labels must be dense and below the compile-time group capacity; sparse labels need remapping first, and the `summary` words report dropped edges and label overflow.

## Reading these maps

Arcs and bundles are visual summaries. Check the readouts next to the map: the share of weight carried by the drawn flows, whether the top-K list was truncated, path stretch and map coverage for bundling, the fill and brightest cell of a matrix. Then change one option at a time and watch what moves; each scene's story does exactly that.
