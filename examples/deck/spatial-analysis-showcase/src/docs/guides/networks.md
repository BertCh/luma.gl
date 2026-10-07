---
title: Networks, part one - routing, isochrones and accessibility
summary: Shortest paths, service areas, drive-time polygons and cost matrices on a street graph, and when to reach for which.
order: 11
---

## What this family does

A street network is a graph: intersections are nodes, street segments are directed edges, and every edge has a *cost* (metres, seconds, anything non-negative). Almost every question about movement reduces to **shortest-path costs over that graph**: how long from A to B, what can I reach within 15 minutes, which station is nearest, how many jobs can a person reach without a car.

The contributors in this family answer those questions on the GPU, over the whole Chicago drive network (about 30,000 intersections and 77,000 directed edges from OpenStreetMap), and they all share one data model: a **CSR** (compressed sparse row) adjacency made of `offsets`, `neighbors` and `weights`. The weights are an ordinary buffer. Closing a road, raising congestion or changing walking speed is a buffer write, not a rebuild.

Three scenes show the family in order: [From the Loop to O’Hare](#/story/routing), [Who is within four minutes of a fire station?](#/story/isochrones) and [How many jobs can you reach without a car?](#/story/job-accessibility). The sibling guide on clustering, map matching and centrality covers the rest of the chapter.

## The search engine: reachability

[`GPUNetworkReachability`](#/reference/GPUNetworkReachability) is the workhorse. From one or many sources it computes, for every node, the minimum cost and (optionally) the predecessor on a shortest path, isochrone band classes and a convergence flag. It is a frontier-based Bellman-Ford: only nodes whose cost just improved are relaxed, in a fixed number of unrolled rounds, and a round with an empty frontier dispatches nothing. Costs are the unique fixpoint of the float32 relaxations, so they are identical for every tuning.

Two compile-time numbers tune it: `maxIterations` (rounds) and `localIterations` (hops chained inside one round). A network whose longest shortest path has `h` hops needs about `ceil(h / (0.7 * localIterations)) + 4` rounds; Chicago converges in 5 to 8 rounds at 32. The per-frame `costLimit` stops the search early, and `activeIterations` caps the rounds. Read the **Solver** readout in the routing scene to see the rounds used.

Everything else builds on its output:

- [`GPUNetworkPathExtraction`](#/reference/GPUNetworkPathExtraction) walks the predecessor array back from up to many targets in parallel and packs the node and edge lists. One tree, many routes.
- [`GPUNetworkNeighborhood`](#/reference/GPUNetworkNeighborhood) is the hop-count neighbourhood (ego network): every node within *k* edges, regardless of length.
- [`GPUNetworkServiceAreas`](#/reference/GPUNetworkServiceAreas) searches from many facilities at once and labels every node with its nearest facility, breaking ties to the lowest row.
- [`GPUNetworkLineGraph`](#/reference/GPUNetworkLineGraph) rebuilds the graph so that a node is a directed edge and an arc is a turn. Turn costs, U-turn bans and explicit banned turns become ordinary edge weights, and every routing contributor runs on the result unchanged.

## From costs to shapes: isochrones

Costs on nodes are not yet a map. [`GPUNetworkIsochrones`](#/reference/GPUNetworkIsochrones) turns them into polygons in two ways. The **raster path** samples every edge, interpolates the cost along it, writes the minimum to a raster (with an optional walking buffer) and contours the raster into filled bands. The **cell path** outlines the H3 or Quadbin cells that hold reached nodes and assembles closed rings with shells and holes. Use the raster path for smooth drive-time bands, the cell path for a clean coverage polygon you can join demand against.

The recipe [`addDriveTimeCatchmentRecipe`](#/reference/addDriveTimeCatchmentRecipe) wires the whole pipeline: [`GPUNetworkSnapping`](#/reference/GPUNetworkSnapping) puts facilities and demand points on street edges, service areas label the network, isochrones draw it, and a group-statistics step counts the demand in each time band. Its counterpart [`addStraightLineCatchmentsRecipe`](#/reference/addStraightLineCatchmentsRecipe) does the same with distance (a raster Voronoi of [`GPUDistanceField`](#/reference/GPUDistanceField) plus zonal statistics), which is the right baseline to show how much a simple buffer misleads.

## Many-to-all: matrices and accessibility

When the question is "for every place, how much can I reach", run the searches from the *opportunities* instead and keep the results. [`GPUNetworkCostMatrix`](#/reference/GPUNetworkCostMatrix) computes one bounded search per row over a (usually reversed) CSR, batching rows into lanes of a lane-expanded graph; the matrix is bit-identical for every lane count. [`GPUNetworkAccessibility`](#/reference/GPUNetworkAccessibility) then scores each node from the retained matrix: cumulative opportunities within a threshold, gravity measures with exponential or power decay, and the two-step floating catchment area (2SFCA).

The split matters for performance. The matrix is the expensive part and is encoded only when the network or the opportunity set changes. Scoring is a few linear passes, so thresholds, decay and measure are a four-float parameter buffer that you can drag at frame rate. The job-accessibility scene shows both timings side by side.

## Which tool when

| Question | Contributor |
| --- | --- |
| How long from A to B, and by which streets? | `GPUNetworkReachability` + `GPUNetworkPathExtraction` |
| What lies within k blocks? | `GPUNetworkNeighborhood` |
| Which facility serves each place, and who is not served in time? | `GPUNetworkServiceAreas`, `addDriveTimeCatchmentRecipe` |
| Draw the 5, 10, 15 minute areas | `GPUNetworkIsochrones` |
| Route with turn penalties or bans | `GPUNetworkLineGraph` |
| Put points on the street first | `GPUNetworkSnapping` |
| How many jobs or beds can each place reach? | `GPUNetworkCostMatrix` + `GPUNetworkAccessibility` |

## Key options and pitfalls

- **Weights decide the answer.** A free-flow drive time ignores intersections, signals and traffic. The scenes add a per-intersection delay and a congestion multiplier; real analyses need real speed profiles. Negative or NaN weights are impassable, so closures are `-1`, not removed edges.
- **Costs are floored above zero.** Equal-cost plateaus (zero-weight edges) are handled, but they cost extra tie rounds; the scenes floor every weight at 0.1 s.
- **Direction matters.** Edges are directed. For "cost *to* a destination" or opportunity-side accessibility, search a reversed CSR. Walking on one-way streets needs the reverse edges added, which the job-accessibility scene does.
- **Snap before you search.** A point is not a node. Snapping gives two seed costs so the search starts mid-edge. Check the snap distance: points beyond the maximum snap to nothing and silently drop out.
- **Raster isochrones are planar and approximate.** They use straight chords between intersections and are accurate to about a pixel; the walking buffer is capped at 16 pixels, so zoom in to see its full radius. The cell path samples nodes only, so on sparse networks long edges can leave cells out.
- **Memory is rows times nodes.** A cost matrix of 256 rows over 51,000 nodes is 52 MiB. The row count and lane count are compile-time.
- **Two options do not exist as controls here.** Origin-row matrices (rows are origins, columns hold opportunities) are supported by `GPUNetworkAccessibility` but not shown, because 791 tract origins over 51,000 nodes would exceed the default storage-binding limit.

## Data and parity

Streets come from the `chicago-roads` dataset (OSM, ODbL), facilities from the City of Chicago and Overture, population from the CDC/ATSDR Social Vulnerability Index, jobs from LEHD LODES, and transit from the CTA GTFS. The reference tools are pgRouting (`pgr_dijkstra`, `pgr_drivingDistance`), OSMnx and networkx for routing and ego networks, QGIS and ArcGIS service-area polygons, `networkx.line_graph` and momepy for turn graphs, pandana and access for accessibility; the contributor documentation states where parity is bitwise and where it is only conceptual.

## Where to go next

Open the reference pages for [`GPUNetworkReachability`](#/reference/GPUNetworkReachability), [`GPUNetworkIsochrones`](#/reference/GPUNetworkIsochrones) and [`GPUNetworkAccessibility`](#/reference/GPUNetworkAccessibility), and the three stories above. Then try the clustering, matching and centrality stories, which reuse the same CSR.
