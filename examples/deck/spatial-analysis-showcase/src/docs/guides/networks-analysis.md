---
title: Networks, part two - clustering, matching and centrality
summary: Network K functions, HMM map matching, and node analytics with subgraph filters, and when to reach for which.
order: 12
---

## What this family does

Street networks are not a plane. Distances run along streets, events sit on streets, and a GPS fix is only meaningful once it is tied to one. The contributors in this guide answer three questions that routing and isochrones do not:

| Question | Tool | Story |
| --- | --- | --- |
| Are events clustered along the streets more than chance? | `GPUNetworkKFunction` | [Do Chicago crashes cluster along the streets?](#/story/crash-k-function) |
| Which street did this noisy GPS trace follow? | `GPUMapMatching`, `GPULineMerge` | [Which streets did these GPS traces follow?](#/story/map-matching) |
| Which streets matter, and what remains if you filter the network? | `GPUNetworkAnalyticsColumns`, `GPUNetworkSubgraphFilter`, `GPUNetworkStatistics` | [Which streets does Chicago lean on?](#/story/street-centrality) |

All three consume the same input: a **CSR** (compressed sparse row) adjacency. `offsets` has one row per node plus one, `neighbors` holds the destination of every directed edge slot, and `weights` holds a non-negative cost such as length in meters. A two-way street appears in both directions. The scenes build this layout once on the CPU from the `chicago-roads` dataset; after that, everything is GPU buffers and parameter writes.

## Network K function

The planar Ripley K function counts how many pairs of events lie within distance `d` and compares that with random points in the plane. For crashes, shops or incidents that can only occur on streets, that comparison is unfair: any street-bound data looks clustered against a uniform plane.

`GPUNetworkKFunction` fixes both halves. Distance is measured along the network by bounded shortest-path searches (one per event), and the yardstick is an envelope of random patterns placed on the *same streets* with probability proportional to street length. The result is `K(d) = 2 * pairs(d) * L / n^2`, where `L` is the network length and `n` the events, evaluated at `bandCount` distances up to `maxDistance`.

Key options:

- **Per-frame parameters:** `maxDistance`, `maxSnapDistance`, the simulation `seed`, the number of `activeSimulations`, and the event positions themselves. Nothing recompiles when they change.
- **Compile-time options:** `bandCount`, `simulationCount`, `rowsPerBlock` (scratch memory against number of passes), `spatialSort` and the snapping `candidateCapacity`.
- **Reading it:** observed K above the maximum of all simulated patterns at a distance means clustering there; below the minimum means dispersion. With 19 simulations the test is pointwise at about 5 percent, so a handful of triangles among many bands can be chance.

Pitfalls: a sample of a few hundred events is enough, and a larger one is paid for in shortest-path rows (events times patterns). Events farther than the snap distance from any street are dropped. The envelope is the plain minimum, mean and maximum, which differs from `spaghetti`, whose extremes are scaled.

## Map matching

`GPUMapMatching` implements the hidden Markov model of Newson and Krumm. Each fix gets candidate edges within a search radius from a GPU edge grid. The **emission** probability is Gaussian in the perpendicular distance (`sigma`), and the **transition** probability is exponential in the difference between the straight distance of two fixes and the route distance between their candidate edges (`beta`). Viterbi runs with one thread per track and restarts after a **break** when no transition is possible.

Things to know:

- The CSR **row is the edge id**, and it is directed, so a two-way street takes two candidate slots. The matcher sees an edge as a straight line between its nodes. The scene therefore splits every street polyline into one edge per segment (and long segments into pieces of at most 300 m, which the edge grid needs) so curved streets are matched against their true geometry.
- `sigma`, `beta`, `searchRadius`, `routeFactor` and `routeSlack` are per-frame parameters. `candidateCount`, `routeNodeBudget` and the grid `cellSize` are compile-time: a small node budget is faster but can overestimate a route in dense streets.
- Scoring needs ground truth. The scene uses simulated traces whose true edge is known for every fix, so it can color each matched segment right, right-street-wrong-direction or wrong.

`GPULineMerge` is a geometry helper that shares the scene: it joins line segments at endpoints shared by exactly two line ends into maximal chains, with exact float32 endpoint comparison and no tolerance. Use it to turn segment soup into streets before labelling or per-street statistics.

## Analytics columns, subgraph filter and statistics

`GPUNetworkAnalyticsColumns` publishes node-aligned columns from a CSR: degree, in-degree, **PageRank**, **k-core** number, weak **components** and label-propagation **communities**, each optionally normalized to the range 0 to 1 by a GPU min/max. The columns feed deck.gl styling directly. Without a reverse CSR the graph is treated as undirected and the forward CSR must be symmetric; passing `reverseOffsets` and `reverseNeighbors` makes it directed.

`GPUNetworkSubgraphFilter` turns half-open attribute ranges on vertex and edge columns (and optionally a time window) into a vertex mask and an edge mask: an edge is live only if it passes its ranges and both endpoints are live. Optional outputs include counts, compact live ids and an induced CSR other contributors can read. `GPUNetworkStatistics` consumes the masks and returns one packed summary: live counts, weak components, isolated vertices, maximum degrees, degree histograms (linear or log2 bins) and the modularity of a community labelling.

Why combine them: the three form a pipeline on one GPU timeline. Analytics write PageRank, the filter reads it as a vertex predicate, and the statistics describe what survives, all without a readback. Compile-time options are the algorithm settings (`damping`, iteration counts, `directed`, `dropIsolated`, `degreeBinning`); every range, the modularity resolution and the histogram bin width are parameters.

Pitfalls: PageRank on a street grid is nearly uniform because almost every intersection has degree three or four, so use it where the structure has hubs and read it as *structural* importance, not traffic. Label propagation is a bounded heuristic, not modularity optimization; the iteration count changes the result. Always check the convergence flags (components, core number, communities) before quoting a number.

## Which one to use

- Events on streets, "is this unusual": network K function.
- Raw GPS, "which roads": map matching, then aggregate over matched edges.
- A network and a question about its structure or a what-if filter: analytics columns, then filter, then statistics.
- A route, a service area or a travel-time catchment: see the routing and isochrone stories in the same chapter.

Related reference pages: `GPUNetworkSnapping` (the snapping step used inside the K function), `GPUNetworkReachability` (the shortest-path search it repeats) and `GPUNetworkCostMatrix`.
