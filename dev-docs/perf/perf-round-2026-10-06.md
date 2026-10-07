# Analysis contributors: perf round 2026-10-06 measurement plan

This round was theory first. Each contributor was audited for work, depth, dispatch count, atomic
contention and memory traffic, then the best known parallel algorithm was implemented where it was
principled and verifiable. Nothing was timed. Correctness is verified by the existing and new
specs (CPU and f64 oracles, Shapely, PyKrige, spopt and similar pinned references), not by
benchmarks. The expected wins below come from complexity or traffic arguments and are the claims
to confirm or reject. Paths are relative to `modules/experimental/`.

How to read the tables:

- "Expected win" is the theoretical change as stated in the stream handoffs. A factor means a
  traffic, dispatch or atomic count ratio, not a measured speedup.
- "A/B" names a prop or input condition that selects the old path. "none" means the change has no
  switch, so compare against the previous commit.
- **Risk** marks a change that can regress. The same items are collected in the checklist below.
- Many fused passes changed node IDs (for example `-sizes` and `-sums` in k-means, `-apply` in map
  coloring, `-level-N` in the extrema pyramid). Look for the new IDs when reading per-node timings.

## Cleanup after the streams

- One `sorted keys -> segment offsets` helper in `src/utils/sorted-segment-offsets.ts` replaces the
  copies in group geometry, geometry measures, spatial weights slot grouping and network noding.
- Dead per-row count atomics are gone from raster zonal statistics, raster join, flow aggregation,
  line density, line length per polygon and zonal statistics (sorted sums).
- Line-segmentize path prefix, validity `rings` and ring-assembly `ring-stats` use the 64-item block
  hybrid, so many small items keep one thread each.

## Regression checklist

Measure these first. Each one is a change where the win is conditional or a threshold was guessed.

| Contributor | What can regress | Check |
|---|---|---|
| `GPUKMeans` update | Update is O(n k) shared-memory reads, close to the assign cost near k = 256 | n = 1e6, k = 256; gate by k if slower (one line in `gpu-kmeans.ts`) |
| `GPUGeographicDistribution` | Chunked path is O(n x groups); the 64-group `auto` switch (`AUTO_CHUNKED_MAXIMUM_GROUPS`) is unmeasured | 8, 32, 64, 128 groups, `reduction: 'chunked'` against `'sorted'` |
| `GPUCellCover` slabs | Default on when `candidateCapacity * vertexCount >= 4e6` is a guess; the index adds about 12 dispatches | Sweep size around the threshold with `edgeSlabs` true and false |
| `GPUCellAggregation` | Tile scan adds 8 shared steps per 256 rows when cells are nearly unique | 5e6 points at quadbin resolution 20 |
| Fused kernels at the binding limit | Fusions fall back to the old kernels above `maxStorageBuffersPerShaderStage`; the fallback runs only in node specs | Confirm the limit-8 path on a real device (local Moran epilogues, line graph with bans, adjacency with all options, map-matching `routes`, vertex snap, cell-cover test kernel) |
| Workgroup-private histograms and tables | Thresholds (bins up to 1024, groups up to 256, slots up to 256, transition cells + 1 up to 2048, network-statistics bins up to 341, zonal `columns * 4 * Z` against workgroup storage) lower occupancy near the limit | Test just below and just above each threshold |
| `GPUSegregation` | `terms` keeps `2 + 2K + K^2` private floats per invocation (290 at K = 16) and may spill | K = 16, n >= 1e6 |
| Many tiny rings | Validity and ring-assembly `ring-stats` run as 64-ring blocks with a cooperative queue for large rings | 1e6 rings of 4 vertices, against one 1M-vertex ring |
| `GPUGeometryMeasures` | With more than 512 rows the cooperative kernel launches up to `rows / 513` workgroups that scan `ringOffsets` even if no feature is large | 1M rows of small features, `cooperativeRingRows: 0` against default |
| `GPUShapeDescriptors` convexity | `auto` picks monotone chain when `rows >= 256 * features`; threshold is a guess | Features of 100 to 1000 vertices, `convexityMethod: 'gift-wrapping'` against `'auto'` |
| `GPUPolygonTriangulation` | Z-order hash starts above 80 vertices; scratch grows from 32 to 56 B per vertex | 80 to 500 vertex polygons, `useZOrderHash: false` |
| `GPULineSimplification` | `finishSpanLimit` default 32 is a guess | Values 0, 8, 32, 128 on random-walk tracks |
| `GPUMinimumBounds` | Calipers start above 16 hull vertices | Hulls of 16, 17, 64, 256 vertices |
| `GPUSimilarLocations` rank | Sort path starts at 4096 rows; the quadratic path stays below | n = 2048, 4096, 8192 |
| Line segmentize, arcs, chunk | Emit dispatches `output.positions.length` invocations; oversized capacity costs idle lanes | Capacity 10x the output |
| `GPUTerrainHorizon` fused march | One thread now does sectors x steps; chunked by `maximumStepsPerDispatch` | Small radius and 64 sectors, default against `maximumStepsPerDispatch: 1` |
| `GPUMapMatching` | `routes` kernel is at 8 bindings; A* changes route lengths where Dijkstra overflowed its node budget | Gap length 10 to 60 s, `routeNodeBudget` 64 |
| `GPUMapColoring` | Async coloring depends on atomics for safety; the oracle spec is the only guard | Large grid and a random graph, compare `roundCount` |
| Fused zonal and stretch | Z or `binCount` just below the workgroup limit gives one workgroup per SM | Z = 1000 against 5000, `binCount` 1024 against 8192 |

## gpu-spatial-analysis

### Proximity, pair statistics and indexing

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| Pair histogram (variogram, Ripley K, correlogram), unordered | Forward half stencil over cell-ordered points | About 2x fewer candidate loads and tests; one dependent gather becomes three sequential loads | `GPUVariogram`, 1M points, `maximumDistance` about 3 cells | `gpu-variogram.spec.ts` | none |
| `GPURipley` border and isotropic | Register tally per focus; skip raw channel | Atomics per pair 3 to 2 (border), 3 to 1 (isotropic without `pairCounts`) | 1M points | `gpu-ripley.spec.ts` | none |
| `GPUPointPatternIndices` nearest neighbor | Clearance-based ring stop, cell-order threads | 9 to 25 fewer cells visited on dense data | 1M points at about 5 and about 50 per cell | `gpu-point-pattern-indices.spec.ts` | none |
| Ripley F reference | Reads sorted points | Removes one gather per candidate | Ripley F on 1M points | `gpu-ripley-distance-functions` specs | none |
| `GPUNeighborSearch` kNN | Skip rows and end cells beyond k-th best | Fewer cell visits from ring 2 on; more gain on clusters | k = 16 and 32, clustered | `neighbor-search-benchmark.spec.ts` | none |
| `GPUHilbertKeys` and segment BVH bounds | Two-level reduction, up to 256 workgroups | One workgroup (1 of N compute units) becomes the whole GPU | 1M to 10M points or segments, no `bounds` | hilbert and segment specs; explorer `spatialSort` runs | pass `bounds` |
| `GPUHilbertKeys` key | Branch-free parallel-prefix index | About 60 ops against 16 dependent iterations; likely memory bound, small win | 10M points | `gpu-hilbert-keys.spec.ts` | none |
| `GPUSegmentIntersection` self mode | Probe in leaf order; reject `right <= left` before table read | Coherent traversal; saves a 32 B gather for half the candidates | 1M vertices, shuffled short features | `gpu-segment-intersection.spec.ts` | `spatialSort` on and off |

### Weights, autocorrelation and permutation inference

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUEmergingHotSpots` Gi* | Per-slice neighbor sums once, then window sums in workgroup memory | n T k (W+1) to n T (k+W+1); about 8x fewer neighbor reads at W = 11, k = 8. **Risk:** 3 KB workgroup memory, assumes size 256 | Lattice 512 x 512, T = 24, W = 11, radius 3; weights n = 100k, k = 8 | `GPUEmergingHotSpots` specs | vary `temporalWindow` |
| `GPUEmergingHotSpots` Mann-Kendall, classify, moments | Series in private memory, fused classify, tree moments | Half the pair loops; one dispatch and one re-read less; moments serial per 1024 bins to tree | n = 1M cells, T = 64 to 256 | emerging hot spots weights-mode spec | none |
| `GPULocalPermutationTest` | Included-neighbor weights hoisted to private array | Per draw, 4 CSR gathers + 1 value gather become 1 value gather | n = 100k, k = 8, P = 999 | `gpu-local-permutation-test.spec.ts` | none |
| `GPUGlobalPermutationTest` finalize | 256-thread workgroup, tree sums | About 4M serial iterations to about 16K per thread at P = 2^20 | P = 2^20, small n | `gpu-global-permutation-test.spec.ts` | none |
| `GPULocalMoran`, `GPUHotSpotAnalysis` front end | 12 dispatches to 4; no term buffers | About 8n floats less traffic; 4 buffers of n removed | n >= 1M | local Moran and hot spot specs | none |
| Local Moran and hot-spot epilogues | Local I and classify fused into neighbor kernel | 1 to 2 dispatches and about 12n bytes less. **Risk:** falls back above the binding limit | n >= 1M, with and without FDR | same specs | request FDR to force the unfused path |
| `GPUContiguityWeights` | Narrow radix keys; binary-search row offsets; fused pair fill | 2 to 4x fewer radix passes; no clear, atomics or scan | 100k polygons, queen | contiguity specs in `gpu-spatial-weights.spec.ts` | none |
| Weights transpose, global statistics, summary | `slot-grouping.ts`: sort then lower bound per column | Replaces clear, atomic counts and scan (3 to 5 dispatches); no hub-column contention | n = 1M, nnz = 8M, plus a star | `gpu-spatial-weights-transpose.spec.ts` | none |
| `GPUSpatialWeightsSummary` | Thread-per-column sums; 7 reductions fused to 2 | Removes 256 x rows launched threads for k-element segments | n = 1M, k = 6 | spatial weights algebra spec | none |
| `GPUNeighborhoodSummary` categorical | Members decoded once into a private array | Global loads for modes and entropy O(k^2) to O(k). **Risk:** only rows with k up to `maximumNeighbors` | n = 100k, k = 16 to 64 | neighborhood-summary spec | rows above `maximumNeighbors` use the old path |

### Spatial statistics tests, rates, similarity

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUSpatialScanStatistic` evaluate | Window max fused: workgroup tree max, one `atomicMax` per workgroup | Removes (P+1) n x 4 B write and read (400 MB at P = 999, n = 1e5) and an n-deep serial loop | P = 999, zones >= 1e4 | scan spec (add a bigger scene) | none |
| `GPUKnoxTest`, `GPUMantelTest` | Slot-fastest threads, rows interleaved across blocks | Up to about 32x fewer CSR transactions (warp broadcast); skew averages out | n >= 1e5, >= 1e6 pairs, P = 999, clustered | Knox and Mantel specs; explorer space-time mode | none |
| `GPUMantelTest` prelude | Up to 4096 blocks; invariant moments once (5 words to 3 per slot) | Observed-moment depth m/256 to m/4096 | same, compare `pair-stats` | same | none |
| `GPUEmpiricalBayesRates` | Fused partials and totals, no 4n matrix | 11 dispatches to 5; 16 B per row each way removed | n >= 1M | `gpu-empirical-bayes-rates.spec.ts` | none |
| `GPUSegregation` | Fused terms and workgroup reduce | About 300 MB less traffic per scale at K = 5, n = 1e6; about 4 dispatches per scale. **Risk:** K = 16 spills | n >= 1M, K = 5 to 16, 1 and 5 scales | `gpu-segregation*.spec.ts` | none |
| `GPUTransitionMatrix` count | Workgroup-private histogram | About 1e6 contended atomics to about 6000 at K = 5 | rows x periods >= 1e6, K = 5 and K = 64 | `gpu-distribution-dynamics.spec.ts` | K = 64 (cells + 1 above 2048) |
| `GPUClassificationFit` stats | Workgroup per class | Depth n/K (three passes) to segment/256 + 3 log 256 | n >= 1e6, K = 7, one dominant class | `gpu-classification-fit.spec.ts` | none |
| `GPUSimilarLocations` columns | Workgroup-per-attribute tree reductions | Serial n per thread to n/256 | n >= 1e5, 8 attributes | `gpu-similar-locations.spec.ts` | none |
| `GPUSimilarLocations` rank | K radix sorts + 2 binary searches per row | O(n^2 K) to O(K n log n). **Risk:** threshold 4096; -0/+0 ties on the quadratic path | n = 1e4 to 1e5, rank mode | same | n below 4096 |

### Regression, interpolation, regionalization

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUKriging`, `GPUInverseDistanceWeighting` (k mode) | Ring kNN walk; sort pass removed; symmetric matrix fill | cells x N to cells x k at large radius; k^2/2 variogram evals | 1M samples, 512 x 512, `searchRadius` Infinity, k = 8 to 16; a small-radius control | `gpu-kriging.spec.ts`, `gpu-inverse-distance-weighting.spec.ts`; explorer interpolation | IDW `neighborCount` 0 against above 0 |
| `GPUGeographicallyWeightedRegression` adaptive | Ring kNN plus grid weights | k-th neighbor search O(n) to O(k), twice per location | 20k to 200k rows, adaptive bisquare, ladder 8 to 32 | `geographically-weighted-regression-adaptive-grid.spec.ts` | fewer than 1024 rows (no grid) |
| GWR fixed bisquare | Threads in grid-cell order | Memory coherence only | 200k to 1M rows, random order | `geographically-weighted-regression-grid.spec.ts` | none |
| GWR nonstationarity test | Shared grid for refits | O(P n^2 p^2) to O(P n k p^2) for bisquare | 5k to 50k rows, P = 99 to 999 | `geographically-weighted-regression-nonstationarity.spec.ts` | `indexGridSize` |
| `GPUSpatialWeightsMinimumSpanningTree` | Fused reset, break-pairs, last jump, label reset; workgroup standardize | About 3 dispatches per round (about 60 at 1M rows); standardize depth O(n) to O(n/256 + 8) | 1M-row lattice or contiguity graph | `gpu-spatial-regionalization.spec.ts`, `-scale.spec.ts` | `standardize` on and off |
| `GPUSkaterRegions` | Euler-tour interval tests; incremental statistics | Per step O(E D) to O(E); 100k path tree about 1e10 to 1e5 steps | 100k path-like tree and 100k lattice, 50 to 200 regions | `-scale.spec.ts` | none |
| `GPURegionPartitionEvaluation` | Stable sort by label; thread per label | O(L n) to O(n) + radix sort. **Risk:** few huge regions stay serial per label | 1M rows, default `labelCapacity`, 5 and 50k labels | `-scale.spec.ts` | none |

### Clustering and aggregation

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUKMeans` update | Sort-free chunked reduction | About 15 to 3 dispatches per iteration; 12 B per row read; no scratch growth per iteration. **Risk:** k = 256 | n = 1e6, k = 8, 64, 256; 20 to 64 iterations | `gpu-clustering-extras.spec.ts`; explorer k-means | none |
| `GPUKMeans` seeding | Workgroup `atomicMin`, one atomic per workgroup | n atomics on one word to n/64, times k rounds | n = 1e6, k = 16 | `kmeans++` test | none |
| `GPUGeographicDistribution` | Fused chunked group sums; 2-dispatch Weiszfeld step | One group: 1 workgroup to up to 1024; about 6 passes per iteration to 2 dispatches. **Risk:** 64-group `auto` switch | n = 1e6 to 4e6; 1, 8 to 64, and 1000 groups | `gpu-geographic-distribution.spec.ts`; explorer directional distribution | `reduction: 'sorted'` |
| `GPUCellAggregation` | Tile segmented scan before atomics | Hot cell: n same-address atomics to n/256 x 3 to 4. **Risk:** fine cells | 5e6 points, quadbin resolution 4 to 8, then 20; roll-up pyramid | `gpu-cell-aggregation.spec.ts`; explorer cell aggregation | none |
| `GPUMapColoring` | Fused select and apply, asynchronous Jones-Plassmann | One kernel plus gate per round instead of two; rounds only drop | 1e5 polygons, grid | `gpu-map-coloring.spec.ts` | none |

### Geometry construction and line operations

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUGroupConvexHull`, `GPUGroupGeometry` | Offsets by lower bound over sorted keys | n serial atomics + scan + fill become G log n reads | 5M points in 1 to 10 labels; 1e5 mostly empty groups | `gpu-group-convex-hull.spec.ts`, `gpu-group-geometry.spec.ts` | none |
| `GPUSegmentRingAssembly` `ring-stats` | Block-of-64 hybrid; cooperative large rings; ring bounds | One huge ring: n to n/64 depth. **Risk:** many tiny rings | One 1M-vertex ring; 1e6 tiny rings | `gpu-segment-ring-assembly.spec.ts` | none |
| `GPUSegmentRingAssembly` `ring-shells` | Bounding-box reject before vertex walk | 1000 holes over 1000 distant shells: about 1e6 x V to about 1e6 box tests | 5k to 50k holes over 5k to 50k shells | same, `runPolygons` | none |
| `GPUShapeDescriptors` convexity | Monotone chain via `GPUGroupConvexHull` | O(V h) per feature to O(n log n); 2500-vertex feature about 2.5e6 serial steps. **Risk:** `auto` threshold | Features of 2k to 100k vertices; one huge among 100k small | `gpu-shape-descriptors.spec.ts` | `convexityMethod: 'gift-wrapping'` |
| `GPULineSplit` ordering | Heapsort above 16 events per segment | O(k^2) to O(k log k); k = 100k about 1e10 to 1.7e6 | One segment crossing 10k to 100k lines | `gpu-line-split.spec.ts` | none |
| `GPULineSegmentize`, arcs, `GPULineChunk` emit | One invocation per output vertex, windowed owner search | Imbalance up to 1024:1 to 1:1; `substring` of 1M-row path from one lane to parallel. **Risk:** capacity | Mixed 1-piece and 1000-piece segments; arcs; 1M-row `substring` | `line-segmentize` specs | none |
| Path prefix (segmentize, chunk, measures) | One 64-lane workgroup per path, tiled scan | n dependent steps to n/64 tiles x about 12 barriers; about 20x on 1M rows | One 1M-row path, spherical | `line-segmentize` specs | none |
| `GPULineDensity` count | Return before the cell walk | O(pieces) to O(1) per segment | Long segments over a fine grid | `line-density` specs | none |
| `GPUGeometryMeasures` | Workgroup per feature with a ring over 512 rows | 1M-vertex ring: 1M serial steps to 16k; geodesic is seconds to tens of ms. **Risk:** see checklist | One 1M-vertex ring among 100k small features; 200k-vertex geodesic polygon | `gpu-geometry-measures-cooperative.spec.ts` | `cooperativeRingRows: 0` |
| `GPULineSimplification` Douglas-Peucker | Depth-first finish of small open intervals | Rounds fall from split-tree depth to depth above leaf intervals; 300-row spiral about 270 to about 240 rounds; each skipped round saves 4 full dispatches. **Risk:** default 32 | 1M-row GPS random-walk tracks; 100k-vertex noisy curve; spirals | `gpu-line-simplification.spec.ts` | `finishSpanLimit: 0` |
| `GPUTrackSimilarity` Hausdorff, `maxDistance` | Early break with hinted start; lanes over the longer track | O(n m) toward O(n + m) on smooth tracks; 5-point against 100k-vertex from 5 lanes to 64 | Two smooth 5k to 50k tracks; random clouds; 5 against 100k points | `gpu-track-similarity.spec.ts` | none |
| `GPULinearReferencing` spherical | Bounding-cap pruning, nearest-cap seed | Exact test (2 atan2, sqrt, 12 loads) on a handful of segments instead of S; still O(P S) cheap tests | 100k points against 100k segments, clustered | `gpu-linear-referencing-spherical.spec.ts` | none |
| `GPUPolygonTriangulation` | Z-order hashed ear test above 80 vertices | O(n^2) to about O(n log n) ear tests per ring. **Risk:** footprint 56 against 32 B per vertex | One polygon with 5k to 50k spiky vertices | `gpu-polygon-triangulation.spec.ts` | `useZOrderHash: false` |
| `GPUMinimumBounds` | Rotating calipers above 16 hull vertices | h = 256: about 65k to about 1.5k distance evaluations per group | Groups with 128 to 256 hull vertices | `gpu-minimum-bounds.spec.ts` | none (hulls up to 16 use the old path) |
| `GPUOutlineGeometry` | Disc rim vertex computed once | sin and cos per vertex 32 to 16; ALU-bound only | Large point or line set | outline-geometry specs | none |
| `GPULineClipByPolygon` | Inverse rank table for sub-piece end points | Sub-piece work O(k^2) to O(k) per segment | Few long lines over a dense mask, k >= 1000 crossings | line-clip specs | none |
| `GPUGeometryPredicates` unique points | Feature sort uses log2(F+1) key bits | Radix passes of the third sort 8 to 2 at 10k features; total sort work down 1/5 to 1/4 | Many vertices, moderate features | geometry-predicates spec | none |
| `GPUGeometryValidity` | Block hybrid rings; workgroup hole containment with bbox prefilter | Huge ring n to about n/64; hole x shell O(shell) to O(shell/64). **Risk:** many tiny rings | One 100k-vertex shell with thousands of holes; 1e6 tiny rings | `gpu-geometry-validity.spec.ts` | none |
| `GPULineMerge` | Pointer-jumping list ranking | 1M-line serpentine: 1M serial steps x 3 to about 20 rounds x 2; O(L log L) work | 100k shuffled lines, one chain and many short chains | `gpu-line-merge.spec.ts` | none |
| `GPUCoverageSimplification` | Parallel span scoring; skip final measuring round | Serial span walks to O(n) work, O(1) depth; 1 of (rounds + 1) BVH passes saved | Coastline-like coverage, default `topologyRounds`, with and without `topologyStats` | coverage-simplification specs | request `topologyStats` |
| `GPUGridGenerator` intersects | Bbox cull per extent edge; one point-in-extent test | Per cell E bbox tests; PIP scans 6 to 1 | 1M cells, 100k-vertex extent | grid-generators spec | none |
| `GPURandomPointsOnLine` | Cumulative arc length + binary search | O(V) to O(log V) per point | Few lines of 100k vertices, many points | dot-density specs | none |
| `sorted-segment-sums` (`src/utils`) | Offsets by lower bound; callers' counts ignored | Removes scan and total fixup; no hot-segment atomics (callers still build counts) | Zonal statistics, flow aggregation with one hot zone | zonal and importer specs | none |

### Zones, cell sets, joins, trajectories

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUCellCover` | Per-feature y-slab edge index | O(C E) to O(E + C k); 100k-vertex country at 1e6 candidates about 1e11 to about 1e7 edge tests. **Risk:** 4e6 threshold | One 50k to 500k vertex polygon, Quadbin `center`; also `full`, `intersects`, `core` | `gpu-cell-cover.spec.ts`; explorer cell cover | `edgeSlabs: false`; `edgeSlabEntryCapacity` |
| Areal interpolation, coverage dissolve | Pair sort with `keyBits` per half | Passes 16 to 6 (1000 x 1000 zones); dissolve edge sort at 1M vertices 16 to 10 | >= 1e6 cells, >= 1000 zones; >= 1e6 vertices | areal and `gpu-coverage-topology.spec.ts` | none |
| Areal interpolation zone counts | Fold 8 cells per run before the atomic | Same-address contention down to 8x | Raster with few huge zones | areal specs scaled up | none |
| `GPUCellSetOutline` | Per-row bitmask and counts | Scan input and flags 10x smaller (H3), 4x (quadbin) | H3 set >= 1e6 cells | `cell-set-outline.spec.ts`; explorer outline | none |
| `GPUCellTableCompare` | Reuse lower bound from matches pass | 1 of 3 binary searches per before row, about 25% of search traffic | Two tables >= 1e6 rows | `gpu-cell-table-compare.spec.ts` | none |
| `GPUTrajectoryEncounters` scan | Half-shell traversal | Cells 9 to 5; distance tests about half | 20k tracks x 256 buckets, distance near cell size | `gpu-trajectory-encounters.spec.ts` | none |
| `GPUTrajectoryEncounters` sort | One packed-key sort when track + partner + bucket bits <= 32 | Saves 2 sorts' fixed cost, 2 key kernels, about 3 gathers per hit | `hitCapacity` >= 1M, bits <= 32 | same | above 32 bits (old chain) |
| Spatial join relate engine | Y-slab pruning of edge-edge scans | nx ny to about nx ny (span + 1)/32; up to about 10x at 200 to 1000 vertices | Polygon-polygon `relate`, `intersects`, `touches` | `gpu-spatial-relate-scale.spec.ts`, `spatial-relate-benchmark.ts` | features under 48 edges |
| `GPUMinimumClearance` | Seed bound from incident segments; test children at push | Visits only boxes within the shortest incident edge, not descent from infinity | Dense overlapping polygons or lines | minimum-clearance specs | none |
| `GPUVertexSnap` global mode | BVH over references, bound = tolerance squared | O(V R) to about O(V log R). **Risk:** 8 bindings | R >= 10k references, V = 100k, small tolerance | `gpu-vertex-snap.spec.ts` | R up to 64, or pairwise mode |

## gpu-network

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUMapMatching` routes | A* with Euclidean heuristic | Settled nodes: disc to corridor (several times fewer); cost per pop is O(S), so work falls about as the square; fewer budget overflows. **Risk:** 8 bindings, results change on overflow | Road network, 10 to 60 s gaps, `routeNodeBudget` 64 | `gpu-map-matching.spec.ts` | none |
| `GPUNetworkCostMatrix`, `GPUNetworkReachability` | Lanes share one CSR (`laneCount`) | Removes 2 dispatches and `laneCount (N + 2E)` words written and read (about 40 MB at 434 lanes, 2304 nodes) | 48 x 48 grid (2304 rows, 434 lanes); road 224 with lane 32 | `gpu-network-accessibility-bench.spec.ts` | `laneCount: 1` |
| `GPUNetworkKFunction` count | Workgroup band histogram | Global atomics per block from pairs to at most bandCount per 64 pairs | `eventCount` >= 4k, `bandCount` 32 | `gpu-network-k-function.spec.ts` | none |
| `GPUNetworkIsochrones` splat | Load before `atomicMin` and `atomicMax` | Skips most read-modify-writes where windows overlap | Wide `radius` raster | `gpu-network-isochrones.spec.ts`; explorer isochrones | none |
| `GPUNetworkSubgraphFilter` | Slot-parallel endpoints; one shared slot scan | Hub thread of 1e10 serial reads spread over 1e5 invocations; CSR build shares one m-scan | Star or power-law, hub 1e4 to 1e5, `pairUndirectedSlots`, `inducedCSR` + `liveEdgeSlots`; m >= 1e6 | `network-subgraph-filter` spec | none |
| `GPUNetworkLineGraph` | Weights fused into fill; skip atan2 on forward turns | Turn-cost evaluations per arc 3 to 2; per-arc binary search removed. **Risk:** bans need 9 bindings | Road grid 1e5+ edges, default and `uTurnCost: 0` | `network-line-graph` spec | none |
| `GPUNetworkNoding` | CSR from endpoint-sort order; third sort removed | One 2E radix sort becomes one scan and two linear kernels | 1e5+ pieces, tolerance 0 and above 0 | `network-noding` spec | none |
| `GPUNetworkAdjacencyMatrix` | Fused bins; row window cull; workgroup maxima | 2 slot walks to 1; zoomed window O(m) to O(window rows + slots); maxima R^2 atomics to R^2/256. **Risk:** 9 bindings with all options | n = 1M, m = 5M, 1% window, resolution 256 and 1024, `mirrorSlots` | `gpu-adjacency-matrix.spec.ts` | none |
| `GPUEdgeBundling` box | Workgroup min and max | 4E contended atomics to 4E/256 | E = 1e6 | `edge-bundling` spec; explorer bundling | none |
| `GPUFlowAggregation` | Slot sort keys `bits(Z^2)`; count-mode sort 2 `bits(rowCount)` | Radix passes 8 to about 5 per sort (about 37% fewer) | C = 2^22, Z = 1000, rowCount 1e6 | `flow-aggregation` specs | none |
| `GPUNetworkCoarsening` | Fold intra weights per row; superedge runs of 16 slots | About 16x less contention on heavy superedges | 1e6 slots, 3 to 100 groups, weighted | `network-coarsening` spec | none |
| `GPUNetworkStatistics` | Workgroup-private counters, maxima, histograms | About 10 global atomics per vertex on about 40 words to about 40 per workgroup (about 2500x fewer at 1M). **Risk:** thresholds | n = 1M, m = 1e7, power-law, 32 bins | `network-statistics` spec | bins above 341 |

## gpu-raster

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUCostDistance` relaxation | Edge costs computed once per tile visit | Up to 32 x 8 `getD8Distance` recomputations per thread removed, with `cos`, `cosh`, `sqrt` and `settings` loads | 2048^2+ friction, `'geographic'` and `'web-mercator'`; serpentine and spiral | `gpu-cost-distance.spec.ts`; explorer cost distance | none |
| `GPURasterPatchMetrics`, sieve | Run aggregation over 16-column segments | Atomics n pixels to about n / min(16, run length); dominant patch no longer serializes | 4096^2 labels, one patch over 50%; clumps as control | `gpu-raster-patches.spec.ts` | none |
| `GPUIsolines` stitching | One 16 B record per node; convergence-gated rounds | Rounds log2(capacity)+1 to about log2(longest chain)+2 (21 to about 16); 1 cache line instead of 4 per gather. **Risk:** 16 B stride binding limits capacity near 8M | Capacity 1M with short chains; chains of 1e3 to 1e5 segments | `gpu-isolines.spec.ts` | none |
| `GPURasterZonalStatistics` | Fused counts, min, max in workgroup tables | Zone reads 4n to n, mask traffic 6n words to 0, atomics n per column to at most Z per tile. **Risk:** large Z | 2048^2 to 4096^2, Z = 10 to 1000; Z = 5000 | explorer zonal statistics | Z above the workgroup limit |
| `GPURasterExtremaPyramid` | 4 levels per dispatch from a 16 x 16 tile | Dispatches L-1 to ceil((L-1)/4); 11 levels to 4 | 4096^2, `firstBlockSize` 1 to 4 | viewshed, line-of-sight, horizon modes | none |
| `GPURasterStretch` | Register fold + tree reduce; private histogram | Same-address atomics n to n/4096; histogram n to at most binCount per tile. **Risk:** large `binCount` | 4096^2 float, smooth and skewed; `binCount` 1024 and 8192 | `gpu-raster-stretch.spec.ts`; stretch mode | `binCount` above the limit |
| `GPUWeightedOverlay` range | Tree-reduced `scoreRange` keys | 2 atomics per cell to 2 per workgroup | 4096^2 x layers | overlay mode | omit `scoreRange` |
| `GPURasterReclassify` counts | Workgroup-private class counts | Atomics per cell to per workgroup | 4096^2, 5 to 20 classes | reclassify mode | none |
| `GPURasterCellStatistics` | One sweep per kernel; private frequency copy | Moments 3 to 2 sweeps; frequencies T^2 to T global reads per cell | 32 to 64 layers, majority, minority, variety | cell-statistics mode | none |
| `GPUStreamlines` pruning | GPU-gated rounds | Rounds after convergence cost about 0; win scales with `roundCount` minus rounds needed | Default 32 rounds, 4k to 60k seeds, large grid | streamlines mode | none |

## gpu-terrain

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUTerrainFlow` fill | Seed only tiles with a boundary cell | One full-grid tile pass of about 5 to 15 removed | 2048^2 DEM, `fillDepressions` | `GPUTerrainFlow` specs | none |
| `GPUTerrainFlow` flats | Seed tiles with finite start values (3 relaxations) | 3 full-grid iteration-0 passes to work proportional to flat area plus a one-tile halo | 2048^2 noisy DEM, `resolveFlats`, epsilon 0; one huge plateau as control (no gain) | `gpu-terrain-flow-flats.spec.ts` | none |
| `GPUTerrainSummits` | 3 x 3 rejection before the disc scan | Only 3 x 3 local maxima (about 10 to 15%) pay O(r^2) | 4096^2 rough terrain, radius 16 | `gpu-terrain-summits.spec.ts` | none |
| `GPUTerrainPeakSnap` | Workgroup per candidate, argmax tree | Per-candidate latency r^2 to r^2/256 + 8 steps; about 50x on the critical path at radius 64 | 100 to 1000 candidates, `maximumRadiusPixels` 64 | `gpu-terrain-peak-snap.spec.ts` | none |
| `GPUProfilePeaks` NMS | GPU-gated rounds | Saves `nmsRounds` minus rounds needed full-size dispatches (default 16, up to 1024) | Dense profiles, 1e6 samples | `gpu-profile-peaks.spec.ts` | none |
| Viewshed, line of sight, point horizon | NaN test on 4 corners instead of 4 validity loads | Loads per sample 8 to 4 (about 2x traffic) | 1024^2 with `maxDistance`; point horizon 2048^2 | `shared-pyramid-benchmark.spec.ts`, viewshed and point-horizon specs | none |
| `GPUGeomorphons` | One value load per sample | Inner loop 3 loads to 1 | 1024^2, `searchRadius` 50 to 100 | `gpu-geomorphons.spec.ts` | none |
| `GPUTerrainHorizon` march | One kernel over all sectors | Dispatches D to about 1; sum traffic 2 to 4 buffers x D x (read + write) to 1 write. **Risk:** per-thread work | 1024^2 to 2048^2, 16 sectors, radius 16 to 64, all outputs | `terrain-illumination-benchmark.spec.ts`; sky-view and openness modes | `maximumStepsPerDispatch: 1` |
| `GPUTerrainVectorRuggedness` | Separable window from radius 3 | O(r^2) to O(r) per pixel; radius 8 is 289 to 34 gathers (about 8x) | `radius` 5 to 10 on 2048^2 | `terrain-ruggedness` GPU specs (no benchmark; A/B radius 2 against 6) | radius 1 to 2 (direct loop) |
| `GPUTerrainSpikeRepair` | Strip of 16 pixels per invocation, one atomic per label run | Hot-address atomics down to 16x; valid count n atomics to about one per component | 2048^2 Terrarium tile, one large component | `gpu-terrain-spike-repair.spec.ts` | none |

## gpu-dataframe

| Contributor | Change | Expected win (theory) | Measure | Spec or mode | A/B |
|---|---|---|---|---|---|
| `GPUCompositeScore` validate and merge | Min, max, count inside `tile-sums`; merged in tile order | 2d + 1 atomics per row on d + 1 hot addresses to 0; one dispatch less | 1M rows, d = 8 to 32 | `gpu-composite-score.spec.ts`, `validate` and `tile-sums` nodes | none |
| `GPUCompositeScore` PCA covariance | Upper triangle only | d^2 to d(d+1)/2 pairs per row (d = 12: 144 to 78) | 1M rows, d = 24, `enablePrincipalComponent` | same, `tile-covariance` node | none |
| `GPUInequality` pooled Gini | Tile totals, scan over tiles, tile area, sum | Depth 2n to about n/256 + 512 + n/256 (1M rows: 2,000,000 to about 12,000 serial steps) | 1M to 4M rows, `globalSummary` | `gpu-inequality.spec.ts`, `global-gini-*` nodes | none |
| `GPUColumnProfile` histogram and category counts | Workgroup-private table | One atomic per row to at most one per bin per workgroup. **Risk:** bins up to 1024 only | 4M rows, 64 to 1024 bins, skewed | `gpu-column-profile.spec.ts` | bins above 1024 |
| `GPUGroupStatistics` accumulate | Workgroup-private count, sum, min, max | rows/groups atomics per address to 1 per workgroup | 4M rows, 4 to 64 groups | `gpu-group-statistics.spec.ts`, `accumulate` node | capacity above 256 |
| `GPUTemporalReduction` extremes | Same privatization | Same, for cells x buckets up to 256 | 4M rows, 8 x 16 slots | `gpu-temporal-reduction.spec.ts`, `extremes` node | slots above 256 |

## Numeric changes

Everything not listed is bit-identical to before or exact integer arithmetic. All changes below
stay deterministic on one device. None changed a pinned reference value (Shapely, pyproj, turf,
spopt, mgwr, spreg). The first group are changes of summation order only.

Summation order (f32 rounding level, within existing tolerances):

- k-means centers and geographic distribution sums: new fixed chunked order.
- Mantel partial sums and the `shift` mean (4096 blocks); within the existing 2e-4 oracle tolerance.
- Empirical Bayes rates, segregation, classification fit, similar-locations z-score and reference:
  tree order, about 1 ulp.
- Emerging hot spots moments, global permutation mean, SD and z (counts, min and max exact), weights
  summary S0, S1 and S2 (column sums now follow slot order, closer to the oracle).
- Emerging hot spots Gi* z: sums of per-slice sums; oracle tolerance 2e-3 absolute + 1e-3 relative
  holds.
- Pooled Gini: tile order, within the spec tolerance 1e-3, typically more accurate than serial f32.
- MST `standardize: true`: lane-strided tree sums. Edge sets with exact cost ties on standardized
  data can differ; `standardize: false` is unchanged.
- SKATER: subtree statistics by subtraction after the first walk; counts exact, sums carry a few more
  roundings in deep regions.
- Region partition evaluation: `withinSsd` and `totalSsd` merge 256-element tiles in tile order.
- GWR adaptive bisquare and nonstationarity with a grid: cell-order sums, about 1e-5 in
  coefficients (unchanged below 1024 rows).
- Ring areas of rings above 64 vertices, and path measures of paths above 64 rows: spec tolerance
  1e-3 relative holds.
- Features with a ring above 512 rows in `GPUGeometryMeasures`: planar at f32 level, geographic
  bounds differ by about 3e-5 degrees from unwrap order.
- Fused horizon march: separately compiled kernel, so fused multiply-add choices differ by 1 to 2
  ulp.

Results that change beyond order:

- `GPUGeographicDistribution` median: Vardi-Zhang step (see bugs below). Medians of small groups move
  to the optimum; the oracle is not the optimum.
- `GPUMapMatching` route lengths: exact shortest paths; they differ from the old output only where
  the old Dijkstra overflowed the node budget, or on float-level ties.
- `GPUShapeDescriptors` convexity under `monotone-chain` (and `auto` for large features): hull snapped to
  a 2^29 lattice, so convexity can differ from gift wrapping by about 2e-9 of the extent; spec
  tolerance 1e-3.
- `GPUMinimumBounds` rectangle and longest line on hulls above 16 vertices: ulp-level differences,
  and f32 near-ties in the diameter can pick another pair.
- `GPULineSimplification` `finishSpanLimit`: importance bits identical for any value; only
  `roundCount` and what a `maximumRounds` cap leaves undecided change.
- `GPUGeometryValidity`: a vertex far outside the shell bounds is now always outside instead of
  possibly uncertain.
- Spatial relate: when more than 24 breaks overflow, visiting order differs, which can change a later
  uncertain-orientation flag (non-finite input only).

Spec changes:

- `gpu-terrain-horizon.spec.ts`: two exact-equality assertions between separately compiled kernels are
  now tolerance checks (sky-view 1e-6, openness 1e-4, horizon 1e-4 degrees).
- Line simplification specs that assert the oracle's `roundCount` now pass `finishSpanLimit: 0`; the
  same scenes also run at the default and must give identical importance bits.
- `gpu-geographic-distribution.spec.ts`: the 200 and 300 group objective checks pass at 1e-5 relative
  on both reductions after the Weiszfeld fix.
- Node specs track the new node IDs. No other tolerance was loosened.

## Bugs found and fixed

- Weiszfeld median (`GPUGeographicDistribution`, both reductions). Plain Weiszfeld drops a row that
  lies within 1e-6 of the iterate, so for groups of a few dozen rows, where the median is often a data
  point, the step lost that row's pull and jumped away from the optimum (objective 3101.750 at 24
  iterations, 3107.28 at 48, oscillating). The f64 oracle and the f32 GPU hit the jump on different
  iterations, so they disagreed by up to +2.3e-3 in objective. Fixed with the Vardi-Zhang step: the
  coincident weight is carried as a fourth summed quantity and the step is scaled by
  `1 - min(1, eta / R)`. No extra passes on the chunked path; one extra array and sum node on the
  sorted path. It also showed that 24 plain steps are under-converged by up to 3.7e-3 against the
  true optimum on elongated clusters.
- WGSL reserved words. `active` broke the fused extrema-pyramid kernel and was caught by another
  stream's GPU run mid-round; `attribute` was avoided in similar-locations (renamed `column`). The
  existing roadmap list of reserved words that fail silently now includes both.
- Shader compiler folded `select(value, 0.0, value == 0.0)` (merging -0 and +0) on the test GPU. The
  rank path in similar-locations uses an integer bit test instead. The quadratic path below 4096 rows
  still uses a float compare; check -0/+0 ties on other GPUs.
- Extrema-pyramid pass: a shadowed variable, fixed in the same stream before the final run.
- Polygon triangulation mid-edit bug: the hashed ear test ran on an unindexed list. Fixed before the
  round closed; the spec passes 5 of 5.
- Open, not fixed: `GPULineChunk` substring mode with a start measure past the path length disagrees
  with the oracle by two vertices. The `piece-range` kernel is untouched by this round and no spec
  covers it. Tracked in `dev-docs/roadmaps/gpugraph-roadmap.md`.

## Not parallel-friendly

- DBSCAN: lock-free union-find already; dense cells skew, use `denseBoxShortcut`.
- Boruvka MST: fixed ceil(log2 n) rounds without readback; read a completion flag every few rounds, or
  use indirect dispatch over a compacted edge list.
- SKATER greedy cuts: step k depends on step k-1; O(1)-depth dispatches in a chain.
- Viterbi forward and backward (map matching): sequential in time, already parallel over tracks; for
  few long tracks, chunked max-plus matrix scan with an augmented semiring for the restart rule.
- Scan statistic secondary clusters: greedy by definition, at most 64 clusters.
- Mann-Kendall and Sen slope: pairwise over T up to 64 or 256; the private quadratic loop wins.
- Priority-flood fill: global priority queue; tile-local flood plus a spillover graph merge.
- D8 accumulation and Strahler order on very long stems: chain compression or tree contraction
  change float order; topological peeling needs cross-address ordering that WGSL lacks.
- Distance-field row envelope: serial stack per row; parallel banding algorithm or row chunks.
- Tile relaxation inner loop: Jacobi is barrier bound; in-place chaotic relaxation is a data race.
- LIC and streamline tracing: sequential along a trajectory; parallel over pixels and seeds already.
- Visvalingam-Whyatt: global heap order; independent-set rounds with compaction every few rounds.
- Frechet: anti-diagonal wavefront is the best known exact parallel form.
- Weiszfeld and pycnophylactic smoothing: fixed iteration counts are the API.
- Exact medoid: O(g^2) per group; sample, or tile the cost kernel (must keep summation order).
- Label point (polylabel): sequential priority queue; the grid-and-refine schedule is the GPU shape.
- GWR with Gaussian kernels: unbounded support, O(n^2); truncating at about 5 bandwidths changes
  results.
- OLS, 2SLS and GM solves: k x k with k up to about 10; latency only, lower `tileRowCount` below
  about 250k rows.
- BH-FDR: needs global ranks; the full sort is the exact route.
- Global permutation pairs: Feistel per neighbor each permutation; materializing permuted arrays
  needs an indirect-dispatch loop over permutation batches.
- Cell-topology grid path, geometry-edit cleanup flags, shared-paths chains: sequential along a path
  or greedy chain; pointer jumping for shared paths, parallel compare for tolerance 0 cleanup.
- Trajectory zone walk, dwell and span: sequential per track; opt-in parity by grouped rank if tracks
  are few and long.
- Line density emit: Amanatides-Woo is a sequential merge of crossings; closed form would break bitwise
  lengths.
- Dot density rejection sampling: needs triangulation and a per-polygon CDF, which changes positions.
- Network path extraction and statistics hub rows: parent walks and one-row loops; pointer jumping and
  CSR-vector only for extreme skew.
- Delta-stepping for network reachability: extra dependent dispatches per bucket; only if a probe
  shows more than 3x work inflation.
- Group moments, sorted zonal sums: fixed reduction order is the reproducibility contract.
- Time-window classify, calendar buckets: per-row output; nothing to reduce.
- Particle advection, point sampling: already embarrassingly parallel.
