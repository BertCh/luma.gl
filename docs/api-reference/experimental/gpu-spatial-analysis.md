import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Spatial Analysis

<ExperimentalDocsTabs active="gpu-spatial-analysis" />

:::caution Experimental
`@luma.gl/experimental/gpu-spatial-analysis` is an experimental entry point. APIs may change between
releases without a deprecation period.
:::

The `@luma.gl/experimental/gpu-spatial-analysis` entry point is the analysis layer above luma.gl's
GPU spatial primitives. It does not reimplement them. It composes the
[geospatial kernels](/docs/api-reference/experimental/geospatial) (projection, distances,
point-in-polygon, `GPUGridIndex`, `GPUPointSpatialQuery`) and the GPU Core primitives (sort, scan,
compaction, hash and grid indexes, `GPUBVH`, segmented reductions) into higher-level workflows:
spatial weights and their algebra, statistics computed on those weights, predicate and nearest
joins, geometry validity, density, discrete global grid cells, line and trajectory geometry,
regression, change of support, distribution dynamics, accessibility, and prebuilt recipes. New WGSL
is written only where no primitive fits.

Reach for the geospatial kernels when you need one primitive (a distance, a containment test, one
indexed query per frame). Reach for this entry point when the question is about every row at once,
or about statistics over the neighborhood structure.

## Overview

Each contributor is an algorithm or workflow that declares
resources and command nodes into a caller-owned `GPUCommandGraph` through `getCommandNodes(graph)`
(`GPUCommandNodeProducer`), and is added with `graph.add(contributor)`. They are composed from GPU
Core primitives (`GPUVisibilityWorkflow`, `GPUBVH`, `GPUGridBinning`, `GPUHistogram`, `GPUReduction`,
`GPUAncestorProjection`, and others) and the raster operators; new WGSL is written only where no
primitive fits.

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, GPUPointDensity} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'map'});
const viewport = new GPUParameterBuffer(device, {id: 'viewport', format: 'float32', length: 4});

// `positions`, `values`, `extent`, and `histogram` are graph views created on `graph`.
graph.add(new GPUPointDensity({
  positions, bounds: viewport.importToGraph(graph), gridSize: [256, 256],
  output: {values, extent, histogram}
}));
const compiled = graph.compile(); // once

// every frame: rewrite parameters, encode the same compiled graph
viewport.write(new Float32Array([minX, minY, maxX, maxY]));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

The sections below group the contributors by task.

| Section | Contributors |
| --- | --- |
| Spatial weights | `GPUNeighborSearch`, `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsTransform`, `GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsSummary`, `GPUSpatialWeightsTranspose`, `GPUSpatialLag`, `GPUMapColoring` |
| Spatial statistics | `GPUGlobalSpatialStatistics`, `GPUHotSpotAnalysis`, `GPULocalMoran`, `GPULocalPermutationTest`, `GPUGlobalPermutationTest`, `GPUNeighborhoodSummary`, `GPUEmpiricalBayesRates`, `GPUSpatialEmpiricalBayesRates`, `GPUSpatialClustering`, `GPUKMeans`, `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, `GPURegionPartitionEvaluation`, `GPUGroupGeometry`, `GPUGroupConvexHull`, `GPUEmergingHotSpots`, `GPUGeographicDistribution`, `GPURegionStatistics`, `GPUZonalStatistics`, `GPUFocalStatistics`, `GPUSimilarLocations` |
| Distribution dynamics and space-time tests | `GPUClassAssignment`, `GPUClassificationFit`, `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov`, `GPUKnoxTest`, `GPUMantelTest`, `GPUSpatialScanStatistic` |
| Accessibility and change of support | `GPUCatchmentAccessibility`, `GPUHuffTradeAreas`, `GPUArealInterpolation`, `GPUPycnophylactic`, `GPUSegregation` |
| Spatial joins | `GPUSpatialPredicateJoin`, `GPUSpatialJoinPrepared`, `GPUSpatialJoinCandidates`, `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin`, `GPUNearestFeatureWeights`, `GPUBufferSelection` |
| Geometry validity and intersections | `GPUSegmentIntersection`, `GPUGeometryValidity` |
| Density & interpolation | `GPUPointDensity`, `GPUInverseDistanceWeighting`, `GPUKriging`, `GPUDotDensity`, `GPURandomPointsInPolygon` |
| Cells (DGGS) | `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid`, `GPUCellLevelSelection`, `GPUPointToCell`, `GPUCellGeometry`, `GPUCellTopology`, `GPUCellCompaction`, `GPUCellCover`, `GPUCellSetOutline`, `GPUSegmentRingAssembly`, `GPUCellTableCompare` |
| Lines & trajectories | `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`, `GPULineSplit`, `GPULineMerge`, `GPUGeometryMeasures`, `GPUGeodesicPairs`, `GPUGeodesicDestination`, `GPULinearReferencing`, `GPULineLocate`, `GPULineSimplification`, `GPUCoverageSimplification`, `GPUTrajectoryMetrics`, `GPUTrajectoryPlayhead`, `GPUTrajectoryResample`, `GPUZoneEvents`, `GPUTrajectoryEncounters`, `addClockEncounters`, `GPUTrackSimilarity` |
| Geometry utilities | `GPUOutlineGeometry`, `GPULabelPoint`, `GPUShapeDescriptors`, `GPULineDensity`, `GPULineLengthPerPolygon`, `GPUGridGenerator`, `GPUShapeGenerator`, `GPUHilbertKeys`, `GPURectangleClip` |
| Regression | `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialErrorGM`, `GPUGeographicallyWeightedRegression`, `GPUGeographicallyWeightedRegressionNonstationarityTest` |
| Recipes | `addHotSpotAnalysisRecipe` and eleven more prebuilt chains |
| Cross-reference | turf, PostGIS, GeoPandas, PySAL, QGIS and ArcGIS tools mapped to contributors |

Contributors for networks, rasters and dataframe columns live in their own entries:
[GPU Network](/docs/api-reference/experimental/gpu-network),
[GPU Raster](/docs/api-reference/experimental/gpu-raster/operations-regions-contours) and
[GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe-analysis).

### Conventions

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

## Spatial weights

A `GPUSpatialWeights` is the sparse `W` that spatial statistics are built on, in CSR form: row `i`
lists the neighbors `j` of focus row `i` and the weight `w_ij` of each. Producers write it on the
GPU, transforms rewrite it, and the statistics below read it, so one statistic works with any
neighborhood definition. Applications can also upload weights built elsewhere.

Every producer guarantees: `offsets` holds `rows + 1` non-decreasing entries starting at 0; row `i`
occupies slots `[offsets[i], offsets[i + 1])`; neighbor IDs are strictly ascending within a row; a
self join never lists a row as its own neighbor (`w_ii = 0`); weights are finite and non-negative.
Capacity overflow clamps offsets to `neighbors.length` and sets an `overflow` flag on the GPU.

| Producer | Neighborhood | PySAL equivalent |
| --- | --- | --- |
| `GPUNeighborSearch` | k nearest neighbors or distance band between points | `KNN`, `DistanceBand`, `Kernel` |
| `GPUContiguityWeights` | polygons sharing a vertex (queen) or an edge (rook) | `Queen`, `Rook` |
| `GPULatticeWeights` | regular grid cells | `lat2W` |
| `GPUSpatialPredicateJoin` with `weights` | any predicate join (intersects, within, dwithin, ...) | - |
| `GPUSpatialWeightsAlgebra` | union, intersection, difference, higher order, self weights, subgraph, block | `w_union`, `higher_order`, `w_subset`, `block_weights` |

### `GPUContiguityWeights`

Polygon contiguity as binary weights. `criterion: 'queen'` joins polygons that share at least one
vertex; `'rook'` joins polygons that share a ring edge (both endpoints, either direction). Polygons
use the flat layout of `GPUPointInPolygonJoin` (`positions`, `ringOffsets`, `polygonOffsets`), with
one output row per polygon; hole rings count. Flatten multipolygons to one row per part.

- `snapTolerance` (default 0, exact f32 equality) snaps vertices with `floor(x / tolerance + 0.5)`
  per axis. This is a grid snap, not a distance test: two vertices closer than the tolerance can
  fall in different cells.
- Exact 64-bit vertex keys, stable radix sorts and run detection; no hashing, so no collisions, and
  the output is deterministic and symmetric.
- A point shared by `m` polygons emits `m^2` candidate pairs, bounded by `pairCapacity` (default
  four times the neighbor capacity); exceeding it also sets `overflow`.

```ts
graph.add(new GPUContiguityWeights({
  criterion: 'rook', positions, ringOffsets, polygonOffsets,
  weights: {offsets, neighbors, weights}, overflow
}));
```

### `GPULatticeWeights`

Weights over a `width x height` grid with row-major IDs `y * width + x`. `criterion: 'rook'` uses
Manhattan distance `<= radius`, `'queen'` Chebyshev distance `<= radius` (`radius` defaults to 1, at
most 32). A zero in the optional `mask` removes the cell as a row and as a neighbor. With `cellSize`
it also writes Euclidean `distances`, so kernel transforms apply.

### `GPUSpatialWeightsTransform`

Rewrites weights in place or into a separate `output`. Chain several instances, for example a kernel
followed by row standardization.

- `'row'`: `w_ij / sum_j w_ij`; rows with a zero or non-finite sum become 0.
- `'binary'`: `w_ij > 0 ? 1 : 0`, keeping every slot.
- `'kernel'`: `K(d_ij / h_i)` from `weights.distances`, with `kernel` one of `gaussian`
  (`exp(-z^2 / 2) / sqrt(2 pi)`), `triangular` (`max(1 - z, 0)`), `epanechnikov` (`3/4 max(1 - z^2, 0)`,
  PySAL quadratic), `bisquare` (`15/16 max(1 - z^2, 0)^2`, PySAL quartic) or `uniform` (`1/2` for
  `z <= 1`). `bandwidth` is a fixed number or `'adaptive'` (default): the row's largest distance, which
  is the k-th neighbor distance of a kNN row.
- `'double'`: `w_ij / S0`, with `S0` the sum of all weights, so the weights sum to 1 (PySAL `'D'`).
  `doubleSum: 'rows'` scales by `n / S0` instead, so the weights sum to `n`.
- `'variance'`: `s_ij = w_ij / sqrt(sum_j w_ij^2)`, then `w'_ij = s_ij n / sum s` over the whole
  matrix, with `n` the row count (PySAL `'V'`).
- `'symmetrize'`: `(w_ij + w_ji) / 2`, treating a missing reverse slot as 0. It keeps the sparsity
  pattern, so it is only symmetric for symmetric patterns (contiguity, lattice, distance band), and
  needs a separate `output`. For a directed pattern such as kNN, take the union of the weights and
  their `GPUSpatialWeightsTranspose` with `GPUSpatialWeightsAlgebra`.

`'double'` and `'variance'` compute a per-row partial, reduce it with `GPUReduction`, and apply it in
a second pass, in place or into `output`. A zero or non-finite total gives zero weights. The
operation, kernel and bandwidth are compile-time options.

### `GPUSpatialLag`

`lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`). An optional `mask` skips masked neighbors and writes
0 for masked rows (square weights only); `normalize: true` divides by the included weight sum, which
gives the row-standardized lag without a transformed copy of the weights. Sums run in slot order,
so the result is deterministic. A weights matrix with a self term (`GPUSpatialWeightsAlgebra`
`'selfWeight'`) includes it in the sum.

Weights need not be square. With `sourceCount` (the number of neighbor-ID rows of `values`, default
the weights row count) the rows of `weights` are targets and the neighbor IDs index source rows;
neighbors at or above `sourceCount` are skipped. `columnCount` (default 1) lags several columns in one
pass: `values` is `sourceCount * columnCount` row-major, `output` is `rows * columnCount`, and
`normalize` shares one weight sum per row. This is how the area shares of `GPUArealInterpolation`
transfer values between zone systems.

```ts
graph.add(new GPUContiguityWeights({criterion: 'queen', positions, ringOffsets, polygonOffsets, weights, overflow}));
graph.add(new GPUSpatialWeightsTransform({operation: 'row', weights}));
graph.add(new GPUSpatialLag({weights, values, output: lag}));
```

### `GPUNeighborSearch`

`GPUNeighborSearch` finds exact k nearest neighbors (`mode: 'knn'`, `k` up to 32) or every point
within a distance band (`mode: 'radius'`) between planar points. A self join never lists a row as its
own neighbor; pass `queryPositions` for a cross join. It writes a `GPUSpatialWeights` CSR:
`offsets` (`queryRows + 1`), `neighbors` (the slot capacity), `weights`, and optional `distances`.

- Rows list neighbor IDs in strictly ascending order with distances and weights aligned, so
  consumers can binary-search `w_ji`.
- kNN keeps the `k` smallest `(d^2, id)` pairs, so ties go to the lowest ID. An optional
  `radius` bounds the search. Fewer valid targets give a shorter row.
- Weights: `binary`, `inverseDistance` (`max(d, distanceFloor)^-power`), or `kernel` (gaussian,
  triangular, epanechnikov, bisquare, uniform). The bandwidth is `radius` in radius mode and the
  row's k-th neighbor distance in kNN mode (PySAL's adaptive bandwidth). `rowStandardize` divides
  each row by its sum. Non-finite weights are written as 0.
- Capacity: offsets are clamped to `neighbors.length`, so readers always see consistent, possibly
  truncated rows. `overflow` is set on the GPU, and `totalNeighbors` and `neighborCounts` stay
  unclamped.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'knn', format: 'float32', length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  values: getGPUNeighborSearchParameterValues({bounds, rowStandardize: true})
});
graph.add(new GPUNeighborSearch({
  mode: 'knn', k: 8, gridSize: [256, 256], positions,
  parameters: parameters.importToGraph(graph),
  weights: {offsets, neighbors, weights, distances}, overflow
}));
```

Targets and queries count only when their mask is nonzero, their coordinates are finite, and they
lie inside the per-frame `bounds`. A self join applies `mask` to the query rows too. Bounds,
radius, the weight function and row standardization are per-frame parameters, so changing them
never rebuilds the graph. Targets are bucketed with `GPUGridIndex`, whose `boundsBuffer` carries the
per-frame lattice (cells at least `radius` wide) without recompiling. kNN
searches expanding cell rings with a private sorted top-k list and stops once the k-th distance is
strictly inside the visited box, with a slack for f32 rounding. Radius mode counts, scans, emits, and
insertion-sorts each row by ID. No float atomics are used, so outputs are bitwise reproducible.

### `GPUSpatialWeightsAlgebra`

Builds a new `GPUSpatialWeights` from existing ones with a capacity-bounded `output` (rows + 1
offsets; it must not alias an input). `operation` is one of:

- `'union'`, `'intersection'`, `'difference'`, `'symmetricDifference'` with `left` and `right` (same
  row count) and an optional `weightRule`: `'left'` (default), `'right'`, `'sum'`, `'min'`, `'max'`,
  `'product'` or `'binary'`. The rule applies where both operands list a link, and one-sided links keep
  their weight. `'binary'` gives 1 everywhere, which reproduces libpysal 4.15 `Graph` set operations.
  Distances come from `left` where it lists the link, else from `right`.
- `'higherOrder'` with `weights`, `order` (1 to 32), `cumulative` (default false) and `workCapacity`
  (intermediate capacity, default the output capacity). Writes the neighbors exactly `order` steps away
  by shortest path, excluding the diagonal, with weight 1; `cumulative` matches `lower_order=True`.
  Cost grows with the squared frontier degree per row, so it suits sparse contiguity and kNN weights.
- `'selfWeight'` with `weights` and `selfWeight` (a number at least 0, or a per-row `float32` view).
  Inserts or replaces `w_ii` with distance 0. It relaxes the `w_ii = 0` invariant. `GPUSpatialLag`,
  `GPUSpatialWeightsTransform`, `GPUSpatialWeightsAlgebra` and `GPUSpatialWeightsSummary` accept it;
  `GPUGlobalSpatialStatistics`, the permutation tests and the local autocorrelation kernels ignore the
  self slot.
- `'subgraph'` with `weights` and a per-row `mask`. Rows keep their IDs: masked rows become empty and
  slots that point at masked rows are dropped.
- `'block'` with `groupIds` and `groupCount`. Row `i` lists every other row of the same group with
  weight 1; IDs at or above `groupCount` have no group. The output holds the sum of squared group
  sizes minus the row count.

`overflow` (one `uint32`) is 1 when the capacity was too small, and the optional `totalNeighbors` holds
the slots needed. On overflow the output is still a valid CSR: the leading rows are complete, one row is
a prefix and the rest are empty. Output rows are sorted and unique with finite non-negative weights.

```ts
graph.add(new GPUSpatialWeightsAlgebra({
  operation: 'union', left: contiguity, right: knn, weightRule: 'binary', output, overflow
}));
```

### `GPUSpatialWeightsSummary`

Summary statistics of a weights matrix. `statistics` (`float32`, at least 3) receives `S0`,
`S1 = 1/2 sum (w_ij + w_ji)^2` and `S2 = sum_i (row_i + col_i)^2`, as in libpysal. `counts` (`uint32`,
at least 5, see `GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT`) receives the used slots, the asymmetric slots,
the isolates, and the minimum and maximum cardinality. The optional `cardinality` is the per-row neighbor
count and `symmetryTolerance` (default 0) decides when two weights differ. Sums run in a fixed order.
Isolates are empty rows; connected components are not computed. A pair of unequal weights counts as two
asymmetric slots, where libpysal `asymmetry()` returns pairs.

### `GPUSpatialWeightsTranspose`

The CSR transpose `W'` of any `GPUSpatialWeights`, square or rectangular (libpysal `Graph` with focal and
neighbor swapped). `columnCount` (default: the row count) sets the number of rows of the result for
cross weights whose neighbor IDs index target rows; slots with a neighbor ID at or above it are dropped,
and columns that nothing reaches give empty rows. `output` is a caller-owned CSR with `columnCount + 1`
offsets and at least the input capacity in `neighbors` and `weights` (and `distances` when the input has
them); slots past `offsets[columnCount]` are zeroed. Slots are keyed by column, stable-sorted, counted, scanned and gathered, so every output
row is ascending and the result is bitwise reproducible with no atomics. The optional `asymmetricSlots`
(one `uint32`, square weights only, with `symmetryTolerance`, default 0) counts the positions `(i, j)` of
the union of both patterns whose weights differ or that are present in one only, so a one-way link counts
twice and 0 means `W = W'`. Reference: libpysal 4.15 (no deviations; the check is exact by default).

```ts
graph.add(new GPUSpatialWeightsTranspose({weights, output: transposed, asymmetricSlots}));
```

### `GPUMapColoring`

Greedy parallel graph coloring (Jones-Plassmann with seeded hashed priorities, lowest ID on ties) over a
symmetric `GPUSpatialWeights` such as contiguity, for choropleth maps where neighbors need distinct
colors (QGIS topological coloring). `colors` holds one `uint32` per row, `GPU_MAP_COLORING_UNCOLORED`
for rows left unfinished. Optional outputs are `colorCount`, `conflictCount` (pairs sharing a color,
0 for a proper coloring), `converged` and `roundCount`. `seed` selects the priorities and
`maximumRounds` (default 64, at most 1024) bounds the rounds. The result equals sequential greedy
coloring in priority order, so it uses at most the maximum degree plus one colors, not the minimum
number. Different seeds give different valid colorings.

```ts
graph.add(new GPUContiguityWeights({criterion: 'rook', positions, ringOffsets, polygonOffsets, weights, overflow}));
graph.add(new GPUMapColoring({weights, colors, conflictCount, converged}));
```

## Spatial statistics


### `GPUGlobalSpatialStatistics`

Global spatial autocorrelation from a `GPUSpatialWeights` CSR and a value column, each statistic with
its analytic expectation, variance, z-score and two-sided normal p-value. Pick the statistics with
`statistics` (compile-time). The `results` layout (`GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`) is fixed,
and blocks that were not requested hold NaN.

- `moran`: Moran's I with esda's `VI_norm` and `VI_rand` (sample kurtosis).
- `geary`: Geary's C with `VC_norm` and `VC_rand`.
- `getisOrdG`: Getis-Ord General G (values should be non-negative). Its randomization variance
  equals esda's `VG`. It is evaluated in a centered form so that f32 does not lose it to the
  cancellation in `E[G^2] - E[G]^2`.
- `bivariateMoran`: `(n / S0) sum w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`, which equals esda's
  `Moran_BV` for row-standardized weights. The exact variance under randomization of `y` is
  `E = 0`. esda itself offers only permutation inference.
- `joinCount`: BB, BW and WW joins of a binary column (nonzero is black) over binarized weights.
  Counts are exact integers, also available as ordered-pair `joinCounts`. Cliff-Ord randomization
  moments are given for BB and BW.

```ts
graph.add(new GPUGlobalSpatialStatistics({
  weights, values, statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'], results, joinCounts
}));
// results[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization]
```

Weights are used as given, so apply row standardization in `GPUNeighborSearch` if wanted. A row is
included when the optional mask is nonzero and every provided value is finite. Excluded rows are
neither foci nor neighbors, and `n`, `S0`, `S1` and `S2` cover included pairs only. The summary block
reports `n`, `S0`, `S1`, `S2`, the mean, the variance, the black count and the island count.
`w_ji` comes from a binary search of row `j`. Column sums come from a deterministic transpose: a
stable sort of slots by neighbor. Per-row partials are reduced by fixed-shape workgroup trees, and
join counts by u32 atomics. Results are bitwise reproducible, and values, weights and the mask
change per frame without a rebuild. Inference is NaN when `n < 4`, `S0 = 0`, or the values are
constant.

### `GPUHotSpotAnalysis` and `GPULocalMoran`

Local spatial autocorrelation on caller-supplied square `weights`, used as given: build them with any
producer above and apply binary, row or kernel weighting first. Rows count when their `mask` is
nonzero and their value is finite; excluded rows are neither foci nor neighbors. `n`, the mean `X`
and the population standard deviation `S` are taken over included rows, or from per-frame
`fixedMoments`. Sums run in slot order with fixed-order reductions, so outputs are bitwise
reproducible.

- `GPUHotSpotAnalysis` writes the Getis-Ord Gi* z-score per row for arbitrary weights:
  `z_i = sum_j w_ij (x_j - X) / (S sqrt((n S1_i - W_i^2) / (n - 1)))`, with `W_i = sum_j w_ij` and
  `S1_i = sum_j w_ij^2`. The focal row enters through the compile-time `selfWeight` (default 1,
  classic Gi*; 0 gives Gi), never through the matrix. The z-score is unchanged when every value is
  shifted by a constant. It equals esda's `G_Local(star=True)` analytic `Zs` for positive data
  (esda normalizes by the sum of values, which flips the sign when the mean is negative). Optional
  outputs: the two-sided p-value, the neighbor count and a `sint32` confidence bin in `-3..3`
  (99/95/90% hot or cold), with Benjamini-Hochberg correction when `falseDiscoveryRate: true`.
- `GPULocalMoran` writes the z-score of local Moran's I with the analytic moments of conditional
  randomization, the null esda's permutation test samples: `E[L_i] = W_i mu`,
  `Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)` over the other `N = n - 1` rows. Optional outputs
  are `localI` (`(n - 1) c_i L_i / sum c^2`, esda's `Is`), the spatial lag `L_i = sum_j w_ij c_j`,
  the p-value, the neighbor count and the esda quadrant code (1 HH, 2 LH, 3 LL, 4 HL; 0 when not
  significant at the per-frame `significanceLevel`, optionally with BH-FDR).
  `quadrantGating` (`'analytic'` default or `'none'`, compile-time) chooses what `quadrants` holds. With
  `'analytic'` it is nonzero only where the analytic p-value (or BH-FDR) is significant. With `'none'` it
  is the ungated quadrant from the signs of the centered value and the spatial lag, for every included
  row with a nonzero centered value and lag, and 0 otherwise, so another test can gate it, for example the
  `significant` mask of `GPULocalPermutationTest` (esda's `q` with `p_sim`). `zScores` and `pValues` do
  not change. `'none'` cannot be combined with `falseDiscoveryRate` (the constructor throws).

Checked against esda 2.10 on 200 points with 6-nearest-neighbor weights: `localI` matches `Is` to
3e-7, the z-score tracks esda's 99,999-permutation `z_sim` (correlation 0.999997, largest gap 0.03),
and Gi* matches `Zs` to 1e-7.

```ts
graph.add(new GPUNeighborSearch({
  mode: 'radius', gridSize: [256, 256], positions, parameters: searchParameters,
  weights, overflow
}));
graph.add(new GPUHotSpotAnalysis({weights, values, parameters, zScores, bins, pValues}));
graph.add(new GPULocalMoran({weights, values, parameters, zScores, quadrants}));
searchParameters.write(getGPUNeighborSearchParameterValues({bounds, radius: 500}));
```

### `GPULocalPermutationTest` and `GPUGlobalPermutationTest`

Permutation inference (pseudo p-values) over a `GPUSpatialWeights` CSR, matching PySAL esda's
`p_sim`. Both contributors read a uint32 parameter view written with `getGPUPermutationParameterValues`:
`{seed, permutations, significanceLevel}`. Changing the seed, the permutation count (up to the
compile-time `maximumPermutations`), the values or the weights never rebuilds the graph.

- `GPULocalPermutationTest` does conditional randomization for `localMoran`, `localG` or
  `localGStar` (a self weight of 1). For each tested row and permutation it draws an ordered sample
  of `k_i` distinct other included rows (a partial Fisher-Yates over their compacted positions) and
  assigns them to the row's weight slots. It counts simulated values `>= observed`, folds the count
  as esda does, and writes `exceedances`, `pseudoPValues = (count + 1) / (P + 1)`, an optional
  `observed` statistic, and a `significant` mask (`p <= level`, or Benjamini-Hochberg with
  `falseDiscoveryRate: true`). Islands and rows with more than `maximumNeighbors` (64 at most)
  neighbors are not tested; the latter set `overflow`.
- `GPUGlobalPermutationTest` handles `moran`, `geary`, `getisOrdG` and `bivariateMoran` (`y`
  permuted). Each permutation relabels the included values through a keyed Feistel bijection, so no
  sort runs per permutation and nothing of size `P x n` is stored. `results`
  (`GPU_GLOBAL_PERMUTATION_RESULT`) holds the observed statistic, `p_sim`, the folded count, the
  simulated mean and standard deviation, `z_sim`, `p_z_sim`, and the minimum and maximum. Optional
  outputs are the `referenceDistribution` and its `histogram` (GeoDa's permutation plot).

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'permutations', format: 'uint32', length: GPU_PERMUTATION_PARAMETER_LENGTH,
  values: getGPUPermutationParameterValues({seed: 1, permutations: 99})
});
graph.add(new GPULocalPermutationTest({
  weights, values, statistic: 'localMoran', parameters: parameters.importToGraph(graph),
  maximumPermutations: 999, exceedances, pseudoPValues, significant, falseDiscoveryRate: true, overflow
}));
// Preview with 99 permutations, then refine on idle:
parameters.write(getGPUPermutationParameterValues({seed: 1, permutations: 999}));
```

`alternative` (compile-time; a change recompiles) selects the tail, with esda's definitions. With `R`
permutations, `g` is the number of simulated values `>= observed` and `l` the number `<= observed`:

| `alternative` | pseudo p-value |
| --- | --- |
| `'directed'` (default) | `(min(g, R - g) + 1) / (R + 1)`, the previous behavior |
| `'greater'` | `(g + 1) / (R + 1)` |
| `'lesser'` (alias `'less'`) | `(l + 1) / (R + 1)` |
| `'two-sided'` | `min(2 (min(g, l) + 1) / (R + 1), 1)` |
| `'folded'` | `(f + 1) / (R + 1)`, `f` = number of simulated values with `abs(sim - mean) >= abs(observed - mean)` |

`exceedances` holds the numerator count of the chosen tail, and `significant` and the Benjamini-Hochberg
path use the same p-value. `'two-sided'` doubles the p-value against `'directed'` and cannot go below
`2 / (R + 1)`, so raise `permutations` when testing a small level.

`'folded'` matches esda 2.10 exactly: `mean` is the mean of the `R` simulated statistics (the observed
value is excluded, as esda computes it from the reference distribution), ties count, and the test is
symmetric about that mean. The kernels run two sweeps over the same deterministic stream (a sum, then
the count); the other tails keep one. `'two-sided'` is not esda's definition: esda 2.10 places the
observed value at a percentile of the simulated statistics and counts simulations beyond the
interpolated percentiles, which would need every simulated value of a row stored and sorted. The two
can differ by a few counts (a 20-permutation fixture gives 3/21 in esda and 2/21 here). Use `'folded'`
when an undirected test must agree with esda.

On skewed data `'folded'` can flag far fewer rows than `'two-sided'`. With local Moran's I over a
right-skewed variable (hot spots on a low background), a low-value row's conditional distribution is
left-skewed, so its observed value sits in the short tail and the long tail holds simulations just as
far from the mean. Low-low rows that `'greater'` flags then mostly fail `'folded'`, while high-high rows
pass. esda 2.10 behaves the same: on a 1,500-point hot-spot field it flags 211 rows with `'folded'` and
551 with `'two-sided'`.

Random numbers come from Philox 4x32-10 counter streams, keyed by the 64-bit seed with the counter
`(block, row, permutation, tag)`. Results are therefore a pure function of the seed and inputs,
independent of the dispatch shape. The observed and simulated statistics run through the same f32
code, positive constant factors are dropped from the comparisons, and exceedances are integers. The
significance and BH tests compare correctly rounded f32 products, never quotients. Global pair sums
use fixed-shape workgroup trees. Every output is bitwise reproducible for a seed.

### `GPUSpatialClustering`

DBSCAN density clustering of planar points. Bounds, `epsilon`, and `minimumPoints` are per-frame
parameters (`getGPUSpatialClusteringParameterValues`). `gridSize` is only a compile-time maximum
lattice. Each encoding derives an active lattice whose cells are at least `epsilon` wide, so the
3×3 neighborhood search is always exact and `epsilon` changes never recompile. Clusters are joined
by a lock-free union-find that hooks larger roots under smaller ones. Labels are therefore
canonical: compact cluster IDs follow the smallest core row of each cluster, and a border point
joins the adjacent cluster with the smallest root. Outputs are per-point labels, roots, and core
flags, the cluster count, a compact list of cluster representatives, and cluster sizes and
centroids.

```ts
graph.add(new GPUSpatialClustering({
  positions, parameters: parameters.importToGraph(graph), gridSize: [128, 128],
  labels, clusterCount, clusters: {ids: clusterRoots, count, overflow}, clusterSizes, clusterCentroids
}));
parameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: 25, minimumPoints: 8}));
```

With `minimumPoints = 1` every point is a core point, so clusters are the connected components of the
graph that joins points within `epsilon`, with no noise. That is PostGIS `ST_ClusterWithin(geom,
distance)` (single-linkage clustering by distance, with the same transitive chaining); larger values
are `ST_ClusterDBSCAN(geom, eps, minpoints)`. `GPUSpatialClustering` writes labels only: pass them to
[`GPUGroupGeometry`](#gpugroupgeometry) and [`GPUGroupConvexHull`](#gpugroupconvexhull) for cluster
bounds, centers and outlines.

`denseBoxShortcut: true` enables the FDBSCAN dense-box shortcut. Points are binned into boxes of side
`0.7 * epsilon`, and a box with at least `minimumPoints` points makes all of its points core without a
neighbor scan. Labels, roots and core flags are identical to the plain algorithm. The union kernel
still scans neighbors, so the measured gain on 200,000 clustered points is only 3 to 9 percent, and
the option is off by default. Points whose box index exceeds 2^16 skip the shortcut.

`drawInstanceCount` (a one-row `uint32`, needs `clusters`) receives the clamped cluster count, so a layer
can draw one instance per cluster: import an indirect draw record's instance count with
`graph.importGPUData(id, drawCommands.getInstanceCountData(0))` and pass the view.

`sumOrder` defaults to `'sorted'` (stable sort of members by cluster, then a fixed-order segmented sum):
because labels are canonical, `clusterCentroids` is bitwise reproducible across encodings on one device,
and cluster sizes are exact either way. `'atomic'` skips the sorts but uses float atomics, so last bits vary
between runs. Behavior change: last-bit differences versus the previous atomic default.

### `GPUKMeans`

Deterministic k-means over planar points (scikit-learn `KMeans`, PostGIS `ST_ClusterKMeans`). Props:
`positions`, `k` (1 to 256), `iterations` (1 to 64, a maximum), `tolerance`, `initialization`
(`'first-valid'` by default, or `'kmeans++'` with `seed`) and the outputs `labels` (cluster or noise),
`centers` (`k` rows), optional `sizes` and `squaredDistances` (the explain column; its sum is the
inertia) and `convergence`. Each point goes to the nearest center with lowest-ID ties, means are fixed-order sorted
segmented sums, and an empty center keeps its position. Results are bitwise reproducible per seed.
Each iteration adds a sort, so memory grows with `iterations * points`. `'kmeans++'` uses `log`,
which WGSL does not pin down across devices.

```ts
graph.add(new GPUKMeans({positions, k: 8, iterations: 20, initialization: 'kmeans++', seed: 3,
  labels, centers, sizes}));
```

`iterations` is a cap: a cap that is too low reports `converged` 0 (first-valid initialization needs 20 to 30
iterations on taxi data at `k = 16`), and the largest shift usually falls straight to exactly 0, a true
Lloyd fixed point, so a tolerance relative to the data extent is enough. `tolerance` (compile-time, default 0) is a convergence threshold on the largest center shift. After an
iteration whose shift is at most `tolerance`, the later assign and update kernels return early, so the
result equals running every iteration when `tolerance` is 0 (an exact fixed point). The optional
`convergence` output is two `uint32` rows, `[iterationsUsed, converged]`. There is no GPU loop exit yet,
so the per-iteration sorts still run: the saving is kernel work, not dispatches.

### `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, and `GPURegionPartitionEvaluation`

Spatially constrained regionalization over a contiguity graph (PySAL spopt `SpanningForest` and SKATER,
Assuncao et al. 2006).

- `GPUSpatialWeightsMinimumSpanningTree`: Boruvka minimum spanning forest over a symmetric
  `GPUSpatialWeights`. The edge cost is the squared Euclidean distance between (optionally standardized)
  attribute rows. A total order of `(cost, lower-row CSR slot)` makes the forest unique and equal to
  Kruskal under that order, with integer atomics and no float races. Outputs are `treeEdgeFlags` (per
  CSR slot), `componentLabels` (the minimum row of each tree) and, optionally, a compact `edges` list with
  `edgeEndpoints`, `edgeCosts` and `standardizedValues`. It unrolls `ceil(log2 rows)` rounds. Tied costs
  can pick different edges than scipy's traversal order, with the same total cost.
- `GPUSkaterRegions`: greedy SKATER cuts of that tree. Each step evaluates the sum-of-squared-deviation
  reduction of every tree edge in parallel (Euler-tour side sums) and applies the best one, ties to the
  lowest slot, while honoring `minimumRegionSize`. The target region count and the minimum size are
  per-frame `parameters` words (`GPU_SKATER_PARAMETER_REGION_COUNT`, `GPU_SKATER_PARAMETER_MINIMUM_SIZE`);
  `maximumRegionCount - 1` steps are compile-time, so the partition with k regions is a prefix of the one
  with k + 1. Outputs are `labels` (the row index of each region's top), `regionCount`, `cutEdges` and
  `cutGains`. The partitions equal spopt `SpanningForest` (squared-Euclidean metric) exactly, including
  floors and islands; spopt's `islands='increase'` adds the tree count to `n_clusters`, while here the
  target counts every region.
- `GPURegionPartitionEvaluation`: scores any label column (SKATER, `GPUSpatialClustering`, `GPUKMeans`):
  region count, within, between and total sum of squared deviations, smallest and largest region, the
  fraction of weights-graph links that cross regions (a compactness proxy) and the number of ignored
  rows, plus per-label sizes and within-region deviations. The cost is `O(labelCapacity * rows)`.

```ts
graph.add(new GPUSpatialWeightsMinimumSpanningTree({weights, attributes, attributeCount: 3, treeEdgeFlags, componentLabels}));
graph.add(new GPUSkaterRegions({weights, attributes, attributeCount: 3, treeEdgeFlags, parameters: skater.importToGraph(graph),
  maximumRegionCount: 12, labels, regionCount}));
```

Cost: the publish and apply kernels are single-threaded and each SKATER step is
`O(sum of side sizes)`, which suits thousands of units and is quadratic on path-like trees of 100,000
rows or more.

### `GPUGroupGeometry`

Per-label summaries over a `uint32` label column, for example the output of `GPUSpatialClustering`:
count, bounds `[minX, minY, maxX, maxY]`, mean center, weighted center, weight sum, medoid, standard
deviational ellipse and standard distance.

- Props: `positions` (`float32x2`), `labels`, `groupCount`, optional `noiseLabel` and `weights`,
  `parameters` (the `getGPUGeographicDistributionParameterValues` layout: origin, standard deviations,
  ellipse convention), `medoidMaximumGroupSize` (default 4096) and `output`.
- Outputs (all optional, at least one): `counts`, `bounds` (`float32x4`), `meanCenters`,
  `weightedCenters` and `weightSums` (need `weights`), `medoidIndices`, `ellipses`
  (`[angle, sigmaX, sigmaY]`, weighted when `weights` is given), `standardDistances` and `overflow`.
- Labels at or above `groupCount` (for example DBSCAN noise, `0xffffffff`), a label equal to
  `noiseLabel`, and rows with a non-finite coordinate are excluded. Empty groups give count 0, NaN
  floats and `GPU_GROUP_GEOMETRY_NO_MEDOID`.
- The medoid is the included row with the smallest summed Euclidean distance to the group, lowest row
  on ties, unweighted. It is quadratic per group, so a group larger than `medoidMaximumGroupSize` gets
  no medoid and sets `overflow`.
- Sums are fixed-order, so results are bitwise reproducible.

```ts
graph.add(new GPUGroupGeometry({
  positions, labels: clusterLabels, groupCount: 64,
  parameters: parameterBuffer.importToGraph(graph),
  output: {counts, bounds, meanCenters, medoidIndices, ellipses}
}));
```

### `GPUGroupConvexHull`

Convex hull per label (Andrew monotone chain), for cluster and hot-region outlines and trajectory
footprints (PostGIS `ST_ConvexHull` per group, `GeoSeries.convex_hull`). Props: `positions`, `labels`,
`groupCount`, optional `noiseLabel`, `maximumVerticesPerGroup` (at least 3), `totalCapacity` and
`output`: `vertexIndices`, optional `vertexPositions`, `offsets` (`groupCount + 1`), `counts`, optional
`sizes` (true hull sizes) and `overflow` (bit 1: a hull exceeded the per-group cap; bit 2: the total
capacity ran out).

- Hulls are counter-clockwise, not closed, and start at the smallest `(x, y)` vertex. Collinear points
  are excluded, equal points collapse to the lowest row, one distinct point gives one vertex, and
  all-collinear points give the two extremes.
- Orientation is exact: coordinates are snapped to a 2^29 lattice with one global power-of-two scale,
  and the sign is computed with 64-bit integer arithmetic. Output positions are the original floats.
- A hull over the per-group cap is dropped whole. Groups are admitted in group order until the total
  capacity runs out, and the first group that does not fit and all later ones are dropped (`counts` 0,
  `sizes` still true).
- `prefilterLevels` (0 to 4, `GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS`; default 2 from 8192 rows)
  discards points inside nested octagon-like filters before the chain. For one 500k-point disk the median
  went from 551 ms (level 0) to 36, 8.8 and 7.6 ms (levels 1 to 3); 64 groups of 500k: 33 to 7.3 ms. With
  4096 small groups the levels cost 2 to 3 ms (3.0 ms at level 0), so set 0 there.
- The chain is one invocation per group, so a group whose points are all on the hull stays serial. For a
  concave outline, close a raster `GPUDistanceField`; exact alpha shapes stay on the CPU.

### `GPUNeighborhoodSummary`

Per-row summaries of a neighborhood over a `GPUSpatialWeights` CSR (momepy `describe`, generalized and
categorical spatial lag). The members of row `i`, in fixed order, are the focal row (with
`includeFocal`, weight `focalWeight`, default 1) and then the CSR slots in slot order. A listed self
slot, out-of-range slots, zero-mask neighbors and, for numeric statistics, non-finite values are
skipped. A masked-out focal row gives `count` 0 and NaN elsewhere. With `k` members, weights `w` and
values `x`:

- `count` is `k`, `weightSum` is `W`, `sum` is `sum w x` (the lag) and `mean` is `sum / W` (NaN when
  `W = 0`).
- `min` and `max` are unweighted. `standardDeviation` is the weighted population deviation about the
  mean, NaN when `W = 0`.
- `median` is unweighted (the mean of the two middle values for even `k`) for rows with at most
  `maximumNeighbors` members (default 32, at most 64). Larger rows give NaN and set `overflow`, which is
  required when `median` is requested.
- `modes` (`uint32`) is the category with the largest weighted frequency, ties to the lowest category
  (`GPU_NEIGHBORHOOD_SUMMARY_NO_MODE` for none). `entropy` is the Shannon entropy (natural log) of the
  weighted category shares. Categorical cost is `O(k^2)` per row.

`statistics` fixes the order of the `output` columns (no repeats): column `c` of row `i` is at
`i * statistics.length + c`. The numeric and categorical parts are two kernels, each below the eight
storage buffer limit. Sums run in member order in f32, so results are deterministic.

```ts
graph.add(new GPUNeighborhoodSummary({
  weights, values, categories, includeFocal: true,
  statistics: ['count', 'mean', 'standardDeviation', 'median'], output, overflow, modes, entropy
}));
```

### `GPUEmpiricalBayesRates`

Global empirical-Bayes standardization and smoothing of rates (esda `assuncao_rate`, as used by
`Moran_Rate` and `Moran_Local_Rate` with `adjusted=True`, and `Empirical_Bayes`). Over the included rows
(nonzero `mask`, finite event count `e`, finite population `b > 0`; esda gives NaN for `b = 0`, here the
row is excluded): `y = e / b`, `m = sum e / sum b`, `s2 = sum b (y - m)^2 / sum b`, `a = s2 - m / mean(b)`.

- `standardizedRates`: `z = (y - m) / sqrt(v)` with `v = a + m / b`, replaced by `m / b` when negative
  (esda's guard).
- `smoothedRates`: `r = w y + (1 - w) m` with `w = a / (a + m / b)`. esda does not clamp `a`, so `w` can
  leave `[0, 1]` when `a < 0`.
- Optional `rawRates` and `summary` (`GPU_EMPIRICAL_BAYES_SUMMARY`, 8 floats: count, event sum,
  population sum, pooled rate, weighted rate variance, prior variance).

Sums are fixed-order two-level trees, so results are bitwise reproducible. Excluded rows are NaN, which
`GPULocalMoran`, `GPUGlobalSpatialStatistics` and the permutation tests exclude. Write
`standardizedRates` to a transient view and pass it as `values`. For the neighborhood-pooled variant see
[`GPUSpatialEmpiricalBayesRates`](#gpuspatialempiricalbayesrates).

```ts
graph.add(new GPUEmpiricalBayesRates({events, populations, standardizedRates: z}));
graph.add(new GPUGlobalSpatialStatistics({weights, values: z, statistics: ['moran'], results}));
```

### `GPUSpatialEmpiricalBayesRates`

Neighborhood-pooled rate smoothing (esda 2.10 `Spatial_Rate` and `Spatial_Empirical_Bayes`, libpysal 4.15
weights). The neighborhood of an included row `i` is `i` plus its listed neighbors that are included
(nonzero `mask`, finite `events`, finite `populations > 0`); only the weights' structure counts (esda sets
`transform = 'b'`). With `E = sum e`, `B = sum b` and `n` members: the spatial rate is `E / B`
(`Spatial_Rate.r`), the local prior mean is `m = E / B`, the local variance is
`s2 = sum b (e / b - m)^2 / B`, `a = max(s2 - m / (B / n), 0)`, and the smoothed rate is
`w e_i / b_i + (1 - w) m` with `w = a / (a + m / b_i)`. Outputs are the optional `spatialRates`,
`smoothedRates`, `priorMeans` and `priorVariances` (NaN for excluded rows). One invocation reduces each row
over its ascending slots, so results are bitwise reproducible. Isolates smooth to their own rate.

The spec pins esda 2.10.0 on a 6x6 queen lattice (spatial rate within 1e-5 relative, smoothed within 2e-4 in
f32). Deviations: a neighborhood with no events gives `0 / 0 = NaN` in esda and 0 here; rows excluded by the
mask or a non-positive population are skipped as neighbors instead of raising; a neighbor ID equal to the
focus row is skipped, where esda would count it twice.

```ts
graph.add(new GPUSpatialEmpiricalBayesRates({events, populations, weights, smoothedRates}));
```

### `GPUEmergingHotSpots`

ArcGIS-style emerging hot spot analysis over a dense space-time cube, on a regular lattice or on any
spatial weights.
`values` is indexed `cell * sliceCount + slice` (the `GPUTemporalReduction` layout, `cell = row *
gridWidth + column`); a `float32` view treats NaN as a missing bin, a `uint32` view (for example
`GPUTemporalReduction` `counts`) is read as `f32(count)`. An optional per-cell `mask` excludes cells.

Pass exactly one neighborhood: `gridWidth` and `gridHeight` for the lattice, with a per-frame
`radius` in cells, or `weights` over the cells (rows are cells) plus a compile-time `selfWeight`,
which works on H3 cells, polygons or points. With binary weights equal to the lattice disc, weights
mode reproduces lattice mode.

```ts
const hotSpots = new GPUEmergingHotSpots({
  values, gridWidth: 64, gridHeight: 64, sliceCount: 24, maximumRadius: 4,
  parameters, giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount
});
graph.add(hotSpots);
parameterBuffer.write(
  getGPUEmergingHotSpotParameterValues({radius: 2, temporalWindow: 3, confidenceLevel: 0.95})
);
```

Passes (deterministic, no float atomics):

1. Global `n`, mean and standard deviation of valid bins: fixed 1024-bin block sums added in order.
2. Space-time Gi* per bin: the spatial neighbors (lattice cells with `dx^2 + dy^2 <= radius^2`, or
   the cell's weights row plus the cell itself at `selfWeight`) over the current and `temporalWindow`
   previous slices, `z = sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))` over valid bins.
3. Per-cell Mann-Kendall on the finite z series: exact integer `S`, tie-corrected variance
   `[n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18` (ties by exact f32 equality), continuity-corrected z, and a
   two-sided p-value from the Numerical Recipes `erfcc` fit.
4. Classification into the 17 `GPU_EMERGING_HOT_SPOT_CATEGORIES` (0 = no pattern, hot 1..8, cold 9..16:
   new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical).

Per-frame parameters (radius, temporal window, critical z or confidence level, trend significance,
persistence fraction) live in a parameter buffer and never rebuild the graph. Compile-time:
`gridWidth`, `gridHeight`, `sliceCount` (at most 256), `maximumRadius` (at most 32), value format,
optional views. Outputs are caller-owned: `giZScores` (per bin), `trendZ`, `trendP`, `trendS`,
`category`, `hotSliceCount`, `coldSliceCount` (per cell) and optional `globalStatistics`
`[n, mean, variance, standardDeviation]`.

### `GPUGeographicDistribution`

`GPUGeographicDistribution` computes the ArcGIS "Measuring Geographic Distributions" statistics
over planar points, per group or per masked selection: counts, weight sums, mean centre, Weiszfeld
median centre, standard distance, standard deviational ellipse, linear directional mean, and
polygon rings for the ellipse and the standard-distance circle. Project longitude/latitude first:
every statistic is Euclidean.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'distribution-parameters',
  format: 'float32',
  length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
});
graph.add(
  new GPUGeographicDistribution({
    positions, weights, groupIds, groupCount: 8, mask,
    parameters: parameters.importToGraph(graph),
    output: {meanCenters, ellipses, ellipseVertices}
  })
);
parameters.write(
  getGPUGeographicDistributionParameterValues({origin: [x0, y0], standardDeviations: 2})
);
```

- Inputs: `positions` (`float32x2`, fewer than 2^24 rows), optional `weights` (a non-finite,
  negative or zero weight excludes the row), `groupIds` with compile-time `groupCount` (ids at or
  above it are excluded), `mask`, and `lineEnds` for the directional mean.
- Per-frame parameters (no recompile): `origin` (local origin subtracted before every sum for
  float32 stability), `standardDeviations`, `ellipseConvention` (`'arcgis'` scales the axes by
  `sqrt(2)`, `'standard'` does not), `orientationOnly`, `medianTolerance`.
- Divisors are the weight sum, as in ArcGIS (not `n - 1`). Second moments are central (a mean
  pass, then a pass about the mean), never `E[x^2] - E[x]^2`.
- `ellipses` holds `[angle, sigmaX, sigmaY]` per group. As in ArcGIS the "y" axis is the long one
  (`sigmaY >= sigmaX`). `angle` is counter-clockwise from +x of the short x axis, so the long axis
  points at `angle + pi / 2`; a ring vertex is `M + R(angle) (sigmaX cos t, sigmaY sin t)`.
- `directionalMeans` holds `[meanAngle, circularVariance, meanLength]` (angles counter-clockwise
  from +x). With `orientationOnly`, angles are doubled before averaging and the mean angle lies in
  `(-pi/2, pi/2]`.
- `ellipseVertices` and `circleVertices` are counter-clockwise rings of `polygonVertexCount`
  (default 64) vertices per group, not closed, ready for a deck.gl `PolygonLayer` or `PathLayer`.
- The median centre runs `medianIterations` (default 24) fixed Weiszfeld steps from the mean centre
  (six nodes per step); `medianConverged` is 1 when the last step moved at most `medianTolerance`.
- Empty groups report 0 counts and weight sums and NaN in every other output.
- Determinism: rows are sorted by group once and every sum, including each Weiszfeld step, is a
  fixed-order segmented tree sum. No float atomics; repeated encodings are bitwise identical.

### `GPURegionStatistics`

Selects points inside a per-frame rectangle, circle, lasso polygon (world space or screen pixels
through a view-projection transform), pick region (`GPUIndexPickingTarget.addRegionPass` result),
or caller mask, and reduces them into selected count, value count, sum, mean, minimum, maximum, and
a fixed-bin histogram packed into one small `summary` buffer. Optional stable selected IDs stay on
the GPU for highlighting. `GPURegionStatisticsReadback` reads the summary through a
`GPUReadbackRing` without stalling. `GPURegionMask` and `GPUPickRegionMask` are exported as
standalone building blocks.

```ts
graph.add(new GPURegionStatistics({
  selection: {kind: 'polygon', vertices: lasso, vertexCount: lassoCount.importToGraph(graph),
    screenTransform: transform.importToGraph(graph)},
  positions, values, histogram: {binCount: 16}, output,
  summary: graph.importGPUData('summary', summaryData)
}));
```

An optional `spatialIndex: {kind: 'grid', index, candidateCapacity}` takes a caller-built
`GPUGridIndex` over the same positions and reduces only candidates from the cells a world-space
rectangle, circle, or lasso touches. Results match the full scan except for float reassociation in
`sum` and `mean`; the `candidatesTruncated` flag reports capacity overflow or rows missing from the
index.

An optional `drawInstanceCount` (one-row `uint32` view, for example the instance-count word of an
indirect draw record) receives the same capacity-clamped selected count as `output.count` on every
encoding, so selected rows can be drawn instanced through `output.ids` without a readback. It
requires `output`; statistics-only graphs read the count from `summary`.

### `GPUZonalStatistics`

Choropleth aggregation. Each point is joined to the polygon feature that contains it (an internal
`GPUPointInPolygonJoin`), or the caller supplies per-point feature rows. The points are then
reduced per feature to count, valid-value count, sum, weighted sum and weight sum, mean, minimum,
maximum, density (count divided by a caller area or a GPU shoelace area), and a feature-value
`[min, max]` extent for color scales.

The output views you pass select the statistics at compile time. Empty features give a sum of 0 and
NaN for the mean, minimum and maximum. `sumOrder: 'sorted'` stably sorts points by feature and
reduces each segment in a fixed tree order, so sums are bitwise reproducible at a bounded cost. It is
the default when every point-rate input (points, values, weights, `pointFeatureRows`) is packed.
Chunked inputs fall back to `sumOrder: 'atomic'`, whose float atomics can vary in the last bits
between runs and serialize when many points share a feature.

```ts
graph.add(new GPUZonalStatistics({
  features: {kind: 'polygons', polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
    candidateCapacity: points.length * 2},
  points, values: incomes, weights: households,
  output: {counts, means, densities, extent, extentStatistic: 'mean', overflow}
}));
```

`output.uncertainCount` (one `uint32` row) reports the join's uncertain candidate pairs. A nonzero
value means some points could not be classified and the per-feature statistics may be low. It is
always 0 for caller-supplied feature rows.

### `GPUFocalStatistics`

Moving-window statistics over a float32 raster, like ArcGIS `FocalStatistics` and GRASS
`r.neighbors`: mean, sum, min, max, range, population standard deviation, and the count of valid
cells. The radius, window shape, minimum count, and center nodata policy are per-frame.

```ts
graph.add(
  new GPUFocalStatistics({
    values: raster, // row-major float32, NaN = nodata
    validity, // optional uint32, 0 = nodata
    noDataValue: -9999, // optional sentinel
    width,
    height,
    maximumRadius: 16, // compile-time cap, at most 64
    parameters: focalParameters.importToGraph(graph), // 4 float32
    output: {mean, standardDeviation, min, max, count} // any non-empty subset
  })
);
focalParameters.write(
  getGPUFocalStatisticsParameterValues({radius: 3, shape: 'circle', minimumCount: 5})
); // no recompile
```

- Square windows cover `|dx|, |dy| <= floor(radius)`; circles keep offsets with
  `dx^2 + dy^2 <= radius^2`. The radius is clamped to `maximumRadius`.
- Window cells outside the raster and nodata cells are skipped, so edge windows are clipped.
  A cell is nodata (NaN) with fewer than `max(minimumCount, 1)` valid cells, or when its own value
  is nodata and `propagateCenterNoData` is set. `count` is written for every cell.
- Direct gather in fixed row-major window order: deterministic, no atomics. Standard deviation
  subtracts the window mean in a second pass. Moments and extremes are separate kernels, each
  scheduled only when one of its outputs is requested.
- Cost is `(2r + 1)^2` reads per cell; see Status and limitations for the still-open separable and
  summed-area variants.

### `GPUSimilarLocations`

Ranks rows by weighted Euclidean distance, in standardized attribute space, to a reference row or the mean
of several selected rows (the ArcGIS Similarity Search task; checked against NumPy, SciPy and scikit-learn distances). Props: `attributes` (row-major `rowCount * attributeCount`), `attributeCount`,
`selection` (a `uint32` mask of the reference rows), optional `mask`, per-frame `parameters`
(`getGPUSimilarLocationsParameterValues`: result count, most or least similar, reference exclusion and
attribute weights), `standardization` (`'zscore'`, population, or `'rank'`, percentile; compile-time) and
`maximumResultCount`.

- Outputs: `ranks` (0 is the most similar; unranked rows hold `GPU_SIMILAR_LOCATIONS_NO_RANK`),
  `distances` (NaN when unranked), `topIds` (the best rows in rank order) and `count`.
- Rows with a non-finite attribute or a zero mask are never ranked. A stable `GPUSort` on the distance bits
  keeps ties on the lowest ID.
- `'rank'` standardization uses a quadratic scan, so it suits tens of thousands of rows, not millions.

### `GPULocalOutlierFactor`

Local Outlier Factor (Breunig 2000; geo `OutlierDetection`, scikit-learn `LocalOutlierFactor`, Sedona
`ST_LocalOutlierFactor`) over a kNN table. `neighbors` is a `GPUSpatialWeights` with `distances`, typically
`GPUNeighborSearch` in `'knn'` mode as a self join; the weight values are unused. Per row it writes
`kDistance` (the largest neighbor distance), `localReachabilityDensity = 1 / (mean max(kDistance_j, d_ij) +
densityFloor)` and `lof = mean(lrd_j) / lrd_i`, which equals `-negative_outlier_factor_`. Optional outputs
are an `outlier` mask (`lof > threshold`) and a one-row `outlierCount` (integer atomics, deterministic).
`parameters` (`getGPULocalOutlierFactorParameterValues({threshold = 1.5, densityFloor = 1e-10})`) change per
frame without recompiling. The floor keeps coincident clusters finite at density `1 / densityFloor`, and
points whose neighbors are such duplicates score enormous values, as in scikit-learn, so use a `k` above
the duplicate count or a larger floor. A row with no neighbors gets kDistance 0, density 0 and lof 1 and is
never flagged. A radius-mode table works, but the k-distance is then the largest in-band distance. Against
a pinned scikit-learn run (24 points with `k = 5`, and a duplicate case) kDistance was within 2e-6 relative
and lof within 1e-4, with the mask and count exact.

## Distribution dynamics & space-time tests

Distribution dynamics contributors use period-major columns: period `t` of row `i` is at
`t * rows + i`.

### `GPUClassAssignment`

`new GPUClassAssignment({values, breaks, classCount, mask, output})` assigns each value the number of
inner edges `e[1..k-1]` that are `<=` the value, clamped, the same rule as `GPUClassBreaks.classCounts`.
NaN and masked rows get `GPU_CLASS_ASSIGNMENT_NO_CLASS` (`0xffffffff`). Run one `GPUClassBreaks` over the
pooled `n * T` column and assign every period with its breaks to get one legend for a time slider.

### `GPUClassificationFit`

Goodness of fit of a classification (mapclassify 2.11 `adcm`, `get_gadf()`, `get_tss()`): pass `values`, the
`classes` from `GPUClassAssignment`, `classCount` (1 to 256) and an optional `mask`. Outputs are `counts`,
exact per-class `medians` (numpy convention, mean of the two middle values), `absoluteDeviations` and
`squaredDeviations` (one row per class plus a last row for all counted rows) and an optional five-row
`summary` `[GADF, ADCM, ADAM, GVF, TSS]` (`GPU_CLASSIFICATION_FIT_*`). `ADCM` is the sum of absolute
deviations from the class medians, `ADAM` the same from the overall median, `GADF = 1 - ADCM / ADAM` (1 when
`ADAM` is 0), and `TSS` is mapclassify's name for the within-class sum of squares about the class means, not a
total sum of squares. `GVF = 1 - TSS / total sum of squares` is an extension, because mapclassify 2.11 has no
`gvf`. Empty classes add 0, as in mapclassify. Two stable radix sorts (value, then class) leave each class a
sorted segment, so medians are read directly and sums run in sorted order: no atomics, bit-reproducible, no
per-class search passes.

The spec pins mapclassify 2.11.0 Quantiles, NaturalBreaks and EqualInterval fits (about 1e-4 relative).
Deviations: sums run in float32 and in sorted order rather than index order, non-finite values are skipped, and
at least one row is required.

```ts
graph.add(new GPUClassificationFit({values, classes, classCount: 5, output: {counts, medians, absoluteDeviations, squaredDeviations, summary}}));
```

### `GPUTransitionMatrix`

Markov transition counts between classes of consecutive periods (giddy `Markov`).

- Props: `classes`, `rows`, `periods`, `classCount`, optional `periodLag` (default 1),
  `conditionClasses` with `conditionCount`, `mask`, and `output: {counts, probabilities?, rowTotals?, ignored?}`.
- `counts` holds exact `uint32` counts at `(c * K + from) * K + to` for `t = 0 .. T - lag - 1`
  (integer atomics, independent of order). `probabilities` are the row-normalized floats, 0 for empty
  rows. `rowTotals` and `ignored` are explain columns.

```ts
graph.add(new GPUClassAssignment({values, breaks, classCount: 5, output: classes}));
graph.add(new GPUTransitionMatrix({classes, rows, periods, classCount: 5, output: {counts, probabilities}}));
```

### `GPUSpatialMarkov` and `GPULISAMarkov`

- `GPUSpatialMarkov` (giddy `Spatial_Markov`): props `values` (`float32`, `rows * periods`, period-major),
  `classes` (`uint32`, same layout, for example from `GPUClassAssignment`), `classCount` (`K`, 1 to 64),
  `weights` (square self-join `GPUSpatialWeights`, constant across periods), `periods` (at least 2),
  optional `periodLag` (default 1), `normalize` (default true), `mask`, `lagMaximumClassCount`,
  `lagParameters` (`getGPUClassBreaksParameterValues` view), optional `lagMethods` (default
  `['quantile']`) and an `output` of `counts`, `lagBreaks`, `lagClassCount` and the optional
  `probabilities`, `rowTotals`, `ignored`, `lagValues` and `lagClasses`. For each period it runs `GPUSpatialLag` (row-standardized by default,
  `normalize`), classifies the lag with one pooled `GPUClassBreaks`, assigns the lag classes, and builds a
  transition matrix conditioned on the lag class at the start period. `counts` is
  `lagMaximumClassCount x K x K`; slices past the produced `lagClassCount` stay zero.
- `GPULISAMarkov` (giddy `LISA_Markov`): props `values`, `weights`, `periods` (at least `periodLag + 1`),
  optional `periodLag`, `parameters` (the `GPULocalMoran` view from
  `getGPUSpatialAutocorrelationParameterValues`, which carries the significance level), optional `mask`
  and an `output` of `counts` and the optional `probabilities`, `rowTotals`, `ignored` and `quadrants`. It runs `GPULocalMoran` per
  period and counts moves between the five states (`GPU_LISA_MARKOV_STATE_COUNT`: 0 not significant, 1 HH,
  2 LH, 3 LL, 4 HL). Significance is the analytic z-test, not esda permutations. A row with a non-finite
  value or no neighbors is state 0 and is counted; masked rows are not. `quadrants` (`rows * periods`) is
  optional.

Both emit one node chain per period, which suits periods in the tens. Goodness-of-fit tests (giddy `Q`,
`LR`) take the `counts` tensor on the CPU; they are not provided. giddy `relative` rescaling is not built
in: divide each period before classifying.

### `GPUKnoxTest` and `GPUMantelTest`

Space-time interaction tests over the pairs of a self-join `GPUSpatialWeights` (typically radius-mode
`GPUNeighborSearch`): the entries `j > i` of each row are the spatial pairs. Knox pairs are only as
complete as the pair list; the capacity overflow flag lives on `GPUNeighborSearch`.

- `GPUKnoxTest({pairs, times, parameters, maximumPermutations, statistics, summary})` counts the pairs
  that are close in space and have `|t_i - t_j| <= timeThreshold`, then runs a Monte Carlo (modified Knox)
  test over `P` Feistel-permuted time assignments. Counts are exact integers, and the expected value comes
  from the exact count of time-close pairs. `statistics` is `uint32` with `P + 1` entries. The CPU function
  `getKnoxPoissonPValue(observed, expected)` gives the classic Poisson upper tail.
- `GPUMantelTest({pairs, ...})` (the weights need `distances`) is the Pearson correlation between pair
  distances and `|t_i - t_j|`, with permuted times. Sums are f32 shifted partials, within about 1e-4 of a
  double-precision oracle.

`timeThreshold`, `seed` and `permutations` are per-frame, written with
`getGPUSpaceTimeParameterValues({seed, permutations, timeThreshold})` (`uint32[4]`). `times` must be
finite. `summary` (`float32`, 12 entries, `GPU_SPACE_TIME_SUMMARY`) holds the observed statistic, pair
count, time-close pair count, expected value, permutation mean and variance, greater and lesser counts,
the pseudo p-values (greater, lesser, two-sided, `(1 + count) / (P + 1)`) and the z-score. The Kulldorff
space-time scan is not provided.

### `GPUSpatialScanStatistic`

Kulldorff's Poisson scan statistic (SaTScan, CARTO `DETECT_SPACETIME_ANOMALIES`) over zones, which are
cells or points, with `cases` (`uint32`) and a `baseline` (population or expected counts) per zone and
optional time bucket (`zone * timeBuckets + bucket`). The null model is a Poisson process with
`E[cases] = baseline * C / sum(baseline)`.

- Windows: the `k` nearest zones of every center (the `'circle'` shape never splits equidistant zones;
  `'nearest'` is plain k-nearest), up to `maximumWindowZones` (at most 32) and a population share. With
  `timeBuckets > 1` each window is a cylinder over every run of up to `maximumTimeBuckets` buckets.
- Outputs: the most likely cluster plus greedy, non-overlapping secondary clusters (no shared zone) in
  `clusterIndices` (`GPU_SCAN_STATISTIC_CLUSTER_INDEX`) and `clusterStatistics`
  (`GPU_SCAN_STATISTIC_CLUSTER`: LLR, observed, expected, p, radius, ratio); `statistics` (the observed
  maximum then the maximum of every Monte Carlo replicate), `summary` (`GPU_SCAN_STATISTIC_SUMMARY`) and
  an optional per-zone best LLR `zoneStatistics`.
- Monte Carlo: cases are redistributed multinomially in proportion to the baseline with the Philox
  streams of the permutation contributors, and `p = (1 + #{max_rep >= LLR}) / (P + 1)`; secondary clusters
  use the same replicate maxima. The result is deterministic (integer atomics, fixed-order reductions).
  Limits, the seed and the replicate count are per-frame parameters
  (`getGPUSpatialScanParameterValues`), up to the compile-time `maximumPermutations` (at most 16384).
- Limits: only the Poisson model for high rates. Bernoulli and ordinal models, low-rate and
  both-direction scans, elliptic windows and Gumbel p-values are not provided. The replicate table needs
  `(maximumPermutations + 1) * zones * timeBuckets <= 2^25` words and the cost is about
  `(P + 1) * zones * maximumWindowZones * timeBuckets^2`. Draws use `mulhigh`, so replicate maxima match
  an f64 oracle almost always but not bit-exactly; LLR below about 1e-3 is approximate in f32; `baseline`
  must be finite and non-negative.

## Accessibility & change of support

### `GPUCatchmentAccessibility`

Floating catchment accessibility on any `GPUSpatialWeights` (the `access` package `two_step_fca`).
`method: '2sfca'` or `'3sfca'`, with `supply`, `demand`, `facilityWeights` (row `j` lists the demand
points `i` with weight `f(cost)`), optional `demandWeights` (row `i` lists facilities; it defaults to
`facilityWeights`, which is only valid for symmetric self-join weights and is not checked) and the
outputs `accessibility`, `ratios` and `reachableFacilities`.

- 2SFCA: `R_j = S_j / sum_i P_i w_ji`, `A_i = sum_j w_ij R_j`.
- 3SFCA uses selection weights `G_ij = w_ij / T_i` with `T_i = sum_k w_ik`: `R_j = S_j / sum_i P_i G_ij`,
  `A_i = sum_j G_ij R_j`. Demand rows with `T_i = 0` are skipped.
- Supply conservation holds for both methods, with any decay function (binary, Gaussian, a network-cost
  kernel), when the demand-side weights are the transpose of the facility-side weights: `w_ij = w_ji` for
  every pair, with the same set of pairs. Then `sum_i P_i A_i = sum_j S_j` over the facilities that capture
  demand (`sum_i P_i w_ji > 0`, or `sum_i P_i G_ij > 0` for 3SFCA). For 2SFCA this is
  `sum_i P_i A_i = sum_j R_j sum_i P_i w_ij`, and the inner sum is the denominator of `R_j`, so each
  facility contributes exactly `S_j`. It breaks when the two steps use different weights (different
  decays or radii, or a `demandWeights` that is not the transpose), and by the supply of facilities with no
  demand in range, which is not allocated to anyone. Sums are f32, so the identity holds to rounding. The
  default `demandWeights = facilityWeights` satisfies the condition for symmetric self-join weights.
  3SFCA differs in what `A_i` means, a selection-weighted average of the `R_j`, not in conservation.

Network accessibility over a cost matrix is [`GPUNetworkAccessibility`](/docs/api-reference/experimental/gpu-network).

### `GPUHuffTradeAreas`

Huff probabilities `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik` from `attractiveness` and
`demandWeights`, with `alpha` from a one-element `parameters` view. The trade area of demand `i` is its
modal facility (lowest ID on ties, `GPU_HUFF_NO_TRADE_AREA` when none). Props: `attractiveness` (`float32`,
one row per facility), `demandWeights` (row `i` lists the facilities of demand `i`), optional `parameters`
(`float32`, `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` = 1 word `[alpha]`, default 1, must be positive), and the
caller-owned `tradeArea` (`uint32` per demand row, required). Optional outputs are `probabilities` (one
per `demandWeights` slot, aligned with `demandWeights.neighbors`), `tradeAreaProbability` (per demand row)
and `expectedDemand`, `expectedDemand_j = sum_i P_i p_ij`, which needs `facilityWeights` (the transpose
of `demandWeights`) and `demand`.

### `GPUArealInterpolation`

Area-weighted transfer between two zone systems (tobler `area_interpolate`, CARTO `ENRICH_POLYGONS`) on
a common fine raster. Rasterize both systems to the same grid with `GPUPolygonRasterization` (or any zone
raster) and pass the two zone views, so H3 or hexagon rasters work too.

- Props: `sourceZones`, `targetZones` (`uint32`, same length; a value at or above the count means no
  zone), `sourceCount`, `targetCount`, optional `cellWeights` (a dasymetric raster; area becomes a sum of
  weights, and non-finite or non-positive cells carry no area), `denominator` (`'zone'` by default, the
  whole-zone area as in tobler, or `'overlap'`, only the part overlapping the other system, which
  conserves mass), `mode` (`'extensive'` by default or `'intensive'`, which selects `weights.weights`),
  `weights` (a caller-owned CSR with `targetCount + 1` offsets; the neighbor capacity is the maximum
  number of pairs), `alternateWeights` (the other normalization, from the same run), `areas` (raw overlap
  area per slot), `overflow`, `totalPairs` and `categories` (`{sourceCategories, categoryCount, output}`,
  `T x K`).
- The cell weight `m` (1, or `cellWeights`) summed per source `s` and target `t` gives `a_st`. Extensive:
  `w_ts = a_st / A_s`. Intensive: `w_ts = a_st / B_t`. `A_s` and `B_t` are the zone or overlap areas.
  Rows are targets and neighbors are sources, so the CSR invariants hold; it is a cross weights matrix.
- Categorical share `(t, k)` is the overlap area with sources of class `k` divided by the area of `t`
  covered by any source, so it sums to 1 where covered.
- On overflow the offsets clamp to capacity, `overflow` is 1 and `totalPairs` gives the size needed.
- `unweightedFastPath` (default on, used when there are no `cellWeights`) takes pair areas from run lengths
  and zone areas from counts instead of per-cell accumulation. Results are bitwise identical; it was 1.4x
  faster on the explorer's graph, about 35 percent on 384 by 384 zone denominators and about equal for the
  overlap denominator, where the two radix sorts dominate.
- Cell centers decide membership, so every area is a cell count and accuracy is about one cell along every
  boundary; slivers smaller than a cell can be missed. It is never an exact polygon intersection. f32 sums
  are exact up to 2^24 cells per zone. Results are deterministic.

Transfer the values with `GPUSpatialLag` over the area shares, using `sourceCount` and `columnCount`. The
tobler `area_interpolate` extensive and intensive results match to 1e-5 on grid-aligned zones.

```ts
graph.add(new GPUArealInterpolation({
  sourceZones, targetZones, sourceCount, targetCount,
  weights: {offsets, neighbors, weights}, alternateWeights, overflow
}));
graph.add(new GPUSpatialLag({
  weights: {offsets, neighbors, weights}, sourceCount, values: population, output: populationOnTargets
}));
```

### `GPUPycnophylactic`

Tobler pycnophylactic interpolation on a raster. Props: `width`, `height`, `zones` (`uint32`; a value at or
above `zoneCount` is outside), `zoneCount`, `totals` (per zone), `iterations` (0 to 512, compile-time),
`kernel` (`'rook'`, the default 4-neighbor mean, or `'box'`, a 3x3 mean including the cell) and `output`
(mass per cell). Each cell starts at `total / cells`. Each iteration takes the focal mean over in-area
neighbors, restores the zone total additively, clamps negatives to 0 and rescales multiplicatively to the
total; zero-sum zones fill uniformly. Totals hold to f32 accuracy and values are non-negative after every
iteration. Zone sums are fixed-order segmented sums over cells sorted once by zone. There is no ancillary
initial surface, no convergence criterion and no barrier. `'rook'` on a checkerboard has a -1 eigenmode
(Tobler's original); `'box'` is smoother.

### `GPUSegregation`

Segregation indices from a unit-by-group count table, aspatial or spatial (PySAL `segregation`).

- Props: `unitCount`, `groupCount` `K` (2 to 16), `groupCounts` (`units * K` float32, row-major),
  `scales` (default `[null]`; `null` is aspatial, otherwise one square `GPUSpatialWeights` per bandwidth,
  which gives the multiscale profile), `selfWeight` (1), `atkinsonB` (0.5, in `(0, 1)`),
  `spatialForm`, `indices` (`scales * stride` floats, layout from `getGPUSegregationLayout(K)`: entropy,
  multi-group dissimilarity, diversity, dissimilarity `[K]`, isolation `[K]`, Atkinson `[K]`, interaction
  `[K * K]`) and optional `local` outputs (`environment`, `entropy`, `dissimilarity`, `theil`, per scale;
  they sum to the global indices).
- `spatialForm: 'environment'` (default) is the Reardon and O'Sullivan (2004) form: `pi_im = x~_im / t~_i`
  with `x~_im = selfWeight x_im + sum_j w_ij x_jm` replaces `p_im`, and the sums keep the unit populations
  `t_i`. `'smoothed-population'` replaces the whole table by the smoothed table `x~` and runs the aspatial
  formulas on it, which equals PySAL's spatially implicit indices (PySAL also rounds the smoothed counts).
- With `P_m = X_m / T`, `E = sum P ln(1 / P)`, `E_i = sum pi ln(1 / pi)` and `I = sum P (1 - P)`: Theil
  `H = sum_i t_i (E - E_i) / (E T)`; multi-group `D = sum_i sum_m t_i |pi_im - P_m| / (2 T I)`;
  `D_g = sum_i t_i |pi_ig - P_g| / (2 T P_g (1 - P_g))`; isolation `xPx_g = sum_i (x_ig / X_g) pi_ig`;
  interaction `xPy_gh = sum_i (x_ig / X_g) pi_ih`; Atkinson
  `A_g = 1 - P / (1 - P) |sum_i (1 - pi_ig)^(1 - b) pi_ig^b t_i / (P T)|^(1 / (1 - b))`. Undefined indices
  are 0.
- Sums are f32, within 1e-5 to 2e-4 relative of float64 and of PySAL on the test fixtures. Memory is
  `O(units * K^2)` per scale. The boundary-based `SpatialDissim` and permutation inference are not
  provided. `atkinsonB` and `selfWeight` are compile-time.

## Spatial joins

### `GPUSpatialPredicateJoin`

Joins two sets of planar features by a spatial predicate, like GeoPandas `sjoin` or PostGIS
`ST_Intersects`, `ST_Contains`, `ST_Within` and `ST_DWithin`. Either side is
`{kind: 'points', positions}`, `{kind: 'lines', positions, lineOffsets}` (one linestring per
feature) or `{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}`
(multipolygons, holes as later rings, GeoArrow layout); all nine combinations are supported.

`predicate` is `'intersects'`, `'contains'`, `'within'`, `'dwithin'` (with an inclusive planar
`distance`), `'covers'`, `'coveredBy'`, `'touches'`, `'crosses'`, `'overlaps'`, `'equals'`,
`'containsProperly'` or `'relate'` (with a DE-9IM `pattern`). Boundaries follow OGC / DE-9IM: a point has no boundary, a linestring's boundary is its
endpoints unless it is closed, and a polygon's boundary is its rings.

- `intersects`: the closed geometries share a point, so polygons touching at an edge or corner
  intersect, and so does a point on a polygon boundary.
- `contains(a, b)`: `b` lies in the closure of `a` and the interiors intersect. A polygon does not
  contain a point or line on its boundary, and a line does not contain its own endpoints.
- `within(a, b)` is `contains(b, a)`.
- `dwithin`: the minimum distance between the closed geometries is at most `distance`.
- `covers` and `coveredBy`: `contains` and `within` that allow boundary contact.
- `touches`: the boundaries meet and the interiors do not (never true for two points).
- `crosses`: the interiors meet and the intersection has a lower dimension than the larger input, by the
  OGC dimension rules (false for polygon/polygon and point/point).
- `overlaps`: the same dimension, the interiors meet and neither contains the other.
- `equals`: the same point set. `containsProperly`: `b` lies in the interior of `a`.
- `relate` with `pattern`: any DE-9IM pattern (characters `T F * 0 1 2`), as a string or an any-of list.

The pipeline is upstream primitives end to end: per-feature bounds into a `GPUBVH` over the right
side, a count pass, `GPUScan` and an ordered write of candidate pairs, the exact predicate per
candidate (point-polygon pairs use `GPUPairwisePointInPolygon`; points on an edge it reports as
uncertain are decided by an f32 test and counted in `uncertainCount`), then a scan and a
capacity-bounded scatter. Output `pairs` (`leftIds`, `rightIds`, `count`, `overflow`, optional
`totalCount`) are unique and sorted by `(left, right)`. On overflow the output is a sorted subset and
`totalCount` and `candidateCount` give the sizes to resize to.

Pass `weights` to also write the matches as a `GPUSpatialWeights` with left features as rows and
weight 1; set `excludeSameRow` for a self join. A predicate join is then a weights producer, for
example `dwithin` between polygons.

The new predicates are masks over one DE-9IM matrix per candidate pair, computed from structure (vertex
locations, edge crossings, collinear overlaps, and edge pieces located against the other geometry, with
polygon areas decided from edge pieces) for all nine point, line and polygon combinations. Semantics
follow GEOS and Shapely. `intersects`, `contains`, `within` and `dwithin` keep their short-circuiting
kernels and use the matrix only when `relate` output is requested; the specs assert both paths agree.
Only bounding-box candidates are enumerated, so a pattern that admits disjoint geometries (all of II, IB,
BI and BB allow `F`) is rejected at construction; use `how: 'anti'` with `intersects` for disjoint.
Lines are single linestrings with the OGC endpoint boundary (multilinestrings and mod-2 boundaries are not
modeled). The matrix kernel is `O(n * m)` per pair, which suits small features and is not tuned for large
polygons.

Predicates other than point-in-polygon run in f32, so recenter or project coordinates before
joining near-degenerate geometry. Polygons must be valid.

```ts
graph.add(new GPUSpatialPredicateJoin({
  left: {kind: 'points', positions: points},
  right: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},
  predicate: 'within',
  candidateCapacity: 1 << 20,
  pairs: {leftIds, rightIds, count, overflow}
}));
```

**`relate` output.** `relate` is an optional `GraphDataView<'uint32'>` as long as `pairs.leftIds`; slot `k`
holds the matrix of `(leftIds[k], rightIds[k])`. Each of the nine cells (order II IB IE BI BB BE EI EB EE,
row-major, left parts then right parts) takes 2 bits at `[2k, 2k + 1]` with value `dimension + 1` (0
empty, 1 point, 2 line, 3 area). `formatGPUSpatialRelate(value)` decodes it on the CPU to a string such as
`'FF2F11212'`. It is not available for `dwithin`.

```ts
graph.add(new GPUSpatialPredicateJoin({
  left, right, predicate: 'touches', candidateCapacity: 1 << 16,
  pairs: {leftIds, rightIds, count, overflow}, relate, uncertainCount
}));
```

**Anti joins.** `how: 'anti'` with `unmatched: GPUCompactOutput` emits the left rows that have no match, in
ascending order (`pairs`, `weights` and `relate` are not allowed with it). When any capacity overflows the
anti output is incomplete in the unsafe direction, because rows whose candidates were dropped look
unmatched, so check `unmatched.overflow`.

**Exactness.** Relate orientation signs are exact: an f32 filter decides the clear cases and the rest
fall back to exact integer arithmetic (256-bit), so touches, crosses, covers and the other DE-9IM
predicates do not depend on f32 rounding of the orientation test. `uncertainCount` counts only pairs with
non-finite input or an exponent span above 200 bits, and zero means every pair was decided. Edge piece
midpoints and break parameters are still computed in f32: the orientation signs are exact, the location of
a piece shorter than about one ulp is not, so a line starting one ulp outside a polygon edge can have its
`crosses` piece classified as boundary. Point/polygon pairs use the robust double-single classifier alone
(they do not get the exact re-decision of `GPUPointInPolygonJoin`), and pairs it cannot certify are
counted in `uncertainCount`. The legacy kernels for pairs other than point/polygon do not track
uncertainty and report 0.

**Per-frame distance and pattern.** `distance` is a `number` or a `GraphDataView<'float32'>`, and `pattern`
is a `GPUSpatialRelatePattern` or a `GraphDataView<'uint32'>`. A view is read from a parameter buffer, so
it changes without recompiling and gives bit-identical results to the compile-time value (a negative, NaN
or infinite distance selects nothing). Fill a pattern view with `packGPUSpatialRelatePattern(pattern,
slotCount)`, which writes `GPU_SPATIAL_RELATE_PATTERN_WORDS` (2) words per slot; unused slots are zero and
match nothing, and the helper rejects patterns that admit disjoint geometries. Size `candidateCapacity`
for the largest distance. `predicate` stays compile-time because it selects the kernel family; for a
per-frame predicate choice use `predicate: 'relate'` with patterns.

**Relate performance.** Per-edge break parameters are collected in one pass (up to 24 per edge) and the
covers scan runs only when a collinear edge exists. The workgroup relate kernel also builds a per-pair
y-slab index of each polygon side in workgroup memory (32 slabs, 1280 entries per side), so point location
scans the closing edges and the query's slab instead of every edge; features under 48 edges, with a
degenerate y extent or over 1280 entries use the plain scan. Results are identical. On 100 polygon pairs
the batched breaks and the slab index took relate from about 7.5 ms to 4.3 to 5 ms, against about 1 ms for
GEOS: the break collection, covers scan and vertex loops are still `O(n * m)`, and about 2 ms of the
remaining time is fixed graph and readback cost.

**`onAttribute: {left, right}`** (GeoPandas `sjoin(on_attribute=)`). Two `uint32` key views, one per left
row and one per right row. A pair survives only when the keys are equal. The filter empties candidate
slots before the exact kernels, so `candidateCount` stays the bounding-box count before filtering, and
`how: 'anti'` then reports left rows with no key-equal match. Its presence is compile-time and its
contents are per-frame. Defaults are unchanged.

#### Engine selection

`engine?: 'auto' | 'fast' | 'relate'` is compile-time and defaults to `'auto'`. It selects the kernel for
`intersects`, `contains`, `within` and `dwithin`; every other predicate always uses the relate engine.

- `'fast'` runs a short-circuiting kernel with one invocation per candidate. It is quickest for small features.
- `'relate'` runs the DE-9IM engine with one workgroup per candidate and bounding-box pruning. For `dwithin` it
  runs a workgroup distance kernel instead.
- `'auto'` picks `'relate'` when neither side is points and the product of the two sides' average vertices per
  feature is at least 1024.

Requesting `relate` output always uses the relate engine. Every engine uses the exact orientation sign, so
shared boundaries give the same answers on all three. Measured cost is about 10 ms for 100 pairs of 600 x 175
vertices. A pair with hundreds of shared edges costs 5–10 ms.

### `GPUSpatialJoinPrepared`

A static right-hand side for repeated joins. It builds the right-side bounds and `GPUBVH` once and skips
those nodes on later encodings (CPU-conditioned, so no GPU work) until `invalidate()` is called or
`rebuildWhen(parameters)` returns true.

```ts
const prepared = new GPUSpatialJoinPrepared({geometry: polygons});
graph.add(prepared); // before the joins that use it
graph.add(new GPUSpatialPredicateJoin({left: movingPoints, right: polygons, prepared, ...}));
graph.add(new GPUPointInPolygonJoin({..., prepared, ...}));
prepared.invalidate(); // polygons changed: the next encoding rebuilds, later ones reuse
```

It reuses a build, not a result: every join still evaluates all candidates against the current left and
right geometry each encoding. A right side that changed without `invalidate()` keeps the old tree, so
candidates come from the old boxes and a feature that moved outside its old box is missed. Storage is
persistent buffers, allocated by the handle and freed by `destroy()`, or passed as `storage`. One handle
belongs to one graph, and several joins may share it. `spatialSort` is allowed only for
`GPUPointInPolygonJoin`. `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin` (points, lines and polygons, not
segments) and `GPUSpatialJoinCandidates` accept `prepared`. `encodedBuildCount` counts the rebuilds.

### `GPUSpatialJoinCandidates`

The bounding-box candidate stage on its own: `(left, right)` pairs sorted by `(left, right)` whose bounding
boxes overlap, with the left box optionally expanded by `distance`. The output is a `GPUSpatialJoinPairs`
(capacity `pairs.leftIds.length`, `count`, `overflow`, `totalCount`); slots past `count` hold `0xffffffff`.
It is conservative: no intersecting pair is missed, and some candidates do not intersect. It has no
`excludeSameRow`.


### `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin`

Spatial joins over a `GPUBVH` built from feature bounds every encoding. `GPUPointInPolygonJoin`
refines candidates with the robust `GPUPairwisePointInPolygon` predicate and assigns each point the
containing feature with the smallest row; `GPUNearestFeatureJoin` measures candidates with
`GPUPairwisePointSegmentDistance` and keeps the nearest point or segment feature within a per-frame
radius. Both write per-point feature IDs, per-feature counts, optional compact matches, and one
overflow flag covering BVH leaf, candidate, and match capacity.

A candidate pair whose containment the double-single predicate cannot prove (a point within its error
envelope of an edge that can affect the result) is re-decided by an `exact` node between `classify` and
`resolve`: the shared exact orientation sign with even-odd rings, union polygons and boundary on an
exact zero, so near-degenerate points get the right answer instead of being dropped. Edges strictly
above, below, or to one side of the point are decided exactly from coordinate signs. Only candidates
that no arithmetic can decide stay `uncertain`: non-finite input, malformed offsets, or rings with fewer
than three vertices. Uncertain pairs are never assigned. They are counted in the optional
`uncertainCount`, so such a point stays unassigned visibly rather than silently. The extra node adds no
measurable cost, because its kernel returns at once for candidates that are not `uncertain`.

```ts
graph.add(new GPUNearestFeatureJoin({
  points, features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  radius: snapRadius.importToGraph(graph), candidateCapacity: points.length * 4,
  nearestFeatureIds, nearestDistances, overflow
}));
```

Both joins accept a [`prepared`](#gpuspatialjoinprepared) handle, so a static polygon or line set builds its
`GPUBVH` once. The handle must be added to the graph first, and `spatialSort` and `leafCapacity` then come
from the handle (conflicting props throw).

**k nearest features.** Set `neighborIds` and `neighborCounts` on `GPUNearestFeatureJoin` to get the `k`
nearest features per query (PostGIS `a <-> b ORDER BY ... LIMIT k`, GeoPandas `sjoin_nearest`). The
original mode above is unchanged. Each query walks the BVH nearest child first and prunes boxes farther than
the current `k`-th distance, which is exact and deterministic.

- Queries are `points` or `queries` (points, lines or polygons, not chunked); features are `points`,
  `segments`, `lines` or `polygons` (multipolygons, holes, even-odd). Distance is the minimum planar
  distance between the geometries and 0 when a polygon contains the other geometry. Polygons must be valid.
- `k` is 1 to 32. `ties` is `'lowest-id'` (default) or `'all'`, which keeps every feature tied with the
  `k`-th. `neighborCapacity` (slots per query, `>= k`, `<= 64`) can exceed `k` only with `ties: 'all'`.
  `maxDistance` (an alias of `radius`, per-frame, inclusive; omitted means unbounded) bounds the search.
- Outputs are dense, `query * neighborCapacity + slot`, ordered by distance and then feature row:
  `neighborIds` (remapped by `featureIds`), `neighborCounts`, optional `neighborDistances`,
  `neighborFootPoints` (the nearest point on the feature, or the crossing point when geometries cross)
  and `neighborSegmentIndices` (the start-vertex index of the nearest edge; `GPU_NEAREST_NO_SEGMENT` for
  points and containment). Unused slots hold `GPU_SPATIAL_JOIN_NO_FEATURE`, -1, NaN and
  `GPU_NEAREST_NO_SEGMENT`.
- `overflow` is 1 when `ties: 'all'` had more ties than `neighborCapacity` or the BVH leaf capacity was
  exceeded. Distances are f32, so near-equal distances can order differently from a double reference, and
  tie order and the foot point at exact vertex ties depend on evaluation order.

- `exclusive` skips features whose ID equals the query's ID (`featureIds[row]` or the row, and optional
  `queryIds`, which require `exclusive`), so the `k` neighbors are the nearest other features
  (`STRtree.query_nearest(exclusive=True)`, `sjoin_nearest` self-joins). It compares IDs, not geometry
  equality, which is the same when IDs are unique per geometry. Both modes support it.
- `neighborQueryPoints` (`float32x2`, the layout of `neighborIds`) is the point of the query geometry
  nearest to the feature, NaN in unused slots. `(neighborQueryPoints, neighborFootPoints)` is
  `shapely.shortest_line` or `ops.nearest_points` for every point, line and polygon pair. Ties between
  parallel edges return the first in vertex order, which can differ from GEOS.
- `onAttribute: {left, right}` (as on `GPUSpatialPredicateJoin`) finds the nearest key-equal feature:
  the filter runs inside the traversal, and nearest-feature mode filters candidates in the reduce passes.

```ts
graph.add(new GPUNearestFeatureJoin({
  points, features: {kind: 'lines', positions, lineOffsets},
  k: 3, ties: 'all', neighborCapacity: 8, maxDistance: maxDistanceView,
  neighborIds, neighborCounts, neighborDistances, neighborFootPoints, neighborSegmentIndices,
  overflow, spatialSort: true
}));
```

`spatialSort: true` reorders features along a Morton curve of their bound centers before the
`GPUBVH` build. Results are identical, and traversal is 30x to 75x cheaper when features are not
spatially coherent in row order.

#### `GPUNearestFeatureWeights`

Adapter from the k-nearest output of `GPUNearestFeatureJoin` to a `GPUSpatialWeights` CSR (libpysal 4.15 `KNN`
and `Graph.build_knn`, for points, lines or polygons, using the join's exact geometry distances). Pass the
join's `neighborIds`, `neighborCounts`, optional `neighborDistances` and its `neighborCapacity` as
`slotCapacity`. Per row it keeps the first `k` valid slots in the join's distance order, drops padding and
(with `excludeSelf`, for self-joins run with `k + 1`) the row's own ID, then sorts the IDs ascending. Short rows
stay short. `weightType` is `'binary'` (default) or `'inverse-distance'` (`1 / max(d, distanceFloor)^power`),
and `weights.distances` is filled when present. The output is directed, so for `'symmetrize'` or symmetric
statistics take the union of the weights and their `GPUSpatialWeightsTranspose`. `excludeSelf` compares IDs to
the query row, so do not use `featureIds` with it.

The spec pins libpysal on 24 random points with `k = 4` (identical neighbor sets). Deviations: ties at the
`k`-th distance keep the lowest feature rows (libpysal breaks them in KDTree order, and `KNN` and
`Graph.build_knn` disagree with each other on grids); inverse-distance floors the distance instead of
dividing by zero.

```ts
graph.add(new GPUNearestFeatureWeights({
  neighborIds, neighborCounts, neighborDistances, slotCapacity: 5, k: 4, excludeSelf: true,
  weights: knnWeights, overflow
}));
```

#### Spatial sort default

`spatialSort` of `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin` and `GPUBufferSelection` defaults to `true`
from 256 features and `false` below; pass `false` to turn it off. With shuffled features each encoding is
50–125x faster; with spatially coherent features it is about 1 ms slower. `GPUSpatialJoinPrepared` still
defaults to `false`.

`spatialSort` orders the features along a Hilbert curve (the default) or a Morton curve
(`spatialSortCurve: 'morton'`, a prop of the three joins above, type `SpatialSortCurve`) before the BVH build.
Results are identical; in paired runs the Hilbert joins were 10 to 15 percent faster (point-in-polygon,
nearest, buffer selection). The prepared join and `GPUZoneEvents` have no curve prop and use Hilbert.

### `GPUPairGather`, `GPUOffsetExpansion`, and `GPUBoundsFilter`

Dataframe joints around the joins.

- **`GPUPairGather`** merges join output with attribute columns (GeoPandas `sjoin` and `sjoin_nearest`
  merges, `distance_col`). The input is `pairs` (a `GPUSpatialPredicateJoin` pair output) or `neighbors`
  (the dense `[query, slot]` lists of `GPUNearestFeatureJoin`, flattened by stable compaction).
  `how: 'inner'` emits matched rows, and `'left'` appends the unmatched left rows (`unmatchedLeft`, the
  anti-join output, derived automatically for neighbors) with `rightRows = GPU_PAIR_GATHER_NO_ROW` and
  NaN or `0xffffffff` in the right and slot columns. `leftColumns` and `rightColumns` gather `float32` or
  `uint32` columns by row, and `slotColumns` carry slot-aligned columns such as `neighborDistances`. The
  output order is matched pairs then unmatched, which is not GeoPandas' left order (sort by
  `(left, right)` to compare). It is bounded by `output.leftRows.length`, and `overflow` ORs the input
  flags. Chunked `GraphVectorView` inputs are unsupported.
- **`GPUOffsetExpansion`** gives the owner index and local index of every child row from an offsets view
  (`explode`, `get_coordinates(return_index=True)`, `get_parts(return_index=True)`) by binary search per row.
  Empty ranges yield no rows, `ownerMap` chains levels (vertex to ring to polygon to feature), and the
  output is bounded with `count`, `overflow` and `totalCount`. Offsets monotonicity is not checked.
- **`GPUBoundsFilter`** is a `.cx`-style mask and compaction over `float32x4` feature bounds with a
  per-frame box (`getGPUBoundsFilterParameterValues`). Modes are `'intersects'`, `'within'` and
  `'contains'`, with closed comparisons and non-finite bounds rejected. It is bounds-level only, and OGC
  `envelope.within(box)` rejects bounds that only touch the border where this accepts them, so refine
  with `GPUSpatialPredicateJoin`.

Matched GeoPandas 1.2.0 `sjoin` (inner 11 pairs, left 32 rows) and `sjoin_nearest` with ties and
`distance_col`, and `explode` and `get_coordinates` indices, exactly.

### `GPUBufferSelection`

Selects points within a per-frame planar distance (inclusive) of point or segment features. A
polyline is passed as consecutive segment rows. The contributor composes `GPUNearestFeatureJoin`. Unlike
the join's unordered `matches`, it writes a source-aligned 0/1 mask and stable IDs in ascending row
order, with a clamped count and overflow that covers both the output capacity and the join.
`output.drawInstanceCount` (one `uint32` row, requires `output`) receives the clamped selected count
from the publish node, so it can be bound as an indirect draw instance count.

```ts
graph.add(new GPUBufferSelection({
  points, features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  distance: bufferDistance.importToGraph(graph), candidateCapacity: points.length * 4,
  spatialSort: true, outputMask, output
}));
```

## Geometry validity & intersections

### `GPUSegmentIntersection`

Finds every intersecting pair of line segments between two geometries (`left`, `right`) or within one
(`left` only), like turf `lineIntersect` and `kinks`, self-intersection tests and noding points. Each side
is a `GPUSpatialJoinLines` or `GPUSpatialJoinPolygons` layout (one kind per side). A segment is identified
by the index of its start vertex. Polygon rings close implicitly; rings with fewer than three vertices,
zero-length segments and segments with non-finite coordinates are skipped.

| Prop | Meaning |
| --- | --- |
| `left`, `right?` | Geometries. Omit `right` for self mode (`left < right`, adjacent segments skipped) |
| `sameFeatureOnly?` | Self mode: only same-feature pairs (per-feature self-intersection and ring crossings) |
| `pairs` | `{leftIds, rightIds, count, overflow, totalCount?}`, sorted by `(left, right)`; capacity is `leftIds.length` |
| `kinds?` | `GPU_SEGMENT_INTERSECTION_KIND`: proper 1, touch 2, collinearTouch 3, overlap 4, uncertain 5 |
| `points?`, `endPoints?` | The intersection point; for an overlap, the two ends of the shared span |
| `leftFeatures?`, `rightFeatures?`, `leftRings?`, `rightRings?` | Feature and ring (or line) rows of each segment |
| `uncertainCount?` | Count of uncertain pairs within capacity |
| `spatialSort?` | Morton sort before the BVH build; on by default from 1024 vertices; the output is identical |
| `leafCapacity?` | Power-of-two BVH leaves, by default the next power of two of the right vertex count |

`proper` means the interiors cross at one point. `touch` is one shared point where an endpoint of one lies
on the other (non-parallel), `collinearTouch` is collinear with one shared endpoint, and `overlap` is
collinear sharing positive length. The pipeline is a segment table, a `GPUBVH` over the right segments, a
counted and scanned probe, exact classification and a bounded sorted output; nothing is read back.
On overflow the output keeps a sorted prefix (`count = min(total, capacity)`, `overflow = 1`, `totalCount`
unclamped), and the columns beyond `count` are unspecified. `uncertainCount` and the classification
columns cover pairs within capacity only. A collinear chain of `n` segments lists all `n^2` overlap pairs.

```ts
graph.add(new GPUSegmentIntersection({
  left: {kind: 'lines', positions, lineOffsets},
  pairs: {leftIds, rightIds, count, overflow},
  kinds, points
}));
```

### `GPUGeometryValidity`

A per-feature validity bitmask for polygons and multipolygons (`GPUSpatialJoinPolygons` layout), like
PostGIS `ST_IsValidDetail` and Shapely `explain_validity` as a filterable mask. Repair stays on the CPU.

The bits of `GPU_GEOMETRY_VALIDITY_BIT` are `nonFinite` 1, `unclosedRing` 2 (explicit closure only),
`shortRing` 4 (fewer than 3 distinct vertices), `repeatedVertex` 8, `selfIntersection` 16, `crossingRings`
32 (a proper crossing or overlap between different rings of one feature; point touches are allowed),
`holeOutsideShell` 64, `badOrientation` 128 and `uncertain` 256. `GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK` is
every bit except `badOrientation`.

| Prop | Meaning |
| --- | --- |
| `polygons`, `mask` | The input and the `featureCount` output rows |
| `intersectionCapacity` | Capacity of the internal same-feature intersection list; valid data produces almost none |
| `overflow`, `intersectionCount?` | The overflow flag (the self-intersection and crossing bits may then be incomplete) and the unclamped count |
| `ringClosure?` | `'implicit'` (default) or `'explicit'`, which sets `unclosedRing` |
| `orientation?` | `'counter-clockwise-shell'` (default), `'clockwise-shell'` or `'ignore'` |

`lines` (`{kind: 'lines', positions, lineOffsets}`) and `points` replace `polygons` for the other kinds
(Shapely `is_valid`, `is_valid_reason` on lines and points). Lines report `nonFinite` and `tooFewPoints`,
points report `nonFinite`; neither needs an intersection list.

Not checked: interior connectivity (holes touching at two points), nested holes, nested polygons of a
multipolygon, and rings that leave the shell only through boundary vertices. Repeated vertices are reported
although OGC accepts them. Hole containment tests one off-boundary vertex per hole.

```ts
graph.add(new GPUGeometryValidity({polygons, mask, overflow, intersectionCapacity: 1024}));
// on the CPU: (mask[i] & GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK) === 0 means structurally valid
```

### `GPUGeometryPredicates`

Per-feature columns for points, linestrings or polygons (Shapely `is_simple`, `is_ring`, `is_closed`,
`is_ccw`, `get_num_points`, `count_coordinates`, `get_num_interior_rings`, `get_num_geometries`,
`extract_unique_points`, `equals_exact`, `equals_identical`). Each output is optional and `uint32` (0 or 1
for booleans): `isSimple`, `isRing`, `isClosed`, `isCcw`, `numPoints`, `numCoordinates`,
`numInteriorRings` and `numGeometries`.

- `isSimple` and `isRing` use `GPUSegmentIntersection` in self mode, so they need `intersectionCapacity`
  and `overflow`; features after an overflow can be wrong. A polygon is simple when no ring meets itself
  (Shapely 2.1). `isCcw` follows the GEOS `Orientation::isCCW` rule (the JTS 1.19 flat-top algorithm) for
  linestrings and is 0 for other kinds, as in Shapely.
- `other` (same kind and feature count) enables `equalsExact` (per-frame `tolerance` through
  `getGPUGeometryPredicatesParameterValues`) and `equalsIdentical`, which compare feature `i` with
  feature `i`. The distance is f32.
- `uniquePositions` and `uniqueOffsets` give `extract_unique_points`: first-occurrence order, as Shapely,
  with `-0` equal to `0`.
- Limits: one thread per feature in the row-wise kernels (a huge single feature is serial); the
  self-mode look-ahead is 1024; an empty polygon gives 0 geometries where Shapely gives 1. Open:
  `equals_exact(normalize=True)`, multilinestring and multipoint layouts.

### `GPUMinimumClearance`

Per-feature `shapely.minimum_clearance` and `minimum_clearance_line` for linestrings and
(multi)polygons: the smallest distance one vertex could move to make the feature invalid, as the minimum
over vertices of the distance to every other vertex and to every segment that does not have that vertex as
an endpoint (GEOS rule, repeated points ignored). Props: `geometry` (lines or polygons), and at least one
output of `clearances` (`float32`, `Infinity` when none), `lines` (`float32x4`, vertex and nearest point,
NaN when none) and `vertexIds` (`uint32` row of the fragile vertex, `0xffffffff` when none). A segment
table and BVH feed one branch-and-bound probe per vertex, restricted to its own feature, then two integer
`atomicMin` passes (distance, then lowest vertex row), so results are deterministic. Parity with Shapely
2.1.2 is within 2e-5 relative on 52 multipolygons and 69 linestrings; line end points can come out in the
opposite order and tied pairs can differ from GEOS (the distance never does). Coordinates are
origin-relative f32, so tiled copies shifted by thousands of units agree to about 4e-4. Not provided:
multipoints.

### `GPUCoverageValidity`

Polygon coverage validity (Shapely `coverage_is_valid`, `coverage_invalid_edges(gap_width)`). One BVH over
all segments writes a per-segment flag word (`unmatched`, `crossing`, `gap`, `overlap`, `uncertain` bits),
optional per-polygon counts, a total and `isValid`. `gapWidth` is per-frame through
`getGPUCoverageValidityParameterValues`. The output has one row per vertex and cannot overflow. Rules found
by probing GEOS: matched edges are skipped and flagged only when both interiors are on the same side, a
T-junction touch counts, and a gap is endpoints within `gapWidth` plus a mutual projection overlap larger
than `gapWidth` (strict). It agreed with Shapely on 28 scenes with 40 gap cases and on 158 of 160 random
jitter cases. The misses are a vertex of an overlapping neighbor within `gapWidth` of an edge, which GEOS
flags and this does not, and a matched edge lying inside a third polygon, which is not flagged (that
polygon's own segments are).

## Density & interpolation

### `GPUPointDensity`

Bins points into a square grid (`GPUGridBinning`, `GPUGridAggregation`) or pointy-top hexagons
(keys kernel plus `GPUGroupAggregation`) and writes per-cell counts, sums, means, a heatmap field,
its `[min, max]` extent, a histogram, and an optional `r32float` texture. Bounds and hexagon radius
may be GPU views; the field can be smoothed with a per-frame `GPUConvolution` kernel. Instead of a full
`kernel`, pass a separable `separableKernel: {horizontal, vertical}` (one of the two is required), which smooths
in two 1D passes.

Weighted sums default to `sumAccumulation: 'workgroup'`: each 256-row workgroup pre-aggregates in
shared memory before writing to the grid, so a hotspot no longer serializes on one cell (90% of 1M
points in one cell drops from seconds to about 4 ms, at about 2 ms extra on uniform data).
`'atomic'` keeps the direct float-atomic path. Either way, sums can differ from a CPU sum in the
last bits.

```ts
graph.add(new GPUPointDensity({
  positions, weights, bounds: viewport.importToGraph(graph), gridSize: [256, 256],
  statistic: 'mean',
  smoothing: {kernel: kernel.importToGraph(graph), kernelWidth: 5, kernelHeight: 5},
  output: {values, extent, histogram, texture}
}));
```

`mask` is an optional per-row `uint32` view with the same length as `positions`. Rows whose mask
is `0` are excluded from count, sum, mean, smoothing input, extent and histogram; any nonzero value
includes the row, like the mask inputs of `GPURegionStatistics` and `GPUFlowAggregation`. Contents
are per-frame and rewriting them never recompiles. Hexagon binning reads the mask in its keys
kernel; grid binning adds one pass that writes masked positions as NaN into a transient copy.
Without `mask` the graph is unchanged. Pass a residency arena `liveMask`, or
`GPUTimeWindowFilter.outputMask`, to get time- or tile-gated density without poisoning position
columns.

`getGPUPointDensityHexagonGridSize` sizes a hexagon lattice that covers given bounds, and
`GPU_POINT_DENSITY_HEXAGON_WGSL` lets render shaders place hexagon bins.

### `GPUInverseDistanceWeighting`

Interpolates scattered samples onto a raster with inverse distance weighting (`w = 1 / d^p`),
the GPU counterpart of turf `interpolate`, GDAL `invdist`, and GRASS `v.surf.idw`. The output
extent, search radius, power, nearest-neighbor limit, and minimum neighbor count are per-frame.

```ts
graph.add(
  new GPUInverseDistanceWeighting({
    positions, // float32x2 per sample
    values, // float32 per sample, NaN skips
    mask, // optional uint32, 0 skips
    parameters: idwParameters.importToGraph(graph), // 8 float32
    width: 512,
    height: 512,
    indexGridSize: [128, 128], // internal GPUGridIndex over the samples
    indexBounds: [minX, minY, maxX, maxY], // samples outside are ignored
    maximumNeighborCount: 16, // compile-time capacity for k; 0 compiles radius-only
    output: {values: surface, counts} // float32 (NaN = nodata), optional uint32
  })
);
idwParameters.write(
  getGPUInverseDistanceWeightingParameterValues({
    extent: [x0, y0, x1, y1],
    searchRadius: 250,
    power: 2,
    neighborCount: 8, // 0 = every sample within the radius
    minimumNeighborCount: 3
  })
); // no recompile
```

- Gather form: one invocation per output cell samples the cell center
  `extentMin + (x + 0.5, y + 0.5) * cellSize` (row 0 at `minY`) and visits the grid-index rows
  that overlap the search square. No atomics touch values.
- Samples with `d^2 <= radius^2` contribute. With `neighborCount = k > 0` the `k` nearest are kept
  by `(d^2, row)`, so equal distances keep the smallest row. `k` is clamped to
  `maximumNeighborCount` (at most 64).
- An exact hit (`d == 0`) returns that sample's value, the smallest row among several, regardless
  of `minimumNeighborCount`. Otherwise a cell with fewer than `max(minimumNeighborCount, 1)`
  contributors is nodata (NaN). `counts` holds the contributors after the `k` limit.
- Negative or NaN radius, or a negative or non-finite power, produces an all-NaN raster.
  `searchRadius: Infinity` searches the whole index domain.
- Determinism: the grid index's in-cell order depends on atomics, so the contributor re-sorts IDs inside
  each index cell every encoding; sums then run in a fixed order (index row, column, ascending
  row; or `(d^2, row)` with `k`). Weights are accumulated in log space against a running maximum,
  so tiny distances with large powers do not overflow f32.
- The index is rebuilt every encoding, so positions and values may change per frame. Keep a few
  samples per index cell: the in-cell sort is quadratic in cell population.
- `output.values` is a row-major float32 raster that `GPUTerrainContours` accepts directly as
  `elevation: {id, format: 'float32', storage: {kind: 'buffer', values: surface}}`; NaN cells
  clear their marching-squares cells.

### `GPUKriging`

Local ordinary kriging on a raster (PyKrige `OrdinaryKriging` with `n_closest_points`, GSTAT, ArcGIS Kriging).
For every output cell the `k <= 16` nearest samples within the radius form a `(k + 1) x (k + 1)` system
under a fitted variogram, solved per invocation by pivoted Gaussian elimination in f32 (the variogram is
normalized by its total sill and the prediction is relative to the nearest value). The samples come from
the same internal `GPUGridIndex` as `GPUInverseDistanceWeighting`.

- Props: `positions`, `values` (NaN skips), optional `mask`, `parameters` (`getGPUKrigingParameterValues`,
  at least 12 float32), `width`, `height`, `indexGridSize`, `indexBounds`, `maximumNeighborCount` (1 to 16,
  compile-time) and `output: {values, variance?}`.
- Per frame: extent, radius, `k`, minimum neighbors, model (`'spherical'`, `'exponential'` or `'gaussian'`,
  the `fitVariogramModel` convention) and nugget, sill and range. Pass `sill + nugget` as PyKrige's "sill"
  when comparing, since PyKrige's value is the total sill.
- `values` is a row-major raster and nodata (too few neighbors, a singular system, duplicate sample
  positions in a neighborhood) is NaN. `variance` is the kriging variance, 0 at exact hits, NaN at nodata.
- Pinned to PyKrige 1.7.3 with 8 closest points (exponential, spherical and Gaussian within 0.2 on values
  near 70, and 1 percent on variance). Kernels bind at most 8 storage buffers. Global and universal
  kriging, anisotropy and cross-validation are not provided, and `radius: Infinity` scans the whole index
  per cell.

### `GPUDotDensity` and `GPURandomPointsInPolygon`

Dasymetric dot-density maps: `value * dotsPerUnit` random dots inside each polygon, per category,
optionally concentrated by a weight raster such as built-up area. Changing the dot value with zoom
only adds or removes dots; existing dots never move, so there is no flicker.
`GPURandomPointsInPolygon` places an integer count of uniform random points per feature.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'dots', format: 'uint32', length: 8});
graph.add(
  new GPUDotDensity({
    polygonPositions, featureOffsets, polygonOffsets, ringOffsets, // GeoArrow, as GPUPointInPolygonJoin
    values, // float32, featureCount * categoryCount rows, feature-major
    categoryCount: 3,
    parameters: parameters.importToGraph(graph),
    mask: {weights, width: 512, height: 512}, // optional, weights in [0, 1]
    output: {positions, dots: {ids, count, overflow, totalCount}, categories, failedCount}
  })
);
parameters.write(
  getGPUDotDensityParameterValues({seed: 1, dotsPerUnit: 1 / peoplePerDot, maskExtent: [x0, y0, cw, ch]})
);
```

- A slot (feature, category) draws `ceil(value * dotsPerUnit - u)` dots with a per-slot Philox
  uniform `u`: `floor(x)` dots plus one more with probability `frac(x)`, monotone in `dotsPerUnit`.
- Dot `j` of a slot is rejection-sampled in the feature's bounding box from Philox candidates keyed by
  `(seed, slot, j, attempt)`, independent of the dot value and of other slots, so coarser dot values
  give a stable prefix of the finer dots. With a mask a candidate is also kept with probability equal
  to its cell weight.
- Dots are ordered by feature, category and rank (offsets from a prefix scan). `dots.ids` holds the
  feature row of each dot and `dots.count` is safe as an instance count; rows past it are NaN /
  `0xffffffff`. A dot whose `maximumAttempts` candidates all miss keeps its slot with a NaN position
  and is counted in `failedCount`.
- Containment is an even-odd test over all rings of the feature (holes excluded); this equals
  multipolygon containment for valid geometry. Cost per candidate is linear in the feature's vertices.

**`GPURandomPointsOnLine`** places `counts` points per line uniformly in arc length (`GeoSeries.sample_points`
on lines). The Philox key is `(line, rank, 0, 0)` with key `(seed, 3)`, so the prefix of points is stable
as counts grow. Outputs are a bounded `positions` buffer and a compact `points` record (`ids` = line row,
`count`, `overflow`, `totalCount`), plus optional `fractions`. Parameters reuse
`getGPUDotDensityParameterValues` (`seed`). Points matched `LineString.interpolate` at the pinned Philox
fractions to 2e-5 relative. Not provided: `cluster_poisson` sampling.

## Cells (DGGS)

### `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid`, and `GPUCellLevelSelection`

Aggregates rows into discrete global grid cells (Quadbin or H3) and builds a zoom pyramid of
cell tables. 64-bit cell keys are stored as two `uint32` words in little-endian `(low, high)`
order, which is the layout of Arrow `Uint64` columns and of `GPUH3CellProjection` input. An H3
table therefore feeds the projection directly.

- **Inputs.** `positions` (lng/lat degrees, Quadbin only) are keyed in WGSL. `cells` are
  pre-keyed rows, for example CARTO Quadbin or H3 columns, as `uint32x2` (`wordOrder:
  'high-low'` is also accepted). Pre-keyed rows finer than `resolution` are truncated to it.
  Invalid rows, and rows coarser than `resolution`, are skipped. H3 validity comes from
  gpu-dggs `dggs_h3_is_valid_cell_id`. Optional `values` and a `mask` can change every frame.
  Rows with a NaN coordinate or a non-finite value are skipped.
- **Cell table** (`GPUCellTable`). Capacity is `cells.length`. Rows `[0, count)` hold occupied
  cells in ascending key order. Later rows hold the empty key `0xffffffff` in both words, count 0
  and NaN extremes. Columns: `counts`, exact fixed-point `sums` (signed 64-bit words of
  `roundHalfEven(fround(value * sumScale))`, default scale 65536), `sumValues` (f32
  `sums / sumScale`), `minimums` and `maximums`. On overflow the table keeps the smallest keys
  and sets `overflow`.
- **Roll-up.** `GPUCellRollup` reduces a table to a coarser resolution without re-reading rows
  or sorting. Parent keys truncate the cell path (`cellToParent`), which preserves key order.
  Counts and sums are integer sums, so a roll-up equals direct aggregation bit for bit. The
  source table's overflow carries through.
- **Pyramid.** `GPUCellPyramid` aggregates once at the finest of `levels` (listed finest first),
  rolls each level into the next, and can write per-level counts to `levelCounts`. Intermediate
  sums and extremes that a coarser level needs become transients.
- **Level selection.** `GPUCellLevelSelection` reads a per-frame `activeLevel` (one `uint32`) and
  writes that level's count, first row, and the `instanceCount`/`firstInstance` words of indirect
  draw arguments. Keep it in a small per-frame graph: switching levels rewrites one word and never
  re-runs the pyramid.

```ts
graph.add(new GPUCellPyramid({
  family: 'quadbin',
  positions, values, mask,                    // mask and values can change every frame
  levels: [18, 14, 10, 6, 2].map((resolution, i) => ({resolution, output: tables[i]})),
  levelCounts
}));
selectionGraph.add(new GPUCellLevelSelection({
  levelCounts, activeLevel: activeLevel.importToGraph(selectionGraph),
  output: {count, drawArguments}
}));
activeLevel.write(Uint32Array.of(levelForZoom(zoom)));   // no recompile
```

- **Determinism.** Point keys use only integer arithmetic. The longitude column is exact. The
  latitude row is a fixed-point Mercator (`ln(cos t) - ln(sin t)`, Q0.32 polynomials, exact long
  division), so every adapter and the CPU reference agree bit for bit. f32 polynomials would not
  agree, because WebGPU compilers may fuse `a * b + c` (Metal does). Sorting uses one stable
  `GPUSort` when the key path fits 31 bits (Quadbin ≤ 15, H3 ≤ 8) and two otherwise. Rows
  accumulate with u32 atomics only: counts, 64-bit sums as two words with a carry, and extremes
  as order-preserving keys.
- **Accuracy.** Compared with the f64 quadbin-py formula, the fixed-point row differs for 0% of
  random points up to resolution 8, 0.01% at 12, 0.1% at 16, 1.1% at 20, 17% at 24 and 55% at
  26. It is never off by more than 1 row up to resolution 22, 2 rows at 24 and 5 rows at 26
  (f32 latitudes are not that precise). The column always matches.
- Inputs must be single packed views. Forward keying of points is Quadbin only.

### `GPUPointToCell` and `GPUCellGeometry`

Keys points to discrete global grid cells and turns cells back into centres and boundary
polygons. This covers the most common Studio/CARTO preprocessing step ("lng/lat column to H3")
and the geometry needed to draw any DGGS cell column. Keys are 64-bit `uint32x2` little-endian
`(low, high)` words, which is the Arrow `Uint64` layout that `GPUCellAggregation` pre-keyed rows
and `GPUH3CellProjection` read. The zero key means "no cell".

- **Families** (`GPUCellIndexFamily`) and their resolutions:

  | Family | Resolutions | Notes |
  | --- | --- | --- |
  | `'quadbin'` | 0-26 | integer keys, exact |
  | `'h3'` | 0-15 | f32 input: exact to res 4 |
  | `'quadkey'` | 1-29 | packed: length in bits 58-63, base-4 digits |
  | `'geohash'` | 1-12 | packed: length in bits 60-63, base-32 characters, i.e. the lng/lat bit interleave |
  | `'s2'` | levels 0-30 | standard S2CellId |

  The quadkey and geohash packings are the layouts the shadertools `dggs` decoders read.
- **`GPUPointToCell`.**
  - Input is `positions` (lng/lat `float32x2`) and an optional `mask`. Output is `cells` and an
    optional `validity`, one row per input row.
  - Masked and non-finite rows get the zero key. The resolution is compile-time.
  - Rewriting positions or the mask re-encodes without recompiling.
- **`GPUCellGeometry`.**
  - Input is `cells` (`wordOrder` option).
  - Outputs are `centers` (lng/lat) and optional `boundaries` with a fixed `maximumVertexCount`
    stride and `vertexCounts`. Families are the five above plus `'a5'`.
  - H3 boundaries follow h3-js `cellToBoundary` exactly in vertex count and order. That includes
    face-crossing distortion vertices (up to 10, so use stride 10) and pentagons.
  - Quadbin, quadkey, geohash and S2 have 4 vertices; A5 has 5.

```ts
graph.add(new GPUPointToCell({family: 'h3', resolution: 8, positions, output: {cells}}));
graph.add(new GPUCellGeometry({
  family: 'h3', cells, maximumVertexCount: GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  output: {centers, boundaries, vertexCounts}
}));
```

- **Exactness and accuracy.**
  - Quadbin, quadkey and geohash keying is integer-only and equals the BigInt oracles bit for bit
    on every device.
  - S2 and H3 use f32 sphere math (polynomial sine/cosine, no trig builtins for H3) and are exact
    at coarse resolutions. A mismatch is an edge neighbour of the f64 answer:

    | Family | Uniform-point mismatch rate vs the f64 reference (f32 inputs) | Usable limit |
    | --- | --- | --- |
    | H3 vs h3-js | 0 at res 0-4, 1e-4-4e-4 at 5-8, 1e-3 at 9, 2e-3 at 10, 1.8% at 12, 28% at 15 | res ≤ 12 |
    | S2 | 0 at levels 0-10, 1e-4-3e-3 at 11-15, 1.8% at 18, 13% at 21 | level ≤ 20 |

  - Geometry matches h3-js / oracles within about 3e-5 degrees.

### `GPUCellTopology` and `GPUCellCompaction`

Cell topology for Quadbin and H3 keys stored as `uint32x2` little-endian `(low, high)` words.
This is the Arrow `Uint64` layout used by `GPUCellAggregation`. Inputs also accept
`wordOrder: 'high-low'`. A zero key means "no cell" in every output.

- **`GPUCellTopology`** writes a fixed fan-out row per input cell, with stride from
  `getCellTopologyStride(family, operation)`:
  - `{type: 'disk', k}`: `gridDisk` / `gridDiskDistances`, sorted by (distance, key). Optional
    `distances` column.
  - `{type: 'ring', k}`: `gridRing`, sorted by key.
  - `{type: 'parent', resolution}`: `cellToParent`.
  - `{type: 'children', resolution, inputResolution}`: `cellToChildren`, in ascending key order.
  - Rows are zero padded and an optional `counts` column gives the valid entries per row. `k` is at
    most 8 and the children stride at most 4096 (H3 depth 4, Quadbin depth 6). Invalid, masked
    or wrong-resolution rows give zero rows.
- **H3 neighbors are exact.** `H3_NEIGHBOR_WGSL` ports H3's `h3NeighborRotations`: base-cell
  neighbor and rotation tables, class II/III digit carry tables and pentagon clockwise-offset
  faces. Disks and rings are a bounded breadth-first search per thread over the six neighbor
  steps. The search is integer-only and exact across icosahedron faces and pentagons. A pentagon's
  disk has fewer cells, and the rest of the row is zero padded.
- **Quadbin neighborhoods** use Chebyshev distance on tile column and row (`QUADBIN_KRING`
  style). Columns wrap across the antimeridian. Rows are clipped at the poles. A tile that the wrap
  reaches twice at small zoom levels appears once.
- **`GPUCellCompaction`** handles sets of cells:
  - `{type: 'compact', minimumResolution?}` (`compactCells`): input is cells at one resolution,
    which are sorted unless `sorted: true` and then deduplicated. It replaces every complete
    sibling set by its parent, recursively: 4 children for Quadbin, 7 for an H3 hexagon, and for an
    H3 pentagon 6 at depth 1, `(5 * 7^d + 1) / 6` in general.
  - `{type: 'uncompact', resolution, maximumDepth?}` (`uncompactCells`): count, scan, then emit
    the descendants at `resolution`.
  - Both write a capacity-bounded `{cells, count, overflow, totalCount?, droppedCount?}`.
    Compact output is in ascending canonical key order. Uncompact output is ascending when the
    input is in path order.

```ts
graph.add(new GPUCellTopology({
  family: 'h3', operation: {type: 'disk', k: 2},
  cells, output: {cells: diskCells, distances, counts}   // diskCells.length = rows * 19
}));
graph.add(new GPUCellCompaction({
  family: 'h3', operation: {type: 'compact'},
  cells: polyfillCells, output: {cells: compacted, count, overflow}
}));
```

- **Determinism.** All operations are integer-only, with fixed sort orders. GPU results equal the
  CPU oracles (h3-js and BigInt Quadbin) bit for bit, including padding.

### `GPUCellGridPath` and `GPUCellMeasures`

H3 gap kernels (`family: 'h3'`), integer-exact where possible.

- **`GPUCellGridPath`** (h3 `grid_distance`, `grid_path_cells`, `are_neighbor_cells`): per pair,
  `origins` and `destinations` (`uint32x2`) give `output.distances` (the H3 `gridDistance`, or
  `GPU_CELL_GRID_DISTANCE_UNDEFINED` = `0xffffffff` where H3 fails: invalid input, resolution mismatch,
  non-adjacent base cells, behind a pentagon's deleted sector, masked), `output.cells` (`gridPathCells`,
  stride `maximumPathLength` up to 256, zero padded) and `output.counts` (the true length `distance + 1`;
  above `maximumPathLength` means overflow, 0 means no path). Omit `cells` for distance only;
  `are_neighbor_cells` is `distance == 1`. It is `cellToLocalIjk` with pentagon rotations and an
  integer cube-round. Known deviations: about 0.2% of random pairs differ from H3 by one adjacent cell at
  exact rounding ties (H3 rounds with double noise), and paths crossing pentagon base cells fail
  (`counts = 0`) in about 6% of those cases. Distances matched h3 4.5.0 on 355 of 355 pairs, failures
  included.
- **`GPUCellMeasures`** (h3 `cell_area`, `edge_length`, `is_pentagon`): `areas` (km2), `perimeters`,
  `edgeLengths` (stride `maximumEdgeCount` 6 to 10, great-circle km between consecutive
  `cellToBoundary` vertices), `edgeCounts` and `pentagons`. Single-face hexagons are exact from lattice
  offsets, within 1e-6 relative at resolutions 0 to 15. Pentagons and cells straddling a face edge use f32
  boundary vectors: 6e-7 at resolution 0, 2e-5 at 2, 2.5e-4 at 4 and about 1e-2 at 6, so treat them as
  unreliable beyond resolution 5. Open: directed edges and vertexes.

### `GPUCellCover`

Polyfills polygons (with holes) into Quadbin or H3 cells, like CARTO `QUADBIN_POLYFILL` /
`H3_POLYFILL` and h3-js `polygonToCells`. The result is `(feature id, cell)` pairs that feed
`GPUCellAggregation` pre-keyed rows, `GPUKeyJoin` and `GPUCellCompaction`.

- **Inputs.** Polygon features use the `GPUPointInPolygonJoin` layout: `polygonPositions`
  (lng/lat `float32x2`), `featureOffsets`, `polygonOffsets`, `ringOffsets`, and optional
  `featureIds`. Geometry is planar in lng/lat degrees, as in h3-js and turf. Containment is
  even-odd over all rings. Polygons must not cross the antimeridian.
- **Containment.**
  - `'center'`: the cell centre is inside the polygon.
  - `'full'`: the cell lies inside the polygon.
  - `'intersects'`: the cell interior meets the polygon.
  - Quadbin supports all three modes. A Quadbin tile is an axis-aligned lng/lat rectangle, so
    `full` and `intersects` are exact edge-versus-open-rectangle tests. H3 supports `'center'`
    only.
- **Candidates.** Quadbin tests the integer-exact tile range of each feature's bounding box. H3
  lays a lng/lat lattice over the bounding box, with spacing below half the minimum H3 inradius at
  that resolution (measured from h3-js; the minimum sits next to a pentagon). Each lattice point
  indexes its cell with `cellIndexH3FromLngLat`. The cell is emitted only from the lattice point
  nearest its centre, so every cell whose centre lies in the polygon is emitted exactly once with
  no sort.
- **Output.** `{featureIds, cells (uint32x2 little-endian), count, overflow, totalCount?}` with the
  usual capacity-bounded semantics. Order is by feature, then candidate order. The pipeline is
  count, scan, test, scan, write, with no atomic append, so the output is deterministic.
  `candidateCapacity` bounds the candidates per graph, and exceeding it raises `overflow`.

```ts
graph.add(new GPUCellCover({
  family: 'h3', resolution: 8, containment: 'center',
  polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
  candidateCapacity: 1 << 22,
  output: {featureIds, cells, count, overflow}
}));
```

- **Accuracy.**
  - Quadbin results equal an f32-faithful CPU oracle at resolutions 3-10 in all modes. Compared
    with an independent f64 centre test there were 0 disagreements.
  - H3 `center` matches h3-js `polygonToCells` exactly at resolutions 3-7. At 8-10 it differs on
    12 of about 75,000 cells. Each of those cells has its centre within 5e-5 degrees of a polygon
    edge, where the f32 centre differs from the f64 one.
  - An f64 simulation of the lattice rule matches h3-js exactly, so the spacing never misses a
    cell.

`output.core` is an optional `uint32` column parallel to `output.featureIds` (the same length): 1 for a core
cell (a Mosaic chip, provably inside the feature) and 0 for a border cell. It works for every containment
and both families. It is conservative: a cell is core only when its center is inside and no polygon edge
enters the cell (for H3, its boundary bounding box) grown by 2e-4 degrees plus 0.1% of the cell size.
Border cells can still lie entirely inside. A join can skip the exact point-in-polygon test for points whose
cell is core.

### `GPUCellSetOutline`

The boundary edges of a set of H3 or Quadbin cells, the `cellsToMultiPolygon` outline. Set `rings` to chain
the edges into closed rings with shells and holes on the GPU ([`GPUSegmentRingAssembly`](#gpusegmentringassembly)).

```ts
graph.add(new GPUCellSetOutline({
  family: 'h3',
  cells, // uint32x2 little-endian keys, sorted ascending and distinct, e.g. a GPUCellAggregation table
  count: tableCount, // optional one-row valid row count
  groups, // optional uint32 label per row: also emit edges between different groups
  output: {rows, cells: segmentCells, edgeIndices, endpoints, groups: segmentGroups, count, overflow, totalCount},
  rings: {normalizeWinding: true, output: ringOutput} // optional, see below
}));
```

- The output columns share one capacity: `rows` (the input row, also the compact ID column), `cells`,
  `edgeIndices`, `endpoints` (`float32x4`: lng0, lat0, lng1, lat1), optional `groups`, `count` (clamped),
  `overflow` and optional `totalCount`.
- An edge is emitted when the neighbor across it is absent, or has a different group label. With `groups`,
  a border between two groups is emitted from both sides, tagged with each side's group.
- Order is by `(input row, edge index)`: classify, inclusive scan, write, with no atomics, so it is
  deterministic.
- H3: edge `e` is the `e`-th edge of `cellToBoundary` (up to 10 slots with distortion vertices, so such an
  edge appears as two segments). The neighbor across it is the H3 neighbor whose center is nearest the
  edge midpoint, a geometric rule validated against h3-js on hexagons, a pentagon disk and distortion
  cells. Quadbin: 0 north, 1 east, 2 south, 3 west; columns wrap, and rows beyond the poles are absent, so
  the map edge is an outline edge.
- Membership is a binary search, so the input must be sorted and distinct. Invalid keys never emit.
- Endpoints are f32. H3 cells that straddle the antimeridian keep their `cellToBoundary` longitudes, so a
  segment can span more than 180 degrees. Other families (quadkey, geohash, S2, A5) are not supported.
- Segments are directed with the cell interior on a fixed side: `outline.interiorSide` is `'left'` for H3
  (counter-clockwise shells) and `'right'` for Quadbin (clockwise shells).
- `rings: {vertexTolerance?, normalizeWinding?, output: GPUSegmentRingAssemblyOutput}` composes
  `GPUSegmentRingAssembly` after the outline, with `output.endpoints`, `output.count` and `output.groups`
  as its inputs and `interiorSide` taken from the family. Use `normalizeWinding: true` for
  counter-clockwise shells and clockwise holes in both families. Without `rings`, hole detection and ring
  orientation are left to the caller.

### `GPUSegmentRingAssembly`

Chains directed boundary segments into closed rings on the GPU, the ring-assembly step of
`cellsToMultiPolygon`. It consumes `GPUCellSetOutline` output, or any `float32x4` segment list whose
vertices meet within a tolerance.

```ts
graph.add(new GPUSegmentRingAssembly({
  endpoints, // float32x4 (x0, y0, x1, y1), e.g. GPUCellSetOutline output.endpoints
  count, // optional one-row valid segment count
  groups, // optional uint32 per segment: rings never mix groups
  vertexTolerance: 5e-5, // compile-time Chebyshev matching distance, input units
  interiorSide: 'left', // side of each segment that is filled ('left': counter-clockwise shells)
  normalizeWinding: true, // shells counter-clockwise, holes clockwise (RFC 7946)
  splitTouchingRings: true, // default
  geographic: true, // cos(lat) in turn comparisons; false for planar coordinates
  output: {
    ringOffsets, // uint32, ringCapacity + 1
    positions, // float32x2, vertex capacity
    ringAreas, ringIsHole, ringShells, ringGroups, // optional, ringCapacity rows
    segmentRings, segmentVertices, segmentFlags, // optional, one row per input segment
    polygons: {positions, ringOffsets, polygonOffsets, featureOffsets}, // optional
    count, overflow, totalCount, openSegmentCount, touchingSegmentCount
  }
}));
```

- **Layout.** GeoArrow style: ring `r` is `positions[ringOffsets[r] .. ringOffsets[r + 1])`, closed (the last
  vertex repeats the first). Rings are ordered by their lowest input segment index; a ring starts at that
  segment and follows its direction (or is reversed by `normalizeWinding`). Offsets past `count` repeat the
  final offset.
- **Matching.** Each segment end is matched to segments that start within `vertexTolerance` in the same
  group, through a sort of quantized vertex hashes and a 3 x 3 neighborhood lookup. Quadbin vertices are
  bit-exact; H3 vertices of neighboring cells agree within f32 round-off (below 5e-5 degrees). Pick a
  tolerance of at least four times the f32 spacing of the coordinates and well below the shortest segment.
- **Shells and holes.** The signed area (shoelace, relative to the first vertex) is compared with
  `interiorSide`: with `'left'` shells are counter-clockwise. `ringIsHole` flags holes. `ringShells`
  assigns every hole to its innermost enclosing shell (smallest area, lowest index on ties, same group), a
  shell to itself, and a hole without a shell to `GPU_SEGMENT_RING_ASSEMBLY_NONE`. This costs one pass over
  every written vertex per hole.
- **Touching vertices.** When more than one segment starts at a segment's end vertex (Quadbin cells that
  meet at a corner; H3 hexagons never do), the continuation is the outgoing segment that turns tightest
  toward the interior, so regions that touch at a point stay separate rings. With `splitTouchingRings` a
  second tracing pass swaps the continuations of segment pairs that end at one vertex and lie on one ring,
  so two holes meeting at a corner, or a hole pinched to its shell, become simple rings. Choices taken
  twice resolve to the lowest segment index. `touchingSegmentCount` and `segmentFlags`
  (`GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING` 1, `_DANGLING` 2, `_CONFLICT` 4) expose the ambiguity.
- **Open chains** (dangling ends, tolerance misses, truncated input) emit no ring and are counted in
  `openSegmentCount`.
- **Cancelling pairs.** `cancelOpposingSegments` removes exact `a -> b` and `b -> a` pairs within a group
  before tracing (they are flagged `GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED`, 8, and not counted as open).
  `GPUIsobandRings` uses it for the edges shared by two cells. `output.polygons.polygonGroups` gives the
  group of each polygon's shell.
- **H3 pentagons.** Rings of three mid-latitude pentagons (resolutions 2, 3 and 5, alone, in disks, as holes
  and open rings) equal h3-js `cellsToMultiPolygon`. The two polar pentagons enclose a pole, so their rings
  have no planar lng/lat orientation.
- **Bounded output.** Rings are written whole, as a prefix: `count` is the number written, `overflow` is 1
  when ring or vertex capacity dropped rings, and `totalCount` is the full ring count.
- **Polygon layout.** `output.polygons` regroups the rings as GeoArrow polygons (a shell, then its holes;
  the closing vertex dropped; holes without a shell dropped) with GPU-written, flat-padded offsets over
  fixed-length views, and `featureOffsets = [0, polygonCount]` (one feature). It plugs into
  `GPUPointInPolygonJoin` (`polygonPositions`, `polygonOffsets`, `ringOffsets`, `featureOffsets`): the join's
  compile-time topology is the ring capacity and unused polygons and rings are empty.
- Deterministic (integer atomics only). The cost is two pointer-jumping phases of `ceil(log2(rows))`
  dispatches each (plus the second tracing pass), a sort and scans. Rings that cross the antimeridian keep
  the input longitudes. Turn comparison uses a monotone pseudo angle, not `atan2`. Ring areas are f32.

### `GPUCellTableCompare`

`GPUCellTableCompare` outer-joins two sorted cell tables (the outputs of `GPUCellAggregation` or `GPUCellRollup`) into one union table for period-over-period or A-versus-B compare maps. Both tables must use the same grid family and resolution; the contributor cannot check that.

```ts
graph.add(new GPUCellTableCompare({
  before: lastMonth, after: thisMonth,
  measure: 'sum', sumScale: 65536,
  output: {cells, presence, before, after, delta, ratio, percentChange, zScore, count, overflow}
}));
```

- `output.cells` holds the union keys ascending (`uint32x2`, little-endian `(low, high)`); its length is the capacity. When the union is larger, the smallest keys are kept, `count` is clamped and `overflow` is 1. `overflow` also becomes 1 when either input table overflowed. Optional `totalCount` receives the unclamped union size.
- `presence`: bit 1 (`GPU_CELL_COMPARE_PRESENT_BEFORE`) when the cell is in `before`, bit 2 (`GPU_CELL_COMPARE_PRESENT_AFTER`) when in `after`.
- `before` / `after`: the measure (`'count'`, or `'sum'` decoded from the fixed-point `sums`), 0 when absent.
- `delta = after - before`, computed exactly as a 64-bit integer difference (counts and fixed-point sums) and only then rounded to f32, so large sums with small differences are not lost to f32 cancellation.
- `ratio = after / before` and `percentChange = 100 * (after - before) / before`, NaN when `before` is 0.
- `zScore`: `'poisson'` (default for counts) is `delta / sqrt(before + after)`, 0 when both are 0. `'standardized'` (default for sums) is `(delta - mean) / sd` over the union rows with the population standard deviation, 0 when sd is 0. The mean and sd use one workgroup in a fixed order, so a re-encode is bitwise identical.
- Rows past `count` hold the empty key (all ones), presence 0, `before`/`after` 0, and NaN in `delta`, `ratio`, `percentChange` and `zScore`.

The merge is a binary-search merge path with an exclusive scan of matched rows, so every union row is written by exactly one invocation and no atomics are used. Cost is O((n + m) log(n + m)). Table contents are per-frame (re-encode without recompiling); capacities, columns, `measure` and `zScore` are topology.

## Lines & trajectories

### `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, and `GPULineChunk`

Geometry-producing line contributors. Each writes a `GPULinePathOutput`: flat `positions`, clamped
`pathOffsets` (deck.gl `PathLayer` `startIndices`), `count`, `overflow`, and optional `totalCount`,
`pathCount`, `sourcePaths` (output path to input path), `sourceRows` (output vertex to input
segment row) and `measures` (distance along the source path). The vertex capacity is
`positions.length`; offsets are clamped to it, so a truncated result is still a valid layout.
Inputs are packed `positions` sorted by path with `pathOffsets` (`pathCount + 1` rows).

```ts
graph.add(new GPULineSegmentize({
  positions, pathOffsets, coordinateSystem: 'spherical', // or 'planar'
  parameters: segmentizeParameters.importToGraph(graph),
  output: {positions: densified, pathOffsets: densifiedOffsets, count, overflow, measures}
}));
segmentizeParameters.write(getGPULineSegmentizeParameterValues({maximumSegmentLength: 50000}));

graph.add(new GPUGreatCircleArcs({
  sources, targets, // longitude/latitude pairs
  parameters: arcParameters.importToGraph(graph),
  output: {positions: arcPositions, pathOffsets: arcOffsets, count, overflow}
}));
arcParameters.write(getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 32}));

graph.add(new GPULineSmooth({positions, pathOffsets, iterations: 3, closed: false,
  parameters: smoothParameters.importToGraph(graph), output}));

graph.add(new GPULineChunk({positions, pathOffsets, mode: 'chunk', // or 'substring'
  parameters: chunkParameters.importToGraph(graph),
  output: {positions: pieces, pathOffsets: pieceOffsets, count, overflow, pathCount, sourcePaths}}));
chunkParameters.write(getGPULineChunkParameterValues({chunkLength: 1000}));
```

- `GPULineSegmentize` (PostGIS `ST_Segmentize`, densify): every segment of length `L` becomes
  `clamp(ceil(L / maximumSegmentLength), 1, maximumPiecesPerSegment)` equal pieces; input vertices
  are kept. `'spherical'` positions are longitude/latitude degrees: lengths are great-circle
  distances times `radius` (default mean Earth radius, meters), interior points are slerped, and
  longitudes are unwrapped along each path so it stays continuous across the antimeridian.
- `GPUGreatCircleArcs` (turf `greatCircle`): one path of `segments + 1` vertices per
  origin/destination pair, `segments = clamp(max(ceil(distance / maximumSegmentLength),
  minimumSegments), 1, maximumSegments)`. Endpoints are exact; longitudes are unwrapped from the
  source (Tokyo to San Francisco ends at 237.6), never split. Draw with `PathLayer`
  `wrapLongitude: true` or in a globe view. Pairs within about 1e-3 rad of antipodal have no unique
  great circle.
- `GPULineSmooth` (Chaikin corner cutting, turf `polygonSmooth`): `iterations` (compile time,
  1 to 10) each replace edge `(a, b)` with `mix(a, b, r)` and `mix(a, b, 1 - r)` for a per-frame
  ratio `r` (default 0.25). Open paths keep their endpoints (`n` vertices become `n * 2^k`); with
  `closed: true` a repeated closing vertex is ignored and each ring repeats its first vertex.
  Output sizes depend only on the layout, so every level's offsets come from two scans in closed
  form and each level is one kernel; levels ping-pong through two capacity-sized transients.
- `GPULineChunk`: `'chunk'` splits each path into pieces of `chunkLength` (turf `lineChunk`), the
  last piece ending at the path end; consecutive pieces share their boundary point. The piece
  count is data dependent: the path capacity is `output.pathOffsets.length - 1`, and
  `pathCount`/`sourcePaths` report the pieces. `'substring'` extracts `[startMeasure,
  endMeasure]` from every path (turf `lineSliceAlong`, PostGIS `ST_LineSubstring` by distance);
  measures are clamped to the path and a start past the end gives an empty path. Zero-length
  paths are copied as one piece. Both modes support `'spherical'`.
- Everything per frame lives in the parameter buffer, so resolution, ratio, chunk length and
  ranges change with no recompile. All kernels are element-wise or sequential per path:
  deterministic, no float atomics.
- Precision (f32, measured against f64 oracles on Apple Metal): planar densify and chunks within
  2e-5 of the coordinate scale; spherical densify within 2.9 m, arcs up to 16,500 km within 5.7 m.

### `GPUGeometryMeasures`, `GPUGeodesicPairs`, and `GPUGeodesicDestination`

Measurement columns for line and polygon features and for point pairs, in planar coordinates or
on the Earth (turf `area`, `length`, `centroid`, `bbox`, `distance`, `bearing`, `destination`,
`midpoint`; PostGIS `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid`, `ST_Envelope`,
`ST_NPoints`, `ST_Distance(geography)`, `ST_Azimuth`, `ST_Project`).

```ts
graph.add(new GPUGeometryMeasures({
  positions, geometryType: 'polygons',
  ringOffsets, featureRingOffsets, // feature -> rings -> vertices
  coordinateSystem: 'wgs84', // or 'planar', 'spherical', 'geodesic'
  holeRule: 'winding', // or 'first-ring-exterior'
  output: {areas, lengths, centroids, bounds, vertexCounts},
  groupIds, groupCount: 12,
  groupOutput: {areas: groupAreas, centroids: groupCentroids, featureCounts}
}));

graph.add(new GPUGeodesicPairs({origins, targets, model: 'wgs84',
  output: {distances, initialBearings, finalBearings, midpoints, converged}}));
graph.add(new GPUGeodesicDestination({origins, bearings, distances,
  output: {destinations, finalBearings}}));
```

- Features: one invocation per feature walks its rings in row order with Neumaier-compensated f32
  sums, relative to the feature's first vertex, so projected coordinates far from the origin keep
  their precision (a 1 m square at x = 1e6 measures exactly 1). Rings close implicitly; a repeated
  closing vertex adds a zero edge. Outputs: length (lines) or perimeter (polygons), signed and
  absolute area, area-weighted (polygons) or length-weighted (lines) centroid falling back to the
  vertex mean, bounds, vertex count. Empty features give zeros and NaN centroid and bounds.
- Holes: `'winding'` sums signed ring areas (holes wind opposite to the exterior, RFC 7946);
  `'first-ring-exterior'` counts the first ring of each feature as exterior and the others as holes
  whatever their winding.
- `'spherical'` areas are the Chamberlain-Duquette formula turf uses (shoelace in the Lambert
  cylindrical equal-area projection, exact for longitude/latitude boxes; long edges are straight
  in that projection, not geodesics); pass `radius: GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS` for turf
  `area` parity. `'wgs84'` maps latitude to authalic latitude (exact equal-area ellipsoid mapping)
  and measures edges with an f32 Vincenty inverse. Rings around a pole are not supported.
- Groups: features are stably sorted by group ID and one workgroup per group reduces them with a
  fixed tree, so group sums are bitwise reproducible. Group centroids are area- (or length-)
  weighted means of feature centroids.
- `'geodesic'` (`pyproj.Geod.geometry_area_perimeter`, geo `GeodesicArea`): lengths and perimeters are
  Vincenty geodesic lengths, as `'wgs84'`, and polygon areas follow geodesic edges on the WGS84
  ellipsoid. Edges of 20 km or more are sampled along the geodesic (spans of at most 250 km, at most 16)
  and summed as a chord plus the exact parabolic-segment bulge in local equal-area coordinates, so f32
  never subtracts two huge areas. Within 1e-6 relative of pyproj on regional and continental polygons
  (perimeter 8e-6), where `'wgs84'` straight edges are off by up to 30% on very long edges. Centroids
  still use straight equal-area edges, edges where Vincenty does not converge stay straight, and edges
  under 20 km ignore the bulge.
- `output.extremeVertices` (`uint32x4`): per feature `[argMinX, argMinY, argMaxX, argMaxY]` as global
  `positions` rows, lowest row on ties, `0xffffffff` for empty features. Geographic x is the unwrapped
  longitude of `bounds`. Per feature only (rejected in `groupOutput`).
- `geometryType: 'points'`: every ring is one point or multipoint. Outputs are lengths (0), centroids
  (vertex mean, as Shapely `MultiPoint.centroid`), bounds, vertex counts and `extremeVertices`;
  `areas` and `signedAreas` are rejected. Group centroids weight by vertex count.
- `GPUGeodesicPairs`: distance, initial and final bearing (degrees clockwise from north in
  `(-180, 180]`), and midpoint per pair. `'sphere'` uses a Vincenty-form central angle that is well
  conditioned from millimeters to antipodal; `'wgs84'` runs Vincenty's inverse with a compile-time
  iteration cap and reports non-converged (near-antipodal) pairs, which fall back to the sphere.
- `GPUGeodesicDestination`: destination and final bearing from origin, bearing and distance
  columns, on the sphere or by Vincenty's direct solution.
- `model: 'rhumb'` on both classes (turf `rhumbDistance`, `rhumbBearing`, `rhumbDestination`; geo `Rhumb`):
  a line of constant bearing on a sphere of `radius` (default 6371008.8 m). Pairs give `distances` (in
  `radius` units), `initialBearings` equal to `finalBearings` (constant, in `(-180, 180]`) and the rhumb
  `midpoints`. The destination longitude is continuous with the origin, and `finalBearings` flips to the
  opposite north-south sense when a move overshoots a pole and is reflected, as turf does (the reflected
  case is approximate). Longitude differences take the short way across the antimeridian, latitudes clamp
  at 89.9999 degrees, equal-latitude edges use the east-west limit `cos(lat)`, and the Mercator-stretched
  latitude difference is an `atanh` of a cancellation-free ratio. Sphere only (turf and geo are spherical
  too), `converged` is `'wgs84'`-only, and the model switch is compile-time. Distance error against an f64
  closed form was 0.74 m (6e-8 relative) over 12,108 km and bearings were within 1e-4 degrees. Rhumb length
  in `GPUGeometryMeasures` and rhumb densify in `GPULineSegmentize` are not provided.
- Precision (f32 vs f64 oracles, Apple Metal): WGS84 lengths 9e-8 m at 1 m, 0.78 m at 12,000 km;
  WGS84 and spherical box areas within 3e-7 relative from 100 m blocks to 2,000 km boxes;
  sphere pair distances 1e-7 m at 1 m and 0.9 m at 10,000 km; destinations within 1 m, bounded by
  the f32 resolution of output latitudes (about 0.4 m).

### `GPULinearReferencing` and `GPULineLocate`

Linear referencing on many polylines (`positions` sorted by path, `pathOffsets`): snap points to
the nearest path and report where they are along it, or place events on paths by measure.

```ts
graph.add(new GPULinearReferencing({
  points, positions, pathOffsets,
  radius: radiusParameters.importToGraph(graph), // one float32 row, per frame
  candidateCapacity: 1 << 20, spatialSort: true,
  output: {pathIndices, segmentIndices, fractions, footPoints, distances, measures, sides, signedOffsets},
  overflow
}));

graph.add(new GPULineLocate({
  positions, pathOffsets, eventPaths, eventMeasures, eventOffsets,
  measureMode: 'distance', // or 'fraction'
  parameters: locateParameters.importToGraph(graph),
  output: {positions: eventPositions, angles, statuses}
}));
locateParameters.write(getGPULineLocateParameterValues({measureOffset: elapsedMeters})); // animate
```

- `GPULinearReferencing` (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint` and
  `ST_LineLocatePoint`): every vertex row becomes a segment to the next row of its path and
  `GPUNearestFeatureJoin` finds each point's nearest segment within the radius through its BVH.
  Ties go to the smallest segment row (earliest along the path, lowest path). Outputs per point:
  path, segment within the path, fraction along the segment, foot point, distance, measure (distance
  along the path to the foot point), side (`1` left of the path direction, `-1` right, `0` on it)
  and `side * distance`. Points with nothing in range get `0xffffffff` indices, distance `-1`,
  side `0` and NaN elsewhere. `overflow` reports BVH leaf or candidate capacity overflow.
- `GPULineLocate` (turf `along`, PostGIS `ST_LineInterpolatePoint` and `ST_LocateAlong`): each
  event's measure (distance or fraction, optionally `measure * scale + offset` from the parameter
  buffer) is clamped to the path (`statuses` 1) and located by binary search over cumulative vertex
  measures, then moved by an optional lateral offset along the left normal. Outputs: position,
  segment, unit tangent, angle (degrees counter-clockwise from +x, the deck.gl `getAngle`
  convention) and status (`2` for empty or out-of-range paths, NaN position).
- Cumulative measures are a per-path Neumaier-compensated f32 prefix (optionally output as
  `vertexMeasures`). Planar by default: project geographic data first, or use `'spherical'` below.
- `output.normalizedMeasures` (both coordinate systems) is `measures / pathLength`, 0 for zero-length
  paths and NaN when unmatched (Shapely `line_locate_point(normalized=True)`).
  `GPULineLocate({measureMode: 'fraction'})` is `line_interpolate_point(normalized=True)`.
- `coordinateSystem: 'spherical'` on both classes: positions are longitude and latitude in degrees and
  segments are great-circle arcs. `distances`, `measures`, `signedOffsets` and the search `radius` are
  meters on a sphere of `sphereRadius` (default 6371008.8; `radius` already names the search radius).
  The foot point is the closest point on the arc (geo `HaversineClosestPoint`), with a longitude that is
  continuous with the segment start, so it can exceed 180 across the antimeridian. `distances` is the
  cross-track distance and `measures` the along-track measure. `GPULineLocate` slerps along the arc
  (`Geod.fwd`, `npts`): `tangents` are `(east, north)` compass components, offsets move along the left
  compass bearing in meters and `angles` are counter-clockwise from east. Limits: the search is brute
  force, points times segments, with no BVH and no overflow (`candidateCapacity`, `leafCapacity` and
  `spatialSort` are ignored); arcs over about 179 degrees are unsupported. Against `pyproj.Geod(a=6371008.8,
  f=0)` the worst error was 1.5 m for distance, 0.8 m for measure and 1.7 m for the foot point.

### `GPULineSplit` and `GPULineMerge`

`GPULineSplit` splits linestrings at every crossing, touch, shared vertex and overlap end (turf `lineSplit`
against the union of the lines; verified against Shapely `unary_union`). It reuses `GPUSegmentIntersection`
in self mode, groups the split events by segment with a stable `GPUSort`, removes duplicates and writes
GeoArrow-layout pieces. Props: `lines` (`{kind: 'lines', positions, lineOffsets}`), `intersectionCapacity`,
`pieces`, optional `spatialSort`, `leafCapacity` and `uncertainCount`. `pieces` holds `lineIds` (the
source line of each piece), `offsets`, `positions` (split points included), `count`, and optional
`vertexCount`, `overflow`, `totalCount` and `totalVertexCount`. Pieces are ordered by source line, then
traversal order. Splits at line ends are ignored, and a line with fewer than two vertices yields no piece.
When pieces, vertices or `intersectionCapacity` overflow, a suffix of pieces is dropped and `overflow` is
set.

`GPULineMerge` is the inverse (Shapely `linemerge`, turf `lineMerge`): it joins linestrings at endpoints
shared by exactly two line ends (an exact f32 match) into maximal chains. Junctions of three or more ends
stop a chain, a closed line never joins itself, and lines with fewer than two vertices are ignored. With
`directed: true` (Shapely `line_merge(directed=True)`) only a line end meeting a line start joins, chains
follow the input direction and `lineReversed` is all 0.
Props: `positions`, `lineOffsets` and `output: {chainOffsets, positions, count, lineChains?, lineOrders?,
lineReversed?}`. The result is deterministic: chains are ordered by their lowest line, paths start at the
lower endpoint ID and cycles start at their lowest line and are emitted closed. One thread walks each
chain, so a chain of `n` lines takes `n` serial steps. Merging with a tolerance is not provided.

```ts
graph.add(new GPULineSplit({
  lines: {kind: 'lines', positions, lineOffsets}, intersectionCapacity: 1 << 16,
  pieces: {lineIds, offsets, positions: piecePositions, count, vertexCount, overflow}
}));
graph.add(new GPULineMerge({positions, lineOffsets, output: {chainOffsets, positions: merged, count}}));
```

`GPUNetworkNoding` in [GPU Network](/docs/api-reference/experimental/gpu-network) builds a routable
network from the pieces.

### `GPULineClipByPolygon` and `GPUSharedPaths`

`GPULineClipByPolygon` clips linestrings to polygons (GeoPandas `clip(lines, polygons)`, Shapely
`intersection(line, polygon)`); `mode: 'outside'` is `difference`. It is bounded because it never computes
a new intersection between two polygons: every output vertex is an input vertex or a crossing of one line
segment with one polygon edge. `GPUSegmentIntersection` lists the crossings, each segment's crossings are
ranked in place (pairs arrive sorted by line segment) and deduplicated into sub-pieces. Sub-pieces inside a
collinear overlap are boundary and count as inside; the rest are classified by `GPUPointInPolygonJoin` at
their midpoint, and kept sub-pieces that continue each other merge. Several polygon features act as their
union. Props: `lines`, `polygons`, `mode`, `intersectionCapacity`, `candidateCapacity`, `pieces` (the
`GPULineSplitPieces` layout) and `uncertainCount`. Overflow of pairs, candidates, pieces or vertices sets
`pieces.overflow`. Limits: a line is not noded at its own crossings, a line running over itself keeps both
traversals (Shapely dedupes), a point-only touch yields no piece (`keep_geom_type=True`), and a sub-piece a
few f32 ulps long beside an edge can be misclassified. Against Shapely 2.1.2, 52 lines in both modes
matched on total length (under 2e-3) and piece end points.

`GPUSharedPaths` is Shapely `shared_paths(a, b)` for many line pairs. Collinear overlaps from
`GPUSegmentIntersection` are oriented along the left line, tagged forward or backward by the exact sign of
the segment directions, and chained into one polyline per (left line, right line, direction). The output
`GPUSharedPathsRuns` has `leftLineIds`, `rightLineIds`, `forward`, `offsets`, `positions`, counts and
`overflow`. Closed left lines are cut at their first vertex, a chain walk is serial per run, and two
coincident right segments emit duplicate runs.

### `GPULineSimplification`

Douglas-Peucker (and approximate Visvalingam-Whyatt) simplification of many polylines or tracks, stored as rows sorted by line and
delimited by `trackOffsets` (the `GPUTrajectoryMetrics` layout). It runs in two parts: a
per-vertex `importance` column computed once, and a per-frame selection at a tolerance from a
parameter buffer. Changing the tolerance re-runs only the mask, compaction, and publish nodes.

```ts
// One-shot graph: importance.
importanceGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance,
  metric: 'segment', // or 'time-ratio' (TD-TR) with float32 `timestamps`
  maximumRounds: 64,
  status: {converged, roundCount}
}));

// Per-frame graph: selection only.
frameGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance, computeImportance: false,
  parameters: toleranceParameters.importToGraph(frameGraph),
  selection: {output: {ids, count, overflow}, keepMask, lineCounts, lineStarts}
}));
toleranceParameters.write(getGPULineSimplificationParameterValues({tolerance: 2})); // no recompile
```

- Importance: every line starts as one open interval between its endpoints (importance
  `+Infinity`). Each round, every open interval picks its farthest row (segmented argmax with ties
  to the smallest row) and splits there. The row's importance is
  `min(distance, min(importance[a], importance[b]))`, so importance never increases from a split
  to its children. An interval whose largest distance is 0 is resolved at once with importance 0.
- Rounds are GPU gated. They stop when no row is left (`converged = 1`) or after `maximumRounds`.
  A balanced split tree takes about `log2(n)` rounds and a spiral can take up to `n - 2`. If the
  cap is reached, the remaining rows get their interval's split importance, so the kept set is a
  superset of Douglas-Peucker's.
- Selection: a row is kept when `importance > tolerance`. `output.ids` holds the kept rows in
  ascending order, and `count` is clamped so it can drive an indirect draw. `lineCounts` (not
  clamped) and `lineStarts` give each line's range inside `output.ids`.
- Exactness: with `converged = 1`, the kept set at any tolerance `e` equals classic recursive
  Douglas-Peucker at `e` (split when `dmax > e`, first maximum on ties) under the same metric.
  The f32 distances are bit-identical to a `Math.fround` CPU oracle: products are prevented from
  fusing into FMAs, and division and square root are corrected to defined values.
- Metrics: `'segment'` measures the distance to the chord segment, which is the perpendicular
  distance inside the segment and the distance to the nearer anchor outside it (GEOS and
  simplify-js behave the same way). `'time-ratio'` is TD-TR's synchronized Euclidean distance,
  which needs float32 `timestamps`.
- Use tile- or view-local coordinates. Simplified lines may self-intersect, because topology is
  not preserved (Douglas-Peucker and Visvalingam alike).

**Visvalingam-Whyatt.** `method: 'visvalingam'` writes each vertex's effective triangle area (squared
position units) into `importance`, and the per-frame `tolerance` is then an area: a vertex is kept when
`importance > tolerance`, and endpoints are always kept. The selection outputs are the same as for
Douglas-Peucker. Each round removes every surviving vertex whose area is the smallest within
`neighborhoodRadius` steps both ways (1 to 8, default 3, compile-time), then unlinks them; the effective
area is the maximum of its own area and those of its removed neighbors, which is the sequential running
maximum. It is approximate: the sequential heap order cannot be parallelised exactly. Against
`simplification.simplify_coords_vw` on 3000-vertex random walks the kept set differed by 3.1, 0.65 and 0.13
percent of vertices at radius 1, 2 and 3, and lines up to 300 vertices matched exactly at radius 3. It needs
about 31 rounds for 300 vertices and 55 for 3000; `maximumRounds` (default 256) caps them, and undecided
rows get `+Infinity` (a superset). Not supported: `metric: 'time-ratio'` and topology-preserving VW
(`simplify_coords_vwp`).

### `GPUCoverageSimplification`

Gap-free simplification of a polygon coverage (Shapely `simplify_coverage`, PostGIS `ST_CoverageSimplify`).
Every arc shared by two polygons is simplified once, so both neighbors keep the same vertices along it.
Arcs are runs of ring edges with the same partner polygon, broken at each ring start; the lower polygon ID
owns a shared arc. The kept vertices are the Douglas-Peucker survivors of the owned arcs plus the junctions,
shared through point IDs, and the output rings are the input rings filtered by the mask, in the same order
and direction.

Props: `positions` (`float32x2` ring vertices), `ringOffsets`, `polygonOffsets` (as `GPUContiguityWeights`),
`snapTolerance` (default 0), `parameters` (packed with `getGPULineSimplificationParameterValues`; the
tolerance changes per frame without recompiling), `maximumRounds`, `converged`, and `output` with
`positions` (the kept vertices, capacity), `ringOffsets` (`ringCount + 1`), optional `keepMask` (per input
vertex), `overflow` and optional `totalCount`. Polygon offsets are unchanged.

```ts
graph.add(new GPUCoverageSimplification({
  positions, ringOffsets, polygonOffsets,
  parameters: toleranceBuffer.importToGraph(graph),
  output: {positions: outPositions, ringOffsets: outRingOffsets, overflow}
}));
toleranceBuffer.write(getGPULineSimplificationParameterValues({tolerance: 0.05}));
```

Topology preservation: arc endpoints (junctions and ring starts) are always kept and a ring never keeps
fewer than three vertices (the farthest original vertices are restored, even with `topologyRounds: 0`).
Then up to `topologyRounds` (compile-time, default 4, 0 turns the repair off) rounds compact the
simplified rings, run `GPUSegmentIntersection` with exact predicates and, for every pair of segments that
cross, touch or overlap without sharing a coverage point, restore the original vertex farthest from each
segment (one Douglas-Peucker split; decisions go through point IDs, so neighbors still agree).
`output.topologyStats` (4 `uint32`, length `GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATS_LENGTH`) reports
the crossings before repair, the crossings remaining, the vertices restored and a candidate-pair overflow
flag; `topologyPairCapacity` (default `max(256, 4 * vertices)`) sizes the candidate list. Residual crossings
are reported, never hidden, and input that already crosses itself cannot be repaired (it counts in both
numbers). On a wavy-strip scene the repaired output passes Shapely `coverage_is_valid`, where plain
Douglas-Peucker gave three invalid polygons. The criterion is distance, not area (Shapely's
`simplify_coverage` is area based and keeps a different vertex set), and there is no guarantee of zero
residual within the round cap.

`simplifyBoundary: false` (Shapely `coverage_simplify(simplify_boundary=False)`) keeps every vertex that
touches an unshared edge and simplifies only the shared arcs.

One thread walks each arc, so one huge arc is a serial walk (and a long dropped span costs a serial walk
per kept vertex), and non-finite vertices are not supported. A shared arc that crosses a ring start keeps
one extra vertex.

### `GPUCoverageDissolve`

Dissolves a polygon coverage by a per-polygon `uint32` label (GeoPandas `dissolve(by, method='coverage')`,
Shapely `coverage_union_all` per label). An edge is dropped when another polygon with the same label has
the same edge (snapped point IDs, `(min, max)` keys). Kept edges are oriented interior-left and chained by
`GPUSegmentRingAssembly` with the label as the group, so the output has shells, holes and
`polygons.polygonGroups`. Vertices at nodes are kept, as GEOS does. This is not the excluded vector
overlay: a coverage has no crossings, so the result is a subset of the input edges and is bounded by the
input vertex count plus one closing vertex per ring, with ring and vertex capacities and an overflow flag.
The input must be a valid coverage (check it with [`GPUCoverageValidity`](#gpucoveragevalidity)).
`snapTolerance` snaps vertices and the output uses the first vertex's coordinates. Dissolve edge sets,
area, parts and holes matched `gdf.dissolve(method='coverage')` exactly on 5 scenes and 18 labels. The
per-ring area walk is serial per ring. The directory is `polygon-coverage-topology/` because `.gitignore`
ignores `coverage-*/`.

### `GPUTrajectoryMetrics`

Per-track metrics for rows sorted by track and time and delimited by `trackOffsets`: planar length,
duration, average speed, and maximum speed (via `GPUSegmentedReduction`). Stop detection finds
maximal runs of slow steps (speed below a per-frame threshold) that last at least a per-frame
minimum duration. It publishes a bounded stop list in row order (track, start row, end row,
duration, centroid) and unclamped per-track stop counts.

`timestamps` may be `float32`, `float32` plus `timestampsLow` (double-single), or exact `Int64`
words (`uint32x2`). Every time difference (steps, track durations, dwells) is taken exactly or in
double-single before conversion to float32, so multi-year epoch-millisecond tracks keep
millisecond steps. Durations, speeds and the stop parameters use the unit of the timestamps.

```ts
graph.add(new GPUTrajectoryMetrics({
  positions, timestamps, trackOffsets, parameters: stopParameters.importToGraph(graph),
  trackLengths, averageSpeeds, maximumSpeeds, trackStopCounts,
  stops: {output: {ids: stopTracks, count, overflow}, startRows, endRows, centroids, durations}
}));
stopParameters.write(getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 0.5, stopMinimumDuration: 300}));
```

Optional per-vertex columns (`positions.length` float32 rows) describe the step that ends at each row:
`stepSpeeds` (distance over delta time, 0 if the delta time is not positive), `stepHeadings` (`atan2(dy,
dx)` in radians, 0 for a zero-distance step) and `stepAccelerations` (`(speed_i - speed_{i-1}) /
deltaTime_i`, 0 for the first two rows of a track). The first row of a track and rows outside tracks are 0,
so a drawn segment takes the value of its end row. `stops.drawInstanceCount` (one `uint32` row) receives
the clamped stop count, for example an indirect draw record's instance count.

### `GPUTrajectoryPlayhead` and `GPUTrajectoryResample`

`GPUTrajectoryPlayhead` interpolates every track at a per-frame playhead: one invocation per track
binary-searches the track's sorted timestamps and interpolates the bracketing segment. It writes
per-track columns (position, optional elevation, heading in radians, speed, status, segment row and
fraction) and an optional compact, ascending list of active tracks with a clamped count, so a
layer can draw one instance per moving object with an indirect instance count. Moving the playhead
only rewrites the parameter buffer.

```ts
graph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets, // same layout as GPUTrajectoryMetrics
  elevations, // optional float32 z per row
  parameters: playheadParameters.importToGraph(graph),
  currentPositions, currentElevations, headings, speeds, status, segmentRows, segmentFractions,
  activeTracks: {ids, count, overflow}, drawInstanceCount
}));
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead: 125.5, maxGap: 60}));
// Int64 epoch-ms words: a uint32 parameter buffer and
// getGPUTrajectoryPlayheadWordParameterValues({playhead: Date.now() + 0.25, maxGap: 60_000})
```

- Status (`GPU_TRAJECTORY_PLAYHEAD_STATUS`): `empty` (no rows), `active`, `beforeStart`,
  `afterEnd`, and `gap` (the playhead lies strictly inside an interval longer than `maxGap`; 0
  disables the rule). Outside the range the position clamps to the first or last sample and the
  heading is that of the first or last segment; in a gap the position holds at the last fix.
  Speed is non-zero only for active tracks.
- The segment is `[j - 1, j]` with `t[j - 1] <= p < t[j]`, or the last segment when `p` equals the
  last time. A playhead exactly on a sample time returns that sample's position bit-exactly.
- Duplicate timestamps: the last row of a run of equal times wins ("latest row wins"), so the
  interpolation interval always has a positive duration and nothing divides by zero.
- Heading is `atan2(dy, dx)` (0 along +x, counterclockwise); compass bearing is `π/2 - heading`.
  Zero-length segments give 0. Positions are planar; project upstream (no antimeridian handling).
- Time: comparisons are exact in both modes (Int64 words compare integer words, then the f32
  playhead fraction); differences are subtracted exactly and then rounded to f32.

`GPUTrajectoryResample` resamples every track to `sampleCount` samples, uniformly in time
(`spacing: 'time'`, default) or along the planar path (`'arc-length'`), into a dense
`[trackCount × sampleCount]` `float32x2` buffer (row `track * sampleCount + k`), with optional
elevations and sample times relative to the track start. It is the fixed-length input for
trajectory similarity, k-means, and fixed-vertex trails.

```ts
graph.add(new GPUTrajectoryResample({
  positions, timestamps, trackOffsets, sampleCount: 32, spacing: 'arc-length',
  samples, sampleElevations, sampleTimes
}));
```

`spacing: 'clock'` samples every track at the instants of one shared clock (a per-frame `clock` view
`[start, step]`, packed with `getGPUTrajectoryClockParameterValues` or, for Int64 timestamps,
`getGPUTrajectoryClockWordParameterValues`; `GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH` words). Column `k` is the
same instant for every track and the sample is NaN outside a track's own time span, which is the layout
`GPUTrajectoryEncounters` expects.

- Sample `k` targets `total * k / (sampleCount - 1)`; the last sample is the last row exactly. It
  uses the playhead's search and duplicate rule. Single-row tracks repeat that row; empty tracks
  write zeros.
- Arc length sums step lengths sequentially per track (deterministic, O(track length) per
  invocation).
- Inputs must be single packed views; chunked vectors are not supported.

### `GPUZoneEvents`

Entry and exit events of trajectories against polygon zones, with interpolated times, bounded events per
track, and dwell time and visit count per `(track, zone)`.

- Inputs: `positions` (`float32x2`), `timestamps` (`float32` relative, or `uint32x2` Int64 words),
  `trackOffsets` (`trackCount + 1`), the zone boundary edges `edgeStarts`, `edgeEnds` (`float32x2`) and
  `edgeZones` (the zone index; the rings and holes of one zone share it), `zoneCount`, `candidateCapacity`
  and `maxEventsPerTrack`.
- Outputs: `events.output` (`GPUCompactOutput`; `ids` is the track index), optional `eventZones`,
  `eventTypes` (`GPU_ZONE_EVENT_TYPE.enter` 0, `.exit` 1), `eventTimes` (relative to the track's first
  timestamp, f32) and `eventRows` (the segment end row); optional dense `[trackCount * zoneCount]`
  `dwellTimes` and `visitCounts`; and `trackEventCounts` (unclamped).
- Events are sorted by `(track, time, edge row)`. Each segment queries a `GPUBVH` of edge bounds and
  runs an exact segment-edge test with half-open parameters; the time is
  `t[r - 1] + s * (t[r] - t[r - 1])`. Enter versus exit comes from parity, not ring orientation: the first
  sample of every track probes with a +x ray (even-odd), so a track that starts inside a zone works, and
  that first interval counts as a visit with dwell from `t = 0`. Open visits close at the track's last
  sample. Dwell and visits use every event, even when the event list is truncated.
- `events.output.overflow` is 1 when the candidate scratch overflowed, any track exceeded
  `maxEventsPerTrack`, or more events were kept than `ids.length`.

```ts
graph.add(new GPUZoneEvents({
  positions, timestamps, trackOffsets, edgeStarts, edgeEnds, edgeZones,
  zoneCount: 3, candidateCapacity: 65536, maxEventsPerTrack: 16,
  events: {output: {ids, count, overflow}, eventZones, eventTypes, eventTimes},
  dwellTimes, visitCounts
}));
```

`events.eventPositions` (`float32x2`) holds the interpolated crossing position of each event. `visitTable`
is a sparse `(track, zone)` table that replaces the dense matrices for large `trackCount * zoneCount`: rows
are ordered by track, then zone, with `output` (a `GPUCompactOutput`: `ids` is the track, plus `count`,
`overflow` and `totalCount`) and the columns `zones`, `visits`, `dwellTimes`, `firstEnterTimes` and
`lastExitTimes` (see `GPUZoneVisitTableOutput`). `addFleetDwellZoneEventsRecipe` forwards both outputs. Times are
relative to the track start, a track that starts inside has first enter 0, and an open visit has last exit
equal to the track duration. The table is built from the dense per-cell state, so it bounds the output, not
the working memory.

Limits: the scratch of `candidateCapacity` rows is sorted (three stable `GPUSort` passes) every encoding;
the walk and dwell kernels use one invocation per track; the dense matrices cost `trackCount * zoneCount`
rows; samples exactly on an edge follow the half-open rule and can disagree with other predicates by one
event; zones with overlapping rings are not supported (even-odd); at equal timestamps the crossing position is not unique.

Diagnostics: the optional `diagnostics` prop takes four one-row `uint32` views, `candidateCount`,
`candidateOverflow`, `trackOverflow` and `eventOverflow`.
- `candidateCount` is the unclamped number of segment-edge bounding-box candidates, including one ray query
  per track. Set `candidateCapacity` to at least this for exact results.
- `candidateOverflow` means the scratch was too small, so events may be missing.
- `trackOverflow` means a track has more than `maxEventsPerTrack` events. Its list is truncated, but dwell and
  visits stay exact.
- `eventOverflow` means there are more events than `events.output.ids.length`. `events.output.totalCount`
  holds the size needed.

`events.output.overflow` remains the OR of the three flags.

### `GPUTrajectoryEncounters`

Pairs of tracks within `distance` in the same time bucket (MobilityDB `tdwithin` on resampled tracks).

- Inputs: dense `samples` (`[trackCount * bucketCount]` `float32x2`, row `track * bucketCount + bucket`,
  the `GPUTrajectoryResample.samples` layout), `trackCount`, `bucketCount`, optional `trackValid`,
  per-frame `distance` (one float row, clamped to `cellSize`), compile-time `cellSize` and
  `bounds [minX, minY, maxX, maxY]`, `hitCapacity` and optional `bucketTimes`.
- Outputs: `pairs.output` (`ids` is the lower track) and optional `partners`, `firstBuckets`,
  `minimumDistances`, `bucketCounts` and `firstTimes` (needs `bucketTimes`). Pairs have `track < partner`,
  are deduplicated across buckets and sorted by `(track, partner)`; the sentinel is `0xffffffff` (IDs,
  buckets) and 0 after `count`. Overflow covers the hit scratch, the pair capacity and the grid.
- It uses a 3D `GPUGridIndex` (x, y and the bucket as an exact third axis) with a 3x3 scan per sample,
  three stable sorts of the hits and `GPUCompaction` for the runs.
- It is discrete in time: the samples must share a clock (column `k` is one instant for all tracks), and
  encounters between buckets are missed. With `GPUTrajectoryResample` that holds only when the tracks cover
  the same window.

```ts
graph.add(new GPUTrajectoryEncounters({
  samples, trackCount, bucketCount, distance, cellSize: 50, bounds: [0, 0, 1000, 1000],
  hitCapacity: 1 << 16, pairs: {output: {ids, count, overflow}, partners, firstBuckets, minimumDistances}
}));
```

`addClockEncounters(graph, props)` builds the shared-clock chain in one call: it resamples all tracks
onto one clock (`GPUTrajectoryResample` with `spacing: 'clock'`, a per-frame `[start, step]` buffer) and
feeds `GPUTrajectoryEncounters`, so tracks need not cover the same window. Optional `bucketTimes` and
`samples` outputs expose the clock and the resampled tracks (`AddClockEncountersProps`, `ClockEncounters`).

### `GPUTrackSimilarity`

Hausdorff (symmetric, vertex-to-segment as GEOS computes it, unbounded length) and discrete Frechet
(vertex-based) distances for given pairs of tracks or rings (`hausdorff_distance(a, b, densify)`,
`frechet_distance(a, b, densify)`). Behavior change: Hausdorff used to be vertex-to-vertex, which agreed
with Shapely only by coincidence; it now measures from the (densified) vertices to the other polyline's
segments, so values can be slightly smaller than before. Inputs are `positionsA` and `offsetsA` (optionally `positionsB` and `offsetsB`, by
default the same set), `pairA`, `pairB` (`uint32` views) and an optional GPU-written `activePairCount`.
Outputs are `hausdorff`, `frechet`, optional `maxDistance` (largest vertex-to-vertex distance per pair, Sedona
`ST_MaxDistance`, original vertices) and `status` (bits of `GPU_TRACK_SIMILARITY_STATUS`: `emptyTrack` 1,
`frechetCapExceeded` 2, `invalidPair` 4). Skipped pairs get NaN. `densify` (0 or in `[1/4096, 1]`, compile-time) splits each segment into `round(1/densify)` parts, as
Shapely does; the virtual vertices are computed on the fly. Frechet sweeps the longer track in strips of
256 lanes, so only the shorter (densified) track is capped: `maxFrechetVertices` is 1 to 2048 (default
256). A pair over the cap reports `frechetCapExceeded` with a NaN Frechet while its Hausdorff is still
computed. Polygon boundaries are passed as closed vertex lists. Against Shapely 2.1.2 (GEOS 3.13.1) all
three measures were within 2e-4 relative on line and polygon-boundary pairs, including a 700 by 300 Frechet
over three strips. Rings are open vertex
sequences, so Frechet is sensitive to the start and the orientation. Pairs can come straight from
`GPUTrajectoryEncounters` (`ids`, `partners` and `output.count`).

```ts
graph.add(new GPUTrackSimilarity({
  positionsA, offsetsA, pairA: pairs.output.ids, pairB: partners, activePairCount: pairs.output.count,
  hausdorff, frechet, maxDistance, status, maxFrechetVertices: 256, densify: 0.25
}));
```

## Geometry utilities

All of these are contributors: construct, then `graph.add`. Per-frame values go through a
`GPUParameterBuffer` packed with the matching `getGPU...ParameterValues` helper, so nothing recompiles.

### `GPUOutlineGeometry`

Buffer-like outline geometry for drawing (turf `buffer` and `lineOffset`, PostGIS `ST_Buffer` as a
picture). Props: `positions`, `geometryType` (`'points'`, `'lines'` or `'rings'`), `pathOffsets` (lines and
rings), `coordinateSystem` (`'planar'` or `'spherical'`), `radius`, `joinSegments` (default 16),
`parameters` (`getGPUOutlineGeometryParameterValues({distance})`) and `output.positions`. It writes a
non-indexed triangle list with exactly `positions.length * getGPUOutlineGeometryVerticesPerInput(joinSegments)`
vertices: per input vertex a round-join disc fan plus a quad to the next vertex, with unused triangles
degenerate. There is no capacity and no overflow. Overlaps are not unioned, so it is for drawing only;
queries use `dwithin` and `GPUBufferSelection`. Spherical mode converts meters with each vertex's local
east and north scale (equirectangular, not geodesic; wrong near the poles and across the antimeridian).
There are no negative buffers. A one-sided offset line is [`GPUOffsetCurve`](#gpuoffsetcurve-and-gpuvertexsnap).

### `GPUOffsetCurve` and `GPUVertexSnap`

`GPUOffsetCurve` is a one-sided offset of lines and rings (`shapely.offset_curve`, geo `OffsetCurve`);
a positive distance is to the left of the path direction. Per vertex it writes a round arc
(`quadSegments` per quarter circle), a mitre (clipped beyond `mitreLimit`, default 5) or a bevel on the
outside of a turn, the single intersection point on the inside, and a plain normal offset at path ends.
The output has `getGPUOffsetCurveRowsPerVertex` rows per vertex (`2q + 1` round, 2 otherwise); the optional
`output.counts` gives the distinct points per vertex and padding repeats the last point. Distance and mitre
limit are per-frame. It is render-grade: GEOS loop removal is not done, so tight inside corners can
self-cross. GEOS clamps `quad_segs` to at least 8, so round-join parity is pinned at 8 and 16; GEOS
leaves the corner at a ring's first vertex open on the outside and this closes it. Points were within 2e-4
of Shapely 2.1.2. Planar only, rings are implicit-closed.

`GPUVertexSnap` is the vertex-to-vertex subset of `shapely.snap`: each vertex with a reference vertex
strictly closer than the per-frame `tolerance` moves to the nearest one (lowest row on ties) and the layout
is unchanged. Optional pairwise mode (`featureOffsets` and `referenceOffsets`) and `output.referenceRows`.
It is `O(V * R)`. Segment insertion and snapping onto segments, which GEOS also does, are out of scope.
It matched Shapely on 60 random lines.

### `GPUAffineTransform`, `GPUGeometryOrientation`, and `GPUGeometryCleanup`

Geometry edits that keep the GeoArrow layout. `GPUAffineTransform` is `shapely.affinity` (`affine_transform`,
`translate`, `scale`, `rotate`, `skew`): either one per-frame matrix packed by
`getGPUAffineTransformParameters({rotate, scale, skew, translate, origin})` (applied as scale, skew,
rotate, translate about one origin) or per-feature `featureTransforms` (6 floats per feature, Shapely
order). The origin is a point, or `'center'` or `'centroid'` computed per feature on the GPU (compile-time
`origins`); it is computed once from the input, where Shapely recomputes `'center'` at each step, which
differs only for skew about the center. Agreement was within 0.03 at coordinates near 1e5 (f32 spacing
0.0078).

`GPUGeometryOrientation` is `shapely.reverse` (`mode: 'reverse'`) or `orient_polygons` (`mode:
'orient-polygons'`, which needs `polygonOffsets` and a per-frame `exteriorClockwise`). The layout is
unchanged and `reversedRings` flags are optional. The sign comes from an f32 shoelace sum, which differs
from GEOS only for self-intersecting rings.

`GPUGeometryCleanup` is `remove_repeated_points(tolerance)` and `set_precision(grid_size, mode='pointwise')`:
`gridSize` snaps half-up as GEOS does, and `tolerance` removes repeats with the exact GEOS semantics (first
and last kept, inclusive tolerance against the previous kept vertex). The output is compacted `positions`
with republished `ringOffsets`, `count`, `overflow`, `totalCount` and `collapsedRings` (polygon rings with
fewer than 3 surviving vertices are emptied, where GEOS raises). GEOS `set_precision(pointwise)` does not
drop repeats, so use `removeRepeatedPoints: false` for exact parity. Both outputs matched Shapely exactly
on 5 parameter sets by 24 features. Points are not supported, the kernels are one thread per ring, and
`normalize` is not provided.

### `GPUMinimumBounds`

Per feature (from `positions`, `ringOffsets` and optional `featureRingOffsets`) or per `uint32` label group
(`labels` and `groupCount`): the minimum-area rotated rectangle (`rectangleCorners`, four counter-clockwise
corners, and `rectangleSizes` = width, height, angle, area), the minimum bounding circle (`circles` = cx, cy,
r), the longest line (`longestLines`, `diameters`) and the envelope (`bounds`). Equivalents: Shapely
`oriented_envelope`, `minimum_rotated_rectangle`, `minimum_bounding_circle`, `minimum_bounding_radius`;
Sedona `ST_LongestLine`, `ST_MaxDistance`. It is built on `GPUGroupConvexHull` (exact lattice hull) with one
thread per group over the hull. Hulls over `maximumHullVertices` (default 256) set the hull overflow bit in
`overflow` and write NaN; `hullSizes` optionally reports the true hull vertex counts. Rectangle ties (areas
within 1e-6 relative) go to the lowest hull edge, and the circle is Welzl in a hash-shuffled fixed order
with the radius recomputed as the largest hull distance, so it always covers. Parity is by area and radius,
not vertex order: rectangle area within 2e-4 relative, radius 1e-4, diameter 1e-4. Cost is `O(h^2)` per
group, and near-collinear circumcenters fall back to the diametral circle.

### `GPUPolygonTriangulation`

Earcut on the GPU (`earcut`, deck.gl `PolygonLayer` tessellation, geo `TriangulateEarcut`, Sedona
`ST_TriangulatePolygon` as earcut, not constrained Delaunay). Props: `polygons: {positions, polygonOffsets,
ringOffsets}`, `indices`, `valid` and optional `triangleCount` and `maximumWork` (default 4M steps, roughly
rings of a few thousand vertices). The first ring is the shell and the others holes, with any winding.
Polygon `p` owns exactly `3 * (vertices + 2 * holes - 2)` index slots, so no count pass is needed
(`getPolygonTriangulationIndexCount(vertexCount, ringCount, polygonCount)` sizes the buffer), and indices
are rows of `positions`. Unused slots (dropped collinear or duplicate vertices, failed polygons) hold the
degenerate triangle `(first, first, first)`, so the whole buffer is drawable. `valid[p]` is 1 when the
triangulation completed and 0 for shells under 3 vertices, self-intersecting rings, scratch overflow or
exceeded `maximumWork` (the slice is then all degenerate). One invocation per polygon, so it scales with
polygon count, not ring size; there is no z-order hashing. Orientation tests use the exact predicate of
`GPUSegmentIntersection`. Every polygon must have a shell of at least 3 vertices and no empty rings, or
later slices shift. Triangle area equaled Shapely `area` and the triangle count equaled the `earcut` npm
package on 14 hand-built cases (order differs), and area matched on 60 random polygons. Delaunay variants
stay excluded.

### `GPULabelPoint`

A label point inside each polygon, the pole of inaccessibility (turf `pointOnFeature`, Shapely
`representative_point`, polylabel). Props: `positions`, `ringOffsets`, optional `featureRingOffsets`,
`initialGridSize` (16), `refinementGridSize` (8), `refinementRounds` (6), `refinementCandidates` (4) and
`output: {points, distances?, degenerate?}`. The rule is even-odd across rings. It starts from seeds
(scanline interval midpoints, guaranteed interior), then a coarse grid, then refines the best cells whose
bound can still win. `distances` is the inscribed radius and `degenerate` is 1 when no interior point
exists. It is planar and greedy: on long multi-lobed shapes it can miss the true pole (the worst ratio to
Shapely `polylabel` was above 0.97 in the random test), so raise `initialGridSize` or
`refinementCandidates`. Cost is sequential per feature, except that polygons with more than
`smallFeatureVertexLimit` vertices (default 32) use a workgroup path with bit-identical results.

### `GPUShapeDescriptors`

Shape indices per polygon (momepy compactness, QGIS shape metrics, esda `shape`). Props: `positions`,
`ringOffsets`, optional `featureRingOffsets`, `holeRule`, `parameters`
(`getGPUShapeDescriptorsParameterValues({sliverThreshold})`) and an `output` with any of `areas`,
`perimeters`, `polsbyPopper`, `schwartzberg`, `elongation`, `orientation`, `convexity`, `clockwise` and
`sliver`. It builds on `GPUGeometryMeasures` (area, perimeter, centroid) plus central second moments.
`elongation = 1 - sqrt(lambda2 / lambda1)`. `orientation` is the major axis in radians in `(-pi/2, pi/2]`.
`convexity = area / hull area` uses a private gift-wrapping hull (`O(n * h)` per feature). `clockwise` is
read from the first ring (y up), and `sliver` is set when Polsby-Popper is below the per-frame threshold.
Coordinates are planar and degenerate features give NaN.

### `GPULineDensity`

Line length per grid cell and its density (QGIS line density). Props: `positions`, `pathOffsets`,
`columns`, `rows`, `coordinateSystem`, `radius`, `maximumRecords`, `parameters`
(`getGPULineDensityParameterValues({minX, minY, cellWidth, cellHeight})`) and
`output: {lengths, densities?, overflow, totalRecords?}`. Segments are clipped to the grid and walked cell
by cell, and the lengths are summed per cell by a stable sort and a fixed-order reduction, so results are
bitwise reproducible. Spherical mode uses the great-circle meters of each clipped piece and exact
spherical cell areas for the densities. Exceeding `maximumRecords` sets `overflow` and under-counts the
last segments. A segment on a cell boundary goes to the upper or right cell, and closed rings are not
special (pass ring edges as lines). Length per polygon is
[`GPULineLengthPerPolygon`](#gpulinelengthperpolygon).

### `GPULineLengthPerPolygon`

Total line length inside each polygon (QGIS "Sum line lengths", PostGIS
`sum(ST_Length(ST_Intersection(line, polygon)))`). Props: `positions`, `pathOffsets`, `polygons` (the layout of
`GPUSpatialPredicateJoin`), optional `pathWeights`, `coordinateSystem`, `radius`, `maximumCandidatePairs`
(default `max(1024, 8 * positions.length)`), `maximumCrossings` (default 64) and
`output: {lengths, weightedLengths?, segmentCounts?, overflow}`. Every vertex row that is not the last of its path
is a segment. Segments meet polygons through `GPUSpatialJoinCandidates`, are cut at every ring-edge crossing, and
each piece whose midpoint is inside (even-odd over all rings, so holes and multipolygons work) is kept. Lengths
are summed per polygon by a stable sort and a fixed-order segmented sum, so results are bitwise reproducible
(segment counts use integer atomics). `weightedLengths` is `sum(length * pathWeight)` (weights are per path, not
per segment) and `segmentCounts` counts segments with positive length inside. `'spherical'` clips straight in
longitude/latitude, as `GPULineDensity` does, and measures each piece as the great-circle distance between its
endpoints. `overflow` is 1 when the candidate, BVH or crossing capacity was exceeded, and lengths can then be low.

Limits: pieces lying exactly along a ring edge are ambiguous and may or may not count; lines are not
unioned, so overlaps count once per line, as in QGIS; f32 precision (about 1e-6 of the segment length). The spec
checks 150 paths against 40 polygons against a float64 oracle at 5e-4 relative with exact counts, and spherical
lengths against a haversine oracle. No QGIS pin.

```ts
graph.add(new GPULineLengthPerPolygon({positions, pathOffsets, polygons, output: {lengths, segmentCounts, overflow}}));
```

### `GPUGridGenerator`

Grid geometry (turf `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid`, QGIS create grid, PostGIS
`ST_SquareGrid`). Props: `gridType` (`'square'`, `'hex'`, `'triangle'` or `'point'`), `columns`, `rows`,
`parameters` (`getGPUGridGeneratorParameterValues({minX, minY, cellWidth, cellHeight})`) and
`output: {positions?, centers?}`. Size the outputs with `getGPUGridCellCount` and
`getGPUGridVerticesPerCell`. Hexagons are pointy-top by default (`cellWidth` is flat to flat), triangles come in
alternating strips with half-cell row shifts (not turf's exact layout), and the point grid sits at cell
centers. The cell counts are compile-time.

- `hexOrientation: 'flat'` (GeoPandas `make_grid(flat_topped=True)`): `cellWidth` is still the
  flat-to-flat distance (GeoPandas `cell_size`), columns step `1.5 * R`, rows step `cellWidth`, odd
  columns shift up half a cell, and the ring is counter-clockwise from angle 0. The lattice is anchored at
  `(minX - R / 2, minY)`, where GeoPandas starts half a cell off the offset. Matched GeoPandas to 1e-5 on
  16 cells.
- `extent` and `output.intersects` (GeoPandas `make_grid(intersect=True)`): a per-cell `uint32` flag
  against a polygon (even-odd rings, holes, multi-polygon) with exact orientation predicates. Boundary
  contact counts, and point grids test the center. Cost is cells times extent edges. It is a flag, not a
  compaction: compact it with `GPUCompaction` for the subset. Matched Shapely `intersects` on five grid
  types (240 cells). Not provided: `make_grid` `corners`.

### `GPUShapeGenerator`

Circles, sectors and ellipses around per-feature centers and radii (turf `circle`, `sector`, `ellipse`).
Props include `shape` (`GPUShapeType`), `centers`, `radii`, optional sector `bearings` and ellipse
`rotations`, `coordinateSystem` (`'planar'`, or `'geodesic'` with spherical destination and radii in
meters; `GPUShapeCoordinateSystem`), compile-time `maximumSegments` and per-frame `parameters`
(`getGPUShapeGeneratorParameterValues`). The segment count and the radius scale are per-frame, up to
`maximumSegments`; size outputs with `getGPUShapeVertexCount` and `getGPUShapeMinimumSegments`. The output
is packed rings plus GeoArrow-style `offsets` (feature `f` starts at `f * V`) and an optional
`vertexCount`. Circles and sectors match turf 7.4 exactly (pinned). The ellipse spaces its vertices
uniformly in the parameter, not by arc length, and coincident sector bearings keep their center spokes.
Geodesic f32 coordinates round to about 0.5 m.

### `GPUHilbertKeys`

2D Hilbert curve index (order 1 to 16, `uint32`) of points, or of feature box centers (`minima` and
`maxima`), within per-frame `bounds` `[minX, minY, maxX, maxY]` (or the GPU-reduced bounds of the valid
items when omitted; items outside are clamped to the border cells). `output.keys` holds the keys,
bit-exact against the CPU `xy2d` on exactly representable data, and optional `sortedRows` and `sortedKeys`
give the stable curve-order permutation from a radix `GPUSort`. Invalid items get key `4^order`
(`getGPUHilbertInvalidKey(order)`, `0xffffffff` at order 16) and sort last. Cell borders depend on the f32
`(v - min) / extent`, and keys beyond order 16 (two words) are not provided.

The Hilbert order clusters better than Morton: a CPU tree-cost proxy (sum of node perimeters of a complete
binary BVH over 16,000 boxes) was 27 percent lower on uniform, 21 percent on clustered and 15 percent on
road-like data. The joins' `spatialSort` now uses it by default (see
[Spatial sort default](#spatial-sort-default)).

### `GPURectangleClip`

Clips lines or polygons to a rectangle (turf `bboxClip`, Shapely `clip_by_rect`, PostGIS `ST_ClipByBox2D`).
Props: `positions`, `geometryType` (`'lines'` or `'polygons'`), `pathOffsets`, `parameters`
(`getGPURectangleClipParameterValues({minX, minY, maxX, maxY})`) and `output: GPULinePathOutput`
(`positions`, `pathOffsets`, `count`, `overflow`, optional `totalCount`, `pathCount` and, for lines,
`sourcePaths`). Lines use Liang-Barsky, and the pieces are re-joined into paths, so a line that leaves and
re-enters becomes several paths. Polygons use Sutherland-Hodgman in four count, scan and emit stages, with
one output ring per input ring (possibly empty) and zero-width bridges for concave shapes (the area is
exact). The output capacity is `output.positions.length`, shared by all stages. Single-vertex paths are
dropped, degenerate bridges are not removed, and there is no spherical mode.

```ts
graph.add(new GPURectangleClip({positions, geometryType: 'polygons', pathOffsets: ringOffsets, parameters, output}));
viewportBuffer.write(getGPURectangleClipParameterValues({minX, minY, maxX, maxY}));
```

## Regression

### `GPUOrdinaryLeastSquares`

Ordinary least squares with an intercept and a full regression report, in one prebuilt graph:
coefficients, standard errors, t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC,
BIC, optional per-row residual and fitted columns, the Jarque-Bera normality test and the
Breusch-Pagan (Koenker) heteroskedasticity test.

```ts
const graph = new GPUCommandGraph(device);
graph.add(
  new GPUOrdinaryLeastSquares({
    predictors, // GraphDataView<'float32'>, row-major row * predictorCount + column
    response, // GraphDataView<'float32'>
    mask, // optional GraphDataView<'uint32'>; zero excludes the row
    parameters: ridgeBuffer.importToGraph(graph), // optional: [ridgeLambda]
    predictorCount: 3, // compile-time, 1 to 15; the intercept is added internally
    output: {coefficients, standardErrors, tStatistics, summary, status, residuals, fitted}
  })
);
```

- Rows with any non-finite predictor or response value, or a zero mask, are excluded.
- `coefficients`, `standardErrors`, `tStatistics`: `predictorCount + 1` floats, intercept first.
- `summary` (16 floats), indexed by `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*`: row count, R2, adjusted
  R2, sigma squared, log-likelihood, AIC, BIC, Jarque-Bera, its p-value, Breusch-Pagan, its p-value,
  RSS, TSS, skewness, kurtosis, ridge lambda.
- `status` (uint32): 0 ok, 1 singular or ill-conditioned (a predictor within `1 - 1e-6` of the span
  of the others, or constant), 2 too few rows (`n <= predictorCount + 1`). A failed fit leaves every
  statistic and per-row column NaN (the row count is still written).
- `residuals` and `fitted` are optional per-row columns (NaN for excluded rows).
- The optional ridge penalty `lambda >= 0` is added to the centered normal matrix of the
  predictors (never the intercept) and changes between frames without rebuilding the graph. Ridge
  standard errors are `sigma2 (X'X + lambda I)^-1` and ignore the shrinkage bias.

Numerics: no float atomics. Per-tile partial sums (one invocation per tile of consecutive rows, in
row order) are merged in tile order with Kahan compensation; means first, then centered
cross-products about those means; the solve runs on the correlation-scaled normal matrix with a
single-invocation Cholesky. The residual sum of squares comes from the residual pass, not from
`Syy - b'Sxy`, so it stays accurate when R2 is close to 1. Results are bitwise reproducible for a
given `tileRowCount` (default `max(64, ceil(rows / 4096))`).

AIC and BIC count `predictorCount + 1` parameters (GeoDa convention); the log-likelihood is
Gaussian with `sigma2 = RSS / n`. Jarque-Bera is `n/6 (S^2 + (K - 3)^2 / 4)` with p-value
`exp(-JB / 2)`. Breusch-Pagan is the Koenker studentized statistic `n R2` of the regression of `e^2`
on the predictors, with a chi-square p-value on `predictorCount` degrees of freedom computed by
the regularized upper incomplete gamma with a fixed iteration cap. `getChiSquareSurvival(value,
degreesOfFreedom)` is the float64 CPU mirror.

### `GPUGeographicallyWeightedRegression`

Geographically weighted regression with AICc bandwidth selection (Fotheringham, Brunsdon and
Charlton 2002; mgwr, ArcGIS GWR). For every included row `i` it fits a weighted least squares
`y ~ 1 + x` over all included rows `j` with weights `K(d_ij / h)` and writes local coefficients,
local R-squared, fitted values, residuals and the hat-matrix diagonal.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'gwr-parameters',
  format: 'float32',
  length: getGPUGeographicallyWeightedRegressionParameterLength(16)
});
graph.add(
  new GPUGeographicallyWeightedRegression({
    positions, // GraphDataView<'float32x2'>, planar
    predictors, // row-major row * predictorCount + column
    predictorCount: 2,
    response,
    parameters: parameterBuffer.importToGraph(graph),
    maximumBandwidthCount: 16,
    output: {coefficients, selectedBandwidth, summary}
  })
);
parameters.write(
  getGPUGeographicallyWeightedRegressionParameterValues(
    {kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [20, 40, 60, 80]},
    16
  )
);
```

- Kernels: `'gaussian'` `exp(-0.5 (d/h)^2)` and `'bisquare'` `(1 - (d/h)^2)^2` for `d < h`.
- Bandwidths: `'fixed'` (ladder values are distances) or `'adaptive'` (ladder values are neighbour
  counts `k`, `h` is 1.00001 times the distance to the k-th nearest included row, the row itself
  counted as in mgwr; `k` in `[2, maximumNeighborCount]`, at most 128).
- Bandwidth ladder: up to `maximumBandwidthCount` (at most 32) candidates in the parameter buffer.
  Each candidate's AICc, `n ln(RSS/n) + n ln(2 pi) + n (n + tr S) / (n - 2 - tr S)`, is evaluated on
  the GPU, the lowest wins (ties to the lowest index) and the outputs are written at that
  bandwidth. A ladder of one value is a fixed bandwidth. Candidates with any singular location,
  invalid `k` or `n - 2 - tr S <= 0` score NaN.
- Outputs: `coefficients` (`row * (predictorCount + 1) + column`, column 0 intercept), optional
  `localR2`, `fitted`, `residuals`, `hatDiagonal`, `localStatus` (0 ok, 1 singular, 2 excluded),
  `bandwidthScores`, `selectedBandwidth` (`[index, value]`), `localConditionNumber` and `summary` (RSS, trace of S, AICc,
  global R-squared, n, has-valid-candidate flag).
- `localConditionNumber` is the mgwr `local_collinearity` local condition number: the singular-value
  ratio of the kernel-weighted, column-normalized design (above 30 flags collinearity). It is pinned to
  mgwr 2.2.1 (bisquare and gaussian, fixed and adaptive) and is f32-accurate.
- Rows with a zero mask or a non-finite position, predictor or response are excluded as
  calibration locations and as neighbours.
- Numerics: the design is centred on the focal row and Jacobi-equilibrated before the shared
  Cholesky solve, so large coordinates or badly scaled predictors stay well conditioned in f32.
  A predictor that is constant over a neighbourhood makes that location singular.
- Cost: one thread per location scanning every row per candidate: `O(n^2 * ladder * p^2)`, at most
  65536 rows without the grid index (see `indexGridSize` below) (about 80 ms for 4096 rows and 8 candidates on the
  test adapter). Reductions use fixed-order 256-row tiles, so results are bitwise reproducible.
- Grid index: `indexGridSize` (default: automatic from 1024 rows; `false` disables) bins rows into a uniform grid so
  each location scans nearby cells only. The row cap rises from 65,536 to 1,048,576 with the grid, and
  `'bisquare'` fixed-bandwidth fits sum in cell order (about 1e-5 relative to the earlier row-order sums, still
  deterministic).

### `GPUGeographicallyWeightedRegressionNonstationarityTest`

Monte Carlo test of coefficient non-stationarity (mgwr `spatial_variability`, GWR4). It permutes the
observations among the locations with the Philox/Feistel bijection of the permutation contributors, refits
at the selected bandwidth and compares the standard deviation of each local coefficient with the observed
one. The pseudo p-value is `(g + 1) / (P + 1)`. Outputs are `table` (six floats per coefficient), `summary`
(`GPU_GWR_NONSTATIONARITY_SUMMARY`, `GPU_GWR_NONSTATIONARITY_TABLE`, `GPU_GWR_NONSTATIONARITY_TABLE_STRIDE`)
and an optional `standardDeviations` (the reference distribution). The per-permutation standard deviations
match mgwr 2.2.1 to 3e-15 (CPU oracle) and 2e-3 (GPU). The bandwidth stays at the value the regression
selected (mgwr re-selects it for every permutation). The cost is `O(P * n^2 * p^2)` with a full scan (no
grid index).

### `GPUSpatialRegressionDiagnostics`

Spatial diagnostics for an OLS fit (spreg `LMtests` and `MoranRes`): LM-lag, LM-error, robust LM-lag,
robust LM-error, LM-SARMA, and Moran's I of the residuals (mean, variance, z, two-sided normal p).

- Props: `weights` (a square `GPUSpatialWeights`; any sparsity pattern is accepted, directed kNN included,
  and row-standardized weights work; `W'` is formed with `GPUSpatialWeightsTranspose`),
  `predictors` (row-major `n x k`, no intercept), `response`, `residuals` (OLS residuals with intercept,
  for example `GPUOrdinaryLeastSquares` `output.residuals`; finite on every row), `predictorCount` (1 to 15),
  optional `tileRowCount` and `output: {tests, summary, status}`.
- `tests` (18 f32): six rows of `[statistic, df, p]`: LM_LAG, LM_ERROR, ROBUST_LM_LAG, ROBUST_LM_ERROR
  (df 1), LM_SARMA (df 2) and MORAN_RESIDUALS (the statistic is z, the df slot is 0, p is two-sided). The
  row index constants are `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_*`.
- `summary` (12 f32): `n`, `sigma2 = e'e / n`, `T = tr(W'W + WW)`, `J`, `S0`, Moran `I`, `E[I]`, `Var[I]`,
  `z`, `p`, `e'We` and `e'Wy` (`..._SUMMARY_*`).
- `status`: 0 ok, 1 singular, 2 too few rows, 3 degenerate weights, 4 non-finite input or zero residuals.
  Every output is NaN unless the status is 0. A masked OLS fit (NaN residuals) reports status 4.

Sums are fixed-order tile sums with a Kahan merge on centered columns, and the solve is a unit-diagonal
scaled Cholesky, so there are no atomics and no readback. Moran's variance and `J` are f32 with
cancellation (about 1e-2 relative on the variance and z). The statistics match spreg 1.9.1 (`OLS` with
`spat_diag` and `moran`) on a lattice, a symmetrized kNN scene and two directed row-standardized kNN scenes
(`k = 3`, 60 and 90 rows, asymmetric patterns) to 2e-3 relative. Earlier versions accumulated `W'Z` over each
row's own neighbors only, which biased the Moran variance for asymmetric patterns; that is fixed.

```ts
graph.add(new GPUOrdinaryLeastSquares({predictors, response, predictorCount: 2, output: {..., residuals}}));
graph.add(new GPUSpatialRegressionDiagnostics({
  weights, predictors, response, residuals, predictorCount: 2, output: {tests, summary, status}
}));
```

### `GPUSpatialTwoStageLeastSquares`

The spatial lag model `y = rho W y + X b + e` by two-stage least squares (spreg `GM_Lag` with `w_lags = 1`),
with instruments `[1, X, WX]`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 8),
optional `tileRowCount`, `instrumentOrder` and `output: {table, summary, status, residuals?}`.
`instrumentOrder: 2` (compile-time, default 1) adds `W^2 X` (spreg `w_lags=2`, instruments `[1, X, WX,
W^2 X]`), evaluated per row from the two-hop neighborhood with no extra buffers; it is pinned to spreg
1.9.1 `GM_Lag(w_lags=2)` on three scenes.

- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]` in spreg order: the
  intercept, then `X`, then `rho` last.
- `summary` (7): `n`, `sigma2 = u'u / n` (spreg `sig2n`, the `GM_Lag` default), `u'u`, the pseudo R2
  (squared correlation of `y` and `y - u`), Moran's `I` of `u`, the Anselin-Kelejian statistic and its
  p-value. The statistic follows spreg `akTest(case='gen')`; any sparsity pattern is accepted (`W'u` comes from
  `GPUSpatialWeightsTranspose`), and it matches spreg 1.9.1 on directed kNN scenes.
- `status`: 0 ok, 1 singular or ill-conditioned (including non-finite inputs), 2 too few rows, 3 non-finite
  sums.
- `residuals`: the structural residuals `u = y - Z delta`, which can be passed to later diagnostics.

The coefficients, standard errors, `sigma2`, pseudo R2 and Anselin-Kelejian statistic match spreg on a
144-row scene within 5e-3 (coefficients) to 2e-2 (z, AK). Robust and HAC variants,
`GM_Combo` and maximum-likelihood models are not provided. For the spatial error model see
[`GPUSpatialErrorGM`](#gpuspatialerrorgm).

### `GPUSpatialErrorGM`

The spatial error model `y = X b + u`, `u = lambda W u + e` by generalized moments (spreg 1.9.1 `GM_Error`,
Kelejian and Prucha 1998 and 1999, homoskedastic). Props: `weights` (any square CSR; no symmetry or
standardization needed), `predictors`, `response`, `predictorCount` (1 to 8), optional `tileRowCount` and
`output: {table, summary, status, residuals?}`.

- Steps: OLS of `y` on `[1, X]` gives `u`; the moments of `u`, `Wu` and `WWu` give `lambda` (bounded to
  `[-0.99, 0.99]`, `sigma2 >= 0`, spreg's `optim_moments` bounds); then OLS of `y - lambda W y` on the
  filtered `[1, X]` (the constant is filtered too, as `get_spFilter` does) gives the coefficients.
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then
  `lambda` last. The `lambda` row has NaN for se, z and p because `GM_Error` estimates no variance for it.
- `summary` (6, `GPU_SPATIAL_ERROR_GM_SUMMARY_*`): `n`, `lambda`, `sigma2 = e'e / n` (spreg `sig2n`), pseudo R2,
  the residual sum of squares and the moment objective at the solution.
- `status`: 0 ok, 1 singular (OLS or filtered fit), 2 too few rows (`n <= k + 2`), 3 non-finite.
- `residuals`: `u = y - X b`.

Accumulations use the fixed-order tile and Kahan scheme of `GPUOrdinaryLeastSquares`, so results are
deterministic. The spec matches spreg on a 10x10 lattice and an asymmetric kNN scene (`k = 4`, 80 rows):
`lambda` and the coefficients within 5e-3, standard errors within 1e-2 and z within 2e-2 (relative).
Deviation: spreg runs a local L-BFGS-B from `lambda = 0`; here one kernel scans the objective on a fixed grid
and bisects every bracketed minimum, returning the global minimum. The two agree whenever the objective has a
single basin. The `GM_Error_Het` and `GM_Error_Hom` variants are not provided.

```ts
graph.add(new GPUSpatialErrorGM({weights, predictors, response, predictorCount: 2, output: {table, summary, status}}));
```

## Recipes

Recipes are plain functions, not classes: `addXRecipe(graph, props)` constructs the contributors, calls
`graph.add` in chain order, wires outputs to inputs and returns `{contributors, ...named views}`. Views you
pass in (`zScores`, `bins`, `colors`, ...) are caller-owned and readable; every other intermediate is a
graph transient. Per-frame parameter views (`getGPU...ParameterValues`) are passed through, so changing them
never rebuilds the graph.

```ts
const hotSpots = addHotSpotAnalysisRecipe(graph, {
  source: {kind: 'points', positions, family: 'quadbin', resolution: 8, tableCapacity: 4096,
    neighborCapacity: 49152, gridSize: [64, 64], neighborSearchParameters},
  parameters: autocorrelationParameters, zScores, bins,
  permutation: {parameters: permutationParameters, maximumPermutations: 199},
  color: {classBreaksParameters, maximumClassCount: 5, colorScaleParameters, palette, maximumPaletteCount: 5}
});
```

| Recipe | Chain |
| --- | --- |
| `addHotSpotAnalysisRecipe` | (GPUPointToCell) -> GPUCellAggregation -> GPUCellGeometry centers -> GPUNeighborSearch (or GPULatticeWeights for raster source) -> GPUHotSpotAnalysis -> GPULocalPermutationTest -> GPUClassBreaks -> GPUColorScale |
| `addRateClusterMapRecipe` | GPUEmpiricalBayesRates -> GPUContiguityWeights -> GPUSpatialWeightsTransform(row) -> GPULocalMoran (`quadrantGating: 'none'` with permutation) -> GPULocalPermutationTest -> quadrant colors |
| `addPointsInPolygonsChoroplethRecipe` | backend 'zonal' (GPUZonalStatistics) or 'group' (GPUPointInPolygonJoin -> dense GPUGroupStatistics) -> GPUClassBreaks -> GPUColorScale |
| `addSpaceTimeHotSpotsRecipe` | GPUCalendarBuckets -> key adapter -> GPUGroupStatistics -> dense cube -> GPUEmergingHotSpots -> ordinal GPUColorScale |
| `addClusterAndOutlineRecipe` | GPUSpatialClustering -> GPUGroupGeometry + GPUGroupConvexHull -> GPUGeometryMeasures |
| `addSpatialRegressionRecipe` | GPUOrdinaryLeastSquares -> GPUSpatialRegressionDiagnostics + residual GPULocalMoran (+ optional GPUGeographicallyWeightedRegression) |
| `addDriveTimeCatchmentRecipe` | GPUNetworkSnapping -> GPUNetworkServiceAreas -> (GPUNetworkIsochrones raster and/or cell outline with rings -> GPUPointInPolygonJoin demand join) -> snap demand -> band adapters -> GPUGroupStatistics |
| `addStraightLineCatchmentsRecipe` | GPUDistanceField allocation -> GPURasterZonalStatistics |
| `addChangeOfSupportRecipe` | 2x GPUPolygonRasterization -> GPUArealInterpolation -> GPUSpatialLag |
| `addFleetDwellRecipe` | GPUTrajectoryMetrics stops -> GPUPointInPolygonJoin -> stop mask -> GPUGroupStatistics (dense, one row per zone) |
| `addFleetDwellZoneEventsRecipe` | GPUZoneEvents -> GPUGroupStatistics (dense, one row per zone; zone-event variant of the fleet dwell recipe) |
| `addPeriodComparisonRecipe` | 2x GPUCellAggregation -> GPUCellTableCompare -> union mask -> GPUClassBreaks -> GPUColorScale |

Notes:

- The space-time recipe aggregates `cell * sliceCount + slice` keys with `GPUGroupStatistics` into a dense
  cube, because `GPUCellAggregation` keys must be valid Quadbin or H3 cells.
- The drive-time catchment recipe classifies demand in network space (the minimum over the snapped edge
  ends) and also returns the isochrone triangles for drawing. It assumes both directions are in the CSR.
  `isochrones` takes `raster` and/or `cellOutline` (at least one). `cellOutline.rings` returns closed
  isochrone rings in `result.isochroneRings`. `isochrones.joinDemand` (`candidateCapacity` and the
  optional `pointFeatureIds`, `overflow`, `uncertainCount`) joins the demand points to the ring polygon (it needs
  `cellOutline.rings.output.polygons`) with `GPUPointInPolygonJoin`, and `result.isochroneDemand.pointFeatureIds`
  is 0 inside and `GPU_SPATIAL_JOIN_NO_FEATURE` outside. That answers "inside the isochrone at
  `cellCostLimit`" at cell resolution (the union of cells that hold reached nodes), where the band
  classification stays exact at the snapped edge. Isoband triangles cannot feed the join, because their
  topology is GPU-written and has no ring structure; the cell rings can. Sparse networks drop cells, so the
  polygon join is coarser than the network-space bands.
- The rate cluster map recipe runs `GPULocalMoran` with `quadrantGating: 'none'` when `permutation` is
  given and keeps the quadrant where the permutation `significant` mask is set; without `permutation` the
  quadrant is gated by the analytic p-value. `analyticQuadrants` returns the local Moran quadrants (gated
  analytically without `permutation`, ungated with it).
- The points-in-polygons choropleth recipe's `'group'` back-end uses dense `GPUGroupStatistics`
  (`keyCount: featureCount`, so empty polygons keep a row), writing `featureValues` directly. For
  `statistic: 'count'`, `featureValues` is the `uint32` counts view, which `GPUClassBreaks` and
  `GPUColorScale` read as numbers (`integerValues: 'numeric'`); the `featureValues` prop is rejected for
  `'count'`, and the result type is `float32` or `uint32`.
- The spatial regression recipe requires a finite residual on every row; the weights pattern may be asymmetric.
- The cluster recipe drops clusters at or above `maximumClusterCount`, and the caller supplies the
  geometry parameters. The change-of-support recipe's `overflow` covers the overlap list only; the raster
  flags are returned as `rasterOverflow`.

## Cross-reference

See the [task-organized cross-reference](/docs/api-reference/experimental/gpu-spatial-analysis-cross-reference).

Contributors live in `@luma.gl/experimental/gpu-spatial-analysis` unless the entry says
`(network)`, `(raster)` or `(dataframe)`, meaning `gpu-network`, `gpu-raster` and `gpu-dataframe`.
Recipes are the `add...Recipe` builder functions. "Not built" rows point at the library to use
instead (see the GPU Core maintainer roadmap). Every contributor runs on the GPU with
bounded outputs and does not read back; the CPU libraries return unbounded geometry.

### turf

| turf | luma.gl |
| --- | --- |
| `area`, `length`, `centroid`, `bbox`, `center` | `GPUGeometryMeasures` (area, length or perimeter, centroid, bounds per feature and per group) |
| `centerOfMass` | `GPUGeometryMeasures` centroid; `GPUGroupGeometry` for weighted centers of label groups |
| `pointOnFeature` | `GPULabelPoint` (pole of inaccessibility) |
| `distance`, `bearing`, `midpoint`, `rhumbDistance`, `rhumbBearing` | `GPUGeodesicPairs` (`model: 'rhumb'` for the rhumb forms) |
| `destination`, `rhumbDestination` | `GPUGeodesicDestination` |
| `greatCircle` | `GPUGreatCircleArcs` |
| `along`, `lineSliceAlong` | `GPULineLocate` |
| `lineChunk`, `lineSegment` | `GPULineChunk`, `GPULineSegmentize` |
| `nearestPointOnLine` | `GPULinearReferencing` |
| `nearestPoint` | `GPUNearestFeatureJoin`; `GPUNeighborSearch` (knn) |
| `pointsWithinPolygon`, `tag` | `GPUPointInPolygonJoin`; `GPUZonalStatistics` for counts and sums |
| `collect` | `GPUZonalStatistics`, `GPUGroupStatistics` (dataframe) |
| `buffer` | `GPUOutlineGeometry` (render-only buffer geometry); `GPUBufferSelection` for "within distance" selection |
| `simplify` | `GPULineSimplification`; `GPUCoverageSimplification` when neighbors must stay gap-free |
| `polygonSmooth`, `bezierSpline` | `GPULineSmooth` (Chaikin) |
| `bboxClip` | `GPURectangleClip` |
| `kinks`, `booleanValid` | `GPUGeometryValidity`; `GPUSegmentIntersection` for the crossing points |
| `lineIntersect` | `GPUSegmentIntersection` |
| `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid` | `GPUGridGenerator` |
| `interpolate`, `idw` | `GPUInverseDistanceWeighting`; `GPUVariogram` to choose a model |
| `isolines`, `isobands` | `GPUIsolines`, `GPUIsobands` (raster) |
| `convex` | `GPUGroupConvexHull`; recipe `addClusterAndOutlineRecipe` |
| `clustersDbscan` | `GPUSpatialClustering` |
| `clustersKmeans` | `GPUKMeans` |
| `randomPoint` inside polygons, `sample` | `GPURandomPointsInPolygon`, `GPUDotDensity` |
| `hexbin`-style counting, `pointsWithinPolygon` + count | `GPUPointDensity`; `GPUCellAggregation` for H3 or Quadbin cells |
| `standardDeviationalEllipse`, `centralMean`, `medianCenter` (turf-extra) | `GPUGeographicDistribution`, `GPUGroupGeometry` |
| `union`, `intersect`, `difference`, `dissolve`, `voronoi`, `tin`, `concave` | Not built, except `GPUCoverageDissolve` (a valid coverage by label) and `GPULineClipByPolygon` (lines only). Use polyclip-ts, JSTS, d3-delaunay; raster Voronoi is `GPUDistanceField` (raster); boundaries between groups are `GPUCellSetOutline` |
| `polygonize` | `GPUSegmentRingAssembly` for already-noded directed segments (cell outlines, isochrone outlines): closed rings with shells and holes. Crossing or unnoded lines are not built (use JSTS) |

### PostGIS

| PostGIS | luma.gl |
| --- | --- |
| `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid` | `GPUGeometryMeasures` |
| `ST_PointOnSurface`, `ST_MaximumInscribedCircle` | `GPULabelPoint` |
| `ST_Distance(geography)`, `ST_Azimuth` | `GPUGeodesicPairs` |
| `ST_Project` | `GPUGeodesicDestination` |
| `ST_DWithin` | `GPUNeighborSearch` (radius), `GPUBufferSelection`, `GPUSpatialPredicateJoin` with a distance |
| `ST_Within`, `ST_Contains`, `ST_Intersects`, `ST_Covers`, `ST_Touches` joins | `GPUPointInPolygonJoin` (points), `GPUSpatialPredicateJoin` (features); `GPUSpatialJoinCandidates` for the bounding-box stage alone |
| `ST_Distance` KNN (`<->`) | `GPUNeighborSearch` (knn), `GPUNearestFeatureJoin` |
| `ST_ClusterDBSCAN` | `GPUSpatialClustering` |
| `ST_ClusterKMeans` | `GPUKMeans` |
| `ST_ConvexHull` per group | `GPUGroupConvexHull` |
| `ST_Segmentize` | `GPULineSegmentize` |
| `ST_Simplify`, `ST_SimplifyVW` | `GPULineSimplification` (`method: 'visvalingam'` is approximate) |
| `ST_CoverageSimplify` | `GPUCoverageSimplification` |
| `ST_IsValid`, `ST_IsValidDetail` | `GPUGeometryValidity` (polygons, lines, points) |
| `ST_IsSimple`, `ST_IsRing`, `ST_IsClosed`, `ST_NPoints`, `ST_NumInteriorRings`, `ST_NumGeometries` | `GPUGeometryPredicates` |
| `ST_MinimumClearance`, `ST_MinimumClearanceLine` | `GPUMinimumClearance` |
| `ST_MinimumBoundingCircle`, `ST_OrientedEnvelope`, `ST_LongestLine`, `ST_MaxDistance` | `GPUMinimumBounds` |
| `ST_Affine`, `ST_Translate`, `ST_Scale`, `ST_Rotate`, `ST_Reverse`, `ST_ForcePolygonCW`, `ST_RemoveRepeatedPoints`, `ST_SnapToGrid` | `GPUAffineTransform`, `GPUGeometryOrientation`, `GPUGeometryCleanup` |
| `ST_OffsetCurve`, `ST_Snap` | `GPUOffsetCurve` (no loop removal), `GPUVertexSnap` (vertex to vertex) |
| `ST_TriangulatePolygon` | `GPUPolygonTriangulation` (earcut, not constrained Delaunay) |
| `ST_CoverageUnion`, `ST_CoverageInvalidEdges`, `ST_CoverageIsValid` | `GPUCoverageDissolve`, `GPUCoverageValidity` |
| `ST_LocalOutlierFactor` (Sedona) | `GPULocalOutlierFactor` |
| `ST_ClipByBox2D` | `GPURectangleClip` |
| `ST_LineInterpolatePoint`, `ST_LocateAlong` | `GPULineLocate` |
| `ST_LineLocatePoint`, `ST_ClosestPoint` on lines | `GPULinearReferencing` |
| `ST_SquareGrid`, `ST_HexagonGrid` | `GPUGridGenerator` |
| `ST_HausdorffDistance`, `ST_FrechetDistance` | `GPUTrackSimilarity` |
| `ST_Length(ST_Intersection(line, cell))` summed per cell | `GPULineDensity` |
| `ST_Buffer` (for drawing) | `GPUOutlineGeometry` |
| `ST_Intersection`, `ST_Union`, `ST_Difference`, `ST_Buffer` (as geometry), `ST_Voronoi*`, `ST_Delaunay*`, `ST_MakeValid`, `ST_Subdivide` | Not built (unbounded output). Use GEOS-wasm or JSTS |
| `ST_Polygonize`, `ST_BuildArea`, `ST_MakePolygon` (from boundary segments or closed lines) | `GPUSegmentRingAssembly` chains directed, already-noded segments into closed rings with shell and hole classification (`output.polygons` is GeoArrow polygon layout). Noding of crossing lines is not built |
| `ST_Clip`, `ST_MapAlgebra`, `ST_Reclass` (raster) | `GPURasterBandMath`, `GPURasterReclassify`, `GPURasterConditional` (raster) |
| `ST_Slope`, `ST_Aspect`, `ST_Hillshade` | `GPURasterGradient`, `GPURasterGradientMagnitude`, `GPURasterSobel` (raster) |
| `ST_Contour`, `ST_DumpAsPolygons` on thresholds | `GPUIsolines`, `GPUIsobands`, `GPURasterContours` (raster) |
| `ST_ValueCount`, `ST_Histogram` | `GPURasterHistogram`, `GPURasterStatistics` (raster) |
| `ST_SummaryStats` per polygon (zonal) | `GPURasterZonalStatistics` (raster) |
| `ST_AsRaster` | `GPUPolygonRasterization` (raster) |
| `pgRouting` `pgr_dijkstraCost`, `pgr_drivingDistance` | `GPUNetworkReachability`, `GPUNetworkCostMatrix`, `GPUNetworkIsochrones` (network, `cellOutline.rings` for isochrone polygons); recipe `addDriveTimeCatchmentRecipe` (`isochroneRings`, demand join) |
| `pgr_withPointsDD` (points on edges) | `GPUNetworkSnapping` (network) |
| `pgr_connectedComponents`, `pgr_pagerank`, `pgr_degree` | `GPUNetworkAnalyticsColumns`, `GPUNetworkStatistics` (network) |
| `h3_lat_lng_to_cell`, `h3_cell_to_boundary`, `h3_grid_disk`, `h3_polygon_to_cells` | `GPUPointToCell`, `GPUCellGeometry`, `GPUCellTopology`, `GPUCellCover` |
| `h3_cells_to_multi_polygon_wkb`, h3-js `cellsToMultiPolygon` | `GPUCellSetOutline` with `rings` (`GPUSegmentRingAssembly`): closed shells and holes of an H3 or Quadbin cell set on the GPU, with `ringShells` and polygon layout |
| `h3` roll-up `GROUP BY h3_cell_to_parent` | `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid` |

### GeoPandas and Shapely

| GeoPandas / Shapely | luma.gl |
| --- | --- |
| `GeoSeries.area`, `.length`, `.centroid`, `.bounds` | `GPUGeometryMeasures` |
| `GeoSeries.representative_point()` | `GPULabelPoint` |
| `GeoSeries.distance`, `pyproj.Geod.inv` / `.fwd` | `GPUGeodesicPairs`, `GPUGeodesicDestination` |
| `GeoDataFrame.sjoin` | `GPUSpatialPredicateJoin`, `GPUPointInPolygonJoin` |
| `GeoDataFrame.sjoin_nearest` | `GPUNearestFeatureJoin` |
| `sjoin` then `groupby(...).agg` | recipe `addPointsInPolygonsChoroplethRecipe`; `GPUZonalStatistics`; `GPUGroupStatistics` (dataframe) |
| `GeoSeries.convex_hull` per `groupby` | `GPUGroupConvexHull`; recipe `addClusterAndOutlineRecipe` |
| `GeoSeries.simplify`, `shapely.simplify_coverage` | `GPULineSimplification`, `GPUCoverageSimplification` |
| `GeoSeries.is_valid`, `shapely.explain_validity` | `GPUGeometryValidity` |
| `shapely.segmentize` | `GPULineSegmentize` |
| `shapely.clip_by_rect`, `GeoDataFrame.clip` (rectangle) | `GPURectangleClip` |
| `shapely.line_interpolate_point`, `.project` | `GPULineLocate`, `GPULinearReferencing` |
| `shapely.hausdorff_distance`, `.frechet_distance` (with `densify`) | `GPUTrackSimilarity` |
| `shapely.set_precision` (pointwise), `remove_repeated_points`, `reverse`, `orient_polygons`, `affinity.*` | `GPUGeometryCleanup`, `GPUGeometryOrientation`, `GPUAffineTransform` |
| `GeoSeries.buffer` (as geometry) | Not built |
| `shapely.offset_curve`, `snap` | `GPUOffsetCurve`, `GPUVertexSnap` (partial, see their sections) |
| `shapely.minimum_rotated_rectangle`, `oriented_envelope`, `minimum_bounding_circle`, `minimum_bounding_radius` | `GPUMinimumBounds` |
| `shapely.minimum_clearance`, `minimum_clearance_line` | `GPUMinimumClearance` |
| `shapely.is_simple`, `is_ring`, `is_closed`, `is_ccw`, `get_num_points`, `extract_unique_points`, `equals_exact` | `GPUGeometryPredicates` |
| `GeoSeries.dissolve(method='coverage')`, `shapely.coverage_union_all`, `coverage_is_valid`, `coverage_invalid_edges` | `GPUCoverageDissolve`, `GPUCoverageValidity` |
| `geopandas.clip(lines, polygon)`, `shapely.shared_paths` | `GPULineClipByPolygon`, `GPUSharedPaths` |
| `sjoin` merge, `sjoin_nearest(distance_col)`, `explode`, `get_parts`, `.cx` | `GPUPairGather`, `GPUOffsetExpansion`, `GPUBoundsFilter` |
| `GeoSeries.make_grid`, `sample_points` (lines) | `GPUGridGenerator` (`hexOrientation`, `extent`), `GPURandomPointsOnLine` |
| `overlay`, `dissolve`, `unary_union`, `voronoi_polygons`, `delaunay_triangles`, `concave_hull`, `make_valid` | Not built; see turf/PostGIS notes. `dissolve(method='coverage')` is `GPUCoverageDissolve`, and `clip` of lines by a polygon is `GPULineClipByPolygon` |
| `shapely.polygonize`, `polygonize_full` | `GPUSegmentRingAssembly` for already-noded segments (no noding, no cut edges or dangles output; open chains are counted in `openSegmentCount`) |
| `shapely.minimum_bounding_circle`, `oriented_envelope` | `GPUShapeDescriptors` (elongation, orientation); `GPUGroupGeometry` (bounds, ellipse) |
| `tobler.area_interpolate` | `GPUArealInterpolation`; recipe `addChangeOfSupportRecipe` |
| `tobler.pycno` | `GPUPycnophylactic` |
| `rasterstats.zonal_stats` | `GPURasterZonalStatistics` (raster); recipe `addStraightLineCatchmentsRecipe` |
| `rasterio.features.rasterize` | `GPUPolygonRasterization` (raster) |
| `momepy` compactness (`circular_compactness`, `convexity`, `elongation`, `orientation`) | `GPUShapeDescriptors` |
| `momepy.describe_agg`, `libpysal.weights.lag_categorical` | `GPUNeighborhoodSummary` |
| `mapclassify.classify` (`Quantiles`, `EqualInterval`, `FisherJenks`, `HeadTailBreaks`, `BoxPlot`, `StdMean`, `MaximumBreaks`, `UserDefined`) | `GPUClassBreaks` (dataframe); `GPUClassAssignment`; `GPUColorScale` (dataframe) |
| `mapclassify` bivariate / `geoplot` choropleth | `GPUBivariateClassification` (dataframe) |
| `sklearn.cluster.DBSCAN`, `KMeans` on coordinates | `GPUSpatialClustering`, `GPUKMeans` |
| `movingpandas` stop detection, trajectory stats | `GPUTrajectoryMetrics`; `GPUZoneEvents` for zone entry/exit; recipe `addFleetDwellRecipe` |
| `movingpandas` interpolation / resample | `GPUTrajectoryPlayhead`, `GPUTrajectoryResample` |
| `gstat` / `scikit-gstat` `Variogram` | `GPUVariogram` (dataframe) |

### PySAL

| PySAL | luma.gl |
| --- | --- |
| `libpysal.weights.Queen`, `Rook` | `GPUContiguityWeights` |
| `libpysal.weights.lat2W` | `GPULatticeWeights` |
| `KNN`, `DistanceBand`, `Kernel`, `Kernel(fixed=False)` | `GPUNeighborSearch` (knn, radius, kernel weights); `GPUSpatialWeightsTransform` (kernel) |
| `W.transform = 'R'` / `'B'`, symmetrize | `GPUSpatialWeightsTransform` |
| `weights.set_operations` (`w_union`, `w_intersection`), `higher_order`, `w_subset`, `block_weights` | `GPUSpatialWeightsAlgebra` |
| `W.s0`, `.s1`, `.s2`, `.cardinalities`, `.islands`, `.asymmetry()` | `GPUSpatialWeightsSummary` |
| `lag_spatial` | `GPUSpatialLag` |
| `esda.Moran`, `Geary`, `G` (General G), `Moran_BV`, `Join_Counts` | `GPUGlobalSpatialStatistics` |
| `esda.Moran` `.p_sim`, `.z_sim` | `GPUGlobalPermutationTest` |
| `esda.Moran_Local` | `GPULocalMoran`; `GPULocalPermutationTest` for `p_sim` |
| `esda.G_Local` (Gi, Gi*) | `GPUHotSpotAnalysis`; `GPULocalPermutationTest` (`'localG'`, `'localGStar'`); recipe `addHotSpotAnalysisRecipe` |
| `esda.Moran_Rate`, `Moran_Local_Rate`, `smoothing.Empirical_Bayes`, `assuncao_rate` | `GPUEmpiricalBayesRates`; recipe `addRateClusterMapRecipe` |
| `esda.fdr` | `falseDiscoveryRate` option of `GPUHotSpotAnalysis`, `GPULocalMoran`, `GPULocalPermutationTest` |
| `pointpats.PointPattern`, `ripley.k_test`, `f_test`, `g_test`, `j_test` | `GPUPointPatternIndices`, `GPURipley`, `GPURipleyDistanceFunctions` (dataframe) |
| `pointpats.Knox`, `Mantel` | `GPUKnoxTest`, `GPUMantelTest` |
| `esda.Moran` correlogram (`spatial_correlogram`) | `GPUSpatialCorrelogram` (dataframe) |
| `esda.Spatial_Pearson`, `Local_Geary`, `Gamma` | Not built |
| `spreg.OLS` | `GPUOrdinaryLeastSquares` |
| `spreg.diagnostics_sp.LMtests`, `MoranRes` | `GPUSpatialRegressionDiagnostics`; recipe `addSpatialRegressionRecipe` |
| `spreg.GM_Lag` | `GPUSpatialTwoStageLeastSquares` |
| `spreg.ML_Lag`, `ML_Error`, panel, SUR, regimes, probit | Not built (likelihood-based, little value on a web map) |
| `mgwr.GWR` | `GPUGeographicallyWeightedRegression`; `MGWR` backfitting is not built |
| `giddy.Markov`, `LISA_Markov`, `Spatial_Markov` | `GPUTransitionMatrix`, `GPULISAMarkov`, `GPUSpatialMarkov` |
| `giddy.directional`, `Rose` | Not built |
| `segregation` aspatial and spatial indices | `GPUSegregation` |
| `inequality.gini.Gini`, `theil.Theil`, `Gini_Spatial` | `GPUInequality` (dataframe) |
| `esda.shape` (`isoperimetric_quotient`, `convex_hull_ratio`) | `GPUShapeDescriptors` |
| `spopt.region.AZP`, `MaxPHeuristic`, `Skater`, `Ward` | Not built (search problems) |
| `spopt.locate.PMedian`, `MCLP`, `LSCP` | Not built; cost matrices come from `GPUNetworkCostMatrix` (network) |
| `access.Access.two_step_fca`, `weighted_catchment`, `Huff` | `GPUCatchmentAccessibility`, `GPUHuffTradeAreas`, `GPUNetworkAccessibility` (network) |
| `spaghetti`, `pandana` network distance and accessibility | `GPUNetworkReachability`, `GPUNetworkCostMatrix`, `GPUNetworkAccessibility` (network) |
| `PySAL` hot spot workflow (Gi* + FDR + map) | recipe `addHotSpotAnalysisRecipe` |

### QGIS

| QGIS tool | luma.gl |
| --- | --- |
| Create grid (rectangle, hexagon) | `GPUGridGenerator` |
| Heatmap (kernel density) | `GPUPointDensity` |
| Count points in polygon, Zonal statistics (vector) | `GPUZonalStatistics`; recipe `addPointsInPolygonsChoroplethRecipe` |
| Join attributes by location, Join attributes by location (summary) | `GPUSpatialPredicateJoin`, `GPUPointInPolygonJoin`, `GPUZonalStatistics` |
| Join attributes by nearest | `GPUNearestFeatureJoin` |
| Extract by location, Select by location | `GPUSpatialPredicateJoin`, `GPUBufferSelection`, `GPURegionMask` |
| Zonal statistics (raster) | `GPURasterZonalStatistics` (raster) |
| Rasterize (vector to raster) | `GPUPolygonRasterization` (raster) |
| Raster calculator | `GPURasterBandMath`, `GPURasterArithmetic` (raster) |
| Reclassify by table / layer | `GPURasterReclassify` (raster) |
| Slope, Aspect, Hillshade, TRI | `GPURasterGradient`, `GPURasterSobel`, `GPURasterScharr` (raster) |
| Contour, Contour polygons | `GPUIsolines`, `GPUIsobands` (raster) |
| Sieve, Fill / clump | `GPURasterSieve`, `GPURasterConnectedComponents` (raster) |
| Proximity (raster distance), Voronoi (raster) | `GPUDistanceField` (raster); recipe `addStraightLineCatchmentsRecipe` |
| Cost distance / Least-cost path | `GPUCostDistance`, `GPUCostDistancePath` (raster) |
| IDW interpolation | `GPUInverseDistanceWeighting` |
| Service area (from point), Service area (from layer) | `GPUNetworkReachability`, `GPUNetworkIsochrones` (network, ring polygons through `cellOutline.rings`); recipe `addDriveTimeCatchmentRecipe` |
| Polygonize (from already-noded segments) | `GPUSegmentRingAssembly` |
| Shortest path (point to point / layer) | `GPUNetworkReachability` with `GPUNetworkPathExtraction` (network) |
| Distance matrix | `GPUNeighborSearch`; `GPUNetworkCostMatrix` for network cost |
| Nearest neighbour analysis | `GPUPointPatternIndices` (dataframe) |
| Random points in polygons | `GPURandomPointsInPolygon`, `GPUDotDensity` |
| Pole of inaccessibility | `GPULabelPoint` |
| Polygon centroids, Centroids | `GPUGeometryMeasures` |
| Mean coordinate(s), Standard deviational ellipse (Processing / plugin) | `GPUGeographicDistribution`, `GPUGroupGeometry` |
| Convex hull, Minimum bounding geometry | `GPUGroupConvexHull`, `GPUGroupGeometry` |
| Simplify, Smooth, Densify by interval | `GPULineSimplification`, `GPULineSmooth`, `GPULineSegmentize` |
| Clip by extent | `GPURectangleClip` |
| Line intersections | `GPUSegmentIntersection` |
| Check validity | `GPUGeometryValidity` |
| Line density, Sum line lengths | `GPULineDensity` |
| DBSCAN clustering, K-means clustering | `GPUSpatialClustering`, `GPUKMeans` |
| Cluster-and-outline workflow (cluster, hull, area) | recipe `addClusterAndOutlineRecipe` |
| Moran's I, Getis-Ord Gi*, LISA (SAGA, GeoDa plugin) | `GPUGlobalSpatialStatistics`, `GPUHotSpotAnalysis`, `GPULocalMoran` |
| Graduated symbology modes (Quantile, Natural Breaks, Pretty, Std Dev, Equal) | `GPUClassBreaks`, `GPUColorScale` (dataframe) |
| Spatial index, Build virtual raster, Merge, Dissolve, Union, Intersection, Difference, Buffer (geometry), Delaunay, Voronoi (vector), Fix geometries | Not built (see the maintainer roadmap) |
| Model Designer / Processing graphical model | Recipes, or compose contributors in one `GPUCommandGraph` |
| ArcGIS Hot Spot Analysis (Getis-Ord Gi*), Emerging Hot Spot Analysis | `GPUHotSpotAnalysis`, `GPUEmergingHotSpots`; recipes `addHotSpotAnalysisRecipe`, `addSpaceTimeHotSpotsRecipe` |
| ArcGIS Spatial Autocorrelation, Incremental Spatial Autocorrelation | `GPUGlobalSpatialStatistics`, `GPUSpatialCorrelogram` (dataframe) |
| ArcGIS Directional Distribution, Mean Center, Median Center | `GPUGeographicDistribution` |
| ArcGIS Generate Network Spatial Weights, Generate Spatial Weights Matrix | `GPUContiguityWeights`, `GPUNeighborSearch`, `GPUNetworkCostMatrix` |
| ArcGIS Geographically Weighted Regression, Ordinary Least Squares | `GPUGeographicallyWeightedRegression`, `GPUOrdinaryLeastSquares` |
| ArcGIS Calculate Composite Index | `GPUCompositeScore` (dataframe) |

## See also

- [Geospatial kernels](/docs/api-reference/experimental/geospatial) for the projection, distance, point-in-polygon, grid-index and point-query primitives these contributors compose
- [GPUCommandGraph](/docs/api-reference/experimental/gpu-core/gpu-command-graph)
- [GPU Network](/docs/api-reference/experimental/gpu-network) and [GPU Terrain](/docs/api-reference/experimental/gpu-terrain) for the network and elevation analysis contributors
- [GPU Dataframe analysis contributors](/docs/api-reference/experimental/gpu-dataframe-analysis) for the column statistics, joins, and time filters the spatial contributors consume
- [GPUGridIndex](/docs/api-reference/experimental/gpu-core/gpu-grid-index)
