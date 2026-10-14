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
| Spatial weights | `GPUNeighborSearch`, `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsTransform`, `GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsSummary`, `GPUSpatialWeightsTranspose`, `GPUSpatialWeightsValidation`, `GPUSpatialLag`, `GPUMapColoring` |
| Spatial statistics | `GPUGlobalSpatialStatistics`, `GPUMultipleTestingAdjustment`, `GPUAssuncaoReisStandardization`, `GPUHotSpotAnalysis`, `GPULocalMoran`, `GPULocalGeary`, `GPUSpatialPearson`, `GPUGammaStatistic`, `GPURoseStatistic`, `GPULocalPermutationTest`, `GPUGlobalPermutationTest`, `GPUNeighborhoodSummary`, `GPUEmpiricalBayesRates`, `GPUSpatialEmpiricalBayesRates`, `GPUSpatialClustering`, `GPUKMeans`, `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, `GPUAZPRegions`, `GPUMaxPRegions`, `GPUWardRegions`, `GPULocationAllocation`, `GPURegionPartitionEvaluation`, `GPUGroupGeometry`, `GPUGroupConvexHull`, `GPUEmergingHotSpots`, `GPUGeographicDistribution`, `GPURegionStatistics`, `GPUZonalStatistics`, `GPUFocalStatistics`, `GPUSimilarLocations` |
| Distribution dynamics and space-time tests | `GPUClassAssignment`, `GPUClassificationFit`, `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov`, `GPUTransitionDiagnostics`, `GPUKnoxTest`, `GPUMantelTest`, `GPUSpatialScanStatistic` |
| Accessibility and change of support | `GPUCatchmentAccessibility`, `GPUHuffTradeAreas`, `GPUArealInterpolation`, `GPUPycnophylactic`, `GPUSegregation` |
| Spatial joins | `GPUSpatialPredicateJoin`, `GPUSpatialJoinPrepared`, `GPUSpatialJoinCandidates`, `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin`, `GPUNearestFeatureWeights`, `GPUBufferSelection` |
| Geometry validity and topology | `GPUSegmentIntersection`, `GPUGeometryValidity`, `GPULineNoding`, `GPUPolygonize`, `GPUMakeValid`, `GPUPolygonOverlay`, `GPUBufferSurface` |
| Density & interpolation | `GPUPointDensity`, `GPUKernelDensity`, `GPUInverseDistanceWeighting`, `GPUKriging`, `GPUDotDensity`, `GPURandomPointsInPolygon` |
| Cells (DGGS) | `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid`, `GPUCellLevelSelection`, `GPUPointToCell`, `GPUCellGeometry`, `GPUCellTopology`, `GPUCellCompaction`, `GPUCellCover`, `GPUCellSetOutline`, `GPUSegmentRingAssembly`, `GPUCellTableCompare` |
| Lines & trajectories | `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`, `GPULineSplit`, `GPULineMerge`, `GPUGeometryMeasures`, `GPUGeodesicPairs`, `GPUGeodesicDestination`, `GPULocalFrameProjection`, `GPULinearReferencing`, `GPULineLocate`, `GPULineSimplification`, `GPUCoverageSimplification`, `GPUTrajectoryMetrics`, `GPUTrajectoryPlayhead`, `GPUTrajectoryResample`, `GPUZoneEvents`, `GPUTrajectoryEncounters`, `addClockEncounters`, `GPUTrackSimilarity` |
| Geometry utilities and tessellation | `GPUOutlineGeometry`, `GPULabelPoint`, `GPUShapeDescriptors`, `GPULineDensity`, `GPULineLengthPerPolygon`, `GPUGridGenerator`, `GPUShapeGenerator`, `GPUHilbertKeys`, `GPURectangleClip`, `GPUDelaunayTessellation`, `GPUVoronoiDiagram` |
| Regression | `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialErrorGM`, `GPUSpatialLagMaximumLikelihood`, `GPUSpatialErrorMaximumLikelihood`, `GPUSpatialWeightsTraces`, `GPUSpatialLagImpacts`, `GPUGeographicallyWeightedRegression`, `GPUGeographicallyWeightedRegressionNonstationarityTest` |
| Recipes | `addHotSpotAnalysisRecipe` and eleven more prebuilt chains |
| Cross-reference | turf, PostGIS, GeoPandas, PySAL, QGIS and ArcGIS tools mapped to contributors |

Contributors for networks, rasters and dataframe columns live in their own entries:
[GPU Network](/docs/api-reference/experimental/gpu-network),
[GPU Raster](/docs/api-reference/experimental/gpu-raster/operations-regions-contours) and
[GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe-analysis).

### Ontology and capability catalogue

The entry point exports a small ontology over the flat contributor API. It gives applications,
documentation tools and future planners stable names for the data flowing between contributors:
points, lines, polygons, trajectories, DGGS cells, spatial weights, feature pairs, regions, sampled
surfaces, summaries and render geometry. Capabilities then group public exports by domain and
pipeline stage and state their logical inputs, outputs, coordinate assumptions and output-bounding
contract.

The catalogue stores export names, not constructor references, so using it for discovery does not
eagerly retain every implementation in a bundle. Every public `GPU...` contributor and `add...`
recipe belongs to exactly one available capability. Deliberate gaps are `status: 'opportunity'` and
have no operators, which makes coverage and missing connective tissue queryable with the same API.

```ts
import {
  getGPUSpatialAnalysisCapabilityForOperator,
  queryGPUSpatialAnalysisCapabilities
} from '@luma.gl/experimental/gpu-spatial-analysis';

const producers = queryGPUSpatialAnalysisCapabilities({
  status: 'available',
  output: 'spatial-weights'
});
const gaps = queryGPUSpatialAnalysisCapabilities({status: 'opportunity'});
const localMoranFamily = getGPUSpatialAnalysisCapabilityForOperator('GPULocalMoran');
```

The exported vocabulary is:

- `GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY` and `GPU_SPATIAL_ANALYSIS_DATA_KINDS`: logical data nouns.
- `GPU_SPATIAL_ANALYSIS_DOMAINS` and `GPU_SPATIAL_ANALYSIS_STAGES`: subject and pipeline roles.
- `GPU_SPATIAL_ANALYSIS_CAPABILITIES`: available capability families and named opportunities.
- `queryGPUSpatialAnalysisCapabilities(query)`: filters by status, domain, stage, input, output,
  coordinate model or search text.
- `getGPUSpatialAnalysisCapability(id)` and
  `getGPUSpatialAnalysisCapabilityForOperator(name)`: direct lookup from a concept or public export.
- `getGPUSpatialAnalysisCapabilityConnections(id)`: resolves the curated adjacent capabilities
  used to assemble workflows.

Operator-level discovery is available through `GPU_SPATIAL_ANALYSIS_OPERATORS`,
`getGPUSpatialAnalysisOperator(name)` and `queryGPUSpatialAnalysisOperators(query)`. Descriptors
contain export names rather than constructors and distinguish primary capability, secondary tags,
logical ports, spatial support, topology, cardinality, numerical semantics, determinism, status and
explicit readback. Unknown details are reported as `unspecified`, never inferred as a stronger
guarantee. `GPU_SPATIAL_ANALYSIS_AUDIT` and `formatGPUSpatialAnalysisAuditMarkdown()` expose the
same inventory used by package verification.

### Shared ports, status and planning

The entry point exports non-owning structural ports for point, line, polygon and trajectory
geometry, compact pairs, generated geometry, spatial weights, cell tables, sampled surfaces and
classified values. Ports only group existing graph views; they do not allocate, pack, submit or
read back buffers. `GPUSpatialPredicateJoin` outputs are assignable to `GPUCompactPairPort` and feed
`GPUPairGather` directly. Capacity-bounded outputs publish `count`, `requiredCount` when exact, and
the applicable overflow or uncertainty fields from the shared status vocabulary.

`GPUSpatialContext` separates coordinate space, metric, units and sphere/ellipsoid parameters.
Compatibility helpers reject invalid workflow connections during graph construction and contain no
CRS parsing or reprojection; that remains the responsibility of GPU Project.

`GPUSpatialParameterSchema` describes packed layouts without changing their WGSL representation.
Schemas are published for geometry predicates, neighbor search, line segmentizing, great-circle
arcs, spatial autocorrelation and line density. The generic packer validates defaults, offsets,
ranges and units; contributor-specific packers retain any documented normalization that is not yet
expressed by the generic schema.

Capacity helpers cover compact pairs, neighborhoods, generated geometry, cell tables and movement
events. Contributor planners for spatial joins, cell covers, line topology and trajectory
encounters additionally report stage capacities, work items, caller-owned output bytes and peak
transient bytes. Pass counts read from a representative run as `observed...Count`; without them the
same helpers use explicit expansion assumptions. They remain pure: applications own allocations.

Candidate and output capacities recover independently. A candidate overflow means a final
`requiredCount` may describe only the candidates that fit, so growing only the output is invalid:

```ts
const plan = getGPUSpatialJoinCapacityPlan({
  leftCount,
  rightCount,
  candidateCapacity,
  pairCapacity,
  observedCandidateCount,
  observedRequiredCount
});

const recovery = getGPUCapacityRecovery({kind: 'grow-on-next-frame'}, {
  candidate: {capacity: candidateCapacity, requiredCount: observedCandidateCount,
    overflow: candidateOverflow === 1},
  output: {capacity: pairCapacity, count, requiredCount: observedRequiredCount,
    overflow: overflow === 1}
});
```

The recovery decision names `candidate` and `output` in `incompleteStages` and returns the next
capacity for each affected stage. Fixed-budget and fail-closed policies preserve the same stage
diagnostics.

`GPUPartitionDescriptor` retains contiguous source-batch cores, read halos and deterministic seam
ownership without repacking `GPUVector` chunks. `GPUTilePartitionDescriptor` carries spatial core
and halo bounds for tiled consumers. Prepared indexes can be checked against a revision, spatial
context and partitioning before reuse, while `getGPUSpatialQueryCostPlan()` chooses scan, populated
grid or prepared BVH from construction-time measurements without a hidden readback. The Arrow
analytics adapter publishes its record batches through this contract, including empty batches.
`iterateGPUAnalyticsPartitionsFromArrowTable()` lazily allocates one source batch at a time while
retaining its global batch ID and row offset; destroying each yielded result before advancing bounds
GPU residency to one partition. The tile-LOD explorer declares explicit core, halo and cross-tile
ownership rules with the tile contract. Execution remains consumer-owned.

Recipe results use a required common envelope with `outputs`, `intermediates` and stage-owned
`status` in addition to `contributors`. The choropleth, fleet-dwell and drive-time catchment recipes
publish that envelope while retaining their existing named result fields.

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
  (`ids`, `count`, `overflow`, optional `requiredCount`). `count` is clamped to `ids.length` and can be
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

A `GPUSpatialWeights` is a first-class object, not only buffers. Every producer in this section
(`GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsTransform`,
`GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsTranspose`) exposes `outputWeights`: the caller-owned
views with `info` (`rowCount`, `transform` state `'raw' | 'binary' | 'row' | 'double' | 'variance' |
'kernel' | 'symmetric' | 'unknown'`, `symmetric` as `true | false | 'unknown'` only where
guaranteed, `hasSelfWeights`, `provenance` with source, criterion, `k`, `radius` and `order`) and
`status` (`overflow` and, where the producer computes it exactly, `requiredCount`) attached. Pass
`outputWeights` to consumers: statistics read `info.transform` (a `'row'` W is re-standardized over
the neighbors that remain after masking) and gate on `status.overflow`. Uploaded weights may omit
`info`, which reads as `'unknown'`. `validateGPUSpatialWeights` checks on the CPU what it can: view
formats and lengths, `info.rowCount`, the declared transform and symmetry, and the status formats.

One neighbor definition is used everywhere: a slot is an active neighbor when its weight is
positive, its neighbor ID is below the row count and it is not the row itself. A zero-weight slot is
listed but is not a neighbor. An island is a row with no active slot, whatever slots it lists.

Uploaded CSR must satisfy the invariants above; consumers do not re-check them (a binary search
over unsorted neighbors silently gives wrong `S1`, `S2` and symmetry counts). Run
`GPUSpatialWeightsValidation` once after upload, and sort, merge duplicates and drop self slots on
the CPU if it reports violations.

Capacity grow loop: `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsAlgebra` and
`GPUNeighborSearch` take a `totalNeighbors` view, published as `status.requiredCount`. Read
`overflow` and `requiredCount` after a frame; when `requiredCount` exceeds the capacity, allocate
`getGPUSpatialWeightsCapacity(requiredCount, capacity)` slots (the required count plus 25 percent
headroom) and rebuild. Rows past the point of truncation are empty, so never use weights whose
`overflow` is 1.

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
- `totalNeighbors` is published as `outputWeights.status.requiredCount`. It is exact unless the pair
  list overflowed `pairCapacity` (then a lower bound; growing the neighbor capacity grows
  `pairCapacity` with it, so the rebuild loop converges). `outputWeights.info` is binary, symmetric,
  with provenance `{source: 'GPUContiguityWeights', criterion}`.

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
it also writes Euclidean `distances`, so kernel transforms apply. `totalNeighbors` is the exact
`outputWeights.status.requiredCount`; `outputWeights.info` is binary and symmetric, with provenance
`{criterion, order: radius}`.

### `GPUSpatialWeightsTransform`

Rewrites weights in place or into a separate `output`. Chain several instances, for example a kernel
followed by row standardization.

- `'row'`: `w_ij / sum_j w_ij`. A row whose weights sum to zero (or to a non-finite value) stays all
  zero: it has no active neighbor, so it is an island for every statistic (no uniform fallback).
  To restore row standardization after masking or `'subgraph'` removed neighbors, apply `'row'` again
  to the result.
- `'binary'`: `w_ij > 0 ? 1 : 0`, keeping every slot.
- `'kernel'`: `K(d_ij / h_i)` from `weights.distances`, with `kernel` one of `gaussian`
  (`exp(-z^2 / 2) / sqrt(2 pi)`), `triangular` (`max(1 - z, 0)`), `epanechnikov` (`3/4 max(1 - z^2, 0)`,
  PySAL quadratic), `bisquare` (`15/16 max(1 - z^2, 0)^2`, PySAL quartic) or `uniform` (`1/2` for
  `z <= 1`). `bandwidth` is required (no default) and is either a positive number, the same `h` for
  every row (PySAL fixed bandwidth; for distance-band weights pass the band radius), or `'adaptive'`:
  PySAL's adaptive bandwidth, `h_i` = the distance to the k-th nearest neighbor times
  `eps = 1.0000001` (`GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON`). On the CSR that is the row's
  largest distance, but only for kNN rows, so `'adaptive'` requires
  `weights.info.provenance.criterion === 'knn'` and throws otherwise. Because of `eps` the k-th
  neighbor keeps a tiny positive weight, as in libpysal (in float32 `eps` rounds to `1 + 2^-23`, so
  the tail weight is about 1e-7, never exactly 0). Only the stored neighbors take part, so the
  Gaussian is truncated at the neighbor set. Weights never hold self slots, so `K(0)` is not
  written: compute it with `getGPUSpatialWeightsKernelValue(kernel, 0)` and pass it as `selfWeight`
  to `GPUHotSpotAnalysis` for a kernel Gi*. `getGPUSpatialWeightsKernelValue(kernel, z)` is the CPU
  value of every profile.
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
operation, kernel and bandwidth are compile-time options. `output`, when given, must have exactly
`weights.neighbors.length` values.

`outputWeights.info.transform` is the operation (`'symmetrize'` gives `'symmetric'`). `symmetric`
is kept only where it is guaranteed: `'binary'`, `'double'`, `'symmetrize'` and a fixed-bandwidth
`'kernel'` keep a `true` input flag; `'row'`, `'variance'` and an adaptive kernel give `'unknown'`
(row standardization makes a symmetric pattern asymmetric unless all row sums are equal).

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
within a distance band (`mode: 'radius'`) between planar points, or between longitude/latitude points
with great-circle distances (see below). A self join never lists a row as its
own neighbor; pass `queryPositions` for a cross join. It writes a `GPUSpatialWeights` CSR:
`offsets` (`queryRows + 1`), `neighbors` (the slot capacity), `weights`, and optional `distances`.

- Rows list neighbor IDs in strictly ascending order with distances and weights aligned, so
  consumers can binary-search `w_ji`.
- kNN keeps the `k` smallest `(d^2, id)` pairs, so ties go to the lowest ID. An optional
  `radius` bounds the search. Fewer valid targets give a shorter row.
- Weights: `binary`, `inverseDistance` (`max(d, distanceFloor)^-power`), or `kernel` (gaussian,
  triangular, epanechnikov, bisquare, uniform). The bandwidth is `radius` in radius mode and the
  row's k-th neighbor distance times 1.0000001 in kNN mode, which is libpysal `Kernel(fixed=False)`
  with the self slot removed (a row with fewer than `k` neighbors uses its farthest listed
  neighbor). `rowStandardize` divides each row by its sum. Non-finite weights are written as 0.
  Rows never hold a self slot: the kernel's self weight `K(0)` is the consumer's `selfWeight`
  (`getGPUSpatialWeightsKernelValue(kernel, 0)`). A zero-weight slot (for example a triangular
  kernel at `d = radius`, or coincident points under inverse distance with no floor) is not a
  neighbor: consumers skip slots whose weight is not positive.
- Capacity: offsets are clamped to `neighbors.length`, so readers always see consistent, possibly
  truncated rows. `overflow` is set on the GPU, and `totalNeighbors` and `neighborCounts` stay
  unclamped.

```ts
const searchParameters = {bounds, rowStandardize: true};
const parameters = new GPUParameterBuffer(device, {
  id: 'knn', format: 'float32', length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  values: getGPUNeighborSearchParameterValues(searchParameters)
});
graph.add(new GPUNeighborSearch({
  mode: 'knn', k: 8, gridSize: [256, 256], positions,
  parameters: parameters.importToGraph(graph),
  weights: {offsets, neighbors, weights, distances}, overflow,
  weightTransform: getGPUNeighborSearchWeightTransform(searchParameters)
}));
```

**Spatial context.** `spatialContext` (compile-time) defaults to native planar coordinates. For
longitude/latitude pass `{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units?:
'meters' | 'kilometers', sphereRadius?}` (`sphereRadius` in meters, default the mean Earth radius).
Positions are `[longitude, latitude]` degrees, `bounds` are `[minLongitude, minLatitude,
maxLongitude, maxLatitude]` degrees (longitude span at most 360, for example the world), and
`radius`, `distanceFloor` and the output `distances` are great-circle distances in `units`
(meters by default). Any other space or metric throws at construction (project to a planar frame
first). Distances are f32 central angles on a sphere. The lattice stays in degrees: radius mode
visits every cell that can hold a target in the cap (latitude span `radius / R`, longitude span the
exact cap extent `asin(sin(a) / cos(latitude))`, the whole circle when the cap reaches a pole, and
bands wrap across +-180), and kNN stops only when a conservative spherical lower bound of the
distance to every unvisited cell exceeds the k-th distance, so results never depend on `gridSize`.
The bound is deliberately loose near the poles, which only costs speed. Points with latitude beyond
+-90 are invalid.

**Planar precision.** Planar f32 coordinates keep about seven digits: UTM northings (~4e6 m) and
Web Mercator (~1e7) quantize to 0.5 to 1 m and make band edges and ties noisy. Project to a local
frame with `createGPULocalFrame` (or subtract an origin) before upload; the node cannot recover
digits lost in its inputs.

**Row standardization.** The optional compile-time `rowStandardize` prop is the single source of truth
when given: the weights kernel uses it as a constant and ignores the per-frame `rowStandardize`
parameter. Consumers re-standardize rows over the neighbors left after masking only when
`info.transform` is `'row'`, so set it for any row-standardized band. `rowStandardize: true` with a
`weightTransform` other than `'row'`, or `false` with `'row'`, throws. When it is omitted, the per-frame
field and the `weightTransform` declaration apply as before.

**Output.** `outputWeights` is `weights` with `info` and `status` attached, for consumers:
`rowCount`, `transform` (`'row'` when the compile-time `rowStandardize` prop is true, otherwise the
optional compile-time `weightTransform`, else `'unknown'`, because weight kind and row standardization
are per-frame parameters; derive it from the same parameters with
`getGPUNeighborSearchWeightTransform`), `symmetric` (true only for a planar radius self join
without `queryMask` and a declared non-row transform; otherwise `'unknown'`), `provenance`
(`criterion` `'knn'` or `'distance-band'`, and `k`) and `status` (`overflow`, `requiredCount` =
`totalNeighbors`, `invalidCount`). kNN output has criterion `'knn'`, which `bandwidth: 'adaptive'`
kernel transforms require. PySAL's `min_threshold_distance` (the smallest band that leaves no
island) is the maximum of the returned `distances` of a `k = 1` kNN run.

**Dropped rows.** Targets and queries count only when their mask is nonzero, their coordinates are finite, and they
lie inside the per-frame `bounds`; the others silently become isolates and never a neighbor.
Pass the optional `invalidCount` (one uint32) to learn how many masked rows were dropped for a
non-finite coordinate or for lying outside `bounds` (target rows, plus the query rows of a cross
join). A self join applies `mask` to the query rows too. Bounds,
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
the slots needed (published as `outputWeights.status.requiredCount`; omitted for `'higherOrder'` with
`order > 1`, where an overflowing intermediate set would make it a lower bound, so raise
`workCapacity` instead). On overflow the output is still a valid CSR: the leading rows are complete,
one row is a prefix and the rest are empty. Output rows are sorted and unique with finite
non-negative weights. `outputWeights.info` carries the transform state (`'binary'` for
`weightRule: 'binary'`, `'higherOrder'` and `'block'`; an unscaled `'raw'` or `'binary'` input
stays so; a row-standardized input becomes `'unknown'` because slots changed) and `symmetric: true`
only when it follows from symmetric operands.

```ts
graph.add(new GPUSpatialWeightsAlgebra({
  operation: 'union', left: contiguity, right: knn, weightRule: 'binary', output, overflow
}));
```

### `GPUSpatialWeightsSummary`

Summary statistics of a weights matrix. `statistics` (`float32`, at least 3) receives `S0`,
`S1 = 1/2 sum (w_ij + w_ji)^2` and `S2 = sum_i (row_i + col_i)^2`, as in libpysal. `counts` (`uint32`,
at least 6, see `GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT`) receives the used slots, the asymmetric slots,
the isolates, the minimum and maximum cardinality and the empty rows. The optional `cardinality` is the
per-row active neighbor count and `symmetryTolerance` (default 0) decides when two weights differ. Sums
run in a fixed order. Isolates are rows with no active slot, the one island definition of every
statistic (libpysal `islands` for a W built without zero-weight entries); `emptyRows` counts rows with
no slot at all. Cardinality counts active slots, so a zero-weight slot is not a neighbor: this deviates
from libpysal `cardinalities`, which counts every listed neighbor, and agrees with it for a W built
without zero-weight entries (pinned against libpysal in the specs). The used slots count stays the
listed total. Connected components are not computed. A pair of unequal weights counts as two
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

`outputWeights` carries the input `symmetric` flag (a transpose is symmetric exactly when `W` is) and
the input `status` (a truncated `W` gives a truncated `W'`); a `'row'` or `'variance'` state becomes
`'unknown'`.

### `GPUSpatialWeightsValidation`

Checks the `GPUSpatialWeights` invariants of weights that were built elsewhere and uploaded, on the GPU:
`offsets[0] = 0`, non-decreasing offsets with `offsets[rows] <= neighbors.length`, every neighbor ID below
the row count, strictly ascending neighbor IDs within each row (sorted and unique), no self slot, and
finite non-negative weights. `invalidCount` (one `uint32`) receives the number of rows with at least one
violation; the optional `flags` (one `uint32` per row) receives the bitmask of
`GPU_SPATIAL_WEIGHTS_VIOLATION` (`offsets` 1, `neighborRange` 2, `order` 4, `self` 8, `weight` 16). Run it
once after upload and before using the weights; there is no GPU sort or dedupe, so fix a failing CSR on the
CPU. `outputWeights` is the checked weights with `status.invalidCount` attached.

```ts
graph.add(new GPUSpatialWeightsValidation({weights: uploaded, invalidCount, flags}));
```

### `GPUMapColoring`

Consumes an adjacency pattern only, so it takes no `spatialContext`.

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
its analytic expectation, variance, z-score and normal p-value (`alternative`, default two-sided).
Pick the statistics with `statistics` (compile-time). The `results` layout (`GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`) is fixed,
and blocks that were not requested hold NaN.

- `moran`: Moran's I with esda's `VI_norm` and `VI_rand` (sample kurtosis).
- `geary`: Geary's C with `VC_norm` and `VC_rand`.
- `getisOrdG`: Getis-Ord General G, which needs non-negative ratio-scale data: when any analysis
  value is negative the whole G block is NaN. Its randomization variance
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

Neighbors, islands and re-standardization follow the shared rules of the local statistics (see
`GPUHotSpotAnalysis` and `GPULocalMoran`):

- **Neighbors.** A neighbor is a slot with a positive weight to another analysis row. Zero-weight
  slots, the row's own slot and out-of-range IDs are not neighbors.
- **Re-standardization.** Weights are used as given, except when `weights.info.transform` is `'row'`:
  each row is then divided by the sum of its active weights everywhere (`S0`, `S1`, `S2`, the reverse
  weights `w'_ji` and the cross products), so masked, missing and island neighbors drop out and the
  row still sums to 1 (esda on the subset `W`, re-row-standardized).
- **Islands.** An island is an included row (mask nonzero, every value finite) with no active slot.
  The rule looks at out-links only; for symmetric weights it equals the old in-or-out rule.
  `islands: 'exclude'` (default, compile-time) removes islands from `n`, the moments, `S0`, `S1`,
  `S2` and from the neighbors of other rows, in one pass with no cascade (a row whose only neighbors
  were islands keeps its place in `n` and adds no weight). `islands: 'keep'` is esda's convention.
  The summary `islandCount` reports the islands under either policy. esda parity needs `'keep'`.
- **Truncated weights.** If `weights.status.overflow` is nonzero the weights are incomplete, so the
  weight sums, the mean, the variance and every statistic block are NaN (`n` and `islandCount` stay;
  the `joinCounts` view holds 0).
- **`alternative`** (`'two-sided'` default, `'greater'`, `'lesser'`/`'less'`; compile-time) selects
  the tail of every analytic p field, including the join counts: `'greater'` is the upper tail of z
  (for example positive autocorrelation of Moran's I), `'lesser'` the lower tail. `'directed'` and
  `'folded'` are permutation conventions and throw. `getInferenceMetadata()` returns a
  `GPUInferenceMetadata` for a legend. Under non-Gaussian data prefer the randomization fields.

The summary block reports `n`, `S0`, `S1`, `S2`, the mean, the variance, the black count and the
island count. `n` is counted in `u32` and the mean gets a second-pass correction
(`mean += sum(x - mean) / n`), so a large offset does not bias the variance; the row count must be
below `2^24`, the limit of the local statistics, because degrees and counts are f32 columns.
`w_ji` comes from a binary search of row `j`. Column sums come from a deterministic transpose: a
stable sort of slots by neighbor. Per-row partials are reduced by fixed-shape workgroup trees, and
join counts by u32 atomics. Results are bitwise reproducible, and values, weights and the mask
change per frame without a rebuild. Inference is NaN when `n < 4`, `S0 = 0`, or the values are
constant.

### `GPUHotSpotAnalysis` and `GPULocalMoran`

Local spatial autocorrelation on caller-supplied square `weights`. Rows count when their `mask` is
nonzero and their value is finite; excluded rows are neither foci nor neighbors. `n`, the mean `X`
and the population standard deviation `S` are taken over the analysis rows (see islands below), or
from per-frame `fixedMoments`. The mean gets a second-pass correction (`mean += sum(x - mean) / n`),
so a large offset does not bias the variance, and `n` is counted in `u32`. Sums run in slot order
with fixed-order reductions, so outputs are bitwise reproducible.

Shared semantics (also `GPULocalGeary`):

- **Neighbors.** A neighbor is a slot with a positive weight that points at another included row.
  Zero-weight slots, the focal row's own slot and out-of-range IDs are not neighbors: they are not
  counted in `neighborCounts` or `minimumNeighbors`. If `weights.status.overflow` is nonzero the
  weights were truncated and every output is NaN.
- **Re-standardization.** Weights are used as given, except when `weights.info.transform` is
  `'row'`: then each row's weights are divided by their sum over the active neighbors, so masked or
  missing neighbors drop out and the row still sums to 1 (esda on the subset `W`, re-row-standardized,
  not a lag that shrinks toward 0). Without masking this changes nothing beyond rounding. Weights
  with no declared transform are never rescaled.
- **Islands.** An island is an included row with no neighbor among the included rows.
  `islands: 'exclude'` (default, compile-time) removes islands from `n`, the mean and the variance and
  writes NaN for every output. `islands: 'keep'` is the esda convention: islands stay in the moments;
  their local statistic is the formula value (local I and the spatial lag are 0, local Geary is 0),
  and z and p are NaN because they are not tested. The decision is one pass with no cascade: a row
  whose only neighbors were islands has no neighbor among the analysis rows, gets NaN, and stays in
  `n`. `globalStatistics` is `[n, mean, variance, sd, islandCount]` under either policy. esda
  parity specs use `'keep'`.
- **`minimumNeighbors`** (default 1, compile-time integer). Rows with fewer neighbors get NaN z and p
  and bin or quadrant 0; the moments are unchanged. The default only rejects rows without a
  neighbor, since no row can have a z-score from its own value. Esri recommends at least 8
  neighbors for Gi* (pass `minimumNeighbors: 8`); the normal approximation of local Moran is poor
  below about 4.
- **`multipleTesting`** (`'none'` default, `'bonferroni'`, `'holm'`, `'benjamini-hochberg'`,
  `'benjamini-yekutieli'`, compile-time) adjusts the two-sided p-values through
  `GPUMultipleTestingAdjustment`; the family is every row with a defined z-score, independent of any
  viewport. The optional `adjustedPValues` output holds the adjusted values (equal to `pValues` for
  `'none'`). `falseDiscoveryRate: true` is a deprecated alias of `'benjamini-hochberg'`.
  `getInferenceMetadata()` returns a `GPUInferenceMetadata` (two-sided, analytic, the method, the
  tested-rows definition) for building a legend.
- **Legend contract.** Bins and quadrants mean the same thing for every correction method. A Gi*
  bin of level `L` means the adjusted p-value (the plain p-value for `'none'`) is at most `0.10`,
  `0.05` or `0.01` for `L = 1, 2, 3`, and the sign is the sign of z. A local Moran quadrant is the
  sign pair of the centered value and the lag, reported only where the adjusted p-value is at most
  `significanceLevel`. With `'none'`, Gi* bins use the critical values `1.645, 1.960, 2.576`,
  which is the same rule.

- `GPUHotSpotAnalysis` writes the Getis-Ord Gi* z-score per row for arbitrary weights:
  `z_i = sum_j w_ij (x_j - X) / (S sqrt((n S1_i - W_i^2) / (n - 1)))`, with `W_i = sum_j w_ij` and
  `S1_i = sum_j w_ij^2`. The focal row enters through the compile-time `selfWeight` (default 1,
  classic Gi*), never through the matrix; a row's own value alone never produces a z-score. The
  z-score is unchanged when every value is shifted by a constant. It equals esda's
  `G_Local(star=True)` analytic `Zs` for positive data (esda normalizes by the sum of values, which
  flips the sign when the mean is negative). `selfWeight: 0` is the exact Ord-Getis Gi (esda
  `G_Local(star=False)`), which standardizes with the mean and variance of the `n - 1` other
  observations: `z_i = (sum_j w_ij x_j - W_i mu_i) / (s_i sqrt(((n - 1) S1_i - W_i^2) / (n - 2)))`.
  With row-standardized weights `selfWeight: 1` weighs the focal row as much as all its neighbors,
  so prefer `0` there. Optional outputs: the two-sided p-value, the adjusted p-value, the neighbor
  count (the focal row not included) and a `sint32` confidence bin in `-3..3` (99/95/90% hot or cold).
- `GPULocalMoran` writes the z-score of local Moran's I with the analytic moments of conditional
  randomization, the null esda's permutation test samples: `E[L_i] = W_i mu`,
  `Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)` over the other `N = n - 1` rows. Optional outputs
  are `localI` (`(n - 1) c_i L_i / sum c^2`, esda's `Is`), the spatial lag `L_i = sum_j w_ij c_j`,
  the p-value, the adjusted p-value, the neighbor count and the esda quadrant code (1 HH, 2 LH, 3 LL,
  4 HL; 0 when not significant at the per-frame `significanceLevel`).
  `quadrantGating` (`'analytic'` default or `'none'`, compile-time) chooses what `quadrants` holds. With
  `'analytic'` it is nonzero only where the analytic p-value (adjusted when `multipleTesting` is set) is
  significant. With `'none'` it
  is the ungated quadrant from the signs of the centered value and the spatial lag, for every included
  row with a nonzero centered value and lag, and 0 otherwise, so another test can gate it, for example the
  `significant` mask of `GPULocalPermutationTest` (esda's `q` with `p_sim`). `zScores` and `pValues` do
  not change. `'none'` cannot be combined with `multipleTesting` (the constructor throws).
  Local Moran significance does not depend on `|c_i|`: a row near the mean can be significant.

Checked against esda 2.10 on 200 points with 6-nearest-neighbor weights: `localI` matches `Is` to
3e-7, the z-score tracks esda's 99,999-permutation `z_sim` (correlation 0.999997, largest gap 0.03),
and Gi* matches `Zs` to 1e-7. A 60-row fixture with an island and masked rows pins `G_Local` (star
and non-star), `Moran_Local`, `Geary_Local` and the statsmodels adjustments for `islands: 'keep'`,
`'exclude'` and the re-standardized masked case.

```ts
graph.add(new GPUNeighborSearch({
  mode: 'radius', gridSize: [256, 256], positions, parameters: searchParameters,
  weights, overflow
}));
graph.add(new GPUHotSpotAnalysis({weights, values, parameters, zScores, bins, pValues}));
graph.add(new GPULocalMoran({weights, values, parameters, zScores, quadrants}));
searchParameters.write(getGPUNeighborSearchParameterValues({bounds, radius: 500}));
```

### `GPUAssuncaoReisStandardization`

Empirical-Bayes standardized rates of Assuncao and Reis (1999), the transformation behind
`esda.Moran_Rate` and `esda.Moran_Local_Rate`. It writes one standardized value per row and no
spatial statistic; Moran statistics of rate data come from composing it with the existing nodes.
Lives in `spatial-autocorrelation/gpu-rate-standardization`.

```ts
const standardization = new GPUAssuncaoReisStandardization({
  events, // float32 view
  populations, // float32 view, population at risk
  mask, // optional uint32
  standardizedRates, // float32 view, NaN for rows outside the analysis
  statistics, // optional, [n, b, s2, a, totalPopulation]
  invalidCount // optional uint32 scalar
});
```

Definition (identical to `esda.smoothing.assuncao_rate`), over the `m` valid rows with rates
`p_i = e_i / n_i`: `b = sum e / sum n`, `s2 = sum n_i (p_i - b)^2 / sum n`,
`a = s2 - b / (sum n / m)`, `v_i = a + b / n_i` (replaced by `b / n_i` where `v_i < 0`), and
`z_i = (p_i - b) / sqrt(v_i)`.

- **Negative prior variance.** `a` is negative when the rates vary less than Poisson noise predicts.
  esda does not clamp `a`: it keeps the negative `a` wherever `v_i` stays non-negative and falls
  back to the pure Poisson variance `b / n_i` only for rows where `v_i < 0`. This node does the
  same, for parity, and `statistics[3]` reports the unclamped `a`. The EB smoothing in
  `rate-smoothing` clamps the prior variance at zero instead (a different, also standard
  convention for shrinkage); the two are not meant to agree when `a < 0`.
- **Missing rows.** A row is valid when it is unmasked, its events are finite and non-negative, and
  its population is finite and positive. Other rows write NaN, are excluded from `b`, `s2`, `m`
  and the mean population, and are excluded by the downstream statistic. `invalidCount` counts
  the unmasked rows that were rejected. Rows where `v_i` is not positive (for example `b = 0`)
  write NaN, as esda's 0 / 0 does.
- **Precision.** `m` is a u32; sums are fixed-order two-level workgroup trees (no float atomics,
  bitwise reproducible); `b` gets a second-pass correction and `s2` is formed from centered
  deviations. Fewer than 2^24 rows.

Composition (no statistic code is duplicated):

```ts
const standardizedRates = createTransientView(graph, 'rates', 'float32', rows);
graph.add(new GPUAssuncaoReisStandardization({events, populations, standardizedRates}));

// EB Moran's I (esda.Moran_Rate, transformation 'r'): row-standardized weights.
graph.add(
  new GPUGlobalSpatialStatistics({
    weights, // weights.info.transform === 'row'
    values: standardizedRates,
    statistics: ['moran'],
    islands: 'keep', // esda's convention
    results
  })
);
// EB local Moran (esda.Moran_Local_Rate): GPULocalMoran, or GPULocalPermutationTest('localMoran').
graph.add(new GPULocalMoran({weights, values: standardizedRates, parameters, islands: 'keep', zScores, localI}));
```

`results[moran + statistic]`, `results[moran + expected]` and `results[moran + zRandomization]`
equal `Moran_Rate.I`, `.EI` and `.z_rand` (esda recomputes the Moran of the standardized values and
ignores the EB sampling variance); `localI` equals `Moran_Local_Rate.Is`. Row-standardized weights
are re-standardized over the active neighbors, so NaN rows drop out of the neighborhoods of the
rows that remain.

### `GPUMultipleTestingAdjustment`

Adjusts a column of per-row p-values for multiple testing on the GPU. `method` is `'none'`,
`'bonferroni'`, `'holm'`, `'benjamini-hochberg'` or `'benjamini-yekutieli'`, and `adjustedPValues`
holds the same numbers as statsmodels `multipletests` (`bonferroni`, `holm`, `fdr_bh`, `fdr_by`), in
`[0, 1]`. `getMultipleTestingAdjustedPValues` is the float64 CPU reference with the same semantics.

- **Family.** The family is every tested row: a row whose p-value is finite and in `[0, 1]`. NaN
  rows are not tested and get NaN. There is no mask prop by design: run the statistic and this node
  on the full analysis set and apply a viewport or display mask only when rendering, otherwise
  panning changes the family size `m` and makes the visible rows look more or less significant.
- **`minimumFamilySize`.** `m = max(testedRows, minimumFamilySize)`; rows without a p-value count as
  `p = 1`. Use it to pin `m` when the tested set can shrink (for example an analysis mask that
  follows the viewport). The optional `testedCount` output reports the `m` actually used.
- **Which method.** Local statistics of neighboring rows share observations, so their p-values are
  dependent. Benjamini-Hochberg only controls the false discovery rate under independence or
  positive regression dependence; use `'benjamini-yekutieli'` (Benjamini-Hochberg times
  `c(m) = sum 1/i`, a cost of about `ln m` in power) for controlled error under arbitrary
  dependence. Holm controls the family-wise error rate and is uniformly more powerful than
  Bonferroni.
- **Significance.** For every method a row is significant at level `alpha` exactly when
  `adjustedPValue <= alpha`, so 0.10 / 0.05 / 0.01 bins are consistent by construction. Tied
  p-values get identical adjusted values.
- **Cost.** `none` and Bonferroni are O(n). The step procedures add a 30-bit `GPUSort` plus a
  deterministic cumulative maximum (Holm) or minimum (Benjamini) scan in workgroup memory, a few
  linear kernels in all. Fewer than 2^24 rows (ranks are exact f32 integers); results are
  independent of scheduling.

```ts
graph.add(new GPUMultipleTestingAdjustment({
  pValues, method: 'benjamini-yekutieli', adjustedPValues
}));
// significant at 5%: adjustedPValues[i] <= 0.05
```

### `GPULocalPermutationTest` and `GPUGlobalPermutationTest`

Permutation inference (pseudo p-values) over a `GPUSpatialWeights` CSR, following PySAL esda's
`p_sim` with the corrections below. Both contributors read a uint32 parameter view written with
`getGPUPermutationParameterValues`: `{seed, permutations, significanceLevel}`. Changing the seed, the
permutation count (up to the compile-time `maximumPermutations`), the values or the weights never
rebuilds the graph.

- `GPULocalPermutationTest` does conditional randomization for `localMoran`, `localGeary`, `localG`,
  `localGStar` (a self weight of 1), `localBivariateMoran`, `localLeeL`, `localJoinCount` and
  `localBivariateJoinCount` (see **Bivariate, join-count and Lee statistics** below). For each tested row and permutation it draws an ordered sample
  of `k_i` distinct other analysis rows (a partial Fisher-Yates over their compacted positions) and
  assigns them to the row's weight slots. It writes `exceedances`, `pseudoPValues`, an optional
  `observed` statistic, an optional `adjustedPValues`, and a `significant` mask. Rows with more than
  `maximumNeighbors` (64 at most) neighbors are not tested and set `overflow`.
- `GPUGlobalPermutationTest` handles `moran`, `geary`, `getisOrdG`, `bivariateMoran` (`y`
  permuted) and `leesL` (Lee's spatial Pearson, `x` and `y` permuted together). `results` (`GPU_GLOBAL_PERMUTATION_RESULT`) holds the observed statistic, `p_sim`, the
  exceedance count, the permutation count used, the simulated mean and standard deviation, `z_sim`,
  `p_z_sim`, the minimum and maximum, and the island and analysis row counts (`islandCount`,
  `analysisCount`). Optional outputs are the `referenceDistribution` and its `histogram` (GeoDa's
  permutation plot).

**Tail (`alternative`).** The type is the shared `GPUInferenceAlternative` (`GPUPermutationAlternative`
remains as a deprecated alias) and the default is now `'two-sided'`: with `g = #{sim >= observed}`
and `l = #{sim <= observed}`, `p = min(2 (min(g, l) + 1), P + 1) / (P + 1)`. A row (or a global
statistic) whose simulations all tie the observed value has `g = l = P` and `p = 1`. `'directed'`
(`p = (min(g, l) + 1) / (P + 1)`) is esda's legacy default and is anti-conservative: the tail is
chosen from the data and not doubled, so under the null `p` is about twice too small. Its tie
handling deviates from esda on purpose: esda uses `min(g, P - g)`, and `P - g` counts only
simulations strictly below the observed value, so when simulations tie the observed value (discrete
data, zero-inflated counts, a constant neighborhood) esda reports the smallest attainable
`p = 1/(P+1)` (for 20 equal simulations: 0.0476 for a tied observed value), while `min(g, l)` reports
1. The two agree whenever no simulation equals the observed value (pinned against esda 2.10.0).
`'folded'` matches esda's undirected test exactly; `'greater'` and `'lesser'` (`'less'` alias)
match esda's one-sided tests. Do not combine `'directed'` with `multipleTesting`.

**Neighbors, islands and re-standardization.** One rule, shared with the other Stream A statistics:
a neighbor slot is *active* when its neighbor is a valid row other than the row itself, its weight is
positive, and the neighbor is an included row (mask nonzero, finite value). Zero-weight slots are not
neighbors: they are not counted in `k_i` and are never drawn. An *island* is an included row with no
active slot. `islands: 'exclude'` (default) drops islands from the analysis: NaN outputs, and they
are not counted in `n`, the mean, the variance or `S0`, and not neighbors of other rows. There is no
cascade: a row whose only neighbors were islands stays in `n` with no neighbor, so it is not tested
(global tests add 0 to `S0` for it). `islands: 'keep'` reproduces esda: islands stay in `n` and the
moments, their local statistic is the formula value with a zero lag (`observed`, for example local
I = 0), but they are never tested (`exceedances` is `GPU_LOCAL_PERMUTATION_NOT_TESTED`, p-value NaN).
Parity specs against esda use `islands: 'keep'`. Island counts are reported in the optional
one-row `islandCount` output (local) or `results[GPU_GLOBAL_PERMUTATION_RESULT.islandCount]` (global).
When `weights.info.transform` is `'row'`, each analysis row's active weights are re-standardized to
sum to 1, in the lags and in `S0`, so a masked or missing neighbor drops out instead of shrinking
the row sum below 1 (numerically identical when nothing is dropped); other transforms are used as
given. If `weights.status.overflow` is nonzero the weights are truncated: no row is analyzed
(statistics NaN), and the local test sets `overflow`. The mean is a fixed-order tree sum followed by a
second-pass correction of the deviations, and `n` is an exact uint32, so large-offset data keep
`sum(z) = 0` to f32 accuracy.

**Permutation count.** The count is never clamped. A per-frame `permutations` outside
`[1, maximumPermutations]` writes NaN p-values: the local test leaves every row untested and sets
`overflow`; the global test writes NaN for `permutations`, the p-values and the simulated summaries
(the observed statistic is still reported). Validate on the CPU with
`getGPUPermutationParameterValues(parameters, {maximumPermutations})` or the contributor's
`getParameterValues(parameters)`, both of which throw.

**Multiple testing (local).** `multipleTesting` is `'none'` (default), `'bonferroni'`, `'holm'`,
`'benjamini-hochberg'` or `'benjamini-yekutieli'`, run through `GPUMultipleTestingAdjustment` over the
pseudo p-values of all tested rows (never a display mask). `adjustedPValues` receives the adjusted
values, and `significant` is `adjustedPValue <= level`; with `'none'` it is the exact
cross-multiplied integer comparison `p <= level`. `falseDiscoveryRate: true` is a deprecated alias of
`multipleTesting: 'benjamini-hochberg'`. `getInferenceMetadata(parameters)` on either contributor
returns a `GPUInferenceMetadata` (alternative, permutation count, seed, multiple-testing method,
significance level and a `testedRowsDefinition` sentence) for legends.

**Bivariate, join-count and Lee statistics.** Definitions below use `c = x - mean` over the analysis
rows, `SS = sum c^2`, `n` analysis rows and the weights as given (re-standardized for `'row'`
weights). Parity: statistics against esda 2.10 at f32 accuracy, p-values against the twin of the
Philox draws, plus a Monte Carlo bound against esda `p_sim` (R = 9999).

- `localBivariateMoran` (esda `Moran_Local_BV`, needs `secondValues`): `I_i = (n - 1) c^x_i sum_j
  w_ij c^y_j / sqrt(SS_x SS_y)`; the focus keeps `x`, the drawn neighbors supply `y`. The optional
  `quadrants` output is esda's `q` in the PySAL scheme (1 high-high, 2 low-high, 3 low-low, 4
  high-low in `x_i` and the lag of `y`); it is not gated by significance.
- `localLeeL` (esda `Spatial_Pearson_Local`, needs `secondValues`): `L_i = n (sum_j w_ij c^x_j)(sum_j
  w_ij c^y_j) / sqrt(SS_x SS_y)`. A drawn row supplies both its `x` and its `y` (the pair stays
  together), the focus value does not enter. For row-standardized weights the mean of the local
  values is the global `L`. esda's own p-value is the `'directed'` one (anti-conservative); the
  default here is `'two-sided'`.
- `leesL` in `GPUGlobalPermutationTest` (esda `Spatial_Pearson`, element `[0, 1]`): `L = z'W'Wz /
  1'W'W1 = n sum_i (sum_j w_ij c^x_j)(sum_j w_ij c^y_j) / (sqrt(SS_x SS_y) sum_i (sum_j w_ij)^2)`,
  with the rows of `(x, y)` permuted together by the swap-or-not shuffle (esda permutes the rows of
  `Z`). Note that esda's global version does not row-standardize `W`; here `'row'` weights are
  re-standardized over active slots, others are used as given. The smoothing factors esda reports
  at `[0, 0]` and `[1, 1]` are not exposed. (`GPUSpatialPearson` is a different statistic: the
  weighted correlation across edges.)
- `localJoinCount` (esda `Join_Counts_Local`, binary `values`): `LJC_i = x_i sum_j x_j` over the active
  neighbors with binary weights (the weight values are ignored), where a value is a join only when
  it is exactly 1. Only rows with `LJC_i > 0` are tested; the others get NaN p-values and
  `GPU_LOCAL_PERMUTATION_NOT_TESTED`, as esda sets their `p_sim` to NaN. The tail that detects
  clustering is `'greater'`. With ties (discrete statistics) esda's `'two-sided'` p-value
  (percentile based) is not `2 min(g, l)`; the parity specs therefore use `'greater'`.
- `localBivariateJoinCount` (esda `Join_Counts_Local_BV`, `joinCase`): `'CLC'` (co-location) counts
  neighbors with `x_j = y_j = 1` for a focus with `x_i = y_i = 1`; `'BJC'` (no co-location) counts
  neighbors with `x_j = 0, y_j = 1` for a focus with `x_i = 1, y_i = 0`. A drawn row supplies its `x`
  and `y` together. **Deviation from esda for `'BJC'`:** esda observes the count above but
  simulates `x_i sum_j y_j`, which ignores `y_i = 0` and `x_j = 0`, so its p-values do not test the
  statistic it reports (on the specs' fixture they differ by up to 0.09 at R = 9999). This
  implementation simulates the observed statistic; `'CLC'` is identical to esda.
- `localGeary` classification (`classes` output, `GPU_LOCAL_GEARY_CLASS`; Anselin 2019, esda
  `Geary_Local(labels=True)`): with `m` the mean of the local Geary values over the analysis rows, a
  significant row (`p <= level`, or `adjustedPValue <= level` with `multipleTesting`) is 1
  (high-high, positive association: `c_i < m` and `x_i` above the mean), 2 (low-low: `c_i < m`, `x_i`
  below the mean) or 3 (negative association: `c_i > m`); an insignificant row is 4 and a row that
  is not tested (NaN p-value), or exactly at a mean, is 0 (esda: NaN). esda's own docstring calls
  codes 1 and 2 "outlier" and "cluster"; the numeric rule is ported, not the words. Small `c_i` means
  similar neighbors, so the tail of clustering is `'lesser'` (also the tail declared by
  `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER`): with `'lesser'` class 3 is empty because negative
  association never has a small lower-tail p-value, with `'greater'` classes 1 and 2 are empty, and
  with the default `'two-sided'` both appear. The classification is not available analytically.

**Global shuffle.** Each permutation relabels the analysis values through a keyed swap-or-not
shuffle (Hoang, Morris and Rogaway 2012, `6 ceil(log2 n) + 6` rounds, round keys and salts from
Philox 4x32-10 on the seed and the permutation index), so no sort runs per permutation and nothing
of size `P x n` is stored. The earlier four-round Feistel network is no longer used by the global
test because it is not uniform on small domains: with 400000 draws the marginal of `pi(0)` and the
joint placement of fixed pairs reject uniformity far below the 0.1 percent level for domains of
10 to 50 elements (joint chi-square p-values below 1e-100; marginal p-values from 4e-4 at `n = 50`
down to 1e-139 at `n = 10`), while swap-or-not passes (all 150 chi-square p-values above 0.0023,
none below 1e-3). At 20000 draws on rook lattices and random kNN graphs with `n` = 30 to 5000
both shuffles reproduce the analytic randomization mean and variance of Moran's I (`E = -1/(n-1)`,
esda `VI_rand`) to within about 2.7 standard errors, so the Feistel bias is a small-domain,
joint-placement effect that the variance check alone does not see; it is removed rather than
tolerated. The Knox and Mantel tests and the GWR non-stationarity test use the same swap-or-not
shuffle.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'permutations', format: 'uint32', length: GPU_PERMUTATION_PARAMETER_LENGTH,
  values: getGPUPermutationParameterValues({seed: 1, permutations: 99}, {maximumPermutations: 999})
});
const test = new GPULocalPermutationTest({
  weights, values, statistic: 'localMoran', parameters: parameters.importToGraph(graph),
  maximumPermutations: 999, exceedances, pseudoPValues, significant, adjustedPValues,
  multipleTesting: 'benjamini-yekutieli', islands: 'exclude', islandCount, overflow
});
graph.add(test);
// Preview with 99 permutations, then refine on idle:
parameters.write(test.getParameterValues({seed: 1, permutations: 999}));
const metadata = test.getInferenceMetadata({seed: 1, permutations: 999});
```

### Additional spatial statistics and bounded optimization

`GPULocalGeary` writes source-aligned `sum_j w_ij (z_i - z_j)^2` values through the shared
`GPUInferenceResultPort`, with the neighbor, re-standardization and `islands` semantics of
`GPULocalMoran` (an island is NaN under `'exclude'`, the default, and 0 under `'keep'`, esda's
convention; a constant variable gives NaN). It has no analytic classification: use permutation
inference (`GPULocalPermutationTest` with `statistic: 'localGeary'` and `classes` gives the Anselin
2019 positive / negative association classes, see above). `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER` declares the `'lesser'` tail, because small `c_i`
means similar neighbors (a cluster). `GPUSpatialPearson` is the weighted Pearson correlation across
directed spatial edges `(x_i, y_j)`. `GPUGammaStatistic` reports the weighted mean centered
cross-product, squared difference, or absolute difference over those edges. Both skip the focal
row's own slot and zero-weight slots. The local Geary, Pearson, and the two Gamma forms with an
order-equivalent pair sum expose `GPUPermutationStatisticAdapter` descriptors for the existing
permutation contributors; they do not duplicate permutation loops.

`GPURoseStatistic` consumes planar line starts and ends. Its required
`GPUCircularStatisticsContract` makes directional versus axial interpretation, radians, angular
origin, and equal-width bin count explicit. It reports count, mean angle, resultant length,
circular variance, Rayleigh z, and the weighted Rose bins for every group.

`GPUAZPRegions` is a greedy steepest-descent AZP: a bounded, deterministic local search from
caller-supplied contiguous labels with no tabu list, annealing or reactive escape, so it stops at the
first local optimum and the result depends on `initialLabels` (for several starts, run several
instances from different initial partitions and keep the smallest `objective`). Every candidate move
is checked for exact source connectivity. The initial labels are checked on the GPU as well: a region
that is not connected adds its extra components to `invalidCount` and the search does not run. The
operator publishes the final within-region sum of squares (the same quantity as `withinSsd` of
`GPURegionPartitionEvaluation` on the same table) plus packed iteration, convergence,
iteration-limit, and invalid-count status. `seed` fixes tie order and records the reproducible
initialization seed; `maximumIterations` is a hard compile-time bound. Optional `extensiveValues` and
`extensiveThreshold` forbid a move that would leave the source region below the threshold (the max-p
constraint). A multi-start AZP is deferred: one instance already costs `maximumIterations` rounds of
row-parallel candidate searches with a per-candidate reachability bitset, so parallel starts would
multiply the private memory and the dispatch count rather than fill idle lanes.

`GPUAZPRegions`, `GPUMaxPRegions` and `GPUWardRegions` share two input rules. **Attributes are
standardized by default** (`standardize`, default true): each column becomes a z-score over its finite
rows with the population standard deviation (divisor `n`, the `sklearn` `StandardScaler` convention),
a constant column becomes zero, and `objective` is in z-score units. Pass `standardize: false` for the
raw-unit behavior of earlier versions (a default change). **Adjacency is undirected**: row `i` is
adjacent to `j` when either row lists the other, so directed (for example k-nearest-neighbor) weights
are symmetrized inside the operator with `GPUSpatialWeightsTranspose`; weights whose `info.symmetric`
is `true` skip that pass.

`GPUWardRegions` begins with one region per row and repeatedly merges the adjacent pair with the
smallest Ward increase in within-region squared error. `targetRegionCount` and `maximumIterations`
bound the search; disconnected weights can stop it before the target, which is reported through the
shared optimization status. `GPUMaxPRegions` is a randomized multi-start greedy Max-P: `startCount`
(compile time, default 16, at most 64) independent constructions run in one dispatch, one workgroup
each with its own scratch labels, and each start grows contiguous regions until their extensive value
reaches a per-frame threshold, drawing its seed rows in a random order from Philox 4x32-10 keyed by
`(seed, start)`. An under-threshold remainder is merged into an adjacent region when possible. The
returned start has the fewest under-threshold leftover regions, then the most regions, then the
smallest within-region sum of squares, then the lowest start index; the status reports the region
count, any infeasible remainder, the `selectedStart` and the `improvementMoveCount`. With
`improvementIterations > 0` (default 0) the selected construction is then refined by greedy AZP moves
that keep every region at or above the threshold, which lowers `objective` at the same region count
(at most 65,536 rows); spopt's `MaxPHeuristic` anneals instead. The starts run side by side, so the
single-construction watchdog limit (about 50k-100k rows) is unchanged, and scratch memory is
`startCount * rows * (columnCount + 4) * 4` bytes. No `spopt` is installed in the oracle environment,
so Max-P quality is checked against the CPU selection mirror and invariants (contiguity, threshold,
`p` of 16 starts never below the first start) rather than against `spopt`.

`GPULocationAllocation` consumes a row-major demand-by-facility cost matrix and implements bounded
greedy p-median, maximum-coverage, and set-covering selection. Pair it with
`GPUNetworkCostMatrix` for network costs, or provide any dense cost matrix. Outputs include selected
facility flags, nearest assignments, the objective, uncovered demand count, completion, and invalid
input count. The contributor keeps selection on the GPU, but it does not claim the integer-programming
optimum or application-specific facility constraints of a packaged solver.

A `+Infinity` cost marks an unreachable demand-facility pair. Unreachable demand is never summed
into the p-median objective: the objective is the weighted cost of the reachable rows, the
unreachable rows are counted in `uncoveredDemandCount` (not `invalidCount`, which marks invalid input
and stops the run), their assignment is `0xffffffff` and their `assignedCosts` entry is `+Infinity`.
Greedy and swap steps compare p-median candidates lexicographically, first by the demand weight left
unreachable and then by the reachable cost, so one unreachable row cannot swamp the float32 cost
differences of the others.

After the greedy construction, p-median and maximum coverage run a Teitz-Bart vertex-substitution
(swap) phase, bounded by `maximumSwapIterations` (compile-time, integer 0 to 1000, default 100; 0 is
pure greedy; set covering has no swap phase). Each iteration scores every (selected facility, unselected
candidate) swap in parallel, applies the single best strictly improving swap (ties go to the lowest
selected facility, then the lowest candidate) and stops when none improves. A swap must gain more than
1e-6 of the objective scale, which keeps f32 summation noise from cycling. It needs
`maximumFacilityCount * facilityCount <= 2^24` and costs `O(maximumFacilityCount * facilityCount *
demandCount)` per iteration. The `status` output now has seven words: after the four earlier ones,
`swapIterations` (swaps applied), `converged` (1 when no improving swap remains, so the result is a
swap-local optimum) and `iterationLimitReached` (1 when the budget was used up, so local optimality is
not verified); all three are 0 when the phase is off. Behavior change: swaps are on by default, so
p-median and coverage results can differ from (and are never worse than) the old greedy result, and
`output.status` must hold seven words. Greedy maximum coverage alone has the `(1 - 1/e)` guarantee
relative to the optimum (Nemhauser, Wolsey and Fisher), which the swap phase preserves because it never
lowers the covered weight; greedy p-median has no constant-factor guarantee. Either way the result is a
local optimum, not the integer-programming optimum. It is checked against a brute-force optimum on small
instances.

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

`spatialContext` (optional) declares the coordinate space of `positions`. Omitted or planar, the search is
Euclidean in position units. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` clusters
`[longitude, latitude]` degrees with `epsilon` a great-circle distance in meters (kilometers with
`units: 'kilometers'`, or any `sphereRadius`; the mean Earth radius by default). Then `bounds` are
`[minLon, minLat, maxLon, maxLat]` degrees inside `[-180, 180] x [-90, 90]`. The neighbor test is the
haversine term against `sin^2(epsilon / (2 R))`, and each encoding sizes its cells to
`epsilon / R` of latitude by `2 asin(sin(epsilon / (2 R)) / cos(maxAbsLatitude))` of longitude (the whole
extent in one column near a pole), so the 3x3 search stays exact. The search does not wrap at the
antimeridian: a cluster straddling +-180 is split, so keep `bounds` on one side of it. `denseBoxShortcut`
is planar only, and other metrics (rhumb, ellipsoidal) throw. Geographic results match scikit-learn
`DBSCAN(metric='haversine')` on the test scene.
Planar positions are float32, so the representable spacing near a coordinate of magnitude `M` is about
`M * 1.2e-7`: UTM northings near 5e6 are only 0.5 m apart, and an `epsilon` below a few such spacings
(a few metres) is unreliable. Subtract a local origin from the coordinates (and from `bounds`) before
upload so the spacing stays far below `epsilon`. Float32 longitudes near 180 degrees have about 1 m spacing.

### `GPUKMeans`

Deterministic k-means over planar points (scikit-learn `KMeans`, PostGIS `ST_ClusterKMeans`). Props:
`positions`, optional planar `spatialContext` (a longitude/latitude context throws; project first), `k`
(1 to 256), `iterations` (1 to 64, a maximum), `tolerance`, `initialization` (`'kmeans++'` by default with
`seed`, or `'first-valid'`), `restartCount` (1 to 32, default 1) and the outputs `labels` (cluster or
noise), `centers` (`k` rows), optional `sizes`, `squaredDistances` (the explain column), `inertia` (a
one-row `float32`: the total inertia, the sum of `squaredDistances`) and `convergence`.
Behavior change: the default `initialization` was `'first-valid'`, which is order dependent and can place
centers next to each other; `'kmeans++'` matches scikit-learn. Each point goes to the nearest center with lowest-ID ties, means are fixed-order chunked
reductions (no sort). Results are bitwise reproducible per seed. `'kmeans++'` uses `log`, which WGSL does not
pin down across devices.

Behavior change (empty clusters): a center that receives no point is no longer left in place. After every
assignment step the empty clusters (lowest ID first) are relocated like scikit-learn's
`_relocate_empty_clusters_dense`: they take the points with the largest squared distance to their assigned
center, farthest first, and the donor clusters' counts and sums lose those points before the means are
taken. Ties in distance go to the lowest row. scikit-learn's `argpartition` leaves the order among the
relocated points unspecified; here it is always descending distance. If every point sits on its center
nothing is relocated, and a cluster that is still empty (or whose only member was taken) moves to the mean of
the largest cluster (lowest ID on ties). A center that is NaN because it could not be initialized (fewer valid
points than `k`) is never relocated. The selection is a single-workgroup iterated argmax with no atomics and
no readback; it costs `O(n E)` for `E` empty clusters and nothing when none is empty.

`restartCount` runs that many independent seeded initializations, each with its own Lloyd iterations, and
returns the one with the lowest total inertia (a fixed-order compensated sum; a tie goes to the lowest
restart index). Restart `r` uses the seed `getKMeansRestartSeed(seed, r)` (restart 0 uses `seed`), so any
single restart can be reproduced with `restartCount: 1` and that seed. The default of 1 matches
scikit-learn's `n_init='auto'` for k-means++. Restarts are separate node chains in the graph, so graph size
and GPU time grow linearly with `restartCount`; `convergence` and `inertia` describe the returned restart.
`restartCount` above 1 needs `'kmeans++'` (`'first-valid'` has no randomness).

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
so the per-iteration dispatches still run: the saving is kernel work, not dispatches.

### `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, and `GPURegionPartitionEvaluation`

Spatially constrained regionalization over a contiguity graph (PySAL spopt `SpanningForest` and SKATER,
Assuncao et al. 2006).

- `GPUSpatialWeightsMinimumSpanningTree`: Boruvka minimum spanning forest over a symmetric
  `GPUSpatialWeights` (every link in both rows; for directed weights build the union with
  `GPUSpatialWeightsTranspose` first, unlike the AZP, Max-P and Ward operators, which symmetrize
  themselves because the tree's edge IDs are the caller's CSR slots). The edge cost is the squared Euclidean distance between (optionally standardized)
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
  target counts every region. SKATER does not center attributes itself: the tree's `standardize`
  (default true) does, and `standardizedValues` should be passed as the SKATER `values` so the cuts and
  the tree share a metric and the f32 subtree-sum subtraction stays well conditioned.
- `GPURegionPartitionEvaluation`: scores any label column (SKATER, `GPUSpatialClustering`, `GPUKMeans`):
  region count, within, between and total sum of squared deviations, smallest and largest region, the
  fraction of weights-graph links that cross regions (a compactness proxy) and the number of ignored
  rows, plus per-label sizes and within-region deviations. The summary also holds `pseudoF`, the
  Calinski-Harabasz statistic `(between / (k - 1)) / (within / (n - k))`, pinned against
  `sklearn.metrics.calinski_harabasz_score` (NaN where undefined, 1 when the within-region dispersion
  is zero, as sklearn). The cost is `O(labelCapacity * rows)`.

```ts
graph.add(new GPUSpatialWeightsMinimumSpanningTree({weights, attributes, attributeCount: 3, treeEdgeFlags, componentLabels}));
graph.add(new GPUSkaterRegions({weights, attributes, attributeCount: 3, treeEdgeFlags, parameters: skater.importToGraph(graph),
  maximumRegionCount: 12, labels, regionCount}));
```

Cost: the publish and apply kernels are single-threaded and each SKATER step is
`O(sum of side sizes)`, which suits thousands of units and is quadratic on path-like trees of 100,000
rows or more.

### `GPUGroupGeometry`

Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude context
throws at construction (project first with `createGPULocalFrame` and `GPULocalFrameProjection`).
Geographic group centroids live in `GPUGeometryMeasures`.

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

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`.

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
`includeFocal`, weight `focalWeight`, default 1) and then the active CSR slots in slot order. A slot
is active when its weight is positive; zero-weight slots, a listed self slot, out-of-range slots,
zero-mask neighbors and, for numeric statistics, non-finite values are skipped. Zero-weight slots are
therefore not members (`count`, `min`, `max`, `median` and the categorical outputs ignore them),
which differs from libpysal `lag_spatial` and momepy when zero weights are listed. When
`weights.info.transform` is `'row'` the slot weights are divided by the sum of the active slot
weights, so `sum` (the lag) stays a weighted average when neighbors are masked or non-finite (the
focal weight is unchanged). If `weights.status.overflow` is nonzero every output is NaN. A masked-out focal row gives `count` 0 and NaN elsewhere. With `k` members, weights `w` and
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
- `smoothedRates`: `r = w y + (1 - w) m` with `w = a / (a + m / b)`. By default (`clampPriorVariance: true`)
  `a` is clamped at 0 (Marshall 1991, GeoDa), so `w` stays in `[0, 1]` and `a <= 0` (counts no more dispersed
  than Poisson, common for small `n` or low counts) shrinks every rate fully to `m`. esda `Empirical_Bayes`
  does not clamp, so with `a < 0` its `w` leaves `[0, 1]` and rates are pushed away from `m`, possibly below
  zero. Pass `clampPriorVariance: false` for exact esda parity. This is a default change.
- `standardizedRates` never clamps `a`; it keeps esda's `assuncao_rate` guard (`v = m / b` when `a + m / b < 0`).
- Optional `rawRates` and `summary` (`GPU_EMPIRICAL_BAYES_SUMMARY`, 8 floats: count, event sum,
  population sum, pooled rate, weighted rate variance, prior variance, clamped prior variance). The
  `priorVariance` slot stays unclamped (esda `ebi_a`); `clampedPriorVariance` is `max(a, 0)`.

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

esda 2.10.0 `Spatial_Empirical_Bayes` itself sets `r_var[r_var < 0] = 0`, so the clamp matches esda and needs
no option (unlike the global `Empirical_Bayes`, which does not clamp; see
[`GPUEmpiricalBayesRates`](#gpuempiricalbayesrates)). The spec pins esda 2.10.0 on a 6x6 queen lattice (spatial
rate within 1e-5 relative, smoothed within 2e-4 in f32) and on low-dispersion counts where the local `a`
clamps. Deviations: a neighborhood with no events gives `0 / 0 = NaN` in esda and 0 here; rows excluded by the
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
console.log(hotSpots.getInferenceMetadata());
// {alternative: 'two-sided', permutationCount: 0, multipleTesting: 'benjamini-hochberg',
//  testedRowsDefinition: 'valid space-time bins'}
```

Defaults changed: `multipleTesting` is `'benjamini-hochberg'` (it was no correction), `confidenceLevel` is 0.95
(it was 0.9), the trend level is `1 - confidenceLevel`, `minimumTrendSliceCount` is 10, and
`finalSlice` is `'require'`. Pass `multipleTesting: 'none'` for the previous rule.

Passes (deterministic, no float atomics):

1. Global `n`, mean and standard deviation of valid bins: fixed 1024-bin block sums added in order.
2. Space-time Gi* per bin: the spatial neighbors (lattice cells with `dx^2 + dy^2 <= radius^2`, or
   the cell's weights row plus the cell itself at `selfWeight`) over the current and `temporalWindow`
   previous slices, `z = sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))` over valid bins.
3. Two-sided normal p-values `erfc(|z| / sqrt(2))` of every valid Gi* bin, adjusted by
   `GPUMultipleTestingAdjustment` over the whole family of valid bins of the cube (never a display
   mask). A bin is hot or cold when its adjusted p-value is at most `1 - confidenceLevel`, with the
   sign of z. With `multipleTesting: 'none'` the rule is `|z| >= criticalZ` with the same two-sided
   critical values, and the p-value passes are skipped unless `pValues` or `adjustedPValues` is
   requested. Bins within a temporal window overlap and neighboring cells share observations, so the
   tests are dependent: Benjamini-Hochberg is only guaranteed under positive dependence, and
   `'benjamini-yekutieli'` controls the false discovery rate under arbitrary dependence at a cost in
   power of about `ln m`. The correction needs fewer than 2^24 bins.
4. Per-cell Mann-Kendall on the finite z series: exact integer `S`, tie-corrected variance
   `[n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18` (ties by exact f32 equality), continuity-corrected z, and a
   two-sided p-value from the Numerical Recipes `erfcc` fit. Consecutive Gi* values share
   `temporalWindow` slices, so the series is serially correlated and this p-value is anti-conservative
   (ESRI has the same property): treat the trend level as a screening threshold. Cells with fewer
   than `minimumTrendSliceCount` finite slices (default 10, ESRI guidance) get NaN `trendZ` and
   `trendP`.
5. Classification into the 17 `GPU_EMERGING_HOT_SPOT_CATEGORIES` (0 = no pattern, hot 1..8, cold 9..16:
   new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical) plus
   `UNDETERMINED` (17).

`UNDETERMINED` marks a cell whose category cannot be decided: with `finalSlice: 'require'` (the
default) a cell with data whose last slice is missing, and a cell that is persistent in one direction
(Intensifying, Persistent or Diminishing, hot or cold) but has fewer finite slices than
`minimumTrendSliceCount`. It is never silently reported as Persistent. `finalSlice: 'last-valid'`
keeps the old behavior, where the last finite slice stands in for the final slice, so a stale cell can
be labelled New or Persistent for a period it has no data in. Cells without any finite slice stay
`NO_PATTERN`. The optional `lastValidSlice` output gives each cell's last finite slice index
(`GPU_EMERGING_HOT_SPOT_NO_SLICE`, `0xffffffff`, for none).

Legend contract: category codes never change meaning with `multipleTesting`, `confidenceLevel` or
`finalSlice`; those options only change which bins are hot or cold and which cells are undetermined.
`getInferenceMetadata()` returns a plain `GPUInferenceMetadata` for the legend (it omits
`significanceLevel`, which is the per-frame `1 - confidenceLevel`).

`criticalZ` is only accepted with `multipleTesting: 'none'` (the parameter packer throws otherwise).
Without a correction `confidenceLevel` must be 0.9, 0.95 or 0.99 or `criticalZ` must be given; with a
correction any level in `(0, 1)` works.

Per-frame parameters (radius, temporal window, confidence level, critical z, trend significance,
persistence fraction, minimum trend length) live in a parameter buffer and never rebuild the graph.
Compile-time: `gridWidth`, `gridHeight`, `sliceCount` (at most 256), `maximumRadius` (at most 32),
`multipleTesting`, `finalSlice`, value format, optional views. Outputs are caller-owned: `giZScores`
(per bin), `trendZ`, `trendP`, `trendS`, `category`, `hotSliceCount`, `coldSliceCount` (per cell),
optional `pValues` and `adjustedPValues` (per bin), optional `lastValidSlice` (per cell) and optional
`globalStatistics` `[n, mean, variance, standardDeviation]`.

### `addSpaceTimeHotSpotsRecipe`

Timestamped events to an emerging hot spot map: `GPUCalendarBuckets` decodes the event times, an
adapter bins every event into a dense `cell * sliceCount + slice` cube, `GPUEmergingHotSpots`
classifies it and an ordinal `GPUColorScale` colors the categories (18 palette entries, 0 to 17).

Slices are absolute time steps, the ArcGIS "time step interval", so the Mann-Kendall trend runs over
a monotone axis:

```ts
addSpaceTimeHotSpotsRecipe(graph, {
  timestamps, calendarParameters, cells, parameters,
  slices: {step: 'month', interval: 3, origin: {year: 2022, month: 11}, count: 12},
  multipleTesting: 'benjamini-yekutieli'
});
```

Slice `floor((stepIndex(t) - stepIndex(origin)) / (interval * stepsPerUnit))` holds the events at
local time `t` (the same UTC-offset semantics as `GPUCalendarBuckets`). `stepIndex` counts whole
steps: `year`, `year * 4 + quarter - 1`, `year * 12 + month - 1`, days since the epoch for `'day'` and
`'week'` (a week is 7 days anchored to the origin day), hours since the epoch for `'hour'`. It is
integer arithmetic on the GPU (Hinnant's `days_from_civil`) and on the CPU
(`getSpaceTimeSliceIndex`). Events before the origin or past the last slice are skipped and counted
in the `slice-range` status stage (`invalidCount`, or the `skippedEventCount` output).

Cyclic fields (month of year, weekday, hour of day, day of year) are no longer accepted: `month = 1`
would pool every January of a multi-year range, and a trend over hour or weekday is meaningless.
Cyclic profiles belong in a non-trend summary such as `GPUCalendarBuckets` `hourWeekdayCounts`.

`multipleTesting` (default `'benjamini-hochberg'`) and `finalSlice` (default `'require'`) pass through
to `GPUEmergingHotSpots`; write the `parameters` buffer with the same `multipleTesting`. The result
carries a plain `inference: GPUInferenceMetadata` object, and optional `pValues`, `adjustedPValues`
and `lastValidSlice` outputs.

### `GPUGeographicDistribution`

`GPUGeographicDistribution` computes the ArcGIS "Measuring Geographic Distributions" statistics
over planar points, per group or per masked selection: counts, weight sums, mean centre, Weiszfeld
median centre, standard distance, standard deviational ellipse, linear directional mean, and
polygon rings for the ellipse and the standard-distance circle. Every statistic is Euclidean:
`spatialContext` (default native planar) rejects `longitude-latitude` at construction with a
"project positions to a planar frame first" error, and `GPURoseStatistic` does the same. A
spherical mean centre (normalised mean of 3D unit vectors) and a great-circle standard distance are
deferred, because they need a second implementation of every reduction pass in both the chunked and
the sorted path.

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
  pass, then a pass about the mean), never `E[x^2] - E[x]^2`. No small-sample degrees-of-freedom
  correction is applied, so for small `n` the spread is biased low by `sqrt((n - 1) / n)`; there is
  no confidence interval or significance test.
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

Coordinates: `spatialContext` defaults to native planar. Rectangle, lasso and pick selections are
topological tests and also accept `longitude-latitude` (metric `'none'`): edges are straight lines
in degree space and the antimeridian is not handled. A `'radius'` selection is a planar distance in
position units, so it throws at construction in `longitude-latitude` (the selection kind is
compile-time); project positions to a planar frame first, or use a rectangle or lasso. Geographic
metrics are rejected. `GPURegionMask` takes the same `spatialContext`; `GPUPickRegionMask` reads no
positions and has none.

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

Boundary points follow `boundaryRule` (passed to the join). The default `'half-open'` counts a point
on an edge or vertex shared by coverage features exactly once, whatever the feature row order; points
on the outer right or top edge of the coverage are counted by no feature. `'closed'` counts every
boundary point (smallest containing row wins) and `'open'` only strict interiors. The legacy
`includeBoundary: true | false` means `'closed'` / `'open'`.

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

- Variance. `output.variances` and `output.standardDeviations` are the unbiased estimates in two
  passes (mean first, then `sum w (v - m)^2`), so there is no `E[v^2] - E[v]^2` cancellation. Unweighted it
  is `ddof = 1` (`numpy.var(v, ddof=1)`). With `weights`, which act as reliability weights, it is
  `sum w (v - m)^2 / (V1 - V2 / V1)` with `V1 = sum w`, `V2 = sum w^2`, equal to
  `numpy.cov(v, aweights=w, ddof=1)`. A feature with fewer than two valid rows, or a non-positive
  denominator, gives NaN, never 0. Variances need packed point views (not chunked vectors) and follow
  `sumOrder`. Checked against numpy in `zonal_variance.py`.
- Weighted density. `output.weightedDensities` is `weightSums / area` (needs `weights`); the area rules
  of `densities` apply (finite and positive, else NaN).
- Coordinates. `spatialContext` accepts planar (default) or `longitude-latitude` with any metric.
  Membership runs `GPUPointInPolygonJoin` with a containment-only context derived from `spatialContext`
  (metric `'none'`), so in degrees it is topological with straight edges (not geodesics) and polygons
  crossing the antimeridian must be split. The shoelace area of degrees is
  wrong, so a longitude-latitude context rejects GPU areas (`densities`, `weightedDensities`,
  `featureAreas` without `areas`) at construction: pass spherical `areas` in square meters, for example
  from `GPUGeometryMeasures`. Counts, sums, means, extrema and variances need no area.
- Not provided: median and quantiles (they need a per-feature sort of the values), weighted minima and
  maxima, fractional polygon coverage (a point belongs to a feature or it does not).

### `GPUFocalStatistics`

Moving-window statistics over a float32 raster, like ArcGIS `FocalStatistics` and GRASS
`r.neighbors`: mean, sum, min, max, range, population standard deviation, and the count of valid
cells. The radius, window shape, minimum count, and center nodata policy are per-frame.
The operator is raster-only: it reads a lattice of values and never positions, so it takes no
`spatialContext`, and its radius is in cells, not ground units. Cell size and shape (for example
lon/lat cells that shrink with latitude) are the caller's concern.

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
`parameters` (`getGPULocalOutlierFactorParameterValues({threshold, densityFloor = 1e-10})`) change per
frame without recompiling. There is no default `threshold`: LOF has no null distribution, so no cut is
calibrated (the former hidden 1.5 is gone). Omit it and the mask and count are all 0 while `lof` is
unaffected; otherwise pick a cut from the score distribution. LOF is exploratory, not inferential: no
p-value, nothing "significant". Choose `k` (the neighbor count of the table) of at least 10 and larger than
the smallest group that should count as a cluster rather than as outliers (Breunig 2000), and compare
scores only at the same `k`. The input must be a true kNN list; with ties at the k-th distance a list cut
at exactly `k` entries changes the k-distance neighborhood (as in scikit-learn). The class reads only
distances, so it has no `spatialContext`: build the table from projected planar coordinates (raw lon/lat
degrees distort with latitude). The floor keeps coincident clusters finite at density `1 / densityFloor`, and
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
- Optional diagnostics (also available on their own as `GPUTransitionDiagnostics`, taking the `counts`
  tensor): `output.homogeneity` (needs `conditionCount > 1`; five rows `Q`, `Q_p_value`, `LR`,
  `LR_p_value`, `dof`, indexed by `GPU_TRANSITION_HOMOGENEITY`), `output.steadyStates` (ergodic
  distribution of the pooled chain, then one per condition, `chains * K` rows), `output.stateSparsity`
  (two rows, `GPU_TRANSITION_STATE_SPARSITY`: states with 0 < n < `minimumTransitions`, and states with
  n = 0) and `output.status.converged`. `minimumTransitions` defaults to 5. The steady-state solver keeps
  its elimination matrix in a storage scratch buffer the class allocates (`chains * K^2` floats), so 64
  classes fit the 8,192-byte private memory limit of core WebGPU.
- Homogeneity follows giddy's `homogeneity` (Bickenbach and Bode 2003): Q and LR compare each conditional
  matrix with the pooled one and are chi-square with `sum_i (b_i - 1)(A_i - 1)` degrees of freedom. A
  from-state with no transitions adds 0 degrees of freedom; giddy adds 1 for it, so `dof` differs from
  `dof_hom` by the number of empty pooled rows. The p-values are upper-tail chi-square (`'greater'`) from
  an f32 incomplete gamma (relative error below 6e-5 for p above 1e-6); `dof = 0` gives NaN. The test is
  asymptotic: with sparse rows (see `stateSparsity`) it is unreliable.
- Steady states use a direct Grassmann-Taksar-Heyman solve per chain. States with no transition in or out
  are dropped (probability 0, as giddy's `Markov` drops classes absent from the data); the remaining
  states must have an outgoing transition and form one communicating class, otherwise the whole chain is
  NaN (giddy raises or returns several distributions). `converged` is one when every chain that has
  transitions got a steady state. The solver is direct, so `iterationLimitReached` is never written.
  A periodic chain is solved (its stationary distribution, not a limit of powers).

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
  `lagMaximumClassCount x K x K`; slices past the produced `lagClassCount` stay zero. The `output` also
  takes the `GPUTransitionMatrix` diagnostics (`homogeneity`, `steadyStates`, `stateSparsity`,
  `status.converged`) and `minimumTransitions`: `homogeneity` is giddy's `Q`, `Q_p_value`, `LR`,
  `LR_p_value`, `dof_hom` across the lag classes; `steadyStates` is giddy's `s` (rows `0..K-1`, pooled) and
  `S` (rows `(1 + c) * K ..`), NaN where a chain is not ergodic. The pooled chain here sums the counted
  transitions over the lag classes, so it excludes rows without a lag class. A small `Q_p_value` says the
  spatial lag matters; a large one says the conditional matrices cannot be told apart from the pooled one.
  `getInferenceMetadata()` describes the tests (analytic, `'greater'`, no correction).
- `GPULISAMarkov` (giddy `LISA_Markov`): props `values`, `weights`, `periods` (at least `periodLag + 1`),
  optional `periodLag`, `parameters` (the `GPULocalMoran` view from
  `getGPUSpatialAutocorrelationParameterValues`, which carries the significance level), optional `mask`,
  `gating` and an `output` of `counts` and the optional `probabilities`, `rowTotals`, `ignored`,
  `steadyStates`, `stateSparsity` and `quadrants`. It runs `GPULocalMoran` per period and counts moves
  between the five states (`GPU_LISA_MARKOV_STATE_COUNT`: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL). The
  layout is the same for both `gating` modes, so a legend never changes with the option.
  - `gating: 'none'` (default, giddy `LISA_Markov` parity): every valid row is in its quadrant, the signs of
    the centered value and of the lag; state 0 stays empty and the chain is a four-state chain. A row at
    exactly the mean or with an exactly zero lag has no quadrant (esda would call it low) and is ignored.
  - `gating: 'significance'`: rows whose analytic local Moran p-value is above the per-frame significance
    level are state 0. The test is the per-period analytic z at an uncorrected level, not esda
    permutations, and each period is gated on its own, so a transition into or out of state 0 can be a
    threshold crossing rather than a change of the local pattern. Treat it as a map of significant
    clusters, not as a test of dynamics. `getInferenceMetadata(significanceLevel)` describes it and
    returns `undefined` for `'none'`.
  - Quadrant numbering is esda's default (`geoda_quads=False`): 1 HH, 2 LH, 3 LL, 4 HL. With
    `geoda_quads=True` esda swaps 2 and 3 (1 HH, 2 LL, 3 LH, 4 HL).
  - Missing data: a row-period that is masked, non-finite, or has no neighbor among the finite unmasked
    rows of that period is not a state. It is written as `0xffffffff` (`GPU_CLASS_ASSIGNMENT_NO_CLASS`) in
    `quadrants`, and every transition with such an end is counted in `ignored`, never as state 0. Each
    period excludes its invalid rows from the mean and the neighbor sums (`islands: 'exclude'`).
    giddy has no missing-data handling; parity is on the complete rows. `quadrants` (`rows * periods`)
    is optional.

Both emit one node chain per period, which suits periods in the tens. giddy `relative` rescaling, mean first
passage times and the permutation test `x2_rpvalue` are not built in: divide each period before
classifying.

### `GPUKnoxTest` and `GPUMantelTest`

Space-time interaction tests over the pairs of a self-join `GPUSpatialWeights` (typically radius-mode
`GPUNeighborSearch`): every unordered pair listed in either direction is a spatial pair and is counted
exactly once (a symmetric list takes it from the lower row, a pair listed only from the higher row, as in a
kNN list, is taken from that row; duplicate and self entries are ignored; lists need not be sorted). Results
are only as complete as the pair list; the capacity overflow flag lives on `GPUNeighborSearch`.

- `GPUKnoxTest({pairs, times, parameters, maximumPermutations, statistics, summary})` counts the pairs
  that are close in space and have `|t_i - t_j| <= timeThreshold`, then runs a Monte Carlo (modified Knox)
  test over `P` permuted time assignments. Counts are exact integers, and the expected value comes
  from the exact count of time-close pairs. `statistics` is `uint32` with `P + 1` entries. Knox's
  hypothesis is one-sided (an excess of close pairs), so the Knox p-value is `pseudoPGreater`.
  `getInferenceMetadata()` returns the CPU-side `GPUInferenceMetadata` (alternative `'greater'`, Monte Carlo,
  no multiple-testing correction). `getKnoxPoissonPValue(observed, expected)` gives the classic Poisson
  upper tail, but it assumes Poisson variance, ignores ties and ignores the Mantel-Kulldorff variance
  inflation, so it is anti-conservative; prefer the Monte Carlo p-value.
- `GPUMantelTest({pairs, ...})` (the weights need `distances`) is the Pearson correlation between pair
  distances and `|t_i - t_j|`, with permuted times. Sums are f32 shifted partials, within about 1e-4 of a
  double-precision oracle. The statistic runs over the listed pairs only, so with a distance band it is a
  band-truncated Mantel `r` (pairs beyond the band are absent) that differs from the full Mantel `r`;
  report the band with the result. `getInferenceMetadata()` states alternative `'greater'`: the interaction
  hypothesis is a positive space-time correlation, the upper-tail convention of pointpats `mantel`
  (`pseudoPTwoSided` is also in the summary if repulsion matters).

`timeThreshold`, `seed` and `permutations` are per-frame, written with
`getGPUSpaceTimeParameterValues({seed, permutations, timeThreshold})` (`uint32[4]`). `times` must be
finite and are float32, so raw epoch seconds (about 1.7e9) are spaced 128 s apart and epoch milliseconds
are unusable. Subtract an origin on the CPU with `getGPUSpaceTimeRelativeTimes(times, origin?,
requiredResolution?)`, which subtracts a float64 origin (default the minimum), returns
`{times, origin, resolution}` and throws if the span is too coarse for `requiredResolution`; the float32
spacing at a span `S` is `2^(floor(log2 S) - 23)` (about 0.5 ms for an hour, 1 s for 4.6 months of
seconds). `timeThreshold` is in the same relative units. `summary` (`float32`, 12 entries, `GPU_SPACE_TIME_SUMMARY`) holds the observed statistic, pair
count, time-close pair count, expected value, permutation mean and variance, greater and lesser counts,
the pseudo p-values (greater, lesser, two-sided, `(1 + count) / (P + 1)`) and the z-score. The Kulldorff
space-time scan is `GPUSpatialScanStatistic` with `timeBuckets > 1`.

Both tests permute the times with the keyed swap-or-not shuffle of `GPUGlobalPermutationTest`, which is
uniform on small domains (permutation 0 is the identity, permutation `k` is slot `k`). Permuted counts and
p-values for a given seed differ from earlier versions, which used a Feistel bijection that was measurably
non-uniform on small domains. At most `2^24 - 1` events are supported.

### `GPUSpatialScanStatistic`

Kulldorff's scan statistic (SaTScan, CARTO `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or
points, with `cases` (`uint32`) per zone and optional time bucket (`zone * timeBuckets + bucket`). The
compile-time `model` selects the null:

- `'poisson'` (default): needs `baseline` (population or expected counts); `E[cases] = baseline * C /
  sum(baseline)`. LLR `c ln(c/e) + (C - c) ln((C - c)/(C - e))`.
- `'bernoulli'`: needs `controls` (`uint32`, same layout); the population of a cell is `cases + controls`.
  LLR of Kulldorff (1997), computed as the sum of `o ln(o/e)` over the inside/outside x case/control
  table. Monte Carlo permutes the case labels over the fixed population (conditioning on the total
  cases), by drawing individuals without replacement. Window populations are f32 (exact below 2^24).
- `'space-time-permutation'` (Kulldorff et al. 2005): no baseline or population. The expectation of a
  cylinder is `(sum of its zones' totals over all buckets) * (sum of its buckets' totals over all zones) /
  C`, the Poisson LLR is applied to it, and Monte Carlo permutes the case times over the case locations
  (both margins fixed). A trend or seasonal cycle shared by every zone is therefore not a cluster, which
  the Poisson model with a static population would report as an "emerging cluster". Needs
  `timeBuckets > 1` to be meaningful.

Windows and defaults:

- Spatial: the `k` nearest zones of every center (the `'circle'` shape never splits equidistant zones;
  `'nearest'` is plain k-nearest), kept while they hold at most `maximumPopulationFraction` of the total
  population (default 0.5, as in SaTScan). The population share governs: the per-center neighbor
  tables hold up to the compile-time `maximumWindowZones` zones, which defaults to every zone while
  `zones^2 <= 2^24` (4096 zones) and otherwise to the largest capacity that fits
  (`GPU_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS`; use `getGPUSpatialScanLayout` to see it). Selecting the
  neighbors costs about `zones^2 * capacity / 32` per frame, so lower the capacity for large zone sets.
  A window the capacity cut short while the population fraction would still allow it is never silent:
  `summary.overflow` counts the affected centers and the cluster record carries
  `truncatedByZoneCap` (0 or 1). The per-frame `maximumWindowZones` may lower the capacity.
- Temporal: with `timeBuckets > 1` each window is a cylinder over every run of up to
  `maximumTimeBuckets` buckets. The default is 50% of the study period, `max(1, floor(timeBuckets / 2))`
  (SaTScan's default), no longer all buckets: with a global trend, pure-spatial windows over the whole
  period would otherwise dominate.
- Direction: per-frame `rateDirection` `'high'` (default, `c > e`), `'low'` (`c < e`) or `'both'`.
  Windows of the other direction score 0.
- Defaults: `maximumPopulationFraction` 0.5, `maximumWindowZones` every zone (bounded), `maximumTimeBuckets`
  50% of the study period, `windowShape` `'circle'`, `rateDirection` `'high'`, `maximumClusters` 8.

Outputs: the most likely cluster plus greedy, non-overlapping secondary clusters (no shared zone, time
ignored, same maximum distribution, as SaTScan) in `clusterIndices` (`GPU_SCAN_STATISTIC_CLUSTER_INDEX`)
and `clusterStatistics` (`GPU_SCAN_STATISTIC_CLUSTER`: LLR, observed, expected, p, radius, ratio,
`truncatedByZoneCap`); `statistics` (the observed maximum then the maximum of every Monte Carlo
replicate), `summary` (`GPU_SCAN_STATISTIC_SUMMARY`: cluster count, total cases, permutations,
`overflow`) and an optional per-zone best LLR `zoneStatistics`.

Inference: `p = (1 + #{max_rep >= LLR}) / (P + 1)`, a Monte Carlo maximum over every window, so it already
accounts for the search; no further correction applies. `getInferenceMetadata(parameters?)` returns the
CPU-side `GPUInferenceMetadata` (alternative `'greater'` for high rates, `'lesser'` for low and
`'two-sided'` for both, permutation count, `multipleTesting: 'none'`, the tested-window definition).
Poisson replicates are multinomial draws from an integer cumulative table (Philox streams of the
permutation contributors); the Bernoulli and space-time permutation draws run one invocation per
replicate (about `C` sequential draws, times `timeBuckets` for the latter). The result is deterministic
(fixed-order reductions, integer atomics, ties to the lowest index). Limits, the seed and the replicate
count are per-frame parameters (`getGPUSpatialScanParameterValues`), up to the compile-time
`maximumPermutations` (at most 16384).

Limits: ordinal and normal models, elliptic windows and Gumbel p-values are not provided. The replicate
table needs `(maximumPermutations + 1) * zones * timeBuckets <= 2^25` words and the cost is about
`(P + 1) * zones * windowZones * timeBuckets^2`. LLR below about 1e-3 is approximate in f32; `baseline`
must be finite and non-negative. No kernel binds more than 8 storage buffers, so the scan runs on `'core'`
WebGPU devices.

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

The operator reads weights, not positions, so it has no `spatialContext`; distance units and the coordinate
space (planar, or great-circle meters) are those of the weights' producer.

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
of `demandWeights`) and `demand`. Like `GPUCatchmentAccessibility` it reads weights only and has no
`spatialContext`.

### `GPUArealInterpolation`

Area-weighted transfer between two zone systems (tobler `area_interpolate`, CARTO `ENRICH_POLYGONS`) on
a common fine raster. Rasterize both systems to the same grid with `GPUPolygonRasterization` (or any zone
raster) and pass the two zone views, so H3 or hexagon rasters work too.

- Props: `sourceZones`, `targetZones` (`uint32`, same length; a value at or above the count means no
  zone), `sourceCount`, `targetCount`, optional `cellWeights` (a dasymetric raster; area becomes a sum of
  weights, and non-finite or non-positive cells carry no area), `denominator` (per kind, see below),
  `mode` (`'extensive'` by default or `'intensive'`, which selects `weights.weights`),
  `weights` (a caller-owned CSR with `targetCount + 1` offsets; the neighbor capacity is the maximum
  number of pairs), `alternateWeights` (the other normalization, from the same run), `areas` (raw overlap
  area per slot), `overflow`, `totalPairs` and `categories` (`{sourceCategories, categoryCount, output}`,
  `T x K`).
- The cell weight `m` (1, or `cellWeights`) summed per source `s` and target `t` gives `a_st`. Extensive:
  `w_ts = a_st / A_s`. Intensive: `w_ts = a_st / B_t`. `A_s` and `B_t` are the zone or overlap areas.
  Rows are targets and neighbors are sources, so the CSR invariants hold; it is a cross weights matrix.
- Denominators (default change). `denominator` is one value for both kinds or
  `{extensive?, intensive?}`, and the default is `{extensive: 'zone', intensive: 'overlap'}`. `'zone'`
  divides by the whole zone (tobler): the part of a source that falls outside the target system is not
  moved onto the targets, and is now reported (below). `'overlap'` divides by the part overlapping the
  other system. The intensive default is `'overlap'` because a target only partly covered by sources
  should be the mean of what covers it, not diluted toward zero; earlier versions used `'zone'` for both.
  Pass `denominator: 'zone'` to restore the old intensive result.
- Mass-conservation diagnostics. `sourceAreas` (`sourceCount` rows) is the raster area of each source
  (0 when it rasterized to no cell), `sourceTransferredFractions` is the overlap area over the extensive
  denominator (`1 - f` of the extensive mass is lost; with `'zone'`, the share of the source that lies
  in the target system; NaN without raster area), and `invalidCount` (one `uint32`) counts sources with no
  raster area, whose mass is lost. They are exact for the rasterized zones, not for the polygons.
- Sources smaller than a cell. `sourceAnchorCells` (`sourceCount` cell indices; out of range means none)
  gives a zone without cells a fallback cell, for example the cell containing its label point. The claim
  is made in a transient copy of `sourceZones`, in ascending source order, and succeeds only if the cell
  belongs to no source or to one that keeps another cell. Remaining empty sources stay in `invalidCount`.
- Coordinates. Planar by default (areas are cell counts or `cellWeights` sums, which is an equal-share
  only on an equal-area raster). With `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
  'great-circle'}` (or `'none'`), `extent` (the rasterization extent `[originLongitude, originLatitude,
  cellWidth, cellHeight]` in degrees, row 0 at the smallest latitude) and `width`, each cell weighs its
  spherical area `R^2 dLon (sin(top) - sin(bottom))` in square meters, times `cellWeights`, with `R` the
  WGS84 authalic radius or `context.sphereRadius`. Pair `areas` and `sourceAreas` are then square meters.
  Other spaces throw. Checked against a numeric integral of `R^2 cos(lat)` (`spherical_cell_area.py`).
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
(mass per cell). Each cell starts at `total / area(zone)`. Each iteration takes the focal mean over in-area
neighbors, restores the zone total additively, clamps negatives to 0 and rescales multiplicatively to the
total; zero-sum zones fill uniformly. Totals hold to f32 accuracy and values are non-negative after every
iteration. Zone sums are fixed-order segmented sums over cells sorted once by zone. There is no ancillary
initial surface and no barrier. `'rook'` on a checkerboard has a -1 eigenmode (Tobler's original); `'box'`
is smoother. Reference: Tobler (1979), JASA 74(367).

- Deviation from Tobler (1979). The restoring step is additive, then clamped, then multiplicative.
  Tobler alternates smoothing with a volume-restoring adjustment; the multiplicative step is kept
  because it is the only variant here that keeps exact totals and non-negativity together, but where
  the additive step clamped it can leave flat plateaus, and an additive-only variant would go negative
  next to much larger zones.
- Convergence. `iterations` is now the maximum. A one-row `tolerance` (per frame) compares, after each
  iteration, the largest `|new - old| / (total / area(zone))` over all cells with it. Once at or below
  the tolerance the surface is final and every remaining compiled iteration is a no-op (deterministic).
  `converged` and `iterationLimitReached` (one `uint32` each; they need `tolerance`) report which
  happened; with 0 iterations both are 0. A NaN or negative tolerance never converges.
- Zones without cells. A zone smaller than a cell has no raster cell and its total used to vanish
  silently. `zoneAnchorCells` (`zoneCount` cell indices) lets such a zone claim a fallback cell (see
  `GPUArealInterpolation.sourceAnchorCells`; ascending zone order, the cell's owner must keep one cell),
  and `invalidCount` counts zones that still have no cell or no area.
- Coordinates. With a `longitude-latitude` `spatialContext` plus `extent` (as above) the smoothed
  quantity is the density per square meter and the totals are enforced on density times the spherical
  cell area, so `output` is mass per cell (density times area) and the optional `density` view is the
  surface itself. Without it every cell counts the same, which is only right on an equal-area raster.
  The focal mean is still by cell index, so away from the equator it is anisotropic on the ground.
- Limits. Counts are f32 sums, exact to 2^24 cells per zone; a larger zone loses count precision (the
  mass still holds to f32 accuracy). Oracles: a float64 numpy reimplementation of the documented
  algorithm including tolerance and anchors (`pycnophylactic.py`).

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
  are NaN, never 0 (0 means perfectly even): H and the diversity with an empty total, H also with a single
  present group (`E = 0`, while `E` itself is the defined 0), D with an empty total or `I = 0`, `D_g` and
  `A_g` for a group with zero or all of the population, and isolation and interaction of a group with no
  population. A unit with an empty environment (no population in it or its neighbors) has NaN local values
  and contributes nothing to the sums, so use a NaN-skipping sum to add the local outputs up to the global
  indices; unit contributions to `D_g` and H are also NaN when that global index is undefined. Behavior
  change: these were 0.
- Small-sample bias: the indices are not bias corrected. `D` and `H` are biased upward when units hold
  few people, because sampling noise alone makes unit compositions differ from the global one even under
  random allocation (Cortese, Falk and Cohen 1976; Winship 1977; Carrington and Troske 1997). Compare
  units of similar size, or against a random-allocation null, before reading a small `D` or `H` as
  segregation. Isolation and interaction are less affected but depend on the group share.
- Float32 counts and totals are integer exact only up to 2^24 = 16,777,216 total population; above that
  `T` and `X_m` round (about 6e-8 relative per addition).
- Sums are f32, within 1e-5 to 2e-4 relative of float64 and of PySAL on the test fixtures. Memory is
  `O(units * K^2)` per scale. The boundary-based `SpatialDissim` and permutation inference are not
  provided: PySAL's evenness null redraws every person's unit from the unit sizes (a binomial or
  multinomial draw per unit) and reruns the whole index pipeline per simulation, whereas the permutation
  operators here shuffle values across rows of an existing weights structure, so it needs its own
  count-table resampling kernel. `atkinsonB` and `selfWeight` are compile-time.

## Spatial joins

### `GPUSpatialPredicateJoin`

Joins two sets of features by a spatial predicate, like GeoPandas `sjoin` or PostGIS
`ST_Intersects`, `ST_Contains`, `ST_Within` and `ST_DWithin`. Coordinates are planar by default; a
longitude/latitude `spatialContext` selects great-circle `dwithin` in meters (see
[Spatial context](#spatial-context-of-gpuspatialpredicatejoin)). Either side is
`{kind: 'points', positions}`, `{kind: 'lines', positions, lineOffsets}` (one linestring per
feature) or `{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}`
(multipolygons, holes as later rings, GeoArrow layout); all nine combinations are supported.

`predicate` is `'intersects'`, `'contains'`, `'within'`, `'dwithin'` (with an inclusive
`distance`: planar in coordinate units, or meters with a great-circle `spatialContext`), `'covers'`, `'coveredBy'`, `'touches'`, `'crosses'`, `'overlaps'`, `'equals'`,
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
uncertain are decided by the f32 kernel and counted in `uncertainCount`), then a scan and a
capacity-bounded scatter. Output `pairs` (`leftIds`, `rightIds`, `count`, `overflow`, optional
`requiredCount`) are unique and sorted by `(left, right)`. On overflow the output is a sorted subset and
`requiredCount` and `candidateCount` give the sizes to resize to.

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

Orientation tests are exact but distances and edge piece midpoints run in f32, so recenter or project
coordinates before measuring near-degenerate geometry. Polygons must be valid.

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
`'FF2F11212'`. Bit 31 (`GPU_SPATIAL_RELATE_UNCERTAIN_BIT`) of a slot is set when an orientation of that pair
had no sign (see **Exactness**); `formatGPUSpatialRelate` ignores it, and `value & 0x3ffff` is the matrix alone.
It is not available for `dwithin`.

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

**Exactness.** Every orientation sign is exact, in the short-circuiting kernels (`engine: 'fast'`) and in
the relate engine alike: an f32 filter decides the clear cases and the rest fall back to exact integer
arithmetic (256-bit), with one inlined copy per kernel. So touches, crosses, covers, `intersects`,
`contains` and the other predicates do not depend on f32 rounding of the orientation test, and the result
does not depend on the engine or on the feature size that `engine: 'auto'` looks at, including
shared-edge and collinear configurations. The exactness is with respect to the f32 vertices; edge piece
midpoints, break parameters and distances are computed in f32: the location of a piece shorter than about
one ulp is not exact, so a line starting one ulp outside a polygon edge can have its `crosses` piece
classified as boundary.

**Uncertain pairs.** A pair is uncertain when an orientation has no sign: a non-finite coordinate, or
product exponents spanning more than 200 bits. It is never dropped silently. The undecidable orientation is
treated as collinear (sign 0) and the predicate is evaluated with that value, so the pair can come out
wrong in either direction, and it is counted in `uncertainCount` (zero means every decision is certified).
Matched uncertain pairs are identifiable per pair through bit 31 of the `relate` output. Point/polygon
pairs use the robust double-single classifier (they do not get the exact re-decision of
`GPUPointInPolygonJoin`); pairs it cannot certify are decided by the f32 kernel and counted. Not tracked
(reported as 0): the workgroup `dwithin` kernel, and the short-circuiting kernel of a polygon/polygon pair,
whose eight storage bindings leave no room for the counter.

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
vertices. A pair with hundreds of shared edges costs 5–10 ms. A great-circle `dwithin` always uses the
short-circuiting kernel (`'relate'` throws, `'auto'` picks `'fast'`).

#### Spatial context of `GPUSpatialPredicateJoin`

`spatialContext` (omitted: native planar) selects how positions are read. It is validated at construction and
exposed as the readonly field `spatialContext`; unsupported combinations throw.

| Context | Predicates | Meaning |
|---|---|---|
| planar (default) | all | Euclidean plane; `dwithin` `distance` in coordinate units. |
| `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` | `dwithin` only | positions are longitude/latitude degrees; `distance` is meters on the mean Earth sphere (`sphereRadius` overrides the radius; `units: 'kilometers'` throws). |
| `{coordinateSpace: 'longitude-latitude', metric: 'none'}` | every predicate except `dwithin` | topology in the degree plane, as GEOS and PostGIS `geometry` do for coordinates in degrees; no distances. |

Great-circle `dwithin`: the candidate stage widens each left box by the angular radius in latitude and by the
longitude reach of that radius at the box's far latitude (`2 asin(sin(r/2) / cos(lat))`, every longitude when
the radius reaches a pole), and a box that crosses the antimeridian also probes its copy shifted by 360
degrees, with a 0.1% plus 1e-5 degree margin so rounding never drops a candidate. The exact kernel measures
the central angle from a vertex to an edge with `geodesicSegmentFraction` and `geodesicSegmentDistance`
(`GEODESIC_WGSL`): edges are straight in longitude/latitude with the longitude difference wrapped at +-180,
the fraction of the nearest point is found in the local equirectangular frame of the point, and the
distance is the haversine of coordinate differences, which keeps meter-scale offsets well conditioned in f32.
The same edge model is used by the intersection and containment tests (exact orientation in degrees, as the
planar kernels) and by the network snapping contributors, so there is one convention rather than two. The
foot is exact to first order for edges of road length, so densify long edges. Distances are accurate to f32
rounding across the antimeridian and over a pole, up to about 8000 km (a larger compile-time distance throws,
a larger per-frame value selects nothing). Pairs whose geometries intersect in the degree plane count as
distance 0. Polygons that span the antimeridian are not split. Oracle: numpy haversine and
`pyproj.Geod(a=R, f=0)` for point pairs, shapely distances in the azimuthal equidistant projection for
polygons (see the spec fixtures).

The topological predicates take `metric: 'none'`; `metric: 'great-circle'` with them throws, and `dwithin`
with `metric: 'none'` throws, so the intent is explicit. `GPUSpatialJoinCandidates` and `GPUPointInPolygonJoin`
are planar only and throw for longitude/latitude with a hint to project first; `GPUNearestFeatureJoin` and
`GPUBufferSelection` also take `metric: 'great-circle'` (see their section); `GPUSpatialJoinPrepared` and `GPUNearestFeatureWeights` take no positions with a metric
(a prepared tree is a bounding-box tree and may serve a great-circle join) and have no context.

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
(capacity `pairs.leftIds.length`, `count`, `overflow`, `requiredCount`); slots past `count` hold `0xffffffff`.
It is conservative: no intersecting pair is missed, and some candidates do not intersect. It has no
`excludeSameRow`.

`distance` is planar in coordinate units (`spatialContext` defaults to native planar coordinates and a
`longitude-latitude` context throws; use `GPUSpatialPredicateJoin` `dwithin` for great-circle candidates).
It may be a compile-time number or a one-row `GraphDataView<'float32'>`. The view is read every
encoding, so an interactive distance join can reuse one compiled graph. Size the pair capacity for the
largest distance that the application permits. Negative, NaN and infinite view values act as zero; invalid
compile-time numbers throw.

```ts
const candidateDistance = new GPUParameterBuffer(device, {
  id: 'candidate-distance',
  format: 'float32',
  length: 1
});

graph.add(new GPUSpatialJoinCandidates({
  left,
  right,
  distance: candidateDistance.importToGraph(graph),
  pairs: {leftIds, rightIds, count, overflow, requiredCount}
}));

const compiled = graph.compile();
candidateDistance.write(Float32Array.of(2_000));
compiled.encode(commandEncoder, {parameters: undefined});
```

### `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin`

Spatial joins over a `GPUBVH` built from feature bounds every encoding. `GPUPointInPolygonJoin`
classifies each (point, feature) candidate with the shared exact orientation sign (even-odd rings, union
polygons) and assigns each point the containing feature with the smallest row (see the boundary rule
below); `GPUNearestFeatureJoin` measures candidates with
`GPUPairwisePointSegmentDistance` and keeps the nearest point or segment feature within a per-frame
radius. Both write per-point feature IDs, per-feature counts, optional compact matches, and one
overflow flag covering BVH leaf, candidate, and match capacity. `GPUPointInPolygonJoin` takes planar
coordinates (the default) or `{coordinateSpace: 'longitude-latitude', metric: 'none'}`. Containment is
topological with straight edges in degrees, as in GeoJSON, and gives the same result as planar. A
longitude/latitude context with a metric such as `'great-circle'` throws at construction, because the join
has no geodesic edges ("project positions to a planar frame first"). `GPUNearestFeatureJoin` also takes the
great-circle context below.

**Great-circle nearest features (`GPUNearestFeatureJoin`, `GPUBufferSelection`).** With
`spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` positions are longitude and
latitude in degrees and every distance (`radius`, `maxDistance`, `nearestDistances`, `neighborDistances`,
`GPUBufferSelection` `distance` and `distances`) is meters on a sphere of radius
`GPU_EARTH_MEAN_RADIUS_METERS` (`sphereRadius` of the context overrides it; `units: 'kilometers'`, a
`longitude-latitude` context with another metric, and planar contexts with `great-circle` throw at
construction). Both output modes use the branch-and-bound BVH traversal of the neighbors mode (nearest-feature
mode is the `k = 1` traversal followed by the nearest-feature columns, counts and matches), so
`candidateCapacity` is not used on this path (it is accepted and ignored, `candidateCount` reads 0, and
`GPUBufferSelection` still validates it). The traversal orders and bounds by central angle. A node is pruned
with a lower bound of the great-circle distance between its box and the query box,
`hav(d) = hav(dlat) + cos^2(max |lat|) hav(dlon)`, with the latitude gap, the circular longitude gap (through
the antimeridian when that is shorter) and the cosine of the latitude farthest from the equator over both
boxes, which never overestimates; a box 180 degrees or wider in longitude (a feature across the antimeridian)
counts as covering every longitude, and a box touching a pole has cosine 0. Edges are straight in
longitude/latitude with wrapped longitude differences, so the vertex boxes bound them exactly. Point to edge
distance uses `geodesicSegmentFraction` and `geodesicSegmentDistance`; a crossing or containment is decided
in the degree plane with the exact orientation sign after the second edge is shifted by a multiple of 360
degrees to overlap the first, and counts as distance 0. Foot points and query points are on the edges with
longitude wrapped to +-180. The mixed convention of the planar-in-degrees tests is therefore one edge model;
densify edges longer than a few kilometers, because the foot is exact to first order in the local frame of the
query. A polygon that spans the antimeridian is not split. Positions are float32, so a distance carries the
quantization of the coordinates (about 1 m at longitude 100). The planar kernels are unchanged and compile
none of this code. Oracle: `geodesic-nearest-fixtures.ts` (numpy 2.5.4 haversine of the edge densified in
longitude/latitude, `pyproj` 3.8.0 `Geod(a=R, f=0)` cross-check), scenes at the equator, latitude 60, across
the antimeridian and next to the pole. `GPUNearestFeatureWeights` is an adapter over the join's outputs and has
no context: its distances, `distanceFloor` and inverse-distance weights are in meters when the join is
great-circle.

**Exact orientation.** The classifier and the nearest-feature kernels use the same exact orientation sign
(an f32 filter with an exact 256-bit integer fallback), so "on the edge" means the same thing in the
point-in-polygon join, the predicate join and the nearest-feature join: a point exactly on an edge is
distance 0 in all three, and polygon containment (distance 0 inside a polygon) is decided with the exact
sign too. Distances and foot points stay f32. Only candidates that no arithmetic can decide stay
`uncertain`: non-finite input or malformed offsets. In `GPUPointInPolygonJoin` an empty polygon part
or a ring with fewer than three vertices contains nothing and is skipped, as in shapely, so the
feature's other parts still classify the point. Uncertain
pairs are never assigned. They are counted in the optional `uncertainCount`, so such a point stays
unassigned visibly rather than silently.

**Boundary rule (`GPUPointInPolygonJoin`).** `boundaryRule` decides a point exactly on a polygon
boundary:

| Rule | Boundary point | Notes |
|---|---|---|
| `'half-open'` (default) | belongs to exactly one feature of a valid coverage, independent of row order | The owner is the feature containing the point after an infinitesimal move toward +x and +y (+y dominates): left and bottom boundaries are included, right and top boundaries are excluded. A horizontal edge belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the feature whose interior touches its upper right. Points on the outer right or top boundary of a coverage belong to no feature. |
| `'closed'` | contained by every feature it touches, then the smallest row wins | Nothing is lost on the outer boundary; the owner of a shared-boundary point depends on the row order. This was the behavior before the rule existed. |
| `'open'` | belongs to no feature | Only strict interior counts; boundary points are lost, not half-assigned. |

`includeBoundary` is the legacy boolean: `true` is `boundaryRule: 'closed'`, `false` is `'open'`, and it throws
when it disagrees with an explicit `boundaryRule`. It is not a switch for the half-open rule: with neither
prop set the join is half-open. The half-open decision is the crossing-number convention made exact: a
supporting-line tie is broken by the sign of dx (the epsilon term of the move), or of -dy for a vertical
edge, so the result follows from the exact orientation signs and the vertex order does not matter. Overlapping
features (an invalid coverage) still resolve to the smallest containing row. Oracle: shapely
`Polygon.contains(Point(x + 1e-9, y + 1e-6))` on integer scenes, 24 row permutations, in the specs.
`GPUZonalStatistics` and the recipes that pass no `boundaryRule` use the half-open default, so a coverage's
counts partition the points.

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
and `weights.distances` is filled when present (in the units of the join: meters for a great-circle join, so `distanceFloor` is meters there). The output is directed, so for `'symmetrize'` or symmetric
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

Dataframe joints around the joins. `GPUPairGather` consumes join row indices and columns, so it takes no
`spatialContext`.

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
  output is bounded with `count`, `overflow` and `requiredCount`. Offsets monotonicity is not checked.
- **`GPUBoundsFilter`** (`spatialContext`: planar, or longitude/latitude with metric `'none'`, treated
  as plain coordinates with no antimeridian wrap, so a box needs `minX <= maxX`) is a `.cx`-style mask and compaction over `float32x4` feature bounds with a
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
from the publish node, so it can be bound as an indirect draw instance count. The distance is planar in
coordinate units by default; with `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` it is a great-circle distance in meters (forwarded to `GPUNearestFeatureJoin`, whose section
lists the edge model, the sphere radius and the rejected combinations), and `candidateCapacity` is ignored.

```ts
graph.add(new GPUBufferSelection({
  points, features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  distance: bufferDistance.importToGraph(graph), candidateCapacity: points.length * 4,
  spatialSort: true, outputMask, output
}));
```

## Geometry validity & intersections

### `GPUSegmentIntersection`

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). Orientation signs are exact.

Finds every intersecting pair of line segments between two geometries (`left`, `right`) or within one
(`left` only), like turf `lineIntersect` and `kinks`, self-intersection tests and noding points. Each side
is a `GPUSpatialJoinLines` or `GPUSpatialJoinPolygons` layout (one kind per side). A segment is identified
by the index of its start vertex. Polygon rings close implicitly; rings with fewer than three vertices,
zero-length segments and segments with non-finite coordinates are skipped.

| Prop | Meaning |
| --- | --- |
| `left`, `right?` | Geometries. Omit `right` for self mode (`left < right`, adjacent segments skipped) |
| `sameFeatureOnly?` | Self mode: only same-feature pairs (per-feature self-intersection and ring crossings) |
| `pairs` | `{leftIds, rightIds, count, overflow, requiredCount?}`, sorted by `(left, right)`; capacity is `leftIds.length` |
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
On overflow the output keeps a sorted prefix (`count = min(total, capacity)`, `overflow = 1`, `requiredCount`
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

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). Orientation signs are exact.

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

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`).

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

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`.

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
polygon's own segments are). Every orientation sign is exact for the f32 vertices, including the winding
number of the interior probe (`orientSign`), so a probe point exactly on a neighbor's edge is never
misclassified by rounding; the probe is the f32 midpoint of the segment. A pair with no decidable orientation
(non-finite input) sets the `uncertain` bit. The operator is planar: `spatialContext` defaults to native
planar coordinates and `longitude-latitude` throws (project first).

## Density & interpolation

### `GPUPointDensity`

Bins points into a square grid (`GPUGridBinning`, `GPUGridAggregation`) or pointy-top hexagons
(keys kernel plus `GPUGroupAggregation`) and writes per-cell counts, sums, means, a heatmap field,
its `[min, max]` extent, a histogram, and an optional `r32float` texture. Bounds and hexagon radius
may be GPU views; the field can be smoothed with a per-frame `GPUConvolution` kernel. Instead of a full
`kernel`, pass a separable `separableKernel: {horizontal, vertical}` (one of the two is required), which smooths
in two 1D passes.

**This is binned counting, not a kernel density estimate.** `output.values` is the count (or weight
sum or mean) per cell, so it changes with `gridSize` and is not an intensity per unit area or a
probability density. Optional `smoothing` convolves that field with a kernel whose radius and sigma
are in **cells**, not ground units, using a zero boundary: kernel mass leaks out of the domain, so the
field is underestimated within about two sigma of the border, and a grid whose cells are not square on
the ground smooths anisotropically. For intensity per unit area, a bandwidth in ground units, a
choice of kernels, and edge correction use [`GPUKernelDensity`](#gpukerneldensity). `spatialContext`
is planar/native only (omitted means planar/native); longitude/latitude throws, so project first.

Weighted sums default to `sumAccumulation: 'auto'`. The static heuristic uses workgroup
pre-aggregation only for at least 1,024 input rows, at most 256 cells, and at least 32 input rows per
cell on average; other inputs use direct float atomics and avoid the key pass and quadratic
workgroup-local scan. The heuristic cannot inspect GPU-resident cell keys, so set `'workgroup'`
explicitly for a known hotspot distribution or `'atomic'` for known diffuse input. Each 256-row
workgroup pre-aggregates in shared memory before writing to the grid, so a hotspot no longer
serializes on one cell (90% of 1M points in one cell drops from seconds to about 4 ms, at about 2 ms
extra on uniform data). Either way, sums can differ from a CPU sum in the last bits.

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

### `GPUKernelDensity`

A true planar kernel density estimate on a raster, with the bandwidth in ground units. For each output
cell center `u` it evaluates `lambda(u) = sum_i w_i K_h(|u - x_i|)` over the points inside the kernel
support, where `K_h(d) = K(d / h) / h^2` is radially symmetric and integrates to one over the plane
(Silverman 1986, Density Estimation, Table 3.1):

| `kernel` | `K_h(d)` for `t = d^2 / h^2` | Support |
| --- | --- | --- |
| `'gaussian'` (default) | `exp(-t / 2) / (2 pi h^2)` | truncated at `5h`; lost mass `exp(-12.5)`, 3.7e-6 |
| `'quartic'` (biweight) | `3 / pi (1 - t)^2 / h^2` | `t < 1` |
| `'epanechnikov'` | `2 / pi (1 - t) / h^2` | `t < 1` |

```ts
const kdeParameters = new GPUParameterBuffer(device, {id: 'kde', format: 'float32', length: 9});
graph.add(
  new GPUKernelDensity({
    positions, // float32x2, planar frame with equal units on both axes
    weights, // optional float32 (populations, counts); NaN or negative excludes the point
    parameters: kdeParameters.importToGraph(graph),
    width: 512,
    height: 512,
    indexGridSize: [64, 64], // internal GPUGridIndex over the points
    indexBounds: [minX, minY, maxX, maxY], // must cover the points that should count
    kernel: 'quartic',
    edgeCorrection: 'uniform', // default
    normalization: 'intensity', // default
    output: {values, weightSum, invalidCount} // weightSum and invalidCount are optional
  })
);
kdeParameters.write(
  getGPUKernelDensityParameterValues({extent: [x0, y0, x1, y1], bandwidth: 250 /* meters */}) // no recompile
);
```

- **Units.** `normalization: 'intensity'` (default) is the sum of weights per unit area, position
  units to the power -2: points per square meter in a meter frame, so the value does not change with
  the raster resolution. Multiply by a cell area to get expected weight per cell. `'probability'`
  divides by the sum of included weights (`output.weightSum`), giving a probability density that
  integrates to about one over the window (exactly one only without edge correction and with the
  whole kernel inside the window). Cells are evaluated at their centers, not averaged over the cell.
- **Edge correction** repairs the downward bias within one bandwidth of the study window, where kernel
  mass falls outside the data region. `'uniform'` (default) divides each cell by `e(u)`, the kernel
  mass inside the window (Diggle 1985; spatstat `density.ppp(edge = TRUE)`). `'diggle'` instead
  weights each point by `1 / e(x_i)` (Jones 1993; spatstat `diggle = TRUE`), which keeps the total
  closer to the point count but is noisier at the edge. `'none'` is the plain sum. The window is
  `window` in `getGPUKernelDensityParameterValues` and defaults to the extent; with correction, points
  outside the window are excluded and cells whose centers are outside are NaN. `e(u)` is closed form,
  with no quadrature: normal-CDF differences for the gaussian, and an exact inclusion-exclusion of
  four disk-clipped quadrant integrals for quartic and epanechnikov. `getGPUKernelDensityWindowMass`
  is the float64 reference. The window is a rectangle; an irregular study area is not supported (use
  the extent that matches your bounding region and mask the output, accepting the bias at concave edges).
- **Bandwidth.** It is per-frame, in position units, and should reflect the question (a service
  distance, a neighborhood). Normal-reference rules on read-back positions are provided:
  `getScottBandwidth` (`h_i = n_eff^(-1/6) sigma_i`) and `getSilvermanBandwidth`, matching
  `scipy.stats.gaussian_kde` including weights through the effective sample size
  `(sum w)^2 / sum w^2`. In two dimensions the two rules coincide. They return per-axis bandwidths and a
  pooled `isotropic` one for the radial kernels, and oversmooth clustered data. Likelihood
  cross-validation and Diggle's selector are not provided.
- **Exclusions.** A non-finite position or weight, a negative weight, and a position outside
  `indexBounds` exclude the row; `output.invalidCount` (status vocabulary `invalidCount`) is their
  number. Zero-weight rows are included.
- **Determinism and precision.** The index is rebuilt every encoding, IDs are re-ranked inside each
  index cell, and each cell sums in a fixed order (index row, column, ascending row), with no float
  atomics. Distances use offsets from the extent origin to limit f32 cancellation; the f32 inputs
  themselves are not recentered, so use local coordinates for fine resolution far from the origin.
- Pinned to `sklearn.neighbors.KernelDensity` (gaussian, epanechnikov), numpy (quartic), and
  `scipy.integrate.dblquad` (window mass, edge-corrected fields); bandwidth rules to `scipy.stats.gaussian_kde`.
- `spatialContext` is planar/native only (omitted means planar/native).

Not provided: adaptive or space-time bandwidths, network KDE (see the network contributors), reflection
boundaries, and non-rectangular windows.

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
    parameters: idwParameters.importToGraph(graph), // 9 float32
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
    taper: 'none', // or 'modified-shepard' (Franke-Nielson, needs a finite radius)
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
- Units and coordinates: by default positions, extent and radius share one planar unit and
  distances are differences of offsets from the extent minimum, so large absolute coordinates (UTM
  northings) do not quantize them. With `spatialContext: {coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'}` positions are absolute lon/lat degrees, the index and output extent stay
  in degrees, and distances are haversine meters on a sphere (`sphereRadius`, else 6371008.8 m;
  kilometers with `units: 'kilometers'`), so `searchRadius` is in meters. Neighbors and radius
  tests order by the haversine term. The query box is conservative (longitude span widened by the
  cosine of the highest latitude reached, full span near the poles) but does not wrap the
  antimeridian, and the nearest-k limit scans the whole box instead of early-stopping. f32
  absolute lon/lat gives about 1 m accuracy at Earth scale. Other longitude/latitude metrics
  (rhumb, ellipsoidal, none) throw; project to a planar frame (for example `createGPULocalFrame`)
  for them.
- `taper: 'none'` (default) is `w = 1 / d^p` inside the radius, a hard cutoff: a sample entering
  or leaving the radius changes the surface discontinuously. `taper: 'modified-shepard'` is the
  Franke and Nielson (1980) weight `w = ((R - d)+ / (R d))^p` with `R = searchRadius`, which falls
  to zero at the radius (a sample exactly at `R` has weight zero). It needs a finite radius
  (`getGPUInverseDistanceWeightingParameterValues` throws otherwise). `counts` still counts every
  sample inside the radius. Default stays `'none'` for backward compatibility; prefer the taper
  when a finite radius is a modelling choice rather than just a speed limit.
- Parameter guidance: `power` 2 is conventional, not optimal; higher powers flatten the surface
  around samples. `minimumNeighborCount` defaults to 1, so a lone distant sample inside the radius
  paints a constant plateau of its value; use 3 or more to leave sparse cells as nodata. IDW
  has no direction or sector search and no leave-one-out output; use `GPUKriging`'s
  cross-validation to compare models on the same samples.
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

```ts
graph.add(
  new GPUKriging({
    positions, values, parameters: krigingParameters.importToGraph(graph),
    width, height, indexGridSize, indexBounds, maximumNeighborCount: 16,
    output: {
      values: surface, variance, // float32 rasters
      invalidCount, uncertainCount, // optional uint32 scalars
      crossValidationPrediction, crossValidationVariance // optional float32 per SAMPLE row
    }
  })
);
```

- Props: `positions`, `values` (NaN skips), optional `mask`, `parameters` (`getGPUKrigingParameterValues`,
  at least 12 float32), `width`, `height`, `indexGridSize`, `indexBounds`, `maximumNeighborCount` (1 to 16,
  compile-time) and `output: {values, variance?}`.
- Per frame: extent, radius, `k`, minimum neighbors, model (`'spherical'`, `'exponential'` or `'gaussian'`,
  the `fitVariogramModel` convention) and nugget, sill and range. Pass `sill + nugget` as PyKrige's "sill"
  when comparing, since PyKrige's value is the total sill.
- `values` is a row-major raster and nodata is NaN: too few distinct neighbors, an ill-conditioned
  system, or an unreliable variance. `variance` is the kriging variance in squared value units, 0 at exact
  hits, NaN at nodata.
- Units and coordinates: by default positions, extent, radius and variogram range share one planar
  unit; distances are differences of offsets from the extent minimum (cross-validation: from the
  index domain minimum), so UTM-sized coordinates keep full f32 precision relative to the
  neighborhood. With `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
  'great-circle'}` positions are absolute lon/lat degrees, the index and extent stay in degrees,
  and the search radius, variogram range and all sample-sample and sample-cell distances are
  haversine meters (`sphereRadius`, else 6371008.8 m; kilometers with `units: 'kilometers'`).
  Neighbors are ordered by `(haversine term, row)`; exact hits and duplicate coalescing compare
  degree positions exactly; leave-one-out uses the same metric. The query box is conservative and
  does not wrap the antimeridian, and the ring scan's early stop is disabled (the whole box is
  scanned, so an unbounded radius is `O(samples)` per cell). f32 absolute lon/lat gives about 1 m
  accuracy at Earth scale. Other longitude/latitude metrics throw.
- Status: `invalidCount` counts cells that are NaN for lack of data or invalid settings (fewer than
  `minimumNeighborCount` distinct positions, no sample in the radius, outside the index domain).
  `uncertainCount` counts cells rejected by the numerical checks below. Both are computed over the
  raster of the last encoding, so a nonzero `uncertainCount` means the surface has holes that are
  not "no data" but "do not trust the solve here".
- Singularity is a relative test, not an absolute one: a pivot at or below
  `KRIGING_RELATIVE_PIVOT_TOLERANCE` (1e-6) of the system scale is zero, and a system whose smallest
  pivot is below `KRIGING_PIVOT_RATIO_TOLERANCE` (1e-4) of its largest is rejected as ill conditioned,
  since an f32 solve loses about `cond * 6e-8` of the weights' accuracy and the ratio tracks `1 / cond`.
  The usual cause is a Gaussian model with a near-zero nugget and closely spaced samples: add a nugget
  (0.001 of the sill moves the ratio from 1e-6 to 5e-5 for 12 clustered samples) or thin the samples. The
  constants were calibrated against emulated f32 solves compared with float64.
- Variance is `mu + sum(w * gamma)` in units of the total sill and is never negative for a valid
  model. A negative value above `-KRIGING_NEGATIVE_VARIANCE_TOLERANCE` (1e-4) is f32 rounding and is
  clamped to zero; a larger negative variance makes the cell NaN (value and variance) and counts it in
  `uncertainCount`. It is never silently clamped.
- Duplicate sample positions inside a neighborhood are coalesced into one kriging point whose value is
  the mean of their values (summed in `(d^2, row)` order, so the result is deterministic), instead of
  making the system singular. Duplicates still occupy neighbor slots, so a neighborhood of `k` rows can
  hold fewer than `k` points, and `minimumNeighborCount` counts distinct positions. An exact hit returns the
  mean of the samples at that position with variance 0.
- The nugget is treated as micro-scale variation: the predictor is exact at the data even with a
  positive nugget (zero variance at exact hits). Filtered kriging with a measurement-error nugget is not
  provided.
- Leave-one-out cross-validation (`crossValidationPrediction`, optional `crossValidationVariance`, one
  float32 row per sample) krigs each sample from its `k` nearest OTHER samples with the same variogram,
  radius, `k` and minimum-neighbor settings and the same solver code. Rows that are NaN, masked or
  unestimable are NaN. Another sample at exactly the same position is an exact hit (its mean, variance 0).
  Compare models with the standardized error `(value - prediction) / sqrt(variance)`, which should have
  mean near 0 and variance near 1, and the root mean squared error of `value - prediction`.
- A sample `mask` is folded into a transient value copy first (a one-pass kernel), and the per-cell status
  goes to a transient before the counters are accumulated, so every kernel stays within 8 storage buffers.
- Pinned to PyKrige 1.7.3 with 8 closest points (exponential, spherical and Gaussian within 0.2 on values
  near 70, and 1 percent on variance); the pivot-ratio, duplicate-coalescing, mask, leave-one-out and
  translation fixtures are float64 numpy ordinary kriging (Cressie 1993, eq. 3.2.12 to 3.2.16) that
  reproduces the PyKrige numbers to 5e-7. Global and universal kriging, anisotropy, octant search and
  filtered (measurement-error) kriging are not provided.

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
    output: {positions, dots: {ids, count, overflow, requiredCount}, categories, failedCount}
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
  `0xffffffff`. A dot whose `maximumAttempts` (default 32) candidates all miss keeps its slot with a NaN
  position and is counted in `failedCount`.
- **Rejection bias.** A candidate is accepted with probability polygon area / bounding-box area (times the
  mask weight), so a dot fails with probability `(1 - p)^maximumAttempts`. Thin, diagonal, concave or
  mask-starved polygons deterministically lose dots, which biases the map toward compact polygons. Watch
  `failedCount` (a sampling failure rather than an invalid input row, so it keeps its own name instead of
  `invalidCount`), and raise the compile-time `maximumAttempts` until it is zero.
- **Coordinates.** Dots are uniform in coordinate space, so `spatialContext` is planar/native only (omitted
  means planar/native), on `GPUDotDensity`, `GPURandomPointsInPolygon` and `GPURandomPointsOnLine`.
  Longitude/latitude throws: degrees are not an equal-area frame, so project to an equal-area planar frame
  first.
- Containment is an even-odd test over all rings of the feature (holes excluded); this equals
  multipolygon containment for valid geometry. Cost per candidate is linear in the feature's vertices.

**`GPURandomPointsOnLine`** places `counts` points per line uniformly in arc length (`GeoSeries.sample_points`
on lines). The Philox key is `(line, rank, 0, 0)` with key `(seed, 3)`, so the prefix of points is stable
as counts grow. Outputs are a bounded `positions` buffer and a compact `points` record (`ids` = line row,
`count`, `overflow`, `requiredCount`), plus optional `fractions`. Parameters reuse
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
- **Coordinate space.** `spatialContext` is checked at construction and exposed as a resolved
  field. `positions` are longitude/latitude degrees (default `{coordinateSpace:
  'longitude-latitude', metric: 'none'}`); a planar context throws, because Quadbin keying is
  defined on degrees. Pre-keyed `cells`, `GPUCellRollup` and `GPUCellPyramid` use a `discrete-grid`
  context (default `gridFamily` is `family`): a given `gridFamily` must equal `family`, and a given
  `gridResolution` (the finest resolution of the input keys) must be at least `resolution`. Nothing
  is measured, so only metric `'none'` is accepted.
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
  - `spatialContext` (optional) must be longitude/latitude without a metric, which is also the
    default; planar contexts throw. `GPUCellGeometry` takes cell ids, so it has no `spatialContext`
    prop; its decoded `centers` and `boundaries` are described by `outputSpatialContext`.
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

Both consume cell keys, not positions, so neither takes a `spatialContext`.

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
  - Both write a capacity-bounded `{cells, count, overflow, requiredCount?, droppedCount?}`.
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

`GPUCellGridPath` consumes cell keys, so it takes no `spatialContext`.

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
- **`GPUCellMeasures`** (h3 `cell_area`, `edge_length`, `is_pentagon`): `areas` (square meters),
  `perimeters`, `edgeLengths` (stride `maximumEdgeCount` 6 to 10, great-circle meters between
  consecutive `cellToBoundary` vertices), `edgeCounts` and `pentagons`. **Breaking:** outputs used to
  be kilometers; meters are now the default and kilometers need an explicit `units: 'kilometers'`
  (then areas are square kilometers). The sphere is the WGS84 authalic sphere
  (`GPU_EARTH_AUTHALIC_RADIUS_METERS`, replacing the removed `GPU_CELL_MEASURES_EARTH_RADIUS_KM`), and
  the instance's `measures` field is the machine-readable description (`lengthModel: 'great-circle'`,
  `areaModel: 'sphere-polygon'`, units). Cell identifiers carry no coordinates, so the class has no
  `spatialContext`. Single-face hexagons are exact from lattice
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
  even-odd over all rings. `spatialContext` defaults to, and must be, longitude/latitude without a
  metric (planar contexts throw).
- **Antimeridian and poles.** Polygons must not cross the antimeridian. A feature whose bounding
  box is wider than 180 degrees of longitude (every antimeridian-crossing ring, since such a ring has
  an edge with a longitude difference above 180) gets no cells and is counted in the optional
  `output.invalidCount`; split it at +-180 first. A feature reaching past the cover's latitude limit
  (89 degrees for H3, 85.051129 for Quadbin) loses its polar cells and is counted in the optional
  `output.uncertainCount`. Both scalars are cleared on every encoding.
- **Containment.**
  - `'center'`: the cell centre is inside the polygon.
  - `'full'`: the cell lies inside the polygon.
  - `'intersects'`: the cell interior meets the polygon.
  - Quadbin supports all three modes. A Quadbin tile is an axis-aligned lng/lat rectangle, so
    `full` and `intersects` are exact edge-versus-open-rectangle tests. H3 supports `'center'`
    only.
- **Candidates.** Quadbin tests the integer-exact tile range of each feature's bounding box. H3
  lays a lng/lat lattice over the bounding box, with spacing below half the minimum H3 inradius at
  that resolution (measured from h3-js; the minimum sits next to a pentagon). The longitude spacing
  is taken at the feature's most equatorward latitude (0 when it spans the equator), so tall
  features are oversampled toward the pole rather than missing cells nearer the equator. Each lattice point
  indexes its cell with `cellIndexH3FromLngLat`. The cell is emitted only from the lattice point
  nearest its centre, so every cell whose centre lies in the polygon is emitted exactly once with
  no sort.
- **Output.** `{featureIds, cells (uint32x2 little-endian), count, overflow,
  candidateOverflow, requiredCount?}`. `overflow` means only that accepted cells exceeded the final
  cell capacity; `candidateOverflow` means `candidateCapacity` truncated enumeration, in which case
  `requiredCount` is not an exact full-cover requirement. Order is by feature, then candidate order.
  The pipeline is count, scan, test, scan, write, with no atomic append, so it is deterministic.

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

It consumes cell keys, so it takes no `spatialContext`; the segments and rings it produces are
longitude/latitude degrees (the `GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT` convention).

The boundary edges of a set of H3 or Quadbin cells, the `cellsToMultiPolygon` outline. Set `rings` to chain
the edges into closed rings with shells and holes on the GPU ([`GPUSegmentRingAssembly`](#gpusegmentringassembly)).

```ts
graph.add(new GPUCellSetOutline({
  family: 'h3',
  cells, // uint32x2 little-endian keys, sorted ascending and distinct, e.g. a GPUCellAggregation table
  count: tableCount, // optional one-row valid row count
  groups, // optional uint32 label per row: also emit edges between different groups
  output: {rows, cells: segmentCells, edgeIndices, endpoints, groups: segmentGroups, count, overflow, requiredCount},
  rings: {normalizeWinding: true, output: ringOutput} // optional, see below
}));
```

- The output columns share one capacity: `rows` (the input row, also the compact ID column), `cells`,
  `edgeIndices`, `endpoints` (`float32x4`: lng0, lat0, lng1, lat1), optional `groups`, `count` (clamped),
  `overflow` and optional `requiredCount`.
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

`spatialContext` replaces the former `geographic` flag. It defaults to native planar coordinates (**behavior change:** `geographic` defaulted to true). A `longitude-latitude` context with metric `'none'` scales x differences by `cos(latitude)` (clamped at 0.01) when comparing turns at shared vertices; great-circle and other metrics are rejected because ring assembly measures nothing, and `vertexTolerance` stays in position units. Hole detection still uses the f32 shoelace sign of each ring.

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
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}, // cos(lat) in turn comparisons; omit for planar
  output: {
    ringOffsets, // uint32, ringCapacity + 1
    positions, // float32x2, vertex capacity
    ringAreas, ringIsHole, ringShells, ringGroups, // optional, ringCapacity rows
    segmentRings, segmentVertices, segmentFlags, // optional, one row per input segment
    polygons: {positions, ringOffsets, polygonOffsets, featureOffsets}, // optional
    count, overflow, requiredCount, openSegmentCount, touchingSegmentCount
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
  Vertices are hashed relative to the first segment's start vertex into unsigned cells of size
  `max(vertexTolerance, extent * 2^-22)` (with a small slack), where `extent` is the largest coordinate
  offset from that origin, measured on the GPU. Projected coordinates (UTM, Web Mercator) therefore never
  collide at any tolerance, and candidates are still compared by their true distance against
  `vertexTolerance`. `overflow` is also set when a valid segment has a non-finite coordinate.
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
  `GPUIsobandRings` uses it for the edges shared by two cells. `output.polygons.sourceIds` gives the
  group of each polygon's shell.
- **H3 pentagons.** Rings of three mid-latitude pentagons (resolutions 2, 3 and 5, alone, in disks, as holes
  and open rings) equal h3-js `cellsToMultiPolygon`. The two polar pentagons enclose a pole, so their rings
  have no planar lng/lat orientation.
- **Bounded output.** Rings are written whole, as a prefix: `count` is the number written, `overflow` is 1
  when ring or vertex capacity dropped rings, and `requiredCount` is the full ring count.
- **Polygon layout.** `output.polygons` regroups the rings as GeoArrow polygons (a shell, then its holes;
  the closing vertex dropped; holes without a shell dropped) with GPU-written, flat-padded offsets over
  fixed-length views. Every polygon is one feature, `featureOffsets` is an identity prefix padded with
  `polygonCount`, and optional `sourceIds` carries its shell group. It plugs into
  `GPUPointInPolygonJoin` (`polygonPositions`, `polygonOffsets`, `ringOffsets`, `featureOffsets`): the join's
  compile-time topology is the ring capacity and unused polygons and rings are empty.
- Deterministic (integer atomics only). The cost is two pointer-jumping phases of `ceil(log2(rows))`
  dispatches each (plus the second tracing pass), a sort and scans. Rings that cross the antimeridian keep
  the input longitudes. Turn comparison uses a monotone pseudo angle, not `atan2`. Ring areas are f32.

### `GPULineNoding`, `GPUPolygonize` and `GPUMakeValid`

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). The resolved context is passed on to the noding and ring-assembly stages.

These contributors form the bounded planar geometry-topology substrate:

- `GPULineNoding` splits arbitrary finite linework at crossings, touches, both ends of collinear
  overlaps, and repeated vertices. Each atomic `GPUNodedSegmentPort` row carries source feature,
  line/ring, original edge, and normalized start/end parameter provenance.
- `GPUPolygonize` expands each noded segment into two directed edges, assigns deterministic left-face
  ownership, omits the unbounded and zero-area faces, and emits canonical polygon features. It
  emits every bounded face (shapely `polygonize` semantics): the hole of a holed ring comes back as
  its own polygon. Its diagnostics classify source edges as closed-ring, open-chain, cut, dangle,
  and unused.
- `GPUMakeValid` applies the same pipeline independently to every input polygon feature, then keeps a
  face only when the feature's source rings cover it an odd number of times. The fill rule is source
  parity (even-odd), the GEOS `make_valid` linework method that shapely uses: a valid polygon with a
  hole returns the annulus only (no second polygon filling the hole), an island inside a hole is kept
  as its own polygon, and the doubly covered lens of mutually crossing rings is dropped. It repairs
  self-crossing and mutually crossing rings into bounded polygon faces; dimensional collapse remains
  available through diagnostics rather than being converted to mixed geometry. Parity is probed
  exactly at a symbolic point beside the first edge of each face shell (strictly inside the face, no
  rounded sample point), brute force over the noded segments, one thread per face. A face whose
  orientation cannot be certified is kept. `status.count` is the number of kept polygons.

All three require an explicit `GPUTopologyPrecisionPolicy`. The current policy uses the shared exact
orientation predicates with an uncertainty counter, float32 coordinate storage, and a caller-chosen
Chebyshev vertex tolerance. For `GPULineNoding`, `requiredCount` is exact when `candidateOverflow` is
zero. For `GPUPolygonize` and `GPUMakeValid`, `status.requiredCount` is the unclamped ring-assembly
requirement (every face ring, including holes), the ring capacity to allocate, not the number of
polygons written (`count`). `candidateOverflow` is the noding flag only: when the intersection
substrate is incomplete, both `candidateOverflow` and `overflow` are set and callers must grow
`intersectionCapacity` and rerun; ring or segment capacity loss sets `overflow` alone. No CPU readback
or compatibility adapter is involved.

### `GPUCellTableCompare`

It compares tables keyed by cell, so it takes no `spatialContext`.

`GPUCellTableCompare` outer-joins two sorted cell tables (the outputs of `GPUCellAggregation` or `GPUCellRollup`) into one union table for period-over-period or A-versus-B compare maps. Both tables must use the same grid family and resolution; the contributor cannot check that.

```ts
graph.add(new GPUCellTableCompare({
  before: lastMonth, after: thisMonth,
  measure: 'sum', sumScale: 65536,
  output: {cells, presence, before, after, delta, ratio, percentChange, zScore, count, overflow}
}));
```

- `output.cells` holds the union keys ascending (`uint32x2`, little-endian `(low, high)`); its length is the capacity. When the union is larger, the smallest keys are kept, `count` is clamped and `overflow` is 1. `overflow` also becomes 1 when either input table overflowed. Optional `requiredCount` receives the unclamped union size.
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
`pathOffsets` (deck.gl `PathLayer` `startIndices`), `count`, `overflow`, and optional `requiredCount`,
`pathCount`, `sourcePaths` (output path to input path), `sourceRows` (output vertex to input
segment row) and `measures` (distance along the source path). The vertex capacity is
`positions.length`; offsets are clamped to it, so a truncated result is still a valid layout.
Inputs are packed `positions` sorted by path with `pathOffsets` (`pathCount + 1` rows).

```ts
graph.add(new GPULineSegmentize({
  positions, pathOffsets,
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}, // omit for planar
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
  are kept. With a great-circle `spatialContext`, positions are longitude/latitude degrees: lengths
  are great-circle distances times the context's `sphereRadius` (default mean Earth radius,
  `GPU_EARTH_MEAN_RADIUS_METERS`; meters, or kilometers with `units: 'kilometers'`), interior points
  are slerped, and longitudes are unwrapped along each path so it stays continuous across the
  antimeridian. **Breaking:** the `coordinateSystem` and `radius` props of `GPULineSegmentize` and
  `GPULineChunk` (and `radius` of `GPUGreatCircleArcs`) are replaced by `spatialContext`; planar is
  the default. `GPUGreatCircleArcs` defaults to a great-circle longitude/latitude context and
  throws for planar ones; `GPULineSmooth` is planar only and throws for longitude/latitude.
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
  paths are copied as one piece. Both modes support a great-circle `spatialContext`.
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
  // planar (native units), great-circle (spherical), or ellipsoidal (geodesic edges by default)
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'},
  holeRule: 'winding', // or 'first-ring-exterior'
  output: {areas, lengths, centroids, bounds, vertexCounts},
  status: {invalidCount}, // geographic polygons with a pole or half-world edge, see below
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
- Great-circle (`metric: 'great-circle'`) areas are the Chamberlain-Duquette formula turf uses
  (shoelace in the Lambert cylindrical equal-area projection, exact for longitude/latitude boxes;
  long edges are straight in that projection, not geodesics); pass
  `sphereRadius: GPU_WGS84_SEMI_MAJOR_AXIS_METERS` for turf `area` parity. Ellipsoidal with
  `ellipsoidalEdgeModel: 'coordinate-linear'` maps latitude to authalic latitude (exact equal-area
  ellipsoid mapping) and measures edges with an f32 Vincenty inverse.
- **Measure model.** The instance's `measures` field (`GPUMeasureDescription`) says what the length
  and area outputs mean: `lengthModel` (`'planar'`, `'great-circle'`, `'ellipsoidal-geodesic'`),
  `areaModel` (`'planar'`, `'sphere-equal-area'`, `'wgs84-authalic'`, `'geodesic'`) and the length
  and area units (planar: native; geographic: meters and square meters, or kilometers on a sphere
  context with `units: 'kilometers'`).
- **Antimeridian and poles (geographic polygons).** Longitude steps are wrapped per edge, so an
  edge with a longitude difference above 180 degrees (Fiji's 179 to -179) takes the short way and
  the ring is measured correctly. A ring whose wrapped steps sum to +-360 degrees encloses a pole,
  and an edge whose wrapped step reaches 180 degrees (or whose raw step is a full 360) has no
  unique interior side. Features owning such a ring report NaN `areas`, `signedAreas` and
  `centroids` (lengths, bounds and vertex counts stay valid) and are counted in
  `status.invalidCount`, a one-row `uint32` view cleared every encoding. Split such polygons at the
  antimeridian, or project them to a polar frame first. The check covers the serial and cooperative
  kernels. Line features are unaffected.
- Groups: features are stably sorted by group ID and one workgroup per group reduces them with a
  fixed tree, so group sums are bitwise reproducible. Group centroids are area- (or length-)
  weighted means of feature centroids. **Behavior change:** in geographic contexts the mean is
  taken on 3D unit vectors (a spherical centroid), not on longitude and latitude numbers, so a
  group straddling the antimeridian lands on the antimeridian (for example 179 and -179 give 180)
  instead of at longitude 0. An antipodal group has no mean direction and reports NaN; a group
  containing an invalid feature reports NaN. Planar groups are unchanged.
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

### `GEODESIC_WGSL`, `createGPULocalFrame`, and `GPULocalFrameProjection`

Two ways to work on longitude/latitude data, used consistently across the package:

- **Measure on the sphere.** `GEODESIC_WGSL` is the public WGSL source of the spherical functions
  every great-circle path in this package uses; include it once in your own compute shader.
  Positions are `vec2<f32>(longitude, latitude)` degrees and distances are in the units of the
  `radius` argument (`GPU_EARTH_MEAN_RADIUS_METERS` for meters, the package default).
  `geodesicHaversineDistance(a, b, radius)` is the great-circle distance;
  `geodesicSegmentFraction(point, start, end)` and
  `geodesicSegmentDistance(point, start, end, fraction, radius)` give the point-to-segment distance
  for segments that are straight in longitude/latitude (the edge model of the containment tests and
  network edges; exact to first order for kilometer-length segments, densify longer ones). Also
  available: `geodesicHaversineTerm`, `geodesicCentralAngle`, `geodesicInitialBearingDegrees`,
  `geodesicDestination`, `geodesicInterpolate` and `geodesicWrapLongitudeDelta`.
- **Project first.** Planar operators throw on longitude/latitude contexts. `createGPULocalFrame({origin,
  kind})` returns a binary64 CPU frame in meters: `'azimuthal-equidistant'` (spherical, mean radius;
  distances and bearings from the origin exact) or `'lambert-equal-area'` (WGS84 ellipsoidal, areas
  exact), matching PROJ `aeqd +R=6371008.8` and `laea +ellps=WGS84` to a millimeter. It offers
  `project(lng, lat)`, `unproject(x, y)` and `projectArray(lngLats, stride)` for uploads, and a
  `spatialContext` that carries the frame identity: contexts of frames with different origins are
  incompatible, so data projected around different origins is not mixed. Float32 frame coordinates
  keep about 1e-7 of the frame extent (millimeters at 10 km) instead of the 0.4 to 1 m resolution of
  float32 degrees. `GPULocalFrameProjection` projects GPU-resident `float32x2` or binary64
  (`uint32x4`) longitude/latitude rows into the frame without a readback (compiled adaptive patches
  from `gpu-project`, default tolerance 1 cm, output in absolute frame meters).

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
- `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` on both classes
  (planar is the default): positions are longitude and latitude in degrees and segments are
  great-circle arcs. `distances`, `measures`, `signedOffsets` and the search `radius` are meters on the
  context's `sphereRadius` (default 6371008.8; `radius` already names the search radius).
  **Breaking:** the `coordinateSystem` and `sphereRadius` props are removed in favor of
  `spatialContext`; other contexts throw.
  The foot point is the closest point on the arc (geo `HaversineClosestPoint`), with a longitude that is
  continuous with the segment start, so it can exceed 180 across the antimeridian. `distances` is the
  cross-track distance and `measures` the along-track measure. `GPULineLocate` slerps along the arc
  (`Geod.fwd`, `npts`): `tangents` are `(east, north)` compass components, offsets move along the left
  compass bearing in meters and `angles` are counter-clockwise from east. Limits: the search is brute
  force, points times segments, with no BVH and no overflow (`candidateCapacity`, `leafCapacity` and
  `spatialSort` are ignored); arcs over about 179 degrees are unsupported. Against `pyproj.Geod(a=6371008.8,
  f=0)` the worst error was 1.5 m for distance, 0.8 m for measure and 1.7 m for the foot point.

### `GPULineSplit` and `GPULineMerge`

Both classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first.

`GPULineSplit` splits linestrings at every crossing, touch, shared vertex and overlap end (turf `lineSplit`
against the union of the lines; verified against Shapely `unary_union`). It reuses `GPUSegmentIntersection`
in self mode, groups the split events by segment with a stable `GPUSort`, removes duplicates and writes
GeoArrow-layout pieces. Props: `lines` (`{kind: 'lines', positions, lineOffsets}`), `intersectionCapacity`,
`pieces`, optional `spatialSort`, `leafCapacity` and `uncertainCount`. `pieces` is a
`GPUGeneratedGeometryPort`: `geometry` holds `{kind: 'lines', positions, lineOffsets}`, `sourceIds`
identifies the source line, and `status` holds `count`, `overflow`, optional `candidateOverflow` and
optional `requiredCount`.
`vertexCount` and `requiredVertexCount` report the flattened vertex cardinality. Pieces are ordered by
source line, then traversal order. Splits at line ends are ignored, and a line with fewer than two
vertices yields no piece. Piece or vertex truncation sets `status.overflow`; an insufficient
`intersectionCapacity` independently sets `status.candidateOverflow`.

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
  pieces: {
    geometry: {kind: 'lines', positions: piecePositions, lineOffsets: offsets},
    sourceIds: lineIds,
    status: {count, overflow},
    vertexCount
  }
}));
graph.add(new GPULineMerge({positions, lineOffsets, output: {chainOffsets, positions: merged, count}}));
```

`GPUNetworkNoding` in [GPU Network](/docs/api-reference/experimental/gpu-network) builds a routable
network from the pieces.

### `GPULineClipByPolygon` and `GPUSharedPaths`

Both classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first.

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

`spatialContext` defaults to native planar coordinates, with the tolerance in position units. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` (optional `sphereRadius`, default `GPU_EARTH_MEAN_RADIUS_METERS`) runs Douglas-Peucker (both metrics) with great-circle meters: `tolerance` and `importance` are meters, chords are straight in longitude/latitude, and the distance from a vertex to its chord is the haversine distance to the closest point of the chord (found in the local equirectangular frame of the vertex, exact to first order for chords of kilometers). The `'time-ratio'` expected position is interpolated linearly in longitude/latitude with the longitude difference wrapped. Great-circle distances use f32 transcendental functions, so the bit-exactness below holds for planar contexts only. `method: 'visvalingam'` is planar only and throws in a great-circle context (its tolerance is an area, its rounds mirror a bit-exact planar triangle area, and no spherical-excess variant exists); kilometers and other metrics throw. For Visvalingam, or to keep planar bit-exactness, project first (`createGPULocalFrame`, `GPULocalFrameProjection`).

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

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). The tolerance is in position units.

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
vertex), `overflow` and optional `requiredCount`. Polygon offsets are unchanged.

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
`polygons.sourceIds`. Vertices at nodes are kept, as GEOS does. This is not the excluded vector
overlay: a coverage has no crossings, so the result is a subset of the input edges and is bounded by the
input vertex count plus one closing vertex per ring, with ring and vertex capacities and an overflow flag.
The input must be a valid coverage (check it with [`GPUCoverageValidity`](#gpucoveragevalidity)).
Planar only: `spatialContext` defaults to native planar coordinates and `longitude-latitude` throws.
`snapTolerance` snaps vertices and the output uses the first vertex's coordinates. Dissolve edge sets,
area, parts and holes matched `gdf.dissolve(method='coverage')` exactly on 5 scenes and 18 labels. The
per-ring area walk is serial per ring. The directory is `polygon-coverage-topology/` because `.gitignore`
ignores `coverage-*/`.

### `GPUTrajectoryMetrics`

Per-track metrics for rows sorted by track and time and delimited by `trackOffsets`: path length,
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

`spatialContext` is optional. Omitted (or planar) means Euclidean distances in position units and headings
`atan2(dy, dx)`. With `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` (optional
`sphereRadius`, default `GPU_EARTH_MEAN_RADIUS_METERS`) the positions are `(longitude, latitude)` degrees,
step distances and `trackLengths` are great-circle meters, speeds and `stopSpeedThreshold` are meters per
timestamp unit, and `stepHeadings` are initial bearings in the planar heading convention, radians counterclockwise from
east in `(-pi, pi]` (`pi / 2 - bearing`, where `GPUGeodesicPairs` reports the bearing in degrees clockwise
from north), so headings mean the same in every context. Longitude/latitude without the `great-circle` metric, kilometer units and the ellipsoidal metric throw.
Stop `centroids` are arithmetic means of coordinates, so a dwell straddling the antimeridian or a pole has a
meaningless centroid. Positions are float32 degrees (about 1 m resolution at mid-latitudes); for
sub-meter work project to a local frame instead. The `spatialContext` prop used to be required and
planar-only.

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
  Zero-length segments give 0. Positions are planar unless `spatialContext` is great-circle.
- Great-circle (`spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`, optional
  `sphereRadius`, default `GPU_EARTH_MEAN_RADIUS_METERS`): positions are `(longitude, latitude)` degrees and
  each segment is interpolated by spherical linear interpolation instead of a linear blend of degrees. The
  heading is the segment's initial bearing in the planar convention (radians counterclockwise from east,
  `pi / 2 - bearing`, in `(-pi, pi]`), and `speeds` are
  great-circle meters per timestamp unit. Interpolated longitudes are continuous with the segment's start
  row, so a segment that crosses the antimeridian yields longitudes beyond ±180 (the end row is `b` shifted
  by a multiple of 360 degrees) rather than jumping; wrap downstream if needed. Fractions 0 and 1 return
  the start and end row exactly (up to that shift). Other contexts throw. Ellipsoidal interpolation is a
  non-goal.
- Time: comparisons are exact in both modes (Int64 words compare integer words, then the f32
  playhead fraction); differences are subtracted exactly and then rounded to f32.

`GPUTrajectoryResample` resamples every track to `sampleCount` samples, uniformly in time
(`spacing: 'time'`, default) or along the path (`'arc-length'`: planar length, or great-circle meters with a
great-circle `spatialContext`, which also slerps the samples as the playhead does), into a dense
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

`spatialContext` is planar only (the default). Segments and zone edges are straight lines in the coordinate
plane, which is not the geometry of degrees (a great-circle segment is not straight in longitude/latitude, and
edges cross the antimeridian), so any longitude-latitude context throws; project positions and zone edges first
with `createGPULocalFrame`.

- Inputs: `positions` (`float32x2`), `timestamps` (`float32` relative, or `uint32x2` Int64 words),
  `trackOffsets` (`trackCount + 1`), the zone boundary edges `edgeStarts`, `edgeEnds` (`float32x2`) and
  `edgeZones` (the zone index; the rings and holes of one zone share it), `zoneCount`, `candidateCapacity`
  and `maxEventsPerTrack`.
- Outputs: `events.output` (`GPUCompactOutput`; `ids` is the track index), optional `eventZones`,
  `eventTypes` (`GPU_ZONE_EVENT_TYPE.enter` 0, `.exit` 1), `eventTimes` (relative to the track's first
  timestamp, f32) and `eventRows` (the segment end row); optional dense `[trackCount * zoneCount]`
  `dwellTimes` and `visitCounts`; and `trackEventCounts` (unclamped).
- Events are sorted by `(track, time, edge row)`. Each segment queries a `GPUBVH` of edge bounds and
  runs an exact crossing test (`orientSign`, never f32 parameters); the time is
  `t[r - 1] + s * (t[r] - t[r - 1])`. Enter versus exit comes from parity, not ring orientation: the first
  sample of every track probes with a +x ray (even-odd), so a track that starts inside a zone works, and
  that first interval counts as a visit with dwell from `t = 0`. Open visits close at the track's last
  sample. Dwell and visits use every event, even when the event list is truncated.
- Boundary rule: every track sample `(x, y)` is read as `(x - e^2, y - e)` for an infinitesimal `e > 0`
  (the track moves, the polygon does not), for the initial ray and the segment crossings alike, so the
  parity of a track's crossings always equals the change of its inside flag. A track that only touches a
  polygon apex or an edge from outside has no event; one that runs through a vertex into the interior has
  one; a sample on an edge is just below it (inside when the interior lies below) and on a vertical edge
  just west of it. Event times along a collinear overlap use the f32 projection.
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
`overflow` and `requiredCount`) and the columns `zones`, `visits`, `dwellTimes`, `firstEnterTimes` and
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
- `eventOverflow` means there are more events than `events.output.ids.length`. `events.output.requiredCount`
  holds the size needed.

`events.output.overflow` remains the OR of the three flags.

### `GPUTrajectoryEncounters`

Pairs of tracks within `distance` in the same time bucket (MobilityDB `tdwithin` on resampled tracks).

- Inputs: dense `samples` (`[trackCount * bucketCount]` `float32x2`, row `track * bucketCount + bucket`,
  the `GPUTrajectoryResample.samples` layout), `trackCount`, `bucketCount`, optional `trackValid`,
  per-frame `distance` (one float row, clamped to `cellSize`), compile-time `cellSize` and
  `bounds [minX, minY, maxX, maxY]`, `hitCapacity` and optional `bucketTimes`.
- Outputs: `pairs.output` (`ids` is the lower track), required `pairs.candidateOverflow`, and
  optional `partners`, `firstBuckets`,
  `minimumDistances`, `bucketCounts` and `firstTimes` (needs `bucketTimes`). Pairs have `track < partner`,
  are deduplicated across buckets and sorted by `(track, partner)`; the sentinel is `0xffffffff` (IDs,
  buckets) and 0 after `count`. `pairs.output.overflow` reports only pair-capacity truncation;
  `pairs.candidateOverflow` reports hit-scratch or grid incompleteness.
- It uses a 3D `GPUGridIndex` (x, y and the bucket as an exact third axis) with a 3x3 scan per sample,
  three stable sorts of the hits and `GPUCompaction` for the runs.
- Great-circle: with `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`
  (optional `sphereRadius`) the samples are `(longitude, latitude)` degrees, `distance` and the reported
  `minimumDistances` are great-circle meters, and `cellSize` and `bounds` stay in degrees. The test is the
  great-circle distance of the two samples of one bucket, so there is no tangent-plane approximation. Pruning
  is conservative: two points within angle `theta` differ by at most `theta` in latitude and by at most
  `asin(sin(theta) / cos(L))` in longitude, where `L` is the largest `|latitude|` of `bounds`. The lattice
  therefore searches up to `maximumDistance = radius * asin(sin(cellSize) * cos(L))` (times `1 - 1e-6`),
  exposed as `encounters.maximumDistance`, and `distance` is clamped to it. Verified against pyproj: walking
  every azimuth at that distance from latitude ±L stays within one cell. Pick latitude bounds that exclude the
  polar caps (the reach shrinks to 0 as `L` approaches 90). Bounds with longitudes exactly `[-180, 180]` wrap
  the lattice across the antimeridian (sample longitudes are folded into that range first and `cellSize` must
  be below 120 degrees); other longitude bounds do not wrap, and `cellSize` must be below 90. Latitudes must
  lie in `(-90, 90)`. Planar contexts keep `maximumDistance = cellSize`. `addClockEncounters` forwards
  `spatialContext` to the resample and the encounters.
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

`spatialContext` defaults to native planar (Euclidean distances in the f32 input coordinates; the planar kernels
compile no geodesic code). `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` makes Hausdorff,
Frechet and `maxDistance` great-circle distances in meters on a sphere of `sphereRadius` (default
`GPU_EARTH_MEAN_RADIUS_METERS`); kilometers and other metrics throw. Vertices are `[longitude, latitude]` degrees
and segments are straight in longitude/latitude, as everywhere in this package. The vertex-to-segment distance is
the haversine distance to the closest point of the segment, with that point found in the local equirectangular
frame of the vertex (`geodesicSegmentFraction`, `geodesicSegmentDistance`): exact to first order for segments of
kilometers, so `densify` long segments. Densified points interpolate longitude and latitude linearly with the
longitude difference wrapped to +-180, so tracks may cross the antimeridian. Against numpy haversine with a
20000-part segment densification (cross-checked with `pyproj.Geod(a=6371008.8, f=0)`) the f32 results are
expected within 2e-3 relative; the spec covers a mid-latitude, a 66 degree and an antimeridian pair.

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
rings), `spatialContext` (omitted is planar; `{coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` gives meters on a sphere of `sphereRadius`, default the mean Earth radius; this replaces
the removed `coordinateSystem` and `radius` props, and other contexts throw), `joinSegments` (default 16),
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
of Shapely 2.1.2. Planar only (`spatialContext` defaults to planar; a longitude/latitude context throws,
so project first with `createGPULocalFrame` and `GPULocalFrameProjection`), rings are implicit-closed.

`GPUVertexSnap` is the vertex-to-vertex subset of `shapely.snap`: each vertex with a reference vertex
strictly closer than the per-frame `tolerance` moves to the nearest one (lowest row on ties) and the layout
is unchanged. Optional pairwise mode (`featureOffsets` and `referenceOffsets`) and `output.referenceRows`.
It is `O(V * R)`. Segment insertion and snapping onto segments, which GEOS also does, are out of scope.
It matched Shapely on 60 random lines.

### `GPUPolygonOverlay` and `GPUBufferSurface`

Both classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws. `GPUBufferSurface` distance is in position units, so project first (`createGPULocalFrame`, `GPULocalFrameProjection`) and buffer in the frame's meters; buffering degrees gives an ellipse on the ground.

`GPUPolygonOverlay` produces polygon geometry for `intersection`, `union`, `difference`,
`symmetric-difference`, and unary `dissolve`. It combines and nodes both operand boundaries, tests
the Boolean result on both sides of every atomic segment, keeps only edges where membership changes,
orients them with result interior on the left, and assembles those edges into shells and holes. The
result is compact polygon geometry rather than a face mask. `output.boundary` retains the operand and
source-feature provenance of every selected atomic edge.

Operand membership follows `fillRule`. The default `'even-odd'` reads the first ring of a polygon as
its shell and any other ring as a hole (crossing parity per ring). `'positive'`, `'negative'` and
`'nonzero'` instead sum the exact winding numbers of all rings of one polygon (counter-clockwise
+1, clockwise -1, same half-open crossing rule as the parity test) and keep the points where the
sum is greater than, less than, or different from zero, so regions a raw boundary covers twice stay
inside; a point is in the operand when it is inside any polygon. `output.status.requiredCount` is
the unclamped ring (shell plus hole) requirement of the assembly, for sizing the output ring
capacity; it is no longer a copy of `count`.

`GPUBufferSurface` is the topology-producing counterpart of `GPUOutlineGeometry` and
`GPUOffsetCurve`. Consecutive duplicate vertices are dropped from lines and rings first, and a
polygon ring's closing vertex is dropped when it repeats the first (GeoJSON and GeoArrow rings are
closed, the offset curve needs open rings). Polygon rings are normalized (counter-clockwise shells,
clockwise holes) before signed offsetting. Lines receive independent left and right offsets plus
`round`, `flat`, or `square` caps. Both paths finish with the same noding and dissolve as overlay,
using a winding fill rule (`'positive'` for polygons, `'negative'` for the clockwise line outline)
so regions the raw offset covers twice (hairpins closer than twice the distance, self-crossing
lines, slots narrower than twice the distance) stay inside, and the inverted loops of a negative
buffer through a thin neck are discarded, as in `shapely.buffer`. Checked on hairpin, figure-eight,
slot and thin-neck shapes with mitre joins. Known limitation: a polygon hole that a positive buffer
collapses (narrower than twice the distance) is not removed but survives as a smaller reflected hole;
a union of simple pieces would fix it and is deferred. Joins are `round`, `mitre`, or `bevel`; round resolution is controlled by `quadSegments`.
Positive polygon distances grow shells and shrink holes, negative distances do the reverse, and
non-positive line distances produce an empty surface.

Both contributors are planar float32 operations with explicit `intersections` and `nodedSegments`
workspace capacities plus caller-owned boundary, ring, and vertex capacities. `candidateOverflow`
means arrangement construction was incomplete, while `overflow` also includes compact boundary or
polygon capacity. Capacity, optional outputs, join/cap style, and topology are compile-time; distance
and mitre limit remain GPU parameters and can change per encoding.

### `GPUAffineTransform`, `GPUGeometryOrientation`, and `GPUGeometryCleanup`

All three classes are planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; `GPUAffineTransform` forwards the resolved context (and its frame) to its measures. `GPUGeometryOrientation` `'orient-polygons'` judges ring winding at the lexicographically smallest vertex with exact orientation, so sliver rings are not misjudged by f32 shoelace cancellation (a bow-tie ring is judged by that vertex, like `shapely.is_ccw`, not by its signed area).

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
and last kept, inclusive tolerance against the previous kept vertex). Line and polygon output is compacted
`positions` with republished `ringOffsets`, `count`, `overflow`, `requiredCount` and `collapsedRings` (polygon
rings with fewer than 3 surviving vertices are emptied, where GEOS raises). GEOS `set_precision(pointwise)`
does not drop repeats, so use `removeRepeatedPoints: false` for exact parity. With `geometryType: 'points'`,
each position row is an independent Point: the direct O(n) kernel applies only precision snapping, preserves
coincident rows, and publishes `positions`, `count`, `overflow` and optional `requiredCount` without ring
outputs. Line and polygon cleanup matched Shapely exactly on 5 parameter sets by 24 features; point tests
cover dynamic snapping, ties, duplicate preservation and capacity overflow. `normalize` is not provided.

### `GPUMinimumBounds`

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`.

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

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). Orientation signs are exact.

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
are separate because this contributor preserves polygon constraints and holes.

### `GPUDelaunayTessellation` and `GPUVoronoiDiagram`

`GPUDelaunayTessellation` builds an unconstrained planar point triangulation with a bounded
Bowyer-Watson kernel. Exact duplicate coordinates use first-row ownership, collinear input produces
no triangles, cocircular ties follow stable input order, and every output triangle is counter-clockwise.
`output.status` uses canonical `count`, exact `requiredCount`, `overflow`, and optional
`invalidCount`; `duplicateCount` reports discarded site rows. Use
`getDelaunayMaximumTriangleCount(pointCount)` for the general-position upper bound.

The starting super triangle is symbolic: its three vertices are points at infinity (towards +x, +y and
-x, with infinitesimally tilted directions and hierarchically growing radii), never finite coordinates.
Every orientation or incircle test that touches one reduces to exact comparisons of real coordinates:
a triangle with one infinite vertex has the half-plane left of its real edge as circumcircle (a site on
the edge's line is inside only when strictly between the endpoints), and a triangle with two or three
infinite vertices is decided by one exact orientation sign. Real hull triangles are therefore never
lost to a finite super vertex, so the result is the complete Delaunay triangulation of the convex hull,
`2n - 2 - h` triangles for `n` distinct sites and `h` hull-boundary points. An earlier finite
supertriangle (corners at 32 extents) dropped hull triangles, for example 2 of about 1200 for 600
uniform points and 51 of 185 for a 1e-2-thick band.

`GPUVoronoiDiagram` consumes those triangle rows and their GPU-resident count without readback.
Internal edges connect adjacent circumcenters; hull bisectors extend to a dynamic one-row
`clipBounds` view. Its compact `float32x4` segments retain the two source site IDs and canonical
bounded status. Both operations are deterministic for a fixed row order and use finite planar f32
coordinates; they do not claim constrained Delaunay, polygonal Voronoi cells, or concave construction.
Orientation and incircle tests are adaptive exact on the f32 coordinates. Both are planar only: a
longitude/latitude `spatialContext` throws at construction (project first with `createGPULocalFrame`
and `GPULocalFrameProjection`), because a lng/lat Delaunay is not a spherical one.

### `GPULabelPoint`

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`.

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
Coordinates are planar and degenerate features give NaN. A longitude/latitude `spatialContext` throws at
construction; project first with `createGPULocalFrame` and `GPULocalFrameProjection`. The resolved context,
including its `frame` identity, reaches the internal `GPUGeometryMeasures`. The gift-wrapping hull picks
each next vertex with the exact orientation predicate (`orientSign`) on the original positions.

### `GPULineDensity`

Line length per grid cell and its density (QGIS line density). Props: `positions`, `pathOffsets`,
`columns`, `rows`, `spatialContext` (planar/native, or longitude-latitude with the great-circle metric),
`maximumRecords`, `parameters`
(`getGPULineDensityParameterValues({minX, minY, cellWidth, cellHeight})`) and
`output: {lengths, densities?, overflow, totalRecords?}`. Segments are clipped to the grid and walked cell
by cell, and the lengths are summed per cell by a stable sort and a fixed-order reduction, so results are
bitwise reproducible. Spherical mode uses the great-circle meters of each clipped piece and exact
spherical cell areas for the densities. Exceeding `maximumRecords` sets `overflow` and under-counts the
last segments. A segment on a cell boundary goes to the upper or right cell, and closed rings are not
special (pass ring edges as lines). Length per polygon is
[`GPULineLengthPerPolygon`](#gpulinelengthperpolygon).

**Semantics.** `densities` is the line length inside the cell divided by the cell area, a box kernel
with the grid cell as its support. It is not the ArcGIS or QGIS circular-search-radius line density,
which spreads each line over a disk and is smooth; those need a smoothing step. For a smooth field,
rasterize `lengths` (or `densities`) and convolve it with `GPUConvolution` (with the same cell-index and
zero-boundary caveats as [`GPUPointDensity`](#gpupointdensity) smoothing), or sample line points with
`GPURandomPointsOnLine` and use [`GPUKernelDensity`](#gpukerneldensity).

### `GPULineLengthPerPolygon`

Total line length inside each polygon (QGIS "Sum line lengths", PostGIS
`sum(ST_Length(ST_Intersection(line, polygon)))`). Props: `positions`, `pathOffsets`, `polygons` (the layout of
`GPUSpatialPredicateJoin`), optional `pathWeights`, `spatialContext` (omitted: planar/native; or
longitude-latitude with the great-circle metric, replacing the former `coordinateSystem` and `radius`),
`maximumCandidatePairs`
(default `max(1024, 8 * positions.length)`), `maximumCrossings` (default 64) and
`output: {lengths, weightedLengths?, segmentCounts?, overflow}`. Every vertex row that is not the last of its path
is a segment. Segments meet polygons through `GPUSpatialJoinCandidates`, are cut at every ring-edge crossing, and
each piece whose midpoint is inside (even-odd over all rings, so holes and multipolygons work) is kept. Lengths
are summed per polygon by a stable sort and a fixed-order segmented sum, so results are bitwise reproducible
(segment counts use integer atomics). `weightedLengths` is `sum(length * pathWeight)` (weights are per path, not
per segment) and `segmentCounts` counts segments with positive length inside. A longitude-latitude context clips straight in
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
`output: {positions?, centers?, corners?, cornerIntersects?}`. Size cell outputs with
`getGPUGridCellCount` and `getGPUGridVerticesPerCell`; size the unique square or hex vertex lattice
with `getGPUGridCornerCount`. Hexagons are pointy-top by default (`cellWidth` is flat to flat), triangles come in
alternating strips with half-cell row shifts (not turf's exact layout), and the point grid sits at cell
centers. The cell counts are compile-time. The lattice is planar: `spatialContext` accepts only a planar
context (the default) and throws for longitude/latitude, so project first or use `GPUPointToCell`.

- `hexOrientation: 'flat'` (GeoPandas `make_grid(flat_topped=True)`): `cellWidth` is still the
  flat-to-flat distance (GeoPandas `cell_size`), columns step `1.5 * R`, rows step `cellWidth`, odd
  columns shift up half a cell, and the ring is counter-clockwise from angle 0. The lattice is anchored at
  `(minX - R / 2, minY)`, where GeoPandas starts half a cell off the offset. Matched GeoPandas to 1e-5 on
  16 cells.
- `extent` and `output.intersects` (GeoPandas `make_grid(intersect=True)`): a per-cell `uint32` flag
  against a polygon (even-odd rings, holes, multi-polygon) with exact orientation predicates. Boundary
  contact counts, and point grids test the center. Cost is cells times extent edges. It is a flag, not a
  compaction: compact it with `GPUCompaction` for the subset. Matched Shapely `intersects` on five grid
  types (240 cells).
- `output.corners` implements GeoPandas `make_grid(feature_type='corners')` for square and hex grids.
  It emits every shared and boundary vertex exactly once using an integer lattice, without sorting or
  floating-point hashing. `output.cornerIntersects` independently tests those points against `extent`,
  including polygon and hole boundaries. Triangle and point grids do not expose corner output.
- Cell polygons compute every vertex from the same integer-lattice expression as `output.corners`, so
  neighbouring cells share bit-identical vertices (no 1-ulp cracks or overlaps along seams).

### `GPUShapeGenerator`

Circles, sectors and ellipses around per-feature centers and radii (turf `circle`, `sector`, `ellipse`).
Props include `shape` (`GPUShapeType`), `centers`, `radii`, optional sector `bearings` and ellipse
`rotations`, `spatialContext` (planar by default, or `{coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` with spherical destination and radii in meters on the mean Earth radius; this replaces
the removed `coordinateSystem` prop and `GPU_SHAPE_GENERATOR_EARTH_RADIUS`), compile-time `maximumSegments` and per-frame `parameters`
(`getGPUShapeGeneratorParameterValues`). The segment count and the radius scale are per-frame, up to
`maximumSegments`; size outputs with `getGPUShapeVertexCount` and `getGPUShapeMinimumSegments`. The output
is packed rings plus GeoArrow-style `offsets` (feature `f` starts at `f * V`) and an optional
`vertexCount`. Circles and sectors match turf 7.4 exactly (pinned). Ellipses default to the original
equal-parameter sampling; `ellipseSpacing: 'arc-length'` builds a normalized 257-sample quarter-ellipse
table per feature and inverts it for approximately equal edge lengths. The table is built before rotation
and, for geodesic shapes, before spherical destination; it costs 1028 transient bytes per feature and one
additional compute pass. Coincident sector bearings keep their center spokes. Geodesic f32 coordinates
round to about 0.5 m.

### `GPUHilbertKeys`

`spatialContext` accepts planar coordinates or longitude/latitude with metric `'none'` (plain
coordinates for ordering; no antimeridian wrap). Metric contexts throw.

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

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`).

Clips lines or polygons to a rectangle (turf `bboxClip`, Shapely `clip_by_rect`, PostGIS `ST_ClipByBox2D`).
Props: `positions`, `geometryType` (`'lines'` or `'polygons'`), `pathOffsets`, `parameters`
(`getGPURectangleClipParameterValues({minX, minY, maxX, maxY})`) and `output: GPULinePathOutput`
(`positions`, `pathOffsets`, `count`, `overflow`, optional `requiredCount`, `pathCount` and, for lines,
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
BIC, optional per-row residual and fitted columns, the Jarque-Bera normality test, the
Koenker-Bassett and Breusch-Pagan heteroskedasticity tests, and collinearity and precision
diagnostics (Belsley condition number, correlation condition number, maximum VIF, precision
warning). `getGPUOrdinaryLeastSquaresModelResult(readback, {predictorNames})` turns the read-back
arrays into the shared `GPURegressionModelResult` (coefficient table with Student t p-values on
`n - p` degrees of freedom, `sigmaSquared = RSS / (n - p)` and `sigmaSquaredMaximumLikelihood = RSS / n`,
R2, log-likelihood, AIC and BIC equal to spreg, conditioning, and the tests `jarqueBera`,
`breuschPagan`, `koenkerBassett`).

```ts
const graph = new GPUCommandGraph(device);
graph.add(
  new GPUOrdinaryLeastSquares({
    predictors, // GraphDataView<'float32'>, row-major row * predictorCount + column
    response, // GraphDataView<'float32'>
    mask, // optional GraphDataView<'uint32'>; zero excludes the row
    parameters: ridgeBuffer.importToGraph(graph), // optional: [ridgeLambda]
    predictorCount: 3, // compile-time, 1 to 15 (16-wide Cholesky); the intercept is added internally
    output: {coefficients, standardErrors, tStatistics, summary, status, residuals, fitted}
  })
);
```

- Rows with any non-finite predictor or response value, or a zero mask, are excluded.
- `coefficients`, `standardErrors`, `tStatistics`: `predictorCount + 1` floats, intercept first.
- `summary` (22 floats, `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH`), indexed by
  `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*`: row count, R2, adjusted R2, sigma squared, log-likelihood,
  AIC, BIC, Jarque-Bera, its p-value, Koenker-Bassett, its p-value, RSS, TSS, skewness, kurtosis,
  ridge lambda, Breusch-Pagan, its p-value, `CONDITION_NUMBER`, `CORRELATION_CONDITION_NUMBER`,
  `MAXIMUM_VARIANCE_INFLATION_FACTOR`, `PRECISION_WARNING`. The slots 9 and 10 used to be called
  `BREUSCH_PAGAN`; they were always the Koenker-Bassett statistic, and `BREUSCH_PAGAN` now names the
  classic test in slots 16 and 17.
- The predictor count is capped at 15 (16 coefficients with the intercept) by the 16-wide
  function-local Cholesky factorization of the solve; a larger design needs a tiled solver.
- `status` (uint32): 0 ok, 1 singular or ill-conditioned (a predictor within `1 - 1e-6` of the span
  of the others, or constant), 2 too few rows (`n <= predictorCount + 1`). A failed fit leaves every
  statistic and per-row column NaN (the row count is still written).
- `residuals` and `fitted` are optional per-row columns (NaN for excluded rows).
- Conditioning never changes `status`: a warning is not a failure and the fit is still reported.
  `CONDITION_NUMBER` is the Belsley number (spreg `OLS.mulColli`): `sqrt(lambdaMax / lambdaMin)` of the
  raw `[1, X]` Gram matrix scaled to unit diagonal (rebuilt from the centered moments and means; values
  above about 30 mean harmful collinearity; ridge is not included). `CORRELATION_CONDITION_NUMBER` is
  `lambdaMax / lambdaMin` of the predictor correlation matrix the float32 Cholesky factors (with the
  ridge penalty), and `MAXIMUM_VARIANCE_INFLATION_FACTOR` the largest diagonal of its inverse. Float32
  coefficients lose about `correlationConditionNumber * 6e-8` relative accuracy, so
  `PRECISION_WARNING` is 1 when the correlation condition number exceeds 1e4
  (`GPU_REGRESSION_PRECISION_CONDITION_LIMIT`, about three of seven digits lost). Eigenvalues come from a
  cyclic Jacobi routine in float32, so the condition numbers are indicative for very ill-conditioned
  designs (the GPU specs allow 2e-2 to 1e-1 relative against the float64 oracle and spreg). They match spreg `mulColli` and `diagnostics.vif` (max
  over the predictor columns, computed on the centered design) in the float64 oracle.
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
`exp(-JB / 2)`. Both heteroskedasticity tests use the auxiliary regression of `e^2` on `[1, X]` with
`R2_aux`, and a chi-square p-value on `predictorCount` degrees of freedom computed by the regularized
upper incomplete gamma with a fixed iteration cap. Koenker-Bassett (studentized, valid without
normality) is `n R2_aux`; the classic Breusch-Pagan (assumes normal errors) is
`0.5 R2_aux (sum e^4 / s2^2 - n) = 0.5 R2_aux n (kurtosis - 1)` with `s2 = RSS / n`, which is spreg
`breusch_pagan` with `z = [1, X]`. Both match spreg 1.9.1 (`breusch_pagan`, `koenker_bassett` with
`z = reg.x`; spreg's default `z` is the squared design). With skewed errors they differ several
times over, and Jarque-Bera rejecting normality is the cue to read Koenker-Bassett instead.
`getChiSquareSurvival(value, degreesOfFreedom)` is the float64 CPU mirror, and
`getStudentTTwoSidedPValue` / `getStudentTSurvival` (float64 regularized incomplete beta, pinned to
`scipy.stats.t.sf`) back the model-result p-values.

### `GPUGeographicallyWeightedRegression`

Geographically weighted regression with AICc bandwidth selection (Fotheringham, Brunsdon and
Charlton 2002; mgwr, ArcGIS GWR). For every included row `i` it fits a weighted least squares
`y ~ 1 + x` over all included rows `j` with weights `K(d_ij / h)` and writes local coefficients,
local standard errors, t values and a significance mask, local R-squared, fitted values, residuals and
the hat-matrix diagonal.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'gwr-parameters',
  format: 'float32',
  length: getGPUGeographicallyWeightedRegressionParameterLength(16)
});
graph.add(
  new GPUGeographicallyWeightedRegression({
    positions, // GraphDataView<'float32x2'>, planar (or lon/lat degrees with spatialContext)
    predictors, // row-major row * predictorCount + column
    predictorCount: 2,
    response,
    parameters: parameterBuffer.importToGraph(graph),
    maximumBandwidthCount: 16,
    output: {coefficients, standardErrors, tValues, significant, selectedBandwidth, summary}
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
  the GPU and the outputs are written at the selected bandwidth. A ladder of one value is a fixed
  bandwidth. Candidates with a non-positive bandwidth, invalid `k` or `n - 2 - tr S <= 0` score NaN.
- Singular locations (changed): a location whose local system is singular (for example a rare dummy
  that is constant in its neighbourhood) makes only that row NaN (coefficients, standard errors,
  fitted value, residual, hat; `localStatus` 1). It no longer voids the whole candidate. A candidate's
  `n`, RSS, `tr S` and AICc are taken over its non-singular rows, and `bandwidthSingularRowCounts`
  reports the count per candidate. Selection: among the candidates with the fewest singular rows the
  lowest AICc wins (ties to the lowest index), so a bandwidth with singular rows is chosen only when
  every valid candidate has some. This is a deliberate deviation from mgwr, which raises on a singular
  local system.
- Refinement (new, default on; `refinement: false` keeps the fixed ladder, a number from 1 to 4 sets
  the pass count, default 2): after the caller ladder, a GPU kernel writes a new ladder that zooms
  around the best candidate between its nearest evaluated neighbours (it never extrapolates beyond the
  caller's range), the same kernels evaluate it, and the overall best wins. Adaptive ladders hold
  integers: when the integers inside the bracket number at most `maximumBandwidthCount` all of them
  are tried, so the result is the exact integer optimum of the bracket; a wider bracket is sampled at
  equal spacing and the next pass narrows it. Fixed ladders are `maximumBandwidthCount` equally spaced
  distances per pass. Each pass costs up to one more ladder evaluation, so the default is about three
  times the work of the bare ladder; shorten the ladder rather than turning refinement off when cost
  matters. `selectedBandwidth` is `[candidate index, value]`; indices from `maximumBandwidthCount` up
  are refined candidates, and `candidateTable` lists every candidate of every pass. On the
  144-row test scene (adaptive bisquare) mgwr's golden section with its default bounds `[40 + 2 p, n]`
  stops at `k = 47`, while its exhaustive integer AICc optimum, which the refinement finds from a
  five-value ladder, is `k = 26` (and mgwr's golden section with bounds widened to 6..144 agrees).
  The AICc curve in `k` is not unimodal there (local minimum near 30), which is why a bracket that
  fits the ladder is enumerated rather than searched.
- Local inference (new): `standardErrors`, `tValues` and `significant` (all laid out like
  `coefficients`) are mgwr `bse`, `tvalues` and `filter_tvals != 0`. `bse = sqrt(diag(C C') sigma2)`
  with `C C' = (X'WX)^-1 X'W^2 X (X'WX)^-1` (one more weighted Gram matrix in the same neighbourhood
  pass) and `sigma2 = RSS / (n - tr S)` (mgwr `sigma2_v1`). The summary carries `ENP = tr S`,
  `SIGMA_SQUARED`, `LOG_LIKELIHOOD`, the Da Silva-Fotheringham adjusted alpha `alpha * k / tr S`
  (`ADJUSTED_ALPHA`, `k` design columns, `alpha` 0.05 unless the per-frame `significanceLevel` says
  otherwise) and the critical value `t.ppf(1 - adjusted / 2, n - 1)` (`CRITICAL_T`). The Student t
  quantile is inverted on the GPU (continued-fraction incomplete beta with a bisection; the float64
  CPU mirror `getStudentTCriticalValue` is pinned to `scipy.stats.t.ppf` to 1e-9 and the GPU summary is
  checked against mgwr), because the mask needs it there.
  `significant` is 1 where `|t| > CRITICAL_T`. The bandwidth is selected on the observed data and the
  inference is conditional on it, as in mgwr. Pinned to mgwr 2.2.1 (fixed and adaptive, planar and
  spherical); adaptive fits agree to about 5e-4 because mgwr scales the k-th neighbour distance by
  1.0000001 and this contributor by 1.00001.
- `getGPUGeographicallyWeightedRegressionModelResult({summary, selectedBandwidth})` returns the shared
  `GPURegressionModelResult` (`kind: 'geographically-weighted-regression'`, empty `coefficients`,
  `parameterCount = tr S`, `rowCount = n`, `sigmaSquared = RSS / (n - tr S)`, `logLikelihood`,
  `akaikeInformationCriterion` `= -2 llf + 2 (tr S + 1)` and `correctedAkaikeInformationCriterion`
  (mgwr `llf`, `aic`, `aicc`), `rSquared`) plus `effectiveParameterCount`, `singularRowCount`,
  `includedRowCount`, `adjustedAlpha`, `criticalT` and `selectedBandwidth`.
- `spatialContext` (new): omitted is planar. `{coordinateSpace: 'longitude-latitude', metric:
  'great-circle', units: 'meters'}` makes `positions` `(longitude, latitude)` degrees and every
  distance (kernels, fixed bandwidths, the adaptive k-th neighbour) a haversine distance on a sphere of
  `sphereRadius` (default the mean Earth radius 6,371,008.8 m); bandwidths are then metres. Rows with a
  latitude outside [-90, 90] are excluded. The grid index is planar-only and is not built for a
  great-circle context, so every location scans every included row and the 65,536-row scan limit
  applies; the cell walk is not made exact for longitude wrap-around. mgwr `spherical=True` uses a
  6371.0 km radius and kilometres, so pass `sphereRadius: 6371000` and metres to match it.
- Predictors: 1 to 11 (12 design columns; it was 7). No kernel binds more than eight storage buffers,
  because the per-row results (including standard errors) share one row-state buffer; the limit is
  per-thread private memory (`2 P^2 + 6 P` floats in the fit). The 11-predictor shader has not been run
  on every adapter: lower `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT` if one
  rejects it.
- Outputs: `coefficients` (`row * (predictorCount + 1) + column`, column 0 intercept), optional
  `standardErrors`, `tValues`, `significant`, `localR2`, `fitted`, `residuals`, `hatDiagonal`,
  `localStatus` (0 ok, 1 singular, 2 excluded), `bandwidthScores` (AICc of the caller ladder),
  `bandwidthSingularRowCounts`, `candidateTable`, `selectedBandwidth` (`[index, value]`),
  `localConditionNumber` and `summary` (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY`: RSS, `tr S`,
  AICc, global R-squared, n, has-valid-candidate flag, singular rows, sigma2, adjusted alpha, critical
  t, log-likelihood, included rows; 12 floats, it was 6).
- `localConditionNumber` is the mgwr `local_collinearity` local condition number: the singular-value
  ratio of the kernel-weighted, column-normalized design (above 30 flags collinearity). It is pinned to
  mgwr 2.2.1 (bisquare and gaussian, fixed and adaptive) and is f32-accurate. Its eigenvalues come
  from the shared `getSymmetricEigenvalueRangeWGSL` Jacobi helper of the regression solvers.
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
observations among the locations with the Philox-keyed swap-or-not shuffle of `GPUGlobalPermutationTest`
(the earlier Feistel bijection was non-uniform on small row counts), refits
at the selected bandwidth and compares the standard deviation of each local coefficient with the observed
one. The pseudo p-value is `(g + 1) / (P + 1)`. Outputs are `table` (six floats per coefficient), `summary`
(`GPU_GWR_NONSTATIONARITY_SUMMARY`, `GPU_GWR_NONSTATIONARITY_TABLE`, `GPU_GWR_NONSTATIONARITY_TABLE_STRIDE`)
and an optional `standardDeviations` (the reference distribution). The per-permutation standard deviations
match mgwr 2.2.1 to 3e-15 (CPU oracle) and 2e-3 (GPU). The bandwidth stays at the value the regression
selected (mgwr re-selects it for every permutation), and that value was selected on the observed data, so
the test is conditional on it, as mgwr's is. `selectedBandwidth` is read from slot 1 (the selected value),
so it also works when the regression refined its bandwidth beyond the caller ladder. It is planar-only
(a great-circle `spatialContext` throws, with a hint to project first) and takes 1 to 7 predictors even
though the regression accepts 11. The cost is `O(P * n^2 * p^2)` with a full scan (no grid index).

### `GPUSpatialRegressionDiagnostics`

Spatial diagnostics for an OLS fit (spreg `LMtests` and `MoranRes`): LM-lag, LM-error, robust LM-lag,
robust LM-error, LM-SARMA, and Moran's I of the residuals (mean, variance, z, two-sided normal p).

- Props: `weights` (a square `GPUSpatialWeights`; any sparsity pattern is accepted, directed kNN included,
  and row-standardized weights work; `W'` is formed with `GPUSpatialWeightsTranspose`),
  `predictors` (row-major `n x k`, no intercept), `response`, `residuals` (OLS residuals with intercept,
  for example `GPUOrdinaryLeastSquares` `output.residuals`), `predictorCount` (1 to 15), optional `mask`,
  `restandardizeRows`, `tileRowCount` and `output: {tests, summary, status}`.
- Masked and missing fits: a row is analyzed when its `mask` entry is not zero and its response, residual
  and predictors are finite; pass the OLS `mask` and the NaN residuals of excluded rows do the rest. Excluded
  rows are removed from the whole analysis: `n`, `S0`, `T`, every Gram matrix and every `W`-product run over
  the fitted rows, and links from fitted rows to excluded rows are dropped. That is `libpysal.weights.w_subset`
  on the fitted ids. With `restandardizeRows: true` each fitted row's remaining weights are rescaled to sum
  to 1 (`w_subset` then `transform = 'r'`; use it after masking row-standardized weights); the default
  keeps the surviving weights as given. Rows left with no fitted neighbor keep an all-zero row. Results
  match spreg 1.9.1 `OLS(..., spat_diag=True, moran=True)` on the fitted subset in both modes.
- `tests` (18 f32): six rows of `[statistic, df, p]`: LM_LAG, LM_ERROR, ROBUST_LM_LAG, ROBUST_LM_ERROR
  (df 1), LM_SARMA (df 2) and MORAN_RESIDUALS (the statistic is z, the df slot is 0, p is two-sided). The
  row index constants are `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_*`.
- `summary` (13 f32): `n` (analyzed rows), `sigma2 = e'e / n`, `T = tr(W'W + WW)`, `J`, `S0`, Moran `I`,
  `E[I]`, `Var[I]`, `z`, `p`, `e'We`, `e'Wy` and `EXCLUDED_ROW_COUNT` (`..._SUMMARY_*`). The excluded
  count is a summary slot, not a status port, because `status` is this operator's single result code; it is
  also written when the status is not 0. Before this change a masked fit reported status 4 for the whole
  result.
- `status`: 0 ok, 1 singular, 2 too few analyzed rows (`n <= predictorCount + 1`, also a failed OLS fit),
  3 degenerate weights, 4 non-finite moments or exactly zero residuals. Every output is NaN unless the
  status is 0. Conditioning slots are not reported here (see the OLS fit).

Sums are fixed-order tile sums with a Kahan merge on centered columns, and the solve is a unit-diagonal
scaled Cholesky, so there are no atomics and no readback. Moran's variance and `J` are f32 with
cancellation (about 1e-2 relative on the variance and z). The statistics match spreg 1.9.1 (`OLS` with
`spat_diag` and `moran`) on a lattice, a symmetrized kNN scene and two directed row-standardized kNN scenes
(`k = 3`, 60 and 90 rows, asymmetric patterns) to 2e-3 relative. Earlier versions accumulated `W'Z` over each
row's own neighbors only, which biased the Moran variance for asymmetric patterns; that is fixed.

```ts
graph.add(new GPUOrdinaryLeastSquares({predictors, response, predictorCount: 2, output: {..., residuals}}));
graph.add(new GPUSpatialRegressionDiagnostics({
  weights, predictors, response, residuals, mask, restandardizeRows: true, predictorCount: 2,
  output: {tests, summary, status}
}));
```

### `GPUSpatialTwoStageLeastSquares`

The spatial lag model `y = rho W y + X b + e` by two-stage least squares (spreg `GM_Lag` with `w_lags = 1`),
with instruments `[1, X, WX]`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 8 with
`instrumentOrder: 1`, 1 to 5 with `instrumentOrder: 2`: the constant is centered out, so the solve factors `(order + 1) k`
instrument columns in a 16-wide Cholesky; the order-2 cap was previously 8 in the check but failed when building the
graph), optional `tileRowCount`, `instrumentOrder` and `output: {table, summary, status, residuals?}`. No kernel
binds more than 8 storage buffers.
`instrumentOrder: 2` (compile-time, default 1) adds `W^2 X` (spreg `w_lags=2`, instruments `[1, X, WX,
W^2 X]`), evaluated per row from the two-hop neighborhood with no extra buffers; it is pinned to spreg
1.9.1 `GM_Lag(w_lags=2)` on three scenes.

- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]` in spreg order: the
  intercept, then `X`, then `rho` last.
- `summary` (16, `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_*`): `n`, `sigma2 = u'u / n` (spreg `sig2n`, the `GM_Lag`
  default), `u'u`, the pseudo R2 (squared correlation of `y` and `y - u`), Moran's `I` of `u`, the
  Anselin-Kelejian statistic and its p-value (slots 0 to 6). The statistic follows spreg `akTest(case='gen')`; any
  sparsity pattern is accepted (`W'u` comes from `GPUSpatialWeightsTranspose`), and it matches spreg 1.9.1 on
  directed kNN scenes. Diagnostics (slots 7 to 15; none changes `status`):
  - Conditioning of the second stage: the Belsley condition number of the raw `[1, PZ]` Gram matrix (`Z = [X, Wy]`,
    `P` the projection on the instruments), the correlation condition number of `Z'PZ` and its largest variance
    inflation factor, the correlation condition number of the instrument system `H'H` (the instrument-projection
    system), and a precision warning (1.0) when either correlation condition number exceeds 1e4 (float32
    normal equations may have lost three of seven digits). Computed by the Jacobi eigenvalue routine of
    `spatial-regression-solve`.
  - First-stage strength: the classical joint F statistic and the partial R2 of the excluded instruments (`WX`, and
    `W^2 X` for order 2) in the regression of `Wy` on `[1, X, WX, ...]`; `F = ((ESS_full - ESS_restricted) / (m - k)) /
    (RSS_full / (n - m - 1))`, partial R2 `= (ESS_full - ESS_restricted) / (TSS - ESS_restricted)`. The usual
    weak-instrument rule of thumb is `F < 10`. Both are the non-robust (homoskedastic) forms and are pinned to a numpy
    computation of the same definitions.
  - Admissibility of `rho`: `||W||_inf` (largest absolute row sum) and a flag that is 1.0 when `|rho| ||W||_inf >= 1`,
    that is when `I - rho W` is not guaranteed invertible and the reduced form and impacts are unreliable
    (`(-1, 1)` for a row-standardized `W`; sufficient, not necessary, for other `W`). It is a flag, not a failure:
    2SLS is unconstrained, the estimate is still reported, and the reader decides what a boundary or explosive `rho`
    means for the model.
- `status`: 0 ok, 1 singular or ill-conditioned (including non-finite inputs), 2 too few rows, 3 non-finite
  sums.
- `residuals`: the structural residuals `u = y - Z delta`, which can be passed to later diagnostics.

The coefficients, standard errors, `sigma2`, pseudo R2 and Anselin-Kelejian statistic match spreg on a
144-row scene within 5e-3 (coefficients) to 2e-2 (z, AK). The standard errors are the classical
`sigma2 (Z'PZ)^-1`: heteroskedasticity-robust and HAC standard errors are not provided (spreg `robust`),
nor are `GM_Combo` and maximum-likelihood models. For the spatial error model see
[`GPUSpatialErrorGM`](#gpuspatialerrorgm).

`getGPUSpatialTwoStageLeastSquaresModelResult(readback, {predictorNames?})` turns the read-back `table`, `summary`
and `status` into the shared `GPURegressionModelResult`: rows `CONSTANT`, the predictors and `W_y` (role
`'spatial-lag'`), z statistics with float64 normal p-values, `sigmaSquaredMaximumLikelihood = u'u / n` (the scale
of the standard errors, `standardErrorSigmaSquared: 'maximum-likelihood'`), `pseudoRSquared`, no likelihood,
`tests.anselinKelejian` and conditioning (the correlation condition number is the larger of the second-stage
and instrument values, so `illConditioned` equals the precision warning).
`getGPUSpatialTwoStageLeastSquaresDiagnostics(summary)` returns the first-stage F, partial R2, `||W||_inf` and the
`rho` admissibility flag.

### `GPUSpatialWeightsTraces`

Powers of a spatial weights matrix: `powerTraces[q-1] = tr((W/s)^q)` and `powerSums[q-1] = 1' (W/s)^q 1` with `s = ||W||_inf` (largest absolute row sum, written to `output.infinityNorm`; series then run in `rho * s`, which keeps the float32 powers bounded for raw weights) for
`q = 1..Q`. These are the inputs of the spatial-lag log-determinant `ln|I - rho W| = -sum_q rho^q tr(W^q) / q`
and of the impact multipliers (see `GPUSpatialLagImpacts`). Props: `weights` (any square CSR: asymmetric,
not standardized, self-loops allowed), `maximumPower` (`Q`, compile-time, 1 to 256), optional `mode`,
`probeCount`, `seed`, `includeGramTraces`, `tileRowCount`, `probeBatchSize` and
`output: {powerTraces, powerSums, gramTraces?, approximation?, infinityNorm?}` (float32 views of at least `Q`, `Q`, 4 values
and a uint32 scalar).

- `mode: 'exact'` runs one unit-vector chain `v_q = W v_{q-1}` per row and sums the diagonal entries
  `v_q[i]`: exact up to float32 rounding. It is the default for at most 4096 rows
  (`GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT`): the work is `rows * Q * nnz`, which at 4096 rows,
  `Q = 256` and 8 neighbors is about 3e10 multiply-adds in about 520 dispatches, while 16384 rows (the limit
  of an explicit `'exact'`) already costs 8200 dispatches and 5e11 multiply-adds.
- `mode: 'stochastic'` is Hutchinson estimation with `probeCount` (default 64) Rademacher probes from Philox
  (counter `(probe, row / 4, 0, 'PROB')`, key `(seed, 'TRAC')`, so a seeded run is reproducible and
  `getSpatialWeightsTracesOnCPU` replays the same probes). The standard error of `tr(A)` is at most
  `sqrt(2 / probeCount) ||A||_F`. `approximation` is 1 (it is 0 in exact mode).
- `powerSums` is always exact: one chain from the ones vector.
- `gramTraces` (when `includeGramTraces`, 4 values) are `[tr(W'W), tr(W'W W), tr((W'W)^2), sum_i c_i^2]` with
  `c_i = sum_l w_li^2`: the moments of the Kelejian-Prucha and Anselin-Kelejian variance formulas. `tr(W'W)` and
  `sum c_i^2` are exact sums (the column sums use `GPUSpatialWeightsTranspose`); the other two are
  `E[(Wz)'(WWz)]` and `E[||W'(Wz)||^2]` over the same probes, exact in exact mode.

Probes run in batches whose two ping-pong vectors stay within 64 MiB. Every dot product is a fixed-order tile
sum with a Kahan merge, so there are no float atomics and results are bit-identical across batch sizes and
runs. A `W` with spectral radius above 1 grows as `radius^q` and can overflow float32 for large `Q`.

CPU float64 mirrors for specs and later consumers: `getSpatialWeightsTracesOnCPU(weights, Q, options?)`,
`getSpatialLogDeterminantFromTraces(rho, powerTraces)`, `getSpatialMultiplierTraceFromTraces(rho, powerTraces,
n)` (`= tr((I - rho W)^-1) = n + sum_q rho^q tr(W^q)`) and the truncation bounds
`getSpatialLogDeterminantTruncationBound` and `getSpatialMultiplierTruncationBound`. WGSL generators
`getSpatialLogDeterminantWGSL(Q, bindingName)`, `getSpatialMultiplierTraceWGSL(Q, bindingName, n)` and
`getSpatialMultiplierTotalWGSL(Q, bindingName, n)` define `getSpatialLogDeterminant(rho)` and friends by
Horner's rule over an f32 storage binding. The series require `|rho| * spectralRadius(W) < 1` (`|rho| < 1` for
a row-standardized `W`) and are truncated: the log-determinant error is below
`n |rho|^(Q+1) / ((Q+1)(1 - |rho|))` and the multiplier error below `n |rho|^(Q+1) / (1 - |rho|)`, so choose `Q`
from `|rho|^Q` (`Q = 30` is accurate to 1e-9 at `|rho| = 0.5` but needs about 300 terms at `|rho| = 0.9`).

The traces match `np.trace(np.linalg.matrix_power(W, q))` for `q = 1..12` on a row-standardized lattice, a
binary lattice and two directed kNN scenes (numpy 2.5.4), and the log-determinant series matches
`np.linalg.slogdet` within the stated bound.

```ts
graph.add(new GPUSpatialWeightsTraces({
  weights, maximumPower: 30, output: {powerTraces, powerSums, infinityNorm}   // exact for <= 4096 rows
}));
```

### `GPUSpatialLagImpacts`

Average direct, indirect and total impacts of the predictors of a spatial lag model (LeSage and Pace 2009):
`direct = b_k tr((I - rho W)^-1) / n`, `total = b_k 1'(I - rho W)^-1 1 / n`, `indirect = total - direct`, from the
truncated series `n + sum_{q<=Q} rho^q tr(W^q)` and `n + sum_{q<=Q} rho^q 1'W^q 1` evaluated in one kernel. Props:
`coefficients` and `coefficientStride` (the `table` of `GPUSpatialTwoStageLeastSquares` with stride 4, or any
fit with a coefficient per row), `rhoIndex` (`predictorCount + 1` for the 2SLS table), optional
`firstPredictorIndex` (default 1, so the intercept is excluded), `predictorCount`, `rowCount`, `maximumPower`,
`traces: {powerTraces, powerSums, infinityNorm}` (the outputs of `GPUSpatialWeightsTraces`), optional `status` and `rhoBound`
(default 1), and `output: {impacts}`: `predictorCount * 3` f32 values `[direct, indirect, total]` per
predictor.

Impacts are NaN when the fit `status` is nonzero, `rho` is not finite or `|rho| >= rhoBound`. For a `W` that
is not row-standardized set `rhoBound = 1 / spectralRadius`, so the series cannot diverge silently.

The direct impact matches spreg 1.9.1 `GM_Lag(..., spat_impacts='full')`, and so does the total for a
row-standardized `W` (spreg uses `1 / (1 - rho)`, which is exact only there; this node uses the exact
`1'W^q1` sums, so it is also right for other weights). spreg's `'simple'` method fixes the direct multiplier at 1
and is not reproduced; its `'power'` method stops after the first term below tolerance, which is wrong on
bipartite lattices where odd traces vanish. Standard errors of the impacts (simulation of `(rho, b)` from the
coefficient covariance) are not provided: the 2SLS contributor does not output the covariance matrix.

```ts
graph.add(new GPUSpatialTwoStageLeastSquares({..., output: {table, summary, status}}));
graph.add(new GPUSpatialWeightsTraces({weights, maximumPower: 30, output: {powerTraces, powerSums, infinityNorm}}));
graph.add(new GPUSpatialLagImpacts({
  coefficients: table, coefficientStride: 4, rhoIndex: 3, predictorCount: 2,
  rowCount, maximumPower: 30, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}
}));
```

### `GPUSpatialErrorGM`

The spatial error model `y = X b + u`, `u = lambda W u + e` by generalized moments. Props: `weights` (any square
CSR; no symmetry or standardization needed), `predictors`, `response`, `predictorCount` (1 to 15, the OLS limit and
the 16-wide Cholesky; no kernel binds more than 8 storage buffers, so the previous cap of 8 was not needed),
`estimator`, optional `tileRowCount`, `traceMode`, `traceProbeCount`, `traceSeed` and
`output: {table, summary, status, residuals?}`.

**Default change.** `estimator` is `'homoskedastic'` by default (spreg 1.9.1 `GM_Error_Hom`, `A1 = 'hom_sc'`,
one iteration: Drukker, Egger and Prucha 2013, Anselin 2011), which gives `lambda` a standard error, z and p-value.
The previous behavior is `estimator: 'kelejian-prucha'` (spreg `GM_Error`, Kelejian and Prucha 1998 and 1999), whose
`lambda` row keeps NaN for se, z and p; in both cases the standard errors of the betas condition on the
estimated `lambda`.

- Steps (both): OLS of `y` on `[1, X]` gives `u`; moments of `u` give `lambda`, minimized globally on a 1024-interval
  grid with bisection of every bracketed minimum (spreg runs a local L-BFGS-B from zero; the two agree whenever the
  objective has a single basin); OLS of `y - lambda W y` on the filtered `[1, X]` (the constant is filtered too, as
  `get_spFilter` does) gives the coefficients.
- `lambda` bound: `0.99 / ||W||_inf` with `||W||_inf` the largest absolute row sum, computed on the GPU. It equals
  spreg's fixed 0.99 for a row-standardized `W` and keeps `I - lambda W` invertible for any `W`; a fixed 0.99
  admitted explosive `lambda` for raw weights. A solution on the bound is flagged
  (`..._SUMMARY_LAMBDA_AT_BOUND`) and `..._SUMMARY_LAMBDA_BOUND` reports the bound.
- `homoskedastic` replicates spreg step for step: (a) `lambda1` minimizes the unweighted
  `|G [l, l^2]' - g|^2` of the moments `u'A1u`, `u'A2u` with `A1 = (W'W - (t/n) I) / (1 + (t/n)^2)`, `t = tr(W'W)`,
  `A2 = (W + W') / 2`; (b) the FGLS betas at `lambda1` are the reported betas and give `u1 = y - X b1`; (c) the
  moments are recomputed on `u1` and `psi`, the variance of the moments (from `sigma2`, `mu3`, `mu4` of
  `u1 - lambda1 W u1` and the traces `tr W`, `tr W^2`, `tr W'W`, `tr W'WW`, `tr (W'W)^2`, `sum_i c_i^2` of
  [`GPUSpatialWeightsTraces`](#gpuspatialweightstraces)), weights the second minimization
  `(G p - g)' psi^-1 (G p - g)` giving `lambda`; (d) `Var(b) = sigma2 (Xs'Xs)^-1` with `Xs` and `u_s` filtered at
  `lambda` (`sigma2 = u_s'u_s / n`) and `Var(lambda) = (j' psi^-1 j)^-1 / n`, `j = G [1, 2 lambda]'`. The
  closed forms replace spreg's `n x n` `A1`, `A2` and are verified against spreg's own `psi`, `G`, `lambda` and
  standard errors (the betas-`lambda` covariance of spreg is not formed; it changes no reported number).
  Pinned to spreg on a row-standardized lattice, an asymmetric kNN scene and a non-row-standardized (binary) lattice.
- Traces are exact up to 4096 rows and Hutchinson estimates above (`traceMode` overrides). With stochastic traces
  the standard error of `lambda` is approximate and `..._SUMMARY_APPROXIMATION` is 1.0 (betas, `lambda` and the
  betas' standard errors still use the exact moment sums).
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then
  `lambda` last.
- `summary` (13, `GPU_SPATIAL_ERROR_GM_SUMMARY_*`): `n`, `lambda`, `sigma2 = e'e / n` (spreg `sig2n`), pseudo R2,
  the residual sum of squares `u'u` and the moment objective at the solution, then conditioning of the final
  filtered regression (Belsley condition number of the raw filtered `[1, X]`, correlation condition number of the
  factored Gram matrix, largest predictor variance inflation factor, and a precision warning above a correlation
  condition number of 1e4 that never changes `status`), the `lambda` bound, the approximation flag and the
  at-bound flag.
- `status`: 0 ok, 1 singular (OLS or filtered fit), 2 too few rows (`n <= k + 2`), 3 non-finite (including `W`
  without links).
- `residuals`: `u = y - X b`.

`getGPUSpatialErrorGMModelResult(readback, {predictorNames?})` turns the read-back `table`, `summary` and `status`
into the shared `GPURegressionModelResult`: rows `CONSTANT`, the predictors and `lambda` (role `'spatial-error'`),
z statistics with float64 normal p-values, `sigmaSquaredMaximumLikelihood = e'e / n` (and
`residualSumOfSquares = n sigma2` of the filtered residuals, the scale of the standard errors), `pseudoRSquared`,
no likelihood and the conditioning above.

Accumulations use the fixed-order tile and Kahan scheme of `GPUOrdinaryLeastSquares`, so results are
deterministic. The spec matches spreg on a 10x10 lattice and an asymmetric kNN scene (`k = 4`, 80 rows):
`lambda` and the coefficients within 5e-3, standard errors within 1e-2 and z within 2e-2 (relative), for both
estimators. The `GM_Error_Het` variant and the Kelejian-Prucha weighted estimator are not provided.

```ts
graph.add(new GPUSpatialErrorGM({weights, predictors, response, predictorCount: 2, output: {table, summary, status}}));
```

### `GPUSpatialLagMaximumLikelihood`

Maximum-likelihood estimator of the spatial lag model `y = rho W y + X b + e` (spreg 1.9.1 `ML_Lag`, `method='full'`),
with the exact log-determinant replaced by a truncated trace series. Nothing is read back. Props: `weights` (any
square CSR), `predictors`, `response`, `predictorCount` (1 to 14), `maximumPower` (series length `Q`, 2 to 256,
default 128), `traceMode`, `traceProbeCount`, `traceSeed`, optional `tileRowCount` and
`output: {table, summary, status, converged?, residuals?, powerTraces?, powerSums?, infinityNorm?}`. Provide `powerTraces`,
`powerSums` and `infinityNorm` together to share the traces with a downstream [`GPUSpatialLagImpacts`](#gpuspatiallagimpacts).

- Steps: the OLS residuals `e0`, `e1` of `y` and `W y` on `[1, X]` (two `GPUOrdinaryLeastSquares` compositions) make
  the concentrated likelihood `-n/2 ln(s2(rho)) + ln|I - rho W|`, `s2 = (e0 - rho e1)'(e0 - rho e1) / n`, depend on
  three sums. The log-determinant is `-sum_{q<=Q} rho^q tr(W^q) / q` from
  [`GPUSpatialWeightsTraces`](#gpuspatialweightstraces) (exact traces up to 4096 rows, Hutchinson above).
  `rho` is the global maximum on `[-B, B]`: one kernel evaluates the analytic derivative on 1025 grid points,
  bisects (32 steps) every positive-to-non-positive sign change on the derivative, and keeps the best candidate
  (an end point counts when the derivative points outward). It is deterministic and has no iteration budget.
  `b = b0 - rho b1`, `s2 = u'u / n`.
- Variance: spreg's information matrix of `(b, rho, s2)` with `tr1 = tr(W A)` and `tr2 = tr((W A)^2)` as series in
  `rho` of the power traces, and `tr3 = tr((W A)' W A)`, `A = (I - rho W)^-1`, from a second probe pass at the
  estimated `rho` (Horner `u <- z + rho W u` per probe, exact unit probes up to 4096 rows, Rademacher above; the
  probes of `GPUSpatialWeightsTraces`). The `s2` row is eliminated by its Schur complement, the matrix is built on
  centered predictors, unit-scaled and factored with the shared Cholesky, and the intercept variance is mapped back
  to the uncentered model. A positive definite failure leaves the standard errors NaN
  (`..._SUMMARY_VARIANCE_DEFINED` 0) without changing `status`. The coefficient block equals spreg's `vm`.
- Search interval and truncation. `B = min(0.99, seriesLimit) / ||W||_inf` with `||W||_inf` the largest absolute row
  sum computed on the GPU (0.99 for a row-standardized `W`; keeps `I - rho W` invertible for any `W`, whereas spreg's
  fixed `(-1, 1)` admits explosive `rho` for raw weights). The series omits a tail below
  `n s^(Q+1) / ((Q+1) (1 - s))`, `s = |rho| ||W||_inf` (worst case, `|tr(W^q)| <= n s^q`;
  `getSpatialLogDeterminantTruncationBound`), and omitting it biases the likelihood toward large positive `rho`, so
  the interval also stops at the `seriesLimit` where the worst-case tail reaches `1e-5 n`
  (`getSpatialMaximumLikelihoodSeriesLimit`: about 0.93 for `Q = 128`, 0.96 for `Q = 256`). Beyond it the fit is a
  boundary solution (`..._SUMMARY_AT_BOUNDARY`, `..._SERIES_LIMITED`, `converged` 0); raise `maximumPower` when
  `rho` is near the bound. The default `Q = 128` keeps the worst-case truncation below `1e-7 n` log-likelihood units
  at `|rho| <= 0.9` (below `1e-6 |LL|`) at the cost of about `Q (1 + probe batches)` dispatches; the probe pass costs
  `rows x Q x nnz` multiply-adds in exact mode. `..._SUMMARY_TRUNCATION_BOUND` reports the remainder at the estimate.
- Approximation. Above 4096 rows the traces and the `tr3` pass are Hutchinson estimates (`..._SUMMARY_APPROXIMATION`
  1.0): `rho` itself then carries stochastic error (about 1e-3 with 256 probes on a 144-row lattice), unlike
  `GPUSpatialErrorGM` where only the standard error of `lambda` is approximate.
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then `rho` last
  (spreg `W_y`). It is the layout `GPUSpatialLagImpacts` reads (`coefficientStride: 4`, `rhoIndex: predictorCount + 1`).
- `summary` (22, `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*`): `n`, `rho`, `sigma2 = u'u / n`, full log-likelihood
  `-n/2 ln s2 + ln|I - rho W| - n/2 (ln 2 pi + 1)` (spreg `logll`), AIC and Schwarz with spreg's `k = predictorCount + 2`
  (`rho` counted), `pr2` (squared correlation of `y` and `y - u`), `pr2_e` (of `y` and `(I - rho W)^-1 X b`), `u'u`,
  conditioning of the design (Belsley number, correlation condition number, largest VIF, precision warning above
  1e4; never changes `status`), the bound, the approximation, converged, at-boundary and series-limited flags, the
  truncation bound, variance-defined, the number of local maxima found on the grid (more than 1 is multimodal) and
  `ln|I - rho W|`.
- `status`: 0 ok, 1 singular design, 2 too few rows (`n <= k + 2`), 3 non-finite predictor, response or weight (or no
  admissible interval); the outputs are then NaN. `converged` (optional) is 1 for an interior maximum.
- `residuals`: `u = y - rho W y - X b`.

`getGPUSpatialLagMaximumLikelihoodModelResult(readback, {predictorNames?})` builds the shared
`GPURegressionModelResult` (`kind: 'spatial-lag-maximum-likelihood'`, rows `CONSTANT`, predictors and `W_y` with role
`'spatial-lag'`, z statistics with float64 normal p-values, `standardErrorSigmaSquared: 'maximum-likelihood'`,
`logLikelihood`, and AIC and Schwarz equal to spreg's); `getGPUSpatialMaximumLikelihoodDiagnostics(summary)` returns
the flags and `pr2_e`. At most 14 predictors: the `(predictorCount + 2)`-wide information matrix must fit the
16-wide Cholesky. Inputs must be finite.

Verified against spreg `ML_Lag(method='full', spat_impacts='full')` on a 12x12 lattice, a binary (raw) lattice, a
six-predictor lattice and two kNN scenes by a float64 mirror (`rho` to 4e-8, coefficients 3e-7, standard errors 1e-7,
z, log-likelihood, AIC, Schwarz, `pr2`, `pr2_e`, conditioning and `tr1`, `tr2`, `tr3`); the GPU spec checks the float32
output against the same pins at 2e-3 (`rho`, coefficients) and 1e-2 (standard errors), and chains the fit into
`GPUSpatialLagImpacts`. spreg's total multiplier `1 / (1 - rho)` is exact only for a row-standardized `W`; for other
weights the impacts here use the exact `1'W^q 1` sums.

```ts
graph.add(new GPUSpatialLagMaximumLikelihood({weights, predictors, response, predictorCount: 2,
  output: {table, summary, status, powerTraces, powerSums, infinityNorm}}));
graph.add(new GPUSpatialLagImpacts({coefficients: table, coefficientStride: 4, rhoIndex: 3, predictorCount: 2,
  rowCount, maximumPower: 128, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}}));
```

### `GPUSpatialErrorMaximumLikelihood`

Maximum-likelihood estimator of the spatial error model `y = X b + u`, `u = lambda W u + e` (spreg 1.9.1
`ML_Error`, `method='full'`), with the same truncated-series log-determinant, search and approximation rules as
`GPUSpatialLagMaximumLikelihood`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 15),
`maximumPower`, `traceMode`, `traceProbeCount`, `traceSeed`, optional `tileRowCount` and
`output: {table, summary, status, converged?, residuals?}`.

- Steps: with `Z = [1, X - mean, y - mean]`, three tile passes (Kahan-merged, fixed order) accumulate once
  `G0 = Z'Z`, `G1 + G1' = Z'WZ + (Z'WZ)'` and `G2 = (WZ)'(WZ)`. The Gram matrix of the filtered columns
  `Z - lambda W Z` (the filtered constant `1 - lambda sum_j w_ij` is column 0, as in spreg) is then exactly
  `G0 - lambda (G1 + G1') + lambda^2 G2`, so scanning `lambda` needs no pass over the rows. Each grid point solves
  one unit-scaled Cholesky system for `ee(lambda)`, and the derivative follows from the envelope theorem
  (`ee' = yy' - 2 b'r' + b'A'b`); the search is the one of the lag model on `-n/2 ln(ee / n) + ln|I - lambda W|`.
- Variance: `Var(b) = s2 (Xs'Xs)^-1` (centered form mapped back to the uncentered intercept) and
  `Var(lambda) = 1 / (tr2 + tr3 - 2 tr1^2 / n)`, the Schur complement of spreg's 2 x 2 information matrix of
  `(lambda, s2)`, with the traces of `W (I - lambda W)^-1` (`tr3` from the probe pass at the estimated `lambda`). As in
  spreg the coefficient standard errors condition on `lambda`.
- `table`: `(k + 2) x 4` rows, `lambda` last. `summary`: the 22 slots of the lag model; `sigma2 = e'e / n` of the
  filtered residuals, `pr2` of `y` and `X b`, `pr2_e` NaN, `u'u` of `u = y - X b`, and the conditioning of the final
  filtered regression (Belsley number of the raw `[c, X - lambda W X]`, correlation condition number and largest
  predictor VIF of the unit-scaled filtered system). AIC and Schwarz use spreg's `k = predictorCount + 1`: spreg's
  `ML_Error` does not count `lambda` (`k = x.shape[1]`), unlike `ML_Lag`, which counts `rho`.
- `status` as the lag model (the cap is 15 predictors, the OLS and Cholesky limit).

`getGPUSpatialErrorMaximumLikelihoodModelResult(readback, {predictorNames?})` builds the shared model result
(`kind: 'spatial-error-maximum-likelihood'`, `lambda` role `'spatial-error'`, `residualSumOfSquares = n sigma2` of the
filtered residuals, AIC and Schwarz equal to spreg's). Verified against `ML_Error(method='full')` on a
row-standardized lattice, an asymmetric kNN scene and a binary lattice (float64 mirror to 1e-7; GPU spec at 2e-3 and
1e-2, not yet measured).

```ts
graph.add(new GPUSpatialErrorMaximumLikelihood({weights, predictors, response, predictorCount: 2,
  output: {table, summary, status}}));
```

## Recipes

Recipes are plain functions, not classes: `addXRecipe(graph, props)` constructs the contributors, calls
`graph.add` in chain order, wires outputs to inputs and returns `{contributors, ...named views}`. Views you
pass in (`zScores`, `bins`, `colors`, ...) are caller-owned and readable; every other intermediate is a
graph transient. Per-frame parameter views (`getGPU...ParameterValues`) are passed through, so changing them
never rebuilds the graph.

```ts
const hotSpots = addHotSpotAnalysisRecipe(graph, {
  source: {kind: 'points', positions, family: 'quadbin', resolution: 8, cellCapacity: 4096,
    studyArea: {polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity: 16384}},
  parameters: autocorrelationParameters,
  outputs: {zScores, bins},
  permutation: {parameters: permutationParameters, maximumPermutations: 199},
  color: {classBreaksParameters, maximumClassCount: 5, colorScaleParameters, palette, maximumPaletteCount: 5}
});
```

| Recipe | Chain |
| --- | --- |
| `addHotSpotAnalysisRecipe` | GPUCellCover study area -> GPUCellAggregation (sorted unique study cells) + (GPUPointToCell) -> GPUCellAggregation (points) -> join (zeros kept) -> area density (GPUCellMeasures for H3) -> GPUCellTopology disk -> CSR weights (or GPULatticeWeights for raster source) -> GPUHotSpotAnalysis -> GPULocalPermutationTest -> GPUClassBreaks -> GPUColorScale |
| `addRateClusterMapRecipe` | GPUEmpiricalBayesRates -> GPUContiguityWeights -> GPUSpatialWeightsTransform(row) -> GPULocalMoran (`quadrantGating: 'none'` with permutation) -> GPULocalPermutationTest -> quadrant colors |
| `addPointsInPolygonsChoroplethRecipe` | backend 'zonal' (GPUZonalStatistics) or 'group' (GPUPointInPolygonJoin -> dense GPUGroupStatistics) -> GPUClassBreaks -> GPUColorScale |
| `addSpaceTimeHotSpotsRecipe` | GPUCalendarBuckets (absolute time-step slices) -> atomic count into the dense cube (sorted GPUGroupStatistics path when its scratch or overflow outputs are passed) -> GPUEmergingHotSpots -> ordinal GPUColorScale |
| `addClusterAndOutlineRecipe` | GPUSpatialClustering -> GPUGroupGeometry + GPUGroupConvexHull -> GPUGeometryMeasures |
| `addSpatialRegressionRecipe` | GPUOrdinaryLeastSquares -> GPUSpatialRegressionDiagnostics -> GPU specification decision (Anselin's rule: 0 OLS, 1 lag, 2 error, 3 both robust, 4 undetermined) + residual GPULocalMoran (+ optional GPUGeographicallyWeightedRegression); `mask` and `restandardizeRows` flow through OLS and diagnostics |
| `addDriveTimeCatchmentRecipe` | GPUNetworkSnapping -> GPUNetworkServiceAreas -> (GPUNetworkIsochrones raster and/or cell outline with rings -> GPUPointInPolygonJoin demand join) -> snap demand -> band adapters -> GPUGroupStatistics |
| `addStraightLineCatchmentsRecipe` | GPUDistanceField allocation -> GPURasterZonalStatistics |
| `addChangeOfSupportRecipe` | 2x GPUPolygonRasterization -> GPUArealInterpolation (with source conservation diagnostics) -> GPUSpatialLag |
| `addFleetDwellRecipe` | GPUTrajectoryMetrics stops -> GPUPointInPolygonJoin -> stop mask -> GPUGroupStatistics (dense, one row per zone) |
| `addFleetDwellZoneEventsRecipe` | GPUZoneEvents -> GPUGroupStatistics (dense, one row per zone; zone-event variant of the fleet dwell recipe) |
| `addPeriodComparisonRecipe` | 2x GPUCellAggregation -> GPUCellTableCompare (with mask) -> GPUClassBreaks -> GPUColorScale |

Notes:

- The space-time recipe counts events into a dense `cell * sliceCount + slice` cube with integer atomics,
  because `GPUCellAggregation` keys must be valid Quadbin or H3 cells. Passing the sorted-path scratch
  views or `outputs.groupOverflow` selects the `GPUGroupStatistics` path instead.
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
  polygon join is coarser than the network-space bands. With `candidateCapacity`, each snapping reports
  BVH candidate overflow in `result.snapping` and the `snap-facilities` and `snap-demand` status stages,
  so a facility or demand point snapped past a dropped candidate is never silent.
- The drive-time, straight-line catchment and fleet dwell recipes take an optional `spatialContext` and
  return the resolved one as `result.spatialContext`. The straight-line recipe and both fleet dwell recipes
  are planar only (longitude/latitude throws; the join and zone edges are straight lines in the plane, and the
  distance field is Euclidean). The drive-time recipe also accepts longitude/latitude (metric `'none'` or
  `'great-circle'`; the cell-outline isochrones need degrees). Both of its snappings then run great-circle:
  edges are ranked by great-circle distance and `maxSnapDistance` is in meters, with no east-west bias at
  any latitude. Drive times stay in the units of `network.weights` (the snapped fraction splits the edge
  cost). For the other recipes, project with `createGPULocalFrame` and pass its `spatialContext` (planar,
  meters), so the distance-field `settings` and the straight-line `distances` output are in meters.
  The fleet dwell recipes used to hand the trajectory stage a native planar context whatever the data;
  they now resolve one context and pass it to the metrics, join and zone stages.
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

### `addHotSpotAnalysisRecipe`

Points on a study area, or a raster, to Getis-Ord Gi* hot and cold spots.

Points source (breaking change from the occupied-cells design):

- `studyArea` (polygons in GeoArrow offsets plus `candidateCapacity`, optional `coverCapacity`) and
  `cellCapacity` are required. The recipe covers the polygons with `GPUCellCover` (`'center'` containment),
  deduplicates and sorts the cells with `GPUCellAggregation`, aggregates the points the same way and joins them
  by binary search of each 64-bit key. Every study cell is a row, with value 0 when it holds no point, so the
  null mean is not inflated and clusters of empty cells can be cold spots. Points outside the study area are
  excluded; `result.pointsOutsideStudyArea` counts them. `result.table.cells` are the sorted study cells and
  `table.counts` / `table.sumValues` the joined values.
- Neighbors are topological: the cells within `neighborRings` grid steps (`GPUCellTopology` disk) that are also
  study cells, self excluded, binary weights, rows ascending. The defaults are 1 ring for Quadbin (8 neighbors)
  and 2 for H3 (18 neighbors), the smallest disks that reach `minimumNeighbors: 8`. This replaces the degree
  radius and the `neighborSearchParameters`, `gridSize` and `tableCapacity` props. `neighborCapacity` is
  optional (defaults to the largest disk); truncation sets the `weights` status `overflow`.
- `distanceBand` (optional, exclusive with `neighborRings`) uses a metric band instead: `GPUNeighborSearch` in radius
  mode over the cell centers with `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`,
  so the radius in `neighborSearchParameters` is in meters (the context's `units`). Any other context throws: a
  radius in degrees is an ellipse that narrows east-west with latitude. `minimumNeighbors` still applies.
  `distanceBand.rowStandardize` (default `false`, classic binary Gi*) row-standardizes the band through the
  search's compile-time `rowStandardize`, overriding the per-frame field of `neighborSearchParameters`. The
  described weights (`info.transform` `'row'` or `'binary'`, and the search status) feed `GPUHotSpotAnalysis`.
- `analyze: 'density'` (default) analyses value (or count) per square kilometer. Web Mercator tiles shrink
  with the cosine of latitude, so raw Quadbin counts compare cells of different area; `'count'` is allowed for
  H3 only (cells are approximately equal area) and throws for Quadbin. Quadbin area is the exact spherical
  rectangle `R^2 dLon (sin(lat1) - sin(lat2))` and H3 area comes from `GPUCellMeasures`, both on the authalic
  sphere (`GPU_EARTH_AUTHALIC_RADIUS_METERS`, which is H3's own radius), so the families share one model.
  Quadbin areas are within f32 rounding (about 1e-6 relative); H3 pentagons and edge cells above resolution 5 are
  unreliable (see `GPUCellMeasures`).

Statistics and legend (both sources):

- `minimumNeighbors` defaults to 8 (ArcGIS guidance for Gi*). Rows with fewer neighbors other than themselves
  get NaN `zScores` and `pValues`, bin 0, and are outside the multiple-testing family;
  `status` stage `minimum-neighbors` reports `invalidCount`. Quadbin with one ring has exactly 8 neighbors at
  most, so the cells on the study-area border are NaN; use `neighborRings: 2` or a lower minimum for them.
- `multipleTesting` defaults to `'benjamini-hochberg'` (it replaces `falseDiscoveryRate`). Legend contract,
  stable under any setting: bins `+-1`, `+-2`, `+-3` are 90%, 95% and 99% confidence hot (`+`) or cold (`-`)
  spots; without correction that is `|z| >= 1.645, 1.960, 2.576`, with Benjamini-Hochberg the step-up test at
  q = 0.10, 0.05, 0.01. `result.inference` is a `GPUInferenceMetadata` (alternative `'two-sided'`, analytic so
  `permutationCount: 0`, the method, and `testedRowsDefinition`, for example `'study-area cells with at least 8
  neighbors'`); `result.permutationInference` describes the optional permutation confirmation.
- The lattice source keeps its queen/rook weights and gets the same guard and metadata.
- Permutation confirmation tests every study row; it does not apply `minimumNeighbors`.

### `addRateClusterMapRecipe`

Event counts and populations per polygon to significant local Moran clusters of empirical-Bayes rates (esda
`Moran_Local_Rate`).

- `analyze` is `'standardized'` (default, Assuncao-Reis) or `'smoothed'` (global empirical-Bayes rate).
  `clampPriorVariance` (default `true`) is passed to `GPUEmpiricalBayesRates` and only changes the `'smoothed'`
  path: the prior variance is clamped at zero (Marshall), so counts no more dispersed than Poisson shrink fully
  to the pooled rate. `false` reproduces esda `Empirical_Bayes`. The standardized path is unaffected.
- `multipleTesting` defaults to `'benjamini-hochberg'` (it replaces `permutation.falseDiscoveryRate`) and applies
  to whichever inference runs: the analytic `GPULocalMoran` p-values without `permutation`, or the permutation
  pseudo p-values with it. The family is every polygon with a valid rate and at least one neighbor, never a viewport.
- Analytic inference is a preview (two-sided normal approximation of the conditional-randomization moments).
  Permutation inference is the reference; esda uses 999 permutations and so should you. The alternative defaults to
  `'two-sided'`. Analytic p-values are exposed as `result.analytic`, permutation ones as `result.permutation`
  (`pseudoPValues`, and `adjustedPValues` after the correction).
- Islands and polygons with a NaN rate (zero population, masked) end at quadrant 0 and have NaN p-values: they are
  outside the test family, not counted as tested but not significant.
- Legend contract, stable under any setting: palette index 0 is "not significant" and 1, 2, 3, 4 are high-high,
  low-high, low-low, high-low. The correction changes which polygons reach significance, never what a color means.
- `result.inference` is the `getInferenceMetadata` of the `GPULocalPermutationTest` the recipe built (or of
  `GPULocalMoran` without `permutation`), so it comes from the same settings that are packed for the GPU. It holds
  the alternative, `permutationCount` (0 for analytic), `seed` when given in `permutation.seed`,
  `significanceLevel` (default 0.05, recorded on both paths), `multipleTesting` and `testedRowsDefinition`. The
  GPU reads the real seed and count from the parameter views. An invalid `permutation.permutations` or `seed`
  throws when the recipe is built.

### `addPeriodComparisonRecipe`

Two periods of events to a classified, colored change map.

- `delta` (after minus before) is always defined. `ratio` and `percentChange` are NaN, not 0, for cells whose earlier
  value is 0; NaN rows are skipped by `GPUClassBreaks` and take the palette no-data color.
- `zScore` is a descriptive score of the change (`'poisson'`: `delta / sqrt(before + after)`, or `'standardized'`
  deltas). It has no null distribution and no multiple-testing correction, so a threshold on it is not a significance
  level; use `addHotSpotAnalysisRecipe` or `addRateClusterMapRecipe` for inference. With `measure: 'count'` every
  union cell has `before + after > 0`, so the Poisson denominator is never zero.

### `addChangeOfSupportRecipe`

Transfers values between two polygon zone systems by area: both systems are rasterized to one grid
(`GPUPolygonRasterization`), `GPUArealInterpolation` builds the overlap areas and both weight sets, and
`GPUSpatialLag` moves `sourceValues` to the targets as `extensiveValues` and `intensiveValues`.

- Props: `width`, `height`, `extent` (rasterization extent), `source` and `target` zone systems,
  `pairCapacity`, optional `cellWeights`, `sourceValues` and `columnCount`, `categories`, `denominator`,
  `sourceAnchorCells`, `spatialContext` and `outputs` / `scratch` overrides.
- Denominators (default change): extensive `'zone'`, intensive `'overlap'`, so a target only partly
  covered by sources is the mean of what covers it. Pass `denominator: 'zone'` for the old intensive result.
- Conservation: the result always carries `sourceAreas`, `sourceTransferredFractions` and `invalidCount`;
  the last one is also in the `areal-interpolation` stage `status`. A source entirely outside the targets,
  or smaller than a cell, shows up there instead of silently losing mass; `sourceAnchorCells` rescues the
  latter.
- Coordinates: `spatialContext` omitted is native planar (cell counts are areas). `longitude-latitude`
  (metric `'none'` or `'great-circle'`) declares a longitude/latitude raster whose `extent` is in degrees,
  and the areal stage weights each cell by its spherical area in square meters. Rasterization itself only
  tests cell centers in the coordinates as given (straight polygon edges in degrees; split polygons at the
  antimeridian). The resolved context is returned as `result.spatialContext`.
- Accuracy is bounded by the raster resolution (cell centers decide membership), never an exact polygon
  intersection.

### `addPointsInPolygonsChoroplethRecipe`

Points to a per-polygon statistic to class breaks to colors, with `backend: 'zonal'` (default,
`GPUZonalStatistics`) or `'group'` (`GPUPointInPolygonJoin` then dense `GPUGroupStatistics`). `statistic`
is `'count'` (default), `'sum'`, `'mean'`, `'minimum'` or `'maximum'`; features with no points keep NaN (mean,
minimum, maximum) or 0 (count, sum).

- Coordinates: `spatialContext` omitted is native planar. `longitude-latitude` is accepted because the
  mapped statistics are topological: containment uses straight polygon edges in degrees (not geodesics) and
  polygons crossing the antimeridian must be split. No density is mapped, so no polygon area is needed; for
  a density per square meter use `GPUZonalStatistics` directly with spherical `areas`. Other spaces throw.
  The resolved context is returned as `result.spatialContext`.

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
| `buffer` | `GPUBufferSurface` for bounded repaired polygon topology; `GPUOutlineGeometry` for render-only geometry; `GPUBufferSelection` for "within distance" selection |
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
| `union`, `intersect`, `difference`, `dissolve`, `voronoi`, `tin`, `concave` | `GPUPolygonOverlay` for bounded Boolean overlay/dissolve; `GPUDelaunayTessellation` and `GPUVoronoiDiagram` for vector tessellation. Concave construction remains out of scope; coverage-only union is `GPUCoverageDissolve` and raster Voronoi is `GPUDistanceField` |
| `polygonize` | `GPUPolygonize` for arbitrary finite planar linework; `GPULineNoding` exposes the atomic noded substrate and provenance; `GPUSegmentRingAssembly` remains the lower-level directed-boundary primitive |

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
| `ST_Buffer` | `GPUBufferSurface` for repaired polygon output; `GPUOutlineGeometry` for render-only geometry |
| `ST_Intersection`, `ST_Union`, `ST_Difference`, `ST_SymDifference`, general `ST_Dissolve` | `GPUPolygonOverlay` with explicit arrangement and output capacities |
| `ST_DelaunayTriangles`, `ST_VoronoiLines` | `GPUDelaunayTessellation`, then `GPUVoronoiDiagram` with dynamic clip bounds |
| `ST_MakeValid` | `GPUMakeValid` rebuilds planar polygon boundaries from their noded bounded faces and reports collapsed/unused edges |
| `ST_Polygonize`, `ST_BuildArea`, `ST_MakePolygon` (from boundary segments or closed lines) | `GPUPolygonize`; `GPULineNoding` and `GPUSegmentRingAssembly` expose the independently measurable noding and face-extraction stages |
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
| `GeoSeries.buffer` (as geometry) | `GPUBufferSurface` (planar; line and polygon joins/caps plus repair) |
| `shapely.offset_curve`, `snap` | `GPUOffsetCurve`, `GPUVertexSnap` (partial, see their sections) |
| `shapely.minimum_rotated_rectangle`, `oriented_envelope`, `minimum_bounding_circle`, `minimum_bounding_radius` | `GPUMinimumBounds` |
| `shapely.minimum_clearance`, `minimum_clearance_line` | `GPUMinimumClearance` |
| `shapely.is_simple`, `is_ring`, `is_closed`, `is_ccw`, `get_num_points`, `extract_unique_points`, `equals_exact` | `GPUGeometryPredicates` |
| `GeoSeries.dissolve(method='coverage')`, `shapely.coverage_union_all`, `coverage_is_valid`, `coverage_invalid_edges` | `GPUCoverageDissolve`, `GPUCoverageValidity` |
| `geopandas.clip(lines, polygon)`, `shapely.shared_paths` | `GPULineClipByPolygon`, `GPUSharedPaths` |
| `sjoin` merge, `sjoin_nearest(distance_col)`, `explode`, `get_parts`, `.cx` | `GPUPairGather`, `GPUOffsetExpansion`, `GPUBoundsFilter` |
| `GeoSeries.make_grid`, `sample_points` (lines) | `GPUGridGenerator` (`hexOrientation`, `extent`), `GPURandomPointsOnLine` |
| `overlay`, `dissolve`, `unary_union` | `GPUPolygonOverlay`; coverage-only dissolve remains `GPUCoverageDissolve` |
| `voronoi_polygons`, `delaunay_triangles`, `make_valid` | `GPUVoronoiDiagram`, `GPUDelaunayTessellation`, `GPUMakeValid` |
| `concave_hull` | Not built; a definition and parameter contract remain evidence-gated |
| `shapely.polygonize`, `polygonize_full` | `GPULineNoding` plus `GPUPolygonize`; diagnostics distinguish rings, open chains, cut edges, dangles, unused edges, and degenerates |
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
| `smoothing.Empirical_Bayes` | `GPUEmpiricalBayesRates` (`clampPriorVariance: false` for exact parity); recipe `addRateClusterMapRecipe` |
| `esda.Moran_Rate`, `Moran_Local_Rate`, `assuncao_rate` | `GPUAssuncaoReisStandardization` followed by `GPUGlobalSpatialStatistics` or `GPULocalMoran` |
| `esda.fdr`, `statsmodels.multipletests` | `GPUMultipleTestingAdjustment`; `multipleTesting` option of `GPUHotSpotAnalysis`, `GPULocalMoran`, `GPULocalPermutationTest` |
| `pointpats.PointPattern`, `ripley.k_test`, `f_test`, `g_test`, `j_test` | `GPUPointPatternIndices`, `GPURipley`, `GPURipleyDistanceFunctions` (dataframe) |
| `pointpats.Knox`, `Mantel` | `GPUKnoxTest`, `GPUMantelTest` |
| `esda.Moran` correlogram (`spatial_correlogram`) | `GPUSpatialCorrelogram` (dataframe) |
| `esda.Spatial_Pearson`, `Spatial_Pearson_Local` (Lee's L) | `GPUGlobalPermutationTest` (`'leesL'`), `GPULocalPermutationTest` (`'localLeeL'`) |
| `esda.Local_Geary`, `Gamma` | `GPULocalGeary`, `GPUGammaStatistic`; shared permutation adapters use `GPUGlobalPermutationTest` or `GPULocalPermutationTest` |
| `spreg.OLS` | `GPUOrdinaryLeastSquares` |
| `spreg.diagnostics_sp.LMtests`, `MoranRes` | `GPUSpatialRegressionDiagnostics`; recipe `addSpatialRegressionRecipe` |
| `spreg.GM_Lag` | `GPUSpatialTwoStageLeastSquares` |
| `spreg.ML_Lag`, `ML_Error` (`method='full'`), `spat_impacts` | `GPUSpatialLagMaximumLikelihood`, `GPUSpatialErrorMaximumLikelihood`; impacts `GPUSpatialLagImpacts` over `GPUSpatialWeightsTraces` |
| `spreg` panel, SUR, regimes, probit | Use caller-partitioned `GPUOrdinaryLeastSquares` for panel or regime fits and `spreg` when SUR or probit inference is required |
| `mgwr.GWR` | `GPUGeographicallyWeightedRegression`; `MGWR` backfitting is not built |
| `giddy.Markov`, `LISA_Markov`, `Spatial_Markov` | `GPUTransitionMatrix`, `GPULISAMarkov`, `GPUSpatialMarkov` |
| `giddy.directional`, `Rose` | `GPURoseStatistic` with explicit directional or axial circular semantics |
| `segregation` aspatial and spatial indices | `GPUSegregation` |
| `inequality.gini.Gini`, `theil.Theil`, `Gini_Spatial` | `GPUInequality` (dataframe) |
| `esda.shape` (`isoperimetric_quotient`, `convex_hull_ratio`) | `GPUShapeDescriptors` |
| `spopt.region.AZP`, `MaxPHeuristic`, `Skater`, `Ward` | `GPUAZPRegions` (greedy steepest-descent AZP), `GPUMaxPRegions` (randomized multi-start greedy construction with optional AZP refinement), `GPUSkaterRegions`, `GPUWardRegions` (bounded adjacent Ward linkage) |
| `spopt.locate.PMedian`, `MCLP`, `LSCP` | `GPULocationAllocation` provides bounded greedy p-median, maximum-coverage, and set-covering selection (plus a swap improvement phase) over `GPUNetworkCostMatrix` or another dense cost matrix; use `spopt` when an integer-programming optimum or richer constraints are required |
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
| Polygonize / Fix geometries | `GPULineNoding`, `GPUPolygonize`, and `GPUMakeValid`; `GPUSegmentRingAssembly` remains the lower-level already-noded primitive |
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
| Dissolve, Union, Intersection, Difference, Buffer (geometry), Delaunay, Voronoi (vector), Fix geometries | `GPUPolygonOverlay`, `GPUBufferSurface`, `GPUDelaunayTessellation`, `GPUVoronoiDiagram`, and `GPUMakeValid`; spatial-index and virtual-raster ownership remain in their dedicated packages |
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
