# Upgrade Guide

The upgrade guide lists breaking changes in each major and minor version of the luma.gl API, and provides information on how to update applications.

Upgrade instructions assume that you are upgrading from the immediately previous release.
If you are upgrading across multiple releases you will want to consider the release notes for all
intermediary releases.

luma.gl largely follows [SEMVER](https://semver.org) conventions. Breaking changes are typically only done in major versions, minor version bumps bring new functionality but few breaking changes, and patch releases typically contain only low-risk fixes.

*For detailed commit level logs that include alpha and beta releases, see the [CHANGELOG](https://github.com/visgl/luma.gl/blob/master/CHANGELOG.md) in the github repository.*

## Upgrading to v10.0

### Render pass draw offsets and counts

Indexed WebGL draws now use `firstIndex` in index elements. Replace uses of
`firstVertex` as an indexed byte offset with `firstIndex: byteOffset / indexByteSize`.
`firstVertex` only applies to non-indexed draws. WebGL now rejects `indexCount`
unless the selected vertex array has an index buffer; attribute-less non-indexed
draws should use `vertexCount`.
WebGPU render passes and render bundles now honor explicit zero draw and instance
counts instead of drawing an implicit instance.

### Model parameters on WebGL

On WebGL, `Model.setParameters()` no longer replaces `model.pipeline`. `Model.draw()` passes
the model's current parameters to every draw call, as before. Code that draws
`model.pipeline` directly without passing `parameters` gets the parameters the retained
pipeline was created with; pass `model.parameters` to the draw call instead.

### CPU geometry validation

Engine geometry now delegates CPU storage and tessellation to `@math.gl/geometry`.
Indices must reference valid vertices across the entire supplied index array, including entries
after an explicit draw count. Remove invalid unused indices before constructing a geometry.
Uint8 indices supplied to the engine compatibility adapter are widened to Uint16.
`TruncatedConeGeometry` defaults to a bottom radius of 1; provide explicit nonzero radii for
frusta. Primitive tessellation order can change, so derive face metadata from normals rather
than relying on vertex or index ordering.

**@luma.gl/gpgpu**

### GPUFFT2D graph lifecycle

`GPUFFT2D` now takes `{input, output, width, height, batchCount?, direction?}` graph views.
Replace `new GPUFFT2D(device, dimensions)` and `transform.encode(encoder, buffers)` with
`graph.add(new GPUFFT2D(props))`, `graph.compile()`, and `compiled.encode(encoder, {parameters: undefined})`.
Destroy the compiled graph instead of the primitive. `GPUFFT2DEncodeOptions` is removed.
Forward and inverse transforms are separate graph operations; pass parameters are immutable.
See [GPUFFT2D](./api-reference/experimental/gpu-core/gpu-fft2d) for the migration example.

- `GPULZByteDecompressor` descriptors now contain five uint32 words per record:
  `[outputOffset, byteLength, literalSourceOffset, literalPeriod, matchOffset]`. Code that creates
  descriptor buffers directly must add `literalPeriod` and use `GPU_LZ_BYTE_DESCRIPTOR_WORDS` when
  allocating and indexing them. Prefer `planGPULZByteDescriptors()` to convert the unchanged
  four-word parser spans into GPU descriptors.

### GPUSort radix tiling

- Radix `GPUSort` processes `elementsPerThread` keys per thread, default `8`, so one workgroup
  covers 2048 keys. Dispatch sizes, histogram lengths and scratch sizes shrink accordingly, and
  ranges that previously exceeded a small `maxComputeWorkgroupsPerDimension` now fit. Pass
  `elementsPerThread: 1` for the previous tiling. `GPUSort.digitBits` is typed `4 | 8`.

**@luma.gl/shadertools**

- `ShaderPassPipeline`, `ShaderPassPipelineStep`, and `ShaderPassComputeOptimization` have been
  renamed to `CompositeShaderPass`, `CompositeShaderPassStep`, and
  `CompositeShaderPassComputeOptimization`. Effect factories and values likewise replace their
  `ShaderPassPipeline` suffix with `CompositeShaderPass`.

**@luma.gl/webgpu**

- `WebGPUDevice.adapter` is now typed `GPUAdapter | null`. It is `null` for devices wrapped with `webgpuAdapter.attach()` / `luma.attachDevice()`, because a `GPUDevice` does not reference its `GPUAdapter`. Devices from `luma.createDevice()` still have an adapter; TypeScript code that reads `device.adapter` needs a null check or non-null assertion. Use `device.adapterInfo` for adapter metadata.

**@luma.gl/splats**

- `SPLAT_DEPTH_KEY_BITS` dropped from `24` to `16`. `packSplatDepthKey(depth, options)` now takes
  `{mode, keyBits, depthMin, depthMax, tileId}`; `mode` defaults to `'linear'` (the old behavior)
  but keys are 16 bits and the largest visible key is `2 ** keyBits - 2`, because the all-ones key
  is reserved for culled rows (`getSplatInvalidDepthKey()`). A `tileId` now sits directly above
  `keyBits`. `'linear'` and `'ndc'` keys are clamped to at most 24 bits.
- `GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH` is `32` (was `48`): a projected record is a `vec4<f32>`
  clip center followed by half-precision axes and color (`packedAxis0`, `packedAxis1`,
  `packedColorRG`, `packedColorBA`). `GPU_SPLAT_GRAPH_UNIFORM_BYTE_LENGTH` is `160` (was `128`)
  with a new `GraphSplatUniforms` layout. Custom shaders reading renderer buffers should include
  `GPU_SPLAT_GRAPH_SHARED_WGSL` and use its accessors rather than hard-coding offsets.
- Visual defaults changed in `GPUSplatGraphRenderer` and `GPUPagedSplatRenderer`, and the screen
  filter default in `SplatRenderer`:
  - The screen-space filter adds `0.3` px² of variance (the reference rasterizer), not
    `0.3² = 0.09`. `kernel2DSize` (a standard deviation) is deprecated in favor of
    `screenSpaceFilterVariance`; pass `screenSpaceFilterVariance: 0.09` for the old look.
    `projectSplatCovarianceToScreen()` still adds no filter unless one is passed.
  - Mip-Splatting opacity compensation is on (`antialiasing: 'mip-splatting'`; use `'none'` for
    base 3DGS), as are `dynamicSupportRadius` and `compensateScreenSpaceClamp`.
  - The GPU sort key defaults to `depthKeyMode: 'float16'` (view-space distance) instead of
    hyperbolic device depth. Under an orthographic (affine) projection every mode keys on device
    depth instead, since clip `w` is constant there.
  - `pickingAlphaThreshold` defaults to `0`, so picking uses `alphaCutoff` like rendering.
  - `alphaMode: 'stochastic'` sorts a one-bit visibility key (`renderer.depthKeyBits === 1`), and
    `GPUSplatGraphMixedRenderer` always enables depth writes and disables blending for it.
- `getSplatDilationCompensation()` returns `0` rather than `1` when the pre-dilation covariance is
  degenerate, so point- or line-like projections no longer render at full opacity.
- `GPUPagedSplatRendererProps` no longer accepts graph-renderer props the paged renderer ignored
  (`clipRegion`, `antialiasing`, `fragmentKernel`, `dynamicSupportRadius`,
  `compensateScreenSpaceClamp`, `depthKeyMode`, `depthKeyBits`, `depthRange`, `alphaMode`,
  `pickingAlphaThreshold`, `batchParams`, `renderPath`, `presentation`).
- `packSplatClipUniforms()` drops degenerate planes before enforcing the 8-plane limit.
- `SplatHierarchyFrontierEntry` gained required `filterVariance` and `fadeOpacity` fields, and
  `SplatHierarchyStats` gained required `budgetedSplatCount`, `budgetExhausted` and
  `framesSinceBudgetPlan`. Code that constructs these objects itself must supply them.
- `getSplatHierarchyFoveatedPriority()` now measures to the nearest edge of a page's projected
  bounds rather than its center, so large or nearby pages are relaxed less than before.

**@luma.gl/experimental**

- `sumOrder` now defaults to `'sorted'`, so float sums and means are bitwise reproducible, for
  `GPURasterZonalStatistics`, `GPUFlowAggregation` and `GPUSpatialClustering`. `GPUZonalStatistics`
  defaults to `'sorted'` when every point-rate input is packed and to `'atomic'` when chunked.
  Last-bit differences against previous atomic output are expected. Pass `sumOrder: 'atomic'` for the
  old behavior, which is faster only with many scattered zones or nearly unique flow pairs.
- Iteration defaults rose: `GPUCostDistance.maxIterations` 64 to 512 (new `sweepAfterIteration`,
  default 96), `GPUTerrainFlow.maxFillIterations` and `maxFlatIterations` 128 to 512 (sweeps after 96).
  Converged results are unchanged; unconverged graphs now unroll more nodes per phase.
- `GPURasterStretch` `gamma` is now the display gamma `t ** (1 / gamma)` (it was `t ** gamma`), so
  `gamma > 1` brightens. Pass `1 / gamma` to keep old output; a GDAL `-exponent e` is `gamma: 1 / e`.
- Capped terrain and raster loops no longer return partial results that read as converged ones.
  `GPUCostDistance` writes NaN costs (and `GPU_COST_DISTANCE_UNRESOLVED` bands and band counts) on
  passable cells when `maxIterations` stops it. `GPUTerrainFlow` writes `GPU_TERRAIN_FLOW_UNRESOLVED`
  directions, class `GPU_TERRAIN_FLOW_CELL_CLASS.unresolved` (5), NaN filled elevation and
  accumulation and unresolved streams on valid cells when filling or flat resolution stops early;
  HAND, watersheds and stream order propagate it. `GPUProfilePeaks`, `GPUStreamlines`,
  `GPUTerrainSpikeRepair` and the raster patch contributors mark undecided output with their new
  `*_UNRESOLVED` / `*_UNCONVERGED` constants. Converged runs are unchanged.
- `GPUPointHorizonVisibility` `targetIgnoreDistance` defaults to one ground pixel along the ray (was
  150 m, which classed every target within 150 m as visible). Pass
  `GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE` for mt-image behaviour.
- `GPUGeomorphons` ignores `flatDistance` at or inside the skip distance and at or beyond the search
  distance, as GRASS `r.geomorphon` does.
- Isolines and isobands treat ±Infinity samples as nodata, and the isobands packer rejects y-down or
  zero-size extents.
- `GPUTerrainCumulativeViewshed` sizes its dispatches by sight-line samples (2^28 per dispatch), so
  `observersPerDispatch` is fractional on large grids; the default node count changes.
- `GPUTextureShading` computes coarse cascade levels on a decimated grid by default
  (`downsampleLevels: true`). Output differs from before by at most 1% of the output range (measured
  0.1-0.27%); pass `downsampleLevels: false` for the exact cascade.
- `GPUGeographicallyWeightedRegression` bisquare-fixed fits sum in grid cell order, shifting results
  by about 1e-5 relative to before; they stay deterministic.
- Network defaults changed. `GPUNetworkReachability.localIterations` 16 to 32 (costs are identical;
  rounds needed drop). `GPUNetworkCostMatrix.laneCount` is `recommendLaneCount()` (at least 32, about
  1M expanded nodes per batch, capped by 128 MB scratch and `rowCount`) instead of 32; results are
  identical, scratch memory grows. `GPUNetworkServiceAreas` labels with a frontier pass bounded by new
  `labelIterations`.
- `GPUNetworkStatistics` drops `countSelfLoopsTwice`: on undirected graphs a self-loop now always adds
  2 to degrees and modularity volumes (networkx), so degrees and modularity change on graphs with loops.
- Network status outputs are required where a capacity can drop work: `GPUNetworkLineGraph.overflow`
  always, and `overflow` of `GPUNetworkSnapping` and `GPUNetworkKFunction` whenever `candidateCapacity` is
  set. `GPUMapMatching.candidateCount` defaults to 8 (was 4; a directed edge row is one candidate).
- OIT fullscreen resolution is now exposed as `createABufferResolveCompositeShaderPass()` and
  `createWBOITResolveCompositeShaderPass()`. `WBOITRenderer.capture()` returns the accumulation and
  revealage bindings for inserting the WBOIT resolve into a larger shader-pass stack.

**@luma.gl/experimental/gpu-spatial-analysis**

The correctness pass changes several defaults to the statistically principled choice. Results differ
from earlier alphas even where the API is unchanged.

- Coordinates: every class that reads positions takes an optional `spatialContext`. Omitting it means
  planar. A longitude/latitude context throws at construction unless the class has a lon/lat path, so
  project to a planar frame first (`createGPULocalFrame`). The `coordinateSystem`, `radius`,
  `sphereRadius` and ring-assembly `geographic` props are removed from `GPULineSegmentize`,
  `GPULineChunk`, `GPUGreatCircleArcs`, `GPULinearReferencing`, `GPULineLocate`, `GPUShapeGenerator`,
  `GPUOutlineGeometry`, `GPULineLengthPerPolygon` and `GPUSegmentRingAssembly`; pass
  `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` instead.
  `GPUSegmentRingAssembly` is planar by default. `GPU_SHAPE_GENERATOR_EARTH_RADIUS` and
  `GPU_CELL_MEASURES_EARTH_RADIUS_KM` are removed; use the constants in `contracts/earth`.
  `GPUCellMeasures` reports meters and m² unless `units: 'kilometers'`.
- Inference: the default `alternative` is `'two-sided'` (was `'directed'`, which is anti-conservative).
  `falseDiscoveryRate` is a deprecated alias of `multipleTesting: 'benjamini-hochberg'`, and
  `getFalseDiscoveryRateNodes` is removed. The global permutation test draws with a new shuffle, so a
  given seed gives different draws.
- Spatial weights: rows without neighbors are excluded by default (`islands: 'exclude'`; pass
  `islands: 'keep'` for esda parity). Zero-weight slots are not neighbors. The weights summary
  `counts` output grows from 5 to 6 values and `globalStatistics` from 4 to 5. Hot spot `neighborCounts`
  no longer counts the focal row. `GPUSpatialWeightsTransform` with `'kernel'` requires `bandwidth`,
  and `'adaptive'` requires kNN weights. `GPUGlobalSpatialStatistics` accepts at most 2^24 rows.
- Recipes: `addHotSpotAnalysisRecipe` points sources need `studyArea` and `cellCapacity`;
  `neighborSearchParameters`, `gridSize` and `tableCapacity` are removed. It analyzes density per km²,
  requires 8 neighbors and applies Benjamini-Hochberg by default, as does `addRateClusterMapRecipe`.
  `addSpaceTimeHotSpotsRecipe` takes absolute `{step, interval, origin, count}` slices; cyclic fields
  and `GPUSpaceTimeSliceField` are removed.
- `GPUEmergingHotSpots` uses 95% confidence, Benjamini-Hochberg and a minimum of 10 trend slices by
  default, and adds category 17 (`UNDETERMINED`), so palettes need 18 entries.
- `GPUSpatialScanStatistic`: `GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES` is removed. Windows default to
  every zone, and time windows default to 50% of the period.
- `GPUEmpiricalBayesRates` clamps the prior variance by default (`clampPriorVariance: false` for esda
  parity). `GPULISAMarkov` is ungated by default and writes `0xffffffff` for rows outside the analysis.
  `GPULocalOutlierFactor` has no default threshold. Knox and Mantel counts change for asymmetric pair
  lists. Knox and Mantel permute with the swap-or-not shuffle, so a given seed gives different draws,
  and they accept at most 2^24 - 1 events.
- Point patterns and density: border-corrected Ripley's K is lower by a factor of (n−1)/n. Clark–Evans
  defaults to the Donnelly correction and a two-sided test, and its output and the quadrat output grow
  from 6 to 7 values. The IDW parameter length grows from 8 to 9.
- Kriging averages duplicate positions and writes NaN for ill-conditioned cells and materially
  negative variances. Intensive areal interpolation defaults to `denominator: 'overlap'`.
  `GPUPycnophylactic.iterations` is now a maximum. `GPUZonalStatistics` densities in lon/lat need
  caller-supplied `areas`.
- `GPUPointInPolygonJoin` defaults to `boundaryRule: 'half-open'`: a point on a shared edge belongs to
  exactly one polygon, and points on the outer right or top edge belong to none. Pass
  `boundaryRule: 'closed'` for the old inclusive behavior.
- Regression summaries grow and some slots change meaning: OLS 16 to 22 values, and
  `BREUSCH_PAGAN` is now the classic Breusch-Pagan test (Koenker-Bassett moved to slots 9/10 as
  `KOENKER_BASSETT`); regression diagnostics 12 to 13 (a failed OLS reports status 2, not 4); 2SLS 7 to 16;
  `GPUSpatialErrorGM` 6 to 13; GWR 6 to 12. Size summaries with the exported `*_LENGTH` constants.
- `GPUSpatialErrorGM` defaults to `estimator: 'homoskedastic'` (spreg `GM_Error_Hom`); pass
  `'kelejian-prucha'` for the previous `GM_Error`. GWR runs bandwidth refinement passes by default, so
  `selectedBandwidth` may lie between ladder entries, and its non-stationarity test draws with the
  swap-or-not shuffle.
- `GPUKMeans` initializes with `'kmeans++'` by default. `GPUMaxPRegions`, `GPUAZPRegions` and
  `GPUWardRegions` standardize attributes by default (`standardize: true`) and symmetrize the adjacency;
  the max-p status grows from 4 to 6 words and the partition evaluation summary from 8 to 9.
  `GPUAZPRegions` (and `GPUMaxPRegions` improvement) accept at most 32,768 rows (was 65,536).
  `GPUGeographicDistribution` stops the Weiszfeld median once every group moved at
  most `medianTolerance`, instead of always running `medianIterations` steps.
- `GPULocationAllocation` runs a Teitz-Bart swap phase by default (`maximumSwapIterations` 100), its status
  grows from 4 to 7 words, and unreachable demand is excluded from the objective and counted in
  `uncoveredDemandCount`, with `assignedCosts` of +Infinity.
- `GPUSegregation` writes NaN, not 0, for undefined indices. Relate outputs may carry
  `GPU_SPATIAL_RELATE_UNCERTAIN_BIT`. Rings that enclose a pole or span about 180° measure as NaN.
- `addDriveTimeCatchmentRecipe` with longitude/latitude positions snaps great-circle, so
  `maxSnapDistance` is in meters (was degrees). `GPUMapMatching` `breaks` can be 2 (a route-budget
  break), and `overflow` also reports route-table drops through the new `routeOverflow` output.

### Analysis contributors vocabulary

The analysis entries of `@luma.gl/experimental` (`gpu-spatial-analysis`, `gpu-network`, `gpu-terrain`,
`gpu-raster`, `gpu-dataframe`, `geospatial`) now share one vocabulary. Old names are removed, not aliased.
The rules below apply to every entry; the sub-headings after them list the renames per entry.

#### Common rules

- **Output bags.** Every caller-owned view a contributor writes is passed in one `output` prop
  (`GPU<Class>Output`) instead of loose props such as `weights`, `labels` or `result(s)`.
- **Status.** Every GPU-written status scalar (`count`, `requiredCount`, `overflow` and prefixed
  `*Overflow`, `converged`, `iterationCount`, `iterationLimitReached`, `invalidCount`, `uncertainCount`,
  `approximation` and the stage forms) lives in the `status` of the bag that owns it:
  `output.overflow` → `output.status.overflow`, `output.pairOverflow` → `output.status.pairOverflow`, a
  nested list's `output.pieces.requiredVertexCount` → `output.pieces.status.requiredVertexCount`, and a
  recipe's `result.weightsOverflow` → `result.status.weightsOverflow` (next to `result.status.stages`).
  Per-row columns use plural names (`count` → `counts`).
- **Compact lists.** `GPUCompactOutput` is `{ids, status: {count, overflow, requiredCount?}}`;
  `GPUCompactPairPort` is `{leftIds, rightIds, status}` and `GPUCellTablePort` is `{cells, counts?, status}`.
- **Parameters.** A packed per-frame view is `parameters` (`uint32Parameters` for a word view), written by
  `getGPU<Class>ParameterValues(settings)` from a CPU `GPU<Class>Settings` object (formerly
  `GPU<Class>Parameters`), with length `GPU_<CLASS>_PARAMETER_LENGTH`. Seeds are per frame.
- **Caps and counts.** `max*`/`min*` → `maximum*`/`minimum*`; loop caps are `maximumIterations`, executed
  counts `iterationCount`; buffer sizes are `<noun>Capacity`; per-frame permutation counts
  `permutationCount`. Coordinate components (`minX` ... `maxZ`) keep their names.
- **Inputs.** Polygons use the four-level `GPUPolygonGeometryPort` layout (`positions`, `featureOffsets`,
  `polygonOffsets`, `ringOffsets`; rows are features); lines are `positions` + `lineOffsets`, tracks
  `positions` + `trackOffsets`; raster and terrain contributors take `grid: GPURasterMetadata`; network
  contributors take `topology: GPUNetworkTopology`, graph starts `start*` and cost bands `costBreaks`.
- **Composition.** Every analysis contributor is added with `graph.add(contributor)` (`getCommandNodes`);
  its `addToGraph` is removed. The `gpu-raster` base operations (band math, morphology, contours, ...)
  keep their `addToGraph` API.

#### Status placement

These moves follow the status rule above. The full old → new path list is the rename map of this
change.

- Shared ports: `GPUCompactOutput`, `GPUCompactPairPort` and `GPUCellTablePort` (and `GPUCellTable`)
  `count`/`overflow`/`requiredCount`/`candidateOverflow` → `status.*`. Every compact list follows:
  geospatial point query, time-window filter, group statistics, key join `rows`, temporal reduction
  `occupiedSlots`, tile LOD and resident row selection, bounds filter, line simplification selection,
  trajectory stops, zone events and visit tables, encounter pairs, playhead `activeTracks`, dot density
  `dots`, random points `points`, join `matches`/`pairs`/`unmatched`, pair gather, spatial clustering
  `clusters`, MST `edges`, region statistics `selectedRows`, network path extraction, neighborhood,
  coarsening and subgraph filter lists, cost-distance `path`, streamlines `lines`, profile `peaks`,
  terrain `summits`.
- Model fits: `GPUSolverStatusPort.status` → `code`; OLS, 2SLS, GM, ML and spatial diagnostics
  `output.status` → `output.status.code`; ML `output.converged` → `output.status.converged`.
- Convergence and iteration words: `converged`/`iterationCount` of k-means, map coloring, pycnophylactic,
  coverage simplification, geodesic pairs and destination, network reachability, cost matrix, service
  areas and k-function → `output.status.*`; `GPUNetworkIsochronesProps.converged` →
  `output.status.converged` (new `GPUNetworkIsochronesOutput`); analytics columns
  `coreNumber`/`components`/`communities.converged` → `.status.converged`. `GPUNetworkPathExtractionProps.converged`
  (an input) → `predecessorsComplete`.
- Overflow, invalid and uncertainty words: areal interpolation (`overflow`, `invalidCount`,
  `requiredPairCount`), cell table compare, cell cover and compaction (`count`, `requiredCount`), cell level
  selection and line merge `count`, similar locations `count`, line density (`overflow`; `totalRecords` →
  `status.requiredCount`), line length per polygon, offset expansion, line geometry (`lineCount`), line split
  pieces (`vertexCount`, `requiredVertexCount` → `output.pieces.status.*`), ring assembly and coverage
  dissolve segment counts, isolines/isobands/isoband rings (`count`, `requiredCount`, `vertexCount`,
  `edgeCount`, `edgeTotalCount` → `status.edgeRequiredCount`), raster profile, patch labels, terrain
  contour levels (`segmentCount`/`requiredSegmentCount` → `status.count`/`status.requiredCount`),
  correlogram, variogram, spatial weights traces `approximation`, flow aggregation (`pairOverflow`), map
  matching (`routeOverflow`), snapping, cost weights, line graph, noding (`uncertainCount`, `nodes.count`)
  and induced topology → `status.*`.
- Recipes: result-level words move into `result.status` next to `stages` (hot spot `weightsOverflow`,
  `weightsRequiredCount`, `weightsInvalidCount`, `invalidCount`, `pointsOutsideStudyArea`; rate cluster;
  choropleth `overflow`/`uncertainCount`; change of support including `rasterOverflow`/`rasterRequiredCount`;
  cluster and outline; fleet dwell `joinOverflow`/`joinUncertainCount`; zone-events recipe `diagnostics` →
  `status`; straight-line catchments `overflow`; drive-time `serviceAreaConverged`); sub-results own a
  `status` (`permutation`, `table`, `stops`, `events`, `bands`, `isochroneDemand`, `snapping`, period
  comparison `before`/`after`/`comparison`); the `outputs` overrides mirror these paths (space-time
  `groupOverflow`, `groupRequiredCount` and `skippedEventCount` → `outputs.status.*`).
- Per-row columns renamed to plural: `GPUFocalStatisticsOutput.count`, `GPUInequalityOutput.count` and
  `GPURasterCellStatisticsOutput.count` → `counts`.
- Validation messages name the new paths (for example `output.status.count must contain one uint32 row`).
- Removed from the contracts exports (no consumer): `getGPUCapacityPlan`, `getGPUCapacityRecovery` and the
  cell, cell-cover, event, generated-geometry, line-topology, neighborhood, pair and trajectory-encounter
  planners with their option and result types; `GPUCapacityPlan` and `GPUContributorCapacityPlan` stay.
- Unchanged: the upstream `gpu-raster` base classes (connected, dense and cross-tile components, contours,
  region measurements, global statistics) keep their flat status props.

#### Layering and shared contracts

- `addDriveTimeCatchmentRecipe` and its types and constants moved from `gpu-spatial-analysis` to
  `@luma.gl/experimental/gpu-network`.
- `GPUTerrainCellSizeMode` → `GPURasterCellSizeMode` (exported from `gpu-raster` and `gpu-terrain`).
- `geospatial` `GPUSpatialQueryOutput` → `GPUCompactOutput` (same fields).
- `GPUIdentifierRows` → `GPUUint32Rows`, `GPUCoordinateRows` → `GPUFloat32Positions`.
- `GPUPointPatternAlternative` and `GPUPermutationAlternative` → `GPUInferenceAlternative`.
- Removed the alternative spelling `'less'`; use `'lesser'`.
- Removed `getScottBandwidth`, `getSilvermanBandwidth` and `GPUKernelDensityBandwidth`; use
  `GPUKernelDensityProps.bandwidthRule`.
- `addRateClusterMapRecipe` takes `GPUParameterBuffer` `parameters` and `permutation.parameters` with
  `settings` only. Removed `significanceLevel`, `permutation.permutations` and `permutation.seed`;
  `result.setSettings` is always present.
- Removed `GPUEdgeBundlingProps.geographic`; use
  `spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}`.
- Removed `falseDiscoveryRate` from `GPULocalPermutationTest`, `GPULocalMoran` and `GPUHotSpotAnalysis`,
  and `permutation.falseDiscoveryRate` from `addHotSpotAnalysisRecipe`; use
  `multipleTesting: 'benjamini-hochberg'`.
- The `coordinateSpace` and `sphereRadius` instance fields of `GPULinearReferencing`, `GPULineLocate` and
  `GPUShapeGenerator` are private; read `spatialContext` and `getGPUSphereRadius(spatialContext)`.
- Longitude/latitude with a metric other than `'great-circle'` on great-circle contributors now fails with
  "positions need the great-circle metric". `GPUNeighborSearch` uses an explicit `sphereRadius` as given in
  the result units with `units: 'kilometers'` (it was divided by 1000).

#### Dataframe and geospatial

- The geospatial classes (`GPUSinusoidalProjection`, `GPUHaversineDistance`, `GPUPairwisePointDistance`,
  `GPUPairwisePointSegmentDistance`, `GPUPairwisePointInPolygon`, `GPUPairwisePointLinestringNearest`,
  `GPUPointSpatialQuery`) no longer have `addToGraph`; call `graph.add(contributor)`.
- `GPUHaversineDistance` `radius` → `sphereRadius` (prop and instance field).
- `GPUPairwisePointSegmentDistance` `points` → `positions`.
- `GPUPairwisePointInPolygon` `points` → `queryPositions`, `polygonPositions` → `positions`,
  `geometryOffsets` → `featureOffsets` (the `GPUPolygonGeometryPort` names).
- `GPUPairwisePointLinestringNearest` `points` → `queryPositions`, `linestringPositions` → `positions`,
  `geometryOffsets` → `featureOffsets`, `linestringOffsets` → `lineOffsets`. Its outputs move into
  `output: {distances, nearestPositions?, lineIndices?, segmentIndices?}` (was `output`, `nearestPoints`,
  `linestringIndices`, `segmentIndices`).
- `GPUPointSpatialQuery` `query` → `parameters`; `intersectedCellCount` and `candidateCount` move into
  `output` (`GPUPointSpatialQueryOutput`).
- `GPURipleyParameters`, `GPURipleyDistanceParameters`, `GPUPointPatternIndicesParameters`,
  `GPUSpatialCorrelogramParameters`, `GPUVariogramParameters`, `GPUClassBreaksParameters`,
  `GPUColorScaleParameterOptions`, `GPUBivariateClassificationParameterOptions` and
  `GPUColumnQuantilesParameterInput` → `GPU<Class>Settings` (`GPURipleyDistanceFunctionsSettings` for the
  distance functions).
- `getGPURipleyDistanceParameterValues` and `GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH` →
  `getGPURipleyDistanceFunctionsParameterValues` and `GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH`.
- `GPURipley`, `GPURipleyDistanceFunctions`, `GPURipleyEnvelope`, `GPUPointPatternIndices`,
  `GPUSpatialCorrelogram` and `GPUVariogram` take their output views (`k`, `g`, `lower`, `clarkEvans`,
  `moransI`, `semivariances`, `statistics`, `overflow`, ...) in one `output` prop (`GPU<Class>Output`).
- The Ripley `seed` is any uint32, stored as uint32 bits in slot `GPU_RIPLEY_SEED_SLOT`;
  `GPU_RIPLEY_MAXIMUM_SEED` (2^24 - 1) is removed. Envelopes for a given seed differ from before.
- `GPUColumnQuantilesOutput.filterMask` → `outputMask`; `GPUColumnProfileDomain` `{lo, hi}` →
  `{minimum, maximum}`.
- `GPUTimeWindow` → `GPUTimeWindowFilterSettings`, `GPUTimeWordWindow` → `GPUTimeWindowFilterWordSettings`,
  `getGPUTimeWindowParameterValues` → `getGPUTimeWindowFilterParameterValues`,
  `getGPUTimeWindowWordParameterValues` → `getGPUTimeWindowFilterWordParameterValues`,
  `GPU_TIME_WINDOW_PARAMETER_LENGTH` → `GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH` and
  `GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH` → `GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH`.
- `GPUTimeWindowFilter` `outputMask`, `fadeWeights`, `clipFractions`, `trackVisibleCounts` and
  `drawInstanceCount` move into `output` (`GPUTimeWindowFilterOutput`, the compact fields plus these).
- `GPUTemporalReductionOutput` `min`/`max` → `minimums`/`maximums`.
- Philox purpose IDs come from one registry: `GPUParticleAdvection`, `GPUDotDensity` and `GPUMaxPRegions`
  draw new random streams for the same seed (they shared IDs with other features).

#### Clustering, allocation, regionalization and accessibility

- `GPUGeographicDistributionParameters` → `GPUGeographicDistributionSettings`;
  `GPUGeographicDistributionProps.medianIterations` → `maximumMedianIterations` (also the instance field).
- `GPURoseStatistic` `summary` and `bins` move into `output: GPURoseStatisticOutput`.
- `GPU_LOCATION_ALLOCATION_STATUS.swapIterations` → `swapIterationCount`;
  `GPU_LOCATION_ALLOCATION_DEFAULT_SWAP_ITERATIONS` → `GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS`.
- `GPUSimilarLocations`: `attributes` → `values`, `attributeCount` → `columnCount`, `selection` →
  `referenceMask`, `maximumResultCount` → `resultCapacity` (also the instance field);
  `getGPUSimilarLocationsParameterValues` takes `columnCount`.
- `GPUSegregation`: `unitCount` is removed (derived from `groupCounts.length / groupCount`); `atkinsonB` →
  `atkinsonInequalityAversion`; `indices` and `local` merge into `output: GPUSegregationOutput` with
  `indices`, `localEnvironment`, `localEntropy`, `localDissimilarity` and `localTheil`
  (`GPUSegregationLocalOutputs` is removed).
- `GPUCatchmentAccessibility`: `accessibility`, `ratios` and `reachableFacilities` moved into
  `output: GPUCatchmentAccessibilityOutput`.
- `GPUHuffTradeAreas`: `tradeArea`, `probabilities`, `tradeAreaProbability` and `expectedDemand` moved into
  `output: GPUHuffTradeAreasOutput`; pack `parameters` with `getGPUHuffTradeAreasParameterValues`.
- `GPUArealInterpolation`: `sourceCount`/`targetCount` → `sourceZoneCount`/`targetZoneCount`; `weights`,
  `alternateWeights`, `areas`, `sourceAreas` and `sourceTransferredFractions` moved into
  `output: GPUArealInterpolationOutput`, `overflow` and `invalidCount` into `output.status`, `totalPairs` →
  `output.status.requiredPairCount` and `categories.output` → `output.categoryShares`.
- `GPUPycnophylactic`: `iterations` → `maximumIterations`; the `output` view → `output.mass`, and
  `density` moved into `output` and `converged` and `invalidCount` into `output.status`.
  `iterationLimitReached` is replaced by `output.status.iterationCount` (iterations that ran).
- `GPUKMeans`: `iterations` → `maximumIterations`; `seed` and `tolerance` moved to a per-frame `uint32`
  `parameters` view (`getGPUKMeansParameterValues`, `GPU_KMEANS_PARAMETER_LENGTH`); outputs moved into
  `output`, and the two-word `convergence` view is replaced by one-row `output.status.iterationCount` and
  `output.status.converged`.
- `GPUSpatialClustering`: every output view moved into `output`; `GPUSpatialClusteringParameters` →
  `GPUSpatialClusteringSettings`.
- `GPUMapColoring`: `seed` moved to a per-frame `uint32` `parameters` view
  (`getGPUMapColoringParameterValues`); `maximumRounds` → `maximumIterations`, `roundCount` →
  `output.status.iterationCount`, `GPU_MAP_COLORING_DEFAULT_MAXIMUM_ROUNDS` →
  `GPU_MAP_COLORING_DEFAULT_MAXIMUM_ITERATIONS`; outputs moved into `output`.
- `GPULocalOutlierFactor`: outputs moved into `output`; `GPULocalOutlierFactorParameters` →
  `GPULocalOutlierFactorSettings`.
- `GPUAZPRegions`, `GPUWardRegions`, `GPUMaxPRegions`: `seed` moved to a per-frame `uint32` `parameters`
  view (`getGPU{AZP,Ward,MaxP}RegionsParameterValues`, `GPU_{AZP,WARD,MAX_P}_REGIONS_PARAMETER_LENGTH`);
  AZP and Ward `labels` + `optimization` → `output: {labels, objective, status}`.
- `GPUMaxPRegions`: `improvementIterations` → `maximumIterations` (prop and instance field).
- `GPUSkaterRegions`: `labels`, `regionCount`, `cutEdges`, `cutGains` moved into `output`;
  `GPU_SKATER_PARAMETER_LENGTH` → `GPU_SKATER_REGIONS_PARAMETER_LENGTH`, and the
  `GPU_SKATER_PARAMETER_REGION_COUNT`/`_MINIMUM_SIZE` indices are replaced by the packer
  `getGPUSkaterRegionsParameterValues({regionCount, minimumRegionSize})`.
- `GPUSpatialWeightsMinimumSpanningTree`: every output moved into `output`; `edgeCosts` →
  `output.edgeDissimilarities`.
- `GPURegionPartitionEvaluation`: `summary`, `regionSizes` and `regionWithinSsd` →
  `output.{summary, regionSizes, regionWithinSumOfSquares}`; layout keys `withinSsd`, `betweenSsd`,
  `totalSsd` → `withinSumOfSquares`, `betweenSumOfSquares`, `totalSumOfSquares`.
- `GPUNetworkAccessibility` `catchment` now runs `GPUNetworkCostWeights` plus `GPUCatchmentAccessibility`
  (new `catchment.method`, `'3sfca'` available) and allocates two transient weights of
  `rowCount x nodeCount` slots. `GPUCatchmentAccessibility` skips a non-finite demand or ratio, as the
  network path did.
- `GPUCatchmentAccessibility` `'3sfca'` now applies the decay with the selection weight (`G_ij W_ij`, Wan
  et al. 2012 and PySAL `access`); results with a non-binary decay change (binary weights are unchanged).

#### Statistics, weights and recipes

- `GPUSpatialWeightsInfo` is now `GPUSpatialWeightsInformation`, and the `GPUSpatialWeights` field `info` is `information` (also in `describeGPUSpatialWeights(weights, {information, status})`).
- `GPUContiguityWeights`, `GPULatticeWeights` and `GPUSpatialWeightsAlgebra` write one `output: GPUSpatialWeightsOutput` (CSR views plus `status: {overflow, requiredCount?, candidateOverflow?}`); the `weights` output prop (contiguity, lattice), `overflow` and `totalNeighbors` props are removed.
- `GPUContiguityWeights` takes `polygons: GPUPolygonGeometryPort` (four-level layout with `featureOffsets`) instead of `positions`/`ringOffsets`/`polygonOffsets`, and its rows are features, not polygon parts; `getPolygonCount()` is `getFeatureCount()`.
- `getGPUSpatialWeightsCapacity(requiredCount, capacity, headroom)` is replaced by `getGPUSpatialWeightsCapacityPlan({rowCount, capacity, requiredCount, distances?, headroom?}).capacity`; contiguity and lattice add `getCapacityPlan(observation)`.
- `GPUSpatialWeightsSummary` takes `output: {statistics, counts, cardinality?}`; `GPUSpatialWeightsValidation` takes `output: {flags?, status: {invalidCount}}`; `GPUSpatialWeightsTranspose` takes `asymmetricSlots` inside `output`.
- `GPUSpatialLag` has no `sourceCount`: the value row count is `values.length / columnCount`; `values` may be `uint32`.
- `GPUSpatialAutocorrelationParameters` is `GPUSpatialAutocorrelationSettings` and `GPUPermutationParameters` is `GPUPermutationSettings`; its `permutations` field is `permutationCount`.
- `GPU_GLOBAL_PERMUTATION_RESULT` is `GPU_GLOBAL_PERMUTATION_SUMMARY` (field `permutations` is `permutationCount`), the layout of `GPUGlobalPermutationTest` `output.summary` (was `results`).
- `GPULocalMoran`, `GPUHotSpotAnalysis`, `GPULocalPermutationTest`, `GPUGlobalPermutationTest`, `GPUMultipleTestingAdjustment` and `GPUAssuncaoReisStandardization` take their result views in one `output` bag; the local test's `overflow` is `output.status.overflow` and Assuncao-Reis `invalidCount` is `output.status.invalidCount`.
- `GPULocalGeary`, `GPUSpatialPearson` and `GPUGammaStatistic` take `output: {statistic}` (Geary also `output.globalStatistics`) instead of `result`.
- `GPULocalMoran.getInferenceMetadata` and `GPUHotSpotAnalysis.getInferenceMetadata` take the `GPUSpatialAutocorrelationSettings` that are packed (was a bare significance level / nothing).
- `GPUSpatialScanStatistic`: outputs move into `output: {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics?}`; `GPU_SCAN_STATISTIC_*` constants are now `GPU_SPATIAL_SCAN_STATISTIC_*`; `GPUSpatialScanStatisticParameters` is `GPUSpatialScanStatisticSettings` (`permutations` → `permutationCount`, also the summary field), packed by `getGPUSpatialScanStatisticParameterValues`; `getGPUSpatialScanLayout`/`GPUSpatialScanLayout` are `getGPUSpatialScanStatisticLayout`/`GPUSpatialScanStatisticLayout`.
- `GPUKnoxTest`, `GPUMantelTest`: `statistics`/`summary` move into `output`; `GPUSpaceTimeParameters` is `GPUSpaceTimeSettings` with `permutationCount`; `getGPUSpaceTimeInferenceMetadata` is removed (use `test.getInferenceMetadata(settings)`).
- `getInferenceMetadata(settings)` on the scan, Knox and Mantel tests takes the packed settings object (required) instead of optional partial parameters.
- `GPUGlobalSpatialStatistics`: `results` is `output.summary` and `joinCounts` is `output.joinCounts`.
- `GPUNeighborhoodSummary`: `output` (the table) is `output.table`; `modes`, `entropy` move into `output`; `overflow` is `output.status.overflow`.
- `GPUEmpiricalBayesRates` and `GPUSpatialEmpiricalBayesRates` outputs (`standardizedRates`, `smoothedRates`, `rawRates`, `summary`; `spatialRates`, `priorMeans`, `priorVariances`) move into `output`; `events` and `populations` also accept `uint32`.
- `GPUEmergingHotSpots` outputs (`giZScores`, `trendZ`, `trendP`, `trendS`, `category`, `hotSliceCount`, `coldSliceCount`, `pValues`, `adjustedPValues`, `lastValidSlice`, `globalStatistics`) move into `output`.
- `GPUEmergingHotSpotParameters` is `GPUEmergingHotSpotsSettings`, `getGPUEmergingHotSpotParameterValues` is `getGPUEmergingHotSpotsParameterValues`, and every `GPU_EMERGING_HOT_SPOT_*` constant is `GPU_EMERGING_HOT_SPOTS_*`.
- `GPUEmergingHotSpots.getInferenceMetadata(settings)` takes the packed settings and reports `significanceLevel`; pack with `getParameterValues(settings)`, which takes `multipleTesting` from the node.
- `GPUTransitionMatrix` drops `rows` (derived from `classes.length / periodCount`); `periods` is `periodCount` on `GPUTransitionMatrix`, `GPUSpatialMarkov` and `GPULISAMarkov`; `GPUSpatialMarkov` `lagMaximumClassCount` is `maximumLagClassCount`; the `rows` fields are `rowCount`.
- `GPULISAMarkov.getInferenceMetadata(settings)` takes the `GPUSpatialAutocorrelationSettings` packed into `parameters` instead of a significance level.
- `GPUSpatialWeightsTraces` `seed` is removed: pass an optional `parameters` uint32 view written with `getGPUSpatialWeightsTracesParameterValues({seed})` (`GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` words); the seed is now per frame.
- `GPUSpatialLagMaximumLikelihood`, `GPUSpatialErrorMaximumLikelihood` and `GPUSpatialErrorGM` `traceSeed` is replaced by `traceParameters` (same per-frame layout).
- `getGPUOrdinaryLeastSquaresParameterValues(ridgeLambda)` now takes `{ridgeLambda}` (`GPUOrdinaryLeastSquaresSettings`).
- `GPU_GWR_NONSTATIONARITY_SUMMARY.permutations` is renamed `permutationCount`.
- The OLS, GM, 2SLS, spatial diagnostics and spatial maximum-likelihood `output.status` is a bag: the solver code view `output.status` → `output.status.code` (`GPUSolverStatusPort.status` → `code`), and the maximum-likelihood `output.converged` → `output.status.converged`. The spatial regression recipe's `result.ols.status` and `result.diagnostics.status` → `.status.code`.
- `getSpatialWeightsTracesProbeWGSL(stochastic)` no longer takes a seed; its `getProbeValue(seed, probe, row)` reads the seed at run time.
- `GPUZonalStatistics`: polygon `features` use the `GPUPolygonGeometryPort` layout (`polygonPositions` is now `positions`; rows are features); `points` is now `positions`; `includeBoundary` is removed (use `boundaryRule`).
- `GPUZonalStatistics`: `output.overflow` and `output.uncertainCount` moved to `output.status`.
- `GPURegionStatistics`: `selection` is now `region`; `summary`, `outputMask`, `output` and `drawInstanceCount` moved into `output: {summary, mask, selectedRows, drawInstanceCount}`.
- `GPURegionMask` and `GPUPickRegionMask`: `outputMask` and `overflow` are now `output: {mask, status: {overflow}}`; the pick result prop `result` (also on `{kind: 'pick-region'}` regions) is now `picks`.
- `GPUKnoxTest`, `GPUMantelTest` and `GPUGeographicallyWeightedRegressionNonstationarityTest` draw their permutations from their own Philox purposes, so their permutation results for a given seed differ from before (no longer shared with `GPUGlobalPermutationTest`).
- `addPointsInPolygonsChoroplethRecipe`: `points` is a `GPUPointGeometryPort`; `polygons` is a `GPUPolygonGeometryPort` (`polygonPositions` → `positions`, `kind: 'polygons'`); `includeBoundary` is replaced by `boundaryRule` (default `'half-open'`).
- `addRateClusterMapRecipe`: `positions`/`ringOffsets`/`polygonOffsets` are replaced by `polygons: GPUPolygonGeometryPort`; rows are features (multipart features are one row). `settings.permutations` is `permutationCount`.
- `addHotSpotAnalysisRecipe`: `parameters` and `permutation.parameters` are `GPUParameterBuffer`s filled from the new `settings`; `studyArea` is a `GPUPolygonGeometryPort` (`polygonPositions` → `positions`); `result.setSettings` repacks; `weightsRequiredCount` is always present.
- `addSpaceTimeHotSpotsRecipe`: takes `settings` (required) and a `GPUParameterBuffer` `parameters`; `result.setSettings` repacks.
- `addSpatialRegressionRecipe`: the compile-time `significanceLevel` is removed; pass `settings.significanceLevel` and a `GPUParameterBuffer` `parameters` (shared by the specification search and the residual local Moran).
- `addStraightLineCatchmentsRecipe`: `width`/`height` → `grid`, `settings` → `parameters`, `seedPositions`/`seedIds`/`seedCount` → `facilities` (point port, `sourceIds`) and `facilityCount`.
- `addClusterAndOutlineRecipe`: `positions` → `points` (point port).
- `addFleetDwellRecipe` / `addFleetDwellZoneEventsRecipe`: `positions`/`timestamps`/`timestampsLow`/`trackOffsets` → `trajectories`; zones are polygon ports (`featureIds` → `sourceIds`); the zone-events variant takes polygon `zones` instead of `edgeStarts`/`edgeEnds`/`edgeZones`/`zoneCount`, and `maxEventsPerTrack` is `trackEventCapacity`.
- `addChangeOfSupportRecipe`: `width`/`height`/`extent` → `grid` + `parameters`; zones are polygon ports; `categories.output` → `outputs.categoryShares`; result `sourceCount`/`targetCount`/`totalPairs` → `sourceZoneCount`/`targetZoneCount`/`requiredCount`.
- Corrected results (not renames): constant values now give NaN Moran's I and Geary's C in `GPUGlobalSpatialStatistics` (was 1 and 0) and NaN `GPULocalMoran` `localI` (was 0); `E[I]` and `E[C]` are NaN below two rows (was 1).

#### Terrain and raster

- Every `gpu-terrain` contributor and the `gpu-raster` analysis contributors (cost distance, distance
  field, zonal statistics, polygon rasterization, raster join, sampling, profile, stretch, isolines,
  isobands, isoband rings, line integral convolution, extrema pyramid, patch metrics, sieve) take
  `grid: GPURasterMetadata` instead of `width`/`height` (`GPUTerrainContours.metadata` becomes the
  required `grid`). New helpers `makeGPURasterMetadata({width, height, cellSize, origin, rowDirection})`
  and `makeGPURasterBand(view, grid)` live in `gpu-raster`.
- Cell size, latitude band, extent and Web Mercator pixel size are read from the grid: `cellSize`,
  `northEdge`, `southEdge`, `origin`, `bounds`, `gridSize`, `extent`, `width`/`height` and
  `worldPixelSize`/`originY` are removed from the settings types, and every packer whose layout holds
  them takes the frame's grid as its second argument, `getGPU<Class>ParameterValues(settings, grid,
  target?)`. Web Mercator row edges are EPSG:3857 meters (they were normalized y).
- `settings` → `parameters` and `wordParameters` → `uint32Parameters` (packers
  `getGPU<Class>Uint32ParameterValues`, constants `GPU_<CLASS>_UINT32_PARAMETER_LENGTH`).
- Outputs move into `output: GPU<Class>Output`; the validity output is `output.outputValidity` (an input
  mask is `validity`, for example `GPUTerrainRGBDecode.inputValidity` → `validity`), and one-row status
  words move into `output.status` (`converged`, `iterationCount`, `overflow`, `iterationLimitReached`,
  phase forms such as `fillConverged`/`fillIterationCount`). Compact lists are named after their noun
  (`GPUTerrainSummits` `output` → `output.summits`, `outputDrop` → `output.summitDrop`, `overflow` →
  `output.status.radiusClamped`; `GPUProfilePeaks` `output` → `output.peaks`; `GPUTerrainPeakSnap`
  `status` → `output.snapStatus`; `GPUCostDistancePath` `output` → `output.path`).
- `max*`/`min*` → `maximum*`/`minimum*` in props, settings and constants (`maxFillIterations` →
  `maximumFillIterations`, `maxIterations` → `maximumIterations`, `maxDistance` → `maximumDistance`,
  `minProminence` → `minimumProminence`, `*_MAX_*` → `*_MAXIMUM_*`); `GPUTerrainSpikeRepair`
  `componentIterations` → `maximumComponentIterations`; `GPUStreamlines` `roundCount` →
  `maximumIterations`; `GPUDistanceField` `jumpFloodRefinementPasses` →
  `jumpFloodRefinementIterationCount`; `GPUProfilePeaks` `minSide`/`nms`/`nmsRounds` →
  `minimumSide`/`suppressionRadius`/`maximumSuppressionIterations`.
- `GPUSimpleLocalRelief.radius` and `GPUTerrainVectorRuggedness.radius` → compile-time `maximumRadius`;
  the radius used is the per-frame `settings.radius` (default `maximumRadius`).
- Packer names follow their class: `getGPUTerrainSightLineParameterValues` →
  `getGPUTerrainLineOfSightParameterValues`, `getGPUPointHorizonParameterValues` →
  `getGPUPointHorizonProfileParameterValues`, `getGPUTerrainVisibilityToleranceParameterValues` →
  `getGPUTerrainViewshedToleranceParameterValues` (prop `tolerance` → `toleranceParameters`),
  `getGPUPolygonRasterizationExtentValues` → `getGPUPolygonRasterizationParameterValues(grid)`,
  `GPUChangeDetectionParameters`/`GPURasterSieveParameters` → `*Settings`. The deprecated row-direction
  overload of `getGPUTerrainCastShadowParameterValues` is removed.
- Line and polygon names: `GPURasterProfile` `pathPositions`/`pathOffsets` → `positions`/`lineOffsets`,
  `GPUStreamlines` `output.pathOffsets`/`points`/`pointCount` → `lineOffsets`/`positions`/`positionCount`,
  `GPUPolygonRasterization` `polygonPositions` → `positions`; `GPUCostDistance` `bandThresholds` →
  `costBreaks`. `GPUTerrainRowDirection` → `GPURasterRowDirection`.
- Polygon rasterization, raster join, sampling, profile and distance field now accept north-up grids
  (negative row step); north-down grids give the same results as before.
- `GPURasterProfile` outputs and field use the line vocabulary: `samplePathIds` → `sampleLineIds`,
  `pathSampleOffsets` → `lineSampleOffsets`, `pathLength` → `lineLengths`, `pathGain` → `lineGains`,
  `pathLoss` → `lineLosses`, `pathMinimum` → `lineMinimums`, `pathMaximum` → `lineMaximums`,
  `GPU_RASTER_PROFILE_NO_PATH_ID` → `GPU_RASTER_PROFILE_NO_LINE_ID`, instance field `pathCount` → `lineCount`.

#### Raster grids in spatial analysis

- `GPUKernelDensity`, `GPUInverseDistanceWeighting`, `GPUKriging` and `GPUFocalStatistics` take
  `grid: GPURasterMetadata` instead of `width` and `height`. The first three need an axis-aligned
  grid; focal statistics works on cells only and accepts any grid.
- `extent` is removed from `GPUKernelDensitySettings`, `GPUInverseDistanceWeightingSettings` and
  `GPUKrigingSettings`; their packers take the frame's grid, `getGPU<Class>ParameterValues(settings,
  grid, target?)`. Parameter slots 0 to 3 are now the grid's outer corners `[startX, startY, endX,
  endY]` in grid order.
  - Output rows follow the grid's row order. A grid whose rows advance north from `[minX, minY]`
    (`makeGPURasterMetadata({..., origin: [minX, minY], rowDirection: 'north'})`) packs the previous
    extent, so the raster is unchanged, with row 0 at `minY`; the default north-up grid puts row 0 at the top.
  - The `GPUKernelDensity` window defaults to the grid's ordered rectangle.
- `GPUArealInterpolation` and `GPUPycnophylactic` take `grid: GPURasterMetadata` instead of `width`
  (and `height`). The zone rasters and `cellWeights` hold exactly `grid.width * grid.height` rows; a
  planar raster needs the grid too.
- `GPUArealInterpolation.extent` and `GPUPycnophylactic.extent` are renamed `parameters`: the
  per-frame `getGPUPolygonRasterizationParameterValues(grid)` view, the same view as
  `GPUPolygonRasterization.parameters`.
  - A north-up grid packs a negative cell height. Cell areas now follow the mirrored rows; before,
    a north-up longitude/latitude raster had non-positive areas and every weight was 0. Rasters
    stored with rows advancing north are unchanged.

#### Geometry, topology, joins, cells and movement

- Removed `GPUSpatialJoinPoints`, `GPUSpatialJoinLines`, `GPUSpatialJoinPolygons`, `GPUNearestFeaturePoints` and
  `GPUSpatialJoinPairs`; use `GPUPackedGeometryPort<GPUPointGeometryPort | GPULineGeometryPort |
  GPUPolygonGeometryPort>` and `GPUCompactPairPort`.
- `GPUPointInPolygonJoin`: `points` + `sourceIds` → `points: GPUPointGeometryPort`; `polygonPositions`,
  `featureOffsets`, `polygonOffsets`, `ringOffsets`, `featureIds` → `polygons` (a polygon port; `featureIds` is
  `polygons.sourceIds`); outputs → `output: {pointFeatureIds, featureCounts, matches, status: {overflow,
  candidateCount, uncertainCount}}`. Removed `includeBoundary`; use `boundaryRule: 'closed' | 'open'`.
- `GPUNearestFeatureJoin`: `points`, `queries` and `sourceIds` → `queries` (a point port, possibly chunked, or a
  line/polygon port); `featureIds` → `features.sourceIds`; removed `maxDistance` (use `radius`); outputs →
  `output: {..., status: {overflow, candidateCount}}`.
- `GPUSpatialPredicateJoin`: `pairs`, `relate`, `weights`, `unmatched` → `output.*`; `overflow`,
  `candidateCount`, `uncertainCount` → `output.status.*`. `candidateCapacity` is optional (plan estimate).
- `GPUSpatialJoinCandidates`: `pairs` → `output`. `GPUBufferSelection`: `points` + `sourceIds` → `points` (a point
  port); `output` → `output.matches`; `outputMask`, `drawInstanceCount`, `distances`, `nearestFeatureIds` →
  `output.*`; `overflow` → `output.status.overflow`. `GPUNearestFeatureWeights`: `weights` + `overflow` →
  `output: GPUSpatialWeightsOutput`.
- `getGPUSpatialJoinCapacityPlan` moved next to the join and now counts the join's real allocation (BVH about
  52 bytes per leaf, probe, dispatch and robust rows); use `GPUSpatialPredicateJoin.getCapacityPlan()`.
- `GPUTrajectoryTimestampFormat` is `'float32' | 'uint32x2'` (Int64 words; `'uint32'` was accepted by no
  contributor), and `GPUPackedGeometryPort` of a trajectory port packs `timestamps` too.
- Removed `GPUOptimizationStatusPort` (no consumer). `GPUPartitionedGeometryPort`, `GPUPreparedIndexDescriptor`,
  `getGPUSpatialQueryCostPlan` (and its types), `assertCompatibleGPUPreparedIndexes`, `isGPUSeamResultOwner`
  and `getGPUTileSeamOwner` are no longer exported (no consumer).
- Polygon inputs of `GPUGeometryMeasures`, `GPUShapeDescriptors`, `GPULabelPoint`, `GPUMinimumBounds`,
  `GPUAffineTransform`, `GPUGeometryOrientation` (`'orient-polygons'`), `GPUGeometryCleanup` and
  `GPUPolygonTriangulation` take the four-level layout (`positions`, `featureOffsets`, `polygonOffsets`,
  `ringOffsets`; rows are features). `featureRingOffsets` → `featureOffsets` (feature → polygons), with
  `polygonOffsets` (polygon → rings) now required. A feature that used to own several rings as parts of a
  multipolygon needs one polygon per part.
- Removed `holeRule` (`GPUGeometryMeasures`, `GPUShapeDescriptors`, `GPUAffineTransform`), the
  `GPUGeometryHoleRule` type and the `holeRule` instance fields. The first ring of every polygon is its shell and
  later rings are holes, whatever the winding, so `'winding'` results change for multipolygons and for holes
  wound like their shell, and `areas` is never negative.
- `GPUGeometryMeasuresProps` is a union by `geometryType`: `'lines'` takes `lineOffsets` (was `ringOffsets`) and
  optional `featureOffsets` (multi-lines, was `featureRingOffsets`); `'points'` takes `positions` and optional
  `featureOffsets` (multi-points). `status.invalidCount` → `output.status.invalidCount`.
- `GPUMinimumBounds` geometry input is polygons only; use `labels` and `groupCount` for lines and points.
  `output.overflow` → `output.status.overflow`.
- `GPUAffineTransform`, `GPUGeometryOrientation` (`'reverse'`) and `GPUGeometryCleanup` take `lineOffsets` for lines
  (was `ringOffsets`); `GPUAffineTransform` points use `featureOffsets`. Polygon mode of orientation and cleanup
  needs `polygonOffsets` and `featureOffsets`; the output keeps the input offsets.
- `GPUGeometryCleanup` output: `count`, `overflow`, `requiredCount` → `output.status.*`; `collapsedRings` →
  `output.status.collapsedCount`; line output `ringOffsets` → `lineOffsets`. `GPUGeometryCleanupOutput` is now the
  union of `GPUGeometryCleanupPointOutput`, `GPUGeometryCleanupLineOutput` and `GPUGeometryCleanupPolygonOutput`.
- `GPUPolygonTriangulation`: `polygons: {...}` → flat `positions`, `featureOffsets`, `polygonOffsets`,
  `ringOffsets`; `indices`, `valid`, `triangleCount` → `output.*` (still one row per polygon). Removed
  `GPUPolygonTriangulationPolygons`.
- `GPUGeometryValidity`: `mask` → `output.mask`; `overflow`, `intersectionCount` → `output.status.overflow`,
  `output.status.intersectionCount`.
- `GPUGeometryPredicates`: every output column (`isSimple` … `uniqueOffsets`) → `output.*`; `overflow`,
  `intersectionCount` → `output.status.*`.
- `GPUMinimumClearance`: `clearances`, `lines`, `vertexIds` → `output.*`.
- `GPUGroupGeometry` and `GPUGroupConvexHull`: `output.overflow` → `output.status.overflow`.
  `GPUDelaunayTessellation`: `output.duplicateCount` → `output.status.duplicateCount`.
- `GPUGeodesicPairs` and `GPUGeodesicDestination`: `iterations` → `maximumIterations` (a cap, in props and
  instance fields).
- `GPULocalFrameProjection.addToGraph(graph)` → `graph.add(projection)` (`getCommandNodes`).
- Settings types and packers: `GPUShapeDescriptorsParameters` → `GPUShapeDescriptorsSettings`,
  `GPUAffineTransformParameters` → `GPUAffineTransformSettings`, `GPUGeometryOrientationParameters` →
  `GPUGeometryOrientationSettings`, `GPUGeometryCleanupParameters` → `GPUGeometryCleanupSettings`,
  `GPUGeometryPredicatesParameters` → `GPUGeometryPredicatesSettings`, `getGPUAffineTransformParameters` →
  `getGPUAffineTransformParameterValues`.
- Lines: `pathOffsets` → `lineOffsets` on `GPULineDensity`, `GPULineLengthPerPolygon`, `GPULineSegmentize`,
  `GPULineChunk`, `GPULineSmooth`, `GPULinearReferencing`, `GPULineLocate`, `GPUOffsetCurve`,
  `GPUOutlineGeometry` and `GPURectangleClip`; `GPULineSimplification` `trackOffsets` → `lineOffsets`.
- `GPURectangleClip`, `GPUOutlineGeometry` and `GPUOffsetCurve`: `geometryType: 'polygons'` (`'rings'` in the
  outline and offset curve) takes the four-level layout `positions`, `featureOffsets`, `polygonOffsets`,
  `ringOffsets` instead of one-level ring offsets. Ring counts are preserved, so the input `featureOffsets` and
  `polygonOffsets` describe the output rings. `GPUOffsetCurve` `geometryType` `'rings'` → `'polygons'` and
  `GPUOutlineGeometry` `'rings'` → `'polygons'`; the props are a discriminated union on `geometryType`.
- `GPULinearReferencing`: `points` → `queryPositions`; `overflow` and `candidateCount` → `output.status.overflow`
  and `output.status.candidateCount`.
- `GPULineLocate`: `vertexMeasures` → `output.vertexMeasures`.
- `GPULineSmooth`: `iterations` → `iterationCount` (an exact pass count, not a cap).
- `GPULineClipByPolygon`: `pieces` → `output.pieces`, `uncertainCount` → `output.status.uncertainCount`.
- `GPUSharedPaths`: `runs` → `output.runs`; `runs.count`, `vertexCount`, `overflow`, `requiredCount`,
  `requiredVertexCount` and `uncertainCount` → `output.status.*`.
- `GPULineSimplification`: `selection` and `status` → one `output` (`ids`, `outputMask`, `lineCounts`,
  `lineStarts`, `status: {count, overflow, requiredCount, converged, iterationCount}`);
  `selection.keepMask` → `output.outputMask`; `maximumRounds` → `maximumIterations`; status `roundCount` →
  `iterationCount`; `GPULineSimplificationSelection` → `GPULineSimplificationSelectionOutput`;
  `GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ROUNDS`, `GPU_LINE_SIMPLIFICATION_MAXIMUM_ROUNDS` and
  `GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_ROUNDS` →
  `GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ITERATIONS`, `GPU_LINE_SIMPLIFICATION_MAXIMUM_ITERATIONS` and
  `GPU_LINE_SIMPLIFICATION_DEFAULT_VISVALINGAM_MAXIMUM_ITERATIONS`.
- Capacities and caps: `GPULineDensity` `maximumRecords` → `recordCapacity`; `GPULineLengthPerPolygon`
  `maximumCandidatePairs` → `pairCapacity` and `maximumCrossings` → `maximumCrossingCount`;
  `GPULineSegmentize` `maximumPiecesPerSegment` → `maximumPieceCount` and
  `GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECES` → `GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECE_COUNT`;
  `GPUGreatCircleArcs` `maximumSegments` → `maximumSegmentCount` and
  `GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENTS` → `GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENT_COUNT`.
- CPU description types `GPU<Class>Parameters` → `GPU<Class>Settings` for `GPULineDensity`, `GPUOffsetCurve`,
  `GPUOutlineGeometry`, `GPURectangleClip`, `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`,
  `GPULineChunk`, `GPULineLocate` and `GPULineSimplification`.
- `GPUSegmentIntersection`: `pairs`, `kinds`, `points`, `endPoints`, `leftFeatures`, `rightFeatures`, `leftRings`, `rightRings` → `output.<name>`, and `uncertainCount` → `output.status.uncertainCount` (new type `GPUSegmentIntersectionOutput`)
- `GPULineSplit`: `pieces` → `output.pieces`, `uncertainCount` → `output.status.uncertainCount` (new type `GPULineSplitOutput`)
- `GPULineNoding`, `GPUPolygonize`, `GPUMakeValid`, `GPUPolygonOverlay`, `GPUBufferSurface`: top-level `uncertainCount` → `output.status.uncertainCount` (`GPUTopologyStatusPort` and the overlay status gain an optional `uncertainCount`)
- `GPUCoverageSimplification`, `GPUCoverageDissolve`, `GPUCoverageValidity`: polygon input is the four-level layout; `featureOffsets` is now required and rows are features (coverage identity, `labels` rows and `featureInvalidCounts` rows are features, not polygon parts). For one-polygon-per-feature data pass the identity `[0 .. polygonCount]`
- `GPUCoverageSimplification`: `maximumRounds` → `maximumIterations`, `topologyRounds` → `topologyIterationCount`, `converged` → `output.status.converged`, `output.keepMask` → `output.outputMask`, `output.overflow` / `output.requiredCount` → `output.status.overflow` / `output.status.requiredCount`, `output.topologyStats` → `output.topologyStatistics`, `GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATS_LENGTH` → `GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATISTICS_LENGTH`
- `GPUCoverageValidity`: `output.polygonInvalidCounts` → `output.featureInvalidCounts`, `output.invalidSegmentCount` → `output.status.invalidCount`
- `GPUCellSetOutline`: `output.count` / `output.overflow` / `output.requiredCount` → `output.status.count` / `output.status.overflow` / `output.status.requiredCount`
- Prop validation messages shortened (`leafCapacity`, `quadSegments`, `intersectionCapacity`, `capacity.*`, `topologyIterationCount` use the shared validators); a chunked vector in a packed-view prop now fails with `... must be one packed view`
- `GPUCellPyramidProps.levels` (`{resolution, output}[]`) → `resolutions` (`number[]`) plus `output.levels` (`GPUCellTable[]`); `levelCounts` → `output.levelCounts`. `GPUCellPyramidLevel` is removed.
- `GPUCellLevelSelection` moved to its own file `cell-aggregation/gpu-cell-level-selection.ts` (barrel export unchanged).
- `GPUCellCoverProps.polygonPositions` → `positions` (typed `GPUPolygonGeometryFields`). `GPUCellCoverOutput.overflow`, `candidateOverflow`, `invalidCount`, `uncertainCount` → `output.status.*`, as are `count` and `requiredCount`.
- `GPUCellCompactionOutput.overflow`, `droppedCount` → `output.status.overflow`, `output.status.droppedCount`. `GPUCellCompactionWordOrder` → `GPUCellWordOrder`.
- `GPUHilbertKeysProps.points` → `positions`.
- `GPUDotDensityProps` and `GPURandomPointsInPolygonProps`: `polygonPositions` → `positions`; `GPUDotDensityPolygons` → `GPUPolygonGeometryFields`; `output.failedCount` → `output.status.failedCount`. `GPURandomPointsInPolygon` moved to `dot-density/gpu-random-points-in-polygon.ts`.
- `GPUKernelDensityOutput.invalidCount`, `GPUPointDensityOutput.overflow`, `GPUKrigingOutput.invalidCount`/`uncertainCount` → `output.status.*`.
- `GPUGridGeneratorParameters` → `GPUGridGeneratorSettings`; `GPUShapeGeneratorParameters` → `GPUShapeGeneratorSettings`; the packers take `settings`.
- `GPUGridGeneratorProps.extent` `{positions, ringOffsets}` → `{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}` (`GPUGridGeneratorExtent` removed; all rings stay even-odd).
- `GPUBoundsFilterProps.box` → `parameters`; `GPUBoundsFilterBox` → `GPUBoundsFilterSettings`; `mask` → `output.outputMask`.
- `GPUNeighborSearchParameters` → `GPUNeighborSearchSettings`. `GPUNeighborSearchProps.weights`, `overflow`, `totalNeighbors`, `invalidCount`, `neighborCounts` → `output: GPUSpatialWeightsOutput & {neighborCounts?}` (`output.status.overflow`, `requiredCount`, `invalidCount`). `outputWeights.info` is now `outputWeights.information`.
- `GPUPairGatherOutput.overflow`, `matchedCount` → `output.status.*`; `GPUPairGatherProps.overflow` (extra flags) → `overflowSources`.
- Movement contributors (`GPUTrajectoryMetrics`, `GPUTrajectoryResample`, `GPUTrajectoryPlayhead`, `GPUZoneEvents`,
  `GPUTrajectoryEncounters`, `addClockEncounters`, `GPUTrackSimilarity`) write every caller-owned view through
  one `output: {...}` bag. `GPUTrajectoryMetrics`: `trackLengths`, `trackDurations`, `averageSpeeds`,
  `maximumSpeeds`, `stepSpeeds`, `stepHeadings`, `stepAccelerations`, `trackStopCounts`, `stops` → `output.*`.
  `GPUTrajectoryResample`: `samples`, `sampleElevations`, `sampleTimes` → `output.*`. `GPUTrajectoryPlayhead`:
  `currentPositions`, `currentElevations`, `headings`, `speeds`, `segmentRows`, `segmentFractions`,
  `activeTracks`, `drawInstanceCount` → `output.*`, and `status` → `output.trackStatus`.
- `GPUTrajectoryStopOutput`, `GPUZoneEventOutput`, `GPUZoneVisitTableOutput` are now `GPUCompactOutput` plus
  the optional columns: `stops.output.ids` → `output.stops.ids` (`count`, `overflow`, `requiredCount` → `output.stops.status.*`),
  `events.output.ids` → `output.events.ids`, `visitTable.output.ids` → `output.visitTable.ids`.
- `GPUZoneEvents`: `positions`, `timestamps`, `trackOffsets` → `tracks: {kind: 'trajectories', ...}`; the zone
  boundary edges `edgeStarts`, `edgeEnds`, `edgeZones` and `zoneCount` → `zones: {kind: 'polygons', positions,
  featureOffsets, polygonOffsets, ringOffsets}` (zone `z` is feature `z`, the zone count is
  `featureOffsets.length - 1`, rings close implicitly). Edge lists that were not closed rings are no longer
  expressible. `maxEventsPerTrack` → `trackEventCapacity`; `events`, `visitTable`, `dwellTimes`, `visitCounts`,
  `trackEventCounts` → `output.*`; `diagnostics` → `output.status` (type `GPUZoneEventsDiagnostics` →
  `GPUZoneEventsStatus`).
- `GPUTrajectoryEncounters` and `addClockEncounters`: `pairs` → `output.pairs` (flattened: `pairs.output.ids` →
  `output.pairs.ids`), `pairs.candidateOverflow` → `output.status.candidateOverflow`. `addClockEncounters`
  `samples` and `bucketTimes` outputs → `output.samples` and `output.bucketTimes`. The type
  `GPUTrajectoryEncounterOutput` is now the output bag; the pair list type is `GPUTrajectoryEncounterPairs`.
- `GPUTrackSimilarity`: `positionsA`/`offsetsA` → `linesA: {kind: 'lines', positions, lineOffsets}`,
  `positionsB`/`offsetsB` → `linesB`; `hausdorff`, `frechet` → `output.*`; `maxDistance` →
  `output.maximumDistance`; `status` → `output.pairStatus`; `maxFrechetVertices` → `maximumFrechetVertexCount`
  (also the instance field).
- Playhead packers: `maxGap` → `maximumGap`. CPU settings types: `GPUTrajectoryStopParameters` →
  `GPUTrajectoryMetricsSettings`, `GPUTrajectoryPlayheadTime` → `GPUTrajectoryPlayheadSettings`,
  `GPUTrajectoryClock` → `GPUTrajectoryClockSettings`.
- Several validation messages changed (`sampleCount`, `hitCapacity`, `trackCount`, `bucketCount`,
  `candidateCapacity`, `trackEventCapacity`, `maximumFrechetVertexCount` now say `must be an integer in [min, max]`).
- `GPUGroupConvexHullProps.totalCapacity` → `vertexCapacity` (the row capacity of `output.vertexIndices` and
  `output.vertexPositions`).
- `GPUPolygonOverlayOutput.sourceIds` is removed; `output.geometry.sourceIds` is the only source-ID view, so the
  "must be the same view" check is gone.
- `GPUGeometryMeasuresOutput.signedAreas` (and the group output's) is removed. With positional ring roles it is
  `sum(|shell|) - sum(|hole|)`, never negative, and equal to `areas`; read `areas`.
- Line geometry uses `line*` names instead of `path*`. `GPULinePathOutput` → `GPULineGeometryOutput`;
  `GPULineSegmentize`, `GPULineChunk`, `GPULineSmooth`, `GPUGreatCircleArcs` and `GPURectangleClip`
  `status.pathCount` → `status.lineCount`, `sourcePaths` → `sourceLineIndices`, `pathCapacity` → `lineCapacity`;
  `GPULinearReferencingOutput.pathIndices` → `lineIndices` and its `pathCount` field → `lineCount`;
  `GPULineLocateProps.eventPaths` → `eventLineIndices`; `GPULineLengthPerPolygonProps.pathWeights` →
  `lineWeights`. Graph and grid paths (`GPUCostDistancePath`, network shortest paths, edge bundling,
  `GPUCellGridPath`) keep `path`.
- `GPUSegmentRingAssembly` decides `ringIsHole` (and so shells, holes and `polygons`) from the exact turn at
  each ring's lowest vertex instead of the f32 sign of `ringAreas`. Results change only for thin rings whose
  f32 shoelace sum had the wrong sign or was zero; `ringAreas` itself is unchanged.

#### Network

- Every `gpu-network` contributor takes the network as one `topology: GPUNetworkTopology`
  (`offsets`, `neighbors`, `weights`, optional `nodePositions`; `GPUNetworkAdjacency` where weights are
  optional) instead of loose `offsets`/`neighbors`/`weights`/`nodePositions` props. Snapping and map
  matching drop `edgeTargets`/`edgeCosts` (use `topology.neighbors`/`topology.weights`); snapping drops
  `edgeSources` (sources are always expanded from `topology.offsets`). The drive-time recipe's
  `network` (`GPUDriveTimeNetwork`) is now `topology`.
- Every GPU-written view moved into `output` (reachability, cost matrix, snapping, service areas,
  neighborhood, path extraction, k-function, line graph, noding, coarsening, flow aggregation, edge
  bundling, adjacency-matrix order). Output CSRs are topologies: line graph
  `lineOffsets`/`lineNeighbors`/`lineWeights` → `output.topology`, noding `csr` → `output.topology`
  (`GPUNetworkNodingCSR` → `GPUNetworkNodingTopology`), subgraph filter `inducedCSR` →
  `inducedTopology` (`GPUNetworkSubgraphFilterInducedCSR` → `GPUNetworkSubgraphFilterInducedTopology`).
- Graph starts are `start*`: `sources`/`sourceCosts`/`sourceCount`, `seedNodes`/`seedCosts`/`seedRows`/
  `seedsPerRow`, neighborhood `seeds`/`seedCount` → `startNodes`/`startCosts`/`startCount`/`startRows`/
  `startsPerRow`; snapping `seedDirection` → `startDirection` (`GPUNetworkSnappingSeedDirection` →
  `GPUNetworkSnappingStartDirection`); recipe `seedNodes`/`seedCosts` → `startNodes`/`startCosts`.
- Cost bands are `costBreaks`: reachability `bandThresholds`, isochrones `breaks`, drive-time
  `bandBreaks` and isochrone-option `breaks`.
- `max*` → `maximum*`: `maxIterations`, `maxTieIterations`, `maxSnapDistance`, `maxHops`,
  `maxPathLength`, k-function `maxDistance`, `maxProbeCount`, adjacency `output.maxCount`/`maxWeightSum`,
  the readonly class fields, `ReachabilityIterations*.maxIterations`, and the statistics words
  `maxOutDegree`/`maxInDegree`/`maxTotalDegree`. Caps: `labelIterations` → `maximumLabelIterations`,
  `componentIterations` → `maximumComponentIterations`, analytics `coreNumber`/`components`/
  `communities.iterations` → `maximumIterations`, edge bundling `iterations` → `maximumIterations`;
  PageRank `iterations` (a fixed count) → `iterationCount`.
- `startCount`, `facilityCount` and `targetCount` accept a number (compiled in) or a one-row view.
- Points are `positions` (snapping, k-function, map matching `points`); path offsets are `lineOffsets`
  (path extraction `pathOffsets`, edge bundling `startIndices`); edge bundling `positions`/
  `sourceVertices`/`targetVertices` → `nodePositions`/`sourceNodes`/`targetNodes`.
- Network rows are nodes: `vertexMask` → `nodeMask`, `vertexColumns` → `nodeColumns`, `vertexValues` →
  `nodeValues`, `groupVertexCount` → `groupNodeCount`, `liveVertices` → `liveNodes`, and the
  `liveVertexCount`/`isolatedVertexCount`/`overflowedVertexCount`/`vertexRanges`/`vertexColumnCount`
  fields → `*Node*`. Coarsening `edgeTargets` → `output.edgeTargetGroups`.
- Packers: `getGPUNetworkIsochroneParameterValues` → `getGPUNetworkIsochronesParameterValues`
  (`GPUNetworkIsochrone{Settings,Raster,CellOutline}` → `GPUNetworkIsochrones*`),
  `encodeGPUMapMatchingParameters` → `getGPUMapMatchingParameterValues` (`GPUMapMatchingSettings`),
  `encodeGPUNetworkStatisticsParameters` → `getGPUNetworkStatisticsParameterValues`,
  `encodeGPUNetworkAccessibilityParameters` → `getGPUNetworkAccessibilityParameterValues`
  (`GPUNetworkAccessibilityParameters` → `GPUNetworkAccessibilitySettings`),
  `createGPUEdgeBundlingParameterValues` → `getGPUEdgeBundlingParameterValues`; adjacency-matrix
  `window` → `parameters`, `encodeGPUAdjacencyMatrixWindow(window, resolution)` →
  `getGPUAdjacencyMatrixParameterValues({...window, resolution})`, `GPU_ADJACENCY_MATRIX_WINDOW_LENGTH` →
  `GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH`; subgraph filter `timeWordParameters` → `uint32Parameters`.
- `GPUNetworkStatistics.destroy()`, `GPUNetworkAnalyticsColumns.destroy()` and edge bundling's public
  `geographic` field are removed: statistics scratch is now graph transients.
- `GPUDriveTimeCatchmentRecipeResult` → `GPUDriveTimeCatchmentRecipeOutput`.

#### Algorithms (2026-10-14): statistics, dataframe and crossfilter

- `GPUAZPRegions` replaces the exact per-candidate reachability search and its single best move per
  iteration with a local articulation test and conflict-free batched moves (every region in at most
  one move per round). Results differ from earlier versions: still a monotone greedy descent to a
  local optimum that keeps every region contiguous, but a different (and usually lower) optimum, and
  `status.iterationCount` now counts rounds, not moves, so far fewer `maximumIterations` are needed.
  The test is sufficient only, so a safe move whose reconnecting path is longer than two steps from the
  row is refused. `GPU_AZP_MAXIMUM_ROWS` (32,768) is removed: only the attribute and region tables must
  fit one binding (`getGPUAZPMaximumTableWordCount(device)`). `GPUMaxPRegions` improvement is no longer
  limited by it either.
- `GPUMaxPRegions` plants many seeds at once and grows all regions in parallel (seed waves, one claimed
  row per region and round, enclave rounds) instead of one sequential region per start, so the regions
  differ from earlier versions for the same seed; the region count is within about 3 percent of spopt's
  `MaxPHeuristic` on the oracle instances. `GPU_MAX_P_MAXIMUM_ROWS` (32,768) is removed: only the scratch
  tables must fit one storage binding. Labels are dense `0..regionCount-1`. A region now needs at most
  `maximumGrowthSteps` (default 256, new) rows; larger regions are reported under the threshold unless
  it is raised. The start-selection sum of squares counts the grown regions before enclave rows join.
- `GPULocationAllocation` scores swaps with the Whitaker / Resende-Werneck gain decomposition, one pass per
  candidate instead of one per (selected facility, candidate) pair. The sums are grouped differently, so
  where float32 rounding separates a near-tie the applied swap can differ from earlier versions (integer
  data is unchanged); the swap kernels are now `list`, `prepare`, `removal`, `score`, `choose` and `apply`.
- `GPUSpatialScanStatistic` draws and evaluates its replicates in batches. The `scatter`, `draw-labels` and
  `draw-times` graph nodes become `scatter-<batch>`, `draw-labels-<batch>` and `draw-times-<batch>`, each batch
  adds `batch-control`, `batch-clear` and `batch-evaluate` nodes, and `evaluate` now covers the observed data
  only. Outputs are unchanged. Code that looked nodes up by ID needs the new IDs.
- `GPUClassBreaks` `'natural-breaks'` no longer snaps breaks to bin edges: each inner edge is now the
  smallest actual data value of its class and the partition is Fisher-Jenks to within a local optimum
  (the previous edges were `min + start * range / naturalBreaksBinCount`, up to one bin width off). The
  default `naturalBreaksBinCount` rises from 1024 to 2048 and the `class-costs` table node is gone; a
  class count above the number of distinct values now produces one class per distinct value. Pass
  `naturalBreaksBinCount: 1024` to keep the old bin grid for the coarse stage.

#### Algorithms (2026-10-14): geometry, joins and cells

- `GPUPointDensity` `sumAccumulation: 'auto'` now always selects the deterministic fixed-point
  `'workgroup'` path (it picked float atomics for diffuse input). Pass `'atomic'` to opt back in to the
  faster, non-deterministic sums. Weighted deterministic grids hold at most
  `GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT` (`floor(2^27 / 12)`) cells; larger ones need `'atomic'`.
- `GPUMapColoring` colors largest-degree-first by default (`priority: 'largest-degree-first'`), so colors
  differ from earlier releases for the same seed, usually with one color fewer on planar maps. The result is
  still a proper, seed-deterministic coloring. Pass `priority: 'random'` for the previous order.

#### Algorithms (2026-10-14): movement and clustering

- `GPUZoneEvents` dwell times are now reduced in a fixed tree order per `(track, zone)` instead of
  a sequential sum per track: the same pair differences, so values can differ in the last bits;
  event types, ranks, visit counts and spans are unchanged.
- `GPULinearReferencing` with a great-circle context now honors `leafCapacity` and `spatialSort`
  (previously ignored); a `leafCapacity` below the position count sets `output.status.overflow`.

#### Algorithms (2026-10-14): topology

- `GPUDelaunayTessellation` builds the triangulation by parallel insertion and exact flipping instead
  of one Bowyer-Watson step per site. Output rows are now rotated so each triangle's smallest row comes
  first and sorted by their first two rows (they were in the serial insertion order), and cocircular
  ties (lattices, regular polygons) follow Simulation of Simplicity on row order instead of insertion
  order: the diagonal chosen for four cocircular sites can differ. The triangle set of input in general
  position is unchanged.
- `GPUPolygonOverlay` (and so dissolve and `GPUBufferSurface`) nodes by snap rounding and labels faces
  instead of probing every ring per segment. `vertexTolerance` is now the snap grid: rounded up to a
  power of two and to at least `2^-22` of the input extent, with every output vertex on that lattice.
  The result equals GEOS OverlayNG with `grid_size` on the pointwise lattice-rounded operands. Even-odd
  is now the parity of all rings of one polygon (unchanged for valid polygons; differs for holes outside
  the shell or overlapping holes). `output.boundary` has one row per group of coincident edges, with the
  provenance of the group's first piece. Input rows + rings + `capacity.intersections` are limited to a
  quarter of the segment-table maximum (1,048,576 on a default device), and
  `getGPUBufferSurfaceMaximumBoundaryRowCount` drops to the same bound.

#### Algorithms (2026-10-14): terrain

- `GPUTerrainFlow` D8 `'cells'` accumulation without `runoff` and `GPUTerrainStreamOrder` run on an
  Euler tour of the receiver forest. Long stems no longer truncate at 64 rounds: 'cells' D8
  accumulation is exact and unbounded, and `maximumAccumulationIterations` and `maximumIterations`
  now bound only the pull paths (`runoff`, `'area'`, MFD and D-infinity routing, and grids above
  4096 x 4096 cells). `status.accumulationIterationCount` reports the ranking rounds.
- `GPUSimpleLocalRelief` and `GPUMultiScaleRelief` compute the window mean from an exact 64-bit
  summed-area table on grids up to `getGPUTerrainSummedAreaMaximumCellCount(device)` cells. Elevations are
  quantized to `quantum` (default `2^-16`), so results differ from the previous filter by at most
  `quantum / 2` plus float32 rounding. The GPU selects the first valid elevation as a grid anchor
  (snapped to the quantum); only offsets from that anchor reaching `2^31 * quantum` (32768 by
  default) are invalid (NaN), regardless of absolute elevation. The radius is limited to 23170.
  Larger grids keep the previous filter.
- `GPURasterCrossTileComponents` now uses deterministic fixed-point intensity and coordinate
  sums. Its capacity helper additionally accounts for 12 accumulator bytes per region per storage
  binding (11184810 regions with 128 MiB bindings); larger declarations are rejected.

#### Core binding maxima

Every documented size maximum of the analysis contributors now holds on the device the graph runs
on, and a size above it throws while the contributor is declared (`graph.add` / `getCommandNodes`),
with `<id> <prop> must be at most <limit>` or `must be an integer in [1, <limit>]`. Before, such
sizes threw later from a kernel, a graph buffer or a dispatch check, or failed at bind time.

- Binding-sized maxima are functions of the device, `getGPU<Contributor>Maximum<Thing>(device)`,
  computed from `device.limits.maxStorageBufferBindingSize` (or `maxComputeWorkgroupsPerDimension` for
  the one-dispatch raster reductions). A device created with larger binding limits accepts larger
  inputs, and algorithm choices such as the exact summed-area relief filter follow the same limit.
  The values quoted in the module docs are those of a default device (128 MiB bindings).
- The shared ones: `getStorageBindingElementLimit(device, bytesPerElement, extra)`,
  `getGPUGridIndexMaximumCellCount(device)` (grid indexes of kernel density, inverse distance
  weighting, kriging, spatial clustering, neighbor search, trajectory encounters; `2^25 - 1` by
  default, encounters used to document `2^27`), `getGPUSegmentIntersectionMaximumVertexCount(device)`
  and `getGPULineSplitMaximumPieceBound(device)` (the segment-intersection, noding and overlay family,
  `2^22` by default, was `2^30`/`2^31`), `getGPUPairStatisticsMaximumRowCount(device)` (variogram,
  correlogram, Ripley, point-pattern indices; also capped at `2^24 - 1`, the float32-exact row count),
  `getGPUNearestFeatureJoinMaximumLeafCapacity(device)` and `getGPURegionalizationMaximumSlotCount(device)`.
- Default-device values that are much lower than before: terrain spike repair 4194304 cells and the
  summed-area paths 11184810 cells (a 4096 x 4096 grid needs a device with larger bindings or tiling),
  `GPUNetworkNoding` `intersectionCapacity` `2^20`, `GPUBufferSurface` boundary rows `2^22`.
- Limits that come from a uint32 invocation index stay constants:
  `GPU_LINE_DENSITY_MAXIMUM_CELL_COUNT`, `GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT`,
  `GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT`, `GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT`,
  `GPU_SPATIAL_CLUSTERING_MAXIMUM_SORTED_CLUSTER_COUNT`, `GPU_AREAL_INTERPOLATION_MAXIMUM_WEIGHTED_PAIR_COUNT`,
  and the `capacity x lanes <= 2^32 - 1` checks of flow aggregation, the network K function and similar
  locations.
- Defaults derived from a limit (map-matching `entryCapacity`, cell-cover `edgeSlabEntryCapacity`,
  line-length `pairCapacity`) are clamped to the device value.
- Count scans: a candidate or pair total that reaches `2^32` now reports overflow with a saturated
  `requiredCount` of `0xffffffff` in `GPUSegmentIntersection` (and everything built on it) and the
  spatial join candidate and probe passes, instead of wrapping to a small count.

## Upgrading to v9.4

**@luma.gl/core**

- WebGPU device creation now defaults to the portable `DeviceProps.featureLevel: 'core'`. Applications that relied on luma.gl requesting every adapter feature and supported limit should pass `featureLevel: 'max'`.
- Render draw state is now owned by `RenderPass`. `RenderPipelineProps.bindings`, `RenderPipelineProps.bindGroups`, `RenderPipeline.setBindings()`, and `RenderPipeline.draw()` are deprecated compatibility APIs. Migrate low-level rendering code to `renderPass.setPipeline()`, `renderPass.setBindings()`, `renderPass.setVertexArray()`, and `renderPass.draw()`.
- `CommandEncoder.finish()` no longer accepts command-buffer properties, and the `CommandBufferProps` type has been removed. Set `id` and `userData` on the command encoder; the finished command buffer inherits them.

**@luma.gl/engine**

- `BufferTransform.run()` now creates its render pass with `discard: true` by default, avoiding
  unnecessary attachment stores for transform-feedback-only workloads. Applications that attach a
  framebuffer and consume rasterized fragment output must pass `discard: false` to `run()`.
- `Model.predraw(commandEncoder)` now requires an explicit command encoder. Call it with the encoder that will be submitted when ordered pre-draw uploads must be shared across multiple draws or viewports. Normal `Model.draw(renderPass)` calls continue to perform their own pre-draw work.
- `makeGPUGeometry()` now interleaves CPU geometry attributes into a single vertex buffer by default. Callers that require separate attribute buffers should create those buffers and construct `GPUGeometry` explicitly with the corresponding `bufferLayout`.

**@luma.gl/webgl**

- WebGLDeveloperTools and Spector integration now require `import '@luma.gl/webgl/debug'` before enabling `debugWebGL` or `debugSpectorJS`. This keeps debug-only code and the full GL enum out of normal adapter application bundles.

**@luma.gl/webgpu**

- `getShaderLayoutFromWGSL()` now uses lightweight interface scanning and returns `null` when WGSL is ambiguous or outside the supported subset. Raw render and compute pipelines must provide an explicit `shaderLayout` in that case. Uniform-buffer member reflection is no longer included in the returned layout.

**@luma.gl/shadertools**

- `ShaderAssembler` is now abstract and can no longer be constructed directly. Replace
  `new ShaderAssembler()` with `new GLSLShaderAssembler()` for GLSL or
  `new WGSLShaderAssembler()` for WGSL.
- `ShaderAssembler.getDefaultShaderAssembler()` now requires an explicit shader language.
  Replace calls without an argument with `ShaderAssembler.getDefaultShaderAssembler('glsl')`
  or `ShaderAssembler.getDefaultShaderAssembler('wgsl')`.
- `assembleGLSLShaderPair()` is available only on `GLSLShaderAssembler`, and
  `assembleWGSLShader()` is available only on `WGSLShaderAssembler`. Narrow existing
  `ShaderAssembler` references with `instanceof GLSLShaderAssembler` or
  `instanceof WGSLShaderAssembler` before assembling shader source.

## Upgrading to v9.3

**Potentially breaking behavior**
- `AsyncTexture` has been renamed to `DynamicTexture`.
- Scenegraph creation API has been improved, see [`createScenegraphsFromGLTF()`](/docs/api-reference/gltf).
- gltf module now creates `DynamicTexture` instances rather than raw `Texture`s.
- glTF texture sampling now defaults to linear filtering when a glTF sampler omits explicit filter settings. Applications relying on the previous nearest-neighbor default should verify visual output and set sampler filters explicitly when nearest sampling is required.
- The legacy feature flag `timer-query-webgl` has been removed. Replace checks for `timer-query-webgl` with `timestamp-query` for GPU timestamp/query support on both WebGPU and WebGL.
- `PipelineFactory` and `ShaderFactory` now import from `@luma.gl/core` instead of `@luma.gl/engine`.

## Upgrading to v9.2

v9.2 brings full WebGPU support. Some additional deprecations and breaking changes have been necessary, but apart from the `Texture` -> `AsyncTexture` split, impact on most applications should be minimal.

**New VertexFormats**
- `VertexFormat` Replace `'unorm8-webgl'` with `'unorm8'`.

**Texture and AsyncTexture**
- The `Texture` class has been simplified to the minimum API required for GPU portability. The `AsyncTexture` texture class provides a higher-level API and is recommended for most applications.
- `device.createTexture()` no longer accepts `props.data`: Use `AsyncTexture` or call `texture.setImageData()`
- `device.createTexture()` no longer accepts `props.mipmaps`: Use `AsyncTexture` (or call `texture.generateMipmapsWebGL()`)
- On WebGPU, mipmap generation now lives in `AsyncTexture.generateMipmaps()`, not in core `Texture`.
- WebGPU `AsyncTexture` uses render passes for `2d`, `2d-array`, `cube`, and `cube-array`, and a compute path for `3d`.
- Unsupported WebGPU formats now fail explicitly when mipmap generation is requested, instead of silently acting as a no-op.
- `TextureFormat` Correct the PVRTC 2bpp RGB format spelling from `pvrtc-rbg2unorm-webgl` to `pvrtc-rgb2unorm-webgl`.

**Removal of WebGL uniform support**
- The transition from uniforms to uniform buffers is complete, and remaining support for non-buffer uniforms has been removed.
- `core`: `RenderPipeline.setUniformsWebGL()` dropped, use uniform buffer bindings
- `engine`: `Model.setUniformsWebGL()` deprecated, use uniform buffer bindings
- `shadertools`: WebGL1 shader modules have been removed, use the new modules uniform buffer-based counterparts.

**`CanvasContext` simplifications**
- `canvasContext.devicePixelWidth` and `canvasContext.devicePixelHeight` are now kept updated to exact device pixel size of underlying canvas.
- Instead `canvasContext.setDrawingBufferSize()` to explicitly control drawing buffer size, if not using `CanvasContextProps.autoResize`
- A new `DeviceProps.onResize` callback can be used to react to changes.

**Minor changes**
- `core`: The shader types has been refactored, some shader type names have changed. These are typically not used directly by applications.

## Upgrading to v9.1

v9.1 continues to build out WebGPU support. Some additional deprecations and breaking changes have been necessary, but impact on most applications should be minimal.

**Major change: Adapters**

- When initializing luma.gl, applications now import an `Adapter` singleton from the WebGPU and/or the WebGL module, and passes the adapter object(s) to `luma.createDevice()`, `makeAnimationLoop` etc.
- `luma.registerDevices()` can be replaced with `luma.registerAdapters()` if global registration is still desired.

**Major change: Texture and AsyncTextures**

- The texture API is being streamlined to work symmetrically across WebGPU and WebGL.
- `Texture.copyExternalImage()` and `Texture.copyImageData()` replaces `Texture.setImageData()` when initializing texture memory with image data.
- `Textures` no longer accept promises when setting data (e.g. from `loadImageBitmap(url)`.
- Instead, a new `AsyncTexture` class does accept promises and creates actual `Textures` once the promise resolves and data is available.
- The `Model` class now accepts `AsyncTextures` as bindings and defers rendering until the underlying texture has been created.

**@luma.gl/core**

| Updated API                   | Status     | Replacement                                  | Comment                                                         |
| ----------------------------- | ---------- | -------------------------------------------- | --------------------------------------------------------------- |
| `luma.registerDevices()` | Deprecated | [`luma.registerAdapters()`][adapters]. | Adapters provide a cleaner way to work with GPU backends. |
| `DeviceProps.canvas` | Moved | [`DeviceProps.createCanvasContext`][canvas]. | Move canvas related props to `props.createCanvasContext: {}`. |
| `DeviceProps.<webgl options>` | Moved | [`DeviceProps.webgl.<options>`][webgl]. | Move canvas related props to `props.webgl: {}`. |
| `DeviceProps.break` | Removed | — | Use an alternative [debugger][debugging] |
| `TextureProps.data` (Promise) | Removed | `AsyncTexture` class | `Texture` no longer accept promises. Use `AsyncTexture` |
| `Parameters.blend` | New | — | Explicit activation of color blending |
| `triangle-fan-webgl` topology | Removed | `triangle-strip`. | Reorganize your geometries |
| `line-loop-webgl` topology | Removed | `line-list`. | Reorganize your geometries |
| `glsl` shader template string | Removed | `/* glsl */` comment | Enable syntax highlighting in vscode using before shader string |
| `depth24unorm-stencil8` | Removed | `depth24plus-stencil8` | The `TextureFormat` was dropped from the WebGPU spec |
| `rgb8unorm-unsized` | Removed | `rgb8unorm` | Drop support for unsized WebGL1 `TextureFormat` |
| `rgba8unorm-unsized` | Removed | `rgba8unorm` | Drop support for unsized WebGL1 `TextureFormat` |

[adapters]: /docs/api-reference/core/luma#lumaregisteradapters
[canvas]: /docs/api-reference/core/canvas-context#canvascontextprops
[webgl]: https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext#contextattributes
[debugging]: /docs/developer-guide/debugging

**@luma.gl/shadertools**

| Updated API                          | Status  | Replacement                             | Comment                                            |
| ------------------------------------ | ------- | --------------------------------------- | -------------------------------------------------- |
| `ShaderModuleInstance` | Removed | Use `ShaderModule` instead. | Type has been removed. |
| `initializeShaderModule()` | Changed | — | Initializes the original shader module object |
| `ShaderModuleInstance.getUniforms()` | Removed | `getShaderModuleUniforms(module, ...)`. | Interact directly with the shader module |
| `getDependencyGraph()` | Removed | `getShaderModuleDependencies(module)` . | Interact directly with the shader module |
| `glsl` template string | Removed | `/* glsl */` comment | Enable syntax highlighting in vscode using comment |

**@luma.gl/effects**

New module. All postprocessing effects that were previously in `@luma.gl/shadertools` are now exported from `@luma.gl/effects`.

**@luma.gl/webgl**

- `WebGLDeviceContext` - Note that luma.gl v9.1 and onwards set `DeviceProps.webgl.preserveDrawingBuffers` to `true` by default. This can be disabled for some (potential) memory savings and a (potential) minor performance boost on resource limited devices, such as mobile phones, at the cost of not being able to take screenshots or rendering to the screen without clearing it.

## Upgrading to v9.0

luma.gl v9 is a major modernization of the luma.gl API, with many breaking changes, so the upgrade notes for this release are unusually long. To facilitate porting to the v9 release we have also provided a
[Porting Guide](/docs/legacy/porting-guide) that also provides more background information and discusses porting strategies.

## Upgrading to v8 and earlier releases

This page only covers luma.gl v9 and later releases.
For information on upgrading to from v8 and earlier releases, see the [Legacy Upgrade Guide](/docs/legacy/legacy-upgrade-guide).

## GPU Core composition

GPU Core primitives now expose `getCommandNodes(graph)` instead of `addToGraph(graph)`. Schedule
a primitive with `graph.add(primitive)`, which calls `getCommandNodes(graph)` internally. To inspect
or modify nodes before scheduling, use `addGPUCommandNodes(graph, nodes)`. Graph-independent
primitives can also be added to `GPUProgram`. `GPUCommandGraphContributor` and the compiler's legacy mutation fallback were
removed. The three range/scatter primitives that also publish scratch views expose
`getCommands(graph)` returning `{nodes, ...views}`; schedule `nodes` explicitly.

`GPUProgramBindings.vectors` now accepts typed `GPUData`, readonly `GPUData[]`, or `GPUVector`
instead of raw buffers. Compiler vector resolution returns `GraphVectorView`; access physical
chunks through `.data` or canonical logical descriptors through `.chunks`.
