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
const compiled = await graph.compileAsync(); // once, without blocking the page

// every frame: rewrite parameters, encode the same compiled graph
viewport.write(new Float32Array([minX, minY, maxX, maxY]));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

Read [GPU Spatial Analysis concepts](./concepts.md) for the ontology, shared ports, conventions,
coordinate precision and the compile, encode and submit pattern. The operator pages group the
contributors by task; the [operators index](./operators.md) lists them all.

| Section | Contributors |
| --- | --- |
| [Spatial weights](./operators-weights.md) | `GPUNeighborSearch`, `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsTransform`, `GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsSummary`, `GPUSpatialWeightsTranspose`, `GPUSpatialWeightsValidation`, `GPUSpatialLag`, `GPUMapColoring` |
| [Spatial statistics](./operators-statistics.md) | `GPUGlobalSpatialStatistics`, `GPUMultipleTestingAdjustment`, `GPUAssuncaoReisStandardization`, `GPUHotSpotAnalysis`, `GPULocalMoran`, `GPULocalGeary`, `GPUSpatialPearson`, `GPUGammaStatistic`, `GPURoseStatistic`, `GPULocalPermutationTest`, `GPUGlobalPermutationTest`, `GPUNeighborhoodSummary`, `GPUEmpiricalBayesRates`, `GPUSpatialEmpiricalBayesRates`, `GPUSpatialClustering`, `GPUHierarchicalDensityClustering`, `GPUKMeans`, `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, `GPUAZPRegions`, `GPUMaxPRegions`, `GPUWardRegions`, `GPULocationAllocation`, `GPURegionPartitionEvaluation`, `GPUGroupGeometry`, `GPUGroupConvexHull`, `GPUEmergingHotSpots`, `GPUGeographicDistribution`, `GPURegionStatistics`, `GPUZonalStatistics`, `GPUFocalStatistics`, `GPUSimilarLocations` |
| [Distribution dynamics and space-time tests](./operators-dynamics.md) | `GPUClassAssignment`, `GPUClassificationFit`, `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov`, `GPUTransitionDiagnostics`, `GPUKnoxTest`, `GPUMantelTest`, `GPUSpatialScanStatistic` |
| [Accessibility and change of support](./operators-accessibility.md) | `GPUCatchmentAccessibility`, `GPUHuffTradeAreas`, `GPUArealInterpolation`, `GPUPycnophylactic`, `GPUSegregation` |
| [Spatial joins](./operators-joins.md) | `GPUSpatialPredicateJoin`, `GPUSpatialJoinPrepared`, `GPUSpatialJoinCandidates`, `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin`, `GPUNearestFeatureWeights`, `GPUBufferSelection` |
| [Geometry validity and topology](./operators-validity.md) | `GPUSegmentIntersection`, `GPUGeometryValidity`, `GPULineNoding`, `GPUPolygonize`, `GPUMakeValid`, `GPUPolygonOverlay`, `GPUBufferSurface` |
| [Density & interpolation](./operators-density.md) | `GPUPointDensity`, `GPUKernelDensity`, `GPUInverseDistanceWeighting`, `GPUKriging`, `GPUDotDensity`, `GPURandomPointsInPolygon` |
| [Cells (DGGS)](./operators-cells.md) | `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid`, `GPUCellLevelSelection`, `GPUPointToCell`, `GPUCellGeometry`, `GPUCellTopology`, `GPUCellCompaction`, `GPUCellCover`, `GPUCellSetOutline`, `GPUSegmentRingAssembly`, `GPUCellTableCompare` |
| [Lines & trajectories](./operators-lines.md) | `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`, `GPULineSplit`, `GPULineMerge`, `GPUGeometryMeasures`, `GPUGeodesicPairs`, `GPUGeodesicDestination`, `GPULocalFrameProjection`, `GPULinearReferencing`, `GPULineLocate`, `GPULineSimplification`, `GPUCoverageSimplification`, `GPUTrajectoryMetrics`, `GPUTrajectoryPlayhead`, `GPUTrajectoryResample`, `GPUZoneEvents`, `GPUTrajectoryEncounters`, `GPUTrajectoryClosestApproach`, `addClockEncounters`, `GPUTrackSimilarity` |
| [Geometry utilities and tessellation](./operators-geometry.md) | `GPUOutlineGeometry`, `GPULabelPoint`, `GPUShapeDescriptors`, `GPULineDensity`, `GPULineLengthPerPolygon`, `GPUGridGenerator`, `GPUShapeGenerator`, `GPUHilbertKeys`, `GPURectangleClip`, `GPUDelaunayTessellation`, `GPUVoronoiDiagram` |
| [Regression](./operators-regression.md) | `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialErrorGM`, `GPUSpatialLagMaximumLikelihood`, `GPUSpatialErrorMaximumLikelihood`, `GPUSpatialWeightsTraces`, `GPUSpatialLagImpacts`, `GPUGeographicallyWeightedRegression`, `GPUGeographicallyWeightedRegressionNonstationarityTest` |
| [Recipes](./recipes.md) | `addHotSpotAnalysisRecipe` and eleven more prebuilt chains |
| [Cross-reference](#cross-reference) | turf, PostGIS, GeoPandas, PySAL, QGIS and ArcGIS tools mapped to contributors |

Contributors for networks, rasters and dataframe columns live in their own entries:
[GPU Network](/docs/api-reference/experimental/gpu-network),
[GPU Raster](/docs/api-reference/experimental/gpu-raster/operations-regions-contours) and
[GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe-analysis).

## Catalog

One figure per contributor, each drawn from a recorded GPU run on real data; click through to its card.

::lens-catalog{entry=gpu-spatial-analysis}

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
