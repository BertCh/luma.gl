---
title: "GPU Spatial Analysis: Recipes"
sidebar_label: "Recipes"
description: "Prebuilt GPU spatial analysis recipes, documented by the contributors they compose."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Recipes

<ExperimentalDocsTabs active="gpu-spatial-analysis-recipes" />

Recipes are plain functions, not classes: `addXRecipe(graph, props)` constructs the contributors, calls
`graph.add` in chain order, wires outputs to inputs and returns `{contributors, ...named views}`. Views you
pass in (`zScores`, `bins`, `colors`, ...) are caller-owned and readable; every other intermediate is a
graph transient. Inference recipes (hot spot, rate cluster, space-time hot spots, spatial regression) take
one `settings` object and `GPUParameterBuffer`s they fill themselves, derive `result.inference` from the same
object and repack it with `result.setSettings(settings)`; the other recipes pass per-frame parameter views
(`getGPU...ParameterValues`) through. Neither rebuilds the graph. Geometry inputs are the shared ports:
`GPUPointGeometryPort`, `GPUPolygonGeometryPort` (four-level layout, one row per feature) and
`GPUPackedGeometryPort<GPUTrajectoryGeometryPort>`.

```ts
const hotSpots = addHotSpotAnalysisRecipe(graph, {
  source: {kind: 'points', positions, family: 'quadbin', resolution: 8, cellCapacity: 4096,
    studyArea: {kind: 'polygons', positions: studyPositions, featureOffsets, polygonOffsets, ringOffsets,
      candidateCapacity: 16384}},
  settings: {significanceLevel: 0.05, seed: 7, permutationCount: 199},
  parameters: autocorrelationParameters, // GPUParameterBuffers the recipe fills from settings
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
| `addDriveTimeCatchmentRecipe` (from `gpu-network`) | GPUNetworkSnapping -> GPUNetworkServiceAreas -> (GPUNetworkIsochrones raster and/or cell outline with rings -> GPUPointInPolygonJoin demand join) -> snap demand -> band adapters -> GPUGroupStatistics |
| `addStraightLineCatchmentsRecipe` | GPUDistanceField allocation -> GPURasterZonalStatistics |
| `addChangeOfSupportRecipe` | 2x GPUPolygonRasterization -> GPUArealInterpolation (with source conservation diagnostics) -> GPUSpatialLag |
| `addFleetDwellRecipe` | GPUTrajectoryMetrics stops -> GPUPointInPolygonJoin -> stop mask -> GPUGroupStatistics (dense, one row per zone) |
| `addFleetDwellZoneEventsRecipe` | GPUZoneEvents -> GPUGroupStatistics (dense, one row per zone; zone-event variant of the fleet dwell recipe) |
| `addPeriodComparisonRecipe` | 2x GPUCellAggregation -> GPUCellTableCompare (with mask) -> GPUClassBreaks -> GPUColorScale |

Notes:

- The space-time recipe counts events into a dense `cell * sliceCount + slice` cube with integer atomics,
  because `GPUCellAggregation` keys must be valid Quadbin or H3 cells. Passing the sorted-path scratch
  views or `outputs.status.groupOverflow` selects the `GPUGroupStatistics` path instead.
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
  so a facility or demand point snapped past a dropped candidate is never silent. The other stages are
  `service-areas`, the caller-owned isochrone outputs (`isochrone-bands`, `isochrone-band-edges`,
  `isochrone-band-rings` on the raster path; `isochrone-cells`, `isochrone-outline`, `isochrone-rings` on
  the cell path) and `band-statistics` (count, overflow, required count). The service-area `assignments`
  are passed to `GPUNetworkIsochrones`, so `raster.output.pixelFacilities`, `triangleFacilities` and
  `cellOutline.byFacility` work through the recipe; facility rows are seed rows (`2 * facility + endpoint`).
- Every recipe reports each inner contributor's status as a `status.stages` entry (rule 10): the
  cluster recipe ends with `hull-measures` (`invalidCount`), the straight-line recipe has
  `zonal-statistics` (`overflow`), and the table, join and rasterization stages also carry `requiredCount`
  or `uncertainCount` where the inner contributor computes them. Result-level status words sit in
  `result.status` next to `stages` (for example `status.weightsOverflow`, `status.invalidCount`,
  `status.rasterOverflow`), sub-results own a `status` (`result.permutation.status.overflow`,
  `result.table.status.count`), and the `outputs` overrides mirror those paths (`outputs.status.weightsOverflow`).
- The drive-time, straight-line catchment and fleet dwell recipes take an optional `spatialContext` and
  return the resolved one as `result.spatialContext`. The straight-line recipe and both fleet dwell recipes
  are planar only (longitude/latitude throws; the join and zone edges are straight lines in the plane, and the
  distance field is Euclidean). The drive-time recipe also accepts longitude/latitude (metric `'none'` or
  `'great-circle'`; the cell-outline isochrones need degrees). Both of its snappings then run great-circle:
  edges are ranked by great-circle distance and `maximumSnapDistance` is in meters, with no east-west bias at
  any latitude. Drive times stay in the units of `network.weights` (the snapped fraction splits the edge
  cost). For the other recipes, project with `createGPULocalFrame` and pass its `spatialContext` (planar,
  meters), so the distance-field `parameters` and the straight-line `distances` output are in meters.
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
  Optional local fits pass their `spatialContext` to `GPUGeographicallyWeightedRegression`; omitted means
  native planar, and fixed bandwidths use the context's length unit.
  Its `settings.significanceLevel` (default 0.05) is packed once into the autocorrelation `parameters`
  buffer and read per frame by both the specification search and the residual `GPULocalMoran` (it used to
  be a compile-time constant of the search kernel); `result.inference` and `result.setSettings` follow it.
- The fleet dwell recipes take `trajectories` (`GPUPackedGeometryPort<GPUTrajectoryGeometryPort>`) and polygon `zones`
  (`GPUPolygonGeometryPort`, `sourceIds` as stable zone IDs); the zone events variant passes the zones to
  `GPUZoneEvents` directly (no edge lists) with `trackEventCapacity` events kept per track. The
  straight-line catchments recipe takes `grid`, `parameters` and `facilities` (`GPUPointGeometryPort`,
  `sourceIds` as zone IDs); the cluster recipe takes `points` (`GPUPointGeometryPort`).
- The cluster recipe's optional `spatialContext` defaults to native planar and is returned as
  `result.spatialContext`. It is planar only because group summaries and convex hulls are planar:
  longitude/latitude throws, so project first. Clustering measures distance and rejects
  `local-lambert-equal-area` frames; `epsilon` uses the context's length unit.
- The cluster recipe drops clusters at or above `maximumClusterCount`, and the caller supplies the
  geometry parameters. The change-of-support recipe's `status.overflow` covers the overlap list only; the raster
  flags are returned as `status.rasterOverflow` and `status.rasterRequiredCount` (stages `source-rasterization` and
  `target-rasterization`).

### `addHotSpotAnalysisRecipe`

Bins observations onto Quadbin cells inside a study area and returns Getis-Ord Gi* z-scores and confidence bins per cell.

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=result caption="2,344 observations over 13 community areas; each analysed cell is coloured by its Gi* z-score."}

Points on a study area, or a raster, to Getis-Ord Gi* hot and cold spots.

Points source (breaking change from the occupied-cells design):

- Recipe `spatialContext` describes study-area polygons (default longitude/latitude, metric `'none'`,
  linear edges). For antimeridian crossings or great-circle edges, set `prepare: true` or prepare with
  `GPULonLatPolygonPreparation`. Preparation requires `preparedPositionCapacity` and a per-frame
  `preparationParameters` view from `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)`.
  It runs before cell cover and preserves feature rows; `status.stages` adds `study-area-preparation`
  with `overflow`, `requiredCount`, `invalidCount` and `approximation`. Preparation requires a points
  source and a longitude/latitude context. The cover's `invalidCount` counts features with unsplit
  non-pole antimeridian-crossing edges, not wide bounding boxes.
- `studyArea` (a `GPUPolygonGeometryPort` plus `candidateCapacity`, optional `coverCapacity`) and
  `cellCapacity` are required. The recipe covers the polygons with `GPUCellCover` (`'center'` containment),
  deduplicates and sorts the cells with `GPUCellAggregation`, aggregates the points the same way and joins them
  by binary search of each 64-bit key. Every study cell is a row, with value 0 when it holds no point, so the
  null mean is not inflated and clusters of empty cells can be cold spots. Points outside the study area are
  excluded; `result.status.pointsOutsideStudyArea` counts them. `result.table.cells` are the sorted study cells and
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
  described weights (`information.transform` `'row'` or `'binary'`, and the search status) feed `GPUHotSpotAnalysis`.
  Every source publishes the unclamped neighbor total as the `weights` stage `requiredCount` (the grid-disk
  adapter included), for `getGPUSpatialWeightsCapacityPlan`.
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
  neighbors'`); `result.permutationInference` describes the optional permutation confirmation. Both come from
  the contributors' `getInferenceMetadata` of the one `settings` object (`GPUHotSpotAnalysisRecipeSettings`:
  `significanceLevel`, optional fixed moments, `seed`, `permutationCount`) that the recipe packs into its
  `parameters` and `permutation.parameters` `GPUParameterBuffer`s; `result.setSettings` repacks them per frame.
- The lattice source keeps its queen/rook weights and gets the same guard and metadata.
- Permutation confirmation tests every study row; it does not apply `minimumNeighbors`.

#### How it computes

{/* lens:addHotSpotAnalysisRecipe */}

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addRateClusterMapRecipe`

Turns bird and total observation counts per tract into smoothed rates and significant local Moran clusters.

::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=result caption="Bird observations over all observations in 199 tracts; quadrants mark significant clusters."}

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
- `events` and `populations` accept `float32` or `uint32` (for example the `counts` of
  `addPointsInPolygonsChoroplethRecipe`); `uint32` columns are converted by one node (exact below 2^24).
- Rows are features: `polygons` is a `GPUPolygonGeometryPort` (four-level layout, one packed `positions`
  view), so multipart features get one row and the `counts` of `addPointsInPolygonsChoroplethRecipe` on the same
  features chain in directly. The `weights` stage publishes `requiredCount` for `getGPUSpatialWeightsCapacityPlan`.
- Settings have one source of truth. Pass `GPUParameterBuffer`s as `parameters` and `permutation.parameters`
  plus `settings: {significanceLevel?, seed?, permutationCount?}`: the recipe packs the settings into both buffers
  and builds `result.inference` from the same object, and `result.setSettings(settings)` repacks them per frame
  (an invalid setting throws and writes nothing). Caller-packed views are not accepted.
- `result.inference` is the `getInferenceMetadata` of the `GPULocalPermutationTest` the recipe built (or of
  `GPULocalMoran` without `permutation`). It holds the alternative, `permutationCount` (0 for analytic), `seed`,
  `significanceLevel` (default 0.05), `multipleTesting` and `testedRowsDefinition`. An invalid permutation
  count or seed throws when the recipe is built.

#### How it computes

{/* lens:addRateClusterMapRecipe */}

::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.rate-cluster-map-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addPeriodComparisonRecipe`

Counts two periods of observations per Quadbin cell and returns the classified, coloured change.

::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=result caption="655 spring and 566 autumn observations compared per cell; classes diverge from no change."}

Two periods of events to a classified, colored change map.

- `delta` (after minus before) is always defined. `ratio` and `percentChange` are NaN, not 0, for cells whose earlier
  value is 0; NaN rows are skipped by `GPUClassBreaks` and take the palette no-data color.
- `zScore` is a descriptive score of the change (`'poisson'`: `delta / sqrt(before + after)`, or `'standardized'`
  deltas). It has no null distribution and no multiple-testing correction, so a threshold on it is not a significance
  level; use `addHotSpotAnalysisRecipe` or `addRateClusterMapRecipe` for inference. With `measure: 'count'` every
  union cell has `before + after > 0`, so the Poisson denominator is never zero.

#### How it computes

{/* lens:addPeriodComparisonRecipe */}

::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.period-comparison-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addChangeOfSupportRecipe`

Moves tract observation counts onto community areas by raster overlap area and reports what each tract could not transfer.

::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=result caption="199 census tracts rasterized with 13 community areas; counts are redistributed by overlap area."}

Transfers values between two polygon zone systems by area: both systems are rasterized to one grid
(`GPUPolygonRasterization`), `GPUArealInterpolation` builds the overlap areas and both weight sets, and
`GPUSpatialLag` moves `sourceValues` to the targets as `extensiveValues` and `intensiveValues`.

- Props: `grid` (`GPURasterMetadata`), `parameters` (`getGPUPolygonRasterizationParameterValues(grid)`),
  `source` and `target` zone systems (`GPUPolygonGeometryPort` plus `crossingCapacity`),
  `pairCapacity`, optional `cellWeights`, `sourceValues` and `columnCount`, `categories`, `denominator`,
  `sourceAnchorCells`, `spatialContext` and `outputs` / `scratch` overrides.
- Denominators (default change): extensive `'zone'`, intensive `'overlap'`, so a target only partly
  covered by sources is the mean of what covers it. Pass `denominator: 'zone'` for the old intensive result.
- Conservation: the result always carries `sourceAreas`, `sourceTransferredFractions` and `status.invalidCount`;
  the last one is also in the `areal-interpolation` stage `status`. A source entirely outside the targets,
  or smaller than a cell, shows up there instead of silently losing mass; `sourceAnchorCells` rescues the
  latter.
- Coordinates: `spatialContext` omitted is native planar (cell counts are areas). `longitude-latitude`
  requires metric `'great-circle'` and declares a longitude/latitude raster whose `parameters` are in degrees.
  The areal stage weights each cell by its area on the authalic sphere in the context's length unit squared
  (square meters by default; square kilometers with `units: 'kilometers'`). It measures area, so
  `local-azimuthal-equidistant` frames throw. Rasterization itself only
  tests cell centers with straight polygon edges in degrees. For antimeridian crossings or great-circle
  edges, set `prepare: true` or prepare with `GPULonLatPolygonPreparation`.
  The resolved context is returned as `result.spatialContext`.
- `prepare` defaults to false and requires a longitude/latitude context, `preparedPositionCapacity`
  (applied separately to each zone system), and a per-frame `preparationParameters` view from
  `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)`. Preparation runs before each
  rasterization and preserves zone IDs. Stages `source-preparation` and `target-preparation` expose
  `overflow`, `requiredCount`, `invalidCount` and `approximation`.
- Accuracy is bounded by the raster resolution (cell centers decide membership), never an exact polygon
  intersection.

#### How it computes

{/* lens:addChangeOfSupportRecipe */}

::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.change-of-support-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addPointsInPolygonsChoroplethRecipe`

Counts points per polygon and classifies the counts into a coloured choropleth.

::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=result caption="2,344 observations counted in 199 tracts and shaded in five quantile classes."}

Points to a per-polygon statistic to class breaks to colors, with `backend: 'zonal'` (default,
`GPUZonalStatistics`) or `'group'` (`GPUPointInPolygonJoin` then dense `GPUGroupStatistics`). `statistic`
is `'count'` (default), `'sum'`, `'mean'`, `'minimum'` or `'maximum'`; features with no points keep NaN (mean,
minimum, maximum) or 0 (count, sum).

- Coordinates: `spatialContext` omitted is native planar. `longitude-latitude` is accepted because the
  mapped statistics are topological: containment uses straight polygon edges in degrees (not geodesics) and
  for antimeridian crossings or great-circle edges, set `prepare: true` or prepare with
  `GPULonLatPolygonPreparation`. No density is mapped, so no polygon area is needed; for
  a density per square meter use `GPUZonalStatistics` directly with spherical `areas`. Other spaces throw.
  The resolved context is returned as `result.spatialContext` (linear edges after preparation).
- `prepare` defaults to false and requires a longitude/latitude context, `preparedPositionCapacity`,
  and a per-frame `preparationParameters` view from
  `getGPULonLatPolygonPreparationParameterValues(edgeModel, settings)`. Both backends prepare before
  containment; feature rows and output counts retain their original IDs. Stage `polygon-preparation`
  exposes `overflow`, `requiredCount`, `invalidCount` and `approximation`.

#### How it computes

{/* lens:addPointsInPolygonsChoroplethRecipe */}

::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.points-in-polygons-choropleth-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addClusterAndOutlineRecipe`

Clusters points with DBSCAN and outlines each cluster with its convex hull, area and perimeter.

::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=result caption="2,344 observations clustered at 250 m with six points minimum; each cluster is outlined."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addClusterAndOutlineRecipe */}

::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.cluster-and-outline-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addFleetDwellRecipe`

Finds where vessels stop and totals the dwell time per anchorage zone.

::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=result caption="12 vessel tracks over 13 anchorage zones; zones are shaded by total stop time."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addFleetDwellRecipe */}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.fleet-dwell-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addSpatialRegressionRecipe`

Fits observations per tract on income and car-free households and tests the residuals for spatial dependence.

::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=result caption="199 tracts; residuals of the fit show where observations exceed or fall short of the model."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addSpatialRegressionRecipe */}

::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.spatial-regression-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}

### `addStraightLineCatchmentsRecipe`

Allocates every raster cell to its nearest facility and sums a value raster per catchment.

::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=result caption="150 facilities split the North Side into nearest-facility catchments on a 250 m grid."}

The chain of this recipe is listed in the chain table above.

#### How it computes

{/* lens:addStraightLineCatchmentsRecipe */}

::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=kernel caption="Read the kernels of the composed contributors, in dispatch order."}

::lens{scenario=gpu-spatial-analysis.straight-line-catchments-recipe rung=graph caption="Follow the contributor-grain command graph from inputs to outputs."}
