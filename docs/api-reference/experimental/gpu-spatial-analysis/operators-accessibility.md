---
title: "GPU Spatial Analysis: Accessibility & change of support"
sidebar_label: "Accessibility & change of support"
description: "Accessibility & change of support contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Accessibility & change of support

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

### `GPUCatchmentAccessibility`

Scores each demand location by the supply-to-demand ratios of the facilities whose catchments reach it, with two-step floating catchment areas.

::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=result caption="14 libraries and 199 tract centroids: 2 km straight-line catchments, not travel time or network distance."}

Floating catchment accessibility on any `GPUSpatialWeights` (the `access` package `two_step_fca`).
`method: '2sfca'` or `'3sfca'`, with `supply`, `demand`, `facilityWeights` (row `j` lists the demand
points `i` with weight `f(cost)`), optional `demandWeights` (row `i` lists facilities; it defaults to
`facilityWeights`, which is only valid for symmetric self-join weights and is not checked) and the
caller-owned `output: {accessibility, ratios?, reachableFacilities?}` (`GPUCatchmentAccessibilityOutput`).

- 2SFCA: `R_j = S_j / sum_i P_i w_ji`, `A_i = sum_j w_ij R_j`.
- 3SFCA (Wan, Zou and Sternberg 2012; PySAL `access` `three_stage_fca`) applies the selection weight
  `G_ij = w_ij / T_i`, `T_i = sum_k w_ik` over demand row `i`, together with the decay `w`. The ratio step
  reads the facility weights, `R_j = S_j / sum_i P_i w_ji^2 / T_i`, and the score step the demand weights,
  `A_i = sum_j (w_ij^2 / T_i) R_j`. Both are `G_ij W_ij` only when the demand weights are the transpose of
  the facility weights. Demand rows with `T_i = 0` are skipped. Matches `access` 1.1.10 on network costs
  for binary, exponential and Gaussian decay.
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

A non-finite demand, or a non-finite ratio from a non-finite supply, is a missing value and adds nothing.
The operator reads weights, not positions, so it has no `spatialContext`; distance units and the coordinate
space (planar, or great-circle meters) are those of the weights' producer.
When both `facilityWeights` and `demandWeights` carry `spatialContext`, the contexts must be
compatible; otherwise construction throws.

This is the one 2SFCA/3SFCA kernel. Over a network cost matrix, turn the matrix into weights with
`GPUNetworkCostWeights`, or use the `catchment` option of
[`GPUNetworkAccessibility`](/docs/api-reference/experimental/gpu-network), which composes the two.

#### How it computes

{/* lens:GPUCatchmentAccessibility */}

::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=datum datum=22 caption="Inspect tract 17031020802, the most populous tract with 8,715 residents."}

::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=formula caption="Facility supply over catchment demand, then summed back to each tract."}

::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=kernel caption="Read the ratio and score kernels, one invocation per facility or tract."}

::lens{scenario=gpu-spatial-analysis.catchment-accessibility rung=graph caption="Follow the two radius searches, then the ratio and score passes."}

### `GPUHuffTradeAreas`

Estimates the probability that each demand location patronises each facility from facility attractiveness and distance-decay weights, and each facility's expected demand.

::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=result caption="Tracts shaded by modal-park probability: 102 parks weighted by area, inverse-square distance within 2.5 km."}

Huff probabilities `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik` from `attractiveness` and
`demandWeights`, with `alpha` from a one-element `parameters` view. The trade area of demand `i` is its
modal facility (lowest ID on ties, `GPU_HUFF_NO_TRADE_AREA` when none). Props: `attractiveness` (`float32`,
one row per facility), `demandWeights` (row `i` lists the facilities of demand `i`), optional `parameters`
(`float32`, `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` = 1 word written by
`getGPUHuffTradeAreasParameterValues({exponent})`; without the view `alpha = 1`), and the caller-owned
`output` (`GPUHuffTradeAreasOutput`) with the required `tradeArea` (`uint32` per demand row). Optional
outputs are `probabilities` (one per `demandWeights` slot, aligned with `demandWeights.neighbors`),
`tradeAreaProbability` (per demand row) and `expectedDemand`, `expectedDemand_j = sum_i P_i p_ij`, which
needs `facilityWeights` (the transpose of `demandWeights`) and `demand`. Like `GPUCatchmentAccessibility` it reads weights only and has no
`spatialContext`. When both `demandWeights` and `facilityWeights` carry `spatialContext`, the contexts
must be compatible; otherwise construction throws.

{/* parameters: GPUHuffTradeAreas generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUHuffTradeAreas` parameters

- Packer: `getGPUHuffTradeAreasParameterValues(settings?)`, writes `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUHuffTradeAreasParameterValues`:

| Field | Type | Description |
|---|---|---|
| `exponent` (optional) | `number` | Attractiveness exponent `alpha`, positive and finite. Defaults to 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUHuffTradeAreas */}

::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=datum datum=22 caption="Inspect tract 17031020802: its modal park and that park's probability."}

::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=formula caption="Park attraction times decay weight, normalised over the tract's reachable parks."}

::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=kernel caption="Read the trade-area kernel that totals each row and picks the modal park."}

::lens{scenario=gpu-spatial-analysis.huff-trade-areas rung=graph caption="Follow the two radius searches, then the trade-area, probability and demand passes."}

### `GPUArealInterpolation`

Transfers source-zone values to target zones by rasterized overlap area, with extensive or intensive area-share weights.

::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=result caption="199 tracts (719,651 residents) shared to 13 community areas by overlap area on a 100 m raster."}

Maximum: at most `getGPUArealInterpolationMaximumCellCount(device)` (`2^25 - 1` on a default device) grid cells; with `cellWeights`, `unweightedFastPath: false` or a longitude/latitude raster at most `GPU_AREAL_INTERPOLATION_MAXIMUM_WEIGHTED_PAIR_COUNT` (`2^24 - 1`) pair slots.

Area-weighted transfer between two zone systems (tobler `area_interpolate`, CARTO `ENRICH_POLYGONS`) on
a common fine raster. Rasterize both systems to the same grid with `GPUPolygonRasterization` (or any zone
raster) and pass the two zone views, so H3 or hexagon rasters work too.

- Props: `grid` (`GPURasterMetadata`; its `width * height` is the cell count), `sourceZones`, `targetZones`
  (`uint32`, one row per grid cell; a value at or above the count means no zone), `sourceZoneCount`, `targetZoneCount`, optional `cellWeights` (a dasymetric raster; area becomes a
  sum of weights, and non-finite or non-positive cells carry no area), `denominator` (per kind, see
  below), `mode` (`'extensive'` by default or `'intensive'`, which selects `output.weights.weights`),
  `categories` (`{sourceCategories, categoryCount}`) and the caller-owned `output`
  (`GPUArealInterpolationOutput`): `weights` (a CSR with `targetZoneCount + 1` offsets; the neighbor
  capacity is the maximum number of pairs), `status.overflow`, and the optional `status.requiredPairCount`,
  `alternateWeights` (the other normalization, from the same run), `areas` (raw overlap area per slot),
  the diagnostics below and `categoryShares` (`T x K`, required with `categories`).
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
- Mass-conservation diagnostics. `output.sourceAreas` (`sourceZoneCount` rows) is the raster area of each source
  (0 when it rasterized to no cell), `sourceTransferredFractions` is the overlap area over the extensive
  denominator (`1 - f` of the extensive mass is lost; with `'zone'`, the share of the source that lies
  in the target system; NaN without raster area), and `status.invalidCount` (one `uint32`) counts sources with no
  raster area, whose mass is lost. They are exact for the rasterized zones, not for the polygons.
- Sources smaller than a cell. `sourceAnchorCells` (`sourceZoneCount` cell indices; out of range means none)
  gives a zone without cells a fallback cell, for example the cell containing its label point. The claim
  is made in a transient copy of `sourceZones`, in ascending source order, and succeeds only if the cell
  belongs to no source or to one that keeps another cell. Remaining empty sources stay in `status.invalidCount`.
- Coordinates. Planar by default (areas are cell counts or `cellWeights` sums, which is an equal-share
  only on an equal-area raster). With `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
  'great-circle'}` and `parameters` (the per-frame `getGPUPolygonRasterizationParameterValues(grid)`
  view, the same view as `GPUPolygonRasterization.parameters`), each cell weighs its spherical area
  `R^2 dLon (sin(top) - sin(bottom))`, times `cellWeights`. `R` comes from `getGPUAuthalicSphereRadius`:
  the WGS84 authalic radius or `context.sphereRadius`, in the context's length unit. Pair `areas` and
  `sourceAreas` use that unit squared: square meters by default, square kilometers with
  `units: 'kilometers'` (previously still square meters).
  **Breaking:** longitude/latitude now requires `metric: 'great-circle'`; `'none'` throws.
  Declares `measure: 'area'`: an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.
  North-up grids (a negative packed cell height) are handled: the area of a stored row follows the mirrored row.
  Other spaces throw. Checked against a numeric integral of `R^2 cos(lat)` (`spherical_cell_area.py`).
- Categorical share `(t, k)` is the overlap area with sources of class `k` divided by the area of `t`
  covered by any source, so it sums to 1 where covered.
- On overflow the offsets clamp to capacity, `status.overflow` is 1 and `status.requiredPairCount` gives the size
  needed.
- `unweightedFastPath` (default on, used when there are no `cellWeights`) takes pair areas from run lengths
  and zone areas from counts instead of per-cell accumulation. Results are bitwise identical; it was 1.4x
  faster on the explorer's graph, about 35 percent on 384 by 384 zone denominators and about equal for the
  overlap denominator, where the two radix sorts dominate.
- Cell centers decide membership, so every area is a cell count and accuracy is about one cell along every
  boundary; slivers smaller than a cell can be missed. It is never an exact polygon intersection. f32 sums
  are exact up to 2^24 cells per zone. Results are deterministic.

Transfer the values with `GPUSpatialLag` over the area shares (its source count is the `values` row
count, `sourceZoneCount`; `columnCount` transfers several variables at once). The tobler `area_interpolate` extensive and intensive results match to 1e-5 on
grid-aligned zones.

```ts
graph.add(new GPUArealInterpolation({
  grid, sourceZones, targetZones, sourceZoneCount, targetZoneCount,
  output: {weights: {offsets, neighbors, weights}, alternateWeights, status: {overflow}}
}));
graph.add(new GPUSpatialLag({
  weights: {offsets, neighbors, weights}, values: population, output: populationOnTargets
}));
```

#### How it computes

{/* lens:GPUArealInterpolation */}

::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=datum datum=5 caption="Inspect community area row 5, Lake View, and its tract weights."}

::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=formula caption="Each tract's value is split by the share of its raster cells in each area."}

::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=kernel caption="Read the weight kernel that divides each overlap area by the source area."}

::lens{scenario=gpu-spatial-analysis.areal-interpolation rung=graph caption="Follow rasterization, pair sorting and run detection, then the transfer."}

### `GPUPycnophylactic`

Smooths zone totals into a non-negative raster surface whose cells sum back to each zone total.

::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=result caption="199 tract totals (719,651 residents) smoothed onto a 100 m raster; tract outlines for reference."}

Tobler pycnophylactic interpolation on a raster. Props: `grid` (`GPURasterMetadata`), `zones` (`uint32`; a value at or
above `zoneCount` is outside), `zoneCount`, `totals` (per zone), `maximumIterations` (0 to 512,
compile-time), `kernel` (`'rook'`, the default 4-neighbor mean, or `'box'`, a 3x3 mean including the cell)
and the caller-owned `output` (`GPUPycnophylacticOutput`) whose `mass` is the mass per cell. Each cell starts at `total / area(zone)`. Each iteration takes the focal mean over in-area
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
- Convergence. `maximumIterations` is the cap. A one-row `tolerance` (per frame) compares, after each
  iteration, the largest `|new - old| / (total / area(zone))` over all cells with it. Once at or below
  the tolerance the surface is final: the per-cell kernels of every remaining compiled iteration are
  skipped through a GPU-written indirect dispatch, and the surface is not copied forward (one settle pass
  copies the scratch raster only when the last iteration that ran wrote it). Deterministic.
  `output.status.converged` and `output.status.iterationCount` (one `uint32` each; they need `tolerance`) report
  whether it converged and how many iterations ran; with 0 iterations both are 0. A NaN or negative tolerance never converges.
- Zones without cells. A zone smaller than a cell has no raster cell and its total used to vanish
  silently. `zoneAnchorCells` (`zoneCount` cell indices) lets such a zone claim a fallback cell (see
  `GPUArealInterpolation.sourceAnchorCells`; ascending zone order, the cell's owner must keep one cell),
  and `output.status.invalidCount` counts zones that still have no cell or no area.
- Coordinates. With a `longitude-latitude` `spatialContext` plus `parameters` (as above; north-up grids are handled) the smoothed
  quantity is density per context length unit squared (per square meter by default, per square kilometer
  with `units: 'kilometers'`, previously still per square meter). Cell areas use
  `getGPUAuthalicSphereRadius` in the context's length unit, and totals are enforced on density times the spherical
  cell area, so `output.mass` is mass per cell (density times area) and the optional `output.density`
  view is the surface itself. Without it every cell counts the same, which is only right on an equal-area raster.
  **Breaking:** longitude/latitude now requires `metric: 'great-circle'`; `'none'` throws.
  Declares `measure: 'area'`: an azimuthal-equidistant local frame throws; project into a Lambert equal-area frame.
  The focal mean is still by cell index, so away from the equator it is anisotropic on the ground.
- Limits. Counts are f32 sums, exact to 2^24 cells per zone; a larger zone loses count precision (the
  mass still holds to f32 accuracy). Oracles: a float64 numpy reimplementation of the documented
  algorithm including tolerance and anchors (`pycnophylactic.py`).

#### How it computes

{/* lens:GPUPycnophylactic */}

::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=datum datum=3576 caption="Inspect the cell containing the centroid of tract 17031020802."}

::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=formula caption="One iteration: smooth, restore the zone total, clamp at zero, rescale."}

::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=kernel caption="Read the smoothing, restore and rescale kernels of one iteration."}

::lens{scenario=gpu-spatial-analysis.pycnophylactic rung=graph caption="Follow the zone sort, then up to 30 smoothing iterations gated by the tolerance."}

### `GPUSegregation`

Computes aspatial and spatial segregation indices from a unit-by-group count table, with per-unit contributions.

::lens{scenario=gpu-spatial-analysis.segregation rung=result caption="Five population groups across 199 tracts; tracts shaded by their contribution to the entropy index H."}

Segregation indices from a unit-by-group count table, aspatial or spatial (PySAL `segregation`).

- Props: `groupCount` `K` (2 to 16), `groupCounts` (`units * K` float32, row-major; the unit count is
  `groupCounts.length / K`),
  `scales` (default `[null]`; `null` is aspatial, otherwise one square `GPUSpatialWeights` per bandwidth,
  which gives the multiscale profile), `selfWeight` (1), `atkinsonInequalityAversion` (`b`, 0.5, in
  `(0, 1)`), `spatialForm` and `output`: `indices` (`scales * stride` floats, layout from `getGPUSegregationLayout(K)`: entropy,
  multi-group dissimilarity, diversity, dissimilarity `[K]`, isolation `[K]`, Atkinson `[K]`, interaction
  `[K * K]`) and optional per-unit views (`localEnvironment`, `localEntropy`, `localDissimilarity`,
  `localTheil`, per scale; they sum to the global indices).
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
  count-table resampling kernel. `atkinsonInequalityAversion` and `selfWeight` are compile-time.

#### How it computes

{/* lens:GPUSegregation */}

::lens{scenario=gpu-spatial-analysis.segregation rung=datum datum=22 caption="Inspect tract 17031020802: its diversity and contribution to H."}

::lens{scenario=gpu-spatial-analysis.segregation rung=formula caption="Each tract's share of H: population times the diversity gap to the whole."}

::lens{scenario=gpu-spatial-analysis.segregation rung=kernel caption="Read the per-tract term kernel that also writes local entropy and Theil."}

::lens{scenario=gpu-spatial-analysis.segregation rung=graph caption="Follow the totals, environment, block reductions and index finalization."}
