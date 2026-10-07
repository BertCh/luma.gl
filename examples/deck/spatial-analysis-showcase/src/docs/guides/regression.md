---
title: Regression, local models and regions
summary: When to use ordinary, spatial and geographically weighted regression, how SKATER builds contiguous regions, and how floating catchments and Huff areas measure access.
order: 6
---

## Four families, one question: what changes from place to place?

The tools in this chapter model a quantity (prevalence of a disease, access to food) as a function of other quantities, and then ask what the geography adds.

| Question | Tool | Story |
| --- | --- | --- |
| What explains the outcome, and are the leftovers spatial? | `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialErrorGM` | [Does the model leave a spatial pattern?](#/story/health-regression) |
| Does the relationship itself change across space? | `GPUGeographicallyWeightedRegression` and its Monte Carlo test | [Does the income-health link change across America?](#/story/local-relationships) |
| Which places are alike *and* adjacent? | `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, `GPURegionPartitionEvaluation` | [Can Chicago be divided into similar regions?](#/story/regionalization) |
| Who can reach a service, and where will they go? | `GPUCatchmentAccessibility`, `GPUHuffTradeAreas` | [Who can reach a grocery store?](#/story/trade-areas) |

## Global regression and what the residuals say

[`GPUOrdinaryLeastSquares`](#/reference/GPUOrdinaryLeastSquares) fits one line for the whole study area. It writes coefficients, standard errors, t statistics, R², AIC and BIC (GeoDa convention), the Jarque-Bera test of normal residuals and the Breusch-Pagan test of constant variance, plus residuals and fitted values per row. An optional **ridge penalty** is a one-float parameter buffer: it shrinks coefficients of overlapping predictors, so sliding it is a buffer write and nothing is recompiled.

Ordinary least squares assumes independent errors. On map data that is usually false. [`GPUSpatialRegressionDiagnostics`](#/reference/GPUSpatialRegressionDiagnostics) tests the residuals against a spatial weights matrix and returns the Lagrange multiplier tests LM-lag and LM-error, their robust versions, LM-SARMA and Moran’s I of the residuals (the same tests as spreg’s `LMtests` and `MoranRes`). [`addSpatialRegressionRecipe`](#/reference/addSpatialRegressionRecipe) wires OLS, those diagnostics and a residual local Moran map into one graph.

If the tests call for a spatial model:

- **Spatial lag**, `y = ρWy + Xβ + ε`, means a neighbour’s outcome spills into yours. [`GPUSpatialTwoStageLeastSquares`](#/reference/GPUSpatialTwoStageLeastSquares) estimates ρ with `WX` (optionally `W²X`) as instruments, like spreg `GM_Lag`.
- **Spatial error**, `u = λWu + ε`, means omitted factors are shared by neighbours. [`GPUSpatialErrorGM`](#/reference/GPUSpatialErrorGM) finds λ by a global scan of the generalized-moments objective, like spreg `GM_Error`.

The classical rule: if neither LM test is significant keep OLS; if one is, choose it; if both are, compare the robust tests. The weights you choose (queen, rook or k nearest neighbours) change every one of these numbers.

## Local regression: geographically weighted

[`GPUGeographicallyWeightedRegression`](#/reference/GPUGeographicallyWeightedRegression) fits a weighted regression around every location. Weights fall with distance through a bisquare or Gaussian kernel, and the **bandwidth** (a distance, or the k nearest rows) is chosen by the lowest corrected AIC over a ladder of candidates, as in mgwr and ArcGIS GWR. It reports local coefficients, local R², the hat-matrix diagonal, and the local condition number (mgwr’s `local_collinearity`: above 30 the local fit is unreliable).

Pitfalls to keep in mind:

- A small bandwidth is local but noisy; a large one approaches the global model. Always read the AICc scores of the ladder.
- Local coefficients vary even in data with no real spatial structure. [`GPUGeographicallyWeightedRegressionNonstationarityTest`](#/reference/GPUGeographicallyWeightedRegressionNonstationarityTest) permutes the observations, refits at the selected bandwidth and counts how often the shuffled surface varies as much as the real one.
- Correlated predictors make local coefficients unstable. Check the condition-number map before reading a coefficient map.
- A row mask lets you fit a subset (metro counties only, say); masked rows neither get a fit nor act as neighbours.

## Regionalization

Clustering groups similar places wherever they are. **Regionalization** adds the constraint that a region must be connected. [`GPUSpatialWeightsMinimumSpanningTree`](#/reference/GPUSpatialWeightsMinimumSpanningTree) builds the cheapest tree through the contiguity graph where an edge costs the squared attribute difference of its two ends. [`GPUSkaterRegions`](#/reference/GPUSkaterRegions) then makes greedy cuts, each removing the edge that most reduces the within-region sum of squares. Every region is a piece of the tree, so it is connected by construction.

The partition with k regions is the one with k+1 regions minus its last cut, so the compiled graph holds the whole cut log: the region count and the minimum region size are parameter writes. [`GPURegionPartitionEvaluation`](#/reference/GPURegionPartitionEvaluation) scores any labelling (explained variance, region sizes, the share of neighbour links that cross a boundary), so you can compare SKATER with [`GPUKMeans`](#/reference/GPUKMeans), which ignores geography and usually fragments into many disconnected pieces.

Use SKATER when regions must be operational (service districts, sampling strata); use plain clustering for statistical typologies. Standardize attributes, or the column with the largest units decides the regions. SKATER is greedy and a tree keeps only n-1 links, so it can miss the optimal partition.

## Access to services

A count of facilities within a radius ignores competition. **Floating catchment** methods fix that in two steps: each facility’s supply is divided by the demand that can reach it, then each demand location adds up the ratios of the facilities it reaches. [`GPUCatchmentAccessibility`](#/reference/GPUCatchmentAccessibility) does this for the 2SFCA and, with demand split by selection weights, the 3SFCA. Supply is conserved when the demand weights are the transpose of the facility weights. [`GPUHuffTradeAreas`](#/reference/GPUHuffTradeAreas) turns the same weights into a probability that a tract uses each facility, `p_ij = A_j^α w_ij / Σ A_k^α w_ik`, plus a modal facility (the trade area) and the expected demand per facility.

Both read weights written by [`GPUNeighborSearch`](#/reference/GPUNeighborSearch): the kernel, the bandwidth and the facility mask are parameters, so exploring catchments never recompiles a graph.

Things to check: the distances are straight lines unless you supply network costs; supply is whatever you give it (open data rarely has floor area or beds); and the answer depends on where demand is placed, here one point per tract.

## Compile once, explore freely

Every story in this chapter follows the library’s pattern: graphs compile once and sliders write parameter buffers. The choices that change the *shape* of a computation (the number of predictors, the instrument order of two-stage least squares, the contiguity criterion, whether the tree standardizes, the k of k-means, the grid index of GWR) are compile-time and show a rebuild badge in the panel; each variant is compiled the first time you choose it. Results stay in storage buffers that the map layers read directly. Only small summaries and per-row result columns are read back, once per change, for the readouts and tooltips.
