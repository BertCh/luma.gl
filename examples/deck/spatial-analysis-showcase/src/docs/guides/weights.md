---
title: Spatial weights and autocorrelation
summary: Who counts as a neighbour, whether similar values cluster, and where the clusters are, from weights matrices to Moran's I, Gi* and local permutation tests.
order: 8
---

## What this family does

Almost every spatial statistic starts from a **weights matrix** `W`: for each place, who its neighbours are and how much each counts. Three questions follow, and each has a family of GPU contributors.

1. **Who is my neighbour?** Build `W` from shared borders, from distances, or from a grid: [`GPUContiguityWeights`](#/reference/GPUContiguityWeights) (queen and rook), [`GPUNeighborSearch`](#/reference/GPUNeighborSearch) (k nearest and distance bands, with kernel weights) and [`GPULatticeWeights`](#/reference/GPULatticeWeights). Then rescale, combine and check it with [`GPUSpatialWeightsTransform`](#/reference/GPUSpatialWeightsTransform), [`GPUSpatialWeightsAlgebra`](#/reference/GPUSpatialWeightsAlgebra), [`GPUSpatialWeightsSummary`](#/reference/GPUSpatialWeightsSummary) and [`GPUSpatialWeightsTranspose`](#/reference/GPUSpatialWeightsTranspose), and describe what surrounds each place with [`GPUSpatialLag`](#/reference/GPUSpatialLag) and [`GPUNeighborhoodSummary`](#/reference/GPUNeighborhoodSummary). Story: [Who is my neighbour?](#/story/spatial-weights).
2. **Is there a pattern at all?** One number for the whole map, with its significance: [`GPUGlobalSpatialStatistics`](#/reference/GPUGlobalSpatialStatistics) (Moran's I, Geary's C, Getis-Ord General G, bivariate Moran and join counts), a simulated reference from [`GPUGlobalPermutationTest`](#/reference/GPUGlobalPermutationTest), and the distance at which it fades from [`GPUSpatialCorrelogram`](#/reference/GPUSpatialCorrelogram). Story: [Is it clustered at all?](#/story/global-autocorrelation).
3. **Where is it?** A test for every place: [`GPUHotSpotAnalysis`](#/reference/GPUHotSpotAnalysis) (Getis-Ord Gi*), [`GPULocalMoran`](#/reference/GPULocalMoran) (clusters and outliers) and [`GPULocalPermutationTest`](#/reference/GPULocalPermutationTest), with [`addHotSpotAnalysisRecipe`](#/reference/addHotSpotAnalysisRecipe) chaining points, cells, neighbours, Gi* and permutation. Story: [Hot spots beyond chance](#/story/hot-spots).

The stories use real data: the 3,109 counties of the contiguous United States with CDC PLACES health measures, Chicago's 791 census tracts, and 43,557 Chicago nature observations (iNaturalist) from 2023.

## Choosing between them

| You want to know | Reach for | It gives you |
| --- | --- | --- |
| Which polygons touch | `GPUContiguityWeights` | Queen (shared vertex) or rook (shared edge) neighbours, exact |
| Neighbours that ignore borders | `GPUNeighborSearch` | k nearest or a distance band, binary, inverse-distance or kernel weights |
| Neighbours on a raster | `GPULatticeWeights` | Rook or queen cells up to a radius, with a mask for holes |
| A different weight scheme | `GPUSpatialWeightsTransform` | Row, binary, double, variance-stabilising, kernel, symmetrised |
| Neighbours of neighbours, unions, blocks | `GPUSpatialWeightsAlgebra` | Set operations, higher order, self weights, subgraphs, block weights |
| Is my `W` sane | `GPUSpatialWeightsSummary`, `GPUSpatialWeightsTranspose` | S0, S1, S2, islands, cardinalities, asymmetry |
| What surrounds each place | `GPUSpatialLag`, `GPUNeighborhoodSummary` | Lag, mean, median, spread, dominant category, entropy |
| One verdict on the whole map | `GPUGlobalSpatialStatistics` | Statistic, expectation, z and p |
| Is that verdict robust | `GPUGlobalPermutationTest` | `p_sim`, `z_sim`, the null histogram |
| At what distance | `GPUSpatialCorrelogram` | Moran's I and z per distance band, the first peak |
| Hot and cold spots | `GPUHotSpotAnalysis` | Gi* z-scores and 90, 95, 99 percent bins, optional FDR |
| Clusters and outliers | `GPULocalMoran` | Quadrants, z-scores, optional FDR |
| Significance without assumptions | `GPULocalPermutationTest` | Pseudo p-values and a significance mask |

Global statistics say *whether*; local statistics say *where*. A strong global I with no significant local places usually means weak, widespread structure; a few significant local places with a small global I means a small number of pockets.

## Key options and what they cost

All of these follow the compile-once pattern: shapes, statistics and tests are compile-time, while the numbers an analyst steers are parameter-buffer writes.

- **Weights.** The rule (queen or rook, k nearest, band, lattice), `k`, the snap tolerance of contiguity, the lattice radius and criterion, and each transform are compile-time, so each is compiled the first time you use it and cached. The distance band, the weight function and its kernel, the power, the distance floor and row standardisation inside the search are parameter writes.
- **Algebra.** The operation, the weight rule, the order and the cumulative flag are compile-time. The self weight is a per-row view and the subgraph mask a buffer, so both are writes.
- **Global tests.** The set of statistics, the permutation statistic and its tail are compile-time. The number of permutations and the seed are writes; the same seed reproduces the same null distribution exactly. The correlogram's band count and band mode are compile-time; the maximum distance and the variance assumption are writes.
- **Local tests.** The statistic (Gi*, Gi, local Moran), the gating (analytic, permutation, none), the tail, the neighbour limit and the false-discovery option are compile-time. Radius, significance level, permutations and seed are writes.

## Pitfalls that matter for real data

- **Weights are a modelling choice.** Queen and rook differ at corners; k nearest never leaves a place isolated but is not symmetric; a distance band gives a dense downtown many neighbours and a rural place none. Check *Islands*, *Asymmetric slots* and the neighbour count before trusting a statistic, and re-run with a different rule.
- **Islands.** Nantucket and San Juan County (Washington) touch no other county, and one Chicago tract (O'Hare) has no neighbour. Their local statistics are undefined; `GPUGlobalSpatialStatistics` reports the island count.
- **Row standardisation changes the question.** With `R` weights a lag is a weighted mean and Moran's I is the regression slope of the lag on the value; with binary weights a place with many neighbours counts more. Gi* is classically defined on binary weights with the place included; with row-standardised weights prefer Gi (exclude the place).
- **Many tests at once.** At the 95 percent level about one place in twenty is flagged by chance. Use the Benjamini-Hochberg option, and read a pseudo p-value as no finer than `1 / (permutations + 1)`.
- **The unit is not the process.** Counts per cell mix exposure with risk, and results move with the cell size or the boundaries (the modifiable areal unit problem). Nature observations follow where people look; model-based health estimates are smoothed.
- **Distances are flattened.** The scenes measure distance in Web Mercator metres around a local origin. That is adequate for choosing neighbours; use an equal-area projection for survey-grade distances.

## Reading the results

Maps of weights show lines between centroids; maps of local statistics use red for hot or high-high, blue for cold or low-low, light orange and blue for outliers, and gray for places that are indistinguishable from chance. Sparklines in the readout panel show the null distribution of a permutation test and the correlogram by distance. Compare the observed statistic with the range of the null distribution: an observed value far outside it is the strong evidence.

Try them in order: look at the neighbourhood of one county, test whether diabetes clusters at all, then find where, and change the weights at each step to see what the answer depends on.
