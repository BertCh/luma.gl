---
title: Rates, classes and indices
summary: Smoothing noisy rates, classifying values for maps, measuring segregation and inequality, building composite scores, finding similar places and grouping tables, and which contributor answers which question.
order: 9
---

## What this family does

Most maps of statistics fail in the same few ways: a rate computed from three people, a color scale stretched by one outlier, an index whose weights nobody chose, or a "total" that hides how unevenly something is spread. The contributors in this chapter are the remedies. All of them work on columns of numbers per place (a county, a tract, a community area) and keep every intermediate result in GPU buffers, so changing an option recomputes the whole map in a frame.

1. **Is this rate real or small-number noise?** Smooth it with empirical Bayes: [`GPUEmpiricalBayesRates`](#/reference/GPUEmpiricalBayesRates) (shrink toward the pooled rate) and [`GPUSpatialEmpiricalBayesRates`](#/reference/GPUSpatialEmpiricalBayesRates) (shrink toward the neighbourhood), then test clusters with [`addRateClusterMapRecipe`](#/reference/addRateClusterMapRecipe). Story: [Traffic deaths without the small-number noise](#/story/rate-smoothing).
2. **How should these values be colored?** [`GPUClassBreaks`](#/reference/GPUClassBreaks) chooses class edges, [`GPUColorScale`](#/reference/GPUColorScale) paints them, [`GPUColumnQuantiles`](#/reference/GPUColumnQuantiles) trims outliers, [`GPUClassAssignment`](#/reference/GPUClassAssignment) and [`GPUClassificationFit`](#/reference/GPUClassificationFit) score the result, [`GPUBivariateClassification`](#/reference/GPUBivariateClassification) maps two variables and [`GPUColumnProfile`](#/reference/GPUColumnProfile) summarises a field. Story: [Choosing classes for a county map](#/story/choropleth-classes).
3. **How separated are groups?** [`GPUSegregation`](#/reference/GPUSegregation) computes dissimilarity, entropy, isolation, interaction and Atkinson indices, globally, locally and across scales. Story: [How segregated is Chicago, and at what scale?](#/story/segregation).
4. **How do I combine indicators, and how unequal is a distribution?** [`GPUCompositeScore`](#/reference/GPUCompositeScore) builds an index; [`GPUInequality`](#/reference/GPUInequality) measures Gini, Theil, Atkinson, Hoover, Palma and the Lorenz curve per zone. Story: [A vulnerability index you can reweight](#/story/vulnerability-index).
5. **Which places are like this one?** [`GPUSimilarLocations`](#/reference/GPUSimilarLocations). Story: [Which counties are most like yours?](#/story/similar-places).
6. **Group by and join?** [`GPUGroupStatistics`](#/reference/GPUGroupStatistics) and [`GPUKeyJoin`](#/reference/GPUKeyJoin). Story: [Wildlife by community area, joined to who lives there](#/story/group-statistics).

## Choosing between them

| You want to know | Reach for | Key options |
| --- | --- | --- |
| A rate per place that is stable for small populations | `GPUEmpiricalBayesRates` | Events and population at risk; a mask |
| The same, borrowing from neighbours | `GPUSpatialEmpiricalBayesRates` | A weights CSR (queen, rook, kNN) |
| Whether a rate map has real clusters | `addRateClusterMapRecipe` | Permutations, tail, FDR, queen or rook |
| Class edges that suit the distribution | `GPUClassBreaks` | Method (eight), class count, histogram bins for Jenks |
| Colors from classes or a continuous scale | `GPUColorScale` | Scale type, palette, blending, clamp, log floor |
| Which classing fits best | `GPUClassificationFit` | GADF, GVF, ADCM, ADAM |
| Two variables on one map | `GPUBivariateClassification` | Classes per axis, value-by-alpha |
| How unevenly a group is spread | `GPUSegregation` | Scales, kernel, self weight, spatial form, Atkinson b |
| A weighted index from many columns | `GPUCompositeScore` | Scaler, aggregation, weights, directions |
| Inequality inside zones | `GPUInequality` | Atkinson epsilon, Palma cuts, Lorenz knots |
| Peers of a place | `GPUSimilarLocations` | Weights, standardisation, reference set |
| Statistics per category or area | `GPUGroupStatistics` | Statistics set, percentiles, variance kind |
| Attributes from another table | `GPUKeyJoin` | Left or inner, gather or aggregate |

## Rates: why the raw number lies

A county rate is `deaths / person-years`. A county with 60 residents that has one fatal crash has a rate five hundred times the national rate, and it will be the brightest county on the map. The cure is not to drop small places but to **shrink** their rates toward a reference in proportion to how uncertain they are: `r = w y + (1 − w) m` with `w = a / (a + m / b)`. Large populations keep their own rate; tiny ones fall back. The *global* version shrinks toward the national rate and the *spatial* version toward the rate of the county and its neighbours. Both match PySAL esda. The **standardized rate** (Assuncao and Reis) divides the excess by its own sampling error and is the right input for local Moran.

## Classes: a modelling choice, not a default

Equal intervals, quantiles and natural breaks answer different questions. Equal intervals preserve magnitudes and fail on skewed data. Quantiles show rank and split similar values. Natural breaks (Jenks) and head/tail breaks follow the shape of the distribution. The fit scores make the comparison measurable: **GADF** (absolute deviations explained) is the fairer score on heavy tails, **GVF** flatters schemes that isolate a single outlier. A percentile filter before classing removes the extremes that stretch the scale. Bivariate classing shows two variables at once, and value-by-alpha fades places whose data are unreliable.

## Indices: weights and scale are judgements

Segregation indices depend on the **scale** at which a neighbourhood is defined, so `GPUSegregation` returns a profile across a ladder of bandwidths; the shape of that profile says whether a pattern is local or regional. Composite scores depend on **scaler, direction and weights**, and the first principal component shows what weights the data would choose. Inequality indices depend on which tail you care about: Gini is middle-sensitive, Theil T top-sensitive, Atkinson tunable.

## Pitfalls

- **Denominators.** Per-resident rates are wrong where residents are not the people at risk (downtown, airports, tourist counties).
- **Zones are arbitrary.** Tracts, counties and community areas are modifiable units: a different partition gives different indices.
- **Estimates have errors.** ACS tract values carry margins of error; smoothing and ranking do not remove them.
- **Weights are not neutral.** Equal weights are an opinion. Show the weights next to the map.
- **Always show the legend.** Class edges and counts are part of the message.

## Using them in your own app

Every contributor takes **packed float32 columns** (one value per place) and writes caller-owned output buffers. Per-frame options such as weights, class counts, percentiles and the reference place are parameter-buffer writes, so interactive sliders cost nothing. Compile-time options (the contiguity criterion, the permutation tail, the kernel, the standardisation, the join kind) select among graphs; every scene here compiles the default variant up front and the others the first time they are chosen, which the options panel marks as a rebuild.
