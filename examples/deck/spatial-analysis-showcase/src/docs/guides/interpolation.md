---
title: Interpolation and change of support
summary: Turning scattered samples into surfaces, and moving values between zone systems that do not line up.
order: 20
---

Two questions sit behind almost every map of something measured at a few places or reported in someone else's zones.

1. **Interpolation.** I have values at points (rain gauges, soil samples, air sensors). What is the value *between* them, and how sure am I?
2. **Change of support.** I have values for one set of zones (census tracts). I need them for another (hexagons, community areas, a school district). How do I move them?

The interpolation chapter has three stories: [rainfall interpolation](#/story/rainfall-interpolation), [change of support](#/story/change-of-support) and [dot density](#/story/dot-density).

## Points to a surface

| Tool | What it does | Reach for it when |
| --- | --- | --- |
| `GPUInverseDistanceWeighting` | Weighted average of nearby samples, weight `1 / d^p`. | You want a fast, assumption-light surface. |
| `GPUKriging` | Ordinary kriging: the k nearest samples solve a small linear system under a fitted variogram, and the kriging variance comes out too. | You need an uncertainty map, or the samples are clustered. |
| `GPUVariogram` | The empirical semivariogram `γ(h) = ½ · mean (zᵢ − zⱼ)²` over all pairs, in lag bins and direction sectors. | Before kriging, to see how quickly values stop resembling each other. |
| `GPUFocalStatistics` | Moving-window mean, min, max, range and standard deviation over a raster. | You want to smooth, find peaks or measure local roughness. |

**IDW** has two knobs. The *power* `p` sets how fast a sample's say fades with distance: `p = 0` is a plain average, `p = 2` is the usual default, and a large `p` snaps each cell to its nearest sample and leaves bullseyes. The *search radius* and *nearest k* decide who gets to vote. IDW can never predict outside the range of its samples, and gives no uncertainty.

**Kriging** replaces the fixed decay with the data's own. The variogram has three numbers: the **nugget** (noise even at zero distance), the **sill** (the variance that is spatially structured) and the **range** (how far samples stay similar). `GPUVariogram` computes the empirical curve on the GPU; the model (spherical, exponential or Gaussian) is fitted on the CPU to the few read-back bins and written into the kriging parameters, so changing it never recompiles. The prediction variance depends only on where the samples are, so the error map is a map of where a new sample would help.

Pitfalls:

- A variogram that never levels off (it keeps climbing past the sample variance) means a **regional trend**; kriging a trend with a stationary model gives a poor range. Shorten the maximum lag, or remove the trend first. Helene's rain has one.
- A random hold-out test flatters both methods when samples are dense and close together.
- Cells with too few samples inside the search radius are *no data*. That is a feature: it stops the map inventing rain over the ocean.
- `GPUKriging` is isotropic. The direction sectors of `GPUVariogram` tell you whether that is a fair assumption.

## Zones to zones

`addChangeOfSupportRecipe` chains `GPUPolygonRasterization` (both zone systems onto one fine raster), `GPUArealInterpolation` (how much of each source lies in each target) and `GPUSpatialLag` (the transfer itself). The weights are area shares:

- **Extensive** variables (counts, totals) are split: `w = a_st / A_s`, so mass is divided between targets, never duplicated.
- **Intensive** variables (densities, means) are averaged: `w = a_st / B_t`, the area-weighted mean of the sources a target overlaps.
- **Categories** get the share of each target covered by each class.

A *rate* is neither. The safe route is to transfer the numerator and the denominator as counts, then divide. Treating a percentage as if it were a count gives meaningless numbers, which the change-of-support story shows on purpose.

Options that matter:

- **Dasymetric weights** (`cellWeights`): every area becomes a sum of cell weights, so mass follows an ancillary raster (street density, land cover, building footprints) instead of being spread evenly over a zone.
- **Denominator** (`'zone'` or `'overlap'`): whether mass that falls outside the other system is lost (tobler) or conserved over the shared extent.
- **Target grid**: `GPUGridGenerator` makes hexagons, squares or triangles; the cell width and offset are per-frame parameters. Shift the grid and the numbers change though the data did not: the modifiable areal unit problem.

`GPUPycnophylactic` is the other route: instead of zones it builds a smooth density surface that preserves every source total (Tobler 1979). The number of iterations and the smoothing kernel are compile-time.

Accuracy is bounded by the raster. Cell centres decide membership, so a boundary is accurate to about one cell, and slivers smaller than a cell can be missed. Choose a raster whose cells are small against the smallest zone of either system, and read the Resolution readout in the story.

## Dots

`GPUDotDensity` draws `value × dotsPerUnit` dots per (zone, category) at random positions inside each polygon. Counter-based random numbers make each dot a pure function of (seed, zone, category, rank), so changing the dot value, or tying it to the zoom, only *adds or removes* dots at the end of each slot and never moves one. A weight raster (a mask) thins candidate positions, which is a dasymetric dot map. `GPURandomPointsInPolygon` is the same sampler for plain uniform points, for example a synthetic population or a Monte Carlo sample.

A dot is not a person's location: read the mixture and the density, not a single dot.

## Which to use

| I have | I want | Use |
| --- | --- | --- |
| Point samples of a continuous field | A surface | `GPUInverseDistanceWeighting`, or `GPUKriging` when uncertainty matters |
| Point samples | To know how far they correlate | `GPUVariogram` |
| A raster | Smoother, peaks, roughness | `GPUFocalStatistics` |
| Counts per zone | The same counts in other zones | `addChangeOfSupportRecipe` (extensive) |
| Rates or means per zone | The same in other zones | Transfer numerator and denominator, or the intensive rule |
| Counts per zone | A smooth density surface that keeps the totals | `GPUPycnophylactic` |
| Counts by category per zone | A picture of density and mix | `GPUDotDensity` |
| Polygons | Random points in them | `GPURandomPointsInPolygon` |

Related reading: [How contributors work](#/guide/how-contributors-work) explains the compile-once pattern all of these follow, and [Reading the maps](#/guide/reading-the-maps) covers legends and no-data cells. The reference pages for each tool are linked from the stories: [`GPUKriging`](#/reference/GPUKriging), [`GPUArealInterpolation`](#/reference/GPUArealInterpolation) and [`GPUDotDensity`](#/reference/GPUDotDensity).
