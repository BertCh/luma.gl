---
title: Raster analysis
summary: Band math, conditionals, reclassification, stretches, patches, weighted overlays, cell and zonal statistics, sampling, trend tests and flow drawing on raster grids, and which contributor answers which question.
order: 85
---

## What this family does

A raster is a grid of numbers with a place on Earth: a Sentinel-2 band, a DEM, a land cover class, a wind component. Most raster questions fall into a few kinds, and each has a small GPU contributor. **Local** operations treat every cell on its own (arithmetic, conditionals, reclassification). **Stack** operations look down through several layers at one cell (weighted overlay, cell statistics, change detection over time). **Region** operations group cells (connected patches, zones). **Sampling** reads a raster at arbitrary points. **Flow** operations turn a vector grid, such as wind, into pictures.

Every contributor is a node in a compiled command graph. You compile once; every slider afterwards is a write into a small parameter buffer, so the maps on this chapter's four stories respond at the frame rate. They run on real data around **Greenville, California**, where the 2021 Dixie Fire burned (Sentinel-2, ESA WorldCover and a 3DEP-derived DEM), and on NOAA GFS winds during **Hurricane Helene**.

## The tools and when to use which

| Question | Tool | Story |
| --- | --- | --- |
| Compute a band ratio, an index or a difference | [`GPURasterArithmetic`](#/reference/GPURasterArithmetic) | [Burn severity](#/story/burn-severity) |
| Where condition X, take A, else B (clouds, thresholds, masks) | [`GPURasterConditional`](#/reference/GPURasterConditional) | [Burn severity](#/story/burn-severity) |
| Bin values into classes with a break table | [`GPURasterReclassify`](#/reference/GPURasterReclassify) | [Burn severity](#/story/burn-severity) |
| Make a long-tailed raster look good | [`GPURasterStretch`](#/reference/GPURasterStretch) | [Burn severity](#/story/burn-severity) |
| How big and how long-edged is each patch? | [`GPURasterPatchMetrics`](#/reference/GPURasterPatchMetrics) | [Burn severity](#/story/burn-severity) |
| Remove salt-and-pepper specks | [`GPURasterSieve`](#/reference/GPURasterSieve) | [Burn severity](#/story/burn-severity) |
| Combine criteria into one suitability score | [`GPUWeightedOverlay`](#/reference/GPUWeightedOverlay) | [Post-fire priorities](#/story/site-suitability) |
| What do the layers say about each cell? | [`GPURasterCellStatistics`](#/reference/GPURasterCellStatistics) | [Post-fire priorities](#/story/site-suitability) |
| Read a raster at points | [`GPURasterSampling`](#/reference/GPURasterSampling) | [Post-fire priorities](#/story/site-suitability) |
| Summarise a raster by region | [`GPURasterZonalStatistics`](#/reference/GPURasterZonalStatistics) | [Post-fire priorities](#/story/site-suitability) |
| Is a trend over many dates real? | [`GPUChangeDetection`](#/reference/GPUChangeDetection) | [Vegetation trends](#/story/vegetation-trends) |
| Animated flow from a vector field | [`GPUParticleAdvection`](#/reference/GPUParticleAdvection) | [Wind flow](#/story/wind-flow) |
| The whole flow pattern as a texture | [`GPULineIntegralConvolution`](#/reference/GPULineIntegralConvolution) | [Wind flow](#/story/wind-flow) |
| Evenly spaced flow lines | [`GPUStreamlines`](#/reference/GPUStreamlines) | [Wind flow](#/story/wind-flow) |

## Local operations: index, condition, class

`GPURasterArithmetic` evaluates `op(a * scaleA + offsetA, b * scaleB + offsetB)` with a clamp. The operation (add, subtract, normalized difference, power, logarithm and more) is a parameter, so a **normalized difference** `(A - B) / (A + B)` is NDVI with near-infrared and red, and NBR with near-infrared and shortwave-infrared. Domain errors give NaN. `GPURasterConditional` is `where(condition, a, b)`; `GPURasterReclassify` finds the class of every value in a break table by binary search, with a left- or right-closed interval choice. All three treat NaN as no data.

**Pitfall.** A threshold like dNBR 0.27 is only meaningful for the index it was derived for. If you change the formula from a normalized difference to a plain ratio, the USGS classes no longer apply; the burn story warns about this and turns off its CPU comparison.

## Stretching and patches

`GPURasterStretch` finds the exact range, a histogram and its cumulative distribution on the GPU and then maps values through a linear, percentile or equalizing curve, optional gamma and a sigmoid. Its statistics window can follow the camera ("stretch to visible extent"). Percentile bounds are accurate to one histogram bin.

To measure patches, label a mask with `GPURasterConnectedComponents` and `GPURasterDenseComponents`, then `GPURasterPatchMetrics` gives area, perimeter and bounding box in ground units (the FRAGSTATS metrics) and `GPURasterSieve` removes or merges patches below a size. Connectivity (4 or 8 neighbours) and the sieve mode are compile-time. Because patches from one mask never touch, `remove` is the useful mode for them; `merge` is for segmentations in which labels touch.

## Stacks: overlay, statistics and trends

`GPUWeightedOverlay` scores each cell as `sum(weight x remap(value))` over up to 16 layers. A layer is remapped linearly (with `invert`) or through a break table; a NaN table value marks a **restricted** class that removes the cell. `GPURasterCellStatistics` summarises the same stack: mean, standard deviation, minimum, maximum, range and valid count. The standard deviation of the rescaled criteria is an audit: where it is high, the score hides a disagreement.

`GPUChangeDetection` works along time. For each cell it computes a two-date difference, a Welch t-test between dates before and after a split, the Theil-Sen slope and Mann-Kendall S, Z and p. Significance comes from either test (a compile-time choice). The tests drop NaN dates, so clouds and a user date window simply shorten the series.

**Pitfall.** A trend map tests every cell at once. At alpha 0.05 about one cell in twenty passes for pure noise; the contributor does not correct for multiple comparisons, and neighbouring cells are not independent.

## Regions and points

`GPURasterSampling` reads a raster at points with nearest, bilinear or bicubic (Catmull-Rom) interpolation and a strict or renormalizing nodata policy; the active point count is a parameter. `GPURasterZonalStatistics` summarises a raster by dense zone ids. Its `sumOrder` is compile-time: **sorted** sums are bitwise reproducible, **atomic** sums depend on accumulation order and are far slower when many neighbouring cells share a zone, which is the normal case for land cover. A zone id at or above the capacity sets an overflow flag.

## Flow

One velocity buffer can feed all three flow tools. `GPUParticleAdvection` moves particles with a midpoint step and a ring buffer of trail positions; `GPULineIntegralConvolution` smears noise along the flow; `GPUStreamlines` extracts evenly spaced polylines. Particle count, trail length, LIC taps and the streamline grid are compile-time; speed, drop rate, age, spacing, seeds and phase are parameters. The wind story integrates in degrees per second on the longitude and latitude grid, so the field is not resampled.

**Pitfall.** Particle paths are not air-parcel trajectories when the field changes in time, and a 0.25 degree model smooths a hurricane's eye wall.

## Reading the maps

Every legend shows the real value range, read back from the GPU once the controls settle. Hover for the value of the cell under the pointer. Wherever a CPU reference exists (the Dixie Fire means and fractions stored with the dataset, Mann-Kendall and Theil-Sen on a sample of pixels, burned share per land cover), the readouts show the GPU number next to it. For the shared compile-once-and-write-parameters pattern see [How contributors work](#/guide/how-contributors-work).
