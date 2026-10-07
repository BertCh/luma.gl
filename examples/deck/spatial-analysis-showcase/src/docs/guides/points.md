---
title: Points and density
summary: Binning, clustering, point-pattern tests, region statistics and line density, and which one answers which question about where things are.
order: 5
---

## What this family does

A pile of points (wildlife sightings, shops, trips) raises the same few questions, and each has a GPU contributor that answers it without sending the points back to the CPU.

1. **How much is where?** Bin the points into cells and count them: [`GPUPointDensity`](#/reference/GPUPointDensity). Story: [Where does Chicago wildlife get noticed?](#/story/nature-density).
2. **Which groups of points form places?** Cluster them and describe each cluster: [`GPUSpatialClustering`](#/reference/GPUSpatialClustering) (DBSCAN), [`GPUKMeans`](#/reference/GPUKMeans), then [`GPUGroupGeometry`](#/reference/GPUGroupGeometry), [`GPUGroupConvexHull`](#/reference/GPUGroupConvexHull) and [`GPUGeographicDistribution`](#/reference/GPUGeographicDistribution). Story: [Where do sightings form hot spots?](#/story/nature-clusters).
3. **Is the pattern clustered, random or regular, and at what scale?** [`GPURipley`](#/reference/GPURipley), [`GPURipleyDistanceFunctions`](#/reference/GPURipleyDistanceFunctions) and [`GPUPointPatternIndices`](#/reference/GPUPointPatternIndices). Story: [Are the points clustered, random or regular?](#/story/point-patterns).
4. **What is inside the shape I just drew?** [`GPURegionStatistics`](#/reference/GPURegionStatistics), with [`GPURegionMask`](#/reference/GPURegionMask) and [`GPUPickRegionMask`](#/reference/GPUPickRegionMask) as alternative ways to produce the selection. Story: [What gets logged inside the area I draw?](#/story/lasso-explorer).
5. **How much line is where?** The same questions for streets and other lines: [`GPULineDensity`](#/reference/GPULineDensity) and [`GPULineLengthPerPolygon`](#/reference/GPULineLengthPerPolygon). Story: [How much street does each part of Chicago have?](#/story/street-density).

All five stories use real Chicago data: 43,557 iNaturalist observations of wild plants, animals and fungi from 2023, Overture places, the OpenStreetMap drive network, and the city's community areas and census tracts.

## Choosing between them

| You want to know | Reach for | It gives you |
| --- | --- | --- |
| Where activity is concentrated, as a surface | `GPUPointDensity` | A count, sum or mean per square or hexagonal cell, optionally smoothed |
| Hot spots as named places with outlines | `GPUSpatialClustering` + hulls | Labels, noise, per-cluster size, centre, hull |
| A forced partition into k groups | `GPUKMeans` | Labels and centres for every point, no noise |
| A single summary of a whole pattern | `GPUGeographicDistribution` | Mean and median centre, standard distance, standard ellipse |
| Whether the pattern departs from randomness | `GPURipley` and friends | K, L, G, F, J curves, Clark-Evans R, quadrat variance-to-mean |
| Numbers for a shape the user draws | `GPURegionStatistics` | Count, sum, mean, min, max, histogram, optional mask and id list |
| Street or line length per cell or polygon | `GPULineDensity`, `GPULineLengthPerPolygon` | Length, density, weighted length, segment count |

Density is a descriptive map; clustering and point-pattern statistics are *inferential* about structure; region statistics is interactive. They compose: a lasso can restrict a density, and a cluster label can feed a rate.

## Key options and what they cost

Every contributor here follows the compile-once pattern: counts, grid sizes and the *kind* of thing computed are compile-time, while the numbers an analyst steers are parameter-buffer writes.

- **Density.** Cell shape (square or hexagon), grid resolution and the statistic (`count`, `sum`, `mean`) are compile-time. The viewport bounds, hexagon radius, Gaussian sigma, the mask and the weights are buffer writes. A Gaussian is separable, so a horizontal and a vertical pass give the same field as the dense 2D kernel with far fewer taps.
- **Clustering.** DBSCAN `epsilon` and `minimumPoints` are per-frame; the dense-box shortcut and summation order are compile-time and never change the labels. K-means takes `k`, the iteration cap, tolerance, initialization and seed at compile time.
- **Point patterns.** The window, largest radius and edge corrections are per-frame; the number of radii, the quadrat grid and the reference lattice of F are compile-time. A mask selects the observation group without recompiling.
- **Region statistics.** The shape (a few numbers, or up to 256 lasso vertices) is per-frame. The shape *kind*, the selection path, the histogram bins and domain, the grid index and the optional mask and id outputs are compile-time, so each combination is its own cached graph.
- **Line density.** The grid origin and cell size are four numbers per frame; the coordinate system (planar or spherical), the grid size and the polygon set are compile-time.

## Pitfalls that matter for real data

- **Shared coordinates.** About one observation in six shares its exact coordinate with another, from repeat visits to one spot and reused map pins. Counts per cell are reliable; nearest-neighbour statistics are sensitive to the ties. The point-pattern story has a *jitter* control that shows how much Clark-Evans and G change when shared points are separated, while L(r) at larger scales hardly moves.
- **Density is not abundance.** Observations follow observers: a count says where people looked and logged what they saw, not how much life is there. Normalise by visitors or survey effort before comparing places.
- **The window matters.** Ripley's functions assume a stationary pattern in a rectangle. A city with a lake, parks and rail yards is not stationary, so "clustered" often means "inhabited". Use the map-view window to test a single neighbourhood and an edge correction at large radii.
- **One epsilon for all.** DBSCAN uses a single radius, so dense downtown and sparse outer areas cannot both be resolved. K-means assumes round, similar-sized groups.
- **Hulls and ellipses summarise, they do not prove.** A convex hull can swallow empty space; a standard ellipse always exists, even for a pattern that is not elliptical.
- **Shapes see what you ask for.** A world-space lasso selects what is under it on the ground; a screen-space lasso selects what is under it on screen. A pick selects only what is visible, so overlapping points hide each other.
- **Lines are not unioned.** Street length counts every line separately; overlapping lines count twice, and two-way streets are counted once here by choice.

## Reading the results

Most of these scenes colour a surface or points with a perceptually uniform ramp whose legend range is read back from the GPU once the camera settles. Curves (L minus r, G, F, J, histograms) are drawn as small bar charts in the readout panel. A readout that says *match* next to a CPU recount, as in the lasso story, is the GPU being checked against a slow reference on the same shape.

Try them in order: start with the density map to see the shape of the data, cluster one group of wildlife to name its hot spots, test whether the pattern is clustered at all, then draw your own lasso and compare hours of the day and months of the year.
