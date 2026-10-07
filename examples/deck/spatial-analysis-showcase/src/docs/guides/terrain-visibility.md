---
title: Visibility and summits
summary: Viewsheds, sight lines, skylines and summits from a DEM, how they relate, and which one to reach for.
order: 41
---

A digital elevation model answers two kinds of question. *What does the ground look like here?* (slope, shading, landforms) is local: each cell only needs its neighbours. *What can be seen from there?* and *which cell is the top of that mountain?* are different: the answer for one cell depends on terrain **far away**. The visibility family of contributors solves these on the GPU for every cell, or for a handful of chosen points, from the same elevation buffer.

All three stories in this chapter use one DEM: the Matterhorn, Gornergrat and the Gorner Glacier at Zermatt, 13.6 km square at 6.6 m ground resolution, stored as a Terrarium PNG in **Web Mercator** (see [Your own data](#/guide/your-own-data)). The Mercator pixel is 9.6 m wide but the ground size shrinks with the cosine of latitude, which matters in two ways described below.

## The tools and when to use which

| Question | Tool | Story |
| --- | --- | --- |
| Which cells can one observer see? | [`GPUTerrainViewshed`](#/reference/GPUTerrainViewshed) | [What can you see from Gornergrat?](#/story/viewshed) |
| Can A see B, and by how many metres? | [`GPUTerrainLineOfSight`](#/reference/GPUTerrainLineOfSight) | [Viewshed](#/story/viewshed) |
| How many of several observers see each cell? | [`GPUTerrainCumulativeViewshed`](#/reference/GPUTerrainCumulativeViewshed) | [Viewshed](#/story/viewshed) |
| What is the 360 degree skyline from a point? | [`GPUPointHorizonProfile`](#/reference/GPUPointHorizonProfile) | [Which peaks can I see from Gornergrat?](#/story/horizon) |
| Is each of these peaks visible, hidden or on the skyline? | [`GPUPointHorizonVisibility`](#/reference/GPUPointHorizonVisibility) | [Horizon](#/story/horizon) |
| Which bumps of a 1-D profile are peaks? | [`GPUProfilePeaks`](#/reference/GPUProfilePeaks) | [Horizon](#/story/horizon) |
| How much sky does every cell see? | [`GPUTerrainHorizon`](#/reference/GPUTerrainHorizon) | [Horizon](#/story/horizon) |
| Where are the summits, and what is a real one? | [`GPUTerrainSummits`](#/reference/GPUTerrainSummits) | [Where are the summits?](#/story/summits) |
| Which catalogue peak is which DEM summit? | [`GPUTerrainPeakSnap`](#/reference/GPUTerrainPeakSnap) | [Summits](#/story/summits) |
| Peaks, saddles and pits of the surface | [`GPUTerrainCriticalPoints`](#/reference/GPUTerrainCriticalPoints) | [Summits](#/story/summits) |
| Contour lines | [`GPUTerrainContours`](#/reference/GPUTerrainContours) | [Summits](#/story/summits) |
| Profile along a line, with gain and loss | [`GPURasterProfile`](#/reference/GPURasterProfile) | [Viewshed](#/story/viewshed) |

## Viewsheds: one observer, every cell

A viewshed is a test per cell. The kernel marches the straight line from the cell to the observer, one sample per pixel with bilinear heights, and keeps the largest slope `(h - c s² - zo) / s` it meets, where `zo` is the eye elevation and `c s²` the earth-curvature drop. If no sample rises above the line from the eye to the cell, the cell is **visible**; otherwise it is **hidden**.

Three details separate a usable viewshed from a toy:

- **Earth curvature and refraction.** The drop is `c d²` with `c = (1 - k) / 2R` and `R = 6371008.8 m`. The refraction coefficient `k = 0.13` is the geodetic default; GDAL's `gdal_viewshed -cc 0.85714` is the same model with `k = 1/7`. Over a 14 km window the drop is about 13 m at the far edge; over 100 km it decides whether a peak shows at all.
- **A tolerance band.** DEM heights have errors, so a cell that is hidden by 20 cm is not really hidden. With a tolerance of a few metres (mt-image uses 2 m plus 1 m per kilometre) a third class appears, **marginal**, between visible and hidden. A short stretch before the target is also ignored so a summit is not hidden by its own flank.
- **Heights.** The eye height and the height of what you want to see are separate parameters. A viewshed for a 2 m person differs from one for a 30 m tower standing on the same cells.

The line-of-sight contributor adds a number: the **clearance**, the metres by which the target could sink and stay visible (negative: how much taller it would have to be). The cumulative viewshed counts, per cell, how many of many observers see it; it is the analogue of GDAL's cumulative mode and the basis of scenic-exposure maps.

### Pitfall: Web Mercator is not metres

The planar contributors take one ground cell size. A Web Mercator raster does not have one: the pixel is 9.6 m on the page but 6.6 m of ground at 46 degrees north. These stories pass the ground size of the central row (the cosine of latitude changes by 0.1 percent across a 13.6 km window). For larger windows resample to a projected raster, or use a contributor with a Web Mercator model: the skyline contributors take a `'web-mercator'` projection and `worldPixelSize`, and the grid contributors take `cellSizeMode: 'web-mercator'` with the normalized y of the north and south edges.

## The min-max pyramid: faster, never different

`GPURasterExtremaPyramid` builds an exact min-max mip chain of the DEM. A ray can skip a whole block of samples when the block's maximum is below the line it is testing. The result is **bit-identical** to the exhaustive march (Tevs, Ihrke and Seidel 2008), and one pyramid is built once and shared by every consumer through the `pyramid` prop. How much it saves depends on the terrain, the observer and the GPU: on the 1024 × 1024 Matterhorn grid a viewshed ran about twice as fast with the pyramid in our test, and a skyline skips most of its samples. The viewshed story has a **Verify** button that runs both traversals and compares every output word, and a **Time** button that reports what the pyramid really buys.

## Skylines and peaks

A skyline is the viewshed turned inside out: from one eye, the highest elevation angle in every direction. `GPUPointHorizonProfile` casts a ray per azimuth and keeps the largest apparent tangent `(h - h_eye) / d - c d`; with the Web Mercator model rays follow great circles on the sphere over the world-pixel grid. `GPUProfilePeaks` treats the circular skyline as a 1-D profile and finds its peaks with a windowed prominence, a parabolic refinement and an exact greedy suppression. `GPUPointHorizonVisibility` then asks of each catalogue peak: is its angle above the highest angle met before it? It reports **hidden**, **marginal**, **visible**, and whether the peak touches the skyline, following the classifier of mt-image.

`GPUTerrainHorizon` does the horizon for every cell instead: horizon angles in up to 64 sectors, the **sky-view factor** `1 - mean(sin max(h, 0))` and openness. Use the point contributors when you need a long range for a few eyes, the grid contributor when you need a map of enclosure.

## Summits

A summit is not just a local maximum: on a glacier there are thousands. `GPUTerrainSummits` marks a cell when it is the highest point of a disc of a **ground** radius and stands a **minimum drop** above the highest cell on the disc's rim; the drop is a radius-limited lower bound of prominence. The radius is the scale of the question, so it is a parameter, not a fact. `GPUTerrainPeakSnap` then moves catalogue points onto those summits, rejecting moves that are too long, too different in height, or that end on the rim of the disc (a flank).

`GPUTerrainCriticalPoints` classifies every cell by the sign changes of the height difference around its ring: peak, pit, regular or saddle. The 8-neighbour ring is the classic Peucker and Douglas one; the 6-neighbour Freudenthal ring is the consistent triangulation whose counts obey the Euler relation. The summits story also uses a **summed-area table** (`addTerrainSummedAreaTableNodes`) to read the mean and spread of the height in any window in constant time, the machinery behind `GPUTerrainTopographicPosition`.

## Pitfalls

- **Scale.** Summits, critical points and saddles all depend on the DEM resolution and the radius. The stories run them on a 26.6 m average of the 6.6 m DEM and say so.
- **Cost.** A viewshed is O(cells x range) and a cumulative one multiplies that by the observer count; the stories run on a 2 x 2 average and re-run only when a parameter changes.
- **Surfaces.** A DEM has no trees, buildings or glacier change, so viewsheds from villages are optimistic.
- **Window edges.** Features beyond the data cannot be seen or found; the Liskamm flank in the south-east corner is cut.

## Next

Continue with the surface-derivative stories of the same chapter ([terrain-basics](#/story/terrain-basics) and its siblings), or open any class above in the [reference](#/reference).
