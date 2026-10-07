---
title: Hydrology and cost surfaces
summary: Flow routing, watersheds, stream order, flood height and wetness, terrain-steered wind, least-cost travel and contours, and which contributor answers which question.
order: 42
---

## What this family does

A digital elevation model (DEM) is a function of place, and almost every question about water, wind, access or the shape of the land is a question *about that function*: where does water go, how far is it from here to there when slope costs effort, where do the contour lines run. The contributors in this chapter answer those questions on the GPU, over rasters of a few million cells, with every slider a parameter-buffer write.

Four stories use them, on two real landscapes: the 32 km central **Grand Canyon** (Terrarium tiles, 15 m cells) and the **Dixie fire** window around Greenville, California (20 m cells with ESA WorldCover).

| You want to know | Contributor | Story |
| --- | --- | --- |
| How much land drains through each cell, and where the streams are | [`GPUTerrainFlow`](#/reference/GPUTerrainFlow) | [Where does the Canyon’s rain go?](#/story/watersheds) |
| Which basin a cell belongs to, for chosen outlets | [`GPUTerrainWatersheds`](#/reference/GPUTerrainWatersheds) | [watersheds](#/story/watersheds) |
| How big a stream is (Strahler order) | [`GPUTerrainStreamOrder`](#/reference/GPUTerrainStreamOrder) | [watersheds](#/story/watersheds) |
| How high a cell is above the nearest stream | [`GPUTerrainHeightAboveDrainage`](#/reference/GPUTerrainHeightAboveDrainage) | [watersheds](#/story/watersheds) |
| Where the ground stays wet, and where flow erodes | [`GPUTerrainHydrologicIndices`](#/reference/GPUTerrainHydrologicIndices) | [watersheds](#/story/watersheds) |
| How a wind bends around terrain | [`GPUTerrainFlowField`](#/reference/GPUTerrainFlowField) + [`GPUParticleAdvection`](#/reference/GPUParticleAdvection) | [Wind through the Canyon](#/story/terrain-flow) |
| What it costs to reach every cell, and the cheapest route | [`GPUCostDistance`](#/reference/GPUCostDistance) + [`GPUCostDistancePath`](#/reference/GPUCostDistancePath) | [The easiest way down](#/story/least-cost-paths) |
| How far every cell is from the nearest seed, and which seed | [`GPUDistanceField`](#/reference/GPUDistanceField) | [least-cost paths](#/story/least-cost-paths) |
| Lines, filled bands and polygon rings at chosen levels | [`GPUIsolines`](#/reference/GPUIsolines), [`GPUIsobands`](#/reference/GPUIsobands), [`GPUIsobandRings`](#/reference/GPUIsobandRings) | [Contours, bands and rings](#/story/contours) |

## Drainage: fill, route, accumulate

`GPUTerrainFlow` is the workhorse. It **fills depressions** so that every cell has a downhill way out (Planchon-Darboux relaxation), optionally **resolves flats** (Barnes, Lehman and Mulla), **routes** each cell to its lower neighbours and **accumulates** the area of everything upstream. Everything downstream of it uses its outputs.

- **Routing** is a compile-time choice: `d8` (steepest neighbour, thin crisp channels), `d-infinity` (Tarboton, flow split between two neighbours by angle) and the multiple-flow-direction family `mfd-freeman` and `mfd-quinn` (flow shared among all lower neighbours, spreading over benches). Pick D8 when you need a clean network and stream order; pick a spreading routing when you care about the distribution of wetness.
- **Accumulation units** are cells or ground area in square metres. With area, the **stream threshold** is a real drainage area (a tenth of a square kilometre is a small gully in this canyon), and it is a per-frame parameter.
- **Cell size mode** matters for Web Mercator tiles: a Terrarium pixel is 19.1 m on the sphere but 15.4 m on the ground at this latitude. Use `cellSizeMode: 'web-mercator'` with the north and south edges of the window and every contributor evaluates ground spacing per row.

The relaxations are iterated by the GPU and stop when nothing changes. The *Converged* readout reports iterations used out of the compile-time limit; if it says no, raise `maxFillIterations` or `maxAccumulationIterations`.

## Products on the filled surface

- **Watersheds** label every cell with its nearest downstream pour point, so nested outlets give nested basins; with no pour points each outlet gets a basin.
- **Strahler order** ranks the stream network: order rises when two streams of equal order meet.
- **HAND** (height above nearest drainage) is the elevation difference to the stream cell you drain to; thresholding it at a stage gives a screening flood map with no hydraulic model. It depends on which cells count as streams, so it moves with the stream threshold.
- **Hydrologic indices** combine specific catchment area `a` with slope: wetness `ln(a / tan β)`, stream power `a tan β`. Use the minimum-slope option to keep flat cells finite.

## Wind and particles

`GPUTerrainFlowField` projects one uniform wind onto the tangent plane of the DEM (`v = w − (w · g) g / (1 + |g|²)`), a single stateless kernel; `GPUParticleAdvection` moves tens of thousands of particles through any velocity field with RK2 steps and writes a trail ring buffer the layer draws directly. The same pair serves wind data (see the raster chapter); here the field is idealised, so read it as where terrain *can* steer air, not as a forecast.

## Cost, paths and distance

`GPUCostDistance` needs a **friction raster**: the cost of crossing one metre of each cell. Build it from slope (Tobler’s hiking function in the story), land cover multipliers (a small table in a parameter buffer) and barriers (NaN friction). The contributor accumulates cost from one or more starts, with optional **back-links**, **isochrone bands**, a **cost limit**, per-start head starts and a per-frame friction **scale and offset** so a pace slider re-prices the map without a recompile. `GPUCostDistancePath` follows the back-links from any target in one GPU thread, which is why a draggable destination is free.

`GPUDistanceField` is the unweighted partner: exact Euclidean distance (separable Felzenszwalb-Huttenlocher) or a jump-flood preview, with allocation to the nearest seed. Seeds can be points or a raster mask such as "all water cells".

When to use which: use **distance field** when only geometry matters (nearest hydrant), **cost distance** when the ground resists (travel time, spread, corridors). Comparing the two shows the *detour* that terrain forces.

## Contours

`GPUIsolines` extracts marching-squares segments (optionally stitched into polylines), `GPUIsobands` the filled bands between levels as triangles, and `GPUIsobandRings` the band boundaries as closed shell and hole rings ready for export. All three read the same raster and the same level buffer; intervals, a band window and the palette are parameter writes. Capacities are compile-time and flagged in the readouts.

## Pitfalls

- **Resolution and quantisation.** Terrarium is 0.5 m in elevation and often resampled; flats and spurious pits are normal. Fill and flat resolution exist for that reason; do not read single-cell features as real.
- **Edges.** Water that leaves the window is not routed further, and watersheds touching the window edge are cut off.
- **Models, not measurements.** Tobler’s function is symmetric here, cliffs are a slope threshold, HAND is not a hydraulic model and the wind field has no pressure. State the model when you show the map.
- **Raster orientation.** Contributors that assume row 0 at the minimum y (distance field, isolines, particles) need the raster flipped or an extent with negated y; the stories show both patterns.

## Next steps

Open [Where does the Canyon’s rain go?](#/story/watersheds) to see the drainage products on one DEM, or [The easiest way down](#/story/least-cost-paths) to drag a destination and watch the cheapest route re-snap.
