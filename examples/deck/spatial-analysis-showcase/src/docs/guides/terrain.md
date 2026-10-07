---
title: Terrain surfaces
summary: Decoding elevation tiles, slope and ruggedness, relief shading, landform classification and sun and shadow, and which contributor answers which question.
order: 40
---

## What this family does

An elevation model is a grid of heights, and almost every useful question about terrain is a small computation on that grid. *How steep is it?* needs the neighbours of each cell. *How should I draw it?* needs light. *Is this a ridge or a valley?* needs a rule and a scale. *When does the sun arrive?* needs the horizon in every direction. The terrain contributors answer these on the GPU, for every cell, from one elevation buffer, so the sliders in the stories are parameter writes and the analysis never leaves the GPU.

The four stories of this chapter share one DEM: the Matterhorn, Gornergrat and the Gorner glacier at Zermatt, 13.6 km square, 2048 x 2048 pixels at 6.6 m on the ground. The viewshed, horizon and summit stories (see [Visibility and summits](#/guide/terrain-visibility)) use the same tile.

## The tools and when to use which

| Question | Tool | Story |
| --- | --- | --- |
| Turn a Terrarium or Mapbox PNG into meters | [`GPUTerrainRGBDecode`](#/reference/GPUTerrainRGBDecode) | [How steep is the Matterhorn?](#/story/terrain-basics) |
| Fix +/-256 m errors in a noisy tile | [`GPUTerrainSpikeRepair`](#/reference/GPUTerrainSpikeRepair) | [Terrain basics](#/story/terrain-basics) |
| Slope, aspect, hillshade | [`GPUTerrainDerivatives`](#/reference/GPUTerrainDerivatives) | [Terrain basics](#/story/terrain-basics) |
| Is it rugged or merely steep? | [`GPUTerrainRuggedness`](#/reference/GPUTerrainRuggedness), [`GPUTerrainVectorRuggedness`](#/reference/GPUTerrainVectorRuggedness) | [Terrain basics](#/story/terrain-basics) |
| One light, many lights, or no light | [`GPUReliefShading`](#/reference/GPUReliefShading), [`GPUTextureShading`](#/reference/GPUTextureShading) | [Seeing terrain](#/story/relief-visualization) |
| How enclosed is each cell? | [`GPUTerrainHorizon`](#/reference/GPUTerrainHorizon) (sky-view factor, openness) | [Seeing terrain](#/story/relief-visualization) |
| Small bumps on a gentle surface | [`GPUSimpleLocalRelief`](#/reference/GPUSimpleLocalRelief), [`GPUMultiScaleRelief`](#/reference/GPUMultiScaleRelief), [`GPULocalDominance`](#/reference/GPULocalDominance) | [Seeing terrain](#/story/relief-visualization) |
| Combine several relief layers | [`GPUReliefBlend`](#/reference/GPUReliefBlend) | [Seeing terrain](#/story/relief-visualization) |
| Ridge, hollow, valley | [`GPUGeomorphons`](#/reference/GPUGeomorphons) | [Landform classification](#/story/landforms) |
| How does the surface bend? | [`GPUTerrainCurvature`](#/reference/GPUTerrainCurvature) | [Landforms](#/story/landforms) |
| Above or below the surroundings, at which scale? | [`GPUTerrainTopographicPosition`](#/reference/GPUTerrainTopographicPosition) | [Landforms](#/story/landforms) |
| Ten landform classes from two scales | [`GPUTerrainWeissLandforms`](#/reference/GPUTerrainWeissLandforms) | [Landforms](#/story/landforms) |
| Where is the sun? | [`GPUSolarPosition`](#/reference/GPUSolarPosition) | [When does the sun reach Zermatt?](#/story/sun-and-shadow) |
| Is a cell in shadow right now? | [`GPUSolarShadowMask`](#/reference/GPUSolarShadowMask), [`GPUTerrainCastShadow`](#/reference/GPUTerrainCastShadow) | [Sun and shadow](#/story/sun-and-shadow) |
| Sun hours and clear-sky energy over a day | [`GPUSolarIrradiance`](#/reference/GPUSolarIrradiance) | [Sun and shadow](#/story/sun-and-shadow) |

## Reading an elevation tile correctly

**Decode first, then filter.** A Terrarium tile stores a height in three bytes (`R * 256 + G + B / 256 - 32768`). Never resample or blur the encoded image, because the bytes do not carry: the average of two colors is not the average of two heights. `GPUTerrainRGBDecode` reads the bytes exactly (it is bit-identical to a float64 decode, which the first story shows as a 0 m error) and marks anything outside a valid range, or with alpha 0, as *nodata* (NaN plus a validity flag) rather than drawing nonsense. Clamping heights below sea level to zero is an application choice that destroys real land below sea level, so it is opt-in.

**Cell size depends on the row.** The DEM is in Web Mercator, so a pixel is 9.55 m wide on the projected grid but only 6.6 m on the ground at 46 degrees north. The terrain contributors take `cellSizeMode: 'web-mercator'` with the equatorial pixel size and the normalized Mercator `northEdge` and `southEdge` of the tile, and scale each row by the cosine of its latitude. Passing 9.55 m as a uniform cell size would shrink every gradient by about 30 %. Contributors with a planar model (viewsheds) take the ground size of the central row instead; across a 13.6 km window the difference is 0.1 %.

**Noise is a property of the source.** Terrarium tiles from some browsers carry +/-256 m spikes; 8-bit DEMs show terraces; interpolated DEMs are smoother than the real surface. Slope, curvature and TPI amplify all of this because they are derivatives. Repair spikes first (the repair is opt-in and cannot tell a bad pixel from a real 250 m butte), and choose analysis scales that are larger than the data's noise.

## Choosing a relief visualization

No single image is best, because each answers a different question.

- A **hillshade** from one light is the familiar image, but it hides every face parallel to the light. The **multidirectional** version (GDAL `-multidirectional`, USGS MDOW weighting) lights each face from the best of several directions.
- **Texture shading** needs no light: it weights the spectrum of the elevation with `|f|^alpha`, so every ridge is bright and every groove dark whatever its direction. The `detail` option moves the emphasis from large landforms to fine texture.
- **Sky-view factor** and **openness** are properties of the horizon: how much sky a cell sees. They need a search radius, which is the main thing to tune; the march algorithm is cheap at short radii and the sweep wins near the tile size.
- The **Swiss relief** blends hillshade, sky-view, texture, curvature and an elevation tint with a light that swings toward each slope (Imhof's trick). **VAT** is the archaeologist's blend of hillshade, inverted slope, openness and sky-view. Both are one parameter write over layers computed elsewhere.
- **SLRM**, **MSRM** and **local dominance** show what is small relative to its surroundings (a moraine on a slope). They were built for 0.5 m lidar; on a 6.6 m alpine DEM they show moraines, rock-glacier lobes and avalanche tracks.

## Landforms depend on scale

A rock rib is a ridge at 100 m and part of a slope at 2 km, so every classification here has a scale that you choose, and changing it changes the map. Geomorphons (`searchRadius`, flatness angle) look along eight lines of sight; Weiss landforms use a small and a large topographic position; curvature depends on the derivative estimator and, for the ring form, the ring radii. **TPI** and **DEV** are evaluated at up to 64 scales from one summed-area table, so asking for many scales is nearly free, and **DEVmax** reports the scale at which each cell is most distinctive. Treat the classes as a way to look at the terrain, not as ground truth.

## Sun and shadow

Shadow needs the horizon in the direction of the sun. **`GPUTerrainHorizon`** computes it once per pixel and compass sector, which is the expensive step (134 MB for 16 sectors at 2048 x 2048 with 16-bit codes, 268 MB with float32). After that, **`GPUSolarShadowMask`** needs two reads per pixel to place any sun, so animating the day costs a few milliseconds a frame. If only one sun matters, **`GPUTerrainCastShadow`** sweeps along the sun azimuth and needs no map. **`GPUSolarIrradiance`** integrates a whole day of sun positions at once for sun hours and clear-sky insolation. The model sees terrain only: no trees, buildings or clouds, and a shadow cast by a peak outside the tile is missing near its edge.

## Pitfalls

- **Tile edges.** Windows and rays that leave the tile see nothing. Contributors report a required halo; pass tiles with it when mosaicking, and treat the outer ring of every product with suspicion.
- **Units.** Slope is degrees or percent, curvature is 1/m, TPI is meters, insolation is Wh/m2. The legends and tooltips of the stories carry the unit.
- **Percentile stretches** make images pretty but not comparable. Use a fixed range when you compare two maps.
- **Compile-time options** (search radii, window sizes, number of sectors, algorithms) rebuild one graph; the options panel marks them. Everything else is a parameter write.
