---
title: "GPU Terrain operators"
description: "Every GPU Terrain contributor: inputs, outputs, parameters and how it computes."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Terrain operators

<ExperimentalDocsTabs active="gpu-terrain-operators" />

### `GPUTerrainDerivatives`, `GPUTerrainContours`, `GPUTerrainViewshed`, `GPUTerrainLineOfSight`, and `GPUTerrainCumulativeViewshed`

Elevation-tile analysis composed from `gpu-raster` operators. `GPUTerrainDerivatives` computes
Horn slope, aspect, and hillshade (uniform, Web Mercator, or geographic cell sizes) with an optional
hillshade texture. `GPUTerrainContours` extracts marching-squares segments for several fixed or
per-frame levels with one aggregate overflow flag (`output.status.overflow`). Every contributor
takes a `grid` (`GPURasterMetadata`), reads `parameters`, and writes into an `output` bag. The grid
supplies the width, height, cell size and latitude band, so the packers take it as their second
argument. The three visibility contributors share one sight-line model, with `cellSizeMode` selecting
how grid world units become ground meters:

- `GPUTerrainViewshed` writes a `GPU_TERRAIN_VISIBILITY` code (`output.visibility`, or
  `output.visibilityTexture`) per pixel from one per-frame observer.
- `GPUTerrainLineOfSight` tests a batch of `[observerColumn, observerRow, targetColumn, targetRow]`
  pairs. Optional per-pair heights override the defaults. It writes `output.visibility` and can also
  write `output.clearance`: how many metres the target could sink and stay visible.
- `GPUTerrainCumulativeViewshed` makes observers the dispatch dimension, like the `gdal_viewshed`
  cumulative mode. It counts, per pixel, the observers that see it (`output.visibleCount`, optional
  `output.marginalCount`), with `observersPerDispatch` observers per node. It uses the packer of
  `GPUTerrainLineOfSight` (`getGPUTerrainLineOfSightParameterValues`) because the heights, distance,
  cell size, curvature and tolerance semantics are shared.

All three accept `heightReference` (default `'ground'`) and, for absolute observer heights,
`observerVerticalDatum`, checked against the elevation band's `verticalDatum`. Targets remain
ground-relative. See [Heights and datums](./concepts.md#heights-and-datums).

```ts
const grid = makeGPURasterMetadata({width, height, cellSize: 30});
graph.add(new GPUTerrainDerivatives({
  grid, elevation, parameters: parameters.importToGraph(graph), output: {slope, hillshade}
}));
parameters.write(getGPUTerrainDerivativesParameterValues({azimuthDegrees: 315}, grid));
```

`GPUTerrainContours` takes a required `grid` (its former optional `metadata`), which also sets
`pixelInterpretation` for the vertex offset, and reports the aggregate flag in
`output.status.overflow`. Each level keeps its own `vertices`, `status` (`count`, optional `requiredCount` and `overflow`) and
optional `draw` members.

Each contour level's optional `draw` record is rewritten on every encoding from the
capacity-clamped GPU segment count. `drawLayout: 'instanced'` (default) writes
`[verticesPerInstance, segments, 0, 0]` for renderers that read segment endpoints from storage
(`verticesPerInstance` defaults to 2; use 4 or 6 for quad-per-segment lines). `drawLayout:
'line-list'` writes `[2 * segments, 1, 0, 0]` for one non-instanced `line-list` draw over the
`vertices` column used as a vertex buffer.

The three sight-line contributors accept `cellSizeMode?: GPURasterCellSizeMode`:
`'uniform'` (default) means grid world units are ground meters on a projected planar grid;
`'geographic'` means a degree grid; `'web-mercator'` means an EPSG:3857 grid. The latter two
require an axis-aligned affine and use WGS84 ground meters for `maximumDistance`, curvature drop
`c·d²`, tolerance bands, `targetIgnoreDistance`, and clearance. Elevations and heights must use
meters too. A ray uses one constant ground cell evaluated at its middle row,
`0.5 * (observerRow + targetRow) + 0.5`. Along-ray scale change is ignored, so range boundaries
and curvature drops are approximate on long rays, especially at high latitudes; the error depends
on the latitude span and ray bearing.

Each sight line joins pixel positions at one-pixel steps with bilinear elevation and ignores
samples with an invalid corner. There is no azimuth input and no pixel-direction conversion.
Row order comes from the affine, so these contributors have no `rowDirection` prop. Holding the
ground cell constant along a ray preserves the monotone sample distances used by pyramid skipping.

- **Tolerance band.** `GPUTerrainViewshed` takes an optional per-frame `toleranceParameters`
  (`[toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction]`, all 0 by
  default, packed by `getGPUTerrainViewshedToleranceParameterValues`; `GPUTerrainLineOfSight` and
  `GPUTerrainCumulativeViewshed` carry the same four values inside their 12-float layout) that turns
  the hard cut into a band
  `β = (toleranceMeters + tolerancePerKilometer·D/1000)/D` around the target slope. Codes are
  `hidden` above `T + β`, `visible` at or below `T − β`, and `marginal` (4) in between.
- **Ignored final stretch.** Samples within `max(targetIgnoreDistance, targetIgnoreFraction·D)` of
  the target are ignored. This removes the self-occlusion speckle of targets just behind their own
  lip; mt-image uses `2 m + 1 m/km` and ignores `max(150 m, 2 %·d)`.
- **Traversal.** `traversal: 'pyramid'` builds a min-max
  [`GPURasterExtremaPyramid`](../gpu-raster/operations-analysis.md) and skips blocks that provably
  cannot change the code. Results are bit-identical to `'march'` (the default).
- **Shared pyramid.** Pass `pyramid` (a `GPURasterExtremaPyramidOutput`) to reuse one pyramid across
  `GPUTerrainViewshed`, `GPUTerrainLineOfSight`, `GPUTerrainCumulativeViewshed`,
  `GPUPointHorizonProfile` and `GPUPointHorizonVisibility`. It must cover the same grid with a
  bilinear footprint. Building one costs 0.3 to 0.9 ms, but the pyramid-skip kernels are still about
  2x slower than `'march'` for viewsheds, so prefer `'march'` there.

With the default `'uniform'` mode, all sight-line kernels retain their previous WGSL and variants.

Earth curvature and refraction lower terrain at ground distance `d` by `c·d²` with
`c = getGPUTerrainCurvatureCoefficient(k) = (1 − k)/(2R)`, R = 6371008.8 m. The default k = 0.13
is the geodetic standard used by mt-image. GDAL's `gdal_viewshed -cc 0.85714` is the same model
with `cc = 1 − k`, k = 1/7, so use `getGPUTerrainCurvatureCoefficient(1 / 7)` to match GDAL. The
difference matters little in practice. Changing k by ±0.1 moves a skyline by 0.011°, 0.022° and
0.034° at 50, 100 and 150 km, and k ± 0.3 (inversions) by 0.03–0.10°. GDAL and mt-image differ by
Δk ≈ 0.013, about 0.003° at 50 km. Range truncation (120 instead of 150 km: up to 0.28°) and DEM
bias (30 m is 0.034° at 50 km) dominate (mt-image measurements).

```ts
graph.add(new GPUTerrainLineOfSight({
  grid, elevation, pairs, parameters: parameters.importToGraph(graph), traversal: 'pyramid',
  output: {visibility, clearance}
}));
parameters.write(getGPUTerrainLineOfSightParameterValues({
  curvatureCoefficient: getGPUTerrainCurvatureCoefficient(1 / 7),
  toleranceMeters: 2, tolerancePerKilometer: 1, targetIgnoreDistance: 150, targetIgnoreFraction: 0.02
}, grid));
```

{/* parameters: GPUTerrainCumulativeViewshed generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainCumulativeViewshed` parameters

- Packer: `getGPUTerrainLineOfSightParameterValues(settings, grid)`, writes `GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainLineOfSightParameterValues`:

| Field | Type | Description |
|---|---|---|
| `observerHeight` (optional) | `number` | Eye height above terrain, or absolute eye elevation with `heightReference: 'absolute'`. In elevation-band units. Defaults to 1.7; set explicitly in absolute mode. |
| `targetHeight` (optional) | `number` | Height added to the target elevation. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground radius in metres; `<= 0` means unlimited. Defaults to 0. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/metres. Defaults to 0. |
| `toleranceMeters` (optional) | `number` | Constant tolerance band half-width in metres. Defaults to 0. |
| `tolerancePerKilometer` (optional) | `number` | Additional band half-width per kilometre of distance, in metres. Defaults to 0. |
| `targetIgnoreDistance` (optional) | `number` | Metres at the target end that are not tested. Defaults to 0. |
| `targetIgnoreFraction` (optional) | `number` | Fraction of the distance at the target end that is not tested. Defaults to 0. |

{/* /parameters */}

Compile-time prop: `cellSizeMode` defaults to `'uniform'`; use `'geographic'` for degrees or
`'web-mercator'` for EPSG:3857, with per-ray WGS84 ground-meter distances as described above.

Parameter layout: `[observerHeight, targetHeight, maximumDistance, cellSizeX, cellSizeY,
curvatureCoefficient, toleranceMeters, tolerancePerKilometer, targetIgnoreDistance,
targetIgnoreFraction, northEdge, southEdge]`.
The packer always writes cell sizes and row edges in grid world units, regardless of mode.
`northEdge` is the top edge of row 0 and `southEdge` the bottom edge of the last row, in latitude
degrees or EPSG:3857 y meters for the respective modes; point grids include the half-row shift.

{/* parameters: GPUTerrainDerivatives generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainDerivatives` parameters

- Packer: `getGPUTerrainDerivativesParameterValues(settings, grid)`, writes `GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainDerivativesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier applied to derivatives. Defaults to 1. |
| `azimuthDegrees` (optional) | `number` | Direction the light comes from, degrees clockwise from north. Defaults to 315. |
| `altitudeDegrees` (optional) | `number` | Light elevation above the horizon in degrees. Defaults to 45. |

{/* /parameters */}

{/* parameters: GPUTerrainLineOfSight generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainLineOfSight` parameters

- Packer: `getGPUTerrainLineOfSightParameterValues(settings, grid)`, writes `GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainLineOfSightParameterValues`:

| Field | Type | Description |
|---|---|---|
| `observerHeight` (optional) | `number` | Eye height above terrain, or absolute eye elevation with `heightReference: 'absolute'`. In elevation-band units. Defaults to 1.7; set explicitly in absolute mode. |
| `targetHeight` (optional) | `number` | Height added to the target elevation. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground radius in metres; `<= 0` means unlimited. Defaults to 0. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/metres. Defaults to 0. |
| `toleranceMeters` (optional) | `number` | Constant tolerance band half-width in metres. Defaults to 0. |
| `tolerancePerKilometer` (optional) | `number` | Additional band half-width per kilometre of distance, in metres. Defaults to 0. |
| `targetIgnoreDistance` (optional) | `number` | Metres at the target end that are not tested. Defaults to 0. |
| `targetIgnoreFraction` (optional) | `number` | Fraction of the distance at the target end that is not tested. Defaults to 0. |

{/* /parameters */}

Compile-time prop: `cellSizeMode` defaults to `'uniform'`; use `'geographic'` for degrees or
`'web-mercator'` for EPSG:3857, with per-ray WGS84 ground-meter distances as described above.

Parameter layout: `[observerHeight, targetHeight, maximumDistance, cellSizeX, cellSizeY,
curvatureCoefficient, toleranceMeters, tolerancePerKilometer, targetIgnoreDistance,
targetIgnoreFraction, northEdge, southEdge]`.
The packer always writes cell sizes and row edges in grid world units, regardless of mode.
`northEdge` is the top edge of row 0 and `southEdge` the bottom edge of the last row, in latitude
degrees or EPSG:3857 y meters for the respective modes; point grids include the half-row shift.

{/* parameters: GPUTerrainViewshed generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainViewshed` parameters

- Packer: `getGPUTerrainViewshedParameterValues(settings, grid)`, writes `GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH` (10) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges
- Packer: `getGPUTerrainViewshedToleranceParameterValues(settings)`, writes `GPU_TERRAIN_VIEWSHED_TOLERANCE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTerrainViewshedParameterValues`:

| Field | Type | Description |
|---|---|---|
| `observer` | `readonly [number, number]` | Observer `[column, row]` in pixel-center index space; fractional values allowed. |
| `observerHeight` (optional) | `number` | Eye height above terrain, or absolute eye elevation with `heightReference: 'absolute'`. In elevation-band units. Defaults to 1.7; set explicitly in absolute mode. |
| `targetHeight` (optional) | `number` | Height added to every target cell. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground radius in meters; `<= 0` means unlimited. Defaults to 0. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/meters: terrain at distance `d` is lowered by `c * d^2`. Defaults to 0. |

Settings of `getGPUTerrainViewshedToleranceParameterValues`:

| Field | Type | Description |
|---|---|---|
| `toleranceMeters` (optional) | `number` | Constant half-width of the tolerance band in metres of sight-line height at the target. mt-image uses 2. Defaults to 0. |
| `tolerancePerKilometer` (optional) | `number` | Additional band half-width per kilometre of target distance, in metres. mt-image uses 1. Defaults to 0. |
| `targetIgnoreDistance` (optional) | `number` | Samples closer than this many metres to the target are not tested (`viewPeaks` ignores the last `max(150 m, 2 %)` of the ray). Defaults to 0. |
| `targetIgnoreFraction` (optional) | `number` | Fraction of the target distance at the target end that is not tested. Defaults to 0. |

{/* /parameters */}

Compile-time prop: `cellSizeMode` defaults to `'uniform'`; use `'geographic'` for degrees or
`'web-mercator'` for EPSG:3857, with per-ray WGS84 ground-meter distances as described above.

Parameter layout: `[observerColumn, observerRow, observerHeight, targetHeight, maximumDistance,
cellSizeX, cellSizeY, curvatureCoefficient, northEdge, southEdge]`.
The packer always writes cell sizes and row edges in grid world units, regardless of mode.
`northEdge` is the top edge of row 0 and `southEdge` the bottom edge of the last row, in latitude
degrees or EPSG:3857 y meters for the respective modes; point grids include the half-row shift.

#### `GPUTerrainDerivatives`

Turns an elevation grid into Horn slope, aspect and hillshade with two Sobel passes and one fused per-cell kernel.

::lens{scenario=gpu-terrain.terrain-derivatives rung=result caption="Slope drawn over the elevation greys; the steepest cell is marked."}

#### How it computes

{/* lens:GPUTerrainDerivatives */}

::lens{scenario=gpu-terrain.terrain-derivatives rung=datum datum=2907 caption="One cell's 3×3 window: Sobel weights 1-2-1, divided by 8 cell sizes."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=formula caption="Horn slope with the z factor and the row-direction sign as the kernel applies them."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=kernel caption="The fused shade kernel: one invocation per cell reads both gradient buffers."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=graph caption="Two GPURasterGradient passes feed one shade node; no loop."}

#### `GPUTerrainContours`

Extracts contour segment endpoints from elevation at seven levels, 1,100–1,700 m, spaced every 100 m.

::lens{scenario=gpu-terrain.terrain-contours rung=result caption="Seven contour levels shown as endpoints; the GPU publishes no line offsets."}

#### How it computes

{/* lens:GPUTerrainContours */}

::lens{scenario=gpu-terrain.terrain-contours rung=datum caption="Inspect one marching square and its crossed edges."}
::lens{scenario=gpu-terrain.terrain-contours rung=formula caption="Edge interpolation clamps fractions and rescales overflowing height differences."}
::lens{scenario=gpu-terrain.terrain-contours rung=kernel caption="Classify, scan, scatter and publish segment counts for each level."}
::lens{scenario=gpu-terrain.terrain-contours rung=graph caption="Seven raster contour pipelines feed the aggregate overflow flag."}

#### `GPUTerrainViewshed`

Classifies terrain visibility from ridge cell (45, 49), with an eye 1.7 m above the DEM.

::lens{scenario=gpu-terrain.terrain-viewshed rung=result caption="Visibility codes over elevation; the observer at column 45, row 49 is marked."}

#### How it computes

{/* lens:GPUTerrainViewshed */}

::lens{scenario=gpu-terrain.terrain-viewshed rung=datum caption="Follow the samples between the ridge observer and one target cell."}
::lens{scenario=gpu-terrain.terrain-viewshed rung=formula caption="Compare target tangent with curvature-adjusted terrain tangents."}
::lens{scenario=gpu-terrain.terrain-viewshed rung=kernel caption="One invocation traces each target cell from the observer."}
::lens{scenario=gpu-terrain.terrain-viewshed rung=graph caption="Canonical elevation feeds the sight-line visibility kernel."}

#### `GPUTerrainLineOfSight`

Tests twelve observer-to-target pairs against terrain, returning visibility codes and clearance in metres.

::lens{scenario=gpu-terrain.terrain-line-of-sight rung=result caption="Twelve UTM sight lines coloured by visibility and target clearance."}

#### How it computes

{/* lens:GPUTerrainLineOfSight */}

::lens{scenario=gpu-terrain.terrain-line-of-sight rung=datum caption="Inspect the target and intervening samples for one pair."}
::lens{scenario=gpu-terrain.terrain-line-of-sight rung=formula caption="Clearance measures target height above the maximum intervening tangent."}
::lens{scenario=gpu-terrain.terrain-line-of-sight rung=kernel caption="Each pair marches its ray without early exit to retain exact clearance."}
::lens{scenario=gpu-terrain.terrain-line-of-sight rung=graph caption="The same caller-owned segments carry both GPU result columns."}

#### `GPUTerrainCumulativeViewshed`

Counts how many of six ridge observers can see each terrain cell.

::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=result caption="Darker cells are visible from more of the six ridge observers."}

#### How it computes

{/* lens:GPUTerrainCumulativeViewshed */}

::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=datum caption="Inspect one target cell across all six observers."}
::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=formula caption="Count only visible codes; marginal and invalid pairs contribute zero."}
::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=kernel caption="Observer-cell invocations atomically increment the visible count."}
::lens{scenario=gpu-terrain.terrain-cumulative-viewshed rung=graph caption="Zero the counts, then accumulate the observer batches."}


### `GPUPointHorizonProfile` and `GPUPointHorizonVisibility`

A 360° horizon profile from any number of observers, ported from mt-image's horizon march.

- **Inputs.** The elevation raster is the `elevation` prop on a `grid`. Observers are
  `[column, row, height, 0]`. `heightReference` is `'ground'` (the default) or `'absolute'`.
  In absolute mode, `observerVerticalDatum` declares the observer heights' datum and is checked
  against the elevation band's `verticalDatum`; see [Heights and datums](./concepts.md#heights-and-datums).
- **Outputs.** In `output`, for each (observer, azimuth bin) the profile writes the maximum
  apparent-elevation tangent `t = (h − h0)/d − c·d`, the `skylineAngle` in degrees (−90 when nothing was sampled), and the
  crest distance. Azimuth bin `i` is at `i·360/azimuthCount` degrees clockwise from north. Sectors
  are set with `firstAzimuth` and `azimuthSpan`.
- **Projections.**
  - `'planar'`: projected metres, a straight ray. The packer takes the cell size from the grid.
  - `'web-mercator'`: a window of the EPSG:3857 grid (square pixels, rows increasing south). The
    packer derives the world size `worldPixelSize = 2πa/cellSize` (pixels per 360°, a = 6378137 m)
    and the Mercator ordinate of the top edge `northEdge/a` from the grid, so no `worldPixelSize`
    or `originY` is passed. The ray is a piecewise great circle on a sphere of R = 6371008.8 m.
    Breakpoints are baked from `maximumLatitude` and `segmentTolerance`, and positions are linear in
    pixels between them.
  - `getGPUPointHorizonProfileParameterValues(settings, grid)` and
    `getGPUPointHorizonVisibilityParameterValues(settings, grid)` tag the words with
    `settings.projection` (default `'planar'`); a contributor whose `projection` differs reports NaN
    (profile) or no-data (visibility) instead of misreading them. The lattice `cellSize` prop stays
    a topology value: it sizes the distance lattice, and on Web Mercator the true pixel size varies
    with latitude.
- **Sample distances.** Samples sit on an exact power-of-two octave lattice: within
  `[2^k, 2^(k+1))` the step is a power of two following mt-image's step rule
  `max(stepFactor·d, clamp(nearFactor·d, minimumStep, cellSize·cellSteps))`, from 20 m out to
  `maximumDistance`. Every distance is exact in float32.
- **Precision.** Precision follows mt-image:
  - the eye is held as an integer pixel plus a float32 fraction;
  - heights are taken relative to the eye as a hi/lo pair behind an `opaque()` guard, against
    Metal fast math;
  - breakpoint offsets use the cancellation-free `bp()` with series `atan` and `atanh`;
  - azimuth sin/cos use an exact integer octant reduction, and degrees use an accurate `atan`
    (the WGSL builtins are only 2^-11 or about 4096 ULP accurate).
- **Traversal.** `traversal: 'pyramid'` (the default) skips min-max pyramid blocks whose
  conservative bound cannot raise the running maximum. Tangent and distance bits are identical to
  `'march'`. A shared `pyramid` can be passed as for the viewshed contributors.

`GPUPointHorizonVisibility` classifies targets `[column, row, height, observerIndex]` like
mt-image's peak classifier:

- **The ray.** It marches the full ray along the observer→target bearing. On Web Mercator the
  bearing and distance come from a cancellation-free spherical inverse.
- **What it records.** It keeps the occluder maximum before `q = d − max(targetIgnoreDistance,
  targetIgnoreFraction·d)` and the full skyline maximum.
- **Codes.** With `tol = toleranceDegrees + σz/d`, a target is `hidden` below the occluder by more
  than `tol`, `marginal` within `tol`, and otherwise `visible`.
- **Skyline.** `onSkyline` is set when the target is within `skylineToleranceDegrees` of the
  skyline.
- **Details.** The optional `output.details` holds `[targetElevation°, occluderElevation°,
  skylineElevation°, onSkyline]`.

Measured against float64 oracles, the elevation error is at most 3.3e-5° (planar) and 3.0e-5°
(Web Mercator).

{/* parameters: GPUPointHorizonProfile generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointHorizonProfile` parameters

- Packer: `getGPUPointHorizonProfileParameterValues(settings, grid)`, writes `GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUPointHorizonProfileParameterValues`:

| Field | Type | Description |
|---|---|---|
| `projection` (optional) | `GPUPointHorizonProjection` | Projection the packed grid words describe. Must equal the contributor's `projection`, else the kernel reports NaN or no-data. Defaults to `'planar'`. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/meters. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ray length cap in meters; `<= 0` or absent means the topology `maximumDistance`. |

{/* /parameters */}

{/* parameters: GPUPointHorizonVisibility generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUPointHorizonVisibility` parameters

- Packer: `getGPUPointHorizonVisibilityParameterValues(settings, grid)`, writes `GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUPointHorizonVisibilityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `projection` (optional) | `GPUPointHorizonProjection` | Projection the packed grid words describe. Must equal the contributor's `projection`, else the kernel reports NaN or no-data. Defaults to `'planar'`. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/meters. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ray length cap in meters; `<= 0` or absent means the topology `maximumDistance`. |
| `toleranceDegrees` (optional) | `number` | Angular tolerance in degrees of the occlusion test. Defaults to 0.02. |
| `sigmaZ` (optional) | `number` | Vertical standard error in meters, widening the tolerance by `sigmaZ / d` radians. Defaults to 5. |
| `skylineToleranceDegrees` (optional) | `number` | Tolerance in degrees of the skyline test. Defaults to 0.05. |
| `targetIgnoreDistance` (optional) | `number` | Meters before the target where the occlusion test stops: terrain in this last stretch of the ray is never tested, so a target inside it cannot be hidden by it. Defaults to one ground pixel along the ray at the target (library rule 4: derived from the cell size), which keeps the target's own bilinear cell from occluding it. mt-image's fixed 150 m is the named preset `GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE`. |
| `targetIgnoreFraction` (optional) | `number` | Fraction of the target distance added to the ignore stretch. Defaults to 0. |

{/* /parameters */}

#### `GPUPointHorizonProfile`

Reduces terrain rays to skyline angles and tangents for three ridge observers and 36 azimuth bins.

::lens{scenario=gpu-terrain.point-horizon-profile rung=result caption="A 36-bin by three-observer strip shows skyline angles; columns advance clockwise by 10 degrees."}

#### How it computes

{/* lens:GPUPointHorizonProfile */}

::lens{scenario=gpu-terrain.point-horizon-profile rung=datum caption="Inspect a single observer and azimuth bin."}
::lens{scenario=gpu-terrain.point-horizon-profile rung=formula caption="Maximise relative terrain height divided by distance, minus the curvature term."}
::lens{scenario=gpu-terrain.point-horizon-profile rung=kernel caption="March the exact distance lattice and retain the largest apparent tangent."}
::lens{scenario=gpu-terrain.point-horizon-profile rung=graph caption="Ray maxima feed the skyline-angle conversion pass."}

#### `GPUPointHorizonVisibility`

Classifies four fixed targets from each of three ridge observers and records target, occluder and skyline angles.

::lens{scenario=gpu-terrain.point-horizon-visibility rung=result caption="Twelve target classifications share the three observers used by the skyline profile."}

#### How it computes

{/* lens:GPUPointHorizonVisibility */}

::lens{scenario=gpu-terrain.point-horizon-visibility rung=datum caption="Inspect one target row and its observer index."}
::lens{scenario=gpu-terrain.point-horizon-visibility rung=formula caption="Angular tolerances separate hidden, marginal and visible targets."}
::lens{scenario=gpu-terrain.point-horizon-visibility rung=kernel caption="March the full bearing while retaining the pre-target occluder maximum."}
::lens{scenario=gpu-terrain.point-horizon-visibility rung=graph caption="Canonical elevation feeds per-target visibility and angle details."}


### `GPUTerrainRGBDecode` and `GPUTerrainSpikeRepair`

These contributors turn RGB-encoded elevation tiles into float32 heights with uint32 validity.

`GPUTerrainRGBDecode` handles two encodings:

- Terrarium: `R·256 + G + B/256 − 32768`.
- Mapbox terrain-RGB: `−10000 + 0.1·(R·65536 + G·256 + B)`.

It reads either an `rgba8unorm` texture (with `textureLoad`) or a buffer of packed little-endian
RGBA8 words, which is `getImageData().data` viewed as a `Uint32Array`. It writes heights and
validity. A pixel is nodata when any of these hold, and nodata pixels hold NaN bits and validity 0:

- its alpha is 0;
- it matches the optional `noDataRGB`;
- the optional input `validity` marks it invalid;
- its height falls outside `validRange` (default `[-11000, 9000]` m, which removes blank-canvas pixels).

`clampBathymetry` is opt-in. It flattens heights in (−12000, 0) to sea level, which also flattens
the Dead Sea and polders.

Decode before you filter. Never sample the encoded texture bilinearly or through mips, because
filtering byte channels mixes carries. Decode with nearest/`textureLoad`, then resample the float
heights.

Terrarium decoding is bit-exact in f32 for all 2^24 codes: every partial sum is a multiple of 1/256
below 2^16. Mapbox decoding is one correctly rounded multiply, `f32(N − 100000) · f32(0.1)`. Its
error is at most `½ulp + |M|·1.5e-9`, which is below 0.6 mm for |h| ≤ 9000 m.

`GPUTerrainSpikeRepair` is opt-in. `grid` holds at most `getGPUTerrainSpikeRepairMaximumCellCount(device)`
cells (4194304, 2048 × 2048 on a default device), because the per-component vote bit counts take 32
bytes per cell in one storage binding. It fixes ±256·k m Terrarium red-byte errors, such as canvas
anti-fingerprinting noise:

1. It labels 4-connected components across edges with |Δh| ≤ `jump`, using
   `GPUGraphConnectedComponents` on a fixed-stride grid CSR.
2. A component that is neither the largest nor bigger than 25 % of the valid pixels is shifted by
   −`step`·k when at least 80 % of its border seams agree on a jump of `step`·k ± `tolerance`.

Run it on full-resolution decoded heights, before any filtering. Nodata is never filled or
repaired. It can shift a real enclosed butte whose walls happen to measure 216–296 m, so enable it
only when the source is known to be noisy.

```ts
const grid = makeGPURasterMetadata({width, height});
graph.add(new GPUTerrainRGBDecode({
  grid, encoding: 'terrarium', input: {texture}, output: {values, outputValidity: validity}
}));
graph.add(new GPUTerrainSpikeRepair({
  grid,
  elevation: makeGPURasterBand(values, grid, {id: 'dem', validity}),
  output: {values: repaired, outputValidity: repairedValidity, statistics}
}));
```

`output.statistics` holds `[jumpCount, repairedPixelCount, shiftedComponentCount,
remainingJumpCount, converged]`. If the labelling does not converge within
`maximumComponentIterations`, nothing is repaired and the valid pixels are marked instead of
copied: heights are `+Infinity`, validity and labels hold `GPU_TERRAIN_SPIKE_REPAIR_UNRESOLVED`,
and `converged` is 0.

#### `GPUTerrainRGBDecode`

Decodes raw Terrarium RGBA words into elevation metres and validity on the 32×32 grid.

::lens{scenario=gpu-terrain.terrain-rgb-decode rung=result caption="Packed Terrarium input becomes decoded elevation on 20 m cells."}

#### How it computes

{/* lens:GPUTerrainRGBDecode */}

::lens{scenario=gpu-terrain.terrain-rgb-decode rung=datum caption="Inspect the RGB bytes and decoded height of one cell."}
::lens{scenario=gpu-terrain.terrain-rgb-decode rung=formula caption="Reassemble the integer RGB code before the exact power-of-two height conversion."}
::lens{scenario=gpu-terrain.terrain-rgb-decode rung=kernel caption="One invocation unpacks, decodes and checks validity for each cell."}
::lens{scenario=gpu-terrain.terrain-rgb-decode rung=graph caption="One decode pass writes elevation and validity buffers."}

#### `GPUTerrainSpikeRepair`

Repairs three injected +900 m spikes by shifting small components whose border jumps agree on the configured 900 m step.

::lens{scenario=gpu-terrain.terrain-spike-repair rung=result caption="Compare the spiked DEM with repaired elevation; no per-cell repair mask is published."}

#### How it computes

{/* lens:GPUTerrainSpikeRepair */}

::lens{scenario=gpu-terrain.terrain-spike-repair rung=datum caption="Inspect component membership and seam votes around an injected spike."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=formula caption="A majority of agreeing border seams selects a whole-component height shift."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=iterations caption="Component-label rounds precede the final height repair; earlier height frames remain unavailable."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=kernel caption="Component sizes and seam votes determine which heights shift."}
::lens{scenario=gpu-terrain.terrain-spike-repair rung=graph caption="Connected components feed voting, shift decisions and repair statistics."}


### `GPUTerrainPeakProminence`

Computes elder-rule prominence of every peak of an elevation grid, entirely on the GPU, from a join tree of sorted saddle edges.

::lens{scenario=gpu-terrain.terrain-peak-prominence rung=result caption="Prominence per cell in metres; the most prominent finite peak is marked, the elder peak is infinite."}

Computes eight-connected upper-excursion-set prominence (zero-dimensional persistence): a peak
dies at the highest saddle connecting its component to an elder peak, and prominence is peak
height minus that saddle. Higher peaks are elder; equal heights prefer the lowest cell index.
Plateau members and nonpeaks have zero prominence except for the plateau's representative.
Nodata is a barrier and receives NaN. Each disconnected component's surviving peak has infinite
prominence and a key saddle of negative infinity; the boundary has no artificial sea level.

The GPU sorts the depression-fill spill graph's neighbor edges by descending saddle, then builds
a join tree with the parallel parent-splicing helper shared by HDBSCAN. One label per cell preserves
peaks inside tiles that the fill-only outlet contraction would erase. Cycle edges create unary nodes;
gated ancestor doubling propagates elder peaks in logarithmic dispatch depth, and each binary merge
records one losing peak. Stable saddle-edge ordering preserves `parentPeak` ties. There is no CPU
readback or fallback. Contended splices can traverse long chains; performance measurements are pending.
Storage is linear and the maximum grid size is `getGPUTerrainPeakProminenceMaximumCellCount(device)`:
the child table requires 40 bytes per cell per binding (3,355,443 cells on a 128 MiB binding limit),
compared with the previous serial implementation's 16 bytes per cell.
The NumPy vertex-activation reference and definition are in
`modules/experimental/test/oracles/terrain-peak-prominence.py`.

```ts
graph.add(new GPUTerrainPeakProminence({
  grid,
  elevation,
  output: {prominence, keySaddle, parentPeak}
}));
```

#### Parameters

No packed settings or packer: `grid` fixes dimensions and connectivity is eight neighbors.
`elevation` is per-frame. `output.prominence` is required; `output.keySaddle` and
`output.parentPeak` are optional, each one value per cell. `parentPeak` records the winning peak
at the merge, with `0xffffffff` for surviving peaks and nodata; equal-saddle parent links follow
stable cell/direction edge order. Output buffers must be disjoint from inputs and each other.

#### How it computes

{/* lens:GPUTerrainPeakProminence */}

::lens{scenario=gpu-terrain.terrain-peak-prominence rung=datum datum=3181 caption="The highest summit in the window: its neighbourhood before the join tree decides."}
::lens{scenario=gpu-terrain.terrain-peak-prominence rung=kernel caption="Spill edges are keyed by saddle height, sorted, then spliced into a join tree."}
::lens{scenario=gpu-terrain.terrain-peak-prominence rung=graph caption="Edge keys, a radix sort, parent splicing and ancestor doubling, all on the GPU."}

### `GPUTerrainSummits`, `GPUTerrainPeakSnap`, `GPUTerrainCriticalPoints`, and `GPUProfilePeaks`

These contributors find terrain features. Summit and snap discs are metric: the packers read the
cell size and latitude band from the `grid` they are given (per frame), in `uniform`,
`web-mercator` or `geographic` mode, evaluated at the centre row. Nodata never contributes a
height.

**`GPUTerrainSummits`** marks pixel `p` as a summit when all of these hold:

- `p` is the strict (height, −index) maximum of its radius-`r` disc;
- `drop = h_p − max(ring) ≥ minimumDrop`, where the ring is the disc pixels with an 8-neighbour
  outside the disc. Every path leaving the disc crosses the ring, so `drop` is a radius-limited
  lower bound on prominence.

Under `incompleteNeighborhood: 'reject'` (the default), discs clipped by the grid edge or touching
nodata are rejected. `output` holds an optional `summitMask`, an optional per-pixel `drop`, an
optional compact list `summits` with an optional `summitDrop` column, and
`status.radiusClamped`, set when the radius was clamped to `maximumRadiusPixels`.

Cost is O(R) per candidate pixel in the disc radius R (pixels), not O(R²): the maximum test
descends a `'cell'`-footprint max pyramid (first block 2) and reads the maximum of every block that
lies wholly inside the disc, and the ring is visited from per-row half-widths. A 3×3 pre-test
rejects almost every pixel first. The optional `pyramid` prop reuses a prebuilt
`GPURasterExtremaPyramid.output` (`footprint: 'cell'`, `incompleteNeighborhood: 'ignore'` only);
results are bit-identical to the previous full disc scan.

**`GPUTerrainPeakSnap`** takes at most `getGPUTerrainPeakSnapMaximumCandidateCount(device)`
candidates (6710886 on a default device), because it keeps five result words per candidate in one
storage binding. It moves caller
candidate points, given as `[column, row]`, to the highest
valid pixel within a per-candidate or per-frame radius. It keeps the original point in four
cases:

- the maximum lies on the disc ring (a flank, not a summit);
- the move would exceed `maximumMove`;
- the height would change by more than `maximumHeightChange`;
- there is no valid data.

Each candidate gets `output.snapStatus` from `GPU_TERRAIN_PEAK_SNAP_STATUS`, beside
`output.positions`, `output.heights` and the optional `output.snapDistance`. mt-image's distance
rule
`min(250, 60 + 0.004·d)` m can be supplied through `candidateRadii`.

Optional `candidateHeights` supplies absolute catalogue elevations. `candidateVerticalDatum`
requires `candidateHeights` and is checked against the elevation band's `verticalDatum`;
see [Heights and datums](./concepts.md#heights-and-datums).

**`GPUTerrainCriticalPoints`** classifies each pixel as regular, peak, pit, saddle, boundary or
noData by counting sign changes around its ring. Equal heights are ordered by index (simulation of
simplicity). There are two rings:

- 8 is the classic Peucker–Douglas ring.
- 6 is the Freudenthal triangulation, which keeps the Euler relation peaks − saddles + pits
  consistent.

`output.classes` is required; `output.signChanges` (per pixel) and `output.counts` (per class)
are optional.

**`GPUProfilePeaks`** finds peaks along 1-D profiles in CSR spans, for example `GPURasterProfile`
`sampleValues` + `lineSampleOffsets`, or a horizon profile with `wrap: true`. For each candidate it:

- tests for a local maximum over ±2 samples;
- computes windowed topographic prominence;
- refines the position with a parabola.

Non-maximum suppression within `suppressionRadius` runs in GPU rounds (at most
`maximumSuppressionIterations`) and reaches exactly the greedy (prominence-descending) result. If
it does not converge, `output.status.converged` is 0, undecided candidates carry the
`GPU_PROFILE_PEAKS_UNRESOLVED_*` sentinels and `output.peaks.status.overflow` has
`GPU_PROFILE_PEAKS_OVERFLOW_UNCONVERGED` set. `minimumProminence` is set per frame and
`minimumSide` at compile time.

{/* parameters: GPUProfilePeaks generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUProfilePeaks` parameters

- Packer: `getGPUProfilePeaksParameterValues(settings)`, writes `GPU_PROFILE_PEAKS_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUProfilePeaksParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minimumProminence` | `number` | Minimum prominence, in profile-value units. A candidate with `prominence < minimumProminence` is rejected. Must not be NaN; a value at or below zero accepts every candidate. |

{/* /parameters */}

{/* parameters: GPUTerrainPeakSnap generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainPeakSnap` parameters

- Packer: `getGPUTerrainPeakSnapParameterValues(settings, grid)`, writes `GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainPeakSnapParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` (optional) | `number` | Default disc radius in ground metres, used for candidates without a radius. Defaults to 150. |
| `maximumMove` (optional) | `number` | Largest accepted move in ground metres. Defaults to 300. |
| `maximumHeightChange` (optional) | `number` | Largest accepted `\|best - reference\|` in elevation units. Defaults to 80. |
| `interior` (optional) | `boolean` | Keep the candidate when the best pixel lies on the disc ring (a flank). Defaults to true. |

{/* /parameters */}

{/* parameters: GPUTerrainSummits generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainSummits` parameters

- Packer: `getGPUTerrainSummitsParameterValues(settings, grid)`, writes `GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainSummitsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` | `number` | Disc radius in ground metres. Finite and positive. |
| `minimumDrop` (optional) | `number` | Smallest accepted drop in elevation units. Finite and at least 0. Defaults to 0. |

{/* /parameters */}

#### `GPUTerrainSummits`

Marks pixels that are the highest point of a metric disc and stand out from its boundary ring.

::lens{scenario=gpu-terrain.terrain-summits rung=result caption="Summit cells: the strict maxima of a 600 m disc that stand 5 m above its ring."}

#### How it computes

{/* lens:GPUTerrainSummits */}

::lens{scenario=gpu-terrain.terrain-summits rung=datum datum=3181 caption="The highest summit: its disc maximum and the ring drop it is measured against."}
::lens{scenario=gpu-terrain.terrain-summits rung=formula caption="Drop below the highest ring pixel, tested against the minimum drop."}
::lens{scenario=gpu-terrain.terrain-summits rung=kernel caption="A pyramid descent finds the disc maximum; the ring is scanned row by row."}
::lens{scenario=gpu-terrain.terrain-summits rung=graph caption="Maximum pyramid levels feed one summit kernel; no loop."}

#### `GPUTerrainPeakSnap`

Moves candidate points to the highest pixel of a metric disc around them, subject to move and height limits.

::lens{scenario=gpu-terrain.terrain-peak-snap rung=result caption="Three candidates and their snapped positions; the flank candidate keeps its place."}

#### How it computes

{/* lens:GPUTerrainPeakSnap */}

::lens{scenario=gpu-terrain.terrain-peak-snap rung=kernel caption="Each candidate scans its disc, then an ordered chain of accept rules sets the status."}
::lens{scenario=gpu-terrain.terrain-peak-snap rung=graph caption="Canonical elevation feeds one search kernel and one unpack kernel; no loop."}

#### `GPUTerrainCriticalPoints`

Classifies every pixel as peak, pit, saddle, regular, boundary or nodata by counting ring sign changes.

::lens{scenario=gpu-terrain.terrain-critical-points rung=result caption="Critical-point classes per cell; the table counts cells in each class."}

#### How it computes

{/* lens:GPUTerrainCriticalPoints */}

::lens{scenario=gpu-terrain.terrain-critical-points rung=datum datum=1957 caption="A saddle: four sign changes around its eight-neighbour ring."}
::lens{scenario=gpu-terrain.terrain-critical-points rung=formula caption="Class from the number of sign changes and the count of higher neighbours."}
::lens{scenario=gpu-terrain.terrain-critical-points rung=kernel caption="One kernel walks the ring in cyclic order and counts sign changes."}
::lens{scenario=gpu-terrain.terrain-critical-points rung=graph caption="Canonical elevation feeds one classification kernel; no loop."}

#### `GPUProfilePeaks`

Finds local maxima along a 1-D profile with windowed prominence, parabolic refinement and non-maximum suppression.

::lens{scenario=gpu-terrain.profile-peaks rung=result caption="Prominence of kept peaks along row 40; the most prominent is marked."}

#### How it computes

{/* lens:GPUProfilePeaks */}

::lens{scenario=gpu-terrain.profile-peaks rung=formula caption="Prominence is the peak minus the higher of the two side minima."}
::lens{scenario=gpu-terrain.profile-peaks rung=iterations caption="Suppression rounds that decide which candidates survive, with the convergence word."}
::lens{scenario=gpu-terrain.profile-peaks rung=kernel caption="The candidate kernel walks both sides of each local maximum inside the window."}
::lens{scenario=gpu-terrain.profile-peaks rung=graph caption="Candidate pass, gated suppression rounds, then output publication."}

### `GPUTerrainCurvature`

Estimates profile and plan curvature from local partial derivatives of an elevation raster; convex ground is positive.

::lens{scenario=gpu-terrain.terrain-curvature rung=result caption="Profile curvature over the elevation greys; convex ground is positive, hollows negative."}

Florinsky's curvature system from local partial derivatives `p, q, r, s, t`, estimated by
Evans–Young (3×3, default), Zevenbergen–Thorne (3×3), or Florinsky's 5×5 polynomial (the
WhiteboxTools estimator). `curvatures` maps any of 15 kinds to a float32 output: `profile`,
`plan`, `tangential`, `mean`, `gaussian`, `minimal`, `maximal`, `unsphericity`, `difference`,
`horizontal-excess`, `vertical-excess`, `accumulation`, `ring`, `rotor`, `laplacian`. Signs follow
WhiteboxTools: a convex hill is positive, a valley floor negative. Where `p² + q² ≤ flatGradient²`
the direction-dependent kinds are 0, as in Whitebox. Window samples are differenced against the
centre before any sum, so a 1000 m base height costs no float32 precision.
`ringCurvature` ports mt-image's multi-radius ring curvature: up to four rings of 8 samples,
`Σ gain_k · Σ(z_c − z_i) / (8 r_k · cell)`, with an optional Padé-tanh squash. Cell sizes are
uniform, Web Mercator, or geographic per row, as for `GPUTerrainDerivatives`. Outputs go into
`output` (`curvatures`, `ringCurvature`, and the per-pixel `outputValidity`). `zFactor`,
`flatGradient` and `ringGains` are per-frame settings; the ring radii are compile time.

```ts
graph.add(new GPUTerrainCurvature({
  grid, elevation, parameters: parameters.importToGraph(graph),
  output: {curvatures: {profile, plan: planCurvature, mean}, ringCurvature, outputValidity},
  ringRadii: [2, 8]
}));
// Cell size and latitude band come from the grid.
parameters.write(getGPUTerrainCurvatureParameterValues({zFactor: 1}, grid));
```

{/* parameters: GPUTerrainCurvature generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainCurvature` parameters

- Packer: `getGPUTerrainCurvatureParameterValues(settings, grid)`, writes `GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainCurvatureParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier converting elevation units into ground units. Defaults to 1. |
| `flatGradient` (optional) | `number` | Gradients with `p^2 + q^2 <= flatGradient^2` are flat: the direction-dependent curvatures (profile, plan, tangential, difference, excesses, accumulation, ring, rotor) are 0 there. Defaults to 1e-6. |
| `ringGains` (optional) | `readonly number[]` | Up to four per-ring gains of the multi-radius ring curvature, one per entry of `ringRadii`. Defaults to `[0.55, 0.45, 0, 0]`. These are mt-image's pre-division weights. mt-image additionally divides ring 0 by 0.35 and ring 1 by 0.3; reproduce it with `[0.55 / 0.35, 0.45 / 0.3]`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUTerrainCurvature */}

::lens{scenario=gpu-terrain.terrain-curvature rung=datum datum=2576 caption="One cell's 3×3 window feeds the five partial derivatives behind profile and plan curvature."}
::lens{scenario=gpu-terrain.terrain-curvature rung=formula caption="Evans-Young partials p, q, r, s and t feed profile and plan curvature, with the flat-gradient guard."}
::lens{scenario=gpu-terrain.terrain-curvature rung=kernel caption="One invocation per cell reads its window and writes up to five curvature outputs."}
::lens{scenario=gpu-terrain.terrain-curvature rung=graph caption="Canonical elevation feeds one curvature kernel that writes both outputs."}

### `GPUGeomorphons`

Classifies each cell of an elevation raster into one of ten landforms from eight lines of sight.

::lens{scenario=gpu-terrain.geomorphons rung=result caption="Landform codes over the elevation greys; the five largest classes are coloured, the rest grouped."}

Jasiewicz & Stepinski (2013) geomorphons, mirroring GRASS `r.geomorphon`: eight line-of-sight
scans per cell out to `searchRadius` cells (skipping `skipRadius`), a flatness threshold in degrees
with optional `flatDistance`, GRASS's `anglev1` (default), `anglev2`, or `anglev2-distance`
comparison, and GRASS's 9×9 table from (minus, plus) counts to ten forms (`GPU_GEOMORPHON_FORMS`:
flat 1 … pit 10, 0 invalid). `pattern` writes the raw base-3 code (GRASS `ternary_6561`, directions
NE, N, NW, W, SW, S, SE, E) and `ternary` its rotation- and mirror-invariant minimum (GRASS's 498
classes). Comparisons use cross-multiplied heights and step counts instead of `atan`, so integer
DEMs with exact ties classify identically to the float64 oracle. Cells within `skipRadius + 1` of
the border are invalid, as in GRASS. A line of sight with no valid sample after the skip gives
digit 0 (flat) in every mode, where GRASS `anglev2` gives +1 from its initial ±90° angles. Outputs `forms`, `ternary`, `pattern` and `outputValidity` go
into `output`. The flatness threshold, `flatDistance` and `zFactor` are per-frame settings
(`getGPUGeomorphonsParameterValues(settings, grid)` takes the cell size and latitude band from the
grid); `searchRadius` and `skipRadius` size the line-of-sight loops and are compile time.

{/* parameters: GPUGeomorphons generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeomorphons` parameters

- Packer: `getGPUGeomorphonsParameterValues(settings, grid)`, writes `GPU_GEOMORPHONS_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUGeomorphonsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier converting elevation units into ground units. Defaults to 1. |
| `flatThresholdDegrees` (optional) | `number` | Flatness threshold in degrees, greater than 0 and below 90. Defaults to 1. |
| `flatDistance` (optional) | `number` | Distance in ground metres beyond which the flatness threshold is lowered to the height `tan(threshold) * flatDistance`. 0 disables the rule. Defaults to 0. As in GRASS `r.geomorphon` (`main.c`: `flat_distance <= skip_distance \|\| flat_distance >= search_distance` sets it to 0), a distance at or inside the skip distance (`skipRadius` cells) or at or beyond the search distance (`searchRadius` cells) disables the rule; both are measured with the row's north-south ground cell size. |

{/* /parameters */}

#### How it computes

{/* lens:GPUGeomorphons */}

::lens{scenario=gpu-terrain.geomorphons rung=datum datum=3181 caption="Eight lines of sight from the ridge cell at column 45, row 49: zenith, nadir and digit."}
::lens{scenario=gpu-terrain.geomorphons rung=formula caption="Zenith and nadir per direction give a ternary digit; the digit counts index the landform table."}
::lens{scenario=gpu-terrain.geomorphons rung=kernel caption="One invocation per cell scans eight rays and looks up the landform."}
::lens{scenario=gpu-terrain.geomorphons rung=graph caption="Canonical elevation feeds one fused classification kernel."}

### `GPUTerrainRuggedness` and `GPUTerrainVectorRuggedness`

`GPUTerrainRuggedness` reproduces gdaldem's TPI, TRI (Riley, the default since GDAL 3.3, or Wilson)
and roughness on 3×3 windows, including both edge modes: `'nodata'` (default) and `'extrapolate'`
(gdaldem `-compute_edges`, with GDAL's linear edge extrapolation and centre substitution for nodata
neighbours). It needs no parameters buffer; its outputs (`topographicPositionIndex`,
`terrainRuggednessIndex`, `roughness`, `outputValidity`) go into `output`.
`GPUTerrainVectorRuggedness` is Sappington et al.'s (2007) vector ruggedness measure,
`1 − |Σn| / N` over Horn unit normals in a `(2·radius + 1)²` window. The window radius is a
per-frame setting: `maximumRadius` (compile time, default 1) sizes `requiredHalo`, bounds the loops
and selects the direct square sum (`maximumRadius` below 3) or the separable row and column passes
(from 3), and `getGPUTerrainVectorRuggednessParameterValues({radius}, grid)` picks any integer in
`[1, maximumRadius]` per frame with the same pipelines. An omitted radius uses `maximumRadius`, the
packer throws for a value that is not an integer of at least 1, and a larger word written to the
buffer by other means clamps to `maximumRadius`.

{/* parameters: GPUTerrainVectorRuggedness generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainVectorRuggedness` parameters

- Packer: `getGPUTerrainVectorRuggednessParameterValues(settings, grid)`, writes `GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainVectorRuggednessParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier converting elevation units into ground units. Defaults to 1. |
| `radius` (optional) | `number` | Window half-width in pixels: the window is `(2 * radius + 1)` square. Integer in `[1, maximumRadius]`. Defaults to `maximumRadius`. The packer throws for a value that is not an integer of at least 1; a value above `maximumRadius` written to the buffer by other means is clamped to `maximumRadius` by the kernel. |

{/* /parameters */}

#### `GPUTerrainRuggedness`

Computes the Riley terrain ruggedness index of an elevation raster over 3×3 windows, in elevation units.

::lens{scenario=gpu-terrain.terrain-ruggedness rung=result caption="Ruggedness over the elevation greys; the most rugged cell is marked and the outer ring is empty."}

#### How it computes

{/* lens:GPUTerrainRuggedness */}

::lens{scenario=gpu-terrain.terrain-ruggedness rung=datum datum=2907 caption="The steepest cell's nine heights: squared differences from the centre sum to the index."}
::lens{scenario=gpu-terrain.terrain-ruggedness rung=formula caption="Riley ruggedness is the root of the summed squared differences from the centre cell."}
::lens{scenario=gpu-terrain.terrain-ruggedness rung=kernel caption="One invocation per cell reads its 3×3 window and writes the index."}
::lens{scenario=gpu-terrain.terrain-ruggedness rung=graph caption="Canonical elevation feeds a single window kernel."}

#### `GPUTerrainVectorRuggedness`

Measures how much unit surface normals diverge inside a window of an elevation raster, from 0 smooth to 1 rugged.

::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=result caption="Vector ruggedness in a 5 × 5 window; the roughest cell is marked."}

#### How it computes

{/* lens:GPUTerrainVectorRuggedness */}

::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=datum datum=2835 caption="One cell's 5 × 5 window of unit normals; the length of their sum sets the value."}
::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=formula caption="One minus the length of the summed unit normals, divided by the number of valid cells."}
::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=kernel caption="A normals pass feeds a direct window sum, one invocation per cell."}
::lens{scenario=gpu-terrain.terrain-vector-ruggedness rung=graph caption="Horn normals feed the window sum that returns the ruggedness."}

### `GPUTerrainTopographicPosition` and `GPUTerrainWeissLandforms`

`GPUTerrainTopographicPosition` evaluates any number of square scales (up to 64) in O(1) per pixel
from one summed-area table: TPI over the annulus `(innerRadius, radius]`, the deviation from mean
elevation (DEV) over the full window, and Lindsay et al.'s (2015) DEVmax with the radius at which
it occurs. A float32 summed-area table cannot do this: on a 512×384 tile at 3800 m its window means
are off by 5.4 m. Elevations are instead quantized (`quantum`, default 1/256) into an exact
64-bit modular table built with `GPUScanUint64`, `GPUScan`, and `GPUTranspose`; variances are formed
relative to the centre cell in integer arithmetic. Measured: TPI within 2.2 mm (the quantization)
and DEV within 2.4e-4 of the float64 definition, and within 4.6e-6 m / 7.3e-7 of the quantized
oracle. Windows are clipped to the raster and to valid cells.

Both contributors accept at most `getGPUTerrainSummedAreaMaximumCellCount(device)` (11184810,
about 3344 × 3344 on a default device) cells: the table keeps 12 bytes per cell in one storage binding, and its scans and
transpose run over the whole grid, so a 4096 × 4096 grid must be split into tiles with a halo of the
largest radius. `GPUTerrainTopographicPosition` also limits `grid` cells × `scales` to
`getGPUTerrainTopographicPositionMaximumPlaneCount(device)` (33554432 on a default device) when it writes
`topographicPositionIndex` or `deviationFromMean`.

`GPUTerrainWeissLandforms` classifies Weiss's (2001) ten landforms from a small and a large TPI
scale and Horn slope. `'global'` standardization (default) uses the grid mean and population
standard deviation of each scale (two-pass `GPUReduction`); `'local'` uses DEV. Thresholds
(`standardThreshold`, default 1; `slopeThresholdDegrees`, default 5) are per-frame settings, and
`getGPUTerrainWeissLandformsParameterValues(settings, grid)` takes the cell size for the slope from
the grid. Both contributors write into `output` (`outputValidity` is the per-pixel mask); the scales
are compile time.

```ts
graph.add(new GPUTerrainTopographicPosition({
  grid, elevation, scales: [{radius: 3}, {radius: 25}, {radius: 100}],
  output: {deviationFromMean, maximumDeviation, maximumDeviationRadius}
}));
```

{/* parameters: GPUTerrainWeissLandforms generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainWeissLandforms` parameters

- Packer: `getGPUTerrainWeissLandformsParameterValues(settings, grid)`, writes `GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainWeissLandformsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier converting elevation units into ground units. Defaults to 1. |
| `standardThreshold` (optional) | `number` | Standardized position threshold separating low, mid, and high. Defaults to 1. |
| `slopeThresholdDegrees` (optional) | `number` | Slope in degrees at or below which a mid/mid cell is a plain. Defaults to 5. |

{/* /parameters */}

#### `GPUTerrainTopographicPosition`

Subtracts the mean elevation of a square annulus from each cell of an elevation raster, per scale.

::lens{scenario=gpu-terrain.terrain-topographic-position rung=result caption="Cells above (positive) or below (negative) the mean of their 11 × 11 window."}

#### How it computes

{/* lens:GPUTerrainTopographicPosition */}

::lens{scenario=gpu-terrain.terrain-topographic-position rung=datum datum=2836 caption="The cell's 11 × 11 window minus its centre: the annulus mean from the summed-area table."}
::lens{scenario=gpu-terrain.terrain-topographic-position rung=formula caption="Centre minus annulus mean, computed exactly in quantized integers from the summed-area table."}
::lens{scenario=gpu-terrain.terrain-topographic-position rung=kernel caption="The evaluate kernel reads box sums from the summed-area table once per scale."}
::lens{scenario=gpu-terrain.terrain-topographic-position rung=graph caption="Elevation feeds the summed-area table scans, then one evaluate kernel."}

#### `GPUTerrainWeissLandforms`

Classifies each cell of an elevation raster into ten Weiss landforms from standardized topographic position at two scales and slope.

::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=result caption="Ten landform classes from small- and large-scale position; the five largest are coloured, the rest grouped."}

#### How it computes

{/* lens:GPUTerrainWeissLandforms */}

::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=datum datum=3181 caption="The ridge cell at column 45, row 49: two z-scores, their classes and the slope pick mountain top."}
::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=formula caption="Standardize both scales, class each at plus or minus the threshold, then look up the landform."}
::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=kernel caption="Classification reads both position planes, the slope and the global statistics."}
::lens{scenario=gpu-terrain.terrain-weiss-landforms rung=graph caption="Position, slope and global statistics feed the final classification node."}

### `GPUTerrainHorizon`, `GPUSolarShadowMask`, `GPUTerrainCastShadow`, `GPUSolarIrradiance`, `GPUReliefShading`, `GPUTextureShading`, and `GPUSolarPosition`

Cartographic relief and terrain light in data space, for "sun and shadow" maps on DEM tiles. Every
style and sun value is a per-frame parameters write; none of them recompile the graph. Each class
takes the raster `grid` (its width and height size the kernels; the cell size and the latitude band
of the packers come from it), packed `parameters`, and a caller-owned `output` bag. Outputs are
float32 rasters, each with an optional `r32float` or `rgba32float` storage texture (channel 0);
validity outputs are named `outputValidity`.

- `GPUTerrainHorizon` marches `directionCount` (4 to 64) azimuth sectors per pixel, one node per
  sector, over a baked step schedule (`maximumRadius` pixels, optional geometric `stepGrowth`) with
  bilinear samples, z factor, earth curvature, and an optional ground `maximumDistance`. It writes
  a pixel-major horizon map (`horizon[pixel * directionCount + sector]`, degrees, sector `d` at
  azimuth `d * 360 / directionCount`), the sky-view factor `1 - mean(sin(max(h, 0)))`, and positive
  openness `mean(90 - h)`, all in `output`. Sky-view-only use needs no `output.horizon` buffer. Sector azimuths are ground
  azimuths on any cell shape: on anisotropic cells (`'geographic'` off the equator, or non-square
  grid cells) each ray's pixel step is `normalize(groundDirection / groundCell)` at the pixel's own
  row, so a 45 degree sector at 60 N steps about two columns per row instead of along the pixel
  diagonal (a 26.6 degree ground ray). The cell size is not re-evaluated along a ray; at 60 N and
  30 km that leaves a 0.25 degree bearing and 0.14% distance error. Options:
  - `algorithm: 'sweep'` replaces the per-pixel march with an exact upper-hull sweep (Stewart 1998
    style) over digital lines through pixel centres: amortised O(1) per pixel and sector, exact
    against brute force over the same samples, 4x faster at radius 256 and 36x at full-tile radius on
    1024² (16 sectors). Samples are pixel centres within half a pixel of the ray, so sky-view
    factors differ from the bilinear march by about 0.01 on rough terrain. The sweep has a large
    fixed cost per pixel and sector: it wins at large radii (1024² at radius 1023: 41 ms against
    about 1300 ms; 512² at full radius: 36 against 109 ms) and loses at small ones (512² at radius
    128: 60–125 ms against about 20 ms). Prefer the march for small radii and the sweep near the
    full tile radius. One line family has one pixel slope, so on anisotropic cells the sweep uses
    the ground cell of row `height / 2` for every row; on a `'geographic'` tile the other rows see
    sectors rotated by up to about `tan(lat) * halfSpan / 2` radians (0.44 degrees at the edges of
    a 1 degree tall tile at 60 N).
  - `horizonFormat: 'unorm16'` halves the horizon map (two 16-bit codes per `uint32`, 0.00275 degree
    steps, code 0 = nodata; `unpackGPUTerrainHorizonUnorm16` decodes on the CPU). `GPUSolarShadowMask`
    and `GPUSolarIrradiance` read it.
  - `output.negativeOpenness` is openness of the inverted DEM (Yokoyama), and
    `output.anisotropicSkyViewFactor` is
    RVT's weighted SVF `1 - sum(w sin h+) / sum(w)`, `w = (1 - wMin) cos((a - A)/2)^level + wMin`,
    computed exactly as RVT's `sky_view_factor_compute`. `anisotropyAzimuthDegrees` (`A`) uses RVT's
    direction convention, so the same number means the same direction as RVT's `asvf_dir`: angles
    run counterclockwise from north (90 west, 270 east), and the default 315 is RVT's default, which
    is compass north-east. `a` is each sector's RVT angle, `(360 - t) mod 360` for compass sector
    azimuth `t`. `anisotropyLevel` is RVT's integer `a_poly_level` with RVT's signed power (presets:
    low 4 with minimum weight 0.4, high 8 with 0.1). Parameter slots 8-10 hold them per frame
    (`parameters` must then hold 12 floats). This is the only counterclockwise angle in gpu-terrain; sun
    azimuths already match RVT's compass `shadow_az`.
  - Breaking change: `anisotropyAzimuthDegrees` used to be a compass azimuth, so the default 315
    weighted the north-west sky. It now weights the north-east sky, as RVT does. To keep a previous
    compass azimuth `t`, pass `(360 - t) mod 360`. Non-integer levels now throw.
- `GPUSolarShadowMask` reads the horizon map and a per-frame sun: it interpolates the horizon at
  the sun azimuth and writes `output.sunVisibility`, the solar-disk area fraction above the horizon
  (soft penumbra from the disk's angular radius, hard when 0). With `slope` and `aspect` from
  `GPUTerrainDerivatives` it also writes `output.illumination = ambient * svf + sun * sunVisibility *
  max(cos(incidence), 0)`. `horizon`, `slope`, `aspect`, and `skyViewFactor` are inputs. One frame
  costs two reads per pixel.
- `GPUTerrainCastShadow` casts one sun's shadow with no horizon map: an exact sweep along the
  per-frame sun azimuth gives the horizon in that direction, then the same solar-disk penumbra as
  `GPUSolarShadowMask`; it writes `output.sunVisibility`, `output.horizonAngle`, and
  `output.outputValidity`. The CPU packer `getGPUTerrainCastShadowParameterValues(settings, grid)`
  computes the line geometry, so moving the sun never recompiles. The packer also stores the sun's
  ground direction and takes no row direction or cell size mode: the contributor mirrors the line
  for its `rowDirection: 'north'` and, on anisotropic cells, turns the direction into a pixel slope
  with its own `cellSizeMode` and the ground cell of row `height / 2`. Use it when only one sun
  matters and a full horizon map is too large.
- `GPUSolarIrradiance` integrates a sun path over a horizon map: `output.sunHours` (disk-visible
  hours) and `output.insolation` (Wh/m², direct normal irradiance times visibility times cos incidence, plus an
  optional diffuse term scaled by the sky-view factor). `getGPUSolarIrradianceSunTable` samples
  `getSolarPosition` over a day or a date range (Meinel clear-sky DNI with Kasten–Young air mass by
  default). `samplesPerNode` splits long tables across dispatches.
- `GPUReliefShading` computes a hillshade from up to 8 lights with fixed weights or the USGS
  multidirectional oblique weighting (`lights: 'mdow'`, per-pixel `sin^2(aspect - azimuth)` as in
  GDAL `-multidirectional`). It writes `output.hillshade`, and blends a Swiss/Imhof relief
  (`output.relief` luminance and packed RGBA8 `output.color`, plus `output.outputValidity`) from the hillshade, an optional sky-view factor, an optional texture shade, an
  elevation tint ramp of up to 8 stops, and a warm-lit/cool-shaded aspect tint. Imhof options:
  `lightWeighting: 'imhof-swing'` (with the `imhofSwing: true` prop) swings each light toward the
  slope's side by up to `imhofSwingDegrees` (default 65), an optional `curvature` raster adds
  `curvatureStrength * curvature`, and `contrastStrength` with
  `contrastLowElevation`/`contrastHighElevation` raises contrast with elevation. Zero values keep the
  previous output bit for bit.
- `GPUTextureShading` approximates Leland Brown's texture shading (fractional Laplacian
  `|f|^alpha`) with a cascade of separable Gaussian levels (`sigma_k = baseSigma * 2^k`) summed as
  weighted band differences. Nodata uses normalized convolution. `detail` (alpha), the band
  weights, and gain are per frame; it writes `output.textureShade` and `output.outputValidity`.
  `hasNodata: false` (default `true`) promises an all-valid
  elevation and blurs one channel instead of value plus validity, which is exact and about halves
  the blur cost. `downsampleLevels` (default `true`) computes levels with incremental sigma of at
  least 4 pixels on a 2x decimated grid and upsamples bilinearly; output stays within 1% of the exact
  output range (measured 0.1-0.27%), and `false` restores the exact full-resolution cascade.
- `GPUSolarPosition` evaluates the NOAA solar position for every row of a longitude/latitude
  column at one per-frame instant. It has no grid (it takes positions): `parameters` holds the time
  and `output` the azimuth, altitude (with optional refraction), and a daylight flag.
  `getSolarPosition` is the float64 CPU version used to drive the shadow mask.
  Both use the **NOAA daily spreadsheet's Meeus-based approximation**, without observer parallax,
  rather than NREL SPA. [NOAA's published accuracy](https://gml.noaa.gov/grad/solcalc/calcdetails.html)
  is about one minute for sunrise/set within ±72° latitude and ten minutes outside; NOAA does
  not give a universal 0.01° angular guarantee. Float32 orbital-angle error increases with time
  from J2000, and azimuth is ill-conditioned near zenith/nadir. Time is transported as separate
  whole days and day fraction. `daylight` compares **geometric sun-center altitude** with a strict
  `>` threshold (default −0.833°), independently of the optional refraction applied to `altitude`.
  Near that threshold, NOAA and SPA can legitimately produce different daylight flags.

```ts
const sun = getSolarPosition(Date.now(), longitude, latitude);
const grid = makeGPURasterMetadata({width, height, cellSize: 30});
graph.add(new GPUTerrainHorizon({
  grid, elevation, parameters: horizonParameters.importToGraph(graph),
  directionCount: 16, maximumRadius: 256, stepGrowth: 1.1,
  output: {horizon, skyViewFactor}
}));
graph.add(new GPUSolarShadowMask({
  grid, directionCount: 16, horizon, parameters: shadowParameters.importToGraph(graph),
  slope, aspect, skyViewFactor, output: {illumination}
}));
shadowParameters.write(getGPUSolarShadowMaskParameterValues({
  azimuthDegrees: sun.azimuthDegrees, altitudeDegrees: sun.altitudeDegrees, ambientIntensity: 0.3
}));
```

The horizon map needs `width * height * directionCount * 4` bytes in one binding (268 MB for a
2048² tile with 16 sectors). The contributor checks `maxStorageBufferBindingSize`. Pixels closer than
`requiredHalo` to a tile edge see a truncated horizon or blur, so pass tiles with that halo for
seamless mosaics. Typical 2048² timings on an Apple-silicon laptop are about 100 ms for the horizon
(16 sectors, radius 256, growth 1.1), 3.4 ms per frame for the shadow mask, 3.7 ms for relief
shading, and 28 ms for six-level texture shading.

```ts
graph.add(new GPUTerrainHorizon({
  grid, elevation, parameters, directionCount: 16, maximumRadius: 1023,
  algorithm: 'sweep', horizonFormat: 'unorm16', output: {horizon, skyViewFactor}
}));
const {values, sampleCount} = getGPUSolarIrradianceSunTable({
  longitude: 8.2, latitude: 46.8, start: Date.UTC(2026, 5, 21), end: Date.UTC(2026, 5, 22)
});
sunTableBuffer.write(values);
graph.add(new GPUSolarIrradiance({
  grid, directionCount: 16, horizonFormat: 'unorm16', horizon,
  sunTable: sunTableBuffer.importToGraph(graph), sampleCapacity: 288, parameters: irradianceParameters,
  slope, aspect, skyViewFactor, output: {sunHours, insolation}
}));
```

At 1024² with 16 sectors the sweep horizon takes 41 ms at full radius (98.6 ms at radius 256), and
a 288-sample day of irradiance takes about 29 ms.

{/* parameters: GPUReliefShading generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUReliefShading` parameters

- Packer: `getGPUReliefShadingParameterValues(settings, grid)`, writes `GPU_RELIEF_SHADING_PARAMETER_LENGTH` (80) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUReliefShadingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier applied to derivatives. Defaults to 1. |
| `lights` (optional) | `'mdow' \| readonly GPUReliefShadingLight[]` | Lights, or `'mdow'` for the USGS set: azimuths 225, 270, 315, 360 at 30 degrees altitude with `'aspect'` weighting. Defaults to `'mdow'`. |
| `lightWeighting` (optional) | `GPUReliefShadingWeighting` | Light weighting. Defaults to `'aspect'` for `'mdow'` and `'fixed'` for custom lights. |
| `imhofSwingDegrees` (optional) | `number` | Largest azimuth swing of `'imhof-swing'` weighting in degrees (settings slot 19). Defaults to 65 for `'imhof-swing'` and 0 otherwise; ignored by the other weightings. 0 disables the swing. |
| `curvatureStrength` (optional) | `number` | Multiplier of the optional `curvature` raster added to the shade (settings slot 76): `shade += curvatureStrength * curvature`. Defaults to 1; ignored when no curvature is bound. |
| `contrastLowElevation` (optional) | `number` | Elevation (calibrated units, before `zFactor`) where the elevation-dependent contrast starts fading in (settings slot 77). Defaults to 0. |
| `contrastHighElevation` (optional) | `number` | Elevation where the contrast reaches full strength (slot 78). When it is not above `contrastLowElevation` the contrast steps at the low elevation. Defaults to 0. |
| `contrastStrength` (optional) | `number` | Strength in `[0, 1]` of the elevation-dependent contrast (slot 79): with `k = mix(1 - strength, 1, smoothstep(low, high, z))` the shade becomes `mix(GPU_RELIEF_SHADING_CONTRAST_PIVOT, shade, k)`, so low ground is flattened toward grey and high ground keeps full relief. Defaults to 0, which is off and leaves output unchanged. |
| `hillshadeStrength` (optional) | `number` | Blend of the hillshade into `relief`: 0 ignores it, 1 uses it fully. Defaults to 1. |
| `skyViewStrength` (optional) | `number` | Blend of the sky-view factor into `relief`. Defaults to 1 (only when the input is bound). |
| `textureShadeStrength` (optional) | `number` | Multiplier of the texture shade added to `relief`. Defaults to 0.5 (only when bound). |
| `exposure` (optional) | `number` | Final `relief` multiplier before clamping to `[0, 1]`. Defaults to 1. |
| `tintStrength` (optional) | `number` | Imhof aspect tint strength in `[0, 1]`. Defaults to 0.3. |
| `warmColor` (optional) | `readonly [number, number, number]` | Tint toward which lit slopes shift. Defaults to `[1, 0.95, 0.85]`. |
| `coolColor` (optional) | `readonly [number, number, number]` | Tint toward which shaded slopes shift. Defaults to `[0.8, 0.85, 1]`. |
| `elevationStops` (optional) | `readonly GPUReliefShadingStop[]` | Ascending elevation tint stops; empty gives a 0.85 grey base. Defaults to `[]`. |

{/* /parameters */}

{/* parameters: GPUSolarIrradiance generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSolarIrradiance` parameters

- Packer: `getGPUSolarIrradianceParameterValues(settings)`, writes `GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSolarIrradianceParameterValues`:

| Field | Type | Description |
|---|---|---|
| `sampleCount` | `number` | Number of valid sun table rows, an integer in `[0, sampleCapacity]`. |
| `angularRadiusDegrees` (optional) | `number` | Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to `GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES`. |
| `diffuseIrradiance` (optional) | `number` | Diffuse horizontal irradiance in W/m^2 added while the sun is up, scaled by the sky-view factor when one is bound. Defaults to 0. |

{/* /parameters */}

{/* parameters: GPUSolarPosition generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSolarPosition` parameters

- Packer: `getGPUSolarPositionParameterValues(settings)`, writes `GPU_SOLAR_POSITION_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUSolarPositionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `timestamp` | `number \| Date` | Unix epoch milliseconds (UTC) or a `Date`. |
| `daylightAltitudeDegrees` (optional) | `number` | Geometric altitude above which `daylight` is 1. Defaults to `GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES`; use -6, -12, or -18 for civil, nautical, or astronomical twilight. |

{/* /parameters */}

{/* parameters: GPUSolarShadowMask generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSolarShadowMask` parameters

- Packer: `getGPUSolarShadowMaskParameterValues(settings)`, writes `GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSolarShadowMaskParameterValues`:

| Field | Type | Description |
|---|---|---|
| `azimuthDegrees` | `number` | Direction of the sun, degrees clockwise from north. See `getSolarPosition`. |
| `altitudeDegrees` | `number` | Elevation of the sun center above the astronomical horizon in degrees. |
| `angularRadiusDegrees` (optional) | `number` | Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to `GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES`. |
| `sunIntensity` (optional) | `number` | Direct light multiplier for `illumination`. Defaults to 1. |
| `ambientIntensity` (optional) | `number` | Ambient light multiplier for `illumination`, scaled by the sky-view factor. Defaults to 0. |

{/* /parameters */}

{/* parameters: GPUTerrainCastShadow generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainCastShadow` parameters

- Packer: `getGPUTerrainCastShadowParameterValues(settings, grid)`, writes `GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH` (16) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainCastShadowParameterValues`:

| Field | Type | Description |
|---|---|---|
| `azimuthDegrees` | `number` | Direction of the sun, degrees clockwise from north. See `getSolarPosition`. |
| `altitudeDegrees` | `number` | Elevation of the sun center above the astronomical horizon in degrees. |
| `angularRadiusDegrees` (optional) | `number` | Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to `GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES`. |
| `zFactor` (optional) | `number` | Elevation multiplier. Defaults to 1. |
| `curvatureCoefficient` (optional) | `number` | Earth curvature and refraction drop in 1/meters, see `GPUTerrainHorizonSettings`. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground search radius in meters; `<= 0` is unlimited. Defaults to 0. |

{/* /parameters */}

{/* parameters: GPUTerrainHorizon generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainHorizon` parameters

- Packer: `getGPUTerrainHorizonParameterValues(settings, grid)`, writes `GPU_TERRAIN_HORIZON_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainHorizonParameterValues`:

| Field | Type | Description |
|---|---|---|
| `zFactor` (optional) | `number` | Elevation multiplier. Defaults to 1. |
| `curvatureCoefficient` (optional) | `number` | Earth curvature and refraction drop in 1/meters: a sample at ground distance `d` is lowered by `c * d^2`. See `getGPUTerrainCurvatureCoefficient`. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground search radius in meters; `<= 0` searches the full pixel radius. Defaults to 0. |
| `anisotropyAzimuthDegrees` (optional) | `number` | Direction of maximum weight for `anisotropicSkyViewFactor`, in RVT's convention: the same number as RVT's `asvf_dir` / `a_main_direction` means the same direction. RVT direction angles run counterclockwise from north (0 north, 90 west, 180 south, 270 east), so `A` is compass azimuth `(360 - A) mod 360`. Defaults to `GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES` (315, RVT's default, which is compass north-east). This is the only counterclockwise angle in gpu-terrain: sector azimuths, sun azimuths and aspect are compass azimuths, clockwise from north. |
| `anisotropyLevel` (optional) | `number` | Weight exponent of `anisotropicSkyViewFactor`: a non-negative integer, RVT's `a_poly_level`. RVT's presets are low (level 4, minimum weight 0.4) and high (level 8, minimum weight 0.1). The power is signed as in RVT, `cos((a - A) / 2)^level`, so an odd level gives a weight below the minimum weight on the far side of the main direction, exactly like RVT. Defaults to `GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL`. |
| `anisotropyMinimumWeight` (optional) | `number` | Weight floor of `anisotropicSkyViewFactor` in `[0, 1]` (RVT: low 0.4, high 0.1). Defaults to `GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT`. |

{/* /parameters */}

{/* parameters: GPUTextureShading generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTextureShading` parameters

- Packer: `getGPUTextureShadingParameterValues(settings?)`, writes `GPU_TEXTURE_SHADING_PARAMETER_LENGTH` (12) `float32` values.

Settings of `getGPUTextureShadingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `detail` (optional) | `number` | Fractional Laplacian order `alpha` (Leland Brown's "detail"): band `k` gets weight `2^(-k * detail)`. 0 weighs every band equally; 1 approaches a Laplacian. Defaults to 0.5. |
| `gain` (optional) | `number` | Output multiplier. Defaults to 1. |
| `levelWeights` (optional) | `readonly number[]` | Explicit per-band weights (up to 8), overriding `detail`. |

{/* /parameters */}

#### `GPUTerrainHorizon`

Marches 16 azimuth sectors from every cell over an elevation grid to find horizon angles, then reduces them to sky-view factor and openness.

::lens{scenario=gpu-terrain.terrain-horizon rung=result caption="Sky-view factor, 1 on open ridges; the most enclosed cell is marked."}

#### How it computes

{/* lens:GPUTerrainHorizon */}

::lens{scenario=gpu-terrain.terrain-horizon rung=datum datum=2972 caption="A valley cell: the highest elevation angle in each of 16 sectors, then their mean sine."}
::lens{scenario=gpu-terrain.terrain-horizon rung=formula caption="Sky view is one minus the mean sine of the non-negative horizon angles."}
::lens{scenario=gpu-terrain.terrain-horizon rung=kernel caption="One fused kernel marches every sector of a cell, then a finalize kernel writes the outputs."}
::lens{scenario=gpu-terrain.terrain-horizon rung=graph caption="Elevation canonicalization, a fused horizon march and a sky-view finalize node; no loop."}

#### `GPUSolarShadowMask`

Turns a horizon map and a sun position into the visible fraction of the solar disk for every cell.

::lens{scenario=gpu-terrain.solar-shadow-mask rung=result caption="Sun visibility at 06:45 PDT; west-facing slopes in ridge shadow are dark."}

#### How it computes

{/* lens:GPUSolarShadowMask */}

::lens{scenario=gpu-terrain.solar-shadow-mask rung=datum datum=2832 caption="A shadowed cell: the horizon blended at the sun azimuth stands above the sun altitude."}
::lens{scenario=gpu-terrain.solar-shadow-mask rung=formula caption="The solar disk is clipped by the horizon; its visible area fraction is the shadow value."}
::lens{scenario=gpu-terrain.solar-shadow-mask rung=kernel caption="One invocation per cell blends two horizon sectors and clips the solar disk."}
::lens{scenario=gpu-terrain.solar-shadow-mask rung=graph caption="The horizon march feeds one shadow node; the sun is a parameters write."}

#### `GPUTerrainCastShadow`

Computes the visible fraction of the solar disk per cell with one hull sweep toward the sun, without a horizon map.

::lens{scenario=gpu-terrain.terrain-cast-shadow rung=result caption="Sun visibility from the sweep; the shadow edges match the horizon-map result."}

#### How it computes

{/* lens:GPUTerrainCastShadow */}

::lens{scenario=gpu-terrain.terrain-cast-shadow rung=datum datum=2832 caption="A shadowed cell: the horizon angle along the sun line exceeds the sun altitude."}
::lens{scenario=gpu-terrain.terrain-cast-shadow rung=formula caption="The horizon toward the sun is the largest elevation tangent along the sun line."}
::lens{scenario=gpu-terrain.terrain-cast-shadow rung=kernel caption="The sweep kernel walks digital lines along the sun azimuth and clips the solar disk."}
::lens{scenario=gpu-terrain.terrain-cast-shadow rung=graph caption="Elevation canonicalization and one sweep node; the sun azimuth sets the line family."}

#### `GPUSolarIrradiance`

Sums sun hours and direct insolation over a day of sun positions from a horizon map.

::lens{scenario=gpu-terrain.solar-irradiance rung=result caption="Direct insolation on the sloped surface over 5 August 2021; north slopes and ridge shade receive least."}

#### How it computes

{/* lens:GPUSolarIrradiance */}

::lens{scenario=gpu-terrain.solar-irradiance rung=datum datum=2908 caption="A shaded cell: 96 quarter-hour sun rows, each clipped by the horizon at its azimuth."}
::lens{scenario=gpu-terrain.solar-irradiance rung=formula caption="Insolation sums DNI times visibility times incidence times duration over the sun table."}
::lens{scenario=gpu-terrain.solar-irradiance rung=kernel caption="Each invocation loops over the 96 table rows with two horizon reads each."}
::lens{scenario=gpu-terrain.solar-irradiance rung=graph caption="Horizon march and Horn slope and aspect feed one accumulation node that reads the sun table."}

#### `GPUReliefShading`

Computes a multidirectional hillshade and a relief luminance from Horn normals of an elevation grid.

::lens{scenario=gpu-terrain.relief-shading rung=result caption="Relief luminance from four USGS lights weighted by aspect."}

#### How it computes

{/* lens:GPUReliefShading */}

::lens{scenario=gpu-terrain.relief-shading rung=datum datum=2907 caption="The steepest cell's 3×3 window and the four light weights."}
::lens{scenario=gpu-terrain.relief-shading rung=formula caption="Aspect-weighted mean of four lights, then the hillshade strength and exposure blend."}
::lens{scenario=gpu-terrain.relief-shading rung=kernel caption="The hillshade kernel loops over the lights; the compose kernel blends and clamps."}
::lens{scenario=gpu-terrain.relief-shading rung=graph caption="Two Sobel passes feed the hillshade node, then the compose node."}

#### `GPUTextureShading`

Approximates Brown's fractional Laplacian of an elevation grid with a Gaussian band-pass pyramid.

::lens{scenario=gpu-terrain.texture-shading rung=result caption="Texture shade in metres; ridges are positive and hollows negative."}

#### How it computes

{/* lens:GPUTextureShading */}

::lens{scenario=gpu-terrain.texture-shading rung=datum datum=3181 caption="A ridge cell: the band-pass differences at five scales and their weights."}
::lens{scenario=gpu-terrain.texture-shading rung=formula caption="Weighted band differences of five Gaussian blurs, scaled by the gain."}
::lens{scenario=gpu-terrain.texture-shading rung=kernel caption="Each level accumulates its weighted band into one buffer; finalize applies gain and validity."}
::lens{scenario=gpu-terrain.texture-shading rung=graph caption="An anchor and premultiply node, then blur passes and an accumulate node per level."}

#### `GPUSolarPosition`

Computes the sun azimuth, altitude and daylight flag for each longitude and latitude row at one instant.

::lens{scenario=gpu-terrain.solar-position rung=result caption="Sun altitude at eight northern California sites at 13:00 PDT; the table below lists all outputs."}

#### How it computes

{/* lens:GPUSolarPosition */}

::lens{scenario=gpu-terrain.solar-position rung=datum datum=0 caption="Greenville at 20:00 UTC: hour angle, declination and the resulting angles."}
::lens{scenario=gpu-terrain.solar-position rung=formula caption="NOAA hour angle, altitude and azimuth with refraction, in float32."}
::lens{scenario=gpu-terrain.solar-position rung=kernel caption="One invocation per site evaluates the full NOAA series."}
::lens{scenario=gpu-terrain.solar-position rung=graph caption="A single kernel node; time is a parameters write."}

### `GPUSimpleLocalRelief`, `GPUMultiScaleRelief`, `GPULocalDominance`, and `GPUReliefBlend`

Archaeological and micro-relief visualizations with Relief Visualization Toolbox (RVT; Kokalj and
Somrak 2019) parity. The float64 test oracles follow RVT-py's formulas, including edge padding, NaN
handling and Python's half-even rounding.

- `GPUSimpleLocalRelief` (SLRM; Hesse 2010): `ve * (z - mean(z))` over a `(2r + 1)²` window with
  edge-clamped coordinates and nodata excluded. On grids up to
  `getGPUTerrainSummedAreaMaximumCellCount(device)` cells (about 3344 x 3344 on a default device) the mean is an O(1) box query
  over an exact 64-bit summed-area table of elevations quantized to `quantum` (a power of two,
  default `2^-16`), with RVT's edge replication applied as edge-strip weights and the deviation formed
  from exact integers; the error against RVT is at most `quantum / 2` (zero for `|z| >= 128`) plus a
  few float32 ULP, at any radius up to 23170. The GPU selects the first finite valid elevation,
  snaps it to the quantum, and subtracts it before building the table. Only offsets from that
  anchor of magnitude `2^31 * quantum` or more (32768 by default) are invalid; absolute elevation
  is unrestricted. Larger grids use a separable O(radius) filter with anchored,
  centre-relative sums, so any radius keeps sub-millimetre precision at alpine elevations. The radius is a
  per-frame setting: `maximumRadius` (compile time) sizes `requiredHalo` and bounds the loops, and
  `getGPUSimpleLocalReliefParameterValues({radius})` picks any integer in `[1, maximumRadius]` per
  frame with the same pipelines (an omitted radius uses `maximumRadius`).
- `GPUMultiScaleRelief` (MSRM; Orengo and Petrie 2018): RVT's radii from `resolution`,
  `featureMinimum`, `featureMaximum` and `scalingFactor` (`getGPUMultiScaleReliefRadii`). RVT's sum of
  consecutive low-pass differences telescopes, so the contributor runs only two mean filters, which
  share one summed-area table on grids within the cell cap (same `quantum` option and bounds as SLRM).
  `cellSizeMode` (default `'uniform'`, unchanged square windows) accepts `'geographic'` and
  `'web-mercator'` on axis-aligned grids: the RVT index ladder is unchanged, each nonzero radius `k^s`
  becomes `k^s * resolution` ground metres, and every row converts it to rectangular pixel half-widths
  with its WGS84 ground cell (rounded, at least 1). The summed-area path evaluates these rectangles
  exactly; the separable path (large grids) averages rows filtered with their own horizontal radius.
  `requiredHalo` is the largest coarsest half-width over all rows.
- `GPULocalDominance` (Hesse 2016): the mean dominance of the observer (`observerHeight` above the
  cell) over an annulus of baked shifts. Distances are in pixels, as in RVT.
- `GPUReliefBlend` (VAT): up to five float32 layers, bottom to top, each stretched
  `clamp((x - min)/(max - min))`, optionally inverted, and blended (normal, multiply, screen, overlay,
  soft light, luminosity) with an opacity. Presets `GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL` (hillshade;
  slope 0-50° inverted, luminosity 50 %; positive openness 68-93°, overlay 50 %; SVF 0.7-1, multiply
  25 %) and `GPU_RELIEF_BLEND_VAT_FLAT`. Feed it from `GPUReliefShading`, `GPUTerrainDerivatives` and
  `GPUTerrainHorizon`. Percent-clip stretches need a histogram, so pass min and max (for example from
  `GPURasterStatistics`).

At 1024²: SLRM r=20 takes 3.3 ms, MSRM at radius 100 takes 5.4 ms, local dominance with the RVT
defaults (264 taps) takes 84.5 ms, and a four-layer VAT blend takes 1.2 ms. All style values are
per-frame `parameters`. Each contributor takes `grid` instead of `width` and `height` and writes into
`output` (`relief`, `dominance`, `blend`, `color`, and `outputValidity`); the blend `layers` stay an
input. The blend layer limit is `GPU_RELIEF_BLEND_MAXIMUM_LAYER_COUNT` and the local dominance
offset limit `GPU_LOCAL_DOMINANCE_MAXIMUM_SHIFT_COUNT`.

{/* parameters: GPULocalDominance generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalDominance` parameters

- Packer: `getGPULocalDominanceParameterValues(settings?)`, writes `GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULocalDominanceParameterValues`:

| Field | Type | Description |
|---|---|---|
| `observerHeight` (optional) | `number` | Height of the observer above the surface, in elevation units. Positive. Defaults to 1.7. |
| `verticalExaggeration` (optional) | `number` | Vertical exaggeration applied to the elevation. Defaults to 1. |

{/* /parameters */}

{/* parameters: GPUMultiScaleRelief generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMultiScaleRelief` parameters

- Packer: `getGPUMultiScaleReliefParameterValues(settings?)`, writes `GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUMultiScaleReliefParameterValues`:

| Field | Type | Description |
|---|---|---|
| `verticalExaggeration` (optional) | `number` | Vertical exaggeration applied to the relief. Defaults to 1. |

{/* /parameters */}

{/* parameters: GPUReliefBlend generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUReliefBlend` parameters

- Packer: `getGPUReliefBlendParameterValues(layers)`, writes `GPU_RELIEF_BLEND_PARAMETER_LENGTH` `float32` values.
  - `layers` (`readonly GPUReliefBlendLayerSettings[]`)

{/* /parameters */}

{/* parameters: GPUSimpleLocalRelief generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSimpleLocalRelief` parameters

- Packer: `getGPUSimpleLocalReliefParameterValues(settings?)`, writes `GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUSimpleLocalReliefParameterValues`:

| Field | Type | Description |
|---|---|---|
| `verticalExaggeration` (optional) | `number` | Vertical exaggeration applied to the relief. Defaults to 1. |
| `radius` (optional) | `number` | Window half-width in pixels, an integer in `[1, maximumRadius]`. Defaults to `maximumRadius`. The packer throws for a value that is not an integer of at least 1; a value above `maximumRadius` written to the buffer by other means is clamped to `maximumRadius` by the kernel. |

{/* /parameters */}

#### `GPUSimpleLocalRelief`

Subtracts the local window mean from an elevation raster, leaving small features such as banks and mounds.

::lens{scenario=gpu-terrain.simple-local-relief rung=result caption="Elevation minus the 17 × 17 window mean; ridges are positive, hollows negative."}

#### How it computes

{/* lens:GPUSimpleLocalRelief */}

::lens{scenario=gpu-terrain.simple-local-relief rung=datum datum=3181 caption="The ridge cell at column 45, row 49 against its 17 × 17 window mean, edge samples included."}
::lens{scenario=gpu-terrain.simple-local-relief rung=formula caption="Centre elevation minus the edge-replicated window mean, times vertical exaggeration."}
::lens{scenario=gpu-terrain.simple-local-relief rung=kernel caption="The deviation kernel reads box sums from the exact summed-area table."}
::lens{scenario=gpu-terrain.simple-local-relief rung=graph caption="Elevation feeds the summed-area table, then one deviation kernel."}

#### `GPUMultiScaleRelief`

Averages the differences between consecutive low-pass filters of an elevation raster to isolate features between two sizes.

::lens{scenario=gpu-terrain.multi-scale-relief rung=result caption="Relief between feature sizes of 360 and 1200 metres; ridges are positive, hollows negative."}

#### How it computes

{/* lens:GPUMultiScaleRelief */}

::lens{scenario=gpu-terrain.multi-scale-relief rung=datum datum=2973 caption="A valley cell: its deviations from the radius 1 and radius 5 means, divided by four."}
::lens{scenario=gpu-terrain.multi-scale-relief rung=formula caption="The telescoped sum: the difference of the coarsest and finest deviations over n minus i."}
::lens{scenario=gpu-terrain.multi-scale-relief rung=kernel caption="Two mean filters share one table; a combine kernel scales their difference."}
::lens{scenario=gpu-terrain.multi-scale-relief rung=graph caption="One summed-area table feeds two deviation filters and the combine node."}

#### `GPULocalDominance`

Measures how far an observer standing above each cell of an elevation raster rises over rings of surrounding terrain.

::lens{scenario=gpu-terrain.local-dominance rung=result caption="Local dominance in rings of 5 to 10 cells; the most dominant cell is marked."}

#### How it computes

{/* lens:GPULocalDominance */}

::lens{scenario=gpu-terrain.local-dominance rung=datum datum=3181 caption="The ridge cell's 144 ring samples: each positive rise adds its ring-weighted share."}
::lens{scenario=gpu-terrain.local-dominance rung=formula caption="Positive rises above each ring sample, ring-weighted and normalized so a flat surface scores one."}
::lens{scenario=gpu-terrain.local-dominance rung=kernel caption="One invocation per cell sums 144 baked ring samples."}
::lens{scenario=gpu-terrain.local-dominance rung=graph caption="Canonical elevation feeds a single sampling node."}

#### `GPUReliefBlend`

Combines hillshade and slope layers from GPUTerrainDerivatives into one grey image with stretches, inversion and blend modes.

::lens{scenario=gpu-terrain.relief-blend rung=result caption="Hillshade blended with inverted slope into one grey image over the elevation context."}

#### How it computes

{/* lens:GPUReliefBlend */}

::lens{scenario=gpu-terrain.relief-blend rung=datum datum=2907 caption="The steepest cell: both layers stretched, the slope inverted, then mixed at 50 percent."}
::lens{scenario=gpu-terrain.relief-blend rung=formula caption="Each layer is stretched, then composited bottom to top with its blend mode and opacity."}
::lens{scenario=gpu-terrain.relief-blend rung=kernel caption="A single kernel stretches each layer and composites them bottom to top."}
::lens{scenario=gpu-terrain.relief-blend rung=graph caption="Derivatives write the layers; the blend node composites them."}

### `GPUTerrainFlow`

Turns an elevation grid into filled elevation, D8 directions, cell classes, upstream-cell accumulation and a stream mask.

::lens{scenario=gpu-terrain.terrain-flow rung=result caption="Upstream cells per cell after the fill; the outlet with the most upstream cells is marked."}

Hydrology on an elevation tile: optional Planchon–Darboux depression filling with a per-frame
epsilon, D8 flow directions (ESRI codes, cell-size-aware diagonal distances, uniform, Web Mercator,
or geographic cells) with flat, pit, and outlet classes, flow accumulation in cells or ground area
optionally weighted by a per-cell runoff view, and a stream mask with a per-frame threshold.
With `fillEpsilon` 0, filling is a tiled spill graph (Barnes 2016): tile-local fill with outlet
labels, a Borůvka spanning tree of the spill graph, and bottleneck paths to the grid boundary, in
O(log n) gated GPU rounds whatever the basin shape, bit-identical to the Planchon–Darboux fixpoint
(and to pysheds `fill_depressions`); `output.status.fillIterationCount` then reports Borůvka rounds.
A `fillEpsilon` above 0 selects, per frame on the GPU, the GPU-gated tiled min-relaxation, which is
also used on grids whose `8 · (cells + 1)` bytes exceed one storage binding. D8 `'cells'` accumulation without `runoff` is tree
contraction on the Euler tour of the receiver forest: every cell enumerates its children in fixed
direction order, and gated pointer jumping ranks the `2 n` tour edges in `ceil(log2(2 n)) + 1`
rounds, so a cell's count is an exact integer (written as float32) for any stem length and
`maximumAccumulationIterations` does not apply. Weighted accumulation (`runoff`, `'area'`) and the
other routings are a deterministic pull over dependency order with downstream walking, bounded by
`maximumAccumulationIterations`. Everything stops early on the GPU and reports convergence flags, so
nothing is read back and accumulation is bit-reproducible. `maximumFillIterations` and
`maximumFlatIterations` default to 512 (up to 1024); after 96 plain tiled iterations each phase adds
directional row and column sweeps so serpentine and spiral basins converge. Check
`output.status.fillConverged` and `output.status.flatsConverged` for longer ones. The optional
one-row outputs `output.status.fillIterationCount`, `output.status.flatsIterationCount` (the sum of
the three flat relaxations) and `output.status.accumulationIterationCount` report the iterations
executed, including the final unchanged one, so a budget can be tuned (the SF terrain used 14 of 256
fill and 2 of 128 accumulation rounds). Every unrolled loop has one gate node per four rounds; rounds
after convergence inside a group are idempotent no-ops, so results and counts are unchanged and the
graph has about a third fewer nodes.

The grid supplies the width, height, cell size and latitude band, so the settings carry only the
algorithm values (`fillEpsilon`, `streamThreshold`, `flowExponent`) and the packer takes the grid as
its second argument.

When `maximumAccumulationIterations` runs out on the pull path, unresolved cells keep NaN `accumulation`, and their
`streams` word is `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` (`0xffffffff`), never 0: the unresolved cells
are the downstream trunk, so 0 would silently drop the main river. Test `streams == 1` for "is a
stream". `GPUTerrainHeightAboveDrainage` and `GPUTerrainStreamOrder` propagate the unresolved word.

```ts
const grid = makeGPURasterMetadata({width, height, cellSize: 30});
graph.add(new GPUTerrainFlow({
  grid, elevation, parameters: parameters.importToGraph(graph), fillDepressions: true,
  output: {flowDirections, accumulation, streams, status: {fillConverged, accumulationConverged}}
}));
parameters.write(getGPUTerrainFlowParameterValues({fillEpsilon: 0.01, streamThreshold: 500}, grid));
```

`resolveFlats: true` routes flow across flats without changing elevations (Barnes, Lehman, and
Mulla 2014). Hop distances toward the flat's draining edge and away from its higher edge are GPU
relaxations over equal-height cells. Each drainable flat cell drains to the equal-height neighbor
whose `2 * towardLower + (flatMaximum - awayFromHigher)` is strictly lower. Pair it with
`fillDepressions` and `fillEpsilon: 0` to drain filled depressions as true flats;
`output.status.flatsConverged` reports the three relaxations. `flowRouting` selects how `accumulation` and `streams` split flow:
`'d8'` (default), `'d-infinity'` (Tarboton 1997 facets), `'mfd-freeman'` (Freeman 1991,
`tan(beta)^p`, `p = 1.1`) or `'mfd-quinn'` (Quinn et al. 1991, contour-length weighted, `p = 1`).
Set `p` per frame with `flowExponent`. Non-D8 accumulation uses the same deterministic pull, with
each donor's fraction recomputed from the surface, so it is reproducible bit for bit.
`flowDirections` and `cellClasses` stay D8.

`flatContinuation` opts into 64 rounds per pass per execution (override with
`maximumFlatIterations`). Supply `lower`, `higher`, and `maximum`, each with a zero-initialized
`uint32` state of `getGPURasterRelaxationContinuationStateLength(grid)` words and a `float32`
values view with one element per cell. All six views must be distinct imported buffers, since
transients can alias between executions. Completed stages stay frozen; the maximum pass starts
only after the higher-edge distances converge. `status.flatsIterationCount` is cumulative across
executions. Outputs retain the unresolved contract until all three stages converge. Keep inputs
unchanged while resuming, or set `GPU_RASTER_RELAXATION_RESTART_WORD` in any stage state to restart
all three. The execution after complete convergence also restarts all three, as cost distance does.

Outputs may share one buffer over disjoint byte ranges. For example, `status.fillConverged` and
`status.accumulationConverged` can be two one-row views of an 8-byte summary buffer. Outputs that one
pass writes together (`filledElevation`, `flowDirections` and `cellClasses`; `accumulation` and
`streams`) must start in different 256-byte binding windows. No output may share a buffer with
an input.

{/* parameters: GPUTerrainFlow generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainFlow` parameters

- Packer: `getGPUTerrainFlowParameterValues(settings, grid)`, writes `GPU_TERRAIN_FLOW_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainFlowParameterValues`:

| Field | Type | Description |
|---|---|---|
| `fillEpsilon` (optional) | `number` | Minimum rise per D8 step on filled surfaces, in elevation units. Negative or non-finite values act as 0. Defaults to 0. Must exceed the float32 spacing at the elevation magnitude to create drainable gradients; 0 leaves flats. |
| `streamThreshold` (optional) | `number` | Accumulation at or above which `streams` is 1. Defaults to `Infinity` (no streams). |
| `flowExponent` (optional) | `number` | Exponent `p` of the multiple-flow-direction weights. Values that are not finite and positive (including the default 0) select the routing's published exponent: 1.1 for `'mfd-freeman'`, 1 for `'mfd-quinn'`. Ignored by `'d8'` and `'d-infinity'`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUTerrainFlow */}

::lens{scenario=gpu-terrain.terrain-flow rung=datum datum=1826 caption="A main-stream cell: drops to eight neighbours, diagonals over 1.414 cell sizes, receiver outlined."}
::lens{scenario=gpu-terrain.terrain-flow rung=formula caption="Steepest D8 receiver on the filled surface, the epsilon fill and accumulation as the kernels apply them."}
::lens{scenario=gpu-terrain.terrain-flow rung=iterations caption="Filled surface in each recorded relaxation round; accumulation then resolves in a few pull rounds."}
::lens{scenario=gpu-terrain.terrain-flow rung=kernel caption="The direction kernel: one invocation per cell scans eight neighbours and keeps the steepest drop."}
::lens{scenario=gpu-terrain.terrain-flow rung=graph caption="Fill relaxation loop, direction pass, gated accumulation rounds and the stream mask."}

### `GPUTerrainHeightAboveDrainage`, `GPUTerrainWatersheds`, `GPUTerrainStreamOrder`, and `GPUTerrainHydrologicIndices`

Products derived from `GPUTerrainFlow` outputs. Each takes a `grid` and ESRI D8 `flowDirections`
(any D8 grid, for example one imported from TauDEM or Whitebox), plus `streams` or `accumulation`
where needed, and writes into an `output` bag. Their inputs stay inputs; the convergence words are
`output.status.converged` and `output.status.iterationCount`, and the iteration caps are
`maximumIterations`.

- `GPUTerrainHeightAboveDrainage`: HAND (Rennó et al. 2008), the height of each cell above the
  first stream cell on its D8 path, with an optional `output.drainageCells` index output. A path that
  first meets an unresolved stream word gives NaN and `GPU_TERRAIN_DRAINAGE_UNRESOLVED`.
- `GPUTerrainWatersheds`: labels every cell (`output.labels`) with its nearest downstream pour point (nested pour
  points give nested watersheds). Without pour points it labels drainage basins by their outlet or
  pit cell.
- `GPUTerrainStreamOrder`: Strahler order on the stream mask (`output.streamOrder`), with an optional
  `output.status.iterationCount`.
  Unresolved stream words and the stream cells below them keep the unresolved order `0xffffffff`.
- `GPUTerrainHydrologicIndices`: computes, from contributing area in m²
  (`accumulationUnits: 'area'`, any routing), into `output.slope`, `output.specificCatchmentArea`,
  `output.wetnessIndex` and `output.streamPowerIndex`:
  - specific catchment area `a = A / b`;
  - topographic wetness `ln(a / tan β)`;
  - stream power `a · tan β`;

  with `tan β` the D8 descent slope clamped to a per-frame `minimumSlope` (default 0.001).

HAND and watersheds resolve D8 paths by GPU pointer jumping, which needs about log₂(path length)
gated rounds. Stream order is Strahler by levels on the Euler tour of the stream forest: one
list ranking, then per level a scatter, a prefix scan (`GPUScan`) and a gather, at most
`floor(log2(n))` levels, each gated on the GPU; `maximumIterations` bounds only the pull rounds
used when the `2 n` tour words exceed one storage binding (grids above 4096 x 4096 cells), and
`status.iterationCount` reports the ranking rounds. A receiver cycle (caller-supplied directions
only) leaves every cell of its component at `0xffffffff` and clears `converged`. Every iterative contributor stops early on the
GPU and can publish `output.status.converged`.

Chain the contributors by wrapping a flow output with `makeGPURasterBand(view, grid)` where a
downstream contributor takes a band (the `elevation` of HAND and the hydrologic indices); the
direction, stream and accumulation views feed the other inputs directly.

```ts
const grid = makeGPURasterMetadata({width, height, cellSize: 30});
graph.add(new GPUTerrainFlow({
  grid, elevation, parameters: flowParameters.importToGraph(graph),
  fillDepressions: true, resolveFlats: true, flowRouting: 'mfd-freeman', accumulationUnits: 'area',
  output: {filledElevation, flowDirections, accumulation, streams}
}));
const surface = makeGPURasterBand(filledElevation, grid, {id: 'filled'});
graph.add(new GPUTerrainHeightAboveDrainage({
  grid, elevation: surface, flowDirections, streams, output: {heightAboveDrainage}
}));
graph.add(new GPUTerrainWatersheds({grid, flowDirections, pourPoints, output: {labels}}));
graph.add(new GPUTerrainStreamOrder({grid, flowDirections, streams, output: {streamOrder}}));
graph.add(new GPUTerrainHydrologicIndices({
  grid, elevation: surface, accumulation, parameters: indexParameters.importToGraph(graph),
  output: {wetnessIndex}
}));
```

{/* parameters: GPUTerrainHydrologicIndices generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainHydrologicIndices` parameters

- Packer: `getGPUTerrainHydrologicIndicesParameterValues(settings, grid)`, writes `GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainHydrologicIndicesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minimumSlope` (optional) | `number` | Lower bound applied to `tan(beta)` before the indices are formed, so filled flats do not give an infinite wetness index. Negative or non-finite values act as 0 (no clamp: zero slopes give `+Infinity` wetness and zero stream power). Defaults to `GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE` (0.001, about 0.06 degrees). |

{/* /parameters */}

#### `GPUTerrainHeightAboveDrainage`

Subtracts the elevation of each cell's first downstream stream cell from its own, found by pointer jumping along D8 paths.

::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=result caption="Height above the nearest stream cell; the highest cell is marked."}

#### How it computes

{/* lens:GPUTerrainHeightAboveDrainage */}

::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=datum datum=3443 caption="A ridge cell: its D8 receiver and the filled surface it drains over."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=formula caption="HAND is one subtraction of two elevations once the drainage cell is known."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=iterations caption="Pointer-jumping rounds executed, with the convergence word the kernel publishes."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=kernel caption="The height kernel reads the converged pointer and subtracts the drainage elevation."}
::lens{scenario=gpu-terrain.terrain-height-above-drainage rung=graph caption="Flow routing feeds a pointer-jumping loop, then one height kernel."}

#### `GPUTerrainWatersheds`

Labels every cell with the outlet or pit cell its D8 path reaches, resolved by pointer jumping.

::lens{scenario=gpu-terrain.terrain-watersheds rung=result caption="Drainage basins, one label per outlet or pit cell."}

#### How it computes

{/* lens:GPUTerrainWatersheds */}

::lens{scenario=gpu-terrain.terrain-watersheds rung=datum datum=1826 caption="A main-stream cell: the receiver it follows toward its basin outlet."}
::lens{scenario=gpu-terrain.terrain-watersheds rung=iterations caption="Pointer-jumping rounds executed, with the convergence word the kernel publishes."}
::lens{scenario=gpu-terrain.terrain-watersheds rung=kernel caption="The label kernel copies the terminal cell index reached by each pointer."}
::lens{scenario=gpu-terrain.terrain-watersheds rung=graph caption="Flow routing feeds a pointer-jumping loop, then the label kernel."}

#### `GPUTerrainStreamOrder`

Assigns Strahler order to stream cells by levels on the Euler tour of the stream forest.

::lens{scenario=gpu-terrain.terrain-stream-order rung=result caption="Strahler order along the 40-cell stream network."}

#### How it computes

{/* lens:GPUTerrainStreamOrder */}

::lens{scenario=gpu-terrain.terrain-stream-order rung=datum datum=1761 caption="A confluence cell: its stream donors and the receiver it drains to."}
::lens{scenario=gpu-terrain.terrain-stream-order rung=iterations caption="List-ranking rounds of the Euler tour, with the convergence word the kernel publishes."}
::lens{scenario=gpu-terrain.terrain-stream-order rung=kernel caption="Per-level scatter and gather kernels: the same code runs for every Strahler level."}
::lens{scenario=gpu-terrain.terrain-stream-order rung=graph caption="Tour link, gated ranking rounds, then gated order levels."}

#### `GPUTerrainHydrologicIndices`

Combines upstream area and D8 descent slope into specific catchment area, topographic wetness and stream power.

::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=result caption="Topographic wetness: high in valley bottoms; the wettest cell is marked."}

#### How it computes

{/* lens:GPUTerrainHydrologicIndices */}

::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=datum datum=1826 caption="A valley cell: its steepest D8 descent and the upstream area it receives."}
::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=formula caption="Wetness and stream power from catchment width, area and the clamped D8 slope."}
::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=kernel caption="One kernel scans eight neighbours for the steepest descent, then forms the indices."}
::lens{scenario=gpu-terrain.terrain-hydrologic-indices rung=graph caption="Flow routing in square metres feeds one index kernel; no loop."}

### `GPUTerrainFlowField`

Deflects a uniform horizontal wind over an elevation grid onto the terrain tangent plane, giving a wind vector per cell.

::lens{scenario=gpu-terrain.terrain-flow-field rung=result caption="Wind speed from an 8 m/s south-westerly; slopes facing the wind slow it."}

Terrain-following wind for the flow contributors. A uniform horizontal wind is projected onto the
terrain tangent plane, `v = w − (w·g) g / (1 + |g|²)`, using central-difference gradients in ground
meters (uniform, Web Mercator, or geographic cells, one-sided at the border). Wind across a slope
is unchanged; wind into a slope slows, so flow bends around ridges. The output is the `velocities`
format of `GPUParticleAdvection`, `GPUStreamlines`, and `GPULineIntegralConvolution`, written in the
raster frame (u along columns, v along rows) to `output.velocities`. Cells next to nodata are NaN.
Wind and vertical exaggeration are per-frame settings; the cell size and latitude band come from
the `grid`, which the packer takes as its second argument.

```ts
const grid = makeGPURasterMetadata({width, height, cellSize: 30});
graph.add(new GPUTerrainFlowField({
  grid, elevation, parameters: windParameters.importToGraph(graph), output: {velocities}
}));
windParameters.write(getGPUTerrainFlowFieldParameterValues({wind: [4, -2]}, grid));
graph.add(new GPUParticleAdvection({
  velocities, fieldWidth: width, fieldHeight: height,
  parameters: advectionParameters.importToGraph(graph),
  uint32Parameters: advectionWords.importToGraph(graph),
  state: {positions, ages, generations}
}));
```

{/* parameters: GPUTerrainFlowField generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainFlowField` parameters

- Packer: `getGPUTerrainFlowFieldParameterValues(settings, grid)`, writes `GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUTerrainFlowFieldParameterValues`:

| Field | Type | Description |
|---|---|---|
| `wind` | `readonly [number, number]` | Horizontal wind `[x, y]` in the raster frame: x along increasing columns, y along increasing rows (south for a north-up raster whose row 0 is the northern edge). Any speed unit; the output velocities share it. |
| `verticalExaggeration` (optional) | `number` | Factor applied to the elevation gradient before deflection. Defaults to 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUTerrainFlowField */}

::lens{scenario=gpu-terrain.terrain-flow-field rung=datum datum=3100 caption="A slope facing the wind: the gradient from four neighbours and the removed component."}
::lens{scenario=gpu-terrain.terrain-flow-field rung=formula caption="The wind loses its component along the gradient, divided by one plus the squared gradient."}
::lens{scenario=gpu-terrain.terrain-flow-field rung=kernel caption="One invocation per cell reads four neighbours, forms the gradient and writes (u, v)."}
::lens{scenario=gpu-terrain.terrain-flow-field rung=graph caption="Elevation canonicalization feeds one deflection kernel; no loop."}

### Parameter packers

Every contributor reads its per-frame settings from a packed `float32` view, `parameters`; the
packer writes it and its `*_PARAMETER_LENGTH` constant sizes it. Each contributor section ends with a
generated "parameters" subsection listing its packer, length constant and settings fields. Packers
with a `grid` argument take the frame's `GPURasterMetadata` and write its cell size and row edges, so
those never appear in the settings object.

Helpers that derive packer inputs or kernel tables:

- point horizon: `getGPUPointHorizonDistanceLattice`, `getGPUPointHorizonSegments`;
- horizon: `getGPUTerrainHorizonDirection`, `getGPUTerrainHorizonStepDistances`;
- solar: `getGPUSolarIrradianceSunTable`, `getGPUSolarDirectNormalIrradiance`,
  `getSolarRefractionDegrees`, `getSolarTimeParameter`;
- texture shading: `getGPUTextureShadingKernel`, `getGPUTextureShadingCascadeSigmas`;
- relief: `getGPUMultiScaleReliefRadii`, `getGPULocalDominanceShifts`.

### Capped loops and compile cost

- A loop stopped by its iteration limit never writes a value that reads as converged (library
  rule 9): unresolved cells carry NaN or a `*_UNRESOLVED` constant, see each contributor's TSDoc.
- A loop index never reaches WGSL text: gated rounds, horizon sectors, cumulative-viewshed
  batches, point-horizon chunks and jump-flood steps read their position from a GPU word, so
  every iteration of a kernel shares one pipeline (default `GPUTerrainFlow` with fill and flats
  compiles 15 relaxation pipelines, not about 60).
- `GPUTerrainCumulativeViewshed` sizes each dispatch to about 2^28 sight-line samples
  (`GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH`); a large grid splits one observer
  across dispatches.
- `GPUPointHorizonVisibility` stops the occlusion test one ground pixel before the target by
  default; mt-image's 150 m is `GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE`.
- `GPUSimpleLocalRelief` and `GPUTerrainVectorRuggedness` compile for `maximumRadius` and read
  the window `radius` per frame from `parameters`, so changing the radius does not recompile.
- D-infinity and MFD accumulation compute each cell's routing once (`routing-prepare`) instead of
  re-deriving every donor's facets in every round.
