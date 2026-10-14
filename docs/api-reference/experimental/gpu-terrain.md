import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Terrain

<ExperimentalDocsTabs active="gpu-terrain" />

:::caution Experimental
`@luma.gl/experimental/gpu-terrain` is an initial, experimental entry point. APIs may change between
releases without a deprecation period.
:::

## Overview

`@luma.gl/experimental/gpu-terrain` provides analysis contributors for elevation tiles: Horn slope,
aspect and hillshade, marching-squares contours, viewshed visibility, hydrology (depression
filling, flat resolution, D8, D-infinity and multiple-flow-direction routing, flow accumulation,
streams, height above drainage, watersheds, stream order, hydrologic indices), batched line of sight
and cumulative viewsheds, point horizon profiles, horizon angles, solar position, shadow masks and
irradiance, relief shading, texture shading, archaeological relief visualizations (RVT), terrain-RGB
decoding, summits and critical points, curvature, geomorphons, ruggedness, topographic position,
Weiss landforms, and terrain-following wind fields. Each contributor is an algorithm that declares its
resources and command nodes into a caller-owned
[`GPUCommandGraph`](./gpu-core/gpu-command-graph.md) through `getCommandNodes(graph)`
(`GPUCommandNodeProducer`). It never compiles, submits, encodes, or reads back.

The contributors compose public primitives from [GPU Raster](./gpu-raster/README.md) and
[GPU Core](./gpu-core/concepts.md); new WGSL is written only where no primitive fits.

## When to use it

Use GPU Terrain when the elevation values themselves are the product: slope and aspect for
suitability models, contour or viewshed geometry for analysis, flow accumulation for hydrology, or
analytic shading textures that stay on the GPU. Per-frame observers, light directions, contour
levels, and cell-size settings are read from storage views, so interaction never recompiles.

Prefer the generic GPU Raster operations when the field is not an elevation model or you need a
different kernel:

- `GPUTerrainDerivatives` is Horn's method (two Sobel `GPURasterGradient` passes plus a fused shade
  kernel) with terrain cell-size modes, a z factor, and latitude handling. Use `GPURasterGradient`
  directly for a single signed derivative of any raster.
- `GPUTerrainContours` runs one `GPURasterContours` pipeline per elevation level and reports one
  overflow flag. Use `GPUIsolines` and `GPUIsobands` from GPU Raster when you need stitched
  polylines or filled bands, and `GPURasterContours` for a single level.
- The solar and shading contributors write analytic values and textures. If you only need lit
  terrain for presentation, the lighting of a terrain renderer is simpler.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUParameterBuffer,
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';
import {makeGPURasterMetadata} from '@luma.gl/experimental/gpu-raster';

const graph = new GPUCommandGraph(device, {id: 'terrain'});
const parameters = new GPUParameterBuffer(device, {
  id: 'derivatives-parameters',
  format: 'float32',
  length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
});
// 30 m cells, top-left corner at (originX, originY), rows advancing south.
const grid = makeGPURasterMetadata({width, height, cellSize: 30, origin: [originX, originY]});

// `elevation`, `slope`, and `hillshade` are graph views created on `graph`.
graph.add(new GPUTerrainDerivatives({
  grid, elevation, parameters: parameters.importToGraph(graph), output: {slope, hillshade}
}));
const compiled = await graph.compileAsync(); // once

// every frame: rewrite parameters (cell size and latitude band come from `grid`)
parameters.write(getGPUTerrainDerivativesParameterValues({azimuthDegrees: 315}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

The application compiles, encodes, submits, and decides whether any bounded result is read back.
[Running contributors](./gpu-spatial-analysis.md#running-contributors) covers `compileAsync()`,
rebuilding without a stall, submission and why `parameters` is `undefined`.

## Conventions

- **One grid.** Every contributor takes `grid: GPURasterMetadata` (`makeGPURasterMetadata` builds
  an axis-aligned one); its width and height size the kernels. Cell sizes and the latitude band
  are never typed twice: each packer that needs them takes the frame's grid as its second
  argument, `getGPU<Class>ParameterValues(settings, grid)`, and reads them from `grid.affine`
  in world units (meters for `'uniform'`, EPSG:3857 meters for `'web-mercator'`, degrees for
  `'geographic'`, see `GPURasterCellSizeMode`).
- **`parameters` in, `output` out.** The packed per-frame view is `parameters` (a second view of
  another format is `uint32Parameters`). Outputs live in one `output` bag; the validity output is
  `output.outputValidity` (a `validity` prop is always an input mask) and one-row status words are
  in `output.status` (`converged`, `iterationCount`, `overflow`). `makeGPURasterBand(view, grid)`
  turns one contributor's output view into the next contributor's input band.
- **Compile-time limits say so.** Values that size kernels are named `maximum*`
  (`maximumRadius`, `maximumFillIterations`); the value actually used each frame (`radius`, a
  distance) is in `parameters`.
- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids` and a `status` with `count`, `overflow` and optional `requiredCount`). `status.count` is clamped
  to `ids.length` and can be an indirect draw instance count; `status.overflow` is rewritten every
  encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](./gpu-terrain.md#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](./gpu-network.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](./gpu-spatial-analysis.md#shared-ports-status-and-planning).

## API

### `GPUTerrainDerivatives`, `GPUTerrainContours`, `GPUTerrainViewshed`, `GPUTerrainLineOfSight`, and `GPUTerrainCumulativeViewshed`

Elevation-tile analysis composed from `gpu-raster` operators. `GPUTerrainDerivatives` computes
Horn slope, aspect, and hillshade (uniform, Web Mercator, or geographic cell sizes) with an optional
hillshade texture. `GPUTerrainContours` extracts marching-squares segments for several fixed or
per-frame levels with one aggregate overflow flag (`output.status.overflow`). Every contributor
takes a `grid` (`GPURasterMetadata`), reads `parameters`, and writes into an `output` bag. The grid
supplies the width, height, cell size and latitude band, so the packers take it as their second
argument. The three visibility contributors share one sight-line model on planar, projected-metre
grids (cell size in metres from the grid):

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

Each sight line samples the straight segment at one-pixel steps with bilinear elevation and ignores
samples with an invalid corner.

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
  [`GPURasterExtremaPyramid`](./gpu-raster/operations-analysis.md) and skips blocks that provably
  cannot change the code. Results are bit-identical to `'march'` (the default).
- **Shared pyramid.** Pass `pyramid` (a `GPURasterExtremaPyramidOutput`) to reuse one pyramid across
  `GPUTerrainViewshed`, `GPUTerrainLineOfSight`, `GPUTerrainCumulativeViewshed`,
  `GPUPointHorizonProfile` and `GPUPointHorizonVisibility`. It must cover the same grid with a
  bilinear footprint. Building one costs 0.3 to 0.9 ms, but the pyramid-skip kernels are still about
  2x slower than `'march'` for viewsheds, so prefer `'march'` there.

Without `toleranceParameters` and with `'march'`, `GPUTerrainViewshed` behaves as before.

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
| `observerHeight` (optional) | `number` | Eye height above the terrain in elevation units. Defaults to 1.7. |
| `targetHeight` (optional) | `number` | Height added to the target elevation. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground radius in metres; `<= 0` means unlimited. Defaults to 0. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/metres. Defaults to 0. |
| `toleranceMeters` (optional) | `number` | Constant tolerance band half-width in metres. Defaults to 0. |
| `tolerancePerKilometer` (optional) | `number` | Additional band half-width per kilometre of distance, in metres. Defaults to 0. |
| `targetIgnoreDistance` (optional) | `number` | Metres at the target end that are not tested. Defaults to 0. |
| `targetIgnoreFraction` (optional) | `number` | Fraction of the distance at the target end that is not tested. Defaults to 0. |

{/* /parameters */}

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
| `observerHeight` (optional) | `number` | Eye height above the terrain in elevation units. Defaults to 1.7. |
| `targetHeight` (optional) | `number` | Height added to the target elevation. Defaults to 0. |
| `maximumDistance` (optional) | `number` | Ground radius in metres; `<= 0` means unlimited. Defaults to 0. |
| `curvatureCoefficient` (optional) | `number` | Drop coefficient `c` in 1/metres. Defaults to 0. |
| `toleranceMeters` (optional) | `number` | Constant tolerance band half-width in metres. Defaults to 0. |
| `tolerancePerKilometer` (optional) | `number` | Additional band half-width per kilometre of distance, in metres. Defaults to 0. |
| `targetIgnoreDistance` (optional) | `number` | Metres at the target end that are not tested. Defaults to 0. |
| `targetIgnoreFraction` (optional) | `number` | Fraction of the distance at the target end that is not tested. Defaults to 0. |

{/* /parameters */}

{/* parameters: GPUTerrainViewshed generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTerrainViewshed` parameters

- Packer: `getGPUTerrainViewshedParameterValues(settings, grid)`, writes `GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges
- Packer: `getGPUTerrainViewshedToleranceParameterValues(settings)`, writes `GPU_TERRAIN_VIEWSHED_TOLERANCE_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUTerrainViewshedParameterValues`:

| Field | Type | Description |
|---|---|---|
| `observer` | `readonly [number, number]` | Observer `[column, row]` in pixel-center index space; fractional values allowed. |
| `observerHeight` (optional) | `number` | Eye height above the terrain in elevation units. Defaults to 1.7. |
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

### `GPUPointHorizonProfile` and `GPUPointHorizonVisibility`

A 360° horizon profile from any number of observers, ported from mt-image's horizon march.

- **Inputs.** The elevation raster is the `elevation` prop on a `grid`. Observers are
  `[column, row, height, 0]`. `heightReference` is `'ground'` (the default) or `'absolute'`.
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

### `GPUTerrainPeakProminence`

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

### `GPUTerrainCurvature`

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

### `GPUGeomorphons`

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

### `GPUTerrainFlow`

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

### `GPUTerrainFlowField`

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

## Limits and compatibility

- GPU Terrain is experimental and WebGPU-only.
- Inputs are planar `float32` elevation values; projection is done upstream (for example with
  [`GPUProjection`](./gpu-project.md)).
- Each contributor's TSDoc lists its non-goals. The GPU Core maintainer roadmap
  (`dev-docs/roadmaps/gpugraph-roadmap.md`) tracks what is still open.

## Related modules

- [GPU Raster](./gpu-raster/README.md) supplies the raster operators that terrain contributors compose.
- [GPU Core](./gpu-core/concepts.md) defines contributors, composition levels, and graph ownership.
- [GPU Network](./gpu-network.md) shares the conventions above for road and flow analysis.
