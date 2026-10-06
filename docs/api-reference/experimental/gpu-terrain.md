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

const graph = new GPUCommandGraph(device, {id: 'terrain'});
const settings = new GPUParameterBuffer(device, {
  id: 'derivatives-settings',
  format: 'float32',
  length: GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
});

// `elevation`, `slope`, and `hillshade` are graph views created on `graph`.
graph.add(new GPUTerrainDerivatives({
  width, height, elevation, settings: settings.importToGraph(graph), slope, hillshade
}));
const compiled = graph.compile(); // once

// every frame: rewrite parameters, encode the same compiled graph
settings.write(getGPUTerrainDerivativesParameterValues({cellSize: [30, 30], azimuthDegrees: 315}));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

The application compiles, encodes, submits, and decides whether any bounded result is read back.

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids`, `count`, `overflow`, optional `totalCount`). `count` is clamped to `ids.length` and can be
  an indirect draw instance count; `overflow` is rewritten every encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.

## API

### `GPUTerrainDerivatives`, `GPUTerrainContours`, `GPUTerrainViewshed`, `GPUTerrainLineOfSight`, and `GPUTerrainCumulativeViewshed`

Elevation-tile analysis composed from `gpu-raster` operators. `GPUTerrainDerivatives` computes
Horn slope, aspect, and hillshade (uniform, Web Mercator, or geographic cell sizes) with an optional
hillshade texture. `GPUTerrainContours` extracts marching-squares segments for several fixed or
per-frame levels with one aggregate overflow flag. The three visibility contributors share one
sight-line model on planar, projected-metre grids (`cellSize` in metres):

- `GPUTerrainViewshed` writes a `GPU_TERRAIN_VISIBILITY` code per pixel from one per-frame observer.
- `GPUTerrainLineOfSight` tests a batch of `[observerColumn, observerRow, targetColumn, targetRow]`
  pairs. Optional per-pair heights override the defaults. It can also write a clearance: how many
  metres the target could sink and stay visible.
- `GPUTerrainCumulativeViewshed` makes observers the dispatch dimension, like the `gdal_viewshed`
  cumulative mode. It counts, per pixel, the observers that see it (`visibleCount`, optional
  `marginalCount`), with `observersPerDispatch` observers per node.

```ts
graph.add(new GPUTerrainDerivatives({
  width, height, elevation, settings: settings.importToGraph(graph), slope, hillshade
}));
settings.write(getGPUTerrainDerivativesParameterValues({cellSize: [30, 30], azimuthDegrees: 315}));
```

Each contour level's optional `draw` record is rewritten on every encoding from the
capacity-clamped GPU segment count. `drawLayout: 'instanced'` (default) writes
`[verticesPerInstance, segments, 0, 0]` for renderers that read segment endpoints from storage
(`verticesPerInstance` defaults to 2; use 4 or 6 for quad-per-segment lines). `drawLayout:
'line-list'` writes `[2 * segments, 1, 0, 0]` for one non-instanced `line-list` draw over the
`vertices` column used as a vertex buffer.

Each sight line samples the straight segment at one-pixel steps with bilinear elevation and ignores
samples with an invalid corner.

- **Tolerance band.** An optional per-frame `tolerance` (`[toleranceMeters, tolerancePerKilometer,
  targetIgnoreDistance, targetIgnoreFraction]`, all 0 by default) turns the hard cut into a band
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

Without `tolerance` and with `'march'`, `GPUTerrainViewshed` behaves as before.

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
  width, height, elevation, pairs, settings: settings.importToGraph(graph), traversal: 'pyramid',
  visibility, clearance
}));
settings.write(getGPUTerrainSightLineParameterValues({
  cellSize: [30, 30], curvatureCoefficient: getGPUTerrainCurvatureCoefficient(1 / 7),
  toleranceMeters: 2, tolerancePerKilometer: 1, targetIgnoreDistance: 150, targetIgnoreFraction: 0.02
}));
```

### `GPUPointHorizonProfile` and `GPUPointHorizonVisibility`

A 360° horizon profile from any number of observers, ported from mt-image's horizon march.

- **Inputs.** The elevation grid is the `elevation` prop. Observers are `[column, row, height, 0]`. `heightReference` is `'ground'` (the
  default) or `'absolute'`.
- **Outputs.** For each (observer, azimuth bin) the profile writes the maximum apparent-elevation
  tangent `t = (h − h0)/d − c·d`, the `skylineAngle` in degrees (−90 when nothing was sampled), and the
  crest distance. Azimuth bin `i` is at `i·360/azimuthCount` degrees clockwise from north. Sectors
  are set with `firstAzimuth` and `azimuthSpan`.
- **Projections.**
  - `'planar'`: projected metres, a straight ray.
  - `'web-mercator'`: a Web Mercator pixel window, with per-frame `worldPixelSize` and `originY`.
    The ray is a piecewise great circle on a sphere of R = 6371008.8 m. Breakpoints are baked from
    `maximumLatitude` and `segmentTolerance`, and positions are linear in pixels between them.
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
- **Details.** The optional `details` output holds `[targetElevation°, occluderElevation°,
  skylineElevation°, onSkyline]`.

Measured against float64 oracles, the elevation error is at most 3.3e-5° (planar) and 3.0e-5°
(Web Mercator).

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
- the optional input validity marks it invalid;
- its height falls outside `validRange` (default `[-11000, 9000]` m, which removes blank-canvas pixels).

`clampBathymetry` is opt-in. It flattens heights in (−12000, 0) to sea level, which also flattens
the Dead Sea and polders.

Decode before you filter. Never sample the encoded texture bilinearly or through mips, because
filtering byte channels mixes carries. Decode with nearest/`textureLoad`, then resample the float
heights.

Terrarium decoding is bit-exact in f32 for all 2^24 codes: every partial sum is a multiple of 1/256
below 2^16. Mapbox decoding is one correctly rounded multiply, `f32(N − 100000) · f32(0.1)`. Its
error is at most `½ulp + |M|·1.5e-9`, which is below 0.6 mm for |h| ≤ 9000 m.

`GPUTerrainSpikeRepair` is opt-in. It fixes ±256·k m Terrarium red-byte errors, such as canvas
anti-fingerprinting noise:

1. It labels 4-connected components across edges with |Δh| ≤ `jump`, using
   `GPUGraphConnectedComponents` on a fixed-stride grid CSR.
2. A component that is neither the largest nor bigger than 25 % of the valid pixels is shifted by
   −`step`·k when at least 80 % of its border seams agree on a jump of `step`·k ± `tolerance`.

Run it on full-resolution decoded heights, before any filtering. Nodata is never filled or
repaired. It can shift a real enclosed butte whose walls happen to measure 216–296 m, so enable it
only when the source is known to be noisy.

```ts
graph.add(new GPUTerrainRGBDecode({width, height, encoding: 'terrarium', input: {texture}, values, validity}));
graph.add(new GPUTerrainSpikeRepair({
  width, height,
  elevation: {id: 'dem', format: 'float32', storage: {kind: 'buffer', values}, validity},
  values: repaired, validity: repairedValidity, statistics
}));
```

`statistics` holds `[jumpCount, repairedPixelCount, shiftedComponentCount, remainingJumpCount,
converged]`. If the labelling did not converge, the contributor fails closed: the output equals the
input and `converged` is 0.

### `GPUTerrainSummits`, `GPUTerrainPeakSnap`, `GPUTerrainCriticalPoints`, and `GPUProfilePeaks`

These contributors find terrain features. Summit and snap discs are metric: cell sizes come from
per-frame settings in `uniform`, `web-mercator` or `geographic` mode, evaluated at the centre row.
Nodata never contributes a height.

**`GPUTerrainSummits`** marks pixel `p` as a summit when all of these hold:

- `p` is the strict (height, −index) maximum of its radius-`r` disc;
- `drop = h_p − max(ring) ≥ minimumDrop`, where the ring is the disc pixels with an 8-neighbour
  outside the disc. Every path leaving the disc crosses the ring, so `drop` is a radius-limited
  lower bound on prominence.

Under `incompleteNeighborhood: 'reject'` (the default), discs clipped by the grid edge or touching
nodata are rejected. Outputs are an optional mask, an optional per-pixel drop, and an optional
compact list with an optional drop column.

**`GPUTerrainPeakSnap`** moves caller candidate points, given as `[column, row]`, to the highest
valid pixel within a per-candidate or settings radius. It keeps the original point in four cases:

- the maximum lies on the disc ring (a flank, not a summit);
- the move would exceed `maximumMove`;
- the height would change by more than `maximumHeightChange`;
- there is no valid data.

Each candidate gets a status from `GPU_TERRAIN_PEAK_SNAP_STATUS`. mt-image's distance rule
`min(250, 60 + 0.004·d)` m can be supplied through `candidateRadii`.

**`GPUTerrainCriticalPoints`** classifies each pixel as regular, peak, pit, saddle, boundary or
noData by counting sign changes around its ring. Equal heights are ordered by index (simulation of
simplicity). There are two rings:

- 8 is the classic Peucker–Douglas ring.
- 6 is the Freudenthal triangulation, which keeps the Euler relation peaks − saddles + pits
  consistent.

Optional outputs are per-pixel sign changes and per-class counts.

**`GPUProfilePeaks`** finds peaks along 1-D profiles in CSR spans, for example `GPURasterProfile`
`sampleValues` + `pathSampleOffsets`, or a horizon profile with `wrap: true`. For each candidate it:

- tests for a local maximum over ±2 samples;
- computes windowed topographic prominence;
- refines the position with a parabola.

Non-maximum suppression runs in GPU rounds and reaches exactly the greedy (prominence-descending)
result. If it does not converge, `converged` is 0 and undecided peaks are dropped. `minProminence`
is set per frame.

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
uniform, Web Mercator, or geographic per row, as for `GPUTerrainDerivatives`.

```ts
graph.add(new GPUTerrainCurvature({
  width, height, elevation, settings: settings.importToGraph(graph),
  curvatures: {profile, plan: planCurvature, mean}, ringCurvature, ringRadii: [2, 8]
}));
settings.write(getGPUTerrainCurvatureParameterValues({cellSize: [10, 10]}));
```

### `GPUGeomorphons`

Jasiewicz & Stepinski (2013) geomorphons, mirroring GRASS `r.geomorphon`: eight line-of-sight
scans per cell out to `searchRadius` cells (skipping `skipRadius`), a flatness threshold in degrees
with optional `flatDistance`, GRASS's `anglev1` (default), `anglev2`, or `anglev2-distance`
comparison, and GRASS's 9×9 table from (minus, plus) counts to ten forms (`GPU_GEOMORPHON_FORMS`:
flat 1 … pit 10, 0 invalid). `pattern` writes the raw base-3 code (GRASS `ternary_6561`, directions
NE, N, NW, W, SW, S, SE, E) and `ternary` its rotation- and mirror-invariant minimum (GRASS's 498
classes). Comparisons use cross-multiplied heights and step counts instead of `atan`, so integer
DEMs with exact ties classify identically to the float64 oracle. Cells within `skipRadius + 1` of
the border are invalid, as in GRASS.

### `GPUTerrainRuggedness` and `GPUTerrainVectorRuggedness`

`GPUTerrainRuggedness` reproduces gdaldem's TPI, TRI (Riley, the default since GDAL 3.3, or Wilson)
and roughness on 3×3 windows, including both edge modes: `'nodata'` (default) and `'extrapolate'`
(gdaldem `-compute_edges`, with GDAL's linear edge extrapolation and centre substitution for nodata
neighbours). `GPUTerrainVectorRuggedness` is Sappington et al.'s (2007) vector ruggedness measure,
`1 − |Σn| / N` over Horn unit normals in a `(2·radius + 1)²` window (direct sum, intended for
radius ≤ ~8).

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

`GPUTerrainWeissLandforms` classifies Weiss's (2001) ten landforms from a small and a large TPI
scale and Horn slope. `'global'` standardization (default) uses the grid mean and population
standard deviation of each scale (two-pass `GPUReduction`); `'local'` uses DEV. Thresholds
(`standardThreshold`, default 1; `slopeThresholdDegrees`, default 5) are per-frame settings.

```ts
graph.add(new GPUTerrainTopographicPosition({
  width, height, elevation, scales: [{radius: 3}, {radius: 25}, {radius: 100}],
  deviationFromMean, maximumDeviation, maximumDeviationRadius
}));
```

### `GPUTerrainHorizon`, `GPUSolarShadowMask`, `GPUTerrainCastShadow`, `GPUSolarIrradiance`, `GPUReliefShading`, `GPUTextureShading`, and `GPUSolarPosition`

Cartographic relief and terrain light in data space, for "sun and shadow" maps on DEM tiles. Every
style and sun value is a per-frame settings write; none of them recompile the graph. Outputs are
float32 rasters, each with an optional `r32float` or `rgba32float` storage texture (channel 0).

- `GPUTerrainHorizon` marches `directionCount` (4 to 64) azimuth sectors per pixel, one node per
  sector, over a baked step schedule (`maximumRadius` pixels, optional geometric `stepGrowth`) with
  bilinear samples, z factor, earth curvature, and an optional ground `maximumDistance`. It writes
  a pixel-major horizon map (`horizon[pixel * directionCount + sector]`, degrees, sector `d` at
  azimuth `d * 360 / directionCount`), the sky-view factor `1 - mean(sin(max(h, 0)))`, and positive
  openness `mean(90 - h)`. Sky-view-only use needs no horizon buffer. Options:
  - `algorithm: 'sweep'` replaces the per-pixel march with an exact upper-hull sweep (Stewart 1998
    style) over digital lines through pixel centres: amortised O(1) per pixel and sector, exact
    against brute force over the same samples, 4x faster at radius 256 and 36x at full-tile radius on
    1024² (16 sectors). Samples are pixel centres within half a pixel of the ray, so sky-view
    factors differ from the bilinear march by about 0.01 on rough terrain. The sweep has a large
    fixed cost per pixel and sector: it wins at large radii (1024² at radius 1023: 41 ms against
    about 1300 ms; 512² at full radius: 36 against 109 ms) and loses at small ones (512² at radius
    128: 60–125 ms against about 20 ms). Prefer the march for small radii and the sweep near the
    full tile radius.
  - `horizonFormat: 'unorm16'` halves the horizon map (two 16-bit codes per `uint32`, 0.00275 degree
    steps, code 0 = nodata; `unpackGPUTerrainHorizonUnorm16` decodes on the CPU). `GPUSolarShadowMask`
    and `GPUSolarIrradiance` read it.
  - `negativeOpenness` is openness of the inverted DEM (Yokoyama), and `anisotropicSkyViewFactor` is
    RVT's weighted SVF `1 - sum(w sin h+) / sum(w)`, `w = (1 - wMin) |cos((t - tMain)/2)|^level + wMin`.
    `tMain` is a compass azimuth (RVT's counterclockwise `a_main_direction = A` is `360 - A` here),
    set per frame in settings slots 8-10 (settings must then hold 12 floats).
- `GPUSolarShadowMask` reads the horizon map and a per-frame sun: it interpolates the horizon at
  the sun azimuth and writes `sunVisibility`, the solar-disk area fraction above the horizon (soft
  penumbra from the disk's angular radius, hard when 0). With `slope` and `aspect` from
  `GPUTerrainDerivatives` it also writes `illumination = ambient * svf + sun * sunVisibility *
  max(cos(incidence), 0)`. One frame costs two reads per pixel.
- `GPUTerrainCastShadow` casts one sun's shadow with no horizon map: an exact sweep along the
  per-frame sun azimuth gives the horizon in that direction, then the same solar-disk penumbra as
  `GPUSolarShadowMask`. The CPU settings packer computes the line geometry, so moving the sun never
  recompiles. Use it when only one sun matters and a full horizon map is too large.
- `GPUSolarIrradiance` integrates a sun path over a horizon map: `sunHours` (disk-visible hours)
  and `insolation` (Wh/m², direct normal irradiance times visibility times cos incidence, plus an
  optional diffuse term scaled by the sky-view factor). `getGPUSolarIrradianceSunTable` samples
  `getSolarPosition` over a day or a date range (Meinel clear-sky DNI with Kasten–Young air mass by
  default). `samplesPerNode` splits long tables across dispatches.
- `GPUReliefShading` computes a hillshade from up to 8 lights with fixed weights or the USGS
  multidirectional oblique weighting (`lights: 'mdow'`, per-pixel `sin^2(aspect - azimuth)` as in
  GDAL `-multidirectional`). It also blends a Swiss/Imhof relief (`relief` luminance and packed
  RGBA8 `color`) from the hillshade, an optional sky-view factor, an optional texture shade, an
  elevation tint ramp of up to 8 stops, and a warm-lit/cool-shaded aspect tint. Imhof options:
  `lightWeighting: 'imhof-swing'` (with the `imhofSwing: true` prop) swings each light toward the
  slope's side by up to `imhofSwingDegrees` (default 65), an optional `curvature` raster adds
  `curvatureStrength * curvature`, and `contrastStrength` with
  `contrastLowElevation`/`contrastHighElevation` raises contrast with elevation. Zero values keep the
  previous output bit for bit.
- `GPUTextureShading` approximates Leland Brown's texture shading (fractional Laplacian
  `|f|^alpha`) with a cascade of separable Gaussian levels (`sigma_k = baseSigma * 2^k`) summed as
  weighted band differences. Nodata uses normalized convolution. `detail` (alpha), the band
  weights, and gain are per frame. `hasNodata: false` (default `true`) promises an all-valid
  elevation and blurs one channel instead of value plus validity, which is exact and about halves
  the blur cost. `downsampleLevels` (default `true`) computes levels with incremental sigma of at
  least 4 pixels on a 2x decimated grid and upsamples bilinearly; output stays within 1% of the exact
  output range (measured 0.1-0.27%), and `false` restores the exact full-resolution cascade.
- `GPUSolarPosition` evaluates the NOAA solar position for every row of a longitude/latitude
  column at one per-frame instant. It writes azimuth, altitude (with optional refraction), and a
  daylight flag. `getSolarPosition` is the float64 CPU version used to drive the shadow mask.

```ts
const sun = getSolarPosition(Date.now(), longitude, latitude);
graph.add(new GPUTerrainHorizon({
  width, height, elevation, settings: horizonSettings.importToGraph(graph),
  directionCount: 16, maximumRadius: 256, stepGrowth: 1.1, horizon, skyViewFactor
}));
graph.add(new GPUSolarShadowMask({
  width, height, directionCount: 16, horizon, settings: shadowSettings.importToGraph(graph),
  slope, aspect, skyViewFactor, illumination
}));
shadowSettings.write(getGPUSolarShadowMaskParameterValues({
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
  width, height, elevation, settings, directionCount: 16, maximumRadius: 1023,
  algorithm: 'sweep', horizonFormat: 'unorm16', horizon, skyViewFactor
}));
const {values, sampleCount} = getGPUSolarIrradianceSunTable({
  longitude: 8.2, latitude: 46.8, start: Date.UTC(2026, 5, 21), end: Date.UTC(2026, 5, 22)
});
sunTableBuffer.write(values);
graph.add(new GPUSolarIrradiance({
  width, height, directionCount: 16, horizonFormat: 'unorm16', horizon,
  sunTable: sunTableBuffer.importToGraph(graph), sampleCapacity: 288, settings: irradianceSettings,
  slope, aspect, skyViewFactor, sunHours, insolation
}));
```

At 1024² with 16 sectors the sweep horizon takes 41 ms at full radius (98.6 ms at radius 256), and
a 288-sample day of irradiance takes about 29 ms.

### `GPUSimpleLocalRelief`, `GPUMultiScaleRelief`, `GPULocalDominance`, and `GPUReliefBlend`

Archaeological and micro-relief visualizations with Relief Visualization Toolbox (RVT; Kokalj and
Somrak 2019) parity. The float64 test oracles follow RVT-py's formulas, including edge padding, NaN
handling and Python's half-even rounding.

- `GPUSimpleLocalRelief` (SLRM; Hesse 2010): `ve * (z - mean(z))` over a `(2r + 1)²` window with
  edge-clamped coordinates and nodata excluded. The mean is separable with anchored, centre-relative
  sums, so any radius keeps sub-millimetre precision at alpine elevations.
- `GPUMultiScaleRelief` (MSRM; Orengo and Petrie 2018): RVT's radii from `resolution`,
  `featureMinimum`, `featureMaximum` and `scalingFactor` (`getGPUMultiScaleReliefRadii`). RVT's sum of
  consecutive low-pass differences telescopes, so the contributor runs only two mean filters.
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
per-frame settings.

### `GPUTerrainFlow`

Hydrology on an elevation tile: optional Planchon–Darboux depression filling with a per-frame
epsilon, D8 flow directions (ESRI codes, cell-size-aware diagonal distances, uniform, Web Mercator,
or geographic cells) with flat, pit, and outlet classes, flow accumulation in cells or ground area
optionally weighted by a per-cell runoff view, and a stream mask with a per-frame threshold.
Filling is a GPU-gated tiled min-relaxation and accumulation a deterministic pull over dependency
order with downstream walking; both stop early on the GPU and report convergence flags, so
nothing is read back and accumulation is bit-reproducible. `maxFillIterations` and
`maxFlatIterations` default to 512 (up to 1024); after 96 plain tiled iterations each phase adds
directional row and column sweeps so serpentine and spiral basins converge. Check `fillConverged`
and `flatsConverged` for longer ones.

```ts
graph.add(new GPUTerrainFlow({
  width, height, elevation, settings: settings.importToGraph(graph), fillDepressions: true,
  flowDirections, accumulation, streams, fillConverged, accumulationConverged
}));
settings.write(getGPUTerrainFlowParameterValues({cellSize: [30, 30], fillEpsilon: 0.01, streamThreshold: 500}));
```

`resolveFlats: true` routes flow across flats without changing elevations (Barnes, Lehman, and
Mulla 2014). Hop distances toward the flat's draining edge and away from its higher edge are GPU
relaxations over equal-height cells. Each drainable flat cell drains to the equal-height neighbor
whose `2 * towardLower + (flatMaximum - awayFromHigher)` is strictly lower. Pair it with
`fillDepressions` and `fillEpsilon: 0` to drain filled depressions as true flats; `flatsConverged`
reports the three relaxations. `flowRouting` selects how `accumulation` and `streams` split flow:
`'d8'` (default), `'d-infinity'` (Tarboton 1997 facets), `'mfd-freeman'` (Freeman 1991,
`tan(beta)^p`, `p = 1.1`) or `'mfd-quinn'` (Quinn et al. 1991, contour-length weighted, `p = 1`).
Set `p` per frame with `flowExponent`. Non-D8 accumulation uses the same deterministic pull, with
each donor's fraction recomputed from the surface, so it is reproducible bit for bit.
`flowDirections` and `cellClasses` stay D8.

Outputs may share one buffer over disjoint byte ranges. For example, `fillConverged` and
`accumulationConverged` can be two one-row views of an 8-byte summary buffer. Outputs that one
pass writes together (`filledElevation`, `flowDirections` and `cellClasses`; `accumulation` and
`streams`) must start in different 256-byte binding windows. No output may share a buffer with
an input.

### `GPUTerrainHeightAboveDrainage`, `GPUTerrainWatersheds`, `GPUTerrainStreamOrder`, and `GPUTerrainHydrologicIndices`

Products derived from `GPUTerrainFlow` outputs. Each takes ESRI D8 `flowDirections` (any D8 grid,
for example one imported from TauDEM or Whitebox), plus `streams` or `accumulation` where needed.

- `GPUTerrainHeightAboveDrainage`: HAND (Rennó et al. 2008), the height of each cell above the
  first stream cell on its D8 path, with an optional `drainageCells` index output.
- `GPUTerrainWatersheds`: labels every cell with its nearest downstream pour point (nested pour
  points give nested watersheds). Without pour points it labels drainage basins by their outlet or
  pit cell.
- `GPUTerrainStreamOrder`: Strahler order on the stream mask.
- `GPUTerrainHydrologicIndices`: computes, from contributing area in m²
  (`accumulationUnits: 'area'`, any routing):
  - specific catchment area `a = A / b`;
  - topographic wetness `ln(a / tan β)`;
  - stream power `a · tan β`;

  with `tan β` the D8 descent slope clamped to a per-frame `minimumSlope` (default 0.001).

HAND and watersheds resolve D8 paths by GPU pointer jumping, which needs about log₂(path length)
gated rounds. Stream order is a deterministic pull. Every iterative contributor stops early on the
GPU and can publish a `converged` flag.

```ts
graph.add(new GPUTerrainFlow({
  width, height, elevation, settings: flowSettings.importToGraph(graph),
  fillDepressions: true, resolveFlats: true, flowRouting: 'mfd-freeman', accumulationUnits: 'area',
  filledElevation, flowDirections, accumulation, streams
}));
const surface = {id: 'filled', format: 'float32', storage: {kind: 'buffer', values: filledElevation}};
graph.add(new GPUTerrainHeightAboveDrainage({width, height, elevation: surface, flowDirections, streams, heightAboveDrainage}));
graph.add(new GPUTerrainWatersheds({width, height, flowDirections, pourPoints, labels}));
graph.add(new GPUTerrainStreamOrder({width, height, flowDirections, streams, streamOrder}));
graph.add(new GPUTerrainHydrologicIndices({
  width, height, elevation: surface, accumulation, settings: indexSettings.importToGraph(graph), wetnessIndex
}));
```

### `GPUTerrainFlowField`

Terrain-following wind for the flow contributors. A uniform horizontal wind is projected onto the
terrain tangent plane, `v = w − (w·g) g / (1 + |g|²)`, using central-difference gradients in ground
meters (uniform, Web Mercator, or geographic cells, one-sided at the border). Wind across a slope
is unchanged; wind into a slope slows, so flow bends around ridges. The output is the `velocities`
format of `GPUParticleAdvection`, `GPUStreamlines`, and `GPULineIntegralConvolution`, written in the
raster frame (u along columns, v along rows). Cells next to nodata are NaN. Wind and vertical
exaggeration are per-frame settings.

```ts
graph.add(new GPUTerrainFlowField({
  width, height, elevation, settings: windSettings.importToGraph(graph), velocities
}));
windSettings.write(getGPUTerrainFlowFieldParameterValues({cellSize: [30, 30], wind: [4, -2]}));
graph.add(new GPUParticleAdvection({velocities, fieldWidth: width, fieldHeight: height, ...}));
```

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
