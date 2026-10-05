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
filling, D8 flow directions, flow accumulation, streams), horizon angles, solar position and shadow
masks, relief shading, and texture shading. Each contributor is an algorithm that declares its
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

### `GPUTerrainDerivatives`, `GPUTerrainContours`, and `GPUTerrainViewshed`

Elevation-tile analysis composed from `gpu-raster` operators. `GPUTerrainDerivatives` computes
Horn slope, aspect, and hillshade (uniform, Web Mercator, or geographic cell sizes) with an optional
hillshade texture. `GPUTerrainContours` extracts marching-squares segments for several fixed or
per-frame levels with one aggregate overflow flag. `GPUTerrainViewshed` writes line-of-sight
visibility codes from one per-frame observer with optional earth curvature.

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

### `GPUTerrainHorizon`, `GPUSolarShadowMask`, `GPUReliefShading`, `GPUTextureShading`, and `GPUSolarPosition`

Cartographic relief and terrain light in data space, for "sun and shadow" maps on DEM tiles. Every
style and sun value is a per-frame settings write; none of them recompile the graph. Outputs are
float32 rasters, each with an optional `r32float` or `rgba32float` storage texture (channel 0).

- `GPUTerrainHorizon` marches `directionCount` (4 to 64) azimuth sectors per pixel, one node per
  sector, over a baked step schedule (`maximumRadius` pixels, optional geometric `stepGrowth`) with
  bilinear samples, z factor, earth curvature, and an optional ground `maximumDistance`. It writes
  a pixel-major horizon map (`horizon[pixel * directionCount + sector]`, degrees, sector `d` at
  azimuth `d * 360 / directionCount`), the sky-view factor `1 - mean(sin(max(h, 0)))`, and positive
  openness `mean(90 - h)`. Sky-view-only use needs no horizon buffer.
- `GPUSolarShadowMask` reads the horizon map and a per-frame sun: it interpolates the horizon at
  the sun azimuth and writes `sunVisibility`, the solar-disk area fraction above the horizon (soft
  penumbra from the disk's angular radius, hard when 0). With `slope` and `aspect` from
  `GPUTerrainDerivatives` it also writes `illumination = ambient * svf + sun * sunVisibility *
  max(cos(incidence), 0)`. One frame costs two reads per pixel.
- `GPUReliefShading` computes a hillshade from up to 8 lights with fixed weights or the USGS
  multidirectional oblique weighting (`lights: 'mdow'`, per-pixel `sin^2(aspect - azimuth)` as in
  GDAL `-multidirectional`). It also blends a Swiss/Imhof relief (`relief` luminance and packed
  RGBA8 `color`) from the hillshade, an optional sky-view factor, an optional texture shade, an
  elevation tint ramp of up to 8 stops, and a warm-lit/cool-shaded aspect tint.
- `GPUTextureShading` approximates Leland Brown's texture shading (fractional Laplacian
  `|f|^alpha`) with a cascade of separable Gaussian levels (`sigma_k = baseSigma * 2^k`) summed as
  weighted band differences. Nodata uses normalized convolution. `detail` (alpha), the band
  weights, and gain are per frame.
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

### `GPUTerrainFlow`

Hydrology on an elevation tile: optional Planchon–Darboux depression filling with a per-frame
epsilon, D8 flow directions (ESRI codes, cell-size-aware diagonal distances, uniform, Web Mercator,
or geographic cells) with flat, pit, and outlet classes, flow accumulation in cells or ground area
optionally weighted by a per-cell runoff view, and a stream mask with a per-frame threshold.
Filling is a GPU-gated tiled min-relaxation and accumulation a deterministic pull over dependency
order with downstream walking; both stop early on the GPU and report convergence flags, so
nothing is read back and accumulation is bit-reproducible.

```ts
graph.add(new GPUTerrainFlow({
  width, height, elevation, settings: settings.importToGraph(graph), fillDepressions: true,
  flowDirections, accumulation, streams, fillConverged, accumulationConverged
}));
settings.write(getGPUTerrainFlowParameterValues({cellSize: [30, 30], fillEpsilon: 0.01, streamThreshold: 500}));
```

Outputs may share one buffer over disjoint byte ranges. For example, `fillConverged` and
`accumulationConverged` can be two one-row views of an 8-byte summary buffer. Outputs that one
pass writes together (`filledElevation`, `flowDirections` and `cellClasses`; `accumulation` and
`streams`) must start in different 256-byte binding windows. No output may share a buffer with
an input.

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
