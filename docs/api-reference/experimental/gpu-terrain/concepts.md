---
title: "GPU Terrain concepts"
description: "Conventions and execution model of GPU Terrain contributors."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Terrain concepts

<ExperimentalDocsTabs active="gpu-terrain-concepts" />

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
    the cell size from it (see [GPU Terrain conventions](#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](../gpu-network/concepts.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](../gpu-spatial-analysis/concepts.md#shared-ports-status-and-planning).

## Heights and datums

Declare height datums when combining DEM elevations with absolute observer or catalogue heights.

Absolute heights and DEM heights must share one vertical datum, and observer positions must share the DEM grid's horizontal datum. Declare both (`verticalDatum` on the band, `observerVerticalDatum` on the operator, `datum` on contexts). The library checks the declarations and never converts. Heights above the ground (`heightReference: 'ground'`) need no conversion. For a few observers, convert them on the CPU in binary64: h_orthometric = h_ellipsoidal − N(lat, lon), with N taken from the DEM's own geoid model (PROJ, pyproj or GeographicLib). For many observers or targets, convert the DEM on the GPU: resample N onto the DEM grid, add it with `GPURasterBandMath('add')`, and declare `'ellipsoidal'`. Convert horizontal datums (NAD27, GCJ-02) on the CPU before upload.

| Source | Vertical datum to declare |
| --- | --- |
| SRTM/NASADEM/AW3D30 | EGM96 (`'EPSG:5773'`) |
| Copernicus GLO-30 | EGM2008 (`'EPSG:3855'`) |
| USGS 3DEP | NAVD88 (`'EPSG:5703'`) |
| Terrarium / Mapbox Terrain-RGB | "above sea level", unspecified geoid — leave `verticalDatum` undeclared |
| GNSS receivers | `'ellipsoidal'` |
| drone metadata | vendor-dependent, verify before declaring |

- **Absolute inputs.** `GPUPointHorizonProfile`, `GPUPointHorizonVisibility`, `GPUTerrainViewshed`,
  `GPUTerrainLineOfSight`, and `GPUTerrainCumulativeViewshed` accept `heightReference: 'absolute'`
  (default: `'ground'`). Their `observerVerticalDatum` is checked against the elevation band's
  `verticalDatum`; declaring it in ground mode throws. `GPUTerrainPeakSnap` checks
  `candidateVerticalDatum` against the band's `verticalDatum` for absolute `candidateHeights`;
  declaring it without `candidateHeights` throws.
- **Compatibility.** A mismatch between two declared vertical datums throws; either undeclared
  passes. Comparison trims whitespace and is case-insensitive, with no aliases.
- **Absolute observers.** Set the observer height explicitly: the `observerHeight` default of 1.7
  is meaningless in absolute mode. The observer component of LOS `pairHeights` and cumulative
  `observerHeights` also becomes absolute. Observers must be inside the grid but may sit over
  no-data ground. Targets stay ground-relative; an eye below the ground hides most targets.
- **Conversion.** There is no geoid-undulation input: the contract performs no CRS conversion,
  GPU Project rejects geoid grids, and a constant undulation can be silently wrong across a DEM.
  Use `GPURasterBandMath` with `'add'` or `'subtract'` and an undulation raster on the same grid;
  a band `offset` is appropriate only when the correction really is constant.


## GPU execution model

How terrain contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.

{/* lens:execution-model */}

A local operator is one kernel over every cell; a propagating one (fill, flats, accumulation) is an
unrolled loop of gated rounds whose indirect dispatch drops to zero once a round changes nothing.

::lens{scenario=gpu-terrain.terrain-derivatives rung=graph caption="A local operator: two Sobel passes feed one fused shade kernel, each one invocation per cell."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=kernel caption="One shade invocation per cell; the dispatch is the grid, not a loop."}
