---
title: GPURaster analysis contributors
description: Zonal statistics, stretches, isolines, distance and cost surfaces, rasterization, change detection, and flow textures.
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPURaster analysis contributors

<ExperimentalDocsTabs active="gpu-raster-operations" />

## Overview

The `@luma.gl/experimental/gpu-raster` entry point exports analysis contributors that build on the
operations in the other families: zonal statistics, contrast stretch, isolines and isobands,
Euclidean distance fields, cost distance, polygon rasterization, raster joins, reclassification,
sampling and profiles, change detection, particle advection, streamlines, and line integral
convolution, and exact min/max pyramids. Each is an algorithm or workflow that declares resources and command nodes into a
caller-owned `GPUCommandGraph` through `getCommandNodes(graph)` (`GPUCommandNodeProducer`). It never
compiles, submits, encodes, or reads back.

## When to use it

Use these contributors when a map analysis needs a finished, bounded result rather than a
building block, and when its per-frame inputs (levels, seeds, windows, thresholds) change without
recompiling.

- `GPUIsolines` and `GPUIsobands` extract several levels at once, order segments
  deterministically, and can stitch polylines or filled bands. Use `GPURasterContours` for a single
  level of raw marching-squares segments, and `GPUTerrainContours` (see
  [GPU Terrain](/docs/api-reference/experimental/gpu-terrain)) for elevation levels with one overflow flag.
- `GPURasterStretch` computes linear, percentile, and equalization stretches together with a colormap
  lookup and statistics row, using integer atomics only. Use `GPURasterContrast` for a
  calibrated, nodata-aware contrast or gamma that keeps the scalar domain, and `GPURasterHistogram`
  when you only need the histogram.
- `GPURasterZonalStatistics` reduces a raster band over a pre-rasterized grid of zone IDs. Use
  `GPUZonalStatistics` from [GPU Spatial Analysis](/docs/api-reference/experimental/gpu-spatial-analysis) when the input
  is points and polygon features instead of a raster.
- `GPUDistanceField` and `GPUCostDistance` are the Euclidean and friction-weighted surfaces. Use
  [GPU Network](/docs/api-reference/experimental/gpu-network) reachability when travel follows a
  road network rather than a grid.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUDistanceField,
  GPUParameterBuffer,
  getGPUDistanceFieldParameterValues
} from '@luma.gl/experimental/gpu-raster';

const graph = new GPUCommandGraph(device, {id: 'raster-analysis'});
const parameters = new GPUParameterBuffer(device, {id: 'df-parameters', format: 'float32', length: 8});

// `grid` is a `GPURasterMetadata` (see `makeGPURasterMetadata`). `seedPositions`, `seedCount`,
// `seedMask`, and the output views are graph views created on `graph`.
graph.add(new GPUDistanceField({
  grid, parameters: parameters.importToGraph(graph),
  seedPositions, seedCount, seedMask,
  output: {distances, allocation, withinDistance}
}));
const compiled = await graph.compileAsync(); // once

parameters.write(getGPUDistanceFieldParameterValues({maximumDistance: 500}, grid));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

`encode()` only records; submit with `device.submit()`. [Running contributors](../gpu-spatial-analysis/concepts.md#running-contributors)
covers `compileAsync()`, rebuilding without a stall, submission and why `parameters` is `undefined`.

## Conventions

- **One grid.** Grid contributors take `grid: GPURasterMetadata`; `makeGPURasterMetadata({width,
  height, cellSize, origin, rowDirection})` builds an axis-aligned area grid. Packers whose layout
  holds a cell size, extent or latitude band take the frame's grid as their second argument and
  read it from `grid.affine`, so it is never typed twice. `makeGPURasterBand(view, grid)` wraps an
  output view as the input band of the next contributor.
- **`parameters` in, `output` out.** The packed per-frame view is `parameters` (a second view of
  another format is `uint32Parameters`); outputs live in one `output` bag, the validity output is
  `outputValidity`, and one-row status words are in `output.status`.
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
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
  - **Grid.** Raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](../gpu-terrain/concepts.md#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](../gpu-network/concepts.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](../gpu-spatial-analysis/concepts.md#shared-ports-status-and-planning).
- **Extent-local coordinates.** Positions, extents, seeds, sample points, and output vertices are
  float32, whose ulp is 0.5 m at a UTM northing of 4.6e6 m and 2 m at 2e7 m in EPSG:3857. The
  parameter packers of `GPUDistanceField`, `GPUIsolines`, `GPUIsobands` (and `GPUIsobandRings`),
  `GPURasterSampling`, and `GPURasterProfile` take an optional float64 `localOrigin`: they subtract
  it from the extent or grid origin in float64, seeds, points, and line vertices are uploaded
  relative to it, and contour, band, and sample-position outputs come back relative to it. Add it
  back on the CPU or as the layer's coordinate origin. With the raster or tile corner as origin,
  positions keep about `1.2e-7` of their distance from it (a 2.5 m UTM raster places contour
  vertices within 0.02 mm of a float64 reference, against 0.34 m with absolute coordinates). The
  default `[0, 0]` keeps absolute coordinates. `GPUPolygonRasterization` follows the same rule
  with caller-subtracted coordinates.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.

## `GPUCostDistance` and `GPUCostDistancePath`

Accumulated cost surface over a friction raster from per-frame source cells, a source mask, or
both: 8-connected moves cost ground distance times mean friction, invalid or negative friction is
a barrier, and a per-frame cost limit, isoline-ready bands with counts, and D8 back-links are
optional. Relaxation runs in 16x16 workgroup tiles with workgroup-memory repeats and per-tile
activity, gated on the GPU like `GPUNetworkReachability`. `GPUCostDistancePath` walks the
back-links from a per-frame target into a compact cell list. The cost surface feeds
`GPUTerrainContours` directly for isolines. Both take a `grid` and write into an `output` bag
(`costs`, `backLinks`, `bands`, `bandCounts` and `status` for `GPUCostDistance`; `path` and
`status` for `GPUCostDistancePath`). The grid supplies the width, height, cell size and latitude
band, and the packer takes it as its second argument.

When `maximumIterations` stops the relaxation before a fixpoint, passable costs are NaN and bands
and band counts are `GPU_COST_DISTANCE_UNRESOLVED`, so a truncated surface never reads as a
converged one. Every relaxation round shares one pipeline per pass variant: its gate-group position is a GPU
word, not WGSL text.

```ts
const grid = makeGPURasterMetadata({width, height, cellSize: 10});
graph.add(new GPUCostDistance({
  grid, friction, parameters: parameters.importToGraph(graph),
  sources: sources.importToGraph(graph), maximumIterations: 512,
  output: {costs, backLinks, status: {converged}}
}));
graph.add(new GPUCostDistancePath({
  grid, backLinks, target: target.importToGraph(graph), output: {path: output}
}));
parameters.write(getGPUCostDistanceParameterValues({costLimit: 5000}, grid));
```

`frictionParameters` (a `GraphDataView` of `[scale, offset]`, per frame; its presence is compile-time)
applies `friction = value * scale + offset` on top of the band's own calibration, so a friction slider is a
buffer write. Results that are negative or non-finite are impassable. Relaxation uses one gate node per four
rounds, so the unrolled loop has about a third fewer nodes (271 to 189 in the explorer, converging after 43
of 128 iterations).

Back-links across zero-friction plateaus come from a second, tie-level relaxation. It now defaults to
`maximumTieIterations = maximumIterations` (it was 8) and uses the cost phase's schedule, with directional tile
sweeps after `sweepAfterIteration`, so long or winding plateaus resolve. A link that a budget cut short
is `GPU_COST_DISTANCE_UNRESOLVED` (`0xfffffffe`), never 0, because 0 would claim a source. That covers
every passable cell when the cost relaxation stops at `maximumIterations`, and plateau cells whose tie
level did not converge. `GPUCostDistancePath` returns an empty path when the walk meets an unresolved
link, and sets its optional one-row `output.status.iterationLimitReached`. It also returns an empty path for
links that do not form a valid forest (an invalid code, a link off the grid or into an unreached
cell, a cycle), so it only ever reports a complete path from the target to a source.

`maximumIterations` defaults to 512. After `sweepAfterIteration` iterations (default 96) relaxation adds
directional tile sweeps that carry a front along a whole row or column of tiles, so long corridors and
spirals converge in far fewer iterations; open terrain converges before the switch. Pass
`sweepAfterIteration >= maximumIterations` for plain tiled passes only.

`continuation: {state, values}` (opt-in) spreads one relaxation over several submissions instead of
unrolling every round: `maximumIterations` then defaults to 64 and is the budget of one execution.
`state` is a zero-filled imported `uint32` buffer of `getGPURasterRelaxationContinuationStateLength(grid)`
words and `values` an imported `float32` per-cell buffer (neither can be a graph transient, which the
planner may alias). An execution resumes the previous one when that stopped unconverged, skipping
initialization and seeding; after a converged run, or after the caller writes a nonzero value at
element `GPU_RASTER_RELAXATION_RESTART_WORD` of `state` (the GPU clears it), it starts from scratch.
Set the restart word whenever an input changes. `output.status` is cumulative, the sweep schedule
follows the cumulative count, and `output.costs` stays NaN for passable cells until the converging
execution. Poll `converged` and submit again until it is 1. Not supported with `output.backLinks`.

Back-links are cycle-safe across zero-cost plateaus: a cell links to its strictly cheaper tight
neighbor or, when it was reached only across zero-cost moves, to the equal-cost neighbor one tie
level closer to a strict entry or a source. The tie phase runs only when `output.backLinks` is
requested and takes up to `maximumTieIterations` (default `maximumIterations`) extra gated
iterations; `output.status.converged` is 0 if either phase was truncated. The isochrone breaks are
the per-frame `costBreaks` input, required exactly when `output.bands` or `output.bandCounts` is given.

Generated transient IDs are prefixed with the contributor `id`, for example `<id>-total`. If one
collides with a caller resource, the error names the contributor and the generated ID.

| Output | Type | Description |
|---|---|---|
| `output.settledAt` (optional) | `GraphDataView<'uint32'>` | Packed, one value per cell: the 1-based gated relaxation round in which its cost last decreased; 0 for sources and cells whose cost never decreased. For a converged run, visualizes the propagation wavefront. Adds one store per improved cell; omit in production. Must be distinct from inputs and other outputs. Not supported with `continuation`. |

{/* parameters: GPUCostDistance generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCostDistance` parameters

- Packer: `getGPUCostDistanceParameterValues(settings, grid)`, writes `GPU_COST_DISTANCE_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): the frame grid, which supplies the cell size and row edges

Settings of `getGPUCostDistanceParameterValues`:

| Field | Type | Description |
|---|---|---|
| `costLimit` (optional) | `number` | Candidate costs above this value stay unreached. Defaults to `Infinity`. |

{/* /parameters */}

#### `GPUCostDistance`

Relaxes accumulated travel cost outward from a source cell over a friction raster until no cell improves.

::lens{scenario=gpu-raster.cost-distance rung=result caption="Accumulated cost from Greenville; water is impassable and stays unreached."}

#### How it computes

{/* lens:GPUCostDistance */}

::lens{scenario=gpu-raster.cost-distance rung=datum datum=2157 caption="One cell and its eight neighbours: ground length times mean friction, then the minimum."}
::lens{scenario=gpu-raster.cost-distance rung=formula caption="Least accumulated cost over eight-connected moves, written as a fixpoint of the relaxation."}
::lens{scenario=gpu-raster.cost-distance rung=iterations caption="The cost surface at key rounds; later gates skip work once nothing improves."}
::lens{scenario=gpu-raster.cost-distance rung=kernel caption="Tiled relaxation: each 16×16 tile repeats the update in workgroup memory."}
::lens{scenario=gpu-raster.cost-distance rung=graph caption="Friction preparation, seeding, gated relaxation rounds and the finalize node."}

#### `GPUCostDistancePath`

Walks D8 back-links from each target cell to the source and lists the least-cost route cell by cell.

::lens{scenario=gpu-raster.cost-distance-path rung=result caption="Three least-cost routes from the Greenville source to tree-cover targets."}

#### How it computes

{/* lens:GPUCostDistancePath */}

::lens{scenario=gpu-raster.cost-distance-path rung=datum datum=2930 caption="Follow one target cell by cell through its back-links until the walk reaches the source."}
::lens{scenario=gpu-raster.cost-distance-path rung=iterations caption="Cost relaxation rounds that produce the back-links the walk reads."}
::lens{scenario=gpu-raster.cost-distance-path rung=kernel caption="One thread walks the chain of back-links, bounded by the cell count."}
::lens{scenario=gpu-raster.cost-distance-path rung=graph caption="Cost distance with back-links feeds three walks; a histogram turns each path into a mask."}

## `GPURasterZonalStatistics`

Counts cells and computes the mean, lowest and highest elevation of each land-cover zone from a zone raster.

::lens{scenario=gpu-raster.raster-zonal-statistics rung=result caption="Elevation statistics per land-cover zone; tree cover spans the widest range."}

Per-zone cell count, valid-value count, sum, mean, minimum, and maximum of a value band over a
dense `uint32` zone raster (for example rasterized administrative areas). Calibration, nodata,
validity, and non-finite samples are honored; one ignored zone ID is skipped silently, and zone IDs
at or above the compile-time capacity raise a GPU overflow flag. Composes `GPUGroupAggregation`.

```ts
graph.add(new GPURasterZonalStatistics({
  grid, zones, values: {id: 'ndvi', format: 'float32', storage: {kind: 'buffer', values}},
  zoneCapacity: 256, ignoredZone: 0,
  output: {valueCounts, means, minimums, maximums, status: {overflow}}
}));
```

`sumOrder` defaults to `'sorted'`: cells are stably sorted by zone ID and each zone is reduced in a
fixed tree (the same sorted segmented sum as `GPUZonalStatistics`), so `sums` and `means` are bitwise
reproducible on one device, at about 6-25 ms at 2048x2048 for any zone layout. `'atomic'` skips the
sort and is at most about 1.5x faster, only with 1000 or more spatially scattered zones; it can be
10-60x slower when neighboring cells share a zone, because float compare-exchange atomics contend on
one address, and its last bits can vary between runs. Counts, minimums, and maximums are exact either way.

Longitude/latitude rasters set `cellSizeMode: 'geographic'` and a per-frame `parameters` view
`[cellSizeX, cellSizeY, northEdge, southEdge]` in degrees (the grid prefix of `GPUCostDistance`),
packed from the grid with `getGPURasterZonalStatisticsParameterValues(grid)`.
Each cell is then weighted by its area on the WGS84 authalic sphere: `sums` are `sum(a * v)` in
value times square meters, `means` are `sum(a * v) / sum(a)`, and the optional `weightSums` column is
the valid area of each zone. Counts, minimums, and maximums are never weighted, and the default
`'uniform'` mode is unchanged. Optional `variances` and `standardDeviations` are unbiased two-pass
estimates with the `GPUZonalStatistics` convention: `sum (v - mean)^2 / (n - 1)`, or with cell areas
as reliability weights `sum a (v - m)^2 / (V1 - V2 / V1)` (`numpy.cov(v, aweights=a, ddof=1)`). They
add a second pass over the cells, so request them only when needed.

{/* parameters: GPURasterZonalStatistics generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterZonalStatistics` parameters

- Packer: `getGPURasterZonalStatisticsParameterValues(grid)`, writes `GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH` (4) `float32` values.
  - `grid` (`GPURasterMetadata`): The frame's grid; its width and height must match the contributor's.

{/* /parameters */}

#### How it computes

{/* lens:GPURasterZonalStatistics */}

::lens{scenario=gpu-raster.raster-zonal-statistics rung=datum datum=0 caption="Zone 0, tree cover: valid cells are summed and divided by their count, with the lowest and highest kept."}
::lens{scenario=gpu-raster.raster-zonal-statistics rung=formula caption="The zone mean is the sum of valid values over their count; extremes are taken over the same cells."}
::lens{scenario=gpu-raster.raster-zonal-statistics rung=kernel caption="A mask pass flags in-zone valid cells, then sorted segment sums and a fused pass reduce each zone."}
::lens{scenario=gpu-raster.raster-zonal-statistics rung=graph caption="Canonicalize the band, mask, sort and sum by zone, then reduce counts and extremes."}

## `GPUDistanceField`

Turns a seed mask into the exact Euclidean distance in metres from every cell to its nearest seed.

::lens{scenario=gpu-raster.distance-field rung=result caption="Distance to the nearest water body, from the WorldCover water cells of the Dixie window."}

Euclidean distance transform and nearest-seed allocation (Voronoi zones) on a raster grid, from
seed points snapped to cells, a labeled seed raster, or both. `distances` holds the grid distance
from each cell center to the nearest seed cell center (`+Infinity` when unreached), `allocation`
the nearest seed's ID, and `nearestCells` its cell; ties break on the smallest seed ID. A
per-frame `maximumDistance` clears cells beyond it and fills an optional `withinDistance` mask, and an
optional `r32float` texture receives a copy of `distances`. Cell sizes may differ in x and y.

Distances use grid units (world units of the affine), so degree grids yield degree distances, not
meters; ground-meter conversion is a non-goal of `GPUDistanceField`—use `GPUCostDistance` with
`cellSizeMode: 'geographic' | 'web-mercator'` and unit friction for ground-meter distances.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'df-parameters', format: 'float32', length: 8});
graph.add(new GPUDistanceField({
  grid, parameters: parameters.importToGraph(graph),
  seedPositions, seedCount, seedMask,  // capacity is seedPositions.length
  output: {distances, allocation, withinDistance}
}));
parameters.write(getGPUDistanceFieldParameterValues({maximumDistance: 500}, grid));
```

`mode: 'exact'` (default) is the separable Felzenszwalb-Huttenlocher transform, banded as in the
Parallel Banding Algorithm (Cao et al. 2010): banded column scans find the nearest seed row, then
32-column band envelopes of those candidates merge pairwise in `ceil(log2(width / 32))` rounds and
a banded running maximum colors each row. Every pass runs about `width * height / 32` invocations
instead of one per row or column, and the result is bit-identical to the sequential envelope.
Candidates compare by exact integer squared offsets when the cell is square
and by f32 squared ground distance otherwise, so allocation matches a brute-force oracle exactly
on square cells. `'exact'` is exact on isotropic grids only: with unequal cell width and height the
f32 keys can misorder near-ties, and envelope crossings can be off by a cell. A correctly rounded square root keeps distances within 1 ULP of an f64 reference
(0 ULP when the cell size is a power of two). `mode: 'jump-flood'` is a cheaper preview for
dragging seeds: `ceil(log2(max(width, height)))` full-grid passes plus
`jumpFloodRefinementIterationCount` extra passes (exactly 0, 1, or 2; default 1). It always reports the distance to a real seed, so it can only overestimate; on random
256x256 scenes JFA+1 misallocated at most 0.009% of cells with errors up to 1.24 cells, and JFA+2
was exact in the measured scenes (not guaranteed).

A seed mask value `v` is the seed ID `v - 1`: write `1` for a plain mask or `label + 1` for
per-zone allocation. Seed point IDs default to the row index or come from `seedIds`. Rewriting
seeds, the count, the mask, or parameters never recompiles the graph. The grid origin and cell size
come from `grid` (axis-aligned, north-up or north-down), not from the settings.

For projected grids pass `localOrigin` (for example the raster corner) to
`getGPUDistanceFieldParameterValues` and upload `seedPositions` relative to it, so seeds near a
cell edge snap to the same cell as in float64. The grid origin stays absolute.

{/* parameters: GPUDistanceField generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUDistanceField` parameters

- Packer: `getGPUDistanceFieldParameterValues(settings, grid)`, writes `GPU_DISTANCE_FIELD_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): The frame's grid; its width and height must match the contributor's.

Settings of `getGPUDistanceFieldParameterValues`:

| Field | Type | Description |
|---|---|---|
| `maximumDistance` (optional) | `number` | Cells farther than this ground distance from every seed get distance `+Infinity`, allocation `GPU_DISTANCE_FIELD_NONE`, and a `0` in the within-distance mask. Defaults to `Infinity`. |
| `localOrigin` (optional) | `readonly [number, number]` | Float64 point that seed positions are uploaded relative to. The packer subtracts it from the grid origin in float64, so seeds snap to cells from origin-relative float32 values. Defaults to `[0, 0]` (absolute seeds). Pass the raster or tile corner for projected grids: absolute float32 seeds and origins are quantized to 0.5 m at a UTM northing of 4.6e6 m, so seeds within that of a cell edge snap to the wrong cell; origin-relative ones keep about `1.2e-7` of their distance from the origin (1.2 cm at 100 km). |

{/* /parameters */}

#### How it computes

{/* lens:GPUDistanceField */}

::lens{scenario=gpu-raster.distance-field rung=datum datum=3228 caption="One cell: the squared offsets to the nearest seed, then the correctly rounded root."}
::lens{scenario=gpu-raster.distance-field rung=formula caption="Distance is the cell size times the root of the smallest squared cell offset to any seed."}
::lens{scenario=gpu-raster.distance-field rung=kernel caption="Banded column scans and row envelopes find the nearest seed, then one finalize pass scales it."}
::lens{scenario=gpu-raster.distance-field rung=graph caption="Seed rasterization, banded exact passes and the finalize node."}

## `GPUPolygonRasterization` and `GPURasterJoin`

`GPUPolygonRasterization` scan-converts polygon features (the GeoArrow layout of
`GPUPointInPolygonJoin`: `positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`) into
a dense `uint32` zone raster on the GPU, plus optional boundary-cell flags. `GPURasterJoin` then
aggregates points by looking up their cell: O(1) per point, independent of polygon complexity.
The raster placement (origin, cell size, and row direction, north-up or north-down) comes from the
`grid`, packed by `getGPUPolygonRasterizationParameterValues(grid)`.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'raster-parameters', format: 'float32', length: 4});
parameters.write(getGPUPolygonRasterizationParameterValues(grid));
const parametersView = parameters.importToGraph(graph);
graph.add(new GPUPolygonRasterization({
  grid, parameters: parametersView,
  positions, featureOffsets, polygonOffsets, ringOffsets,
  crossingCapacity: 1 << 20,
  output: {zones, boundary, status: {overflow, crossingCount}}
}));
graph.add(new GPURasterJoin({
  grid, parameters: parametersView, points, values, zones, boundary, zoneCount: featureCount,
  output: {counts, sums, boundaryCounts, unassignedBoundaryCount, outsideCount, pointBoundaryMask}
}));
parameters.write(getGPUPolygonRasterizationParameterValues(nextGrid)); // no recompile
```

**Longitude/latitude polygons — rasterization.** Pass `spatialContext: {coordinateSpace:
'longitude-latitude', metric: 'none'}` for linear edges in degrees; omitted contexts remain planar.
The resolved context is exposed as `spatialContext`. Great-circle metrics and great-circle edges
are rejected. Prepare polygons crossing ±180 or enclosing a pole with
`GPULonLatPolygonPreparation`. Rasterization uses longitude copies shifted by -360, 0, and +360
degrees, selected on the GPU from the per-frame grid extent, so a split Fiji polygon covers both
sides of a 170..190-degree grid. Crossings outside the grid remain clamped to its border to preserve
fill parity. Grids inside [-180, 180] retain their existing output. A grid straddling ±180 can count
each edge twice; size `crossingCapacity` accordingly and inspect `status.crossingCount`.
Empty rings contribute no crossings.

**Longitude/latitude polygons — raster join.** `GPURasterJoin` accepts and exposes the same optional
`spatialContext`, with linear edges and metric `'none'`. Give the rasterization and join the same
context. The join consumes caller-provided `zones` and `boundary`; it does not rasterize polygons
itself or shift point coordinates. Points must use the grid's longitude branch. Prepare crossing
or pole rings with `GPULonLatPolygonPreparation` before producing the zone raster; rasterization's
±180 seam handling supplies the shifted polygon copies.

{/* parameters: GPUPolygonRasterization generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUPolygonRasterization` parameters

- Packer: `getGPUPolygonRasterizationParameterValues(grid)`, writes `GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH` (4) `float32` values.
  - `grid` (`GPURasterMetadata`): The frame's grid: axis-aligned; its width and height must match the contributor's.

{/* /parameters */}

{/* parameters: GPURasterJoin generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterJoin` parameters

- Packer: `getGPUPolygonRasterizationParameterValues(grid)`, writes `GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH` (4) `float32` values.
  - `grid` (`GPURasterMetadata`): The frame's grid: axis-aligned; its width and height must match the contributor's.

{/* /parameters */}

#### `GPUPolygonRasterization`

Scan-converts polygon rings into a zone raster whose cells hold the feature covering their centre.

::lens{scenario=gpu-raster.polygon-rasterization rung=result caption="The Dixie perimeter on the 120 m grid; about 89 percent of cell centres fall inside."}

#### How it computes

{/* lens:GPUPolygonRasterization */}

::lens{scenario=gpu-raster.polygon-rasterization rung=datum datum=3360 caption="A cell on the perimeter edge: its row centre line meets the edge crossings that pair into a span."}
::lens{scenario=gpu-raster.polygon-rasterization rung=kernel caption="One thread per vertex counts the rows its edge crosses, then spans are filled with an atomic minimum."}
::lens{scenario=gpu-raster.polygon-rasterization rung=graph caption="Edge counts, scan, crossing records, radix sort, ranking and span fill, plus boundary flags."}

#### `GPURasterJoin`

Assigns each point the zone stored in its raster cell, counting points per zone in O(1) per point.

::lens{scenario=gpu-raster.raster-join rung=result caption="Burned cell centres coloured by zone; the few outside the perimeter take no zone."}

#### How it computes

{/* lens:GPURasterJoin */}

::lens{scenario=gpu-raster.raster-join rung=datum datum=112 caption="One burned point near the perimeter: its cell index, the zone stored there and the boundary status."}
::lens{scenario=gpu-raster.raster-join rung=formula caption="The point takes the zone of the cell that contains it, found by flooring the offset from the origin."}
::lens{scenario=gpu-raster.raster-join rung=kernel caption="One thread per point floors its position into a cell and reads the zone and boundary words."}
::lens{scenario=gpu-raster.raster-join rung=graph caption="Rasterization fills the zone raster, then the join looks up points and tallies counts."}

## `GPURasterReclassify`, `GPUWeightedOverlay`, and local raster operations

Per-cell map algebra over float32 rasters or plain columns, with every table, weight, threshold,
and operation code in a per-frame parameter view, so sliders never recompile. Rasters are packed
row-major float32; NaN is always nodata, and an optional finite `noDataValue` sentinel is
compared exactly. Stacks are band-sequential: layer `i` occupies rows `[i * cellCount, (i + 1) * cellCount)`.

- `GPURasterReclassify` maps values through `n` ascending per-frame breaks: the class is the number
  of breaks `<= value` (left-closed `[b[k - 1], b[k])`, default) or `< value` (right-closed), found
  by binary search. Outputs: `classes` (`0xffffffff` for nodata), `reclassified`
  (`classValues[class]`), and deterministic `classCounts` (integer atomics).
- `GPUWeightedOverlay` (suitability) scores up to 16 layers: each value is remapped linearly from
  `[inputMinimum, inputMaximum]` to `[0, 1]` (clamped, optionally inverted) or through a per-layer break
  table, then `score = sum(weight * remapped)` in fixed layer order. A NaN table value marks a
  restricted class (ArcGIS "Restricted"). Options: normalize by the sum of absolute weights,
  nodata `'propagate'` or `'ignore'`. `scoreRange` is the exact min/max of the defined scores
  (order-preserving integer atomics) and `outputValidity` flags the defined scores.
- `GPURasterCellStatistics` (ArcGIS "Cell Statistics") reduces up to 64 layers per cell:
  minimum, maximum, range, sum, mean, population standard deviation (centred second pass),
  majority, minority (ties pick the smallest value), variety, and valid `counts`. Frequencies are an
  `O(layerCount^2)` exact-equality scan per cell.
- `GPURasterConditional` is `where(condition, a, b)`: a `uint32` mask, or a float raster compared
  per frame (`<`, `<=`, `>`, `>=`, `==`, `!=`, inclusive `between`); `a` and `b` are rasters or
  per-frame constants. The optional condition result is `output.outputMask`.
- `GPURasterArithmetic` evaluates `op(a * scaleA + offsetA, b * scaleB + offsetB)` with a per-frame
  operation: add, subtract, multiply, divide, minimum, maximum, power, absolute difference,
  normalized difference (NDVI), or a unary absolute/square root/log/exp/floor/ceil/round
  (half to even) on `a`, then an optional clamp (`clampMinimum`, `clampMaximum`). Domain errors give NaN.

```ts
const overlayParameters = new GPUParameterBuffer(device, {
  id: 'overlay', format: 'float32', length: getGPUWeightedOverlayParameterLength(3)
});
graph.add(new GPUWeightedOverlay({
  stack, layerCount: 3, cellCount: width * height,
  parameters: overlayParameters.importToGraph(graph),
  remapBreaks, remapValues, maximumBreakCount: 4, // optional per-layer tables
  output: {score, scoreRange}
}));
overlayParameters.write(getGPUWeightedOverlayParameterValues({
  layers: [
    {weight: 0.5, inputMinimum: 0, inputMaximum: 30, invert: true}, // slope: flatter is better
    {weight: 0.3, mode: 'table', breakCount: 4},            // land cover classes
    {weight: 0.2, inputMinimum: 0, inputMaximum: 5000}            // distance to roads
  ],
  normalizeWeights: true
})); // no recompile
```

Every contributor is one invocation per cell reading inputs in fixed order, so results are
deterministic per device. Backends may contract `a * b + c` into an FMA (Metal does), so scores
and scaled operands can differ from an unfused CPU evaluation by a few roundings; reclassify,
conditional, extremes, sums, frequencies, and counts match a CPU oracle bit for bit.

{/* parameters: GPURasterArithmetic generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterArithmetic` parameters

- Packer: `getGPURasterArithmeticParameterValues(settings)`, writes `GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURasterArithmeticParameterValues`:

| Field | Type | Description |
|---|---|---|
| `operation` | `GPURasterBinaryOperation \| GPURasterUnaryOperation` | Operation applied to the scaled operands. |
| `scaleA` (optional) | `number` | `a' = a * scaleA + offsetA`. Defaults to 1. |
| `offsetA` (optional) | `number` | Defaults to 0. |
| `scaleB` (optional) | `number` | `b' = b * scaleB + offsetB`. Defaults to 1. |
| `offsetB` (optional) | `number` | Defaults to 0. |
| `constantB` (optional) | `number` | `b` when no `b` view is bound. Defaults to 0. |
| `clampMinimum` (optional) | `number` | Optional output clamp lower bound. Defaults to `-Infinity`. |
| `clampMaximum` (optional) | `number` | Optional output clamp upper bound. Defaults to `+Infinity`. |

{/* /parameters */}

{/* parameters: GPURasterCellStatistics generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterCellStatistics` parameters

- Packer: `getGPURasterCellStatisticsParameterValues(settings?)`, writes `GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPURasterCellStatisticsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `noDataPolicy` (optional) | `'ignore' \| 'propagate'` | `'ignore'` (default, ArcGIS "DATA"): statistics use the valid layers of each cell. `'propagate'`: any nodata layer makes the cell nodata. |
| `minimumValidCount` (optional) | `number` | Cells with fewer valid layers are nodata. Defaults to 1. |

{/* /parameters */}

{/* parameters: GPURasterConditional generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterConditional` parameters

- Packer: `getGPURasterConditionalParameterValues(settings?)`, writes `GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURasterConditionalParameterValues`:

| Field | Type | Description |
|---|---|---|
| `comparison` (optional) | `GPURasterComparison` | Comparison of `conditionValues` against `threshold`. Ignored with a `mask`. Defaults to `'>'`. |
| `threshold` (optional) | `number` | Threshold, or the inclusive lower bound for `'between'`. Defaults to 0. |
| `upperThreshold` (optional) | `number` | Inclusive upper bound for `'between'`. Defaults to `threshold`. |
| `constantA` (optional) | `number` | Value used where the condition holds when no `a` view is bound. Defaults to 1. |
| `constantB` (optional) | `number` | Value used where the condition fails when no `b` view is bound. Defaults to 0. |

{/* /parameters */}

{/* parameters: GPURasterReclassify generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterReclassify` parameters

- Packer: `getGPURasterReclassifyParameterValues(settings)`, writes `GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPURasterReclassifyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `breakCount` | `number` | Number of active ascending breaks, at most the contributor's `maximumBreakCount`. |
| `closed` (optional) | `'left' \| 'right'` | Which side of each interval is closed. `'left'` (default) gives intervals `[b[k - 1], b[k])`, so the class is the number of breaks `<= value`; `'right'` gives `(b[k - 1], b[k]]`, so the class is the number of breaks `< value`. |

{/* /parameters */}

{/* parameters: GPUWeightedOverlay generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUWeightedOverlay` parameters

- Packer: `getGPUWeightedOverlayParameterValues(settings, options?)`, writes `getGPUWeightedOverlayParameterLength(...)` `float32` values.
  - `options` (`{hasTables?: boolean}`): `hasTables: false` declares that the overlay was built without remap tables, so a `'table'` layer throws here instead of scoring NaN (nodata) on every cell. Defaults to true.

Settings of `getGPUWeightedOverlayParameterValues`:

| Field | Type | Description |
|---|---|---|
| `layers` | `readonly GPUWeightedOverlayLayerSettings[]` | One record per input layer, in stack order. |
| `normalizeWeights` (optional) | `boolean` | Divide the weighted sum by the sum of absolute weights of the contributing layers. |
| `noDataPolicy` (optional) | `'propagate' \| 'ignore'` | `'propagate'` (default): a nodata input makes the cell nodata. `'ignore'`: nodata layers are skipped (and excluded from weight normalization); a cell with no valid layer is nodata. |

{/* /parameters */}

#### `GPURasterArithmetic`

Scales two rasters and combines them per cell; here near-infrared reflectance after the fire minus before it.

::lens{scenario=gpu-raster.raster-arithmetic rung=result caption="Change in near-infrared reflectance; the burn scar is the strongest decrease."}

#### How it computes

{/* lens:GPURasterArithmetic */}

::lens{scenario=gpu-raster.raster-arithmetic rung=datum datum=2400 caption="One burn-scar cell: raw digital numbers are checked for no data, scaled, then subtracted."}
::lens{scenario=gpu-raster.raster-arithmetic rung=formula caption="Check the raw values for no data, scale each, subtract, and keep NaN for missing input."}
::lens{scenario=gpu-raster.raster-arithmetic rung=kernel caption="One invocation per cell evaluates the operation chosen in the parameters."}
::lens{scenario=gpu-raster.raster-arithmetic rung=graph caption="A single per-cell node reads both dates and writes the difference."}

#### `GPURasterCellStatistics`

Reduces a stack of aligned rasters to one statistic per cell; here the mean of near-infrared before and after.

::lens{scenario=gpu-raster.raster-cell-statistics rung=result caption="Mean near-infrared reflectance over both dates; cells missing either date are NaN."}

#### How it computes

{/* lens:GPURasterCellStatistics */}

::lens{scenario=gpu-raster.raster-cell-statistics rung=datum datum=2400 caption="One cell: both layers are checked for validity, summed in order and divided by the count."}
::lens{scenario=gpu-raster.raster-cell-statistics rung=formula caption="Mean of the valid layers, written only when enough layers are valid."}
::lens{scenario=gpu-raster.raster-cell-statistics rung=kernel caption="Each invocation sweeps its layers in fixed order and counts the valid ones."}
::lens{scenario=gpu-raster.raster-cell-statistics rung=graph caption="One moments node reads the stack and writes the mean."}

#### `GPURasterConditional`

Chooses one of two values per cell by comparing a raster with a threshold; here friction from built-up land.

::lens{scenario=gpu-raster.raster-conditional rung=result caption="Friction by land cover: 0.4 where WorldCover says built-up, 1 elsewhere."}

#### How it computes

{/* lens:GPURasterConditional */}

::lens{scenario=gpu-raster.raster-conditional rung=datum datum=1893 caption="Greenville cell: the comparison is true, so the first value is chosen."}
::lens{scenario=gpu-raster.raster-conditional rung=formula caption="Compare the land cover with the threshold and take the branch it selects."}
::lens{scenario=gpu-raster.raster-conditional rung=kernel caption="One invocation compares and selects; the unchosen branch is never read for no data."}
::lens{scenario=gpu-raster.raster-conditional rung=graph caption="A single select node reads the land cover and writes the friction."}

#### `GPURasterReclassify`

Maps raster values through ascending breaks to class values; here land cover codes to suitability ranks.

::lens{scenario=gpu-raster.raster-reclassify rung=result caption="Suitability rank per land cover class, from 1 for water to 5 for cropland."}

#### How it computes

{/* lens:GPURasterReclassify */}

::lens{scenario=gpu-raster.raster-reclassify rung=datum datum=1526 caption="One cropland cell: count the breaks at or below its code, then look up the rank."}
::lens{scenario=gpu-raster.raster-reclassify rung=formula caption="The interval index is the number of breaks at or below the value; the rank is its table entry."}
::lens{scenario=gpu-raster.raster-reclassify rung=kernel caption="A binary search over the breaks classifies each cell in one invocation."}
::lens{scenario=gpu-raster.raster-reclassify rung=graph caption="A single classify node reads the codes, breaks and class values."}

#### `GPUWeightedOverlay`

Scores each cell as a weighted sum of remapped layers; here land cover and elevation suitability.

::lens{scenario=gpu-raster.weighted-overlay rung=result caption="Suitability from land cover (weight 0.7) and elevation (weight 0.3), on a 0–1 scale."}

#### How it computes

{/* lens:GPUWeightedOverlay */}

::lens{scenario=gpu-raster.weighted-overlay rung=datum datum=1893 caption="Greenville cell: each layer is remapped through its table, then weighted and summed."}
::lens{scenario=gpu-raster.weighted-overlay rung=formula caption="Each layer is remapped by its break table; the score is the weighted sum."}
::lens{scenario=gpu-raster.weighted-overlay rung=kernel caption="One invocation remaps every layer and accumulates the weighted sum in layer order."}
::lens{scenario=gpu-raster.weighted-overlay rung=graph caption="A single score node reads the stack, break tables and weights."}

## `GPURasterStretch`

Rescales a raster to 0–1 between its smallest and largest valid values, with optional percentile, equalize, gamma and sigmoid.

::lens{scenario=gpu-raster.raster-stretch rung=result caption="Near-infrared reflectance after the fire, stretched between its smallest and largest valid values."}

Contrast stretches and colormap lookups for a float32 raster, computed on the GPU every frame with
no readback: a linear min/max stretch, a percentile stretch (for example 2%/98%), and histogram
equalization, with optional display gamma (`t^(1/gamma)`, so `gamma > 1` brightens, as in rio-color
and GIMP levels) and sigmoidal contrast (normalized so 0 and 1 are fixed). GDAL has no gamma; its
`-exponent e` (`gdal_translate`, VRT) is `t^e`, so it equals `gamma: 1 / e`. Outputs (any subset): per-cell `stretched` values in `[0, 1]`, packed rgba8 `colors`
(`r | g << 8 | b << 16 | a << 24`) through a per-frame `palette` (nearest or linear), a `lut` of
`lutSize` normalized values with optional `lutColors`, the `histogram`, and an 8-row `statistics`
row `[domainMinimum, domainMaximum, lo, hi, validCount, binWidth, 0, 0]` (`GPU_RASTER_STRETCH_STATISTICS_INDEX`).

Statistics (exact min/max, a `binCount` histogram, its CDF) come from the valid, finite cells inside a
per-frame window `[column0, row0, column1, row1)` and an optional per-frame `regionMask`; the apply
step always covers every cell. A viewport mask (for example from `GPURegionMask`) gives QGIS-style
"stretch to visible extent" per frame. The domain is `'auto'` (exact min/max) or an explicit
`[min, max]`. Percentile bounds find the bin where the CDF crosses `p * count` and interpolate inside
it, so they are within one `binWidth` of the exact percentile. NaN, `noDataValue`, and `validity`
cells are nodata (`stretched` NaN, `colors` 0). A constant raster (`lo == hi`) maps values below to 0,
equal to 0.5, above to 1; with an automatic domain and no included cell, `statistics` is NaN and
`validCount` 0.

```ts
const stretch = new GPURasterStretch({
  values, grid, noDataValue: -9999,
  binCount: 1024, lutSize: 256, // topology
  parameters: stretchParameters.importToGraph(graph),
  regionMask: viewportMask, // optional, contents per frame
  palette, // uint32 rgba8, contents per frame
  output: {colors, lut, histogram, statistics}
});
graph.add(stretch);
stretchParameters.write(getGPURasterStretchParameterValues({
  mode: 'percentile', percentiles: [2, 98], gamma: 0.8, sigmoidContrast: 4, paletteInterpolation: 'linear'
})); // no recompile
```

Pipeline: order-preserving integer min/max keys, integer-atomic histogram, `GPUScan` CDF, finalize,
lookup table, apply; every step is deterministic. Mode, domain, percentiles, window, gamma,
sigmoid, palette interpolation, mask, and palette contents never recompile.

{/* parameters: GPURasterStretch generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterStretch` parameters

- Packer: `getGPURasterStretchParameterValues(settings?)`, writes `GPU_RASTER_STRETCH_PARAMETER_LENGTH` (16) `float32` values.

Settings of `getGPURasterStretchParameterValues`:

| Field | Type | Description |
|---|---|---|
| `window` (optional) | `readonly [number, number, number, number]` | Statistics window `[column0, row0, column1, row1)` in cells; clamped to the raster. Defaults to the whole raster. Only statistics (min/max, histogram) use it: the apply step runs on all cells. |
| `domain` (optional) | `'auto' \| readonly [number, number]` | `'auto'` (default) takes the exact minimum and maximum of the included valid cells, or an explicit `[min, max]` with `min <= max`. Cells outside an explicit domain are not counted in the histogram and clamp to 0 or 1 when applied. |
| `mode` (optional) | `GPURasterStretchMode` | `'linear'` (default) maps the domain to `[0, 1]`, `'percentile'` maps the `percentiles` of the histogram, `'equalize'` maps through the histogram CDF. |
| `percentiles` (optional) | `readonly [number, number]` | Percentile mode: `[low, high]` in percent, `0 <= low <= high <= 100`. Defaults to `[2, 98]`. |
| `gamma` (optional) | `number` | Display gamma: applies `t' = t ** (1 / gamma)` after the mode, as rio-color and GIMP levels do, so `gamma > 1` brightens and `gamma < 1` darkens. Finite and positive, defaults to 1. GDAL has no gamma; its `-exponent e` (`gdal_translate`, VRT) is `t ** e`, so it equals `gamma: 1 / e`. |
| `sigmoidContrast` (optional) | `number` | Sigmoidal contrast (rio-color / ImageMagick style) applied after gamma, normalised so 0 maps to 0 and 1 to 1. Zero or undefined disables it; otherwise `(0, 80]`. |
| `sigmoidMidpoint` (optional) | `number` | Sigmoid midpoint in `(0, 1)`. Defaults to 0.5. |
| `paletteInterpolation` (optional) | `'nearest' \| 'linear'` | Palette lookup: `'nearest'` (default) or `'linear'` channel interpolation. |

{/* /parameters */}

#### How it computes

{/* lens:GPURasterStretch */}

::lens{scenario=gpu-raster.raster-stretch rung=datum datum=1791 caption="The brightest cell: its value sits at the top of the domain and maps to 1."}
::lens{scenario=gpu-raster.raster-stretch rung=formula caption="Subtract the lower bound, divide by the domain width and clamp to 0–1."}
::lens{scenario=gpu-raster.raster-stretch rung=kernel caption="The apply pass maps every cell through the bounds the statistics passes found."}
::lens{scenario=gpu-raster.raster-stretch rung=graph caption="Extent, histogram, scan, finalize and apply nodes run in order."}

## `GPUIsolines` and `GPUIsobands`

Marching-squares contours of any float32 raster (terrain, density, interpolation output) with
per-frame levels. Both contributors share one definition, so filled bands and lines drawn together
coincide bit for bit on a device. Samples sit at cell centres of the per-frame world extent (row 0
at `minY`). A cell with a nodata corner (NaN or an infinity, optional `noDataValue`, optional
`validity`) emits nothing; infinities are nodata because a crossing cannot interpolate them. The
extent must be finite with `max > min` on both axes (a y-down or zero-size extent throws). Corners with
`value >= level` are high; saddles are resolved by the cell-centre average
`((v0 + v1) + (v2 + v3)) / 4`. Crossings are computed once per canonical edge (left to right,
bottom to top), so neighbouring cells produce identical vertices.

A corner exactly on a level gives a zero-length `GPUIsolines` segment. The raw segment output
keeps it, because the edge-based stitching needs it as the bridge between the neighbouring cells'
chains; a contour through a sample stays one polyline. Stitched polyline vertex lists collapse
bit-identical consecutive vertices (a polyline made only of such segments keeps one vertex), and
the vertex capacity is checked before that collapse.

`GPUIsolines` writes proper two-endpoint segments `[x0, y0, x1, y1]` (high side on the left),
their level index, and optionally the global edge ids `[startEdge, endEdge]`, ordered by
(cell, level, slot). Unlike `GPUTerrainContours`, a record is one segment, not a vertex pair to
reassemble. Passing `output.polylines` stitches segments of the same level into deterministic polylines
by pointer jumping in a fixed `ceil(log2(capacity)) + 1` rounds: open chains start at the segment
with no predecessor, closed rings at their smallest segment index and repeat the first vertex;
polylines are ordered by head segment.

`GPUIsobands` writes `bandClasses` (number of breaks `<=` each sample, `0xffffffff` for nodata) for
fragment-side shading, and band geometry as counter-clockwise triangles with a band index:
each cell's fragment of band `[b[k - 1], b[k])` is `Above(b[k - 1]) ∩ Below(b[k])`, built
combinatorially from the cell boundary walk (no float clipping), at most two convex pieces per
band and cell. An optional `status.vertexCount` (`3 * status.count`) feeds a non-indexed indirect draw, and
`firstBand`/`lastBand` limit the emitted bands per frame.

```ts
const lineParameters = new GPUParameterBuffer(device, {
  id: 'isolines', format: 'float32', length: GPU_ISOLINES_PARAMETER_LENGTH
});
const grid = makeGPURasterMetadata({width, height});
graph.add(new GPUIsolines({
  grid, values, levels, // levels: maximumLevelCount rows, contents per frame
  parameters: lineParameters.importToGraph(graph),
  output: {
    segments, segmentLevels,
    polylines: {vertices, polylineOffsets, polylineLevels, polylineClosed, polylineCount, vertexCount},
    status: {count, requiredCount, overflow, polylineOverflow}
  }
}));
graph.add(new GPUIsobands({
  grid, values, breaks: levels,
  parameters: bandParameters.importToGraph(graph),
  output: {
    bandClasses, triangles, triangleBands,
    status: {count: triangleCount, vertexCount: drawVertexCount, overflow: triangleOverflow}
  }
}));
lineParameters.write(getGPUIsolinesParameterValues({levelCount: 5, extent}, grid));
bandParameters.write(getGPUIsobandsParameterValues({breakCount: 5, extent}, grid)); // no recompile
```

The packers take the frame's grid (its width and height give the cell sizes and must match the
contributor's). On projected rasters add `localOrigin: [extent[0], extent[1]]` to both packers'
settings. Segments,
polyline vertices, triangles and ring vertices are then relative to it, so draw them with it as the
layer's coordinate origin. It also keeps `GPUIsobandRings` vertex hashes within their i32 range,
which saturates beyond `2^31 * vertexTolerance` (214,748 units at the default tolerance).

`GPUIsobands` can also write the band boundary as edges: `output.edges` (`float32x4`), `edgeBands`,
`status.edgeCount`, `status.edgeOverflow` and an optional `status.edgeRequiredCount`. Edges are counter-clockwise with the band on
the left; an edge shared by two cells cancels, and a zero-area cell (one or two samples exactly on the
lower break) writes none, so edges lying on a break leave no slits.

`GPUIsobandRings` chains those edges into closed rings with `GPUSegmentRingAssembly`, grouped by band, so
every band comes as shells and holes (holes attach within a band) and, optionally, GeoArrow polygons with
`sourceIds`. Props: `grid`, `values`, `breaks`, `parameters` (same packing as
`GPUIsobands`), a compile-time `edgeCapacity`, `vertexTolerance`, `splitTouchingRings` and `output` (the
`GPUSegmentRingAssembly` outputs plus optional `status.edgeCount` and `status.edgeOverflow`). Maximum: `edgeCapacity` and the ring capacity (`output.ringOffsets.length - 1`) are each at most `getGPUIsobandRingsMaximumEdgeCapacity(device)` and `getGPUIsobandRingsMaximumRingCapacity(device)` (8388608 on a default device). Unlike the triangle fill it
gives true boundary geometry, but filling it for display needs polygon triangulation or rasterization
(`GPUPolygonRasterization` gives an exact even-odd fill at raster resolution).

Outputs are ordered by count, `GPUScan`, and scatter, so they are deterministic; capacities are
`segments.length` and `triangleBands.length`, with clamped `status.count`, `status.overflow`, and `status.requiredCount`
rewritten every encoding. If segments overflow, stitching produces no polylines and sets its own
`status.polylineOverflow`; a short `vertices` buffer keeps only the complete polylines that fit (the worst case is
`2 * segmentCapacity` vertices). Coordinates match an f32 CPU oracle within a few ULP (GPU
division and FMA contraction); topology, ordering, and band indices match exactly.

{/* parameters: GPUIsobands generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUIsobands` parameters

- Packer: `getGPUIsobandsParameterValues(settings, grid)`, writes `GPU_ISOBANDS_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): Frame raster grid; its width and height give the cell sizes and must match the contributor's.

Settings of `getGPUIsobandsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `breakCount` | `number` | Number of active ascending breaks, at most the contributor's `maximumBreakCount` (larger values clamp on the GPU). `breakCount + 1` bands exist: band `k` covers `[b[k - 1], b[k])`. |
| `extent` | `readonly [number, number, number, number]` | Raster extent `[minX, minY, maxX, maxY]` in world units. Row 0 is at `minY`. |
| `firstBand` (optional) | `number` | First band emitted as geometry. Defaults to 0. Does not affect `bandClasses`. |
| `lastBand` (optional) | `number` | Last band (inclusive) emitted as geometry. Defaults to `breakCount` (every band). |
| `localOrigin` (optional) | `readonly [number, number]` | Float64 point that triangle vertices (and `GPUIsobandRings` ring vertices) are written relative to; add it back on the CPU or as the layer's coordinate origin. Defaults to `[0, 0]` (absolute vertices). Pass the raster or tile corner `[minX, minY]` for projected rasters: absolute float32 vertices are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about `1.2e-7` of their distance from the origin (1.2 cm at 100 km). |

{/* /parameters */}

{/* parameters: GPUIsolines generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUIsolines` parameters

- Packer: `getGPUIsolinesParameterValues(settings, grid)`, writes `GPU_ISOLINES_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): Frame raster grid; its width and height give the cell sizes and must match the contributor's.

Settings of `getGPUIsolinesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `levelCount` | `number` | Number of active leading rows of the `levels` view. Clamped by the contributor to its maximum. |
| `extent` | `readonly [number, number, number, number]` | World extent `[minX, minY, maxX, maxY]` of the raster. Sample `(column, row)` sits at the centre of its cell, `minX + (column + 0.5) * cellWidth`; row 0 is at `minY`. |
| `localOrigin` (optional) | `readonly [number, number]` | Float64 point that segment and polyline vertices are written relative to; add it back on the CPU or as the layer's coordinate origin. Defaults to `[0, 0]` (absolute vertices). Pass the raster or tile corner `[minX, minY]` for projected rasters: absolute float32 vertices are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about `1.2e-7` of their distance from the origin (1.2 cm at 100 km). |

{/* /parameters */}

#### `GPUIsolines`

Traces marching-squares contour lines of an elevation raster at several levels and stitches them into polylines.

::lens{scenario=gpu-raster.isolines rung=result caption="Contours every 100 m; closed rings ring the summits, open chains run to the window edge."}

#### How it computes

{/* lens:GPUIsolines */}

::lens{scenario=gpu-raster.isolines rung=datum datum=12 caption="Polyline 12, the 1300 m chain: its level, whether it is closed and where its vertices start."}
::lens{scenario=gpu-raster.isolines rung=formula caption="Each vertex sits where linear interpolation along a cell edge reaches the contour level."}
::lens{scenario=gpu-raster.isolines rung=kernel caption="Cells count and emit segments by their corner mask; stitching links segments by shared edge ids."}
::lens{scenario=gpu-raster.isolines rung=graph caption="Count, scan, scatter and publish segments, then link, jump, rank and gather polylines."}

#### `GPUIsobands`

Fills the elevation bands between consecutive breaks with counter-clockwise triangles tagged by band.

::lens{scenario=gpu-raster.isobands rung=result caption="Triangles of the 1300 to 1600 m bands, coloured by band index."}

#### How it computes

{/* lens:GPUIsobands */}

::lens{scenario=gpu-raster.isobands rung=datum datum=120 caption="Triangle 120: its band index and the three vertices relative to the window corner."}
::lens{scenario=gpu-raster.isobands rung=kernel caption="Each cell walks its boundary, joins level crossings and fan-triangulates the band fragments."}
::lens{scenario=gpu-raster.isobands rung=graph caption="Per-cell triangle counts, scan, scatter and publish for the emitted band window."}

#### `GPUIsobandRings`

Chains the band boundary edges of an elevation raster into closed shell and hole rings grouped by band.

::lens{scenario=gpu-raster.isoband-rings rung=result caption="Closed rings of all eight bands, coloured by band index."}

#### How it computes

{/* lens:GPUIsobandRings */}

::lens{scenario=gpu-raster.isoband-rings rung=datum datum=0 caption="Ring 0: its band, vertex range and the closed vertex list."}
::lens{scenario=gpu-raster.isoband-rings rung=kernel caption="Boundary vertices are hashed and matched, then rings are traced and ranked by pointer jumping."}
::lens{scenario=gpu-raster.isoband-rings rung=graph caption="Band boundary edges feed hash, sort, match, ring tracing and placement nodes."}

## `GPURasterSampling` and `GPURasterProfile`

`GPURasterSampling` samples a float32 raster at points ("extract raster values to points", draping,
tooltips). The pixel coordinate is `u = (x - minX) * (1 / cellWidth) - 0.5`, so integer `u` is a cell
centre; the extent, cell size, and row direction (row 0 at `maxY` for a north-up grid) come from the
`grid`. Points outside the closed extent (or with NaN coordinates) give NaN;
inside it, indices beyond the outermost centres clamp to the edge cell. `nearest` is the cell
containing the point (the `maxX`/`maxY` edge belongs to the last cell), `bilinear` uses the four
surrounding centres, `bicubic` is Catmull-Rom (`a = -0.5`) over the 4x4 support. A neighbour
participates only when both axis weights are nonzero, so a point on a cell centre returns that cell.
`noDataPolicy: 'strict'` gives NaN when a participating neighbour is nodata; `'renormalize'` divides
the valid bilinear weights by their sum; bicubic with nodata in its support falls back to the bilinear
rule of the same policy. Nodata is NaN, an optional finite `noDataValue`, or a zero `validity` flag.
`method`, `noDataPolicy`, and the packed grid placement are per-frame; `pointCount` is an optional per-frame active
count (rows beyond it are NaN and 0).

`GPURasterProfile` builds elevation profiles along polylines (`positions` plus CSR `lineOffsets`)
at a per-frame `spacing` (planar distances in grid world units). Each line emits samples at
`0, s, 2s, ...` below its length plus the final vertex at exactly the length; a zero-length line emits
one sample, an empty line none. Outputs (any subset): sample positions, distances, values, line IDs,
cumulative gain and loss; per line: sample offsets (clamped to capacity), length, gain, loss, minimum,
and maximum over finite samples (gain and loss sum positive and negative differences between
consecutive finite samples); plus `status.count`, `status.overflow`, and `status.requiredCount`.

```ts
const samplingParameters = new GPUParameterBuffer(device, {
  id: 'sampling', format: 'float32', length: GPU_RASTER_SAMPLING_PARAMETER_LENGTH
});
graph.add(new GPURasterSampling({
  grid, values: elevation, positions, parameters: samplingParameters.importToGraph(graph),
  output: {values: pointElevations, outputValidity}
}));
graph.add(new GPURasterProfile({
  grid, values: elevation, positions: lineVertices, lineOffsets,
  parameters: profileParameters.importToGraph(graph),
  output: {status: {count, overflow}, sampleValues, sampleDistances, lineGains, lineLosses}
}));
samplingParameters.write(getGPURasterSamplingParameterValues({method: 'bicubic'}, grid));
profileParameters.write(getGPURasterProfileParameterValues({method: 'bilinear', spacing: 25}, grid));
```

On projected rasters pass `localOrigin` (for example the raster corner) to either packer and
upload point `positions` and line `positions` relative to it; `samplePositions` come back relative to it.
Distances are unaffected.

The profile pipeline is a serial per-line walk, `GPUScan`, a per-sample binary search and sample,
then a serial per-line cumulative pass, all in fixed order, so results are deterministic. Exactly
representable (dyadic) inputs match an f32 CPU oracle bit for bit; elsewhere GPU division and FMA
contraction give a few ULP.

{/* parameters: GPURasterProfile generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterProfile` parameters

- Packer: `getGPURasterProfileParameterValues(settings, grid)`, writes `GPU_RASTER_PROFILE_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): The frame's grid; its width and height must match the contributor's.

Settings of `getGPURasterProfileParameterValues`:

| Field | Type | Description |
|---|---|---|
| `method` (optional) | `GPURasterSamplingMethod` | Interpolation method. Defaults to `'bilinear'`. |
| `noDataPolicy` (optional) | `GPURasterSamplingNoDataPolicy` | Nodata handling for bilinear and bicubic. Defaults to `'strict'`. |
| `localOrigin` (optional) | `readonly [number, number]` | Float64 point that query positions (sampling points, profile line vertices) are given relative to, and that profile sample positions are written relative to. Defaults to `[0, 0]` (absolute positions). Pass the raster or tile corner `[minX, minY]` for projected rasters: absolute float32 positions are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about `1.2e-7` of their distance from the origin (1.2 cm at 100 km). |
| `spacing` | `number` | Distance between samples along a line, in grid world units. Must be finite and positive. |

{/* /parameters */}

{/* parameters: GPURasterSampling generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURasterSampling` parameters

- Packer: `getGPURasterSamplingParameterValues(settings, grid)`, writes `GPU_RASTER_SAMPLING_PARAMETER_LENGTH` (12) `float32` values.
  - `grid` (`GPURasterMetadata`): The frame's grid; its width and height must match the contributor's.

Settings of `getGPURasterSamplingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `method` (optional) | `GPURasterSamplingMethod` | Interpolation method. Defaults to `'bilinear'`. |
| `noDataPolicy` (optional) | `GPURasterSamplingNoDataPolicy` | Nodata handling for bilinear and bicubic. Defaults to `'strict'`. |
| `localOrigin` (optional) | `readonly [number, number]` | Float64 point that query positions (sampling points, profile line vertices) are given relative to, and that profile sample positions are written relative to. Defaults to `[0, 0]` (absolute positions). Pass the raster or tile corner `[minX, minY]` for projected rasters: absolute float32 positions are quantized to 0.5 m at a UTM northing of 4.6e6 m, origin-relative ones to about `1.2e-7` of their distance from the origin (1.2 cm at 100 km). |

{/* /parameters */}

#### `GPURasterSampling`

Reads a raster at point positions with nearest, bilinear or bicubic interpolation, giving one value per point.

::lens{scenario=gpu-raster.raster-sampling rung=result caption="25 DEM sample points coloured by elevation; point 7 sits off cell centres and averages four cells."}

#### How it computes

{/* lens:GPURasterSampling */}

::lens{scenario=gpu-raster.raster-sampling rung=datum datum=7 caption="Point 7 sits on a cell corner, so four cells each weigh 0.25."}
::lens{scenario=gpu-raster.raster-sampling rung=formula caption="Each point blends the four surrounding cell centres by distance; nodata gives NaN under strict."}
::lens{scenario=gpu-raster.raster-sampling rung=kernel caption="One invocation per point finds its cell and weights, then reads the raster."}
::lens{scenario=gpu-raster.raster-sampling rung=graph caption="A single sampling node reads the raster, the points and the parameters."}

#### `GPURasterProfile`

Samples a raster along a polyline at a fixed spacing, giving elevation and distance per sample and total ascent per line.

::lens{scenario=gpu-raster.raster-profile rung=result caption="33 samples every 250 m along the line, coloured by elevation, fall from the ridge to the valley."}

#### How it computes

{/* lens:GPURasterProfile */}

::lens{scenario=gpu-raster.raster-profile rung=datum datum=14 caption="Sample 14, 3500 m along the line, after the steepest drop of the profile."}
::lens{scenario=gpu-raster.raster-profile rung=formula caption="Samples sit every spacing along the polyline, with a last one at its end."}
::lens{scenario=gpu-raster.raster-profile rung=kernel caption="Edge lengths are scanned, each sample finds its segment, then the sampling rule and a gain scan run."}
::lens{scenario=gpu-raster.raster-profile rung=graph caption="Vertex distances, sample counts, scan, locate, sample and gain nodes run in order."}

## `GPUParticleAdvection`

Moves particles through a vector field with midpoint steps and records each position in a trail ring buffer.

::lens{scenario=gpu-raster.particle-advection rung=result caption="200 particles drift for 24 hours of wind, coloured by final speed; none left the window."}

Advects particles through a 2D vector field (wind, ocean currents, flow directions) one frame per
encoding, with an optional trail ring buffer that a `PathLayer` or `LineLayer` can draw without CPU
readback. Unlike screen-space fading trails, the trails are in data space, so they survive panning
and zooming.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'wind', format: 'float32', length: 12});
const words = new GPUParameterBuffer(device, {id: 'wind-words', format: 'uint32', length: 4});
graph.add(
  new GPUParticleAdvection({
    velocities, // GraphDataView<'float32x2'>, (u, v) per cell, row 0 = smallest y
    fieldWidth: 360,
    fieldHeight: 180,
    parameters: parameters.importToGraph(graph),
    uint32Parameters: words.importToGraph(graph),
    state: {positions, ages, generations}, // particleCount rows each, updated in place
    output: {
      previousPositions, // optional: segment start for a LineLayer
      speeds, // optional: colour by speed
      trails: {positions: trailPositions, length: 16} // optional ring, particleCount * 16 rows
    }
  })
);
// Every frame, no recompile:
parameters.write(
  getGPUParticleAdvectionParameterValues(
    {fieldExtent: [-180, -90, 1, 1], timeStep: 1 / 60, speedScale: 0.2, dropRate: 0.003},
    [360, 180]
  )
);
words.write(getGPUParticleAdvectionUint32ParameterValues({seed: 1, frame, maximumAge: 120, reset: frame === 0}));
```

- Each frame takes one RK2 (midpoint) step with `h = timeStep * speedScale`, sampling the field by
  manual bilinear interpolation between cell centres (no filterable float texture needed).
- A particle respawns inside `spawnBounds` (default: the field bounds) when `reset` is set, its age
  reaches `maximumAge` (0 disables ageing), any sample leaves the field or touches a NaN cell, it is
  slower than `minimumSpeed`, or a per-frame drop test fires with probability `dropRate`. On `reset`
  ages are staggered so particles do not all expire together.
- Exact replays: random numbers come from a Philox 4x32-10 counter generator, keyed
  `(seed, particle, generation)` for spawn positions and `(seed, particle, frame)` for drop tests,
  with no persistent RNG state. The same parameter sequence replays bit for bit.
- Trails: particle `i` owns rows `i * L` to `i * L + L - 1`; frame `f` writes slot `f % L` (newest),
  the oldest is `(f + 1) % L`. A respawn fills the whole ring with the spawn position.
- State is updated in place: every invocation touches only its own particle, so no ping-pong copy is
  needed.

{/* parameters: GPUParticleAdvection generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUParticleAdvection` parameters

- Packer: `getGPUParticleAdvectionParameterValues(settings, fieldSize)`, writes `GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH` (12) `float32` values.
  - `fieldSize` (`readonly [number, number]`): `[width, height]` of the field in cells, used for the default spawn bounds.
- Packer: `getGPUParticleAdvectionUint32ParameterValues(settings)`, writes `GPU_PARTICLE_ADVECTION_UINT32_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUParticleAdvectionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `fieldExtent` | `readonly [number, number, number, number]` | Field raster placement `[originX, originY, cellWidth, cellHeight]`. Row 0 is the row with the smallest y; cell `(i, j)` is centred at `origin + (i + 0.5, j + 0.5) * cellSize`. |
| `timeStep` | `number` | Time step of this frame, in the time unit of the field velocities. |
| `speedScale` (optional) | `number` | Multiplier on the time step, for example a playback speed. Defaults to 1. |
| `dropRate` (optional) | `number` | Probability in `[0, 1]` that a particle respawns this frame regardless of age. Defaults to 0. |
| `minimumSpeed` (optional) | `number` | Particles slower than this (world units per time unit, after `speedScale`) respawn. Defaults to 0. |
| `spawnBounds` (optional) | `readonly [number, number, number, number]` | Respawn rectangle `[minX, minY, maxX, maxY]`. Defaults to the field bounds. Particles spawned outside the field respawn again on the next frame. |

Settings of `getGPUParticleAdvectionUint32ParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed. Every random decision is a pure function of the seed and particle counters. |
| `frame` | `number` | Frame counter. Keys the per-frame drop test and selects the trail ring slot `frame % L`. |
| `maximumAge` | `number` | Particles respawn when their age reaches this many frames. `0` disables ageing. |
| `reset` (optional) | `boolean` | When true, every particle respawns with generation 0 and a staggered age, which initialises the state buffers. Pass it on the first frame and whenever the scene is reset. |

{/* /parameters */}

#### How it computes

{/* lens:GPUParticleAdvection */}

::lens{scenario=gpu-raster.particle-advection rung=datum datum=11 caption="Particle 11, the farthest traveller: its trail of twelve positions and final wind speed."}
::lens{scenario=gpu-raster.particle-advection rung=formula caption="One midpoint step per frame: sample the wind at the start, then at the half step."}
::lens{scenario=gpu-raster.particle-advection rung=kernel caption="Each particle samples the field three times, then moves or respawns; a second pass writes its trail slot."}
::lens{scenario=gpu-raster.particle-advection rung=graph caption="Twelve step nodes and twelve trail nodes run in order on the same particle state."}

## `GPULineIntegralConvolution`

Smears seeded white noise along the streamlines of a vector field, giving a 0 to 1 texture that shows flow direction.

::lens{scenario=gpu-raster.line-integral-convolution rung=result caption="Noise streaks follow the wind across the 16 by 12 degree window."}

Line integral convolution: smears white noise along the streamlines of a vector field so the flow
direction is visible everywhere. Output is a float raster in `[0, 1]` (plus an optional `r32float`
texture and a speed raster) for a `BitmapLayer`, typically modulated by a speed colour ramp.

```ts
graph.add(
  new GPULineIntegralConvolution({
    velocities, fieldWidth: 360, fieldHeight: 180,
    grid: makeGPURasterMetadata({width: 1024, height: 512}), // output pixels
    stepCount: 20, // steps per direction
    parameters: parameters.importToGraph(graph), // float32, 12
    uint32Parameters: words.importToGraph(graph), // uint32, 4
    output: {values, speeds, texture}
  })
);
parameters.write(
  getGPULineIntegralConvolutionParameterValues({
    fieldExtent: [-180, -90, 1, 1],
    outputExtent: [-180, -90, 360 / 1024, 180 / 512],
    stepLength: 0.5, // output pixels
    period: 8, // animated ripple, in steps; 0 = static LIC
    phase: frame / 30 // advance every frame to make the texture flow
  })
);
```

- Noise is Philox white noise per output pixel keyed by `(seed, column, row)`; the kernel is a Hann
  window, optionally times an animated ripple `0.5 * (1 + cos(2 * pi * (s / period - phase)))`.
- Streamlines step along the normalised field (RK2) and stop at the field or output edge, NaN data,
  or speeds at or below `minimumSpeed`. Pixels whose centre has no data are NaN.
- The output extent is independent of the field, so a viewport renders at screen resolution from a
  coarse field.

{/* parameters: GPULineIntegralConvolution generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPULineIntegralConvolution` parameters

- Packer: `getGPULineIntegralConvolutionParameterValues(settings)`, writes `GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH` (12) `float32` values.
- Packer: `getGPULineIntegralConvolutionUint32ParameterValues(settings)`, writes `GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPULineIntegralConvolutionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `fieldExtent` | `readonly [number, number, number, number]` | Field placement `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. |
| `outputExtent` | `readonly [number, number, number, number]` | Output raster placement `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. |
| `stepLength` (optional) | `number` | Streamline step in output pixels. Defaults to 0.5. |
| `minimumSpeed` (optional) | `number` | A streamline stops where the field is slower than this. Defaults to 0. |
| `phase` (optional) | `number` | Animation phase in kernel periods; advance it every frame to make the texture flow. Defaults to 0. |
| `period` (optional) | `number` | Ripple period in steps. `0` disables the animated ripple (static LIC). Defaults to 0. |
| `seed` (optional) | `number` | Seed of the white noise. Defaults to 0. |

Settings of `getGPULineIntegralConvolutionUint32ParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Seed of the white noise. Defaults to 0. |

{/* /parameters */}

#### How it computes

{/* lens:GPULineIntegralConvolution */}

::lens{scenario=gpu-raster.line-integral-convolution rung=datum datum=6208 caption="The centre pixel: its wind speed and the value from its weighted noise samples."}
::lens{scenario=gpu-raster.line-integral-convolution rung=formula caption="A Hann-weighted mean of the noise along the streamline, 24 steps each way."}
::lens{scenario=gpu-raster.line-integral-convolution rung=kernel caption="Each pixel traces its streamline both ways and averages the noise it visits."}
::lens{scenario=gpu-raster.line-integral-convolution rung=graph caption="A noise node fills the white noise, then the convolution node reads it."}

## `GPUStreamlines`

Traces evenly spaced streamlines from jittered seeds of a vector field and prunes them to a separation distance.

::lens{scenario=gpu-raster.streamlines rung=result caption="Seeded lines pruned to 0.5 degree separation, coloured by seed index."}

Evenly spaced streamlines (after Jobard and Lefer) whose pruning is resolved by a random priority
per seed instead of processing order, so the result is deterministic. Output is CSR polylines for a
`PathLayer`.

```ts
graph.add(
  new GPUStreamlines({
    velocities, fieldWidth: 360, fieldHeight: 180,
    gridWidth: 180, gridHeight: 90, // occupancy grid; its cell is the separation distance
    seedColumns: 90, seedRows: 45, stepsPerDirection: 60, maximumIterations: 16,
    parameters: parameters.importToGraph(graph), // float32, 12
    uint32Parameters: words.importToGraph(graph), // uint32, 4
    output: {
      lines: {ids, status: {count, overflow, requiredCount}}, lineOffsets, positions, positionCount,
      status: {iterationLimitReached}
    }
  })
);
parameters.write(
  getGPUStreamlinesParameterValues({
    fieldExtent: [-180, -90, 1, 1],
    gridExtent: [-180, -90, 2, 2],
    stepLength: 0.5
  })
);
words.write(getGPUStreamlinesUint32ParameterValues({seed: 1, minimumPoints: 8}));
```

- Seeds form a Philox-jittered lattice over the grid extent; each is traced `L` RK2 steps backward and
  forward. Priorities are Philox keys with ties toward the lower seed index.
- The result equals a greedy pass in descending priority: reject a line whose seed cell is occupied,
  cut each direction before its first occupied cell, accept it if it keeps `minimumPoints` points,
  and occupy its cells. The GPU reaches it in at most `maximumIterations` GPU-gated rounds of claim (`atomicMax` of
  keys over the cells a line can still keep) and decide (accept lines that win all their cells);
  `status.iterationLimitReached` is 1 if rounds ran out (undecided lines are dropped).
- Published lines are in ascending seed order: line `i` is `positions[lineOffsets[i] .. lineOffsets[i + 1])`.
  `lines.status.overflow` is 1 when lines were dropped because the line or point capacity was full.
- Optional `candidates` outputs expose the traced lines before pruning, for diagnostics.

{/* parameters: GPUStreamlines generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUStreamlines` parameters

- Packer: `getGPUStreamlinesParameterValues(settings)`, writes `GPU_STREAMLINES_PARAMETER_LENGTH` (12) `float32` values.
- Packer: `getGPUStreamlinesUint32ParameterValues(settings)`, writes `GPU_STREAMLINES_UINT32_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUStreamlinesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `fieldExtent` | `readonly [number, number, number, number]` | Field placement `[originX, originY, cellWidth, cellHeight]`; row 0 has the smallest y. |
| `gridExtent` | `readonly [number, number, number, number]` | Occupancy grid placement `[originX, originY, cellWidth, cellHeight]`. The cell size is the separation distance between streamlines; seeds and traced points stay inside the grid. |
| `stepLength` | `number` | Integration step in world units. |
| `minimumSpeed` (optional) | `number` | A streamline stops where the field is slower than this. Defaults to 0. |
| `seed` (optional) | `number` | Seed of the jittered seed lattice and the line priorities. Defaults to 0. |
| `minimumPoints` (optional) | `number` | Streamlines with fewer points after trimming are rejected. Defaults to 2. |

Settings of `getGPUStreamlinesUint32ParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Seed of the jittered seed lattice and the line priorities. Defaults to 0. |
| `minimumPoints` (optional) | `number` | Streamlines with fewer points after trimming are rejected. Defaults to 2. |

{/* /parameters */}

#### How it computes

{/* lens:GPUStreamlines */}

::lens{scenario=gpu-raster.streamlines rung=datum datum=11 caption="Line 11: its seed index and where its points start in the shared position list."}
::lens{scenario=gpu-raster.streamlines rung=kernel caption="Seeds are traced both ways, then rounds claim and decide occupancy-grid cells."}
::lens{scenario=gpu-raster.streamlines rung=graph caption="Trace, pruning rounds, scans and publish nodes build the CSR polylines."}

## `GPUChangeDetection`

Subtracts the before slice from the after slice for each band; here near-infrared and shortwave-infrared reflectance around the fire.

::lens{scenario=gpu-raster.change-detection rung=result caption="After minus before for near-infrared; strong decreases mark the fire scar."}

`GPUChangeDetection` computes per-cell change statistics over a dense float32 stack of time slices
indexed `(cell * sliceCount + slice) * bandCount + band` (cell-major, like `GPUTemporalReduction`).
NaN means missing. The modes are selected at compile time by which outputs are present; each
present output group adds one compute node (`two-slice`, `multiband`, `t-test`, `sen-slope`,
`mann-kendall`), and one thread handles one cell with fixed-order loops, so results are
deterministic and no atomics are used.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'change-parameters',
  format: 'float32',
  length: GPU_CHANGE_DETECTION_PARAMETER_LENGTH
});
graph.add(
  new GPUChangeDetection({
    slices: stack, // cellCount * sliceCount float32 rows
    parameters: parameters.importToGraph(graph),
    cellCount,
    sliceCount,
    output: {difference, logRatio, tStatistic, tPValue, senSlope, mannKendallS, significance}
  })
);
parameters.write(
  getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 11, splitSlice: 6, alpha: 0.05})
);
```

- `compile-time`: `cellCount`, `sliceCount` (2 to 256, at most 64 with `senSlope`), `bandCount`
  (default 1), the optional cell `mask`, `significanceSource`, and which outputs exist.
- `per frame` (parameter buffer, no rebuild): `beforeSlice`, `afterSlice`, `epsilon`, `alpha`,
  `splitSlice`.
- Two-slice outputs (any `bandCount`, `cellCount * bandCount` rows): `difference` (after minus
  before), `logRatio` (`ln((after + epsilon) / (before + epsilon))`, NaN when either side is
  `<= -epsilon`) and `percentChange` (`100 * (after - before) / |before|`, NaN when before is 0).
- Welch t-test between slices `[0, splitSlice)` and `[splitSlice, sliceCount)`: `tStatistic`,
  `tDegreesOfFreedom` (Welch-Satterthwaite) and two-sided `tPValue`. NaN slices are dropped; each
  group needs 2 valid slices and a positive standard error. Variance is two-pass. The p-value is
  the regularized incomplete beta `I_x(df / 2, 1 / 2)`, `x = df / (df + t^2)`, by continued
  fraction (at most 64 iterations) with a Stirling log-gamma, in float32 (about 1e-4 absolute for
  `df` up to 254).
- `senSlope`: exact Theil-Sen median of the pairwise slopes over valid slices, by in-thread bounded
  max-heap selection. `O(T^2 log T)` per cell, so `sliceCount <= 64`.
- `mannKendallS` (`sint32`, integer exact), `mannKendallZ` and `mannKendallP`: tie-corrected
  variance, `Z = (S - sign(S)) / sqrt(var)`, `p = erfc(|Z| / sqrt(2))` (Numerical Recipes `erfcc`,
  relative error below 1.2e-7). Fewer than 2 valid slices give NaN; all values tied give Z 0, p 1.
- Multiband (`bandCount >= 2`, at most 16): `changeMagnitude` (Euclidean norm of the band
  difference vector) and `changeDirection` (float32 `atan2(d1, d0)` for 2 bands; uint32 sign code
  for 3 to 16 bands with bit `b` set for an increase and bit `16 + b` for a decrease; NaN or
  `0xffffffff` when any band is missing).
- `significance` (uint32): 1 significant increase, 2 significant decrease, 0 otherwise, from the
  t-test p-value (default) or Mann-Kendall p-value (`significanceSource: 'mann-kendall'`) below
  `alpha`; the sign follows the t statistic or S.
- Masked cells write NaN (float), 0 (`mannKendallS`, `significance`) or `0xffffffff` (multiband
  direction code).

{/* parameters: GPUChangeDetection generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUChangeDetection` parameters

- Packer: `getGPUChangeDetectionParameterValues(settings)`, writes `GPU_CHANGE_DETECTION_PARAMETER_LENGTH` (5) `float32` values.

Settings of `getGPUChangeDetectionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `beforeSlice` | `number` | Slice index of the "before" slice for two-slice comparisons. Out of range makes them NaN. |
| `afterSlice` | `number` | Slice index of the "after" slice for two-slice comparisons. Out of range makes them NaN. |
| `epsilon` (optional) | `number` | Denominator guard of the log ratio. Defaults to `1e-6`. |
| `alpha` (optional) | `number` | Two-sided significance level for the `significance` class. Defaults to `0.05`. |
| `splitSlice` (optional) | `number` | First slice of the "after" group of the Welch t-test: slices `[0, splitSlice)` are "before" and `[splitSlice, sliceCount)` are "after". Must be in `[1, sliceCount - 1]`. Defaults to 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUChangeDetection */}

::lens{scenario=gpu-raster.change-detection rung=datum datum=2400 caption="One burn-scar cell: select the before and after slices, then subtract."}
::lens{scenario=gpu-raster.change-detection rung=formula caption="Difference of the two selected slices; masked, out-of-range or missing input gives NaN."}
::lens{scenario=gpu-raster.change-detection rung=kernel caption="One thread per cell reads both slices and writes the difference."}
::lens{scenario=gpu-raster.change-detection rung=graph caption="One two-slice node per band, sharing one parameter buffer."}

## `GPURasterExtremaPyramid`

Builds a min and max mip chain of a raster band, each cell holding the extremes of its pixel block.

::lens{scenario=gpu-raster.raster-extrema-pyramid rung=result caption="Highest elevation of each 960 m block, level 1 of the pyramid."}

An exact min/max mip chain (Tevs et al. 2008 maximum mipmaps) for conservative ray skipping. The
[GPU Terrain](../gpu-terrain/README.md) sight-line and point-horizon contributors use it for
`traversal: 'pyramid'`.

- **Grid and outputs.** The raster is `grid` (a `GPURasterMetadata`; its width and height size the
  pyramid, cells live in pixel index space so no cell size applies). The caller-owned views go in
  `output` (`GPURasterExtremaPyramidViews`): `maximum`, `minimum`, or one `combined` view of
  `2 * layout.length` floats. With `combined`, `pyramid.output` (a `GPURasterExtremaPyramidOutput`,
  `{layout, combined}`) is what the consumers take as their `pyramid` prop.
- **Levels.** Level `L` has blocks of `firstBlockSize·2^L` pixels (default 4), down to 1×1 or
  `levelCount`. Levels are packed into one float32 view per extremum, with offsets from
  `getGPURasterExtremaPyramidLayout(grid, options)`.
- **Footprint.** With `footprint: 'bilinear'` (the default), each cell also covers one extra
  pixel column and row. Every bilinear sample whose base pixel lies in the block is then bounded by
  that one cell. `'cell'` gives the plain block extrema.
- **Validity.** Invalid pixels are excluded. Cells with no valid pixel hold ∓FLT_MAX
  (`GPU_RASTER_EXTREMA_PYRAMID_EMPTY_*`).
- **Exactness.** The GPU output is bit-identical to a CPU build. Level 0 reduces from pixels and
  each higher level reduces the 2×2 cells below in a fixed order.

#### How it computes

{/* lens:GPURasterExtremaPyramid */}

::lens{scenario=gpu-raster.raster-extrema-pyramid rung=datum datum=48 caption="Block 48, the highest block: its four children at level 0 and their maximum."}
::lens{scenario=gpu-raster.raster-extrema-pyramid rung=formula caption="A parent maximum is the largest of its 2 by 2 children; level 0 reduces valid pixels of a block."}
::lens{scenario=gpu-raster.raster-extrema-pyramid rung=kernel caption="Level 0 scans each block; a fused workgroup pass reduces 16 by 16 tiles through the higher levels."}
::lens{scenario=gpu-raster.raster-extrema-pyramid rung=graph caption="Canonicalize the band, reduce level 0, then one fused pass for the remaining levels."}

## Related pages

- [GPURaster operations reference](/docs/api-reference/experimental/gpu-raster/operations)
- [GPU Terrain](/docs/api-reference/experimental/gpu-terrain)
- [GPU Core](/docs/api-reference/experimental/gpu-core)
