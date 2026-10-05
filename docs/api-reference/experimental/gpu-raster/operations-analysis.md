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
const settings = new GPUParameterBuffer(device, {id: 'df-settings', format: 'float32', length: 8});

// `seedPositions`, `seedCount`, `seedMask`, and the output views are graph views created on `graph`.
graph.add(new GPUDistanceField({
  width, height, settings: settings.importToGraph(graph),
  seedPositions, seedCount, seedMask,
  output: {distances, allocation, withinDistance}
}));
const compiled = graph.compile(); // once

settings.write(getGPUDistanceFieldParameterValues({
  bounds: [minX, minY, maxX, maxY], gridSize: [width, height], maxDistance: 500
}));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

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

## `GPUCostDistance` and `GPUCostDistancePath`

Accumulated cost surface over a friction raster from per-frame source cells, a source mask, or
both: 8-connected moves cost ground distance times mean friction, invalid or negative friction is
a barrier, and a per-frame cost limit, isoline-ready bands with counts, and D8 back-links are
optional. Relaxation runs in 16x16 workgroup tiles with workgroup-memory repeats and per-tile
activity, gated on the GPU like `GPUNetworkReachability`. `GPUCostDistancePath` walks the
back-links from a per-frame target into a compact cell list. The cost surface feeds
`GPUTerrainContours` directly for isolines.

```ts
graph.add(new GPUCostDistance({
  width, height, friction, settings: settings.importToGraph(graph),
  sources: sources.importToGraph(graph), costs, backLinks, converged, maxIterations: 64
}));
graph.add(new GPUCostDistancePath({width, height, backLinks, target: target.importToGraph(graph), output}));
settings.write(getGPUCostDistanceParameterValues({cellSize: [10, 10], costLimit: 5000}));
```

Back-links are cycle-safe across zero-cost plateaus: a cell links to its strictly cheaper tight
neighbor or, when it was reached only across zero-cost moves, to the equal-cost neighbor one tie
level closer to a strict entry or a source. The tie phase runs only when `backLinks` is requested
and takes up to `maxTieIterations` (default 8) extra gated iterations; `converged` is 0 if either
phase was truncated.

Generated transient IDs are prefixed with the contributor `id`, for example `<id>-total`. If one
collides with a caller resource, the error names the contributor and the generated ID.

## `GPURasterZonalStatistics`

Per-zone cell count, valid-value count, sum, mean, minimum, and maximum of a value band over a
dense `uint32` zone raster (for example rasterized administrative areas). Calibration, nodata,
validity, and non-finite samples are honored; one ignored zone ID is skipped silently, and zone IDs
at or above the compile-time capacity raise a GPU overflow flag. Composes `GPUGroupAggregation`.

```ts
graph.add(new GPURasterZonalStatistics({
  width, height, zones, values: {id: 'ndvi', format: 'float32', storage: {kind: 'buffer', values}},
  zoneCapacity: 256, ignoredZone: 0, output: {valueCounts, means, minimums, maximums}, overflow
}));
```

Sums and means use float atomics by default, so their last bits can vary between runs.
`sumOrder: 'sorted'` stably sorts cells by zone ID and reduces each zone in a fixed tree (the same
sorted segmented sum as `GPUZonalStatistics`), making `sums` and `means` bitwise reproducible on
one device at the cost of a radix sort, scan, and gather over all cells. Counts, minimums, and
maximums are exact either way.

## `GPUDistanceField`

Euclidean distance transform and nearest-seed allocation (Voronoi zones) on a raster grid, from
seed points snapped to cells, a labeled seed raster, or both. `distances` holds the ground distance
from each cell center to the nearest seed cell center (`+Infinity` when unreached), `allocation`
the nearest seed's ID, and `nearestCells` its cell; ties break on the smallest seed ID. A
per-frame `maxDistance` clears cells beyond it and fills an optional `withinDistance` mask, and an
optional `r32float` texture receives a copy of `distances`. Cell sizes may differ in x and y.

```ts
const settings = new GPUParameterBuffer(device, {id: 'df-settings', format: 'float32', length: 8});
graph.add(new GPUDistanceField({
  width, height, settings: settings.importToGraph(graph),
  seedPositions, seedCount, seedMask,  // capacity is seedPositions.length
  output: {distances, allocation, withinDistance}
}));
settings.write(getGPUDistanceFieldParameterValues({bounds: [minX, minY, maxX, maxY], gridSize: [width, height], maxDistance: 500}));
```

`mode: 'exact'` (default) is the separable Felzenszwalb-Huttenlocher transform: one invocation per
column finds the nearest seed row, then one invocation per row builds the discrete lower envelope
of those candidates. Candidates compare by exact integer squared offsets when the cell is square
and by f32 squared ground distance otherwise, so allocation matches a brute-force oracle exactly
on square cells. A correctly rounded square root keeps distances within 1 ULP of an f64 reference
(0 ULP when the cell size is a power of two). `mode: 'jump-flood'` is a cheaper preview for
dragging seeds: `ceil(log2(max(width, height)))` full-grid passes plus `jumpFloodRefinementPasses`
(default 1). It always reports the distance to a real seed, so it can only overestimate; on random
256x256 scenes JFA+1 misallocated at most 0.009% of cells with errors up to 1.24 cells, and JFA+2
was exact in the measured scenes (not guaranteed).

A seed mask value `v` is the seed ID `v - 1`: write `1` for a plain mask or `label + 1` for
per-zone allocation. Seed point IDs default to the row index or come from `seedIds`. Rewriting
seeds, the count, the mask, or settings never recompiles the graph.

## `GPUPolygonRasterization` and `GPURasterJoin`

`GPUPolygonRasterization` scan-converts polygon features (the GeoArrow layout of
`GPUPointInPolygonJoin`: `polygonPositions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`) into
a dense `uint32` zone raster on the GPU, plus optional boundary-cell flags. `GPURasterJoin` then
aggregates points by looking up their cell: O(1) per point, independent of polygon complexity.

```ts
const extent = new GPUParameterBuffer(device, {id: 'extent', format: 'float32', length: 4});
extent.write(getGPUPolygonRasterizationExtentValues(originX, originY, cellWidth, cellHeight));
const extentView = extent.importToGraph(graph);
graph.add(new GPUPolygonRasterization({
  width, height, extent: extentView,
  polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
  crossingCapacity: 1 << 20, zones, boundary, overflow, crossingCount
}));
graph.add(new GPURasterJoin({
  width, height, extent: extentView, points, values, zones, boundary, zoneCount: featureCount,
  output: {counts, sums, boundaryCounts, unassignedBoundaryCount, outsideCount, pointBoundaryMask}
}));
extent.write(getGPUPolygonRasterizationExtentValues(...nextExtent)); // no recompile
```

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
  `[inputMin, inputMax]` to `[0, 1]` (clamped, optionally inverted) or through a per-layer break
  table, then `score = sum(weight * remapped)` in fixed layer order. A NaN table value marks a
  restricted class (ArcGIS "Restricted"). Options: normalize by the sum of absolute weights,
  nodata `'propagate'` or `'ignore'`. `scoreRange` is the exact min/max of the defined scores
  (order-preserving integer atomics).
- `GPURasterCellStatistics` (ArcGIS "Cell Statistics") reduces up to 64 layers per cell:
  minimum, maximum, range, sum, mean, population standard deviation (centred second pass),
  majority, minority (ties pick the smallest value), variety, and valid count. Frequencies are an
  `O(layerCount^2)` exact-equality scan per cell.
- `GPURasterConditional` is `where(condition, a, b)`: a `uint32` mask, or a float raster compared
  per frame (`<`, `<=`, `>`, `>=`, `==`, `!=`, inclusive `between`); `a` and `b` are rasters or
  per-frame constants.
- `GPURasterArithmetic` evaluates `op(a * scaleA + offsetA, b * scaleB + offsetB)` with a per-frame
  operation: add, subtract, multiply, divide, minimum, maximum, power, absolute difference,
  normalized difference (NDVI), or a unary absolute/square root/log/exp/floor/ceil/round
  (half to even) on `a`, then an optional clamp. Domain errors give NaN.

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
    {weight: 0.5, inputMin: 0, inputMax: 30, invert: true}, // slope: flatter is better
    {weight: 0.3, mode: 'table', breakCount: 4},            // land cover classes
    {weight: 0.2, inputMin: 0, inputMax: 5000}              // distance to roads
  ],
  normalizeWeights: true
})); // no recompile
```

Every contributor is one invocation per cell reading inputs in fixed order, so results are
deterministic per device. Backends may contract `a * b + c` into an FMA (Metal does), so scores
and scaled operands can differ from an unfused CPU evaluation by a few roundings; reclassify,
conditional, extremes, sums, frequencies, and counts match a CPU oracle bit for bit.

## `GPURasterStretch`

Contrast stretches and colormap lookups for a float32 raster, computed on the GPU every frame with
no readback: a linear min/max stretch, a percentile stretch (for example 2%/98%), and histogram
equalization, with optional gamma (`t^gamma`) and sigmoidal contrast (normalized so 0 and 1 are
fixed). Outputs (any subset): per-cell `stretched` values in `[0, 1]`, packed rgba8 `colors`
(`r | g << 8 | b << 16 | a << 24`) through a per-frame `palette` (nearest or linear), a `lut` of
`lutSize` normalized values with optional `lutColors`, the `histogram`, and an 8-row `statistics`
row `[domainMin, domainMax, lo, hi, validCount, binWidth, 0, 0]` (`GPU_RASTER_STRETCH_STATISTICS_INDEX`).

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
  values, width, height, noDataValue: -9999,
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

## `GPUIsolines` and `GPUIsobands`

Marching-squares contours of any float32 raster (terrain, density, interpolation output) with
per-frame levels. Both contributors share one definition, so filled bands and lines drawn together
coincide bit for bit on a device. Samples sit at cell centres of the per-frame world extent (row 0
at `minY`). A cell with a nodata corner (NaN, optional `noDataValue`, optional `validity`) emits
nothing. Corners with `value >= level` are high; saddles are resolved by the cell-centre average
`((v0 + v1) + (v2 + v3)) / 4`. Crossings are computed once per canonical edge (left to right,
bottom to top), so neighbouring cells produce identical vertices.

`GPUIsolines` writes proper two-endpoint segments `[x0, y0, x1, y1]` (high side on the left),
their level index, and optionally the global edge ids `[startEdge, endEdge]`, ordered by
(cell, level, slot). Unlike `GPUTerrainContours`, a record is one segment, not a vertex pair to
reassemble. Passing `polylines` stitches segments of the same level into deterministic polylines
by pointer jumping in a fixed `ceil(log2(capacity)) + 1` rounds: open chains start at the segment
with no predecessor, closed rings at their smallest segment index and repeat the first vertex;
polylines are ordered by head segment.

`GPUIsobands` writes `bandClasses` (number of breaks `<=` each sample, `0xffffffff` for nodata) for
fragment-side shading, and band geometry as counter-clockwise triangles with a band index:
each cell's fragment of band `[b[k - 1], b[k])` is `Above(b[k - 1]) ∩ Below(b[k])`, built
combinatorially from the cell boundary walk (no float clipping), at most two convex pieces per
band and cell. An optional `vertexCount` (`3 * count`) feeds a non-indexed indirect draw, and
`firstBand`/`lastBand` limit the emitted bands per frame.

```ts
const lineParameters = new GPUParameterBuffer(device, {
  id: 'isolines', format: 'float32', length: GPU_ISOLINES_PARAMETER_LENGTH
});
graph.add(new GPUIsolines({
  width, height, values, levels, // levels: maximumLevelCount rows, contents per frame
  parameters: lineParameters.importToGraph(graph),
  output: {segments, segmentLevels, count, overflow, totalCount},
  polylines: {vertices, polylineOffsets, polylineLevels, polylineClosed, polylineCount, vertexCount, overflow: polylineOverflow}
}));
graph.add(new GPUIsobands({
  width, height, values, breaks: levels,
  parameters: bandParameters.importToGraph(graph),
  output: {bandClasses, triangles, triangleBands, count: triangleCount, overflow: triangleOverflow, vertexCount: drawVertexCount}
}));
lineParameters.write(getGPUIsolinesParameterValues({width, height, levelCount: 5, extent}));
bandParameters.write(getGPUIsobandsParameterValues({width, height, breakCount: 5, extent})); // no recompile
```

Outputs are ordered by count, `GPUScan`, and scatter, so they are deterministic; capacities are
`segments.length` and `triangleBands.length`, with clamped `count`, `overflow`, and `totalCount`
rewritten every encoding. If segments overflow, stitching produces no polylines and sets its own
overflow; a short `vertices` buffer keeps only the complete polylines that fit (the worst case is
`2 * segmentCapacity` vertices). Coordinates match an f32 CPU oracle within a few ULP (GPU
division and FMA contraction); topology, ordering, and band indices match exactly.

## `GPURasterSampling` and `GPURasterProfile`

`GPURasterSampling` samples a float32 raster at points ("extract raster values to points", draping,
tooltips). The pixel coordinate is `u = (x - minX) * (1 / cellWidth) - 0.5`, so integer `u` is a cell
centre and row 0 is at `minY`. Points outside the closed extent (or with NaN coordinates) give NaN;
inside it, indices beyond the outermost centres clamp to the edge cell. `nearest` is the cell
containing the point (the `maxX`/`maxY` edge belongs to the last cell), `bilinear` uses the four
surrounding centres, `bicubic` is Catmull-Rom (`a = -0.5`) over the 4x4 support. A neighbour
participates only when both axis weights are nonzero, so a point on a cell centre returns that cell.
`noDataPolicy: 'strict'` gives NaN when a participating neighbour is nodata; `'renormalize'` divides
the valid bilinear weights by their sum; bicubic with nodata in its support falls back to the bilinear
rule of the same policy. Nodata is NaN, an optional finite `noDataValue`, or a zero `validity` flag.
`method`, `noDataPolicy`, and the extent are per-frame; `pointCount` is an optional per-frame active
count (rows beyond it are NaN and 0).

`GPURasterProfile` builds elevation profiles along polylines (`pathPositions` plus CSR `pathOffsets`)
at a per-frame `spacing` (planar distances in extent units). Each path emits samples at
`0, s, 2s, ...` below its length plus the final vertex at exactly the length; a zero-length path emits
one sample, an empty path none. Outputs (any subset): sample positions, distances, values, path IDs,
cumulative gain and loss; per path: sample offsets (clamped to capacity), length, gain, loss, minimum,
and maximum over finite samples (gain and loss sum positive and negative differences between
consecutive finite samples); plus `count`, `overflow`, and `totalCount`.

```ts
const samplingParameters = new GPUParameterBuffer(device, {
  id: 'sampling', format: 'float32', length: GPU_RASTER_SAMPLING_PARAMETER_LENGTH
});
graph.add(new GPURasterSampling({
  width, height, values: elevation, positions, parameters: samplingParameters.importToGraph(graph),
  output: {values: pointElevations, validity}
}));
graph.add(new GPURasterProfile({
  width, height, values: elevation, pathPositions, pathOffsets,
  parameters: profileParameters.importToGraph(graph),
  output: {count, overflow, sampleValues, sampleDistances, pathGain, pathLoss}
}));
samplingParameters.write(getGPURasterSamplingParameterValues({width, height, extent, method: 'bicubic'}));
profileParameters.write(getGPURasterProfileParameterValues({width, height, extent, method: 'bilinear', spacing: 25}));
```

The profile pipeline is a serial per-path walk, `GPUScan`, a per-sample binary search and sample,
then a serial per-path cumulative pass, all in fixed order, so results are deterministic. Exactly
representable (dyadic) inputs match an f32 CPU oracle bit for bit; elsewhere GPU division and FMA
contraction give a few ULP.

## `GPUParticleAdvection`

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
    wordParameters: words.importToGraph(graph),
    state: {positions, ages, generations}, // particleCount rows each, updated in place
    previousPositions, // optional: segment start for a LineLayer
    speeds, // optional: colour by speed
    trails: {positions: trailPositions, length: 16} // optional ring, particleCount * 16 rows
  })
);
// Every frame, no recompile:
parameters.write(
  getGPUParticleAdvectionParameterValues(
    {fieldExtent: [-180, -90, 1, 1], timeStep: 1 / 60, speedScale: 0.2, dropRate: 0.003},
    [360, 180]
  )
);
words.write(getGPUParticleAdvectionWordParameterValues({seed: 1, frame, maximumAge: 120, reset: frame === 0}));
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

## `GPULineIntegralConvolution`

Line integral convolution: smears white noise along the streamlines of a vector field so the flow
direction is visible everywhere. Output is a float raster in `[0, 1]` (plus an optional `r32float`
texture and a speed raster) for a `BitmapLayer`, typically modulated by a speed colour ramp.

```ts
graph.add(
  new GPULineIntegralConvolution({
    velocities, fieldWidth: 360, fieldHeight: 180,
    width: 1024, height: 512, stepCount: 20, // output pixels, steps per direction
    parameters: parameters.importToGraph(graph), // float32, 12
    wordParameters: words.importToGraph(graph), // uint32, 4
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

## `GPUStreamlines`

Evenly spaced streamlines (after Jobard and Lefer) whose pruning is resolved by a random priority
per seed instead of processing order, so the result is deterministic. Output is CSR polylines for a
`PathLayer`.

```ts
graph.add(
  new GPUStreamlines({
    velocities, fieldWidth: 360, fieldHeight: 180,
    gridWidth: 180, gridHeight: 90, // occupancy grid; its cell is the separation distance
    seedColumns: 90, seedRows: 45, stepsPerDirection: 60, roundCount: 16,
    parameters: parameters.importToGraph(graph), // float32, 12
    wordParameters: words.importToGraph(graph), // uint32, 4
    output: {lines: {ids, count, overflow, totalCount}, pathOffsets, points, pointCount, unconverged}
  })
);
parameters.write(
  getGPUStreamlinesParameterValues({
    fieldExtent: [-180, -90, 1, 1],
    gridExtent: [-180, -90, 2, 2],
    stepLength: 0.5
  })
);
words.write(getGPUStreamlinesWordParameterValues({seed: 1, minimumPoints: 8}));
```

- Seeds form a Philox-jittered lattice over the grid extent; each is traced `L` RK2 steps backward and
  forward. Priorities are Philox keys with ties toward the lower seed index.
- The result equals a greedy pass in descending priority: reject a line whose seed cell is occupied,
  cut each direction before its first occupied cell, accept it if it keeps `minimumPoints` points,
  and occupy its cells. The GPU reaches it in `roundCount` GPU-gated rounds of claim (`atomicMax` of
  keys over the cells a line can still keep) and decide (accept lines that win all their cells);
  `unconverged` is 1 if rounds ran out (undecided lines are dropped).
- Published lines are in ascending seed order: line `i` is `points[pathOffsets[i] .. pathOffsets[i + 1])`.
  `lines.overflow` is 1 when lines were dropped because the line or point capacity was full.
- Optional `candidates` outputs expose the traced lines before pruning, for diagnostics.

## `GPUChangeDetection`

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

## `GPURasterExtremaPyramid`

An exact min/max mip chain (Tevs et al. 2008 maximum mipmaps) for conservative ray skipping. The
[GPU Terrain](../gpu-terrain.md) sight-line and point-horizon contributors use it for
`traversal: 'pyramid'`.

- **Levels.** Level `L` has blocks of `firstBlockSize·2^L` pixels (default 4), down to 1×1 or
  `levelCount`. Levels are packed into one float32 view per extremum, with offsets from
  `getGPURasterExtremaPyramidLayout`.
- **Footprint.** With `footprint: 'bilinear'` (the default), each cell also covers one extra
  pixel column and row. Every bilinear sample whose base pixel lies in the block is then bounded by
  that one cell. `'cell'` gives the plain block extrema.
- **Validity.** Invalid pixels are excluded. Cells with no valid pixel hold ∓FLT_MAX
  (`GPU_RASTER_EXTREMA_PYRAMID_EMPTY_*`).
- **Exactness.** The GPU output is bit-identical to a CPU build. Level 0 reduces from pixels and
  each higher level reduces the 2×2 cells below in a fixed order.

## Related pages

- [GPURaster operations reference](/docs/api-reference/experimental/gpu-raster/operations)
- [GPU Terrain](/docs/api-reference/experimental/gpu-terrain)
- [GPU Core](/docs/api-reference/experimental/gpu-core)
