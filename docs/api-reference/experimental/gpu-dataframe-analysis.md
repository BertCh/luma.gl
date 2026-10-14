---
title: GPU Dataframe analysis contributors
description: Column profiles, classification, group statistics, key joins, composite indicators, time windows, and pair statistics.
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Dataframe analysis contributors

<ExperimentalDocsTabs active="gpu-dataframe-analysis" />

## Overview

The `@luma.gl/experimental/gpu-dataframe` entry point exports analysis contributors for GPU-resident
columns: column profiles, quantiles and class breaks, color scales, group statistics, key joins,
composite scores, inequality measures, calendar buckets, time-window filters, temporal reductions,
and pair-statistics tools (variograms, correlograms, Ripley's K, point-pattern indices). Each is an
algorithm or workflow that declares resources and command nodes into a caller-owned
`GPUCommandGraph` through `getCommandNodes(graph)` (`GPUCommandNodeProducer`). It never compiles,
submits, encodes, or reads back.

## When to use it

Use these contributors for the fixed analyses that map and dashboard applications repeat every
frame, where inputs are packed graph views and results must stay on the GPU. Use the
[`GPUDataFrame` query API](./gpu-dataframe-operations) when the question is a general expression,
filter, or aggregation plan.

- `GPUGroupStatistics` groups by a 32- or 64-bit key and computes many statistics per group (moments,
  median, percentiles, mode, unique count). Use `GPUDataFrameGroupedAggregationQuery` for dense
  categorical keys and standard aggregations.
- `GPUKeyJoin` joins a left table to a right table by key and supports 1:n aggregates and 64-bit
  keys. Use `GPUDataFrameJoinQuery` for a unique right-side key lookup between dataframes.
- `GPUColumnProfile` fuses count, nulls, moments, histogram, distinct estimate, and top categories
  into one statistics panel. Use `GPUDataFrameHistogramQuery` when you only need an explicit
  histogram.
- `GPUTimeWindowFilter` publishes compacted IDs, masks, and fade weights for a window that moves
  every frame, including exact `Int64` time words. Express a fixed time predicate as a dataframe
  filter expression instead.

## Quick start

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUParameterBuffer,
  GPUTimeWindowFilter,
  getGPUTimeWindowFilterParameterValues,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'time'});
const window = new GPUParameterBuffer(device, {
  id: 'time-window',
  format: 'float32',
  length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
});

// `timestamps` and `output` are graph views created on `graph`.
graph.add(new GPUTimeWindowFilter({timestamps, window: window.importToGraph(graph), output}));
const compiled = await graph.compileAsync(); // once

// every frame: rewrite parameters, encode the same compiled graph
window.write(getGPUTimeWindowFilterParameterValues({start: now - 60, end: now}));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

`encode()` only records; submit with `device.submit()`. [Running contributors](./gpu-spatial-analysis.md#running-contributors)
covers `compileAsync()`, rebuilding without a stall, submission and why `parameters` is `undefined`.

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

## `GPUTimeWindowFilter`

Filters timestamped rows (instants) or time intervals (trail segments) against a per-frame window
and publishes stable compacted IDs, a clamped count, and optional mask, fade weights, segment clip
fractions, per-track visible counts, and an indirect `instanceCount`. All outputs live in one
`output` prop (`GPUTimeWindowFilterOutput`: `GPUCompactOutput` plus `outputMask`, `fadeWeights`,
`clipFractions`, `trackVisibleCounts`, `drawInstanceCount`); `trackIds` is an input. The per-frame settings object is `GPUTimeWindowFilterSettings` (`GPUTimeWindowFilterWordSettings` for Int64 words), packed by `getGPUTimeWindowFilterParameterValues` or `getGPUTimeWindowFilterWordParameterValues`. Timestamps come in three
forms, chosen by the view format:

- `float32` relative to an application epoch, the cheapest path;
- `float32` plus `timestampsLow` (double-single, from `splitTimestamps`) for inputs that arrive as
  float64;
- exact `Int64` words: a `uint32x2` view holding each value's little-endian `(low, high)` words,
  for example Arrow `Int64` or `Timestamp` epoch milliseconds uploaded without a CPU pass
  (`getInt64TimeWords`, or `makeArrowTemporalWordGPUVector` in `@luma.gl/arrow`). The window is
  then a `uint32` parameter buffer written with `getGPUTimeWindowFilterWordParameterValues`, whose start
  and end are a `number` (with a sub-unit fraction for a smooth playhead) or a `bigint`. Kernels
  order times lexicographically on the words and subtract with a borrow before converting to
  float32, so acceptance is exact for every `Int64` value, including an `INT64_MIN`/`INT64_MAX`
  "unbounded" window over pre-1970 rows, with no epoch to choose; fade and clip differences have the
  right sign for any two values and are exact below 2^24 units.
- The window is inclusive. A reversed window (`end < start`) accepts nothing, interval rows that
  span it included, and NaN timestamps are never accepted (edge-policy table
  `TIME_WINDOW_EDGE_POLICY` in the tests).

### Sorted range mode

When the instants are sorted by time (the usual layout of a recorded track or event log), set
`sorted: true`. The accepted rows are then one contiguous range, found by two binary searches in a
single one-thread kernel: O(log n) per frame for any row count, with no mask pass, scan or
compaction. The range is written to `output.rowRange` (`[firstRow, rowCount]`, `rowCount` unclamped),
`output.drawFirstInstance` (the draw record's `firstInstance`, so the instance index is the row
index) and the usual status words and `output.drawInstanceCount`; `output.ids` receives only its
first `min(count, ids.length)` entries (`sourceIds` if given, else row indices) from a kernel that
returns early past the count. A caller that reads only the range passes a small `ids` view. Acceptance
uses the same comparison helpers as the mask path (double-single `timestampsLow` and Int64 words
included), so the rows are exactly those the mask path accepts.

- Precondition: `timestamps` (with `timestampsLow`) is non-decreasing in row order and has no NaN. The
  default path does not scan for this. Set `validateOrder: true` with `output.orderViolations` (one
  `uint32`) to count adjacent descents on every encoding; it is an O(n) pass, as expensive as the mask
  path, so use it while developing or after an upload, not per frame.
- Instants only: `endTimestamps`, `additionalPredicates`, `trackIds`, `output.trackVisibleCounts` and
  `output.clipFractions` are refused, and views must be single packed (unchunked).
- `output.outputMask` and `output.fadeWeights` are still allowed and still cost O(n), so leave them
  out when scrubbing very large data.
- A reversed window (`end < start`) gives an empty range.

```ts
graph.add(new GPUTimeWindowFilter({
  sorted: true,
  timestamps: sortedInstants,
  window: window.importToGraph(graph),
  output: {...compactOutput, rowRange, drawInstanceCount}
}));
```

```ts
graph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,
  window: window.importToGraph(graph),
  // `output` is a `GPUTimeWindowFilterOutput`: the compact list plus the optional views.
  output: {
    ...compactOutput, // ids, status: {count, overflow, optional requiredCount}
    fadeWeights, clipFractions,
    drawInstanceCount: graph.importGPUData('trail-count', drawCommands.getInstanceCountData(0))
  }
}));
window.write(getGPUTimeWindowFilterParameterValues({start: now - trailLength, end: now, startFadeDuration: trailLength}));
```

```ts
import {makeArrowTemporalWordGPUVector} from '@luma.gl/arrow';

const window = new GPUParameterBuffer(device, {
  id: 'window', format: 'uint32', length: GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH
});
const {vector: departures} = makeArrowTemporalWordGPUVector(device, table.getChild('departure')!);
graph.add(new GPUTimeWindowFilter({
  timestamps: graph.importGPUVector('departures', departures),
  window: window.importToGraph(graph), output
}));
window.write(getGPUTimeWindowFilterWordParameterValues({start: playheadMs - 3_600_000, end: playheadMs}));
```

{/* parameters: GPUTimeWindowFilter generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUTimeWindowFilter` parameters

- Packer: `getGPUTimeWindowFilterParameterValues(settings)`, writes `GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH` (8) `float32` values.
- Packer: `getGPUTimeWindowFilterWordParameterValues(settings)`, writes `GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH` (8) `uint32` values.

Settings of `getGPUTimeWindowFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number` | Inclusive window start. |
| `end` | `number` | Inclusive window end. A window with `end < start` accepts nothing. |
| `startFadeDuration` (optional) | `number` | Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. |
| `endFadeDuration` (optional) | `number` | Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. |

Settings of `getGPUTimeWindowFilterWordParameterValues`:

| Field | Type | Description |
|---|---|---|
| `start` | `number \| bigint` | Inclusive window start. |
| `end` | `number \| bigint` | Inclusive window end. A window with `end < start` accepts nothing. |
| `startFadeDuration` (optional) | `number` | Fade weight ramps 0 to 1 over `[start, start + startFadeDuration]`. 0 (default) disables the ramp. |
| `endFadeDuration` (optional) | `number` | Fade weight ramps 1 to 0 over `[end - endFadeDuration, end]`. 0 (default) disables the ramp. |

{/* /parameters */}

## `GPUColumnQuantiles`, `GPUClassBreaks`, `GPUColorScale`, and `GPUBivariateClassification`

Choropleth classification on the GPU, every frame, with no readback: exact quantiles and
percentile filters, class breaks, and colour scales that write a packed `rgba8` column a deck.gl
layer can bind as an attribute. Each output can feed the next contributor as a graph view, so a brush
or filter change recolours the map without a CPU round trip.

```ts
graph.add(new GPUColumnQuantiles({
  values, mask, quantileCount: 3,
  parameters: quantileParameters.importToGraph(graph),
  output: {quantiles, validCount, outputMask, filterBounds} // outputMask may be a transient
}));
graph.add(new GPUClassBreaks({
  values, mask: outputMask, maximumClassCount: 9,
  methods: ['quantile', 'natural-breaks', 'head-tail'], // default: every method
  parameters: breakParameters.importToGraph(graph),
  output: {breaks, classCount, classCounts}
}));
graph.add(new GPUColorScale({
  values, mask: outputMask, domain: breaks, domainCount: classCount, palette,
  parameters: scaleParameters.importToGraph(graph),
  maximumDomainCount: 10, maximumPaletteCount: 9,
  output: {colors, classIndices, classCounts}
}));
quantileParameters.write(getGPUColumnQuantilesParameterValues({quantiles: [0.25, 0.5, 0.75], filterRange: [0.05, 0.95]}));
breakParameters.write(getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 7}, 9));
scaleParameters.write(getGPUColorScaleParameterValues({scale: 'threshold', domainCount: 8, paletteCount: 9}));
// Any of these writes takes effect on the next encoding, with no recompile.
```

- Class rule shared by all four contributors: breaks are edges `e[0..k]` (`e[0]` the smallest and
  `e[k]` the largest finite value). A value is in class `i` when `i` inner edges `e[1..k-1]` are at
  or below it (d3 `scaleThreshold`/`bisectRight`), compared on order-preserving `u32` keys, so
  `-0` sorts below `+0`. Values outside `[e[0], e[k]]` clamp to the end classes. NaN and masked
  rows are skipped (class `0xffffffff`, the no-data colour). Colours are `r | g << 8 | b << 16 | a << 24`.
- `GPUColumnQuantiles`: exact order statistics by four 8-bit radix-select passes over the ordered
  keys (no sort): up to 64 per-frame probabilities, `'lower' | 'higher' | 'nearest' | 'linear' |
  'midpoint'` (numpy names; `'linear'` is d3 and R-7). The rank is `fround(fround(n - 1) * p)` and
  every interpolation step is rounded separately (no fused multiply-add), so results match a
  sorted CPU column bit for bit. `filterRange` gives deck.gl/kepler
  lower/upper percentile filtering: bounds `x[floor(n * lower)]` and `x[ceil(n * upper) - 1]`, and a
  `outputMask` of the rows inside them. At most 2^24 rows. About 1.8 ms at 1M rows and 3.5 ms at
  4M rows for 9 quantiles plus the filter (Apple M-series, headless Chromium).
- One order-statistic policy for `GPUColumnQuantiles`, the `GPUClassBreaks` quantile and box-plot
  methods, and the `GPUGroupStatistics` medians and percentiles: the sample is the finite unmasked
  values (NaN and ±Infinity are missing), `-0` equals `+0`, and a probability that is NaN or outside
  `[0, 1]`, or an empty sample, gives NaN. The result is `numpy.quantile(x[numpy.isfinite(x)], p)`;
  numpy itself keeps infinities and raises on an invalid `p`. When `x[high] - x[low]` overflows f32
  the convex form `x[low] (1 - t) + x[high] t` keeps the result finite.
- `GPUClassBreaks`: per-frame `method` and `classCount` among the compiled `methods`:
  `'equal-interval'`, `'quantile'` (exact linear quantiles at `i / k`), `'standard-deviation'`
  (classes of `interval` sample standard deviations centred on the mean, as mapclassify `StdMean`
  and R classInt `'sd'`), `'head-tail'` (Jiang's
  breaks with a per-frame head ratio, default 0.4; 1 gives mapclassify's rule), `'box-plot'`
  (six classes, per-frame hinge), `'maximum-breaks'` (midpoints of the `k - 1` largest gaps
  between distinct sorted values), `'natural-breaks'` (Fisher-Jenks, see below), and
  `'custom'`. Head/tail and maximum breaks may return fewer classes; `classCount` reports `k`
  (0 with no finite values, 1 when they are all equal). Infinities are missing for every edge, so
  `e[0]` and `e[k]` bound the inner edges. Kernels of unselected methods return at
  once, except the two radix sorts of `'maximum-breaks'`, which run every frame when compiled.
  Natural-breaks layers and refinement rounds and head/tail rounds read their index from a GPU
  state word and are dispatched indirectly, so the pipeline count does not grow with
  `maximumClassCount` and finished rounds launch no workgroups. Moments use a fixed-order Chan merge, so every output is bitwise
  reproducible. About 3 to 4 ms per frame at 1M rows with six methods compiled. Every closed-form
  method is checked against mapclassify 2.11 and natural breaks against the exact Fisher-Jenks
  partition (an f64 dynamic program, agreeing with jenkspy 0.4.1) by the committed generator
  `test/oracles/dataframe-class-breaks.py`.
  - `'natural-breaks'` finds the partition into `k` classes of least total within-class squared
    deviation. A coarse dynamic program over `naturalBreaksBinCount` equal-width bins of
    `[min, max]` (2 to 2048, default 2048) sums count, position and squared position of each bin as
    exact 64-bit integers, so the class cost is evaluated without the cancellation of `S2 - S1^2 / n`
    in f32 and without a bin-by-bin cost table. A layer of the dynamic program is one workgroup that
    exploits the monotone split points of the cost (the quadrangle inequality) over `log2(bins)`
    barrier-separated levels: `O(k B log B)` work in all, against `O(k B^2)` before. The bins cannot
    see structure finer than `range / bins`, so the breaks are then refined on the rows, and no
    longer snap to bin edges: red-black coordinate descent in which each break, with its two
    neighbours fixed, takes the best split of the values in its two adjacent classes (odd breaks and
    then even breaks, so a pass never moves two interacting breaks and the cost only falls). The
    first window of a break covers the data extent of its two classes, so a heavy outlier cannot hide
    the structure, and the window then shrinks around the chosen cell boundary until a cell is
    below an f32 ulp. If empty classes show that the bins collapsed the most populated class (one
    extreme outlier puts the whole bulk into one bin), the coarse stage runs again over the range of
    that class with the classes the empty ones would have held. Rounds end once no cut moved with
    every break zoomed twice, at most eight; a column usually settles in two rounds, that is five
    passes over the rows (one for the class extents, two per round). Every inner edge `e[i]` is an
    actual data value, the smallest value of class `i`; mapclassify and jenkspy report the
    largest value of each class (`b[i-1]`), so `e[i]` is the smallest data value above `b[i-1]`
    and the classes are the same. A class count above the number of distinct values (which
    mapclassify and jenkspy refuse) yields one class per distinct value, like `'maximum-breaks'`;
    the produced count is `classCount`.
  - Accuracy: on the 34 reference cases (10 columns from lognormal and cluster mixtures to heavy
    ties, an extreme outlier of 1e6 times the spread, an outlier pair and a Pareto tail, `k` from 2
    to 10) 32 equal the exact partition, bit for bit in the edges. The other two stop in a
    neighbouring local optimum of the coordinate descent (a goodness of variance fit within 2e-6
    of the optimum; one of them is the partition mapclassify 2.11 itself returns, which is not
    optimal). On random well-scaled columns about 1 in 300 does. On 180 random heavy-tailed
    columns (Pareto tails, outliers, clusters beside an outlier) 179 were exact and the other was
    within a fit of 4e-8. Structure narrower than a bin that is not the most populated class (for
    example several clusters in one bin beside an outlier, when a lighter class also collapses)
    keeps the coarse allocation and is refined locally only.
    The previous implementation snapped breaks to bin edges (an error up to `range / 1024`) and
    cost `O(k B^2)` for the cost table and the layers.
- `GPUColorScale`: `linear`, `sqrt`, `pow`, `log` (kepler's `logFloor` for `v <= 0`), `symlog`,
  `quantize`, `threshold`/`quantile` (binary search over the active edges), and `ordinal` (a
  `uint32` code column). Scale, domain, palette, `clamp`, and `'step'`/`'linear'` interpolation are
  per frame. A multi-stop domain with as many stops as palette entries is piecewise linear, like a
  d3 multi-stop scale. A `domainCount` view (for example `GPUClassBreaks.classCount`) holds a
  class count `k`, and the domain then has `k + 1` edges. `'step'` colours are exact; `'linear'`
  blends round half up (within 1 per channel of a double-precision reference).
- `GPUBivariateClassification`: two columns and two edge views give `classId = yClass *
  classCountX + xClass` (up to 16 per axis), a colour from an `n x n` palette, per-class counts,
  and optional value-by-alpha (`alphaValues` mapped through a per-frame `[low, high]` to
  `[minimumAlpha, 1]` times the palette alpha).
- Integer columns: `GPUClassBreaks.values` accepts `uint32` and `sint32` in addition to `float32`. One
  internal pass converts the column to `float32` (exact to 2^24) and every method behaves as for that
  float column. `GPUColorScale.values` accepts `sint32` (always numeric) and `uint32` with
  `integerValues: 'numeric'`, which reads the integers as numbers and applies the `scale` parameter; the
  default `'ordinal'` keeps `uint32` as category codes. The value format is topology (fixed at compile
  time). This is what lets a count column (for example dense `GPUGroupStatistics` `counts`) feed a
  choropleth without a float copy.
- Per-frame settings are the CPU objects `GPUColumnQuantilesSettings`, `GPUClassBreaksSettings`,
  `GPUColorScaleSettings`, and `GPUBivariateClassificationSettings`, packed by the matching
  `getGPU<Class>ParameterValues(settings, target?)`.
- Inputs must be single packed views; chunked vectors are rejected.

{/* parameters: GPUBivariateClassification generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUBivariateClassification` parameters

- Packer: `getGPUBivariateClassificationParameterValues(settings)`, writes `GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUBivariateClassificationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `classCountX` | `number` | Active classes along X, `1..maximumClassCount`; `breaksX` holds `classCountX + 1` edges. |
| `classCountY` | `number` | Active classes along Y, `1..maximumClassCount`; `breaksY` holds `classCountY + 1` edges. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `valueByAlpha` (optional) | `GPUBivariateValueByAlpha` | Enables value-by-alpha. Omit to leave the palette alpha unchanged. |

{/* /parameters */}

{/* parameters: GPUClassBreaks generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUClassBreaks` parameters

- Packer: `getGPUClassBreaksParameterValues(settings, maximumClassCount)`, writes `getGPUClassBreaksParameterLength(...)` `float32` values.
  - `maximumClassCount` (`number`): Compile-time class capacity of the contributor.

Settings of `getGPUClassBreaksParameterValues`:

| Field | Type | Description |
|---|---|---|
| `method` | `GPUClassBreaksMethod` | Classification method. It must be one of the contributor's compiled `methods`. |
| `classCount` (optional) | `number` | Requested class count `k`. Ignored by `'box-plot'` (always 6) and `'custom'` (one less than the number of edges). For `'head-tail'` and `'maximum-breaks'` it is an upper bound: fewer classes are produced when the data runs out of heads or distinct gaps. |
| `standardDeviationInterval` (optional) | `number` | `'standard-deviation'`: width of one class in standard deviations. Default 1. |
| `headTailRatio` (optional) | `number` | `'head-tail'`: largest head fraction that keeps splitting. Default 0.4 (Jiang's 40% rule). Use 1 for mapclassify's rule, which splits until a head has a single distinct value. |
| `boxPlotHinge` (optional) | `number` | `'box-plot'`: whisker length in interquartile ranges. Default 1.5. |
| `customEdges` (optional) | `ArrayLike<number>` | `'custom'`: ascending class edges `e[0..k]`, at most `maximumClassCount + 1` of them. |

{/* /parameters */}

{/* parameters: GPUColorScale generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUColorScale` parameters

- Packer: `getGPUColorScaleParameterValues(settings)`, writes `GPU_COLOR_SCALE_PARAMETER_LENGTH` (9) `float32` values.

Settings of `getGPUColorScaleParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scale` | `GPUColorScaleType` | Scale type. |
| `domainCount` | `number` | Number of active domain entries, `1..maximumDomainCount`. Ignored when the contributor was given a `domainCount` view, which then supplies a CLASS count `k` and the domain has `k + 1` edges. |
| `paletteCount` | `number` | Number of active palette entries, `1..maximumPaletteCount`. |
| `interpolation` (optional) | `GPUColorScaleInterpolation` | `'step'` picks `palette[class]` exactly; `'linear'` blends adjacent entries per channel. Default `'step'`. |
| `clamp` (optional) | `boolean` | Clamp out-of-domain values to the first or last colour instead of no-data. Default `false`. |
| `noDataColor` (optional) | `number` | Packed `rgba8` colour (`r \| g << 8 \| b << 16 \| a << 24`) for no-data rows. Default `0`. |
| `logFloor` (optional) | `number` | Value that replaces `v <= 0` (and non-positive domain entries) on the `log` scale, as kepler does with a small positive floor. `NaN` treats such rows as no-data. Default `1e-5`. |
| `exponent` (optional) | `number` | Exponent of the `pow` scale. Default `1`. |

{/* /parameters */}

{/* parameters: GPUColumnQuantiles generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUColumnQuantiles` parameters

- Packer: `getGPUColumnQuantilesParameterValues(settings)`, writes `getGPUColumnQuantilesParameterLength(...)` `float32` values.

Settings of `getGPUColumnQuantilesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `quantiles` | `ArrayLike<number>` | Probabilities in `[0, 1]`; others (and NaN) produce a NaN quantile. |
| `interpolation` (optional) | `GPUColumnQuantileInterpolation` | Defaults to `'linear'`. |
| `filterRange` (optional) | `[number, number]` | Percentile filter as fractions `[lower, upper]` in `[0, 1]`. Defaults to `[0, 1]`. |
| `quantileCount` (optional) | `number` | Number of probability slots. Defaults to `quantiles.length`. Unused slots become NaN. |

{/* /parameters */}

## `GPUColumnProfile`

A dataset statistics panel (kepler.gl / Foursquare Studio field summaries) for several columns of
one table, recomputed on the GPU every frame so it follows filters and brushes without readback
of the rows.

```ts
graph.add(new GPUColumnProfile({
  columns: [
    {values: fare}, // float32, numeric
    {values: vendorCodes, kind: 'category', categoryCount: 500} // uint32 dictionary codes
  ],
  mask, // optional uint32, shared by every column
  parameters: domainParameters.importToGraph(graph), // optional histogram domains
  histogramBinCount: 64,
  hyperLogLogPrecision: 12, // 2^12 registers per column
  topCategoryCount: 10,
  output: {statistics, counts, histograms, hyperLogLogRegisters, topCategories, topCategoryCounts}
}));
domainParameters.write(getGPUColumnProfileParameterValues([[0, 100], [NaN, NaN]])); // NaN = auto
```

- `statistics` holds `GPU_COLUMN_PROFILE_STATISTIC_COUNT` (11) floats per column, indexed by
  `GPU_COLUMN_PROFILE_STATISTIC`: count, nullCount, minimum, maximum, sum, mean, variance
  (population), sampleVariance, standardDeviation, distinctEstimate, overflowCount. `counts`
  holds the exact `uint32` count and null count per column.
- NaN (numeric) or `0xffffffff` (category) rows are nulls. ±Infinity counts toward count, minimum
  and maximum but is excluded from the moments. With no rows, the float statistics are NaN and the
  distinct estimate is 0. Category columns report their smallest and largest code and NaN moments.
- Moments are a fixed-order two-level Chan/Welford reduction (workgroup trees, then one
  fixed-order merge), with no float atomics: bitwise identical across runs and stable for large
  offsets (values near 1e6 with a spread of 100).
- Histograms: equal-width bins over a per-frame `[minimum, maximum]` per column (a
  `GPUColumnProfileDomain`: a pair or `{minimum, maximum}`), or the frame's finite minimum and
  maximum when a bound is NaN; the last bin includes the maximum and values outside are not counted.
  Bin assignment uses correctly rounded products, so it is identical on every adapter. Category
  columns get per-code counts in their first bins.
- Distinct count: HyperLogLog over a murmur3 `fmix32` hash of the value bits (`-0` folded to
  `+0`) with `atomicMax` registers and linear-counting small-range correction. Registers are
  exact, and the estimate is within about `1.04 / sqrt(2^p)` relative error.
- Top categories: exact per-code counts with `uint32` atomics, then the top `topCategoryCount`
  ordered by count (descending) and code (ascending). Codes at or above `categoryCount` are
  counted in `overflowCount`.
- At most 16 columns and 2^24 rows, and at most `getGPUColumnProfileMaximumCategoryTotal(device)` (2^25 on a default device)
  `categoryCount` summed over the category columns (one 4-byte slot per category in a 128 MiB binding).
  Inputs must be single packed views. The per-column kernels read
  the current column from a GPU cursor word, so the pipeline count does not grow with the column
  count.

{/* parameters: GPUColumnProfile generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUColumnProfile` parameters

- Packer: `getGPUColumnProfileParameterValues(domains, columnCount?)`, writes `getGPUColumnProfileParameterLength(...)` `float32` values.
  - `domains` (`readonly (GPUColumnProfileDomain | undefined)[]`): One entry per column, `undefined` for fully automatic.
  - `columnCount` (`number`): Number of columns. Defaults to `domains.length`.

{/* /parameters */}

## `GPUGroupStatistics`

Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and value column on the GPU: count, sum, mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles, mode, unique count and a per-row group z-score. Use it for "aggregate this column by that key" panels, choropleth values keyed by a region id or an H3 or Quadbin cell, and per-group outlier styling.

```ts
const fractions = new GPUParameterBuffer(device, {id: 'fractions', format: 'float32', length: 3});
graph.add(
  new GPUGroupStatistics({
    keys, // uint32, or uint32x2 (low, high) words; all-ones is "no key"
    mask, // optional uint32
    columns: [
      {
        values, // float32 per row; non-finite rows are excluded for this column only
        statistics: ['count', 'mean', 'variance', 'median', 'percentiles', 'mode', 'zScore'],
        output: {counts, means, variances, medians, percentiles, modes, zScores}
      }
    ],
    variance: 'sample', // or 'population'
    percentiles: fractions.importToGraph(graph),
    output: {keys: groupKeys, counts: groupCounts, status: {count, overflow, requiredCount}} // GPUGroupStatisticsOutput
  })
);
fractions.write(Float32Array.of(0.1, 0.5, 0.9)); // no recompile
```

- Output groups are ascending by key, capacity is `output.keys.length`, at most `getGPUGroupStatisticsMaximumGroupCount(device)` (2^23 on a default device; the 16-byte moment sums per group fit one 128 MiB binding) and, with `percentiles`, `capacity * P` at most `getGPUGroupStatisticsMaximumPercentileTotal(device)` (2^25 on a default device). When more groups exist, the smallest keys are kept, `status.overflow` is 1 and `status.requiredCount` is the unclamped number. Tail rows hold the all-ones key, count 0, sums 0 and NaN for every other float statistic.
- `output.counts` counts valid rows (mask set, key valid); each column's `counts` counts finite values. Zero to four columns; each column lists the statistics it needs (a compile-time set) and supplies exactly the matching output views: `counts`, `sums` (exact signed 64-bit fixed point, `uint32x2`, scale `sumScale`, default 65536) and/or `sumValues` for `sum`, `means`, `minimums`, `maximums`, `variances`, `standardDeviations`, `skewness`, `kurtosis`, `medians`, `percentiles` (`capacity * P`, group-major), `modes`, `uniqueCounts`, and `zScores` (one row per source row).
- Exactness: counts, fixed-point sums, minimum, maximum, median, percentiles, mode and unique counts are bit-identical to a CPU reference. Percentiles use the numpy `linear` rule (`h = (n - 1) p`, `lo + (hi - lo) * (h - floor h)` in f32) and the order-statistic policy shared with `GPUColumnQuantiles`, with `p` read per frame; a fraction that is NaN or outside [0, 1] gives NaN (it is no longer clamped). Mode is the longest run of equal values (ties: smallest value). `-0` is treated as `+0`; NaN and infinities are excluded.
- Moments are corrected two-pass: `d = v - mean` is summed in a fixed order (one 256-thread workgroup per group, strided partials and a binary tree), so a second encode is bitwise identical, and central moments about the true mean come from the shifted power sums. Measured agreement with a float64 reference on mixed data is about 1e-7 for mean and variance, 2e-6 for skewness and kurtosis, 7e-6 for z-scores (relative). Skewness is Fisher-Pearson `g1`, kurtosis is excess `g2`; both are NaN for constant groups, a group whose variance is under 2^-19 of its shifted sum of squares is treated as constant (the fixed-point mean carries 2^-17 absolute error). Sample variance of one value is NaN; z-score is 0 when the deviation is 0 or a sample group has one value, NaN for masked, invalid-key, non-finite or capacity-dropped rows.
- Dense mode: `keyCount` (compile-time) requires `uint32` keys in `[0, keyCount)`. `output.keys.length` and every per-group output hold exactly `keyCount` rows, and row `k` is key `k` (`output.keys[k] = k`, `output.status.count` and `status.requiredCount` are `keyCount`, `status.overflow` is 0). Keys with no rows keep a row: `counts` 0, `sums` and `uniqueCounts` 0, and NaN for mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles and mode. Keys outside the range and the reserved all-ones key are skipped like masked rows and are not reported as overflow. It costs one binary search per key instead of a group scan, and suits region IDs or other small dense key spaces (the choropleth recipe uses it so empty polygons keep a row). The default compact table of occupied keys is unchanged.
- Cost: one stable key sort (two 32-bit radix sorts for 64-bit keys); each column that asks for median, percentiles, mode or unique count adds a value sort plus a key re-sort. Inputs must be single packed views. Approximate distinct counts (HyperLogLog) are not included.

## `GPUKeyJoin`

Attribute join of a left table onto a right table by key, for example "join CSV statistics onto
boundaries". Keys are `uint32` or 64-bit `uint32x2` little-endian `(low, high)` words (the same
format on both sides). The all-ones key (`0xffffffff`, or both words all-ones) means "no key" and
never matches; masked rows (`mask == 0`) never match.

```ts
const join = new GPUKeyJoin({
  leftKeys, rightKeys,
  kind: 'left', // or 'inner'
  gather: [{column: rightPopulation, output: leftPopulation}],
  aggregates: [{operation: 'sum', column: rightIncome, output: leftIncomeSum}],
  output: {matched, rightRows, matchCounts, rightMatched}
});
graph.add(join);
```

Valid right rows are sorted by key with a stable radix sort (one pass per key word), so equal keys
stay in right-row order. Segment heads and a scan produce the unique right keys with row ranges;
each left row binary-searches them on the unsigned `(high, low)` order.

- `gather` copies a right column (`float32` or `uint32`, bit for bit) from the smallest matching
  right row. Unmatched rows get NaN or `0xffffffff`.
- `aggregates` run over all matching right rows: `count`, `sum`, `mean`, `minimum`, `maximum`.
  Sums are exact 64-bit fixed point (`sums` output, scale `sumScale`, default 65536); extremes use
  ordered-key atomics. Non-finite values are excluded per aggregate. Unmatched rows get 0 for
  `count` and NaN otherwise.
- `output.rightRows`, `matchCounts`, `matched` (usable as a crossfilter live mask) are left-aligned;
  `rightMatched` is right-aligned (1 when some valid left row has the same key).
- `kind: 'inner'` additionally compacts matched left row IDs, ascending, into `output.rows`
  (`status.count` clamped, `status.overflow`, optional `status.requiredCount`; ids past `status.count` are `0xffffffff`). Other
  outputs stay left-aligned. `'left'` rejects `output.rows`.

All results are integer or fixed-point, independent of thread order, and bitwise reproducible.
Input contents (keys, masks, values) are per-frame and need no rebuild; view lengths, key format,
`kind`, `sumScale` and the set of outputs are topology.

## `GPUVariogram`, `GPUSpatialCorrelogram`, `GPURipley`, `GPURipleyDistanceFunctions`, and `GPUPointPatternIndices`

Every contributor here accepts at most `getGPUPairStatisticsMaximumRowCount(device)` (11,184,810 on a default device, never above 2^24 - 1) `positions`
rows: the cell-sorted points table holds 12 bytes per row and must fit one 128 MiB storage binding.

Pair statistics over planar points share one deterministic pair-histogram pass: rows are sorted
into a per-frame lattice whose cells are at least the per-frame `maximumDistance` wide, one thread
per included row visits the 3x3 cell neighbourhood, and each pair adds integer amounts into
per-workgroup shared-memory 64-bit accumulators that are then added into global 64-bit
accumulators. Float terms are fixed point (a bounded term such as a distance maps its maximum to
2^24; the variogram and correlogram unit comes from the typical term, see below, never from the
maximum), so every result is exact or has a documented fixed-point bound, independent of
scheduling, and bitwise reproducible. A 64-bit sum that would wrap raises an `overflow` word
(`readPairOverflow`) instead of wrapping silently, and pair counts saturate at `0xffffffff`.
When `maximumDistance` covers the extent the pass is a plain all-pairs loop. A row is included when
its mask is nonzero, its value (if any) finite and its position finite and inside the per-frame
`bounds`. Parameters (`[minX, minY, maxX, maxY, maximumDistance, ...]`) are per-frame, packed from a
`GPU<Class>Settings` object (`GPUVariogramSettings`, `GPUSpatialCorrelogramSettings`,
`GPURipleySettings`, `GPURipleyDistanceFunctionsSettings`, `GPUPointPatternIndicesSettings`) by
`getGPU<Class>ParameterValues(settings)`; bin counts and the set of outputs are compile-time. Every
caller-owned output view goes in one `output` prop (`GPUVariogramOutput`,
`GPUSpatialCorrelogramOutput`, `GPURipleyOutput`, `GPURipleyEnvelopeOutput`,
`GPURipleyDistanceFunctionsOutput`, `GPUPointPatternIndicesOutput`), including the `overflow`
status word.

All five classes (and `GPURipleyEnvelope`) take an optional `spatialContext` and are planar only:
distances and window areas are Euclidean in the units of `positions`. Omitting it means native
planar coordinates; a `longitude-latitude` context throws at construction ("project positions to a
planar frame first"), because degrees give latitude-dependent distances, wrong areas and biased
indices. The resolved context is exposed as `spatialContext`. The window is always the `bounds`
rectangle: for an irregular study area the bounding box overstates the area and makes every
pattern look clustered (polygon windows are not implemented).

```ts
graph.add(
  new GPUVariogram({
    positions, values, parameters, gridSize: [64, 64], lagCount: 24, directionCount: 4,
    output: {semivariances, pairCounts, meanDistances}
  })
);
parameters.write(getGPUVariogramParameterValues({bounds, maximumDistance: 500, azimuthOffset: 0}));
// After readback: CPU weighted least squares.
const model = fitVariogramModel({distances, semivariances, pairCounts}, {model: 'spherical'});
```

- `GPUVariogram`: Matheron semivariance, pair count, mean lag distance and Cressie-Hawkins robust
  semivariance per lag and optional azimuth sector (`directionCount` equal sectors over `[0, pi)`
  from a per-frame `azimuthOffset`). Matheron precision: a term `(z_i - z_j)^2 / 2` is scaled by
  `2^24 / u` and rounded to an integer, where `u` is the robust variance `(IQR / 1.349)^2` of the
  included values (exact quartiles), clamped to at most the population variance, so the absolute
  error of a bin mean is at most `u * 2^-25` (never more than `variance * 2^-25`, itself below the
  former `(max - min)^2 * 2^-26`) plus f32 rounding of the term. The former unit
  `2^24 / (0.5 * (max - min)^2)` rounded every typical term to 0 once one outlier or sentinel
  stretched the range (-83% at an isolated outlier of 2e4 standard deviations on 4000 points,
  -100% at 1e5) with no status. Terms above 256 `u` (a difference beyond 22 standard deviations)
  are added exactly in a second 64-bit channel at `2^12` units, with relative error at most
  `2^-21`; the worst-case relative error of `gamma` per bin is `max(u * 2^-25 / gamma, 5e-7 +
  2^-24)`. Capacity is `2^40 u` for the first channel and `2^52 u` for the second. The optional
  `overflow` output is 1 when a sum, a clamped term (`2^51 u`) or a pair count (saturated at
  `0xffffffff`; previously the count wrapped modulo 2^32 and became the `N_h` weight of the fit)
  left its range. The Cressie-Hawkins channel scales with `sqrt(max - min)`. The pass uses 4
  channels per bin (5 with `robustSemivariances`), so `lagCount * directionCount` is at most 512
  (409 with the robust channel). Model fitting is CPU post-processing: `fitVariogramModel` runs on
  the CPU over the read-back bins, fits spherical, exponential or Gaussian models
  (practical-range convention) with Cressie, pair-count or uniform weights, and should be skipped
  when `output.status.overflow` is 1. The Matheron and Cressie-Hawkins bins, pair counts and mean distances are
  checked against scikit-gstat 1.0.24 (including an embedded 2e4 sentinel) by the committed
  generator `test/oracles/dataframe-variogram.py`, with tolerances derived from the bounds above.
- `GPUSpatialCorrelogram`: global Moran's I per distance band (`'cumulative'` bands as PySAL
  `DistanceBand` / ArcGIS Incremental Spatial Autocorrelation, or `'annulus'`), with `E[I]`,
  normality or randomization variance (per-frame), z, p, unordered pair counts, and
  `[firstPeakBand, maximumBand]`. Variances lost to f32 cancellation (for example a band holding
  every pair) are NaN. Pair products use the same typical-term unit as the variogram, `2^24 / u`
  with `u` the robust variance (the former `2^24 / max|c|^2` rounded every product to 0, so `I` was
  0, once one outlier or sentinel dominated), so `|I|` errs by at most `2^-25 * u / variance` plus
  `2^-21` of the contribution of products above 256 `u`, which are summed exactly in a second
  channel. The optional `overflow` output is 1 when a product sum, a clamped product or a band pair
  count (saturated at `0xffffffff`) left its range.
- `GPURipley`: K, L, L - r and an annulus pair-correlation estimate over `radiusCount` radii in a
  rectangular window, with `'none'`, `'border'` (reduced sample, spatstat `Kest` border estimator
  with `lambda = n / A`, so `K = A * numerator / (n * borderCount)`) or `'isotropic'` (Ripley 1977
  rectangle, weights capped at 100 like spatstat `edge.Ripley` `maxweight`) edge correction
  switchable per frame. Border `K` changed from `lambda = (n - 1) / A` to `n / A` to match spatstat,
  which lowers it by the factor `(n - 1) / n`. The isotropic rectangle formula is exact at every
  radius (checked against brute-force circle sampling), so there is no radius guard; the practical
  limits are variance and the weight cap, which biases `K` low once less than 1% of a circle is
  inside, so keep `maximumDistance` well below the shorter window side (spatstat's default `rmax` is
  a quarter of it). The pair correlation is an unsmoothed annulus difference and is noisy.
- `GPURipleyEnvelope`: Monte Carlo CSR envelopes of K, L or `L - r` (`statistic`, default
  `'l-minus-r'`), with `simulationCount` (at most 199) binomial CSR patterns of the observed
  included count `n`, drawn with Philox from the per-frame `seed` (`getGPURipleyParameterValues({..,
  seed})`, a uint32 stored as a word) and measured by a full `GPURipley` pass each, with the same window,
  radii and edge correction, without CPU readback. Outputs (`output`): `lower` / `upper` (pointwise: the
  `rank`-th smallest / largest simulated value, spatstat `envelope(nrank)`, not simultaneous),
  `globalLower` / `globalUpper` (Ripley 1981; Baddeley et al. 2014: `L(r) = r` plus or minus the
  `rank`-th largest `max_r |L_j(r) - r|`, a simultaneous test of level `rank / (simulationCount + 1)`),
  `theoretical`, and `statistics` `[n, validSimulationCount, observedMaximumDeviation,
  criticalMaximumDeviation, pValue, rank]` with the maximum absolute deviation (MAD) Monte Carlo
  p-value `(1 + #{T_j >= T_obs}) / (s + 1)` when `observedK` (an input, the observed `GPURipley` `output.k`) is given.
  Cost is `simulationCount + 1` times one K pass and about 25 graph nodes per simulation; the graph
  aliases transient buffers across simulations. The simulation index is a GPU counter word, not an
  override constant, so every simulation shares the same pipelines (an override still compiles one
  pipeline per value). Radii where an estimate is NaN are skipped per
  curve, so prefer `'isotropic'` or `'none'` for exact Monte Carlo validity. Cross-type K and
  inhomogeneous K (Kinhom) are not implemented.
- `GPURipleyDistanceFunctions`: the distance functions next to `GPURipley` (spatstat `Gest`, `Fest`,
  `Jest`). `G` is the nearest-neighbour distance function of the events, `F` the empty-space function
  over a `referenceGrid` lattice of reference locations, and `J = (1 - G) / (1 - F)` (below 1 is
  clustered, above 1 regular). `parameters` (`getGPURipleyDistanceFunctionsParameterValues`, from a `GPURipleyDistanceFunctionsSettings`) hold `bounds`,
  `maximumDistance` and `edgeCorrection` (`'border'`, the default, `'kaplan-meier'`, `'hanisch'` or
  `'none'`, switchable per frame without recompiling); `gridSize`, `referenceGrid` and `radiusCount` are
  compile-time. Outputs (`output.g`, `output.f`, `output.j`, `output.radii`; at least one of `g`, `f`, `j`) hold NaN where a value is undefined. Integer atomics
  keep it bitwise reproducible; the F pass sums each workgroup's reference locations in workgroup
  memory and adds one exact 64-bit partial per bin and workgroup. `'border'` is spatstat `rs`. `'kaplan-meier'` is spatstat `km`, with hazards
  binned on the radius grid, exact including the censoring of points that have no neighbor within
  `maximumDistance`. `'hanisch'` is spatstat `han` for G and Chiu-Stoyan style weights for F, with 14-bit
  fixed-point weights (relative error about 1e-4); a point with no neighbor within `maximumDistance` but a
  border distance of at least `maximumDistance` enters the denominator at `d = maximumDistance`, so G is
  slightly high unless every point has a neighbor in range. The corrections are checked against an f64
  oracle written from the spatstat definitions, not against spatstat itself, and K and L (`GPURipley`) have
  no Kaplan-Meier or Hanisch option. The pass uses 8 channels per radius (a `radiusCount` of 256 still fits
  the default 16 KB of workgroup memory exactly). F depends on the reference lattice, so keep it coprime with the
  window so no reference location sits exactly on a radius.
- `GPUPointPatternIndices`: exact nearest neighbour per row (ring search, ties to the smallest row),
  Clark-Evans `[n, observed, expected, R, standardError, z, pValue]`, and quadrat counts with
  `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom, pValue]`. The per-frame
  `clarkEvansCorrection` defaults to `'donnelly'` (Donnelly 1978 edge correction for the rectangular
  window: `expected = 0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance
  `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the perimeter; spatstat uses 0.0412 for the second
  constant). The plain infinite-plane index (`'none'`, `0.5 sqrt(A / n)`, `0.26136 sqrt(A) / n`,
  pointpats / ArcGIS) overstates clustering in small or elongated windows and is no longer the
  default. The per-frame `alternative` (default `'two-sided'`; `'greater'` tests for regularity in
  Clark-Evans and for clustering in the quadrat chi-square, `'lesser'` the opposite) sets the tail
  of both p-values. The quadrat p-value is exact (regularized incomplete gamma) up to 100 degrees
  of freedom and the Wilson-Hilferty approximation (absolute error below 2e-4) above; assume
  expected counts of about 5 or more per quadrat. `GPU_CLARK_EVANS_LENGTH` and
  `GPU_QUADRAT_STATISTICS_LENGTH` are now 7.

Measured on an Apple M3 Pro (headless Chromium), `GPUVariogram` with 32 lags including the cell
sort: 100k points with 54M pairs in range in 10 ms; 20k points all-pairs (200M pairs) in 12 ms.

{/* parameters: GPUPointPatternIndices generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUPointPatternIndices` parameters

- Packer: `getGPUPointPatternIndicesParameterValues(settings)`, writes `GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUPointPatternIndicesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the Clark-Evans index and the quadrats tile it. |
| `maximumDistance` | `number` | Size hint of the neighbor-search cells (cells are at least this wide). Results never depend on it, only speed: a value near the expected nearest-neighbor distance is fastest. |
| `clarkEvansCorrection` (optional) | `GPUClarkEvansCorrection` | Clark-Evans edge correction. Defaults to `'donnelly'` (Donnelly 1978: expected mean distance `0.5 sqrt(A / n) + (0.0514 + 0.041 / sqrt(n)) P / n`, variance `0.0703 A / n^2 + 0.037 P sqrt(A / n^5)`, `P` the window perimeter), because the plain expectation is biased toward apparent clustering for windows with a large perimeter. `'none'` gives the classic `0.5 sqrt(A / n)` with standard error `0.26136 sqrt(A) / n` (pointpats, ArcGIS). |
| `alternative` (optional) | `GPUInferenceAlternative` | Tail of the Clark-Evans z-test and the quadrat chi-square test. Defaults to `'two-sided'`. `'greater'` tests for a statistic above its complete-spatial-randomness expectation (Clark-Evans: regular; quadrat chi-square: clustered), `'lesser'` for one below it. The permutation-only tails `'folded'` and `'directed'` throw. |

{/* /parameters */}

{/* parameters: GPURipley generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURipley` parameters

- Packer: `getGPURipleyParameterValues(settings)`, writes `GPU_RIPLEY_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside. |
| `edgeCorrection` (optional) | `GPURipleyEdgeCorrection` | Edge-correction mode. Defaults to `'isotropic'`. |
| `seed` (optional) | `number` | Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly. |

{/* /parameters */}

{/* parameters: GPURipleyDistanceFunctions generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURipleyDistanceFunctions` parameters

- Packer: `getGPURipleyDistanceFunctionsParameterValues(settings)`, writes `GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyDistanceFunctionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` observation window. Events outside it, or with a non-finite coordinate, are excluded. The reference lattice of F covers this rectangle. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. |
| `edgeCorrection` (optional) | `GPURipleyDistanceEdgeCorrection` | `'border'` (default) divides by the points farther from the window edge than `r`; `'kaplan-meier'` and `'hanisch'` use every point (see `GPURipleyDistanceFunctions`). |

{/* /parameters */}

{/* parameters: GPURipleyEnvelope generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURipleyEnvelope` parameters

- Packer: `getGPURipleyParameterValues(settings)`, writes `GPU_RIPLEY_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURipleyParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` observation window. Rows outside it, or with a non-finite coordinate, are excluded. Its area is the study area of the estimators. |
| `maximumDistance` | `number` | Largest radius: radius `b` is `maximumDistance * (b + 1) / radiusCount`. Keep it well below the shorter window side (spatstat's default is a quarter): K is unbiased at any radius, but its variance grows and the isotropic weight cap biases it low once a circle is mostly outside. |
| `edgeCorrection` (optional) | `GPURipleyEdgeCorrection` | Edge-correction mode. Defaults to `'isotropic'`. |
| `seed` (optional) | `number` | Monte Carlo seed of `GPURipleyEnvelope`, a uint32. `GPURipley` ignores it. Defaults to 0. The same seed reproduces the same simulated patterns exactly. |

{/* /parameters */}

{/* parameters: GPUSpatialCorrelogram generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpatialCorrelogram` parameters

- Packer: `getGPUSpatialCorrelogramParameterValues(settings)`, writes `GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialCorrelogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. |
| `maximumDistance` | `number` | Upper distance of the last band. Band `b` ends at `maximumDistance * (b + 1) / bandCount`. |
| `varianceAssumption` (optional) | `GPUSpatialCorrelogramVarianceAssumption` | Variance assumption of the z-scores and p-values. Defaults to `'randomization'`. |

{/* /parameters */}

{/* parameters: GPUVariogram generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUVariogram` parameters

- Packer: `getGPUVariogramParameterValues(settings)`, writes `GPU_VARIOGRAM_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUVariogramParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` extent. Rows outside it, or with a non-finite coordinate or value, are excluded. The extent also sizes the neighbor-search lattice. |
| `maximumDistance` | `number` | Largest lag: pairs farther apart are ignored. Lag bins split `[0, maximumDistance]`. |
| `azimuthOffset` (optional) | `number` | Start of the first direction sector, in radians counterclockwise from +x. Sector `s` holds pairs whose undirected direction (taken modulo pi) minus this offset, again modulo pi, lies in `[s * pi / directionCount, (s + 1) * pi / directionCount)`. Ignored when `directionCount` is 1. Defaults to 0. |

{/* /parameters */}

## `GPUCompositeScore`

`GPUCompositeScore` builds a composite indicator per row from up to 16 indicator columns
(row-major `row * indicatorCount + column`), as CARTO's composite-indicator procedures and ArcGIS
index builders do: scale each column, flip "higher is worse" indicators, and combine them. Weights,
directions, the scaler and the aggregation live in a parameter buffer, so weight sliders recolor
every row without rebuilding the graph.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'composite-parameters',
  format: 'float32',
  length: GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
});
graph.add(
  new GPUCompositeScore({
    indicators, // rowCount * 4 float32
    indicatorCount: 4,
    parameters: parameters.importToGraph(graph),
    enableRank: true,
    enablePrincipalComponent: true,
    output: {score, columnStatistics, loadings, principalComponentSummary}
  })
);
parameters.write(
  getGPUCompositeScoreParameterValues({
    scaler: 'rank',
    aggregation: 'weighted-sum',
    weights: [1, 2, 0.5, 1],
    directions: [1, -1, 1, 1]
  })
);
```

- A row is included when its mask is non-zero and every indicator is finite; excluded rows write
  NaN and do not enter the statistics.
- Scalers: `'min-max'` (`(x - min) / (max - min)`, 0 for a constant column), `'z-score'`
  (population standard deviation, 0 for a constant column), `'rank'` (percentile rank
  `r / (n - 1)` with tied ranks averaged, exact; needs `enableRank`). A negative direction flips the
  scaled value (`1 - s`, or `-s` for z-scores).
- Aggregations: `'weighted-sum'` (`sum(w * s) / sum(|w|)`), `'weighted-geometric-mean'`
  (`exp(sum(w * ln(s + epsilon)) / sum(w))`; pair it with min-max or rank; the packer refuses a
  negative weight here, use a direction to flip an indicator),
  `'principal-component'` (projection of direction-adjusted z-scores on the first eigenvector of
  their correlation matrix; needs `enablePrincipalComponent`; weights are ignored).
- The first component comes from 64 fixed power-iteration steps from `1 / sqrt(d)` in one thread,
  signed so the loadings sum to a non-negative value. `principalComponentSummary` holds
  `[eigenvalue, explainedVarianceRatio, residual]`; the ratio divides by the number of
  non-constant columns.
- Outputs: `score` (required), `scaled`, `columnStatistics` (`[minimum, maximum, mean, standardDeviation]` per column),
  `loadings`, `principalComponentSummary`.
- Determinism: min and max use u32 order-key atomics; means and centered (co)variances are
  two-pass fixed-order sums over 256-row tiles, merged by one 256-lane workgroup per output
  (strided tiles, then a fixed binary tree); ranks use one stable radix
  sort per column plus a binary search per tie run. No float atomics.
- `compile-time`: row count, `indicatorCount`, `enableRank`, `enablePrincipalComponent`, the mask
  and which outputs exist. `per frame`: scaler, aggregation, weights, directions, epsilon.

{/* parameters: GPUCompositeScore generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCompositeScore` parameters

- Packer: `getGPUCompositeScoreParameterValues(settings)`, writes `GPU_COMPOSITE_SCORE_PARAMETER_LENGTH` `float32` values.

Settings of `getGPUCompositeScoreParameterValues`:

| Field | Type | Description |
|---|---|---|
| `scaler` (optional) | `GPUCompositeScoreScaler` | Per-indicator scaler. Defaults to `'min-max'`. |
| `aggregation` (optional) | `GPUCompositeScoreAggregation` | How scaled indicators combine into the score. Defaults to `'weighted-sum'`. |
| `weights` | `ArrayLike<number>` | One weight per indicator (at most 16). Missing trailing weights are 0. A negative weight penalizes an indicator in `'weighted-sum'`; `'weighted-geometric-mean'` refuses negative weights (a weighted geometric mean needs `w >= 0`): use `directions` to flip an indicator. |
| `directions` (optional) | `ArrayLike<number>` | One direction per indicator: a negative value means "higher is worse" and flips the scaled value (`1 - s` for min-max and rank, `-s` for z-scores). Defaults to all `+1`. |
| `epsilon` (optional) | `number` | Positive shift added before the logarithm of the geometric mean. Defaults to `1e-6`. |

{/* /parameters */}

## `GPUInequality`

`GPUInequality` computes per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L, Atkinson, Hoover, Palma, Lorenz curve knots, plus the between/within-zone decomposition of Theil T and pooled statistics.

Inputs: `values` (float32), `zoneIds` (uint32; `0xffffffff` or any id `>= zoneCount` skips the row), `zoneCount` (at most `getGPUInequalityMaximumZoneCount(device)`, 3,050,402 on a default device) and `lorenzKnotCount` (compile-time; `zoneCount * lorenzKnotCount` at most `getGPUInequalityMaximumLorenzKnotTotal(device)`, 2^25 on a default device), optional population `weights` (float32) and `mask` (uint32), and a `parameters` view written with `getGPUInequalityParameterValues({epsilon, palmaTopShare, palmaBottomShare})`. Changing the parameters never rebuilds or recompiles the graph. A row is included when its mask is non-zero, its zone is valid, its value is finite and non-negative and, with weights, its weight is finite and positive. With `w` the weight (1 without weights), `W = sum w`, `S = sum w x` and `mu = S / W`:

| Output | Definition |
| --- | --- |
| `gini` | `1 - sum (P_i - P_(i-1)) (L_i + L_(i-1))`, trapezoid area under the Lorenz curve of the value-sorted rows; equals `2 sum i x_(i) / (n S) - (n + 1) / n` for unit weights, `(n - 1) / n` for one holder of everything |
| `theilT` | `(1 / W) sum w r ln r`, `r = x / mu`, `0 ln 0 = 0` |
| `theilL` | `-(1 / W) sum w ln r`; NaN when the zone contains a zero (zeros are never silently dropped) |
| `atkinson` | `1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))`, `1 - exp(mean ln r)` at `e = 1`; zeros are fine for `e < 1`, NaN for `e >= 1`; `e` in `[0, 32]` per frame |
| `hoover` | `sum w abs(x - mu) / (2 S)` |
| `palma` | `(1 - L(1 - top)) / L(bottom)`, cuts per frame (0.1 and 0.4 by default); NaN when `L(bottom) = 0` |
| `mean`, `counts` | weighted mean, included row count per zone (`counts` is `uint32`) |
| `lorenzKnots` | `L(k / (K - 1))`, row-major `zone * K + k`, `L(0) = 0`, `L(1) = 1` |
| `globalSummary` | slots `GPU_INEQUALITY_GLOBAL_SUMMARY`: `TOTAL_THEIL_T`, `BETWEEN_THEIL_T`, `WITHIN_THEIL_T`, `GINI`, `COUNT`, `MEAN`, `TOTAL_WEIGHT`, `TOTAL_INCOME` |

The Lorenz curve is linear inside each row (a row's income spreads over its population share), so a Palma cut or knot inside a row takes a fractional share of that row. Empty zones, and zones with zero total income, report NaN for every index (zero-income zones keep `mean = 0`). The Theil decomposition is `between = sum_g s_g ln(s_g / p_g)` and `within = sum_g s_g T_g` with income share `s_g` and population share `p_g`; the pooled Theil T is computed independently from all included rows, so `total = between + within` is a real check.

Algorithm and cost: rows are sorted by an order-preserving u32 value key (`GPUSort`), re-keyed by zone and stably radix-sorted by zone, so every zone is one value-ordered segment. One thread per zone binary-searches its segment and walks it twice in fixed order. There are no atomics, so results are bitwise reproducible on one adapter. A zone is one thread of serial work (O(segment)): a zone holding most rows serializes the pass. The pooled Gini sums 256-row tiles, then scans and sums the tile totals in one 256-lane workgroup, and the global sums reduce the zones in one 256-lane workgroup (strided reads, fixed binary tree), so their depth is rows / 256 or zones / 256 plus 8 rather than serial. Sums are plain f32.

{/* parameters: GPUInequality generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUInequality` parameters

- Packer: `getGPUInequalityParameterValues(settings?)`, writes `GPU_INEQUALITY_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUInequalityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `epsilon` (optional) | `number` | Atkinson inequality aversion, `0` to `GPU_INEQUALITY_MAXIMUM_EPSILON`. `1` selects the geometric-mean form. Defaults to 1. |
| `palmaTopShare` (optional) | `number` | Population share of the Palma numerator, in `(0, 1]`. Defaults to 0.1. |
| `palmaBottomShare` (optional) | `number` | Population share of the Palma denominator, in `(0, 1]`. Defaults to 0.4. |

{/* /parameters */}

## `GPUTemporalReduction`

Reduces timestamped rows into one record per (cell, coarse time bucket) so a scrubbing view draws one feature per occupied slot instead of one per row. Bucket `b` covers `[origin + b * width, origin + (b + 1) * width)`; the slot is `cell * bucketCount + bucket`.

```ts
graph.add(
  new GPUTemporalReduction({
    cellIds, // uint32 per row, 0xffffffff skips
    timestamps, // float32 relative times, or uint32x2 Int64 words
    values, // float32 per row
    mask, // optional uint32
    parameters: bucketParameters.importToGraph(graph), // [origin, width]
    cellCount,
    bucketCount,
    output: {counts, minimums, maximums, first, last, occupiedSlots: {ids, status: {count, overflow}}}
  })
);
bucketParameters.write(getGPUTemporalReductionParameterValues(origin, width)); // no recompile
```

- Outputs are dense columns of `cellCount * bucketCount` rows. Empty slots have count 0 and NaN minimums, maximums, first and last. `occupiedSlots.ids` lists occupied slots in ascending order (`cell = id / bucketCount`, `bucket = id % bucketCount`), clamped to its length with `overflow` set when more are occupied.
- `first` and `last` are the values at the earliest and latest timestamp in the slot; ties are broken by the lower row index.
- Rows with a bucket outside `[0, bucketCount)`, a skipped cell, a zero mask, or a NaN value or time are dropped, not clamped.
- Exact Int64 times: pass `uint32x2` words and write `getGPUTemporalReductionWordParameterValues(origin, width)` (integer origin, integer width up to 2^32-1). Bucket assignment is exact integer division.
- Determinism: u32 atomics only, so results do not depend on thread order. In float mode bucket edges are defined by correctly rounded products (`fround(b * width) <= fround(t - origin) < fround((b + 1) * width)`), so assignment is identical on every adapter; at most 2^20 buckets.
- Inputs must be single packed views; chunked vectors are not supported.
- `sorted: true` (default false, compile-time) declares `timestamps` non-decreasing and non-NaN. The rows with bucket in `[0, bucketCount)` are then one contiguous range found by two binary searches (same bucket rules as above: the correctly rounded `fround(t - origin) < fround(bucketCount * width)` test in float mode, the exact quotient in word mode), and the classify, extreme and tie kernels are dispatched indirectly over just that range, with the first row read from a GPU word. A scrub costs O(range), not O(rows), and the result is bit-identical to the unsorted path on sorted input. It adds one `uint32` scratch column of `rows` entries. The order is not checked; unsorted input silently reduces the wrong rows.

{/* parameters: GPUTemporalReduction generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUTemporalReduction` parameters

- Packer: `getGPUTemporalReductionParameterValues(origin, width)`, writes `GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH` (2) `float32` values.
  - `origin` (`number`): Start of bucket 0, in the unit of the timestamps.
  - `width` (`number`): Bucket width, in the unit of the timestamps.
- Packer: `getGPUTemporalReductionWordParameterValues(origin, width)`, writes `GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH` (4) `uint32` values.
  - `origin` (`number | bigint`): Integer start of bucket 0, any signed 64-bit value.
  - `width` (`number | bigint`): Integer bucket width in `[1, 2^32 - 1]`.

{/* /parameters */}

## `GPUCalendarBuckets`

Decodes epoch-millisecond timestamps into calendar columns (year, month, day of month, hour, minute,
weekday, day of year, ISO week, ISO week-year, quarter) and an hour by weekday count matrix, so
time-of-day and weekday charts and calendar bucket keys need no per-row `Date` on the CPU. The
columns feed `GPUTemporalReduction`, `GPUCellAggregation` keys and heatmaps.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'calendar', format: 'sint32', length: 2});
graph.add(
  new GPUCalendarBuckets({
    timestamps, // GraphDataView<'uint32x2'>, Int64 epoch ms words, e.g. from getInt64TimeWords
    parameters: parameters.importToGraph(graph),
    output: {hour, weekday, hourWeekdayCounts}
  })
);
parameters.write(getGPUCalendarBucketsParameterValues(-300 /* UTC-5 */, 0 /* Monday first */));
```

- `timestamps`: signed 64-bit two's complement epoch milliseconds as `uint32x2` `(low, high)` rows;
  times before 1970 are fine. Optional `mask` (`uint32`, zero makes the row invalid).
- `parameters` (`sint32`, 2 elements, per frame, no rebuild): fixed UTC offset in minutes and the first
  day of the week for `weekday` and the matrix (0 = Monday, the ISO default, to 6 = Sunday). ISO week
  numbers always start on Monday.
- Time zones: the per-frame offset is a fixed offset. Daylight saving time is supported by either
  the optional per-row `utcOffsets` column (`sint32` minutes, within +-1440, precomputed per row on
  the CPU) or the optional `timeZone` transition table, which needs no per-row CPU work. They are
  mutually exclusive and both override the fixed offset.
- `timeZone: {transitions, offsets}`: `transitions` is a `uint32x2` view of strictly increasing Int64
  epoch-millisecond instants (the format of `timestamps`), `offsets` a `sint32` view of UTC offsets in
  seconds with `transitions.length + 1` elements. `offsets[0]` applies before the first transition
  and `offsets[i + 1]` from transition `i` on. Each row binary-searches its instant (signed 64-bit
  compare, about `log2(n)` steps) and uses that offset; seconds keep pre-standardization mean time
  such as +00:19:32 exact. A table offset beyond +-86400 s makes the row invalid. The table length
  is topology (a table of another length needs a new graph); its contents can change per frame.
  `getGPUTimeZoneTransitionTable(zone, {start, end})` builds the table on the CPU once per zone
  and range from `Intl.DateTimeFormat` (offset sampled daily and bisected to the millisecond; a
  zone without a transition in the range gets one sentinel transition at the Int64 minimum;
  an unknown zone throws). Instants outside the range use the offset at its start or end. The
  helper is as exact as the ICU data in the runtime: Node 22 reproduces Python `zoneinfo` (tzdata
  2026d) transition for transition for New_York, London, Lord_Howe, Kathmandu, Amsterdam and
  Dublin over 1900 to 2040, including seconds offsets; it differs only where the databases are
  different releases (Africa/Casablanca after 2026-09).
- Outputs, each optional but at least one required (compile-time): `year` and `isoWeekYear`
  (`sint32`), `month` (1-12), `dayOfMonth` (1-31), `hour`, `minute`, `weekday` (0-6),
  `dayOfYear` (1-366), `isoWeek` (1-53), `quarter` (1-4) as `uint32`, and `hourWeekdayCounts`
  (168 `uint32`, `weekday * 24 + hour`, cleared every encoding).
- Invalid rows (masked, local day beyond +-1e9, Int64 overflow with the offset, per-row offset beyond
  +-1440) write `0xffffffff` to `uint32` outputs and `-2^31` to `year` and `isoWeekYear`, and are not
  counted in the matrix.
- Exact integer arithmetic only (nibble-wise 64-bit division, Hinnant `civil_from_days`), identical on
  every adapter. Outputs are split over several decode kernels to respect the 8 storage binding limit.

{/* parameters: GPUCalendarBuckets generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUCalendarBuckets` parameters

- Packer: `getGPUCalendarBucketsParameterValues(utcOffsetMinutes, firstDayOfWeek?, target?)`, writes `GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH` (2) values.
  - `utcOffsetMinutes` (`number`): Fixed offset east of UTC in minutes, an integer in `[-1440, 1440]`.
  - `firstDayOfWeek` (`number`): First day of the week for the `weekday` output and the hour by weekday matrix: 0 = Monday (ISO, default) to 6 = Sunday. ISO week numbers always use Monday.
  - `target` (`Int32Array`): Optional destination of at least 2 elements.

{/* /parameters */}

## Related pages

- [GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe)
- [Indexes and joins](/docs/api-reference/experimental/gpu-dataframe-indexes-joins)
- [GPU Spatial Analysis contributors](/docs/api-reference/experimental/gpu-spatial-analysis)
