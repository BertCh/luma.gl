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
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

const graph = new GPUCommandGraph(device, {id: 'time'});
const window = new GPUParameterBuffer(device, {
  id: 'time-window',
  format: 'float32',
  length: GPU_TIME_WINDOW_PARAMETER_LENGTH
});

// `timestamps` and `output` are graph views created on `graph`.
graph.add(new GPUTimeWindowFilter({timestamps, window: window.importToGraph(graph), output}));
const compiled = graph.compile(); // once

// every frame: rewrite parameters, encode the same compiled graph
window.write(getGPUTimeWindowParameterValues({start: now - 60, end: now}));
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

## `GPUTimeWindowFilter`

Filters timestamped rows (instants) or time intervals (trail segments) against a per-frame window
and publishes stable compacted IDs, a clamped count, and optional mask, fade weights, segment clip
fractions, per-track visible counts, and an indirect `instanceCount`. Timestamps come in three
forms, chosen by the view format:

- `float32` relative to an application epoch, the cheapest path;
- `float32` plus `timestampsLow` (double-single, from `splitTimestamps`) for inputs that arrive as
  float64;
- exact `Int64` words: a `uint32x2` view holding each value's little-endian `(low, high)` words,
  for example Arrow `Int64` or `Timestamp` epoch milliseconds uploaded without a CPU pass
  (`getInt64TimeWords`, or `makeArrowTemporalWordGPUVector` in `@luma.gl/arrow`). The window is
  then a `uint32` parameter buffer written with `getGPUTimeWindowWordParameterValues`, whose start
  and end are a `number` (with a sub-unit fraction for a smooth playhead) or a `bigint`. Kernels
  subtract with a borrow before converting to float32, so acceptance is exact for any `Int64`
  value with no epoch to choose; fade and clip differences are exact below 2^24 units.

```ts
graph.add(new GPUTimeWindowFilter({
  timestamps: segmentStarts, endTimestamps: segmentEnds,
  window: window.importToGraph(graph),
  output, fadeWeights, clipFractions,
  drawInstanceCount: graph.importGPUData('trail-count', drawCommands.getInstanceCountData(0))
}));
window.write(getGPUTimeWindowParameterValues({start: now - trailLength, end: now, startFadeDuration: trailLength}));
```

```ts
const window = new GPUParameterBuffer(device, {
  id: 'window', format: 'uint32', length: GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH
});
const {vector: departures} = makeArrowTemporalWordGPUVector(device, table.getChild('departure')!);
graph.add(new GPUTimeWindowFilter({
  timestamps: graph.importGPUVector('departures', departures),
  window: window.importToGraph(graph), output
}));
window.write(getGPUTimeWindowWordParameterValues({start: playheadMs - 3_600_000, end: playheadMs}));
```

## `GPUColumnQuantiles`, `GPUClassBreaks`, `GPUColorScale`, and `GPUBivariateClassification`

Choropleth classification on the GPU, every frame, with no readback: exact quantiles and
percentile filters, class breaks, and colour scales that write a packed `rgba8` column a deck.gl
layer can bind as an attribute. Each output can feed the next contributor as a graph view, so a brush
or filter change recolours the map without a CPU round trip.

```ts
graph.add(new GPUColumnQuantiles({
  values, mask, quantileCount: 3,
  parameters: quantileParameters.importToGraph(graph),
  output: {quantiles, validCount, filterMask, filterBounds} // filterMask may be a transient
}));
graph.add(new GPUClassBreaks({
  values, mask: filterMask, maximumClassCount: 9,
  methods: ['quantile', 'natural-breaks', 'head-tail'], // default: every method
  parameters: breakParameters.importToGraph(graph),
  output: {breaks, classCount, classCounts}
}));
graph.add(new GPUColorScale({
  values, mask: filterMask, domain: breaks, domainCount: classCount, palette,
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
  'midpoint'` (numpy names; `'linear'` is d3 and R-7). The rank is `fround(fround(n - 1) * p)`, so
  order statistics match a sorted CPU column bit for bit; `'linear'` and `'midpoint'` can differ by
  1 ulp where a driver fuses the final multiply-add. `filterRange` gives deck.gl/kepler
  lower/upper percentile filtering: bounds `x[floor(n * lower)]` and `x[ceil(n * upper) - 1]`, and a
  `filterMask` of the rows inside them. At most 2^24 rows. About 1.8 ms at 1M rows and 3.5 ms at
  4M rows for 9 quantiles plus the filter (Apple M-series, headless Chromium).
- `GPUClassBreaks`: per-frame `method` and `classCount` among the compiled `methods`:
  `'equal-interval'`, `'quantile'` (exact linear quantiles at `i / k`), `'standard-deviation'`
  (classes of `interval` standard deviations centred on the mean), `'head-tail'` (Jiang's
  breaks with a per-frame head ratio, default 0.4; 1 gives mapclassify's rule), `'box-plot'`
  (six classes, per-frame hinge), `'maximum-breaks'` (midpoints of the `k - 1` largest gaps
  between distinct sorted values), `'natural-breaks'` (exact Fisher-Jenks over
  `naturalBreaksBinCount` equal-width bins, default 1024, so breaks snap to bin edges), and
  `'custom'`. Head/tail and maximum breaks may return fewer classes; `classCount` reports `k`
  (0 with no finite values, 1 when they are all equal). Kernels of unselected methods return at
  once, except the two radix sorts of `'maximum-breaks'`, which run every frame when compiled.
  Moments use a fixed-order Chan merge, so every output is bitwise reproducible. About 3 to 4 ms
  per frame at 1M rows with six methods compiled.
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
- Inputs must be single packed views; chunked vectors are not supported.

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
- Histograms: equal-width bins over a per-frame `[lo, hi]` per column, or the frame's finite
  `[min, max]` when a bound is NaN; the last bin includes `hi` and values outside are not counted.
  Bin assignment uses correctly rounded products, so it is identical on every adapter. Category
  columns get per-code counts in their first bins.
- Distinct count: HyperLogLog over a murmur3 `fmix32` hash of the value bits (`-0` folded to
  `+0`) with `atomicMax` registers and linear-counting small-range correction. Registers are
  exact, and the estimate is within about `1.04 / sqrt(2^p)` relative error.
- Top categories: exact per-code counts with `uint32` atomics, then the top `topCategoryCount`
  ordered by count (descending) and code (ascending). Codes at or above `categoryCount` are
  counted in `overflowCount`.
- At most 16 columns and 2^24 rows. Inputs must be single packed views.

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
    output: {keys: groupKeys, counts: groupCounts, count, overflow, totalCount}
  })
);
fractions.write(Float32Array.of(0.1, 0.5, 0.9)); // no recompile
```

- Output groups are ascending by key, capacity is `output.keys.length`. When more groups exist, the smallest keys are kept, `overflow` is 1 and `totalCount` is the unclamped number. Tail rows hold the all-ones key, count 0, sums 0 and NaN for every other float statistic.
- `output.counts` counts valid rows (mask set, key valid); each column's `counts` counts finite values. Zero to four columns; each column lists the statistics it needs (a compile-time set) and supplies exactly the matching output views: `counts`, `sums` (exact signed 64-bit fixed point, `uint32x2`, scale `sumScale`, default 65536) and/or `sumValues` for `sum`, `means`, `minimums`, `maximums`, `variances`, `standardDeviations`, `skewness`, `kurtosis`, `medians`, `percentiles` (`capacity * P`, group-major), `modes`, `uniqueCounts`, and `zScores` (one row per source row).
- Exactness: counts, fixed-point sums, minimum, maximum, median, percentiles, mode and unique counts are bit-identical to a CPU reference. Percentiles use the numpy `linear` rule (`h = (n - 1) p`, `lo + (hi - lo) * (h - floor h)` in f32), with `p` read per frame and clamped to [0, 1] (NaN gives NaN). Mode is the longest run of equal values (ties: smallest value). `-0` is treated as `+0`; NaN and infinities are excluded.
- Moments are corrected two-pass: `d = v - mean` is summed in a fixed order (one 256-thread workgroup per group, strided partials and a binary tree), so a second encode is bitwise identical, and central moments about the true mean come from the shifted power sums. Measured agreement with a float64 reference on mixed data is about 1e-7 for mean and variance, 2e-6 for skewness and kurtosis, 7e-6 for z-scores (relative). Skewness is Fisher-Pearson `g1`, kurtosis is excess `g2`; both are NaN for constant groups, a group whose variance is under 2^-19 of its shifted sum of squares is treated as constant (the fixed-point mean carries 2^-17 absolute error). Sample variance of one value is NaN; z-score is 0 when the deviation is 0 or a sample group has one value, NaN for masked, invalid-key, non-finite or capacity-dropped rows.
- Dense mode: `keyCount` (compile-time) requires `uint32` keys in `[0, keyCount)`. `output.keys.length` and every per-group output hold exactly `keyCount` rows, and row `k` is key `k` (`output.keys[k] = k`, `output.count` and `totalCount` are `keyCount`, `overflow` is 0). Keys with no rows keep a row: `counts` 0, `sums` and `uniqueCounts` 0, and NaN for mean, minimum, maximum, variance, standard deviation, skewness, kurtosis, median, percentiles and mode. Keys outside the range and the reserved all-ones key are skipped like masked rows and are not reported as overflow. It costs one binary search per key instead of a group scan, and suits region IDs or other small dense key spaces (the choropleth recipe uses it so empty polygons keep a row). The default compact table of occupied keys is unchanged.
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
  (`count` clamped, `overflow`, optional `totalCount`; ids past `count` are `0xffffffff`). Other
  outputs stay left-aligned. `'left'` rejects `output.rows`.

All results are integer or fixed-point, independent of thread order, and bitwise reproducible.
Input contents (keys, masks, values) are per-frame and need no rebuild; view lengths, key format,
`kind`, `sumScale` and the set of outputs are topology.

## `GPUVariogram`, `GPUSpatialCorrelogram`, `GPURipley`, `GPURipleyDistanceFunctions`, and `GPUPointPatternIndices`

Pair statistics over planar points share one deterministic pair-histogram pass: rows are sorted
into a per-frame lattice whose cells are at least the per-frame `maximumDistance` wide, one thread
per included row visits the 3x3 cell neighbourhood, and each pair adds integer amounts into
per-workgroup shared-memory 64-bit accumulators that are then added into global 64-bit
accumulators. Float terms are fixed point (each term's maximum maps to 2^24), so every result is
exact or has a documented fixed-point bound, independent of scheduling, and bitwise reproducible.
When `maximumDistance` covers the extent the pass is a plain all-pairs loop. A row is included when
its mask is nonzero, its value (if any) finite and its position finite and inside the per-frame
`bounds`. Parameters (`[minX, minY, maxX, maxY, maximumDistance, ...]`) are per-frame; bin counts
and the set of outputs are compile-time.

```ts
graph.add(
  new GPUVariogram({
    positions, values, parameters, gridSize: [64, 64], lagCount: 24, directionCount: 4,
    semivariances, pairCounts, meanDistances
  })
);
parameters.write(getGPUVariogramParameterValues({bounds, maximumDistance: 500, azimuthOffset: 0}));
// After readback: CPU weighted least squares.
const model = fitVariogramModel({distances, semivariances, pairCounts}, {model: 'spherical'});
```

- `GPUVariogram`: Matheron semivariance, pair count, mean lag distance and Cressie-Hawkins robust
  semivariance per lag and optional azimuth sector (`directionCount` equal sectors over `[0, pi)`
  from a per-frame `azimuthOffset`). Bin error is at most `(max - min)^2 * 2^-26`.
  `fitVariogramModel` fits spherical, exponential or Gaussian models (practical-range convention)
  with Cressie, pair-count or uniform weights on the CPU.
- `GPUSpatialCorrelogram`: global Moran's I per distance band (`'cumulative'` bands as PySAL
  `DistanceBand` / ArcGIS Incremental Spatial Autocorrelation, or `'annulus'`), with `E[I]`,
  normality or randomization variance (per-frame), z, p, unordered pair counts, and
  `[firstPeakBand, maximumBand]`. Variances lost to f32 cancellation (for example a band holding
  every pair) are NaN.
- `GPURipley`: K, L, L - r and an annulus pair-correlation estimate over `radiusCount` radii in a
  rectangular window, with `'none'`, `'border'` (reduced sample, `lambda = (n - 1) / A`) or
  `'isotropic'` (Ripley 1977 rectangle, weights capped at 100) edge correction switchable per frame.
- `GPURipleyDistanceFunctions`: the distance functions next to `GPURipley` (spatstat `Gest`, `Fest`,
  `Jest`). `G` is the nearest-neighbour distance function of the events, `F` the empty-space function
  over a `referenceGrid` lattice of reference locations, and `J = (1 - G) / (1 - F)` (below 1 is
  clustered, above 1 regular). `parameters` (`getGPURipleyDistanceParameterValues`) hold `bounds`,
  `maximumDistance` and `edgeCorrection` (`'border'`, the default, `'kaplan-meier'`, `'hanisch'` or
  `'none'`, switchable per frame without recompiling); `gridSize`, `referenceGrid` and `radiusCount` are
  compile-time. Outputs are `g`, `f`, `j` and `radii`, with NaN where a value is undefined. Integer atomics
  keep it bitwise reproducible. `'border'` is spatstat `rs`. `'kaplan-meier'` is spatstat `km`, with hazards
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
  Clark-Evans `[n, observed, expected, R, standardError, z]`, and quadrat counts with
  `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom]`.

Measured on an Apple M3 Pro (headless Chromium), `GPUVariogram` with 32 lags including the cell
sort: 100k points with 54M pairs in range in 10 ms; 20k points all-pairs (200M pairs) in 12 ms.

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
  (`exp(sum(w * ln(s + epsilon)) / sum(w))` over positive weights; pair it with min-max or rank),
  `'principal-component'` (projection of direction-adjusted z-scores on the first eigenvector of
  their correlation matrix; needs `enablePrincipalComponent`; weights are ignored).
- The first component comes from 64 fixed power-iteration steps from `1 / sqrt(d)` in one thread,
  signed so the loadings sum to a non-negative value. `principalComponentSummary` holds
  `[eigenvalue, explainedVarianceRatio, residual]`; the ratio divides by the number of
  non-constant columns.
- Outputs: `score` (required), `scaled`, `columnStatistics` (`[min, max, mean, std]` per column),
  `loadings`, `principalComponentSummary`.
- Determinism: min and max use u32 order-key atomics; means and centered (co)variances are
  two-pass fixed-order sums over 256-row tiles merged in tile order; ranks use one stable radix
  sort per column plus a binary search per tie run. No float atomics.
- `compile-time`: row count, `indicatorCount`, `enableRank`, `enablePrincipalComponent`, the mask
  and which outputs exist. `per frame`: scaler, aggregation, weights, directions, epsilon.

## `GPUInequality`

`GPUInequality` computes per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L, Atkinson, Hoover, Palma, Lorenz curve knots, plus the between/within-zone decomposition of Theil T and pooled statistics.

Inputs: `values` (float32), `zoneIds` (uint32; `0xffffffff` or any id `>= zoneCount` skips the row), `zoneCount` and `lorenzKnotCount` (compile-time), optional population `weights` (float32) and `mask` (uint32), and a `parameters` view written with `getGPUInequalityParameterValues({epsilon, palmaTopShare, palmaBottomShare})`. Changing the parameters never rebuilds or recompiles the graph. A row is included when its mask is non-zero, its zone is valid, its value is finite and non-negative and, with weights, its weight is finite and positive. With `w` the weight (1 without weights), `W = sum w`, `S = sum w x` and `mu = S / W`:

| Output | Definition |
| --- | --- |
| `gini` | `1 - sum (P_i - P_(i-1)) (L_i + L_(i-1))`, trapezoid area under the Lorenz curve of the value-sorted rows; equals `2 sum i x_(i) / (n S) - (n + 1) / n` for unit weights, `(n - 1) / n` for one holder of everything |
| `theilT` | `(1 / W) sum w r ln r`, `r = x / mu`, `0 ln 0 = 0` |
| `theilL` | `-(1 / W) sum w ln r`; NaN when the zone contains a zero (zeros are never silently dropped) |
| `atkinson` | `1 - ((1 / W) sum w r^(1 - e))^(1 / (1 - e))`, `1 - exp(mean ln r)` at `e = 1`; zeros are fine for `e < 1`, NaN for `e >= 1`; `e` in `[0, 32]` per frame |
| `hoover` | `sum w abs(x - mu) / (2 S)` |
| `palma` | `(1 - L(1 - top)) / L(bottom)`, cuts per frame (0.1 and 0.4 by default); NaN when `L(bottom) = 0` |
| `mean`, `count` | weighted mean, included row count (`count` is `uint32`) |
| `lorenzKnots` | `L(k / (K - 1))`, row-major `zone * K + k`, `L(0) = 0`, `L(1) = 1` |
| `globalSummary` | slots `GPU_INEQUALITY_GLOBAL_SUMMARY`: `TOTAL_THEIL_T`, `BETWEEN_THEIL_T`, `WITHIN_THEIL_T`, `GINI`, `COUNT`, `MEAN`, `TOTAL_WEIGHT`, `TOTAL_INCOME` |

The Lorenz curve is linear inside each row (a row's income spreads over its population share), so a Palma cut or knot inside a row takes a fractional share of that row. Empty zones, and zones with zero total income, report NaN for every index (zero-income zones keep `mean = 0`). The Theil decomposition is `between = sum_g s_g ln(s_g / p_g)` and `within = sum_g s_g T_g` with income share `s_g` and population share `p_g`; the pooled Theil T is computed independently from all included rows, so `total = between + within` is a real check.

Algorithm and cost: rows are sorted by an order-preserving u32 value key (`GPUSort`), re-keyed by zone and stably radix-sorted by zone, so every zone is one value-ordered segment. One thread per zone binary-searches its segment and walks it twice in fixed order. There are no atomics, so results are bitwise reproducible on one adapter. A zone is one thread of serial work (O(segment)): a zone holding most rows serializes the pass, and the pooled Gini and global sums are single-thread passes. Sums are plain f32.

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
    output: {counts, min, max, first, last, occupiedSlots: {ids, count, overflow}}
  })
);
bucketParameters.write(getGPUTemporalReductionParameterValues(origin, width)); // no recompile
```

- Outputs are dense columns of `cellCount * bucketCount` rows. Empty slots have count 0 and NaN min, max, first and last. `occupiedSlots.ids` lists occupied slots in ascending order (`cell = id / bucketCount`, `bucket = id % bucketCount`), clamped to its length with `overflow` set when more are occupied.
- `first` and `last` are the values at the earliest and latest timestamp in the slot; ties are broken by the lower row index.
- Rows with a bucket outside `[0, bucketCount)`, a skipped cell, a zero mask, or a NaN value or time are dropped, not clamped.
- Exact Int64 times: pass `uint32x2` words and write `getGPUTemporalReductionWordParameterValues(origin, width)` (integer origin, integer width up to 2^32-1). Bucket assignment is exact integer division.
- Determinism: u32 atomics only, so results do not depend on thread order. In float mode bucket edges are defined by correctly rounded products (`fround(b * width) <= fround(t - origin) < fround((b + 1) * width)`), so assignment is identical on every adapter; at most 2^20 buckets.
- Inputs must be single packed views; chunked vectors are not supported.

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
- Time zones: the per-frame offset is a fixed offset. Daylight saving time is supported only through
  the optional per-row `utcOffsets` column (`sint32` minutes, within +-1440), which overrides the
  fixed offset; the caller precomputes it from its zone rules. There is no zone table on the GPU.
- Outputs, each optional but at least one required (compile-time): `year` and `isoWeekYear`
  (`sint32`), `month` (1-12), `dayOfMonth` (1-31), `hour`, `minute`, `weekday` (0-6),
  `dayOfYear` (1-366), `isoWeek` (1-53), `quarter` (1-4) as `uint32`, and `hourWeekdayCounts`
  (168 `uint32`, `weekday * 24 + hour`, cleared every encoding).
- Invalid rows (masked, local day beyond +-1e9, Int64 overflow with the offset, per-row offset beyond
  +-1440) write `0xffffffff` to `uint32` outputs and `-2^31` to `year` and `isoWeekYear`, and are not
  counted in the matrix.
- Exact integer arithmetic only (nibble-wise 64-bit division, Hinnant `civil_from_days`), identical on
  every adapter. Outputs are split over several decode kernels to respect the 8 storage binding limit.

## Related pages

- [GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe)
- [Indexes and joins](/docs/api-reference/experimental/gpu-dataframe-indexes-joins)
- [GPU Spatial Analysis contributors](/docs/api-reference/experimental/gpu-spatial-analysis)
