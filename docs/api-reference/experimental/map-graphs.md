# Map Graph Toolkit

:::caution Experimental
`@luma.gl/experimental/map-graphs` is an initial, experimental toolkit. APIs may change between
releases without a deprecation period.
:::

The map graph toolkit packages common map tasks as prebuilt GPU Core command graphs ("recipes").
Each recipe is a class with typed graph-view inputs and outputs that a map application adds to its
own [`GPUCommandGraph`](./gpu-core/gpu-command-graph.md) with `graph.add(recipe)`. Recipes are
composed from public GPU Core primitives (`GPUVisibilityWorkflow`, `GPUBVH`, `GPUGridBinning`,
`GPUHistogram`, `GPUReduction`, `GPUAncestorProjection`, ...) and the experimental geospatial and
raster modules; new WGSL is written only where no primitive fits.

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer} from '@luma.gl/experimental/geospatial';
import {GPUTimeWindowFilter, getGPUTimeWindowParameterValues, GPU_TIME_WINDOW_PARAMETER_LENGTH} from '@luma.gl/experimental/gpu-dataframe';
import {submitGraph} from '@luma.gl/experimental/UNRESOLVED';

const graph = new GPUCommandGraph(device, {id: 'map'});
const window = new GPUParameterBuffer(device, {
  id: 'time-window',
  format: 'float32',
  length: GPU_TIME_WINDOW_PARAMETER_LENGTH
});
graph.add(
  new GPUTimeWindowFilter({timestamps, window: window.importToGraph(graph), output})
);
const compiled = graph.compile(); // once

// every frame: rewrite parameters, encode the same compiled graph
window.write(getGPUTimeWindowParameterValues({start: now - 60, end: now}));
submitGraph(device, compiled, undefined);
```

## Conventions

- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView` / `GraphVectorView` /
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Recipes never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, time windows, thresholds, radii, observer
  positions, budgets, and region shapes are read from storage views, usually a
  `GPUParameterBuffer` that the application rewrites with `write()`. Lengths, capacities,
  grid sizes, and which optional views exist are compile-time topology; each prop's TSDoc says which
  category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids`, `count`, `overflow`, optional `totalCount`). `count` is clamped to `ids.length` and can be
  an indirect draw instance count; `overflow` is rewritten every encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** `importGraphBuffer` (and `GPUParameterBuffer.importToGraph`)
  registers a buffer under one graph resource ID. Importing the same buffer again under a different
  ID throws `already in use`; import it once and pass the returned view to every recipe that reads
  it.

## Recipes

### GPUTimeWindowFilter

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

### GPUPointDensity

Bins points into a square grid (`GPUGridBinning`, `GPUGridAggregation`) or pointy-top hexagons
(keys kernel plus `GPUGroupAggregation`) and writes per-cell counts, sums, means, a heatmap field,
its `[min, max]` extent, a histogram, and an optional `r32float` texture. Bounds and hexagon radius
may be GPU views; the field can be smoothed with a per-frame `GPUConvolution` kernel.

```ts
graph.add(new GPUPointDensity({
  positions, weights, bounds: viewport.importToGraph(graph), gridSize: [256, 256],
  statistic: 'mean',
  smoothing: {kernel: kernel.importToGraph(graph), kernelWidth: 5, kernelHeight: 5},
  output: {values, extent, histogram, texture}
}));
```

`mask` is an optional per-row `uint32` view with the same length as `positions`. Rows whose mask
is `0` are excluded from count, sum, mean, smoothing input, extent and histogram; any nonzero value
includes the row, like the mask inputs of `GPURegionStatistics` and `GPUFlowAggregation`. Contents
are per-frame and rewriting them never recompiles. Hexagon binning reads the mask in its keys
kernel; grid binning adds one pass that writes masked positions as NaN into a transient copy.
Without `mask` the graph is unchanged. Pass a residency arena `liveMask`, or
`GPUTimeWindowFilter.outputMask`, to get time- or tile-gated density without poisoning position
columns.

`getGPUPointDensityHexagonGridSize` sizes a hexagon lattice that covers given bounds, and
`GPU_POINT_DENSITY_HEXAGON_WGSL` lets render shaders place hexagon bins.

### GPUCellAggregation, GPUCellRollup, GPUCellPyramid, and GPUCellLevelSelection

Aggregates rows into discrete global grid cells (Quadbin or H3) and builds a zoom pyramid of
cell tables. 64-bit cell keys are stored as two `uint32` words in little-endian `(low, high)`
order, which is the layout of Arrow `Uint64` columns and of `GPUH3CellProjection` input. An H3
table therefore feeds the projection directly.

- **Inputs.** `positions` (lng/lat degrees, Quadbin only) are keyed in WGSL. `cells` are
  pre-keyed rows, for example CARTO Quadbin or H3 columns, as `uint32x2` (`wordOrder:
  'high-low'` is also accepted). Pre-keyed rows finer than `resolution` are truncated to it.
  Invalid rows, and rows coarser than `resolution`, are skipped. H3 validity comes from
  gpu-dggs `dggs_h3_is_valid_cell_id`. Optional `values` and a `mask` can change every frame.
  Rows with a NaN coordinate or a non-finite value are skipped.
- **Cell table** (`GPUCellTable`). Capacity is `cells.length`. Rows `[0, count)` hold occupied
  cells in ascending key order. Later rows hold the empty key `0xffffffff` in both words, count 0
  and NaN extremes. Columns: `counts`, exact fixed-point `sums` (signed 64-bit words of
  `roundHalfEven(fround(value * sumScale))`, default scale 65536), `sumValues` (f32
  `sums / sumScale`), `minimums` and `maximums`. On overflow the table keeps the smallest keys
  and sets `overflow`.
- **Roll-up.** `GPUCellRollup` reduces a table to a coarser resolution without re-reading rows
  or sorting. Parent keys truncate the cell path (`cellToParent`), which preserves key order.
  Counts and sums are integer sums, so a roll-up equals direct aggregation bit for bit. The
  source table's overflow carries through.
- **Pyramid.** `GPUCellPyramid` aggregates once at the finest of `levels` (listed finest first),
  rolls each level into the next, and can write per-level counts to `levelCounts`. Intermediate
  sums and extremes that a coarser level needs become transients.
- **Level selection.** `GPUCellLevelSelection` reads a per-frame `activeLevel` (one `uint32`) and
  writes that level's count, first row, and the `instanceCount`/`firstInstance` words of indirect
  draw arguments. Keep it in a small per-frame graph: switching levels rewrites one word and never
  re-runs the pyramid.

```ts
graph.add(new GPUCellPyramid({
  family: 'quadbin',
  positions, values, mask,                    // mask and values can change every frame
  levels: [18, 14, 10, 6, 2].map((resolution, i) => ({resolution, output: tables[i]})),
  levelCounts
}));
selectionGraph.add(new GPUCellLevelSelection({
  levelCounts, activeLevel: activeLevel.importToGraph(selectionGraph),
  output: {count, drawArguments}
}));
activeLevel.write(Uint32Array.of(levelForZoom(zoom)));   // no recompile
```

- **Determinism.** Point keys use only integer arithmetic. The longitude column is exact. The
  latitude row is a fixed-point Mercator (`ln(cos t) - ln(sin t)`, Q0.32 polynomials, exact long
  division), so every adapter and the CPU reference agree bit for bit. f32 polynomials would not
  agree, because WebGPU compilers may fuse `a * b + c` (Metal does). Sorting uses one stable
  `GPUSort` when the key path fits 31 bits (Quadbin ≤ 15, H3 ≤ 8) and two otherwise. Rows
  accumulate with u32 atomics only: counts, 64-bit sums as two words with a carry, and extremes
  as order-preserving keys.
- **Accuracy.** Compared with the f64 quadbin-py formula, the fixed-point row differs for 0% of
  random points up to resolution 8, 0.01% at 12, 0.1% at 16, 1.1% at 20, 17% at 24 and 55% at
  26. It is never off by more than 1 row up to resolution 22, 2 rows at 24 and 5 rows at 26
  (f32 latitudes are not that precise). The column always matches.
- Inputs must be single packed views. Forward keying of points is Quadbin only.

### GPUPointToCell and GPUCellGeometry

Keys points to discrete global grid cells and turns cells back into centres and boundary
polygons. This covers the most common Studio/CARTO preprocessing step ("lng/lat column to H3")
and the geometry needed to draw any DGGS cell column. Keys are 64-bit `uint32x2` little-endian
`(low, high)` words, which is the Arrow `Uint64` layout that `GPUCellAggregation` pre-keyed rows
and `GPUH3CellProjection` read. The zero key means "no cell".

- **Families** (`GPUCellIndexFamily`) and their resolutions:

  | Family | Resolutions | Notes |
  | --- | --- | --- |
  | `'quadbin'` | 0-26 | integer keys, exact |
  | `'h3'` | 0-15 | f32 input: exact to res 4 |
  | `'quadkey'` | 1-29 | packed: length in bits 58-63, base-4 digits |
  | `'geohash'` | 1-12 | packed: length in bits 60-63, base-32 characters, i.e. the lng/lat bit interleave |
  | `'s2'` | levels 0-30 | standard S2CellId |

  The quadkey and geohash packings are the layouts the shadertools `dggs` decoders read.
- **`GPUPointToCell`.**
  - Input is `positions` (lng/lat `float32x2`) and an optional `mask`. Output is `cells` and an
    optional `validity`, one row per input row.
  - Masked and non-finite rows get the zero key. The resolution is compile-time.
  - Rewriting positions or the mask re-encodes without recompiling.
- **`GPUCellGeometry`.**
  - Input is `cells` (`wordOrder` option).
  - Outputs are `centers` (lng/lat) and optional `boundaries` with a fixed `maximumVertexCount`
    stride and `vertexCounts`. Families are the five above plus `'a5'`.
  - H3 boundaries follow h3-js `cellToBoundary` exactly in vertex count and order. That includes
    face-crossing distortion vertices (up to 10, so use stride 10) and pentagons.
  - Quadbin, quadkey, geohash and S2 have 4 vertices; A5 has 5.

```ts
graph.add(new GPUPointToCell({family: 'h3', resolution: 8, positions, output: {cells}}));
graph.add(new GPUCellGeometry({
  family: 'h3', cells, maximumVertexCount: GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  output: {centers, boundaries, vertexCounts}
}));
```

- **Exactness and accuracy.**
  - Quadbin, quadkey and geohash keying is integer-only and equals the BigInt oracles bit for bit
    on every device.
  - S2 and H3 use f32 sphere math (polynomial sine/cosine, no trig builtins for H3) and are exact
    at coarse resolutions. A mismatch is an edge neighbour of the f64 answer:

    | Family | Uniform-point mismatch rate vs the f64 reference (f32 inputs) | Usable limit |
    | --- | --- | --- |
    | H3 vs h3-js | 0 at res 0-4, 1e-4-4e-4 at 5-8, 1e-3 at 9, 2e-3 at 10, 1.8% at 12, 28% at 15 | res ≤ 12 |
    | S2 | 0 at levels 0-10, 1e-4-3e-3 at 11-15, 1.8% at 18, 13% at 21 | level ≤ 20 |

  - Geometry matches h3-js / oracles within about 3e-5 degrees.

### GPUCellTopology and GPUCellCompaction

Cell topology for Quadbin and H3 keys stored as `uint32x2` little-endian `(low, high)` words.
This is the Arrow `Uint64` layout used by `GPUCellAggregation`. Inputs also accept
`wordOrder: 'high-low'`. A zero key means "no cell" in every output.

- **`GPUCellTopology`** writes a fixed fan-out row per input cell, with stride from
  `getCellTopologyStride(family, operation)`:
  - `{type: 'disk', k}`: `gridDisk` / `gridDiskDistances`, sorted by (distance, key). Optional
    `distances` column.
  - `{type: 'ring', k}`: `gridRing`, sorted by key.
  - `{type: 'parent', resolution}`: `cellToParent`.
  - `{type: 'children', resolution, inputResolution}`: `cellToChildren`, in ascending key order.
  - Rows are zero padded and an optional `counts` column gives the valid entries per row. `k` is at
    most 8 and the children stride at most 4096 (H3 depth 4, Quadbin depth 6). Invalid, masked
    or wrong-resolution rows give zero rows.
- **H3 neighbors are exact.** `H3_NEIGHBOR_WGSL` ports H3's `h3NeighborRotations`: base-cell
  neighbor and rotation tables, class II/III digit carry tables and pentagon clockwise-offset
  faces. Disks and rings are a bounded breadth-first search per thread over the six neighbor
  steps. The search is integer-only and exact across icosahedron faces and pentagons. A pentagon's
  disk has fewer cells, and the rest of the row is zero padded.
- **Quadbin neighborhoods** use Chebyshev distance on tile column and row (`QUADBIN_KRING`
  style). Columns wrap across the antimeridian. Rows are clipped at the poles. A tile that the wrap
  reaches twice at small zoom levels appears once.
- **`GPUCellCompaction`** handles sets of cells:
  - `{type: 'compact', minimumResolution?}` (`compactCells`): input is cells at one resolution,
    which are sorted unless `sorted: true` and then deduplicated. It replaces every complete
    sibling set by its parent, recursively: 4 children for Quadbin, 7 for an H3 hexagon, and for an
    H3 pentagon 6 at depth 1, `(5 * 7^d + 1) / 6` in general.
  - `{type: 'uncompact', resolution, maximumDepth?}` (`uncompactCells`): count, scan, then emit
    the descendants at `resolution`.
  - Both write a capacity-bounded `{cells, count, overflow, totalCount?, droppedCount?}`.
    Compact output is in ascending canonical key order. Uncompact output is ascending when the
    input is in path order.

```ts
graph.add(new GPUCellTopology({
  family: 'h3', operation: {type: 'disk', k: 2},
  cells, output: {cells: diskCells, distances, counts}   // diskCells.length = rows * 19
}));
graph.add(new GPUCellCompaction({
  family: 'h3', operation: {type: 'compact'},
  cells: polyfillCells, output: {cells: compacted, count, overflow}
}));
```

- **Determinism.** All operations are integer-only, with fixed sort orders. GPU results equal the
  CPU oracles (h3-js and BigInt Quadbin) bit for bit, including padding.

### GPUCellCover

Polyfills polygons (with holes) into Quadbin or H3 cells, like CARTO `QUADBIN_POLYFILL` /
`H3_POLYFILL` and h3-js `polygonToCells`. The result is `(feature id, cell)` pairs that feed
`GPUCellAggregation` pre-keyed rows, `GPUKeyJoin` and `GPUCellCompaction`.

- **Inputs.** Polygon features use the `GPUPointInPolygonJoin` layout: `polygonPositions`
  (lng/lat `float32x2`), `featureOffsets`, `polygonOffsets`, `ringOffsets`, and optional
  `featureIds`. Geometry is planar in lng/lat degrees, as in h3-js and turf. Containment is
  even-odd over all rings. Polygons must not cross the antimeridian.
- **Containment.**
  - `'center'`: the cell centre is inside the polygon.
  - `'full'`: the cell lies inside the polygon.
  - `'intersects'`: the cell interior meets the polygon.
  - Quadbin supports all three modes. A Quadbin tile is an axis-aligned lng/lat rectangle, so
    `full` and `intersects` are exact edge-versus-open-rectangle tests. H3 supports `'center'`
    only.
- **Candidates.** Quadbin tests the integer-exact tile range of each feature's bounding box. H3
  lays a lng/lat lattice over the bounding box, with spacing below half the minimum H3 inradius at
  that resolution (measured from h3-js; the minimum sits next to a pentagon). Each lattice point
  indexes its cell with `cellIndexH3FromLngLat`. The cell is emitted only from the lattice point
  nearest its centre, so every cell whose centre lies in the polygon is emitted exactly once with
  no sort.
- **Output.** `{featureIds, cells (uint32x2 little-endian), count, overflow, totalCount?}` with the
  usual capacity-bounded semantics. Order is by feature, then candidate order. The pipeline is
  count, scan, test, scan, write, with no atomic append, so the output is deterministic.
  `candidateCapacity` bounds the candidates per graph, and exceeding it raises `overflow`.

```ts
graph.add(new GPUCellCover({
  family: 'h3', resolution: 8, containment: 'center',
  polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
  candidateCapacity: 1 << 22,
  output: {featureIds, cells, count, overflow}
}));
```

- **Accuracy.**
  - Quadbin results equal an f32-faithful CPU oracle at resolutions 3-10 in all modes. Compared
    with an independent f64 centre test there were 0 disagreements.
  - H3 `center` matches h3-js `polygonToCells` exactly at resolutions 3-7. At 8-10 it differs on
    12 of about 75,000 cells. Each of those cells has its centre within 5e-5 degrees of a polygon
    edge, where the f32 centre differs from the f64 one.
  - An f64 simulation of the lattice rule matches h3-js exactly, so the spacing never misses a
    cell.

### GPUColumnQuantiles, GPUClassBreaks, GPUColorScale, and GPUBivariateClassification

Choropleth classification on the GPU, every frame, with no readback: exact quantiles and
percentile filters, class breaks, and colour scales that write a packed `rgba8` column a deck.gl
layer can bind as an attribute. Each output can feed the next recipe as a graph view, so a brush
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

- Class rule shared by all four recipes: breaks are edges `e[0..k]` (`e[0]` the smallest and
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
- Inputs must be single packed views; chunked vectors are not supported.

### GPUColumnProfile

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

### GPUGroupStatistics

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
- Cost: one stable key sort (two 32-bit radix sorts for 64-bit keys); each column that asks for median, percentiles, mode or unique count adds a value sort plus a key re-sort. Inputs must be single packed views. Approximate distinct counts (HyperLogLog) are not included.

### GPUKeyJoin

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

### GPUCellTableCompare

`GPUCellTableCompare` outer-joins two sorted cell tables (the outputs of `GPUCellAggregation` or `GPUCellRollup`) into one union table for period-over-period or A-versus-B compare maps. Both tables must use the same grid family and resolution; the recipe cannot check that.

```ts
graph.add(new GPUCellTableCompare({
  before: lastMonth, after: thisMonth,
  measure: 'sum', sumScale: 65536,
  output: {cells, presence, before, after, delta, ratio, percentChange, zScore, count, overflow}
}));
```

- `output.cells` holds the union keys ascending (`uint32x2`, little-endian `(low, high)`); its length is the capacity. When the union is larger, the smallest keys are kept, `count` is clamped and `overflow` is 1. `overflow` also becomes 1 when either input table overflowed. Optional `totalCount` receives the unclamped union size.
- `presence`: bit 1 (`GPU_CELL_COMPARE_PRESENT_BEFORE`) when the cell is in `before`, bit 2 (`GPU_CELL_COMPARE_PRESENT_AFTER`) when in `after`.
- `before` / `after`: the measure (`'count'`, or `'sum'` decoded from the fixed-point `sums`), 0 when absent.
- `delta = after - before`, computed exactly as a 64-bit integer difference (counts and fixed-point sums) and only then rounded to f32, so large sums with small differences are not lost to f32 cancellation.
- `ratio = after / before` and `percentChange = 100 * (after - before) / before`, NaN when `before` is 0.
- `zScore`: `'poisson'` (default for counts) is `delta / sqrt(before + after)`, 0 when both are 0. `'standardized'` (default for sums) is `(delta - mean) / sd` over the union rows with the population standard deviation, 0 when sd is 0. The mean and sd use one workgroup in a fixed order, so a re-encode is bitwise identical.
- Rows past `count` hold the empty key (all ones), presence 0, `before`/`after` 0, and NaN in `delta`, `ratio`, `percentChange` and `zScore`.

The merge is a binary-search merge path with an exclusive scan of matched rows, so every union row is written by exactly one invocation and no atomics are used. Cost is O((n + m) log(n + m)). Table contents are per-frame (re-encode without recompiling); capacities, columns, `measure` and `zScore` are topology.

### GPURegionStatistics

Selects points inside a per-frame rectangle, circle, lasso polygon (world space or screen pixels
through a view-projection transform), pick region (`GPUIndexPickingTarget.addRegionPass` result),
or caller mask, and reduces them into selected count, value count, sum, mean, minimum, maximum, and
a fixed-bin histogram packed into one small `summary` buffer. Optional stable selected IDs stay on
the GPU for highlighting. `GPURegionStatisticsReadback` reads the summary through a
`GPUReadbackRing` without stalling. `GPURegionMask` and `GPUPickRegionMask` are exported as
standalone building blocks.

```ts
graph.add(new GPURegionStatistics({
  selection: {kind: 'polygon', vertices: lasso, vertexCount: lassoCount.importToGraph(graph),
    screenTransform: transform.importToGraph(graph)},
  positions, values, histogram: {binCount: 16}, output,
  summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'uint32')
}));
```

An optional `spatialIndex: {kind: 'grid', index, candidateCapacity}` takes a caller-built
`GPUGridIndex` over the same positions and reduces only candidates from the cells a world-space
rectangle, circle, or lasso touches. Results match the full scan except for float reassociation in
`sum` and `mean`; the `candidatesTruncated` flag reports capacity overflow or rows missing from the
index.

An optional `drawInstanceCount` (one-row `uint32` view, for example the instance-count word of an
indirect draw record) receives the same capacity-clamped selected count as `output.count` on every
encoding, so selected rows can be drawn instanced through `output.ids` without a readback. It
requires `output`; statistics-only graphs read the count from `summary`.

### GPUPointInPolygonJoin and GPUNearestFeatureJoin

Spatial joins over a `GPUBVH` built from feature bounds every encoding. `GPUPointInPolygonJoin`
refines candidates with the robust `GPUPairwisePointInPolygon` predicate and assigns each point the
containing feature with the smallest row; `GPUNearestFeatureJoin` measures candidates with
`GPUPairwisePointSegmentDistance` and keeps the nearest point or segment feature within a per-frame
radius. Both write per-point feature IDs, per-feature counts, optional compact matches, and one
overflow flag covering BVH leaf, candidate, and match capacity.

A candidate pair whose containment cannot be proven (non-finite input, malformed offsets, or a
point within the predicate's double-single error envelope of an edge that can affect the result)
is classified `uncertain`. Uncertain pairs are never assigned. They are counted in the optional
`uncertainCount`, so a point whose only candidates were uncertain stays unassigned visibly rather
than silently. Edges strictly above, below, or to one side of the point are decided exactly from
coordinate signs, so a far edge whose line passes near the point does not make it `uncertain`.

```ts
graph.add(new GPUNearestFeatureJoin({
  points, features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  radius: snapRadius.importToGraph(graph), candidateCapacity: points.length * 4,
  nearestFeatureIds, nearestDistances, overflow
}));
```

`spatialSort: true` reorders features along a Morton curve of their bound centers before the
`GPUBVH` build. Results are identical, and traversal is 30x to 75x cheaper when features are not
spatially coherent in row order.

### GPULineSegmentize, GPUGreatCircleArcs, GPULineSmooth, and GPULineChunk

Geometry-producing line recipes. Each writes a `GPULinePathOutput`: flat `positions`, clamped
`pathOffsets` (deck.gl `PathLayer` `startIndices`), `count`, `overflow`, and optional `totalCount`,
`pathCount`, `sourcePaths` (output path to input path), `sourceRows` (output vertex to input
segment row) and `measures` (distance along the source path). The vertex capacity is
`positions.length`; offsets are clamped to it, so a truncated result is still a valid layout.
Inputs are packed `positions` sorted by path with `pathOffsets` (`pathCount + 1` rows).

```ts
graph.add(new GPULineSegmentize({
  positions, pathOffsets, coordinateSystem: 'spherical', // or 'planar'
  parameters: segmentizeParameters.importToGraph(graph),
  output: {positions: densified, pathOffsets: densifiedOffsets, count, overflow, measures}
}));
segmentizeParameters.write(getGPULineSegmentizeParameterValues({maximumSegmentLength: 50000}));

graph.add(new GPUGreatCircleArcs({
  sources, targets, // longitude/latitude pairs
  parameters: arcParameters.importToGraph(graph),
  output: {positions: arcPositions, pathOffsets: arcOffsets, count, overflow}
}));
arcParameters.write(getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 32}));

graph.add(new GPULineSmooth({positions, pathOffsets, iterations: 3, closed: false,
  parameters: smoothParameters.importToGraph(graph), output}));

graph.add(new GPULineChunk({positions, pathOffsets, mode: 'chunk', // or 'substring'
  parameters: chunkParameters.importToGraph(graph),
  output: {positions: pieces, pathOffsets: pieceOffsets, count, overflow, pathCount, sourcePaths}}));
chunkParameters.write(getGPULineChunkParameterValues({chunkLength: 1000}));
```

- `GPULineSegmentize` (PostGIS `ST_Segmentize`, densify): every segment of length `L` becomes
  `clamp(ceil(L / maximumSegmentLength), 1, maximumPiecesPerSegment)` equal pieces; input vertices
  are kept. `'spherical'` positions are longitude/latitude degrees: lengths are great-circle
  distances times `radius` (default mean Earth radius, meters), interior points are slerped, and
  longitudes are unwrapped along each path so it stays continuous across the antimeridian.
- `GPUGreatCircleArcs` (turf `greatCircle`): one path of `segments + 1` vertices per
  origin/destination pair, `segments = clamp(max(ceil(distance / maximumSegmentLength),
  minimumSegments), 1, maximumSegments)`. Endpoints are exact; longitudes are unwrapped from the
  source (Tokyo to San Francisco ends at 237.6), never split. Draw with `PathLayer`
  `wrapLongitude: true` or in a globe view. Pairs within about 1e-3 rad of antipodal have no unique
  great circle.
- `GPULineSmooth` (Chaikin corner cutting, turf `polygonSmooth`): `iterations` (compile time,
  1 to 10) each replace edge `(a, b)` with `mix(a, b, r)` and `mix(a, b, 1 - r)` for a per-frame
  ratio `r` (default 0.25). Open paths keep their endpoints (`n` vertices become `n * 2^k`); with
  `closed: true` a repeated closing vertex is ignored and each ring repeats its first vertex.
  Output sizes depend only on the layout, so every level's offsets come from two scans in closed
  form and each level is one kernel; levels ping-pong through two capacity-sized transients.
- `GPULineChunk`: `'chunk'` splits each path into pieces of `chunkLength` (turf `lineChunk`), the
  last piece ending at the path end; consecutive pieces share their boundary point. The piece
  count is data dependent: the path capacity is `output.pathOffsets.length - 1`, and
  `pathCount`/`sourcePaths` report the pieces. `'substring'` extracts `[startMeasure,
  endMeasure]` from every path (turf `lineSliceAlong`, PostGIS `ST_LineSubstring` by distance);
  measures are clamped to the path and a start past the end gives an empty path. Zero-length
  paths are copied as one piece. Both modes support `'spherical'`.
- Everything per frame lives in the parameter buffer, so resolution, ratio, chunk length and
  ranges change with no recompile. All kernels are element-wise or sequential per path:
  deterministic, no float atomics.
- Precision (f32, measured against f64 oracles on Apple Metal): planar densify and chunks within
  2e-5 of the coordinate scale; spherical densify within 2.9 m, arcs up to 16,500 km within 5.7 m.

### GPUGeometryMeasures, GPUGeodesicPairs, and GPUGeodesicDestination

Measurement columns for line and polygon features and for point pairs, in planar coordinates or
on the Earth (turf `area`, `length`, `centroid`, `bbox`, `distance`, `bearing`, `destination`,
`midpoint`; PostGIS `ST_Area`, `ST_Length`, `ST_Perimeter`, `ST_Centroid`, `ST_Envelope`,
`ST_NPoints`, `ST_Distance(geography)`, `ST_Azimuth`, `ST_Project`).

```ts
graph.add(new GPUGeometryMeasures({
  positions, geometryType: 'polygons',
  ringOffsets, featureRingOffsets, // feature -> rings -> vertices
  coordinateSystem: 'wgs84', // or 'planar', 'spherical'
  holeRule: 'winding', // or 'first-ring-exterior'
  output: {areas, lengths, centroids, bounds, vertexCounts},
  groupIds, groupCount: 12,
  groupOutput: {areas: groupAreas, centroids: groupCentroids, featureCounts}
}));

graph.add(new GPUGeodesicPairs({origins, targets, model: 'wgs84',
  output: {distances, initialBearings, finalBearings, midpoints, converged}}));
graph.add(new GPUGeodesicDestination({origins, bearings, distances,
  output: {destinations, finalBearings}}));
```

- Features: one invocation per feature walks its rings in row order with Neumaier-compensated f32
  sums, relative to the feature's first vertex, so projected coordinates far from the origin keep
  their precision (a 1 m square at x = 1e6 measures exactly 1). Rings close implicitly; a repeated
  closing vertex adds a zero edge. Outputs: length (lines) or perimeter (polygons), signed and
  absolute area, area-weighted (polygons) or length-weighted (lines) centroid falling back to the
  vertex mean, bounds, vertex count. Empty features give zeros and NaN centroid and bounds.
- Holes: `'winding'` sums signed ring areas (holes wind opposite to the exterior, RFC 7946);
  `'first-ring-exterior'` counts the first ring of each feature as exterior and the others as holes
  whatever their winding.
- `'spherical'` areas are the Chamberlain-Duquette formula turf uses (shoelace in the Lambert
  cylindrical equal-area projection, exact for longitude/latitude boxes; long edges are straight
  in that projection, not geodesics); pass `radius: GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS` for turf
  `area` parity. `'wgs84'` maps latitude to authalic latitude (exact equal-area ellipsoid mapping)
  and measures edges with an f32 Vincenty inverse. Rings around a pole are not supported.
- Groups: features are stably sorted by group ID and one workgroup per group reduces them with a
  fixed tree, so group sums are bitwise reproducible. Group centroids are area- (or length-)
  weighted means of feature centroids.
- `GPUGeodesicPairs`: distance, initial and final bearing (degrees clockwise from north in
  `(-180, 180]`), and midpoint per pair. `'sphere'` uses a Vincenty-form central angle that is well
  conditioned from millimeters to antipodal; `'wgs84'` runs Vincenty's inverse with a compile-time
  iteration cap and reports non-converged (near-antipodal) pairs, which fall back to the sphere.
- `GPUGeodesicDestination`: destination and final bearing from origin, bearing and distance
  columns, on the sphere or by Vincenty's direct solution.
- Precision (f32 vs f64 oracles, Apple Metal): WGS84 lengths 9e-8 m at 1 m, 0.78 m at 12,000 km;
  WGS84 and spherical box areas within 3e-7 relative from 100 m blocks to 2,000 km boxes;
  sphere pair distances 1e-7 m at 1 m and 0.9 m at 10,000 km; destinations within 1 m, bounded by
  the f32 resolution of output latitudes (about 0.4 m).

### GPULinearReferencing and GPULineLocate

Linear referencing on many polylines (`positions` sorted by path, `pathOffsets`): snap points to
the nearest path and report where they are along it, or place events on paths by measure.

```ts
graph.add(new GPULinearReferencing({
  points, positions, pathOffsets,
  radius: radiusParameters.importToGraph(graph), // one float32 row, per frame
  candidateCapacity: 1 << 20, spatialSort: true,
  output: {pathIndices, segmentIndices, fractions, footPoints, distances, measures, sides, signedOffsets},
  overflow
}));

graph.add(new GPULineLocate({
  positions, pathOffsets, eventPaths, eventMeasures, eventOffsets,
  measureMode: 'distance', // or 'fraction'
  parameters: locateParameters.importToGraph(graph),
  output: {positions: eventPositions, angles, statuses}
}));
locateParameters.write(getGPULineLocateParameterValues({measureOffset: elapsedMeters})); // animate
```

- `GPULinearReferencing` (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint` and
  `ST_LineLocatePoint`): every vertex row becomes a segment to the next row of its path and
  `GPUNearestFeatureJoin` finds each point's nearest segment within the radius through its BVH.
  Ties go to the smallest segment row (earliest along the path, lowest path). Outputs per point:
  path, segment within the path, fraction along the segment, foot point, distance, measure (distance
  along the path to the foot point), side (`1` left of the path direction, `-1` right, `0` on it)
  and `side * distance`. Points with nothing in range get `0xffffffff` indices, distance `-1`,
  side `0` and NaN elsewhere. `overflow` reports BVH leaf or candidate capacity overflow.
- `GPULineLocate` (turf `along`, PostGIS `ST_LineInterpolatePoint` and `ST_LocateAlong`): each
  event's measure (distance or fraction, optionally `measure * scale + offset` from the parameter
  buffer) is clamped to the path (`statuses` 1) and located by binary search over cumulative vertex
  measures, then moved by an optional lateral offset along the left normal. Outputs: position,
  segment, unit tangent, angle (degrees counter-clockwise from +x, the deck.gl `getAngle`
  convention) and status (`2` for empty or out-of-range paths, NaN position).
- Cumulative measures are a per-path Neumaier-compensated f32 prefix (optionally output as
  `vertexMeasures`). Planar only: project geographic data first.

### GPUBufferSelection

Selects points within a per-frame planar distance (inclusive) of point or segment features. A
polyline is passed as consecutive segment rows. The recipe composes `GPUNearestFeatureJoin`. Unlike
the join's unordered `matches`, it writes a source-aligned 0/1 mask and stable IDs in ascending row
order, with a clamped count and overflow that covers both the output capacity and the join.

```ts
graph.add(new GPUBufferSelection({
  points, features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  distance: bufferDistance.importToGraph(graph), candidateCapacity: points.length * 4,
  spatialSort: true, outputMask, output
}));
```

### GPUZonalStatistics

Choropleth aggregation. Each point is joined to the polygon feature that contains it (an internal
`GPUPointInPolygonJoin`), or the caller supplies per-point feature rows. The points are then
reduced per feature to count, valid-value count, sum, weighted sum and weight sum, mean, minimum,
maximum, density (count divided by a caller area or a GPU shoelace area), and a feature-value
`[min, max]` extent for color scales.

The output views you pass select the statistics at compile time. Empty features give a sum of 0 and
NaN for the mean, minimum and maximum. With `sumOrder: 'atomic'` (the default), sums use float
atomics, so their last bits can vary between runs. `sumOrder: 'sorted'` stably sorts points by
feature and reduces each segment in a fixed tree order, so sums are bitwise reproducible.

```ts
graph.add(new GPUZonalStatistics({
  features: {kind: 'polygons', polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
    candidateCapacity: points.length * 2},
  points, values: incomes, weights: households,
  output: {counts, means, densities, extent, extentStatistic: 'mean', overflow}
}));
```

`output.uncertainCount` (one `uint32` row) reports the join's uncertain candidate pairs. A nonzero
value means some points could not be classified and the per-feature statistics may be low. It is
always 0 for caller-supplied feature rows.

### GPUNetworkReachability

Single- or multi-source shortest-path costs over a directed CSR road network with isochrone bands,
band counts, predecessors, and a convergence flag. Relaxation is chaotic Bellman-Ford over a compact
GPU frontier queue with a round-stamped visited set: each of the `maxIterations` rounds is one
indirect dispatch sized by the previous round's pushes, so converged frames dispatch nothing and
nothing is read back. Within a round each workgroup chains up to `localIterations` (default 16)
hops through workgroup memory, so a path of `h` hops needs about `h / localIterations` rounds; costs
are identical for every setting. Predecessors are cycle-safe: the smallest strictly cheaper tight
in-neighbor, or across equal-cost (zero-weight) edges the smallest tight in-neighbor one BFS level
closer to a strict entry or a source, found in up to `maxTieIterations` extra rounds (default 4).
`maxIterations` counts rounds, one graph node each, not hops. With the default `localIterations`,
a budget of about `h / 16` plus a few rounds suffices; 48 covers the demo networks.
Build CSR on the GPU with `GPUCOOToCSR`; a reverse CSR gives "cost to reach" isochrones.

```ts
graph.add(new GPUNetworkReachability({
  offsets, neighbors, weights, costs, bands, bandCounts, converged,
  sources: sources.importToGraph(graph), costLimit: costLimit.importToGraph(graph),
  bandThresholds: thresholds.importToGraph(graph), maxIterations: 48
}));
```

### GPUNetworkPathExtraction, GPUNetworkServiceAreas, GPUNetworkNeighborhood, and GPUNetworkAnalyticsColumns

Network analysis over the same directed CSR road network as `GPUNetworkReachability`.

- `GPUNetworkPathExtraction` walks a reachability predecessor array from per-frame targets. It publishes ordered (source to target) node lists and resolved CSR edge lists as `GPUCompactOutput`s, together with per-target offsets, costs, and found flags. Each walk is bounded by a compile-time `maxPathLength`, so cyclic or garbage predecessors terminate and raise `overflow`.
- `GPUNetworkServiceAreas` assigns every node to its nearest per-frame facility. Ties go to the smallest facility row. It also reports per-facility node counts and the total cost each facility serves. It composes `GPUNetworkReachability` with a GPU-gated min-label pass over the tight shortest-path edges.
- `GPUNetworkNeighborhood` publishes k-hop ego networks around per-frame seeds, with per-frame `k` up to a compile-time `maxHops`. Outputs are hop distances, masks, and compact node and induced-edge IDs.
- `GPUNetworkAnalyticsColumns` runs `gpu-graph` degree, PageRank, connected components, core number, and label-propagation communities directly on the caller's CSR views through a `GPUGraphTopologyView`, so a transient CSR built in the same graph by `GPUCOOToCSR` works as well as imported buffers. It publishes node-aligned columns, optionally normalized to `[0, 1]` with a GPU extent.

```ts
graph.add(new GPUNetworkReachability({offsets, neighbors, weights, costs, predecessors,
  sources: origin.importToGraph(graph), maxIterations: 48}));
graph.add(new GPUNetworkPathExtraction({predecessors, costs,
  targets: destination.importToGraph(graph), output: route,
  edges: {offsets, neighbors, weights, output: routeEdges}}));
graph.add(new GPUNetworkServiceAreas({offsets, neighbors, weights, assignments, facilityNodeCounts,
  facilities: stations.importToGraph(graph), costLimit: limit.importToGraph(graph)}));
graph.add(new GPUNetworkNeighborhood({offsets, neighbors, hopDistances, nodes, edges,
  seeds: picked.importToGraph(graph), hops: hops.importToGraph(graph), maxHops: 8}));
graph.add(new GPUNetworkAnalyticsColumns({offsets, neighbors,
  pageRank: {output: rank, normalized: rankSize}, coreNumber: {output: core}}));
```

### GPUNetworkStatistics

Statistics for a graph panel over the same CSR as `GPUNetworkAnalyticsColumns`, reduced on the GPU
into one caller-owned `uint32` summary. It reports live vertex, edge and slot counts, weak component
count and largest component size, isolated vertices, maximum degrees, out-, in- and total-degree
histograms (`'linear'` or `'log2'` bins), and the modularity of a caller-supplied community
labeling. Optional vertex and edge masks restrict every statistic to live rows. Components run
`GPUGraphConnectedComponents` on a masked copy of the adjacency, so masking a bridge vertex splits
its component. Modularity is accumulated with exact integer atomics and a fixed-order f32 finish,
so it is deterministic; a self-loop is counted once, unlike `GPUGraphModularity`. The resolution
and linear bin width are per-frame parameters (`encodeGPUNetworkStatisticsParameters`). Read the
summary back and decode it with `decodeGPUNetworkStatistics`. The recipe owns small scratch
buffers imported under fixed IDs, so one instance belongs to one graph.

```ts
const summary = importGraphBuffer(graph, 'stats', statsBuffer, 'uint32', getGPUNetworkStatisticsLength(32));
graph.add(new GPUNetworkStatistics({offsets, neighbors, vertexMask, communities,
  parameters: statsParameters.importToGraph(graph), degreeBinning: 'log2', output: summary}));
// after submit
const stats = decodeGPUNetworkStatistics(new Uint32Array(await statsBuffer.readAsync()));
```

### GPUNetworkSubgraphFilter

Turns attribute and time predicates on vertex and edge rows into a consistent induced subgraph of a forward CSR network.

- Vertex `v` is live when its caller mask is nonzero and every enabled vertex range accepts its f32 column value. Ranges are half-open (`min <= value < max`); NaN never passes.
- Slot `u -> v` is live when its caller mask is nonzero, every enabled edge range and time gate accepts it, and both endpoints are live. The f32 `edgeTimes` window and the exact Int64 `edgeTimeWords` window are closed (`start <= t <= end`).
- Undirected graphs list each edge in both slots. By default an edge is live only if both slots pass, so `edgeMask` is always symmetric; set `pairUndirectedSlots: false` to skip the O(degree) pairing when columns are already symmetric.
- `dropIsolated` also kills vertices with no live incident slot.
- Ranges live in one f32 `parameters` view (`getGPUNetworkSubgraphFilterParameterValues`) and the time-word window in `timeWordParameters` (`getGPUTimeWindowWordParameterValues`); rewriting them never recompiles.
- Outputs: `vertexMask` and `edgeMask` (1 live, 0 dead; feed `GPUNetworkStatistics` and `filterMask`), `counts` (decode with `decodeGPUNetworkSubgraphFilterCounts`), compact `liveVertices` and `liveEdgeSlots` (stable ascending, overflow-reporting), and an `inducedCSR` (`offsets`, `neighbors` over the original vertex ids, optional `sourceSlots`, `overflow`) built with a prefix sum over live degree.
- Results are exact: integer arithmetic and comparisons only.

```ts
const filter = new GPUNetworkSubgraphFilter({
  offsets, neighbors,
  vertexColumns: [population], edgeColumns: [speed], parameters,
  dropIsolated: true,
  output: {vertexMask, edgeMask, counts}
});
graph.add(filter);
parameterBuffer.write(
  getGPUNetworkSubgraphFilterParameterValues(layout, {vertexRanges: [[1000, Infinity]], edgeRanges: [[30, 90]]})
);
```

### GPUNetworkCoarsening

Summarizes a CSR network by a vertex group label (a community, component or spatial cell) for compound nodes when zoomed out. Supernode outputs: `groupVertexCount` plus optional `groupIntraEdgeCount`, `groupIntraWeight`, `groupCentroid` (`float32x2`), `groupBounds` (`float32x4`) and `groupValueSum`, each with exactly `groupCapacity` rows. Superedges are a bounded `edges` output (`ids` is the source group) with `edgeTargets`, `edgeCounts` and optional `edgeWeights`, sorted by `(source, target)`; undirected superedges have `source < target`. An optional 8-word `summary` reports live and overflowed vertices, counted/dropped/intra/inter edges, group count and the unclamped superedge count.

```ts
const coarsening = new GPUNetworkCoarsening({
  offsets, neighbors, labels, positions, groupCapacity: 1024,
  groupVertexCount, groupCentroid, groupBounds,
  edges: {ids: sourceGroups, count, overflow}, edgeTargets, edgeCounts, edgeWeights
});
graph.add(coarsening);
```

Labels are dense group IDs below the compile-time `groupCapacity` (at most 65535); a live vertex above it is excluded and sets `edges.overflow`. Undirected CSRs list each edge in both directions and each edge counts once. Counts are integer exact. Weight, position and value sums use 64-bit fixed-point atomics (`fixedPointScale`, default 65536), so they are bitwise reproducible. Superedges use a stable sort and segment, so order and the kept prefix on overflow are deterministic. Inputs must be finite.

### GPUAdjacencyMatrix and GPUAdjacencyMatrixOrder

Bins a forward CSR into an `R x R` matrix image for a matrix view linked to a node-link view. Row-major `uint32` counts (`row * R + column`), optional fixed-point weight sums, one-row maxima for color normalization, and an optional `r32float` storage texture. The optional `order` view maps vertex to matrix position (a permutation computed by the caller, for example with `GPUAdjacencyMatrixOrder` or `computeAdjacencyMatrixOrder`; identity by default). Per-frame zoom is a four-word window `[rowStart, rowEnd, colStart, colEnd)` in matrix positions (`encodeGPUAdjacencyMatrixWindow`); position `p` lands in bin `floor((p - start) * R / (end - start))` and positions outside the window are dropped, so panning and zooming never recompile.

```ts
const matrix = new GPUAdjacencyMatrix({
  offsets, neighbors, weights, order, window: windowParameters.importToGraph(graph),
  resolution: 64,
  output: {counts, weightSums, maxCount, maxWeightSum}
});
graph.add(matrix);
windowParameters.write(encodeGPUAdjacencyMatrixWindow({rowStart: 0, rowEnd: 500, colStart: 0, colEnd: 500}, 64));
```

Undirected graphs follow the toolkit convention that the CSR lists both directions, so each slot is written once and the matrix is symmetric; pass `mirrorSlots: true` for an edge-list CSR that lists each edge once and each slot is also written transposed (self-loops once). Directed: row = source. Masks follow the `GPUNetworkStatistics` liveness rule. Counts are exact `u32` atomics. Weight sums are fixed-point `u32` atomics of `round(weight * weightScale)` (default scale 1024), so results are bit-identical regardless of thread order; the cost is half a step of quantization per edge and wraparound at 2^32. `GPUAdjacencyMatrixOrder` builds `order` from a `uint32` group label and an optional tie key with two stable `GPUSort` passes and matches `computeAdjacencyMatrixOrder` exactly.

### GPUNetworkSnapping, GPUNetworkCostMatrix, and GPUNetworkAccessibility

Accessibility over the same directed CSR road network as `GPUNetworkReachability`, in three steps.

- `GPUNetworkSnapping` snaps planar points (origins, facilities, opportunities) to the nearest
  network edge, given node positions and a COO edge list or CSR `offsets`. For each point it
  outputs the edge row (ties go to the smallest row, so a point on a node takes the smallest
  incident edge), the fraction along the edge, the snap distance, the snapped position, and the
  two seed costs: `fraction * edgeCost` to the edge source and `(1 - fraction) * edgeCost` to the
  target. The edge cost defaults to the planar length. `seedNodes` / `seedCosts` (two rows per point)
  feed a search directly. `seedDirection` picks which endpoints count: `'both'` for undirected
  networks, `'forward'` for a point that leaves along a one-way edge, `'reverse'` for a point that
  a one-way edge arrives at. Without `candidateCapacity`, every point scans every edge, which is
  exact. With it, candidates come from `GPUNearestFeatureJoin` over the edge segments within a
  per-frame `maxSnapDistance`, and `overflow` reports when capacity runs out.
- `GPUNetworkCostMatrix` computes a bounded many-to-all cost matrix: one shortest-path search per
  row, from that row's seeds (`seedRows`, or `seedsPerRow: 2` for snapped points) to every node.
  It batches `laneCount` rows (default 32) into one `GPUNetworkReachability` over a lane-expanded
  copy of the CSR. Lane `l`, node `u` becomes node `l * nodeCount + u`, so the frontier rounds,
  hop chaining, per-frame `costLimit`, and convergence flag serve all lanes at once. The matrix is
  bit-identical for every `laneCount`. Rows set the number of searches, so put the smaller side on
  rows. To score every node, put opportunities or facilities on rows and search the reverse CSR
  (an undirected network can reuse its CSR).
- `GPUNetworkAccessibility` scores a retained matrix with per-frame parameters
  (`encodeGPUNetworkAccessibilityParameters`: threshold, decay `'none' | 'exponential' | 'power'`,
  beta, minimum cost for power decay). It outputs cumulative opportunities within the threshold,
  gravity accessibility, and the two-step floating catchment area (`catchment`: supply on rows,
  demand per node, per-facility ratios, and per-node accessibility, with the decay applied inside
  each catchment). `orientation: 'origin-rows'` scores a forward matrix whose rows are origins.
  Per-node sums gather rows in ascending order. Per-row sums use one workgroup and a fixed tree.
  Neither uses float atomics, so results are bitwise reproducible.

Encode the matrix graph when the network or the opportunity set changes. Encode the scoring graph
every frame: threshold, decay, and beta changes never re-run a search. The threshold must not
exceed the `costLimit` the matrix was built with.

```ts
matrixGraph.add(new GPUNetworkSnapping({points: jobs, nodePositions, offsets, edgeTargets: neighbors,
  edgeCosts: weights, snappedEdges, seedNodes, seedCosts}));
matrixGraph.add(new GPUNetworkCostMatrix({offsets: reverseOffsets, neighbors: reverseNeighbors,
  weights: reverseWeights, seedNodes, seedCosts, seedsPerRow: 2, costs: matrix,
  costLimit: limit.importToGraph(matrixGraph), maxIterations: 48}));
scoreGraph.add(new GPUNetworkAccessibility({costs: matrixView, opportunityWeights: jobCounts,
  parameters: scoring.importToGraph(scoreGraph), cumulative, gravity}));
scoring.write(encodeGPUNetworkAccessibilityParameters({threshold: 1800, decay: 'exponential', beta: 0.002}));

### Rendering network recipe outputs with deck.gl

`@deck.gl-community/arrow-layers` renders the network recipes' node-aligned outputs without
copying or reading them back. `GPUGraphNodeLayer` keeps node positions as its only vertex
attribute and binds every other input as a read-only storage buffer indexed by instance:

| Layer prop | Typical recipe output |
| --- | --- |
| `colorColumn` / `sizeColumn` (`{buffer, format: 'uint32' \| 'float32'}`) | `GPUNetworkAnalyticsColumns` normalized degree, PageRank, core number; component or community labels; `GPUNetworkReachability` `bands` |
| `highlightMask` | `GPUNetworkNeighborhood` `nodeMask` |
| `pathRanks` (0 = off path, k = 1-based order) | `GPUNetworkPathExtraction` compact `ids`, scattered by rank |
| `filterMask` (0 hides the row from drawing **and** picking) | any `uint32` node mask |

`colorScale` (`linear` or `categorical`, domain, up to 8 palette stops, `nullColor` for
`0xffffffff`) and `sizeScale` live in a uniform block. Swapping a column, changing a domain or a
mask rebinds or rewrites that block and never rebuilds a pipeline; the block is uploaded only when
its contents change, and `layer.getRenderStats()` reports draws, style uploads and rebindings.
`GPUGraphEdgeLayer` draws an edge only when both endpoints pass `filterMask`, highlights it when both
endpoints are in `highlightMask`, and marks it on the path when both endpoints have consecutive
`pathRanks`.

`GPUGraphRecipeColumns` (owned by `GPUGraphDeckEffect`) wires these recipes onto a symmetrized
topology of the original edge batches and encodes them inside deck.gl's frame encoder: analytics
once, and the neighborhood, reachability and path recipes only when `setHoverVertex`,
`setNeighborhoodHops` or `setPathEndpoints` change their imported input buffers.

The layers require WebGPU and throw on WebGL2: the recipes are WebGPU compute, and WebGL2 cannot
read storage buffers in the vertex stage. Render CPU-side columns with standard deck.gl layers.

### GPUEdgeBundling

Kernel-density edge bundling (KDEEB) as WebGPU compute, producing renderable polylines. Each edge
starts as a straight subdivision. Every iteration does four things:

1. Splats Epanechnikov weights of all control points into a fixed-point `atomic<u32>` density
   buffer.
2. Advects interior points one kernel radius along the bilinearly sampled normalized gradient.
3. Resamples each edge to uniform arc length and applies one Laplacian smoothing pass.
4. Anneals the radius by `lambda`.

Endpoints are pinned. Density is a storage buffer, so there is no float-renderable texture
requirement and no texture size cap.

Lengths are relative to a square work box computed on the GPU every encoding from live-edge
endpoints, so results are scale invariant. Dead edges (masked, out-of-range or non-finite) do not
bundle, and their path collapses onto the source position. Outputs:

- `paths`: `edgeCount * pointsPerEdge` `float32x2` rows, edge-major, in caller coordinates.
- `startIndices` (optional): for PathLayer binary attributes.
- `drawRecord` (optional): a four-word `drawIndirect` record `[pointsPerEdge, edgeCount, 0, 0]`
  for instanced line strips.

`pointsPerEdge` (2 to 64), the maximum `iterations` (1 to 64) and `densityResolution` are
compile-time. The per-frame `parameters` view holds `[activeIterations, kernelRadius, lambda,
smoothing, stepScale]`; pack it with `createGPUEdgeBundlingParameterValues`.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'bundling', format: 'uint32', length: 5,
  values: createGPUEdgeBundlingParameterValues({kernelRadius: 0.03}, 'uint32')});
graph.add(new GPUEdgeBundling({positions, sourceVertices, targetVertices, edgeMask,
  paths, startIndices, drawRecord, pointsPerEdge: 16, iterations: 15, densityResolution: 256,
  parameters: parameters.importToGraph(graph)}));
parameters.write(createGPUEdgeBundlingParameterValues({activeIterations: 6, kernelRadius: 0.05}, 'uint32'));
```

The normalized-gradient schedule is chaotic in f32. The f32 CPU oracle matches below 1e-5 of the
box after one iteration, and only statistically after that. Gradients below four fixed-point
quanta are treated as zero so a lone edge does not drift.

### GPUAttributeCrossfilter

Linked histograms over vertex or edge attribute columns, one recipe per table. Each of up to 32
dimensions is its own `float32` or `uint32` column view, with no packed multi-column buffer
required. Each dimension's histogram counts rows that are live and pass every other dimension's
brush, so brushing one chart updates the others and never itself.

Brushes are `[min, max)`. Brushes, enabled flags and parameter domains live in a per-frame
`float32` view with 8 floats per dimension (`getGPUAttributeCrossfilterParameterValues`), so
brushing never recompiles. `'auto'` domains are the min and max of live, finite rows, computed on
the GPU every encoding, so dead rows never widen a histogram. NaN and infinite values fail their own
dimension. `uint32` columns are binned as f32, which is exact only below 2^24.

Optional outputs are the domains used, selected and live counts, a per-row selection mask, and a
compact `GPUCompactOutput` of selected row IDs.

```ts
graph.add(new GPUAttributeCrossfilter({
  dimensions: [{column: degree, binCount: 32}, {column: rank, binCount: 64},
    {column: year, binCount: 40, domain: 'parameters'}],
  liveMask: timeWindowMask, parameters: brushes.importToGraph(graph),
  histograms, selectedCount, selection
}));
brushes.write(getGPUAttributeCrossfilterParameterValues([{brush: [0.2, 0.6]}, {}, {domain: [1990, 2030]}]));
```


### GPUTerrainDerivatives, GPUTerrainContours, and GPUTerrainViewshed

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

### GPUTerrainHorizon, GPUSolarShadowMask, GPUReliefShading, GPUTextureShading, and GPUSolarPosition

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
2048² tile with 16 sectors). The recipe checks `maxStorageBufferBindingSize`. Pixels closer than
`requiredHalo` to a tile edge see a truncated horizon or blur, so pass tiles with that halo for
seamless mosaics. Typical 2048² timings on an Apple-silicon laptop are about 100 ms for the horizon
(16 sectors, radius 256, growth 1.1), 3.4 ms per frame for the shadow mask, 3.7 ms for relief
shading, and 28 ms for six-level texture shading.

### GPUTerrainFlow

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

### GPUCostDistance and GPUCostDistancePath

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

Generated transient IDs are prefixed with the recipe `id`, for example `<id>-total`. If one
collides with a caller resource, the error names the recipe and the generated ID.

### GPURasterZonalStatistics

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

### GPUDistanceField

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

### GPUInverseDistanceWeighting

Interpolates scattered samples onto a raster with inverse distance weighting (`w = 1 / d^p`),
the GPU counterpart of turf `interpolate`, GDAL `invdist`, and GRASS `v.surf.idw`. The output
extent, search radius, power, nearest-neighbor limit, and minimum neighbor count are per-frame.

```ts
graph.add(
  new GPUInverseDistanceWeighting({
    positions, // float32x2 per sample
    values, // float32 per sample, NaN skips
    mask, // optional uint32, 0 skips
    parameters: idwParameters.importToGraph(graph), // 8 float32
    width: 512,
    height: 512,
    indexGridSize: [128, 128], // internal GPUGridIndex over the samples
    indexBounds: [minX, minY, maxX, maxY], // samples outside are ignored
    maximumNeighborCount: 16, // compile-time capacity for k; 0 compiles radius-only
    output: {values: surface, counts} // float32 (NaN = nodata), optional uint32
  })
);
idwParameters.write(
  getGPUInverseDistanceWeightingParameterValues({
    extent: [x0, y0, x1, y1],
    searchRadius: 250,
    power: 2,
    neighborCount: 8, // 0 = every sample within the radius
    minimumNeighborCount: 3
  })
); // no recompile
```

- Gather form: one invocation per output cell samples the cell center
  `extentMin + (x + 0.5, y + 0.5) * cellSize` (row 0 at `minY`) and visits the grid-index rows
  that overlap the search square. No atomics touch values.
- Samples with `d^2 <= radius^2` contribute. With `neighborCount = k > 0` the `k` nearest are kept
  by `(d^2, row)`, so equal distances keep the smallest row. `k` is clamped to
  `maximumNeighborCount` (at most 64).
- An exact hit (`d == 0`) returns that sample's value, the smallest row among several, regardless
  of `minimumNeighborCount`. Otherwise a cell with fewer than `max(minimumNeighborCount, 1)`
  contributors is nodata (NaN). `counts` holds the contributors after the `k` limit.
- Negative or NaN radius, or a negative or non-finite power, produces an all-NaN raster.
  `searchRadius: Infinity` searches the whole index domain.
- Determinism: the grid index's in-cell order depends on atomics, so the recipe re-sorts IDs inside
  each index cell every encoding; sums then run in a fixed order (index row, column, ascending
  row; or `(d^2, row)` with `k`). Weights are accumulated in log space against a running maximum,
  so tiny distances with large powers do not overflow f32.
- The index is rebuilt every encoding, so positions and values may change per frame. Keep a few
  samples per index cell: the in-cell sort is quadratic in cell population.
- `output.values` is a row-major float32 raster that `GPUTerrainContours` accepts directly as
  `elevation: {id, format: 'float32', storage: {kind: 'buffer', values: surface}}`; NaN cells
  clear their marching-squares cells.

### GPUFocalStatistics

Moving-window statistics over a float32 raster, like ArcGIS `FocalStatistics` and GRASS
`r.neighbors`: mean, sum, min, max, range, population standard deviation, and the count of valid
cells. The radius, window shape, minimum count, and center nodata policy are per-frame.

```ts
graph.add(
  new GPUFocalStatistics({
    values: raster, // row-major float32, NaN = nodata
    validity, // optional uint32, 0 = nodata
    noDataValue: -9999, // optional sentinel
    width,
    height,
    maximumRadius: 16, // compile-time cap, at most 64
    parameters: focalParameters.importToGraph(graph), // 4 float32
    output: {mean, standardDeviation, min, max, count} // any non-empty subset
  })
);
focalParameters.write(
  getGPUFocalStatisticsParameterValues({radius: 3, shape: 'circle', minimumCount: 5})
); // no recompile
```

- Square windows cover `|dx|, |dy| <= floor(radius)`; circles keep offsets with
  `dx^2 + dy^2 <= radius^2`. The radius is clamped to `maximumRadius`.
- Window cells outside the raster and nodata cells are skipped, so edge windows are clipped.
  A cell is nodata (NaN) with fewer than `max(minimumCount, 1)` valid cells, or when its own value
  is nodata and `propagateCenterNoData` is set. `count` is written for every cell.
- Direct gather in fixed row-major window order: deterministic, no atomics. Standard deviation
  subtracts the window mean in a second pass. Moments and extremes are separate kernels, each
  scheduled only when one of its outputs is requested.
- Cost is `(2r + 1)^2` reads per cell; see Status and limitations for the still-open separable and
  summed-area variants.

### GPUPolygonRasterization and GPURasterJoin

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

### GPURasterReclassify, GPUWeightedOverlay, and local raster operations

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

Every recipe is one invocation per cell reading inputs in fixed order, so results are
deterministic per device. Backends may contract `a * b + c` into an FMA (Metal does), so scores
and scaled operands can differ from an unfused CPU evaluation by a few roundings; reclassify,
conditional, extremes, sums, frequencies, and counts match a CPU oracle bit for bit.

### GPURasterStretch

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

### GPUIsolines and GPUIsobands

Marching-squares contours of any float32 raster (terrain, density, interpolation output) with
per-frame levels. Both recipes share one definition, so filled bands and lines drawn together
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

### GPURasterSampling and GPURasterProfile

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

### GPUParticleAdvection

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

### GPULineIntegralConvolution

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

### GPUStreamlines

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

### GPUDotDensity and GPURandomPointsInPolygon

Dasymetric dot-density maps: `value * dotsPerUnit` random dots inside each polygon, per category,
optionally concentrated by a weight raster such as built-up area. Changing the dot value with zoom
only adds or removes dots; existing dots never move, so there is no flicker.
`GPURandomPointsInPolygon` places an integer count of uniform random points per feature.

```ts
const parameters = new GPUParameterBuffer(device, {id: 'dots', format: 'uint32', length: 8});
graph.add(
  new GPUDotDensity({
    polygonPositions, featureOffsets, polygonOffsets, ringOffsets, // GeoArrow, as GPUPointInPolygonJoin
    values, // float32, featureCount * categoryCount rows, feature-major
    categoryCount: 3,
    parameters: parameters.importToGraph(graph),
    mask: {weights, width: 512, height: 512}, // optional, weights in [0, 1]
    output: {positions, dots: {ids, count, overflow, totalCount}, categories, failedCount}
  })
);
parameters.write(
  getGPUDotDensityParameterValues({seed: 1, dotsPerUnit: 1 / peoplePerDot, maskExtent: [x0, y0, cw, ch]})
);
```

- A slot (feature, category) draws `ceil(value * dotsPerUnit - u)` dots with a per-slot Philox
  uniform `u`: `floor(x)` dots plus one more with probability `frac(x)`, monotone in `dotsPerUnit`.
- Dot `j` of a slot is rejection-sampled in the feature's bounding box from Philox candidates keyed by
  `(seed, slot, j, attempt)`, independent of the dot value and of other slots, so coarser dot values
  give a stable prefix of the finer dots. With a mask a candidate is also kept with probability equal
  to its cell weight.
- Dots are ordered by feature, category and rank (offsets from a prefix scan). `dots.ids` holds the
  feature row of each dot and `dots.count` is safe as an instance count; rows past it are NaN /
  `0xffffffff`. A dot whose `maximumAttempts` candidates all miss keeps its slot with a NaN position
  and is counted in `failedCount`.
- Containment is an even-odd test over all rings of the feature (holes excluded); this equals
  multipolygon containment for valid geometry. Cost per candidate is linear in the feature's vertices.

### GPUTileLODSelection

Walks a breadth-level tile hierarchy (bounding spheres, geometric errors, child ranges) once per
level and selects the tiles to draw and to load. Refinement uses projected screen-space error,
optionally relaxed by foveation and focus-distance falloff, and optionally capped by a per-frame
cost and node budget spent deterministically by priority bucket. Resident refined tiles stand in
for missing children; every visible non-resident tile is requested with a priority. Outputs include
drawn IDs, requests, masks, nearest drawn ancestors, statistics, and indirect draw or dispatch
records. `makeGPUTileLODQuadtree` builds a Morton-ordered quadtree hierarchy on the CPU.

```ts
graph.add(new GPUTileLODSelection({
  hierarchy: {sphereBounds, geometricErrors, children, levelOffsets, tileIds},
  view: view.importToGraph(graph), residency: residency.importToGraph(graph),
  output, requests, indirectDraw: {commands: drawCommands.importToGraph(graph)}
}));
view.write(getGPUTileLODViewParameterValues({viewProjectionMatrix, cameraPosition, viewportSize,
  maximumScreenSpaceError: 2}));
```

### GPUFlowAggregation

Origin–destination aggregation for flow maps. Each row's origin and destination are assigned to
square-grid cells, hexagons, or caller zone IDs. Rows are then grouped by zone pair through a
compile-time-capacity `GPUHashIndex`, counted and weighted with `GPUGroupAggregation`, and ranked by
two stable `GPUSort` passes. The result is a deterministic top-K list (weight descending, ties by
origin then destination zone) with zone columns, counts, and weights for an `ArcLayer`, plus
per-zone outgoing and incoming totals. An optional caller mask and per-frame time window
(`getGPUTimeWindowParameterValues`) gate rows without recompiling. `pairOverflow` reports a full
pair table, while `totalCount > ids.length` only means the list was truncated to the top K.

The time gate accepts the same three timestamp forms as `GPUTimeWindowFilter`, including exact
`Int64` words with a `uint32` word window. Weight sums use atomic float addition by default;
`sumOrder: 'sorted'` sorts rows by pair and by zone and sums each group in a fixed tree, so
`flowWeights`, `zoneOutWeights` and `zoneInWeights` are bitwise reproducible, at the cost of
three sorts and scans per encoding (38 instead of 22 nodes).

```ts
graph.add(new GPUFlowAggregation({
  zones: {kind: 'hexagon', bounds: viewport.importToGraph(graph), gridSize: [64, 48], radius: 250},
  origins, destinations, weights: tripCounts,
  timeWindow: {timestamps: departures, window: window.importToGraph(graph)},
  pairCapacity: 8192, excludeSelfFlows: true,
  output: {ids: flowKeys, count, overflow, totalCount},
  flowOriginZoneIds, flowDestinationZoneIds, flowWeights, zoneOutCounts, zoneInCounts,
  drawInstanceCount: graph.importGPUData('arcs', arcDraws.getInstanceCountData(0))
}));
```

### GPUSpatialClustering

DBSCAN density clustering of planar points. Bounds, `epsilon`, and `minimumPoints` are per-frame
parameters (`getGPUSpatialClusteringParameterValues`). `gridSize` is only a compile-time maximum
lattice. Each encoding derives an active lattice whose cells are at least `epsilon` wide, so the
3×3 neighborhood search is always exact and `epsilon` changes never recompile. Clusters are joined
by a lock-free union-find that hooks larger roots under smaller ones. Labels are therefore
canonical: compact cluster IDs follow the smallest core row of each cluster, and a border point
joins the adjacent cluster with the smallest root. Outputs are per-point labels, roots, and core
flags, the cluster count, a compact list of cluster representatives, and cluster sizes and
centroids.

```ts
graph.add(new GPUSpatialClustering({
  positions, parameters: parameters.importToGraph(graph), gridSize: [128, 128],
  labels, clusterCount, clusters: {ids: clusterRoots, count, overflow}, clusterSizes, clusterCentroids
}));
parameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: 25, minimumPoints: 8}));
```

Centroid sums use float atomics by default. Because labels are canonical, `sumOrder: 'sorted'`
(stable sort of members by cluster, then a fixed-order segmented sum) makes `clusterCentroids`
bitwise reproducible across encodings on one device; cluster sizes are exact either way.

### GPUHotSpotAnalysis and GPULocalMoran

Local spatial autocorrelation over planar points with a distance-band weight. Row `j` is a neighbor
of row `i` when their distance is at most `radius`. The weights are never materialized: each
encoding builds a cell index whose cells are at least `radius` wide and gathers every focus row's
3×3 cell neighborhood, so `radius`, bounds, and every other parameter in the shared
`getGPUSpatialAutocorrelationParameterValues` layout change per frame without a rebuild.
`gridSize` only caps the lattice.

- `GPUHotSpotAnalysis` writes the Getis-Ord Gi* z-score per row. The row itself counts as a
  neighbor and weights are binary. The formula is the ArcGIS Hot Spot Analysis one, evaluated on
  centered values. Optional outputs are the two-sided p-value, the neighbor count, and a `sint32`
  confidence bin in `-3..3` (99/95/90% hot or cold). With `falseDiscoveryRate: true` the bins use
  Benjamini-Hochberg correction at 0.10, 0.05, and 0.01.
- `GPULocalMoran` writes the z-score of local Moran's I. The analytic mean and variance are those of
  conditional randomization, the null that esda's permutation test samples. Optional outputs are
  `localI` (esda scaling, row-standardized or binary weights), the spatial lag, the p-value, the
  neighbor count, and the esda quadrant code (1 HH, 2 LH, 3 LL, 4 HL; 0 when not significant at the
  per-frame `significanceLevel`, optionally with BH-FDR).

```ts
graph.add(new GPUHotSpotAnalysis({
  positions, values, mask, parameters: parameters.importToGraph(graph), gridSize: [256, 256],
  zScores, bins, pValues, globalStatistics
}));
parameters.write(getGPUSpatialAutocorrelationParameterValues({bounds, radius: 500}));

### GPUNeighborSearch and GPUSpatialWeights

`GPUNeighborSearch` finds exact k nearest neighbors (`mode: 'knn'`, `k` up to 32) or every point
within a distance band (`mode: 'radius'`) between planar points. A self join never lists a row as its
own neighbor; pass `queryPositions` for a cross join. It writes a `GPUSpatialWeights` CSR:
`offsets` (`queryRows + 1`), `neighbors` (the slot capacity), `weights`, and optional `distances`.
This is the weights structure that `GPUGlobalSpatialStatistics`, `GPULocalPermutationTest` and
`GPUGlobalPermutationTest` read. Applications can also upload one built elsewhere, for example
polygon contiguity.

- Rows list neighbor IDs in strictly ascending order with distances and weights aligned, so
  consumers can binary-search `w_ji`.
- kNN keeps the `k` smallest `(d^2, id)` pairs, so ties go to the lowest ID. An optional
  `radius` bounds the search. Fewer valid targets give a shorter row.
- Weights: `binary`, `inverseDistance` (`max(d, distanceFloor)^-power`), or `kernel` (gaussian,
  triangular, epanechnikov, bisquare, uniform). The bandwidth is `radius` in radius mode and the
  row's k-th neighbor distance in kNN mode (PySAL's adaptive bandwidth). `rowStandardize` divides
  each row by its sum. Non-finite weights are written as 0.
- Capacity: offsets are clamped to `neighbors.length`, so readers always see consistent, possibly
  truncated rows. `overflow` is set on the GPU, and `totalNeighbors` and `neighborCounts` stay
  unclamped.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'knn', format: 'float32', length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  values: getGPUNeighborSearchParameterValues({bounds, rowStandardize: true})
});
graph.add(new GPUNeighborSearch({
  mode: 'knn', k: 8, gridSize: [256, 256], positions,
  parameters: parameters.importToGraph(graph),
  weights: {offsets, neighbors, weights, distances}, overflow
}));
```

Targets and queries count only when their mask is nonzero, their coordinates are finite, and they
lie inside the per-frame `bounds`. A self join applies `mask` to the query rows too. Bounds,
radius, the weight function and row standardization are per-frame parameters, so changing them
never rebuilds the graph. Targets are bucketed by a stable sort into a per-frame lattice. kNN
searches expanding cell rings with a private sorted top-k list and stops once the k-th distance is
strictly inside the visited box, with a slack for f32 rounding. Radius mode counts, scans, emits, and
insertion-sorts each row by ID. No float atomics are used, so outputs are bitwise reproducible.

### GPUGlobalSpatialStatistics

Global spatial autocorrelation from a `GPUSpatialWeights` CSR and a value column, each statistic with
its analytic expectation, variance, z-score and two-sided normal p-value. Pick the statistics with
`statistics` (compile-time). The `results` layout (`GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`) is fixed,
and blocks that were not requested hold NaN.

- `moran`: Moran's I with esda's `VI_norm` and `VI_rand` (sample kurtosis).
- `geary`: Geary's C with `VC_norm` and `VC_rand`.
- `getisOrdG`: Getis-Ord General G (values should be non-negative). Its randomization variance
  equals esda's `VG`. It is evaluated in a centered form so that f32 does not lose it to the
  cancellation in `E[G^2] - E[G]^2`.
- `bivariateMoran`: `(n / S0) sum w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`, which equals esda's
  `Moran_BV` for row-standardized weights. The exact variance under randomization of `y` is
  `E = 0`. esda itself offers only permutation inference.
- `joinCount`: BB, BW and WW joins of a binary column (nonzero is black) over binarized weights.
  Counts are exact integers, also available as ordered-pair `joinCounts`. Cliff-Ord randomization
  moments are given for BB and BW.

```ts
graph.add(new GPUGlobalSpatialStatistics({
  weights, values, statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'], results, joinCounts
}));
// results[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization]
```

Weights are used as given, so apply row standardization in `GPUNeighborSearch` if wanted. A row is
included when the optional mask is nonzero and every provided value is finite. Excluded rows are
neither foci nor neighbors, and `n`, `S0`, `S1` and `S2` cover included pairs only. The summary block
reports `n`, `S0`, `S1`, `S2`, the mean, the variance, the black count and the island count.
`w_ji` comes from a binary search of row `j`. Column sums come from a deterministic transpose: a
stable sort of slots by neighbor. Per-row partials are reduced by fixed-shape workgroup trees, and
join counts by u32 atomics. Results are bitwise reproducible, and values, weights and the mask
change per frame without a rebuild. Inference is NaN when `n < 4`, `S0 = 0`, or the values are
constant.

### GPULocalPermutationTest and GPUGlobalPermutationTest

Permutation inference (pseudo p-values) over a `GPUSpatialWeights` CSR, matching PySAL esda's
`p_sim`. Both recipes read a uint32 parameter view written with `getGPUPermutationParameterValues`:
`{seed, permutations, significanceLevel}`. Changing the seed, the permutation count (up to the
compile-time `maximumPermutations`), the values or the weights never rebuilds the graph.

- `GPULocalPermutationTest` does conditional randomization for `localMoran`, `localG` or
  `localGStar` (a self weight of 1). For each tested row and permutation it draws an ordered sample
  of `k_i` distinct other included rows (a partial Fisher-Yates over their compacted positions) and
  assigns them to the row's weight slots. It counts simulated values `>= observed`, folds the count
  as esda does, and writes `exceedances`, `pseudoPValues = (count + 1) / (P + 1)`, an optional
  `observed` statistic, and a `significant` mask (`p <= level`, or Benjamini-Hochberg with
  `falseDiscoveryRate: true`). Islands and rows with more than `maximumNeighbors` (64 at most)
  neighbors are not tested; the latter set `overflow`.
- `GPUGlobalPermutationTest` handles `moran`, `geary`, `getisOrdG` and `bivariateMoran` (`y`
  permuted). Each permutation relabels the included values through a keyed Feistel bijection, so no
  sort runs per permutation and nothing of size `P x n` is stored. `results`
  (`GPU_GLOBAL_PERMUTATION_RESULT`) holds the observed statistic, `p_sim`, the folded count, the
  simulated mean and standard deviation, `z_sim`, `p_z_sim`, and the minimum and maximum. Optional
  outputs are the `referenceDistribution` and its `histogram` (GeoDa's permutation plot).

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'permutations', format: 'uint32', length: GPU_PERMUTATION_PARAMETER_LENGTH,
  values: getGPUPermutationParameterValues({seed: 1, permutations: 99})
});
graph.add(new GPULocalPermutationTest({
  weights, values, statistic: 'localMoran', parameters: parameters.importToGraph(graph),
  maximumPermutations: 999, exceedances, pseudoPValues, significant, falseDiscoveryRate: true, overflow
}));
// Preview with 99 permutations, then refine on idle:
parameters.write(getGPUPermutationParameterValues({seed: 1, permutations: 999}));
```

Random numbers come from Philox 4x32-10 counter streams, keyed by the 64-bit seed with the counter
`(block, row, permutation, tag)`. Results are therefore a pure function of the seed and inputs,
independent of the dispatch shape. The observed and simulated statistics run through the same f32
code, positive constant factors are dropped from the comparisons, and exceedances are integers. The
significance and BH tests compare correctly rounded f32 products, never quotients. Global pair sums
use fixed-shape workgroup trees. Every output is bitwise reproducible for a seed.

### GPUVariogram, GPUSpatialCorrelogram, GPURipley, and GPUPointPatternIndices

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
- `GPUPointPatternIndices`: exact nearest neighbour per row (ring search, ties to the smallest row),
  Clark-Evans `[n, observed, expected, R, standardError, z]`, and quadrat counts with
  `[m, mean, variance, varianceToMeanRatio, chiSquare, degreesOfFreedom]`.

Measured on an Apple M3 Pro (headless Chromium), `GPUVariogram` with 32 lags including the cell
sort: 100k points with 54M pairs in range in 10 ms; 20k points all-pairs (200M pairs) in 12 ms.

### GPUGeographicDistribution

`GPUGeographicDistribution` computes the ArcGIS "Measuring Geographic Distributions" statistics
over planar points, per group or per masked selection: counts, weight sums, mean centre, Weiszfeld
median centre, standard distance, standard deviational ellipse, linear directional mean, and
polygon rings for the ellipse and the standard-distance circle. Project longitude/latitude first:
every statistic is Euclidean.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'distribution-parameters',
  format: 'float32',
  length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
});
graph.add(
  new GPUGeographicDistribution({
    positions, weights, groupIds, groupCount: 8, mask,
    parameters: parameters.importToGraph(graph),
    output: {meanCenters, ellipses, ellipseVertices}
  })
);
parameters.write(
  getGPUGeographicDistributionParameterValues({origin: [x0, y0], standardDeviations: 2})
);
```

- Inputs: `positions` (`float32x2`, fewer than 2^24 rows), optional `weights` (a non-finite,
  negative or zero weight excludes the row), `groupIds` with compile-time `groupCount` (ids at or
  above it are excluded), `mask`, and `lineEnds` for the directional mean.
- Per-frame parameters (no recompile): `origin` (local origin subtracted before every sum for
  float32 stability), `standardDeviations`, `ellipseConvention` (`'arcgis'` scales the axes by
  `sqrt(2)`, `'standard'` does not), `orientationOnly`, `medianTolerance`.
- Divisors are the weight sum, as in ArcGIS (not `n - 1`). Second moments are central (a mean
  pass, then a pass about the mean), never `E[x^2] - E[x]^2`.
- `ellipses` holds `[angle, sigmaX, sigmaY]` per group. As in ArcGIS the "y" axis is the long one
  (`sigmaY >= sigmaX`). `angle` is counter-clockwise from +x of the short x axis, so the long axis
  points at `angle + pi / 2`; a ring vertex is `M + R(angle) (sigmaX cos t, sigmaY sin t)`.
- `directionalMeans` holds `[meanAngle, circularVariance, meanLength]` (angles counter-clockwise
  from +x). With `orientationOnly`, angles are doubled before averaging and the mean angle lies in
  `(-pi/2, pi/2]`.
- `ellipseVertices` and `circleVertices` are counter-clockwise rings of `polygonVertexCount`
  (default 64) vertices per group, not closed, ready for a deck.gl `PolygonLayer` or `PathLayer`.
- The median centre runs `medianIterations` (default 24) fixed Weiszfeld steps from the mean centre
  (six nodes per step); `medianConverged` is 1 when the last step moved at most `medianTolerance`.
- Empty groups report 0 counts and weight sums and NaN in every other output.
- Determinism: rows are sorted by group once and every sum, including each Weiszfeld step, is a
  fixed-order segmented tree sum. No float atomics; repeated encodings are bitwise identical.

### GPUEmergingHotSpots

ArcGIS-style emerging hot spot analysis over a dense space-time cube on a regular lattice.
`values` is indexed `cell * sliceCount + slice` (the `GPUTemporalReduction` layout, `cell = row *
gridWidth + column`); a `float32` view treats NaN as a missing bin, a `uint32` view (for example
`GPUTemporalReduction` `counts`) is read as `f32(count)`. An optional per-cell `mask` excludes cells.

```ts
const hotSpots = new GPUEmergingHotSpots({
  values, gridWidth: 64, gridHeight: 64, sliceCount: 24, maximumRadius: 4,
  parameters, giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount
});
graph.add(hotSpots);
parameterBuffer.write(
  getGPUEmergingHotSpotParameterValues({radius: 2, temporalWindow: 3, confidenceLevel: 0.95})
);
```

Passes (deterministic, no float atomics):

1. Global `n`, mean and standard deviation of valid bins: fixed 1024-bin block sums added in order.
2. Space-time Gi* per bin (binary weights, focal bin included): neighbors are valid bins in cells
   with `dx^2 + dy^2 <= radius^2` (lattice offsets) over the current and `temporalWindow` previous
   slices, `z = sum(x - X) / (S sqrt((n k - k^2) / (n - 1)))`.
3. Per-cell Mann-Kendall on the finite z series: exact integer `S`, tie-corrected variance
   `[n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18` (ties by exact f32 equality), continuity-corrected z, and a
   two-sided p-value from the Numerical Recipes `erfcc` fit.
4. Classification into the 17 `GPU_EMERGING_HOT_SPOT_CATEGORIES` (0 = no pattern, hot 1..8, cold 9..16:
   new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical).

Per-frame parameters (radius, temporal window, critical z or confidence level, trend significance,
persistence fraction) live in a parameter buffer and never rebuild the graph. Compile-time:
`gridWidth`, `gridHeight`, `sliceCount` (at most 256), `maximumRadius` (at most 32), value format,
optional views. Outputs are caller-owned: `giZScores` (per bin), `trendZ`, `trendP`, `trendS`,
`category`, `hotSliceCount`, `coldSliceCount` (per cell) and optional `globalStatistics`
`[n, mean, variance, standardDeviation]`.

### GPUOrdinaryLeastSquares

Ordinary least squares with an intercept and a full regression report, in one prebuilt graph:
coefficients, standard errors, t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC,
BIC, optional per-row residual and fitted columns, the Jarque-Bera normality test and the
Breusch-Pagan (Koenker) heteroskedasticity test.

```ts
const graph = new GPUCommandGraph(device);
graph.add(
  new GPUOrdinaryLeastSquares({
    predictors, // GraphDataView<'float32'>, row-major row * predictorCount + column
    response, // GraphDataView<'float32'>
    mask, // optional GraphDataView<'uint32'>; zero excludes the row
    parameters: ridgeBuffer.importToGraph(graph), // optional: [ridgeLambda]
    predictorCount: 3, // compile-time, 1 to 15; the intercept is added internally
    output: {coefficients, standardErrors, tStatistics, summary, status, residuals, fitted}
  })
);
```

- Rows with any non-finite predictor or response value, or a zero mask, are excluded.
- `coefficients`, `standardErrors`, `tStatistics`: `predictorCount + 1` floats, intercept first.
- `summary` (16 floats), indexed by `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*`: row count, R2, adjusted
  R2, sigma squared, log-likelihood, AIC, BIC, Jarque-Bera, its p-value, Breusch-Pagan, its p-value,
  RSS, TSS, skewness, kurtosis, ridge lambda.
- `status` (uint32): 0 ok, 1 singular or ill-conditioned (a predictor within `1 - 1e-6` of the span
  of the others, or constant), 2 too few rows (`n <= predictorCount + 1`). A failed fit leaves every
  statistic and per-row column NaN (the row count is still written).
- `residuals` and `fitted` are optional per-row columns (NaN for excluded rows).
- The optional ridge penalty `lambda >= 0` is added to the centered normal matrix of the
  predictors (never the intercept) and changes between frames without rebuilding the graph. Ridge
  standard errors are `sigma2 (X'X + lambda I)^-1` and ignore the shrinkage bias.

Numerics: no float atomics. Per-tile partial sums (one invocation per tile of consecutive rows, in
row order) are merged in tile order with Kahan compensation; means first, then centered
cross-products about those means; the solve runs on the correlation-scaled normal matrix with a
single-invocation Cholesky. The residual sum of squares comes from the residual pass, not from
`Syy - b'Sxy`, so it stays accurate when R2 is close to 1. Results are bitwise reproducible for a
given `tileRowCount` (default `max(64, ceil(rows / 4096))`).

AIC and BIC count `predictorCount + 1` parameters (GeoDa convention); the log-likelihood is
Gaussian with `sigma2 = RSS / n`. Jarque-Bera is `n/6 (S^2 + (K - 3)^2 / 4)` with p-value
`exp(-JB / 2)`. Breusch-Pagan is the Koenker studentized statistic `n R2` of the regression of `e^2`
on the predictors, with a chi-square p-value on `predictorCount` degrees of freedom computed by
the regularized upper incomplete gamma with a fixed iteration cap. `getChiSquareSurvival(value,
degreesOfFreedom)` is the float64 CPU mirror.

### GPUGeographicallyWeightedRegression

Geographically weighted regression with AICc bandwidth selection (Fotheringham, Brunsdon and
Charlton 2002; mgwr, ArcGIS GWR). For every included row `i` it fits a weighted least squares
`y ~ 1 + x` over all included rows `j` with weights `K(d_ij / h)` and writes local coefficients,
local R-squared, fitted values, residuals and the hat-matrix diagonal.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'gwr-parameters',
  format: 'float32',
  length: getGPUGeographicallyWeightedRegressionParameterLength(16)
});
graph.add(
  new GPUGeographicallyWeightedRegression({
    positions, // GraphDataView<'float32x2'>, planar
    predictors, // row-major row * predictorCount + column
    predictorCount: 2,
    response,
    parameters: parameterBuffer.importToGraph(graph),
    maximumBandwidthCount: 16,
    output: {coefficients, selectedBandwidth, summary}
  })
);
parameters.write(
  getGPUGeographicallyWeightedRegressionParameterValues(
    {kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [20, 40, 60, 80]},
    16
  )
);
```

- Kernels: `'gaussian'` `exp(-0.5 (d/h)^2)` and `'bisquare'` `(1 - (d/h)^2)^2` for `d < h`.
- Bandwidths: `'fixed'` (ladder values are distances) or `'adaptive'` (ladder values are neighbour
  counts `k`, `h` is 1.00001 times the distance to the k-th nearest included row, the row itself
  counted as in mgwr; `k` in `[2, maximumNeighborCount]`, at most 128).
- Bandwidth ladder: up to `maximumBandwidthCount` (at most 32) candidates in the parameter buffer.
  Each candidate's AICc, `n ln(RSS/n) + n ln(2 pi) + n (n + tr S) / (n - 2 - tr S)`, is evaluated on
  the GPU, the lowest wins (ties to the lowest index) and the outputs are written at that
  bandwidth. A ladder of one value is a fixed bandwidth. Candidates with any singular location,
  invalid `k` or `n - 2 - tr S <= 0` score NaN.
- Outputs: `coefficients` (`row * (predictorCount + 1) + column`, column 0 intercept), optional
  `localR2`, `fitted`, `residuals`, `hatDiagonal`, `localStatus` (0 ok, 1 singular, 2 excluded),
  `bandwidthScores`, `selectedBandwidth` (`[index, value]`) and `summary` (RSS, trace of S, AICc,
  global R-squared, n, has-valid-candidate flag).
- Rows with a zero mask or a non-finite position, predictor or response are excluded as
  calibration locations and as neighbours.
- Numerics: the design is centred on the focal row and Jacobi-equilibrated before the shared
  Cholesky solve, so large coordinates or badly scaled predictors stay well conditioned in f32.
  A predictor that is constant over a neighbourhood makes that location singular.
- Cost: brute force, one thread per location scanning every row per candidate:
  `O(n^2 * ladder * p^2)`, at most 65536 rows (about 80 ms for 4096 rows and 8 candidates on the
  test adapter). Reductions use fixed-order 256-row tiles, so results are bitwise reproducible.

### GPUCompositeScore

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

### GPUInequality

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

### GPUTrajectoryMetrics

Per-track metrics for rows sorted by track and time and delimited by `trackOffsets`: planar length,
duration, average speed, and maximum speed (via `GPUSegmentedReduction`). Stop detection finds
maximal runs of slow steps (speed below a per-frame threshold) that last at least a per-frame
minimum duration. It publishes a bounded stop list in row order (track, start row, end row,
duration, centroid) and unclamped per-track stop counts.

`timestamps` may be `float32`, `float32` plus `timestampsLow` (double-single), or exact `Int64`
words (`uint32x2`). Every time difference (steps, track durations, dwells) is taken exactly or in
double-single before conversion to float32, so multi-year epoch-millisecond tracks keep
millisecond steps. Durations, speeds and the stop parameters use the unit of the timestamps.

```ts
graph.add(new GPUTrajectoryMetrics({
  positions, timestamps, trackOffsets, parameters: stopParameters.importToGraph(graph),
  trackLengths, averageSpeeds, maximumSpeeds, trackStopCounts,
  stops: {output: {ids: stopTracks, count, overflow}, startRows, endRows, centroids, durations}
}));
stopParameters.write(getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 0.5, stopMinimumDuration: 300}));
```

### GPUTrajectoryPlayhead and GPUTrajectoryResample

`GPUTrajectoryPlayhead` interpolates every track at a per-frame playhead: one invocation per track
binary-searches the track's sorted timestamps and interpolates the bracketing segment. It writes
per-track columns (position, optional elevation, heading in radians, speed, status, segment row and
fraction) and an optional compact, ascending list of active tracks with a clamped count, so a
layer can draw one instance per moving object with an indirect instance count. Moving the playhead
only rewrites the parameter buffer.

```ts
graph.add(new GPUTrajectoryPlayhead({
  positions, timestamps, trackOffsets, // same layout as GPUTrajectoryMetrics
  elevations, // optional float32 z per row
  parameters: playheadParameters.importToGraph(graph),
  currentPositions, currentElevations, headings, speeds, status, segmentRows, segmentFractions,
  activeTracks: {ids, count, overflow}, drawInstanceCount
}));
playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead: 125.5, maxGap: 60}));
// Int64 epoch-ms words: a uint32 parameter buffer and
// getGPUTrajectoryPlayheadWordParameterValues({playhead: Date.now() + 0.25, maxGap: 60_000})
```

- Status (`GPU_TRAJECTORY_PLAYHEAD_STATUS`): `empty` (no rows), `active`, `beforeStart`,
  `afterEnd`, and `gap` (the playhead lies strictly inside an interval longer than `maxGap`; 0
  disables the rule). Outside the range the position clamps to the first or last sample and the
  heading is that of the first or last segment; in a gap the position holds at the last fix.
  Speed is non-zero only for active tracks.
- The segment is `[j - 1, j]` with `t[j - 1] <= p < t[j]`, or the last segment when `p` equals the
  last time. A playhead exactly on a sample time returns that sample's position bit-exactly.
- Duplicate timestamps: the last row of a run of equal times wins ("latest row wins"), so the
  interpolation interval always has a positive duration and nothing divides by zero.
- Heading is `atan2(dy, dx)` (0 along +x, counterclockwise); compass bearing is `π/2 - heading`.
  Zero-length segments give 0. Positions are planar; project upstream (no antimeridian handling).
- Time: comparisons are exact in both modes (Int64 words compare integer words, then the f32
  playhead fraction); differences are subtracted exactly and then rounded to f32.

`GPUTrajectoryResample` resamples every track to `sampleCount` samples, uniformly in time
(`spacing: 'time'`, default) or along the planar path (`'arc-length'`), into a dense
`[trackCount × sampleCount]` `float32x2` buffer (row `track * sampleCount + k`), with optional
elevations and sample times relative to the track start. It is the fixed-length input for
trajectory similarity, k-means, and fixed-vertex trails.

```ts
graph.add(new GPUTrajectoryResample({
  positions, timestamps, trackOffsets, sampleCount: 32, spacing: 'arc-length',
  samples, sampleElevations, sampleTimes
}));
```

- Sample `k` targets `total * k / (sampleCount - 1)`; the last sample is the last row exactly. It
  uses the playhead's search and duplicate rule. Single-row tracks repeat that row; empty tracks
  write zeros.
- Arc length sums step lengths sequentially per track (deterministic, O(track length) per
  invocation).
- Inputs must be single packed views; chunked vectors are not supported.

### GPULineSimplification

Douglas-Peucker simplification of many polylines or tracks, stored as rows sorted by line and
delimited by `trackOffsets` (the `GPUTrajectoryMetrics` layout). It runs in two parts: a
per-vertex `importance` column computed once, and a per-frame selection at a tolerance from a
parameter buffer. Changing the tolerance re-runs only the mask, compaction, and publish nodes.

```ts
// One-shot graph: importance.
importanceGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance,
  metric: 'segment', // or 'time-ratio' (TD-TR) with float32 `timestamps`
  maximumRounds: 64,
  status: {converged, roundCount}
}));

// Per-frame graph: selection only.
frameGraph.add(new GPULineSimplification({
  positions, trackOffsets, importance, computeImportance: false,
  parameters: toleranceParameters.importToGraph(frameGraph),
  selection: {output: {ids, count, overflow}, keepMask, lineCounts, lineStarts}
}));
toleranceParameters.write(getGPULineSimplificationParameterValues({tolerance: 2})); // no recompile
```

- Importance: every line starts as one open interval between its endpoints (importance
  `+Infinity`). Each round, every open interval picks its farthest row (segmented argmax with ties
  to the smallest row) and splits there. The row's importance is
  `min(distance, min(importance[a], importance[b]))`, so importance never increases from a split
  to its children. An interval whose largest distance is 0 is resolved at once with importance 0.
- Rounds are GPU gated. They stop when no row is left (`converged = 1`) or after `maximumRounds`.
  A balanced split tree takes about `log2(n)` rounds and a spiral can take up to `n - 2`. If the
  cap is reached, the remaining rows get their interval's split importance, so the kept set is a
  superset of Douglas-Peucker's.
- Selection: a row is kept when `importance > tolerance`. `output.ids` holds the kept rows in
  ascending order, and `count` is clamped so it can drive an indirect draw. `lineCounts` (not
  clamped) and `lineStarts` give each line's range inside `output.ids`.
- Exactness: with `converged = 1`, the kept set at any tolerance `e` equals classic recursive
  Douglas-Peucker at `e` (split when `dmax > e`, first maximum on ties) under the same metric.
  The f32 distances are bit-identical to a `Math.fround` CPU oracle: products are prevented from
  fusing into FMAs, and division and square root are corrected to defined values.
- Metrics: `'segment'` measures the distance to the chord segment, which is the perpendicular
  distance inside the segment and the distance to the nearer anchor outside it (GEOS and
  simplify-js behave the same way). `'time-ratio'` is TD-TR's synchronized Euclidean distance,
  which needs float32 `timestamps`.
- Use tile- or view-local coordinates. Simplified lines may self-intersect, because topology is
  not preserved.

### GPUTemporalReduction

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

### GPUCalendarBuckets

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

### GPUChangeDetection

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

## Tiled and streamed rows

### GPUResidencyArena and GPUResidentRowSelection

Runs recipes over tiled and streamed rows without changing the graph. `GPUResidencyArena` owns one
fixed-capacity buffer per column plus a live mask and a per-row tile slot, and pages tiles in and
out (`insertTile`, `evictTile`, `replaceTile`) with queue writes only: no buffer, pipeline, or graph
is created during churn. `importToGraph(graph)` returns one packed view per column, so every existing
recipe runs over arena columns unchanged and its node count does not depend on how many tiles are
resident. Pages are fixed-size and need not be contiguous, so fragmentation never blocks an insert;
`ResidencyArenaAllocator` is the CPU allocator and oracle (`resolveRow`, `getLiveRows`). Which tiles
to evict is the application's policy (for example relative to a playhead or a request priority).

Dead rows must not reach a recipe. Pass `liveMask` wherever a recipe takes a mask or predicate
(`GPUTimeWindowFilter.additionalPredicates`, `GPURegionStatistics` `selection: {kind: 'mask'}`,
`GPUPointDensity` `mask`), and give columns read by mask-less recipes a `deadValue`. Otherwise
stale rows are counted, including in automatic histogram domains.

`GPUResidentRowSelection` is the arena's mask, compaction, and indirect-draw step: live mask, an
optional per-tile gate (`tileMask` indexed by tile slot), and optional predicates reduce to bounded
stable IDs and an indirect `instanceCount`.

```ts
const arena = new GPUResidencyArena(device, {rowCapacity: 1 << 20, columns: [
  {name: 'positions', format: 'float32x2', deadValue: NaN},
  {name: 'times', format: 'float32'}
]});
const views = arena.importToGraph(graph);
graph.add(new GPUTimeWindowFilter({
  timestamps: views.columns.times, window: window.importToGraph(graph),
  additionalPredicates: [{kind: 'selection', mask: views.liveMask}],
  output, drawInstanceCount: graph.importGPUData('count', drawCommands.getInstanceCountData(0))
}));
const compiled = graph.compile();
arena.insertTile('7/65/42', {rowCount, columns: {positions, times}}); // no recompile
```

Over 1,048,576 rows on an Apple M3 Pro, the arena graph stays at 10 nodes for any tile or chunk
count, while `GPUCompaction` over a 32-chunk `GraphVectorView` emits 1,249; time mask, compaction,
and publish take about 0.8 ms per queued encoding against 120 ms for the chunked graph.


## Status and limitations

These are initial versions. Inputs are planar `float32` coordinates; projection is done upstream
(for example with [`GPUProjection`](./gpu-project.md)). Hierarchies and CSR graphs use packed views
only; tiled and streamed rows go through `GPUResidencyArena`. Each recipe's TSDoc lists its non-goals; the GPU Core maintainer roadmap
(`dev-docs/roadmaps/gpugraph-roadmap.md`) tracks what is still open.
