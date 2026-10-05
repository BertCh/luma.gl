import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';
import {SpatialBenchmark} from '@site/src/components/docs/spatial-benchmark';

# WebGPU Geospatial Kernels

<ExperimentalDocsTabs active="geospatial" />

The `@luma.gl/experimental/geospatial` entry point provides small, side-effect-free WebGPU
algorithms that add compute nodes to a `GPUCommandGraph`. This first set includes fixed-output
projection and distance kernels plus a flat grid index and point-query workflow:

- `GPUSinusoidalProjection`
- `GPUHaversineDistance`
- `GPUPairwisePointDistance`
- `GPUPairwisePointSegmentDistance`
- `GPUPairwisePointInPolygon`
- `GPUPairwisePointLinestringNearest`
- `GPUGridIndex`
- `GPUPointSpatialQuery`

The entry point also exports [analysis contributors](#analysis-contributors) for density, joins,
spatial statistics, line geometry, and trajectories.

These classes structurally implement `GPUCommandGraphContributor`. Calling `addToGraph()` declares
work, but does not compile the graph, submit commands, allocate caller-visible outputs, or read
results back. Projection and simple pairwise distance kernels accept caller-allocated
`GraphDataView` objects or matching `GraphVectorView` objects. The nested-offset point-in-polygon
and nearest-linestring APIs, grid index, and point query consume fixed-width flat `GraphDataView`
objects in V1.

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUHaversineDistance} from '@luma.gl/experimental/geospatial';

const graph = new GPUCommandGraph(device);

new GPUHaversineDistance({
  left: pickupLocations,
  right: dropoffLocations,
  output: distances
}).addToGraph(graph);

const compiled = graph.compile();
```

The geospatial entry point is intentionally separate from the experimental root and standalone
bundle. Importing the subpath is the explicit opt-in.

## When to use it

Use the fixed-output kernels (`GPUSinusoidalProjection`, `GPUHaversineDistance`, the pairwise
kernels) when the caller already owns the paired input and output views. Use `GPUGridIndex` with
`GPUPointSpatialQuery` for one query region per frame: bounds, a radius, or a polygon that returns
the matching IDs, with work proportional to the intersected cells.

Use the [analysis contributors](#analysis-contributors) when the question is about many rows at
once or about statistics over them:

- `GPUNeighborSearch` answers a neighbor question for every point: the k nearest or all targets in a
  distance band, as a `GPUSpatialWeights` CSR with distances and weights that the spatial statistics
  read. `GPUPointSpatialQuery` answers one region per frame and returns IDs only, so use it for
  selection and filtering, and `GPUNeighborSearch` for kNN, weights, and clustering inputs.
- `GPUPointInPolygonJoin` builds a `GPUBVH` over feature bounds and refines candidates with
  `GPUPairwisePointInPolygon`, so it assigns every point to a polygon without a caller-built pair
  list. Use `GPUPairwisePointInPolygon` when you already have the point and polygon pairs.
- `GPUPointDensity` writes per-cell statistics, a heatmap field, and a histogram. Use `GPUGridIndex`
  when you need the per-cell member lists instead.
- `GPUGeodesicPairs` adds WGS84 distance, bearings, and midpoints to the spherical result of
  `GPUHaversineDistance`.

## Attribution and licensing

These geospatial operations are inspired by [NVIDIA RAPIDS cuSpatial](https://github.com/rapidsai/cuspatial)
and the NVIDIA and RAPIDS contributors who pioneered GPU-accelerated spatial analytics. cuSpatial is
distributed under the [Apache License 2.0](https://github.com/rapidsai/cuspatial/blob/branch-25.04/LICENSE).

The luma.gl implementation is independently written TypeScript and WGSL for browser-native WebGPU;
it does not copy or translate cuSpatial source code and remains
[MIT-licensed](https://github.com/visgl/luma.gl/blob/master/LICENSE). Compatibility with selected
documented mathematical conventions does not imply CUDA support, cuSpatial API compatibility,
feature parity, NVIDIA affiliation, or NVIDIA endorsement.

## Coordinate and result formats

Two packed position formats are accepted:

| Position format | Storage | Intended use |
| --- | --- | --- |
| `float32x2` | Local XY or longitude/latitude values | Fast, tile-local calculations |
| `uint32x4` | Raw browser `Float64Array` words: `[xLow, xHigh, yLow, yHigh]` | Preserve small deltas between large coordinates |

Projection and haversine outputs are `float32x2` and `float32`, respectively. Their trigonometric
steps are f32 because portable WGSL does not provide f64 trigonometry.

Planar distance kernels pair their input and output formats:

| Position input | Distance output |
| --- | --- |
| `float32x2` | `float32` |
| raw binary64 `uint32x4` | double-single `float32x2` (`high + low`) |

The TypeScript property unions enforce these pairs. Raw binary64 subtraction occurs before the
planar calculation, preventing nearby large coordinates from first collapsing to the same f32
value. Double-single results provide approximately 48 significand bits over the f32 exponent
range; they are not general IEEE binary64 values.

Projection and simple pairwise distance inputs and outputs must have the same row count and view
kind. Vector inputs must also have identical chunk topology. Empty chunks are retained without
dispatching work. Nested geometry offsets cannot safely span independently chunked vectors, so the
two pairwise geometry APIs intentionally accept flat data views only. Each output must use storage
separate from its inputs, and view byte offsets must be naturally aligned to the row format.

## `GPUSinusoidalProjection`

`GPUSinusoidalProjection` matches the cuSpatial longitude/latitude convention. Coordinates and the
origin are in degrees; output is in kilometres. For longitude `lon`, latitude `lat`, and
`origin = [originLon, originLat]`, it computes:

```text
kilometresPerDegree = 40000 / 360
x = (originLon - lon) * kilometresPerDegree * cos((lat + originLat) / 2)
y = (originLat - lat) * kilometresPerDegree
```

The cosine argument is converted to radians. The sign, fixed 40,000 km circumference, and mean
latitude are part of the compatibility contract. Longitude is not implicitly wrapped at the
antimeridian. `originLon` must be in `[-180, 180]`, `originLat` must be in `[-90, 90]`, and both
values must be finite.

Raw binary64 inputs preserve each origin-minus-coordinate delta through binary64 subtraction before
rounding that delta to f32. The absolute latitude used to evaluate the midpoint cosine and the
remaining arithmetic are f32. This kernel does not claim f64-transcendental accuracy.

## `GPUHaversineDistance`

`GPUHaversineDistance` calculates pairwise great-circle distance from longitude/latitude degrees.
The configurable positive `radius` defaults to `6371` km and must be finite and representable as a
finite f32 value.

For angular deltas below `1e-4` radians, the implementation uses the local equirectangular limit so
that adapters do not collapse tiny f32 trigonometric inputs to zero. Intermediate paths use the
standard clamped `asin(sqrt(h))` haversine form. Paths beyond a central angle of π/2 use independent
spherical cross and dot products with `atan2`, avoiding the sensitivity of a rounded haversine near
antipodes. The acceptance suite requires at most 2 km absolute error against a double-precision CPU
oracle for difficult finite paths, including antimeridian, near-polar, and nearly antipodal cases.
This is a practical V1 error envelope rather than a cross-adapter mathematical bound; applications
needing tighter geodesic guarantees should verify their target devices and coordinate domain.

Raw binary64 inputs avoid premature CPU conversion and preserve small longitude/latitude deltas
through subtraction, but the transcendental inputs and final scalar result remain f32.

## `GPUPairwisePointDistance`

`GPUPairwisePointDistance` calculates the Euclidean distance between aligned point rows. Local
f32 inputs write one f32 distance per row. Raw binary64 inputs write a double-single result as
`[high, low]`; add the two limbs when reading the result on the CPU.

## `GPUPairwisePointSegmentDistance`

`GPUPairwisePointSegmentDistance` calculates the Euclidean distance from each point to its aligned
closed segment. Projections before the start or after the end are clamped to the corresponding
endpoint. A degenerate segment returns the point-to-endpoint distance. Its f32 and raw binary64
input/output pairings are the same as `GPUPairwisePointDistance`.

## `GPUPairwisePointInPolygon`

`GPUPairwisePointInPolygon` classifies one point against one polygon or multipolygon per row. It
accepts flat point and vertex views plus three caller-owned `uint32` offset views:

- `geometryOffsets` maps each point row to a range of polygons;
- `polygonOffsets` maps each polygon to a range of rings; and
- `ringOffsets` maps each ring to a range of flattened polygon positions.

Each offsets view starts at zero and ends at the next hierarchy level's row count. Offsets must be
nondecreasing. Rings close implicitly, use even/odd fill semantics, and may repeat their first
vertex explicitly. Polygon components within a multipolygon are unioned. Empty geometries are
outside, while malformed reachable spans and rings with fewer than three effective vertices are
uncertain. Rings whose effective vertices do not contain a provably non-collinear triple are also
uncertain, including rings with three or more distinct vertices on one line.

The caller-owned `uint32` output uses these values:

| Value | Classification |
| ---: | --- |
| `0` | `outside` |
| `1` | `inside` |
| `2` | `boundary` |
| `3` | `uncertain` |

Applications must handle `uncertain` explicitly. Non-finite coordinates and predicates too close
to the double-single arithmetic error envelope are never silently forced to inside or outside.
For both f32 and raw binary64 positions, exact endpoints and axis-aligned boundaries can be proven
as boundary; an exactly zero non-axis-aligned determinant remains uncertain without an adaptive
exact predicate. A cuSpatial-compatible boolean projection is
`classification === GPU_POINT_IN_POLYGON_CLASSIFICATION.inside`, so boundary remains false.

## `GPUPairwisePointLinestringNearest`

`GPUPairwisePointLinestringNearest` finds the nearest point on one paired multipart linestring for
each input point. `geometryOffsets` maps each point row to linestring parts, and
`linestringOffsets` maps each part to flattened positions. Parts are never joined or closed
implicitly. Empty and singleton parts contain no segment; repeated-vertex segments remain valid
distance candidates. Equal-distance ties preserve the first part and segment.

Both offset views start at zero, are nondecreasing, and end at the next level's row count.
Malformed reachable spans invalidate the complete paired row rather than being clamped into a
different geometry.

The required `output` follows the planar distance format pairing: f32 inputs write `float32`, while
raw binary64 inputs write double-single `float32x2`. Optional outputs include the nearest point,
the local linestring-part index, and the local segment index. F32 nearest points use `float32x2`;
raw binary64 nearest points use absolute `[xHigh, xLow, yHigh, yLow]` `float32x4` rows. When no
finite segment remains, numeric outputs are NaN and indices are `0xffffffff`.

Every output view must have a disjoint aligned storage-binding footprint from every input and from
the other optional outputs. This also applies when distinct graph handles refer to the same
physical buffer.

Both pairwise geometry kernels currently assign one GPU invocation to each row and scan that row's
rings or segments serially. This keeps the first API and storage contract small and works well for
bounded per-row geometries. Very large individual geometries should use a future flattened-segment
and segmented-reduction path.

## `GPUGridIndex`

`GPUGridIndex` rebuilds a flat, row-major uniform grid for packed `float32x2` or `float32x3`
points each time the compiled build graph is encoded. Its caller-owned outputs are:

- `cellOffsets`: `cellCount + 1` exclusive offsets;
- `objectIds`: capacity-bounded source IDs grouped by cell;
- `count`: the number of finite, in-domain positions accepted by the build; and
- `overflow`: `1` when `count` exceeds `objectIds.length`, otherwise `0`.

The `bounds` array stores every minimum followed by every maximum: `[minX, minY, maxX, maxY]` in
2D or `[minX, minY, minZ, maxX, maxY, maxZ]` in 3D. Positions outside these inclusive bounds and
positions with non-finite components are ignored. Supply `sourceIds` to retain application IDs, or
use `firstSourceIndex` to generate consecutive IDs. IDs within a cell have unspecified order.

Index construction is intended for a separate graph that runs when a dataset or tile changes. The
compact cell offsets require a complete rebuild after point positions or membership change.

## `GPUPointSpatialQuery`

`GPUPointSpatialQuery` selects rows from packed `float32x2` or `float32x3` positions. It accepts a
mutable f32 `query` view with one of these layouts:

| Kind | 2D layout | 3D layout |
| --- | --- | --- |
| `bounds` | `[minX, minY, maxX, maxY]` | `[minX, minY, minZ, maxX, maxY, maxZ]` |
| `radius` | `[centerX, centerY, radius]` | `[centerX, centerY, centerZ, radius]` |
| `polygon` | Polygon bounds: `[minX, minY, maxX, maxY]` | Not supported |

An optional `GPUGridIndex` view restricts refinement to cells intersecting the query envelope. The
query prepares an indirect dispatch on the GPU and refines only those candidates. Without an index,
it scans every position. The query-facing index view exposes `rowIndices`, and every stored value
must address a row in the supplied `positions` view. A `GPUGridIndex` can produce this buffer by
using its default zero-based generated IDs and passing its `objectIds` output as query
`rowIndices`. Keep application IDs out of that index build; instead, provide them through the
query's optional packed `sourceIds` view, aligned one-to-one with `positions`. Matching outputs use
`sourceIds[rowIndex]` when supplied and the zero-based row index otherwise. `GPUGridIndex` itself
remains a generic index and can still store arbitrary IDs for other consumers.

Polygon positions use packed `float32x2` rows. `ringOffsets` contains a start offset for each ring
plus one terminal offset; rings close implicitly and use even/odd fill semantics. Boundary points
are selected. This V1 API returns matching IDs rather than robust-topology classifications, so it
does not distinguish `inside`, `boundary`, or an ambiguous result.

All query outputs are caller-owned:

```ts
type GPUSpatialQueryOutput = {
  ids: GraphDataView<'uint32'>;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  totalCount?: GraphDataView<'uint32'>;
};
```

`count` is clamped to `ids.length` and can alias an indirect draw count. `totalCount`, when
provided, receives the unclamped number of matches among candidates actually examined by
refinement. If the index overflowed, its stored candidates are only a subset of the accepted source
rows, so `totalCount` is incomplete relative to the original positions. `overflow` is set when
either the index or result capacity overflows. The four writable output views must have mutually
disjoint aligned storage-binding ranges and must not overlap positions, source IDs, query values,
index storage, or polygon storage. This includes the one-row binding footprint of a zero-capacity
`ids` view. Result order is unspecified; no CPU readback is required for rendering.

Two optional packed scalar views can be supplied directly to `GPUPointSpatialQuery` for
GPU-resident broad-phase diagnostics:

```ts
new GPUPointSpatialQuery({
  // ...query inputs and output...
  intersectedCellCount,
  candidateCount
});
```

`intersectedCellCount` is the exact number of indexed cells dispatched for refinement. It is zero
for scan queries, invalid queries, and indexed query envelopes outside the index domain.
`candidateCount` is the exact number of rows presented to the narrow phase: every position row for
a valid scan, or every retained row ID in the intersected indexed cells. It deliberately includes
duplicate row IDs and invalid retained row IDs that the narrow phase later rejects. When an index
overflows, the count covers only candidates retained by the capacity-bounded `rowIndices` view, not
the missing source rows. These diagnostics are finalized by the existing query passes, require no
CPU synchronization, and do not change `GPUSpatialQueryOutput` or its indirect-draw layout. Like the
query outputs, both diagnostic views must be disjoint packed `uint32` scalars belonging to the same
graph as every other binding.

### Run a live spatial-query benchmark

Compare this real query kernel against an equivalent CPU bounds scan using your browser and GPU.
The indexed path builds a reusable grid once, reports that construction cost separately, and checks
all compacted IDs against the CPU result before displaying fence-synchronized timings.

<SpatialBenchmark />

## Non-finite data

The fixed-output distance and projection kernels do not silently replace non-finite arithmetic
with finite coordinates or distances. Grid construction ignores non-finite positions, and point
queries do not select them. Applications should still filter or classify non-finite rows before
rendering or consuming fixed-output results.

## Scale and dispatch

Fixed-output kernels linearize bounded multidimensional WebGPU workgroup dispatches. This avoids
the usual 65,535-workgroup single-dimension ceiling (16,776,960 rows at a workgroup size of 256)
without allocating or packing a source-sized intermediate. For vector-capable kernels, each chunk
remains an independent graph node, so large streamed vectors retain their original topology.
Nested-geometry kernels dispatch their flat paired rows directly. Indexed point queries instead
generate their candidate dispatch dimensions on the GPU.

## Analysis contributors

The entry point also exports analysis contributors. Each is an algorithm or workflow that declares
resources and command nodes into a caller-owned `GPUCommandGraph` through `getCommandNodes(graph)`
(`GPUCommandNodeProducer`), and is added with `graph.add(contributor)`. They are composed from GPU
Core primitives (`GPUVisibilityWorkflow`, `GPUBVH`, `GPUGridBinning`, `GPUHistogram`, `GPUReduction`,
`GPUAncestorProjection`, and others) and the raster operators; new WGSL is written only where no
primitive fits.

```ts
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, GPUPointDensity} from '@luma.gl/experimental/geospatial';

const graph = new GPUCommandGraph(device, {id: 'map'});
const viewport = new GPUParameterBuffer(device, {id: 'viewport', format: 'float32', length: 4});

// `positions`, `values`, `extent`, and `histogram` are graph views created on `graph`.
graph.add(new GPUPointDensity({
  positions, bounds: viewport.importToGraph(graph), gridSize: [256, 256],
  output: {values, extent, histogram}
}));
const compiled = graph.compile(); // once

// every frame: rewrite parameters, encode the same compiled graph
viewport.write(new Float32Array([minX, minY, maxX, maxY]));
compiled.encode(device.commandEncoder, {parameters: undefined});
```

### Conventions

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

### `GPUPointDensity`

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

### `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid`, and `GPUCellLevelSelection`

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

### `GPUPointToCell` and `GPUCellGeometry`

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

### `GPUCellTopology` and `GPUCellCompaction`

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

### `GPUCellCover`

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

### `GPUCellTableCompare`

`GPUCellTableCompare` outer-joins two sorted cell tables (the outputs of `GPUCellAggregation` or `GPUCellRollup`) into one union table for period-over-period or A-versus-B compare maps. Both tables must use the same grid family and resolution; the contributor cannot check that.

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

### `GPURegionStatistics`

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
  summary: graph.importGPUData('summary', summaryData)
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

### `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin`

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

### `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, and `GPULineChunk`

Geometry-producing line contributors. Each writes a `GPULinePathOutput`: flat `positions`, clamped
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

### `GPUGeometryMeasures`, `GPUGeodesicPairs`, and `GPUGeodesicDestination`

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

### `GPULinearReferencing` and `GPULineLocate`

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

### `GPUBufferSelection`

Selects points within a per-frame planar distance (inclusive) of point or segment features. A
polyline is passed as consecutive segment rows. The contributor composes `GPUNearestFeatureJoin`. Unlike
the join's unordered `matches`, it writes a source-aligned 0/1 mask and stable IDs in ascending row
order, with a clamped count and overflow that covers both the output capacity and the join.

```ts
graph.add(new GPUBufferSelection({
  points, features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  distance: bufferDistance.importToGraph(graph), candidateCapacity: points.length * 4,
  spatialSort: true, outputMask, output
}));
```

### `GPUZonalStatistics`

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

### `GPUInverseDistanceWeighting`

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
- Determinism: the grid index's in-cell order depends on atomics, so the contributor re-sorts IDs inside
  each index cell every encoding; sums then run in a fixed order (index row, column, ascending
  row; or `(d^2, row)` with `k`). Weights are accumulated in log space against a running maximum,
  so tiny distances with large powers do not overflow f32.
- The index is rebuilt every encoding, so positions and values may change per frame. Keep a few
  samples per index cell: the in-cell sort is quadratic in cell population.
- `output.values` is a row-major float32 raster that `GPUTerrainContours` accepts directly as
  `elevation: {id, format: 'float32', storage: {kind: 'buffer', values: surface}}`; NaN cells
  clear their marching-squares cells.

### `GPUFocalStatistics`

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

### `GPUDotDensity` and `GPURandomPointsInPolygon`

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

### `GPUSpatialClustering`

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

### `GPUHotSpotAnalysis` and `GPULocalMoran`

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
```

### `GPUNeighborSearch` and `GPUSpatialWeights`

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

### `GPUGlobalSpatialStatistics`

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

### `GPULocalPermutationTest` and `GPUGlobalPermutationTest`

Permutation inference (pseudo p-values) over a `GPUSpatialWeights` CSR, matching PySAL esda's
`p_sim`. Both contributors read a uint32 parameter view written with `getGPUPermutationParameterValues`:
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

### `GPUGeographicDistribution`

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

### `GPUEmergingHotSpots`

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

### `GPUOrdinaryLeastSquares`

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

### `GPUGeographicallyWeightedRegression`

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

### `GPUTrajectoryMetrics`

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

### `GPUTrajectoryPlayhead` and `GPUTrajectoryResample`

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

### `GPULineSimplification`

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

## See also

- [GPUCommandGraph](/docs/api-reference/experimental/gpu-core/gpu-command-graph)
- [GPU Network](/docs/api-reference/experimental/gpu-network) and [GPU Terrain](/docs/api-reference/experimental/gpu-terrain) for the network and elevation analysis contributors
- [GPU Dataframe analysis contributors](/docs/api-reference/experimental/gpu-dataframe-analysis) for the column statistics, joins, and time filters the spatial contributors consume
- [GPUGridIndex](/docs/api-reference/experimental/gpu-core/gpu-grid-index)
- [GPU floating-point precision](/docs/api-guide/shaders/gpu-floating-point-precision)
- [`fp64arithmetic`](/docs/api-reference/shadertools/shader-modules/fp64-arithmetic)
