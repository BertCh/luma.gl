---
title: "GPU Spatial Analysis: Spatial weights"
sidebar_label: "Spatial weights"
description: "Spatial weights contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Spatial weights

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

A `GPUSpatialWeights` is the sparse `W` that spatial statistics are built on, in CSR form: row `i`
lists the neighbors `j` of focus row `i` and the weight `w_ij` of each. Producers write it on the
GPU, transforms rewrite it, and the statistics below read it, so one statistic works with any
neighborhood definition. Applications can also upload weights built elsewhere.

Every producer guarantees: `offsets` holds `rows + 1` non-decreasing entries starting at 0; row `i`
occupies slots `[offsets[i], offsets[i + 1])`; neighbor IDs are strictly ascending within a row; a
self join never lists a row as its own neighbor (`w_ii = 0`); weights are finite and non-negative.
Capacity overflow clamps offsets to `neighbors.length` and sets an `overflow` flag on the GPU.

A `GPUSpatialWeights` is a first-class object, not only buffers. Every producer in this section
(`GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsTransform`,
`GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsTranspose`) exposes `outputWeights`: the caller-owned
views with `information` (`rowCount`, `transform` state `'raw' | 'binary' | 'row' | 'double' | 'variance' |
'kernel' | 'symmetric' | 'unknown'`, `symmetric` as `true | false | 'unknown'` only where
guaranteed, `hasSelfWeights`, `provenance` with source, criterion, `k`, `radius` and `order`) and
`status` (`overflow` and, where the producer computes it exactly, `requiredCount`) attached. Pass
`outputWeights` to consumers: statistics read `information.transform` (a `'row'` W is
re-standardized over the neighbors that remain after masking) and gate on `status.overflow`.
Uploaded weights may omit `information`, which reads as `'unknown'`. `validateGPUSpatialWeights`
checks on the CPU what it can: view formats and lengths, `information.rowCount`, the declared
transform and symmetry, and the status formats.

The optional `spatialContext` describes the coordinates and metric used to build the weights;
`distances` and `information.provenance.radius` use its length unit. `GPUNeighborSearch` sets it;
`GPUNearestFeatureWeights` sets it from the producing join's context passed to the adapter.
Transforms, transpose and algebra propagate it; algebra checks declared input contexts with
`assertCompatibleGPUSpatialContexts`. Topology-only producers and uploaded weights may omit it.

One neighbor definition is used everywhere: a slot is an active neighbor when its weight is
positive, its neighbor ID is below the row count and it is not the row itself. A zero-weight slot is
listed but is not a neighbor. An island is a row with no active slot, whatever slots it lists.

Uploaded CSR must satisfy the invariants above; consumers do not re-check them (a binary search
over unsorted neighbors silently gives wrong `S1`, `S2` and symmetry counts). Run
`GPUSpatialWeightsValidation` once after upload, and sort, merge duplicates and drop self slots on
the CPU if it reports violations.

Capacity grow loop: `GPUContiguityWeights`, `GPULatticeWeights` and `GPUSpatialWeightsAlgebra`
write into one caller-owned `output: GPUSpatialWeightsOutput`, the CSR views plus
`status: {overflow, requiredCount?, candidateOverflow?}` (`overflow` required). Read `overflow` and
`requiredCount` after a frame; when `requiredCount` exceeds the capacity, allocate
`getGPUSpatialWeightsCapacityPlan({rowCount, capacity, requiredCount}).capacity` slots (the required
count plus 25 percent headroom, `GPU_SPATIAL_WEIGHTS_CAPACITY_HEADROOM`) and rebuild. The plan's
`estimatedOutputBytes` is the bytes of the CSR views at that capacity
(`GPU_SPATIAL_WEIGHTS_SLOT_BYTE_LENGTH` per slot, plus `GPU_SPATIAL_WEIGHTS_DISTANCE_BYTE_LENGTH`
with distances, plus the offsets). The contiguity and lattice producers also expose
`getCapacityPlan(observation)`, which adds their own transient bytes (from the same constants their
`getCommandNodes` allocates with; the device-dependent gpgpu sort and scan scratch is not included)
and, for contiguity, the intermediate pair stage. Rows past the point of truncation are empty, so
never use weights whose `overflow` is 1.

| Producer | Neighborhood | PySAL equivalent |
| --- | --- | --- |
| `GPUNeighborSearch` | k nearest neighbors or distance band between points | `KNN`, `DistanceBand`, `Kernel` |
| `GPUContiguityWeights` | polygons sharing a vertex (queen) or an edge (rook) | `Queen`, `Rook` |
| `GPULatticeWeights` | regular grid cells | `lat2W` |
| `GPUSpatialPredicateJoin` with `weights` | any predicate join (intersects, within, dwithin, ...) | - |
| `GPUSpatialWeightsAlgebra` | union, intersection, difference, higher order, self weights, subgraph, block | `w_union`, `higher_order`, `w_subset`, `block_weights` |

### `GPUContiguityWeights`

Join Chicago tract polygons that share a vertex (queen) or an edge (rook) into binary neighbour weights.

::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=result caption="Queen links join 791 Chicago tracts that share a vertex; tracts with no link are islands."}

Polygon contiguity as binary weights. `criterion: 'queen'` joins features that share at least one
vertex; `'rook'` joins features that share a ring edge (both endpoints, either direction). Input is
`polygons: GPUPolygonGeometryPort` (the four-level layout `positions`, `ringOffsets`,
`polygonOffsets`, `featureOffsets`), with one output row per FEATURE: every ring of every part
counts, holes included, so a multipart feature is the neighbor of anything any of its parts touches,
and parts of one feature are never self neighbors. This is the row space of
`GPUZonalStatistics` and the points-in-polygons recipe, so points -> counts -> W -> Moran chain on
one feature table without a cast.

- `snapTolerance` (default 0, exact f32 equality) snaps vertices with `floor(x / tolerance + 0.5)`
  per axis. This is a grid snap, not a distance test: two vertices closer than the tolerance can
  fall in different cells.
- Exact 64-bit vertex keys, stable radix sorts and run detection; no hashing, so no collisions, and
  the output is deterministic and symmetric.
- A point shared by `m` features emits `m^2` candidate pairs, bounded by `pairCapacity` (default
  `GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR` = four times the neighbor capacity); exceeding it
  sets `output.status.overflow` and, when given, `output.status.candidateOverflow`.
- `output.status.requiredCount` is exact unless the pair list overflowed (then a lower bound).
  `getCapacityPlan({requiredCount, candidateOverflow})` returns the next `slots` and `pairs`
  capacities (pairs keep the default ratio and double after a candidate overflow), so the rebuild
  loop converges. `outputWeights.information` is binary, symmetric, with provenance
  `{source: 'GPUContiguityWeights', criterion}`.

```ts
graph.add(new GPUContiguityWeights({
  criterion: 'rook',
  polygons: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},
  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}
}));
```

#### How it computes

{/* lens:GPUContiguityWeights */}

::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen neighbours, and how many of the 791 tracts are islands."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=formula caption="Follow the rule: tracts are neighbours when their snapped vertex keys intersect."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=kernel caption="Inspect the key sort, pair emission and CSR compaction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=graph caption="Trace the command graph from tract vertices to the CSR neighbour lists."}

### `GPULatticeWeights`

Connect the cells of a width by height grid to their rook or queen neighbours as binary CSR weights.

::lens{scenario=gpu-spatial-analysis.lattice-weights rung=result caption="Queen neighbours on a 32 by 32 terrain lattice: 8 inside, 5 along edges, 3 in corners."}

Weights over a `width x height` grid with row-major IDs `y * width + x`. `criterion: 'rook'` uses
Manhattan distance `<= radius`, `'queen'` Chebyshev distance `<= radius` (`radius` defaults to 1, at
most 32). A zero in the optional `mask` removes the cell as a row and as a neighbor. With `cellSize`
it also writes Euclidean `distances`, so kernel transforms apply. `output.status.requiredCount` is
the exact unclamped neighbor total; `getCapacityPlan({requiredCount})` plans the next slots.
`outputWeights.information` is binary and symmetric, with provenance `{criterion, order: radius}`.

#### How it computes

{/* lens:GPULatticeWeights */}

::lens{scenario=gpu-spatial-analysis.lattice-weights rung=datum bind=page datum=528 view=neighbourhood-view caption="Inspect cell 528: its 8 queen neighbours, with the degree histogram of all 1,024 cells."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=formula caption="Follow the rule: two cells are neighbours within the rook or queen radius."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=kernel caption="Inspect the neighbour count and fill kernels over 1,024 cells."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=graph caption="Trace the command graph from lattice size to CSR neighbours."}

### `GPUSpatialWeightsTransform`

Rewrite CSR weights by row, binary, kernel, symmetric, double or variance standardisation into a new weights graph.

::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=result caption="Row standardisation keeps the queen links of 791 tracts and rescales each row to sum to one."}

Rewrites weights in place or into a separate `output`. Chain several instances, for example a kernel
followed by row standardization. Numeric `bandwidth` uses the length unit of
`weights.spatialContext`, matching `weights.distances`. The optional `spatialContext` declares the
bandwidth's context and is checked against `weights.spatialContext` with
`assertCompatibleGPUSpatialContexts` when both are supplied.

- `'row'`: `w_ij / sum_j w_ij`. A row whose weights sum to zero (or to a non-finite value) stays all
  zero: it has no active neighbor, so it is an island for every statistic (no uniform fallback).
  To restore row standardization after masking or `'subgraph'` removed neighbors, apply `'row'` again
  to the result.
- `'binary'`: `w_ij > 0 ? 1 : 0`, keeping every slot.
- `'kernel'`: `K(d_ij / h_i)` from `weights.distances`, with `kernel` one of `gaussian`
  (`exp(-z^2 / 2) / sqrt(2 pi)`), `triangular` (`max(1 - z, 0)`), `epanechnikov` (`3/4 max(1 - z^2, 0)`,
  PySAL quadratic), `bisquare` (`15/16 max(1 - z^2, 0)^2`, PySAL quartic) or `uniform` (`1/2` for
  `z <= 1`). `bandwidth` is required (no default) and is either a positive number, the same `h` for
  every row (PySAL fixed bandwidth; for distance-band weights pass the band radius), or `'adaptive'`:
  PySAL's adaptive bandwidth, `h_i` = the distance to the k-th nearest neighbor times
  `eps = 1.0000001` (`GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON`). On the CSR that is the row's
  largest distance, but only for kNN rows, so `'adaptive'` requires
  `weights.information.provenance.criterion === 'knn'` and throws otherwise. Because of `eps` the k-th
  neighbor keeps a tiny positive weight, as in libpysal (in float32 `eps` rounds to `1 + 2^-23`, so
  the tail weight is about 1e-7, never exactly 0). Only the stored neighbors take part, so the
  Gaussian is truncated at the neighbor set. Weights never hold self slots, so `K(0)` is not
  written: compute it with `getGPUSpatialWeightsKernelValue(kernel, 0)` and pass it as `selfWeight`
  to `GPUHotSpotAnalysis` for a kernel Gi*. `getGPUSpatialWeightsKernelValue(kernel, z)` is the CPU
  value of every profile.
- `'double'`: `w_ij / S0`, with `S0` the sum of all weights, so the weights sum to 1 (PySAL `'D'`).
  `doubleSum: 'rows'` scales by `n / S0` instead, so the weights sum to `n`.
- `'variance'`: `s_ij = w_ij / sqrt(sum_j w_ij^2)`, then `w'_ij = s_ij n / sum s` over the whole
  matrix, with `n` the row count (PySAL `'V'`).
- `'symmetrize'`: `(w_ij + w_ji) / 2`, treating a missing reverse slot as 0. It keeps the sparsity
  pattern, so it is only symmetric for symmetric patterns (contiguity, lattice, distance band), and
  needs a separate `output`. For a directed pattern such as kNN, take the union of the weights and
  their `GPUSpatialWeightsTranspose` with `GPUSpatialWeightsAlgebra`.

`'double'` and `'variance'` compute a per-row partial, reduce it with `GPUReduction`, and apply it in
a second pass, in place or into `output`. A zero or non-finite total gives zero weights. The
operation, kernel and bandwidth are compile-time options. `output`, when given, must have exactly
`weights.neighbors.length` values.

`outputWeights.information.transform` is the operation (`'symmetrize'` gives `'symmetric'`). `symmetric`
is kept only where it is guaranteed: `'binary'`, `'double'`, `'symmetrize'` and a fixed-bandwidth
`'kernel'` keep a `true` input flag; `'row'`, `'variance'` and an adaptive kernel give `'unknown'`
(row standardization makes a symmetric pattern asymmetric unless all row sums are equal).

#### How it computes

{/* lens:GPUSpatialWeightsTransform */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen weights after row standardisation, and the islands among 791 tracts."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=formula caption="Follow the row rule: each weight is divided by its row sum; empty rows stay zero."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=kernel caption="Inspect the row-sum and rescale kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=graph caption="Trace queen contiguity into row standardisation on the command graph."}

### `GPUSpatialLag`

Average neighbouring values through row-standardised weights, turning a tract variable into its spatial lag.

::lens{scenario=gpu-spatial-analysis.spatial-lag rung=result caption="Poverty percentage beside its neighbour-weighted lag across 791 tracts; the lag smooths local extremes."}

`lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`). An optional `mask` skips masked neighbors and writes
0 for masked rows (square weights only); `normalize: true` divides by the included weight sum, which
gives the row-standardized lag without a transformed copy of the weights. Rows are summed by a
load-balanced (merge-path) CSR loop, so a hub row with 1e5 neighbors spreads over many workgroups
instead of one thread. The association depends only on the offsets, so the result is bitwise
reproducible; it matches a sequential sum within f32 rounding. A weights matrix with a self term (`GPUSpatialWeightsAlgebra`
`'selfWeight'`) includes it in the sum.

Weights need not be square. The value row count is `values.length / columnCount`: the rows of
`weights` are targets and the neighbor IDs index those value rows; neighbors at or above it are
skipped. `columnCount` (default 1) lags several columns in one pass: `values` is row-major,
`output` is `rows * columnCount`, and `normalize` shares one weight sum per row. This is how the area
shares of `GPUArealInterpolation` transfer values between zone systems. `values` may be `float32` or
`uint32` (counts, exact below 2^24; one conversion node is added).

```ts
graph.add(new GPUContiguityWeights({criterion: 'queen', polygons, output: {...weights, status: {overflow}}}));
graph.add(new GPUSpatialWeightsTransform({operation: 'row', weights}));
graph.add(new GPUSpatialLag({weights, values, output: lag}));
```

#### How it computes

{/* lens:GPUSpatialLag */}

::lens{scenario=gpu-spatial-analysis.spatial-lag rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its neighbours and weights, and the islands that receive a zero lag."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=formula caption="Follow the lag: the sum of each neighbour weight times its value."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=kernel caption="Inspect the load-balanced segment sum over the CSR slots."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=graph caption="Trace queen weights, row standardisation and the lag on the command graph."}

### `GPUNeighborSearch`

Find the k nearest tract centroids by great-circle distance and write them as CSR neighbour weights.

::lens{scenario=gpu-spatial-analysis.neighbor-search rung=result caption="Each of 791 tract centroids links to its 4 nearest others, giving 3,164 directed links."}

Maximum: `gridSize[0] * gridSize[1]` is at most `getGPUGridIndexMaximumCellCount(device)` (2^25 - 1 on a default device), checked when the contributor is added to a graph.

`GPUNeighborSearch` finds exact k nearest neighbors (`mode: 'knn'`, `k` up to 32) or every point
within a distance band (`mode: 'radius'`) between planar points, or between longitude/latitude points
with great-circle distances (see below). A self join never lists a row as its
own neighbor; pass `queryPositions` for a cross join. It writes a `GPUSpatialWeights` CSR:
`offsets` (`queryRows + 1`), `neighbors` (the slot capacity), `weights`, and optional `distances`.

- Rows list neighbor IDs in strictly ascending order with distances and weights aligned, so
  consumers can binary-search `w_ji`.
- kNN keeps the `k` smallest `(d^2, id)` pairs, so ties go to the lowest ID. An optional
  `radius` bounds the search. Fewer valid targets give a shorter row.
- Weights: `binary`, `inverseDistance` (`max(d, distanceFloor)^-power`), or `kernel` (gaussian,
  triangular, epanechnikov, bisquare, uniform). The bandwidth is `radius` in radius mode and the
  row's k-th neighbor distance times 1.0000001 in kNN mode, which is libpysal `Kernel(fixed=False)`
  with the self slot removed (a row with fewer than `k` neighbors uses its farthest listed
  neighbor). `rowStandardize` divides each row by its sum. Non-finite weights are written as 0.
  Rows never hold a self slot: the kernel's self weight `K(0)` is the consumer's `selfWeight`
  (`getGPUSpatialWeightsKernelValue(kernel, 0)`). A zero-weight slot (for example a triangular
  kernel at `d = radius`, or coincident points under inverse distance with no floor) is not a
  neighbor: consumers skip slots whose weight is not positive.
- Capacity: offsets are clamped to `neighbors.length`, so readers always see consistent, possibly
  truncated rows. `output.status.overflow` is set on the GPU, and `output.status.requiredCount` and
  `output.neighborCounts` stay unclamped.

```ts
const searchParameters = {bounds, rowStandardize: true};
const parameters = new GPUParameterBuffer(device, {
  id: 'knn', format: 'float32', length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  values: getGPUNeighborSearchParameterValues(searchParameters)
});
graph.add(new GPUNeighborSearch({
  mode: 'knn', k: 8, gridSize: [256, 256], positions,
  parameters: parameters.importToGraph(graph),
  output: {offsets, neighbors, weights, distances, status: {overflow}},
  weightTransform: getGPUNeighborSearchWeightTransform(searchParameters)
}));
```

**Spatial context.** `spatialContext` (compile-time) defaults to native planar coordinates. Planar
contexts require metric `'native'`; local frames with `units: 'meters'` are accepted. This operator
measures distance, so `local-lambert-equal-area` frames throw. The resolved context is attached to
`outputWeights.spatialContext`. For
longitude/latitude pass `{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units?:
'meters' | 'kilometers', sphereRadius?}` (`sphereRadius` in the context's length unit, default the mean Earth radius).
Positions are `[longitude, latitude]` degrees, `bounds` are `[minLongitude, minLatitude,
maxLongitude, maxLatitude]` degrees (longitude span at most 360, for example the world), and
`radius`, `distanceFloor` and the output `distances` are great-circle distances in `units`
(meters by default). Any other space or metric throws at construction (project to a planar frame
first). Distances are f32 central angles on a sphere. The lattice stays in degrees: radius mode
visits every cell that can hold a target in the cap (latitude span `radius / R`, longitude span the
exact cap extent `asin(sin(a) / cos(latitude))`, the whole circle when the cap reaches a pole, and
bands wrap across +-180), and kNN stops only when a conservative spherical lower bound of the
distance to every unvisited cell exceeds the k-th distance, so results never depend on `gridSize`.
The bound is deliberately loose near the poles, which only costs speed. Points with latitude beyond
+-90 are invalid.

**Planar precision.** Planar f32 coordinates keep about seven digits: UTM northings (~4e6 m) and
Web Mercator (~1e7) quantize to 0.5 to 1 m and make band edges and ties noisy. Project to a local
frame with `createGPULocalFrame` (or subtract an origin) before upload; the node cannot recover
digits lost in its inputs.

Pass optional `spatialContext` to `getGPUNeighborSearchParameterValues(settings)` to warn on short `radius` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

**Row standardization.** The optional compile-time `rowStandardize` prop is the single source of truth
when given: the weights kernel uses it as a constant and ignores the per-frame `rowStandardize`
parameter. Consumers re-standardize rows over the neighbors left after masking only when
`info.transform` is `'row'`, so set it for any row-standardized band. `rowStandardize: true` with a
`weightTransform` other than `'row'`, or `false` with `'row'`, throws. When it is omitted, the per-frame
field and the `weightTransform` declaration apply as before.

**Output.** `outputWeights` is `output` with `information` and `status` attached, for consumers:
`rowCount`, `transform` (`'row'` when the compile-time `rowStandardize` prop is true, otherwise the
optional compile-time `weightTransform`, else `'unknown'`, because weight kind and row standardization
are per-frame parameters; derive it from the same parameters with
`getGPUNeighborSearchWeightTransform`), `symmetric` (true only for a planar radius self join
without `queryMask` and a declared non-row transform; otherwise `'unknown'`), `provenance`
(`criterion` `'knn'` or `'distance-band'`, and `k`) and `status` (`overflow`, `requiredCount`,
`invalidCount`). kNN output has criterion `'knn'`, which `bandwidth: 'adaptive'`
kernel transforms require. PySAL's `min_threshold_distance` (the smallest band that leaves no
island) is the maximum of the returned `distances` of a `k = 1` kNN run.

**Dropped rows.** Targets and queries count only when their mask is nonzero, their coordinates are finite, and they
lie inside the per-frame `bounds`; the others silently become isolates and never a neighbor.
Pass the optional `output.status.invalidCount` (one uint32) to learn how many masked rows were dropped for a
non-finite coordinate or for lying outside `bounds` (target rows, plus the query rows of a cross
join). A self join applies `mask` to the query rows too. Bounds,
radius, the weight function and row standardization are per-frame parameters, so changing them
never rebuilds the graph. Targets are bucketed with `GPUGridIndex`, whose `boundsBuffer` carries the
per-frame lattice (cells at least `radius` wide) without recompiling. kNN
searches expanding cell rings with a private sorted top-k list and stops once the k-th distance is
strictly inside the visited box, with a slack for f32 rounding. Radius mode counts, scans, emits, and
insertion-sorts each row by ID. No float atomics are used, so outputs are bitwise reproducible.

**Search structure.** The optional compile-time `searchStructure` (`'auto'` by default, `'grid'` or
`'bvh'`) only changes speed: all three give the same IDs, offsets, distances and weights (bitwise),
except which neighbors the one row cut by an exceeded capacity keeps. A grid cell that holds most of
the points (clustered data) makes every query in it scan that cell, so `'bvh'` instead walks a
Hilbert-packed point BVH in near-first depth-first order with the same `(d^2, id)` top-k
list and radius limit, then shares the grid path's count, scan, emit and sort-by-ID stages. Invalid
targets (mask, non-finite, outside the per-frame bounds) get empty boxes, so per-frame inputs still
never recompile. `'auto'` now uses exactly the grid graph, without allocating or building a BVH or
computing an occupancy gate. All four measured 200k-point cases (clustered/uniform, kNN/radius)
favored the grid; occupancy alone did not predict traversal cost. `'bvh'` remains opt-in.
`GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD` is retained for compatibility but no longer used.
The BVH path is planar
only: `'bvh'` with a longitude-latitude `spatialContext` throws, and great-circle search always uses
the grid. The new prop is compile-time and not part of the per-frame parameter
packer, so the Parameters section below is unchanged.

{/* parameters: GPUNeighborSearch generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNeighborSearch` parameters

- Packer: `getGPUNeighborSearchParameterValues(settings)`, writes `GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH` (12) `float32` values.

Settings of `getGPUNeighborSearchParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `bounds` | `readonly [number, number, number, number]` | `[minX, minY, maxX, maxY]` extent of the search lattice. Targets and queries outside it (or with non-finite coordinates) are excluded: they become isolates and never a neighbor, so it must cover every point of interest (count the dropped rows with `invalidCount`). For a great-circle search these are degrees `[minLongitude, minLatitude, maxLongitude, maxLatitude]` with a longitude span of at most 360 (`[-180, -90, 180, 90]` for the world). |
| `radius` (optional) | `number` | In the units of the spatial context (position units, or meters/kilometers of great-circle distance). Radius mode: the distance band (neighbors satisfy `d <= radius`); a non-positive or non-finite radius yields empty rows. kNN mode: an optional maximum distance; `Infinity` (the default) or any non-positive value means unbounded. |
| `weightKind` (optional) | `GPUNeighborSearchWeightKind` | Weight function. Defaults to `'binary'`. |
| `power` (optional) | `number` | Inverse-distance exponent. Defaults to 1. |
| `distanceFloor` (optional) | `number` | Inverse distance uses `max(d, distanceFloor)`. Defaults to 0, in which case coincident points (`d = 0`) get weight 0 because their weight is not finite. |
| `kernel` (optional) | `GPUNeighborSearchKernel` | Kernel profile for `weightKind: 'kernel'`. Defaults to `'triangular'`. |
| `rowStandardize` (optional) | `boolean` | Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false. |

{/* /parameters */}

#### How it computes

{/* lens:GPUNeighborSearch */}

::lens{scenario=gpu-spatial-analysis.neighbor-search rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its four nearest centroids, and the degree histogram of all 791 tracts."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=formula caption="Follow the rule: keep the k smallest distance and ID pairs, excluding the tract itself."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=kernel caption="Inspect the grid search kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=graph caption="Trace centroids into the grid and k-nearest search on the command graph."}

### `GPUSpatialWeightsAlgebra`

Combine or expand CSR weights by union, intersection, higher order, self weights, subgraph or block into a new graph.

::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=result caption="Second-order queen neighbours of 791 tracts: each tract links to tracts exactly two steps away."}

Builds a new `GPUSpatialWeights` from existing ones with a capacity-bounded `output` (rows + 1
offsets; it must not alias an input). `operation` is one of:

- `'union'`, `'intersection'`, `'difference'`, `'symmetricDifference'` with `left` and `right` (same
  row count) and an optional `weightRule`: `'left'` (default), `'right'`, `'sum'`, `'min'`, `'max'`,
  `'product'` or `'binary'`. The rule applies where both operands list a link, and one-sided links keep
  their weight. `'binary'` gives 1 everywhere, which reproduces libpysal 4.15 `Graph` set operations.
  Distances come from `left` where it lists the link, else from `right`.
- `'higherOrder'` with `weights`, `order` (1 to 32), `cumulative` (default false) and `workCapacity`
  (intermediate capacity, default the output capacity). Writes the neighbors exactly `order` steps away
  by shortest path, excluding the diagonal, with weight 1; `cumulative` matches `lower_order=True`.
  Cost grows with the squared frontier degree per row, so it suits sparse contiguity and kNN weights.
- `'selfWeight'` with `weights` and `selfWeight` (a number at least 0, or a per-row `float32` view).
  Inserts or replaces `w_ii` with distance 0. It relaxes the `w_ii = 0` invariant. `GPUSpatialLag`,
  `GPUSpatialWeightsTransform`, `GPUSpatialWeightsAlgebra` and `GPUSpatialWeightsSummary` accept it;
  `GPUGlobalSpatialStatistics`, the permutation tests and the local autocorrelation kernels ignore the
  self slot.
- `'subgraph'` with `weights` and a per-row `mask`. Rows keep their IDs: masked rows become empty and
  slots that point at masked rows are dropped.
- `'block'` with `groupIds` and `groupCount`. Row `i` lists every other row of the same group with
  weight 1; IDs at or above `groupCount` have no group. The output holds the sum of squared group
  sizes minus the row count.

`output.status.overflow` (one `uint32`, required) is 1 when the capacity was too small, and the
optional `output.status.requiredCount` holds the slots needed (published as
`outputWeights.status.requiredCount`; omitted there for `'higherOrder'` with
`order > 1`, where an overflowing intermediate set would make it a lower bound, so raise
`workCapacity` instead). On overflow the output is still a valid CSR: the leading rows are complete,
one row is a prefix and the rest are empty. Output rows are sorted and unique with finite
non-negative weights. `outputWeights.information` carries the transform state (`'binary'` for
`weightRule: 'binary'`, `'higherOrder'` and `'block'`; an unscaled `'raw'` or `'binary'` input
stays so; a row-standardized input becomes `'unknown'` because slots changed) and `symmetric: true`
only when it follows from symmetric operands.

```ts
graph.add(new GPUSpatialWeightsAlgebra({
  operation: 'union', left: contiguity, right: knn, weightRule: 'binary',
  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}
}));
```

#### How it computes

{/* lens:GPUSpatialWeightsAlgebra */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its second-order neighbours, and the degree histogram of all 791 tracts."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=kernel caption="Inspect the frontier expansion, sort and compaction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=graph caption="Trace first-order queen weights into second-order expansion on the command graph."}

### `GPUSpatialWeightsSummary`

Summarise CSR weights into S0, S1, S2, slot counts, asymmetry, isolates and degree range.

::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=result caption="Read S0, S1, S2 and the isolate count of the queen weights of 791 tracts."}

Summary statistics of a weights matrix, written into `output: {statistics, counts, cardinality?}`.
`statistics` (`float32`, at least 3) receives `S0`,
`S1 = 1/2 sum (w_ij + w_ji)^2` and `S2 = sum_i (row_i + col_i)^2`, as in libpysal. `counts` (`uint32`,
at least 6, see `GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT`) receives the used slots, the asymmetric slots,
the isolates, the minimum and maximum cardinality and the empty rows. The optional `cardinality` is the
per-row active neighbor count and `symmetryTolerance` (default 0) decides when two weights differ. Sums
run in a fixed order. Isolates are rows with no active slot, the one island definition of every
statistic (libpysal `islands` for a W built without zero-weight entries); `emptyRows` counts rows with
no slot at all. Cardinality counts active slots, so a zero-weight slot is not a neighbor: this deviates
from libpysal `cardinalities`, which counts every listed neighbor, and agrees with it for a W built
without zero-weight entries (pinned against libpysal in the specs). The used slots count stays the
listed total. Connected components are not computed. A pair of unequal weights counts as two
asymmetric slots, where libpysal `asymmetry()` returns pairs.

#### How it computes

{/* lens:GPUSpatialWeightsSummary */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=datum bind=page datum=420 caption="Inspect tract 420: its degree among the 791 tracts, beside the global moments."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=formula caption="Follow three moments: S0, S1 and S2 from row sums, column sums and reverse weights."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=kernel caption="Inspect the row, column and blocked reduction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=graph caption="Trace the row, column and global reduction passes on the command graph."}

### `GPUSpatialWeightsTranspose`

Reverse every link of CSR weights, turning outgoing weights into incoming ones.

::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=result caption="Transposed row-standardised queen weights: rows no longer sum to one, since they hold incoming weights."}

The CSR transpose `W'` of any `GPUSpatialWeights`, square or rectangular (libpysal `Graph` with focal and
neighbor swapped). `columnCount` (default: the row count) sets the number of rows of the result for
cross weights whose neighbor IDs index target rows; slots with a neighbor ID at or above it are dropped,
and columns that nothing reaches give empty rows. `output` is a caller-owned CSR with `columnCount + 1`
offsets and at least the input capacity in `neighbors` and `weights` (and `distances` when the input has
them); slots past `offsets[columnCount]` are zeroed. Slots are keyed by column, stable-sorted, counted, scanned and gathered, so every output
row is ascending and the result is bitwise reproducible with no atomics. The optional
`output.asymmetricSlots`
(one `uint32`, square weights only, with `symmetryTolerance`, default 0) counts the positions `(i, j)` of
the union of both patterns whose weights differ or that are present in one only, so a one-way link counts
twice and 0 means `W = W'`. Reference: libpysal 4.15 (no deviations; the check is exact by default).

```ts
graph.add(new GPUSpatialWeightsTranspose({weights, output: {...transposed, asymmetricSlots}}));
```

`outputWeights` carries the input `symmetric` flag (a transpose is symmetric exactly when `W` is) and
the input `status` (a truncated `W` gives a truncated `W'`); a `'row'` or `'variance'` state becomes
`'unknown'`.

#### How it computes

{/* lens:GPUSpatialWeightsTranspose */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its incoming weights, which were outgoing weights of its neighbours."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=kernel caption="Inspect the column sort and row boundary search kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=graph caption="Trace row standardisation into transposition on the command graph."}

### `GPUSpatialWeightsValidation`

Flag rows of uploaded CSR weights whose offsets, neighbour order, range, diagonal or weights break the invariants.

::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=result caption="Validation flags per tract for queen weights; expect zero of 791 rows flagged."}

Checks the `GPUSpatialWeights` invariants of weights that were built elsewhere and uploaded, on the GPU:
`offsets[0] = 0`, non-decreasing offsets with `offsets[rows] <= neighbors.length`, every neighbor ID below
the row count, strictly ascending neighbor IDs within each row (sorted and unique), no self slot, and
finite non-negative weights. `output.status.invalidCount` (one `uint32`) receives the number of rows with at least one
violation; the optional `output.flags` (one `uint32` per row) receives the bitmask of
`GPU_SPATIAL_WEIGHTS_VIOLATION` (`offsets` 1, `neighborRange` 2, `order` 4, `self` 8, `weight` 16). Run it
once after upload and before using the weights; there is no GPU sort or dedupe, so fix a failing CSR on the
CPU. `outputWeights` is the checked weights with `status.invalidCount` attached.

```ts
graph.add(new GPUSpatialWeightsValidation({
  weights: uploaded,
  output: {flags, status: {invalidCount}}
}));
```

#### How it computes

{/* lens:GPUSpatialWeightsValidation */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=datum bind=page datum=420 caption="Inspect tract 420: its violation flags, and the total number of invalid rows."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=kernel caption="Inspect the per-row invariant check kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=graph caption="Trace queen contiguity into validation on the command graph."}

### `GPUMapColoring`

Assign each tract a colour index so that no two neighbouring tracts share one.

::lens{scenario=gpu-spatial-analysis.map-coloring rung=result caption="At most five colours separate 791 rook-adjacent tracts; neighbouring tracts never match."}

Consumes an adjacency pattern only, so it takes no `spatialContext`. Weights may carry their own
context; no compatibility check is needed because coloring uses only adjacency.

Greedy parallel graph coloring (Jones-Plassmann, largest degree first by default) over a
symmetric `GPUSpatialWeights` such as contiguity, for choropleth maps where neighbors need distinct
colors (QGIS topological coloring). `output.colors` holds one `uint32` per row,
`GPU_MAP_COLORING_UNCOLORED` for rows left unfinished. Optional outputs are `colorCount`, `conflictCount`
(pairs sharing a color, 0 for a proper coloring) and `status: {converged, iterationCount}`. The per-frame
`parameters` view (`getGPUMapColoringParameterValues({seed})`, `GPU_MAP_COLORING_PARAMETER_LENGTH` `uint32`
words) selects the priorities without recompiling, and `maximumIterations` (default 64, at most 1024)
bounds the rounds. The compile-time `priority` prop selects the row order:
`'largest-degree-first'` (default; Welsh-Powell order, degree is the CSR row length, ties broken by the seeded
hash and then the lowest ID) or `'random'` (seeded hash, then lowest ID). The result equals sequential greedy
coloring in that order, a proper and seed-deterministic coloring that uses at most the maximum degree plus one
colors, not the minimum number. Largest-degree-first typically saves a color on planar maps (6 versus 6 to 7
on Delaunay graphs, matching networkx `largest_first`) at about the same number of rounds. Different seeds
give different valid colorings.

```ts
graph.add(new GPUContiguityWeights({criterion: 'rook', polygons, output: {...weights, status: {overflow}}}));
graph.add(new GPUMapColoring({weights, parameters, output: {colors, conflictCount, status: {converged}}}));
parameters.write(getGPUMapColoringParameterValues({seed: 3}));
```

{/* parameters: GPUMapColoring generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMapColoring` parameters

- Packer: `getGPUMapColoringParameterValues(settings?)`, writes `GPU_MAP_COLORING_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUMapColoringParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Seed of the hashed priorities, a uint32. Default 0. Different seeds give different valid colorings (with `'largest-degree-first'` the seed only orders rows of equal degree). |

{/* /parameters */}

#### How it computes

{/* lens:GPUMapColoring */}

::lens{scenario=gpu-spatial-analysis.map-coloring rung=datum bind=page datum=420 caption="Inspect tract 420: its colour and its neighbours, which all differ."}
::lens{scenario=gpu-spatial-analysis.map-coloring rung=kernel caption="Inspect the priority round kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.map-coloring rung=graph caption="Trace rook contiguity into rounds of greedy colouring on the command graph."}
