---
title: "GPU Spatial Analysis: Spatial joins"
sidebar_label: "Spatial joins"
description: "Spatial joins contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Spatial joins

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

### `GPUSpatialPredicateJoin`

Finds feature pairs that satisfy a spatial predicate such as intersects, contains or within, and counts matches per feature.

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=result caption="102 parks against 199 tracts; tracts shaded by the number of parks they intersect."}

Joins two sets of features by a spatial predicate, like GeoPandas `sjoin` or PostGIS
`ST_Intersects`, `ST_Contains`, `ST_Within` and `ST_DWithin`. Coordinates are planar by default; a
longitude/latitude `spatialContext` selects great-circle `dwithin` in meters (see
[Spatial context](#spatial-context-of-gpuspatialpredicatejoin)). Either side is
`{kind: 'points', positions}`, `{kind: 'lines', positions, lineOffsets}` (one linestring per
feature) or `{kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets}`
(multipolygons, holes as later rings, GeoArrow layout); all nine combinations are supported.

`predicate` is `'intersects'`, `'contains'`, `'within'`, `'dwithin'` (with an inclusive
`distance`: planar in coordinate units, or meters with a great-circle `spatialContext`), `'covers'`, `'coveredBy'`, `'touches'`, `'crosses'`, `'overlaps'`, `'equals'`,
`'containsProperly'` or `'relate'` (with a DE-9IM `pattern`). Boundaries follow OGC / DE-9IM: a point has no boundary, a linestring's boundary is its
endpoints unless it is closed, and a polygon's boundary is its rings.

- `intersects`: the closed geometries share a point, so polygons touching at an edge or corner
  intersect, and so does a point on a polygon boundary.
- `contains(a, b)`: `b` lies in the closure of `a` and the interiors intersect. A polygon does not
  contain a point or line on its boundary, and a line does not contain its own endpoints.
- `within(a, b)` is `contains(b, a)`.
- `dwithin`: the minimum distance between the closed geometries is at most `distance`.
- `covers` and `coveredBy`: `contains` and `within` that allow boundary contact.
- `touches`: the boundaries meet and the interiors do not (never true for two points).
- `crosses`: the interiors meet and the intersection has a lower dimension than the larger input, by the
  OGC dimension rules (false for polygon/polygon and point/point).
- `overlaps`: the same dimension, the interiors meet and neither contains the other.
- `equals`: the same point set. `containsProperly`: `b` lies in the interior of `a`.
- `relate` with `pattern`: any DE-9IM pattern (characters `T F * 0 1 2`), as a string or an any-of list.

The pipeline is upstream primitives end to end: per-feature bounds into a `GPUBVH` over the right
side, a count pass, `GPUScan` and an ordered write of candidate pairs, the exact predicate per
candidate (point-polygon pairs use `GPUPairwisePointInPolygon`; points on an edge it reports as
uncertain are decided by the f32 kernel and counted in `uncertainCount`), then a scan and a
capacity-bounded scatter. Outputs go in `output`; `output.pairs` (a `GPUCompactPairPort`: `leftIds`,
`rightIds`, `status` with `count`, `overflow`, optional `requiredCount`) are unique and sorted by `(left, right)`, and
scalars go in `output.status` (`overflow`, `candidateCount`, `uncertainCount`). The sides are geometry
ports (`GPUPackedGeometryPort` of the point, line or four-level polygon port). On overflow the output is a
sorted subset; `join.getCapacityPlan({candidateCount, requiredCount})` turns the read-back counts into the
next capacities (see [Shared ports, status and planning](./concepts.md#shared-ports-status-and-planning)).
`candidateCapacity` defaults to that plan's estimate, `min(rightCount, 8)` per left feature.

Pass `output.weights` to also write the matches as a `GPUSpatialWeights` with left features as rows and
weight 1; set `excludeSameRow` for a self join. A predicate join is then a weights producer, for
example `dwithin` between polygons.

The new predicates are masks over one DE-9IM matrix per candidate pair, computed from structure (vertex
locations, edge crossings, collinear overlaps, and edge pieces located against the other geometry, with
polygon areas decided from edge pieces) for all nine point, line and polygon combinations. Semantics
follow GEOS and Shapely. `intersects`, `contains`, `within` and `dwithin` keep their short-circuiting
kernels and use the matrix only when `relate` output is requested; the specs assert both paths agree.
Only bounding-box candidates are enumerated, so a pattern that admits disjoint geometries (all of II, IB,
BI and BB allow `F`) is rejected at construction; use `how: 'anti'` with `intersects` for disjoint.
Lines are single linestrings with the OGC endpoint boundary (multilinestrings and mod-2 boundaries are not
modeled). The matrix kernel is `O(n * m)` per pair, which suits small features and is not tuned for large
polygons.

Orientation tests are exact. Edge pieces between crossings are located from an exactly located endpoint
and the exact crossings between it and the piece, not from their rounded midpoint, so edges within a few
ulps of each other (near-parallel or nearly shared) classify like Shapely. Distances and the parameters
that order crossings along an edge run in f32, so recenter or project coordinates before measuring
near-degenerate geometry. Polygons must be valid.

```ts
graph.add(new GPUSpatialPredicateJoin({
  left: {kind: 'points', positions: points},
  right: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},
  predicate: 'within',
  candidateCapacity: 1 << 20,
  output: {pairs: {leftIds, rightIds, status: {count, overflow}}}
}));
```

**`relate` output.** `output.relate` is an optional `GraphDataView<'uint32'>` as long as `pairs.leftIds`; slot `k`
holds the matrix of `(leftIds[k], rightIds[k])`. Each of the nine cells (order II IB IE BI BB BE EI EB EE,
row-major, left parts then right parts) takes 2 bits at `[2k, 2k + 1]` with value `dimension + 1` (0
empty, 1 point, 2 line, 3 area). `formatGPUSpatialRelate(value)` decodes it on the CPU to a string such as
`'FF2F11212'`. Bit 31 (`GPU_SPATIAL_RELATE_UNCERTAIN_BIT`) of a slot is set when an orientation of that pair
had no sign (see **Exactness**); `formatGPUSpatialRelate` ignores it, and `value & 0x3ffff` is the matrix alone.
It is not available for `dwithin`.

```ts
graph.add(new GPUSpatialPredicateJoin({
  left, right, predicate: 'touches', candidateCapacity: 1 << 16,
  output: {pairs: {leftIds, rightIds, status: {count, overflow}}, relate, status: {uncertainCount}}
}));
```

**Anti joins.** `how: 'anti'` with `output.unmatched: GPUCompactOutput` emits the left rows that have no match, in
ascending order (`pairs`, `weights` and `relate` are not allowed with it). When any capacity overflows the
anti output is incomplete in the unsafe direction, because rows whose candidates were dropped look
unmatched, so check `output.unmatched.overflow`.

**Exactness.** Every orientation sign is exact, in the short-circuiting kernels (`engine: 'fast'`) and in
the relate engine alike: an f32 filter decides the clear cases and the rest fall back to exact integer
arithmetic (256-bit), with one inlined copy per kernel. So touches, crosses, covers, `intersects`,
`contains` and the other predicates do not depend on f32 rounding of the orientation test, and the result
does not depend on the engine or on the feature size that `engine: 'auto'` looks at, including
shared-edge and collinear configurations. The exactness is with respect to the f32 vertices; edge piece
midpoints, break parameters and distances are computed in f32: the location of a piece shorter than about
one ulp is not exact, so a line starting one ulp outside a polygon edge can have its `crosses` piece
classified as boundary.

**Uncertain pairs.** A pair is uncertain when an orientation has no sign: a non-finite coordinate, or
product exponents spanning more than 200 bits. It is never dropped silently. The undecidable orientation is
treated as collinear (sign 0) and the predicate is evaluated with that value, so the pair can come out
wrong in either direction, and it is counted in `uncertainCount` (zero means every decision is certified).
Matched uncertain pairs are identifiable per pair through bit 31 of the `relate` output. Point/polygon
pairs use the robust double-single classifier (they do not get the exact re-decision of
`GPUPointInPolygonJoin`); pairs it cannot certify are decided by the f32 kernel and counted. Not tracked
(reported as 0): the workgroup `dwithin` kernel, and the short-circuiting kernel of a polygon/polygon pair,
whose eight storage bindings leave no room for the counter.

**Per-frame distance and pattern.** `distance` is a `number` or a `GraphDataView<'float32'>`, and `pattern`
is a `GPUSpatialRelatePattern` or a `GraphDataView<'uint32'>`. A view is read from a parameter buffer, so
it changes without recompiling and gives bit-identical results to the compile-time value (a negative, NaN
or infinite distance selects nothing). Fill a pattern view with `packGPUSpatialRelatePattern(pattern,
slotCount)`, which writes `GPU_SPATIAL_RELATE_PATTERN_WORDS` (2) words per slot; unused slots are zero and
match nothing, and the helper rejects patterns that admit disjoint geometries. Size `candidateCapacity`
for the largest distance. The exact kernels dispatch by the live candidate count, but the cooperative
workgroup kernels (`dwithin` with `engine: 'relate'` and the relate engine, 64 and 256 lanes per
candidate) index their lanes with a u32, so the constructor rejects a `candidateCapacity` above
`floor((2^32 - 1) / lanes)` for them, naming the prop. `predicate` stays compile-time because it selects the kernel family; for a
per-frame predicate choice use `predicate: 'relate'` with patterns.

**Relate performance.** Per-edge break parameters are collected in one pass (up to 24 per edge) and the
covers scan runs only when a collinear edge exists. The workgroup relate kernel also builds a per-pair
y-slab index of each polygon side in workgroup memory (32 slabs, 1280 entries per side), so point location
scans the closing edges and the query's slab instead of every edge; features under 48 edges, with a
degenerate y extent or over 1280 entries use the plain scan. Results are identical. On 100 polygon pairs
the batched breaks and the slab index took relate from about 7.5 ms to 4.3 to 5 ms, against about 1 ms for
GEOS: the break collection, covers scan and vertex loops are still `O(n * m)`, and about 2 ms of the
remaining time is fixed graph and readback cost.

**`onAttribute: {left, right}`** (GeoPandas `sjoin(on_attribute=)`). Two `uint32` key views, one per left
row and one per right row. A pair survives only when the keys are equal. The filter empties candidate
slots before the exact kernels, so `candidateCount` stays the bounding-box count before filtering, and
`how: 'anti'` then reports left rows with no key-equal match. Its presence is compile-time and its
contents are per-frame. Defaults are unchanged.

#### Engine selection

`engine?: 'auto' | 'fast' | 'relate'` is compile-time and defaults to `'auto'`. It selects the kernel for
`intersects`, `contains`, `within` and `dwithin`; every other predicate always uses the relate engine.

- `'fast'` runs a short-circuiting kernel with one invocation per candidate. It is quickest for small features.
- `'relate'` runs the DE-9IM engine with one workgroup per candidate and bounding-box pruning. For `dwithin` it
  runs a workgroup distance kernel instead.
- `'auto'` picks `'relate'` when neither side is points and the product of the two sides' average vertices per
  feature is at least 1024.

Requesting `relate` output always uses the relate engine. Every engine uses the exact orientation sign, so
shared boundaries give the same answers on all three. Measured cost is about 10 ms for 100 pairs of 600 x 175
vertices. A pair with hundreds of shared edges costs 5–10 ms. A great-circle `dwithin` always uses the
short-circuiting kernel (`'relate'` throws, `'auto'` picks `'fast'`).

#### Spatial context of `GPUSpatialPredicateJoin`

`spatialContext` (omitted: native planar) selects how positions are read. It is validated at construction and
exposed as the readonly field `spatialContext`; unsupported combinations throw.

| Context | Predicates | Meaning |
|---|---|---|
| planar (default) | all | Euclidean plane; `dwithin` `distance` in coordinate units. |
| `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` | `dwithin` only | positions are longitude/latitude degrees; `distance` is meters on the mean Earth sphere (`sphereRadius` overrides the radius; `units: 'kilometers'` throws). |
| `{coordinateSpace: 'longitude-latitude', metric: 'none'}` | every predicate except `dwithin` | topology in the degree plane, as GEOS and PostGIS `geometry` do for coordinates in degrees; no distances. |

Great-circle `dwithin`: the candidate stage widens each left box by the angular radius in latitude and by the
longitude reach of that radius at the box's far latitude (`2 asin(sin(r/2) / cos(lat))`, every longitude when
the radius reaches a pole), and a box that crosses the antimeridian also probes its copy shifted by 360
degrees, with a 0.1% plus 1e-5 degree margin so rounding never drops a candidate. The exact kernel measures
the central angle from a vertex to an edge with `geodesicSegmentFraction` and `geodesicSegmentDistance`
(`GEODESIC_WGSL`): edges are straight in longitude/latitude with the longitude difference wrapped at +-180,
the fraction of the nearest point is found in the local equirectangular frame of the point, and the
distance is the haversine of coordinate differences, which keeps meter-scale offsets well conditioned in f32.
The same edge model is used by the intersection and containment tests (exact orientation in degrees, as the
planar kernels) and by the network snapping contributors, so there is one convention rather than two. The
foot is exact to first order for edges of road length, so densify long edges. Distances are accurate to f32
rounding across the antimeridian and over a pole, up to about 8000 km (a larger compile-time distance throws,
a larger per-frame value selects nothing). Pairs whose geometries intersect in the degree plane count as
distance 0. Prepare polygons that cross ±180 or enclose a pole with `GPULonLatPolygonPreparation`.
Longitude/latitude contexts require `edgeModel: 'linear'` (the default); `'great-circle'` edges throw
with a preparation hint. Optional `output.status.antimeridianUncertainCount` and
`output.status.poleUncertainCount` report unprepared rings, accumulated over both polygon inputs.
They are cleared each encoding in longitude/latitude contexts; planar contexts leave these fields
untouched and build no ring check. A numeric great-circle `dwithin` distance below ten float32
longitude/latitude quanta (about 17 m on the default sphere) warns to use `createGPULocalFrame`.
Oracle: numpy haversine and
`pyproj.Geod(a=R, f=0)` for point pairs, shapely distances in the azimuthal equidistant projection for
polygons (see the spec fixtures).

The topological predicates take `metric: 'none'`; `metric: 'great-circle'` with them throws, and `dwithin`
with `metric: 'none'` throws, so the intent is explicit. `GPUSpatialJoinCandidates` is planar only;
`GPUPointInPolygonJoin` also accepts longitude/latitude with `metric: 'none'`. `GPUNearestFeatureJoin` and
`GPUBufferSelection` also take `metric: 'great-circle'` (see their section); `GPUSpatialJoinPrepared` and `GPUNearestFeatureWeights` take no positions with a metric
(a prepared tree is a bounding-box tree and may serve a great-circle join) and have no context.

#### How it computes

{/* lens:GPUSpatialPredicateJoin */}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=datum datum=46 caption="Tract 17031031400 (row 46): its candidate parks and how many intersect it."}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=formula caption="Intersects means some interior or boundary cell of the DE-9IM matrix is non-empty."}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=kernel caption="The mask kernel reads one DE-9IM matrix per candidate pair and keeps intersecting pairs."}

::lens{scenario=gpu-spatial-analysis.spatial-predicate-join rung=graph caption="Candidate pairs from bounds, an exact relate per pair, a match mask, then scan and scatter."}

### `GPUSpatialJoinPrepared`

Builds a reusable bounds index for one side of a join, so repeated joins against it skip the rebuild.

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=result caption="One tract index, built once, serves both the candidate and the exact park join."}

A static right-hand side for repeated joins. It builds the right-side bounds and `GPUBVH` once and skips
those nodes on later encodings (CPU-conditioned, so no GPU work) until `invalidate()` is called or
`rebuildWhen(parameters)` returns true.

```ts
const prepared = new GPUSpatialJoinPrepared({geometry: polygons});
graph.add(prepared); // before the joins that use it
graph.add(new GPUSpatialPredicateJoin({
  left: movingPoints, right: polygons, prepared, predicate: 'within', candidateCapacity: 1 << 20,
  output: {pairs: {leftIds, rightIds, status: {count, overflow}}}
}));
graph.add(new GPUPointInPolygonJoin({
  points: movingPoints, polygons, prepared, candidateCapacity: 1 << 20,
  output: {pointFeatureIds, status: {overflow: pointOverflow}}
}));
prepared.invalidate(); // polygons changed: the next encoding rebuilds, later ones reuse
```

It reuses a build, not a result: every join still evaluates all candidates against the current left and
right geometry each encoding. A right side that changed without `invalidate()` keeps the old tree, so
candidates come from the old boxes and a feature that moved outside its old box is missed. Storage is
persistent buffers, allocated by the handle and freed by `destroy()`, or passed as `storage`. One handle
belongs to one graph, and several joins may share it. `spatialSort` is allowed only for
`GPUPointInPolygonJoin`. `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin` (points, lines and polygons, not
segments) and `GPUSpatialJoinCandidates` accept `prepared`. `encodedBuildCount` counts the rebuilds.

#### How it computes

{/* lens:GPUSpatialJoinPrepared */}

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=datum datum=46 caption="Tract 17031031400 (row 46): matched through the prepared index, not a rebuilt one."}

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=kernel caption="Prepare kernels copy tract bounds and edges once; the joins only read those buffers."}

::lens{scenario=gpu-spatial-analysis.spatial-join-prepared rung=graph caption="Prepared nodes sit behind a rebuild condition; both joins read the same prepared buffers."}

### `GPUSpatialJoinCandidates`

Lists feature pairs whose bounding boxes overlap: the filter step before exact predicates.

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=result caption="Park and tract pairs with overlapping boxes; boxes can overlap where polygons do not."}

Maximum: `output.leftIds.length` is at most `getGPUSpatialJoinCandidatesMaximumPairCapacity(device)` (2^24 on a default device).

The bounding-box candidate stage on its own: `(left, right)` pairs sorted by `(left, right)` whose bounding
boxes overlap, with the left box optionally expanded by `distance`. The `output` is a `GPUCompactPairPort`
(capacity `output.leftIds.length`, `status.count`, `status.overflow`, `status.requiredCount`); slots past `status.count` hold `0xffffffff`.
It is conservative: no intersecting pair is missed, and some candidates do not intersect. It has no
`excludeSameRow`.

`distance` is planar in coordinate units (`spatialContext` defaults to native planar coordinates and a
`longitude-latitude` context throws; use `GPUSpatialPredicateJoin` `dwithin` for great-circle candidates).
It may be a compile-time number or a one-row `GraphDataView<'float32'>`. The view is read every
encoding, so an interactive distance join can reuse one compiled graph. Size the pair capacity for the
largest distance that the application permits. Negative, NaN and infinite view values act as zero; invalid
compile-time numbers throw.

```ts
const candidateDistance = new GPUParameterBuffer(device, {
  id: 'candidate-distance',
  format: 'float32',
  length: 1
});

graph.add(new GPUSpatialJoinCandidates({
  left,
  right,
  distance: candidateDistance.importToGraph(graph),
  output: {leftIds, rightIds, status: {count, overflow, requiredCount}}
}));

const compiled = await graph.compileAsync();
candidateDistance.write(Float32Array.of(2_000));
compiled.encode(commandEncoder, {parameters: undefined});
```

#### How it computes

{/* lens:GPUSpatialJoinCandidates */}

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=datum datum=46 caption="Tract 17031031400 (row 46): every park whose box overlaps the tract's box."}

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=kernel caption="Count and write kernels test box overlap; the write pass fills each row's scanned slots."}

::lens{scenario=gpu-spatial-analysis.spatial-join-candidates rung=graph caption="Count candidates per row, prefix-scan to offsets, then write each row's pairs in place."}

### `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin`

**`GPUPointInPolygonJoin`**: Assigns each point the polygon that contains it, under an explicit boundary rule, and counts points per polygon.

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=result caption="2,344 observations (grey) assigned to the containing tract; colour is the tract."}

**`GPUNearestFeatureJoin`**: Finds each point's nearest point or segment feature within a radius and reports its id and distance.

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=result caption="Observations coloured by planar distance to the nearest library."}

**`GPUNearestFeatureWeights`**: Turns each point's k nearest features into a CSR spatial-weights matrix with inverse-distance weights.

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=result caption="Each observation linked to its three nearest libraries, weighted by inverse distance."}

Maximum (`GPUNearestFeatureJoin`, neighbors mode and great-circle): the node table `5 * (2 * leafCapacity - 1) + 2 * queries` (plus `2 * features` with a filter) is at most `getGPUNearestFeatureJoinMaximumNodeTableWordCount(device)` (2^25 words on a default device), which allows 2^21 leaves without queries; `leafCapacity` is at most `getGPUNearestFeatureJoinMaximumLeafCapacity(device)` (2^23).

Spatial joins over a `GPUBVH` built from feature bounds every encoding. `GPUPointInPolygonJoin`
classifies each (point, feature) candidate with the shared exact orientation sign (even-odd rings, union
polygons) and assigns each point the containing feature with the smallest row (see the boundary rule
below); `GPUNearestFeatureJoin` measures candidates with
`GPUPairwisePointSegmentDistance` and keeps the nearest point or segment feature within a per-frame
radius. Both take the points as a `GPUPointGeometryPort` (`points` and `queries`; chunked positions and
`sourceIds` allowed) and the features as a geometry port whose optional `sourceIds` replace feature rows in
the outputs. Both write, in `output`, per-point feature IDs, per-feature counts, optional compact matches,
and one `output.status.overflow` flag covering BVH leaf, candidate, and match capacity. `GPUPointInPolygonJoin` takes planar
coordinates (the default) or `{coordinateSpace: 'longitude-latitude', metric: 'none'}`. Containment is
topological with straight edges in degrees, as in GeoJSON, and gives the same result as planar. A
longitude/latitude context with a metric such as `'great-circle'` throws at construction, because the join
has no geodesic edges ("project positions to a planar frame first"). `GPUNearestFeatureJoin` also takes the
great-circle context below.

Both joins require linear longitude/latitude edges (omitted `edgeModel` means `'linear'`);
`edgeModel: 'great-circle'` throws with a preparation hint. Prepare polygons that cross ±180 or enclose
a pole with `GPULonLatPolygonPreparation`. Optional `output.status.antimeridianUncertainCount` and
`output.status.poleUncertainCount` report unprepared rings; nearest joins count both polygon inputs.
These counters reset each encoding for longitude/latitude and remain untouched for planar contexts,
which build no ring check. Empty rings and polygons in the preparation's fixed 2x topology are accepted.
`getPointInPolygonContainmentContext` preserves the input edge model and datum when selecting
`metric: 'none'`, so it cannot bypass edge-model validation.

**Great-circle nearest features (`GPUNearestFeatureJoin`, `GPUBufferSelection`).** With
`spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` positions are longitude and
latitude in degrees and every distance (`radius`, `nearestDistances`, `neighborDistances`,
`GPUBufferSelection` `distance` and `output.distances`) is in the context's length unit (meters by default;
kilometers with `units: 'kilometers'`) on the mean Earth sphere (`sphereRadius` overrides the radius in
that unit). A `longitude-latitude` context with another metric and planar contexts with `great-circle`
throw at construction. Both distance operators reject `local-lambert-equal-area` frames. Both output modes use the branch-and-bound BVH traversal of the neighbors mode (nearest-feature
mode is the `k = 1` traversal followed by the nearest-feature columns, counts and matches), so
`candidateCapacity` is not used on this path (it is accepted and ignored, `candidateCount` reads 0, and
`GPUBufferSelection` still validates it). The traversal orders and bounds by central angle. A node is pruned
with a lower bound of the great-circle distance between its box and the query box,
`hav(d) = hav(dlat) + cos^2(max |lat|) hav(dlon)`, with the latitude gap, the circular longitude gap (through
the antimeridian when that is shorter) and the cosine of the latitude farthest from the equator over both
boxes, which never overestimates; a box 180 degrees or wider in longitude (a feature across the antimeridian)
counts as covering every longitude, and a box touching a pole has cosine 0. Edges are straight in
longitude/latitude with wrapped longitude differences, so the vertex boxes bound them exactly. Point to edge
distance uses `geodesicSegmentFraction` and `geodesicSegmentDistance`; a crossing or containment is decided
in the degree plane with the exact orientation sign after the second edge is shifted by a multiple of 360
degrees to overlap the first, and counts as distance 0. Foot points and query points are on the edges with
longitude wrapped to +-180. The mixed convention of the planar-in-degrees tests is therefore one edge model;
densify edges longer than a few kilometers, because the foot is exact to first order in the local frame of the
query. Prepare crossing and pole-enclosing polygons as described above.
No warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).
The planar kernels are unchanged and compile
none of this code. Oracle: `geodesic-nearest-fixtures.ts` (numpy 2.5.4 haversine of the edge densified in
longitude/latitude, `pyproj` 3.8.0 `Geod(a=R, f=0)` cross-check), scenes at the equator, latitude 60, across
the antimeridian and next to the pole. `GPUNearestFeatureWeights` adapts the join's outputs; pass the
join's `spatialContext` so its distances and `distanceFloor` share the context's length unit and
inverse-distance weights use those distances.

**Exact orientation.** The classifier and the nearest-feature kernels use the same exact orientation sign
(an f32 filter with an exact 256-bit integer fallback), so "on the edge" means the same thing in the
point-in-polygon join, the predicate join and the nearest-feature join: a point exactly on an edge is
distance 0 in all three, and polygon containment (distance 0 inside a polygon) is decided with the exact
sign too. Distances and foot points stay f32. Only candidates that no arithmetic can decide stay
`uncertain`: non-finite input or malformed offsets. In `GPUPointInPolygonJoin` an empty polygon part
or a ring with fewer than three vertices contains nothing and is skipped, as in shapely, so the
feature's other parts still classify the point. Uncertain
pairs are never assigned. They are counted in the optional `uncertainCount`, so such a point stays
unassigned visibly rather than silently.

**Boundary rule (`GPUPointInPolygonJoin`).** `boundaryRule` decides a point exactly on a polygon
boundary:

| Rule | Boundary point | Notes |
|---|---|---|
| `'half-open'` (default) | belongs to exactly one feature of a valid coverage, independent of row order | The owner is the feature containing the point after an infinitesimal move toward +x and +y (+y dominates): left and bottom boundaries are included, right and top boundaries are excluded. A horizontal edge belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the feature whose interior touches its upper right. Points on the outer right or top boundary of a coverage belong to no feature. |
| `'closed'` | contained by every feature it touches, then the smallest row wins | Nothing is lost on the outer boundary; the owner of a shared-boundary point depends on the row order. |
| `'open'` | belongs to no feature | Only strict interior counts; boundary points are lost, not half-assigned. |

The half-open decision is the crossing-number convention made exact: a
supporting-line tie is broken by the sign of dx (the epsilon term of the move), or of -dy for a vertical
edge, so the result follows from the exact orientation signs and the vertex order does not matter. Overlapping
features (an invalid coverage) still resolve to the smallest containing row. Oracle: shapely
`Polygon.contains(Point(x + 1e-9, y + 1e-6))` on integer scenes, 24 row permutations, in the specs.
`GPUZonalStatistics` and the recipes that pass no `boundaryRule` use the half-open default, so a coverage's
counts partition the points.

```ts
graph.add(new GPUNearestFeatureJoin({
  queries: {kind: 'points', positions: points},
  features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  radius: snapRadius.importToGraph(graph), candidateCapacity: points.length * 4,
  output: {nearestFeatureIds, nearestDistances, status: {overflow}}
}));
```

Both joins accept a [`prepared`](#gpuspatialjoinprepared) handle, so a static polygon or line set builds its
`GPUBVH` once. The handle must be added to the graph first, and `spatialSort` and `leafCapacity` then come
from the handle (conflicting props throw).

**k nearest features.** Set `output.neighborIds` and `output.neighborCounts` on `GPUNearestFeatureJoin` to get the `k`
nearest features per query (PostGIS `a <-> b ORDER BY ... LIMIT k`, GeoPandas `sjoin_nearest`). The
original mode above is unchanged. Each query walks the BVH nearest child first and prunes boxes farther than
the current `k`-th distance, which is exact and deterministic.

- `queries` are points (possibly chunked), lines or polygons (one packed view each); features are `points`,
  `segments`, `lines` or `polygons` (multipolygons, holes, even-odd). Distance is the minimum planar
  distance between the geometries and 0 when a polygon contains the other geometry. Polygons must be valid.
- `k` is 1 to 32. `ties` is `'lowest-id'` (default) or `'all'`, which keeps every feature tied with the
  `k`-th. `neighborCapacity` (slots per query, `>= k`, `<= 64`) can exceed `k` only with `ties: 'all'`.
  `radius` (per-frame, inclusive; omitted means unbounded) bounds the search.
- Outputs are dense, `query * neighborCapacity + slot`, ordered by distance and then feature row:
  `neighborIds` (remapped by `features.sourceIds`), `neighborCounts`, optional `neighborDistances`,
  `neighborFootPoints` (the nearest point on the feature, or the crossing point when geometries cross)
  and `neighborSegmentIndices` (the start-vertex index of the nearest edge; `GPU_NEAREST_NO_SEGMENT` for
  points and containment). Unused slots hold `GPU_SPATIAL_JOIN_NO_FEATURE`, -1, NaN and
  `GPU_NEAREST_NO_SEGMENT`.
- `output.status.overflow` is 1 when `ties: 'all'` had more ties than `neighborCapacity` or the BVH leaf capacity was
  exceeded. Distances are f32, so near-equal distances can order differently from a double reference, and
  tie order and the foot point at exact vertex ties depend on evaluation order.
- The per-query scratch (result slots, and the foot-point scratch when a foot-point output is requested)
  is one binding while it fits `maxStorageBufferBindingSize` (128 MiB by default). Beyond that, for
  example 838k queries with `k = 8` and foot points, the scratch holds a fixed chunk of queries (a multiple
  of 64) that every chunk of queries reuses in turn, so the query count is not limited by the scratch. The
  caller-owned output views still have to fit their own bindings.

- `exclusive` skips features whose ID equals the query's ID (`features.sourceIds[row]` or the row, and optional
  `queryIds`, which require `exclusive`), so the `k` neighbors are the nearest other features
  (`STRtree.query_nearest(exclusive=True)`, `sjoin_nearest` self-joins). It compares IDs, not geometry
  equality, which is the same when IDs are unique per geometry. Both modes support it.
- `neighborQueryPoints` (`float32x2`, the layout of `neighborIds`) is the point of the query geometry
  nearest to the feature, NaN in unused slots. `(neighborQueryPoints, neighborFootPoints)` is
  `shapely.shortest_line` or `ops.nearest_points` for every point, line and polygon pair. Ties between
  parallel edges return the first in vertex order, which can differ from GEOS.
- `onAttribute: {left, right}` (as on `GPUSpatialPredicateJoin`) finds the nearest key-equal feature:
  the filter runs inside the traversal, and nearest-feature mode filters candidates in the reduce passes.

```ts
graph.add(new GPUNearestFeatureJoin({
  queries: {kind: 'points', positions: points},
  features: {kind: 'lines', positions, lineOffsets},
  k: 3, ties: 'all', neighborCapacity: 8, radius: radiusView, spatialSort: true,
  output: {
    neighborIds, neighborCounts, neighborDistances, neighborFootPoints, neighborSegmentIndices,
    status: {overflow}
  }
}));
```

`spatialSort: true` reorders features along a Morton curve of their bound centers before the
`GPUBVH` build. Results are identical, and traversal is 30x to 75x cheaper when features are not
spatially coherent in row order.

#### `GPUNearestFeatureWeights`

Adapter from the k-nearest output of `GPUNearestFeatureJoin` to a `GPUSpatialWeights` CSR (libpysal 4.15 `KNN`
and `Graph.build_knn`, for points, lines or polygons, using the join's exact geometry distances). Pass the
join's `neighborIds`, `neighborCounts`, optional `neighborDistances` and its `neighborCapacity` as
`slotCapacity`. Pass the join's resolved `spatialContext` to the adapter (omitted means native planar);
`distanceFloor` uses its length unit and `outputWeights.spatialContext` carries it to consumers. Per row it keeps the first `k` valid slots in the join's distance order, drops padding and
(with `excludeSelf`, for self-joins run with `k + 1`) the row's own ID, then sorts the IDs ascending. Short rows
stay short. `weightType` is `'binary'` (default) or `'inverse-distance'` (`1 / max(d, distanceFloor)^power`),
and `output.distances` is filled when present (in the join's context length unit, also used by `distanceFloor`). The output is directed, so for `'symmetrize'` or symmetric
statistics take the union of the weights and their `GPUSpatialWeightsTranspose`. `excludeSelf` compares IDs to
the query row, so do not use feature `sourceIds` in the join with it. `output` is a
`GPUSpatialWeightsOutput` (the CSR views plus `status.overflow`).

The spec pins libpysal on 24 random points with `k = 4` (identical neighbor sets). Deviations: ties at the
`k`-th distance keep the lowest feature rows (libpysal breaks them in KDTree order, and `KNN` and
`Graph.build_knn` disagree with each other on grids); inverse-distance floors the distance instead of
dividing by zero.

```ts
graph.add(new GPUNearestFeatureWeights({
  neighborIds, neighborCounts, neighborDistances, slotCapacity: 5, k: 4, excludeSelf: true,
  output: {...knnWeights, status: {overflow}}
}));
```

#### Spatial sort default

`spatialSort` of `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin` and `GPUBufferSelection` defaults to `true`
from 256 features and `false` below; pass `false` to turn it off. With shuffled features each encoding is
50–125x faster; with spatially coherent features it is about 1 ms slower. `GPUSpatialJoinPrepared` still
defaults to `false`.

`spatialSort` orders the features along a Hilbert curve (the default) or a Morton curve
(`spatialSortCurve: 'morton'`, a prop of the three joins above, type `SpatialSortCurve`) before the BVH build.
Results are identical; in paired runs the Hilbert joins were 10 to 15 percent faster (point-in-polygon,
nearest, buffer selection). The prepared join and `GPUZoneEvents` have no curve prop and use Hilbert.

`querySort` orders queries the same way: each point chunk, or the bounds centers of line/polygon queries, is
processed along a Hilbert curve over the root bounds of the feature BVH (non-finite points and empty bounds last,
equal keys in row order), so neighboring invocations walk
neighboring subtrees. Every query still reads and writes its own row, so all outputs are identical, including which
candidate pairs a `candidateCapacity` overflow drops. It is a prop of `GPUPointInPolygonJoin`, `GPUNearestFeatureJoin`
and `GPUBufferSelection` and defaults to `true` for point queries from 4096 queries and 256 features; line and polygon queries
default to `false` because the measured sort cost exceeds the traversal saving.
Lines default off: the measured 8192-line workload took 0.80 ms without sorting and 1.20 ms with sorting;
explicit `querySort: true` remains available. Point sorting stays enabled; polygon defaults are unchanged.
Line/polygon queries use the same source-row output addressing, including
chunked scratch and foot-point outputs; polygon query topology is packed to keep every pass within eight storage
bindings. Geometry sorting adds bounds and center passes. With a planar `radius` the second candidate pass keeps
input order.

`edgeSlabs` (`GPUPointInPolygonJoin`) makes the exact refine test only the edges of the y-slab at the point's height
in each polygon part, instead of every edge of the candidate feature: O(slab) per candidate rather than O(vertices).
Results are identical (all boundary rules, the half-open owner, malformed offsets as UNCERTAIN); the one exception is
a pair that the full walk reports UNCERTAIN only because an unrelated far edge defeats the exact orientation test
(terms more than 200 binary orders apart). It defaults on when `polygons.positions.length * candidateCapacity` is at
least 4,000,000 and features average 8 vertices. `new GPUSpatialJoinPrepared({geometry, edgeSlabs: true})` builds the
index once with the BVH, and joins given that handle reuse it. Slab counts are limited by GPU-computed
total vertical edge travel and the entry budget, including ring closure. This prevents jagged long rings
from overflowing the index and reverting to a full walk through indirect edge references.

#### How it computes

{/* lens:GPUPointInPolygonJoin */}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=datum datum=0 caption="Observation 0: its candidate tracts and the one whose rings contain it."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=formula caption="Crossing parity with the half-open boundary rule; rings XOR within a part, parts OR."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=kernel caption="The classify kernel walks each candidate tract's edges with exact orientation signs."}

::lens{scenario=gpu-spatial-analysis.point-in-polygon-join rung=graph caption="Bounding-box candidates first, then one exact classify pass per candidate pair."}
#### How it computes

{/* lens:GPUNearestFeatureJoin */}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=datum datum=0 caption="Observation 0: its nearest library row and the planar distance in metres."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=formula caption="Argmin of squared distance, ties to the lower feature row; distance published as metres."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=kernel caption="The traverse kernel descends the library BVH, pruning boxes farther than the best so far."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-join rung=graph caption="BVH build over libraries, then one traversal per observation."}
#### How it computes

{/* lens:GPUNearestFeatureWeights */}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=datum datum=0 caption="Observation 0: its CSR row of three library neighbours and their weights."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=formula caption="Weight is one over distance (floored at 1 m) to the power; no row standardisation."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=kernel caption="The emit kernel writes each row's neighbour ids in ascending order with their weights."}

::lens{scenario=gpu-spatial-analysis.nearest-feature-weights rung=graph caption="Nearest search, per-row counts, offset scan, then the CSR emit pass."}

### `GPUPairGather`, `GPUOffsetExpansion`, and `GPUBoundsFilter`

**`GPUPairGather`**: Copies attributes from the source rows of matched pairs onto the pair rows.

::lens{scenario=gpu-spatial-analysis.pair-gather rung=result caption="Park area copied onto each park and tract pair, without clipping or apportioning."}

**`GPUOffsetExpansion`**: Expands CSR offsets into one row per member, with its owner index and local index.

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=result caption="Observations grouped by tract receive their tract index and position within it."}

**`GPUBoundsFilter`**: Flags features whose bounds intersect, lie within, or contain an axis-aligned box: a bounds test, not exact geometry.

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=result caption="Observations inside the Lincoln Park bounding box (border included) are selected."}

Dataframe joints around the joins. `GPUPairGather` consumes join row indices and columns, so it takes no
`spatialContext`.

- **`GPUPairGather`** merges join output with attribute columns (GeoPandas `sjoin` and `sjoin_nearest`
  merges, `distance_col`). The input is `pairs` (a `GPUSpatialPredicateJoin` pair output) or `neighbors`
  (the dense `[query, slot]` lists of `GPUNearestFeatureJoin`, flattened by stable compaction).
  `how: 'inner'` emits matched rows, and `'left'` appends the unmatched left rows (`unmatchedLeft`, the
  anti-join output, derived automatically for neighbors) with `rightRows = GPU_PAIR_GATHER_NO_ROW` and
  NaN or `0xffffffff` in the right and slot columns. `leftColumns` and `rightColumns` gather `float32` or
  `uint32` columns by row, and `slotColumns` carry slot-aligned columns such as `neighborDistances`. The
  output order is matched pairs then unmatched, which is not GeoPandas' left order (sort by
  `(left, right)` to compare). It is bounded by `output.leftRows.length`, and `output.status.overflow` ORs the
  input flags and `overflowSources`. Chunked `GraphVectorView` inputs are unsupported.
- **`GPUOffsetExpansion`** gives the owner index and local index of every child row from an offsets view
  (`explode`, `get_coordinates(return_index=True)`, `get_parts(return_index=True)`) by binary search per row.
  Empty ranges yield no rows, `ownerMap` chains levels (vertex to ring to polygon to feature), and the
  output is bounded with `status.count`, `status.overflow` and `status.requiredCount`. Offsets monotonicity is not checked.
- **`GPUBoundsFilter`** (`spatialContext`: planar, or longitude/latitude with metric `'none'`, treated
  as plain coordinates with no antimeridian wrap, so a box needs `minX <= maxX`) is a `.cx`-style mask and compaction over `float32x4` feature bounds with a
  per-frame box (`parameters`, written with `getGPUBoundsFilterParameterValues`; the optional 0/1 mask is
  `output.outputMask`). Modes are `'intersects'`, `'within'` and
  `'contains'`, with closed comparisons and non-finite bounds rejected. It is bounds-level only, and OGC
  `envelope.within(box)` rejects bounds that only touch the border where this accepts them, so refine
  with `GPUSpatialPredicateJoin`.

Matched GeoPandas 1.2.0 `sjoin` (inner 11 pairs, left 32 rows) and `sjoin_nearest` with ties and
`distance_col`, and `explode` and `get_coordinates` indices, exactly.

{/* parameters: GPUBoundsFilter generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUBoundsFilter` parameters

- Packer: `getGPUBoundsFilterParameterValues(settings)`, writes `GPU_BOUNDS_FILTER_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUBoundsFilterParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minX` | `number` | Left edge. |
| `minY` | `number` | Bottom edge. |
| `maxX` | `number` | Right edge, at least `minX`. |
| `maxY` | `number` | Top edge, at least `minY`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUPairGather */}

::lens{scenario=gpu-spatial-analysis.pair-gather rung=datum datum=46 caption="Tract 17031031400 (row 46): the park areas gathered onto its pairs."}

::lens{scenario=gpu-spatial-analysis.pair-gather rung=kernel caption="Each pair row reads its source row and copies the attribute into its slot column."}

::lens{scenario=gpu-spatial-analysis.pair-gather rung=graph caption="Slot assignment, one gather pass per attribute column, then publication of the gathered rows."}
#### How it computes

{/* lens:GPUOffsetExpansion */}

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=datum datum=0 caption="The first observation in tract order: its tract and its local index within it."}

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=kernel caption="One invocation per member binary-searches the offsets for its owner, then subtracts the start."}

::lens{scenario=gpu-spatial-analysis.offset-expansion rung=graph caption="A single expansion pass over the CSR offsets; no sort or scan is needed."}
#### How it computes

{/* lens:GPUBoundsFilter */}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=datum datum=0 caption="Observation 0: its position against the Lincoln Park box, and the mask bit."}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=formula caption="Within: every bound inside the box, borders included, compared in float32."}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=kernel caption="One invocation per feature compares its bounds with the box and writes one mask word."}

::lens{scenario=gpu-spatial-analysis.bounds-filter rung=graph caption="One mask pass, then a visibility workflow compacts and publishes the selected rows."}

### `GPUBufferSelection`

Selects features within a distance of any reference feature, without building buffer polygons.

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=result caption="Observations within 300 m of a library are selected; distance is straight-line, not walking."}

No warning is possible because the distance is a GPU view; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

Selects points within a per-frame distance (inclusive) of point or segment features. A
polyline is passed as consecutive segment rows. The contributor composes `GPUNearestFeatureJoin`. Unlike
the join's unordered `matches`, it writes a source-aligned 0/1 mask and stable IDs in ascending row
order (`output.outputMask`, `output.matches`), with a clamped count and overflow that covers both the
output capacity and the join. `points` is a `GPUPointGeometryPort` whose `sourceIds` become the match IDs.
`output.drawInstanceCount` (one `uint32` row, requires `output.matches`) receives the clamped selected count
from the publish node, so it can be bound as an indirect draw instance count. The distance is planar in
coordinate units by default; with `spatialContext: {coordinateSpace: 'longitude-latitude', metric:
'great-circle'}` it is a great-circle distance in the context's length unit (meters by default; kilometers
with `units: 'kilometers'`). The context is forwarded to `GPUNearestFeatureJoin`, whose section lists the
edge model, sphere radius and rejected combinations; `candidateCapacity` is ignored.
It measures distance, so `local-lambert-equal-area` frames throw.

```ts
graph.add(new GPUBufferSelection({
  points: {kind: 'points', positions: points},
  features: {kind: 'segments', starts: roadStarts, ends: roadEnds},
  distance: bufferDistance.importToGraph(graph), candidateCapacity: points.length * 4,
  spatialSort: true, output: {outputMask, matches}
}));
```

#### How it computes

{/* lens:GPUBufferSelection */}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=datum datum=0 caption="Observation 0: distance to its nearest library and whether it is within 300 m."}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=formula caption="Selected when the nearest-library distance is between 0 and r, here 300 m."}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=kernel caption="The mask kernel turns each nearest distance into a selection bit."}

::lens{scenario=gpu-spatial-analysis.buffer-selection rung=graph caption="A nearest-library join within 300 m, a mask pass, then compaction of the selected rows."}
