---
title: "GPU Spatial Analysis: Cells (DGGS)"
sidebar_label: "Cells (DGGS)"
description: "Cells (DGGS) contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Cells (DGGS)

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

### `GPUCellAggregation`, `GPUCellRollup`, `GPUCellPyramid`, and `GPUCellLevelSelection`

**`GPUCellAggregation`** — Counts and sums rows per Quadbin or H3 cell into a sorted, capacity-bounded cell table.

::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=result caption="Blue: Quadbin zoom-18 cells shaded by place count · grey: 1,275 Wicker Park places"}

**`GPUCellRollup`** — Reduces a cell table to a coarser resolution by summing child counts and sums, without re-reading the rows.

::lens{scenario=gpu-spatial-analysis.cell-rollup rung=result caption="Grey: 115 zoom-18 cells · blue: 12 zoom-16 parents shaded by summed place count"}

**`GPUCellPyramid`** — Aggregates rows once at the finest resolution, then rolls each level into the next coarser table.

::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=result caption="Grey: zoom-18 level · blue: zoom-16 level of the same pyramid, shaded by place count"}

**`GPUCellLevelSelection`** — Publishes the cell count and first row of a pyramid level chosen by one per-frame index.

::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=result caption="Blue: the selected zoom-17 level, shaded by place count · grey: places"}

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
- **Coordinate space.** `spatialContext` is checked at construction and exposed as a resolved
  field. `positions` are longitude/latitude degrees (default `{coordinateSpace:
  'longitude-latitude', metric: 'none'}`); a planar context throws, because Quadbin keying is
  defined on degrees. Pre-keyed `cells`, `GPUCellRollup` and `GPUCellPyramid` use a `discrete-grid`
  context (default `gridFamily` is `family`): a given `gridFamily` must equal `family`, and a given
  `gridResolution` (the finest resolution of the input keys) must be at least `resolution`. Nothing
  is measured, so only metric `'none'` is accepted.
- **Cell table** (`GPUCellTable`). Capacity is `cells.length`. Rows `[0, status.count)` hold occupied
  cells in ascending key order. Later rows hold the empty key `0xffffffff` in both words, count 0
  and NaN extremes. Columns: `counts`, exact fixed-point `sums` (signed 64-bit words of
  `roundHalfEven(fround(value * sumScale))`, default scale 65536), `sumValues` (f32
  `sums / sumScale`), `minimums` and `maximums`. On overflow the table keeps the smallest keys
  and sets `status.overflow`.
- **Roll-up.** `GPUCellRollup` reduces a table to a coarser resolution without re-reading rows
  or sorting. Parent keys truncate the cell path (`cellToParent`), which preserves key order.
  Counts and sums are integer sums, so a roll-up equals direct aggregation bit for bit. The
  source table's `status.overflow` carries through.
- **Pyramid.** `GPUCellPyramid` aggregates once at the finest of `resolutions` (listed finest first),
  rolls each level into the next, writes one table per level to `output.levels` and can write
  per-level counts to `output.levelCounts`. Intermediate
  sums and extremes that a coarser level needs become transients.
- **Level selection.** `GPUCellLevelSelection` reads a per-frame `activeLevel` (one `uint32`) and
  writes that level's `status.count`, first row, and the `instanceCount`/`firstInstance` words of indirect
  draw arguments. Keep it in a small per-frame graph: switching levels rewrites one word and never
  re-runs the pyramid.

```ts
graph.add(new GPUCellPyramid({
  family: 'quadbin',
  positions, values, mask,                    // mask and values can change every frame
  resolutions: [18, 14, 10, 6, 2],
  output: {levels: tables, levelCounts}
}));
selectionGraph.add(new GPUCellLevelSelection({
  levelCounts, activeLevel: activeLevel.importToGraph(selectionGraph),
  output: {status: {count}, drawArguments}
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

#### How it computes

{/* lens:GPUCellAggregation */}

::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=datum datum=0 caption="The first occupied cell: its key and how many places fell inside"}

::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=formula caption="Cell count is the number of accepted rows keyed into the cell"}

::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=kernel caption="Key, sort, scan segment heads, then atomic adds into table rows"}

::lens{scenario=gpu-spatial-analysis.cell-aggregation rung=graph caption="Keying, radix sort, head scan and the per-cell reduction nodes"}

{/* lens:GPUCellRollup */}

::lens{scenario=gpu-spatial-analysis.cell-rollup rung=datum datum=0 caption="The first parent cell: its count is the sum of its children"}

::lens{scenario=gpu-spatial-analysis.cell-rollup rung=formula caption="Parent counts are integer sums of child counts, so totals are conserved"}

::lens{scenario=gpu-spatial-analysis.cell-rollup rung=kernel caption="Truncate each key to its parent, scan heads, reduce children per parent"}

::lens{scenario=gpu-spatial-analysis.cell-rollup rung=graph caption="One roll-up node reads the source table and writes the parent table"}

{/* lens:GPUCellPyramid */}

::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=datum datum=0 caption="The first zoom-16 cell: its count rolled up through zoom 17"}

::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=formula caption="Each level sums the counts of the finer level below it"}

::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=kernel caption="Aggregation kernels at level 0, roll-up kernels for each coarser level"}

::lens{scenario=gpu-spatial-analysis.cell-pyramid rung=graph caption="One aggregation, two chained roll-ups and the level-count packing node"}

{/* lens:GPUCellLevelSelection */}

::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=datum datum=0 caption="The first cell of the selected zoom-17 level and its count"}

::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=formula caption="The level is the active index clamped to the last level"}

::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=kernel caption="One thread reads the active level and writes its count and first row"}

::lens{scenario=gpu-spatial-analysis.cell-level-selection rung=graph caption="A single selection node beside the pyramid it reads"}

### `GPUPointToCell` and `GPUCellGeometry`

**`GPUPointToCell`** — Keys longitude/latitude points to 64-bit cell keys of a discrete global grid family at one resolution.

::lens{scenario=gpu-spatial-analysis.point-to-cell rung=result caption="Grey: every 4th Wicker Park place · blue: the H3 resolution-9 cell each falls in"}

**`GPUCellGeometry`** — Decodes cell keys to centres and boundary polygons in longitude/latitude degrees.

::lens{scenario=gpu-spatial-analysis.cell-geometry rung=result caption="Blue: H3 resolution-9 cell boundaries · grey: the places they were keyed from"}

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
  - `spatialContext` (optional) must be longitude/latitude without a metric, which is also the
    default; planar contexts throw. `GPUCellGeometry` takes cell ids, so it has no `spatialContext`
    prop; its decoded `centers` and `boundaries` are described by `outputSpatialContext`.
- **`GPUCellGeometry`.**
  - Input is `cells` (`wordOrder` option).
  - Outputs are `centers` (lng/lat) and optional `boundaries` with a fixed `maximumVertexCount`
    stride and `vertexCounts`. Families are the five above plus `'a5'`.
  - H3 boundaries follow h3-js `cellToBoundary` exactly in vertex count and order. That includes
    face-crossing distortion vertices (up to 10, so use stride 10) and pentagons.
  - Quadbin, quadkey, geohash and S2 have 4 vertices; A5 has 5.
  - `edgeModel` and `outputSpatialContext.edgeModel` are `'great-circle'` for H3, S2 and A5,
    and `'linear'` for quadbin, quadkey and geohash. Before linear polygon consumers, prepare
    great-circle boundaries with `GPULonLatPolygonPreparation`; see the
    [H3 edge-deviation table](#gpucellsetoutline) below.

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

#### How it computes

{/* lens:GPUPointToCell */}

::lens{scenario=gpu-spatial-analysis.point-to-cell rung=datum datum=0 caption="Place 0: its longitude/latitude and the H3 key written for it"}

::lens{scenario=gpu-spatial-analysis.point-to-cell rung=kernel caption="One thread per place runs the H3 face, IJK and digit encoder"}

::lens{scenario=gpu-spatial-analysis.point-to-cell rung=graph caption="A single keying node: positions in, 64-bit cell keys out"}

{/* lens:GPUCellGeometry */}

::lens{scenario=gpu-spatial-analysis.cell-geometry rung=datum datum=0 caption="Cell 0: its centre and the six boundary vertices decoded from its key"}

::lens{scenario=gpu-spatial-analysis.cell-geometry rung=kernel caption="One thread per key decodes the centre and boundary vertices into fixed strides"}

::lens{scenario=gpu-spatial-analysis.cell-geometry rung=graph caption="A single decode node: cell keys in, centres, boundaries and counts out"}

### `GPUCellTopology` and `GPUCellCompaction`

**`GPUCellTopology`** — Expands each cell key to its grid disk, ring, parent or children as a fixed-stride row.

::lens{scenario=gpu-spatial-analysis.cell-topology rung=result caption="Blue: the radius-4 disk around the central place, shaded by grid steps from its cell"}

**`GPUCellCompaction`** — Merges complete sibling sets of cells into parents (compact) or expands cells to descendants (uncompact).

::lens{scenario=gpu-spatial-analysis.cell-compaction rung=result caption="Grey: 49 resolution-9 cells, two masked out · blue: the merged set, resolutions 8 and 9"}

Cell topology for Quadbin and H3 keys stored as `uint32x2` little-endian `(low, high)` words.
This is the Arrow `Uint64` layout used by `GPUCellAggregation`. Inputs also accept
`wordOrder: 'high-low'`. A zero key means "no cell" in every output.

Both consume cell keys, not positions, so neither takes a `spatialContext`.

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
  - Both write a capacity-bounded `{cells, status: {count, requiredCount?, overflow, droppedCount?}}`.
    Compact output is in ascending canonical key order. Uncompact output is ascending when the
    input is in path order.

```ts
graph.add(new GPUCellTopology({
  family: 'h3', operation: {type: 'disk', k: 2},
  cells, output: {cells: diskCells, distances, counts}   // diskCells.length = rows * 19
}));
graph.add(new GPUCellCompaction({
  family: 'h3', operation: {type: 'compact', resolution: 9},
  cells: polyfillCells, output: {cells: compacted, status: {count, overflow}}
}));
```

- **Determinism.** All operations are integer-only, with fixed sort orders. GPU results equal the
  CPU oracles (h3-js and BigInt Quadbin) bit for bit, including padding.

#### How it computes

{/* lens:GPUCellTopology */}

::lens{scenario=gpu-spatial-analysis.cell-topology rung=datum datum=0 caption="Disk cell 0: the input cell itself at grid distance 0"}

::lens{scenario=gpu-spatial-analysis.cell-topology rung=kernel caption="One thread per input cell walks the six neighbour steps breadth first"}

::lens{scenario=gpu-spatial-analysis.cell-topology rung=graph caption="A single topology node: cell keys in, disk cells and distances out"}

{/* lens:GPUCellCompaction */}

::lens{scenario=gpu-spatial-analysis.cell-compaction rung=datum datum=0 caption="Compacted cell 0: the lowest key after sorting and merging"}

::lens{scenario=gpu-spatial-analysis.cell-compaction rung=kernel caption="Sort, deduplicate, then ancestor-completeness checks merge sibling sets"}

::lens{scenario=gpu-spatial-analysis.cell-compaction rung=graph caption="Sort, scan and compaction passes between input cells and the bounded table"}

### `GPUCellGridPath` and `GPUCellMeasures`

**`GPUCellGridPath`** — Computes the H3 grid distance and the cells of the grid path between origin and destination cells.

::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=result caption="Grey: 7 origin places · blue: grid-path cell centres to the central place"}

**`GPUCellMeasures`** — Computes the spherical area, perimeter, edge lengths and pentagon flag of H3 cells.

::lens{scenario=gpu-spatial-analysis.cell-measures rung=result caption="Blue: 160 H3 resolution-9 cells by spherical area, 106,248–106,278 m², near-equal here"}

H3 gap kernels (`family: 'h3'`), integer-exact where possible.

`GPUCellGridPath` consumes cell keys, so it takes no `spatialContext`.

- **`GPUCellGridPath`** (h3 `grid_distance`, `grid_path_cells`, `are_neighbor_cells`): per pair,
  `origins` and `destinations` (`uint32x2`) give `output.distances` (the H3 `gridDistance`, or
  `GPU_CELL_GRID_DISTANCE_UNDEFINED` = `0xffffffff` where H3 fails: invalid input, resolution mismatch,
  non-adjacent base cells, behind a pentagon's deleted sector, masked), `output.cells` (`gridPathCells`,
  stride `maximumPathLength` up to 256, zero padded) and `output.counts` (the true length `distance + 1`;
  above `maximumPathLength` means overflow, 0 means no path). Omit `cells` for distance only;
  `are_neighbor_cells` is `distance == 1`. It is `cellToLocalIjk` with pentagon rotations and an
  integer cube-round. Known deviations: about 0.2% of random pairs differ from H3 by one adjacent cell at
  exact rounding ties (H3 rounds with double noise), and paths crossing pentagon base cells fail
  (`counts = 0`) in about 6% of those cases. Distances matched h3 4.5.0 on 355 of 355 pairs, failures
  included.
- **`GPUCellMeasures`** (h3 `cell_area`, `edge_length`, `is_pentagon`): `areas` (square meters),
  `perimeters`, `edgeLengths` (stride `maximumEdgeCount` 6 to 10, great-circle meters between
  consecutive `cellToBoundary` vertices), `edgeCounts` and `pentagons`. **Breaking:** outputs used to
  be kilometers; meters are now the default and kilometers need an explicit `units: 'kilometers'`
  (then areas are square kilometers). The sphere is the WGS84 authalic sphere
  (`GPU_EARTH_AUTHALIC_RADIUS_METERS`, replacing the removed `GPU_CELL_MEASURES_EARTH_RADIUS_KM`), and
  the instance's `measures` field is the machine-readable description (`lengthModel: 'great-circle'`,
  `areaModel: 'sphere-polygon'`, units). Cell identifiers carry no coordinates, so the class has no
  `spatialContext`. Single-face hexagons are exact from lattice
  offsets, within 1e-6 relative at resolutions 0 to 15. Pentagons and cells straddling a face edge use f32
  boundary vectors: 6e-7 at resolution 0, 2e-5 at 2, 2.5e-4 at 4 and about 1e-2 at 6, so treat them as
  unreliable beyond resolution 5. Open: directed edges and vertexes.

#### How it computes

{/* lens:GPUCellGridPath */}

::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=datum datum=0 caption="Pair 0: origin place, grid distance and the cells on its path"}

::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=formula caption="Grid distance is the spread of the cube-coordinate differences"}

::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=kernel caption="Local IJK conversion, distance, then cube-rounded interpolation of path cells"}

::lens{scenario=gpu-spatial-analysis.cell-grid-path rung=graph caption="A single grid-path node: origin and destination keys in, paths out"}

{/* lens:GPUCellMeasures */}

::lens{scenario=gpu-spatial-analysis.cell-measures rung=datum datum=0 caption="Cell 0: area and perimeter, with the hexagon flag"}

::lens{scenario=gpu-spatial-analysis.cell-measures rung=formula caption="Area sums spherical triangles from the cell centre to each edge"}

::lens{scenario=gpu-spatial-analysis.cell-measures rung=kernel caption="One thread per cell decodes the boundary and sums edge lengths and areas"}

::lens{scenario=gpu-spatial-analysis.cell-measures rung=graph caption="A single measures node: cell keys in, areas and perimeters out"}

### `GPUCellCover`

Polyfills polygons into H3 or Quadbin cells, emitting (feature id, cell) pairs.

::lens{scenario=gpu-spatial-analysis.cell-cover rung=result caption="Grey: 17 community areas · blue: 163 resolution-8 H3 cells whose centres fall inside"}

Polyfills polygons (with holes) into Quadbin or H3 cells, like CARTO `QUADBIN_POLYFILL` /
`H3_POLYFILL` and h3-js `polygonToCells`. The result is `(feature id, cell)` pairs that feed
`GPUCellAggregation` pre-keyed rows, `GPUKeyJoin` and `GPUCellCompaction`.

Maximum: `candidateCapacity` is at most `getGPUCellCoverMaximumCandidateCapacity(device)` (2^24 on a default device), and with the edge-slab index `positions.length + edgeSlabEntryCapacity` is at most `getGPUCellCoverMaximumEdgeSlabEntryCount(device)` (2^24).

- **Inputs.** Polygon features use the four-level layout: `positions`
  (lng/lat `float32x2`), `featureOffsets`, `polygonOffsets`, `ringOffsets`, and optional
  `featureIds`. Geometry is planar in lng/lat degrees, as in h3-js and turf. Containment is
  even-odd over all rings, decided exactly (exact orientation signs, a cell centre exactly on an edge
  resolved by the shared Simulation-of-Simplicity tie rule of `utils/exact-geometry-wgsl.ts`).
  `spatialContext` defaults to, and must be, longitude/latitude without a
  metric and with linear edges (planar contexts and great-circle edges throw).
- **Antimeridian and poles.** A feature with a non-pole edge whose raw longitude difference exceeds
  180 degrees gets no cells and increments `output.status.invalidCount`; prepare it with
  `GPULonLatPolygonPreparation` first. Pre-split multipolygons and same-pole edges are accepted,
  including split Fiji and Natural-Earth Antarctica. Empty rings contribute no bounds.
  Full-width features enumerate full-width lattices; with `candidateEnumeration: 'bounds'`,
  provision `candidateCapacity` accordingly and inspect `candidateCount` and `candidateOverflow`.
  A feature reaching past the cover's latitude limit
  (89 degrees for H3, 85.051129 for Quadbin) loses its polar cells and is counted in the optional
  `output.status.uncertainCount`. Both scalars are cleared on every encoding.
- **Containment.**
  - `'center'`: the cell centre is inside the polygon.
  - `'full'`: the cell lies inside the polygon.
  - `'intersects'`: the cell interior meets the polygon.
  - Quadbin supports all three modes. A Quadbin tile is an axis-aligned lng/lat rectangle, so
    `full` and `intersects` are exact edge-versus-open-rectangle tests. H3 supports `'center'`
    only.
- **Candidates.** Quadbin tests the integer-exact tile range of each feature's bounding box. H3
  lays a lng/lat lattice over the bounding box, with spacing below half the minimum H3 inradius at
  that resolution (measured from h3-js; the minimum sits next to a pentagon). The longitude spacing
  is taken at the feature's most equatorward latitude (0 when it spans the equator), so tall
  features are oversampled toward the pole rather than missing cells nearer the equator. Each lattice point
  indexes its cell with `cellIndexH3FromLngLat`. The cell is emitted only from the lattice point
  nearest its centre, so every cell whose centre lies in the polygon is emitted exactly once with
  no sort.
- **Sparse polygons (`candidateEnumeration`).** A thin diagonal polygon (a river, a road buffer, a
  coastline strip) or a sparse multipolygon has a bounding box that is mostly empty, so testing every
  box cell is wasteful and can overflow `candidateCapacity`. Both lattices are axis aligned in
  degree space (Quadbin tile rows are latitude bands, H3 sample rows are constant-latitude lines),
  so per lattice row the cells that can be accepted lie in x intervals that are found by a boundary
  trace plus a scanline fill, and only cells in those intervals are candidates:
  - The boundary trace emits, for every edge and every row its y range touches, the x span of the
    edge part inside the row band. The scanline fill takes the crossings of every edge with the
    row's test line (half-open in y, so a vertex on the line counts once), sorts them by row and x
    (stable radix sorts) and pairs them even-odd over all rings, which gives the interior intervals.
  - Spans and interior intervals are swept per row, grown by a proven f32 slack (a few ulps of
    the crossing, plus half a column for H3, whose accepted centre rounds to the lattice point) and
    converted to disjoint column runs. The exact test, the cell order and the output are unchanged,
    bit for bit, because the intervals contain every cell the test can accept.
  - `candidateEnumeration: 'bounds' | 'scanline' | 'auto'` (compile-time, default `'auto'`). With
    `'auto'` the stage is built when `candidateCapacity` is at least 65536, and each feature picks
    the scanline runs on the GPU when its box has more than four times as many cells as the
    scanline work (twice the rows plus the edge-row slots). A feature that does not fit
    `scanlineSlotCapacity` (default `min(candidateCapacity, 2^20, max(16 * vertices, 16384))`),
    has a non-finite vertex or would enumerate 2^32 candidates keeps its bounding box. The built
    stage runs a fixed set of passes (two sorts and several scans over three events per slot)
    whether or not a feature uses it, so size `scanlineSlotCapacity` to the polygons at hand or use
    `'bounds'` for compact polygons only.
  - `candidateCapacity` counts enumerated candidates, so with the stage it covers far larger
    bounding boxes of thin or sparse polygons. `output.status.candidateCount` (optional) reports the
    unclamped enumerated total.
- **Output.** `{featureIds, cells (uint32x2 little-endian),
  status: {count, requiredCount?, overflow, candidateOverflow, candidateCount?, invalidCount?, uncertainCount?}}`. `status.overflow` means only that accepted cells exceeded the final
  cell capacity; `status.candidateOverflow` means `candidateCapacity` truncated enumeration, in which case
  `requiredCount` is not an exact full-cover requirement. Order is by feature, then candidate order.
  The pipeline is count, scan, test, scan, write, with no atomic append, so it is deterministic.

```ts
graph.add(new GPUCellCover({
  family: 'h3', resolution: 8, containment: 'center',
  positions, featureOffsets, polygonOffsets, ringOffsets,
  candidateCapacity: 1 << 22,
  output: {featureIds, cells, status: {count, overflow, candidateOverflow}}
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

`output.core` is an optional `uint32` column parallel to `output.featureIds` (the same length): 1 for a core
cell (a Mosaic chip, provably inside the feature) and 0 for a border cell. It works for every containment
and both families. It is conservative: a cell is core only when its center is inside and no polygon edge
enters the cell (for H3, its boundary bounding box) grown by 2e-4 degrees plus 0.1% of the cell size.
Border cells can still lie entirely inside. A join can skip the exact point-in-polygon test for points whose
cell is core.

#### How it computes

{/* lens:GPUCellCover */}

::lens{scenario=gpu-spatial-analysis.cell-cover rung=datum datum=0 caption="Cell 0: its H3 key and the area it covers"}

::lens{scenario=gpu-spatial-analysis.cell-cover rung=kernel caption="Count, scan, test, scan and write passes enumerate and test candidate cells"}

::lens{scenario=gpu-spatial-analysis.cell-cover rung=graph caption="Candidate counting, exact containment and bounded emission as one graph"}

### `GPUCellSetOutline`

Finds the boundary edges of a sorted H3 or Quadbin cell set and chains them into closed rings.

::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=result caption="Grey: 163 covering H3 cells · blue: one closed ring chained from 140 exposed edges"}

The boundary edges of a set of H3 or Quadbin cells, the `cellsToMultiPolygon` outline. Set `rings` to chain
the edges into closed rings with shells and holes on the GPU ([`GPUSegmentRingAssembly`](#gpusegmentringassembly)).

It consumes cell keys, so it takes no `spatialContext`; the segments and rings it produces are
longitude/latitude degrees. `edgeModel` and `outputSpatialContext.edgeModel` are `'great-circle'`
for H3 and `'linear'` for Quadbin; `outputSpatialContext.metric` is `'none'`.

```ts
graph.add(new GPUCellSetOutline({
  family: 'h3',
  cells, // uint32x2 little-endian keys, sorted ascending and distinct, e.g. a GPUCellAggregation table
  count: tableCount, // optional one-row valid row count
  groups, // optional uint32 label per row: also emit edges between different groups
  output: {rows, cells: segmentCells, edgeIndices, endpoints, groups: segmentGroups, status: {count, overflow, requiredCount}},
  rings: {normalizeWinding: true, output: ringOutput} // optional, see below
}));
```

- The output columns share one capacity: `rows` (the input row, also the compact ID column), `cells`,
  `edgeIndices`, `endpoints` (`float32x4`: lng0, lat0, lng1, lat1), optional `groups` and `status` with `count` (clamped),
  `overflow` and optional `requiredCount`.
- An edge is emitted when the neighbor across it is absent, or has a different group label. With `groups`,
  a border between two groups is emitted from both sides, tagged with each side's group.
- Order is by `(input row, edge index)`: classify, inclusive scan, write, with no atomics, so it is
  deterministic.
- H3: edge `e` is the `e`-th edge of `cellToBoundary` (up to 10 slots with distortion vertices, so such an
  edge appears as two segments). The neighbor across it is the H3 neighbor whose center is nearest the
  edge midpoint, a geometric rule validated against h3-js on hexagons, a pentagon disk and distortion
  cells. Quadbin: 0 north, 1 east, 2 south, 3 west; columns wrap, and rows beyond the poles are absent, so
  the map edge is an outline edge.
- Membership is a binary search, so the input must be sorted and distinct. Invalid keys never emit.
- Endpoints are f32. H3 cells that straddle the antimeridian keep their `cellToBoundary` longitudes, so a
  segment can span more than 180 degrees. Other families (quadkey, geohash, S2, A5) are not supported.
- Segments are directed with the cell interior on a fixed side: `outline.interiorSide` is `'left'` for H3
  (counter-clockwise shells) and `'right'` for Quadbin (clockwise shells).
- `rings: {vertexTolerance?, normalizeWinding?, output: GPUSegmentRingAssemblyOutput}` composes
  `GPUSegmentRingAssembly` after the outline, with `output.endpoints`, `output.status.count` and `output.groups`
  as its inputs, `outputSpatialContext` as its context and `interiorSide` taken from the family. Use `normalizeWinding: true` for
  counter-clockwise shells and clockwise holes in both families. Without `rings`, hole detection and ring
  orientation are left to the caller.

For an east-west edge of ground length `L` at latitude `phi`, the great-circle arc deviates from
the degree-straight edge by approximately `tan(phi) * L² / (8R)`. For average H3 edge lengths at 45°N:

| H3 resolution | Average edge length | Approximate deviation |
| --- | --- | --- |
| 0 | 1281 km | 32 km |
| 1 | 483 km | 4.6 km |
| 2 | 183 km | 660 m |
| 3 | 69 km | 93 m |
| 4 | 26 km | 13 m |
| 5 | 9.9 km | 1.9 m |

Run outlines of H3 resolution &lt;= 4 through `GPULonLatPolygonPreparation` with
`spatialContext.edgeModel: 'great-circle'` before linear consumers. At resolution >= 5, the
average-edge approximation at 45°N is within 2 m; this is not a global bound (multiply the table's
deviations by about 1.7 at 60°). Preparing preserves the declared great-circle meaning while
producing a linear output context. Global or antimeridian-crossing rings also need preparation.

#### How it computes

{/* lens:GPUCellSetOutline */}

::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=datum datum=0 caption="Exposed edge 0: its owning table row and edge index"}

::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=kernel caption="Neighbour lookup, scan and write passes keep the edges with no neighbour"}

::lens{scenario=gpu-spatial-analysis.cell-set-outline rung=graph caption="Outline classification, edge packing and ring assembly as one graph"}

### `GPUSegmentRingAssembly`

Chains directed boundary segments into closed rings with shell and hole classification.

::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=result caption="Grey: 17 community areas · blue: one outer shell from 3,184 edges once shared borders cancel"}

Chains directed boundary segments into closed rings on the GPU, the ring-assembly step of
`cellsToMultiPolygon`. It consumes `GPUCellSetOutline` output, or any `float32x4` segment list whose
vertices meet within a tolerance.

`spatialContext` replaces the former `geographic` flag. It defaults to native planar coordinates (**behavior change:** `geographic` defaulted to true). A `longitude-latitude` context with metric `'none'` scales x differences by `cos(latitude)` (clamped at 0.01) when comparing turns at shared vertices; great-circle and other metrics are rejected because ring assembly measures nothing, and `vertexTolerance` stays in position units. Hole detection uses the exact orientation of the turn at each ring's lowest vertex (smallest y, then x; the f32 area sign only when that turn is zero), so a sliver ring whose f32 shoelace sum cancels is still classified correctly.

The resolved `spatialContext` describes both input and output. For longitude/latitude it preserves
the input `edgeModel`, including omission; both `'linear'` and `'great-circle'` edges are accepted
with metric `'none'`. Pass `outline.outputSpatialContext` when assembling cell outline segments.
Prepare global or antimeridian-crossing rings with `GPULonLatPolygonPreparation` before polygon
consumers, and prepare great-circle rings before consumers that require linear edges.

```ts
graph.add(new GPUSegmentRingAssembly({
  endpoints, // float32x4 (x0, y0, x1, y1), e.g. GPUCellSetOutline output.endpoints
  count, // optional one-row valid segment count
  groups, // optional uint32 per segment: rings never mix groups
  vertexTolerance: 5e-5, // compile-time Chebyshev matching distance, input units
  interiorSide: 'left', // side of each segment that is filled ('left': counter-clockwise shells)
  normalizeWinding: true, // shells counter-clockwise, holes clockwise (RFC 7946)
  splitTouchingRings: true, // default
  spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}, // cos(lat) in turn comparisons; omit for planar
  output: {
    ringOffsets, // uint32, ringCapacity + 1
    positions, // float32x2, vertex capacity
    ringAreas, ringIsHole, ringShells, ringGroups, // optional, ringCapacity rows
    segmentRings, segmentVertices, segmentFlags, // optional, one row per input segment
    polygons: {kind: 'polygons', positions: polygonPositions, ringOffsets: polygonRingOffsets,
      polygonOffsets, featureOffsets}, // optional four-level polygon port
    status: {count, overflow, requiredCount, openSegmentCount, touchingSegmentCount}
  }
}));
```

- **Layout.** GeoArrow style: ring `r` is `positions[ringOffsets[r] .. ringOffsets[r + 1])`, closed (the last
  vertex repeats the first). Rings are ordered by their lowest input segment index; a ring starts at that
  segment and follows its direction (or is reversed by `normalizeWinding`). Offsets past `status.count` repeat the
  final offset.
- **Matching.** Each segment end is matched to segments that start within `vertexTolerance` in the same
  group, through a sort of quantized vertex hashes and a 3 x 3 neighborhood lookup. Quadbin vertices are
  bit-exact; H3 vertices of neighboring cells agree within f32 round-off (below 5e-5 degrees). Pick a
  tolerance of at least four times the f32 spacing of the coordinates and well below the shortest segment.
  Without `vertexTolerance` the tolerance is four f32 ulps of the frame extent (`extent * 2^-21`, measured
  on the GPU), so only separately rounded copies of one vertex match (**behavior change:** the default was
  an absolute 5e-5). `GPUCellSetOutline` keeps 5e-5 degrees as its own preset
  (`GPU_CELL_SET_OUTLINE_VERTEX_TOLERANCE`), valid up to H3 resolution 12 and Quadbin zoom 22; pass about
  a quarter of the shortest edge for finer cells. Vertices are hashed relative to the first segment's start vertex into unsigned cells of size
  `max(vertexTolerance, extent * 2^-22)` (with a small slack), where `extent` is the largest coordinate
  offset from that origin, measured on the GPU. Projected coordinates (UTM, Web Mercator) therefore never
  collide at any tolerance, and candidates are still compared by their true distance against
  `vertexTolerance`. `status.overflow` is also set when a valid segment has a non-finite coordinate.
- **Shells and holes.** The signed area (shoelace, relative to the first vertex) is compared with
  `interiorSide`: with `'left'` shells are counter-clockwise. `ringIsHole` flags holes. `ringShells`
  assigns every hole to its innermost enclosing shell (smallest area, lowest index on ties, same group), a
  shell to itself, and a hole without a shell to `GPU_SEGMENT_RING_ASSEMBLY_NONE`. This costs one pass over
  every written vertex per hole. Containment is exact: the probe is the symbolic point infinitesimally
  beside the midpoint of the hole's first edge, on the side away from the hole, classified with the exact
  point-in-ring rule of `utils/exact-geometry-wgsl.ts`, so a hole whose edge runs along another ring is
  still assigned correctly.
- **Touching vertices.** When more than one segment starts at a segment's end vertex (Quadbin cells that
  meet at a corner; H3 hexagons never do), the continuation is the outgoing segment that turns tightest
  toward the interior, so regions that touch at a point stay separate rings. With `splitTouchingRings` a
  second tracing pass swaps the continuations of segment pairs that end at one vertex and lie on one ring,
  so two holes meeting at a corner, or a hole pinched to its shell, become simple rings. Choices taken
  twice resolve to the lowest segment index. `status.touchingSegmentCount` and `segmentFlags`
  (`GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING` 1, `_DANGLING` 2, `_CONFLICT` 4) expose the ambiguity.
- **Open chains** (dangling ends, tolerance misses, truncated input) emit no ring and are counted in
  `status.openSegmentCount`.
- **Cancelling pairs.** `cancelOpposingSegments` removes exact `a -> b` and `b -> a` pairs within a group
  before tracing (they are flagged `GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED`, 8, and not counted as open).
  `GPUIsobandRings` uses it for the edges shared by two cells. `output.polygons.sourceIds` gives the
  group of each polygon's shell.
- **H3 pentagons.** Rings of three mid-latitude pentagons (resolutions 2, 3 and 5, alone, in disks, as holes
  and open rings) equal h3-js `cellsToMultiPolygon`. The two polar pentagons enclose a pole, so their rings
  have no planar lng/lat orientation.
- **Bounded output.** Rings are written whole, as a prefix: `status.count` is the number written, `status.overflow` is 1
  when ring or vertex capacity dropped rings, and `status.requiredCount` is the full ring count.
- **Polygon layout.** `output.polygons` regroups the rings as GeoArrow polygons (a shell, then its holes;
  the closing vertex dropped; holes without a shell dropped) with GPU-written, flat-padded offsets over
  fixed-length views. Every polygon is one feature, `featureOffsets` is an identity prefix padded with
  `polygonCount`, and optional `sourceIds` carries its shell group. It plugs into
  the `polygons` port of `GPUPointInPolygonJoin` (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`): the join's
  compile-time topology is the ring capacity and unused polygons and rings are empty.
- Deterministic (integer atomics only). The cost is two pointer-jumping phases of `ceil(log2(rows))`
  dispatches each (plus the second tracing pass), a sort and scans. Rings that cross the antimeridian keep
  the input longitudes; use `GPULonLatPolygonPreparation` for global or antimeridian-crossing rings.
  Turn comparison uses a monotone pseudo angle, not `atan2`. Ring areas are f32.

#### How it computes

{/* lens:GPUSegmentRingAssembly */}

::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=datum datum=0 caption="Ring 0: its signed area and whether it is a hole"}

::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=kernel caption="Vertex matching, pointer jumping and scans link segments into rings"}

::lens{scenario=gpu-spatial-analysis.segment-ring-assembly rung=graph caption="Cancel, match, trace, split touching rings and write as one graph"}

### `GPULineNoding`, `GPUPolygonize` and `GPUMakeValid`

**`GPULineNoding`** — Splits planar linework at crossings, touches and overlaps into atomic segments with source provenance.

::lens{scenario=gpu-spatial-analysis.line-noding rung=result caption="Grey: 223 Wicker Park streets · blue: 294 segments split at every crossing"}

**`GPUPolygonize`** — Builds every bounded polygon face from planar linework, as shapely `polygonize` does.

::lens{scenario=gpu-spatial-analysis.polygonize rung=result caption="Grey: 223 Wicker Park streets · blue: 73 bounded faces, the city blocks"}

**`GPUMakeValid`** — Repairs each polygon feature into valid faces, keeping those covered by an odd number of source rings.

::lens{scenario=gpu-spatial-analysis.make-valid rung=result caption="Blue: 51 kept faces coloured by source park; all 51 parks were already valid"}

These contributors form the bounded planar geometry-topology substrate:

- `GPULineNoding` splits arbitrary finite linework at crossings, touches, both ends of collinear
  overlaps, and repeated vertices. Each atomic `GPUNodedSegmentPort` row carries source feature,
  line/ring, original edge, and normalized start/end parameter provenance. Segment pairs the exact
  predicates cannot certify (no split is made for them) and non-finite input vertices set `overflow`
  whether or not `output.status.uncertainCount` is wired; `overflow` with `candidateOverflow` clear and
  `requiredCount` within capacity means such input, which no capacity fixes. `GPULineSplit` reports
  the same in `pieces.status.overflow`.
- `GPUPolygonize` expands each noded segment into two directed edges, assigns deterministic left-face
  ownership, omits the unbounded and zero-area faces, and emits canonical polygon features. It
  emits every bounded face (shapely `polygonize` semantics): the hole of a holed ring comes back as
  its own polygon. Its diagnostics classify source edges as closed-ring, open-chain, cut, dangle,
  and unused.
- `GPUMakeValid` applies the same pipeline independently to every input polygon feature, then keeps a
  face only when the feature's source rings cover it an odd number of times. The fill rule is source
  parity (even-odd), the GEOS `make_valid` linework method that shapely uses: a valid polygon with a
  hole returns the annulus only (no second polygon filling the hole), an island inside a hole is kept
  as its own polygon, and the doubly covered lens of mutually crossing rings is dropped. It repairs
  self-crossing and mutually crossing rings into bounded polygon faces; dimensional collapse remains
  available through diagnostics rather than being converted to mixed geometry. Parity is probed
  exactly at the symbolic point infinitesimally left of the midpoint of the first edge of each face
  shell (the shared probe of `utils/exact-geometry-wgsl.ts`; strictly inside the face, no rounded
  sample point), brute force over the noded segments, one thread per face. A face whose
  orientation cannot be certified is kept. `status.count` is the number of kept polygons.

All three require an explicit `GPUTopologyPrecisionPolicy`. The current policy uses the shared exact
orientation predicates with an uncertainty counter, float32 coordinate storage, and a caller-chosen
Chebyshev vertex tolerance. For `GPULineNoding`, `requiredCount` is exact when `candidateOverflow` is
zero. For `GPUPolygonize` and `GPUMakeValid`, `status.requiredCount` is the unclamped ring-assembly
requirement (every face ring, including holes), the ring capacity to allocate, not the number of
polygons written (`count`). `candidateOverflow` is the noding flag only: when the intersection
substrate is incomplete, both `candidateOverflow` and `overflow` are set and callers must grow
`intersectionCapacity` and rerun; ring or segment capacity loss sets `overflow` alone. No CPU readback
or compatibility adapter is involved.

Maximum: `positions` has at most `getGPUSegmentIntersectionMaximumVertexCount(device)` (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows, and the line (or ring) count plus four times `intersectionCapacity` is at most `getGPULineSplitMaximumPieceBound(device)` (4194304 on a default device, the 32-byte piece-table row). Each contributor validates both under its own ID.

Planar only: `spatialContext` defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (`createGPULocalFrame`, `GPULocalFrameProjection`). The resolved context is passed on to the noding and ring-assembly stages.

#### How it computes

{/* lens:GPULineNoding */}

::lens{scenario=gpu-spatial-analysis.line-noding rung=datum datum=0 caption="Noded segment 0: the road and edge it came from"}

::lens{scenario=gpu-spatial-analysis.line-noding rung=kernel caption="Pair intersections, piece scans and provenance reconstruction emit atomic segments"}

::lens{scenario=gpu-spatial-analysis.line-noding rung=graph caption="Line splitting, segment flags, scan and writing as one graph"}

{/* lens:GPUPolygonize */}

::lens{scenario=gpu-spatial-analysis.polygonize rung=datum datum=0 caption="Noded segment 0: its closed-ring, cut or dangle class bits"}

::lens{scenario=gpu-spatial-analysis.polygonize rung=kernel caption="Twin edges, left-face traversal and ring writing build the faces"}

::lens{scenario=gpu-spatial-analysis.polygonize rung=graph caption="Noding, directed edges, ring assembly and diagnostics as one graph"}

{/* lens:GPUMakeValid */}

::lens{scenario=gpu-spatial-analysis.make-valid rung=datum datum=0 caption="Face 0: the source park it was kept for"}

::lens{scenario=gpu-spatial-analysis.make-valid rung=kernel caption="One thread per face probes parity against every noded segment"}

::lens{scenario=gpu-spatial-analysis.make-valid rung=graph caption="Noding, polygonize, parity filter and repacking as one graph"}

### `GPUCellTableCompare`

Outer-joins two sorted cell tables into one union table of before, after, delta, ratio and z-score columns.

::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=result caption="35 zoom-17 union cells coloured by shops minus restaurants · grey: the places counted"}

`GPUCellTableCompare` outer-joins two sorted cell tables (the outputs of `GPUCellAggregation` or `GPUCellRollup`) into one union table for period-over-period or A-versus-B compare maps. Both tables must use the same grid family and resolution; the contributor cannot check that.

It compares tables keyed by cell, so it takes no `spatialContext`.

```ts
graph.add(new GPUCellTableCompare({
  before: lastMonth, after: thisMonth,
  measure: 'sum', sumScale: 65536,
  output: {cells, presence, before, after, delta, ratio, percentChange, zScore, status: {count, overflow}}
}));
```

- `output.cells` holds the union keys ascending (`uint32x2`, little-endian `(low, high)`); its length is the capacity. When the union is larger, the smallest keys are kept, `status.count` is clamped and `status.overflow` is 1. `status.overflow` also becomes 1 when either input table overflowed. Optional `status.requiredCount` receives the unclamped union size.
- `presence`: bit 1 (`GPU_CELL_COMPARE_PRESENT_BEFORE`) when the cell is in `before`, bit 2 (`GPU_CELL_COMPARE_PRESENT_AFTER`) when in `after`.
- `before` / `after`: the measure (`'count'`, or `'sum'` decoded from the fixed-point `sums`), 0 when absent.
- `delta = after - before`, computed exactly as a 64-bit integer difference (counts and fixed-point sums) and only then rounded to f32, so large sums with small differences are not lost to f32 cancellation.
- `ratio = after / before` and `percentChange = 100 * (after - before) / before`, NaN when `before` is 0.
- `zScore`: `'poisson'` (default for counts) is `delta / sqrt(before + after)`, 0 when both are 0. `'standardized'` (default for sums) is `(delta - mean) / sd` over the union rows with the population standard deviation, 0 when sd is 0. The mean and sd are reduced in three fixed-order stages (per-workgroup partial sums, per-workgroup partial squares around the combined mean, then one combining workgroup), so a re-encode is bitwise identical.
- Rows past `status.count` hold the empty key (all ones), presence 0, `before`/`after` 0, and NaN in `delta`, `ratio`, `percentChange` and `zScore`.

The merge is a binary-search merge path with an exclusive scan of matched rows, so every union row is written by exactly one invocation and no atomics are used. Cost is O((n + m) log(n + m)). Table contents are per-frame (re-encode without recompiling); capacities, columns, `measure` and `zScore` are topology.

#### How it computes

{/* lens:GPUCellTableCompare */}

::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=datum datum=0 caption="Union cell 0: restaurants, shops, difference and z-score"}

::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=formula caption="Delta from exact integer difference, ratio, percent change and the Poisson z-score"}

::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=kernel caption="A merge path pairs the two tables and writes each union row once"}

::lens{scenario=gpu-spatial-analysis.cell-table-compare rung=graph caption="Merge, scan, statistics reduction and column writes as one graph"}
