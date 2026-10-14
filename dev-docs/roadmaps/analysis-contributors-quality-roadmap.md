# Analysis contributors: quality roadmap

Status: implemented 2026-10-14 (see [Implementation status](#implementation-status-2026-10-14)); proposed
2026-10-13. Based on a read-only, five-stream review of every analysis contributor
(`gpu-spatial-analysis`, `gpu-network`, `gpu-terrain`, the `gpu-raster` and `gpu-dataframe` subdirs, `geospatial`
and `src/utils`) for consistency, correctness, performance and ergonomics. The evidence (file and line, a reproducer
or oracle replay, and a proposed fix for each finding) is in
[the stream reports](../reviews/analysis-contributors-2026-10-13/README.md). Finding IDs below (A-1, B-01, C-01,
D-01, E-01) refer to those reports.

Related plans. This document only adds what they do not already cover, and corrects them where the code says
otherwise:
- [GPU algorithm roadmap](./gpu-analysis-algorithms-roadmap.md): algorithm choice, P0 to P3.
- [GPU spatial analysis roadmap](./gpu-spatial-analysis-roadmap.md): ontology, contracts and tranche status.
- [gpugraph roadmap](./gpugraph-roadmap.md), "Analysis contributors" → "Still open".
- [Perf round 2026-10-06](../perf/perf-round-2026-10-06.md).

The statistics methodology (inference, weights, islands, missing values, spatial context) was settled by the
2026-10-13 correctness round, so this review treats it as given.

## Implementation status (2026-10-14)

Every tranche was implemented on `map-contributors` (uncommitted) on top of the 2026-10-13 correctness round. Each item
was checked against the code before it was fixed. Each fix has a node spec and, where GPU results change, a headless
GPU spec run serially. Oracles come from reference packages.

| Tranche | Status | Notes |
|---|---|---|
| Q0 silent wrong results | Done | D-01 to D-15, B-01 to B-11, C-01 to C-09, A-1/A-2, E-10/E-11 and the algorithms P0s. C-09 now writes unresolved sentinels (NaN costs, UNRESOLVED codes). |
| Q1 compile cost | Done | Rule 12 holds library-wide. Each fix has a node spec asserting the pipeline count is unchanged at N and N+1. Reachability: 64 → 1 relax pipeline. Default terrain flow: ~60 → 15. Class breaks: 62/65 → 45. GWR pass cursor. E-05: no synchronous `.compile()` in either app, which swap graphs through `GraphSwap`. |
| Q2 wasted GPU work | Done, except one item | Rule 13 gates, two-stage reductions and per-workgroup atomics. Scan `nearest` uses radix select. Ripley F is 24% faster with bitwise-equal output. The Delaunay rebuild is 1.9x faster at 3,000 sites. Texture bindings in `createWGSLKernelNode` retire the escape hatches. **Open:** coverage-repair rounds are not gated, because upstream `GPUScan` and `GPUSegmentIntersection` take no `condition`. |
| Q3 one library | Done | `src/analysis-core/` sits below every entry, and the import DAG is enforced by a spec. The barrels are `export *` with a diff spec. The canonical vocabulary is enforced by a Props-parsing spec; its allowlist holds only upstream-owned files. All status words live in `output.status`. `GPUNetworkTopology`, `grid: GPURasterMetadata` and the four-level polygon port are used everywhere. Feature-row contiguity and `uint32 \| float32` counts let points → counts → W → Moran chain with no cast. There is one 2SFCA/3SFCA kernel, plus the new `GPUNetworkCostWeights`. The capacity plans drive the W and join grow loops. Construction uses `getCommandNodes` only. Errors go through the shared validators and `assert`. B-22 has one Philox purpose registry. |
| Q4 test and doc integrity | Done | `requireTestDevice` covers 1,448 guards, and the canary fails headless. The kernel-audit node spec builds all 295 public contributors on a fake `'core'` device and checks limits, reserved words, NaN self-compares, const non-finite bitcasts and undeclared calls. All 142 doc `ts` blocks type-check. Parameters subsections are generated for 131 classes. Committed generators under `test/oracles/` cover networkx, GDAL 3.12 via rasterio, pysheds, shapely, scipy, h3-py, esda, spreg, libpysal, mgwr, mapclassify, scikit-gstat, statsmodels and PySAL `access`. |
| Smaller items | Done | C-14 is now done. C-21 is rejected, because corrupt links already return an empty path. Of the 34 unverified suspicions across the reports, 7 were real and are fixed, and 2 are upstream. |

Bugs found on the way that the review had not listed:
- WGSL reserved words used as identifiers (`target`, `active`) in four kernels that never compiled on Dawn.
- 3SFCA missing its decay factor (it now matches PySAL `access` to 3e-16).
- Moran/Geary on constant values.
- North-up lon/lat areal weights of 0.
- Multipart ZIP attributes in the explorer.
- Line-noding pair overflow never reported.
- k-means convergence reported on an f32 overflow.

Still open:
- AZP articulation test (algorithms P3 #3). The interim row cap is 32,768.
- D-06 crossing points, superseded by snap rounding (algorithms roadmap item 2).
- A-18 for `GPUTerrainSpikeRepair`: gpgpu has no graph-owned persistent buffer API.
- Upstream items:
  - `@luma.gl/gpgpu` kernels still emit folded `v == v` NaN tests (about 6,300 generated kernels).
  - `GPUSort` bakes a constant per pass.
  - Dawn does not enforce the 8 KB private-memory limit.
  - `GPUGridIndex` keeps flat `count`/`overflow`.

## Verdict

The numerical core is good and was checked against references:
- Exact orientation and incircle, Philox, ordered-float keys and dispatch folding.
- Terrain derivatives and curvature (Horn, Zevenbergen-Thorne, Evans-Young, Florinsky), geomorphons and GRASS tie
  rules, D8/D-inf/MFD, Planchon-Darboux, Barnes flats, the Felzenszwalb EDT, the isoline saddle rule, H3/S2/geohash
  forward indexing, Hilbert keys, and the reachability engine.
- Resource lifecycle is clean: four owned buffers, all destroyed, aliased transients, no readback for control flow.

The defects are at three seams, and most are silent:

1. **Topology decided by f32 next to an exact predicate.**
   - Ten hand-written point-in-polygon loops use inexact crossings, and three of them decide topology.
   - `edgeBreaks` branches on f32.
   - Buffers and make-valid rebuild areas with the wrong fill rule.
   - Delaunay uses a finite supertriangle and drops hull triangles.
2. **Bounded work that does not say it stopped early.**
   - Capped loops write partial results into the same outputs as converged runs.
   - Budgets drop work with no counter (map matching).
   - Wrappers and recipes swallow inner status (isochrones, drive-time recipe).
   - In two places the truncated value is a legitimate code: cost-distance back-link 0 means "source", and NaN
     accumulation becomes "not a stream".
3. **Pipeline compilation.**
   - Shared helpers bake round, gate-slot, batch-offset and seed values into WGSL.
   - Overrides do not share pipelines either.
   - A single reachability compiles 64 relax pipelines where 2 would do.
   - Both apps and every doc page compile synchronously.

The API is one construction shape (`getCommandNodes`, caller-owned outputs, construction-time validation, complete
TSDoc) with a different dialect per entry:
- `parameters` vs `settings`;
- `output` bag vs loose output props;
- three spellings each for CSR, polygons and cost breaks;
- `max*` vs `maximum*`.

The contracts layer exists but is mostly unconsumed. `GPUSpatialContext` is the exception. The 11 capacity plans and
the point, feature and trajectory ports have no callers. Recipes do not chain: zonal statistics has one row per
feature while contiguity W has one row per polygon part, and the uint32 counts would need a cast to float32 events.

## Library rules

These rules close the patterns the review found. Each one is cheap to check with a grep or a node spec, and each new
or touched contributor should follow them. They extend the
[algorithm roadmap's library-wide decisions](./gpu-analysis-algorithms-roadmap.md#library-wide-engineering-decisions).

### Geometry and numerics

1. **Topology is decided by exact signs only.** Every branch that changes a topological classification uses
   `orientSign` or exact incircle; f32 values only place points. Point-in-polygon and segment-crossing parity go
   through one shared exact helper (`pointInRingParity`, `segmentCrossesSoS` with a Simulation-of-Simplicity tie
   rule). f32 crossings are allowed only in sampling code (labels, dots), with a comment. (D-04, D-05, D-09, and
   pattern 1 in report D)
2. **Every boundary→area rebuild states its fill rule**: nonzero for offsets and buffers, source parity for repair.
   It also has a fixture where the rule matters. (D-02, D-03)
3. **GPU geometry is origin-relative.** The extent origin lives on the CPU in f64, and kernels see cell-space or
   origin-relative coordinates. No `i32(floor(v * k))` hash without subtracting an origin. (C-04, C-08, B-10)
4. **Default tolerances are derived from cell size or ulp, never absolute literals.** A semantic default that differs
   from sibling contributors must be a named preset. (C-04, C-05, D-18)
5. **Bearings become pixel steps through the ground cell**: `normalize(sin az / cx, ±cos az / cy)`. (C-01)
6. **Finite sentinels (`1e30`) only enter min/max**, never sums. Reachability is a separate count or flag. (D-08)
7. **Fixed point and 64-bit words**:
   - never derive a rounding unit from the data maximum;
   - 64-bit accumulator readers saturate or return both words;
   - Int64 ordering compares words with the sign bit flipped, never the sign of a difference. (B-01, B-06, B-07)
8. **One missing-value predicate.** Non-finite means missing in any kernel that interpolates or averages. Tests use
   `isNanBits` and `isFiniteBits` from `utils/wgsl-common.ts`. No `x == x`, no local `isFiniteValue` copies, and no
   NaN-only bit tests where infinities are interpolated. (B pattern 1, about 36 sites; C-06, C pattern 4)

### Status and completeness

9. **Every budget, capacity or capped loop reports itself.** Either its `converged`/`overflow` output is required
   when it can trigger, or the data carries an "unresolved" sentinel that no converged run can produce. A truncated
   value must never equal a legitimate code. (B-02, B-05, C-02, C-03, C-09, D-10, D-11)
10. **Wrappers and recipes re-export every inner status as a stage.** (B-05, B-08)
11. **A node that owns an output writes all of its rows.** Per-edge loops are bounded by `offsets[NODE_COUNT]`,
    never by `neighbors.length`, because transients alias and unwritten slots hold stale data, not zero. (B-03,
    B-04, B-09)

### Performance

12. **A loop index never reaches WGSL text or an override constant.**
    - Rounds, gate slots, seeds, layers, observer and batch indices come from a GPU state or params word.
    - Batch sub-views use 64-element-aligned strides, so `createWGSLKernelNode`'s offset residue stays constant.
    - Input sizes may stay baked or become overrides.
    - Check: a node spec builds a graph at N and N+1 rounds and asserts the count of distinct
      `(source, constants)` is unchanged. (E-01 to E-07)
13. **Every unrolled loop with a data-dependent exit is gated** with `condition: {source: 'gpu', mode: 'indirect'}`
    on its state word. (E-08)
14. **Reductions over n are two-stage.** No single-workgroup reductions over all rows, and no per-row atomics onto a
    handful of words. (E-12, E-14)
15. **Apps and docs call `compileAsync()`.** Scenes build, compile asynchronously, swap, and only then release the
    old graph. (E-05, E-19)

### Device limits and validation

16. **Default device limits are checked at construction, by a node spec on a fake `'core'` device**:
    - 8 storage buffers per stage;
    - 8,192 B private memory;
    - 16,384 B workgroup memory;
    - `maxStorageBufferBindingSize` for every binding at the maximum allowed props;
    - `capacity × lanes ≤ 2³²−1`.

    The headless device runs at `featureLevel: 'max'` and hides all of these. (A-1, A-2, C-11, E-10, E-11, B-18)
17. **Invariants use `assert(condition)` with the reason in a comment.** User-facing prop errors go through shared
    validators (`validatePackedView`, `validateCapacity`, `validateExtent`, `validatePowerOfTwo`) with short
    messages. (A-16, C-07)

### API

18. **One vocabulary** (see [Canonical vocabulary](#canonical-vocabulary)), enforced by a node spec that parses
    `export type GPU*Props`. (A-8 to A-12)
19. **One source of truth per setting.** CPU-side metadata (for example `GPUInferenceMetadata`) is derived from the
    same settings object that is packed for the GPU. (A-6)
20. **A contract is public only with an in-tree consumer.** Barrels are `export * from './<dir>/index'`, with a
    barrel-diff spec. Shared contracts live below every entry (`src/utils` or an `analysis-core` folder), not inside
    `gpu-spatial-analysis`. (A-4, A-13, A-15)
21. **One construction idiom**: `graph.add(contributor)` with `getCommandNodes`, or `add<Name>Recipe(graph, props)`.
    No `addToGraph`. Device resources come from the graph. (A-17, A-18)

### Tests and docs

22. **Oracles come from reference packages through committed generators** (`test/oracles/*.py`: shapely,
    networkx, scipy.ndimage, gdaldem/rasterio, h3-py, esda). Each fixture records the package version and the
    tolerance.
    - Every topology operator gets one randomized or property fixture (counts and invariants), not only hand cases.
    - Each statistic has one edge-policy table (NaN, ±Inf, -0, out-of-range parameters, empty input) that every
      implementation and oracle imports.
    (A-23, B-14, B-21, D pattern 6)
23. **GPU specs skip visibly and fail under headless when no device exists.** Use a shared `requireTestDevice`
    helper in place of about 1,600 `if (!device) return` guards in 367 spec files. (A-3)
24. **Doc `ts` blocks type-check against the barrels in a node spec.** (A-7)

## Tranches

Effort: S is hours, M is a day or two, L is several days. This follows the user's working rules: change code first,
then run headless specs serially at the end with one GPU spec at a time, with A/B checks only where a change claims a
speedup. Q0 and Q1 can run in parallel because their files are disjoint.

### Q0: silent wrong results and device-limit failures

| Item | Where | Fix | Effort |
|---|---|---|---|
| D-01 Delaunay loses hull triangles (finite supertriangle; 2 missing on 600 uniform points, 51 of 185 on a thin band) | delaunay-tessellation | Symbolic super vertices: ghost triangles whose incircle reduces to exact `orientSign`. Spec `T = 2n−2−h` plus empty-circle checks on uniform and band inputs. The planned gDel2D rewrite (algorithms P3 #1) must keep the same rule. | M |
| D-02 buffer self-overlap becomes holes (even-odd) | polygon-buffer → overlay `insidePolygons` | Nonzero-winding membership mode, or the buffer as a union of simple pieces | M |
| D-03 `GPUMakeValid` emits hole interiors as polygons | geometry-topology | Keep faces by exact source parity; add a holed-polygon spec | M |
| D-04 `edgeBreaks` branches on f32 `den` | spatial-join/spatial-predicate-wgsl.ts | Collinear iff both exact orients are 0; `den` only places `t` | S |
| D-07 H3 cover lattice too coarse on tall high-latitude polygons (26% of cells missing at 0–85°) | cell-cover | Spacing from the most equatorward latitude, or per-row column count; add a 0–80° spec | S |
| D-08 p-median `1e30` sentinel summed | location-allocation | Exclude unreachable rows from the objective; add an ∞ row to the spec | S |
| D-09 track touch/turn-back counted as a crossing | trajectory-zones | Exact crossing with the SoS tie rule shared with the ray test | M |
| B-01 variogram/correlogram fixed-point unit pegged to max−min (−83% at a 2e4 sd outlier) | pair-statistics | Unit from a typical magnitude, rare large terms into an overflow word | M |
| B-02 map-matching route budget silently drops nodes (186 of 227 routes missed at 1 km spacing) | map-matching | `routeOverflow` count, worst-key eviction, documented supported spacing | M |
| B-03, B-04, B-09 unwritten CSR tail slots (phantom snap edges, out-of-range K-function events, stale sort keys) | network snapping, k-function, subgraph filter, coarsening, line graph, map-matching edge table, drive-time recipe copy | Rule 11. Delete the recipe's kernel copy. | S each |
| B-05 isochrones expose no `converged` | network-isochrones | Plumb `converged` and `localIterations` | S |
| B-06 variogram `pairCounts` wraps at 2³² | pair-histogram | Saturate | S |
| B-07 Int64 window compare wraps (unbounded windows reject pre-1970 rows) | time-window-filter | Word-wise compare | S |
| B-08 drive-time recipe drops both snapping `overflow`s | recipes | Two stages | S |
| C-01 azimuth applied in pixel space on anisotropic grids (45° → 26.6° at lat 60) | terrain-illumination horizon, cast shadow, sweep | Rule 5; add a lat-60 geographic spec | M |
| C-02 cost-distance back-link 0 ("source") on an unresolved tie plateau | cost-distance | Write NONE; give the tie phase the sweep schedule | S |
| C-03 NaN accumulation becomes `streams = 0` (main trunk vanishes, HAND/Strahler inherit it) | hydrology | Unresolved third state, or a mandatory `accumulationConverged` | S |
| C-04 isoband ring hash saturates above 214,748 (O(n²) on projected coordinates) | isolines → ring-assembly | Exact mode (`vertexTolerance: 0` hashes bits) and origin-relative quantizing | M |
| A-1, A-2 scan-statistic `evaluate` binds 9 storage buffers; nearest-feature join 9–10 in some configs; 46 kernels at exactly 8 | scan-statistics, spatial-join | Merge buffers; add the rule-16 node spec | S / M |
| E-10 transition diagnostics needs 16.9 KB of private memory at 64 classes | distribution-dynamics | Storage scratch, or cap at 44 classes | S |
| E-11 scratch above 128 MiB fails at dispatch (global stats 1.24M rows, nearest extras 838k queries, hot-spot recipe ~880k H3 rows) | several | Check at construction (rule 16); chunk nearest-nodes extras | S / M |
| Open algorithms-roadmap P0s (Max-P cap, `log1p`, Kahan spec) | | Unchanged; see that roadmap | S |

### Q1: compile cost

Every item here is an A/B of pipeline count, compile time and time-to-first-frame. The GPU time should not change.

| Item | Fix | Expected |
|---|---|---|
| E-01 frontier round baked (6 network contributors) | Round from a GPU cursor word; parity is the only variant | 64 → 2 relax pipelines per reachability |
| E-02 cost-matrix and k-function batch offsets (offset residue mod 256 B baked per binding) | 64-element-aligned batch stride; full-lane short batch | ~400 → ~5 pipelines at 5 batches |
| E-03 raster gate group slot baked (`1 << groupOffset`) | Slot from a word | Default terrain flow ~60 → ~15 large pipelines |
| E-04 cumulative viewshed `FIRST_OBSERVER` | Params word | 50 → 1 sight-line pipeline at 1024² × 200 observers |
| E-06 k-means seeding, class breaks, SKATER, column profile, segregation, rank columns, terrain horizon sectors | Rule 12 | Up to thousands → 1 per kernel |
| E-07 Ripley envelope and weights-trace overrides | Counter in `parameters` | 2 × simulations → 2 |
| E-05 522 synchronous `.compile()` sites in 238 app files; none of the six contributor doc pages shows `compileAsync` | Async scene mount (rule 15); one shared "Running contributors" doc section | Main-thread stalls removed |

### Q2: wasted GPU work

| Item | Fix | Effort |
|---|---|---|
| E-08 k-means, location-allocation swap (~500 dispatches when converged), Weiszfeld, scan rounds, pycnophylactic (full copy per round) | Gate (rule 13); drop the pycnophylactic copy | S each |
| E-12 single-workgroup reductions (Pearson/Gamma, cell-table compare, k-means restart scoring, network statistics finish) | Two-stage | S each |
| E-13 scan-statistic `nearest` is Z²·⌈L/32⌉ (~2e9 at Z=4096), not Z² | Raise algorithms-roadmap P1 grid kNN priority | M |
| E-14 hot-address atomics (join counts, Knox) | Workgroup reduce, one atomic per workgroup | S |
| E-09 AZP 8 KB private bitset per improving row | Algorithms P3 #3; meanwhile lower the row cap | S / L |
| C-10 cumulative viewshed batch sized by pixels, not samples (~4e9 samples per dispatch at 1024²) | Size by `observers × pixels × max(w,h)` | S |
| C-12 raster profile: one lane per path | Segmented scans | M |
| B-18 `pair-sum` dispatches per hash slot × 256 | Dispatch by live count | M |
| B-19 line-graph ban scan in both passes; B-20 isochrone splat without extent culling | Sorted bans with binary search; bbox cull | S / M |
| E-15 escape-hatch `Computation` kernels rebuild bind groups every encode (geospatial point query, terrain RGB decode) | Texture bindings in `createWGSLKernelNode`, then retire the escape hatches | S |

### Q3: one library (input to the correctness round's wave 2)

Session 32's planned wave 2 ("naming and port consistency") can execute this directly. Order the work so that
mechanical renames land in one pass per entry:

1. **Barrels and layering** (A-4, A-13): `export *` barrels plus a diff spec. Move `contracts/`, cell-key WGSL and
   grid utilities below all entries. Break the spatial-analysis↔network, spatial-analysis↔dataframe and
   raster↔terrain import cycles. Rename `GPUTerrainCellSizeMode` to `GPURasterCellSizeMode`.
2. **Vocabulary renames** (A-8 to A-12, A-26): the table below. Most of the churn is in terrain (`settings` →
   `parameters`, loose outputs → `output`) and network (CSR names, `max*`). Delete leftover synonyms.
3. **Shared inputs** (A-11, A-20): a `GPUNetworkTopology` object assignable to the CSR part of
   `GPUSpatialWeights`. A `grid: GPURasterMetadata` prop for every raster and terrain contributor, with cell size
   read from it rather than from 27 packed layouts. Terrain outputs typed as raster bands.
4. **Support and composition** (A-5, A-6, A-19):
   - The feature is the unit of support everywhere: contiguity takes the four-level polygon layout and emits
     feature rows.
   - Count-like inputs accept `uint32 | float32`.
   - Recipes pack their own settings and derive inference metadata from them.
   - One 2SFCA/3SFCA kernel on W, plus `GPUNetworkCostWeights` (cost matrix → W).
5. **Contracts with consumers** (A-15, E-16): wire the capacity plans into the W and join grow loop, computing each
   plan from the same constants as the allocation, or make them internal. Promote `GPUCompactOutput` to the contract
   and delete its twins (A-14).
6. **Construction** (A-17, A-18, A-21): port the eight `addToGraph` classes. Owned resources come from the graph.
   Algorithm-shape values are compile-time and named `maximum*`; every other scalar, including `seed`, is per frame.
7. **Errors** (A-16): shared validators and `assert` for invariants (rule 17). There are 2,848 throws today, with
   about 155 KB of message text.

### Q4: test and doc integrity

| Item | Effort |
|---|---|
| `requireTestDevice` helper; make the WebGPU canary fail under headless (A-3) | M |
| Fake-`'core'`-limits node spec over every contributor's maximal options (rule 16) | M |
| Doc `ts` blocks type-check (A-7). Fix the four stale snippets now: SKATER/MST `attributes`, `GPUGeodesicPairs({model})`, the raster README quick start, `GPUCommandGraphContributor` | S |
| Generated "Parameters" subsection per class: packer, length constant, settings fields. 106 exported functions are undocumented, 33 of 44 in terrain (A-22) | M |
| Oracle generators under `test/oracles/` (A-23); first targets are network (networkx), terrain (gdaldem/richdem), class breaks (mapclassify/jenkspy) and the variogram (scikit-gstat) | L |
| Randomized and property fixtures for Delaunay, make-valid, buffer, cover, p-median and zones (D pattern 6) | M |
| GPU specs for `GPUCellRollup` and `GPULineNoding` (A-24) | S |

### Smaller items

Each is S unless marked. Full rows are in the reports.
- **Status:** D-10 `requiredCount` echoes `count` in overlay and polygonize.
- **Geometry:**
  - D-12 an empty part makes a whole feature uncertain;
  - D-13 hole rules differ across measures, labels and triangulation (M);
  - D-14 grid-generator polygons do not share bit-identical vertices;
  - D-15 buffers keep closing duplicates;
  - D-16 dot-density rejection bias (M: sample in triangles).
- **Network:**
  - B-11 isochrone `atomicMin` on -0 and negative costs;
  - B-12 service-areas `localIterations` and tie mismatch;
  - B-15 self-loops count once by default;
  - B-16 NaN opportunity weights;
  - B-17 directed twins use candidate slots.
- **Dataframe:**
  - B-13 reversed windows with interval rows;
  - B-14 the two quantile engines disagree (M);
  - B-22 Philox purpose IDs collide across modules.
- **Terrain and raster:**
  - C-05 point-horizon `targetIgnoreDistance` defaults to 150 m;
  - C-06 infinities in contours;
  - C-07 y-down isoband extents;
  - C-11 unused `elevationValidity` binding;
  - C-13 geomorphon `flatDistance` vs GRASS;
  - C-26 stretch `gamma` is inverted relative to GDAL.
- **Policy:** D-21 and E-17 put a CPU oracle and a CPU bandwidth helper in shipped `src`; move them to `test/`, and
  compute bandwidth with a GPU moments reduction.
- **Lifetime:** E-21 `SpatialJoinPrepared` allocates before validating.

## Corrections made to other roadmaps

These claims were checked against code and corrected in place on 2026-10-13:

- **Algorithms roadmap, decision 1.** Override constants do not share pipelines: `PipelineFactory` hashes
  `constants`. The rule is now rule 12. The compile-site count is 522 sites in 238 files (165 showcase + 73 explorer),
  not 167.
- **Algorithms roadmap, decision 4.** One float CAS add remains in scope (point density), not 3–4.
- **Algorithms roadmap, verdict.**
  - "64-bit fixed-point pair statistics" is qualified by B-01 and B-06.
  - The exact-predicate review note now says the predicate is sound but ten PIP loops are not (D pattern 1).
  - "Node drift, the only real robustness gap" is wrong (D-01 to D-09).
- **Algorithms roadmap, P0 and P1.**
  - Map-matching budget: worse than stated (B-02).
  - Viewshed pyramid: the in-graph build already exists, so only the default flip remains (S).
  - Scan-statistic `nearest`: cubic, not quadratic.
  - p-median swap: already exists; what remains is gating and quality.
  - k-means reseed: contradicts the documented contract, so decide the API first.
  - Accumulation NaN tails: worse than stated (C-03).
- **GPU spatial analysis roadmap, status table.**
  - Tranches 1, 2, 3, 5 and 6 are marked "Partial", with the adoption and fixture gaps (report A §5, report D §4).
  - Tranche 2's "mandatory `GPUSpatialContext`" is false: about 99 declarations are optional with a planar default,
    and that default is the decision now in force.
- **gpugraph roadmap.**
  - The Philox note is stale: it lives in `utils/philox-random.ts` with 14 importers; what remains is purpose-ID
    collisions (B-22).
  - The terrain-flow iteration numbers are stale: the defaults are now 512 and 64.

## Canonical vocabulary

This is the subset of report A §4 that wave 2 should apply. Counts are library-wide. AGENTS.md takes precedence
(no abbreviations), then the dominant spelling.

| Concept | Canonical | Replaces |
|---|---|---|
| Packed per-frame view | `parameters` (second view: `uint32Parameters`) | `settings` (terrain, 36 files), `wordParameters` |
| CPU settings and packer | `GPU<Class>Settings`, `getGPU<Class>ParameterValues(settings)`, `GPU_<CLASS>_PARAMETER_LENGTH` | `*Parameters` types, `encode*`/`create*`/`pack*`, 15 drifted packer names |
| Outputs | `output: {...}`, status in `output.status` | loose outputs (125 of 259 classes), `result(s)` |
| Compact result | `GPUCompactOutput` promoted to a contract | `GPUSpatialQueryOutput`, unused status leaves |
| Limits | `maximum*`/`minimum*`; `maximumIterations`, output `iterationCount`, flag `converged` | `max*` (109), `min*` (15), `iterations`, `maxRounds`, `roundCount`, `*Passes` |
| Capacity | `<noun>Capacity` (memory) vs `maximum<Noun>Count` (algorithm) | `maximumCapacity`, `neighborCapacity`/`maximumNeighbors`/`maximumNeighborCount` for one quantity |
| Seeds and starts | `seed` in settings, always per frame; graph starts are `start*` | compile-time `seed` (6 classes), `seed*` start nodes |
| Permutations | `maximumPermutations` (compile time), `permutationCount` (per frame), `simulationCount` | `permutations` |
| Row selection | input `mask` (`vertexMask`/`edgeMask`/`nodeMask`), output `outputMask` | `keepMask`, `selection`, `valueMask`, `included` |
| Raster validity | input `validity`, output `outputValidity` | terrain `validity` used as an output |
| Polygons | `GPUPolygonGeometryPort` four-level layout; rows are features | 3 encodings in 37 types, `polygonPositions`, `featureRingOffsets` |
| Lines and tracks | `positions` + `lineOffsets`, `trackOffsets` | `pathOffsets`, `startIndices` |
| Network | `topology: {offsets, neighbors, weights, nodePositions}`; `costBreaks`, `costLimit` | `edgeTargets`/`edgeCosts`; `bandThresholds`/`breaks`/`bandBreaks` |
| Raster grid | `grid: GPURasterMetadata`; `GPURasterCellSizeMode` | `width`/`height` props + `cellSize` in 27 layouts; `GPUTerrainCellSizeMode` |
| Counts | derived from view length, else `rowCount`/`featureCount`/`nodeCount`/`cellCount`; GPU-side counts `number \| GraphDataView<'uint32'>` | `sourceCount` as a number in spatial-analysis vs a view in network |
| Construction | `graph.add(contributor)` (`getCommandNodes`), `add<Name>Recipe` | `addToGraph` (8 classes) |
| Abbreviations | `Statistics`, `Information` | `*_STATS_*`, `GPUSpatialWeightsInfo` |

## Quick A/B candidates

Run these after the matching Q1 and Q2 items, one GPU spec at a time:
1. Showcase scene load (network isochrones, terrain watersheds): sync compile vs `compileAsync`.
2. Reachability round cursor: pipeline count and compile time at `maxIterations` 64.
3. Cost matrix with a padded batch stride.
4. Raster gate slot as data for `GPUTerrainFlow` at 2048².
5. k-means gating at n=1e6, k=8, converging in about 10 of 64 iterations.
6. Location-allocation swap gating on a converged instance.
7. Pycnophylactic gating with no copy.
8. Pearson/Gamma two-stage reduction at 1e6 rows.
9. Scan-statistic `nearest` at Z = 1024, 2048 and 4096.
10. Cumulative viewshed observer word.
11. Join-count atomics.
12. Point-query bind-group reuse.
13. D8 vs D-infinity accumulation at 2048² (C suspicion: 10–50x).

## Open questions for the owner

- **k-means empty clusters**: reseed (algorithms P0) or keep the documented "keeps its previous position" contract?
- **Variogram model fitting** reads back bins and fits on the CPU. It is post-processing, not a fallback. Is that
  within the GPU-only policy?
- **Horizon default azimuth**: 315 equals RVT's 45, not RVT's default of 315. Should the default follow RVT parity?
