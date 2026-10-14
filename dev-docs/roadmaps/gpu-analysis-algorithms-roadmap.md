# Analysis contributors: GPU algorithm roadmap

Status: proposal, 2026-10-13. Based on an eight-family algorithm review (code read at `map-contributors`
731e2a066, compared against GPU and CPU state of the art). Related plans:
[GPU spatial analysis roadmap](./gpu-spatial-analysis-roadmap.md) (ontology and contracts),
[gpugraph roadmap](./gpugraph-roadmap.md) "Analysis contributors" (open items), and
[perf round 2026-10-06](../perf/perf-round-2026-10-06.md). The 2026-10-13 correctness, consistency and
performance review is in the [quality roadmap](./analysis-contributors-quality-roadmap.md); it corrected several statements below in place. Scope is our contributors in
`gpu-spatial-analysis`, `gpu-network`, `gpu-raster` subdirs, `gpu-terrain`, `gpu-dataframe`; luma core
primitives are trusted as given.

## Policy: everything runs on the GPU

Every analysis runs end to end on the GPU, including the ones a CPU would do faster today. This keeps
data GPU-resident and composable in one command graph, gives one precision and failure contract, and
spares applications from juggling devices. It follows that:

- There are no CPU fallbacks, no `backend` options and no CPU-build / GPU-query hybrids in the library.
  Upload of caller-supplied input data (geometry, rasters, a timezone transition table) is not a hybrid.
  It is input.
- Readback happens only for results the application asks to read, never for control flow. Convergence,
  sizing and loop exits stay on the GPU (status words, indirect dispatch, capacity plus overflow).
- CPU state of the art informs but does not replace. Reference CPU algorithms (GEOS, PySAL, Delaunator,
  Priority-Flood, Dijkstra) provide oracles for specs and a speed bar for A/B checks. CPU oracles stay
  out of the public barrels.
- When the GPU loses to the CPU, that is a GPU work item: find the work-efficient parallel formulation,
  as with Priority-Flood → tiled fill with a spill graph, or Bowyer-Watson → insertion plus flips.

## Verdict

Most of the library already uses the right GPU algorithm, so this plan doesn't call for a rewrite.
Across about 150 contributors:

- **Optimal or near-optimal (about 80%).** These include the Hilbert-packed BVH joins, the DE-9IM relate
  engine, segment intersection, line simplification, hulls and calipers, DBSCAN (FDBSCAN-class), SKATER,
  OLS/GM/2SLS and local stats on CSR, kriging, the horizon sweep, SAT topographic position, geomorphons,
  isobands, zonal stats, reachability (chaotic Bellman-Ford with indirect gating), isochrones (the
  Valhalla method), edge bundling (KDEEB) and segmentize/chunk.
- **Numerics are a strength.** Philox counters, exact integer orient/incircle, fixed-order tile sums with
  Kahan merge, 64-bit fixed-point pair statistics, no float atomics on oracle-compared paths. This is
  ahead of most GPU geometry codes (gDel2D, CudaHull fall back to CPU f64 for exactness). Exceptions
  found on 2026-10-13: the variogram/correlogram fixed-point unit is pegged to the data range, and
  variogram pair counts wrap at 2³². Also, inexact point-in-polygon and branch tests sit next to the
  exact predicates in overlay, ring assembly, zones and `edgeBreaks` (quality roadmap Q0).
- **The gaps are concentrated.** They fall into four kinds:
  1. Quadratic cores behind good APIs: Delaunay, overlay classification, Max-P, scan-statistic kNN,
     point-in-polygon refine on large polygons.
  2. Missing adaptivity, where every run does the worst-case work: permutation tests, the GWR bandwidth
     ladder, the time-window filter, crossfilter.
  3. Sequential chains run as long dispatch sequences: Ward, AZP, p-median greedy, accumulation on long
     stems.
  4. Over-budget cliffs: triangulation of large polygons fails, and Max-P can lose the device.

## Library-wide engineering decisions

1. **Compile cost is the hidden one-shot tax.** Upstream `GPUCommandGraph.compileAsync()` already
   batches all pipeline creation (`PipelineFactory.beginAsyncCompilation`), but our docs (e.g.
   `gpu-spatial-analysis.md:47`) and the apps call `graph.compile()` (522 sites in 238 files: 165 showcase
   and 73 explorer, with 0 `compileAsync`).
   - Switch docs and examples to `compileAsync()`.
   - Add a rule and a check: loop-varying values (round, gate slot, seed, layer, batch or observer
     index, sub-view offset residue) never reach WGSL source **or override constants**. They come from
     a GPU state or params word. Overrides share the shader module but not the pipeline, because
     `PipelineFactory` hashes `constants`. Input sizes may stay baked or become overrides. Today a
     single reachability compiles 64 relax pipelines where 2 would do (quality roadmap Q1, rule 12).
   - No contributor-level caches (repo rule).
2. **No silent cliffs.** Any per-item work budget (triangulation vertex budget, map-matching node
   budget, unrolled iteration caps) reports overflow in a status output and has a parallel path that
   scales past it. A budget is never just a limit.
3. **One shared load-balancing utility.** Add `createLoadBalancedForEachNode` in `src/utils` (scan +
   load-balanced search, Merrill–Garland / moderngpu). It replaces the ad-hoc 64-item block hybrids and
   fixes hub-row skew in CSR loops. That covers spatial lag, catchment, network statistics, geometry
   measures, region evaluation, and candidate probes for huge left boxes.
4. **One shared fixed-point accumulator.** It takes a data-derived scale (ReproBLAS-style) and replaces
   the per-file scales and the one remaining in-scope CAS float add (point density), which gets it
   first. The other CAS sites are set-once publishes or union-find links.
5. **GPU-resident control loops.** Iterative searches (bandwidth selection, local search, swap
   heuristics, convergence) run as fixed-depth gated loops or indirect-dispatch loops, not CPU loops
   with readback.
6. **Subgroups: later, opt-in only.** They shipped in Chrome 134 and we don't use them yet.
   Candidates: tree reductions, ballot compaction and workgroup histograms. Results are not
   bit-identical to the fallback, so they go behind an explicit option after the items below. Use f16
   only for raster band math. Request raised storage-buffer limits when offered and keep the 8-binding
   path.

## Implementation status (2026-10-14)

Every tranche below was implemented on `map-contributors` (uncommitted). Each item has node specs and headless GPU
specs (run serially), oracles from reference packages, and a same-run A/B against the path it replaced. Timings are
medians on one Apple-silicon WebGPU adapter.

| Item | Status | Evidence |
|---|---|---|
| Decisions 3–4 | Done | `createLoadBalancedForEachNode` / `createLoadBalancedSegmentSumNodes` (merge-path) and a 96-bit fixed-point accumulator, both in `src/utils`. Spatial lag hub: 18.1 → 4.6 ms. Network statistics hub: 21.5 → 0.9 ms. Point density sums are now bitwise deterministic. |
| P0 | Done | See the quality roadmap. |
| P1 `compileAsync` | Done | No synchronous compile in either app. |
| P1 time-window `sorted` range mode, timezone table, Besag–Clifford stopping, crossfilter skip | Done | All opt-in; defaults keep esda parity. |
| P1 cost-matrix `targetNodes`, reachability relaxation counter | Done | The inflation probe is run; the delta-stepping verdict stays deferred per the rejected list. |
| P1 viewshed `'pyramid'` default | **Rejected (measured)** | At 512², viewshed `'march'` takes 1.1 ms against 3.3 ms for the pyramid, so `'march'` stays the default. |
| P1 scan `nearest` | Done | Radix select instead of grid kNN: at the default capacity a grid cannot prune. |
| P1 DBSCAN cell order, Hilbert query sort | Done | Point-query sort: nearest k=1/k=4 0.70×/0.68×, PIP 0.92×. Line and polygon query sort defaults off: it measured slower (1.2 → 1.7 ms). |
| P1 Akl–Toussaint, degree-first coloring, parallel cleanup | Done / partial | Cleanup at tolerance 0: 184 → 2.6 ms. Akl–Toussaint filters points but cannot shrink the sorts, because `GPUSort` has a fixed length (upstream). |
| P1 SLRM/MSRM via the 64-bit SAT | Done | 4.6 ms at 1024², with an origin-relative anchor. |
| P2 y-slab PIP refine | Done | 5 × 50k-vertex polygons × 1e6 points: walk 1,833 ms → slabs 115 ms. Slabs are chosen from GPU edge-travel statistics. |
| P2 large-polygon triangulation | Done | Skyline-50k: earcut 542 ms → monotone 118 ms. Double-comb-50k: 34,080 ms → 115 ms. |
| P2 point-BVH kNN/radius | Done, opt-in | The BVH loses to the grid on every measured workload (1.6–4.7×), so `'auto'` selects the grid without building the BVH. |
| P2 Fisher–Jenks D&C | Done (histogram approximation) | 32/34 fixtures are exact. Two are single-cut local minima (worst GVF shortfall 3.4e-8), documented. |
| P2 p-median swap quality, sparse costs; GWR golden section; scan replicate batching; Ward mutual merges | Done | The scan replicate cap is removed (permutations up to 99,999). |
| P2 tree contraction for accumulation and Strahler | Done | Exact on integers; no 64-round cap. D∞ costs 3.4× D8 at 2048² (55.7 vs 16.3 ms), not 10–50×. |
| P2 summits range max, PBA EDT, relaxation continuation | Done | EDT exact at 1024² with 1,000 seeds: 2.3 ms. |
| P2 KDE linear binning | Done | Exact 395 ms → binned 4.4 ms; max error 5e-3 against a 0.8 peak. |
| P2 zone-event segmented scan, spherical BVH line locate, map-matching cost table | Done | Route table 4.9 ms per frame after a 51 ms build. |
| P3 #1 GPU Delaunay (parallel insertion and flipping) | Done | Faster at every measured size: 1e3 sites 36.5 → 15.1 ms, 1e4 1,094 → 495 ms, 1e5 517 ms (the legacy path is infeasible). A gated continuation covers collinear inputs. |
| P3 #2 OverlayNG face labelling and snap rounding | Done | 182 shapely fixtures; three invalid-input contract differences are documented. Closes D-06. Robustness costs speed at small sizes: intersection 1e3 7.5 → 13.0 ms and 1e4 48 → 62 ms. It scales where the legacy path cannot: 1e5 vertices in 651 ms for intersection and 181 ms for dissolve. The legacy f32 path is not kept for small inputs, because it carries the D-06 precision bug. |
| P3 #3 Max-P rewrite and AZP articulation | Done | Both row caps are removed. |
| P3 #4 exact CPA encounters | Done | 2,000 tracks: 10.1 ms. |
| P3 #5 Fréchet decision and banded DTW | Done | 10k pairs × 200 vertices: distance 39 ms, decision 1–3 ms. |
| P3 #6 radius-bounded betweenness and closeness | Done | `GPUNetworkCentrality`: 4.5–7× faster than CPU Brandes (networkx oracle). |
| P3 #7 HDBSCAN on the GPU | Done | Parallel merge tree and EOM; matches sklearn exactly on 4 of 6 scenes, ARI ≥ 0.983 on the other two. 100k points: 125 ms. |
| P3 #8 tiled spill-graph depression fill and peak prominence | Done | Spirals 425 → 36 ms (relaxation did not converge). 2048² pits 241 → 126 ms. Prominence shares the merge-tree helper. |
| P3 #9 Falcon cube | Done | Crossfilter histogram views. |
| P3 #10 scanline cell cover | Done | H3 res 9: 487 → 26 ms; 0.01°×20° strip: 628 ms (overflow) → 88 ms. |

Upstream items that blocked parts of this work:
- `GPUSort` with a GPU-side row count.
- Gateable `GPUBVH`, `GPUSort` and `GPUScan` builds.
- Folded `v == v` NaN tests in gpgpu kernels.

## Tranches

Effort: S ≈ hours, M ≈ a day or two, L ≈ multi-day. Each item lands with a quick same-run A/B against
the current path (one GPU spec at a time), not a benchmark suite. Where a CPU reference exists, it is
the oracle and the speed bar.

### P0: safety and silent-wrong results (all S)

| Item | Where | Why |
|---|---|---|
| Hard row cap with a clear status in `GPUMaxPRegions` until the P3 rewrite | spatial-regionalization | Documented device loss (watchdog) from about 50k rows |
| `routeNodeBudget` overflow flag **and drop policy** in map matching | gpu-network/map-matching | Worse than approximate: at 1 km sample spacing the default budget of 64 missed 186 of 227 routes in a grid mimic, and they show as ordinary breaks. Needs a counter plus worst-key eviction (quality roadmap B-02) |
| Empty-cluster reseed in `GPUKMeans` | spatial-clustering | Correctness. The code documents the opposite contract ("a center that never received a point keeps its previous position"), so decide the API first |
| Check scan-statistic LLR uses `log1p` near c≈e; Kahan-merge spec at 1e6–1e7 rows on Metal | scan-statistics, spatial-regression | Precision guards (unverified, check first) |

### P1: cheap, high-usage wins (S each)

| Item | Where | Expected effect |
|---|---|---|
| `compileAsync()` in docs, explorer and showcase | docs, examples | Cuts cold start for every one-shot analysis |
| Time-window `sorted: true` range mode (2 binary searches → indirect draw record) | gpu-dataframe/time-window-filter, temporal-reduction | O(log n) scrubbing, no compaction |
| Besag–Clifford sequential stopping, opt-in `stopAfterExceedances` | permutation-inference, Knox/Mantel, global perms | About 5–25x less work on LISA maps; default P stays for esda parity |
| `GPUNetworkCostMatrix` compact `targetNodes` output | gpu-network/network-accessibility | Removes the dense rows×nodes ceiling (accessibility, K function, OD) |
| Relaxations-per-edge counter in reachability | gpu-network/network-reachability | Decides near-far / delta-stepping from data |
| Viewshed: default to `'pyramid'` above ~64 px (the in-graph pyramid build already exists) | gpu-terrain/terrain-analysis | Bit-identical, sub-linear; only the `'march'` default remains |
| Grid kNN in the scan-statistic `nearest` kernel | scan-statistics | Removes a Z²·⌈L/32⌉ cliff, roughly cubic (~2e9 evaluations at Z=4096) |
| DBSCAN cell-ordered position copy | spatial-clustering | Coalesced reads (pair-statistics already does it) |
| Crossfilter: skip unchanged dimensions | gpu-crossfilter | O(changed·n) per brush |
| Timezone transition table uploaded as input, binary search per row | calendar-buckets | Removes per-row CPU DST offsets |
| Sort queries by Hilbert key in nearest and PIP joins | spatial-join | Traversal coherence |
| Akl–Toussaint prefilter; degree-first map coloring; tolerance-0 parallel cleanup | group-geometry, map-coloring, geometry-edit | 10–100x smaller hull sorts on blobs; fewer colors |
| SLRM/MSRM via the existing int64 SAT (verify current code first) | relief-visualization | O(1) per pixel |

### P2: structural replacements (M each)

| Item | Where | From → to |
|---|---|---|
| Prepared y-slab edge index in the PIP refine (code exists in relate engine and cell cover) | spatial-join + `GPUSpatialJoinPrepared` | O(candidates·E) → O(log E) per pair |
| Large-polygon triangulation: workgroup-cooperative path (parallel y-monotone decomposition, then per-piece monotone triangulation); per-invocation earcut stays for small polygons | polygon-triangulation | Over-budget polygons return `valid=0` today |
| Point-BVH kNN/radius path chosen on the GPU by grid occupancy (max/mean cell count) | neighbor-search | Robust on clustered points; reuses `nearest-nodes.ts` |
| Fisher–Jenks monotone divide-and-conquer DP + local fine-bin refinement | gpu-dataframe/column-classification | O(kB²) → O(kB log B); removes bin-snapped breaks |
| p-median swap quality (Whitaker / Resende–Werneck gains evaluated in parallel) and gating of the existing 100-iteration swap loop; sparse cost option | location-allocation | Ungated swap (~500 dispatches when converged) → gated interchange quality |
| Tree contraction for `'cells'` accumulation and Strahler order (exact on integers) | gpu-terrain/hydrology | 64-round cap and NaN tails on long rivers → O(log n) rounds. Today the NaN tail also becomes `streams = 0` downstream (quality roadmap C-03) |
| Summits: row-run range max over `GPURasterExtremaPyramid` | terrain-features | O(R²) → O(R) per pixel |
| GWR golden-section bandwidth search as a fixed-depth gated loop on the GPU; Gaussian cutoff | spatial-regression | Ladder of L full fits → about 12, no readback |
| Scan-statistic replicate batching (running max) | scan-statistics | Removes the P·Z·T ≤ 2²⁵ cap |
| KDE linear binning, then box-blur cascade or IIR for radius > 32 | point-density | Accuracy, then bandwidth-independent cost |
| Ward: merge all mutually-cheapest pairs per round | spatial-regionalization | n−k serial rounds → about O(log n) |
| Relaxation engine graph size (lower default unroll + GPU-gated continuation) | raster-relaxation.ts | About 640 nodes per relaxation × fill, 3 flats, cost distance |
| PBA-style banding for exact EDT | distance-field | Parallelism W or H → full |
| Load-balanced-search and fixed-point utilities (decisions 3–4) | src/utils | Shared |
| Zone-event walk as segmented scan; line-merge pointer jumping; spherical BVH line locate | trajectory-zones, line-merge, linear-referencing | Serial tails → log depth |
| Map matching: GPU-precomputed bounded cost table (UBODT-style) for static networks via batched reachability | gpu-network/map-matching | Replaces per-pair bounded A* for large track batches |

### P3: large items, ordered by value

1. **GPU Delaunay by parallel insertion and flipping** (gDel2D-style), using the existing exact
   incircle for the flip test and iterated flips in place of gDel2D's CPU star-splaying repair. This
   replaces the O(n²) Bowyer-Watson with 3 graph nodes per site; Voronoi already consumes triangles.
   It is the largest gap against the CPU bar (Delaunator).
2. **Overlay classification by face labelling** (OverlayNG), plus snap-rounded nodes on an integer
   lattice. This removes the quadratic membership probe for overlay, dissolve and buffer. It also fixes
   node drift and the f32 crossing points. It is not the only robustness gap. Delaunay hull loss,
   even-odd buffers, make-valid hole faces, `edgeBreaks` branching, f32 overlay side probes and
   zone crossings are separate fixes (quality roadmap Q0).
3. **Max-P rewrite.** Parallel multi-seed construction with a frontier queue (one workgroup per seed),
   then AZP improvement. AZP gets a local articulation test and conflict-free batched moves, which also
   lifts its 65,536-row cap.
4. **Exact CPA encounters.** Use a space-time segment grid and a closest-approach quadratic per segment
   pair, as MobilityDB `tdwithin` does. Keep clock mode as the preview. This is the biggest accuracy gap
   in movement.
5. **Fréchet decision mode** (`epsilon`, free-space early exit, bound pruning) and **banded DTW**.
6. **Radius-bounded betweenness and closeness.** Lane-batched reachability with tight-edge σ/δ
   accumulation and sampled pivots. It is the highest-usage missing network analysis, and adding it
   means reversing a roadmap non-goal.
7. **HDBSCAN fully on the GPU.** kNN core distances, then a Borůvka MST over mutual reachability, then a
   single-linkage dendrogram from the sorted MST via union-find, then condensing and EOM extraction by
   tree contraction.
8. **Tiled depression fill with a GPU spill graph** (Barnes 2016 structure). Tile-local fill, then a
   minimax shortest-path solve over tile-perimeter cells on the GPU, then one propagation pass. It
   replaces sweeps for spirals and large lakes and unlocks seamless cross-tile flow. GPU peak prominence
   (sorted union-find over the spill graph) follows from the same structure.
9. **Falcon-style cube for histogram views** in crossfilter, built on the GPU (above ~10–50M rows).
10. **Boundary-trace plus scanline-fill cell cover** for large or thin polygons.

## Rejected or deferred (with reasons)

- **CPU fallbacks, `backend` options and CPU-built structures.** Rejected by policy; see the top
  section.
- **Contraction hierarchies with PHAST/RPHAST.** Deferred. Building the hierarchy is a sequential
  CPU-class preprocessing step, so it conflicts with the policy. The metric is also fixed, while
  per-frame weights are the product. Revisit only if a GPU contraction or customization method proves
  practical. Until then, many-to-many cost is improved through `targetNodes` and the reachability
  inflation probe.
- **Replacing the Hilbert-packed BVH with Karras/Apetrei LBVH.** Same tree quality for our workloads; no
  win.
- **Delta-stepping before the inflation probe says >3x.**
- **Chunked max-plus Viterbi and a map-matching private heap.** Viterbi is O(T·K²); routing dominates.
- **Bound-pruned k-means (Elkan, Hamerly, Yinyang).** No gain in 2D on a GPU.
- **A GPU hash join.** It needs 64-bit CAS; sort plus binary search stays.
- **ML lag/error now.** The roadmap gate stands. When admitted: Chebyshev or Monte Carlo
  log-determinant on the CSR lag with a GPU-resident 1-D search; never LU.

## Review notes

Facts checked during the review, recorded here so later passes do not re-propose settled items:

- Exact integer orientation and incircle are already used across the geometry family (16 files), so a
  new filtered `orient2d` predicate is not needed. They are not used everywhere topology is decided:
  10 of 15 hand-written point-in-polygon loops use an f32 crossing (quality roadmap rule 1).
- `raster-relaxation.ts` already skips inactive tiles via per-tile activity stamps. Its real cost is the
  unrolled graph size (P2 item).
- No new warmup API is needed: upstream `GPUCommandGraph.compileAsync()` exists. We only need to call it
  (P1).
- The viewshed's extrema pyramid is already built in-graph (`gpu-terrain-viewshed.ts`), so making
  `'pyramid'` the default needs only an A/B check.
- Where the review measured the GPU against CPU references (Delaunator, earcut, Dijkstra,
  Priority-Flood, GEOS overlay), the CPU numbers are speed bars for the A/B, not routing thresholds.
  They are estimates until each item's A/B runs.
