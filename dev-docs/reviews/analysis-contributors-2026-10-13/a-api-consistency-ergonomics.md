# Stream A: API consistency, ergonomics, docs and tests (analysis contributors, library-wide)

Reviewed 2026-10-13 on the live tree `vis.gl-build/luma-contributors` (map-contributors). Read-only. Paths are relative to
`modules/experimental/src/` unless prefixed `docs/` or `test/` (= `modules/experimental/test/`). Sub-review notes with
full counts and scripts: `work/A/A1-naming-lifecycle.md`, `work/A/A2-contracts-errors.md`, `work/A/A3-docs-tests.md`,
`work/A/walkthroughs.md`. Every row in section 2 was re-checked by the coordinator against the source.
Baseline: deepdive `06-architecture.md`. Section 2 does not repeat it except where it was wrong, has since been fixed, or turned out worse
library-wide. Node specs run: 42 files and 202 tests passed. No GPU, tsc or build was run.

## 1. Verdict

The contributors share one solid construction shape (`constructor(props)` + `getCommandNodes(graph)` in 255 classes,
caller-owned outputs, GPU-side overflow, and every exported class and function has TSDoc). Above that layer there is
no library. Each entry has its own dialect:
- the per-frame view is `parameters` in four entries and `settings` in terrain;
- outputs go in an `output` bag in raster and dataframe, but are loose props in terrain and network;
- the same CSR, polygon and cost-break data is spelled three or more ways;
- `max*` and `maximum*` coexist (109 vs 215 props).

As a result, the three realistic pipelines I wrote out (section 6) hit 5 to 11 friction points each. The two worst are:
- **Support mismatch:** the zonal and choropleth recipes have one row per feature, while contiguity W and the rate recipe have one row per polygon part.
- **Conflicting contracts:** recipes that carry the same value twice, once packed for the GPU and once as CPU "metadata only".

The contracts layer is still mostly aspirational: `GPUSpatialContext` is the only contract in wide use (109 files). About
30 contract exports, including all capacity planning, have no consumer, and the roadmap's "Complete" status claims for
tranches 1 to 3 overstate adoption. Hygiene is good: TSDoc coverage is complete and construction-time validation is
thorough.

Four problems are real hazards:
- one kernel exceeds the default 8-storage-buffer limit;
- the GPU spec suite passes silently when no WebGPU device is present;
- the hand-maintained entry barrel has drifted from its subdirectories, so documented classes are unreachable;
- several doc examples no longer compile.

## 2. Findings (verified, ranked)

| ID | Sev | Kind | Where (path:line) | Finding | Evidence | Fix (principled) | Effort |
|---|---|---|---|---|---|---|---|
| A-1 | P1 | correctness | `gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts:~982-996` (`${id}-evaluate`) | The kernel binds **9** storage buffers. `createWGSLKernelNode` (`utils/wgsl-kernel-nodes.ts:217-224`) throws at declaration on any default-limit device (8 per stage), so `GPUSpatialScanStatistic` cannot be built in a stock browser. Specs pass because the test device is `featureLevel 'max'`. | Bindings: parameters, neighbors, distances, windowInfo, counts, baseline, totals, statistics, centerBest. No scan-statistics spec uses `getWebGPUTestDevice('core')`. | Fold `totals` and `parameters` into one buffer, or split the evaluate pass. Run every contributor's maximal-option spec once on a `'core'` device. | S |
| A-2 | P1 | correctness | `gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join.ts` (`reduce-distance`, `reduce-feature`) | Config-dependent overflow: `exclusive` with `queryIds`+`featureIds` plus `onAttribute` gives 5+4 or 6+4 bindings, which throws on default devices. 46 more kernels sit at exactly 8 with no headroom. | A2 static scan of 1,011 `bindings:` arrays. Only 8 of 524 analysis specs use the `'core'` device. | Same as A-1. Add a node spec that builds each contributor against a fake device with `maxStorageBuffersPerShaderStage: 8`. | M |
| A-3 | P1 | tests | `test/**` (1,555 `if (!device)` sites in 318 spec files); `test/utils/webgpu-test-device.spec.ts:13` | Every GPU spec returns green with zero assertions when no WebGPU device is present. The canary that should catch this is itself `it.skipIf(!hasWebGPU)`, so a mis-provisioned headless CI stays green. | `grep -rEn "if \(!device\)" test \| wc -l` = 1555; sampled bodies are `return;` (e.g. `test/gpu-terrain/relief-visualization/gpu-local-dominance.spec.ts:107`). | A shared `requireTestDevice(ctx)` helper that calls `ctx.skip()` (visible skip count) on node and throws under the headless project. Make the canary fail in headless. | M |
| A-4 | P1 | consistency | `gpu-spatial-analysis/index.ts` (1,390-line explicit barrel) | The entry barrel has drifted from the subdir indexes: 74 spatial-analysis names are missing (plus 15 dataframe, 13 raster, 2 network). Documented but unreachable from the package: `GPUKernelDensity`, `GPUMultipleTestingAdjustment`, `GPUSpatialWeightsValidation`, `GPUTransitionDiagnostics`, `GPURipleyEnvelope`, `GPULocalFrameProjection`. `GPUSpatialWeightsTraces` and `GPUSpatialLagImpacts` are in **no** index yet are documented. Several of these files are untracked (in flight), but the mechanism guarantees recurrence. | Coordinator grep: each name has `barrel=0` and exists in one subdir index. The last two have `anyindex=0`. | `export * from './<subdir>/index'` per subdir, so name conflicts become compile errors, plus a node spec that diffs entry exports against subdir exports. | S |
| A-5 | P1 | ergonomics | `spatial-weights/gpu-contiguity-weights.ts:47-77` vs `zonal-statistics/gpu-zonal-statistics.ts` `features` vs `recipes/points-in-polygons-choropleth-recipe.ts:34-46` vs `recipes/rate-cluster-map-recipe.ts:51-56` | **Support mismatch.** Zonal and choropleth rows are features (`polygonPositions, featureOffsets, polygonOffsets, ringOffsets`), while contiguity W and the rate recipe rows are polygon parts (`positions, ringOffsets, polygonOffsets`; docs say "Flatten multipolygons to one row per part"). For any multipart tract, points -> counts -> W -> Moran fails at the Moran constructor (`length must equal the weights row count`), and no public node collapses part-W to feature-W. On top of that, the choropleth recipe emits `uint32` counts but `GPUEmpiricalBayesRates.events` and the rate recipe take `float32`, so the two recipes cannot be chained without a hand-written cast kernel. | `gpu-empirical-bayes-rates.ts:55,118` (float32 only); choropleth result `counts: GraphDataView<'uint32'>`; 3 polygon encodings in 37 exported types across 18 dirs (A1 section 1.5, A2 table T1). | Make the feature the unit of support everywhere: contiguity takes the 4-level layout and emits feature rows, since a shared vertex between any parts makes features neighbours. Accept `uint32 \| float32` for count-like inputs (events, populations), as `GPUEmergingHotSpots.values` already does. This is the concrete form of baseline recommendation 1. | M |
| A-6 | P1 | ergonomics | `recipes/rate-cluster-map-recipe.ts` (`significanceLevel`, `permutation.permutations`, `permutation.seed`) | **Two sources of truth.** The GPU reads significance level, permutation count and seed from packed parameter views, while `result.inference` is built from separate CPU props documented as "only recorded ... Keep them equal". A mismatch is silent, so the legend and inference metadata can describe a test that was not run. | TSDoc on those props: "CPU-side significance level, only recorded in `result.inference.significanceLevel`; the GPU reads the level from `parameters` ... Keep them equal." | The recipe takes one settings object, packs it itself (it owns or receives the `GPUParameterBuffer`), and derives `result.inference` from the same object. Contributors should expose `getInferenceMetadata(settings)` rather than parallel props. | S |
| A-7 | P2 | docs | `docs/.../gpu-spatial-analysis.md:1031-1032`, `:3269`; `docs/.../gpu-raster/README.md:25-32`; `docs/.../geospatial.md:21`, `gpu-raster/operations.md:41`, `operations-statistics.md:34` | Doc examples that would not compile: SKATER and MST use `attributes, attributeCount` (real props are `values, columnCount`: `gpu-skater-regions.ts:77,79`); `GPUGeodesicPairs({model: 'wgs84'})` (real prop is `spatialContext`, `gpu-geodesic-pairs.ts:62`); the raster README quick start does `new GPURasterBand(...)` (a type), passes wrong props and calls `addPasses()`, which does not exist; three pages name the removed `GPUCommandGraphContributor`. | Coordinator re-read each. Doc line numbers move as peers edit. | Fix the four snippets. Add a docs-example type-check (compile ```ts blocks against the barrels in a node spec); A3's TS-API script (`work/A/scripts/docprops.cjs`) is a start. | S |
| A-8 | P2 | consistency | 255 Props types | **Per-frame view naming.** The packed view is `parameters` in 115 files (spatial-analysis 71, dataframe 19, raster 16, network 9) and `settings` in 36 (terrain 31 of 31, raster 4, spatial-analysis 1). CPU struct types are `*Settings` 58 vs `*Parameters` 39. Packers are `getGPU*ParameterValues` 117, `encode*` 3, `create*` 3, `pack*` 1; 15 names drift from their class (`getGPUNetworkIsochroneParameterValues` for `GPUNetworkIsochrones`, `getGPUTimeWindowParameterValues` for `GPUTimeWindowFilter`); 9 contributors have no packer at all. | A1 section 1.1-1.2; coordinator: `grep -E "^\s*settings\??:"` gives terrain 84 declaration lines and terrain `parameters` 0. | One rule: the prop is `parameters`, the CPU object is `GPU<Class>Settings`, and the packer is `getGPU<Class>ParameterValues(settings)`, plus `GPU_<CLASS>_PARAMETER_LENGTH`. Mechanical rename. | S |
| A-9 | P2 | consistency | terrain 34 of 36, network 14 of 20, spatial-analysis 64 of 156, pair-statistics 4 | **Output bag vs loose output views.** Five classes mix a bag with loose outputs (`GPUFlowAggregation`, `GPUIsolines`, `GPUTerrainSummits` with `output`/`outputDrop`/`drop`, `GPUTimeWindowFilter`, `GPURasterZonalStatistics`). The baseline said "98 of ~130" for spatial-analysis only; library-wide only 130 of 259 Props use `output`. | A1 section 0 and 1.3 (parser over 259 `GPU*Props`). | `output: {...}` for every contributor, and status in `output.status`. This is baseline recommendation 2 extended to terrain and network, which are the bulk of the renames. | M |
| A-10 | P2 | consistency | network (`maxIterations`, `maxSnapDistance`, `maxHops`, ...), terrain (`maxFillIterations`, `_MAX_` constants) | `max*` 109 vs `maximum*` 215 and `min*` 15 vs `minimum*` 52, against AGENTS.md "never abbreviate". The iteration cap has six spellings (`maxIterations` 50, `iterations` 21 sometimes meaning a cap, `maximumRounds`, `maxRounds`, `roundCount`, `*Passes`), and terrain alone mixes `maxFillIterations` with `maximumRadius`. | A1 section 1.8-1.9; coordinator count `max` 116 vs `maximum` 243 (all declaration lines). | `maximum*`/`minimum*` everywhere, with `minX`-style coordinate components exempt. Use `maximumIterations` for the cap, `iterationCount` for executed rounds, and `converged` as the flag. | S |
| A-11 | P2 | consistency | `gpu-network/network-accessibility/gpu-network-snapping.ts` Props; `network-reachability/gpu-network-reachability.ts:133`; `network-isochrones/gpu-network-isochrones.ts` Props; `recipes/drive-time-catchment-recipe.ts` | The same data has different names in one entry. CSR is `offsets/neighbors/weights` in 17 network Props but `offsets/edgeTargets/edgeCosts` in snapping (the docs pass `edgeTargets: neighbors, edgeCosts: weights`). Cost breaks are `bandThresholds` (reachability), `breaks` (isochrones) and `bandBreaks` (recipe). Across entries CSR has three shapes: loose network props, `GPUSpatialWeights` object, and gpu-graph `GPUGraphTopologyView`. | Coordinator Props dump of the three classes. | A `GPUNetworkTopology` object (`offsets, neighbors, weights, nodePositions`) structurally assignable to `GPUSpatialWeights`'s CSR part, plus one name `costBreaks`. | M |
| A-12 | P2 | consistency | `gpu-network/network-reachability/gpu-network-reachability.ts:74` vs `spatial-weights/gpu-spatial-lag.ts:28`, `areal-interpolation/gpu-areal-interpolation.ts:84` | `sourceCount` is a one-row GPU view in network and raster but a compile-time number in spatial-analysis. `validity` is an **output** in terrain (`gpu-terrain-curvature.ts:211`, "1 where every requested output is valid") but an **input** mask in raster and spatial-analysis (`gpu-raster/isolines/gpu-isolines.ts:103`). | Coordinator re-read. | A name never changes direction or kind: input `validity`, output `outputValidity`; count views typed `number \| GraphDataView<'uint32'>`, the dual form already used by 9 props. | S |
| A-13 | P2 | consistency | `geospatial/*.ts` (5 imports of `gpu-spatial-analysis/contracts/*`); `gpu-dataframe/group-statistics/*`, `key-join/gpu-key-join.ts:34` (import `cell-aggregation/cell-table`); `gpu-raster/cost-distance/*`, `raster-zonal-statistics`, `raster-pyramid` (import `gpu-terrain/terrain-analysis/*`) | **Layering inversions and cycles.** Shared contracts (`spatial-context`, `earth`) live inside the top-level analysis entry, yet the lower-level geospatial primitives, dataframe and network import them. Dataframe imports spatial-analysis internals (`ORDERED_KEY_WGSL`). The raster subdirs import terrain utilities and type their grid as `GPUTerrainCellSizeMode`. Import cycles exist between spatial-analysis and network, spatial-analysis and dataframe, and raster and terrain. | Coordinator cross-entry import map (`grep -rhoE "from '(\.\./)+(gpu-…)"`). | Move `contracts/`, the cell-key WGSL and grid/cell-size utilities into `utils/` (or an `analysis-core` folder) that every entry imports and that imports none of them. Rename `GPUTerrainCellSizeMode` to `GPURasterCellSizeMode`. | M |
| A-14 | P2 | consistency | `utils/gpu-contributor-types.ts:11,17,29`; `geospatial/gpu-spatial-query-types.ts:14`; `gpu-network/network-spatial-context.ts:20,32,57`; 7 resolvers; 11 `sphereRadius ??` sites | **Parallel twins.** Twins in use: `GPUUint32Rows` = `GPUIdentifierRows` (14 vs 0 users), `GPUFloat32Positions` = `GPUCoordinateRows` (19 vs 0), and `GPUCompactOutput` = `GPUSpatialQueryOutput`, which overlaps the status ports (33 vs 3 users). Duplicated logic: 7 spatial-context resolver variants and 11 sphere-radius defaults, of which the network, trajectory and join copies ignore `units` where `getGPUSphereRadius` honours it. Duplicated alternative types: `GPUPointPatternAlternative` (dataframe) re-declares `GPUInferenceAlternative`, including the `less` alias. | A2 section 2; coordinator spot-checked the utils and contracts types. | One definition per concept. `GPUCompactOutput` is the de-facto port, so make it the contract and delete the unused twins. Have one resolver with a support table, and one radius helper. | S |
| A-15 | P2 | consistency | `gpu-spatial-analysis/contracts/capacity-planning.ts` (608 lines), `contracts/partitioning.ts`, `contracts/ports.ts` | About 30 contract exports have **zero** consumers in src: all 11 `getGPU*CapacityPlan` and `getGPUCapacityRecovery` (used only by contracts specs and one benchmark spec), 8 partition and seam helpers, the Point/Feature/Trajectory geometry ports, and `GPUCardinalityPort` plus 4 status leaves. Docs present them as the capacity-recovery workflow (`gpu-spatial-analysis.md` "Shared ports, status and planning"). | Coordinator: `getGPUSpatialJoinCapacityPlan` appears only in `test/.../spatial-join-benchmark.spec.ts`; `getGPUCapacityRecovery` and `getGPUCellCoverCapacityPlan` have 0 hits. | Rule: a contract is public only with an in-tree consumer. Either wire the capacity plans into the W and join grow loop (baseline recommendation 3) or make them internal. | M |
| A-16 | P2 | consistency | all entries | Errors vs the luma AGENTS.md rule ("minimal runtime validation, simple assertions, explanation in a comment"): **2,848 `throw new Error`, 0 `assert(`, 0 `log.warn`**. That is about 155 KB of message text in bundles; only 13 throws carry the comment; 384 messages lack the `${id}` prefix (mostly packers). There are 62 copies of "must be a single packed view" and 9 hand-copied power-of-two checks. Strength: 74% of throws are in constructors, so validation happens at build time, not silently at dispatch. | Coordinator count 2848 / 0 / 0; A2 section 3 message scan. | Shared validators in `utils/gpu-contributor-utils.ts` (`validatePackedView`, `validateCapacity`, `validatePowerOfTwo`). Convert invariant checks to `assert(cond)` with the explanation in a comment, and keep `throw` only for user-facing prop errors with short messages. | M |
| A-17 | P2 | ergonomics | `geospatial/*.ts` (7 classes `addToGraph`), `geometry-measures/gpu-local-frame-projection.ts:78`; `recipes/*.ts` (12 `add*Recipe`) | **Three construction idioms.** `graph.add(new X())` with `getCommandNodes` (255 classes), `new X().addToGraph(graph)` (geospatial, one spatial-analysis class, and 43 raster-doc snippets), and the `add*Recipe(graph, props)` free functions (spatial-analysis only). No contributor has `setProps`; every non-view prop change is a rebuild. | A1 section 2.1, coordinator grep `addToGraph` in geospatial. | `GPUCommandNodeProducer` only. Port the 8 `addToGraph` classes. If geospatial is upstream-owned, wrap it with an adapter in `utils/`. | S |
| A-18 | P2 | ergonomics | `gpu-network/network-statistics/gpu-network-statistics.ts:334-343`; `terrain-decode/gpu-terrain-spike-repair.ts:302`; `spatial-join/spatial-join-prepared.ts:213`; `gpu-network/network-analysis/gpu-network-analytics-columns.ts:331` | Three contributors own device buffers, allocated at three different times (constructor, first `getCommandNodes`, first use) and with two different "one graph" rules. `GPUNetworkStatistics` derives its device from the default buffer of `props.offsets` rather than from the graph. `GPUNetworkAnalyticsColumns.destroy() {}` is an empty method "kept for API compatibility" in an unpublished package. | Coordinator re-read. | Allocate through graph-owned persistent resources, or a single `GPUOwnedResources` helper taking the device from `graph`. Delete the empty `destroy`. | S |
| A-19 | P2 | ergonomics | `gpu-network/network-accessibility/gpu-network-accessibility.ts:72-123` vs `catchment-accessibility/*` (`GPUCatchmentAccessibility`) | There are two 2SFCA implementations in two entries. One works over a cost matrix (2SFCA only); the other over `GPUSpatialWeights` (2SFCA and 3SFCA). No node turns a network cost matrix or reachability result into `GPUSpatialWeights`, so 3SFCA on network costs is unreachable. This extends the baseline's "catchment belongs in gpu-network". | Coordinator read both docs and the TSDoc. | Keep one 2SFCA/3SFCA kernel on W, and add `GPUNetworkCostWeights` (cost matrix to W with a decay). | M |
| A-20 | P2 | ergonomics | `gpu-terrain/**` Props; `gpu-raster/types.ts:67` `GPURasterMetadata` | Grid georeferencing is spread out: `width`/`height` are props, `cellSize` is repeated inside every terrain contributor's packed settings layout (27 layouts), and `GPURasterMetadata` (width, height, geotransform) is accepted only by `GPUTerrainContours`. Terrain outputs are `GraphDataView`s but inputs are `GPURasterBand`, so chaining Flow into HAND needs a hand-written band literal (`docs/.../gpu-terrain.md` hydrology example). | Coordinator grep (`GPURasterMetadata` used in 1 terrain file). | A `grid: GPURasterMetadata` prop for every raster and terrain contributor, with cell size taken from it rather than from per-contributor settings. Outputs typed as `GPURasterBufferBand`, or a `makeGPURasterBand(view, grid)` helper. | M |
| A-21 | P2 | ergonomics | `gpu-terrain/relief-visualization/gpu-simple-local-relief.ts:68`, `topographic-position/gpu-terrain-vector-ruggedness.ts:86`, `spatial-clustering/gpu-kmeans.ts:77`, 6 classes with `seed: number` | The same concept is compile-time in one place and per-frame in another: `radius` (runtime view in the nearest join and linear referencing, compile-time in lattice W, local relief and vector ruggedness), `tolerance`, and `seed` (rebuild in 6 classes, per-frame in the permutation tests, Ripley and streamlines). The docs' shared "Conventions" block says radii are per-frame, which is false for these terrain classes. | A1 section 2.3; coordinator read the relief Props doc ("Topology: ... radius"). | Rule: algorithm-shape values (kernel extents that size workgroup memory) are compile-time and say so in their names (`maximumRadius`); all other scalars are per-frame. `seed` always lives in the packed parameters. | M |
| A-22 | P2 | docs | 106 exported functions | No exported function has a `###` heading anywhere, and 106 are not mentioned at all. Terrain is worst: 33 of 44 undocumented, including nearly all `getGPUTerrain*ParameterValues`, which are the only way to drive those contributors. | A3 section 1 (full list in `work/A/A3-part1.txt`). | A generated "Parameters" subsection per class listing the packer, length constant and settings fields. | M |
| A-23 | P2 | tests | `test/**` oracles | Oracle provenance: 0 `.py` generators and 1 JSON fixture in `test/`. Network (17 of 18), raster (14 of 14), terrain (22 of 27; GDAL and GRASS "literal mirrors" ported by hand) and dataframe (14 of 18) oracles are hand-written TypeScript with no recorded external values. Spatial-analysis pins 42 of 120 (esda, spreg, sklearn, statsmodels), but only by comment. | A3 section 3. | Commit `test/oracles/*.py` generators (networkx/pandana, scipy.ndimage/rasterio/gdaldem, shapely, h3-py) that write fixtures with package versions. The pysal-venv already exists. | L |
| A-24 | P2 | tests | `test/gpu-spatial-analysis/cell-aggregation/*`; `segment-intersection/planar-operators-spatial-context.node.spec.ts` | `GPUCellRollup` and `GPULineNoding` have no GPU execution spec; they are covered only by constructor-validation node specs. | Coordinator: `grep -rln GPUCellRollup test` gives the oracle and the node spec only. | Add oracle specs. | S |
| A-25 | P3 | tests | `test/.../spatial-regression/ordinary-least-squares.spec.ts:357`; `geographically-weighted-regression.spec.ts:375-376`; `gpu-variogram.spec.ts:187` | Loose recovery checks: the intercept is checked with `toBeCloseTo(2, 0)` (±0.5) while the slope gets 1e-3; GWR accepts a mean slope error of 0.4. The parity tests above them carry the real accuracy, so this is not a hidden regression. The variogram `countDifference <= 4` is unexplained. No tolerance was widened in the working-tree diff. | Coordinator re-read OLS lines 330-358. | Tighten to 1e-2, or delete the redundant recovery assertions. Comment or remove the count slack. | S |
| A-26 | P3 | consistency | `gpu-network/edge-bundling/gpu-edge-bundling.ts:181`; `contracts/inference.ts`, `permutation-alternative.ts`, `pair-statistics/point-pattern-indices-parameters.ts` | Leftover synonyms in an unpublished package: deprecated `geographic?: boolean`; the `'less'`/`'lesser'` alias in three unions; `coordinateSystem`/`sphereRadius` public fields on `GPULinearReferencing`, `GPULineLocate` and `GPUShapeGenerator`; `falseDiscoveryRate` boolean alias on three contributors (in flight with session 32). | A2 1b; coordinator grep line 181. | Delete them; breaking changes are allowed in this package. | S |
| A-27 | P3 | docs | the snippets in `gpu-spatial-analysis.md`, `gpu-network.md`, `gpu-terrain.md`, `gpu-dataframe-analysis.md` | The canonical snippet always encodes with `{parameters: undefined}`, which leaves the graph's typed `Parameters` channel unused (every contributor uses storage views instead), and `compileAsync()` (`gpgpu/.../gpu-command-graph.ts:970`) is never shown. The raster docs use a third run style (`addToGraph` + `device.submit`). | Coordinator grep (one `.compile(` per page, 0 `compileAsync`). | One "Running contributors" page linked from every entry, showing `compileAsync`, encode and submit, and stating why `parameters` is unused. | S |

## 3. Patterns and the one rule that prevents each

| Pattern | Count (grep) | Rule |
|---|---|---|
| Same concept, different name per entry (`parameters`/`settings`, CSR names, breaks, `max`/`maximum`, `sourceCount`) | 115 vs 36; 17 vs 3; 3 break names; 109 vs 215 | Section 4 vocabulary enforced by a node spec that parses `export type GPU*Props` (A1's parser) and rejects non-canonical keys. |
| Same data, several layouts (polygon 3 encodings in 37 types; CSR 3 shapes; positions twins) | 37 types, 18 dirs; 43 files declare `ringOffsets` | Contributor input types are `Port & {...}`; the layout is spelled once in contracts. |
| Contract without consumer | ~30 exports | A public contract needs at least one in-tree contributor consumer, or it is internal. |
| Lower layer imports a higher one | geospatial to spatial-analysis 5; raster to terrain 8; dataframe to spatial-analysis 10 | Entries import only `utils/` (shared contracts move there); a lint rule bans `../gpu-*` imports from geospatial and gpu-raster. |
| Explicit hand-maintained barrels | 1,390 lines; 104 missing names | `export *` per subdir, plus a barrel-diff spec. |
| Throw-with-prose for invariants | 2,848 throws, 0 asserts, 155 KB | `assert` + comment for invariants, shared validators for prop errors. |
| Silent-pass GPU specs | 1,555 guards | One test-device helper that skips visibly and fails under headless. |
| Device limit hidden by the `'max'` test device | 1 kernel at 9, 2 conditional, 46 at 8 | Each contributor's maximal-option spec also runs on `'core'`. |
| Docs drift from signatures | 4 stale snippets, 106 undocumented functions | Type-check doc ```ts blocks in a node spec. |
| Duplicated CPU metadata next to GPU parameters | rate-cluster recipe (3 fields) | Metadata is derived from the same settings object that is packed. |

## 4. Canonical vocabulary proposal (for session 32's "wave 2 naming/port consistency")

The rule order is AGENTS.md first (no abbreviations, verbNoun functions), then the dominant spelling. Counts are library-wide
(spatial-analysis / network / terrain / raster subdirs / dataframe subdirs / geospatial). Rows marked † were added by the coordinator.

| Concept | Canonical | Current variants (count) | Note |
|---|---|---|---|
| Packed per-frame view prop | `parameters` | `parameters` 115 files; `settings` 36 (terrain 31); `wordParameters`/`timeWordParameters` 8 | A second view gets a typed suffix: `uint32Parameters`. |
| CPU settings object | `GPU<Class>Settings` | `*Settings` 58, `*Parameters` types 39 | Settings is what the user types; parameters is what is on the GPU. |
| Packer / length | `getGPU<Class>ParameterValues(settings)`, `GPU_<CLASS>_PARAMETER_LENGTH` | get 117 / encode 3 / create 3 / pack 1; 15 name drifts; 8 `get*ParameterLength` functions | Every parameterised class gets one; shared packers stay as aliases only when shared semantics are real. |
| Outputs | `output: {...}`; status in `output.status` | bag 130 of 259; loose 125; `result(s)` 4; bag+loose 5 | Terrain 34 and network 14 are the bulk. |
| Row count (JS) | derived from view length; else `rowCount` (tables), `featureCount` (geometry), `nodeCount`/`edgeCount`/`vertexCount`/`cellCount`; `width`/`height` for grids | rowCount 67, featureCount 43, `sourceCount` number 6 | `sourceCount` number becomes `rowCount`. |
| Runtime count (GPU) | same noun, typed `number \| GraphDataView<'uint32'>` | `sourceCount` view network 4 / raster 1 | The dual form already exists in 9 props. |
| Compact result | `GPUCompactOutput` (`ids`, `count`, `overflow`, `requiredCount`) promoted to a contract | `GPUCompactOutput` 33 users, `GPUSpatialQueryOutput`, status leaves 0 to 3 | Delete the twins. |
| Bounded status | one-row `overflow` for a single capacity, else `status: GPUBoundedResultStatusPort` | `overflow` 191, `*Overflow` ~34 prefixed, `status` 135 (two shapes) | Prefixed overflow names move into `status.<stage>Overflow`. |
| Invalid / uncertain | `invalidCount`, `uncertainCount` | `invalidEdgeCount`, `invalidSegmentCount`, `invalidRings` | |
| Max / min prefix | `maximum*` / `minimum*` (props and `_MAXIMUM_` constants) | `max*` 109, `maximum*` 215; `min*` 15, `minimum*` 52; `_MAX_` 10 | Coordinate components `minX` etc. are exempt. |
| Iteration cap / done / flag | `maximumIterations` / `iterationCount` (output) / `converged` | `maxIterations` 50, `iterations` 21, `maximumRounds` 6, `maxRounds` 4, `roundCount`, `*Passes` 3 | Phase-specific forms such as `fillIterationCount` are fine. |
| Buffer capacity vs algorithm cap | `<noun>Capacity` (memory) vs `maximum<Noun>Count` (algorithm) | `maximumCapacity`, `maximumOutputCapacity`, `maximumCandidatePairs`; `neighborCapacity`/`maximumNeighbors`/`maximumNeighborCount` for one quantity | |
| Seed | `seed` field in settings (always per-frame) | compile-time `seed: number` 6 classes | Start points become `start*` (`startNodes`, `startCosts`), not `seed*` (20+ collisions). |
| Permutations | `maximumPermutations` (compile-time), `permutationCount` (per-frame and echoed), `simulationCount` (Monte Carlo envelopes) | `permutations` 8 runtime fields | |
| Row selection input | `mask` (nonzero = include); `vertexMask`/`edgeMask`/`nodeMask` when there are two entity tables | `keepMask` 10, `selection` 8, `valueMask` 5, `included` 1 | |
| Selection output | `outputMask` | `keepMask`, `outputMask` 7 | |
| Raster validity | input `validity`, output `outputValidity` | terrain `validity` as output (19) vs raster/spatial-analysis input (9); terrain internals use 5 spellings | |
| Nodata | `noDataValue` (sentinel) + NaN rule | `hasNodata`, `noData`, `noDataPolicy`, `alphaNoData`, `noDataRGB` | |
| Polygon input | `GPUPolygonGeometryPort` 4-level (`positions, ringOffsets, polygonOffsets, featureOffsets`); rows = features | 3 encodings; `polygonPositions` 3; `featureRingOffsets` 5 | Coupled with A-5 (support). |
| Points input | `positions`; qualified `queryPositions`, `nodePositions` | bare `points` 11 | |
| Line input | `positions` + `lineOffsets`; `trackOffsets` for tracks | `pathOffsets` 12, `lineOffsets` 3, `startIndices` 1 | |
| † CSR / network | `offsets, neighbors, weights` (+ `nodePositions`), grouped as a `topology` object assignable to `GPUSpatialWeights` | `edgeTargets`/`edgeCosts` (snapping), loose props (network 17), `GPUGraphTopologyView` | |
| † Cost bands | `costBreaks` (view), `costLimit` | `bandThresholds`, `breaks`, `bandBreaks` | |
| † Raster grid | `grid: GPURasterMetadata` | `width`/`height` props in terrain (34) and raster (14), plus `cellSize` inside 27 settings layouts | |
| † Coordinate model | `spatialContext: GPUSpatialContext`; raster form `GPURasterCellSizeMode` | `GPUTerrainCellSizeMode` (terrain, imported by raster), `GPUPointHorizonProjection`, ontology `coordinateModel`, `geographic?: boolean` | Session 8d owns this; record only. |
| Node-builder functions (internal) | `create*Node(s)` | create 227, get 43 (dataframe 12) | |
| Contributor contract | `getCommandNodes` via `graph.add` | `addToGraph` 8 classes | |
| Abbreviations | `Statistics`, `Information` | `INEQUALITY_STAT(S)`, `getInequalityZoneStatsBody`, `GPUSpatialWeightsInfo`, `..._TOPOLOGY_STATS_LENGTH` | Domain acronyms (GWR, LISA, OLS, AZP) are allowed. |
| Files | kebab-case, one public class per file named after it | 0 kebab violations; mismatches such as `gpu-additional-spatial-statistics.ts` (3 classes), `GPURandomPointsInPolygon` in `gpu-dot-density.ts`, `GPUCellLevelSelection` in `gpu-cell-pyramid.ts` | |

## 5. Roadmap and baseline claims checked

`dev-docs/roadmaps/gpu-spatial-analysis-roadmap.md` "Current implementation status":
- **T1 Shared ports, "directly used by join/gather/statistics, line topology, cells, weights and recipe workflows":
  PARTLY.** The join only aliases `GPUSpatialJoinPairs = GPUCompactPairPort` (`spatial-join-types.ts:102`) and its props stay raw. Gather only validates.
  Weights and cells use the port as a base type that re-declares its fields. `GPUInferenceResultPort` is used in one file.
  **No recipe takes a geometry port.** The Point, Feature and Trajectory ports have 0 users.
- **T2 "consume mandatory `GPUSpatialContext`": FALSE as a rule.** About 99 `spatialContext?` declarations are optional with a planar default.
  The context is mandatory in 3 public Props (geometry measures, geodesic pairs, line density). Trajectory metrics, which the claim names, is optional
  (`gpu-trajectory-metrics.ts:86`), and the cluster-outline recipe has no context prop. **"Old synonyms removed": PARTLY** (A-26).
  "Shared schemas drive ... packers" is true for 6 of about 41 packers.
- **T3 "capacity plans ... report work, output and peak-transient": TRUE as functions, FALSE as adoption.** No contributor or recipe calls
  them (A-15). The doc's capacity-recovery snippet describes a workflow nothing in the library performs.
- **T4 recipe envelope: TRUE** (12 of 12 use `GPURecipeOverrides` and `GPURecipeResult`), but the recipes consume loose geometry and A-6 applies.
- **T8 "Permutation work is shared rather than copied": the engine is shared but the vocabulary is not** (`permutations`, `maximumPermutations` and `permutationCount`; 8 files
  copy `alternative` props).
- **T9 `GPUSolverStatusPort` shared by OLS, error GM and 2SLS: TRUE**, but three convergence-status shapes still coexist.

Baseline `06-architecture.md` items that are **now fixed or improved** (update the baseline):
- `parameter-schema.ts` had "no consumer" and now has 6.
- `GPUMultipleTestingMethod` was "not accepted by any contributor" and is now accepted by 6 contributors and 3 recipes. Local Moran adds holm, BY and Bonferroni through `GPUMultipleTestingAdjustment`.
- `GPUSpatialWeights` was "a bare CSR with no identity". Producers now expose `outputWeights` with `info` (transform, symmetric, self weights, provenance) and `status`, and statistics read `info.transform`. This partly implements baseline recommendation 3.

Baseline items that are **worse library-wide** than stated:
- The output-bag mix: 125 of 259 classes take loose outputs.
- `max*` naming, which is concentrated in network and terrain.
- Port adoption outside spatial-analysis is 0.

## 6. Ergonomics walk-through (summary; full snippets in `work/A/walkthroughs.md`)

The tags in the table:
- **G:** geometry re-spelled
- **A:** manual allocation or capacity guess
- **P:** a pack helper plus a parameter buffer
- **C:** a cast kernel
- **M:** missing node
- **N:** name drift
- **R:** a readback needed to trust the result

| Pipeline | Friction points | Dominant causes |
|---|---|---|
| Points -> tracts -> EB -> queen W -> local Moran (BH) -> colors | 11 (G2 A4 P2 C1 M1 R1), about 16 user-allocated views | Support mismatch (A-5), uint32 -> float32 cast, W capacity guess and grow loop, a packed parameters view required even for defaults; recipes do not chain (A-5, A-6). |
| DEM -> hillshade + flow + streams -> HAND | 5 (P2 G1 N1 R1) | `cellSize` written into two packed layouts; output-view to `GPURasterBand` re-wrap; `settings` and `maxFillIterations` naming. Terrain is the most internally coherent entry (`elevation: GPURasterBand` in every contributor). |
| Roads -> snap -> isochrones -> 2SFCA | 8 (N4 P2 M1 + two graphs to sequence) | CSR re-spelling, three break names, two 2SFCA implementations with no cost-matrix-to-W bridge (A-11, A-19). |

Prior art, briefly:
- **PySAL:** `Moran_Local_Rate(e, b, w, permutations=999)` plus `esda.fdr` is two calls; W is keyed by the dataframe index, so multipart rows cannot mismatch.
- **cuSpatial:** one GeoSeries type for every operation, with library-allocated outputs.
- **turf:** GeoJSON in and out.
- **deck.gl aggregation layers:** accessor props plus `aggregation` and `gpuAggregation`; prop diffing re-runs only the changed stage.

The GPU constraints justify caller-owned buffers and packed per-frame views. They do not justify per-contributor spellings
of geometry, CSR and breaks, per-contributor packed layouts for shared quantities such as cell size, or recipes that cannot
compose.

## 7. Unverified suspicions

1. `geospatial/gpu-point-spatial-query.ts` (raw WGSL, not guarded by `createWGSLKernelNode`) may bind 9 storage buffers in the
   refinement pass (indexed polygon query with source IDs and query values). Confirm by counting the branch combinations near lines 405-495, or
   with a node spec on a fake device limited to 8 buffers. The file is partly upstream.
2. `GPUPolygonize`/`GPUMakeValid.intersectionCapacity`, `GPUCoverageSimplification`/`GPUSegmentIntersection.pairCapacity`
   and `GPULinearReferencing.leafCapacity` (documented as a power of two, forwarded at about line 360) may go unvalidated until a delegate fails with the
   delegate's id. Confirm with node specs that pass 0, 1.5 and NaN.
3. A part-to-feature W collapse might be possible through `GPUSpatialWeightsAlgebra` `block`/`subgraph` operations. I did not find a documented recipe;
   if one exists, A-5 drops to "undocumented".
4. Owned-buffer contributors (A-18) may break when added to a second graph, or when re-added after `compiled.destroy()`. This was
   read statically only.
5. A3's coverage regex missed variable-held class references. Other classes besides `GPUCellRollup` and `GPULineNoding` may also have only
   indirect GPU coverage (for example through recipes).
