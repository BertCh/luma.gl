# GPU Core implementation roadmap

This maintainer roadmap tracks the phased development and graduation of GPU Core. The
[user-facing GPU Core guide](../../docs/api-reference/experimental/gpu-core/README.md)
documents shipped behavior, constraints, and reference pages.

The provider-driven projection contributor uses these command-graph contracts but has a separate
[GPU Project roadmap](./gpu-project-roadmap.md) covering CRS planning, greater-than-Float32
precision, native projection operations, and inline shader composition.

## Phased roadmap

The command-graph foundation and hierarchical-trace, analysis, texture, picking, and spatial
filtering v1 milestones are implemented. The remaining work is ordered by dependency so that later
APIs build on measured, reusable contracts instead of demo-specific behavior.

Impact estimates how broadly a phase unlocks GPU-driven applications. Complexity/cost estimates
relative engineering scope, integration risk, and validation effort; it is not a staffing or
schedule commitment.

| Phase | Outcome | Status | Impact | Complexity/cost |
| --- | --- | :---: | :---: | :---: |
| 0 — Current foundation | Command graph, masks, hierarchy layout, graph traversal, ancestor projection, compaction, indirect drawing, picking, analysis primitives, and three working consumers | Implemented | High | Complete |
| 1 — Hardening and observability | GPU timestamps, performance baselines, adapter capability reporting, boundary and overflow validation, memory statistics, and device-loss and resource-lifetime coverage | Implemented | High | Medium |
| 2 — Reusable visibility workflows | Renderer-independent time-range, bounds, LOD, and selection workflows that publish stable IDs, counts, and indirect commands | Implemented | High | Medium |
| 3 — Algorithm and table scaling | Multi-chunk coverage, segmented and inclusive scans, weighted statistics, richer histograms, and batch-preserving algorithms | Core implemented; topology extensions deferred | High | Large |
| 4 — Picking and texture coverage | Region picking, asynchronous staging rings, multisample resolves, frame-scoped swapchain imports, and sampled-only external-image contracts | Implemented | Medium | Large |
| 5 — Spatial acceleration | `GPUGridIndex` and `GPUBVH` with explicit build, update, query, correctness, and cost-comparison contracts | V1 implemented; conditional extensions deferred | High | Large |
| 6 — GPUScene | A flat GPU draw database with stable identity, bounds, transforms, grouping, geometry references, and indirect command slots | Storage, mutation, source adapters, draw generation, and resource grouping implemented; cross-domain consumers planned | High | Large |
| 7 — API graduation | Stable package contracts and a dependency-safe direct move out of experimental packages | Planned | High | Large |

### Spatial v1 milestones

Phase 4 is implemented through Tranche 4.4. The spatial stack now includes compact 2D/3D
`GPUGridIndex` construction, conservative queries, exact point refinement with an unindexed GPU
oracle, deterministic complete-binary `GPUBVH` storage and refit, and exact BVH bounds/point
traversal. These are foundations, not a claim that incremental grid updates or source-order BVH
topology are always profitable. A shared benchmark harness now rejects incomplete or incorrect
paths before reporting timings and records why optional update, topology, and ray extensions remain
deferred.

| Milestone | Implemented outcome |
| --- | --- |
| 5.1a — `GPUGridIndex` build | Stable cell offsets and capacity-bounded IDs for packed 2D/3D points |
| 5.2a — `GPUGridIndex` query | Point, bounds, and radius cell candidates with masks, counts, and overflow |
| 5.2b — Exact query consumers | Indexed and unindexed 2D/3D point predicates feeding one visibility contract |
| 5.3a — `GPUBVH` storage and refit | Flat complete-binary nodes, stable leaf slots, explicit capacity, and bottom-up refit |
| 5.4a — `GPUBVH` bounds and point query | Exact 2D/3D traversal with stable IDs, masks, count, overflow, and visited-node metrics |
| 5.2c / 5.4c — Benchmark and cost model | Correctness-gated scan/grid/BVH timings, phase distributions, memory, work counters, and reuse amortization |
| 5.1b / 5.3b / 5.4b — Decision gates | Incremental grid updates, topology rebuild, and ray traversal deferred pending consumer semantics and positive evidence |

### Remaining tranche map

The first larger-compute slice can proceed independently of package graduation because it uses the
existing table-independent graph contract. Tranche 8.1a is implemented: `GPUHashIndex` rebuilds a
fixed-capacity sparse `uint32` identity map from one packed batch, `GPUBatchHashIndex` builds the
same shared map from preserved right-side chunks without concatenation, and `GPUHashIndexQuery`
performs bounded lookup against either index with deterministic duplicate values, explicit invalid
and overflow counts, and probe statistics.
Tranche 8.2a is also implemented: `GPUHashJoin` composes lookup and stable scan into bounded
many-to-one row-pair materialization with exact required-count and overflow reporting. It
propagates incomplete source indices rather than presenting partial matches as complete. Tranche
8.3a adds `GPUBatchHashJoin`: ordered left chunks independently query one shared right index and
retain per-batch capacities, counts, overflow, and probe statistics without packing. The current
contracts intentionally stop before deletion, independently partitioned right indices, multi-match
joins, and payload materialization establish their own consumer-driven contracts.

The remaining work is divided into reviewable contracts. A tranche should land only when its entry
dependency is present and its measurable exit evidence can be produced. Numbering groups related
contracts; table order and the recommended sequence express dependency order, not staffing or
schedule commitments. Conditional implementation tranches are entered only when the preceding
decision gate shows that their added memory and complexity pay for themselves in representative
consumers.

| Tranche | Outcome | Entry dependency | Measurable exit | Impact | Cost |
| --- | --- | --- | --- | :---: | :---: |
| 8.1b — Mutable hash maintenance | Deletion and tombstone or rebuild-threshold policy grounded in a dynamic consumer | Implemented bounded `GPUHashIndex` build/query plus measured mutation workload | Lookup parity across deletion and reinsertion; bounded degradation and explicit rebuild trigger | Medium | Medium |
| 8.2b — Sparse grouping and multi-match decision | Decide sparse aggregate rows, one-to-many storage, and payload materialization separately | Implemented bounded many-to-one `GPUHashJoin` plus two requesting consumers | Each expansion contract is accepted with output bounds and CPU-oracle evidence or explicitly deferred | High | Large |
| 8.3b — Partitioned-right decision | Decide paired partitions, key routing, or global addressing for multiple right indices | Implemented shared-right `GPUBatchHashJoin` plus a consumer with partitioned right ownership | One routing contract is accepted with empty/uneven batch evidence or explicitly deferred | High | Large |
| 8.4 — Sparse graph analytics | Frontier/visited representations and graph algorithms selected by demonstrated consumers | Partitioned identity contracts from 3.2 and the 8.3b decision | Bounded CPU-oracle parity on disconnected, cyclic, and high-degree graphs | High | Large |
| 8.5 — Field and solver composition | Reusable graph-native stencil, advection, and solver building blocks behind live simulations | Two existing simulation consumers agree on field and boundary contracts | Shared primitives replace consumer-local kernels without hidden submission | High | Large |
| 3.2 — Partitioned topology | Implemented: global-ID and chunk-base contracts for hierarchy and CSR inputs without hidden packing | Implemented Phase 3 primitives plus a preserved-batch consumer | CPU-oracle parity across empty and uneven chunks; no implicit repack | High | Large |
| 3.3 — Extension decision gate | Implemented: sparse/multidimensional histograms, custom scans, and shader predicates explicitly deferred pending consumer evidence | Tranche 3.2 plus at least two requesting consumers | Each proposal is accepted with a fixed contract or explicitly deferred with evidence | Medium | Small |
| 6.3a — Conventional scene consumer | Implemented: an application-owned CPU scene graph uses shared flat storage, GPU visibility and picking, renderer resource groups, measured mutation, and indirect draw generation | Tranche 6.2b and Phase 4 | Stable application IDs, one compiled graph, explicit update costs, and no consumer-specific fields or CPU draw filtering | High | Medium |
| 6.3b — Table-oriented scene consumer | A preserved-batch table application uses the same runtime contracts | Tranches 6.1c, 6.2b, and 3.2 | Shares public primitives with 6.3a without repacking or adapter casts | High | Medium |
| T.1 — Canonical GPU trace scene | Stable spans, process/thread ownership, preserved source partitions, parents, dependency CSR, and generic scene projection | Implemented `GPUScene`, draw generation, and renderer-owned resource groups | Source identity, empty/uneven batches, bidirectional links, ownership, and scene draw/group integration pass GPU tests | High | Medium |
| T.2 — Interactive GPU trace policies | Implemented: time windows, process/thread expansion, linked-span focus, ancestor retention, and stable indirect draws | Tranche T.1 plus hierarchy, mask, traversal, and visibility workflows | Policy-only updates reuse one graph with GPU-tested stable masks, row IDs, hierarchy offsets, ancestry, and indirect draws | High | Large |
| T.3 — Scene-backed trace showcase | Implemented: a bounded live trace explorer combines canonical trace scenes, GPU interactions, picking, resource groups, stable indirect drawing, and command-graph inspection | Tranche T.2 plus existing picking and graph-inspection contracts | Representative traces pan, filter, collapse, focus, and pick without CPU draw selection or graph recompilation | High | Large |
| 7.1 — Dependency audit and API freeze | Freeze names, ownership, failures, capacities, submission, and package graph | Phase 6 exits and two consumers per graduation candidate | Acyclic dependency report and owner for every public resource boundary | High | Medium |
| 7.2 — Scheduling-core extraction | Extract an engine-independent compute runtime beneath engine in `@luma.gl/gpgpu` | Tranche 7.1 | Graph scheduling no longer imports engine; all repository imports use the final owner | High | Large |
| 7.3 — Adapter and algorithm migration | Keep primitive data and algorithms in `@luma.gl/gpgpu`; layer engine and table/model adapters above it | Tranche 7.2 | Package tests enforce dependency direction and examples use final owners | High | Large |
| 7.4 — Documentation and graduation | Stable docs, release notes, and experimental-removal criteria | Tranche 7.3 | API reports and links pass; obsolete experimental exports are absent | High | Medium |

### Recommended execution order

1. Add the preserved-batch table consumer (6.3b) beside the implemented conventional scene
   consumer (6.3a) to prove the same shared visibility, picking, generated draws, and
   renderer-owned resource groups without packing or consumer-specific scene fields.
2. Preserve the implemented canonical trace-scene, reusable interaction, and live scene-backed
   showcase contracts (T.1–T.3); add larger trace features only when a consumer establishes their
   resource bounds and measurable benefit.
3. Preserve the implemented partitioned hierarchy/CSR topology and explicit extension decision
   gate (3.2–3.3); reopen extensions only when new consumers fix their memory and identity costs.
4. Reopen incremental grid maintenance, spatial BVH rebuild, or ray traversal only when the
   documented decision gate gains a requesting consumer and positive evidence.
5. Graduate packages only after both scene consumers prove the final APIs and dependency direction.
6. Develop the larger compute vocabulary independently where contracts are already bounded:
   single-batch `GPUHashIndex`, preserved-right-batch `GPUBatchHashIndex`, bounded many-to-one
   `GPUHashJoin`, and shared-right `GPUBatchHashJoin` are implemented; require consumers and
   measurements before adding mutable
   maintenance, partitioned-right routing, multi-match joins, sparse graph algorithms, or
   generalized field solvers.

### Phase 0 — Current foundation

**Entry dependencies:** None. This is the implemented baseline.

The baseline includes fixed-capacity buffer and logical-texture scheduling across compute, render,
and copy nodes; hazard inference; imported-resource overrides; graph-owned attachments; transient
reuse; ownership validation; and allocation statistics. Implemented algorithms cover exclusive,
inclusive, and segmented scans; stable compaction; chunk-preserving boolean masks; bounded CSR
traversal; nearest-visible parent projection; paired sort; scalar reduction; histogram counting;
spatial grid binning; and GPU-written indirect commands.

`GPUIndexPickingTarget` provides single-pixel and bounded-region integer object and batch picking;
`GPUReadbackRing` overlaps reusable staging slots without mapped-buffer reuse. The hierarchical
trace viewer adds GPU-scanned process/thread layout, source and topology filtering, dependency
focus, click picking, collapsed activity, projected edges, and stable indirect span and edge groups.
The frustum-culling and GPU data-analysis examples provide two additional consumers.

**Exit criteria:** Achieved by the current exported primitives, reference documentation, CPU
oracles, WebGPU tests, and the three independent examples.

### Phase 1 — Hardening and observability

**Status:** Implemented in the experimental API.

**Entry dependencies:** Phase 0 behavior remains experimental but functionally complete.

Add capability-gated timestamp allocation per graph node without adding readback to normal
encoding. Establish repeatable performance and memory baselines for empty inputs, workgroup
boundaries, maximum example capacities, dense dependency graphs, and repeated parameter-only
updates. Extend adapter diagnostics, capacity and overflow tests, device-loss handling, and
resource-lifetime coverage.

`CompiledGPUCommandGraph.capabilities` reports graph-relevant adapter support and limits. Expanded
`stats` account for imported, logical, and owned transient memory. Every `encode()` returns
synchronous whole-graph and per-node CPU costs; timestamp-enabled encoders additionally support an
explicit post-submit `readTimings()` call. The documented benchmark protocol covers boundary and
maximum example capacities without making readback part of the frame loop.

**Exit criteria:** Achieved by capability reporting, encoding and timestamp diagnostics, expanded
memory statistics, safe-range and adapter-limit validation, device-loss rejection, lifecycle tests,
and the repeatable benchmark protocol.

### Phase 2 — Reusable visibility workflows

**Status:** Implemented in the experimental API.

**Entry dependencies:** Phase 1 establishes measurement, failure, and lifetime contracts.

Standardize graph fragments for bounding spheres, axis-aligned boxes, time ranges, LOD thresholds,
and selection masks. A workflow publishes source-aligned masks, compacted stable IDs, counts, and
optional indirect-command fields rather than a renderer-specific object. Refactor the trace viewer
and frustum-culling example to consume the same workflow contract.

General application-defined WGSL predicates remain deferred until fixed-contract workflows reveal
the necessary shader-extension points.

`GPUVisibilityWorkflow` now accepts source-aligned time-range, bounds, LOD, and selection masks,
intersects them, optionally publishes the canonical mask, generates or consumes stable source IDs,
and writes compacted IDs plus one GPU-resident count. Atomic and multi-chunk vector inputs share
the same contract. The hierarchical trace viewer and frustum-culling example both use the workflow
and send its count directly to indirect rendering; changing view and selection data does not
recompile either graph.

**Exit criteria:** Achieved by two consumers sharing one workflow without application-owned scan or
compaction plumbing, parameter-only interaction updates on compiled graphs, and GPU-resident counts
flowing directly into indirect rendering.

### Phase 3 — Algorithm and table scaling

**Status:** Implemented. Inclusive and segmented `uint32` scans, weighted floating-point grid and
categorical statistics, irregular-edge histograms, filtered categorical counts, batch-preserving
paired sort, and partitioned hierarchy and CSR topology are implemented.

**Entry dependencies:** Phase 2 provides real workflow demand for each added variant.

`GPUScan` now supports inclusive output and nonzero segment-start flags for both atomic data views
and chunk-preserving vectors. Segments continue across chunk boundaries, hierarchical summaries
preserve carry-in values when a later row starts a segment, and all arithmetic retains the
documented modulo-2^32 behavior. The data-analysis example uses inclusive scan for a histogram CDF
and a segmented inclusive scan for per-row spatial-grid prefixes.

`GPUGridAggregation` pairs `float32x2` positions with aligned `float32` weights and computes one
row-major sum, minimum, maximum, or mean per cell. Atomic data views and vectors with identical
chunk topology share the same initialize-once contract. Non-finite positions and weights are
ignored. Sum and mean use compare-exchange addition with explicit nondeterministic `float32`
accumulation order; minimum and maximum use ordered float encodings with native integer atomics.
Empty non-sum cells publish NaN. The data-analysis example validates all four operations over the
same imported Arrow batches.

`GPUHistogram` now supports literal and GPU-resident irregular edges in addition to equal-width
domains. It uses binary search with `[edge[i], edge[i + 1])` intervals and includes the final upper
edge. GPU edges are validated for finite, strictly increasing order by a graph pass, so applications
can update thresholds between encodings without a CPU readback or graph rebuild. Invalid GPU edges
produce zero counts. The data-analysis example switches between uniform and GPU-resident threshold
bins and validates both against a CPU oracle.

`GPUGroupAggregation` maps dense `uint32` identity codes directly to caller-owned group rows. Count
uses a `uint32` output; aligned `float32` values add sum, minimum, maximum, and mean outputs with the
same finite-value and empty-result contracts as grid aggregation. An optional source-aligned mask
lets visibility or selection workflows update categorical distributions and statistics without
downloading selected IDs. Atomic and vector inputs share one contract; vectors preserve aligned
source chunks without packing. Large chunks use bounded three-dimensional dispatches. The
data-analysis example groups the same Arrow rows by quadrant while a selectable value mask changes
both the accepted population and its per-group means.

`GPUBatchSort` applies stable paired `uint32` sorting independently to aligned GPU vector chunks.
It preserves the number, order, and length of source batches, never allocates a hidden packed
copy, and selects bitonic or radix sorting independently for each chunk. This supports streaming
record batches, per-tile ordering, and incremental ingestion where partition boundaries are part
of the storage and lifetime contract. The GPU sort example contrasts that behavior with one
explicit packed global sort and validates independently sorted batches against a CPU oracle.

`GPUSegmentedSort` addresses a different partition contract: many small domains already occupy
shared parent key and payload buffers. Explicit per-domain offsets and lengths retain segment
boundaries while equally sized workgroups share one dispatch. Segments of up to 256 rows require
at most eight width-bucket graph nodes regardless of segment count; gaps remain untouched and no
hidden packing or physical allocation occurs. This is useful for independent mesh-local Morton
orders, while separately allocated streaming chunks remain the domain of `GPUBatchSort`.

More batch-aware operations remain consumer-driven rather than being required to complete this
phase. Custom associative scans, sparse histograms, and multidimensional histograms should be added
only with a concrete consumer and an explicit numerical or memory contract.

#### Tranche 3.2 — Partitioned topology

**Status:** Implemented.

Define how chunk-local rows map to stable global IDs, including explicit base offsets and the
ownership of cross-chunk hierarchy or CSR edges. Extend at least one hierarchy primitive and one
CSR primitive to consume that contract without concatenating chunks behind the caller's back.

`GPUHierarchyLayout` now derives stable cumulative bases independently for parent and child
vectors, splits only the intersecting work when their boundaries differ, and scans offsets across
the preserved child topology. `GPUGraphTraversal` accepts one local CSR allocation per output
partition with global neighbor IDs and routes arbitrary cross-partition edges through explicit
source-to-target passes. The trace viewer exercises both contracts using two logical partitions
backed by its existing allocations.

**Exit evidence:** CPU-oracle tests cover empty chunks, cross-chunk references, uneven boundaries,
and incremental replacement of one batch. A hierarchy and graph consumer preserve their source
partitions while producing the same IDs and results as an explicitly packed input.

#### Tranche 3.3 — Extension decision gate

**Status:** Implemented as an explicit deferral decision.

Evaluate custom associative scans, sparse and multidimensional histograms, and shader predicate
callbacks against demonstrated consumers after partitioned topology lands. Each candidate must
state its numerical behavior, memory-growth bounds, composition model, and why existing fixed
contracts are insufficient. Explicit deferral is a valid outcome; this tranche does not require
inventing an extension API merely to complete a checklist.

| Candidate | Decision | Evidence required to reopen |
| --- | --- | --- |
| Custom associative scans | Defer | Two consumers sharing an associative operation, identity value, overflow behavior, and shader value layout that the fixed `uint32` scan cannot express |
| Sparse histograms | Defer | A high-cardinality consumer where dense output is demonstrably the dominant memory cost, plus bounded key storage and overflow semantics |
| Multidimensional histograms | Defer | Two consumers requiring joint distributions that cannot compose `GPUGridBinning`, `GPUGridAggregation`, or dense group IDs without materializing an avoidable column |
| Application WGSL predicate callbacks | Defer | Two visibility consumers sharing binding, validation, cache-key, diagnostic, and composition requirements beyond the fixed workflow masks |

This decision keeps numerical behavior and shader interfaces inspectable. A future proposal should
name the missing fixed-contract capability and its capacity bound rather than exposing an
unconstrained callback as a shortcut.

**Exit evidence:** Every candidate has two motivating consumers or remains documented as deferred,
and any accepted API has a CPU oracle plus explicit capacity, overflow, and shader-compatibility
contracts.

Irregular histogram edges primarily target heavy-tailed measurements such as trace duration and
request latency. Explicit microsecond-to-second boundaries preserve resolution across orders of
magnitude, align results with service-level thresholds, and let applications compare dynamic
filtered subsets without first generating a log-transformed column. The histogram reference
documents the use case, interval semantics, and update contract.

**Exit criteria:** Each new variant has a deterministic CPU oracle, empty and boundary coverage,
multi-chunk tests where applicable, explicit overflow and floating-point behavior, and at least one
application or renderer consumer.

### Phase 4 — Picking and texture coverage

**Entry dependencies:** Phase 1 defines resource-lifetime and asynchronous readback behavior;
Phase 2 defines stable visible identity.

Region picking, reusable asynchronous staging-buffer rings, multisample resolves, frame-scoped
swapchain imports, and sampled-only external-image imports are implemented. Their access,
ownership, and asynchronous or frame lifetime are explicit.
Callback, highlighting, tooltip, and color-encoded fallback policies remain higher-level workflow
or application concerns.

#### Tranche 4.1 — Region picking

`GPUIndexPickingTarget.addRegionPass()` publishes object IDs, batch IDs, a total result count, and
an overflow flag into caller-sized GPU storage. One covered pixel produces one pair, duplicates are
preserved, and atomic append order is unspecified. Selection semantics such as nearest-only,
toggling, deduplication, or highlighting stay above the primitive.

**Exit evidence:** Tests cover empty regions, overlapping primitives, duplicate IDs, exact capacity,
and overflow. An example uses stable IDs from `GPUVisibilityWorkflow` without a CPU-side identity
translation.

#### Tranche 4.2 — Asynchronous readback ring

`GPUReadbackRing` provides reusable staging tickets with immediate and waiting acquisition paths,
explicit cancellation, safe mapped-buffer reuse, destruction, and device-loss propagation. It owns
staging allocations but neither submits command buffers nor silently waits for a mapped slot.

**Exit evidence:** Repeated region picks can overlap rendering and readback without reusing a
mapped buffer or serializing every frame. Tests cover ring exhaustion, out-of-order completion,
cancellation, destruction, and device loss.

#### Tranche 4.3 — Render-target graph contracts

Graph render attachments now model multisample resolve targets with explicit mip, layer, aspect,
access, format, extent, and sample validation. `importFrameTexture()` requires caller-acquired
swapchain textures to carry one coherent, strictly increasing frame ID per encoding. The graph
validates hazards but never acquires, presents, or destroys a swapchain texture on the
application's behalf.

**Exit evidence:** Compute, render, copy, and resolve nodes order conflicting subresource access;
multisample and swapchain examples encode through the graph; invalid same-pass access,
stale-frame, and ownership mistakes fail before submission.

#### Tranche 4.4 — External-texture contracts

`importExternalTexture()` represents external images as a distinct sampled-only resource rather
than pretending they are ordinary texture storage. Each encoding requires a fresh concrete
binding and a strictly increasing frame ID coherent with every other frame resource. Render nodes
resolve the current snapshot through `getExternalTexture()`; views, storage, copies, attachments,
and graph ownership are deliberately unavailable. Media scheduling, frame acquisition, and
fallback conversion stay outside the graph.

This boundary matters for video, camera, and browser-compositor sources. Their native WebGPU path
can avoid a per-frame copy, but the resulting `texture_external` binding is opaque and short-lived.
Making that lifetime explicit prevents an application from caching yesterday's browser binding or
accidentally routing it through APIs that require reusable texture memory. The video-texture
example demonstrates successive native bindings while retaining an explicit copied WebGL fallback.

**Exit evidence:** Validation prevents persistence into incompatible compiled encodings; common
device-loss checks cover encoding; replacement, cross-resource frame coherence, stale IDs, fresh
binding identity, and borrowed destruction are tested; and the video-texture consumer imports
successive native frames without graph-owned destruction or accidental cross-frame reuse.

**Exit criteria:** Region results preserve stable object and batch identity; repeated picks do not
serialize rendering on mapped buffers; and resolve, swapchain, and external resources participate
in graph validation without accidental ownership or cross-frame reuse.

### Phase 5 — Spatial acceleration

**Entry dependencies:** Phase 1 supplies measurement, Phase 2 supplies reusable visibility, and
Phase 3 supplies the required scan, compaction, and batching behavior.

`GPUGridIndex` was implemented before `GPUBVH` to validate build, update, storage, and query
interfaces on a simpler structure. Grid-index consumers established the shared identity, capacity,
overflow, and measurement contracts reused by BVH traversal. Both are library-built storage-buffer
structures, not native WebGPU acceleration resources, and expose their construction and query
costs.

#### Tranche 5.1 — `GPUGridIndex` build and update

**Status:** Compact full-build storage is implemented. Incremental maintenance is deferred pending
a moving-data consumer and a positive measured update-policy decision.

Build a flat index of cell offsets and stable object IDs from bounded positions or bounds. Expose
capacity and overflow, distinguish full rebuilds from supported incremental updates, and keep cell
size and domain policy caller-controlled.

`GPUGridIndex` builds 2D `float32x2` or 3D `float32x3` point inputs into exclusive row-major cell
offsets and capacity-bounded stable IDs. It accepts one packed view or preserved vector chunks,
generates logical IDs or consumes aligned explicit IDs, and reports the full accepted count plus
overflow without writing past capacity. Exact maximum coordinates enter the final cell; non-finite
and out-of-domain rows are ignored.

The current update policy is explicitly `'rebuild'`: callers may upload a bounded input range or
replace one vector chunk, but the next encoding clears, scans, and scatters the complete compact
index. The 5.1b decision gate deferred bounded relocation and reserved-cell designs because no
moving-data consumer yet demonstrates a positive crossover after memory overhead, fragmentation,
adversarial movement, and overflow behavior. Tranche 5.1c reopens only if that evidence justifies an
incremental design; `'rebuild'` is the complete v1 policy.

**Implemented evidence:** Two-dimensional and three-dimensional builds match CPU oracles across
empty, clustered, out-of-domain, and capacity-boundary inputs.

**Decision:** Full rebuild is the supported v1 update contract. The benchmark reports its build cost
separately from query cost. Bounded relocation or reserved-cell maintenance reopens only when a
moving-data consumer demonstrates a crossover that repays added memory, fragmentation, and
overflow complexity.

#### Tranche 5.2 — `GPUGridIndex` query

**Status:** Conservative queries, exact 2D/3D point-refinement consumers, and the shared benchmark
contract are implemented.

Add bounds, radius, and point queries whose masks or compacted IDs compose directly with visibility
and region-picking outputs. Query contracts preserve stable identity and do not require downloading
candidate lists before filtering or drawing.

`GPUGridIndexQuery` consumes the flat grid storage and a mutable GPU-resident point, bounds, or
radius query. It publishes capacity-bounded stable candidate IDs, the stored-prefix candidate count,
propagated index or output overflow, and an optional source-ID-addressed mask. Point queries select
one cell; bounds and radius queries conservatively select intersecting cells. Exact object tests are
deliberately a following application or visibility predicate, so the index does not embed one object
shape or confuse cell overlap with an exact hit.

`GPUPointSpatialFilter` supplies a fixed-contract exact predicate for packed points. It runs over
either every source row or compact candidate row IDs and publishes the same source-aligned mask in
both modes. Two- and three-dimensional tests feed the exact mask into `GPUVisibilityWorkflow`,
intersect it with selection, and compare indexed results with an unindexed GPU scan after dynamic
query changes. Candidate overflow remains visible because a refined result cannot be complete when
its broad phase was truncated.

Tranche 5.2c turns the indexed and unindexed paths into one repeatable benchmark harness. The harness
uses identical data and queries, validates exact result parity, rejects overflow, and reports
distributions rather than a single favorable sample.

**Implemented evidence:** `runGPUSpatialQueryBenchmark` reports build, query, exact-predicate, memory,
candidate, and reuse-amortization metrics with optional GPU timestamps. Representative consumers
still choose and publish their own crossover; the library does not encode one adapter-specific
threshold as policy.

#### Tranche 5.3 — `GPUBVH` build and refit

**Status:** Flat complete-binary storage, deterministic GPU refit, exact traversal, and
topology-quality measurement are implemented. Spatial topology rebuild is deferred pending positive
consumer evidence.

Define flat node and leaf storage, stable leaf identity, bounds encoding, and explicit rebuild and
refit policies. Reuse the grid index's ownership and measurement conventions where possible while
allowing BVH-specific topology.

`GPUBVH` reserves a power-of-two leaf capacity and publishes `2 * leafCapacity - 1` row-major node
bounds and child pairs plus stable leaf IDs. Source order defines leaf slots. Each encoding reloads
the bounded source prefix and reduces parent bounds bottom-up, so changing bounds refits without
graph recompilation or identity changes. Hierarchies of up to 128 leaves fuse the complete build
into one workgroup; larger hierarchies retain explicit, safely ordered level passes. Optional
caller-supplied source identifiers are published without exceeding the default eight-buffer WebGPU
CORE limit. Count, overflow, topology, update policy, level count, and caller-owned output bytes
remain explicit.

`GPUSegmentedBVH` applies that same complete-binary contract to many independent hierarchies
already packed into shared source and destination buffers. It groups trees containing up to 128
leaves by leaf capacity and dispatches one workgroup per tree, so arbitrarily many same-sized mesh
BLASes need one graph node and mixed sizes need at most eight nodes. Packed offsets, invalid leaves,
overflow reporting, two- or three-dimensional bounds, and the eight-storage-buffer CORE limit stay
explicit.

The complete source-order topology is a correctness and refit baseline, not a promised spatial
quality heuristic. Tiled, Morton-ordered, or producer-sorted inputs may already have locality;
arbitrary order may create overlapping parents and poor traversal. The 5.3b decision gate uses
visited nodes, candidate ratios, and build/refit phases to compare source, producer, and externally
preordered input. Tranche 5.3c reopens only after representative measurements demonstrate that a
library spatial builder repays its sorting and topology cost.

**Implemented query evidence:** `GPUBVHQuery` traverses complete-binary 2D/3D hierarchies for exact
point containment and bounds intersection. CPU-oracle tests cover selective pruning, overlap,
invalid queries, output overflow, mutable queries, and source-ID-addressed masks. `visitedCount`
reports topology work independently from matches.

**Decision:** `visitedCount` and phase timings report topology quality separately from storage and
refit cost. A Morton or other topology builder reopens only when representative source, producer,
and preordered inputs demonstrate a material end-to-end win after sorting/build cost. Any future
builder must preserve query equivalence with refit.

#### Tranche 5.4 — `GPUBVH` query and cost model

**Status:** Bounds/point query and the Phase 5 selection cost model are implemented. Ray-like
traversal is deferred as a separate consumer-defined extension.

Tranche 5.4a is implemented. `GPUBVHQuery` connects exact bounds and point traversal to the same
bounded candidate, mask, count, and overflow contracts used by grid queries. It intentionally
precedes topology optimization: `visitedCount` measures whether a new topology improves useful
work.

Tranche 5.4b remains deferred until a picking or simulation consumer fixes ray/segment intersection,
nearest-hit behavior, and bounded traversal semantics. Traversal stack or work-queue capacity must
be explicit and overflow must never look like an empty hit set.

Tranche 5.4c publishes selection guidance rather than claiming one index is universally best. It
includes conditional incremental-grid or spatial-BVH builders only if their decision gates pass.

**Exit evidence:** The shared harness compares unindexed scan, grid, and BVH paths with the same
inputs and correctness oracle, including build amortization, selectivity, memory, candidates,
topology quality, visited nodes, and query time. Update rates remain consumer inputs rather than a
hard-coded library threshold.

**Exit criteria:** Achieved by 2D/3D grid and BVH build/query tests, exact point-filter integration,
visibility composition, stable output contracts, and the correctness-gated cost model. Picking can
consume the same stable-ID masks; ray traversal is not required for spatial filtering v1.

### Phase 6 — GPUScene

**Entry dependencies:** Phase 2 provides visibility output, Phase 4 provides interaction and
texture resource contracts, and Phase 5 provides spatial queries.

Define `GPUScene` as a flat draw database containing stable object IDs, bounds, transforms, group
membership, geometry references, and indirect command slots. CPU scene graphs may update this
database, and table-oriented applications may construct it directly; `GPUScene` does not introduce
a second game-engine hierarchy.

#### Tranche 6.1 — Scene storage and updates

Specify flat draw records, stable IDs, bounded update ranges, group membership, geometry references,
and command-slot ownership. Provide explicit adapters for CPU scene graphs and GPU tables without
making either representation canonical.

The table-independent record contract (6.1a) is implemented as a fixed 128-byte interleaved record
with explicit stable identity, references, bounds, transforms, state, capacity, ownership, and
typed graph views. Transactional CPU-authored insert, patch, removal, stable compaction, overflow,
and exact upload-cost reporting complete 6.1b. Tranche 6.1c adds two explicit source boundaries:
stable preorder callbacks flatten an application-owned CPU hierarchy into normal mutable records,
while canonical interleaved `GPUTable` batches are borrowed as ordered scene partitions without
readback, concatenation, or hidden packing. Empty batches retain their partition slots and global
record bases. Independent buffer ownership lets each adapted scene release its state block while
leaving table record storage with the table.

**Exit evidence:** Insert, update, removal, and compaction tests preserve identity and references;
partial updates have measurable upload bounds; no scene hierarchy or table type enters the core
storage contract.

#### Tranche 6.2 — GPU draw generation

Translate visibility and spatial-query results into capacity-bounded indirect-command slots grouped
by compatible pipeline and resource bindings. WebGPU binding constraints remain explicit rather
than being presented as a bindless renderer.

Tranche 6.2a is implemented by `GPUSceneDrawGeneration`. Active, optionally visible scene rows
claim explicit fixed-capacity indirect-command slots; the lowest scene row deterministically wins a
collision. The graph clears and publishes only instance count and first instance, preserving
renderer-authored geometry arguments. Required and published counts plus overflow distinguish
complete, colliding, and out-of-range results without CPU draw selection, hidden allocation,
submission, or readback. Tranche 6.2b is implemented by `GPUSceneResourceGroups`: immutable
renderer-owned group IDs and command windows preserve pipeline/binding order while generated draw
membership, empty groups, geometry mismatches, misplaced slots, unknown groups, and per-group
overflow remain GPU-resident and observable. Re-encoding after scene mutation reclassifies groups
without claiming bindless WebGPU behavior or hiding resource binding policy.

**Exit evidence:** A compiled graph updates counts and commands after parameter-only changes with
no CPU draw selection. Tests cover empty groups, stable ordering, capacity overflow, and geometry
or material group changes.

#### Tranche 6.3 — Cross-domain scene consumers

Prove that the storage contract serves both a conventional scene graph and a table-oriented
application. Both consumers use the same identity, visibility, picking, and indirect-draw path
while retaining their own update and presentation policies.

Tranche 6.3a is implemented by the live GPU Scene Graph Explorer. An application-owned hierarchy
is flattened through `makeGPUSceneFromCPUScene`; GPU bounds visibility, stable-row compaction,
source-indexed indirect draw generation, renderer-owned resource windows, visibility-aware
picking, and explicit mutation costs share one compiled command graph. The hierarchy stays on the
CPU, stable application IDs differ from physical scene slots, and group diagnostics never drive
CPU draw selection.

The preserved-batch table consumer follows as 6.3b. Phase 6 does not exit until that second
independent consumer uses the same public runtime contracts without casts, hidden packing, or
consumer-specific record fields.

**Exit evidence:** Two independent consumers share the public scene primitives without adapter
casts, hidden packing, or consumer-specific fields in `GPUScene`.

**Exit criteria:** Incremental updates preserve stable identity, visibility and spatial-query
results write draw commands without CPU draw selection, and both scene-graph and table-oriented
consumers use the same storage contract.

### Phase 7 — API graduation

**Entry dependencies:** Phases 1–6 have stable failure, ownership, extension, and package-boundary
contracts, and every candidate abstraction has at least two independent consumers.

Extract the table-independent scheduling core beneath engine in `@luma.gl/gpgpu`, keep primitive
GPU data, graph structures, algorithms, and reusable workflows there, and layer engine resource
adapters plus private table/model integrations above it. Keep Arrow conversion and readback adapters
in `@luma.gl/arrow`. Audit `DrawCommandBuffer` and split its core from rendering integration if that
is required to avoid a gpgpu dependency on engine. These APIs are new and experimental, so
graduation is a direct move: update
repository consumers atomically and do not retain compatibility exports, duplicate public paths,
or a deprecation window.

#### Tranche 7.1 — Dependency audit and API freeze

Freeze ownership, naming, submission, lifetime, failure, capacity, and extension contracts only
after every graduation candidate has at least two consumers. Produce the target package graph and
identify every repository import that must move atomically with the implementation.

**Exit evidence:** The audit demonstrates an acyclic package graph, no Arrow leakage into gpgpu or
generic table layers, no engine dependency in graph scheduling, and an owner for every resource and command
submission boundary.

#### Tranche 7.2 — Scheduling-core extraction

Extract the table-independent command-graph runtime beneath engine in `@luma.gl/gpgpu`, limited to
buffers, textures, passes, generic graph views, scheduling, hazards, and allocation. Move engine
resource and model adapters above that runtime, and remove direct engine imports from graph
scheduling.

**Exit evidence:** Graph scheduling builds without engine, experimental tables, or Arrow; all
repository consumers import the final gpgpu owner; engine consumers use explicit adapters; no
compatibility export preserves the former path.

#### Tranche 7.3 — Adapter and algorithm migration

Keep primitive GPU data, graph structures, optional algorithms, and reusable workflow builders in
`@luma.gl/gpgpu`; keep higher-level tables and rendering models in private experimental subpaths;
and retain Arrow conversion, upload, and readback helpers in `@luma.gl/arrow`. Split
`DrawCommandBuffer` rendering integration if necessary to preserve that direction.

**Exit evidence:** Package-level tests and dependency checks enforce the intended arrows, public
examples import from their final owners, and no algorithm or adapter remains exported by both its
old and final packages.

#### Tranche 7.4 — Documentation and graduation

Publish stable reference pages, release notes, and removal criteria for the experimental surface.
Treat candidate names as provisional until this tranche exits. Because this is a new experimental
surface, describe final package ownership without promising compatibility aliases.

**Exit evidence:** All examples and tests use graduated entry points, links and API reports pass,
the former experimental entry points are absent, and experimental removal has an explicit release
boundary.

**Exit criteria:** The final package graph has no dependency cycle or Arrow leakage into tables or
gpgpu; each API has one public package owner and no compatibility export; public API documentation
names ownership and submission responsibilities; and all existing consumers build against the
graduated packages.

## Analysis contributors (initial versions)

Status: initial, experimental. Analysis contributors ship prebuilt command nodes for common map
tasks, composed from the public primitives above, in the domain entry points
`@luma.gl/experimental/gpu-terrain`, `gpu-network`, `gpu-spatial-analysis`, `gpu-raster`, `gpu-dataframe`,
`gpu-tables`, and `gpu-crossfilter`. Each contributor is a `GPUCommandNodeProducer` class with typed
graph-view inputs and outputs, per-frame parameters in storage views (no recompile), and GPU-side
capacity and overflow reporting. Source lives under `modules/experimental/src/<entry>/<directory>/`;
the directory names below are relative to the entry that exports them. Shared helpers are in
`modules/experimental/src/utils/`. User docs:
[GPU Terrain](../../docs/api-reference/experimental/gpu-terrain.md),
[GPU Network](../../docs/api-reference/experimental/gpu-network.md),
[GPU Spatial Analysis](../../docs/api-reference/experimental/gpu-spatial-analysis.md),
[GPURaster analysis contributors](../../docs/api-reference/experimental/gpu-raster/operations-analysis.md),
[GPU Dataframe analysis contributors](../../docs/api-reference/experimental/gpu-dataframe-analysis.md), and
[GPU residency](../../docs/api-reference/experimental/gpu-tables/gpu-residency.mdx).

Round 8 (DEM analysis ported from mt-image and Rigi): terrain decode and spike repair, summits and critical points, curvature, geomorphons, TPI/TRI/Weiss, hydrology extras and the flow field, and pyramid-accelerated visibility. Every contributor is compared against an f64 or CPU oracle; each has a headless GPU spec and a node spec.

Implemented (initial versions, node and headless WebGPU tests):

- **Viewport tile and LOD selection** — `GPUTileLODSelection`: SSE refinement with foveation and
  distance falloff, deterministic bucketed cost/count budget, residency stand-ins, requests with
  priorities, indirect draw and dispatch; parity-tested against `GPUVirtualGeometrySelection`
- **Point density and binning** — `GPUPointDensity`: grid and hexagon bins, count/sum/mean, smoothing,
  extent, histogram, `r32float` texture; optional per-row `uint32` `mask` (nonzero includes) that
  excludes rows from counts, sums, means, smoothing, extent and histogram with no recompile on mask
  change. Hexagons read it in the keys kernel; grid binning adds one masked-copy pass
- **Spatial join** — `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin` over a per-encoding `GPUBVH`.
  `GPUPairwisePointInPolygon` decides edges strictly above, below, or to one side of the point from
  exact signs, so far near-collinear edges no longer make a point `uncertain`: on the explorer's SF
  bike parking × ZIP data, 24 unassigned points (33 uncertain pairs) → 2,520 of 2,520 assigned,
  0 uncertain. Uncertain pairs are never assigned and are counted in `uncertainCount`
- **Terrain analysis** — `GPUTerrainDerivatives`, `GPUTerrainContours`, `GPUTerrainViewshed`;
  contour draw records take `drawLayout` (`'instanced'` with `verticesPerInstance`, or
  `'line-list'`)
- **Time-windowed filtering** — `GPUTimeWindowFilter`, including trail clipping, double-single
  epochs, and exact Int64 word timestamps (Arrow Int64 epoch ms without a CPU pass, playhead words
  plus an f32 fraction, exact borrow-subtract). `makeArrowTemporalWordGPUVector` in `@luma.gl/arrow`
  uploads Arrow `Int64`, `Timestamp`, `Date(ms)` and `Duration` batches as is; `Int64` fields with
  `visgl:temporal-*` metadata are recognized. Tested across a low-word wrap, epochs near 1.7e12 ms
  and before 1970, and fractional playheads against BigInt oracles
- **Network reachability** — `GPUNetworkReachability`: chaotic Bellman-Ford over a compact frontier
  queue with a round-stamped visited set (8.4-style representations), per-round indirect dispatch
  with no gate nodes, workgroup-local multi-hop (`localIterations`, default 16), bands, cycle-safe
  predecessors (strict-then-tie-level rule, `maxTieIterations`), and a convergence flag. 224×224
  grid: 1,283 → 55 graph nodes and 12.8 → 0.8 ms CPU encode at equal converged costs (31 rounds
  instead of 457). Predecessors cost +9 nodes and +0.2 ms encode. Path extraction now crosses
  zero-weight edges and cycles
- **Region picking and lasso statistics** — `GPURegionStatistics`, `GPURegionMask`,
  `GPUPickRegionMask`, `GPURegionStatisticsReadback`; optional `drawInstanceCount`
- **Zonal statistics** — `GPUZonalStatistics`: choropleth count, sum, weighted mean, min, max,
  density, and extent over a point-in-polygon join; atomic sums or bitwise-reproducible
  `sumOrder: 'sorted'`; `output.uncertainCount` reports unprovable pairs
- **Buffer selection** — `GPUBufferSelection`: points within a per-frame distance of point or
  segment features, as a mask and ascending stable IDs
- **Spatial (Morton) feature sorting for joins** — `spatialSort`; M3 Pro, shuffled features:
  point-in-polygon 14.4k cells × 250k points 218 → 6.7 ms, nearest feature 50k segments × 100k
  points 309 → 4.1 ms; no gain on coherent features, so off by default
- **Grid-index candidates for region statistics** — `GPURegionStatistics.spatialIndex`: statistics
  only, 1.2–1.4x at 1M points and 3.7–4.3x at 4M points for 0.01–1% selections
- **Network analysis** — `GPUNetworkPathExtraction` (bounded predecessor walks to compact routes and
  edges), `GPUNetworkServiceAreas` (nearest facility, smallest-row ties, counts, served cost),
  `GPUNetworkNeighborhood` (k-hop ego networks), `GPUNetworkAnalyticsColumns` (gpu-graph degree,
  PageRank, components, core number, communities as normalized styling columns). Analytics run on a
  `GPUGraphTopologyView`, so transient CSRs from `GPUCOOToCSR` can be analyzed
- **View-based gpu-graph topology** — `GPUGraphTopologyView` for degree, PageRank, core number,
  components, and label propagation over in-graph CSR views; existing `GPUGraphTopology` callers
  unchanged
- **Network statistics** — `GPUNetworkStatistics`: masked live counts, weak components (gpu-graph CC
  on a masked adjacency), degree histograms (linear or log2) and deterministic integer-accumulated
  modularity in one summary buffer. 100k vertices, 400k slots: 12.6 ms median
- **Edge bundling** — `GPUEdgeBundling`: KDEEB as compute, fixed-point `atomic<u32>` density buffer
  (no float-renderable texture or size cap), GPU work box, per-frame radius, lambda, smoothing and
  active iterations; outputs `paths`, `startIndices` and a `drawIndirect` record. 10k edges × 16
  points × 15 iterations at 256²: 9.8 ms, 50 nodes
- **Attribute crossfilter** — `GPUAttributeCrossfilter`: linked histograms over separate column
  views, own-brush exclusion, live-row `'auto'` domains, selection and compact IDs. 524,288 rows × 4
  dimensions × 64 bins: brush to readback 2.8 ms median, 4.4 ms p95
- **Residency arena for tiled and streamed rows** — `GPUResidencyArena` (one fixed buffer per
  column, live mask, row tile slots, page allocator with non-contiguous pages, insert/evict/replace
  with no allocation) and `GPUResidentRowSelection` (live mask, per-tile gate and predicates to
  bounded IDs and an indirect count). M3 Pro, 1,048,576 rows: 10 nodes at any tile or chunk count,
  against 1,249 for `GPUCompaction` over a 32-chunk vector; about 0.8 ms per queued encoding against
  120 ms. `GPUTimeWindowFilter`, `GPUPointDensity` and `GPURegionStatistics` run over arena columns
  unchanged; a negative control reproduces poopdeck.gl's dead-row histogram-domain failure
- **Hydrology** — `GPUTerrainFlow`: Planchon–Darboux fill, D8 directions with flat/pit/outlet
  classes, deterministic accumulation (cells, area, runoff), stream mask; outputs may share one
  buffer over disjoint byte ranges. Round 8 adds flat resolution (Barnes 2014, three GPU-gated
  equal-height relaxations, bit-exact against a BFS oracle), D-infinity and Freeman/Quinn MFD
  accumulation (deterministic pull, fractions recomputed per donor), and the derived contributors
  `GPUTerrainHeightAboveDrainage` (HAND) and `GPUTerrainWatersheds` (pointer jumping),
  `GPUTerrainStreamOrder` (Strahler) and `GPUTerrainHydrologicIndices` (SCA, TWI, SPI).
  `GPUTerrainFlowField` builds a DEM-deflected wind field for particle advection, streamlines and
  LIC (from mt-image). Apple silicon via Dawn/Metal, 512² synthetic DEM, fill ε 0, flats, area
  accumulation, HAND, basins, Strahler, TWI and wind field in one graph: every loop converged;
  no flats 59 ms; flat limit 32 71 ms, limit 128 91 ms; limit 512 212 ms (D8), 228 ms (D-inf),
  237 ms (MFD)
- **DEM analysis, round 8 (ported from mt-image / Rigi)**
  - `GPUTerrainRGBDecode` and `GPUTerrainSpikeRepair` (`terrain-decode/`): Terrarium and Mapbox
    terrain-RGB decode from a texture or packed buffer with nodata via alpha, RGB, validity or
    range (exhaustive 2^24 GPU tests bit-exact on both input paths and encodings); opt-in ±256 m
    spike repair on `GPUGraphConnectedComponents` (10-14 component rounds, up to a 513² serpentine)
  - `GPUTerrainSummits`, `GPUTerrainPeakSnap`, `GPUTerrainCriticalPoints` and `GPUProfilePeaks`
    (`terrain-features/`): disc summits with a ring-drop prominence bound, generic peak snapping,
    8- or 6-neighbour critical points (the 6-ring gives χ = 0 exactly on a torus egg-crate), and 1-D
    profile peaks with parallel exact greedy NMS and a wrap mode
  - `GPUTerrainCurvature` (`terrain-curvature/`): 15 Florinsky curvatures from Evans–Young,
    Zevenbergen–Thorne or Florinsky 5×5 partials (Whitebox signs), plus mt-image multi-radius ring
    curvature; max abs error 4.2e-6 of output scale versus f64
  - `GPUGeomorphons` (`geomorphons/`): GRASS r.geomorphon parity (3 comparison modes, flat
    distance, skip radius, forms, raw and 498-class ternary codes); 0 mismatches versus an f64
    mirror; 2048², L = 20: 15 ms
  - `GPUTerrainRuggedness`, `GPUTerrainVectorRuggedness`, `GPUTerrainTopographicPosition` and
    `GPUTerrainWeissLandforms` (`topographic-position/`): gdaldem TPI/TRI/roughness with both edge
    modes, VRM, multiscale TPI/DEV/DEVmax from an exact integer summed-area table (f32 SAT: 5.4 m
    error, exact table: 2.2 mm; 2048², 3 scales: 31 ms), Weiss 10 landform classes
- **Visibility, round 8 (ported from mt-image / Rigi)**
  - `GPURasterExtremaPyramid` (`raster-pyramid/`): exact min/max mip chain, bilinear-footprint option, validity-aware empty sentinels; bit-identical to the CPU build
  - `GPUPointHorizonProfile` and `GPUPointHorizonVisibility` (`point-horizon/`): planar or Web Mercator piecewise great circle, exact power-of-two octave distance lattice, mt-image f32 precision scheme, curvature and refraction, pyramid skipping bit-identical to the march, tolerance-band peak classification (elevation error at most 3.3e-5 degrees against f64)
  - `GPUTerrainViewshed` gains `traversal: 'pyramid'` (bit-identical) and a tolerance band with an ignored final stretch (code `marginal: 4`); new `GPUTerrainLineOfSight` (batched pairs, clearance) and `GPUTerrainCumulativeViewshed` (observers as the dispatch dimension). Curvature conventions documented (GDAL `cc = 0.85714`, k = 1/7, against k = 0.13)
  - Measured (Apple GPU, Dawn, median of 5, with readback): viewshed 512² march 9.3 ms, pyramid 8.5 ms; 1024² 35.5 / 33.3 ms; 2048² 171 / 175 ms (about 70 ms fixed). Point horizon 1024²: 16 observers × 720 azimuths cell-30 m step 6 ms march, 10 ms pyramid (38 % of samples); 128 × 720 cell-2 m step 65 ms march, 39 ms pyramid (23 %). The pyramid wins only at high ray counts with a dense step rule
- **Relief and illumination, round 8 (ported from mt-image / Rigi / RVT)**
  - `GPUTerrainHorizon` `algorithm: 'sweep'`: exact upper-hull digital-line sweep, amortised O(1) per pixel and sector; 1024² × 16 sectors: 98.6 ms against 411 ms for the march at radius 256, 40.6 ms against 1474 ms at full radius
  - `horizonFormat: 'unorm16'` half-size horizon maps (1-code parity); `negativeOpenness` and anisotropic SVF (RVT) outputs
  - `GPUSolarIrradiance` (sun hours and insolation from the horizon map and `getSolarPosition` sun tables; 29 ms per 288-sample day at 1024²) and `GPUTerrainCastShadow` (single-sun exact sweep shadow without a horizon map, per-frame sun)
  - `GPUReliefShading`: Imhof aspect swing, curvature raster input and elevation contrast, bit-identical when off
  - `GPUSimpleLocalRelief`, `GPUMultiScaleRelief`, `GPULocalDominance` and `GPUReliefBlend` (`relief-visualization/`; SLRM, telescoped MSRM, local dominance, VAT blend with presets) against RVT-formula f64 oracles; 1024²: SLRM 3.3 ms, MSRM radius 100 5.4 ms, local dominance 84.5 ms, 4-layer blend 1.2 ms
  - Precision: the horizon march now interpolates centre-relative heights; off-axis error at 4000 m fell from 1.5e-3° to 3.5e-5°. The inventory's fast-math division claim was tested and not reproduced
- **Cost distance** — `GPUCostDistance` (tiled GPU-gated relaxation, cost limit, bands, back-links)
  and `GPUCostDistancePath`; generated-transient collisions name the contributor; back-links use the
  reachability tie rule, so paths through zero-friction plateaus reach a source (`maxTieIterations`)
- **Raster zonal statistics** — `GPURasterZonalStatistics` over dense zone rasters with overflow;
  bitwise-reproducible `sumOrder: 'sorted'`
- **Origin–destination flows** — `GPUFlowAggregation`: grid, hexagon, or caller zones; hashed pair
  aggregation with GPU overflow; deterministic weight-ranked top-K; zone in/out totals; time-window
  or mask gate; Int64 word time gate; bitwise-reproducible `sumOrder: 'sorted'` (22 → 38 nodes;
  1M rows × 64 zones 34.5 ms atomic vs 25.2 ms sorted, headless Chromium)
- **Density clustering** — `GPUSpatialClustering`: DBSCAN with per-frame epsilon on an adaptive
  lattice, lock-free union-find, canonical labels matching a CPU oracle exactly;
  bitwise-reproducible centroids with `sumOrder: 'sorted'` (30k points: ~25 ms per submit plus
  readback either way, dominated by DBSCAN)
- **Trajectory metrics** — `GPUTrajectoryMetrics`: per-track length, duration, speeds, and stop
  detection with a bounded stop list; f32, double-single, or exact Int64 word timestamps
- **Temporal reduction** — `GPUTemporalReduction`: count, min, max, first and last per (cell,
  coarse time bucket) with u32 atomics only (first/last by time, then lowest row), f32 or exact
  Int64 word times, and an ascending occupied-slot compact list as the scrub draw source. f32
  bucket edges use correctly rounded products, not WGSL division (2.5 ULP), so assignment matches
  the CPU oracle on every adapter. CPU oracle, 1M skewed rows: 6 h buckets draw 4.4x and 24 h
  buckets 16.7x fewer features than rows (1 h: 1.6x)
- **Network subgraph filter** — `GPUNetworkSubgraphFilter`: half-open f32 vertex and edge ranges,
  f32 or exact Int64 edge time windows and caller masks to symmetric vertex and edge masks
  (undirected slot pairing), `dropIsolated`, counts, compact live IDs and an induced CSR, all
  exact and per-frame without recompiling
- **Network coarsening** — `GPUNetworkCoarsening`: dense group labels to supernode counts,
  centroids, bounds and value sums plus a bounded, sorted superedge list (stable sort and segment,
  not a hash table, so the kept prefix on overflow is deterministic); 64-bit fixed-point sums are
  bitwise reproducible. 975k slots: about 15 ms per encode with readback at 1,024 groups, 25 ms at
  60,000
- **Adjacency matrix** — `GPUAdjacencyMatrix`: ordered `R×R` counts, fixed-point weight sums,
  maxima and an optional `r32float` texture over a per-frame zoom window; `GPUAdjacencyMatrixOrder`
  builds the order from group labels with two stable `GPUSort` passes and matches
  `computeAdjacencyMatrixOrder` exactly
- **Deterministic sums** — one shared sorted segmented sum, `getSortedSegmentSumNodes`
  (`utils/sorted-segment-sums.ts`, stable radix sort, scan, gather, fixed 256-wide tree), used by zonal,
  raster zonal, flow aggregation and clustering
- **Kernel helper** — `createWGSLKernelNode` keeps every declared binding in the auto layout
  with a zero-cost phony reference (`_ = &binding;`) and omits an unset workload `variant`; a
  headless test reproduces the former bind-group validation failure
- **deck.gl rendering of network outputs** — analytics columns, neighborhood mask, reachability
  bands and extracted paths render through `GPUGraphNodeLayer`/`GPUGraphEdgeLayer` as
  storage-buffer columns with uniform scales: 0 pipeline rebuilds on column or scale change, 0
  uniform-buffer writes per steady frame. Contributor-side resident cost ≈16E bytes for the symmetrized
  CSR (89 KB at V=1,024/E=2,239; 5.5 MB at V=65,536/E=131,263)
- **Real map demo** — `examples/deck/spatial-analysis-explorer` has 25 modes, one per contributor family, all
  with 0 rebuilds under per-frame parameter changes
- **Euclidean distance and allocation**: `GPUDistanceField`, an exact separable
  (Felzenszwalb-Huttenlocher) distance transform with smallest-ID nearest-seed allocation, anisotropic
  cells, per-frame seeds and `maxDistance` band mask, plus a jump-flood preview mode;
  oracle-tested (allocation exact, distances within 1 ULP)
- **Spatial interpolation and focal statistics** — `GPUInverseDistanceWeighting`: gather-form IDW
  onto a raster over a per-encoding `GPUGridIndex` with in-cell ID sort, per-frame extent, radius,
  power, nearest-`k` (ties by smallest row), minimum neighbors, exact hits, log-space weights, NaN
  nodata, contour-ready output; `GPUFocalStatistics`: per-frame square or circle window mean, sum,
  min, max, range, centered standard deviation, and count with nodata, validity, and edge
  clipping; M3 Pro: IDW 50k samples to 512² in 1.7 ms (r with ~35 neighbors) and 2.9 ms (~138),
  focal 1024² mean/std/min/max/count 0.75 ms (r=1), 5.2 ms (r=5), 17.4 ms (r=10)
- **Polygon rasterization and raster join** — `GPUPolygonRasterization`: compute scan conversion of
  GeoArrow polygons to a zone-ID raster (even-odd rings, unioned multipolygons, smallest feature row
  wins, half-open center rule) with conservative boundary-cell flags, per-frame extent, GPU overflow;
  `GPURasterJoin`: O(1)-per-point polygon aggregation (counts, sorted sums) with per-zone
  boundary-point counts that bound the error against exact point-in-polygon
- **Network accessibility**: `GPUNetworkSnapping` (nearest edge, fraction, snap distance, two seed
  costs, exact scan or BVH join with a max snap distance), `GPUNetworkCostMatrix` (bounded
  many-to-all matrix from lane-batched reachability, bit-identical across lane counts) and
  `GPUNetworkAccessibility` (cumulative, gravity with exponential or power decay, 2SFCA; per-frame
  threshold, decay and beta with no search re-run; deterministic fixed-order sums)
- **Local spatial autocorrelation**: `GPUHotSpotAnalysis` (Getis-Ord Gi* z, p, 90/95/99% bins,
  optional Benjamini-Hochberg FDR) and `GPULocalMoran` (local Moran's I, conditional-randomization
  analytic z, spatial lag, HH/LL/HL/LH quadrants, optional FDR). Distance-band weights are evaluated
  directly from a per-frame cell index, so radius changes need no rebuild. Moments come from
  deterministic two-pass tree sums, with optional fixed moments and a mask. Tested against a CPU
  oracle.
- **Trajectory playhead and resampling** — `GPUTrajectoryPlayhead`: per-track binary search and
  interpolation at a per-frame playhead (position, elevation, heading, speed, status with a
  per-frame gap limit, segment row and fraction), compact active tracks with an indirect instance
  count, f32 and exact Int64-word times; `GPUTrajectoryResample`: fixed-N resampling in time or arc
  length into a dense `[tracks × N]` buffer
- **Line simplification**: `GPULineSimplification`. Level-synchronous parallel Douglas-Peucker and
  TD-TR into a monotone per-vertex importance column (GPU-gated rounds, convergence flag,
  superset fallback at the cap). Per-frame tolerance gives a keep mask, ascending compact row IDs,
  and per-line kept counts and starts with no recompile. Kept sets equal recursive Douglas-Peucker
  at every tolerance once converged, and importance is bit-exact against a CPU oracle.
- **Cell aggregation and zoom pyramids**: `GPUCellAggregation` (Quadbin keys from lng/lat in
  integer-only WGSL, bit-exact against a BigInt reference; pre-keyed Quadbin/H3 `uint32x2` rows
  as in CARTO), `GPUCellRollup` (sort-free `cellToParent` roll-up, bit-equal to direct
  aggregation), `GPUCellPyramid` and `GPUCellLevelSelection` (per-frame level choice and indirect
  draw words); exact counts, 64-bit fixed-point sums and extremes with u32 atomics; capacity and
  overflow that propagates up the levels. Apple M-series, 1M points, capacity 2^20: 4.6–6 ms per
  aggregation and 14 ms for a 6-level pyramid.
- `GPUPointToCell`: lng/lat to H3, Quadbin, quadkey, geohash and S2 keys in WGSL. H3 `latLngToCell` is a new forward port with generated `faceIjkBaseCells` tables. Quadbin, quadkey and geohash are integer-exact. H3 and S2 are f32, with per-resolution mismatch rates measured against h3-js and an S2 oracle. `GPUCellGeometry` gives centres and boundaries for all five families plus A5. The H3 boundary is a port of `_faceIjkToCellBoundary` with face overage and distortion vertices, and it matches h3-js vertex counts for 100% of the cells tested.
- `GPUCellTopology` (H3/Quadbin `gridDisk`, `gridDiskDistances`, `gridRing`, `cellToParent`, `cellToChildren`, all exact on the GPU) and `GPUCellCompaction` (`compactCells`/`uncompactCells`, capacity bounded). H3 neighbor stepping is a WGSL port of `h3NeighborRotations` with tables generated and verified against h3-js. Disks are an exact breadth-first search through pentagons and across faces.
- `GPUCellCover`: polygon (with holes) to cell polyfill with `(feature id, cell)` capacity-bounded output. Quadbin supports `center`, `full` and `intersects`, exact against an f32-faithful oracle. H3 `center` uses a lattice-nearest-centre rule, so there is no sort and no duplicates. It matches h3-js `polygonToCells` exactly at res 3-7 and to within 12 of about 75,000 cells at res 8-10, all of them f32 centre-on-edge ties.
- **Column classification**: `GPUColumnQuantiles` (exact quantiles by 8-bit radix select over
  order-preserving keys, five interpolation rules, deck.gl/kepler percentile filter mask and
  bounds), `GPUClassBreaks` (equal interval, quantile, standard deviation, head/tail, box plot,
  maximum breaks, binned Fisher-Jenks natural breaks, custom; per-frame method and class count),
  `GPUColorScale` (d3 continuous, quantize, threshold, quantile, ordinal scales into a packed
  `rgba8` column, class ids, and legend counts), and `GPUBivariateClassification` (n x n classes,
  value-by-alpha). All per-frame parameters live in buffers; the four compose in one graph with no
  readback, and integer outputs match CPU oracles bit for bit.
- **Column profile**: `GPUColumnProfile`, a fused per-frame statistics panel for up to 16 numeric
  or dictionary-coded columns: exact counts, nulls, min and max, fixed-order Chan/Welford sum, mean
  and variance (bitwise reproducible), per-frame or automatic histogram domains, HyperLogLog
  distinct counts, and exact deterministic top-K categories, all matched against CPU oracles.
- `GPUGroupStatistics`: GPU group-by on 32- or 64-bit keys with count, sum, mean, min, max, variance, standard deviation, skewness, kurtosis, median, per-frame percentiles, mode, unique count and per-row z-scores for up to four columns; exact integer statistics, fixed-order bitwise-reproducible moments, capacity-bounded sorted output; tested against a CPU oracle.
- `GPUKeyJoin` (`gpu-dataframe/key-join/`): left and inner attribute joins on u32 or u64 keys by stable radix sort plus binary search, with 1:1 gathers (f32/u32), 1:n aggregates (count, sum, mean, min, max with exact fixed-point sums), left and right matched flags, match counts, and bounded inner-row compaction; tested against a CPU oracle.
- `GPUCellTableCompare`: outer join of two sorted cell tables (merge path by binary search and scan) with presence, before/after, exact fixed-point delta, ratio, percent change and Poisson or standardized z-scores, capacity-bounded with overflow.
- **Line geometry**: `GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, and `GPULineChunk`. Densify (planar or great-circle), origin/destination great-circle arcs with unwrapped longitudes, Chaikin smoothing for open paths and closed rings, and chunk/substring by measure, all with per-frame resolution through count, scan and emit into a shared capacity-bounded `GPULinePathOutput` (clamped path offsets, overflow, source path and row, measures). Tested against f64 oracles, including antimeridian, capacity overflow and zero-recompile parameter changes.
- **Geometry measures and geodesy**: `GPUGeometryMeasures`, `GPUGeodesicPairs`, and `GPUGeodesicDestination`. Per-feature and per-group length, perimeter, signed and absolute area (two hole rules), centroid, bounds and vertex count in planar, spherical (turf-compatible) and WGS84 (authalic area, Vincenty length) modes with local-origin compensated f32 sums and deterministic group reductions; geodesic distance, bearings, midpoints and destinations with f32 formulations that stay accurate from 1 m to antipodal, each with a measured error budget against f64 oracles.
- **Linear referencing**: `GPULinearReferencing` and `GPULineLocate`. Points snap to the nearest of many polylines through the existing nearest-feature BVH join (ties on the smallest segment) and report path, segment, fraction, foot point, distance, measure along the path, side and signed offset; events are placed by distance or fraction with lateral offsets, tangents and angles, and a per-frame scale/offset for animation. Tested against f64 brute force on random polylines with zero recompiles across radii.
- `GPUTerrainHorizon`, `GPUSolarShadowMask`, `GPUReliefShading`, `GPUTextureShading`, and `GPUSolarPosition` (`gpu-terrain/terrain-illumination/`): bounded ray-march horizon maps with sky-view factor and openness, per-frame soft solar shadows and sun+ambient illumination from the horizon map (two reads per pixel per frame), multidirectional (fixed or USGS MDOW) hillshade with a Swiss/Imhof relief blend and RGBA8 color, multi-scale DoG texture shading with per-frame detail, and NOAA sun position on CPU (`getSolarPosition`) and per row on GPU; every contributor tested against float64 oracles.
- **Raster map algebra**: `GPURasterReclassify` (per-frame break tables, class values, class counts),
  `GPUWeightedOverlay` (suitability over up to 16 layers with linear or table remaps, restricted
  classes, weight normalization, nodata policies, score range), `GPURasterCellStatistics` (local
  min/max/range/sum/mean/stdev/majority/minority/variety/count across up to 64 layers),
  `GPURasterConditional` (`where` with per-frame comparisons and constants) and `GPURasterArithmetic`
  (per-frame operation codes incl. normalized difference); oracle-tested
- **Raster stretch and colormaps**: `GPURasterStretch`, exact GPU min/max, deterministic histogram and CDF,
  linear/percentile/equalize stretches with gamma and sigmoidal contrast, LUT and rgba8 palette outputs, and
  statistics restricted to a per-frame window and region mask ("stretch to visible extent"); oracle-tested
- **Isolines and isobands on any raster**: `GPUIsolines` (two-endpoint marching-squares segments with
  centre-average saddles, level and edge ids, per-frame levels and extent, deterministic polyline
  stitching by fixed-round pointer jumping with closed rings) and `GPUIsobands` (per-sample band classes
  plus capacity-bounded CCW band triangles from a combinatorial boundary walk, band window, indirect-draw
  vertex count); oracle-tested, and band boundaries coincide bit for bit with the isolines
- **Raster values to points and profiles**: `GPURasterSampling` (nearest, bilinear, Catmull-Rom bicubic at
  points; per-frame method, extent, and strict or renormalized nodata handling; active point count) and
  `GPURasterProfile` (elevation profiles along polylines at a per-frame spacing with distances, values,
  cumulative gain/loss, per-path length/gain/loss/min/max, capacity-bounded with overflow); oracle-tested
- `GPUParticleAdvection` (round 7, `gpu-raster/particle-advection/`): RK2 particle advection through a NaN-aware
  bilinear vector field, in-place state, Philox respawn keyed (seed, particle, generation) for exact
  replays, drop rate, maximum age, minimum speed, data-space trail ring buffer, speed column
- `GPULineIntegralConvolution` and `GPUStreamlines` (round 7, `gpu-raster/flow-texture/`): animated LIC (Philox
  noise, Hann window with phase ripple, independent output extent, r32float texture) and evenly
  spaced streamlines pruned by priority in GPU-gated claim/decide rounds, equal to the greedy
  Jobard-Lefer-style pass, as CSR polylines with capacity and convergence flags
- `GPUDotDensity` and `GPURandomPointsInPolygon` (round 7, `geospatial/dot-density/`): dasymetric dot density
  with per-category counts, Bernoulli remainders, zoom-stable dot prefixes, Philox rejection sampling
  in polygons with holes, optional weight mask, capacity-bounded compact output and failure count
- **Neighbor search and spatial weights**: `GPUNeighborSearch` does exact kNN (k <= 32, ties
  to the lowest ID) and distance-band queries, self or cross join, with masks. It writes a
  `GPUSpatialWeights` CSR with rows in ascending ID order, distances, and binary, inverse-distance or
  kernel weights (fixed or adaptive bandwidth) with optional row standardization. Capacity is
  bounded with an overflow flag. Bounds, radius and weights are per-frame. The CSR is the shared
  input of the round-7 global statistics and permutation contributors. Tested against a brute-force
  oracle, with exact ID equality.
- **Global spatial autocorrelation**: `GPUGlobalSpatialStatistics` reads a `GPUSpatialWeights`
  CSR and computes global Moran's I, Geary's C, Getis-Ord General G, bivariate Moran's I and
  BB/BW/WW join counts. Each comes with analytic expectation, normality and randomization variance,
  z and p, following esda's definitions. General G and join-count variances use a cancellation-free
  centered form, and bivariate Moran uses an exact randomization variance. Reductions are
  deterministic and inputs can change per frame. The analytic moments are verified against exact
  permutation enumeration, and GPU output against an f64 oracle.
- **Permutation inference**: `GPULocalPermutationTest` does conditional-permutation pseudo
  p-values for local Moran, local G and G* (esda `Moran_Local` and `G_Local`), with integer
  exceedance counts, an optional Benjamini-Hochberg mask and a neighbor cap with an overflow flag.
  `GPUGlobalPermutationTest` builds reference distributions for global Moran, Geary, General G and
  bivariate Moran, with `p_sim`, `z_sim` and a histogram. Both use a private counter-based Philox
  4x32-10. The global test also uses a keyed Feistel bijection, so there are no per-permutation
  sorts. Results are seed-reproducible, and the seed and P change per frame. The RNG is checked
  against Random123 known-answer vectors and is bit-identical between WGSL and TypeScript. Local
  counts match an f32-emulating oracle exactly.
- **Pair statistics** — `GPUVariogram` (omni/directional Matheron and Cressie-Hawkins semivariogram, CPU `fitVariogramModel`), `GPUSpatialCorrelogram` (Moran's I per cumulative or annulus distance band with normality/randomization z and peak bands), `GPURipley` (K, L, g with none/border/isotropic edge correction), and `GPUPointPatternIndices` (exact nearest neighbour, Clark-Evans, quadrat VMR) over one shared pair-histogram kernel: per-workgroup shared-memory 64-bit integer accumulators, fixed-point float terms, bitwise reproducible; M3 Pro: 100k points / 54M pairs in 10 ms, 20k all-pairs in 12 ms; tested against f64 CPU oracles
- **Geographic distributions** — `GPUGeographicDistribution`: per-group or masked weighted mean centre, Weiszfeld median centre, standard distance, standard deviational ellipse (ArcGIS sqrt(2) convention), linear directional mean, and ellipse/circle polygon rings; fixed-order sorted sums, bitwise reproducible; tested against an f64 CPU oracle
- `GPUEmergingHotSpots`: space-time Gi* over a lattice cube, per-cell Mann-Kendall trend (exact integer S, tie-corrected), and the 17-category ArcGIS emerging hot spot classification, with per-frame radius, window and thresholds.
- `GPUOrdinaryLeastSquares`: deterministic OLS with intercept, optional ridge, standard errors, t statistics, R2, adjusted R2, log-likelihood, AIC, BIC, residual and fitted columns, Jarque-Bera, and Koenker Breusch-Pagan; tile-and-merge fixed-order accumulation, correlation-scaled Cholesky solve, status word for singular or short designs (spatial-regression)
- `GPUGeographicallyWeightedRegression`: GWR with Gaussian and bisquare kernels, fixed or adaptive bandwidths, an on-GPU AICc bandwidth ladder (per-frame, no rebuild), local coefficients, local R-squared, hat diagonal and a global summary; brute-force O(n^2 * ladder) with fixed-order reductions.
- `GPUCompositeScore`: min-max, z-score and exact percentile-rank scalers, per-indicator directions, weighted sum, weighted geometric mean and first-principal-component scores with per-frame weights, fixed-order column statistics, oracle-tested
- `GPUInequality`: per-zone Gini, Theil T and L, Atkinson (any epsilon per frame), Hoover, Palma, Lorenz knots, weighted (population) variants, and the between/within Theil T decomposition with pooled Gini, deterministic with no atomics, tested against a float64 oracle.
- `GPUCalendarBuckets`: exact Int64-millisecond to calendar columns (year, month, day, hour, minute, weekday with configurable week start, day of year, ISO week and week-year, quarter) plus a deterministic hour by weekday count matrix; fixed per-frame offset or per-row offset column for DST; tested bit-exact against a BigInt oracle and JS `Date`.
- `GPUChangeDetection`: two-slice difference/log ratio/percent change, Welch t-test with Student-t p-value, Theil-Sen slope, Mann-Kendall S/Z/p, multiband change magnitude and direction, and significance classes, per-cell deterministic, oracle-tested

Still open:

- Upstream composition, checked 2026-10-05 (local code kept, with reasons in TSDoc):
  - `GPUNeighborSearch`, `GPUSpatialClustering` and the spatial-autocorrelation contributors build
    their own cell keys because `GPUGridIndex` bakes `bounds` and `gridSize` into WGSL. They need
    per-frame bounds and a cell size of at least the per-frame radius, without recompiling. A
    buffer-driven bounds and minimum-cell-size mode on `GPUGridIndex`, plus a stable in-cell order,
    would let them drop their key kernels. Scan, sort and group aggregation are already upstream.
  - `GPUNetworkStatistics` derives masked live degrees inside its masking pass. `GPUGraphDegree`
    counts every stored CSR slot and has no vertex or edge mask.
  - `GPUNetworkReachability` is multi-source, cost-bounded and frontier-driven, and keeps the network
    tie rule. `GPUGraphSingleSourceShortestPath` is single-source dense Bellman-Ford with a
    lowest-parent tie rule.
  - `createWGSLKernelNode` now dispatches through the engine `Kernel`, like gpu-core.
- Terrain performance:
  - Pyramid-skip traversal kernels are about 2x slower than the march for viewshed and cumulative
    viewshed. The pyramid build is only 0.3–0.9 ms and sharing one pyramid does not close the gap.
  - Hydrology's fixed-length relaxation loops make the drainage graph 2,675 nodes; it needs a
    GPU-side loop exit (indirect dispatch or a convergence flag).
- Demo findings from `spatial-analysis-explorer`:
  - Explorer reachability mode should drop to `maxIterations` ≈ 48 (default `localIterations` 16):
    640 rounds is 649 nodes, 48 is 57; it converges in 5.
  - `GPUBufferSelection`, `GPUSpatialClustering` and `GPUTrajectoryMetrics` have no
    `drawInstanceCount` output, so modes copy counts into draw records by hand.
  - `GPUFlowAggregation` has no zone-count extent output, so its zone color range needs a readback;
    `gridSize` is compile-time while the hexagon radius is per-frame.
  - Region statistics grid-index cost scales with `candidateCapacity`, not region size.
  - Unrolled iteration is still the dominant CPU cost: `GPUNetworkServiceAreas` 400 iterations is
    1,212 nodes and 31–36 ms encode; `GPUCostDistance` 128 iterations is 271 nodes against 19–59
    needed. A GPU Core "repeat while" node, or the frontier queue, would fix both.
  - `GPUTerrainFlow` has no iteration count; `GPUCostDistance` friction calibration is
    compile-time; `GPUTrajectoryMetrics` has no per-step speed, heading or acceleration;
    `createWGSLKernelNode` is not exported.
  - Record viewshed and reachability benchmarks on real data.
- deck.gl WebGPU picking is mirrored vertically: `deck-picker` converts the cursor with
  `cssToDevicePixels([x, y], true)` (GL bottom-left) and reads WebGPU's top-left picking texture,
  so the picked row is `height − 1 − y`. Fix in deck.gl core (`yInvert = device.type !== 'webgpu'`)
  and add an off-center pick test; the graph layers add no workaround.
- `GPUNetworkServiceAreas` still runs its own unrolled relax and gate loop; move it onto
  `network-frontier.ts`. `createReachabilityGateNode` stays exported only for it.
- A truncated tie phase leaves the deepest equal-cost plateau nodes without predecessors; the only
  signal is `converged = 0`.
- Extend `GPUGraphTopologyView` to the remaining gpu-graph algorithms (BFS, SSSP, modularity,
  clustering, layouts) and migrate `GPUNetworkStatistics` off the `@internal`
  `createNetworkAnalyticsTopology` / `defaultBuffer` adapter.
- Native `mask` inputs for gpgpu `GPUGridBinning` and `GPUGridAggregation`, removing
  `GPUPointDensity`'s masked-copy pass (one pass and N × 8 bytes).
- Bridge `GPUTileLODSelection` to the residency arena: map its per-tile drawn mask (tile IDs) to arena
  slots so `GPUResidentRowSelection` `tileMask` comes from the GPU.
- Chunked (`GraphVectorView`) hierarchies and CSR inputs; 3D, f64, and geodesic variants. Tiled and
  streamed rows go through `GPUResidencyArena`. `GPUCompaction` still emits chunks² scatter passes
  for vector outputs (1,249 nodes at 32 chunks, remeasured); document it as a non-goal or route
  vector outputs through one packed scatter. Unexplained: 1,249-node encodings queued 20 deep cost
  120–140 ms each against 24 ms synced.
- Attribute generated-ID collisions for every contributor: `createTransientView` and node IDs in gpu-core
  should name the creating contributor. Today only cost-distance and relaxation transients do, through
  a raster-relaxation helper. Importing one buffer twice under different IDs throws "already in
  use" (documented).
- Graph layers: `highlightColor` collides with deck's `LayerProps.highlightColor` (readonly tuples
  fail to compile); rename to `neighborhoodColor`. Add log and sqrt color scales for heavy-tailed
  columns, and hash categorical colors (`value % palette.length` collides on power-of-two labels).
  Hover reruns reachability and path too (3–5 ms encode at ≤65k vertices); split graphs if GPU time
  matters. Port to deck.gl-community once that checkout is usable.
- Edge bundling: a deck.gl layer or example drawing `paths` through the draw record; GPU-gated early
  exit for unused iterations; benchmarks at 64 iterations and on low-contention scenes; chunked
  positions; cos-lat correction for lon/lat input. f32 KDEEB amplifies ulp noise ~1000× per
  iteration, so CPU parity holds only statistically past one iteration; the f64 `EPS` became a
  4-quanta gradient floor.
- `GPUNetworkStatistics` is single-graph (contributor-owned buffers under fixed IDs) and counts a
  self-loop once, unlike `GPUGraphModularity`. `GPUAttributeCrossfilter` bins `uint32` as f32, exact
  only to 2^24.
- Arrow temporal relative-float32 path misreads `Date(ms)` columns (two int32 words per row); the
  word path is correct.
- Flow sorted mode does not share sorts between pair and zone keys (about 3 sorts and 3 scans per
  encode). Raster atomic-versus-sorted cost is not measured, and atomic drift on Apple GPUs is not
  demonstrated.
- `trajectory-metrics-kernels.ts` and `flow-aggregation-kernels.ts` bind only what each body reads,
  a workaround the kernel helper no longer needs.
- Promote `utils/wgsl-kernel-nodes.ts` helpers (linear kernel node builder, bounded compact publish) into
  GPU Core if more consumers appear.
- Package placement stays open (luma.gl versus a deck.gl-side module); luma.gl must not depend on
  deck.gl either way. Projection-dependent contributors take their CRS contract from the
  [GPU Project roadmap](./gpu-project-roadmap.md).
- `GPUGridIndexQuery` dispatches one invocation per indexed object; region statistics uses a local
  cell-range gather instead. Promote a cell-range-driven query into GPU Core. The grid path stays
  O(N) when IDs or a mask are requested, and screen-space and pick selections are not indexed.
- Promote `gpu-raster/cost-distance/raster-relaxation.ts` (tiled min-relaxation, GPU-gated iteration) into
  `gpu-raster` once a second consumer exists (only `GPUCostDistance` uses it); network reachability
  no longer uses a gate, and raster relaxation could adopt the queue design for sparse activity.
  Cross-tile hydrology and cost distance, anisotropic costs, and nearest-source allocation.
  `gpu-raster/cost-distance/raster-relaxation.ts` now has three consumers (cost distance, fill, flat
  resolution); promote it to `gpu-raster`.
- Hydrology open items: Priority-Flood filling and watershed labelling and least-cost breaching
  (Barnes 2014; Lindsay 2016; sequential priority queues, GPU variants need a tile spill graph),
  cross-tile fill, flats, accumulation and watersheds, D-infinity HAND/DistDown (TauDEM) and Shreve
  magnitude, and a GPU loop primitive with a GPU-side break (gated loops cost about 50 µs per
  skipped round pair after convergence).
- Flow pair keys are 32-bit (65535 zones); trajectories need geodesic distances; capacitated
  location-allocation; OPTICS/HDBSCAN.
- Round 4 contributors (`GPUTemporalReduction`, `GPUNetworkSubgraphFilter`, `GPUNetworkCoarsening`,
  `GPUAdjacencyMatrix`) take packed views only, have no GPU timings except coarsening, and were
  parity-tested on Apple Metal only. Open per contributor: a deterministic sum or mean column for
  temporal reduction; u32 and i32 filter columns and feeding the induced CSR to
  `GPUGraphTopologyView`; sparse-label relabelling and multi-level coarsening; per-slot balanced
  binning for hub rows (both the matrix and slot pairing walk a whole row per invocation),
  matrix-cell edge readback and a multi-resolution matrix.
- Round 6, `GPUDistanceField` and its siblings in `gpu-raster/distance-field/`:
  - Euclidean direction output (ArcGIS "Euclidean Direction") and back-to-seed vectors; trivially
    derivable from `nearestCells` but not shipped
  - Sub-cell seed positions (distance to the seed point rather than its cell center), line/polygon
    seeds without rasterizing them into `seedMask` first, and barriers (cost-distance covers those)
  - Geodesic distances on longitude/latitude grids; cells are planar
  - Multi-tile or streamed grids; one grid per contributor, at most 32768 cells per side
  - Exact-mode row pass is one serial invocation per row (O(width log width) per row, uncoalesced
    column reads); a workgroup-cooperative row pass or a transposed layout would be faster on
    large grids
- Round 6, `GPUInverseDistanceWeighting` and its siblings in `geospatial/spatial-interpolation/`:
  - Focal statistics: van Herk/Gil-Werman separable min/max and summed-area (or separable) mean/sum
    for large square windows; workgroup-tile shared memory for the direct gather; majority, median,
    and rank (histogram or sorting-network windows); annulus and custom kernel windows.
  - IDW nearest-`k` keeps a private `(d^2, row)` list per invocation; at k = 16 it is ~6x slower than
    radius-only mode on M3 Pro (16.9 vs 2.9 ms). Try a workgroup-shared candidate list or a radius
    shrink once `k` are found. Barrier-aware IDW (obstacle mask), anisotropic search, and a
    caller-supplied (shared) grid index instead of the per-contributor rebuild.
  - Promote the in-cell ID sort into `GPUGridIndex` (deterministic in-cell order option); region
    statistics and other gathers that sum floats over index cells need it too.
  - Output count is `uint32`; contouring a count surface needs a float conversion.
- Round 6, `GPUPolygonRasterization` and its siblings in `gpu-raster/polygon-rasterization/`:
  - Raster join follow-ups: an epsilon-to-resolution planner (pick cell size from a distance bound);
    a built-in hybrid exact mode that sends boundary points through `GPUPointInPolygonJoin` (today
    the caller wires `pointBoundaryMask`); tiling for rasters beyond one storage binding; per-frame
    raster dimensions (they are compile-time); chunked (`GraphVectorView`) points; a `featureIds`
    remap (zones are feature rows); one ID layer per overlap depth for overlapping-feature joins
    (overlaps resolve to the smallest row only).
  - `GPUPolygonRasterization` sorts and scans always run over the full `crossingCapacity`, and one
    invocation fills one span (up to `width` atomics), so a raster-wide polygon fills rows serially
    per thread. A two-level span split or an indirect dispatch sized by the crossing count would help
    very wide rasters.
  - Feed `GPURasterZonalStatistics` from `GPUPolygonRasterization` in the explorer demo, and reuse the
    scan-conversion front end for polyfill / H3 cover and raster overlay.
- Round 6, `GPUNetworkSnapping` and its siblings in `gpu-network/network-accessibility/`:
  - The matrix is dense `rowCount x nodeCount` f32. The scoring pass binds all of it, so
    `rowCount * nodeCount <= maxStorageBufferBindingSize / 4` (32M entries at 128 MB). Large
    networks need a sparse within-`costLimit` (row, node, cost) list, or row tiling for the scoring
    pass.
  - Every lane batch is a separate `GPUNetworkReachability` with its own frontier transients, so
    graph nodes grow as `ceil(rows / laneCount) * (maxIterations + 3)`: 976 nodes and 13 ms of CPU
    encode for 2,304 rows. A reachability entry point with a lane count (per-lane seeds, a shared
    frontier) would need one set of rounds. Lane expansion also copies the CSR `laneCount` times
    per encoding (`laneCount * (nodeCount + 2 * edgeCount)` words).
  - Opportunities attach to nodes (or to snapped seeds as matrix rows). No per-edge opportunity
    mass and no snapping of origins inside the scoring pass: origin accessibility at a snapped
    point means combining the two endpoint scores on the CPU, or snapping origins onto rows.
  - `GPUNetworkSnapping` is planar (no geodesic distances). The BVH path picks the edge with the
    join's own distance expression, then recomputes fraction and distance. Near-ties within an ulp
    can differ from the exhaustive scan (the tests only exercise exact ties and clear winners).
    Edges with an endpoint out of range become far-away segments in the BVH path, so a radius near
    1e30 would match them.
  - Radius-bounded UNA metrics (reach, closeness, betweenness) and many-to-many OD for
    location-allocation could reuse `GPUNetworkCostMatrix`. Neither is built yet.
- Round 6, `GPUHotSpotAnalysis` and its siblings in `geospatial/spatial-autocorrelation/`:
  - Permutation inference (pseudo p-values) for Gi* and local Moran. It needs a counter-based RNG.
    Proposed design: Philox4x32-10 in WGSL (u32 math only, `mulhi` emulated with 16-bit limbs). The
    counter is `(row, permutation, drawBlock, streamId)` and the key is a 64-bit seed from the
    parameter buffer. Each `(row, permutation)` invocation draws `k_i` distinct indices from the other
    `n - 1` included rows by a partial Fisher-Yates over a virtual index range: rejection on a small
    per-thread set, or a Feistel bijection on `[0, n - 1)` so no state is stored. It sums the permuted
    centered values and counts `|stat| >= |observed|` per row with integer atomics. Integer counts
    and a counter RNG make it exact and seed-reproducible. Add a preview `P = 99` mode with refine on
    idle; the cost is about `N * P * k`.
  - ArcGIS-style total-randomization variance for local Moran as an alternative null. The current
    conditional z takes the sign of `x_i - mean`, so it is discontinuous for values at the mean, as in
    esda's folded permutation test.
  - Global indices over the same neighbor loop: global Moran's I, Geary's C, General G, and an
    incremental-Moran distance sweep for "optimized hot spot".
  - k-nearest and inverse-distance weights, chunked `GraphVectorView` inputs, geodesic distances, and
    space-time neighborhoods (emerging hot spot).
  - The neighbor loop is unbounded. Cost is `O(sum of neighbor counts)`, so a radius covering most of
    the data is `O(n^2)`. There is no neighbor cap or overflow flag yet.
- Round 6, `GPUTrajectoryPlayhead` and its siblings in `geospatial/trajectory-interpolation/`:
  - Double-single (`timestampsLow`) time input for the playhead and resample (f32 relative and Int64
    words are supported).
  - Arc-length resampling sums per track in one invocation; long tracks load-balance poorly. Needs a
    float (or deterministic fixed-point) segmented scan in GPU Core; `GPUSegmentedScan` is uint32-only.
  - Heading of zero-length segments is 0 rather than the last moving heading; no shortest-angle
    smoothing between segments (heading is piecewise constant, as is speed). Curved (Catmull-Rom)
    interpolation and dead-reckoning extrapolation past the last fix are not implemented.
  - Geodesic interpolation and antimeridian handling (planar only, project upstream).
  - Chunked (`GraphVectorView`) inputs; residency-arena columns work today as packed views.
  - Interior resample targets use a GPU f32 division (2.5 ulp), so at a path discontinuity (duplicate
    timestamps with different positions, or a stationary step in arc length) a sample may fall on
    either side; tests accept the oracle within 4 ulps of the target.
  - Performance not measured yet (no benchmark run).
- Round 6, `GPULineSimplification` and its siblings in `geospatial/line-simplification/`:
  - Each round unrolls 5 nodes, so the default 64 rounds is about 325 nodes. Deep shapes such as spirals
    need up to `n - 2` rounds. Options are a persistent per-workgroup loop for short lines and a
    shared GPU-gated loop primitive (the same issue as the network-reachability node count).
  - The per-round dispatch covers every row, including rows already decided. Compacting the active
    rows per round would help when lines are long and the tree is deep.
  - Topology-preserving simplification, Visvalingam-Whyatt, a perpendicular-to-line metric option,
    geodesic distances, double-single or Int64 timestamps for `'time-ratio'`, chunked inputs, and
    polygon rings (a closed ring keeps its seam vertex, and the chord degenerates to a point).
  - Subnormal-range coordinates are unsupported (GPUs may flush them). NaN inputs are undefined.
  - No `drawIndirect` argument writer. Use `output.count` as the instance or vertex count.
- Round 6, `GPUCellAggregation` and its siblings in `geospatial/cell-aggregation/`:
  - Forward H3 and A5 keying of points (`latLngToCell`) in WGSL. gpu-dggs only projects cells to
    centers. H3 pyramids need pre-keyed rows today, and A5 is not supported, since there is no
    parent or forward primitive.
  - deck.gl consumers: binary/Arrow cell columns plus GPU instance counts in `H3HexagonLayer`,
    `QuadbinLayer` and the CARTO `ClusterTileLayer` (see "Consumers" below). There is also no GPU
    Quadbin center or bounds decode.
  - `'any'` aggregation (a representative row value) and averages as a column: average is
    `sumValues / counts` today. Multiple value columns need one contributor per column, and each repeats
    the sort.
  - Chunked (`GraphVectorView`) inputs; per-tile tables for tiled CARTO sources, which could go
    through `GPUResidencyArena`.
  - Hot cells serialize on atomics. A res-0 table of 1M rows hits one address, which costs little
    on Apple but was not measured elsewhere.
- Cross-cutting findings from round 6:
  - Apple Metal fuses `a * b + c` into FMA, so a GPU result can differ by 1–16 ULP from a
    correctly rounded CPU oracle on unquantized input. `line-simplification` blocks fusion with a
    runtime zero loaded from memory; `cell-aggregation` keys points with integer-only WGSL. Audit
    contributors whose parity relies on rounded products (for example `temporal-reduction` bucket
    edges) with unquantized inputs, and add a shared no-fuse helper.
  - A WGSL module that fails to compile (one used the reserved word `target`) produced all-zero
    output with no error through the `createWGSLKernelNode` path. Surface shader compilation
    errors in `createWGSLKernelNode` or the graph compiler.
  - `GPUGridIndex` leaves the order within a cell to atomics, so float sums over neighbors vary
    between runs. `spatial-autocorrelation` builds its own stable cell index and
    `spatial-interpolation` sorts IDs within each cell. Add a stable in-cell order option to
    `GPUGridIndex` and per-frame index bounds.
  - Round 6 contributors carry local versions of missing gpgpu primitives: polygon scan conversion,
    exact distance transform and jump flooding, a two-word (64-bit key) sort, and a fixed-order
    cell gather. Promote them to GPU Core when a second consumer appears.
- Explorer findings for the round 6 contributors (modes `hot-spots`, `distance-field`, `raster-join`,
  `cell-pyramid`, `playhead`, `accessibility`, `simplification`, `interpolation`):
  - `GPUCellLevelSelection.drawArguments` always writes `firstInstance`, so on a device without
    `indirect-first-instance` every level after the first draws nothing, with no validation error.
    Add an option that leaves `firstInstance` at 0 and publishes the first row separately.
  - `GPUCellPyramid` levels cannot share one slab, because outputs are checked against inputs per
    buffer, not per byte range; the explorer ping-pongs two slabs. Make the check range-aware or
    let the pyramid allocate its own level storage. Every level also has the finest capacity.
  - `GPUCellLevelSelection.output.count` rejects the strided instance-count view of a
    `DrawCommandBuffer`; accept a stride for one-row views.
  - `GPURasterJoin` zone values use `NO_ZONE = 0xffffffff`, which cannot index per-zone values in a
    raster layer; add a compact display output. Its cost follows `crossingCapacity`, not the
    crossings used: with 12 convex polygons it was slower than the exact join.
  - `GPUTrajectoryPlayhead` counts only active tracks on the GPU; add per-status counts.
  - `GPUNetworkAccessibility` has no extent output, and `GPUNetworkSnapping` positions need an
    interleave step before a segment layer can draw them.
  - `GPUHotSpotAnalysis` and `GPULocalMoran` take FDR as a compile-time flag, so a toggle compiles
    both variants; make it a parameter.
  - `GPUInverseDistanceWeighting` has no GPU count of nodata cells, and its `counts` output is
    `uint32`, so neither feeds a float colormap without a readback or conversion.
  - `GPULineSimplification` has no indirect-draw writer, and its 64-round default unrolls 324
    nodes where NYC trips converge in 15.
- Round 7, `GPUPointToCell` and its siblings in `geospatial/cell-indexing/`:
  - `precision: 'double-single'` (hi/lo position columns) for exact H3 at res ≥ 10 and S2 at level ≥ 20. f32 math caps H3 at about res 12 and S2 at about level 20.
  - No forward A5 (point to A5 key); A5 is geometry-only.
  - Chunked `GraphVectorView` inputs are not supported (single packed views only, as in `GPUCellAggregation`).
  - Quadkey/quadbin rows at zoom 27-29 differ from the f64 formula by at most one row (1.3-9.6%). Deep-zoom quadkey centres are f32-limited.
  - No string conversion (H3/S2 tokens, geohash text); that stays on the CPU.
- Round 7, `GPUCellTopology` and its siblings in `geospatial/cell-topology/`:
  - H3 disks cost one private breadth-first search per thread, about 217 cells at k = 8. This is unmeasured. A shared-memory or spiral (`gridDiskUnsafe`) fast path for cells away from pentagons is not done.
  - No `gridDistance`, `gridPathCells`, local IJ, directed edges or vertexes.
  - Uncompact has no dedupe of overlapping inputs, and its unclamped total is a u32.
  - Compaction reuses `geospatial/cell-aggregation/cell-table` internals (`getCellTableNodes`) and allocates an unused transient counts column.
  - S2, A5 and geohash topology are not done.
  - GPU tests cover H3 disk and ring only up to k = 4. The k = 5-8 strides are validated in the node spec only.
- Round 7, `GPUCellCover` and its siblings in `geospatial/cell-cover/`:
  - H3 `full` / `intersects` containment (h3-js `containmentFull` / `containmentOverlapping`): needs cell-boundary-versus-polygon tests.
  - `compactToParents` option. Callers can chain `GPUCellCompaction` per feature today, but the contributor has no built-in per-feature compaction.
  - Each candidate loops over all edges of its feature (candidates x edges). There is no edge BVH or scanline acceleration for large polygons.
  - H3 lattice cost grows as 1/cos(latitude) and is clamped at |lat| ≤ 89. The inradius constants for res 11-15 are extrapolated (divided by sqrt 7 per level).
  - No antimeridian-crossing polygons, and no check for them. S2/geohash/quadkey cover is not done.
  - Double-single H3 centres would remove the remaining edge ties at res ≥ 8.
- Round 7, `GPUColumnQuantiles` and its siblings in `gpu-dataframe/column-classification/`:
  - Quantiles: a single-pass `precision: 'histogram'` fast path, grouped (per-cell) medians and
    quantiles over a segmented sort, `uint32`/`sint32` columns, Int64 columns, more than 2^24 rows,
    and chunked vector inputs.
  - Class breaks: exact refinement of natural breaks inside the bins around each break; maximum
    breaks without two full sorts every frame (gate the sorts, or a histogram approximation);
    head/tail rounds unroll three nodes each (`maximumClassCount - 1` rounds), so a GPU-gated loop
    primitive would shrink the graph; mapclassify's de-duplication of repeated quantile edges (ties
    currently give empty classes); geometric/percentile/user-defined interval variants.
  - Colour scale: interpolation in a perceptual colour space (OKLab), diverging scales with a
    midpoint, NaN-safe descending domains, and a texture LUT output for raster layers (the same logic
    as the planned raster stretch).
  - Bivariate: more than 16 classes per axis; trivariate and value-by-saturation variants.
  - deck.gl still needs a GPU-buffer attribute route to consume `colors` without a readback.
- Round 7, `GPUColumnProfile` and its siblings in `gpu-dataframe/column-profile/`:
  - Histogram, HyperLogLog and category counts run as separate row passes per column (about four
    pipelines per column) rather than one fused pass; one hot histogram bin serialises on one atomic.
  - Medians and quantiles in the panel (wire `getColumnQuantileNodes` from `column-classification`),
    skewness and kurtosis (M3/M4 in the moment record), and count-min sketches for very large
    dictionaries.
  - HyperLogLog large-range correction (not needed below 2^24 rows with a 32-bit hash), HLL++ bias
    correction, more than 2^24 rows, and chunked vector inputs.
- Round 7, `GPUGroupStatistics` and its siblings in `gpu-dataframe/group-statistics/`:
  - Approximate distinct counts (HyperLogLog) for very high-cardinality columns; `uniqueCount` is exact and needs a value sort.
  - Weighted statistics and weighted quantiles; histograms / quantile sketches without a full value sort.
  - A sort-free fast path for statistics that need no order (count, sum, mean, min, max could use a hash table instead of a key sort when keys are dense or small).
  - Alternative quantile definitions (nearest rank, lower, higher, midpoint); only linear interpolation is implemented.
  - Float64-grade sums for values beyond the fixed-point range (|v * sumScale| saturates near 2^62) and exact moments (moments use f32 shifted sums).
  - Chunked vector inputs (`GraphVectorView`) are rejected; single packed views only.
  - Groups whose spread is below the fixed-point mean resolution (m2 <= 2^-19 of the shifted sum of squares) are snapped to constant (variance 0, skewness/kurtosis NaN, z 0); a second exact-mean pass would remove this limit.
- Round 7, `GPUKeyJoin` and its siblings in `gpu-dataframe/key-join/`:
  - Right and full outer joins (rightMatched plus compaction of unmatched right rows would cover them; not wired).
  - Dictionary (string) keys and composite multi-column keys; callers must pre-encode to u32/u64.
  - 1:n aggregates beyond count/sum/mean/min/max (median, mode, first/last by order column); hash-index probe path for pre-sorted or tiny right tables.
  - A dedicated N-th-match gather (`many: 'first'` is the only 1:1 mode; no `last`).
  - Aggregates of u32 columns (float32 only).
- Round 7, `GPUCellTableCompare` and its siblings in `geospatial/cell-table-compare/`:
  - Compare of tables from different resolutions or families (needs a rollup of the finer table first; the contributor does not verify that the two tables agree).
  - Compare of more than two tables (time series of N tables) and a mean/variance baseline over several prior periods.
  - Other measures: minimum/maximum deltas, and `sumValues`-only tables (currently sums need the fixed-point `sums`).
  - Standardized z-score reduction is a single workgroup (256 threads loop over the union rows); fine to ~1e6 rows, a multi-workgroup fixed tree would scale further.
  - Ratio, percent change and z-scores are f32 divisions and not bit-exact to the CPU oracle (see section 5); the keys, presence, before, after and delta are exact.
- Round 7, `GPULineSegmentize` and its siblings in `geospatial/line-segmentize/`:
  - Bounding-box clip (Liang-Barsky with count, scan and emit) and `circleVertices` were not built.
  - Arcs are never split at the antimeridian (unwrapped longitudes only); near-antipodal pairs are not detected.
  - `GPULineSmooth` is planar only and does not output `measures` or `sourceRows`; `GPULineChunk` has no per-path measure-range columns and no negative-from-end measures.
  - Densify and arc emit loop over pieces inside one invocation per segment; very uneven piece counts underuse the GPU.
  - Chunked (`GraphVectorView`) inputs are rejected. No `drawIndirect` writer: use `count`.
- Round 7, `GPUGeometryMeasures` and its siblings in `geospatial/geometry-measures/`:
  - Validity checks (closure, orientation, self-intersection), `isClosed`/orientation columns, and per-group mean centre, standard distance and ellipse (stream D builds some of these as `geographic-distribution`).
  - Geodesic-edge polygon areas (Karney) and rings around a pole; geographic group centroids average longitudes naively (wrong across the antimeridian).
  - Cross-track and along-track distance, rhumb lines, and Vincenty direct with relative latitude output (destinations are bounded by the f32 latitude ulp).
  - Per-feature work is one invocation per feature, so latency follows the largest feature; a segmented-reduction path for very large features is not built.
  - Chunked (`GraphVectorView`) inputs are rejected.
- Round 7, `GPULinearReferencing` and its siblings in `geospatial/linear-referencing/`:
  - Geodesic (longitude/latitude) snapping and measures; the join and measures are planar.
  - k nearest candidates per point (map matching needs several candidates with their measures).
  - Foot points could be emitted by `GPUNearestFeatureJoin` itself instead of a second projection pass.
  - Side is relative to the chosen segment, so points exactly beyond a vertex on the outside of a corner report `0` or the side of the earlier segment.
  - Measures at a vertex resolve to the following segment (upper bound); `angles` are `0` for zero-length tangents.
  - Chunked (`GraphVectorView`) inputs are rejected.
- Round 7, `GPUTerrainHorizon` and its siblings in `gpu-terrain/terrain-illumination/`:
  - FFT texture shading (exact `|f|^alpha`) needs a graph-node wrapper for gpu-core `GPUFFT2D` and 2x padding (4096² for a 2048² tile, above its 2048 limit); the DoG pyramid follows the target slope only within about 0.2 in the mid band and rolls off near Nyquist and below `1 / (2 pi sigma_max)`.
  - Texture shading runs at full resolution at every level (about 10 taps x `baseSigma * 2^levelCount` per pixel); a downsampled pyramid would make 7-8 levels cheap.
  - (Round 8 added `algorithm: 'sweep'`, exact and amortised O(1); the max-mip pyramid for the march is still open.) Horizon march cost is O(pixels x sectors x steps) with direct global reads; a max-mip "horizon pyramid" (Timonen/Sloan) or shared-memory line sweep for axis sectors is not implemented.
  - (Resolved in round 8: `horizonFormat: 'unorm16'`.) Horizon storage is float32 only; an `rg16float`/packed-f16 layout would halve the 268 MB of a 2048² x 16 map, and texture-array output for direct shader sampling is not offered.
  - Ground distance uses the center row's cell size along the whole ray (fine for tiles, not for continent-scale geographic rasters).
  - (`GPUSolarIrradiance` landed in round 8.) Per-pixel GPU sun position for globe-scale rasters are not built.
  - `GPUReliefShading` color is packed RGBA8 in a `uint32` buffer only; no `rgba8unorm` storage-texture output helper (needs the shared texture-output helper proposed in research 04 section 4).
  - The relief texture output and the horizon texture output are untested on GPU; texture paths are covered for `GPUSolarShadowMask` and `GPUTextureShading` only.
- Round 7, `GPURasterReclassify` and its siblings in `gpu-raster/raster-algebra/`:
  - Not an expression compiler: longer formulas chain contributors (one storage round trip per step); a fused
    `GPURasterCalculator` lowering a small expression AST to one kernel would remove the round trips
  - Stacks are one packed band-sequential view; separate per-layer views are limited by the 8-binding
    kernel limit (a gather into the stack is the caller's job), and chunked `GraphVectorView` inputs are rejected
  - Weighted overlay supports up to 16 layers and a shared `maximumBreakCount` per layer table; fuzzy
    memberships (ArcGIS Fuzzy Overlay: gaussian, large/small, MS functions) are not implemented
  - Cell-statistics frequencies are `O(layerCount^2)` per cell (64 layers max); percentile/median across the
    stack is not implemented
  - Reclassify break tables must be sorted ascending (unsorted tables give a deterministic but meaningless
    class); range-to-value tables with gaps ("unmatched" values) are not supported
  - Scores with FMA-contracting backends differ from an unfused oracle by a few f32 roundings (bounded in tests)
- Round 7, `GPURasterStretch` and its siblings in `gpu-raster/raster-stretch/`:
  - Classification breaks are not built here: quantile class breaks (inverse CDF, error one bin width) and
    Jenks/natural breaks (histogram DP); stream A's `GPUClassBreaks` covers columns
  - CLAHE (tile-wise clipped equalization) is not built
  - One level of equal-width bins limits percentile resolution to `range / binCount`; an adaptive two-stage
    histogram for heavy-tailed data is open
  - Bin assignment uses a GPU-side `binCount / range` division (not correctly rounded), so cells within an f32
    ULP of a bin edge may land in the neighbouring bin compared with an f64 reference
  - Multi-band/RGB stretch, log/sqrt transfer functions, and an `rgba8unorm` LUT texture output are not offered
  - Infinite cells are excluded from statistics (they clamp to 0 or 1 when applied); `validCount` is float32
    (exact to 2^24); window coordinates are float32 (rasters up to 2^24 columns/rows)
- Round 7, `GPUIsolines` and its siblings in `gpu-raster/isolines/`:
  - `GPUTerrainContours` still emits the old two-vertex records; it could be reimplemented on top of
    `GPUIsolines` (left untouched: existing dir)
  - Polylines are per level and tile-local: no joining across tiles, no smoothing or simplification
    (chain into `GPULineSimplification` by hand)
  - Isobands are a triangle soup: no stitched polygon rings, no vertex dedup or index buffer
  - Neighbour lookup during stitching scans a cell's segments linearly (O(levels) per segment); isoband cells
    loop over the bands between their lowest and highest class (O(bands) on steep cells)
  - Infinite samples and NaN or unsorted levels/breaks give deterministic but unspecified geometry;
    closed-right band intervals are not offered
  - Coordinates are within a few ULP of an f32 oracle, not bit-exact (GPU division, FMA)
- Round 7, `GPURasterSampling` and its siblings in `gpu-raster/raster-sampling/`:
  - Rasters are packed row-major buffers only: no texture input (hardware filtering) and no tiled or chunked
    rasters
  - Profile distances are planar; geodesic (great-circle or ellipsoidal) spacing is not implemented, nor
    draping lines onto a raster in another CRS
  - Profiles have no per-sample `validity` column; non-finite samples are only skipped in summaries
  - Cumulative gain/loss and the path walk are serial per path (one invocation), slow for one very long path;
    a segmented scan would parallelize it
  - Per-path samples are capped at `min(2^24, (2^32 - 1) / (pathCount + 1))`, then truncated with overflow
  - Sample counts can differ by one from an f64 reference when `length / spacing` is within an f32 ULP of an
    integer
- Round 7, `GPUParticleAdvection` and its siblings in `gpu-raster/particle-advection/`:
  - Time interpolation between two fields (`velocitiesNext` plus a blend factor) for animated forecasts
  - Lon/lat-aware stepping (metres per second on a sphere) and wrap-around at the antimeridian
  - Spawn weighted by speed or by a density raster instead of a uniform rectangle
  - deck.gl consumption of the trail and position buffers without readback (generic GPU buffer attribute)
- Round 7, `GPULineIntegralConvolution` and its siblings in `gpu-raster/flow-texture/`:
  - Streamline self-proximity: a line may loop back over its own cells (Jobard-Lefer also stops a line
    near itself); size `stepsPerDirection` to the field or add a self-test
  - Point-distance separation (`d_sep` / `d_test` against neighbouring points) instead of whole grid cells
  - Seeding from neighbouring accepted lines (classic Jobard-Lefer growth) for denser coverage
  - Indirect-dispatch gating of rounds after convergence (all `roundCount` rounds always dispatch)
  - LIC: oriented (OLIC) and fast-LIC variants, contrast enhancement, and lon/lat-aware step lengths
- Round 7, `GPUDotDensity` and its siblings in `geospatial/dot-density/`:
  - Triangle-area-weighted sampling (from caller triangles) or raster-CDF sampling for thin polygons and
    for sparse masks, instead of bounding-box rejection
  - Acceleration for very large polygons (per-candidate cost is linear in vertex count): edge grid or a
    `GPUPolygonRasterization` zone lookup with exact tests only on boundary cells
  - Interleaved draw order across categories (dots are grouped by slot, so later categories draw on top)
  - Blue-noise / Poisson-disk thinning of dots at the current zoom
- Round 7, `GPUNeighborSearch` and its siblings in `geospatial/neighbor-search/`:
  - Lon/lat inputs: equirectangular scaling plus an exact haversine re-rank of the final k.
  - k > 32 (needs workgroup-memory top-k), and a BVH path for very uneven densities.
  - Radius rows are insertion-sorted per thread, which is O(d^2) for a row of degree d. Use a segmented
    sort for very wide bands.
  - kNN scratch is `queryRows * k` ids and distances. A two-pass recompute variant would trade time
    for memory.
  - Targets outside `bounds` are excluded, so bounds must cover the data. There are no chunked
    `GraphVectorView` inputs yet.
  - The weights CSR could feed `GPUHotSpotAnalysis` and `GPULocalMoran`, which today evaluate an
    implicit distance band. That would be a change in another dir.
- Round 7, `GPUGlobalSpatialStatistics` and its siblings in `geospatial/global-spatial-statistics/`:
  - An internal 25-column `float32` scratch matrix (`25 * rows * 4` bytes) must fit the device's
    maximum storage binding size. That allows about 1.3M rows at the 128 MiB default. Tile it by
    column batch for larger inputs.
  - The oracle and tests use dense `n x n` matrices, so GPU-versus-oracle tests stop at a few
    thousand rows. Larger runs are only timed.
  - Integer-valued sums (counts, binary degrees) are exact only below 2^24. Use u32 or
    fixed-point sums beyond that.
  - There is no lon/lat or chunked `GraphVectorView` support. Permutation inference for these
    statistics lives in `GPUGlobalPermutationTest`.
  - A spatial correlogram (statistics over B distance bands in one pass) is not implemented. That
    needs per-band partial columns, or stream D's `GPUSpatialCorrelogram`.
- Round 7, `GPULocalPermutationTest` and its siblings in `geospatial/permutation-inference/`:
  - Local Gi* uses a fixed self weight of 1. esda's float `star` values and its row standardization
    after adding the diagonal are not modeled.
  - Local Geary, local join counts, bivariate LISA, and multivariate or conditional LISA are not
    implemented.
  - The global test evaluates the Feistel bijection for each neighbor term, so it costs
    `(P + 1) * nnz` Feistel evaluations. Caching per-permutation relabeled values in tiles would help
    at large P.
  - `maximumNeighbors <= 64`, because the swap list lives in private memory. Wider rows are not
    tested, and sampling with replacement is not offered.
  - The local test runs one invocation per row and loops over P. Splitting P across invocations, with
    integer `atomicAdd` of counts, would balance skewed degree distributions.
  - Promote the RNG module (see section 6).
  - Wire `inference: 'permutation'` into `GPULocalMoran` and `GPUHotSpotAnalysis` (another dir).
- Round 7, `GPUVariogram` and its siblings in `gpu-dataframe/pair-statistics/`:
  - Monte Carlo CSR envelopes for K/L and permutation inference for the correlogram (need a shared Philox RNG); cross-type K and marked patterns.
  - Non-rectangular windows; kernel-smoothed pair correlation (spatstat `pcf`); quadrat and Clark-Evans p-values.
  - Verify the border `lambda` convention and the isotropic weight cap against spatstat itself (`lambda = (n - 1) / A` chosen for consistency with `'none'`).
  - Variogram `maxPairs` subsampling cap for very dense data, variogram cloud output, and ordinary kriging on top of the fitted model.
  - Correlogram band count is capped at 64 (private per-row degree counters); analytic variance is f32 and loses accuracy when a band holds most pairs.
  - Ship a cell-range-driven neighbour pass: dense clusters make per-focus work uneven (no load balancing beyond cell-ordered foci).
- Round 7, `GPUGeographicDistribution` and its siblings in `geospatial/geographic-distribution/`:
  - Central feature (argmin of summed distances) and geodesic (lng/lat) variants; positions are planar f32, so centres and rings at ~1e5 coordinates are only accurate to a few f32 ulps even with the local origin.
  - Validate the ellipse axis naming and angle against real ArcGIS Directional Distribution output (derived from the published formula, not compared with ArcGIS results).
  - Weiszfeld unrolls six nodes per iteration (about 150 nodes at the default 24); a GPU-gated loop or a single multi-iteration kernel per group would cut node count.
  - Chunked (`GraphVectorView`) inputs are rejected.
- Round 7, `GPUEmergingHotSpots` and its siblings in `geospatial/emerging-hot-spots/`:
  - Permutation (pseudo) p-values and false discovery rate correction for the space-time Gi* bins.
  - Irregular neighborhoods (CSR, k-nearest) and time-step intervals other than one slice; a slice limit of 256 comes from the O(slices^2) Mann-Kendall pass.
  - Zero-count bins from `GPUTemporalReduction` are valid zeros; no sparse-cube input or per-bin missing mask for `uint32` counts.
  - Space-time Gi* is O(bins * disc * window); a prefix-sum formulation would make large radii cheap.
  - Category rule ambiguities are resolved by documented ordering (see TSDoc), not validated against ArcGIS output.
- Round 7, `GPUOrdinaryLeastSquares` and its siblings in `geospatial/spatial-regression/`:
  - Moran's I of the residuals (composes with `GPULocalMoran` and global autocorrelation statistics from the autocorrelation contributors; not wired into this contributor)
  - Original (non-studentized) Breusch-Pagan, White test, F statistic and its p-value, heteroskedasticity-robust (HC) standard errors
  - Weighted least squares and ridge standard errors that account for shrinkage bias
  - More than 15 predictors, and ill-conditioning diagnostics beyond the pivot test (condition number, VIF)
  - Spatial lag and error models (2SLS, GM, ML) build on this contributor; not started
  - GWR neighbour search is brute force over all rows (<= 65536); a grid or kNN index (CSR) would cut the cost for large n.
  - GWR has no multiscale (MGWR) per-coefficient bandwidths, no local standard errors or pseudo-t values, no Poisson/logistic GWR, no geodesic or great-circle distances (planar positions only).
  - GWR candidates with any singular location are rejected wholesale; there is no per-location fallback or ridge.
  - Adaptive `k` is capped at 128 and the ladder at 32 candidates (compile-time).
- Round 7, `GPUCompositeScore` and its siblings in `gpu-dataframe/composite-indicators/`:
  - Rank scaling runs one full `GPUSort` per indicator column (16 sorts at most); a single segmented sort over all columns would cut passes
  - No robust (median/IQR) scaler, no supervised (regression-fitted) weights, no output rank or class column (feed the score to the classification contributors instead)
  - PCA returns only the first component, by a fixed 64-step power iteration; a near-tied second eigenvalue converges slowly (the residual slot reports it)
  - Parallel per-zone segment reduction: one thread walks a zone, so a single huge zone (or the pooled Gini) serializes; a tile-parallel Lorenz scan would fix it.
  - f32 accumulation only: precision degrades for segments of millions of rows; compensated or fixed-point sums not done.
  - Decomposition of Theil L and Atkinson (only Theil T is decomposed), and Gini decomposition with the overlap term.
  - Confidence intervals and bootstrap for the indices; grouped/subgroup decomposition by more than one partition.
  - Zero-valued rows are reported as NaN for Theil L and Atkinson (e >= 1) rather than excluded or epsilon-floored; no option to choose.
- Round 7, `GPUCalendarBuckets` and its siblings in `gpu-dataframe/calendar-buckets/`:
  - Time zone transition table on the GPU (binary search of sorted transitions, R7 original design); today DST needs a caller-precomputed per-row offset column.
  - Float32 relative or float64-split timestamp inputs; only Int64 words are accepted.
  - Calendar bucket index output (month, week or day count from an origin) for direct use as `GPUTemporalReduction` or `GPUCellAggregation` keys.
  - Chunked vector inputs (`GraphVectorView`) are rejected; single packed views only.
  - Fiscal years, locale week rules other than a configurable first weekday, and non-Gregorian calendars.
- Round 7, `GPUChangeDetection` and its siblings in `gpu-raster/change-detection/`:
  - Sen slope and Mann-Kendall are O(T^2) per thread; `sliceCount` for Sen is capped at 64, and large stacks would need a parallel-over-pairs variant
  - Scalar statistics (t-test, Sen, Mann-Kendall, significance) are single band only; per-band variants and a multiband Hotelling/Mahalanobis test are not built
  - Student-t p-value is float32 (about 1e-4 absolute); no f64-quality or exact-permutation path, and no Pettitt/CUSUM change-point or seasonal Mann-Kendall
  - Significance does not apply multiple-comparison control (FDR); two-sample test assumes a fixed split slice (no per-cell breakpoint search)
- Round 8, DEM analysis ported from mt-image and Rigi (`terrain-decode/`, `terrain-features/`,
  `terrain-curvature/`, `geomorphons/`, `topographic-position/`, `hydrology/`, `terrain-flow-field/`):
  - True 2-D prominence and isolation (sequential Priority-Flood or union-find; Kirmse & de Ferranti), a 2× decode downsample with stats (use gpu-raster overview or reduction), and cross-tile spike repair (tile-edge seams are not voted) are not built
  - Spike repair is capped near 2048² tiles (per-component bit counters need 8·N words in one binding), is bit-exact only for power-of-two `step`, and can shift a real enclosed butte with 216-296 m walls; enable it only for known-noisy sources
  - Mapbox `(0,0,0)` decodes to -10000 m, inside the default range; pass `noDataRGB: [0, 0, 0]` when the source uses it as nodata
  - Peak snap keeps unsnapped candidates at the bilinear DEM height (all four corners valid); `max(DEM, catalogue elevation)` is left to the caller. Profile-peak suppression is deterministic only once converged
  - Summed-area table (`topographic-position/terrain-summed-area-table.ts`) should graduate into a gpgpu `GPUSummedAreaTable` / box-sum primitive (`GPUScanUint64` needs separate low and high buffers); TPI neighbourhoods are square annuli, not discs, and there is no Gaussian-weighted DEV
  - Geomorphons lack the GRASS `extended` correction, per-sample geodesic distances (the centre row's cell size is used) and intensity/exposition/range/variance outputs; curvature lacks Florinsky's spheroidal-trapezoid method and a log-transform output
  - Weiss global statistics depend on the tile extent (`'local'` avoids this); multi-tile Weiss and geomorphons need halos (`requiredHalo`) and mosaic-wide reductions
  - HAND follows D8 paths only; hydrologic indices use the D8 descent slope, and zero slope gives +∞ TWI unless `minimumSlope` is set (default 0.001); D-infinity accumulation differs from the f64 oracle by up to 2.5e-4 relative at facet near-ties
  - Oracles are f64 transcriptions of GDAL, GRASS and Whitebox definitions; no comparison against the actual binaries was run
- Round 8, visibility (`raster-pyramid/`, `point-horizon/`, `terrain-analysis/`):
  - Pyramid skip cost: the skip does an analytic exit, up to 4 endpoint re-checks and a bound at every level; try a coarse-to-fine start level per ray or workgroup-uniform skipping, GPU timestamp timing instead of wall clock, and a real-DEM benchmark
  - Point horizon covers one raster band (tile window): no multi-ring mosaic (z15 to z9 out to 300 km), no geographic (lng/lat) great-circle path, no variable-length ridge-crest lists; targets narrower than one lattice step can be missed
  - `GPUPointHorizonVisibility` has no f64 oracle (structural and planar-versus-Web-Mercator tests only); `firstAzimuth` wrap-around and `rowDirection: 'north'` are untested
  - Sight-line contributors are planar only (no Web Mercator or geographic cell size); line of sight needs all four bilinear corners valid at the target
  - `GPUTerrainHorizon` does not use the pyramid yet; the eye-adaptive `maximumDistance` helper is not built
- Round 8, relief and illumination (`terrain-illumination/`, `relief-visualization/`):
  - Max-mip horizon pyramid for the march (uses the extrema pyramid); the sweep covers most of the cost
  - Per-row great-circle geometry for long rays and per-pixel GPU sun position for globe-scale irradiance
  - `GPUReliefBlend` percent-clip stretches need a histogram percentile pass (pass min and max for now); Hesse's full LRM (purged trend surface) is not built, only SLRM and MSRM
  - Local dominance is 84.5 ms at 1024² with 264 taps; a shared-memory tile or summed-area form could speed it up
  - The sweep samples pixel centres within half a pixel of the ray, so SVF differs from the bilinear march by about 0.01 on rough terrain; sweep curvature with latitude-dependent cells uses the middle row's spacing for the argmax; sweep needs `stepGrowth` 1 and extents of at most 32767
  - `'imhof-swing'` needs the `imhofSwing: true` topology flag, otherwise it behaves like `'aspect'`; mt-image's propagated-occluder soft shadow and multi-scale normals are not ported
  - RVT-py's greyscale overlay and soft-light mutate the background in place so opacity has no effect there; `GPUReliefBlend` applies the documented opacity
- Round 7 explorer findings (13 demo modes over the round-7 contributors):
  - `GPUClassBreaks` bound 9 storage buffers in one kernel with all methods compiled; fixed by
    packing head/tail breaks into the head/tail state buffer. `createWGSLKernelNode` now
    rejects kernels over the device's storage-buffer limit at declaration, so node specs (8-buffer
    null device) catch this. GPU specs use the `'max'` feature level and cannot.
  - Raster contributors assume row 0 at minimum y; DEMs are usually north-up. Add `rowOrigin: 'north'`
    to isolines, isobands, sampling, profile and distance field.
  - Contributor-internal resource ids (`${id}-total`) collide with caller imports of the same name.
  - Missing glue: a uint32 to float32 cast node, a CSR-to-segments helper, a column-stack helper,
    indirect draw records for isolines and path outputs, a per-vertex path id, and a quadrant
    class output on `GPULocalPermutationTest`.
  - Compile-time knobs users want live: `GPUNeighborSearch` k (an upper k with a per-frame limit)
    and cell family in `GPUPointToCell`. `GPUSolarShadowMask` cannot take its sun from
    `GPUSolarPosition` on the GPU.
  - Costs: H3 grid disk k=8 is about 11 ms for one cell; adaptive GWR is about 5x a fixed ladder;
    Weiszfeld median centre unrolls to about 150 nodes; streamline and stitching rounds dispatch
    at capacity, not at convergence.

## GPU spatial analysis: prior-art review and next tranches

Reviewed 2026-10-05 against GeoPandas / Shapely 2 / GEOS, DuckDB spatial, Apache Sedona (SedonaDB),
the PySAL family (libpysal, esda, spreg, mgwr, spopt, segregation, tobler, access, momepy, giddy,
pointpats, spaghetti), RAPIDS cuSpatial and other GPU systems (HeavyDB, Kinetica, FDBSCAN, jump
flooding), PostGIS / pgRouting / MobilityDB / h3-pg, the CARTO Analytics Toolbox and BigQuery /
Snowflake / Databricks geospatial, and Turf.js with the browser ecosystem (JSTS, polygon clipping,
d3-delaunay, h3-js, geotiff.js, DuckDB-WASM). Round 7 had already covered turf measurement, esda
statistics and kepler/CARTO parity, so this review looks for what is still missing. Every
"have" below was checked against the source on `map-contributors`.

### Where the field is

- cuSpatial stopped publishing at RAPIDS 25.06 and no replacement has been named. The live GPU
  vector reference is now SedonaDB's GPU spatial join, which runs on NVIDIA RT cores. It evaluates
  every named predicate through one DE-9IM relate path, has no `dwithin` or kNN on the GPU, and
  needs batches of about 100k before it beats the CPU.
- PySAL (meta-package v26.07, libpysal 4.15) is converging on `libpysal.graph.Graph` as its one
  weights object. That is the same joint as our `GPUSpatialWeights` CSR. `mgwr` and `spaghetti`
  are dormant, so a maintained GPU GWR or network point-pattern implementation has no competition.
- GeoPandas 1.1 has no GPU work on its roadmap. Turf 7.4 is shipping correctness releases. JSTS is
  in maintenance.
- No other browser library does WebGPU vector or statistical analysis. Our differentiators are
  bounded outputs with GPU overflow flags, fixed-order deterministic reductions, f64 and CPU
  oracles, and per-frame parameters that need no recompile. SedonaDB and cuSpatial return
  unordered pairs, which makes ours unusual.

### Layering lessons

Each reference project is a small set of joints plus many consumers. Ours already line up with
them, and the gaps are at the joints.

| Joint | Reference | Ours | Action |
| --- | --- | --- | --- |
| Geometry array | Shapely ufuncs, GeoArrow, GEOMETRY column | GeoArrow offsets views | Keep. Factor one shared segment table so the join, measures and new geometry contributors stop each re-deriving segments |
| Candidate / matched pair table | `STRtree.query` index pairs; sjoin, overlay and clip all consume it | `GPUSpatialJoinPairs`, sorted `(left, right)` | Keep. Expose the bounding-box candidate stage on its own so relate, distance and intersection kernels share it |
| Predicate | One DE-9IM engine (GEOS RelateNG, SedonaDB); named predicates are masks over it | One kernel per predicate (`intersects`, `contains`, `within`, `dwithin`) | Move to a single relate classification (tranche S1) |
| Weights | `libpysal.graph.Graph` with builders, transforms and set algebra | `GPUSpatialWeights` CSR, 4 producers, `row`/`binary`/`kernel`/`symmetrize` | Add the algebra on the CSR (tranche S2). Keep the CSR as the only neighbor currency |
| Group key | SQL window `OVER (PARTITION BY cluster)`, GeoPandas `dissolve(by=)` | `GPUGroupStatistics` over a `uint32` label | Add per-group geometry so a cluster label gives outlines and centers (tranche S3) |
| Cell key | CARTO, Mosaic and h3-pg turn joins into equi-joins on H3 or Quadbin | cell indexing, topology, cover, aggregation, `GPUKeyJoin` | Add a core/border flag and set outlines (tranche S3) |
| Prepared geometry | GEOS `PreparedGeometry`; "prepared" is a headline GeoPandas roadmap item | The BVH is rebuilt every encoding | Add a static right-hand side so animated points reuse the polygon BVH (tranche S1) |
| Named tool or workflow | CARTO Workflows, QGIS models, ArcGIS tools | Contributors compose inside a `GPUCommandGraph` | Ship named recipes as builder functions, not classes (see recipes below) |

Conventions to adopt from them:
- Every statistic emits its "explain" columns (neighbor count, z, p, class) next to the primary
  result, as CARTO's procedures do.
- Each contributor documents a measured CPU/GPU crossover and offers `strategy: 'scan' | 'indexed'`
  where both exist. SedonaDB and HeavyDB both lose to the CPU on small joins.
- Parity tables pin a reference release: esda 2.10 / PySAL v26.07 for statistics, Shapely 2.1 /
  GEOS for predicates, h3-js for cells. Keep running those as oracles.

### Tranche S1: predicates and joins (P1)

Three of the five reviews independently ranked this first.

- **`GPUSegmentIntersection`** (M): segment–segment intersections over BVH candidate pairs. Per
  pair it classifies the crossing as proper, touch, collinear or overlap, uses exact orientation
  signs, and flags uncertain pairs as the point-in-polygon join does. Output is capacity-bounded
  and sorted. It gives turf `lineIntersect`, `kinks`, self-intersection and `lineSplit` points.
  It also supplies noding for `gpu-network` and the building block for the next item. turf's naive
  O(nm) `lineIntersect` stalls at about 10^4 segments, so this is a real browser win.
- **`GPUSpatialPredicateJoin` relate engine** (M–L): one per-pair DE-9IM classification for all
  nine geometry-kind pairs. The existing predicates become masks over it, and it adds `covers`,
  `coveredBy`, `touches`, `crosses`, `overlaps`, `equals` and `containsProperly`, plus an optional
  per-pair `relate` output and a pattern filter. Add `how: 'anti'` to emit unmatched left IDs,
  since `disjoint` as a pair list is unbounded. `touches` between polygons cross-checks
  `GPUContiguityWeights`. SedonaDB's predicate list is the acceptance list and Shapely is the
  oracle.
- **`GPUGeometryValidity`** (S once the item above exists): a per-feature bitmask for unclosed or
  short rings, NaN, repeated vertices, ring orientation, self-intersection, holes outside the shell
  and crossing rings. It answers the "polygons must be valid" caveat in the join doc with a mask
  users can filter on, and repair stays on the CPU.
- **`GPUNearestFeatureJoin` v2** (M): `k` nearest (best-first BVH, lowest-ID ties), all-ties mode,
  `maxDistance`, and distance, foot-point and segment columns. This matches `sjoin_nearest` and the
  PostGIS `<->` operator with `LIMIT k`. Also add line/line, line/polygon and polygon/polygon
  minimum distance columns, which cuSpatial had and we do not.
- **Static right-hand side** (S): keep the right-side BVH across encodings when the geometry has
  not changed, as a `prepared` handle on the join and the nearest join. This reuses a build. It
  does not cache results.

### Tranche S2: spatial weights and statistics on weights (P1)

This is the core of the entry's purpose ("our value is spatial weights, stats on weights").

- **`GPUSpatialWeightsAlgebra`** (M): union, intersection, difference and symmetric difference as
  per-row two-pointer merges (rows are already sorted), `higherOrder(k)`, an explicit self weight,
  `subgraph(mask)`, block weights from group IDs, and a summary of cardinality, S0/S1/S2,
  asymmetry and isolate count. This covers the libpysal `Graph` set operations. Also add the
  missing `D` and `V` transforms to `GPUSpatialWeightsTransform`.
- **`GPUNeighborhoodSummary`** (S): per-row weighted mean, sum, min, max, standard deviation,
  median (k ≤ 32), mode (categorical lag) and entropy over a CSR. It is momepy `describe` and the
  generalized lag, and it feeds segregation and morphometrics.
- **`GPUSpatialRegressionDiagnostics`** (M): LM-lag, LM-error, their robust forms, LM-SARMA,
  Anselin-Kelejian and Moran's I of residuals, computed from `GPUOrdinaryLeastSquares` residuals,
  `GPUSpatialLag` and fixed-order traces. This turns our OLS into the GeoDa "fit OLS, then which
  spatial model?" panel.
- **Rate smoothing and tails** (S): empirical-Bayes rate standardization in front of
  `GPULocalMoran` and `GPUGlobalSpatialStatistics` (esda `Moran_Rate` and `Moran_Local_Rate`), and
  esda 2.10's `alternative` tail option on the permutation tests. Raw rates over small
  denominators are the most common source of false hot spots.

### Tranche S3: group geometry, cell outlines and isochrones (P1)

Closes the "cluster, then show it" loop that PostGIS window functions and turf users rely on.

- **`GPUGroupGeometry`** (S): per-label bounds, mean and weighted center, medoid, standard ellipse
  and count. It reuses `GPUGroupStatistics` and the `GPUGeographicDistribution` kernels over a
  label column. Document `GPUSpatialClustering` with `minPoints = 1` as PostGIS
  `ST_ClusterWithin`.
- **`GPUGroupConvexHull`** (M): monotone-chain hulls per label over a segmented sort, bounded with
  overflow and lowest-index ties. Use it for cluster and hot-region outlines and trajectory
  footprints. A raster closing of `GPUDistanceField` is the concave option (P2). Exact alpha shapes
  stay on the CPU.
- **`GPUCellSetOutline`** (S): boundary edges of a cell set (H3 `cellsToMultiPolygon` without ring
  assembly) from `GPUCellTopology` neighbors. It is the cheapest way to draw an isochrone or a
  significant region as one outline.
- **Network isochrone polygons** (M): splat edge-interpolated `GPUNetworkServiceAreas` costs to a
  raster and run `GPUIsobands`, or outline the reached cells with `GPUCellSetOutline`. The
  isochrone outline is the most requested routing visual we cannot draw today.
- **Core/border flag on `GPUCellCover`** (S): mark cells that lie entirely inside the polygon
  (Mosaic "chips"), so tessellated joins skip the exact test for core cells.

### Tranche S4: trajectories against places (P1/P2)

MobilityDB shows a tier between our trajectory contributors and the zone tools.

- **Zone entry and exit events** (M, P1): crossings of trajectory segments with polygon edges
  through the BVH, with interpolated times. Bounded events per track give dwell time per zone and
  visits per POI. Composes `GPUTrajectoryPlayhead`, the BVH and the predicate join.
- **Encounters** (M, P2): pairs of tracks within distance d during the same time bucket, using a
  (cell, time bucket) grid over `GPUTrajectoryResample` output. This is MobilityDB `tdwithin`.
- **Hausdorff and discrete Fréchet distance** (S–M, P2): per-pair track or ring similarity with a
  workgroup reduction, for track grouping and comparing boundary vintages.

### Tranche S5: change of support and neighborhood indices (P2)

- **`GPUArealInterpolation`** (M): area-weighted transfer of extensive, intensive and categorical
  variables between zone systems or onto H3 and hexagons. Shared-cell counts come from
  `GPUPolygonRasterization` on a common fine grid, with an optional dasymetric mask raster. Output
  is an area-share `GPUSpatialWeights`, so `GPUSpatialLag` performs the transfer. Accuracy is
  bounded by resolution, never exact polygon intersection, and is documented as such. Also add
  `GPUPycnophylactic` (focal mean plus a zone rescale, fixed iterations). Together these cover
  tobler and CARTO `ENRICH_POLYGONS`.
- **`GPUSegregation`** (M): aspatial D, isolation, interaction, entropy and Atkinson, spatial
  versions via kernel weights, local per-unit indices, and a multiscale profile over a bandwidth
  ladder. Composes `GPUNeighborhoodSummary`, `GPUInequality` and `GPUGlobalPermutationTest`.
- **Distribution dynamics** (M): `GPUTransitionMatrix`, `GPUSpatialMarkov` and `GPULISAMarkov` as
  exact integer histograms over `GPUClassBreaks`, `GPUSpatialLag` and `GPULocalMoran` quadrants
  per period (giddy), plus pooled class breaks and GADF goodness of fit, so a time slider keeps one
  legend.
- **Space-time tests** (M): Knox, modified Knox and Mantel interaction tests with
  Feistel-permuted times (pointpats). A space-time scan statistic (Kulldorff, CARTO
  `DETECT_SPACETIME_ANOMALIES`) is L, after Knox.
- **Floating catchments on any weights** (S): `GPUNetworkAccessibility` already does 2SFCA with
  decay on network costs. Accept any `GPUSpatialWeights` (Euclidean kernel) and add 3SFCA and Huff
  gravity trade areas.
- **Spatial 2SLS / GM error** (M–L): spreg `GM_Lag` and `GM_Error` with instruments from
  `GPUSpatialLag`, GPU-reduced moments and tiny CPU solves. It follows the S2 diagnostics.

### Tranche S6: geometry utilities (P2)

- **Render-only buffer geometry** (M): per-vertex offset outlines with round joins for points,
  lines and rings, with a per-frame distance (planar or geodesic). Overlaps are left alone; the
  renderer or `GPUDistanceField` handles the union look. Queries keep using `dwithin` and
  `GPUBufferSelection`. This covers turf `buffer` and `lineOffset` for the picture only.
- **Label point** (S–M): pole of inaccessibility through fixed refinement rounds over
  point-segment distance, for labeling 10^5 polygons during pan (polylabel).
- **Coverage simplification** (M): simplify shared arcs once, found through contiguity edge keys,
  with `GPULineSimplification`, so choropleths stay gap-free at low zoom (Shapely 2.1
  `simplify_coverage`, PostGIS `ST_CoverageSimplify`).
- **Shape descriptors** (S): compactness (Polsby-Popper, Schwartzberg), convexity (needs the
  hull), elongation and orientation, clockwise and sliver flags from `GPUGeometryMeasures`. Add
  momepy neighbor alignment through `GPUNeighborhoodSummary`.
- **Line density** (M): line length per grid cell or per polygon (QGIS "Line density", "Sum line
  lengths").
- **Raster connected components** (M): `r.clump`, sieve filters and patch metrics. It builds on
  `GPUGraphConnectedComponents`, which spike repair already uses.
- **Small generators** (S): square, hex, triangle and point grids from an extent ("fishnet then
  join"), rectangle clip (Liang-Barsky, Sutherland-Hodgman per ring), and map coloring (greedy
  parallel coloring over contiguity, via gpu-graph once it exists upstream).
- **Clustering extras** (S–M): FDBSCAN's dense-box shortcut in `GPUSpatialClustering` (a cell
  holding at least `minPoints` points is all core), deterministic k-means, and Ripley F, G and J
  next to K, L and g.

### P3 and later

SKATER-style minimum spanning tree over contiguity (needs an upstream MST), network K function
(spaghetti), GWR Monte Carlo non-stationarity test and local condition number, kNN-neighborhood
kriging prediction, feature-space kNN ("similar locations"), endpoint line merge, a turn-restriction
line graph, map matching, Hilbert keys (upstream, next to `GPUBVH`), and circle and sector
generators.

### Named recipes

Ship these as parameterised builder functions that add a chain to a caller's graph and return
named output views, as CARTO Workflows templates do. They are not new classes. Each one exists
today except where an item above is marked.

| Recipe | Chain |
| --- | --- |
| Hot spot analysis | `GPUPointToCell` → `GPUCellAggregation` → `GPULatticeWeights` or `GPUNeighborSearch` → `GPUHotSpotAnalysis` (FDR) → `GPULocalPermutationTest` → `GPUClassBreaks` → `GPUColorScale` |
| Rate cluster map | EB rates (S2) → `GPUContiguityWeights` → `GPUSpatialWeightsTransform` → `GPULocalMoran` → `GPULocalPermutationTest` → quadrant colors |
| Points-in-polygons choropleth | `GPUPointInPolygonJoin` or `GPUSpatialPredicateJoin` → `GPUZonalStatistics` / `GPUGroupStatistics` → `GPUClassBreaks` → `GPUColorScale` |
| Space-time hot spots | `GPUCalendarBuckets` → `GPUCellAggregation` → `GPUEmergingHotSpots` → `GPUColorScale` |
| Cluster and outline | `GPUSpatialClustering` → `GPUGroupGeometry` + `GPUGroupConvexHull` (S3) → `GPUGeometryMeasures` |
| Spatial regression | `GPUOrdinaryLeastSquares` → `GPUSpatialRegressionDiagnostics` (S2) → residual `GPULocalMoran`; `GPUGeographicallyWeightedRegression` for local fits |
| Drive-time catchment | `GPUNetworkSnapping` → `GPUNetworkServiceAreas` → isochrone polygon (S3) → `GPUPointInPolygonJoin` → `GPUZonalStatistics` |
| Straight-line catchments | `GPUDistanceField` allocation (raster Voronoi) → `GPURasterZonalStatistics` |
| Change of support | `GPUPolygonRasterization` → `GPUArealInterpolation` (S5) → `GPUSpatialLag` |
| Fleet dwell | `GPUTrajectoryMetrics` stops or zone events (S4) → `GPUPointInPolygonJoin` → `GPUGroupStatistics` |
| Period comparison | two `GPUCellAggregation` → `GPUCellTableCompare` → diverging `GPUClassBreaks` |

Also add a cross-reference table to the user doc, keyed by turf, PostGIS, GeoPandas, PySAL and
QGIS tool names, so people moving from those libraries can find the matching contributor.

### Deliberately not building

- **Vector overlay:** union, intersection, difference, dissolve geometry, exact buffer, polygonize,
  noding output, `make_valid` and `ST_Subdivide`. Output is unbounded and needs robust topology.
  cuSpatial never shipped overlay and SedonaDB's GPU path does not attempt it. Use polyclip-ts,
  JSTS or GEOS-wasm through Arrow adapters. The bounded substitutes are areal interpolation, cell
  roll-ups and boundaries drawn where neighboring groups differ.
- **Exact vector Voronoi, Delaunay and concave hulls:** they need exact predicates, and
  d3-delaunay is fast on the CPU. A CPU-built triangulation can be uploaded as `GPUSpatialWeights`
  (Delaunay weights). Raster Voronoi already exists in `GPUDistanceField`.
- **Regionalization search and location solvers:** AZP, max-p, Ward-spatial, p-median, MCLP,
  TSP/VRP and contraction hierarchies are sequential or MILP problems. We contribute cost
  matrices, MST edges and partition evaluation, not the solvers.
- **Likelihood-based spatial models:** ML lag and error, panel, SUR, regimes, probit and MGWR
  backfitting are control-flow heavy with little value on a web map.
- **Other layers:** quadtree indexes (the sort-based BVH and grid subsume them), a SQL front end
  or dataframe layer, CRS handling, S2 spherical predicates, GeometryCollection, and service-style
  features (geocoding, enrichment catalogues, tiling).

Upstream asks for luma gpgpu: minimum spanning tree and graph coloring in gpu-graph, Hilbert keys
and sort next to `GPUBVH`, and a reusable bounding-box candidate-pair stage shared by the join
contributors.

### Status after the 2026-10-05 build round

Every P1 and P2 item above now has a first version on `map-contributors`, except the ones listed as
open below. Each contributor has a headless GPU spec against a CPU oracle. Where a reference
library exists, the fixtures pin its values: spreg 1.9.1, esda 2.10, libpysal 4.15, giddy,
pointpats, segregation 2.6, tobler and Shapely 2.1.2.

| Tranche | Shipped | Still open |
| --- | --- | --- |
| S1 | `GPUSegmentIntersection` and `GPUGeometryValidity` (exact 256-bit orientation fallback, Morton-sorted BVH); relate engine in `GPUSpatialPredicateJoin` (DE-9IM for all nine kind pairs, seven new predicates, `relate` output and patterns, `how: 'anti'`); `GPUSpatialJoinPrepared` for the predicate and point-in-polygon joins; `GPUSpatialJoinCandidates`; `GPUNearestFeatureJoin` k-nearest with lowest-ID or all ties, `maxDistance`, foot points and segment IDs for every kind pair; `GPUNearestFeatureWeights` (k-nearest output to `GPUSpatialWeights`, libpysal 4.15 `KNN`) | Relate is still about 10x slower than GEOS on few pairs of large polygons (about 10 ms for 100 pairs, against 1 ms); pattern and distance are compile-time; `lineSplit` and `gpu-network` noding; nearest join to `GPUSpatialWeights` adapter |
| S2 | `GPUSpatialWeightsAlgebra` (set operations, `higherOrder`, self weight, subgraph, block), `GPUSpatialWeightsSummary`, `double` and `variance` transforms; `GPUNeighborhoodSummary`; `GPUEmpiricalBayesRates`; `alternative` tails on both permutation tests; `GPUSpatialRegressionDiagnostics` (LM tests, residual Moran); `GPUSpatialTwoStageLeastSquares` (`GM_Lag` with Anselin-Kelejian); `folded` tail (esda 2.10); `GPUSpatialEmpiricalBayesRates` (esda `Spatial_Rate`, `Spatial_Empirical_Bayes`); `GPUSpatialErrorGM` (spreg `GM_Error`); `GPUSpatialWeightsTranspose` (CSR transpose, so diagnostics and 2SLS no longer need a symmetric pattern and match spreg 1.9.1 on directed kNN) | `W²X` instruments |
| S3 | `GPUGroupGeometry`, `GPUGroupConvexHull` (exact lattice orientation); `GPUCellSetOutline` (H3, Quadbin) with ring output through `GPUSegmentRingAssembly` (closed shells and holes, GeoArrow offsets, polygon layout for the point-in-polygon join); `output.core` on `GPUCellCover`; `GPUNetworkIsochrones` in gpu-network (raster isobands or cell outlines) | Isoband triangles are not turned into rings; isochrones carry no facility labels; H3 pentagon rings are untested; one hull chain per group runs serially |
| S4 | `GPUZoneEvents` (enter/exit, dwell, visits), `GPUTrajectoryEncounters`, `GPUTrackSimilarity` (Hausdorff, discrete Fréchet up to 256 vertices) | Resample-to-common-clock helper for encounters; crossing positions; sparse (track, zone) output |
| S5 | `GPUArealInterpolation` (area-share weights; `GPUSpatialLag` now takes cross weights and several columns), `GPUPycnophylactic`, `GPUSegregation`; `GPUClassAssignment`, `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov`; `GPUKnoxTest`, `GPUMantelTest`; `GPUCatchmentAccessibility` (2SFCA, 3SFCA), `GPUHuffTradeAreas`; `GPUClassificationFit` (mapclassify 2.11 ADCM, GADF, tss); `GPUSpatialErrorGM` (`GM_Error`) | Kulldorff scan; a fast path for unweighted areal interpolation |
| S6 | `GPUOutlineGeometry`, `GPULabelPoint`, `GPUShapeDescriptors`, `GPULineDensity` (grid), `GPUGridGenerator`, `GPURectangleClip`; `GPUCoverageSimplification`; `GPURasterSieve` and `GPURasterPatchMetrics` in gpu-raster; dense-box option in `GPUSpatialClustering`, `GPUKMeans`, `GPURipleyDistanceFunctions` (F, G, J); `GPUMapColoring`; `GPULineLengthPerPolygon` (QGIS Sum line lengths) | Topology-preserving coverage simplification; k-means convergence test; Ripley edge corrections beyond border |

Fix round (same day), driven by the recipes and the explorer demo:
- **Exact orientation:** relate and the point-in-polygon join use the exact orientation of `GPUSegmentIntersection`. The upstream pairwise classifier was never wrong, but it marked up to 60% of near-edge rows uncertain; the join now re-decides those exactly at no measurable cost.
- **Relate engine speed:** one 64-lane workgroup per pair with box pruning brought realistic polygons from about 1 s to about 10 ms, and a new `engine: 'auto' | 'fast' | 'relate'` prop selects the kernel. Metal fused the fast kernels' f32 orientation test into an FMA, so `contains`/`within` missed polygons sharing boundary runs; every predicate kernel now uses the exact sign.
- **`spatialSort`:** now on by default from 256 features for the point-in-polygon join, the nearest join and buffer selection. Shuffled input runs 50–125x faster; coherent input costs about 1 ms more.
- **Storage-buffer limit:** the `GPUCellSetOutline` groups kernel bound 9 storage buffers. Specs did not catch it because the test device requests `featureLevel: 'max'`. A spec on a `'core'` device now guards it, and an audit at default limits found no other kernel over 8.
- **Recipe adapters:** removed through `GPULocalMoran` `quadrantGating`, `GPUGroupStatistics` `keyCount` (dense tables), and integer columns in `GPUClassBreaks` and `GPUColorScale`. `GPUZoneEvents` reports separate overflow diagnostics.
- **Explorer:** 17 new modes and 9 extended ones, grouped into categories in the panel, so every contributor and recipe of this round has a mode.
- **Graph imports:** importing the same buffer into one graph twice throws. Import it once and share the view.

Platform findings from this round:
- The Metal compiler removes float `twoSum` residuals, so error-free transforms cannot certify orientation signs here. `GPUSegmentIntersection` uses an exact integer fallback instead. Check whether the point-in-polygon join's fp64 uncertainty path is affected the same way. Its spec passes, but it may not exercise the residual.
- WGSL rejects a constant NaN, and `x == x` is not a reliable NaN test, so use bit tests.
- `pass` and `final` are reserved words in WGSL.
- `GPUSpatialRegressionDiagnostics` accumulated `W'Z` over each row's own neighbors only, so a directed pattern (kNN) missed every link `j -> i` without `i -> j` and biased the Moran variance. It now transposes `W` with `GPUSpatialWeightsTranspose`, and the spec pins spreg 1.9.1 on directed kNN scenes.
