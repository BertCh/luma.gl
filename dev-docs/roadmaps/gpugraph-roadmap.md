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
and `gpu-tables`. Each contributor is a `GPUCommandNodeProducer` class with typed
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
- **Spatial join** — `GPUPointInPolygonJoin` and `GPUNearestFeatureJoin` over a `GPUBVH`, rebuilt per
  encoding or reused across encodings through `GPUSpatialJoinPrepared`.
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
- **Spatial feature sorting for joins** — `spatialSort`; M3 Pro, shuffled features:
  point-in-polygon 14.4k cells × 250k points 218 → 6.7 ms, nearest feature 50k segments × 100k
  points 309 → 4.1 ms. On by default from 256 features for the point-in-polygon join, the nearest
  join and buffer selection, with a Hilbert curve (10–15% faster than Morton, which stays available
  as `spatialSortCurve: 'morton'`)
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
- **Attribute crossfilter** — folded into upstream `GPUCrossfilter` (`liveMask`, `rejectNonFinite`,
  `exclusiveMaximum`, count view): linked histograms over separate column
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
- **Real map demo** — `examples/deck/spatial-analysis-explorer` has 67 modes in 10 categories (plus
  "Other"), covering every exported gpu-spatial-analysis contributor and recipe, all with 0 rebuilds
  under per-frame parameter changes
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
  analytic z, spatial lag, HH/LL/HL/LH quadrants, optional FDR, `quadrantGating`). Both read a
  `GPUSpatialWeights` CSR from any weights producer. Moments come from
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
- `GPUDotDensity` and `GPURandomPointsInPolygon` (round 7, `gpu-spatial-analysis/dot-density/`): dasymetric dot density
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
- `GPUGeographicallyWeightedRegression`: GWR with Gaussian and bisquare kernels, fixed or adaptive bandwidths, an on-GPU AICc bandwidth ladder (per-frame, no rebuild), local coefficients, local R-squared, hat diagonal, `localConditionNumber` and a global summary; O(n^2 * ladder) with fixed-order reductions, or grid-indexed for fixed bisquare bandwidths. `GPUGeographicallyWeightedRegressionNonstationarityTest` is the mgwr 2.2.1 Monte Carlo non-stationarity test.
- `GPUCompositeScore`: min-max, z-score and exact percentile-rank scalers, per-indicator directions, weighted sum, weighted geometric mean and first-principal-component scores with per-frame weights, fixed-order column statistics, oracle-tested
- `GPUInequality`: per-zone Gini, Theil T and L, Atkinson (any epsilon per frame), Hoover, Palma, Lorenz knots, weighted (population) variants, and the between/within Theil T decomposition with pooled Gini, deterministic with no atomics, tested against a float64 oracle.
- `GPUCalendarBuckets`: exact Int64-millisecond to calendar columns (year, month, day, hour, minute, weekday with configurable week start, day of year, ISO week and week-year, quarter) plus a deterministic hour by weekday count matrix; fixed per-frame offset or per-row offset column for DST; tested bit-exact against a BigInt oracle and JS `Date`.
- `GPUChangeDetection`: two-slice difference/log ratio/percent change, Welch t-test with Student-t p-value, Theil-Sen slope, Mann-Kendall S/Z/p, multiband change magnitude and direction, and significance classes, per-cell deterministic, oracle-tested

Still open (re-audited against the working tree on 2026-10-06). Items that have shipped were removed
from this list; they appear in the implemented list above or in "Status after the 2026-10-05 build
round" below. Repeated items are listed once under "Cross-cutting", and per-contributor bullets list
only what is specific to that contributor.

#### Under way

- deck.gl WebGPU picking is mirrored vertically: `deck-picker` converts the cursor with
  `cssToDevicePixels([x, y], true)` (GL bottom-left) and reads WebGPU's top-left picking texture.
  Until upstream deck.gl changes, this checkout carries a yarn patch
  (`.yarn/patches/@deck.gl-core-npm-9.4.0-*.patch`) that flips the WebGPU readback offset, row
  order and pick scissor. The core fix and an off-center pick test are in progress on the deck.gl
  branch `fix/webgpu-picking-origin` (worktree `vis.gl-build/round5/deck.gl`, uncommitted). Drop the
  patch once that branch lands. The graph layers add no workaround.

#### Cross-cutting

- **GPU loop primitive.** A "repeat while" node with a GPU-side break would replace the unrolled
  loops that dominate several graphs. An in-kernel fused gate was rejected, because a dispatch buffer
  cannot be both an indirect argument and storage-written. Current costs:
  - Hydrology: one gate per four rounds brought the drainage graph from 2,675 to 1,691 nodes, and
    each skipped round pair after convergence still costs about 50 µs. `GPUTerrainFlow` reports the
    iterations it used (SF terrain: fill 14 of 256, accumulation 2 of 128).
  - `GPULineSimplification`: five nodes per round, about 320 nodes at the default 64 rounds, while
    NYC trips converge in 15. Deep shapes such as spirals need up to `n - 2` rounds.
  - `GPUClassBreaks` head/tail: three nodes per round. Weiszfeld in `GPUGeographicDistribution`:
    about 150 nodes at the default 24 iterations.
  - `GPUKMeans` keeps every iteration in the graph. `GPUSpatialWeightsMinimumSpanningTree` runs a
    fixed ceil(log2 n) Boruvka rounds with no completion flag; a gate (or an indirect dispatch over a
    compacted inter-component edge list, which makes the total work O(E)) would skip converged rounds.
  - `GPUNetworkReachability` unrolls `maxIterations` rounds, so empty rounds still dispatch.
  - `GPUGlobalPermutationTest` needs a loop over batches of permutations to materialize a permuted
    array per batch (n bijections plus nnz gathers) instead of (n + nnz) Feistel evaluations per
    permutation; graph nodes have fixed size today.
- **Indirect dispatch by actual count.** These dispatch at capacity and idle lanes exit early; a
  count-sized indirect dispatch would remove the idle launch:
  - The exact and relate kernels of `GPUSpatialJoin` (relate is 256 lanes per candidate slot).
  - `GPULineClipByPolygon` sub-pieces and `GPULineSegmentize` output-vertex emits.
  - Cost-distance tile scheduling: dispatch only active tiles from a compacted list that `relaxTile`
    appends to (stamp dedupe), shared with hydrology.
  - `GPUFlowAggregation` top-K sorts all C hash slots; a `GPUSort` with a GPU-written count would sort
    only the D occupied ones.
- **`GPUGridIndex`.** Run-time bounds have shipped (`boundsBuffer`), and neighbor search and DBSCAN
  use them. Still missing:
  - A stable in-cell order, which would let `GPUInverseDistanceWeighting` drop its own in-cell sort
    and make float sums over index cells reproducible.
  - A minimum-cell-size mode.
  - A cell-range-driven `GPUGridIndexQuery`. Today it dispatches one invocation per indexed object;
    region statistics uses its own local cell-range gather.
- **Chunked `GraphVectorView` inputs.** Most contributors reject them and take single packed views
  only. Among them: cell indexing, aggregation and topology, column quantiles, column profile, group
  statistics, line geometry, geometry measures, linear referencing, raster algebra, neighbor search,
  global statistics, calendar buckets, geographic distribution and the round-4 network
  contributors. The spatial regression contributors accept them. Tiled and streamed rows go through
  `GPUResidencyArena` instead. `GPUCompaction` still emits one scatter pass per input-chunk ×
  output-chunk pair (1,249 nodes at 32 chunks); document that as a non-goal or route vector outputs
  through one packed scatter. Unexplained: 1,249-node encodings queued 20 deep cost 120–140 ms each,
  against 24 ms synced.
- **Geodesic variants.** The following are planar only:
  - Neighbor search: lon/lat input with an exact haversine re-rank of the final k.
  - Network snapping, linear referencing and line smoothing.
  - Trajectory metrics, playhead and resampling, including antimeridian handling.
  - Distance field and raster profiles; geomorphons and sight lines use the centre row's cell
    size.
  - Geographic distribution centres and rings.
- **GPU Core promotions.** Each item below is a local copy that now has more than one consumer:
  - `utils/wgsl-kernel-nodes.ts`: `createWGSLKernelNode`, the publish and fill nodes, and
    `createTransientUint32Rows`. They are imported by about 250 files and are not exported.
  - `gpu-raster/cost-distance/raster-relaxation.ts`: used by cost distance and eight hydrology
    files; move it out of `cost-distance/`.
  - The Philox RNG (`permutation-inference/permutation-random.ts`): reused by scan statistics,
    space-time tests and the GWR test, with separate copies in flow texture, particle advection
    and dot density.
  - The exact integer summed-area table (`topographic-position/`): a gpgpu `GPUSummedAreaTable`
    or box-sum primitive (`GPUScanUint64` needs separate low and high buffers).
  - Polygon scan conversion, the exact distance transform and jump flooding, the two-word (64-bit
    key) sort, and the fixed-order cell gather.
  - A float or fixed-point segmented scan, since `GPUScan` and `GPUSegmentedScan` are uint32-only.
    Waiting on it: arc-length resampling, profile gain and loss, `GPUInequality` zone statistics (one
    thread per zone today) and per-label sums for skewed `GPURegionPartitionEvaluation` regions.
  - Workgroup memory with a tile and halo in `createWGSLKernelNode` (it is 1-D today), for the 3 × 3
    kernels (curvature, TRI, Weiss, derivatives). Eight neighbour loads hit L1 meanwhile.
  - A union-find or ECL-CC hook-and-compress primitive (CAS parent array) for
    `GPUGraphConnectedComponents`, which still runs relaxation plus pointer jumping for a fixed
    `componentIterations`. `GPUNetworkStatistics` consumes it.
  - A 64-bit-key sort. It would halve the three LSD sorts in `GPUGroupConvexHull` (30-bit keys) and
    the two 32-bit coordinate sorts in network noding and coverage dissolve.
  - A shared no-fuse helper. Apple Metal fuses `a * b + c` into an FMA, which moves results by 1–16
    ULP. Audit contributors whose parity relies on correctly rounded products, such as
    `temporal-reduction` bucket edges, with unquantized inputs.
- **Graph IDs.**
  - `createTransientView` and gpu-core node IDs should name the contributor that created them.
    Today only the raster-relaxation helpers do.
  - Contributor-internal resource IDs (`${id}-total`) collide with caller imports of the same
    name.
  - Importing one buffer twice under different IDs throws "already in use" (documented).
- **Shader compile errors.** The async graph compiler now rejects on pipeline failures. Add a spec
  that compiles invalid WGSL, and cover the sync `compile()` path. Reserved words that already
  failed silently in new kernels: `pass`, `final`, `from`, `target`, `attribute` and `active`.
- **Workgroup-per-item kernels over many small items.** The `segment-sum` kernel in
  `utils/sorted-segment-sums.ts` launches 256 lanes per segment, so many small zones leave most lanes
  idle. The 64-item block hybrid used by `ring-stats`, validity `rings` and the line-segmentize path
  prefix would fix it, but it changes the summation order of small segments, so every caller's
  bit-identical sums spec moves; do it as one change with those specs. Same shape, lower priority:
  flow-aggregation segment sums, group-reduce (x128), label-point, dwithin and relate joins (per
  candidate), track similarity (per pair), group-statistics moments, grouped-sum-reduction, rate
  smoothing and segregation (groups x lanes), terrain peak snap (x256).
- **Glue between contributors.**
  - A uint32-to-float32 cast node: `GPUInverseDistanceWeighting` `counts` and other count outputs
    cannot feed a float colormap or contouring without one.
  - A CSR-to-segments helper and a column-stack helper.
  - A per-vertex path ID.
  - Indirect draw records for `GPUIsolines`, `GPULineSimplification` and the line-geometry
    outputs, which today expose only a clamped `count`.
  - `rowOrigin: 'north'` for isolines, isobands, sampling, profiles and the distance field.
    gpu-terrain already calls this `rowDirection`; reuse that name.
- **Compile-time knobs users want live.**
  - `GPUNeighborSearch` k: an upper k with a per-frame limit.
  - The cell family in `GPUPointToCell`.
  - FDR in `GPUHotSpotAnalysis` and `GPULocalMoran`: a toggle compiles both variants.
  - `GPUSpatialJoinCandidates` `distance`. The nearest join's `maxDistance` is already per-frame.
- **deck.gl consumers.**
  - A GPU-buffer attribute route, so that `GPUColorScale` colors and particle trails and positions
    render without a readback.
  - Binary or Arrow cell columns with GPU instance counts in `H3HexagonLayer`, `QuadbinLayer` and
    the CARTO `ClusterTileLayer`.
  - Graph layers (`deck-arrow-layers/src/gpu-graph/`):
    - Rename `highlightColor`, which collides with deck's `LayerProps.highlightColor` (readonly
      tuples fail to compile), to `neighborhoodColor`.
    - Add log and sqrt scales for heavy-tailed columns.
    - Hash categorical colors: `value % palette.length` collides on power-of-two labels.
    - Hover reruns reachability and path (3–5 ms encode at ≤65k vertices).
    - Port to deck.gl-community.
- **Benchmarks.**
  - Viewshed and reachability on real data.
  - Trajectory playhead and resampling (not measured).
  - Hot-cell atomics off Apple GPUs.
  - The 2026-10-06 algorithmic perf round was not timed. The measurement plan, with expected wins and
    A/B switches, is in [perf-round-2026-10-06](../perf/perf-round-2026-10-06.md).
  - Raster atomic-versus-sorted cost, and whether atomic drift on Apple GPUs is real.
- Package placement stays open (luma.gl versus a deck.gl-side module); luma.gl must not depend on
  deck.gl either way. Projection-dependent contributors take their CRS contract from the
  [GPU Project roadmap](./gpu-project-roadmap.md).

#### Upstream composition (local code kept, with reasons in TSDoc)

- `GPUNetworkStatistics` derives masked live degrees itself, because `GPUGraphDegree` counts every
  stored CSR slot and has no vertex or edge mask. It also still goes through the `@internal`
  `createNetworkAnalyticsTopology` / `defaultBuffer` adapter.
- `GPUGraphTopologyView` is used by degree and PageRank only. Extend it to BFS, SSSP, modularity,
  clustering and layouts, and feed it the induced CSR from `GPUNetworkSubgraphFilter`.
- `GPUNetworkReachability` is multi-source, cost-bounded and frontier-driven, and keeps the network
  tie rule. `GPUGraphSingleSourceShortestPath` is single-source dense Bellman-Ford with a
  lowest-parent tie rule.
- Native `mask` inputs on gpgpu `GPUGridBinning` and `GPUGridAggregation` would remove
  `GPUPointDensity`'s masked-copy pass (one pass and N × 8 bytes).
- Bridge `GPUTileLODSelection` to the residency arena: map its per-tile drawn mask to arena slots so
  that `GPUResidentRowSelection` `tileMask` comes from the GPU instead of the caller.

#### gpu-network

- Network statistics: single-graph (contributor-owned buffers under fixed IDs).
- Edge bundling:
  - Benchmarks at 64 iterations and on low-contention scenes; chunked positions.
  - Bin the points (one atomic each) and convolve when E × P ≫ R². It wins on atomics but loses
    sub-cell position, so numbers change. FDEB is O(E²) and stays rejected.
  - f32 KDEEB amplifies ulp noise about 1000× per iteration, so CPU parity holds only statistically
    after the first iteration.
- Flow aggregation:
  - Sorted mode does not share sorts between pair and zone keys (about three sorts and three scans
    per encode).
  - Top-K sorts all C slots twice. Radix-select the K-th key and sort only the winners, or compact
    occupied slots with a scan (needs the counted `GPUSort` above). Weight-mode sort 2 still has 32
    key bits.
  - Pair keys are 32-bit (65,535 zones).
  - `flow-aggregation-kernels.ts` and `trajectory-metrics-kernels.ts` still bind only what each body
    reads, a workaround the kernel helper no longer needs.
- Round-4 contributors were parity-tested on Apple Metal only and have no GPU timings except
  coarsening:
  - Temporal reduction: a deterministic sum or mean column.
  - Subgraph filter: u32 and i32 filter columns.
  - Coarsening: sparse-label relabelling and multi-level coarsening.
  - Adjacency matrix: per-slot balanced binning for hub rows (the matrix and slot pairing walk a
    whole row per invocation), matrix-cell edge readback and a multi-resolution matrix. Shared-memory
    privatization of small grids (R up to 32) is not done.
- Network accessibility:
  - The cost matrix is dense `rowCount × nodeCount` f32, and the scoring pass binds all of it.
    Large networks need a sparse within-`costLimit` list or row tiling.
  - Lanes share one CSR now (`laneCount`), so the `recommendLaneCount` scratch cap still counts a
    CSR-copy term; drop the edge term (a node spec pins the values).
  - Opportunities attach to nodes only; there is no per-edge opportunity mass and no snapping of
    origins inside the scoring pass.
  - Snapping: BVH near-ties within an ulp can differ from the exhaustive scan. Edges with an
    endpoint out of range become far-away segments, so a radius near 1e30 would match them.
  - Radius-bounded UNA metrics (reach, closeness, betweenness) and many-to-many OD for capacitated
    location-allocation.
  - `GPUNetworkAccessibility` has no extent output. `GPUNetworkSnapping` positions need an interleave
    step before a segment layer can draw them.
- Network statistics and components:
  - Hub rows are one thread each, and component sizes still hit one atomic on the giant component.
    Fix with CSR-vector (workgroup per heavy row, binned by degree) or merge-path over slots, and
    sort-by-label plus a segmented count.
  - Component rounds wait on the ECL-CC primitive above.
- Subgraph filter: the slot pairing is still O(Σ deg(u) + deg(v)), so a hub is quadratic. Two stable
  radix sorts of slots by (max, min) plus two scans make it O(m log m); a cheaper form is an additive
  `sortedNeighbors` flag with binary-search pairing.
- Line graph: ban lookup is O(arcs × bans). Sort bans per frame on (from, to) and binary search, or use
  `GPUHashIndex`.
- Noding: two 32-bit coordinate sorts dominate. A Morton-hash key would halve them but breaks exact
  grouping and node numbering; wait for the 64-bit-key sort.
- Coarsening: the vertex pass still does about 8 atomics per vertex on group cells. Use sort-by-label
  and a segmented reduce, or privatize for small `groupCapacity` (64-bit fixed-point sums make that
  costly).
- Isochrones: the per-edge splat skews on long edges × wide windows. Rasterize distance-to-segment per
  pixel from an edge-binned tile list (changes numerics).
- Reachability: delta-stepping or near-far only after a relaxations-per-edge probe shows more than 3×
  work inflation, because each bucket epoch adds dependent dispatches.
- Map matching:
  - Route search extract-min is a linear scan, O(S) per pop; a private binary heap makes it
    O(S log S).
  - Viterbi forward and backward are one thread per track. For a few very long tracks use chunked
    max-plus transfer matrices with a scan; the restart-from-emission rule is a non-associative reset
    and needs an augmented semiring.
- Network K function: a single-pass count would avoid the per-block cost matrix. A snap
  `candidateCapacity` overflow silently changes K, and only `overflow` signals it. One explorer
  capture of a 192-row block produced no result.
- Region statistics: grid-index cost scales with `candidateCapacity`, not with region size.

#### gpu-raster

- Distance field:
  - No Euclidean direction or back-to-seed vector outputs.
  - No sub-cell seed positions, line or polygon seeds without rasterizing, or barriers
    (cost distance covers barriers).
  - One grid per contributor, at most 32,768 cells per side.
  - The exact-mode row pass is one serial invocation per row (only `height` threads) with
    uncoalesced column reads. Use a banded exact algorithm (PBA) or split each row into 64 chunk
    envelopes merged by binary search, keeping the tie-by-id contract.
- Cost distance and raster relaxation: cross-tile seams, anisotropic costs and nearest-source
  allocation are non-goals today. The sparse-activity queue design from network reachability could
  carry over: an active-tile worklist (see "Indirect dispatch by actual count") would cut each round
  from O(tiles) to O(active tiles).
- Polygon rasterization and raster join:
  - The hybrid exact mode is still caller-wired through `pointBoundaryMask`.
  - An epsilon-to-resolution planner, tiling beyond one storage binding, and per-frame raster
    dimensions (compile-time today).
  - A `featureIds` remap, and one ID layer per overlap depth: overlaps resolve to the smallest row.
  - Sorts and scans run over the full `crossingCapacity`, so with 12 convex polygons the join was
    slower than the exact join. One invocation fills one span, so raster-wide polygons fill rows
    serially; a per-cell lookup of the covering span would balance it. Edge emit loops over the rows
    an edge crosses (tall edges skew); a hybrid of thread-per-edge for up to 8 rows and a
    load-balanced search over `edgeOffsets` for taller edges fixes that.
  - `NO_ZONE = 0xffffffff` cannot index per-zone values in a raster layer; add a compact display
    output.
  - The explorer's raster-zonal mode still rasterizes zones on the CPU. Feed
    `GPURasterZonalStatistics` from `GPUPolygonRasterization`, and reuse the scan-conversion front
    end for `GPUCellCover`.
- Raster algebra:
  - Not an expression compiler: a fused `GPURasterCalculator` (an additive `GPURasterExpression`
    class) lowering a small AST to one kernel would remove the storage round trip per step.
  - Stacks are one packed band-sequential view (8-binding limit).
  - Weighted overlay allows 16 layers and a shared `maximumBreakCount`; no fuzzy memberships.
  - Cell statistics are `O(layerCount^2)`, with no percentile or median across the stack.
  - Reclassify needs ascending break tables, and range-to-value tables with gaps are unsupported.
- Raster stretch:
  - No CLAHE and no adaptive two-stage histogram.
  - Bin assignment uses GPU division, so cells within an ULP of a bin edge may move.
  - No multi-band stretch, log or sqrt transfer, or `rgba8unorm` LUT texture (the same texture
    output `GPUColorScale` lacks).
  - `validCount` is f32, exact to 2^24.
- Isolines and isobands:
  - Polylines are per level and tile-local: no joining across tiles, smoothing or simplification.
  - `GPUIsobands` itself is still a triangle soup with no vertex dedup or index buffer; closed
    rings come from `GPUIsobandRings`.
  - Stitching neighbour lookup is O(levels) per segment, and steep isoband cells loop over their
    bands.
  - NaN or unsorted levels give unspecified geometry. `GPUTerrainContours` could be rebuilt on
    `GPUIsolines`: it runs a full `GPURasterContours` pipeline per level, where one classify pass
    could take the min and max of the four corners and emit segments for every level in range (a
    raster core change).
- Raster sampling and profiles:
  - Packed row-major buffers only: no texture input and no tiled rasters.
  - No per-sample `validity` on profiles.
  - Gain and loss are a serial walk per path, waiting on the float segmented scan above (chunk
    reduce, scan of carries, apply: O(n) work, O(log n) depth, but the f32 summation order would
    change).
  - Samples per path are capped at `min(2^24, (2^32 - 1) / (pathCount + 1))`.
- Particle advection: time interpolation between two fields (`velocitiesNext` plus a blend), lon/lat
  stepping with antimeridian wrap, and speed- or density-weighted spawning instead of the
  `spawnBounds` rectangle.
- Flow texture:
  - Streamlines prune by occupancy cell, not by `d_sep` / `d_test` point distance, and do no
    self-proximity test or neighbour-line seeding.
  - LIC is classic Cabral-Leedom: no OLIC, fast LIC, contrast enhancement or lon/lat steps.
- Zonal statistics: no median or percentiles. Use per-zone radix histograms (about three passes of
  8–11 bits, then a per-zone prefix rank search). Sorted sums keep the radix sort, because bitwise
  determinism fixes the reduction order.
- Change detection:
  - Sen and Mann-Kendall are O(T²) per thread (Sen capped at 64 slices). O(T log T) variants lose at
    T up to 64.
  - Scalar statistics are single-band.
  - The Student-t p-value is f32.
  - No Pettitt/CUSUM, seasonal Mann-Kendall, FDR or per-cell breakpoint.

#### gpu-terrain

- Hydrology:
  - Priority-Flood filling, watershed labelling and least-cost breaching need a tile spill graph
    (tile-local priority flood plus a spillover merge). The tiled Planchon-Darboux fill converges in
    few rounds on real DEMs.
  - D8 accumulation of one very long stem (more than 1024 × `maximumAccumulationIterations` cells)
    needs chain compression or an Euler tour, which change the float sum order and are exact only for
    count units. Strahler order would need rake-and-compress tree contraction for O(log n) rounds.
  - Flats: the lower and higher relaxations are independent and could share one two-field kernel
    (`createRasterTiledRelaxationNodes` with `values2` and a second candidate function), for up to 2×
    on flat-dominated DEMs.
  - Cross-tile fill, flats, accumulation and watersheds.
  - D-infinity HAND/DistDown (TauDEM) and Shreve magnitude are not built. D-infinity and MFD
    accumulation have shipped.
  - HAND follows D8 paths only, and hydrologic indices use the D8 descent slope. TWI is +∞ on flat
    cells only when `minimumSlope` is 0 (default 0.001).
  - D-infinity accumulation differs from the f64 oracle by up to 2.5e-4 relative at facet
    near-ties.
- Visibility:
  - Pyramid-skip traversal is about 2x slower than the march for viewshed and cumulative viewshed,
    and sharing one pyramid (0.3–0.9 ms to build) does not close the gap. Try a coarse-to-fine start
    level per ray or workgroup-uniform skipping, and time with GPU timestamps on a real DEM.
  - An opt-in per-observer radial sweep (Stewart hull, O(n) against the march's O(n^1.5)) needs a new
    result contract on digital lines, like the horizon `'sweep'`. R2, R3, Van Kreveld and XDraw use a
    different sample lattice, so they cannot replace the bit-identical march.
  - Point horizon covers one raster band: no multi-ring mosaic, no geographic great-circle path, no
    ridge-crest lists. Targets narrower than one lattice step can be missed.
  - `GPUPointHorizonVisibility` has no f64 oracle. `firstAzimuth` wrap-around and
    `rowDirection: 'north'` are untested.
  - Line of sight needs all four bilinear corners valid at the target.
- Horizon, illumination and relief:
  - A max-mip horizon pyramid for the march; the sweep already covers most of the cost.
  - The ray's ground distance and geometry use the centre row; per-row great-circle geometry is
    still needed for long rays.
  - `GPUSolarPosition` runs per row on the GPU but is not wired per pixel into irradiance or shadow
    rasters, and `GPUSolarShadowMask` cannot take its sun from it.
  - Neither a horizon texture-array output nor an `rgba8unorm` relief color texture exists, and the
    relief and horizon texture paths have no GPU tests.
  - FFT texture shading needs a graph-node wrapper for `GPUFFT2D`, which is capped at 2048, below
    the 4096² that a padded 2048² tile needs.
  - `GPUReliefBlend` percent-clip needs a histogram percentile pass, and Hesse's full LRM is not
    built.
  - Local dominance takes 84.5 ms at 1024²; a shared-memory or summed-area form could help.
  - The sweep's SVF differs from the bilinear march by about 0.01 on rough terrain, and needs
    `stepGrowth` 1 and extents of at most 32,767. `'imhof-swing'` needs the `imhofSwing` topology
    flag. mt-image's propagated-occluder soft shadow and multi-scale normals are not ported.
- DEM analysis:
  - True 2-D prominence and isolation are not built, nor a 2× decode downsample with statistics.
  - Spike repair is not cross-tile. It is capped near 2048² tiles (per-component vote counters cost
    8n words) and bit-exact only for power-of-two `step`; a compact-by-rank scheme would lift the cap.
  - TPI neighbourhoods are square annuli, and there is no Gaussian-weighted DEV.
  - Geomorphons lack the GRASS `extended` correction and intensity and exposition outputs. Zenith
    and nadir are tangent queries on a windowed hull, O(1) amortised per cell and direction against
    O(R); the window is bounded at both ends and the GRASS first-count tie rule needs care. Build it
    with the sweep line-family machinery (worth it above a search radius of about 30).
    Curvature lacks the spheroidal-trapezoid method.
  - Multi-tile Weiss and geomorphons need halos and mosaic-wide reductions.
  - Documented caveats: Mapbox `(0,0,0)` decodes to -10000 m (pass `noDataRGB: [0, 0, 0]` when the
    source uses it as nodata), and peak snap leaves `max(DEM, catalogue elevation)` to the caller.
  - The oracles are f64 transcriptions; no comparison against GDAL, GRASS or Whitebox binaries was
    run.

#### gpu-spatial-analysis

- Spatial interpolation:
  - IDW nearest-`k` keeps a private list per invocation and was about 6x slower than radius-only at
    k = 16 (16.9 against 2.9 ms, measured before the ring walk replaced the rectangle scan; measure
    again). Radius-only mode and `k = 0` still scan the sorted rectangle, because the summation order
    needs the in-cell sort.
  - No barrier-aware or anisotropic IDW, and no caller-supplied grid index.
  - Focal statistics need van Herk/Gil-Werman min/max (exact, O(1) per cell), summed-area or
    separable mean and sum (compensated sums, which change the documented row-major order), majority,
    median, rank, and annulus or custom windows. They are O(r²) per cell today.
  - `GPUInverseDistanceWeighting` has no GPU count of nodata cells.
- Cell indexing, topology and cover:
  - Forward A5 keying is missing; A5 is geometry-only.
  - `precision: 'double-single'` would extend exact H3 and S2 to fine resolutions. H3 is exact up to
    res 4, with about 1e-3 mismatch at res 9, 1.8% at res 12 and 28% at res 15. Quadkey and quadbin
    rows at zoom 26–29 differ from f64 by at most one row.
  - No string tokens.
  - H3 disks run one private BFS per thread (about 11 ms for one cell at k = 8), with no spiral fast
    path. GPU specs cover k ≤ 4 only.
  - No `gridDistance`, `gridPathCells`, local IJ, directed edges or vertices. S2, A5 and geohash
    topology and cover are not built.
  - Uncompact does not dedupe overlapping inputs, and its u32 total wraps. Compaction allocates an
    unused transient counts column.
  - Cover:
    - H3 supports only `center`, plus the conservative `core` flag. `full` and `intersects` are
      Quadbin-only.
    - No `compactToParents`.
    - No antimeridian-crossing polygons.
    - The res 11–15 H3 inradius constants are extrapolated.
  - Cell aggregation: no `'any'` aggregation or average column, and each value column repeats the
    sort. H3 points are keyed through `GPUPointToCell` first. Tiled CARTO sources could go through
    the residency arena.
  - `GPUCellLevelSelection.drawArguments` always writes `firstInstance`: add an option to leave it
    at 0, for devices without `indirect-first-instance`. Its `output.count` rejects strided one-row
    `DrawCommandBuffer` views.
  - `GPUCellPyramid` levels cannot share one slab, because outputs are checked against inputs per
    buffer, not per byte range.
  - Skewed per-feature walks: zonal-statistics area, ring orientation (validity, dissolve) and the
    cell-cover count node loop over all vertices in one thread, so a 1M-vertex polygon is a serial
    1M-step loop. Use vertex-parallel terms with a segmented reduce, or min and max by workgroup then
    one atomic. It needs a long-ring list or an indirect dispatch, otherwise a workgroup per tiny ring
    regresses.
  - Catchment and Huff: thread per facility row, so a facility with thousands of demand slots
    serializes. Use a slot-parallel contribution kernel plus a segment sum, enabled by a
    `maximumRowLength` hint (changes the deterministic sum order).
  - Dissolve vertex sort: two 32-bit coordinate sorts (16 passes). Use 8-bit digits (needs a 256-bucket
    workgroup table) or a linear-probing hash (needs capacity and a collision fallback).
  - Cell-set outline neighbour lookup is a binary search per edge; galloping from the cell's own row or
    an open-addressing table cuts reads.
  - Cell topology: grid path tests six neighbours with an O(res) `getLocalIjk` each; step in local ijk
    and validate once (pentagon parity with h3 is pinned). Compaction analysis does two binary searches
    per depth; sibling-run detection by scan replaces them.
  - Cell-table compare:
    - A full merge path (diagonal binary search per 256-element tile) would give O(n + m) coalesced
      traffic instead of n log m gathers, and needs a rewrite of the four-kernel pipeline.
    - Tables must share a resolution and family, and the contributor cannot check that.
    - Two tables only, and measures are `count` and `sum` only.
    - The standardized reduction is a single workgroup.
    - Ratio and z are f32 divisions.
- Spatial weights and statistics:
  - The hot spot and local Moran contributors walk a weights CSR and have no neighbor cap of their
    own. `GPUNeighborSearch` and the permutation test have caps and overflow flags.
  - Neighbor search:
    - k ≤ 32, with no BVH path for uneven densities.
    - Radius rows are insertion-sorted (O(d²)); a segmented radix sort by row fixes dense rows.
    - kNN scratch is `queryRows * k`.
    - Targets outside `bounds` are excluded.
  - Local Moran: an ArcGIS-style total-randomization null.
  - Global statistics:
    - A 25-column f32 scratch matrix caps inputs at about 1.3M rows at 128 MiB.
    - GPU-versus-oracle tests stop at a few thousand rows (dense oracle).
    - Degree sums are exact below 2^24 only; join counts are exact u32.
  - Permutation test:
    - Local Gi* uses a fixed self weight of 1.
    - No local Geary, local join counts or local bivariate or multivariate LISA.
    - The global test pays `(P + 1) * nnz` Feistel evaluations (see the loop over permutation
      batches above).
    - The local test could stop early (Besag-Clifford: stop a row after `h` exceedances, per-row cap)
      or sample k of m by Floyd's algorithm. Both change the random stream and break the oracle, so
      they would be opt-in.
    - `maximumNeighbors <= 64`.
    - One invocation per row loops over P.
    - No quadrant class output; callers gate `GPULocalMoran` quadrants with its `significant`
      mask.
  - Spatial weights:
    - Contiguity vertex grouping: two full 32-bit radix sorts of quantised coordinates. A hash sort
      needs collision handling and a GPU hash table is order-nondeterministic. The pair stage is O(m²)
      per shared point (hub points).
    - Weights algebra (higher-order, union): per-row insertion sort, O(d²) per level; use a segmented
      sort plus unique.
    - Global statistics `findWeight` could merge with the transposed rows instead of a binary search
      per slot.
    - Spatial lag stays CSR-scalar (right below degree 32); `GPUProgramSpMV` covers heavy skew.
  - Scan statistics: `nearest` is O(n²) per frame; compose a `GPUGridIndex` k-nearest query for O(n k).
    It only matters above about 10^4 zones. `scatter` draws P × C samples at log(cells) each; a
    conditional-binomial split (O(P × cells)) needs a GPU binomial sampler and changes the random
    stream.
  - Emerging hot spots:
    - Permutation p-values and FDR.
    - Time steps other than one slice; 256 slices maximum.
    - A per-bin missing mask.
    - A prefix-sum Gi* for large radii.
    - Category rules are not validated against ArcGIS output.
  - "Optimized hot spot" (a distance sweep that picks the band) is not composed, though
    `GPUSpatialCorrelogram` reports peak bands.
- Pair statistics and point patterns:
  - Ordered histograms (Ripley G, correlogram degree) need both directions, so they keep the full
    stencil; a pair-action API that evaluates both roles would let Ripley K use the half stencil.
  - A 5 × 5 stencil at half cell width covers 6.25 r² against 9 r², but the lattice is shared by every
    kernel in the directory.
  - Radius search emits per-row ids by count, scan and emit; ids are sorted per row (O(d²)).
  - Monte Carlo CSR envelopes for K and L, permutation inference for the correlogram, cross-type K
    and marked patterns.
  - Non-rectangular windows, kernel-smoothed pair correlation, and quadrat and Clark-Evans
    p-values.
  - Verify the border `lambda` convention and the isotropic weight cap against spatstat.
  - Variogram `maxPairs` subsampling and a variogram cloud.
  - The correlogram has 64 bands at most and an f32 analytic variance.
  - No cell-range-driven neighbour pass for dense clusters.
- Regression:
  - OLS has only Koenker Breusch-Pagan: no White test, F statistic, HC errors or WLS. Ridge
    standard errors ignore shrinkage. 15 predictors at most, with no condition number or VIF.
  - GWR:
    - Gaussian bandwidths are still O(n²) (unbounded support). Truncating at about 5 bandwidths is the
      standard fix, changes results, and needs an opt-in slot in
      `geographically-weighted-regression-parameters.ts`. The grid index and ring kNN now cover fixed
      and adaptive bisquare.
    - The `select` kernel is one thread over tiles × ladder, and the ladder loop re-reads the
      neighbourhood per candidate. A shared pass over the largest bandwidth needs per-candidate normal
      equations in registers.
    - No MGWR, local standard errors, pseudo-t, Poisson/logistic or geodesic GWR.
    - A single singular location rejects the whole candidate.
    - Adaptive `k` ≤ 128, with at most 32 ladder candidates. Adaptive GWR cost about 5x a fixed
      ladder before the ring kNN search (measure again).
- Clustering:
  - OPTICS and HDBSCAN are not built.
  - k-means assign is O(n k) every iteration. Hamerly or Elkan bound pruning needs 8 B more per point
    and a k × k center-distance step, with a safety margin to keep the lowest-id tie rule, and wins
    more as k grows. A real gain on the 64-iteration cap needs the GPU loop exit above.
  - Point density computes cell keys twice when sums are on (`GPUGridBinning` plus `grid-keys`); fusing
    them means dropping luma's `GPUGridBinning`. The sum path's O(256) pre-aggregation per row could
    be a segmented scan like the cell table.
- Geographic distribution: no central feature, and the ellipse axis naming and angle are derived
  from the published formula, not compared with ArcGIS.
- Geometry:
  - Measures:
    - No Karney geodesic-edge areas.
    - No rings around a pole.
    - Geographic group centroids average longitudes naively.
    - No cross-track, along-track or rhumb distances.
    - A feature with a ring above 512 rows is measured cooperatively, but a feature of 100k holes
      still runs on one lane and latency follows it.
    - Shape descriptors' moments are one thread per feature (Neumaier), so one huge feature skews;
      a workgroup per feature changes the compensated order, or fuse into `GPUGeometryMeasures`.
  - Line geometry:
    - Arcs are never split at the antimeridian.
    - `GPULineSmooth` has no `measures` or `sourceRows`, and `GPULineChunk` has no measure-range
      columns or negative measures.
    - Open bug: `GPULineChunk` substring mode with a start measure past the path length disagrees
      with the oracle by two vertices (piece-range kernel), and no spec covers it.
  - Line simplification:
    - Active rows are not compacted per round, so late rounds dispatch over all rows. Fuse `reset` into
      the previous `split` by ping-ponging `bestKeys` and `bestRows` (4 to 3 dispatches per round) and
      use a workgroup per large interval for the argmax (all rows `atomicMax` one key in round 0).
      Visvalingam-Whyatt rounds grow with runs of increasing area; the small-span finish is not exact
      there, so compact surviving rows every k rounds instead.
    - No topology-preserving, Visvalingam-Whyatt or perpendicular metric.
    - No double-single or Int64 `'time-ratio'` times.
    - Closed rings keep their seam vertex.
  - Linear referencing:
    - Spherical projection is O(P × S) cheap cap tests now. A cell or BVH index over the cap table
      (S2-style cube-face grid via `GPUGridIndex`) makes it sub-quadratic.
    - `GPULinearReferencing` returns one candidate and runs its own projection pass, although
      `GPUNearestFeatureJoin` now emits k candidates with foot points.
    - Side and vertex-measure conventions are documented limits.
- Joins, predicates and clearance:
  - The fast one-thread-per-pair kernels have no slab index or segment-chunk splitting; polygon/polygon
    would exceed 8 bindings for the extra bounding boxes. Lines have no slab index.
  - The point-in-polygon join uses the upstream robust classifier; edge bucketing belongs there.
  - Nearest join: Hilbert-sort the query points (radix sort plus permuted `row`) to cut divergence;
    point queries only, because polygon/polygon is at 8 bindings.
  - Minimum clearance per-feature min uses two contended `atomicMin` passes; use a workgroup reduction
    and one atomic.
  - Predicates: pair compare and `is-ccw` are one thread per feature; line length per polygon does an
    O(E) point-in-polygon per piece (needs an edge BVH and different tie semantics); line clip still
    dedupes events in O(k²) per segment (use a segmented sort) and loops over overlap spans per
    sub-piece (use a per-segment flag); shared-paths walk run chains (pointer jumping); geometry-edit
    ring orientation is a serial area and cleanup flags are a greedy chain (tolerance 0 is a parallel
    neighbour compare).
  - Validity holes are O(shell/64) per hole without an index (y-slab buckets would be next); polygon
    triangulation is one thread per polygon with O(holes × n) hole elimination (monotone decomposition
    for one huge polygon); coverage simplification rebuilds the BVH every topology round (a per-round
    gate on "any offending pair" needs a condition node written from classify).
  - Medoid is exact O(g²) per group; a workgroup-tiled cost kernel cuts global reads 256× but must
    keep the summation order. Label point could prune segments per grid cell with a per-feature BVH.
  - Line density `cell-counts` atomics are now dead work (see "Dead work after the perf round").
- Trajectories:
  - Zone walk, dwell and span are one thread per track, so depth is the most events of one track. An
    opt-in `balancedWalk` would sort events by (track, zone) cell and take type from rank parity. The
    three kernels cannot fuse (13 bindings).
  - `isSlowStep` is evaluated three times per row; a flag buffer cuts the gathers about 3×.
  - The playhead and resampling take f32 relative or Int64-word times, not double-single.
  - Arc-length resampling sums per track in one invocation.
  - Zero-length segments report heading 0, with no heading smoothing, spline interpolation or
    dead reckoning.
  - Resample targets at a path discontinuity can land on either side (4 ulp tolerance).
  - `GPUTrajectoryPlayhead` counts only active tracks; add per-status counts.
- Dot density:
  - Triangle-area or raster-CDF sampling for thin polygons and sparse masks, instead of
    bounding-box rejection.
  - Acceleration for very large polygons.
  - Interleaved draw order across categories.
  - Blue-noise thinning.

#### gpu-dataframe

- Column classification:
  - Quantiles: a histogram fast path, `uint32`/`sint32`/Int64 columns, and more than 2^24 rows.
  - Class breaks:
    - Exact refinement of natural breaks (bin-snapped today). The B² cost matrix (4 MB) and k
      dependent layer dispatches could become prefix-sum costs (O(1), shifted by the bin mean or
      compensated against f32 cancellation) with a divide-and-conquer argmin per layer (about
      B²/32 + 32 B instead of B²/2), fused into one workgroup. The lowest-start tie rule is not
      guaranteed under f32 rounding.
    - Maximum breaks without two full sorts every frame.
    - De-duplication of repeated quantile edges.
    - Geometric and percentile methods.
  - Colour scale: OKLab interpolation, diverging scales with a midpoint, NaN-safe descending
    domains and a texture LUT.
  - Bivariate: more than 16 classes per axis; trivariate.
- Column profile:
  - Four pipelines per column instead of one fused pass, and a histogram wider than 1,024 bins still
    serializes on a hot-bin atomic.
  - No quantiles, M3/M4 moments or count-min sketches.
  - No HyperLogLog large-range or HLL++ correction.
- Group statistics:
  - Approximate distinct counts: reuse column profile's HyperLogLog.
  - Weighted statistics and quantiles.
  - A sort-free hash path: dense `keyCount` removes key compaction but still sorts rows.
  - Quantile definitions other than linear: reuse `GPUColumnQuantiles`' rules.
  - Float64-grade sums and exact moments, and an exact-mean pass for groups whose spread is below
    the fixed-point resolution.
  - The moments pass runs a 256-thread workgroup per group, which wastes lanes on many tiny groups. A
    hybrid (workgroup per large group, thread per small group) changes the summation order.
- Key join:
  - Right and full outer joins (`rightMatched` exists).
  - Dictionary and composite keys.
  - Median, mode and first/last aggregates.
  - A hash probe path.
  - A `last` gather.
  - u32 aggregate columns.
- Composite score: one `GPUSort` per indicator for ranks; head flags plus a scan of tie runs would
  replace the two binary searches per row. No robust scaler, supervised weights or
  output class. PCA returns the first component only.
- Inequality:
  - One thread walks each zone (the pooled Gini is tiled now), so one big zone serializes. Use a
    float segmented scan over zone-sorted rows and a segmented reduce, with Lorenz knots by binary
    search on the cumulative share.
  - f32 accumulation.
  - Theil L, Atkinson and Gini decompositions.
  - Bootstrap intervals.
  - Zero-valued rows give NaN for Theil L and Atkinson.
- Calendar buckets:
  - A GPU time-zone transition table (DST needs a per-row offset column today).
  - Float32 or split timestamp inputs.
  - A bucket-index output to key `GPUTemporalReduction` or `GPUCellAggregation`.
  - Fiscal and non-Gregorian calendars.
- Arrow temporal: the relative-float32 path misreads `Date(ms)` columns (two int32 words per row);
  the word path is correct.

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

Shipped in the 2026-10-05 backlog round: SKATER-style minimum spanning tree over contiguity
(`GPUSpatialWeightsMinimumSpanningTree`, built here because upstream has no MST), `GPUSkaterRegions`,
`GPURegionPartitionEvaluation`, network K function (`GPUNetworkKFunction`), GWR Monte Carlo
non-stationarity test and local condition number, kNN-neighborhood kriging with variance (`GPUKriging`),
feature-space similar locations (`GPUSimilarLocations`), endpoint line merge (`GPULineMerge`), a
turn-restriction line graph (`GPUNetworkLineGraph`), map matching (`GPUMapMatching`), Hilbert keys
(`GPUHilbertKeys`, in luma experimental rather than upstream) and circle, sector and ellipse generators
(`GPUShapeGenerator`, including optional equal-arc ellipse spacing).

Still open: global and universal kriging, anisotropic variograms and kriging cross-validation; REDCAP
and other tree-cut criteria and a polygon-geometry compactness (Polsby-Popper) partition measure; a
line-graph seeding helper (origin node to edge seeds), node costs and a network cross-K; exact
(unbudgeted) map-matching routes, geodesic distances, time-aware transitions and line merge with a
tolerance; two-word Hilbert keys (order above 16); a GPU rank sort for large-N rank standardization in
`GPUSimilarLocations`.

### Named recipes

All eleven ship as parameterised builder functions (`add<Name>Recipe` in
`gpu-spatial-analysis/recipes/`) that add a chain to a caller's graph and return named output views,
as CARTO Workflows templates do. They are not new classes. The user doc lists them, and every recipe
has an explorer mode.

| Recipe | Chain |
| --- | --- |
| Hot spot analysis | `GPUPointToCell` → `GPUCellAggregation` → `GPUCellGeometry` (centres) → `GPULatticeWeights` or `GPUNeighborSearch` → `GPUHotSpotAnalysis` (FDR) → `GPULocalPermutationTest` → `GPUClassBreaks` → `GPUColorScale` |
| Rate cluster map | `GPUEmpiricalBayesRates` → `GPUContiguityWeights` → `GPUSpatialWeightsTransform` → `GPULocalMoran` → `GPULocalPermutationTest` → quadrant colors |
| Points-in-polygons choropleth | `GPUPointInPolygonJoin` or `GPUSpatialPredicateJoin` → `GPUZonalStatistics` / `GPUGroupStatistics` → `GPUClassBreaks` → `GPUColorScale` |
| Space-time hot spots | `GPUCalendarBuckets` → `GPUGroupStatistics` (cell × slice counts) → `GPUEmergingHotSpots` → `GPUColorScale` |
| Cluster and outline | `GPUSpatialClustering` → `GPUGroupGeometry` + `GPUGroupConvexHull` → `GPUGeometryMeasures` |
| Spatial regression | `GPUOrdinaryLeastSquares` → `GPUSpatialRegressionDiagnostics` → residual `GPULocalMoran`; `GPUGeographicallyWeightedRegression` for local fits |
| Drive-time catchment | `GPUNetworkSnapping` → `GPUNetworkServiceAreas` → `GPUNetworkIsochrones` → `GPUPointInPolygonJoin` → `GPUGroupStatistics` |
| Straight-line catchments | `GPUDistanceField` allocation (raster Voronoi) → `GPURasterZonalStatistics` |
| Change of support | `GPUPolygonRasterization` → `GPUArealInterpolation` → `GPUSpatialLag` |
| Fleet dwell | `GPUTrajectoryMetrics` stops or `GPUZoneEvents` → `GPUPointInPolygonJoin` → `GPUGroupStatistics` |
| Period comparison | two `GPUCellAggregation` → `GPUCellTableCompare` → diverging `GPUClassBreaks` |

The cross-reference page (`docs/api-reference/experimental/gpu-spatial-analysis-cross-reference.md`)
maps turf, PostGIS, GeoPandas/Shapely, PySAL and QGIS/ArcGIS/CARTO tool names to contributors and
recipes.

### Deliberately not building

- **Vector overlay:** union, intersection, difference, dissolve geometry, exact buffer, polygonize,
  noding output, `make_valid` and `ST_Subdivide`. Output is unbounded and needs robust topology. The two
  carve-outs are bounded because they never compute a new intersection between two polygons:
  `GPUCoverageDissolve` (a valid coverage keeps a subset of its input edges) and `GPULineClipByPolygon`
  (every vertex is an input vertex or one line-segment-by-polygon-edge crossing).
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

Upstream asks for luma gpgpu: a minimum spanning tree in gpu-graph (the local Boruvka version could
move there), Hilbert keys next to `GPUBVH`, and a reusable bounding-box candidate-pair stage shared by the join
contributors.

### Status after the 2026-10-05 build round

Every P1 and P2 item above now has a first version on `map-contributors`, except the ones listed as
open below. Each contributor has a headless GPU spec against a CPU oracle. Where a reference
library exists, the fixtures pin its values: spreg 1.9.1, esda 2.10, libpysal 4.15, giddy,
pointpats, segregation 2.6, tobler and Shapely 2.1.2.

| Tranche | Shipped | Still open |
| --- | --- | --- |
| S1 | `GPUSegmentIntersection` and `GPUGeometryValidity` (exact 256-bit orientation fallback, Morton-sorted BVH); relate engine in `GPUSpatialPredicateJoin` (DE-9IM for all nine kind pairs, seven new predicates, `relate` output and patterns, `how: 'anti'`); `GPUSpatialJoinPrepared` for the predicate and point-in-polygon joins; `GPUSpatialJoinCandidates`; `GPUNearestFeatureJoin` k-nearest with lowest-ID or all ties, `maxDistance`, foot points and segment IDs for every kind pair; `GPUNearestFeatureWeights` (k-nearest output to `GPUSpatialWeights`, libpysal 4.15 `KNN`); backlog round: `GPUSpatialPredicateJoin` `distance` and `pattern` are now per-frame views (`packGPUSpatialRelatePattern`); `GPUSpatialJoinCandidates` `distance` is now a per-frame view; `GPUBufferSelection` `drawInstanceCount`; batched relate break collection and a workgroup y-slab index (100 pairs, about 7.5 to 4.3 to 5 ms); `GPULineSplit` and `GPUNetworkNoding` (Shapely-pinned); `GPULineMerge`. | Relate is about 4 to 5x slower than GEOS on few pairs of large polygons (4.3 to 5 ms for 100 pairs, against 1 ms; about 2 ms of that is fixed graph and readback cost), because the break collection, covers scan and vertex loops are still O(n*m); next step is slab-restricted break collection or pieces across lanes; tolerance-based T-junction repair, cluster-exact noding and bridge control in noding |
| S2 | `GPUSpatialWeightsAlgebra` (set operations, `higherOrder`, self weight, subgraph, block), `GPUSpatialWeightsSummary`, `double` and `variance` transforms; `GPUNeighborhoodSummary`; `GPUEmpiricalBayesRates`; `alternative` tails on both permutation tests; `GPUSpatialRegressionDiagnostics` (LM tests, residual Moran); `GPUSpatialTwoStageLeastSquares` (`GM_Lag` with Anselin-Kelejian); `folded` tail (esda 2.10); `GPUSpatialEmpiricalBayesRates` (esda `Spatial_Rate`, `Spatial_Empirical_Bayes`); `GPUSpatialErrorGM` (spreg `GM_Error`); `GPUSpatialWeightsTranspose` (CSR transpose, so diagnostics and 2SLS no longer need a symmetric pattern and match spreg 1.9.1 on directed kNN); backlog round: `W²X` instruments (`instrumentOrder: 2`, spreg `w_lags=2`); GWR Monte Carlo non-stationarity test and `localConditionNumber` (mgwr 2.2.1); `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions` (spopt `SpanningForest`), `GPURegionPartitionEvaluation` | Robust and heteroskedastic 2SLS; grid-indexed Monte Carlo refits; VDP and VIF from `local_collinearity` |
| S3 | `GPUGroupGeometry`, `GPUGroupConvexHull` (exact lattice orientation); `GPUCellSetOutline` (H3, Quadbin) with ring output through `GPUSegmentRingAssembly` (closed shells and holes, GeoArrow offsets, polygon layout for the point-in-polygon join); `output.core` on `GPUCellCover`; `GPUNetworkIsochrones` in gpu-network (raster isobands or cell outlines); backlog round: `GPUIsobands` edge output and `GPUIsobandRings` (closed band shells and holes), `GPUSegmentRingAssembly` `cancelOpposingSegments` and `polygonGroups`, `GPUNetworkIsochrones` facility assignments and `byFacility` rings, H3 pentagon ring coverage (three mid-latitude pentagons equal h3-js), `GPUGroupConvexHull` `prefilterLevels` (500k disk 551 to 7.6 ms) | GPU polygon triangulation for an exact ring fill without rasterization; the two polar pentagons (no planar orientation); one hull chain per group is serial when all points are on the hull; hull prefilter levels cost 2 to 3 ms on thousands of tiny groups |
| S4 | `GPUZoneEvents` (enter/exit, dwell, visits), `GPUTrajectoryEncounters`, `GPUTrackSimilarity` (Hausdorff, discrete Fréchet up to 256 vertices); backlog round: common-clock helper (`addClockEncounters`, `GPUTrajectoryResample` `spacing: 'clock'`), `GPUZoneEvents` crossing positions (`eventPositions`) and sparse (track, zone) `visitTable`; per-step speed, heading and acceleration in `GPUTrajectoryMetrics` | The visit table is still built from the dense per-cell state (it bounds the output, not the working memory) |
| S5 | `GPUArealInterpolation` (area-share weights; `GPUSpatialLag` now takes cross weights and several columns), `GPUPycnophylactic`, `GPUSegregation`; `GPUClassAssignment`, `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov`; `GPUKnoxTest`, `GPUMantelTest`; `GPUCatchmentAccessibility` (2SFCA, 3SFCA), `GPUHuffTradeAreas`; `GPUClassificationFit` (mapclassify 2.11 ADCM, GADF, tss); `GPUSpatialErrorGM` (`GM_Error`); backlog round: `GPUSpatialScanStatistic` (Kulldorff Poisson space-time scan with secondary clusters and Monte Carlo p-values) and the `GPUArealInterpolation` `unweightedFastPath` (about 1.4x on the explorer graph) | Bernoulli and ordinal scan models, low-rate and both-direction scans, elliptic windows, an alive-cluster option and Gumbel p-values |
| S6 | `GPUOutlineGeometry`, `GPULabelPoint`, `GPUShapeDescriptors`, `GPULineDensity` (grid), `GPUGridGenerator`, `GPURectangleClip`; `GPUCoverageSimplification`; `GPURasterSieve` and `GPURasterPatchMetrics` in gpu-raster; dense-box option in `GPUSpatialClustering`, `GPUKMeans`, `GPURipleyDistanceFunctions` (F, G, J); `GPUMapColoring`; `GPULineLengthPerPolygon` (QGIS Sum line lengths); backlog round: topology-preserving `GPUCoverageSimplification` (fixed arc endpoints, three-vertex ring minimum, bounded exact detect-and-repair with a residual report, Shapely `coverage_is_valid` on the test scene), `GPUKMeans` `tolerance` and `convergence`, `GPURipleyDistanceFunctions` Kaplan-Meier and Hanisch edge corrections, `GPUSpatialClustering` `drawInstanceCount`, `GPUShapeGenerator` (circle, sector, ellipse, optional equal-arc ellipse spacing; turf-pinned), `GPUHilbertKeys` and Hilbert as the joins' default `spatialSort` (10 to 15% faster than Morton); `GPUCostDistance` `frictionParameters` and hydrology iteration outputs | Area-based (Visvalingam or TPVW) coverage criterion, no zero-residual guarantee within the round cap; a pointpats or spatstat pin for the edge corrections (neither is installed here; the oracle is written from the spatstat definitions); Kaplan-Meier and Hanisch for K and L (`GPURipley`); a true GPU loop exit for k-means |

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

Backlog round (2026-10-05): sixteen parallel streams closed the open items above (map matching, noding,
line split and merge, kriging, scan statistic, SKATER, network K and the rest). Findings:
- **Relate and GEOS:** batching the per-edge break parameters and adding a workgroup y-slab index
  closed part of the gap on 100 polygon pairs, from about 7.5 ms to 4.3 to 5 ms against 1 ms for GEOS,
  with identical checksums. The remaining time is about 2 ms of fixed graph and readback cost
  plus the break collection, covers scan and vertex loops, which are still O(n*m).
- **Hilbert versus Morton:** a CPU tree-cost proxy (sum of node perimeters of a complete binary BVH over
  16,000 boxes) is 27% lower for Hilbert on uniform, 21% on clustered and 15% on road-like data.
  The joins' `spatialSort` is now Hilbert by default (paired A/B: point-in-polygon, nearest and buffer
  selection 10 to 15% faster, identical outputs); Morton stays available as `spatialSortCurve: 'morton'`.
- **Network K function:** sharing one lane-expanded CSR across row blocks, one reachability per block and BVH
  snapping took 192 events by 20 patterns on 97,000 edges from 181 to 61 ms (1936 to 931 nodes) with identical
  results. Open: `GPUNetworkCostMatrix` could take an external expanded CSR for other multi-block users, a
  single-pass count would avoid the per-block costs matrix, a snap `candidateCapacity` overflow silently
  changes K (only `overflow` signals it), and a 192-row block produced no result in one explorer capture.
- **k-means "not converged":** not a bug. The shift falls to exactly 0 (a true Lloyd fixed point) after 22 to
  29 iterations, so the explorer's 24-iteration cap was too low; it is now 48 with a tolerance relative to
  the extent.
- **Map matching:** splitting the bounded-Dijkstra route searches into their own pass over
  (point, previous candidate) work items took the New York matching graph from 923 ms (serial) to 420 ms
  (hash lookups) to 30 ms at a node budget of 64, with identical results.
- **Metal `atan2(y, -0.0)`:** WGSL `atan2(y, x)` with `x = -0.0` returned the wrong sign, which
  surfaced as a perpendicular dot product of exactly zero in `GPUNetworkLineGraph`. It special-cases
  `dot == 0`. Treat any `atan2` on a computed dot product as suspect on this adapter.
- **More reserved WGSL words:** `from`, `target` and `active` (alongside `pass` and `final`) failed to
  compile silently in new kernels.
- **Binding limits:** the polygon-polygon relate kernels, the coverage-simplification classify kernel and
  the similar-locations distance kernel bind exactly 8 storage buffers, so per-frame values ride in
  trailing rows of existing buffers. The workgroup memory of the Ripley G pass is now exactly the 16 KB
  default at `radiusCount` 256.

### GeoPandas / GeoPolars parity round (2026-10-06)

A survey of GeoPandas 1.2, Shapely 2.1.2, GeoPolars (a prototype over WKB), polars-st (a GEOS clone that
mirrors Shapely names), geoarrow-rs (which wraps the `geo` crate) and the `geo` crate extras found the
operations below that fit bounded GPU output. Eighteen units were built in parallel on `map-contributors`,
each with a headless GPU spec against a pinned Shapely, GeoPandas, pyproj, h3 or turf oracle.

| Unit | Shipped | Still open |
| --- | --- | --- |
| A | `GPUAffineTransform`, `GPUGeometryOrientation` (`reverse`, `orient_polygons`), `GPUGeometryCleanup` (`remove_repeated_points`, pointwise `set_precision`, points input) | `normalize` |
| B | `GPUMinimumBounds`: rotated rectangle, bounding circle, longest line, envelope, per feature or label | Per-feature hull output |
| C | `GPUNearestFeatureJoin` `exclusive` and `neighborQueryPoints`; `onAttribute` on both joins | Geometry-equality `exclusive` |
| D | `GPUPairGather`, `GPUOffsetExpansion`, `GPUBoundsFilter` | Live wiring to a nearest join, chunked inputs |
| E | `GPUCoverageDissolve`, `GPUCoverageValidity` | Two jitter cases and a third-polygon edge differ from GEOS |
| F | `GPUGeometryPredicates`; `GPUGeometryValidity` for lines and points | `equals_exact(normalize=True)`, multi layouts |
| G | `GPUMinimumClearance` | Multipoints |
| H | `GPULineClipByPolygon`, `GPUSharedPaths` | Noding at a line's own crossings |
| I | `GPULocalOutlierFactor` | Global summary beyond the count |
| J | Rhumb `model` on `GPUGeodesicPairs` and `GPUGeodesicDestination` | Ellipsoidal rhumb |
| K | Spherical `GPULinearReferencing` and `GPULineLocate`; `normalizedMeasures` | A BVH for the spherical scan |
| L | `GPUGeometryMeasures` `'geodesic'`, `extremeVertices`, `'points'` | Geodesic centroid, group extremes |
| M | `GPULineSimplification` `method: 'visvalingam'` (approximate) | Exact heap order, `time-ratio`, VW-preserve |
| N | `GPUOffsetCurve`, `GPUVertexSnap` | Loop removal, snap onto segments, indexed snap |
| O | `GPUTrackSimilarity` `densify`, strip-tiled Fréchet to 2048, `maxDistance` | Re-centred coordinates |
| P | `GPUCellGridPath`, `GPUCellMeasures` | Directed edges and vertexes, exact fine-resolution area |
| Q | `GPUPolygonTriangulation` (earcut) | Workgroup-parallel ears, z-order hashing |
| R | `GPULineMerge` `directed`, `GPUGridGenerator` flat hex, extent mask and unique square/hex corners, `GPURandomPointsOnLine`, `simplifyBoundary` | `line_merge` tolerance, `cluster_poisson` |

Deliberately skipped and still open: a dataframe function-call expression node (it edits the upstream
`gpu-expression` module), a `GPUGeometryColumn` bridge, an element-wise aligned-pair mode for predicates
and distance, monotone chains, `normalize`, directed H3 edges and vertexes, rhumb length in
`GPUGeometryMeasures` and `GPULineSegmentize`, and VW-preserve.

Findings:
- **Hausdorff:** `GPUTrackSimilarity` was vertex to vertex, and the old pinned cases agreed with Shapely by
  coincidence. It now measures to segments as GEOS does, so showcase values get slightly smaller.
- **`.gitignore` trap:** `coverage-*/` silently ignores a `coverage-topology/` directory, also from Biome.
  The unit lives in `polygon-coverage-topology/`; check any new `coverage-*` name.
- **WGSL reserved words:** `target` and `layout` fail to compile, in addition to the ones above.
- **Compiler reassociation:** `(90 - |a|) + (90 - |b|)` became a cancelling `180 - a - b`. Write the mean polar
  colatitude as `colat(a) - 0.5 * (|b| - |a|)`.
- **`is_ccw`:** matching Shapely needs the JTS 1.19 flat-top rule; the older highest-point rule missed 27 of
  171 lines.
- **`set_precision`:** GEOS pointwise mode keeps repeated points, so the survey claim that it drops them was wrong.
- **Visvalingam:** parallel local-minimum rounds do not reproduce heap order; the kept set differs by 3.1,
  0.65 and 0.13 percent of vertices at radius 1, 2 and 3, so it ships as approximate.
- **Spherical linear referencing:** brute force over points times segments, with no BVH and no overflow.
- **Earcut:** one thread per polygon, so it scales with polygon count rather than ring size.
