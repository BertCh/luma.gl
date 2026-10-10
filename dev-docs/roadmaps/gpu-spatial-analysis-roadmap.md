# GPU spatial analysis ontology and implementation roadmap

This maintainer roadmap turns the public ontology in
`modules/experimental/src/gpu-spatial-analysis/ontology` into an implementation plan. The
user-facing [GPU Spatial Analysis reference](../../docs/api-reference/experimental/gpu-spatial-analysis.md)
documents shipped behavior. This file records work that should be implemented, refined or
redesigned; it is not a release or staffing commitment.

## North star

The library should let an application express a spatial question as a bounded, typed flow of
GPU-resident data:

```text
source geometry or field
  -> spatial representation / index
  -> relation, neighborhood or topology
  -> aggregate, transform or model
  -> inference / classification
  -> renderable or readback result
```

The command graph remains the only planner and scheduler. Contributors remain independently useful
operations. Recipes remain explicit compositions of contributors. The ontology describes this
surface; it must not become a second execution framework.

Success means:

- compatible outputs connect without casts, repacking or application-local adapter kernels;
- coordinate, identity, cardinality, precision and failure semantics are visible in types;
- dynamic values update without graph recompilation;
- topology-changing work is capacity-bounded and reports completeness on the GPU;
- algorithms are searchable by task without requiring users to understand the flat export list;
- every implementation is grounded by an oracle, adversarial cases and representative scale data;
- package boundaries stay acyclic: primitive GPU data in `@luma.gl/gpgpu`, higher-level analysis in
  `@luma.gl/experimental`, and Arrow conversion in `@luma.gl/arrow`.

## Architectural layers

New work must name the layer it extends and preserve this dependency direction.

| Layer | Responsibility | Canonical contracts |
| --- | --- | --- |
| GPU Core | Scheduling, resources and reusable parallel primitives | `GPUCommandGraph`, graph views, sort, scan, reduction, compaction, hash/grid indexes and BVHs |
| Spatial substrate | Non-owning spatial shapes and relationship representations | Feature geometry ports, pair tables, `GPUSpatialWeights`, cell tables, sampled surfaces and status ports |
| Analysis contributors | One bounded analytical operation with no hidden submission or readback | `GPUCommandNodeProducer`, caller-owned outputs, explicit topology and dynamic parameters |
| Workflows | Named cross-contributor connective tissue | `add...Recipe`, typed result ports, graph-owned transients and caller-owned intermediate overrides |
| Discovery | Search, documentation and compatibility metadata | Data ontology, operator descriptors, capability families and named opportunities |

Discovery stores export names rather than constructor references, preserving tree shaking. Runtime
objects must not depend on the discovery catalogue.

## What the ontology revealed

The current surface is functionally broad: every public operator and workflow export is assigned
to an available capability family. The original audit covered 24 families; the catalogue now also
names the topology, overlay, tessellation, local-statistics and region-optimization families added
by this work. The implementation already covers most common 2D spatial
analysis workflows. The next step is therefore not to add unrelated algorithms at the same level of
abstraction.

The audit found several structural seams:

- Public operations share logical shapes but define many local geometry, pair and output types.
  There is no canonical typed port for a point set, line set, polygon set, compact pair table or
  capacity-bounded geometry result.
- Coordinate vocabulary varies by family: `spherical`, `geodesic`, `wgs84`, radius-bearing
  spherical modes, grid coordinates and planar native units overlap without one shared spatial
  context contract.
- Overflow reporting is widespread but not uniform. Candidate overflow, final-output overflow,
  truncated neighborhoods, iteration limits, uncertain predicates and non-convergence are different
  conditions but are often represented by similar one-word flags.
- Some outputs include exact required counts for resizing while others expose only clamped counts.
  Capacity estimates and sizing guidance are contributor-specific.
- Dynamic parameter buffers are a successful pattern, but schemas, scalar formats, field naming and
  validation remain local to each contributor.
- The catalogue is capability-family metadata. It cannot yet describe the materially different
  properties of each operator inside a family, such as exactness, determinism, dimensions,
  compile-time topology, dynamic fields or status outputs.
- Recipes compose implementations well, but repeated boundaries are not named. Similar workflows
  expose different intermediate overrides and result shapes.
- `GPURegionStatisticsReadback` is a useful explicit bridge, but it demonstrates that readback and
  asynchronous lifecycle semantics need a separate, consistently named adapter layer.
- The single flat entry point is useful for compatibility but is too large to be the only navigation
  surface. Task-oriented discovery and, eventually, stable domain subpaths are needed.

These seams divide the work into three categories.

## Work classification

### Implement

New algorithms or adapters that do not exist today:

- bounded noding, general polygonization and geometry repair;
- polygon overlay and true polygon buffering;
- vector Delaunay, Voronoi and concave construction;
- additional local statistics and directional dynamics;
- constrained region and facility optimization;
- likelihood, panel and multiscale spatial models, only after shared solver evidence;
- tiled, partition-preserving and prepared-index adapters where representative consumers prove the
  need.

### Refine

Existing behavior whose abstraction is sound but whose contract or evidence should be strengthened:

- spatial weights, DGGS, measurement, interpolation, clustering, regression and inference families;
- completeness reporting, total counts and capacity guidance;
- deterministic tie rules and documented nondeterministic reductions;
- precision, finite-value, empty-input and degenerate-input behavior;
- CPU-oracle coverage, performance baselines and memory/work counters;
- catalogue accuracy, task-oriented documentation and example-to-API traceability.

### Redesign

Surfaces where adding more local variants would make the API harder to compose:

- shared spatial data ports and geometry topology types;
- coordinate/spatial-reference context;
- result status and failure taxonomy;
- parameter schemas and compile-time versus dynamic property metadata;
- operator-level discovery descriptors;
- recipe inputs, outputs and intermediate override conventions;
- movement time representation and trajectory ports;
- readback adapters and asynchronous lifecycle naming;
- navigation beyond the flat export surface.

Redesign does not mean a wholesale rewrite. Each redesign tranche must migrate a small set of
existing contributors without changing their command nodes, then expand only after the shared type
has proved useful.

## Capability-family disposition

| Capability family | Disposition | Required work |
| --- | --- | --- |
| Dynamic parameters | Redesign | Add declarative parameter schemas, format/length metadata and reusable validation while retaining `GPUParameterBuffer` as storage. |
| Geometry measurement | Refine | Reuse a spatial context, align grouped/source outputs and document numerical error envelopes by coordinate model. |
| Geodesic operations | Refine | Share ellipsoid/radius definitions and distinguish spherical, rhumb and ellipsoidal semantics in metadata. |
| Geometry editing | Redesign | Accept canonical feature-geometry ports and return one consistent same-topology or compacted-geometry result. |
| Geometry construction | Redesign | Standardize generated geometry counts, required capacities, provenance IDs and draw-ready versus analysis-ready output. |
| Geometry validation | Refine | Share issue/status layouts and connect invalid locations directly to later repair operations. |
| Line processing | Redesign | Consolidate repeated path output shapes, coordinate options, provenance fields and overflow semantics. |
| Segment and coverage topology | Redesign | Extract reusable noded-segment, half-edge/ring ownership and robust predicate primitives. |
| Spatial ordering | Keep and refine | Retain Hilbert keys; add reusable ordering/index descriptors and cost guidance rather than a new index facade. |
| Spatial joins | Redesign | Unify prepared sources, candidate/final pair ports, completeness, uncertainty and optional weights output. |
| Spatial neighborhoods | Refine | Preserve `GPUSpatialWeights`; standardize producer status, transforms, rectangular matrices and source identity. |
| DGGS indexing/topology | Refine | Align cell-set ports, family/resolution metadata and capacity reports across H3 and Quadbin. |
| DGGS aggregation | Refine | Standardize sparse table occupancy, hierarchy identity and pyramid-level selection ports. |
| DGGS outlines/change | Refine | Return canonical polygon and comparison ports with explicit antimeridian behavior. |
| Density/interpolation | Refine | Share surface metadata, nodata/mask semantics, neighborhood status and uncertainty outputs. |
| Zonal/region statistics | Redesign | Separate GPU result production from optional readback adapters; align vector and raster region ports. |
| Group/regional analysis | Refine | Standardize labels, unassigned rows, objective diagnostics, iteration budgets and group geometry outputs. |
| Spatial autocorrelation/inference | Refine | Share statistic result layouts, alternative/tail conventions, multiple-testing outputs and permutation metadata. |
| Spatial regression | Refine | Share model matrices, coefficient/diagnostic ports, solver status and consistent mask/finite-row behavior. |
| Distribution dynamics | Refine | Align class identity, transition cubes, missing periods and significance outputs. |
| Accessibility/segregation | Refine | Name origin/destination support explicitly and share impedance, normalization and group contracts. |
| Change of support | Refine | Expose overlap weights and conservation diagnostics as standard reusable ports. |
| Movement analysis | Redesign | Define one trajectory/time port, timestamp precision policy and event/pair result contract. |
| Workflow recipes | Redesign | Adopt typed ports, predictable intermediate overrides, status aggregation and a common result envelope. |

## Canonical contracts to design first

The following are TypeScript structural types over existing graph views. They must not own buffers,
erase chunk boundaries or duplicate `GPUVector`.

### Feature geometry ports

Define precise, non-owning ports for:

- point coordinates with optional source IDs;
- line coordinates plus feature offsets;
- polygon coordinates plus ring and polygon offsets;
- trajectory coordinates plus track offsets and a timestamp column;
- generated geometry plus source provenance.

The port names should describe logical topology, not Arrow. Arrow and GeoArrow adapters create these
ports; contributors consume them.

Exit evidence: spatial joins, geometry measures, line processing and one recipe accept the same
ports without casts, copies or new runtime objects.

### Cardinality and status ports

Separate these conditions:

- `count`: usable rows or elements written;
- `requiredCount`: exact unclamped requirement when it can be computed;
- `overflow`: final result is incomplete because output capacity was too small;
- `candidateOverflow`: refinement may be incomplete because an intermediate capacity was too small;
- `uncertainCount`: exact classification was not possible inside the selected numerical policy;
- `converged` or `iterationLimitReached`: an iterative model stopped;
- `invalidCount`: input rows were rejected;
- `approximation`: the algorithm intentionally returns an approximate result.

Do not collapse these into a generic error code. Define small composable status ports and a recipe
status aggregate that points back to the responsible stage.

Exit evidence: joins, generated geometry, neighborhoods and iterative statistics use the shared
terms, and examples can explain whether a displayed result is complete without contributor-specific
logic.

### Spatial context

Separate four concerns that the current `coordinateSystem` options sometimes combine:

1. coordinate space: planar, longitude/latitude, discrete grid or topology-only;
2. metric: native planar units, spherical great-circle, rhumb or ellipsoidal geodesic;
3. model parameters: units, sphere radius or ellipsoid;
4. CRS transformation: owned by `@luma.gl/experimental/gpu-project`, not hidden in an analysis
   contributor.

The first version should be a compile-time descriptor used for validation and metadata. Do not add
per-row CRS objects or runtime reprojection to every contributor.

Exit evidence: measurement, line processing, density and trajectory families use one vocabulary;
incompatible workflow connections fail during graph construction with a short local assertion.

### Parameter schemas

Keep packed GPU parameter views, but describe each layout with reusable metadata:

- scalar name, format, word offset and default;
- units and valid range;
- whether the value may change between encodings;
- helper for packing and validation;
- optional catalogue reference for interactive controls and documentation.

Do not replace packed views with per-field GPU buffers or ship verbose runtime reflection into hot
paths.

Exit evidence: at least five unrelated families use the shared schema utility, generated constants
remain tree-shakable and existing WGSL layouts do not change.

### Operator descriptors

Keep capability families for human navigation, but add a second operator-level descriptor layer.
Each public contributor or recipe should declare, without importing its constructor:

- primary capability and secondary tags;
- accepted and produced ports;
- dimensionality and spatial context support;
- fixed topology versus dynamic parameters;
- source-sized, fixed-size or capacity-bounded cardinality;
- exact, robust-exact, approximate, stochastic or model-based semantics;
- deterministic versus order-dependent outputs;
- status ports and explicit readback behavior;
- relevant package and public export name.

The existing exactly-one-family test becomes exactly one primary capability plus any number of tags.
This avoids forcing multi-role operations into an artificially exclusive taxonomy.

Exit evidence: documentation tables and compatibility checks are generated from operator metadata;
the catalogue still adds no constructor references to bundles.

## Ordered implementation tranches

The sequence below is dependency-ordered. Later algorithm work should not invent local substitutes
for an unfinished earlier contract.

### Tranche 0 — Freeze the evidence baseline

**Disposition:** refine now.

- Inventory every public operator, recipe, output type, parameter layout, overflow/status field,
  coordinate option and explicit readback.
- Record representative input scale, transient memory, dispatch count and correctness oracle for each
  capability family.
- Add catalogue validation for unique IDs, primary ownership and resolvable relationships.
- Mark every catalogue claim as proven, partial or opportunity at the operator level; `available`
  alone is too coarse for detailed planning.

**Exit:** a generated audit report has no uncatalogued public export and every available operator
links to tests, documentation and one supported input/output contract.

### Tranche 1 — Shared ports and status vocabulary

**Disposition:** redesign first.

- Land feature geometry, compact pairs, generated geometry, spatial weights plus status, cell table
  plus occupancy, sampled surface and classified-value ports.
- Land the cardinality/status vocabulary and pure validation helpers.
- Migrate one vertical slice: spatial join -> pair gather -> group statistics or rendering.
- Migrate a topology-changing slice: line split or segment ring assembly -> output geometry.
- Replace divergent experimental props directly; migrated contributors expose only canonical ports.

**Exit:** two real workflows connect only through shared ports, preserve caller ownership and perform
no new copies, readbacks or submissions.

### Tranche 2 — Spatial context and parameter schemas

**Disposition:** redesign, then refine families.

- Introduce the shared spatial context vocabulary without moving CRS projection into this package.
- Migrate geometry measures, geodesics, line processing, line density and movement distance logic.
- Introduce parameter schemas and migrate five representative float32/uint32 layouts.
- Add graph-construction compatibility validation and documentation generated from schemas.

**Exit:** the migrated families use no private synonym for the same coordinate model, and parameter
changes remain allocation-free and recompile-free.

### Tranche 3 — Capacity planning and completeness

**Disposition:** refine broadly.

- Add pure `get...CapacityPlan` helpers for pair, neighborhood, generated geometry, cell and event
  outputs.
- Prefer exact `requiredCount` GPU outputs where an existing scan already computes the number.
- Distinguish intermediate/candidate overflow from final-output overflow.
- Add reusable application policy for fixed budget, grow-on-next-frame and fail-closed behavior;
  resizing remains application-owned.
- Surface capacity, peak transient bytes and work estimates in examples and benchmarks.

**Exit:** representative joins, cell covers, line operations and trajectory encounters can explain
and resize every incomplete result without contributor-specific interpretation.

### Tranche 4 — Recipe and navigation surface

**Disposition:** redesign the surface, not execution.

- Define a common recipe result envelope: contributors, named outputs, named intermediate ports and
  aggregated status.
- Make intermediate overrides predictable (`inputs`, `outputs`, `scratch` only when caller ownership
  is meaningful) and remove recipe-specific synonyms.
- Add task-oriented discovery helpers over operator metadata.
- Evaluate stable domain subpaths such as geometry, relationships, statistics, movement and recipes.
  Keep the flat entry point as a compatibility facade until graduation.
- Do not build a declarative planner until two applications need to assemble unknown workflows from
  metadata. Known workflows should continue calling contributors or recipes directly.

**Exit:** three existing recipes share the envelope and ports; a user can start from a task or data
kind and reach the correct operation without scanning the flat index.

### Tranche 5 — Geometry topology substrate

**Disposition:** implement the highest-value missing foundation.

- Define canonical noded segments with source feature, ring, edge and parameter provenance.
- Reuse exact orientation predicates and make the snap/precision policy explicit.
- Implement bounded segment noding for crossings, touches, collinear overlap and repeated vertices.
- Implement half-edge or equivalent directed-edge ownership sufficient for face/ring extraction.
- Upgrade ring assembly to report closed rings, open chains, cut edges, dangles and unused edges.
- Implement general polygonization and a first repair operation from the same substrate.

Do not begin with a monolithic GEOS-shaped facade. Land measurable primitives with independent
oracles.

**Exit:** arbitrary finite planar linework can be noded and polygonized into the canonical polygon
port on the GPU, with deterministic provenance, exact required counts and explicit incomplete/invalid
status.

### Tranche 6 — Overlay, buffer and tessellation

**Disposition:** implement after Tranche 5.

- Implement polygon intersection first, because it proves face classification and attribution.
- Extend the same substrate to difference, union and general dissolve.
- Implement line/arc offset generation, joins/caps and repair-backed polygon buffer assembly.
- Implement Delaunay with stable duplicate/collinear policies; derive vector Voronoi from the accepted
  triangulation topology.
- Add concave construction only after a specific definition and parameter contract is selected.

**Exit:** overlay operations match GEOS/JSTS oracles on adversarial fixtures, conservation checks pass,
and every topology-changing output is bounded with deterministic feature attribution.

### Tranche 7 — Tiled and partition-preserving scale

**Disposition:** implement only where consumers prove need.

- Accept `GPUVector`-backed or partitioned geometry without implicit packing where chunk identity is
  meaningful.
- Define halo, ownership and seam rules for tiled density, joins and topology.
- Reuse prepared indexes across compatible left-side batches and repeated encodings.
- Add cost models that choose scan, grid or BVH paths from measured workload properties; never hide a
  CPU readback in path selection.
- Keep streaming batch boundaries unless an explicit packing operation is requested.

**Exit:** one Arrow/table consumer and one tiled-map consumer share public primitives, preserve batch
or tile identity and demonstrate lower peak memory or repeated-query cost than packed execution.

### Tranche 8 — Statistical breadth

**Disposition:** implement after shared inference ports.

- Add local Geary, spatial Pearson and Gamma using the shared weights and result layouts.
- Add directional distribution/Rose statistics with an explicit circular-statistics contract.
- Expand permutation inference only through shared statistic adapters, not copied permutation loops.
- Add AZP or Max-P as the first regional optimization only after reproducible initialization,
  iteration budget and objective diagnostics are accepted.
- Add facility-location solvers only when GPU Network cost matrices and a real consumer establish
  useful bounds.

**Exit:** every new statistic reuses shared multiple-testing/permutation infrastructure, and search
algorithms publish objective, convergence and reproducibility evidence rather than only labels.

### Tranche 9 — Advanced models and graduation

**Disposition:** later and evidence-gated.

- Extract shared iterative linear algebra/solver status only when at least two model families need it.
- Evaluate maximum-likelihood lag/error, panel, regimes, SUR, probit and MGWR independently; do not
  treat parity with a CPU package as sufficient justification.
- Audit package dependency direction, public names, ownership, status semantics and bundle cost.
- Graduate stable substrate contracts before specialized models. Experimental algorithms may remain
  behind the stable ports.

**Exit:** two independent consumers use each graduation candidate, cross-package dependencies are
acyclic, supported semantics are frozen and experimental-only exports are clearly separated.

## Missing capability priority

| Priority | Opportunity | Dependency | Required contract before implementation |
| --- | --- | --- | --- |
| 1 | Geometry repair and general polygonization | Tranches 1–3 | Bounded noding, robust degeneracy rules, ring ownership, unused-edge diagnostics and completeness semantics |
| 2 | General polygon overlay and buffering | Geometry topology substrate | Capacity planning, deterministic feature attribution and explicit robust-predicate/precision policy |
| 3 | Vector Delaunay and Voronoi | Shared generated-geometry port | Stable tie rules, bounded topology and duplicate/collinear behavior |
| 4 | Additional local statistics | Shared inference result port | Common statistic, permutation-tail and multiple-testing layouts |
| 5 | Region and facility optimization | Shared iteration/status port | Reproducible initialization, stopping criteria, bounded budgets and objective diagnostics |
| 6 | Likelihood, panel and multiscale models | Shared solver evidence | Convergence/failure API and evidence that browser GPU execution is valuable |

## Scope boundaries

These are connections, not missing implementations in this entry point:

- CRS parsing and reprojection belong to `gpu-project`; spatial analysis consumes coordinates in an
  explicit context.
- Raster storage, tiling, nodata and raster algebra belong to `gpu-raster`; recipes may bridge to
  sampled-surface ports.
- Network topology and routing belong to `gpu-network`; spatial analysis may consume costs, paths or
  regions through shared ports.
- Generic column statistics, group-by, classification and temporal columns belong to
  `gpu-dataframe`.
- Arrow/GeoArrow conversion, upload and Arrow-compatible readback belong to `@luma.gl/arrow`
  adapters.
- Rendering models belong to `experimental/models`, deck.gl or application code. Analysis may
  produce render geometry and stable IDs but should not own map styling.
- File I/O, geocoding, basemap services and catalog management are application concerns.
- Full arbitrary-precision computational geometry is not a WebGPU claim. The library must document
  its finite-coordinate and precision policy and expose uncertainty where exact classification is
  not available.

## API-shaping rules

1. Use an existing ontology noun for inputs and outputs. Add a noun only when topology, identity or
   ownership cannot be expressed by an existing concept.
2. A contributor owns one analytical operation. Use a recipe when value comes from a named chain of
   independently useful operations.
3. Publish useful intermediate ports from recipes and accept caller-owned overrides so workflows
   connect to rendering or later analysis without copies.
4. Put per-frame values in graph views. Keep capacities, optional resources and algorithm topology
   compile-time and describe the boundary in operator metadata.
5. Topology-changing results are capacity-bounded and publish completeness on the GPU. Never hide a
   counting readback, submission or graph rebuild.
6. Keep coordinate space, metric and CRS transformation separate.
7. Keep memory layout and shader value type separate, following the core `GPUVectorFormat` and
   `ShaderLayout` boundary.
8. Preserve source batch, chunk and tile boundaries unless the caller explicitly requests packing.
9. Prefer adapters at package boundaries over duplicate primitive types.
10. Keep casts localized inside low-level adapters; workflow, renderer and example code should use
    precise public generics and ports.
11. Every stochastic or iterative operation publishes seed, budget, determinism and stopping
    semantics.
12. Update operator metadata, capability relationships, API docs, oracle tests and performance
    evidence with every public contributor or recipe.

## Quality gates for every tranche

- `nvm use`, `yarn install` when dependency state changes, `yarn lint fix` and `yarn build`.
- Targeted node tests for validation, empty inputs, capacities and graph declarations.
- Browser/WebGPU oracle parity for actual execution, including workgroup boundaries.
- Adversarial geometry for duplicate points, zero-length segments, touches, overlaps, holes,
  non-finite coordinates and near-degenerate predicates as applicable.
- Determinism checks or an explicit statement of reduction-order nondeterminism.
- Overflow, exact required-count and recovery-policy tests for bounded outputs.
- Memory, dispatch, transient-lifetime and repeated-encode measurements at representative scale.
- Documentation of compile-time topology, per-encoding parameters, units, source identity and known
  non-goals.
- `yarn test` for the final change set; `yarn test-node` is a focused check, not a substitute.
- Website reference/link checks whenever public navigation or generated API tables change.

## Immediate closure work

The original six-step bootstrap sequence and its closure pass are complete in this branch:

1. Record representative scale, transient-memory and dispatch evidence for the operator audit.
2. Migrate the shared spatial context into measurement, geodesic, density and movement
   contributors, with graph-construction compatibility checks at real workflow boundaries.
3. Route contributor parameter packers through shared schemas where their normalization semantics
   are expressible, and extend the schema contract before removing any special-case validation.
4. Expand canonical ports and status across neighborhoods, DGGS, density, regional analysis and
   movement instead of stopping at the join and topology slices.
5. Add adversarial runtime oracles, overflow recovery and conservation tests for topology, overlay,
   buffering and tessellation, including touches, overlaps, holes, duplicates and degeneracies.
6. Prove Tranche 7 with an actual tiled-map consumer and measured repeated-query or peak-memory
   improvement; keep advanced model exports gated on independent consumers.

The evidence catalogue now records scale, transient-memory, dispatch and oracle data for every
available family. Canonical ports are used by joins, topology, cells, weights, surfaces,
classification and recipes. The required measurement, geodesic, density and movement contributors
take `GPUSpatialContext` directly, and representative parameter packers use shared schemas.
Contributor-specific capacity plans cover joins, cell cover, line topology and trajectory
encounters. Arrow record batches can be uploaded and released one partition at a time; the measured
test includes value and validity buffers and demonstrates lower peak residency than packing. The
tile-LOD consumer uses the same core/halo and deterministic seam-ownership descriptors.

## Current implementation status

The contract foundation and the later contributor implementations are present. A tranche is only
marked complete when its exit evidence is also present; contributor availability alone is not the
same as tranche completion:

| Tranche | Status | Evidence in this branch |
| --- | --- | --- |
| 0 — Evidence baseline | Complete | Every public catalogue export has operator metadata. All 29 available families provide representative scale/output, transient-memory, dispatch and repository-oracle evidence; generated audits validate ownership, relationships, contracts and oracle paths. |
| 1 — Shared ports/status | Complete | Canonical geometry, trajectory, pair, weights, cell, surface, classification and composable status ports are public and directly used by join/gather/statistics, line topology, cells, weights and recipe workflows without ownership changes or hidden execution. |
| 2 — Context/schemas | Complete | Geometry measures, geodesic pair/destination, line density, trajectory metrics and cluster-outline consume mandatory `GPUSpatialContext`; old coordinate/model/radius synonyms were removed. Shared schemas drive unrelated trajectory, density and line-processing parameter packers. |
| 3 — Capacity/completeness | Complete | Pure generic plans and measured contributor plans for joins, cell covers, line topology and trajectory encounters report work, output and peak-transient estimates. Candidate and final capacity recover independently; representative contributors expose distinct flags, and the spatial-join benchmark publishes measured plan evidence. |
| 4 — Recipes/navigation | Complete | Operator task queries and the required common recipe envelope are public. All recipe entry points accept caller-owned overrides only through `inputs`, `outputs` and `scratch`, publish named outputs/intermediates, and attribute completeness or convergence to meaningful stages. Recipe tests and application callsites use the normalized surface. |
| 5 — Geometry topology | Complete | `GPULineNoding`, `GPUPolygonize`, and `GPUMakeValid` share bounded atomic-segment, directed-edge, diagnostics, polygon and status contracts. WebGPU fixtures cover crossings, endpoint touches, collinear overlap, repeated vertices, bounded faces, self-crossing repair and exact overflow requirements. |
| 6 — Overlay, buffer and tessellation | Complete | Arrangement-backed intersection, union, difference, symmetric difference and dissolve pass area/conservation fixtures; line and signed polygon buffers exercise repair assembly; Delaunay/Voronoi cover duplicate, collinear, cocircular, clipping and bounded-overflow policies. All topology-changing outputs use canonical bounded status and deterministic attribution. |
| 7 — Tiled/partition scale | Complete | Public partition, halo, seam, prepared-index and cost contracts are tested. Arrow iteration allocates one record batch at a time with stable global identity and measured value-plus-validity peak residency below packed residency; tile LOD consumes the same deterministic ownership vocabulary. |
| 8 — Statistical breadth | Complete | Local Geary, spatial Pearson, Gamma and directional/axial Rose statistics use shared inference vocabulary and deterministic CPU oracles. `GPUAZPRegions` publishes reproducible initialization, objective, iteration and convergence status. Permutation work is shared rather than copied. |
| 9 — Advanced models/graduation | Gate enforced | `GPUSolverStatusPort` is shared by OLS, spatial error GM and spatial 2SLS, and package/ownership/status audits are executable. No likelihood, panel, regimes, SUR, probit or MGWR API is admitted because none meets the roadmap's two-independent-consumer and browser-GPU evidence gate; this is an explicit scope decision, not a stub surface. |
