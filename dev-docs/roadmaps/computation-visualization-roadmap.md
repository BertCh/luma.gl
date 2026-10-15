# Computation visualization: a ladder-of-abstraction system for luma

Status: proposal, 2026-10-13. Research and planning only; no source changed. Two research rounds,
each a fan-out of Opus coordinators with Sonnet researchers:

- **Round 1 (project-focused):** an inventory of what the analysis contributors expose at each level
  of abstraction, external prior art, the docs and showcase platform, and a visual grammar worked
  through four contributor families.
- **Round 2 (general):**
  - the spatial domain (GIScience concepts, pedagogy, misconceptions, operation taxonomies);
  - cartography and interactive mapping;
  - algorithm animation, abstraction theory and learning science;
  - luma graphs as a general substrate compared with other compute systems;
  - visualization system architecture (grammars, coordinated views, provenance, embedding,
    accessibility).

Claims rest on fetched sources or on code read at `map-contributors` 731e2a066. Unverified sources are
listed at the end.

Related plans:
- [GPU analysis algorithms](./gpu-analysis-algorithms-roadmap.md) (GPU-only policy)
- [GPU spatial analysis](./gpu-spatial-analysis-roadmap.md) (ontology and contracts)
- [analysis contributors quality](./analysis-contributors-quality-roadmap.md)
- [gpugraph](./gpugraph-roadmap.md)
- [gpu-trace](./gpu-trace-roadmap.md)

## Goal

luma runs computation and rendering on one device, so it can show **the computation itself**, not
only its result. Any luma computation can be viewed at several levels of abstraction:

- why it is computed, and what function it computes;
- the dependency structure of the algorithm, and what one particular run actually did;
- how the device executed it;
- for one datum, one kernel, one contributor, or a whole workflow.

This plan designs a general visualization system for moving up and down those levels. It is
embedded in documentation and applications, and draws every view from a recorded real execution.
The analysis contributors (spatial statistics, networks, raster and terrain) are the first domain,
not the boundary. The same system serves dataframe and SQL queries, GPU Core programs, graph
algorithms, gpu-trace and, later, rendering pipelines.

Positioning: Roth's 2013 empirical study of map interaction found "calculate" to be the least
supported operator, which he read as "inadequate integration of visualization and computation". No
"process cartography" literature exists, and no published work generates explanatory visuals from
operation types. This system fills a gap the field has named but not filled. It also means there is
no validated design to copy, so every default below is a hypothesis for the user's manual review.

## Policies

1. **Real execution is the visual source.** Views read real compiled graphs, buffers and timestamps
   captured from the real computation. CPU re-simulation is never the visual source; CPU oracles
   remain test-time checks. The justification is correctness, authenticity and freedom from drift,
   not a measured learning gain, and no study shows such a gain. In detail:
   - **Correctness:** chaotic relaxation and frontier Bellman-Ford have intermediate states that a
     textbook replay gets wrong.
   - **Drift:** the docs' hand-modelled `GPUCoreCompilerAnatomy` already names nodes that no longer
     exist.
   - **Precedent:** Python Tutor showed that the recorded-trace architecture works at scale.
2. **Folded state is recovered on the GPU and labelled.** Where state is folded away (one permutation
   row, a reachability run capped at round k), a GPU replay recovers it and is labelled as a replay.
3. **Every mark carries its provenance, and the record states its observer effects.** Uncoalesced
   timing encodings, taps and lens twins all move a run away from production, and the record says
   which were in force.
4. **Instrumentation is compile-time opt-in.** Production graphs pay nothing.
5. **Views are lawful as well as honest.** Guards block misleading glyphs, such as metric distances
   on lon/lat or sums of intensive variables, without blocking the page.
6. **Nothing is upstreamed to visgl.** Core changes are fork-local and additive. The record schema
   therefore carries no external compatibility promise.

## Theory: what a view is

A view is a **typed projection of one recorded execution** (Lerner's Projection Boxes: start from a
full model of the run's semantics and project a subset). Its signature has five independent
coordinates.

| Coordinate | Values | Source |
| --- | --- | --- |
| **Level**: which question it answers | P Purpose (why, here) · M1 Specification (what function) · M2a Skeleton (declared dependency structure, schedule-free) · M2b Realized (what this run did) · M3 Execution (how the device ran it) | Marr's three levels; Poggio's levels above; Merrill and Garland's decoupled look-back, which shows that on GPUs the algorithm level is not independent of the schedule |
| **Grain**: unit of decomposition | datum or lane · kernel · contributor · recipe or workflow | McClamrock: grain is not a level |
| **Analysis stage**: which domain concept it is about (supplied by the domain pack) | Spatial: representation · support · neighbourhood · operator · inference · presentation | GIScience. Every hard misconception sits at a stage, not a level |
| **Bindings and free axes** | Axes such as entity, lane, workgroup, iteration, permutation draw, buffer slot, node, formula term, parameter, dataset. Each is bound to a value or set, or free with a layout role: position, small multiple, overlay, animate, aggregate | Victor's ladder of abstraction; Polaris table algebra; Wilkinson's cross and nest |
| **Record type**: which inspector draws it | buffer, CSR row, raster tile, permutation null, graph node, dispatch, loop trace, formula term… | Moldable development (Glamorous Toolkit): views chosen per object type. Houdini, TouchDesigner and Blender converge on this |

How navigation works with these coordinates:

- **Stepping down** binds an axis, for example by clicking a datum, an iteration or a sweep cell.
- **Stepping up** frees an axis under an aggregator, as small multiples, or as an overlay. These are
  Victor's three step-up forms.
- **Drilling** changes grain along a hierarchy (node → contributor → recipe).

All three are one reversible operator, *toggle binding*. The **datum is a pin, not a rung**: binding
row 17 turns the specification into a worked example, the realized algorithm into that row's journey
and settle time, and the execution into its lane.

The familiar **six rungs survive as a curated reading path for documentation**, not as the engine:

| Rung | Coordinates |
| --- | --- |
| Formula | M1 |
| Datum | data axis bound, any level |
| Kernel | M3 at kernel grain |
| Iterations | M2a and M2b at contributor grain, time free |
| Graph | M2a or M3 at contributor or recipe grain |
| Story | P |

New domains add axes, not levels. Examples:
- gpu-trace adds frame and draw axes;
- splats add tile and sort-key axes;
- dataframes add column and expression-node axes.

## Evidence that shapes the design

| Finding | Source | Consequence |
| --- | --- | --- |
| How learners use a visualization predicts benefit better than how it looks | Hundhausen, Douglas and Stasko 2002 (24 studies) | Spend effort on narrative, captions and correct records, not polish |
| Animation beats matched statics only slightly (g ≈ 0.23); continuous events are understood as discrete steps | Berney and Bétrancourt 2016 (61 studies); Tversky, Morrison and Bétrancourt 2002 | Discrete, reader-controlled steps; animation is never the evidence |
| The median reader of an interactive article interacts zero times | Conlen, Kale and Heer 2019 (50,000+ Idyll sessions) | Every view's static default must carry the argument on its own |
| Worked examples, fading and narration help novices and slow experts down | Kalyuga et al. (expertise reversal); Renkl and Atkinson (fading) | The entry point sets the default depth; analyst and engineer pages differ |
| Predict-then-reveal prompts improve recall; showing other readers' predictions adds nothing | Kim, Reinecke and Hullman 2017 | Optional, skippable prompts with no crowd predictions |
| Each extra representation costs translation effort | Ainsworth 2006 | At most two linked representations plus the native geometry view by default |
| Updates more than about 100 ms apart are not perceived as linked | Wang Baldonado et al. 2000; Shneiderman | Links propagate within 100 ms or the view visibly decouples |
| Fisheye distortion damages metric judgement; overview+detail keeps metric truth | Cockburn, Karlson and Bederson 2008 | Geometry-bearing views are never distorted; only the graph zooms semantically |
| Matched legends across a series improve comparison by about 28% | Brewer and Pickle 2002; Qu and Hullman 2018 | One scale object per variable, frozen per run, shared by every view |
| Crispness, location and value are the intuitive uncertainty variables; saturation and hue are not | MacEachren et al. 2012 | Provisional state is blurred or coarse; never encoded by saturation |
| Readers are frustrated by uncontrollable animated maps and miss changes between frames | Harrower 2003; Fish et al. 2011 | User-paced animation, a static key-frame strip, and an explicit delta as the default comparison |
| People treat maps and computers as "somehow less fallible" | MacEachren 1992 | Every result map can open its process |
| Slideshow stepping beat vertical scrolling for comprehension; leader lines aid retention | Zhi, Ottley and Metoyer 2019; Cartographic Perspectives 2022 | Stepped cards, not scrollytelling; leader lines from receipts to the map |

None of this evidence covers GPU or map-analysis readers. It transfers from CS1, lab and web
studies, so these are defaults to revise, not laws.

## Architecture

```
A  Record        GPUComputationRecord: graph forms, axes and relations, refs, decisions, choices
                 + sidecars (metrics, captures)                          [gpgpu + domain packs]
B  Domain packs  intent form, formulas, datum kinds, analysis stages, guards, custom views
C  Coordinator   params (navigation) + clause selections + scheduler; CPU and GPU backends
D  Views         ViewRegistry: record type -> view with signature (binds, free, layout)
E  Composition   ladder rail, juxtaposed linked views, DOI-folded graph, ScaleRegistry
F  Hosts         <ladder-lens> element, static twin, URL state, accessibility graph, exports
```

Each layer reads only the layer above it, and nothing above layer B ever sees a contributor.

### A. The record

Every system that introspects its programs well has converged on the same shape: a graph per
compilation form with stable ids, hierarchy as a namespace path, a deduplicated source table, metrics
in a separate document joined by node id, and optimizer decisions as data. Examples are XLA
`OpMetadata`, Google Model Explorer, Spark UI, JAX, Unreal RDG Insights and DataFusion. The record
adopts that shape.

```
GPUComputationRecord {
  schema: 'luma.computation/1'; id; runId; createdAt; adapter; domains: string[];
  graphs: GPUComputationGraph[];        // ordered forms: intent -> lowered -> compiled (-> encoded)
  axes: {axes[], keySpaces[], relations[]};
  refs: {formulas: FormulaRef[]; sources: SourceRef[]};   // deduplicated tables
  decisions: GPUDecisionRecord[];       // "why this variant", with rejected candidates
  choices: GPUAnalysisChoice[];         // analyst choices, by analysis stage
  workflow?: GPUWorkflowRecord;         // ordered graph or pass invocations chosen by state
}
GPUComputationGraph {id; form: 'intent'|'lowered'|'compiled'|'encoded'; domain;
  nodes: [{id; label; namespace; kind; op?; datum?; loop?: {id, iteration, phase}; dedupName?;
           derivedFrom?: string[]; formula?: refIndex; source?: refIndex;
           workgroupSize?; dispatch?; reads; writes; workload?; attrs}];
  edges: [{from; to; kind: 'dataflow'|'explicit'|'raw'|'war'|'waw'|'indirect'; resource?}];
  resources?: [{id; bytes; ownership: 'imported'|'transient'; allocation; firstNode; lastNode}]}
Sidecars, keyed by runId + graph + node id:
  GPUComputationMetrics  {timebase; encoding: {coalesced; timestamped; taps; lensTwin};
                          nodes: {id: {gpuBegin?; gpuEnd?; cpuEncodeMs; outcome?; counters}}}
  GPUComputationCaptures {snapshots; stamps; replays}
```

What each element does:

- **Graph forms linked by `derivedFrom`.** One mechanism covers SQL → plan → shader plan → nodes,
  operation tree → lowered → compiled, shader-pass chain → passes, and contributor → nodes. The link
  is many-to-many, because fusion and pass coalescing merge nodes while unrolling splits them. Views
  must show that multiplicity rather than pick one node.
- **`namespace`** is a slash path. It replaces id-prefix parsing (splat stage prefixes, gpu-graph
  `-iteration-N`, dataframe batch ids) and any stored producer tree; group nodes are derived.
- **`loop` and `dedupName`** turn unrolled iteration copies into structured coordinates, so folding
  is a group-by rather than a regex.
- **Metrics are a sidecar.** Structure is deterministic from construction, so it can be diffed and
  tested without a GPU. Two metrics files over one structure give an A/B comparison for free, which
  matches the quick-A/B working style.
- **`encoding.coalesced`** records an observer effect: attaching timestamps disables compute-pass
  coalescing (`gpu-command-graph.ts:1640`). Per-node times are real, but the timed run does not have
  the production pass structure, and every timing view says so.
- **`decisions`** keep the rejected candidates: `{subject, chosen, candidates[{id, score, supported,
  reason}], kind}`. Today `selectGPUStrategy` drops the losers, the reduction strategy drops even the
  reason, and `supportsDeferredScene` returns a bare boolean.
- **`choices`** are the analyst's choices as data: `{stage, name, value, alternatives, sweepable}`.
  Weights type, support, CRS, correction method, permutation count and bandwidth are examples. They
  answer "why this answer" and enable stability sweeps. MAUP has no fix except showing the
  alternatives.
- **`axes`** declare the axes and their hierarchies (lane / workgroup / dispatch, cell / tile,
  node / contributor / recipe), the key spaces, and the **mapping relations** between key spaces:
  `{from, to, cardinality, kind: 'function'|'buffer', ref, timeDependent}`.
  - **Function relations** are pure functions: `invocation = row`, or raster tile arithmetic.
  - **Buffer relations** are data: CSR `offsets/neighbors`, `edgeIds`, permutation buffers.
  - **Time-dependent relations** exist only if captured; otherwise they are declared *unknown*, never
    guessed.
  - Relations route through each family's primary key (a star topology), so N key spaces need about
    N relations, not N².
- **`workflow`** describes rendering as it runs: an ordered list of graph or pass invocations chosen
  by state. Ray tracing's four graphs, incremental execution's batch and merge graphs, and fft-bloom's
  mixed passes fit here. We do **not** retrofit a unified frame graph for visualization's sake.

### B. Domain packs

The core never knows a domain. A pack is a plain-data registration, keyed the way
`GPUOperationLoweringRegistry` keys lowerers:

```
GPUComputationDomainPack {
  id; describeIntent?(object) -> GPUComputationGraph;     // the domain's top form
  datumKinds: [{kind; label; indexSpace: '1d'|'2d'|'csr'|'batch'; toGeometry?}];
  formulaFor?(intentNode) -> FormulaRef;                   // derived or curated
  operators?: {[op]: {scope; scopeSecondary?; reduction; variableKinds?}};
  analysisStages?: string[]; guards?: GuardSpec[]; views?: ViewRegistration[];
}
FormulaRef {tex; ascii; symbols[{id; role; binding?}]; derivation: 'derived'|'curated'}
```

- **Formulas carry their derivation.**
  - *Derived* formulas come from a real AST and cannot drift:
    - dataframe `GPUExpression` and SQL, once the AST-node → `expression{N}` statement map that the
      shader plan already computes is stored;
    - GPU Core scalar operations (`alpha = rr / pq`);
    - gpu-project stages.
  - *Curated* formulas live in sidecars:
    - spatial, network and raster: the formula sidecar;
    - gpu-graph: PageRank, modularity;
    - rendering: BRDF, covariance projection.

    Curated formulas get node specs checking that every binding and WGSL name exists, plus a distinct
    provenance mark.
  - *Query-shaped* packs such as gpu-trace print their aggregation spec as a table; TeX is not forced
    on them.
- **The generic operator descriptor is `scope × reduction`.**
  - Scope values: local, focal, block, zonal, global, frontier, pairwise, solver.
  - Reduction values: none, count, sum, weighted mean, argmin, histogram, mode, select, scan/sort,
    solve.

  Map algebra (local, focal, zonal, global) is the backbone, and the axis transfers across domains:
  - focal becomes a dataframe window or a graph neighbour gather;
  - zonal becomes group-by or community aggregation;
  - global becomes a full reduction or a BFS frontier.

  The descriptor picks the **idiom family**, for example a stencil or CSR star, a funnel, a
  predicate matrix, a frontier, a replicate histogram or a convergence trace. The pack supplies the
  specifics, and a per-operator override covers unusual mechanisms. Fully automatic idiom choice is
  not feasible.
- **The spatial pack** adds:
  - the six analysis stages;
  - `variableKind` (nominal, ordinal, interval, ratio, cyclic; extensive or intensive), which gates
    lawful aggregation;
  - the representation facet (field, object, network, event), noting that contributors convert
    between these: KDE turns events into a field, terrain flow turns a field into a network and back;
  - guards;
  - GeoDa and PySAL visual vocabulary.

### C. Coordinator

`LadderCoordinator` replaces round 1's single-point `SelectionStore`. It is TypeScript with no DOM
and no dependency.

- **Navigation params versus selections** (Improvise, Mosaic).
  - *Params* are single-valued shared variables: iteration, parameter, rung, graph focus, expanded
    set, camera.
  - *Selections* are sets over key spaces built from clauses `{source, clients, predicate, value}`.
    A new clause replaces the previous one from the same source. Resolution is single, intersect or
    union. A view gets crossfilter self-exclusion by omitting its own clauses.
- **Four states, never conflated:**

  | State | Encoding |
  | --- | --- |
  | Hover | Achromatic outline |
  | Selection | Ink core and ground halo; others dim, never hidden |
  | Filter | Hide or soft-dim |
  | Focus | Camera, scrubber position and rail |

  Selection ink is achromatic, so it never collides with a data hue.
- **One writer per gesture, and views never echo.** A `writer` field records whether the narrative
  or the pointer owns the store, so explaining and exploring use one engine and the switch is visible.
- **Linking goes through key spaces and declared relations, never through geometry.**
  - **Highlight** is the image of the selection under a view's mark relation. **Pick** is the
    preimage, so one click on a 1:N mark selects a set.
  - **Coarse marks show partial coverage**, not a boolean.
  - **Keys a view cannot show** get an off-view cue (a count and an edge marker), because silence
    reads as "not involved".
  - **Aggregate selections are lowered** to keys before they link anywhere.
- **Two backends behind one clause API.**
  - CPU sets serve small key spaces: graph nodes, terms, one workgroup's lanes, one datum.
  - `GPUCrossfilter` serves brushes over large index spaces. It already provides clause-like
    dimension state, AND composition, leave-one-out masks and one packed readback per frame.
  - **One new GPU kernel**, `transferMask(relation)`, gathers a mask across a buffer relation (rows to
    cells, nodes or edges) without readback.
  - The map and other native geometry views consume the mask, or an index-keyed `Uint8` state buffer
    packed into the existing per-row `highlight` channel (`engine/layers.ts:140`), never CPU geometry.
- **Link heat** (Roberts):
  - **Hot:** hover and scrub.
  - **Warm:** a parameter edit or reader edit that re-runs the real GPU computation. Dependent views
    show a *stale* badge until the new record lands.
  - **Cold:** export.
- **The 100 ms rule.** A view that cannot update within 100 ms greys itself as *decoupled* and offers
  "update". Overview scrubs preview and commit on release (asymmetric coupling), which also bounds
  lens re-runs.
- **Step to an event by query**, alongside scrubbing, for example "first iteration where this cell
  settled", using per-datum stamps.

### D. Views

**ViewRegistry.** Views register by record type, with an order, an activation predicate on the
selection path, and a signature `{binds, free, layout}`. Every record type gets a typed table view
for free, like Houdini's attribute spreadsheet, so a computation with no custom views still has a
complete inspector. The authoring target is about 10 lines per view (the Moldable Inspector reports
9.2 ± 6.6, and 68% of its views were textual). The drawer enables a rung only when the record
satisfies that rung's signature; otherwise the rung is disabled with a reason, never faked.

**Every view has an axis header** showing its bound and free axes, for example
`row 17 pinned · iteration free (0–11 of 64) · dispatch level-0 · Apple M3`. Clicking a bound chip
frees it; clicking a mark binds it. Free axes are explicit wildcards ("all 64 iterations"), so an
aggregate is never mistaken for an instance.

Core views are generic and work across domains:

| View | Binds | Free | What it is |
| --- | --- | --- | --- |
| FormulaCard | — or e, t | term | KaTeX statement with symbol chips bound to buffers and kernels. A substituted state renders a worked example from real readbacks, with backward fading on later instances |
| DatumReceipt | e, t | term, buffer, neighbour | One entity's receipt, rows ordered by analysis stage, each row with a provenance badge; neighbourhood slot per family (CSR star, 3x3 stencil, candidate list, flag-offset-slot) |
| BufferStrip | b, t | slot | Index-space view: 1-D, 2-D tile or CSR brackets; typed table, diff, histogram, NaN and Inf badges, min/max envelope |
| DispatchView | p, t | workgroup, lane | Real dispatch geometry, at most one workgroup expanded, idle lanes hatched; the id-probe dispatch is the notional-machine demo |
| SkeletonView | — | step, element | M2a: declared step-indexed DAG or tree (GPU Gems 39, Blelloch work and depth), always drawn in outline as "structure, not this run's schedule" |
| RealizedView | e or Agg(e) | iteration | M2b: scrubber, convergence plot, log-spaced key-frame strip, history image from the snapshot arena; frames of chaotic algorithms captioned "a valid execution prefix" |
| CommandGraphView | t | node by hierarchy | Folded by degree of interest (`DOI = API − distance from selection`) under a node budget, with presets: recipe flow, algorithm (loops folded), pass × resource ladder, memory/aliasing. Toggles logical (prospective) and compiled (retrospective) graphs. Tiered inline previews: scalars inline, thumbnail on the selected node, heavy data in a pinned inspector |
| TimingView | run | node, time | One track per level (TensorBoard trace viewer), with the encoding flags stated |
| CompareView | — | — | A projection over two references (k vs k′, run A vs B, GPU vs oracle). Default is an explicit delta on the variable's diverging partner scale with the midpoint stated; swipe and juxtapose are secondary |
| StabilityMap | e | choice | Step-up over any sweepable choice: per-entity share of runs whose class is unchanged. A real GPU sweep over one compiled graph, labelling which axis varies |
| ScopeDiagram | e | input | Map-algebra input shading generated from `scope × reduction`: the fallback datum view for any operator without a custom receipt |
| Native geometry view | t, b | entity in its own space | The map for spatial data; camera space for splats, layout coordinates for graphs, the pixel grid for render targets. Owns position; index, iteration and rank never go on its axes |
| ComputationDrawer | — | rung | "How this ran": a ladder rail plus a **choices strip** in stage order; defaults to the current rung, one above and one below, plus the native geometry view |

Spatial-pack views:
- **NeighbourhoodView:** a connectivity histogram, hover star, island list and edge truncation,
  linked to the map.
- **MoranScatter:** quadrant colours, with a "slope = I" guard that holds only for row-standardised
  weights.
- **FDR staircase:** sorted p against i·α/m, with the achievable p grid k/(M+1).
- **Space-time cube:** wavefronts from `settledAt`, in a separate view so the map keeps pitch 0.
  This one is deferred.

Rejected views:
- force-directed network layouts, because they lose index-space truth;
- a generic thread animator, which is dishonest without instrumentation;
- ping-pong buffer diagrams, because no ping-pong exists;
- a continuous zoom from map to thread, which location primacy and the CSUR evidence both rule out;
- fisheye distortion of geometry-bearing views.

### E. Composition and encoding rules

**ScaleRegistry.** The showcase's class tables, hue registry, ramps and `directionFor` move into the
library, together with its chart renderers. Every view takes a `ScaleRef` and never builds its own
scale. The rules are:

1. **One scale object per variable.** It covers domain, breaks, palette, no-data colour, unit, method
   and midpoint, and is frozen per run. It never changes per frame, filter or view.
2. **Iteration domains come from the final state** or a declared reference. Values that overshoot go
   in a stated "beyond range" class.
3. **Classed by default.** Use quantile classes from published tables, not Jenks, unless argued
   otherwise. Use an unclassed fixed domain only for continuous inspection.
4. **Hue only for nominal data.** Use lightness for ordered data. Use diverging scales only with a
   stated meaningful midpoint.
5. **No turbo or rainbow for ordered data.** Cyclic ramps are only for periodic quantities.
6. **Iteration time gets its own channel**, never the variable's ramp.
7. **Provenance, provisional state and value use separate visual variables:**
   - **Lineage** uses texture: solid for GPU readback, hatched for GPU replay or derived values,
     outline for static metadata.
   - **Provisional or unconverged** state uses crispness, blurred or coarse.
   - **Value** uses lightness.
   - **Uncertainty is never saturation**, and value-by-alpha never doubles as compositing.
8. **Ramp direction follows the ground.** Views placed beside a night map inherit the night tables.
9. **Spatial support is always drawn**: cell size, bandwidth or neighbourhood in true metres.
10. **Structure carries no hue.** Hierarchy tiers apply to graphs too: subject at full ink, inputs at
    half, context faint.
11. **Labels go above the data**, with greedy priority placement.
12. **Animation is never the evidence.** It is user-paced, always paired with a static strip or chart,
    and turned into stepped frames under reduced motion.

A validator checks the Qu and Hullman consistency constraints, and exceptions are declared with a
reason. The showcase's `check-cartography` lint extends to view specs.

**Guards.** A guard renders as a VerdictChip and blocks the misleading glyph, not the page. Missing
metadata gives an explicit *unknown* state, never a guess. Guards:

- **Coordinate model:** no metric glyphs on geographic coordinates.
- **Count versus rate:** an extensive choropleth needs a denominator.
- **Conservation:** extensive sums are equal before and after aggregation.
- **Uncorrected p:** shown with a badge.
- **Classification needs inference:** a cluster class never appears without its significance.
- **Row standardisation:** must hold before "slope = I" is shown.

**Rung descriptor.** Each rung declares `{audience, task, interaction}`, following the map-use cube
and DiBiase's swoopy. Story rungs are public, presentational and low-interaction; kernel rungs are
expert, exploratory and dense.

### F. Hosts, state, accessibility and exports

- **One embed unit: `<ladder-lens>`.** It is a framework-neutral custom element authored as a
  markdown leaf directive, for example `::lens{rung=receipt record=hotspot-tracts bind=page}`. A remark
  plugin serves the Docusaurus site and a markdown-it rule serves the showcase, so one `.md` renders
  identically in both. The element takes string ids only, since React 18 passes strings to custom
  elements. It calls the vanilla `mountX(element, record, options)` API. This replaces per-view React
  wrappers.
- **Lenses on a page link by default**, through one coordinator named by `bind` (Idyll's model).
- **Every lens has a build-time static twin.** The twin is a `<figure>` with the recorded view at its
  permalink state, a generated description as alt text, the receipt as a `<table>`, and the formula
  as LaTeX plus MathML. SSR, print, llms.txt and readers without WebGPU get the twin. Each twin must
  have a caption and an inline key number (a docs contract test).
- **One description generator** feeds alt text, the polite live region (debounced, never per scrub
  frame) and the llms twin, so the three cannot drift.
- **Live lens with recorded fallback.** A lens is a small real GPU run of the real computation on the
  reader's adapter, using docs-sized real open data. Activation is opt-in and never happens during
  render or hydration. The recorded fallback is captured from a real GPU run and labelled with
  adapter, variant and date. A node spec checks that its ids match the scenario's structure.
- **Device and page budget.**
  - One device per page, using a `webgpu-analysis` profile with `timestamp-query` and the showcase's
    raised limits.
  - One canvas per live view; render on change only, and pause when the page is hidden.
  - IntersectionObserver activation; views far offscreen revert to their twin.
  - A page scheduler serializes lens runs.
  - Starting budget: at most 4 live views visible, 8 constructed, 16 MB per view and 96 MB per page.
    These numbers are judgement, to be checked once with an A/B, not benchmarked.
- **URL state:** `{runId, rung, bindings, clauses, params, focus}`, with `runId` mandatory. `runId`
  hashes the structure ids, the input dataset version, the seed and the adapter or variant. Permalinks
  survive record updates or fail visibly.
  - **Showcase:** the hash router gains an `l=` parameter.
  - **History:** rung and binding changes push a history entry; scrubbing replaces the current one.
  - **Action log:** the clause log is a log of labelled semantic actions. A branching provenance graph
    is not a goal.
- **Accessibility: the ladder is a (rung × entity) navigation graph.**
  - Up and Down change rung, which binds or frees an axis.
  - Left and Right move to the next sibling on the free axis; Home, End, PgUp and PgDn move further.
  - Enter pins.
  - Each lens is one tab stop with a roving tabindex.
  - WCAG 2.2: 3:1 contrast for selection ink and hatching; visible, unobscured focus; persistent hover
    content; pause over 5 s; stepped frames under reduced motion.
  - The design borrows Data Navigator's command model and is built in-house.
- **Exports.** All exports are pure functions of the record, with no dependencies:
  - Chrome Trace Event JSON for time: a thread per level, `args.nodeId`, counters for frontier size
    and residual, flows for RAW edges and `derivedFrom`. It opens in Perfetto, and each view offers an
    "Open in Perfetto" download.
  - Model-Explorer-shaped graph JSON.
  - Graphviz DOT, for docs snapshots and issue reports.
  - Text printers per graph form, for golden tests and logs. They are explicitly not an API.
  - Every resource and pass is also labelled and wrapped in `pushDebugGroup` scopes named by
    namespace, so WebGPU Inspector, Xcode and RenderDoc show the same hierarchy for free.
- **Dogfood.** A Chrome-JSON loader for gpu-trace lets luma render its own large traces (512+
  unrolled nodes, multi-frame workflows) in its own GPU timeline.

**Dependencies.**
- **Adopted:** KaTeX (MIT, about 77 KB gzip, at least 0.16.21). It is an app-level dependency of
  the website and showcase only. The library takes `renderTeX` as an option. `trust` is a function
  allowing only `\htmlClass` and `\htmlData`, and a `\term` macro tags symbols.
- **Not adopted initially:** a graph-layout library. Folded graphs are tens of nodes, and the
  compiler already provides a topological order. `@dagrejs/dagre` (MIT, 16 KB) is the pre-approved
  fallback, laid out per group.
- **Rejected:**
  - MathJax: larger and asynchronous.
  - elkjs: 433 KB, EPL-2.0, no incremental layout.
  - xyflow: an editor framework, and these graphs are read-only.
  - Mermaid and Graphviz-wasm: static and heavy.
  - Vega and Vega-Lite: CPU tuples, value-domain selections.
  - Mosaic: needs DuckDB, which would duplicate GPU-resident data.
  - Observable Plot, d3 and vsup: small SVG code suffices; the value-suppressing palette is about
    60 lines.
  - Trrack, Scrollama and Idyll: covered by the URL state, the coordinator and directives.

**Package layout.**
- **gpgpu:** the record, graph forms, capture helper, `transferMask`, exports and schema (no DOM).
- **Domain modules:** each pack beside its owner (`gpu-dataframe`, `gpu-spatial-analysis`, and so
  on).
- **`@luma.gl/experimental/computation-views`:** the coordinator, registry, views, ScaleRegistry,
  chart renderers and the `<ladder-lens>` element, all vanilla TypeScript.
- **Hosts:** directive adapters in the website and the showcase.
- **Rejected:** a separate workspace package (publish surface for no isolation gain while
  experimental), and React in the core.

## Seams to fill in the code

These are ranked by reach. All are additive and compile-time opt-in.

1. **`CompiledGPUCommandGraph.describe()`.**
   - Graph form, nodes with resource uses, and edges captured in `getNodeOrder` before they are
     discarded.
   - Resources with lifetimes and physical allocation.
   - Today the compiler builds RAW, WAR, WAW and explicit edges, then drops them. Compiled nodes are
     private, and `toJSON()` returns one string.
2. **`createWGSLKernelNode` retention** (289 files).
   - Keep `sourceRef` (module id and body hash), binding names and access, workgroup size, dispatch
     shape, `namespace`, `datum` and `loop`.
   - Give sort, grid index and grid binning a `workload`.
   - One hook reaches almost every contributor.
3. **Lowering-scope decorator.** `GPUCommandLoweringGraph.addComputePass` stamps `namespace` and
   `derivedFrom` on every node emitted under a lowering, with no contributor edits. Extend it to render
   and copy nodes.
4. **Operation tree.** Emit loop bounds, lowering and predicate expressions. Give opaque producers a
   `describeIntent` hook instead of a class-name leaf.
5. **Decisions.** `selectGPUStrategy` returns the candidate table. The reduction strategy keeps its
   reason. `supportsDeferredScene` returns the first failing feature. Projection decline codes map in
   directly.
6. **Timing.** `readTimings()` also returns begin and end ticks. `CommandEncoder` keeps
   `{passId, slot}` labels, so imperative renderers report per-pass records without becoming graphs.
7. **Capture helper over stepped execution.**
   - Run through a publication point and copy named views into a snapshot arena (snapshot k at offset
     k·size, mapped once). Add `retain` on transients in lens graphs only.
   - `addObservationTap(resourceId, {afterNode, sink})` is the hazard-correct primitive.
   - Snapshotting the **value arena** per iteration gives conjugate-gradient residual, alpha and beta
     trajectories at one copy each, the cheapest iteration capture in the codebase.
8. **Dataframe and SQL.** `CompiledGPUDataFrameQuery.describe()` returns the graph forms. Store the
   AST-node → `expression{N}` map. `LuSQLQuery` keeps its statement AST.
9. **Contributor `describe()` and `publication`** on each stage's last node, starting with the
   teaching spine. Contributors also declare their key spaces and relations.
10. **Per-datum stamps:** raster `settledAt`, network last-improved round, FDR ranks and m.
11. **Loop traces:**
    - Per-round rows written by the existing gate kernels.
    - Standardize `activeIterations` on every unrolled loop (cost distance, terrain fill, flats,
      accumulation, AZP), copying network reachability's pattern.
12. **Replay kernels:** a single-row Philox permutation replay with an optional `nullDistribution`
    sink, and replay-with-limit for networks.
13. **`GPURecipePlan`.** Built by `RecipeBuilder`, with glue kernels routed through the builder. A
    coverage spec checks that every node in a recipe graph belongs to a plan step.
14. **Choice records** come from `GPUSpatialParameterSchema` plus weights, CRS and correction
    metadata.
15. **`GPUCrossfilter` external-mask dimensions** and the `transferMask` kernel.
16. **Scene workflow.** `Scene.workflow` references plan steps. Pipeline chip strips are generated
    from the plan, with hand labels as overrides; today they are hand-written in 49 scenes and link
    to nothing.

## Teaching spine and proofs

The first three proofs cover three domains and two audiences:

| Proof | Domain | Audience | What it proves |
| --- | --- | --- | --- |
| GPU Core anatomy, rebuilt from a recorded real `GPUScan` | core | engineers | `describe()`, CommandGraphView, the memory and aliasing story; retires hand-modelled data |
| Dataframe expression | dataframe | analysts and engineers | The only exact, non-drifting formula → WGSL statement → node chain, with a per-row receipt |
| Network reachability rounds | network | both | Realized iterations from `activeIterations` with no contributor change |

The spatial teaching spine follows, one new concept per exemplar, on real open data already in the
showcase:

| # | Exemplar | New concept | Misconceptions it carries |
| --- | --- | --- | --- |
| 1 | Spatial lag | Neighbourhood is a choice: NeighbourhoodView with queen vs KNN, then lag | M6, M7 |
| 2 | Kernel density | Composition, intermediates, strategy decisions | M13 |
| 3 | Local Moran, permutation, FDR | Inference; local sums to global | M1–M5 |
| 4 | Network accessibility | Iteration, frontier, decay formula | M15 |
| 5 | Terrain flow (cost distance as the simpler sibling) | Gated loops, multi-stage convergence, skeleton vs realized | M14 |
| 6 | Rate cluster map recipe and story | Workflow, choices, stability over H3 resolution | M9–M11 |

Misconceptions are built into views by default and corrected where they arise, never in footnotes:

| # | Misconception | Correction built into the view |
| --- | --- | --- |
| M1 | A Moran quadrant or Gi value is significant | Class never shown without significance; quadrant and p on separate rows |
| M2 | Pseudo-p is exact | The achievable grid k/(M+1) and floor 1/(M+1) are drawn on the p axis |
| M3 | One 0.05 test per unit with no correction | FDR staircase is the default inference view; "uncorrected" is badged |
| M4 | Local tests are independent | Hover lights neighbours shared with other significant units |
| M5 | A significant unit marks the cluster's extent | Cores solid, neighbours outlined, captioned "core" |
| M6 | Weights are neutral | Neighbourhood is a first-class view; islands hatched and counted |
| M7 | Row standardisation is cosmetic | The slope = I guard |
| M8 | Distances or buffers in degrees | The coordinate-model guard |
| M9 | Counts mapped as if they were rates | The count-vs-rate guard; crude and EB maps side by side |
| M10 | EB reference rate = mean of rates | μ = ΣO/ΣP chips |
| M11 | Zoning is given | Resolution swipe and StabilityMap |
| M12 | Extensive and intensive aggregate the same way | The conservation VerdictChip |
| M13 | Bandwidth or IDW power is a detail | Weight-vs-distance curve beside the surface |
| M14 | D8 diagonals cost 1 | 1.414 on stencil diagonals; "all input becomes runoff" caption |
| M15 | Isochrone ≈ buffer | Euclidean buffer outline overlaid |
| M16 | Patterns are visible by eye | Lineup |

**Page design.**
- **Analyst-facing pages** (spatial analysis, network, raster) are concrete first: map feature →
  DatumReceipt with a substituted formula (a worked example) → symbolic formula → iterations → kernel
  → graph.
- **Engineer-facing pages** (GPU Core, gpgpu) open at DispatchView or CommandGraphView with "up one
  level" controls.
- Each page opens with a short stepped stem of captioned states that reads with zero interaction,
  then hands over to the free drawer (martini glass).
- Each docs family has a "GPU execution model" panel rendered from real `describe()` output: the
  notional machine made explicit.
- Each rung transition offers at most one optional predict-then-reveal prompt, on concept and
  tutorial pages and the showcase, never in operator reference.

## Phases

Each phase ends usable on its own. `gpgpu/gpu-core`, `wgsl-kernel-nodes.ts`, the ontology,
contributor source and the showcase are edited by peer sessions, so each phase claims its directories
before starting. Showcase and docs work is gated on tsc and Biome; the user checks visuals by hand,
with no headless capture sweeps.

| Phase | Deliverable | Exit gate |
| --- | --- | --- |
| V0 Record and seams | Record schema (graph forms, axes and relations, refs, decisions, choices, workflow, sidecars); `describe()`; `createWGSLKernelNode` retention; lowering-scope `namespace`/`derivedFrom`; operation-tree loop fields; decision candidate tables; begin and end ticks; encoder pass labels; capture helper and snapshot arena; `pushDebugGroup` labels | Node specs on hand-built graphs (edge kinds, lifetimes, aliasing, `derivedFrom`, schema validation); `yarn build`; one GPU spec for capture |
| V1 Runtime and hosts | `computation-views`: ViewRegistry with table fallback, axis headers, coordinator (CPU backend), ScaleRegistry and chart renderers moved from the showcase, `<ladder-lens>` with both directive adapters, static-twin and description generators, URL state, accessibility graph, page scheduler and device profile; Chrome JSON, DOT and text exports | tsc, Biome, node specs (registry signatures, coordinator clauses, scale validator, twin contract); showcase typecheck unchanged |
| V2 Three proofs | GPU Core anatomy from a recorded `GPUScan`; dataframe derived-formula pack with receipt; network reachability RealizedView | Docs contract tests; capture-id spec; manual check |
| V3 Spatial statistics slice | Spatial pack (stages, `scope × reduction` and `variableKind` for spine operators, guards, choice records); formula sidecar; FormulaCard with fading; DatumReceipt by stage; NeighbourhoodView; MoranScatter; FDR staircase; Philox replay; StabilityMap; GPUCrossfilter backend and `transferMask`; spatial-analysis concepts page and hot-spot recipe page | Replay bit-identical to the real run; sidecar binding specs; guard specs; coverage spec for declared scope |
| V4 Iteration | Raster and terrain stamps, gate logs and `activeIterations`; SkeletonView and RealizedView split; key-frame strip, crispness for provisional state, step-to-event; CompareView delta; reader edits as warm links (draggable cost-distance origin, bandwidth ring) | Fixpoint results unchanged with instrumentation off and on |
| V5 Workflows | `GPURecipePlan` for all recipes with coverage spec; DOI-folded CommandGraphView with presets and tiered previews; generated pipeline strips in the showcase; Model-Explorer export; gpu-trace Chrome-JSON loader | Coverage spec; `check-story-steps` |
| V6 Kernel rung | DispatchView with id probe and scan lens; kernel source view; notional-machine panels | Manual check |
| V7 Showcase drawer and docs IA | ComputationDrawer with choices strip in every scene that declares a workflow; spatial analysis, network and terrain docs restructured like `gpu-raster/` (Overview, Concepts, Operators, Recipes); the 5,328-line `gpu-spatial-analysis.md` split by family in the same change as the showcase reference glob | `yarn test-website` only when no peer shares the tree; docs and showcase contracts |
| V8 Breadth | Packs for gpu-graph, gpu-trace and rendering (timing-only first, then `CompositeShaderPass` intent; splat stage prefixes migrated to `namespace`); `scope × reduction` across all operators; network, raster and terrain ontology; space-time cube | Ontology and pack coverage specs |

Status V7 docs IA and V1 hosts (stream D, 2026-10-14): `gpu-spatial-analysis.md` split into
`gpu-spatial-analysis/` (overview, concepts, operators index, eleven operator families, recipes); network
and terrain moved to Overview/Concepts/Operators folders; lens slots on every contributor; `::lens` hosts
(website remark plugin, showcase markdown-it rule) print identical static twins; coverage and twin specs
in `modules/experimental/test/computation-docs/`.

Status V0 record and seams (stream R, 2026-10-14): schema, validation, builder, metrics and
captures sidecars, exports (Chrome trace, Model Explorer, DOT, text) and `GPUSnapshotArena` /
`captureGPUCommandGraph` / `GPUTransferMask` in `modules/gpgpu/src/computation-record/`;
`GPUCommandGraph` `recording` opt-in with `withRecordScope`, `describe()` (kinded edges, resources,
allocations), `retain`, `addObservationTap`, namespace debug groups, begin/end ticks, encoded form;
program-compiler loop provenance and `describeGPUProgramCompilation`; strategy candidate tables;
`getDeferredSceneSupport`; `createWGSLKernelNode` retention; GPUCrossfilter external masks.
`CommandEncoder` `{passId, slot}` labels are not done (rendering is timing-only until V8).

Status V1 runtime (stream V, 2026-10-14): `@luma.gl/experimental/computation-views` (ViewRegistry with
table fallback and disabled-with-reason rungs, axis headers, LadderCoordinator with CPU backend and a
GPU-backend interface, ScaleRegistry and §E validator over the copied showcase tables/ramps/charts,
guards and VerdictChip, provenance textures and crispness, FormulaCard, DatumReceipt, BufferStrip,
DispatchView, SkeletonView, RealizedView, CommandGraphView with DOI folding, TimingView, CompareView,
ScopeDiagram, a `result` rung (native geometry view, input → output, from captures tagged `role` +
`geometry`), one description generator, pure static twins with thumbnails, a per-scenario `runExample`
convention, URL state, rung × entity keyboard graph,
page scheduler, `webgpu-analysis` device profile, `<ladder-lens>`) and the scenario contract
`@luma.gl/experimental/computation-lenses`. The GPU selection backend (GPUCrossfilter external masks
plus `GPUTransferMask`) is an interface only; StabilityMap and the Purpose rung are not built.

## Decisions taken on open questions

The research streams raised these. Each is decided here, with its trade-off.

| Question | Decision | Trade-off |
| --- | --- | --- |
| Live or recorded docs | Live lens with labelled recorded fallback | Recorded-only is reproducible but stops being the reader's real GPU; live-only shows nothing without WebGPU |
| Upstreaming | Fork-local, additive | No external review; schema carries no compatibility promise |
| Public or internal views | Public experimental subpath | Experimental carries no stability promise; deck.gl examples can use it |
| Trace outputs on public props | Opt-in compile-time props, no `debug` entry point | A parallel entry point would duplicate every class |
| Shader instrumentation | Targeted stamps, gate logs, id probes and replays only; no per-access trace buffers | Intra-dispatch behaviour stays unobservable, and views say so |
| Audience per docs family | Analysts: concrete first. Engineers: kernel first | Two page templates to maintain |
| Reader telemetry | None; design reviewed by hand | Designing without usage data, accepted for a personal fork |
| Prompts | Concept, tutorial and showcase pages only; never operator reference | Reference readers get no scaffolding |
| Autoplay | Step by default; a "play once" under 5 s for short transitions; off under reduced motion | Follows Berney and Bétrancourt's system-paced result without making play the evidence |
| Narrative genre | Stepped cards with leader lines, not scrollytelling | Matches the showcase; scroll has no comprehension advantage in the fetched evidence |
| Guard severity | Block the glyph, keep the page; *unknown* when metadata is missing | Some views show a guard instead of a picture |
| Default inference view | FDR staircase, with uncorrected results badged | Diverges from the PySAL book's uncorrected 5% |
| Sweeps in docs lenses | Allowed, small (for example 5 resolutions × 3 weights types), opt-in, within the page budget | Device time on the reader's adapter |
| Ontology facets | In the spatial ontology, claimed per phase; the generic `scope × reduction` type lives in `computation-views` | Coordination with peers editing the ontology |
| Cartography guide | Its rules and colour code become the library-wide ScaleRegistry; geo-only rules stay in the spatial pack | One more move out of the showcase |
| Reader edits | Yes in docs and showcase, as warm links, from V4 | Needs lens infrastructure first |
| Space-time cube | A separate view may pitch; deferred to V8 | Occlusion; kept off the map |
| Lens linking | Lenses on a page link by default | Authors opt out for unrelated lenses |
| GPUCrossfilter change | External-mask dimensions in the fork; `transferMask` in gpgpu | Touches a file the fork already patches |
| Branching history | No; URL plus clause log | No saved branching sessions |
| Accessibility library | In-house, borrowing Data Navigator's model | A little more code, no dependency |
| External viewers | In-site views are primary; "Open in Perfetto" is an optional download | Perfetto is a Google property; nothing is uploaded |
| Rendering scope | Timing-only records until V8 | Rendering pipelines get no ladder until late |
| Dataframe proof | In V2 | Smaller showcase presence than spatial, but the only derived formula chain |
| KaTeX accessibility | MathML output plus adjacent prose | Revisit MathJax only if an accessibility review requires speech navigation |

## Risks

- **Schema breadth.** A general record invites over-design. Add a field only when a view consumes it;
  build packs for core, dataframe and spatial first.
- **Many-to-many `derivedFrom`.** A selected intent operation may light ten nodes. Views show
  multiplicity rather than picking one.
- **Observer effects compound.** Uncoalesced timing, taps and lens twins each move further from
  production. The record states each one, and specs prove fixpoint results are unchanged.
- **Chaotic and device-dependent intermediates.** Per-round frames and fused graphs differ across
  adapters. Every view shows adapter and variant; captions say "a valid execution prefix".
- **Memory.** Map-scale snapshots (512² × 100 frames is about 100 MB) need a byte budget and
  decimation; stamps are the default for that reason.
- **Binding ceiling.** Selection state must pack into the existing per-row channel buffer, because
  layers are at 5 to 7 of 8 storage bindings.
- **100 ms on big lens graphs.** Mitigated by commit-on-release and the decouple indicator; measure
  once in V3.
- **Taxonomy brittleness.** Multi-scope operators escape defaults. The override tier and a coverage
  spec contain it.
- **Guard false positives.** CRS metadata is often missing; guards use *unknown*, never a guess.
- **Over-scaffolding experts and over-interaction on public rungs.** Enforced by the rung descriptor
  and per-page templates.
- **Too many views.** The registry makes views cheap. Every registration must declare its axes and
  links, and the default shows two representations plus the native geometry view.
- **Honesty erosion.** The pull toward pretty CPU animations is strong; provenance and lawfulness
  marks are enforced in review.
- **Weak evidence transfer and no precedent.** No study covers GPU or map-analysis readers, and no
  process cartography exists. Every default is a hypothesis for manual review.
- **Peer collisions** in gpu-core, `wgsl-kernel-nodes.ts`, the ontology and the showcase. Claim
  directories per phase.

## Sources

The research reports, with URLs and per-claim verification tags, are in the planning session's
scratchpad (`viz/reports/A–D`, `viz/r2/reports/S1–S5`, raw notes in `streams/`).

**Verified anchors:**
- Victor, *Up and Down the Ladder of Abstraction*, *Learnable Programming* and Tangle.
- Marr via the Stanford Encyclopedia of Philosophy; Poggio 2012; Merrill and Garland 2016.
- Chis et al., *The Moldable Inspector* (Onward! 2015); Lerner, *Projection Boxes* (CHI 2020);
  Guo, Python Tutor (SIGCSE 2013).
- Hundhausen et al. 2002, via Sorva and Shaffer; Berney and Bétrancourt 2016;
  Tversky, Morrison and Bétrancourt 2002; Conlen, Kale and Heer 2019.
- Kalyuga 2007; Kim, Reinecke and Hullman 2017.
- Polaris (Stolte, Tang and Hanrahan 2002); Vega-Lite selections; Mosaic.
- North and Shneiderman, Snap-Together 2000; Roberts 2004; Wang Baldonado et al. 2000;
  Cockburn, Karlson and Bederson 2008.
- Heer et al., *Graphical Histories* 2008; Zong et al. 2022; Data Navigator; Olli; WCAG 2.2.
- Roth 2013 (interaction primitives) and Roth 2015 (visual variables);
  MacEachren 1992 and MacEachren et al. 2012.
- Brewer and Pickle 2002; Qu and Hullman 2018; Harrower 2003; Andrienko et al. 2002; Kraak 2017;
  Ragan et al. 2016; Segel and Heer 2010.
- ColorBrewer; CARTOColors.
- GeoDa workbook (4a, 5a, 6a, 3b); *Geographic Data Science with Python*; esda; GIS&T Body of
  Knowledge (MAUP; map algebra, Mennis 2022); Kuhn 2012; Goodchild, Yuan and Cova 2007;
  Scheider et al. 2020.
- XLA `OpMetadata`; Model Explorer `input_graph.ts`; Perfetto's JSON importer; DataFusion and
  DuckDB EXPLAIN; Unreal RDG; TensorFlow graph visualizer (VAST 2018); Unity Render Graph Viewer;
  GPU Gems 3 ch. 39; Bostock, *Visualizing Algorithms*; Red Blob Games A*.
- KaTeX `trust` and `\htmlClass`; dagre.

**Not verified; check before quoting in shipped docs:**
- Primary texts: Tomlin 1990; Albrecht 1996; Anselin 1995; Câmara 2000; DiBiase 1990;
  Harrower and Fabrikant 2008; Kraak 2003; Crameri 2020.
- Munzner's nested-model details; Wilkinson's operator semantics; Ainsworth's cost sub-tasks;
  Mayer effect medians by edition.
- Trace Event Format flow semantics; Model Explorer custom-graph loading.
