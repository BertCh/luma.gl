// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Schema of `GPUComputationRecord`: one recorded real execution of a luma computation, drawn by
 * the computation views. Structure (graphs, axes, refs, decisions, choices) is deterministic from
 * construction and can be diffed and tested without a GPU. Measurements and readbacks live in
 * separate sidecars (`GPUComputationMetrics`, `GPUComputationCaptures`) joined by `runId`, graph
 * id and node id, so two metrics files over one structure form an A/B comparison.
 *
 * The schema is fork-local and carries no external compatibility promise. Fields exist only when
 * a view consumes them; see `dev-docs/roadmaps/computation-visualization-roadmap.md` §A.
 */

/** Schema identifier of {@link GPUComputationRecord}. */
export const GPU_COMPUTATION_RECORD_SCHEMA = 'luma.computation/1';
/** Schema identifier of {@link GPUComputationMetrics}. */
export const GPU_COMPUTATION_METRICS_SCHEMA = 'luma.computation-metrics/1';
/** Schema identifier of {@link GPUComputationCaptures}. */
export const GPU_COMPUTATION_CAPTURES_SCHEMA = 'luma.computation-captures/1';

/** Plain JSON-compatible attribute value. */
export type GPUComputationAttributeValue = string | number | boolean;

/** Free-form, JSON-compatible node or resource attributes shown by the table fallback view. */
export type GPUComputationAttributes = Readonly<Record<string, GPUComputationAttributeValue>>;

// ---------------------------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------------------------

/** Adapter that produced the run. Every view shows it, since intermediates are device-dependent. */
export type GPUComputationAdapter = {
  /** Backend, such as `webgpu`. */
  backend: string;
  /** Adapter vendor reported by the device, when known. */
  vendor?: string;
  /** Adapter architecture reported by the device, when known. */
  architecture?: string;
  /** Adapter description or device name, when known. */
  description?: string;
  /** Implementation variant label, for example a strategy or autotuned kernel set. */
  variant?: string;
};

/** One recorded real execution: graph forms, axes, refs, decisions and choices. */
export type GPUComputationRecord = {
  /** Schema identifier, always {@link GPU_COMPUTATION_RECORD_SCHEMA}. */
  schema: typeof GPU_COMPUTATION_RECORD_SCHEMA;
  /** Stable record identifier, for example a lens scenario id. */
  id: string;
  /**
   * Run identity. Hashes the structure ids, input dataset version, seed and adapter or variant,
   * so permalinks survive record updates or fail visibly. Sidecars join on it.
   */
  runId: string;
  /** ISO 8601 creation time. */
  createdAt: string;
  /** Adapter that executed the run. */
  adapter: GPUComputationAdapter;
  /** Domain packs that describe this record, such as `gpu-core` or `gpu-spatial-analysis`. */
  domains: readonly string[];
  /** Ordered graph forms, most abstract first: intent, lowered, compiled, encoded. */
  graphs: readonly GPUComputationGraph[];
  /** Declared axes, key spaces and the relations that map key spaces onto each other. */
  axes: GPUComputationAxes;
  /** Deduplicated formula and source tables referenced by node `formula` and `source` ids. */
  refs: GPUComputationRefs;
  /** Implementation decisions with their rejected candidates. */
  decisions: readonly GPUDecisionRecord[];
  /** Analyst choices as data, in analysis-stage order. */
  choices: readonly GPUAnalysisChoice[];
  /** Ordered graph or pass invocations chosen by state, for multi-graph computations. */
  workflow?: GPUWorkflowRecord;
};

/** A record together with its optional sidecars, as produced by a lens scenario run. */
export type GPUComputationRecordWithSidecars = GPUComputationRecord & {
  /** Measurements, one entry per measured graph. */
  metrics?: readonly GPUComputationMetrics[];
  /** Readbacks: snapshots, per-datum stamps and GPU replays. */
  captures?: GPUComputationCaptures;
};

// ---------------------------------------------------------------------------------------------
// Graph forms
// ---------------------------------------------------------------------------------------------

/**
 * Compilation form of a graph. `intent` is the semantic operation tree or contributor intent,
 * `lowered` the backend-lowered operation nodes, `compiled` the scheduled command graph and
 * `encoded` the realized pass structure of one encoding (coalescing, skipped nodes).
 */
export type GPUComputationGraphForm = 'intent' | 'lowered' | 'compiled' | 'encoded';

/** One graph form with stable node ids. */
export type GPUComputationGraph = {
  /** Record-unique graph id. Node ids are unique within a graph. */
  id: string;
  /** Compilation form. */
  form: GPUComputationGraphForm;
  /** Domain pack that owns this graph's node kinds and ops, such as `gpu-core`. */
  domain: string;
  /** Human-readable label. */
  label?: string;
  /** Nodes in schedule order for compiled and encoded forms, declaration order otherwise. */
  nodes: readonly GPUComputationNode[];
  /** Dependencies between nodes of this graph. */
  edges: readonly GPUComputationEdge[];
  /** Logical resources with lifetimes and physical allocation, for compiled forms. */
  resources?: readonly GPUComputationResource[];
  /** Free-form graph attributes. */
  attrs?: GPUComputationAttributes;
};

/**
 * Node kind. Execution kinds are `compute`, `render` and `copy` (compiled) and `pass` (encoded);
 * semantic kinds are `operation`, `composite`, `loop` and `conditional` (intent and lowered).
 * Domain packs may add their own kinds.
 */
export type GPUComputationNodeKind =
  | 'compute'
  | 'render'
  | 'copy'
  | 'pass'
  | 'operation'
  | 'composite'
  | 'loop'
  | 'conditional'
  | (string & {});

/**
 * Structured coordinate of one unrolled loop copy, so folding iterations is a group-by on
 * `loop.id` rather than a regex over node ids.
 */
export type GPUComputationLoopCoordinate = {
  /** Loop identity, stable across its unrolled copies (for example the loop operation id). */
  id: string;
  /** Zero-based iteration index of this copy. */
  iteration: number;
  /**
   * Phase within the iteration: `unconditional` copies always run; `gated` copies run only while
   * the loop predicate holds on the GPU. Packs may add phases such as `gate` or `body`.
   */
  phase?: 'unconditional' | 'gated' | (string & {});
  /** Declared upper bound on iterations, when known. */
  maximumIterations?: number;
};

/** Binding of a node's invocations to a data key space, so a lane maps to a datum. */
export type GPUComputationDatumBinding = {
  /** Key space id in {@link GPUComputationAxes.keySpaces}, for example `row` or `cell`. */
  keySpace: string;
  /**
   * Relation id in {@link GPUComputationAxes.relations} from the `invocation` key space to
   * `keySpace`. Omitted means identity (`invocation = datum`).
   */
  relation?: string;
};

/** Workgroup counts of one dispatch. */
export type GPUComputationDispatch = {
  /** Workgroup counts in x, y and z. */
  workgroups: readonly [number, number, number];
  /** Linear invocations the kernel guards to, when it is a linear kernel. */
  invocationCount?: number;
  /** Whether the GPU writes the dispatch (indirect dispatch); `workgroups` is then the bound. */
  indirect?: boolean;
};

/** One named binding of a kernel, linking its WGSL variable to a graph resource. */
export type GPUComputationBinding = {
  /** Shader variable name, such as `values`. */
  name: string;
  /** Resource id in {@link GPUComputationGraph.resources}. */
  resource: string;
  /** Shader access mode. */
  access: 'read' | 'read_write' | 'write' | 'sampled' | 'uniform';
  /** Shader element type, such as `f32` or `atomic<u32>`, when known. */
  elementType?: string;
  /** Element offset of the bound view within its resource. */
  elementOffset?: number;
  /** Element count of the bound view. */
  length?: number;
};

/** Upper-bound cost of a node, mirroring `GPUCommandGraphNodeWorkloadEstimate`. */
export type GPUComputationWorkload = {
  operation?: string;
  variant?: string;
  commandCount?: number;
  maximumWorkgroupCount?: number;
  maximumInvocationCount?: number;
  readByteLength?: number;
  writeByteLength?: number;
};

/** One node in a graph form. */
export type GPUComputationNode = {
  /** Graph-unique id. */
  id: string;
  /** Display label; defaults to `dedupName` or `id`. */
  label: string;
  /**
   * Slash-separated hierarchy path such as `scan/level-0`. Group nodes are derived from it;
   * `''` means top level.
   */
  namespace: string;
  /** Node kind. */
  kind: GPUComputationNodeKind;
  /** Domain operator name keyed by the pack's `operators` table, such as `GPUScan`. */
  op?: string;
  /** Data key space the node's invocations map to. */
  datum?: GPUComputationDatumBinding;
  /** Unrolled loop coordinate. */
  loop?: GPUComputationLoopCoordinate;
  /** Name shared by every unrolled copy of the same node, such as `relax` for `relax-iteration-3`. */
  dedupName?: string;
  /**
   * Ids of nodes in a more abstract graph form that this node realizes, written
   * `<graphId>:<nodeId>`. Many-to-many: fusion merges and unrolling splits.
   */
  derivedFrom?: readonly string[];
  /** Formula id in {@link GPUComputationRefs.formulas}. */
  formula?: string;
  /** Source id in {@link GPUComputationRefs.sources}. */
  source?: string;
  /** Workgroup size in x, y and z, for compute nodes. */
  workgroupSize?: readonly [number, number, number];
  /** Dispatch shape, for compute nodes. */
  dispatch?: GPUComputationDispatch;
  /** Named shader bindings, for kernels whose bindings are known. */
  bindings?: readonly GPUComputationBinding[];
  /** Resource ids read by the node. */
  reads: readonly string[];
  /** Resource ids written by the node. */
  writes: readonly string[];
  /** Static upper-bound workload. */
  workload?: GPUComputationWorkload;
  /** Conditional execution: CPU skip or GPU indirect gate. */
  condition?: {id: string; source: 'cpu' | 'gpu'; mode: 'skip' | 'indirect'; resource?: string};
  /** Coherent result boundary declared by this node. */
  publication?: {id: string; completeness: 'partial' | 'complete'};
  /** Free-form attributes. */
  attrs: GPUComputationAttributes;
};

/**
 * Dependency kind. `dataflow` links semantic forms; `explicit` is a declared `dependsOn`;
 * `raw`, `war` and `waw` are inferred resource hazards; `indirect` is a read of a GPU-written
 * dispatch or condition buffer.
 */
export type GPUComputationEdgeKind = 'dataflow' | 'explicit' | 'raw' | 'war' | 'waw' | 'indirect';

/** One dependency between two nodes of the same graph. */
export type GPUComputationEdge = {
  /** Predecessor node id. */
  from: string;
  /** Successor node id. */
  to: string;
  /** Dependency kind. */
  kind: GPUComputationEdgeKind;
  /** Resource id that carries the hazard, for resource edges. */
  resource?: string;
};

/** One logical resource of a compiled graph with its lifetime and physical allocation. */
export type GPUComputationResource = {
  /** Graph-unique logical resource id. */
  id: string;
  /** Resource class. */
  kind: 'buffer' | 'texture' | 'external-texture';
  /** Logical byte size (estimated for textures). */
  bytes: number;
  /** `imported` resources are caller-owned; `transient` ones are graph-owned and may alias. */
  ownership: 'imported' | 'transient';
  /**
   * Physical allocation id shared by aliased transients, such as `transient-buffer-0`.
   * Omitted for imported resources.
   */
  allocation?: string;
  /** Byte size of the physical allocation, which may exceed `bytes` when aliased. */
  allocationBytes?: number;
  /** First scheduled node using the resource. Omitted when unused. */
  firstNode?: string;
  /** Last scheduled node using the resource. Omitted when unused. */
  lastNode?: string;
  /** Whether the resource was retained (never aliased after its last use) for observation. */
  retained?: boolean;
  /** Texture format or buffer element format, when known. */
  format?: string;
  /** Free-form attributes such as texture extent. */
  attrs?: GPUComputationAttributes;
};

// ---------------------------------------------------------------------------------------------
// Axes, key spaces and relations
// ---------------------------------------------------------------------------------------------

/** Axes, key spaces and mapping relations. Relations route through each family's primary key. */
export type GPUComputationAxes = {
  axes: readonly GPUComputationAxis[];
  keySpaces: readonly GPUComputationKeySpace[];
  relations: readonly GPUComputationRelation[];
};

/**
 * One navigable axis, such as entity, lane, workgroup, iteration, permutation draw, buffer slot,
 * node, formula term, parameter or dataset. Hierarchies are expressed with `parent`
 * (lane → workgroup → dispatch; node → contributor → recipe).
 */
export type GPUComputationAxis = {
  /** Record-unique axis id. */
  id: string;
  /** Display label. */
  label: string;
  /** What the axis indexes. */
  kind: 'data' | 'execution' | 'time' | 'structure' | 'parameter';
  /** Coarser axis in the same hierarchy. */
  parent?: string;
  /** Key space whose keys are this axis' values. */
  keySpace?: string;
  /** Number of values, when finite and known. */
  size?: number;
};

/** One key space: the set of keys of one kind of entity (rows, cells, nodes, lanes, terms). */
export type GPUComputationKeySpace = {
  /** Record-unique key space id. */
  id: string;
  /** Display label. */
  label: string;
  /** Number of keys, when known. */
  size?: number;
  /** Index shape of the keys. */
  indexSpace?: '1d' | '2d' | 'csr' | 'batch';
  /** Extent of a `2d` index space as `[width, height]`. */
  shape?: readonly number[];
};

/** Cardinality of a relation from its `from` key space to its `to` key space. */
export type GPUComputationRelationCardinality = '1:1' | '1:N' | 'N:1' | 'N:M';

/**
 * Mapping between two key spaces. `function` relations are pure functions (`invocation = row`,
 * tile arithmetic); `buffer` relations are data (CSR offsets and neighbours, permutation
 * buffers). A time-dependent relation that was not captured is `unknown`, never guessed.
 */
export type GPUComputationRelation = {
  /** Record-unique relation id. */
  id: string;
  /** Source key space id. */
  from: string;
  /** Target key space id. */
  to: string;
  /** Cardinality. */
  cardinality: GPUComputationRelationCardinality;
  /** How the mapping is known. */
  kind: 'function' | 'buffer' | 'unknown';
  /** Function name and parameters for `function`, or resource ids for `buffer` relations. */
  ref?: GPUComputationRelationRef;
  /** Whether the mapping changes during the run (for example a frontier). */
  timeDependent: boolean;
};

/** How a relation is evaluated. */
export type GPUComputationRelationRef =
  | {
      /** Pure function name, such as `identity`, `divide` or `tile-2d`. */
      function: string;
      /** Function parameters, such as `{divisor: 256}` or `{width: 512}`. */
      params?: Readonly<Record<string, number>>;
    }
  | {
      /** `index` maps each `from` key to one `to` key; `csr` maps a key to a range of keys. */
      layout: 'index' | 'csr';
      /** Resource holding target keys (`index`) or neighbour keys (`csr`). */
      resource: string;
      /** Resource holding CSR row offsets, for `csr`. */
      offsetsResource?: string;
      /** Capture id holding the relation's values, when it was read back. */
      capture?: string;
    };

// ---------------------------------------------------------------------------------------------
// Refs
// ---------------------------------------------------------------------------------------------

/** Deduplicated formula and source tables. */
export type GPUComputationRefs = {
  formulas: readonly GPUComputationFormulaRef[];
  sources: readonly GPUComputationSourceRef[];
};

/** One symbol of a formula, optionally bound to a resource, kernel variable or scalar. */
export type GPUComputationFormulaSymbol = {
  /** Symbol id used by `\term{id}` in the TeX. */
  id: string;
  /** Role of the symbol. */
  role: 'input' | 'output' | 'parameter' | 'index' | 'intermediate' | 'constant';
  /** Display label, such as `w_ij`. */
  label?: string;
  /** Bound resource id, WGSL variable or scalar, written `resource:<id>` or `wgsl:<name>`. */
  binding?: string;
};

/**
 * One formula. `derived` formulas come from a real AST and cannot drift; `curated` ones live in
 * sidecars with binding-checked specs and carry a distinct provenance mark.
 */
export type GPUComputationFormulaRef = {
  /** Record-unique formula id. */
  id: string;
  /** TeX source; symbols are tagged with `\term{symbolId}{...}`. */
  tex: string;
  /** Plain-text rendering for alt text, logs and readers without math rendering. */
  ascii: string;
  /** Symbols with their bindings. */
  symbols: readonly GPUComputationFormulaSymbol[];
  /** Provenance of the formula. */
  derivation: 'derived' | 'curated';
};

/** One deduplicated source: a shader module, a SQL statement or an expression. */
export type GPUComputationSourceRef = {
  /** Record-unique id, `<language>:<hash>` for generated sources. */
  id: string;
  /** Source language. */
  language: 'wgsl' | 'glsl' | 'sql' | 'expression' | 'typescript';
  /** Producing module, such as `GPUScan/level-0`; shared by sources of one generator. */
  moduleId: string;
  /** Hash of the full `text`, see `hashGPUComputationText`. */
  hash: string;
  /** Hash of the author-written body only, shared across differently-bound instances. */
  bodyHash?: string;
  /** Entry point, such as `main`. */
  entryPoint?: string;
  /** Full source text, for the kernel source view. Omitted when not retained. */
  text?: string;
  /** Pipeline-overridable constants applied at pipeline creation. */
  constants?: Readonly<Record<string, number>>;
};

// ---------------------------------------------------------------------------------------------
// Decisions, choices and workflow
// ---------------------------------------------------------------------------------------------

/** One evaluated candidate of a decision, kept even when rejected. */
export type GPUDecisionCandidate = {
  /** Candidate implementation id. */
  id: string;
  /** Heuristic or measured score; higher is preferred. Omitted for unsupported candidates. */
  score?: number;
  /** Whether the candidate is supported on this adapter and workload. */
  supported: boolean;
  /** Why the candidate scored as it did, or why it is unsupported. */
  reason: string;
  /** First failing feature or limit, for unsupported candidates. */
  failingFeature?: string;
};

/** "Why this variant": one implementation decision with every candidate considered. */
export type GPUDecisionRecord = {
  /** Record-unique decision id. */
  id: string;
  /** Node, operation or contributor the decision applies to, as `<graphId>:<nodeId>` or a label. */
  subject: string;
  /** Decision family, such as `strategy`, `lowering`, `reduction`, `support` or `projection`. */
  kind: 'strategy' | 'lowering' | 'reduction' | 'support' | 'projection' | (string & {});
  /** Chosen candidate id. */
  chosen: string;
  /** Why the chosen candidate won. */
  reason?: string;
  /** How the choice was made: highest score, caller preference or a fixed rule. */
  method?: 'score' | 'preferred' | 'rule' | 'measured';
  /** Every candidate considered, including rejected and unsupported ones. */
  candidates: readonly GPUDecisionCandidate[];
};

/** One analyst choice: weights type, support, CRS, correction method, permutations, bandwidth. */
export type GPUAnalysisChoice = {
  /** Analysis stage defined by the domain pack, such as `neighbourhood` or `inference`. */
  stage: string;
  /** Parameter name. */
  name: string;
  /** Chosen value. */
  value: GPUComputationAttributeValue;
  /** Alternatives the analyst could have chosen. */
  alternatives?: readonly GPUComputationAttributeValue[];
  /** Whether a stability sweep over this choice is meaningful. */
  sweepable: boolean;
  /** Display label. */
  label?: string;
  /** Unit of `value`, such as `m`. */
  unit?: string;
};

/** Ordered graph or pass invocations chosen by state, such as incremental batch and merge graphs. */
export type GPUWorkflowRecord = {
  /** Workflow id. */
  id: string;
  /** Invocations in execution order. */
  invocations: readonly GPUWorkflowInvocation[];
};

/** One invocation of a graph in a workflow. */
export type GPUWorkflowInvocation = {
  /** Graph id in {@link GPUComputationRecord.graphs}. */
  graph: string;
  /** Display label. */
  label?: string;
  /** State that selected this invocation, such as `first batch` or `merge`. */
  reason?: string;
  /** Repetition index when the same graph runs more than once. */
  iteration?: number;
};

// ---------------------------------------------------------------------------------------------
// Sidecars
// ---------------------------------------------------------------------------------------------

/**
 * Observer effects in force during a measured run. Each moves the run away from production and
 * every timing view states them.
 */
export type GPUComputationEncodingFlags = {
  /**
   * Whether compute passes were coalesced as in production. Timestamps disable coalescing, so a
   * timed run has per-node passes and the production pass structure differs.
   */
  coalesced: boolean;
  /** Whether per-pass GPU timestamps were written. */
  timestamped: boolean;
  /** Number of observation taps inserted into the graph. */
  taps: number;
  /** Whether the run used a lens twin (retained transients, small docs-sized data). */
  lensTwin: boolean;
  /** Transient resource ids retained instead of aliased. */
  retained?: readonly string[];
};

/** Measurements of one node in one run. */
export type GPUComputationNodeMetrics = {
  /** GPU begin timestamp in `timebase` units, relative to the earliest begin in the run. */
  gpuBegin?: number;
  /** GPU end timestamp in `timebase` units, relative to the earliest begin in the run. */
  gpuEnd?: number;
  /** CPU encode time in milliseconds. */
  cpuEncodeMs: number;
  /** Execution outcome for conditional nodes. */
  outcome?: 'executed' | 'skipped' | 'indirect';
  /** Zero-based encoded pass index, which shows coalescing. */
  passIndex?: number;
  /** Named counters, such as frontier size or residual. */
  counters?: Readonly<Record<string, number>>;
};

/** Measurement sidecar of one graph in one run, keyed by `runId`, `graph` and node id. */
export type GPUComputationMetrics = {
  /** Schema identifier, always {@link GPU_COMPUTATION_METRICS_SCHEMA}. */
  schema: typeof GPU_COMPUTATION_METRICS_SCHEMA;
  /** Run this sidecar measures. */
  runId: string;
  /** Measured graph id (normally the compiled form). */
  graph: string;
  /** Unit of `gpuBegin` and `gpuEnd`. */
  timebase: 'ns' | 'ms';
  /** Adapter that produced the measurements. */
  adapter?: GPUComputationAdapter;
  /** Observer effects in force. */
  encoding: GPUComputationEncodingFlags;
  /** Per-node measurements keyed by node id. */
  nodes: Readonly<Record<string, GPUComputationNodeMetrics>>;
  /** Per-iteration or per-step counter series, such as `residual` by iteration. */
  series?: readonly GPUComputationSeries[];
};

/** One counter series over an axis such as iteration or execution step. */
export type GPUComputationSeries = {
  /** Series name, such as `residual` or `frontier-size`. */
  name: string;
  /** Axis id the series is indexed by, such as `iteration`. */
  axis: string;
  /** Values in axis order. */
  values: readonly number[];
  /** Unit, when meaningful. */
  unit?: string;
};

/** Element type of captured values. */
export type GPUComputationCaptureElementType = 'f32' | 'u32' | 'i32';

/**
 * One readback. `gpu-readback` values were copied from the real run; `gpu-replay` values were
 * recomputed on the GPU to recover folded state and are drawn with replay provenance.
 */
export type GPUComputationCapture = {
  /** Sidecar-unique capture id. */
  id: string;
  /** Logical name, such as the snapshot slot `distances`. */
  name: string;
  /** Source resource id in the captured graph, when it is a resource copy. */
  resource?: string;
  /** Graph id the resource belongs to. */
  graph?: string;
  /** Node after which the value was observed. */
  afterNode?: string;
  /** Iteration or execution step at which the value was observed. */
  iteration?: number;
  /** Key space the values are indexed by. */
  keySpace?: string;
  /** Element type. */
  elementType: GPUComputationCaptureElementType;
  /** Values per key, when more than one (for example 2 for a vec2). */
  components?: number;
  /** Captured values; typed arrays in memory, number arrays in JSON. */
  values: ArrayLike<number>;
  /** Provenance; drawn as solid (readback) or hatched (replay) texture. */
  provenance: 'gpu-readback' | 'gpu-replay';
  /** Meaning of a stamp, such as `settledAt round` or `last-improved round`. */
  meaning?: string;
  /**
   * Role of the capture in the element's input → output story, read by the result view. Captures
   * without a role are intermediates. `context` is drawn faint under every result panel and is not an
   * input of the element (study-area boundary, road graph, hillshade).
   */
  role?: 'input' | 'output' | 'intermediate' | 'context';
  /** The single answer a card figure points at on an output capture, such as the nearest point. */
  highlight?: {key: number; label: string};
  /** How to draw the capture in the element's own space, for the result view. */
  geometry?: GPUComputationCaptureGeometry;
};

/**
 * Geometry of a captured buffer in the element's own space. Coordinates are `components: 2`
 * (x, y or lon, lat) per vertex; paths and polygons index their vertices through an offsets capture
 * (one start per path, plus a final end); polygons are drawn as single closed rings.
 */
export type GPUComputationCaptureGeometry =
  | {
      kind: 'points' | 'lines' | 'polygons';
      /** `geographic` = lon/lat degrees; `projected` = planar units; `unknown` is stated, never guessed. */
      coordinateModel: 'geographic' | 'projected' | 'unknown';
      /** Capture id or name holding path start offsets, for lines and polygons. */
      offsetsCapture?: string;
      /** Capture id or name holding one value per feature, coloured on the run's scale. */
      valuesCapture?: string;
      /** Unit of projected coordinates, such as `m`. */
      unit?: string;
    }
  | {
      kind: 'raster';
      /** Width and height in cells; values are row-major. */
      shape: readonly [number, number];
      coordinateModel?: 'geographic' | 'projected' | 'unknown';
      /** Cell size in coordinate units, when known (drawn as spatial support). */
      cellSize?: number;
      unit?: string;
      /**
       * World coordinate of the outer corner of cell (0, 0), in `coordinateModel` units, so vector
       * inputs overlay the raster. Omitted: the raster is drawn at the origin of its own space.
       */
      origin?: readonly [number, number];
      /**
       * Direction in which row indices advance on the ground. `south` (default): row 0 is the
       * northern (top) row, as in images. `north`: row 0 is the southern row, as in y-up grids.
       */
      rowDirection?: 'south' | 'north';
    }
  | {
      kind: 'graph';
      /** Capture id or name of CSR row offsets (node count + 1 values). */
      offsetsCapture: string;
      /** Capture id or name of CSR neighbour node ids. */
      targetsCapture: string;
      /** Capture id or name of node layout coordinates (`components: 2`), when the element has a layout. */
      layoutCapture?: string;
      /** Capture id or name holding one value per node, coloured on the run's scale. */
      valuesCapture?: string;
      coordinateModel?: 'geographic' | 'projected' | 'unknown';
    };

/** Readback sidecar: snapshots, per-datum stamps and GPU replays of one run. */
export type GPUComputationCaptures = {
  /** Schema identifier, always {@link GPU_COMPUTATION_CAPTURES_SCHEMA}. */
  schema: typeof GPU_COMPUTATION_CAPTURES_SCHEMA;
  /** Run these captures belong to. */
  runId: string;
  /** Resource copies taken at publication points, steps or taps. */
  snapshots: readonly GPUComputationCapture[];
  /** Per-datum stamps such as settle round, FDR rank or last-improved round. */
  stamps: readonly GPUComputationCapture[];
  /** GPU replays of folded state, such as one permutation row. */
  replays: readonly GPUComputationCapture[];
};

// ---------------------------------------------------------------------------------------------
// Construction-time annotations
// ---------------------------------------------------------------------------------------------

/**
 * Record metadata attached to a command node while a graph records (`GPUCommandGraph` created
 * with `recording`). Production graphs never carry it. `describe()` copies it onto the compiled
 * graph node, moving `source` into the deduplicated source table.
 */
export type GPUComputationNodeAnnotation = {
  /** Display label. */
  label?: string;
  /** Slash path, joined under the enclosing record scopes. */
  namespace?: string;
  /** Node ids in a more abstract graph form, as `<graphId>:<nodeId>`. */
  derivedFrom?: readonly string[];
  /** Domain operator name. */
  op?: string;
  /** Data key space binding of the invocations. */
  datum?: GPUComputationDatumBinding;
  /** Unrolled loop coordinate. */
  loop?: GPUComputationLoopCoordinate;
  /** Name shared by unrolled copies. */
  dedupName?: string;
  /** Formula id. */
  formula?: string;
  /** Full source, including `text` when retained. */
  source?: GPUComputationSourceRef;
  /** Workgroup size. */
  workgroupSize?: readonly [number, number, number];
  /** Dispatch shape. */
  dispatch?: GPUComputationDispatch;
  /**
   * Named bindings. `resource` holds the graph resource id (buffer or texture handle id) of the
   * bound view.
   */
  bindings?: readonly GPUComputationBinding[];
  /** Free-form attributes. */
  attrs?: GPUComputationAttributes;
};

/** One scope pushed with `GPUCommandGraph.withRecordScope`, stamped on nodes added inside it. */
export type GPUComputationRecordScope = {
  /** Namespace segment appended to the enclosing path, such as `scan` or `level-0`. */
  namespace?: string;
  /**
   * Abstract nodes realized by every node added in the scope, as `<graphId>:<nodeId>`. The
   * innermost scope that declares it wins, so nested lowerings link to their own operation.
   */
  derivedFrom?: readonly string[];
  /** Loop coordinate applied to nodes added in the scope; the innermost declaration wins. */
  loop?: GPUComputationLoopCoordinate;
  /** Operator applied to nodes that declare none. */
  op?: string;
  /** Datum binding applied to nodes that declare none. */
  datum?: GPUComputationDatumBinding;
};
