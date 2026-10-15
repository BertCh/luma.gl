// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared contracts of the computation-views runtime. Everything here is plain data: no DOM, no
 * device. Record-specific types come from `@luma.gl/gpgpu` (the computation record); this file
 * holds the view-side vocabulary that every layer (coordinator, registry, views, hosts) shares.
 */

// ---------------------------------------------------------------------------------------------
// Rungs, axes and signatures
// ---------------------------------------------------------------------------------------------

/**
 * Curated reading-path rungs a `::lens` directive can name. `result` is the native geometry view
 * (input → output in the element's own space), the element card's lead figure. The Purpose/Story rung is out of scope
 * for this runtime. `datum` binds the data axis at any level; it is a pin, not a level.
 */
export type LensRungId =
  | 'result'
  | 'formula'
  | 'datum'
  | 'iterations'
  | 'kernel'
  | 'graph'
  | 'timing'
  | 'compare'
  | 'scope'
  | 'table';

/** Every rung id, in reading order (concrete to engine). */
export const LENS_RUNG_IDS: readonly LensRungId[] = [
  'result',
  'datum',
  'formula',
  'iterations',
  'kernel',
  'graph',
  'timing',
  'compare',
  'scope',
  'table'
];

/** Marr/Poggio level a view answers (Purpose is out of scope). */
export type ComputationLevel = 'specification' | 'skeleton' | 'realized' | 'execution';

/** Grain: unit of decomposition (not a level). */
export type ComputationGrain = 'datum' | 'kernel' | 'contributor' | 'recipe';

/**
 * Axis identifiers. Domains add axes, not levels, so this is an open string with well-known
 * members.
 */
export type AxisId =
  | 'entity'
  | 'lane'
  | 'workgroup'
  | 'dispatch'
  | 'iteration'
  | 'permutation'
  | 'buffer'
  | 'slot'
  | 'node'
  | 'term'
  | 'parameter'
  | 'dataset'
  | 'run'
  | 'time'
  | 'step'
  | 'element'
  | 'neighbour'
  | 'input'
  | 'choice'
  | (string & {});

/** How a free axis is laid out (Victor's step-up forms plus position). */
export type AxisLayoutRole = 'position' | 'small-multiple' | 'overlay' | 'animate' | 'aggregate';

/** A view's signature: which axes it requires bound, which it leaves free, and how free axes lay out. */
export type ViewSignature = {
  /** Axes the view needs bound before it can draw. */
  binds: readonly AxisId[];
  /** Axes the view enumerates. */
  free: readonly AxisId[];
  /** Layout role of each free axis. */
  layout: Readonly<Partial<Record<AxisId, AxisLayoutRole>>>;
};

/** Value bound to an axis: one key, an explicit key set, or a named value (e.g. a node id). */
export type AxisBindingValue = number | string | readonly number[];

/** Current axis bindings of a lens (`row 17 pinned`, `iteration 3`, `node scan/upsweep`). */
export type AxisBindings = Readonly<Partial<Record<AxisId, AxisBindingValue>>>;

/** Map-use-cube descriptor of a rung (audience, task, interaction). */
export type RungDescriptor = {
  audience: 'public' | 'analyst' | 'expert';
  task: 'present' | 'explain' | 'explore';
  interaction: 'low' | 'medium' | 'high';
};

// ---------------------------------------------------------------------------------------------
// Operators (generic descriptor; the domain packs fill it in)
// ---------------------------------------------------------------------------------------------

/** Map-algebra scope of an operator, generalised across domains. */
export type OperatorScope =
  | 'local'
  | 'focal'
  | 'block'
  | 'zonal'
  | 'global'
  | 'frontier'
  | 'pairwise'
  | 'solver';

/** Reduction an operator performs over its scope. */
export type OperatorReduction =
  | 'none'
  | 'count'
  | 'sum'
  | 'weighted-mean'
  | 'argmin'
  | 'histogram'
  | 'mode'
  | 'select'
  | 'scan-sort'
  | 'solve';

/** Stevens measurement level plus the extensive/intensive split that gates lawful aggregation. */
export type VariableKind = {
  level: 'nominal' | 'ordinal' | 'interval' | 'ratio' | 'cyclic';
  /** Only meaningful for interval/ratio quantities attached to supports. */
  extent?: 'extensive' | 'intensive';
};

/** Generic operator descriptor (`scope × reduction`) that picks the idiom family of a view. */
export type OperatorDescriptor = {
  scope: OperatorScope;
  scopeSecondary?: OperatorScope;
  reduction: OperatorReduction;
  variableKinds?: Readonly<Record<string, VariableKind>>;
};

// ---------------------------------------------------------------------------------------------
// Provenance and provisional state (§E rule 7)
// ---------------------------------------------------------------------------------------------

/**
 * Lineage of a mark. Readback = solid texture; replay and derived = hatched; metadata = outline.
 * `unknown` is drawn as outline plus an explicit *unknown* label, never guessed.
 */
export type MarkProvenance = 'readback' | 'replay' | 'derived' | 'metadata' | 'unknown';

/** Texture that encodes a {@link MarkProvenance}. */
export type ProvenanceTexture = 'solid' | 'hatched' | 'outline';

/** Convergence state of a value, encoded by crispness (blur or coarseness), never saturation. */
export type ProvisionalState = 'final' | 'provisional' | 'unconverged';

// ---------------------------------------------------------------------------------------------
// Coordinator vocabulary (§C)
// ---------------------------------------------------------------------------------------------

/** The four interaction states, never conflated. */
export type InteractionState = 'hover' | 'selection' | 'filter' | 'focus';

/** Who owns the store for the current gesture. */
export type CoordinatorWriter = 'narrative' | 'pointer' | 'keyboard' | 'url';

/** Link heat (Roberts): hot = hover/scrub, warm = re-run, cold = export. */
export type LinkHeat = 'hot' | 'warm' | 'cold';

/** Coupling state of a view against the 100 ms rule. */
export type ViewCoupling = 'linked' | 'stale' | 'decoupled';

/** A key space id (e.g. `rows`, `cells`, `graph-nodes`, `lanes`). */
export type KeySpaceId = string;

/** Predicate of a selection clause. */
export type ClausePredicate =
  | {kind: 'keys'; keys: readonly number[]}
  | {kind: 'range'; field: string; range: readonly [number, number]}
  | {kind: 'all'};

/** A selection clause (Mosaic/Vega-Lite style). A new clause replaces the previous one from the same source. */
export type SelectionClause = {
  /** View or narrative step that produced the clause. */
  source: string;
  keySpace: KeySpaceId;
  predicate: ClausePredicate;
  /** Views the clause applies to; omitted = all. */
  clients?: readonly string[];
  /** Human label for the action log and live region ("row 17", "iterations 3–5"). */
  label: string;
};

/** How clauses of one selection combine. */
export type SelectionResolution = 'single' | 'intersect' | 'union';

// ---------------------------------------------------------------------------------------------
// URL state (§F)
// ---------------------------------------------------------------------------------------------

/** Serializable lens state for permalinks: `{runId, rung, bindings, clauses, params, focus}`. */
export type LensUrlState = {
  /** Mandatory: hash of structure ids, dataset version, seed and adapter/variant. */
  runId: string;
  rung: LensRungId;
  bindings: AxisBindings;
  clauses: readonly SelectionClause[];
  params: Readonly<Record<string, string | number | boolean>>;
  /** Graph or entity focus (node id, expanded group), if any. */
  focus?: string;
};

// ---------------------------------------------------------------------------------------------
// Descriptions (one generator feeds alt text, the live region and the llms twin)
// ---------------------------------------------------------------------------------------------

/** The one number a twin caption must carry inline. */
export type DescriptionKeyNumber = {
  label: string;
  value: number | string;
  unit?: string;
};

/** Structured facts a view reports about its current model; the description generator words them. */
export type DescriptionFacts = {
  /** What the view shows ("Inclusive scan of 4,096 u32 rows"). */
  subject: string;
  keyNumber: DescriptionKeyNumber;
  /** Further facts in reading order, each a short sentence without trailing period. */
  facts: readonly string[];
  /** Lineage of the shown values. */
  provenance: MarkProvenance;
  /** Observer effects and caveats ("timestamps disabled pass coalescing"). */
  caveats?: readonly string[];
  /** Adapter and variant label ("Apple M3 · recorded 2026-10-14"). */
  adapterLabel?: string;
};

/** Text outputs of the description generator. */
export type LensDescription = {
  /** One or two sentences for `alt`/`aria-label`. */
  alt: string;
  /** One sentence for the polite live region (debounced by the host). */
  live: string;
  /** Markdown paragraph for the llms twin. */
  llms: string;
};

// ---------------------------------------------------------------------------------------------
// Rendering options shared by views and twins
// ---------------------------------------------------------------------------------------------

/** Host-supplied TeX renderer (KaTeX with MathML output in the website and showcase). */
export type RenderTeX = (tex: string, options: {displayMode: boolean}) => string;

/** Basemap/page ground a view sits on (§E rule 8). */
export type ViewGround = 'light' | 'dark';
