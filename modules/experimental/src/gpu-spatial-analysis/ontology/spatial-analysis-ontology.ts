// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Canonical kinds of data exchanged by GPU spatial-analysis contributors. */
export const GPU_SPATIAL_ANALYSIS_DATA_KINDS = [
  'parameters',
  'attributes',
  'points',
  'lines',
  'polygons',
  'trajectories',
  'cells',
  'spatial-weights',
  'pairs',
  'regions',
  'surface',
  'network',
  'summary',
  'render-geometry'
] as const;

/** A logical data shape, independent of its concrete `GraphDataView` storage. */
export type GPUSpatialAnalysisDataKind = (typeof GPU_SPATIAL_ANALYSIS_DATA_KINDS)[number];

/** Stable stages used to organize contributors by the role they play in an analysis. */
export const GPU_SPATIAL_ANALYSIS_STAGES = [
  'bridge',
  'construct',
  'transform',
  'index',
  'relate',
  'aggregate',
  'analyze',
  'infer',
  'orchestrate'
] as const;

/** The role of a capability in an analysis pipeline. */
export type GPUSpatialAnalysisStage = (typeof GPU_SPATIAL_ANALYSIS_STAGES)[number];

/** Subject areas used for capability discovery. */
export const GPU_SPATIAL_ANALYSIS_DOMAINS = [
  'execution',
  'geometry',
  'topology',
  'proximity',
  'discrete-global-grids',
  'density-and-interpolation',
  'regional-analysis',
  'spatial-statistics',
  'spatial-models',
  'movement',
  'workflow'
] as const;

/** A subject area within the spatial-analysis ontology. */
export type GPUSpatialAnalysisDomain = (typeof GPU_SPATIAL_ANALYSIS_DOMAINS)[number];

/** Coordinate assumptions made by a capability family. */
export const GPU_SPATIAL_ANALYSIS_COORDINATE_MODELS = [
  'none',
  'planar',
  'spherical',
  'ellipsoidal',
  'grid',
  'topological',
  'mixed'
] as const;

/** Coordinate model used by an operation. */
export type GPUSpatialAnalysisCoordinateModel =
  (typeof GPU_SPATIAL_ANALYSIS_COORDINATE_MODELS)[number];

/** How a result's storage requirement relates to its input. */
export type GPUSpatialAnalysisOutputBound =
  | 'none'
  | 'source-sized'
  | 'fixed-sized'
  | 'capacity-bounded'
  | 'mixed';

/** Whether a capability exists in this package or is a deliberately recorded gap. */
export type GPUSpatialAnalysisCapabilityStatus = 'available' | 'opportunity';

/** Repeatable scale and execution evidence attached to an implemented capability family. */
export type GPUSpatialAnalysisCapabilityEvidence = {
  /** Source rows exercised by the representative correctness/performance fixture. */
  representativeSourceRows: number;
  /** Capacity or fixed output rows exercised by that fixture when output is not source-sized. */
  representativeOutputRows?: number;
  /** Dominant temporary-storage growth, excluding caller-owned inputs and outputs. */
  transientMemory:
    | 'constant'
    | 'linear-source'
    | 'linear-output'
    | 'linear-source-and-output'
    | 'quadratic-source'
    | 'solver-matrix';
  /** Command-graph dispatch structure used by the representative operator. */
  dispatch:
    | 'single-pass'
    | 'source-parallel'
    | 'workgroup-reduction'
    | 'multi-pass'
    | 'bounded-iterative'
    | 'orchestrated';
  /** Repository-relative executable specification or independent oracle. */
  correctnessOracle: string;
};

/** One named concept in the data ontology. */
export type GPUSpatialAnalysisDataConcept = {
  id: GPUSpatialAnalysisDataKind;
  title: string;
  description: string;
};

/**
 * The shared nouns of the API. Contributors may use different physical views, but compatible
 * outputs and inputs should agree on one of these logical concepts.
 */
export const GPU_SPATIAL_ANALYSIS_DATA_ONTOLOGY: readonly GPUSpatialAnalysisDataConcept[] = [
  {
    id: 'parameters',
    title: 'Dynamic parameters',
    description: 'Small GPU-resident values that change without recompiling a command graph.'
  },
  {
    id: 'attributes',
    title: 'Feature attributes',
    description: 'Source-aligned scalar or vector columns, masks, labels and stable identifiers.'
  },
  {
    id: 'points',
    title: 'Point features',
    description: 'Coordinate rows with optional source-aligned attributes and feature identifiers.'
  },
  {
    id: 'lines',
    title: 'Line features',
    description: 'Flat coordinates plus offsets that preserve path and feature boundaries.'
  },
  {
    id: 'polygons',
    title: 'Polygon features',
    description: 'Flat coordinates with ring and polygon offsets, including holes.'
  },
  {
    id: 'trajectories',
    title: 'Trajectories',
    description: 'Ordered positions with track offsets and, when needed, a shared time coordinate.'
  },
  {
    id: 'cells',
    title: 'Discrete global-grid cells',
    description: 'H3 or Quadbin keys and capacity-bounded cell tables across resolution levels.'
  },
  {
    id: 'spatial-weights',
    title: 'Spatial weights',
    description: 'A CSR neighborhood graph with aligned weights and optional distances.'
  },
  {
    id: 'pairs',
    title: 'Feature pairs',
    description: 'Capacity-bounded left/right row relationships produced by joins and encounters.'
  },
  {
    id: 'regions',
    title: 'Regions and partitions',
    description: 'Dense or sparse labels, masks and summaries that assign rows to spatial groups.'
  },
  {
    id: 'surface',
    title: 'Sampled surfaces',
    description: 'Regular grids or raster-like fields with explicit shape and spatial meaning.'
  },
  {
    id: 'network',
    title: 'Spatial networks',
    description:
      'Graph topology embedded in space; implemented by the related GPU Network entry point.'
  },
  {
    id: 'summary',
    title: 'Statistics and models',
    description: 'Per-row, per-group or global estimates, diagnostics, classifications and tests.'
  },
  {
    id: 'render-geometry',
    title: 'Render geometry',
    description: 'GPU-written vertices, indices or indirect counts intended for visualization.'
  }
];

/** A discoverable family of related contributors or a named missing capability. */
export type GPUSpatialAnalysisCapability = {
  /** Stable kebab-case identifier. */
  id: string;
  title: string;
  status: GPUSpatialAnalysisCapabilityStatus;
  domain: GPUSpatialAnalysisDomain;
  stage: GPUSpatialAnalysisStage;
  summary: string;
  inputs: readonly GPUSpatialAnalysisDataKind[];
  outputs: readonly GPUSpatialAnalysisDataKind[];
  coordinateModels: readonly GPUSpatialAnalysisCoordinateModel[];
  outputBound: GPUSpatialAnalysisOutputBound;
  /** Public runtime exports belonging to this family. Empty for an opportunity. */
  operators: readonly string[];
  /** Other capability IDs that commonly feed or consume this family. */
  relatedCapabilities?: readonly string[];
  /** Important boundary that prevents this descriptor from overclaiming support. */
  limitation?: string;
  /** Required executable evidence for available families; omitted for recorded opportunities. */
  evidence?: GPUSpatialAnalysisCapabilityEvidence;
};

/** Filters accepted by {@link queryGPUSpatialAnalysisCapabilities}. */
export type GPUSpatialAnalysisCapabilityQuery = {
  status?: GPUSpatialAnalysisCapabilityStatus;
  domain?: GPUSpatialAnalysisDomain;
  stage?: GPUSpatialAnalysisStage;
  input?: GPUSpatialAnalysisDataKind;
  output?: GPUSpatialAnalysisDataKind;
  coordinateModel?: GPUSpatialAnalysisCoordinateModel;
  /** Case-insensitive text matched against IDs, titles, summaries and public export names. */
  search?: string;
};

/** A capability plus its resolved, explicitly curated connections. */
export type GPUSpatialAnalysisCapabilityConnections = {
  capability: GPUSpatialAnalysisCapability;
  related: readonly GPUSpatialAnalysisCapability[];
};

/** Maturity of the evidence attached to one public operator descriptor. */
export type GPUSpatialAnalysisEvidenceStatus = 'proven' | 'partial' | 'opportunity';

/** Whether an operator's command topology can change between graph encodings. */
export type GPUSpatialAnalysisTopology = 'fixed' | 'dynamic-parameters' | 'unspecified';

/** Numerical contract advertised by an operator. */
export type GPUSpatialAnalysisSemantics =
  | 'exact'
  | 'robust-exact'
  | 'approximate'
  | 'stochastic'
  | 'model-based'
  | 'unspecified';

/** Readback behavior is explicit because contributors must not hide submissions or mapping. */
export type GPUSpatialAnalysisReadback = 'none' | 'explicit-adapter';

/** Status fields that an operator may publish on GPU-resident one-row views. */
export type GPUSpatialAnalysisStatusKind =
  | 'count'
  | 'requiredCount'
  | 'overflow'
  | 'candidateOverflow'
  | 'uncertainCount'
  | 'invalidCount'
  | 'converged'
  | 'iterationLimitReached'
  | 'approximation';

/** Operator-level descriptor used for discovery and generated audits. */
export type GPUSpatialAnalysisOperator = {
  /** Public export name. Constructors are deliberately not stored here. */
  exportName: string;
  /** Capability that owns this operator. */
  primaryCapability: string;
  /** Additional capability IDs that describe secondary roles. */
  tags: readonly string[];
  package: '@luma.gl/experimental/gpu-spatial-analysis';
  inputs: readonly GPUSpatialAnalysisDataKind[];
  outputs: readonly GPUSpatialAnalysisDataKind[];
  dimensionality: '2d' | '2d+time' | 'topology' | 'mixed' | 'unspecified';
  coordinateModels: readonly (GPUSpatialAnalysisCoordinateModel | 'unspecified')[];
  topology: GPUSpatialAnalysisTopology;
  cardinality: GPUSpatialAnalysisOutputBound | 'unspecified';
  semantics: GPUSpatialAnalysisSemantics;
  deterministic: boolean | 'order-dependent' | 'unspecified';
  status: readonly GPUSpatialAnalysisStatusKind[];
  readback: GPUSpatialAnalysisReadback;
  evidence: {
    status: GPUSpatialAnalysisEvidenceStatus;
    /** Repository-relative test root or focused specification. */
    tests: string;
    /** Repository-relative documentation page. */
    documentation: string;
    /** Public contract implemented by the operator. */
    contract: string;
    /** Family representative scale inherited by this operator descriptor. */
    representativeSourceRows: number;
    representativeOutputRows?: number;
    transientMemory: GPUSpatialAnalysisCapabilityEvidence['transientMemory'];
    dispatch: GPUSpatialAnalysisCapabilityEvidence['dispatch'];
    correctnessOracle: string;
  };
};

/** Filters accepted by {@link queryGPUSpatialAnalysisOperators}. */
export type GPUSpatialAnalysisOperatorQuery = {
  primaryCapability?: string;
  tag?: string;
  input?: GPUSpatialAnalysisDataKind;
  output?: GPUSpatialAnalysisDataKind;
  coordinateModel?: GPUSpatialAnalysisCoordinateModel;
  status?: GPUSpatialAnalysisStatusKind;
  semantics?: GPUSpatialAnalysisSemantics;
  evidence?: GPUSpatialAnalysisEvidenceStatus;
  search?: string;
};

/** Generated, machine-readable baseline for the public spatial-analysis surface. */
export type GPUSpatialAnalysisAudit = {
  capabilityFamilyCount: number;
  availableCapabilityFamilyCount: number;
  opportunityCount: number;
  operatorCount: number;
  provenOperatorCount: number;
  partialOperatorCount: number;
  fullySpecifiedCapabilityFamilyCount: number;
  operatorsByCapability: Readonly<Record<string, number>>;
};
