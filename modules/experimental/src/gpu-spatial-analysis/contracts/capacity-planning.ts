// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Inputs to a pure capacity estimate. */
export type GPUCapacityPlanningOptions = {
  sourceCount: number;
  /** Expected output rows per source row. Defaults to one. */
  expansionFactor?: number;
  minimumCapacity?: number;
  maximumCapacity?: number;
  outputBytesPerRow?: number;
  transientBytesPerRow?: number;
};

/** Deterministic capacity and memory estimate; allocation remains application-owned. */
export type GPUCapacityPlan = {
  capacity: number;
  estimatedRequiredCount: number;
  limited: boolean;
  estimatedOutputBytes: number;
  estimatedPeakTransientBytes: number;
};

/** Application-owned response to an incomplete capacity-bounded result. */
export type GPUCapacityRecoveryPolicy =
  | {kind: 'fixed-budget'}
  | {
      kind: 'grow-on-next-frame';
      growthFactor?: number;
      maximumOutputCapacity?: number;
      maximumCandidateCapacity?: number;
    }
  | {kind: 'fail-closed'};

/** Decoded status for one independently-sized capacity stage. */
export type GPUCapacityStageObservation = {
  capacity: number;
  /** Present for the final output stage, where usable rows are visible to the application. */
  count?: number;
  /** Exact unclamped requirement when the contributor computed it. */
  requiredCount?: number;
  overflow: boolean;
};

/**
 * Decoded completeness values used after an application reads status scalars.
 *
 * Candidate and final output capacity are independent on purpose. A candidate overflow means the
 * final `requiredCount` is not necessarily exact and must grow the candidate stage before the
 * output stage can be sized from a subsequent run.
 */
export type GPUCapacityObservation = {
  output: GPUCapacityStageObservation & {count: number};
  candidate?: GPUCapacityStageObservation;
};

export type GPUIncompleteCapacityStage = 'candidate' | 'output';

/** Pure decision returned by {@link getGPUCapacityRecovery}. */
export type GPUCapacityRecovery =
  | {action: 'accept'; complete: true}
  | {
      action: 'accept';
      complete: false;
      incompleteStages: readonly GPUIncompleteCapacityStage[];
    }
  | {
      action: 'grow';
      complete: false;
      incompleteStages: readonly GPUIncompleteCapacityStage[];
      nextOutputCapacity: number;
      nextCandidateCapacity?: number;
    }
  | {
      action: 'reject';
      complete: false;
      incompleteStages: readonly GPUIncompleteCapacityStage[];
    };

/** Shared summary published by contributor-specific capacity planners. */
export type GPUContributorCapacityPlan = {
  /** Estimated total logical items inspected by the contributor. */
  estimatedWorkItems: number;
  /** Bytes in caller-owned result columns at the planned capacities. */
  estimatedOutputBytes: number;
  /** Peak graph-owned bytes live at once, excluding caller-owned inputs and outputs. */
  estimatedPeakTransientBytes: number;
};

/** Inputs measured or estimated for a bounding-box/refinement spatial join. */
export type GPUSpatialJoinCapacityPlanningOptions = {
  leftCount: number;
  rightCount: number;
  candidateCapacity?: number;
  pairCapacity?: number;
  /** Estimate used before a run has supplied `observedCandidateCount`. Defaults to 8. */
  expectedCandidatesPerLeft?: number;
  /** Estimate used before a run has supplied `observedRequiredCount`. Defaults to 1. */
  expectedMatchesPerLeft?: number;
  /** Exact candidate requirement read from a representative run. */
  observedCandidateCount?: number;
  /** Exact final requirement read from a complete representative run. */
  observedRequiredCount?: number;
};

export type GPUSpatialJoinCapacityPlan = GPUContributorCapacityPlan & {
  candidates: GPUCapacityPlan;
  pairs: GPUCapacityPlan;
};

/** Inputs measured or estimated for a polygon-to-cell cover. */
export type GPUCellCoverCapacityPlanningOptions = {
  featureCount: number;
  vertexCount: number;
  candidateCapacity?: number;
  cellCapacity?: number;
  expectedCandidatesPerFeature?: number;
  expectedCellsPerFeature?: number;
  observedCandidateCount?: number;
  observedRequiredCount?: number;
  edgeSlabs?: boolean;
  edgeSlabEntryCapacity?: number;
  includeCoreColumn?: boolean;
};

export type GPUCellCoverCapacityPlan = GPUContributorCapacityPlan & {
  candidates: GPUCapacityPlan;
  cells: GPUCapacityPlan;
  estimatedEdgeTests: number;
};

/** Inputs measured or estimated for segment noding followed by generated line geometry. */
export type GPULineTopologyCapacityPlanningOptions = {
  segmentCount: number;
  lineCount: number;
  intersectionCapacity?: number;
  pieceCapacity?: number;
  vertexCapacity?: number;
  expectedIntersectionsPerSegment?: number;
  expectedPiecesPerLine?: number;
  expectedVerticesPerPiece?: number;
  observedIntersectionCount?: number;
  observedPieceCount?: number;
  observedVertexCount?: number;
};

export type GPULineTopologyCapacityPlan = GPUContributorCapacityPlan & {
  intersections: GPUCapacityPlan;
  pieces: GPUCapacityPlan;
  vertices: GPUCapacityPlan;
};

/** Inputs measured or estimated for discrete trajectory encounters. */
export type GPUTrajectoryEncounterCapacityPlanningOptions = {
  trackCount: number;
  bucketCount: number;
  latticeCellCount: number;
  hitCapacity?: number;
  pairCapacity?: number;
  expectedHitsPerSample?: number;
  expectedPairsPerTrack?: number;
  observedHitCount?: number;
  observedRequiredCount?: number;
  /** Bytes in requested pair columns. Defaults to every optional encounter column (24). */
  outputBytesPerPair?: number;
};

export type GPUTrajectoryEncounterCapacityPlan = GPUContributorCapacityPlan & {
  hits: GPUCapacityPlan;
  pairs: GPUCapacityPlan;
  sampleCount: number;
};

/** Computes a bounded capacity and byte estimate without allocating resources. */
export function getGPUCapacityPlan(options: GPUCapacityPlanningOptions): GPUCapacityPlan {
  const sourceCount = validateNonNegativeInteger(options.sourceCount, 'sourceCount');
  const expansionFactor = options.expansionFactor ?? 1;
  if (!Number.isFinite(expansionFactor) || expansionFactor < 0) {
    throw new Error('expansionFactor must be a non-negative finite number');
  }
  const minimumCapacity = validateNonNegativeInteger(
    options.minimumCapacity ?? 0,
    'minimumCapacity'
  );
  const maximumCapacity =
    options.maximumCapacity === undefined
      ? Number.MAX_SAFE_INTEGER
      : validateNonNegativeInteger(options.maximumCapacity, 'maximumCapacity');
  if (maximumCapacity < minimumCapacity) {
    throw new Error('maximumCapacity must not be below minimumCapacity');
  }
  const estimatedRequiredCount = Math.min(
    Number.MAX_SAFE_INTEGER,
    Math.ceil(sourceCount * expansionFactor)
  );
  const capacity = Math.min(maximumCapacity, Math.max(minimumCapacity, estimatedRequiredCount));
  const outputBytesPerRow = validateNonNegativeInteger(
    options.outputBytesPerRow ?? 0,
    'outputBytesPerRow'
  );
  const transientBytesPerRow = validateNonNegativeInteger(
    options.transientBytesPerRow ?? 0,
    'transientBytesPerRow'
  );
  return {
    capacity,
    estimatedRequiredCount,
    limited: capacity < estimatedRequiredCount,
    estimatedOutputBytes: saturatingMultiply(capacity, outputBytesPerRow),
    estimatedPeakTransientBytes: saturatingMultiply(capacity, transientBytesPerRow)
  };
}

/** Plans compact `(left, right)` pair rows (two uint32 output columns). */
export function getGPUPairCapacityPlan(options: GPUCapacityPlanningOptions): GPUCapacityPlan {
  return getGPUCapacityPlan({outputBytesPerRow: 8, ...options});
}

/** Plans CSR neighborhood slots (neighbor and weight, plus optional contributor scratch). */
export function getGPUNeighborhoodCapacityPlan(
  options: GPUCapacityPlanningOptions
): GPUCapacityPlan {
  return getGPUCapacityPlan({outputBytesPerRow: 8, transientBytesPerRow: 8, ...options});
}

/** Plans generated float32x2 coordinates plus uint32 provenance. */
export function getGPUGeneratedGeometryCapacityPlan(
  options: GPUCapacityPlanningOptions
): GPUCapacityPlan {
  return getGPUCapacityPlan({outputBytesPerRow: 12, transientBytesPerRow: 8, ...options});
}

/** Plans sparse cell-table rows (two key words, count and value). */
export function getGPUCellCapacityPlan(options: GPUCapacityPlanningOptions): GPUCapacityPlan {
  return getGPUCapacityPlan({outputBytesPerRow: 16, transientBytesPerRow: 8, ...options});
}

/** Plans movement/event rows with identity, time, type and position provenance. */
export function getGPUEventCapacityPlan(options: GPUCapacityPlanningOptions): GPUCapacityPlan {
  return getGPUCapacityPlan({outputBytesPerRow: 20, transientBytesPerRow: 8, ...options});
}

/** Plans the candidate and final pair stages of a spatial join from workload measurements. */
export function getGPUSpatialJoinCapacityPlan(
  options: GPUSpatialJoinCapacityPlanningOptions
): GPUSpatialJoinCapacityPlan {
  const leftCount = validateNonNegativeInteger(options.leftCount, 'leftCount');
  const rightCount = validateNonNegativeInteger(options.rightCount, 'rightCount');
  const candidateRequirement = getMeasuredRequirement(
    leftCount,
    options.expectedCandidatesPerLeft ?? Math.min(rightCount, 8),
    options.observedCandidateCount,
    'observedCandidateCount'
  );
  const pairRequirement = getMeasuredRequirement(
    leftCount,
    options.expectedMatchesPerLeft ?? Math.min(rightCount, 1),
    options.observedRequiredCount,
    'observedRequiredCount'
  );
  const candidates = getExactCapacityPlan(candidateRequirement, options.candidateCapacity, 8, 16);
  const pairs = getExactCapacityPlan(pairRequirement, options.pairCapacity, 8, 8);
  const boundsBytes = saturatingMultiply(saturatingAdd(leftCount, rightCount), 16);
  const bvhBytes = saturatingMultiply(getNextPowerOfTwo(Math.max(rightCount, 1)), 32);
  return {
    candidates,
    pairs,
    estimatedWorkItems: saturatingAdd(saturatingAdd(leftCount, rightCount), candidateRequirement),
    estimatedOutputBytes: pairs.estimatedOutputBytes,
    estimatedPeakTransientBytes: saturatingAdd(
      saturatingAdd(boundsBytes, bvhBytes),
      candidates.estimatedPeakTransientBytes
    )
  };
}

/** Plans candidate tests, accepted cells and optional edge-slab storage for a cell cover. */
export function getGPUCellCoverContributorCapacityPlan(
  options: GPUCellCoverCapacityPlanningOptions
): GPUCellCoverCapacityPlan {
  const featureCount = validateNonNegativeInteger(options.featureCount, 'featureCount');
  const vertexCount = validateNonNegativeInteger(options.vertexCount, 'vertexCount');
  const candidateRequirement = getMeasuredRequirement(
    featureCount,
    options.expectedCandidatesPerFeature ?? 16,
    options.observedCandidateCount,
    'observedCandidateCount'
  );
  const cellRequirement = getMeasuredRequirement(
    featureCount,
    options.expectedCellsPerFeature ?? 8,
    options.observedRequiredCount,
    'observedRequiredCount'
  );
  // Per candidate: flag, accepted scan row and uint64 cell key.
  const candidates = getExactCapacityPlan(candidateRequirement, options.candidateCapacity, 0, 16);
  // Feature ID + uint64 cell key + optional core flag.
  const cells = getExactCapacityPlan(
    cellRequirement,
    options.cellCapacity,
    options.includeCoreColumn ? 16 : 12,
    0
  );
  const edgesPerCandidate =
    featureCount === 0
      ? 0
      : options.edgeSlabs
        ? Math.max(1, Math.ceil(vertexCount / featureCount / 8))
        : Math.max(1, Math.ceil(vertexCount / featureCount));
  const edgeSlabBytes = options.edgeSlabs
    ? saturatingMultiply(
        validateNonNegativeInteger(
          options.edgeSlabEntryCapacity ?? saturatingMultiply(vertexCount, 4),
          'edgeSlabEntryCapacity'
        ),
        8
      )
    : 0;
  const featureScratchBytes = saturatingMultiply(featureCount + 1, 32);
  return {
    candidates,
    cells,
    estimatedEdgeTests: saturatingMultiply(candidateRequirement, edgesPerCandidate),
    estimatedWorkItems: saturatingAdd(
      saturatingAdd(featureCount, candidateRequirement),
      saturatingMultiply(candidateRequirement, edgesPerCandidate)
    ),
    estimatedOutputBytes: cells.estimatedOutputBytes,
    estimatedPeakTransientBytes: saturatingAdd(
      saturatingAdd(candidates.estimatedPeakTransientBytes, featureScratchBytes),
      edgeSlabBytes
    )
  };
}

/** Plans intersection, piece and coordinate stages for line splitting or noding. */
export function getGPULineTopologyCapacityPlan(
  options: GPULineTopologyCapacityPlanningOptions
): GPULineTopologyCapacityPlan {
  const segmentCount = validateNonNegativeInteger(options.segmentCount, 'segmentCount');
  const lineCount = validateNonNegativeInteger(options.lineCount, 'lineCount');
  const intersectionRequirement = getMeasuredRequirement(
    segmentCount,
    options.expectedIntersectionsPerSegment ?? 1,
    options.observedIntersectionCount,
    'observedIntersectionCount'
  );
  const pieceRequirement = getMeasuredRequirement(
    lineCount,
    options.expectedPiecesPerLine ?? 2,
    options.observedPieceCount,
    'observedPieceCount'
  );
  const verticesPerPiece = validateNonNegativeFinite(
    options.expectedVerticesPerPiece ?? 2,
    'expectedVerticesPerPiece'
  );
  const vertexRequirement =
    options.observedVertexCount === undefined
      ? saturatingCeilMultiply(pieceRequirement, verticesPerPiece)
      : validateNonNegativeInteger(options.observedVertexCount, 'observedVertexCount');
  const intersections = getExactCapacityPlan(
    intersectionRequirement,
    options.intersectionCapacity,
    0,
    32
  );
  const pieces = getExactCapacityPlan(pieceRequirement, options.pieceCapacity, 8, 32);
  const vertices = getExactCapacityPlan(vertexRequirement, options.vertexCapacity, 8, 8);
  const segmentTableBytes = saturatingMultiply(segmentCount, 48);
  return {
    intersections,
    pieces,
    vertices,
    estimatedWorkItems: saturatingAdd(
      saturatingAdd(segmentCount, intersectionRequirement),
      saturatingAdd(pieceRequirement, vertexRequirement)
    ),
    estimatedOutputBytes: saturatingAdd(
      saturatingAdd(pieces.estimatedOutputBytes, vertices.estimatedOutputBytes),
      4
    ),
    estimatedPeakTransientBytes: saturatingAdd(
      segmentTableBytes,
      saturatingAdd(
        intersections.estimatedPeakTransientBytes,
        saturatingAdd(pieces.estimatedPeakTransientBytes, vertices.estimatedPeakTransientBytes)
      )
    )
  };
}

/** Plans the hit-sort and distinct-pair stages of trajectory encounter detection. */
export function getGPUTrajectoryEncounterCapacityPlan(
  options: GPUTrajectoryEncounterCapacityPlanningOptions
): GPUTrajectoryEncounterCapacityPlan {
  const trackCount = validateNonNegativeInteger(options.trackCount, 'trackCount');
  const bucketCount = validateNonNegativeInteger(options.bucketCount, 'bucketCount');
  const latticeCellCount = validateNonNegativeInteger(options.latticeCellCount, 'latticeCellCount');
  const sampleCount = saturatingMultiply(trackCount, bucketCount);
  const hitRequirement = getMeasuredRequirement(
    sampleCount,
    options.expectedHitsPerSample ?? 1,
    options.observedHitCount,
    'observedHitCount'
  );
  const pairRequirement = getMeasuredRequirement(
    trackCount,
    options.expectedPairsPerTrack ?? 1,
    options.observedRequiredCount,
    'observedRequiredCount'
  );
  // The encounter pipeline keeps raw hits, sorted fields, orders, flags and compaction rows live.
  const hits = getExactCapacityPlan(hitRequirement, options.hitCapacity, 0, 72);
  const outputBytesPerPair = validateNonNegativeInteger(
    options.outputBytesPerPair ?? 24,
    'outputBytesPerPair'
  );
  const pairs = getExactCapacityPlan(pairRequirement, options.pairCapacity, outputBytesPerPair, 0);
  const sampleAndGridBytes = saturatingAdd(
    saturatingMultiply(sampleCount, 16),
    saturatingMultiply(latticeCellCount + 1, 4)
  );
  return {
    hits,
    pairs,
    sampleCount,
    estimatedWorkItems: saturatingAdd(
      sampleCount,
      saturatingAdd(saturatingMultiply(sampleCount, 9), hitRequirement)
    ),
    estimatedOutputBytes: pairs.estimatedOutputBytes,
    estimatedPeakTransientBytes: saturatingAdd(sampleAndGridBytes, hits.estimatedPeakTransientBytes)
  };
}

/** Applies a fixed-budget, grow-next-frame or fail-closed policy to decoded status. */
export function getGPUCapacityRecovery(
  policy: GPUCapacityRecoveryPolicy,
  observation: GPUCapacityObservation
): GPUCapacityRecovery {
  const output = validateCapacityStageObservation('output', observation.output, true);
  const candidate = observation.candidate
    ? validateCapacityStageObservation('candidate', observation.candidate, false)
    : undefined;
  const incompleteStages: GPUIncompleteCapacityStage[] = [];
  if (candidate?.overflow) {
    incompleteStages.push('candidate');
  }
  if (output.overflow) {
    incompleteStages.push('output');
  }
  if (incompleteStages.length === 0) {
    return {action: 'accept', complete: true};
  }
  if (policy.kind === 'fixed-budget') {
    return {action: 'accept', complete: false, incompleteStages};
  }
  if (policy.kind === 'fail-closed') {
    return {action: 'reject', complete: false, incompleteStages};
  }
  const growthFactor = policy.growthFactor ?? 2;
  if (!Number.isFinite(growthFactor) || growthFactor <= 1) {
    throw new Error('growthFactor must be finite and greater than one');
  }
  const nextOutputCapacity = output.overflow
    ? getGrownCapacity(output, growthFactor, policy.maximumOutputCapacity, 'maximumOutputCapacity')
    : output.capacity;
  const nextCandidateCapacity = candidate?.overflow
    ? getGrownCapacity(
        candidate,
        growthFactor,
        policy.maximumCandidateCapacity,
        'maximumCandidateCapacity'
      )
    : candidate?.capacity;
  if (
    (output.overflow && nextOutputCapacity <= output.capacity) ||
    (candidate?.overflow &&
      nextCandidateCapacity !== undefined &&
      nextCandidateCapacity <= candidate.capacity)
  ) {
    return {action: 'reject', complete: false, incompleteStages};
  }
  return {
    action: 'grow',
    complete: false,
    incompleteStages,
    nextOutputCapacity,
    ...(nextCandidateCapacity === undefined ? {} : {nextCandidateCapacity})
  };
}

function getExactCapacityPlan(
  estimatedRequiredCount: number,
  requestedCapacity: number | undefined,
  outputBytesPerRow: number,
  transientBytesPerRow: number
): GPUCapacityPlan {
  const capacity =
    requestedCapacity === undefined
      ? estimatedRequiredCount
      : validateNonNegativeInteger(requestedCapacity, 'capacity');
  return {
    capacity,
    estimatedRequiredCount,
    limited: capacity < estimatedRequiredCount,
    estimatedOutputBytes: saturatingMultiply(capacity, outputBytesPerRow),
    estimatedPeakTransientBytes: saturatingMultiply(capacity, transientBytesPerRow)
  };
}

function getMeasuredRequirement(
  sourceCount: number,
  expansionFactor: number,
  observedRequiredCount: number | undefined,
  observedName: string
): number {
  if (observedRequiredCount !== undefined) {
    return validateNonNegativeInteger(observedRequiredCount, observedName);
  }
  return saturatingCeilMultiply(
    sourceCount,
    validateNonNegativeFinite(expansionFactor, 'expansionFactor')
  );
}

function validateCapacityStageObservation(
  name: string,
  observation: GPUCapacityStageObservation,
  requiresCount: boolean
): GPUCapacityStageObservation & {count?: number} {
  const capacity = validateNonNegativeInteger(observation.capacity, `${name}.capacity`);
  const count =
    observation.count === undefined
      ? undefined
      : validateNonNegativeInteger(observation.count, `${name}.count`);
  if (requiresCount && count === undefined) {
    throw new Error(`${name}.count is required`);
  }
  if (count !== undefined && count > capacity) {
    throw new Error(`${name}.count must not exceed ${name}.capacity`);
  }
  const requiredCount =
    observation.requiredCount === undefined
      ? undefined
      : validateNonNegativeInteger(observation.requiredCount, `${name}.requiredCount`);
  if (requiredCount !== undefined && count !== undefined && requiredCount < count) {
    throw new Error(`${name}.requiredCount must not be below ${name}.count`);
  }
  if (!observation.overflow && requiredCount !== undefined && requiredCount > capacity) {
    throw new Error(`${name}.requiredCount above capacity requires overflow`);
  }
  return {...observation, capacity, count, requiredCount};
}

function getGrownCapacity(
  observation: GPUCapacityStageObservation,
  growthFactor: number,
  maximum: number | undefined,
  maximumName: string
): number {
  const maximumCapacity =
    maximum === undefined
      ? Number.MAX_SAFE_INTEGER
      : validateNonNegativeInteger(maximum, maximumName);
  const grownCapacity = Math.max(
    observation.requiredCount ?? 0,
    saturatingCeilMultiply(Math.max(observation.capacity, 1), growthFactor)
  );
  return Math.min(maximumCapacity, grownCapacity);
}

function validateNonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative safe integer`);
  }
  return value;
}

function validateNonNegativeFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a non-negative finite number`);
  }
  return value;
}

function saturatingAdd(left: number, right: number): number {
  return Math.min(Number.MAX_SAFE_INTEGER, left + right);
}

function saturatingCeilMultiply(left: number, right: number): number {
  return Math.min(Number.MAX_SAFE_INTEGER, Math.ceil(left * right));
}

function saturatingMultiply(left: number, right: number): number {
  return Math.min(Number.MAX_SAFE_INTEGER, left * right);
}

function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value && result < Number.MAX_SAFE_INTEGER / 2) {
    result *= 2;
  }
  return Math.min(result, Number.MAX_SAFE_INTEGER);
}
