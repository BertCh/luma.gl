// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCapacityPlan,
  GPUContributorCapacityPlan
} from '../../analysis-core/contracts/index';
import {getNextPowerOfTwo, getSpatialJoinBVHByteLength} from './spatial-join-passes';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';

/** Bytes per candidate slot: the `(left, right)` pair (two words), its flag and its flag offset. */
const CANDIDATE_BYTES = 16;
/** Bytes per candidate slot of the DE-9IM matrix column of the relate engine. */
const RELATE_CANDIDATE_BYTES = 4;
/** Bytes per final pair: the caller-owned `leftIds` and `rightIds` columns. */
const PAIR_BYTES = 8;
/** Bytes per feature of the per-side bounds (`minima` and `maxima`, two `float32x2`). */
const BOUNDS_BYTES = 16;
/** Bytes per polygon feature of its `(ringStart, ringEnd)` table. */
const FEATURE_RING_BYTES = 8;
/** Bytes per robust point/polygon pair row: point, geometry offset and classification (two rows per candidate). */
const ROBUST_PAIR_BYTES = 16;
/** Bytes of the join's state words (candidate, uncertain, match and unmatched totals). */
const STATE_BYTES = 16;
/** Bytes per left feature of the candidate probe's per-feature counts and offsets. */
const LEFT_PROBE_BYTES = 8;
/** Bytes of one live-count indirect dispatch (three `uint32` workgroup counts). */
const DISPATCH_BYTES = 12;
/** Fixed bytes of the robust point/polygon pair rows: the padded point, offset and class rows. */
const ROBUST_FIXED_BYTES = 8 + 8 + 4;
/** Default candidate estimate per left feature before a run supplies `observedCandidateCount`. */
const DEFAULT_CANDIDATES_PER_LEFT = 8;

/** Inputs measured or estimated for a {@link GPUSpatialPredicateJoin}. */
export type GPUSpatialJoinCapacityPlanningOptions = {
  /** Number of left features. */
  leftCount: number;
  /** Number of right features. */
  rightCount: number;
  /** Geometry kind of the left side. Defaults to `'lines'`. */
  leftKind?: GPUSpatialJoinGeometry['kind'];
  /** Geometry kind of the right side. Defaults to `'lines'`. */
  rightKind?: GPUSpatialJoinGeometry['kind'];
  /** Whether candidates are classified by the relate engine (it keeps one matrix per candidate). */
  relate?: boolean;
  /** Right-hand BVH leaf slots. Defaults to the join's default, the next power of two of `rightCount`. */
  leafCapacity?: number;
  /** Candidate capacity to report on; defaults to the estimated requirement. */
  candidateCapacity?: number;
  /** Pair capacity to report on; defaults to the estimated requirement. */
  pairCapacity?: number;
  /** Candidates per left feature before a run supplied `observedCandidateCount`. Defaults to `min(rightCount, 8)`. */
  expectedCandidatesPerLeft?: number;
  /** Matches per left feature before a run supplied `observedRequiredCount`. Defaults to `min(rightCount, 1)`. */
  expectedMatchesPerLeft?: number;
  /** Exact candidate requirement read from a representative run (`output.status.candidateCount`). */
  observedCandidateCount?: number;
  /** Exact pair requirement read from a complete run (`output.pairs.requiredCount`). */
  observedRequiredCount?: number;
};

/** Candidate and pair stages of a spatial join, with its graph-owned memory. */
export type GPUSpatialJoinCapacityPlan = GPUContributorCapacityPlan & {
  candidates: GPUCapacityPlan;
  pairs: GPUCapacityPlan;
};

/**
 * Plans the candidate and final pair stages of a {@link GPUSpatialPredicateJoin} from workload
 * measurements: the capacities to allocate, the caller-owned pair bytes and the join's own
 * transient bytes.
 *
 * The transient bytes are computed from the same sizes the join allocates: per-side bounds, polygon
 * ring ranges, the right-hand BVH (`getSpatialJoinBVHByteLength`), the candidate table and, when
 * used, the relate matrices and the robust point/polygon pair rows. Scratch inside the GPU Core
 * primitives the join composes (the BVH build sort and the scans) is not included, and neither is a
 * `spatialSort` reorder. A capacity grow loop reads `candidateCount` and `pairs.requiredCount`
 * after a frame, passes them as `observed...Count`, and rebuilds with the planned capacities (see
 * {@link getGPUCapacityRecovery} for the stage-by-stage decision).
 */
export function getGPUSpatialJoinCapacityPlan(
  options: GPUSpatialJoinCapacityPlanningOptions
): GPUSpatialJoinCapacityPlan {
  const leftCount = validateCount(options.leftCount, 'leftCount');
  const rightCount = validateCount(options.rightCount, 'rightCount');
  const leftKind = options.leftKind ?? 'lines';
  const rightKind = options.rightKind ?? 'lines';
  const candidateRequirement = getRequirement(
    leftCount,
    options.expectedCandidatesPerLeft ?? Math.min(rightCount, DEFAULT_CANDIDATES_PER_LEFT),
    options.observedCandidateCount,
    'observedCandidateCount'
  );
  const pairRequirement = getRequirement(
    leftCount,
    options.expectedMatchesPerLeft ?? Math.min(rightCount, 1),
    options.observedRequiredCount,
    'observedRequiredCount'
  );
  const usesRobustPointPolygon =
    (leftKind === 'points' && rightKind === 'polygons') ||
    (leftKind === 'polygons' && rightKind === 'points');
  const candidateBytes =
    CANDIDATE_BYTES +
    (options.relate ? RELATE_CANDIDATE_BYTES : 0) +
    (usesRobustPointPolygon ? 2 * ROBUST_PAIR_BYTES : 0);
  const candidates = getStagePlan(
    candidateRequirement,
    options.candidateCapacity,
    0,
    candidateBytes
  );
  const pairs = getStagePlan(pairRequirement, options.pairCapacity, PAIR_BYTES, 0);
  const leafCapacity = options.leafCapacity ?? getNextPowerOfTwo(Math.max(rightCount, 1));
  const ringBytes =
    (leftKind === 'polygons' ? leftCount : 0) * FEATURE_RING_BYTES +
    (rightKind === 'polygons' ? rightCount : 0) * FEATURE_RING_BYTES;
  const robustBytes = usesRobustPointPolygon ? ROBUST_FIXED_BYTES + DISPATCH_BYTES : 0;
  const fixedBytes =
    (leftCount + rightCount) * BOUNDS_BYTES +
    leftCount * LEFT_PROBE_BYTES +
    ringBytes +
    getSpatialJoinBVHByteLength(leafCapacity) +
    STATE_BYTES +
    DISPATCH_BYTES +
    (options.relate ? DISPATCH_BYTES : 0) +
    robustBytes;
  return {
    candidates,
    pairs,
    estimatedWorkItems: saturate(leftCount + rightCount + candidateRequirement),
    estimatedOutputBytes: pairs.estimatedOutputBytes,
    estimatedPeakTransientBytes: saturate(fixedBytes + candidates.estimatedPeakTransientBytes)
  };
}

/** Candidate capacity a {@link GPUSpatialPredicateJoin} allocates when `candidateCapacity` is omitted. @internal */
export function getDefaultSpatialJoinCandidateCapacity(
  leftCount: number,
  rightCount: number
): number {
  return Math.max(1, getGPUSpatialJoinCapacityPlan({leftCount, rightCount}).candidates.capacity);
}

function getStagePlan(
  estimatedRequiredCount: number,
  requestedCapacity: number | undefined,
  outputBytesPerRow: number,
  transientBytesPerRow: number
): GPUCapacityPlan {
  const capacity =
    requestedCapacity === undefined
      ? estimatedRequiredCount
      : validateCount(requestedCapacity, 'capacity');
  return {
    capacity,
    estimatedRequiredCount,
    limited: capacity < estimatedRequiredCount,
    estimatedOutputBytes: saturate(capacity * outputBytesPerRow),
    estimatedPeakTransientBytes: saturate(capacity * transientBytesPerRow)
  };
}

function getRequirement(
  sourceCount: number,
  expansionFactor: number,
  observedCount: number | undefined,
  observedName: string
): number {
  if (observedCount !== undefined) {
    return validateCount(observedCount, observedName);
  }
  if (!Number.isFinite(expansionFactor) || expansionFactor < 0) {
    throw new Error('expansionFactor must be a non-negative finite number');
  }
  return saturate(Math.ceil(sourceCount * expansionFactor));
}

function validateCount(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative safe integer`);
  }
  return value;
}

function saturate(value: number): number {
  return Math.min(Number.MAX_SAFE_INTEGER, value);
}
