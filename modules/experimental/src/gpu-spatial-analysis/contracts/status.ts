// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';

/** One GPU-written unsigned scalar. */
export type GPUStatusScalar = GraphDataView<'uint32'>;

/** Usable and, when known, exact required cardinality of a bounded result. */
export type GPUCardinalityPort = {
  /** Number of usable rows or elements written. */
  count: GPUStatusScalar;
  /** Exact unclamped requirement. Omit when the operation cannot compute it exactly. */
  requiredCount?: GPUStatusScalar;
};

/** Final-output completeness for a capacity-bounded result. */
export type GPUOverflowStatusPort = {
  /** One when the final result is incomplete because its output capacity was too small. */
  overflow: GPUStatusScalar;
};

/** Completeness of an intermediate candidate stage. */
export type GPUCandidateStatusPort = {
  /** One when refinement may be incomplete because an intermediate capacity was too small. */
  candidateOverflow?: GPUStatusScalar;
};

/** Numerical classifications that could not be certified under the selected precision policy. */
export type GPUUncertaintyStatusPort = {
  /** Exact number of classifications resolved through the documented uncertainty fallback. */
  uncertainCount?: GPUStatusScalar;
};

/** Input validity diagnostics. */
export type GPUInvalidStatusPort = {
  /** Number of input rows rejected by validation. */
  invalidCount?: GPUStatusScalar;
};

/** Status for bounded iterative work. */
export type GPUIterationStatusPort = {
  /** One when the operation met its stopping condition. */
  converged?: GPUStatusScalar;
  /** One when the iteration budget was exhausted before convergence. */
  iterationLimitReached?: GPUStatusScalar;
};

/** Status for algorithms that intentionally return an approximation. */
export type GPUApproximationStatusPort = {
  /** One when the result uses an intentional approximation rather than exact semantics. */
  approximation?: GPUStatusScalar;
};

/** Composable status vocabulary for a capacity-bounded analytical result. */
export type GPUBoundedResultStatusPort = GPUCardinalityPort &
  GPUOverflowStatusPort &
  GPUCandidateStatusPort &
  GPUUncertaintyStatusPort &
  GPUInvalidStatusPort &
  GPUIterationStatusPort &
  GPUApproximationStatusPort;

/** Status attributed to one named recipe stage. */
export type GPURecipeStageStatusPort = {
  stage: string;
  status: Partial<GPUBoundedResultStatusPort>;
};

/** Aggregate status that preserves the stage responsible for each condition. */
export type GPURecipeStatusPort = {
  stages: readonly GPURecipeStageStatusPort[];
};

/** Validates the format and scalar shape of every present status field. */
export function validateGPUStatusPort(
  id: string,
  status: Partial<GPUBoundedResultStatusPort>
): void {
  for (const [name, scalar] of Object.entries(status)) {
    if (!scalar) {
      continue;
    }
    if (scalar.format !== 'uint32' || scalar.length < 1) {
      // Status values are GPU-written uint32 scalars.
      throw new Error(`${id} ${name} must contain one uint32 row`);
    }
  }
}

/** Validates every stage of a recipe status aggregate. */
export function validateGPURecipeStatusPort(id: string, status: GPURecipeStatusPort): void {
  const stages = new Set<string>();
  for (const stage of status.stages) {
    if (!stage.stage || stages.has(stage.stage)) {
      // Stage names identify the operation responsible for an incomplete result.
      throw new Error(`${id} recipe status stages must have unique non-empty names`);
    }
    stages.add(stage.stage);
    validateGPUStatusPort(`${id} stage ${stage.stage}`, stage.status);
  }
}
