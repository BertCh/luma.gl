// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUIterationStatusPort} from './status';

/**
 * Tail convention shared by analytic and permutation inference. `'two-sided'` is the default.
 * `'lesser'` is the lower tail. `'directed'` picks the tail from the data and is kept
 * only for esda legacy parity; its p-values are anti-conservative.
 */
export type GPUInferenceAlternative = 'two-sided' | 'greater' | 'lesser' | 'folded' | 'directed';

/** Default tail of every test that takes an `alternative`. */
export const GPU_DEFAULT_INFERENCE_ALTERNATIVE = 'two-sided' satisfies GPUInferenceAlternative;

/**
 * Multiple-testing correction applied to a family of p-values. The family is every valid tested
 * row, never a display or viewport mask. `'benjamini-yekutieli'` controls the false discovery rate
 * under arbitrary dependence, which local spatial statistics have.
 */
export type GPUMultipleTestingMethod =
  | 'none'
  | 'bonferroni'
  | 'holm'
  | 'benjamini-hochberg'
  | 'benjamini-yekutieli';

/** CPU-side description of how a result's p-values were produced, enough to build a legend. */
export type GPUInferenceMetadata = {
  alternative: GPUInferenceAlternative;
  /** Number of permutations, or 0 for analytic inference. */
  permutationCount: number;
  /** Seed of the permutation streams; omitted for analytic inference. */
  seed?: number;
  multipleTesting: GPUMultipleTestingMethod;
  /** Significance level used for classifications, when the result classifies rows. */
  significanceLevel?: number;
  /** Which rows form the tested family, for example `'rows with at least one neighbor'`. */
  testedRowsDefinition: string;
};

/** Non-owning result columns shared by local and global spatial statistics. */
export type GPUInferenceResultPort = {
  /** Observed statistic, either one scalar or one value per analyzed row. */
  statistic: GraphDataView<'float32'>;
  pValues?: GraphDataView<'float32'>;
  adjustedPValues?: GraphDataView<'float32'>;
  classifications?: GraphDataView<'uint32'>;
};

/** Reproducibility and interpretation metadata for a permutation distribution. */
export type GPUPermutationMetadata = {
  seed: number;
  permutationCount: number;
  alternative: GPUInferenceAlternative;
  multipleTesting: GPUMultipleTestingMethod;
  includeObserved: true;
};

/** Contract implemented by a statistic that can share a permutation engine. */
export type GPUPermutationStatisticAdapter = {
  scope: 'local' | 'global';
  statistic: string;
  permutedVariable: 'values' | 'secondValues';
  alternative: GPUInferenceAlternative;
};

/** Circular interpretation of line directions. */
export type GPUCircularStatisticsContract = {
  mode: 'directional' | 'axial';
  angleUnit: 'radians';
  origin: 'positive-x-counter-clockwise';
  binCount: number;
};

/**
 * Common status of direct or iterative linear-system model fits: a solver code plus the shared
 * convergence fields of {@link GPUIterationStatusPort}.
 */
export type GPUSolverStatusPort = GPUIterationStatusPort & {
  /** One-row solver code: zero denotes success; nonzero values are documented failure codes. */
  code: GraphDataView<'uint32'>;
};
