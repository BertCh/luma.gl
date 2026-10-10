// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';

/** Tail convention shared by analytic and permutation inference. */
export type GPUInferenceAlternative =
  | 'two-sided'
  | 'greater'
  | 'less'
  | 'lesser'
  | 'directed'
  | 'folded';

/** Multiple-testing correction applied to a family of p-values. */
export type GPUMultipleTestingMethod = 'none' | 'benjamini-hochberg';

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

/** Common status of direct or iterative linear-system model fits. */
export type GPUSolverStatusPort = {
  /** Zero denotes success; nonzero values are solver-specific documented failure codes. */
  status: GraphDataView<'uint32'>;
  iterationCount?: GraphDataView<'uint32'>;
  converged?: GraphDataView<'uint32'>;
};

/** Reproducibility and stopping evidence for bounded search algorithms. */
export type GPUOptimizationStatusPort = {
  objective: GraphDataView<'float32'>;
  /** Packed words `[iterationCount, converged, iterationLimitReached, invalidCount]`. */
  status: GraphDataView<'uint32'>;
  seed: number;
};
