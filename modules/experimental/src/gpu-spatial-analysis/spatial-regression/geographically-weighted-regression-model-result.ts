// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPURegressionModelResult,
  type GPURegressionModelResult
} from '../contracts/model-result';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
} from './geographically-weighted-regression-parameters';

/** Read-back arrays of a `GPUGeographicallyWeightedRegression` fit. */
export type GPUGeographicallyWeightedRegressionReadback = {
  /** `output.summary`, at least `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH` values. */
  summary: ArrayLike<number>;
  /** Optional `output.selectedBandwidth`, `[candidate index, ladder value]`. */
  selectedBandwidth?: ArrayLike<number>;
};

/**
 * {@link GPURegressionModelResult} of a GWR fit plus the quantities specific to local models.
 *
 * `coefficients` is empty (the local coefficient surface is a per-row output) and `parameterCount`
 * is the effective number of parameters `tr(S)`.
 */
export type GPUGeographicallyWeightedRegressionModelResult = GPURegressionModelResult & {
  /** Effective number of parameters `tr(S)` (mgwr `ENP`, `tr_S`); same as `parameterCount`. */
  effectiveParameterCount: number;
  /** Included rows whose local system was singular at the selected bandwidth (NaN rows). */
  singularRowCount: number;
  /** Included rows, singular or not; `rowCount` excludes the singular ones. */
  includedRowCount: number;
  /** Da Silva-Fotheringham adjusted alpha `alpha * k / tr(S)` (mgwr `adj_alpha`). */
  adjustedAlpha: number;
  /** Critical `|t|` of the adjusted alpha with `rowCount - 1` degrees of freedom (mgwr `critical_tval`). */
  criticalT: number;
  /** Selected `[candidate index, ladder value]`; null when `selectedBandwidth` was not read back. */
  selectedBandwidth: {index: number; value: number} | null;
};

/**
 * Builds a {@link GPUGeographicallyWeightedRegressionModelResult} from the read-back summary of a
 * `GPUGeographicallyWeightedRegression`, with the conventions of mgwr 2.2.1 `GWRResults`:
 *
 * - `rowCount` is `n`, the non-singular included rows; `parameterCount` is `ENP = tr(S)`;
 * - `sigmaSquared` is `RSS / (n - tr(S))` (mgwr `sigma2` with `sigma2_v1`, which the standard
 *   errors use: `standardErrorSigmaSquared` is `'degrees-of-freedom'`) and
 *   `sigmaSquaredMaximumLikelihood` is `RSS / n`;
 * - `logLikelihood` is mgwr `llf` and `akaikeInformationCriterion` mgwr `aic`
 *   (`-2 llf + 2 (tr(S) + 1)`);
 * - `correctedAkaikeInformationCriterion` is the selected AICc (mgwr `aicc`);
 * - `rSquared` is the global `1 - RSS / TSS` (mgwr `R2`).
 *
 * `status.code` is 0 when a bandwidth candidate was valid and 1 otherwise (all numbers are then
 * NaN). Singular rows do not change the status; read `singularRowCount`.
 *
 * @param readback Read-back `summary` (and optionally `selectedBandwidth`).
 */
export function getGPUGeographicallyWeightedRegressionModelResult(
  readback: GPUGeographicallyWeightedRegressionReadback
): GPUGeographicallyWeightedRegressionModelResult {
  const {summary} = readback;
  if (summary.length < GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH) {
    throw new Error(
      `GWR summary must hold ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH} values`
    );
  }
  const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
  const hasValid = summary[SUMMARY.HAS_VALID_CANDIDATE] > 0.5;
  const traceOfHat = summary[SUMMARY.TRACE_OF_HAT];
  const result = createGPURegressionModelResult({
    kind: 'geographically-weighted-regression',
    coefficients: [],
    rowCount: summary[SUMMARY.OBSERVATION_COUNT],
    residualSumOfSquares: summary[SUMMARY.RESIDUAL_SUM_OF_SQUARES],
    standardErrorSigmaSquared: 'degrees-of-freedom',
    statusCode: hasValid ? 0 : 1,
    rSquared: summary[SUMMARY.R_SQUARED],
    logLikelihood: hasValid ? summary[SUMMARY.LOG_LIKELIHOOD] : null,
    parameterCount: traceOfHat,
    // mgwr counts the error variance: aic = -2 llf + 2 (tr(S) + 1).
    informationCriterionParameterCount: traceOfHat + 1,
    correctedAkaikeInformationCriterion: hasValid ? summary[SUMMARY.AICC] : null
  });
  const selected = readback.selectedBandwidth;
  return {
    ...result,
    effectiveParameterCount: traceOfHat,
    singularRowCount: summary[SUMMARY.SINGULAR_ROW_COUNT],
    includedRowCount: summary[SUMMARY.INCLUDED_ROW_COUNT],
    adjustedAlpha: summary[SUMMARY.ADJUSTED_ALPHA],
    criticalT: summary[SUMMARY.CRITICAL_T],
    selectedBandwidth: selected ? {index: selected[0], value: selected[1]} : null
  };
}
