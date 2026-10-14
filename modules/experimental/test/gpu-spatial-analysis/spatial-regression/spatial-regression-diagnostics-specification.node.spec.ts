// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSpatialRegressionSpecification,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED
} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {SPREG_ASYMMETRIC_KNN_REFERENCE, SPREG_DIAGNOSTICS_REFERENCE} from './spreg-reference';

/** Flattens `[statistic, p-value]` pairs of the five LM tests into the GPU `tests` layout. */
function toTests(reference: {
  lmLag: readonly number[];
  lmError: readonly number[];
  robustLmLag: readonly number[];
  robustLmError: readonly number[];
  lmSarma: readonly number[];
}): number[] {
  const rows = [
    reference.lmLag,
    reference.lmError,
    reference.robustLmLag,
    reference.robustLmError,
    reference.lmSarma
  ];
  return [...rows.flatMap(([statistic, pValue]) => [statistic, 1, pValue]), 0, 0, 0];
}

/** Test table from the four p-values (statistics are `-log(p)` so larger means more significant). */
function fromPValues(lag: number, error: number, robustLag: number, robustError: number) {
  const row = (p: number) => [-Math.log(p), 1, p];
  return [...row(lag), ...row(error), ...row(robustLag), ...row(robustError), 0, 2, 1, 0, 0, 0];
}

const {
  OLS,
  SPATIAL_LAG: LAG,
  SPATIAL_ERROR: ERROR,
  LAG_AND_ERROR: BOTH,
  UNDETERMINED
} = {
  OLS: GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS,
  SPATIAL_LAG: GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG,
  SPATIAL_ERROR: GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR,
  LAG_AND_ERROR: GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR,
  UNDETERMINED: GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED
};

it('specification search follows Anselin on the spreg-pinned diagnostics', () => {
  // spreg 1.9.1 p-values (spreg-reference.ts): the lattice scene has a significant LM-error only.
  expect(getSpatialRegressionSpecification(toTests(SPREG_DIAGNOSTICS_REFERENCE.lattice), 0)).toBe(
    ERROR
  );
  expect(
    getSpatialRegressionSpecification(toTests(SPREG_DIAGNOSTICS_REFERENCE.nearestNeighbor), 0)
  ).toBe(ERROR);
  // Directed kNN 60/13: LM-lag p = 0.033 significant, LM-error p = 0.081 not.
  expect(
    getSpatialRegressionSpecification(
      toTests(SPREG_ASYMMETRIC_KNN_REFERENCE.scene_60_13.diagnostics),
      0
    )
  ).toBe(LAG);
  // The same tests at alpha 0.01: nothing is significant.
  expect(
    getSpatialRegressionSpecification(
      toTests(SPREG_ASYMMETRIC_KNN_REFERENCE.scene_60_13.diagnostics),
      0,
      0.01
    )
  ).toBe(OLS);
  // At alpha 0.10 both classic tests are significant (p 0.033 and 0.081); robust LM-error
  // (p 0.041) and robust LM-lag (p 0.017) are both significant too.
  expect(
    getSpatialRegressionSpecification(
      toTests(SPREG_ASYMMETRIC_KNN_REFERENCE.scene_60_13.diagnostics),
      0,
      0.1
    )
  ).toBe(BOTH);
});

it('specification search covers the decision tree and treats failures as undetermined', () => {
  const significant = 0.001;
  const quiet = 0.5;
  // Neither classic test significant.
  expect(
    getSpatialRegressionSpecification(fromPValues(quiet, quiet, significant, significant), 0)
  ).toBe(OLS);
  // One classic test significant: robust tests are not consulted.
  expect(
    getSpatialRegressionSpecification(fromPValues(significant, quiet, quiet, significant), 0)
  ).toBe(LAG);
  expect(
    getSpatialRegressionSpecification(fromPValues(quiet, significant, significant, quiet), 0)
  ).toBe(ERROR);
  // Both classic significant: the robust tests decide.
  expect(
    getSpatialRegressionSpecification(fromPValues(significant, significant, significant, quiet), 0)
  ).toBe(LAG);
  expect(
    getSpatialRegressionSpecification(fromPValues(significant, significant, quiet, significant), 0)
  ).toBe(ERROR);
  expect(
    getSpatialRegressionSpecification(
      fromPValues(significant, significant, significant, significant),
      0
    )
  ).toBe(BOTH);
  // Both classic significant, neither robust: the larger classic statistic wins, lag on a tie.
  expect(getSpatialRegressionSpecification(fromPValues(0.0005, 0.002, quiet, quiet), 0)).toBe(LAG);
  expect(getSpatialRegressionSpecification(fromPValues(0.002, 0.0005, quiet, quiet), 0)).toBe(
    ERROR
  );
  expect(getSpatialRegressionSpecification(fromPValues(0.002, 0.002, quiet, quiet), 0)).toBe(LAG);
  // p <= alpha is significant.
  expect(getSpatialRegressionSpecification(fromPValues(0.05, quiet, quiet, quiet), 0, 0.05)).toBe(
    LAG
  );
  // A failed diagnostic is never "OLS adequate".
  expect(getSpatialRegressionSpecification(fromPValues(quiet, quiet, quiet, quiet), 4)).toBe(
    UNDETERMINED
  );
  expect(getSpatialRegressionSpecification(fromPValues(NaN, quiet, quiet, quiet), 0)).toBe(
    UNDETERMINED
  );
  expect(
    getSpatialRegressionSpecification(fromPValues(significant, significant, NaN, quiet), 0)
  ).toBe(UNDETERMINED);
});
