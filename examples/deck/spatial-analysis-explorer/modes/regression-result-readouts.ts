// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Readouts for a `GPURegressionModelResult`: every regression view shows the same fields (the
 * coefficient table, n / k / df, sigma squared with its convention, likelihood and information
 * criteria, collinearity and the named specification tests) read through the shared contract
 * instead of per-estimator summary slots.
 */

import type {GPURegressionModelResult} from '@luma.gl/experimental/gpu-spatial-analysis';

type ReadoutControls = {
  addReadout(label: string, value?: string): {setValue: (value: string) => void};
};

/** Display names of the specification tests a result may carry. */
const TEST_LABELS: Record<string, string> = {
  jarqueBera: 'Jarque-Bera (normal residuals)',
  breuschPagan: 'Breusch-Pagan (classic)',
  koenkerBassett: 'Koenker-Bassett (robust)',
  anselinKelejian: 'Anselin-Kelejian (residual lag)'
};

function formatValue(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '–';
  if (Math.abs(value) >= 1e5 || (value !== 0 && Math.abs(value) < 1e-3)) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
}

function formatProbability(value: number): string {
  if (!Number.isFinite(value)) return '–';
  return value < 0.001 ? '<0.001' : value.toFixed(3);
}

/** What `GPURegressionModelStatus` means for a reader of the panel. */
function describeStatus(result: GPURegressionModelResult): string {
  const {status} = result;
  if (!status.ok) return `fit failed (status ${status.code})`;
  const parts = ['ok'];
  if (status.converged === false) parts.push('not converged');
  if (status.illConditioned) parts.push('ill-conditioned (float32 precision warning)');
  return parts.join(', ');
}

/**
 * Adds the readouts of one model result and returns the function that fills them.
 *
 * @param controls Panel controls.
 * @param prefix Short model name put in front of each label, for views that show several models.
 * @param coefficientNames Row labels of the coefficient table, in table order.
 * @param testNames Keys of `result.tests` to show, in order.
 */
export function addModelResultReadouts(
  controls: ReadoutControls,
  prefix: string,
  coefficientNames: readonly string[],
  testNames: readonly string[] = []
): {update: (result: GPURegressionModelResult) => void} {
  const label = (text: string) => (prefix ? `${prefix} ${text}` : text);
  const fitReadout = controls.addReadout(label('fit'));
  const sizeReadout = controls.addReadout(label('n / k / df'));
  const varianceReadout = controls.addReadout(label('σ² (error scale) / ML σ²'));
  const likelihoodReadout = controls.addReadout(label('log-lik / AIC / BIC'));
  const conditionReadout = controls.addReadout(label('condition (Belsley / corr.) / max VIF'));
  const testReadouts = testNames.map(name => controls.addReadout(label(TEST_LABELS[name] ?? name)));
  const coefficientReadouts = coefficientNames.map(name => controls.addReadout(label(name)));

  return {
    update(result) {
      fitReadout.setValue(describeStatus(result));
      sizeReadout.setValue(
        `${result.rowCount} / ${formatValue(result.parameterCount, result.kind === 'geographically-weighted-regression' ? 1 : 0)} / ${formatValue(result.degreesOfFreedom, 0)}`
      );
      const sigmaConvention =
        result.standardErrorSigmaSquared === 'degrees-of-freedom' ? 'n−k' : 'n';
      varianceReadout.setValue(
        `${formatValue(result.sigmaSquared)} / ${formatValue(result.sigmaSquaredMaximumLikelihood)} (SEs use the ${sigmaConvention} form)`
      );
      likelihoodReadout.setValue(
        result.logLikelihood === null
          ? 'not defined for this estimator'
          : `${formatValue(result.logLikelihood, 1)} / ${formatValue(result.akaikeInformationCriterion, 1)} / ${formatValue(result.schwarzCriterion, 1)}`
      );
      const {conditioning} = result;
      conditionReadout.setValue(
        conditioning
          ? `${formatValue(conditioning.conditionNumber, 1)} / ${formatValue(conditioning.correlationConditionNumber, 1)} / ${formatValue(conditioning.maximumVarianceInflationFactor, 1)}${conditioning.conditionNumber > 30 ? ' (collinear)' : ''}`
          : '–'
      );
      testNames.forEach((name, index) => {
        const test = result.tests[name];
        testReadouts[index].setValue(
          test
            ? `${formatValue(test.statistic, 2)}${test.degreesOfFreedom > 0 ? ` (${test.degreesOfFreedom} df)` : ''}, p ${formatProbability(test.pValue)}`
            : '–'
        );
      });
      coefficientReadouts.forEach((readout, index) => {
        const row = result.coefficients[index];
        readout.setValue(
          row
            ? `${formatValue(row.estimate)} ± ${formatValue(row.standardError)} (${row.statisticKind} ${formatValue(row.statistic, 1)}, p ${formatProbability(row.pValue)})`
            : '–'
        );
      });
    }
  };
}
