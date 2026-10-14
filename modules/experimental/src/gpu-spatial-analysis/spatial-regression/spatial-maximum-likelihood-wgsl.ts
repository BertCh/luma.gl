// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BISECTION_STEPS,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS
} from './spatial-maximum-likelihood-parameters';
import {GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER} from './spatial-weights-traces-parameters';

function validateSeriesArguments(maximumPower: number, bindingName: string): void {
  if (
    !Number.isInteger(maximumPower) ||
    maximumPower < 2 ||
    maximumPower > GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER
  ) {
    throw new Error(
      `maximumPower must be an integer in [2, ${GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER}]`
    );
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(bindingName)) {
    throw new Error('bindingName must be a WGSL identifier');
  }
}

/**
 * WGSL defining `getLogDeterminantDerivative(rho)`, `d/d rho ln|I - rho W| = -s sum_{q=1}^{Q}
 * t^(q-1) c_q`, and the two trace series of the information matrix `getTraceOne(rho)` =
 * `s sum_{q=1}^{Q} t^(q-1) c_q` and `getTraceTwo(rho)` = `s^2 sum_{m=0}^{Q-2} (m+1) t^m c_(m+2)`,
 * all by Horner's rule over the f32 binding `bindingName` holding `c_q = tr((W/s)^q)` at index
 * `q - 1`, with `t = rho * s` and `s` the WGSL expression `normExpression` (the `infinityNorm`
 * output of `GPUSpatialWeightsTraces`; default `'1.0'` for unscaled traces).
 * The float64 mirrors are `getSpatialLogDeterminantDerivativeFromTraces`,
 * `getSpatialMaximumLikelihoodTraceOne` and `getSpatialMaximumLikelihoodTraceTwo`.
 */
export function getTraceSeriesWGSL(
  maximumPower: number,
  bindingName: string,
  normExpression = '1.0'
): string {
  validateSeriesArguments(maximumPower, bindingName);
  return /* wgsl */ `
fn getTraceOne(rho: f32) -> f32 {
  let norm = ${normExpression};
  let t = rho * norm;
  var accumulator = 0.0;
  for (var step = 0u; step < ${maximumPower}u; step++) {
    let q = ${maximumPower}u - step;
    accumulator = accumulator * t + ${bindingName}[${bindingName}Offset + q - 1u];
  }
  return norm * accumulator;
}

fn getLogDeterminantDerivative(rho: f32) -> f32 {
  return -getTraceOne(rho);
}

fn getTraceTwo(rho: f32) -> f32 {
  let norm = ${normExpression};
  let t = rho * norm;
  var accumulator = 0.0;
  for (var step = 0u; step < ${maximumPower - 1}u; step++) {
    let m = ${maximumPower - 2}u - step;
    accumulator = accumulator * t + f32(m + 1u) * ${bindingName}[${bindingName}Offset + m + 1u];
  }
  return norm * norm * accumulator;
}
`;
}

/**
 * WGSL defining `searchParameter(bound) -> SearchResult`, the GPU form of
 * `searchSpatialMaximumLikelihoodParameter`. The including module must define
 * `getObjective(parameter: f32) -> f32` (concentrated log-likelihood) and
 * `getDerivative(parameter: f32) -> f32`, plus `getNaN()`.
 *
 * The derivative is evaluated on `scanSteps + 1` equally spaced points of `[-bound, bound]`;
 * each positive-to-non-positive sign change is refined by bisection on the derivative, an end
 * point is a candidate when the derivative points outward, and the candidate with the largest
 * objective wins (the first on ties). A NaN derivative produces no candidate.
 */
export function getParameterSearchWGSL(): string {
  return /* wgsl */ `
struct SearchResult {
  parameter: f32,
  localMaximumCount: u32,
  atBoundary: bool,
}

fn searchParameter(bound: f32) -> SearchResult {
  var best = getNaN();
  var bestObjective = -3.0e38;
  var bestAtBoundary = false;
  var count = 0u;
  let step = 2.0 * bound / ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS}.0;
  var previousParameter = -bound;
  var previous = getDerivative(previousParameter);
  if (previous <= 0.0) {
    let objective = getObjective(previousParameter);
    count = count + 1u;
    if (objective > bestObjective) {
      best = previousParameter;
      bestObjective = objective;
      bestAtBoundary = true;
    }
  }
  for (var index = 1u; index <= ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS}u; index++) {
    var parameter = bound;
    if (index < ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SCAN_STEPS}u) {
      parameter = -bound + f32(index) * step;
    }
    let current = getDerivative(parameter);
    if (previous > 0.0 && current <= 0.0) {
      var low = previousParameter;
      var high = parameter;
      for (var iteration = 0u; iteration < ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BISECTION_STEPS}u; iteration++) {
        let middle = 0.5 * (low + high);
        if (getDerivative(middle) > 0.0) {
          low = middle;
        } else {
          high = middle;
        }
      }
      let root = 0.5 * (low + high);
      let objective = getObjective(root);
      count = count + 1u;
      if (objective > bestObjective) {
        best = root;
        bestObjective = objective;
        bestAtBoundary = false;
      }
    }
    previous = current;
    previousParameter = parameter;
  }
  if (previous > 0.0) {
    let objective = getObjective(bound);
    count = count + 1u;
    if (objective > bestObjective) {
      best = bound;
      bestObjective = objective;
      bestAtBoundary = true;
    }
  }
  return SearchResult(best, count, bestAtBoundary);
}
`;
}

/**
 * Kahan-compensated sum of `source[source_offset + base + item * stride]` over
 * `item in [0, count)`, as WGSL statements declaring the variable `name` (and `${name}Compensation`).
 */
export function getKahanSumWGSL(
  source: string,
  base: string,
  stride: string,
  count: string,
  name = 'sum'
): string {
  return `var ${name} = 0.0;
  var ${name}Compensation = 0.0;
  for (var item = 0u; item < ${count}; item++) {
    let term = ${source}[${source}Offset + ${base} + item * ${stride}] - ${name}Compensation;
    let next = ${name} + term;
    ${name}Compensation = (next - ${name}) - term;
    ${name} = next;
  }`;
}

/**
 * WGSL defining `writeTableRow(row, estimate, standardError)`, which writes one stride-4 row
 * (estimate, standard error, z, two-sided normal p-value) to a `tableOut` binding. Requires
 * `erfcValue` (include `ERFC_VALUE_WGSL`).
 */
export function getTableRowWGSL(): string {
  return /* wgsl */ `
fn writeTableRow(row: u32, estimate: f32, standardError: f32) {
  let z = estimate / standardError;
  tableOut[tableOutOffset + row * 4u] = estimate;
  tableOut[tableOutOffset + row * 4u + 1u] = standardError;
  tableOut[tableOutOffset + row * 4u + 2u] = z;
  tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
}
`;
}
