// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';

/** Score of one evaluated candidate: AICc (NaN when invalid) and its singular-row count. */
export type GoldenSectionScore = {
  score: number;
  singularRowCount: number;
};

/** One candidate evaluation of the search, in the order of the GPU candidate table. */
export type GoldenSectionEvaluation = GoldenSectionScore & {
  value: number;
  /** Candidate table row (0 and 1: the first interior points; row `i + 1` for iteration `i`). */
  row: number;
};

/** Result of {@link simulateGoldenSectionSearch}. */
export type GoldenSectionTrace = {
  /** Every evaluation, including repeats of a value (the GPU re-evaluates them; mgwr caches). */
  evaluations: GoldenSectionEvaluation[];
  /** Candidate table row of the winner of the last comparison (-1 when no iteration ran). */
  optimumRow: number;
  optimumValue: number;
  optimumScore: number;
  /** Comparisons of the two interior points, mgwr's iteration count. */
  iterationCount: number;
  /** True when the tolerance stopped the search; false when `maximumIterations` did. */
  converged: boolean;
  /** Bracket and interior points after the last comparison. */
  state: {lower: number; lowerInterior: number; upper: number; upperInterior: number};
};

export type GoldenSectionOptions = {
  lower: number;
  upper: number;
  /** Adaptive bandwidths: interior points are rounded half to even (numpy `round`). */
  isAdaptive: boolean;
  maximumIterations: number;
  evaluate: (value: number) => GoldenSectionScore;
  /** Tolerance on `|score_b - score_d|` for the pair just compared. */
  getTolerance: (scoreB: number, scoreD: number) => number;
  /** Round every state value to float32 like the kernels. Defaults to float64. */
  precision?: 'float32' | 'float64';
};

/** mgwr's tolerance, an absolute 1e-6. */
export function getMgwrGoldenSectionTolerance(): number {
  return GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE;
}

/** The kernel's tolerance: mgwr's with the float32 relative floor. */
export function getGPUGoldenSectionTolerance(scoreB: number, scoreD: number): number {
  return Math.max(
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE,
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE *
      Math.max(Math.abs(scoreB), Math.abs(scoreD))
  );
}

/** numpy `round`: nearest integer, ties to even. */
export function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference < 0.5) {
    return floor;
  }
  if (difference > 0.5) {
    return floor + 1;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * CPU mirror of the golden-section state machine of `GPUGeographicallyWeightedRegression`
 * (`gwr-golden-section-nodes.ts`), which is mgwr's `golden_section`: iteration 1 evaluates the
 * interior points `b` and `d`; each iteration compares them (`score_b <= score_d`, fewest singular
 * rows first, an invalid score loses to a valid one, two invalid scores stop), shrinks the
 * bracket, stops on the tolerance or the iteration cap, otherwise evaluates the one new interior
 * point. The winner of the last comparison is the optimum.
 */
export function simulateGoldenSectionSearch(options: GoldenSectionOptions): GoldenSectionTrace {
  const round32 = options.precision === 'float32' ? Math.fround : (value: number) => value;
  const delta = round32(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA);
  const normalize = (value: number) =>
    options.isAdaptive ? roundHalfToEven(round32(value)) : round32(value);
  const low = round32(Math.min(options.lower, options.upper));
  const high = round32(Math.max(options.lower, options.upper));
  let lower = low;
  let upper = high;
  let lowerInterior = normalize(lower + round32(delta * round32(upper - lower)));
  let upperInterior = normalize(upper - round32(delta * round32(upper - lower)));
  const evaluations: GoldenSectionEvaluation[] = [];
  const evaluate = (value: number, row: number): GoldenSectionEvaluation => {
    const evaluation = {value, row, ...options.evaluate(value)};
    evaluations.push(evaluation);
    return evaluation;
  };
  let first = evaluate(lowerInterior, 0);
  let second = evaluate(upperInterior, 1);
  let optimumRow = -1;
  let iterationCount = 0;
  let converged = false;
  for (let iteration = 1; iteration <= options.maximumIterations; iteration++) {
    iterationCount = iteration;
    const isFirstValid = Number.isFinite(first.score);
    const isSecondValid = Number.isFinite(second.score);
    let firstWins = isFirstValid;
    if (isFirstValid && isSecondValid) {
      firstWins =
        first.singularRowCount < second.singularRowCount ||
        (first.singularRowCount === second.singularRowCount && first.score <= second.score);
    }
    const newRow = iteration + 1;
    let newValue: number;
    let nextFirst = first;
    let nextSecond = second;
    if (firstWins) {
      optimumRow = first.row;
      upper = upperInterior;
      upperInterior = lowerInterior;
      nextSecond = first;
      lowerInterior = normalize(lower + round32(delta * round32(Math.abs(upper - lower))));
      newValue = lowerInterior;
    } else {
      optimumRow = second.row;
      lower = lowerInterior;
      lowerInterior = upperInterior;
      nextFirst = second;
      upperInterior = normalize(upper - round32(delta * round32(Math.abs(upper - lower))));
      newValue = upperInterior;
    }
    let keepSearching = true;
    if (isFirstValid && isSecondValid && first.singularRowCount === second.singularRowCount) {
      keepSearching =
        Math.abs(first.score - second.score) > options.getTolerance(first.score, second.score);
    } else if (!isFirstValid && !isSecondValid) {
      keepSearching = false;
    }
    first = nextFirst;
    second = nextSecond;
    if (!keepSearching || iteration >= options.maximumIterations) {
      converged = !keepSearching;
      break;
    }
    const evaluation = evaluate(newValue, newRow);
    if (firstWins) {
      first = evaluation;
    } else {
      second = evaluation;
    }
  }
  const optimum = evaluations.find(evaluation => evaluation.row === optimumRow);
  return {
    evaluations,
    optimumRow,
    optimumValue: optimum ? optimum.value : Number.NaN,
    optimumScore: optimum ? optimum.score : Number.NaN,
    iterationCount,
    converged,
    state: {lower, lowerInterior, upper, upperInterior}
  };
}

/** The distinct values of the evaluations, in first-evaluation order (what mgwr's cache calls). */
export function getDistinctEvaluatedValues(trace: GoldenSectionTrace): number[] {
  return Array.from(new Set(trace.evaluations.map(evaluation => evaluation.value)));
}
