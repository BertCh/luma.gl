// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_SIGNIFICANCE_LEVEL,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getStudentTCriticalValue} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {
  getGPUGoldenSectionTolerance,
  getMgwrGoldenSectionTolerance,
  simulateGoldenSectionSearch
} from './gwr-golden-section-reference';

/** Inputs of {@link computeGeographicallyWeightedRegressionOnCPU}. */
export type GeographicallyWeightedRegressionOracleInput = {
  /** Interleaved `x, y` positions (`longitude, latitude` degrees with `sphereRadius`). */
  positions: Float32Array;
  /** Row-major `row * predictorCount + column`. */
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
  mask?: Uint32Array;
  settings: GPUGeographicallyWeightedRegressionSettings;
  /** Largest adaptive `k`. Defaults to 128. */
  maximumNeighborCount?: number;
  /** Ladder capacity (the refinement ladders have this many entries). Defaults to 32. */
  maximumBandwidthCount?: number;
  /** Refinement passes after the caller ladder. Defaults to 0, the GPU default is 2. */
  refinementPassCount?: number;
  /** Great-circle distances on a sphere of this radius (positions in degrees); planar when omitted. */
  sphereRadius?: number;
  /** Bandwidth search; defaults to the ladder. Golden section follows `bandwidthSearch` of the GPU. */
  bandwidthSearch?: 'ladder' | 'golden-section';
  /** Golden-section iterations, the GPU default is 30. */
  maximumSearchIterations?: number;
  /** Golden-section convergence tolerance: the kernel's float32 floor (default) or mgwr's 1e-6. */
  searchTolerance?: 'gpu' | 'mgwr';
  /** Gaussian weights of rows farther than this many bandwidths are 0; exact Gaussian when omitted. */
  gaussianCutoff?: number;
};

/** One evaluated bandwidth candidate. */
export type GeographicallyWeightedRegressionOracleCandidate = {
  value: number;
  /** NaN when the candidate is invalid. */
  aicc: number;
  singularRowCount: number;
  residualSumOfSquares: number;
  traceOfHat: number;
  observationCount: number;
  includedRowCount: number;
};

/** Float64 results mirroring the GPU outputs. */
export type GeographicallyWeightedRegressionOracleResult = {
  coefficients: number[];
  standardErrors: number[];
  tValues: number[];
  significant: number[];
  localR2: number[];
  fitted: number[];
  residuals: number[];
  hatDiagonal: number[];
  localStatus: number[];
  /** One AICc per caller ladder value, NaN for invalid candidates. */
  bandwidthScores: number[];
  /** Every candidate of every pass in table order (caller ladder first). */
  candidates: GeographicallyWeightedRegressionOracleCandidate[];
  /**
   * Selected index in table order (pass * ladderCapacity + candidate); with golden section the
   * candidate table row (evaluation order).
   */
  selectedIndex: number;
  /** Golden section only: mgwr iterations (comparisons) and whether the tolerance stopped it. */
  searchIterationCount?: number;
  searchConverged?: boolean;
  /** Golden section only: the bounds the search started from. */
  searchBounds?: [number, number];
  selectedValue: number;
  residualSumOfSquares: number;
  traceOfHat: number;
  aicc: number;
  rSquared: number;
  /** Rows entering the statistics (included minus singular). */
  observationCount: number;
  includedRowCount: number;
  singularRowCount: number;
  sigmaSquared: number;
  adjustedAlpha: number;
  criticalT: number;
  logLikelihood: number;
  hasValidCandidate: boolean;
};

type LocalFit = {
  ok: boolean;
  /** Centred coefficients: `beta[0]` is the fitted value, the rest are slopes. */
  beta: number[];
  hat: number;
  weightSum: number;
  weightedResponse: number;
  /** diag(C C') of the original coefficients. */
  variance: number[];
};

/**
 * Gaussian or bisquare weight of distance `d` at bandwidth `h`. A Gaussian `cutoff` zeroes the
 * weight of rows farther than `cutoff * h`.
 */
export function getGeographicallyWeightedRegressionWeight(
  kernel: 'gaussian' | 'bisquare',
  distance: number,
  bandwidth: number,
  cutoff?: number
): number {
  const ratio = distance / bandwidth;
  if (kernel === 'bisquare') {
    if (distance >= bandwidth) {
      return 0;
    }
    return (1 - ratio * ratio) ** 2;
  }
  if (cutoff !== undefined && distance > cutoff * bandwidth) {
    return 0;
  }
  return Math.exp(-0.5 * ratio * ratio);
}

/** Great-circle distance of two (longitude, latitude) points in degrees on a sphere. */
export function getHaversineDistance(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number,
  radius: number
): number {
  const toRadians = Math.PI / 180;
  const deltaLatitude = (latitudeB - latitudeA) * toRadians;
  const deltaLongitude = (longitudeB - longitudeA) * toRadians;
  const a =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(latitudeA * toRadians) *
      Math.cos(latitudeB * toRadians) *
      Math.sin(deltaLongitude / 2) ** 2;
  return 2 * radius * Math.asin(Math.min(1, Math.sqrt(a)));
}

/**
 * Solves the symmetric positive definite system by Cholesky in float64; undefined when singular.
 * Returns the solution, the `(0, 0)` entry of the inverse and the full inverse.
 */
function solveSymmetric(
  matrix: number[][],
  rightHandSide: number[]
): {solution: number[]; inverse00: number; inverse: number[][]} | undefined {
  const p = rightHandSide.length;
  const lower = matrix.map(row => row.slice());
  for (let column = 0; column < p; column++) {
    let diagonal = lower[column][column];
    for (let k = 0; k < column; k++) {
      diagonal -= lower[column][k] ** 2;
    }
    if (!(diagonal > 1e-12 * matrix[column][column])) {
      return undefined;
    }
    const pivot = Math.sqrt(diagonal);
    lower[column][column] = pivot;
    for (let row = column + 1; row < p; row++) {
      let value = lower[row][column];
      for (let k = 0; k < column; k++) {
        value -= lower[row][k] * lower[column][k];
      }
      lower[row][column] = value / pivot;
    }
  }
  const solve = (input: number[]): number[] => {
    const x = input.slice();
    for (let row = 0; row < p; row++) {
      for (let k = 0; k < row; k++) {
        x[row] -= lower[row][k] * x[k];
      }
      x[row] /= lower[row][row];
    }
    for (let row = p - 1; row >= 0; row--) {
      for (let k = row + 1; k < p; k++) {
        x[row] -= lower[k][row] * x[k];
      }
      x[row] /= lower[row][row];
    }
    return x;
  };
  const inverse: number[][] = [];
  for (let column = 0; column < p; column++) {
    const unit = new Array<number>(p).fill(0);
    unit[column] = 1;
    inverse.push(solve(unit));
  }
  return {solution: solve(rightHandSide), inverse00: inverse[0][0], inverse};
}

/**
 * Float64 reference for `GPUGeographicallyWeightedRegression`: the same definitions (focal
 * centring, kernels, adaptive k-th distance including the row itself times 1.00001, hat diagonal,
 * AICc with `n ln(RSS/n)` over the non-singular rows, fewest-singular-rows-then-minimum-AICc
 * selection with ties to the lowest index, ladder refinement, local standard errors from
 * `(X'WX)^-1 X'W^2 X (X'WX)^-1`, mgwr `adj_alpha` and `critical_tval`) evaluated with plain loops.
 */
export function computeGeographicallyWeightedRegressionOnCPU(
  input: GeographicallyWeightedRegressionOracleInput
): GeographicallyWeightedRegressionOracleResult {
  const {positions, predictors, predictorCount, response, mask, settings} = input;
  const p = predictorCount + 1;
  const rowCount = response.length;
  const kernel = settings.kernel ?? 'bisquare';
  const isAdaptive = settings.bandwidthMode === 'adaptive';
  const maximumK =
    input.maximumNeighborCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT;
  const ladderCapacity =
    input.maximumBandwidthCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH;
  const refinementPassCount = ladderCapacity < 2 ? 0 : (input.refinementPassCount ?? 0);
  const ladder = Array.from(settings.bandwidths);
  const level =
    settings.significanceLevel ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_SIGNIFICANCE_LEVEL;

  const valid: boolean[] = [];
  for (let row = 0; row < rowCount; row++) {
    let isValid = mask ? mask[row] !== 0 : true;
    isValid &&= Number.isFinite(positions[2 * row]) && Number.isFinite(positions[2 * row + 1]);
    if (input.sphereRadius !== undefined) {
      isValid &&= Math.abs(positions[2 * row + 1]) <= 90;
    }
    isValid &&= Number.isFinite(response[row]);
    for (let column = 0; column < predictorCount; column++) {
      isValid &&= Number.isFinite(predictors[row * predictorCount + column]);
    }
    valid.push(isValid);
  }
  const distance = (a: number, b: number) =>
    input.sphereRadius !== undefined
      ? getHaversineDistance(
          positions[2 * a],
          positions[2 * a + 1],
          positions[2 * b],
          positions[2 * b + 1],
          input.sphereRadius
        )
      : Math.hypot(
          positions[2 * a] - positions[2 * b],
          positions[2 * a + 1] - positions[2 * b + 1]
        );

  const getBandwidth = (row: number, value: number): number => {
    if (!isAdaptive) {
      return value > 0 ? value : -1;
    }
    if (!(value >= 2) || value > maximumK) {
      return -1;
    }
    const k = Math.floor(value + 0.5);
    const distances: number[] = [];
    for (let other = 0; other < rowCount; other++) {
      if (valid[other]) {
        distances.push(distance(row, other));
      }
    }
    distances.sort((left, right) => left - right);
    if (k > distances.length) {
      return -1;
    }
    return distances[k - 1] * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR;
  };

  const fitLocation = (row: number, bandwidth: number): LocalFit => {
    const a = Array.from({length: p}, () => new Array<number>(p).fill(0));
    const squared = Array.from({length: p}, () => new Array<number>(p).fill(0));
    const b = new Array<number>(p).fill(0);
    const z = new Array<number>(p).fill(0);
    z[0] = 1;
    for (let other = 0; other < rowCount; other++) {
      if (!valid[other]) {
        continue;
      }
      const weight = getGeographicallyWeightedRegressionWeight(
        kernel,
        distance(row, other),
        bandwidth,
        input.gaussianCutoff
      );
      if (!(weight > 0)) {
        continue;
      }
      for (let column = 0; column < predictorCount; column++) {
        z[column + 1] =
          predictors[other * predictorCount + column] - predictors[row * predictorCount + column];
      }
      for (let r = 0; r < p; r++) {
        b[r] += weight * z[r] * response[other];
        for (let c = 0; c <= r; c++) {
          a[r][c] += weight * z[r] * z[c];
          squared[r][c] += weight * weight * z[r] * z[c];
        }
      }
    }
    for (let r = 0; r < p; r++) {
      for (let c = r + 1; c < p; c++) {
        a[r][c] = a[c][r];
        squared[r][c] = squared[c][r];
      }
    }
    const fit: LocalFit = {
      ok: false,
      beta: new Array<number>(p).fill(0),
      hat: 0,
      weightSum: a[0][0],
      weightedResponse: b[0],
      variance: new Array<number>(p).fill(Number.NaN)
    };
    if (!a.every((line, r) => line[r] > 0)) {
      return fit;
    }
    const solved = solveSymmetric(a, b);
    if (!solved) {
      return fit;
    }
    fit.ok = true;
    fit.beta = solved.solution;
    fit.hat = solved.inverse00;
    // Covariance of the centred coefficients is A^-1 B A^-1; the original intercept is
    // gamma_0 - sum x_ic gamma_c, so its direction is t = (1, -x_i).
    const inverse = solved.inverse;
    const covariance = Array.from({length: p}, () => new Array<number>(p).fill(0));
    for (let r = 0; r < p; r++) {
      for (let c = 0; c < p; c++) {
        let sum = 0;
        for (let i = 0; i < p; i++) {
          for (let j = 0; j < p; j++) {
            sum += inverse[r][i] * squared[i][j] * inverse[j][c];
          }
        }
        covariance[r][c] = sum;
      }
    }
    const direction = new Array<number>(p).fill(0);
    direction[0] = 1;
    for (let column = 0; column < predictorCount; column++) {
      direction[column + 1] = -predictors[row * predictorCount + column];
    }
    let interceptVariance = 0;
    for (let r = 0; r < p; r++) {
      for (let c = 0; c < p; c++) {
        interceptVariance += direction[r] * covariance[r][c] * direction[c];
      }
    }
    fit.variance[0] = interceptVariance;
    for (let column = 1; column < p; column++) {
      fit.variance[column] = covariance[column][column];
    }
    return fit;
  };

  const nan = Number.NaN;
  const includedCount = valid.filter(Boolean).length;

  const evaluate = (value: number): GeographicallyWeightedRegressionOracleCandidate => {
    let residualSum = 0;
    let traceSum = 0;
    let singular = 0;
    let invalid = 0;
    for (let row = 0; row < rowCount; row++) {
      if (!valid[row]) {
        continue;
      }
      const bandwidth = getBandwidth(row, value);
      if (!(bandwidth > 0)) {
        invalid++;
        continue;
      }
      const fit = fitLocation(row, bandwidth);
      if (!fit.ok) {
        singular++;
        continue;
      }
      residualSum += (response[row] - fit.beta[0]) ** 2;
      traceSum += fit.hat;
    }
    const count = includedCount - singular;
    const denominator = count - 2 - traceSum;
    let aicc = nan;
    if (invalid === 0 && count > 0 && denominator > 0) {
      const variance = Math.max(
        residualSum / count,
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE
      );
      const score =
        count * Math.log(variance) +
        count * Math.log(2 * Math.PI) +
        (count * (count + traceSum)) / denominator;
      aicc = Number.isFinite(score) ? score : nan;
    }
    return {
      value,
      aicc,
      singularRowCount: singular,
      residualSumOfSquares: residualSum,
      traceOfHat: traceSum,
      observationCount: count,
      includedRowCount: includedCount
    };
  };

  const unused = (): GeographicallyWeightedRegressionOracleCandidate => ({
    value: nan,
    aicc: nan,
    singularRowCount: nan,
    residualSumOfSquares: nan,
    traceOfHat: nan,
    observationCount: nan,
    includedRowCount: nan
  });
  // Best candidate: fewest singular rows, then smallest AICc, ties to the first.
  const getBestIndex = (candidates: GeographicallyWeightedRegressionOracleCandidate[]): number => {
    let best = -1;
    candidates.forEach((candidate, index) => {
      if (!Number.isFinite(candidate.aicc)) {
        return;
      }
      if (
        best < 0 ||
        candidate.singularRowCount < candidates[best].singularRowCount ||
        (candidate.singularRowCount === candidates[best].singularRowCount &&
          candidate.aicc < candidates[best].aicc)
      ) {
        best = index;
      }
    });
    return best;
  };

  /** Mirror of the `refineLadder` kernel. */
  const getRefinedLadder = (
    candidates: GeographicallyWeightedRegressionOracleCandidate[]
  ): number[] => {
    const bestIndex = getBestIndex(candidates);
    if (bestIndex < 0) {
      return [];
    }
    const normalize = (value: number) => (isAdaptive ? Math.floor(value + 0.5) : value);
    const bestValue = normalize(candidates[bestIndex].value);
    let lower = -Infinity;
    let upper = Infinity;
    for (const candidate of candidates) {
      if (!Number.isFinite(candidate.value)) {
        continue;
      }
      const value = normalize(candidate.value);
      if (value < bestValue) {
        lower = Math.max(lower, value);
      } else if (value > bestValue) {
        upper = Math.min(upper, value);
      }
    }
    if (lower === -Infinity) lower = bestValue;
    if (upper === Infinity) upper = bestValue;
    const values: number[] = [];
    if (isAdaptive) {
      const first = Math.max(lower + 1, 2);
      const last = Math.min(upper - 1, maximumK);
      const isInside = first <= bestValue && bestValue <= last;
      const integers = Math.max(last - first + 1 - (isInside ? 1 : 0), 0);
      const count = Math.min(Math.floor(integers), ladderCapacity);
      for (let slot = 0; slot < count; slot++) {
        const position =
          integers > ladderCapacity ? Math.floor(((slot + 0.5) * integers) / ladderCapacity) : slot;
        let value = first + position;
        if (isInside && value >= bestValue) {
          value += 1;
        }
        values.push(value);
      }
    } else {
      const low = Math.max(lower, 0);
      const span = upper - low;
      if (span > 1e-5 * Math.max(Math.abs(upper), 1e-30)) {
        for (let slot = 0; slot < ladderCapacity; slot++) {
          values.push(low + (span * (slot + 1)) / (ladderCapacity + 1));
        }
      }
    }
    return values;
  };

  // Candidate table: caller ladder, then one block of `ladderCapacity` slots per refinement pass.
  const candidates: GeographicallyWeightedRegressionOracleCandidate[] = [];
  const padBlock = (block: GeographicallyWeightedRegressionOracleCandidate[]) => {
    while (block.length < ladderCapacity) {
      block.push(unused());
    }
    candidates.push(...block);
  };
  const isGoldenSection = input.bandwidthSearch === 'golden-section';
  let searchIterationCount: number | undefined;
  let searchConverged: boolean | undefined;
  let searchBounds: [number, number] | undefined;
  let optimumRow = -1;
  let bandwidthScores: number[];
  if (isGoldenSection && ladder.length >= 2) {
    // Bounds: the first two ladder values when positive, else mgwr's defaults (adaptive
    // [40 + 2 p, n capped at the neighbour list], fixed [min_dist / 2, 2 max_dist]).
    let lower = 0;
    let upper = 0;
    if (isAdaptive) {
      lower = 40 + 2 * p;
      upper = Math.min(includedCount, maximumK);
    } else {
      let nearest = Number.POSITIVE_INFINITY;
      let farthest = 0;
      for (let row = 0; row < rowCount; row++) {
        for (let other = 0; other < rowCount && valid[row]; other++) {
          if (!valid[other]) {
            continue;
          }
          const separation = distance(row, other);
          if (separation > 0) {
            nearest = Math.min(nearest, separation);
          }
          farthest = Math.max(farthest, separation);
        }
      }
      if (includedCount > 0 && Number.isFinite(nearest)) {
        lower = nearest / 2;
        upper = 2 * farthest;
      }
    }
    if (ladder[0] > 0) lower = ladder[0];
    if (ladder[1] > 0) upper = ladder[1];
    searchBounds = [Math.min(lower, upper), Math.max(lower, upper)];
    const evaluated: GeographicallyWeightedRegressionOracleCandidate[] = [];
    const trace = simulateGoldenSectionSearch({
      lower,
      upper,
      isAdaptive,
      maximumIterations: input.maximumSearchIterations ?? 30,
      getTolerance:
        input.searchTolerance === 'mgwr'
          ? getMgwrGoldenSectionTolerance
          : getGPUGoldenSectionTolerance,
      evaluate: value => {
        const candidate = evaluate(value);
        evaluated.push(candidate);
        return {score: candidate.aicc, singularRowCount: candidate.singularRowCount};
      }
    });
    const rowCountOfTable = (input.maximumSearchIterations ?? 30) + 1;
    for (let row = 0; row < rowCountOfTable; row++) {
      candidates.push(unused());
    }
    trace.evaluations.forEach((evaluation, index) => {
      candidates[evaluation.row] = evaluated[index];
    });
    optimumRow = trace.optimumRow;
    searchIterationCount = trace.iterationCount;
    searchConverged = trace.converged;
    bandwidthScores = Array.from({length: ladderCapacity}, (_, row) =>
      row < candidates.length ? candidates[row].aicc : nan
    );
  } else {
    padBlock(ladder.map(evaluate));
    for (let pass = 1; pass <= refinementPassCount && !isGoldenSection; pass++) {
      padBlock(getRefinedLadder(candidates).map(evaluate));
    }
    bandwidthScores = candidates.slice(0, ladder.length).map(candidate => candidate.aicc);
  }
  let bestIndex = getBestIndex(candidates);
  if (optimumRow >= 0 && Number.isFinite(candidates[optimumRow].aicc)) {
    // Golden section returns the winner of its last comparison, as mgwr does.
    bestIndex = optimumRow;
  }
  const hasValidCandidate = bestIndex >= 0;
  const selectedIndex = hasValidCandidate ? bestIndex : 0;
  const selectedValue = candidates[selectedIndex].value;
  const selected = hasValidCandidate ? candidates[selectedIndex] : undefined;

  const coefficients: number[] = [];
  const standardErrors: number[] = [];
  const tValues: number[] = [];
  const significant: number[] = [];
  const localR2: number[] = [];
  const fitted: number[] = [];
  const residuals: number[] = [];
  const hatDiagonal: number[] = [];
  const localStatus: number[] = [];
  const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;
  const count = selected ? selected.observationCount : includedCount;
  const traceOfHat = selected ? selected.traceOfHat : nan;
  const residualSumOfSquares = selected ? selected.residualSumOfSquares : nan;
  const sigmaSquared = selected ? residualSumOfSquares / (count - traceOfHat) : nan;
  const adjustedAlpha = selected && traceOfHat > 0 ? (level * p) / traceOfHat : nan;
  const criticalT = Number.isFinite(adjustedAlpha)
    ? getStudentTCriticalValue(adjustedAlpha, count - 1)
    : nan;
  const logLikelihood = selected
    ? -0.5 *
      count *
      (Math.log(2 * Math.PI) +
        Math.log(
          Math.max(
            residualSumOfSquares / count,
            GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE
          )
        ) +
        1)
    : nan;
  const okResponses: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    let status: number = STATUS.SINGULAR;
    let r2 = nan;
    let fittedValue = nan;
    let residual = nan;
    let hat = nan;
    const rowCoefficients = new Array<number>(p).fill(nan);
    const rowErrors = new Array<number>(p).fill(nan);
    const rowT = new Array<number>(p).fill(nan);
    if (!valid[row]) {
      status = STATUS.EXCLUDED;
    } else {
      const bandwidth = getBandwidth(row, selectedValue);
      const fit = bandwidth > 0 ? fitLocation(row, bandwidth) : undefined;
      if (fit && fit.ok) {
        status = STATUS.OK;
        okResponses.push(response[row]);
        let intercept = fit.beta[0];
        for (let column = 1; column < p; column++) {
          rowCoefficients[column] = fit.beta[column];
          intercept -= fit.beta[column] * predictors[row * predictorCount + column - 1];
        }
        rowCoefficients[0] = intercept;
        fittedValue = fit.beta[0];
        residual = response[row] - fittedValue;
        hat = fit.hat;
        for (let column = 0; column < p; column++) {
          const variance = fit.variance[column] * sigmaSquared;
          if (Number.isFinite(variance) && variance >= 0) {
            rowErrors[column] = Math.sqrt(variance);
            if (rowErrors[column] > 0) {
              rowT[column] = rowCoefficients[column] / rowErrors[column];
            }
          }
        }
        const weightedMean = fit.weightedResponse / fit.weightSum;
        let residualSum = 0;
        let totalSum = 0;
        for (let other = 0; other < rowCount; other++) {
          if (!valid[other]) {
            continue;
          }
          const weight = getGeographicallyWeightedRegressionWeight(
            kernel,
            distance(row, other),
            bandwidth,
            input.gaussianCutoff
          );
          if (!(weight > 0)) {
            continue;
          }
          let prediction = fit.beta[0];
          for (let column = 0; column < predictorCount; column++) {
            prediction +=
              fit.beta[column + 1] *
              (predictors[other * predictorCount + column] -
                predictors[row * predictorCount + column]);
          }
          residualSum += weight * (response[other] - prediction) ** 2;
          totalSum += weight * (response[other] - weightedMean) ** 2;
        }
        r2 = totalSum > 0 ? 1 - residualSum / totalSum : nan;
      }
    }
    coefficients.push(...rowCoefficients);
    standardErrors.push(...rowErrors);
    tValues.push(...rowT);
    significant.push(
      ...rowT.map(value => (Number.isFinite(value) && Math.abs(value) > criticalT ? 1 : 0))
    );
    localR2.push(r2);
    fitted.push(fittedValue);
    residuals.push(residual);
    hatDiagonal.push(hat);
    localStatus.push(status);
  }
  const mean = okResponses.reduce((sum, value) => sum + value, 0) / Math.max(okResponses.length, 1);
  const total = okResponses.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  return {
    coefficients,
    standardErrors,
    tValues,
    significant,
    localR2,
    fitted,
    residuals,
    hatDiagonal,
    localStatus,
    bandwidthScores,
    candidates,
    selectedIndex,
    searchIterationCount,
    searchConverged,
    searchBounds,
    selectedValue,
    residualSumOfSquares,
    traceOfHat,
    aicc: selected ? selected.aicc : nan,
    rSquared: selected && total > 0 ? 1 - residualSumOfSquares / total : nan,
    observationCount: count,
    includedRowCount: includedCount,
    singularRowCount: selected ? selected.singularRowCount : nan,
    sigmaSquared,
    adjustedAlpha,
    criticalT,
    logLikelihood,
    hasValidCandidate
  };
}
