// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Variogram model families supported by {@link fitVariogramModel}. */
export type VariogramModelType = 'spherical' | 'exponential' | 'gaussian';

/**
 * Least-squares weighting of {@link fitVariogramModel}:
 * - `'cressie'`: `N_h / gamma_model(h)^2` (Cressie 1985), iterated three times.
 * - `'pairs'`: `N_h`.
 * - `'none'`: ordinary least squares.
 */
export type VariogramModelWeighting = 'cressie' | 'pairs' | 'none';

/**
 * A fitted variogram model `gamma(h) = nugget + sill * shape(h / range)` for `h > 0` and
 * `gamma(0) = 0`.
 */
export type VariogramModel = {
  /** Model family. */
  model: VariogramModelType;
  /** Nugget `c0 >= 0`, the discontinuity at the origin. */
  nugget: number;
  /** Partial sill `c >= 0`; the total sill is `nugget + sill`. */
  sill: number;
  /**
   * Effective (practical) range: the spherical model reaches the sill at `range`, the exponential
   * and Gaussian models reach 95% of it there (gstat ranges `range / 3` and `range / sqrt(3)`).
   */
  range: number;
  /** Weighted sum of squared residuals at the fitted parameters. */
  residual: number;
};

/** Empirical variogram bins, for example read back from `GPUVariogram`. */
export type VariogramModelInput = {
  /** Representative lag per bin, usually the mean pair distance. */
  distances: ArrayLike<number>;
  /** Empirical semivariance per bin. */
  semivariances: ArrayLike<number>;
  /** Pair count per bin. Bins with no pairs or non-finite values are skipped. */
  pairCounts: ArrayLike<number>;
};

/** Options of {@link fitVariogramModel}. */
export type VariogramModelOptions = {
  /** Model family. */
  model: VariogramModelType;
  /** Least-squares weighting. Defaults to `'cressie'`. */
  weighting?: VariogramModelWeighting;
};

const RANGE_LADDER_LENGTH = 200;
const GOLDEN_SECTION_ITERATIONS = 60;
const CRESSIE_PASSES = 3;

/**
 * Returns the normalized shape `shape(h / range)` in `[0, 1]` of a model family.
 *
 * @param model Model family.
 * @param scaledDistance `h / range`, non-negative.
 */
export function getVariogramModelShape(model: VariogramModelType, scaledDistance: number): number {
  switch (model) {
    case 'spherical':
      return scaledDistance >= 1 ? 1 : 1.5 * scaledDistance - 0.5 * scaledDistance ** 3;
    case 'exponential':
      return 1 - Math.exp(-3 * scaledDistance);
    case 'gaussian':
      return 1 - Math.exp(-3 * scaledDistance * scaledDistance);
    default:
      throw new Error(`Unknown variogram model ${String(model)}`);
  }
}

/**
 * Evaluates a fitted variogram model at lag `distance` (`0` at the origin).
 *
 * @param model Fitted model.
 * @param distance Non-negative lag.
 */
export function evaluateVariogramModel(
  model: Pick<VariogramModel, 'model' | 'nugget' | 'sill' | 'range'>,
  distance: number
): number {
  if (distance <= 0) {
    return 0;
  }
  return model.nugget + model.sill * getVariogramModelShape(model.model, distance / model.range);
}

/**
 * Fits a spherical, exponential or Gaussian model to empirical variogram bins by weighted least
 * squares, on the CPU.
 *
 * Fitting runs on at most a few hundred read-back bins and is nonlinear only in the range, so it
 * stays on the CPU: for each range on a 200-step log ladder over `[minLag / 10, 3 * maxLag]` the
 * nugget and partial sill are the closed-form non-negative weighted linear least-squares solution,
 * and the best range is refined by golden-section search between its ladder neighbors. Cressie
 * weighting repeats the search three times with weights from the previous model.
 *
 * @param input Empirical bins.
 * @param options Model family and weighting.
 * @returns The fitted model.
 * @throws If fewer than two bins are usable or the model or weighting is unknown.
 */
export function fitVariogramModel(
  input: VariogramModelInput,
  options: VariogramModelOptions
): VariogramModel {
  const {model} = options;
  const weighting = options.weighting ?? 'cressie';
  getVariogramModelShape(model, 0);
  if (weighting !== 'cressie' && weighting !== 'pairs' && weighting !== 'none') {
    throw new Error(`Unknown variogram weighting ${String(weighting)}`);
  }
  const distances: number[] = [];
  const semivariances: number[] = [];
  const pairCounts: number[] = [];
  const length = Math.min(
    input.distances.length,
    input.semivariances.length,
    input.pairCounts.length
  );
  for (let bin = 0; bin < length; bin++) {
    const distance = input.distances[bin];
    const semivariance = input.semivariances[bin];
    const pairCount = input.pairCounts[bin];
    if (
      Number.isFinite(distance) &&
      Number.isFinite(semivariance) &&
      distance > 0 &&
      pairCount > 0
    ) {
      distances.push(distance);
      semivariances.push(semivariance);
      pairCounts.push(pairCount);
    }
  }
  if (distances.length < 2) {
    throw new Error('Variogram model fitting needs at least two non-empty bins');
  }
  let weights = pairCounts.map(count => (weighting === 'none' ? 1 : count));
  let best = fitRange(model, distances, semivariances, weights);
  if (weighting === 'cressie') {
    for (let pass = 0; pass < CRESSIE_PASSES; pass++) {
      const previous = best;
      weights = pairCounts.map((count, bin) => {
        const fitted = evaluateVariogramModel({...previous, model}, distances[bin]);
        return count / Math.max(fitted * fitted, Number.MIN_VALUE);
      });
      best = fitRange(model, distances, semivariances, weights);
    }
  }
  return {model, ...best};
}

type LinearFit = {nugget: number; sill: number; range: number; residual: number};

function fitRange(
  model: VariogramModelType,
  distances: number[],
  semivariances: number[],
  weights: number[]
): LinearFit {
  const minimumLag = Math.min(...distances);
  const maximumLag = Math.max(...distances);
  const lowest = Math.log(minimumLag / 10);
  const highest = Math.log(3 * maximumLag);
  const ladder: LinearFit[] = [];
  for (let step = 0; step < RANGE_LADDER_LENGTH; step++) {
    const range = Math.exp(lowest + ((highest - lowest) * step) / (RANGE_LADDER_LENGTH - 1));
    ladder.push(fitLinear(model, range, distances, semivariances, weights));
  }
  let bestStep = 0;
  for (let step = 1; step < ladder.length; step++) {
    if (ladder[step].residual < ladder[bestStep].residual) {
      bestStep = step;
    }
  }
  let left = Math.log(ladder[Math.max(bestStep - 1, 0)].range);
  let right = Math.log(ladder[Math.min(bestStep + 1, ladder.length - 1)].range);
  const ratio = (Math.sqrt(5) - 1) / 2;
  let inner = right - ratio * (right - left);
  let outer = left + ratio * (right - left);
  let innerFit = fitLinear(model, Math.exp(inner), distances, semivariances, weights);
  let outerFit = fitLinear(model, Math.exp(outer), distances, semivariances, weights);
  for (let iteration = 0; iteration < GOLDEN_SECTION_ITERATIONS; iteration++) {
    if (innerFit.residual <= outerFit.residual) {
      right = outer;
      outer = inner;
      outerFit = innerFit;
      inner = right - ratio * (right - left);
      innerFit = fitLinear(model, Math.exp(inner), distances, semivariances, weights);
    } else {
      left = inner;
      inner = outer;
      innerFit = outerFit;
      outer = left + ratio * (right - left);
      outerFit = fitLinear(model, Math.exp(outer), distances, semivariances, weights);
    }
  }
  let best = ladder[bestStep];
  for (const candidate of [innerFit, outerFit]) {
    if (candidate.residual < best.residual) {
      best = candidate;
    }
  }
  return best;
}

/** Non-negative weighted least squares of `gamma ~ nugget + sill * shape` for a fixed range. */
function fitLinear(
  model: VariogramModelType,
  range: number,
  distances: number[],
  semivariances: number[],
  weights: number[]
): LinearFit {
  let weightSum = 0;
  let shapeSum = 0;
  let shapeSquareSum = 0;
  let valueSum = 0;
  let crossSum = 0;
  const shapes = distances.map(distance => getVariogramModelShape(model, distance / range));
  for (let bin = 0; bin < distances.length; bin++) {
    const weight = weights[bin];
    const shape = shapes[bin];
    weightSum += weight;
    shapeSum += weight * shape;
    shapeSquareSum += weight * shape * shape;
    valueSum += weight * semivariances[bin];
    crossSum += weight * shape * semivariances[bin];
  }
  const determinant = weightSum * shapeSquareSum - shapeSum * shapeSum;
  let nugget = 0;
  let sill = 0;
  if (Math.abs(determinant) > 1e-300) {
    nugget = (shapeSquareSum * valueSum - shapeSum * crossSum) / determinant;
    sill = (weightSum * crossSum - shapeSum * valueSum) / determinant;
  }
  if (!(nugget >= 0 && sill >= 0)) {
    // Clamp one parameter at zero and re-solve the other; keep the better feasible candidate.
    const sillOnly = shapeSquareSum > 0 ? Math.max(crossSum / shapeSquareSum, 0) : 0;
    const nuggetOnly = weightSum > 0 ? Math.max(valueSum / weightSum, 0) : 0;
    const sillOnlyResidual = getResidual(0, sillOnly, shapes, semivariances, weights);
    const nuggetOnlyResidual = getResidual(nuggetOnly, 0, shapes, semivariances, weights);
    if (sillOnlyResidual <= nuggetOnlyResidual) {
      nugget = 0;
      sill = sillOnly;
    } else {
      nugget = nuggetOnly;
      sill = 0;
    }
  }
  return {nugget, sill, range, residual: getResidual(nugget, sill, shapes, semivariances, weights)};
}

function getResidual(
  nugget: number,
  sill: number,
  shapes: number[],
  semivariances: number[],
  weights: number[]
): number {
  let residual = 0;
  for (let bin = 0; bin < shapes.length; bin++) {
    const difference = semivariances[bin] - nugget - sill * shapes[bin];
    residual += weights[bin] * difference * difference;
  }
  return residual;
}
