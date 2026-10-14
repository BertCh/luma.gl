// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {getGridLookupWGSL, getRingNeighborDistancesWGSL} from './gwr-grid-index';
import {getSymmetricEigenvalueRangeWGSL, getCholeskyWGSL} from './spatial-regression-solve';

/** Options of {@link getGWRFitWGSL}. @internal */
export type GWRFitWGSLOptions = {
  /** Design columns including the intercept, `P`. */
  coefficientCount: number;
  /** Grid index dimensions, or null for the all-rows scan. Planar distances only. */
  indexGridSize: readonly [number, number] | null;
  /** Declare `getLocalConditionNumber`. */
  hasConditionNumber: boolean;
  /** Also accumulate the squared-weight Gram matrix and return `variance` (local standard errors). */
  hasVariance: boolean;
  /** Distances are great-circle `geodesicHaversineDistance(a, b, radius)`; otherwise planar. */
  sphereRadius: number | null;
  /**
   * Gaussian weights of rows farther than `gaussianCutoff * bandwidth` are exactly 0, which also
   * lets the Gaussian kernel use the grid index. Null or omitted is the exact unbounded Gaussian.
   */
  gaussianCutoff?: number | null;
};

/**
 * WGSL of the local weighted least squares shared by the candidate and final-fit kernels.
 *
 * The including kernel must declare, before this text: the constants `ROW_COUNT`,
 * `PREDICTOR_COUNT`, `P`, `MAXIMUM_NEIGHBORS`, `SENTINEL`, `ADAPTIVE_FACTOR`; the bindings
 * `positions`, `predictors`, `response`, `rowValid` (and `cellOffsets`, `sortedIds` with a grid);
 * `getNaN()` and `isFiniteBits()`; and the two functions `getIsBisquare() -> bool` and
 * `getIsAdaptive() -> bool` that read the kernel and bandwidth mode from wherever the kernel keeps
 * them (the parameter buffer for the candidate passes, the row-state tail for the final fit).
 *
 * Declares: `LocalFit`, `getPosition`, `getDistance`, `getWeight`, `selectNeighborDistances`,
 * `getBandwidthForValue`, `fitLocation`, `getLocalR2`, optionally `getLocalConditionNumber`, and
 * the Cholesky and eigenvalue helpers.
 *
 * @internal
 */
export function getGWRFitWGSL(options: GWRFitWGSLOptions): string {
  const {
    coefficientCount: P,
    indexGridSize,
    hasConditionNumber,
    hasVariance,
    sphereRadius
  } = options;
  const gaussianCutoff = options.gaussianCutoff ?? null;
  const hasGrid = Boolean(indexGridSize);
  return /* wgsl */ `
${sphereRadius === null ? '' : GEODESIC_WGSL}
${getCholeskyWGSL(P)}
${hasConditionNumber ? getSymmetricEigenvalueRangeWGSL(P) : ''}
${indexGridSize ? getGridWGSL(indexGridSize, gaussianCutoff) : ''}
struct LocalFit {
  ok: bool,
  hat: f32,
  weightSum: f32,
  weightedResponse: f32,
  beta: array<f32, ${P}>,
  // diag((X'WX)^-1 X'W^2 X (X'WX)^-1) of the original (uncentered) coefficients; NaN when not
  // finite. Only filled when the squared-weight Gram matrix is accumulated.
  variance: array<f32, ${P}>
}

fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

${
  sphereRadius === null
    ? `fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return sqrt(dot(delta, delta));
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return dot(delta, delta);
}`
    : `// Great-circle distance in the units of the sphere radius; positions are (longitude, latitude).
fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  return geodesicHaversineDistance(pointA, pointB, ${getWGSLFloatLiteral(sphereRadius)});
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let distance = getDistance(pointA, pointB);
  return distance * distance;
}`
}

${
  gaussianCutoff === null
    ? ''
    : `// Gaussian weights vanish beyond this many bandwidths (compile-time gaussianCutoff).
const GAUSSIAN_CUTOFF: f32 = ${getWGSLFloatLiteral(gaussianCutoff)};
`
}
fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  let ratio = distance / bandwidth;
  if (getIsBisquare()) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }${
    gaussianCutoff === null
      ? ''
      : `
  if (distance > GAUSSIAN_CUTOFF * bandwidth) {
    return 0.0;
  }`
  }
  return exp(-0.5 * ratio * ratio);
}

${
  indexGridSize && sphereRadius === null
    ? getRingNeighborDistancesWGSL()
    : `// Keeps the maximumK smallest squared distances from row i (itself included) in ascending order.
fn selectNeighborDistances(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let squared = getSquaredDistance(origin, getPosition(row));
    if (squared < (*list)[maximumK - 1u]) {
      var slot = maximumK - 1u;
      loop {
        if (slot == 0u || (*list)[slot - 1u] <= squared) {
          break;
        }
        (*list)[slot] = (*list)[slot - 1u];
        slot = slot - 1u;
      }
      (*list)[slot] = squared;
    }
  }
}`
}

// Positive bandwidth of a ladder value for the focal row of the neighbour list, or -1 when the
// value is invalid there (fixed: not positive; adaptive: k outside [2, MAXIMUM_NEIGHBORS] or more
// than the included rows).
fn getBandwidthForValue(value: f32, isAdaptive: bool, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) -> f32 {
  if (!isAdaptive) {
    return select(-1.0, value, value > 0.0 && isFiniteBits(value));
  }
  if (!(value >= 2.0) || value > f32(MAXIMUM_NEIGHBORS)) {
    return -1.0;
  }
  let squared = (*list)[u32(floor(value + 0.5)) - 1u];
  if (squared >= SENTINEL) {
    return -1.0;
  }
  return sqrt(squared) * ADAPTIVE_FACTOR;
}

fn fitLocation(i: u32, bandwidth: f32) -> LocalFit {
  var a: array<f32, ${P * P}>;
  var b: array<f32, ${P}>;
  var z: array<f32, ${P}>;
  ${hasVariance ? `var squared: array<f32, ${P * P}>;` : ''}
  var fit: LocalFit;
  fit.ok = false;
  fit.hat = 0.0;
  fit.weightSum = 0.0;
  fit.weightedResponse = 0.0;
  for (var column = 0u; column < P; column++) {
    fit.beta[column] = 0.0;
    fit.variance[column] = getNaN();
    b[column] = 0.0;
  }
  for (var entry = 0u; entry < P * P; entry++) {
    a[entry] = 0.0;
    ${hasVariance ? 'squared[entry] = 0.0;' : ''}
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${getCandidateLoopWGSL(
    `let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    let y = response[responseOffset + row];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
        ${hasVariance ? 'squared[r * P + c] = squared[r * P + c] + weighted * weight * z[c];' : ''}
      }
    }`,
    hasGrid
  )}
  fit.weightSum = a[0];
  fit.weightedResponse = b[0];
  var scale: array<f32, ${P}>;
  for (var r = 0u; r < P; r++) {
    let diagonal = a[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return fit;
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      a[r * P + c] = a[r * P + c] * scale[r] * scale[c];
    }
    b[r] = b[r] * scale[r];
  }
  if (!choleskyFactor_${P}(&a)) {
    return fit;
  }
  let inverse00 = choleskyInverseDiagonal_${P}(&a, 0u);
  choleskySolve_${P}(&a, &b);
  var finite = isFiniteBits(inverse00);
  for (var r = 0u; r < P; r++) {
    fit.beta[r] = b[r] * scale[r];
    finite = finite && isFiniteBits(fit.beta[r]);
  }
  fit.hat = inverse00 * scale[0] * scale[0];
  fit.ok = finite && fit.hat >= 0.0;
  ${
    hasVariance
      ? `if (fit.ok) {
    // C C' = A^-1 B A^-1 with A = X'WX (centered design) and B = X'W^2 X. Column c of A^-1 comes
    // from the equilibrated factor: A^-1 e_c = s_c * (s o solve(e_c)). The original intercept is
    // gamma_0 - sum x_ic gamma_c, so its direction is t = (1, -x_i) and A^-1 t = s o solve(s o t);
    // variance_c = w' B w.
    for (var column = 0u; column < P; column++) {
      var w: array<f32, ${P}>;
      for (var r = 0u; r < P; r++) {
        w[r] = 0.0;
      }
      var columnScale = scale[column];
      if (column == 0u) {
        columnScale = 1.0;
        w[0] = scale[0];
        for (var r = 1u; r < P; r++) {
          w[r] = -scale[r] * predictors[originBase + r - 1u];
        }
      } else {
        w[column] = 1.0;
      }
      choleskySolve_${P}(&a, &w);
      for (var r = 0u; r < P; r++) {
        w[r] = scale[r] * w[r] * columnScale;
      }
      var quadratic = 0.0;
      for (var r = 0u; r < P; r++) {
        quadratic = quadratic + w[r] * w[r] * squared[r * P + r];
        for (var c = 0u; c < r; c++) {
          quadratic = quadratic + 2.0 * w[r] * w[c] * squared[r * P + c];
        }
      }
      fit.variance[column] = select(getNaN(), quadratic, quadratic >= 0.0 && isFiniteBits(quadratic));
    }
  }`
      : ''
  }
  return fit;
}

// Weighted R^2 of the local fit over the neighborhood of row i.
fn getLocalR2(i: u32, bandwidth: f32, fit: LocalFit) -> f32 {
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  let weightedMean = fit.weightedResponse / fit.weightSum;
  var residualSum = 0.0;
  var totalSum = 0.0;
  ${getCandidateLoopWGSL(
    `let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    var prediction = fit.beta[0];
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      prediction = prediction + fit.beta[column + 1u] * (predictors[rowBase + column] - predictors[originBase + column]);
    }
    let y = response[responseOffset + row];
    residualSum = residualSum + weight * (y - prediction) * (y - prediction);
    totalSum = totalSum + weight * (y - weightedMean) * (y - weightedMean);`,
    hasGrid
  )}
  return select(getNaN(), 1.0 - residualSum / totalSum, totalSum > 0.0);
}
${hasConditionNumber ? getConditionNumberWGSL(P, hasGrid) : ''}
`;
}

/**
 * Loop over the candidate rows of a location (`origin`, `bandwidth` in scope) running `body` with
 * `row` bound. Without a grid every row is visited in order. With one, bounded fixed-bandwidth
 * fits (bisquare, fixed) visit only the sorted contents of the cells covering the bandwidth, one
 * contiguous slot range per grid row; other encodings still visit every row.
 */
function getCandidateLoopWGSL(body: string, hasGrid: boolean): string {
  if (!hasGrid) {
    return `for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    ${body}
  }`;
  }
  return `let useGrid = isGridBandwidth(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, getSupportRadius(bandwidth)), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = ROW_COUNT;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      if (rowValid[rowValidOffset + row] == 0u) {
        continue;
      }
      ${body}
    }
  }`;
}

/**
 * `getLocalConditionNumber`: mgwr `local_collinearity` condition number. With weights `w_j`
 * (kernel weights, not their square roots, exactly as mgwr's `xw = x * wi`) and the raw design
 * `x_j = (1, X_j)`, the singular values of the column-normalized `diag(w) x` are the square roots
 * of the eigenvalues of `C = D^-1/2 (sum_j w_j^2 x_j x_j') D^-1/2`, `D = diag(sum_j w_j^2 x_jc^2)`.
 * The moment is accumulated in coordinates centered on the focal row (as the fit is) and shifted
 * to raw coordinates afterwards; the extreme eigenvalues of the unit-diagonal `C` come from the
 * shared `getSymmetricEigenvalueRange_P` (cyclic Jacobi). A near-singular `C` returns a very large
 * number rather than NaN.
 */
function getConditionNumberWGSL(coefficientCount: number, hasGrid: boolean): string {
  const P = coefficientCount;
  return /* wgsl */ `
fn getLocalConditionNumber(i: u32, bandwidth: f32) -> f32 {
  var moment: array<f32, ${P * P}>;
  var z: array<f32, ${P}>;
  for (var entry = 0u; entry < P * P; entry++) {
    moment[entry] = 0.0;
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${getCandidateLoopWGSL(
    `let weight = getWeight(getDistance(origin, getPosition(row)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let squaredWeight = weight * weight;
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    for (var r = 0u; r < P; r++) {
      for (var c = 0u; c <= r; c++) {
        moment[r * P + c] = moment[r * P + c] + squaredWeight * z[r] * z[c];
      }
    }`,
    hasGrid
  )}
  // Shift the centered moment to raw coordinates: x = (1, x_i + z), so raw = T moment T'.
  var center: array<f32, ${P}>;
  center[0] = 0.0;
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    center[column + 1u] = predictors[originBase + column];
  }
  var raw: array<f32, ${P * P}>;
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      // T = I + e_r' center, with center[0] = 0 and the intercept row of T equal to e_0.
      let m00 = moment[0];
      let mr0 = moment[r * P];
      let mc0 = moment[c * P];
      var value = moment[r * P + c];
      if (r > 0u && c > 0u) {
        value = value + center[r] * mc0 + center[c] * mr0 + center[r] * center[c] * m00;
      } else if (r > 0u) {
        value = value + center[r] * m00;
      }
      raw[r * P + c] = value;
      raw[c * P + r] = value;
    }
  }
  var scale: array<f32, ${P}>;
  for (var r = 0u; r < P; r++) {
    let diagonal = raw[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return getNaN();
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c < P; c++) {
      raw[r * P + c] = raw[r * P + c] * scale[r] * scale[c];
    }
  }
  let eigenvalues = getSymmetricEigenvalueRange_${P}(&raw);
  return sqrt(eigenvalues.y / max(eigenvalues.x, 1.0e-30));
}
`;
}

function getGridWGSL(
  indexGridSize: readonly [number, number],
  gaussianCutoff: number | null
): string {
  return /* wgsl */ `
${getGridLookupWGSL(indexGridSize)}

// Bisquare has bounded support (fixed or adaptive), so the grid is exact for it; the Gaussian
// kernel is bounded only with a compile-time cutoff.
fn isGridBandwidth(bandwidth: f32) -> bool {
  return ${gaussianCutoff === null ? 'getIsBisquare()' : 'true'} && bandwidth > 0.0 && isFiniteBits(bandwidth);
}

// Distance beyond which the weight is exactly 0.
fn getSupportRadius(bandwidth: f32) -> f32 {
  ${gaussianCutoff === null ? 'return bandwidth;' : 'return select(bandwidth * GAUSSIAN_CUTOFF, bandwidth, getIsBisquare());'}
}

// Included rows in grid-cell order: invocation \`slot\` fits the slot-th indexed row.
fn getFocalCount() -> u32 {
  return min(cellOffsets[cellOffsetsOffset + INDEX_WIDTH * INDEX_HEIGHT], ROW_COUNT);
}

fn getFocalRow(slot: u32) -> u32 {
  return sortedIds[sortedIdsOffset + slot];
}
`;
}
