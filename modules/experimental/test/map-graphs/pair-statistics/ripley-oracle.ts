// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  forEachPair,
  getIncludedRows,
  type PairStatisticsFrame,
  type PairStatisticsScene
} from './pair-statistics-oracle';

const fround = Math.fround;

/** Isotropic weight cap, equal to `GPU_RIPLEY_WEIGHT_CAP`. */
export const RIPLEY_ORACLE_WEIGHT_CAP = 100;

/** Fixed-point error of one isotropic weight: `2^-18`. */
const WEIGHT_QUANTIZATION_ERROR = 2 ** -18;

/** Relative band around a radius edge inside which GPU division rounding may flip an annulus. */
const EDGE_BAND = 2e-6;

/** Edge-correction mode of the oracle. */
export type RipleyOracleEdgeCorrection = 'none' | 'border' | 'isotropic';

/** CPU result of {@link computeRipleyOnCPU}. */
export type RipleyOracleResult = {
  n: number;
  /** f32 window area. */
  area: number;
  radii: number[];
  pairCounts: number[];
  /** Per boundary `j` in `[0, radiusCount]` the pairs within the edge band of scaled distance `j`. */
  boundaryPairs: number[];
  k: number[];
  l: number[];
  lMinusR: number[];
  pairCorrelation: number[];
  /** Absolute error bounds of each output, from f32 rounding, fixed point and edge-band flips. */
  kTolerance: number[];
  lTolerance: number[];
  pairCorrelationTolerance: number[];
  /** Largest pair count tolerance per annulus (pairs near either bounding radius). */
  pairCountTolerance: number[];
};

/**
 * Closed-form isotropic edge-correction weight `1 / f` for a circle of radius `distance` centered at
 * distance `left`, `bottom`, `right`, `top` from the four window sides, capped at the weight cap.
 * Evaluated in f64 from the given f32 distances.
 */
export function getIsotropicWeightOnCPU(
  left: number,
  bottom: number,
  right: number,
  top: number,
  distance: number
): number {
  const angle = (side: number) =>
    distance > side ? Math.atan2(Math.sqrt((distance - side) * (distance + side)), side) : 0;
  const [a, b, c, e] = [angle(left), angle(bottom), angle(right), angle(top)];
  const overlap = (first: number, second: number) => Math.max(0, first + second - Math.PI / 2);
  const inside =
    1 -
    (a + b + c + e) / Math.PI +
    (overlap(a, b) + overlap(b, c) + overlap(c, e) + overlap(e, a)) / (2 * Math.PI);
  return Math.min(1 / Math.max(inside, 1 / RIPLEY_ORACLE_WEIGHT_CAP), RIPLEY_ORACLE_WEIGHT_CAP);
}

/**
 * f64 Ripley's K / Besag's L / annulus pair correlation with the `GPURipley` inclusion, binning
 * (f32 distances, scaled distance `d / maximumDistance * radiusCount`), border counts and isotropic
 * weights, and with error bounds covering f32 rounding, the isotropic fixed point and pairs that
 * lie within the GPU division rounding band of a radius edge.
 */
export function computeRipleyOnCPU(
  scene: PairStatisticsScene,
  frame: PairStatisticsFrame & {edgeCorrection?: RipleyOracleEdgeCorrection},
  radiusCount: number
): RipleyOracleResult {
  const mode = frame.edgeCorrection ?? 'isotropic';
  const [minX, minY, maxX, maxY] = frame.bounds.map(fround);
  const maximumDistance = fround(frame.maximumDistance);
  const included = getIncludedRows(scene, frame);
  const n = included.length;
  const area = fround(fround(maxX - minX) * fround(maxY - minY));
  const radii = Array.from({length: radiusCount}, (_, b) =>
    fround(fround(maximumDistance * (b + 1)) / radiusCount)
  );
  const count = () => new Array<number>(radiusCount).fill(0);
  const rawCounts = count();
  const boundaryPairs = new Array<number>(radiusCount + 1).fill(0);
  // Border: pairs and eligible foci per radius, with per-focus bookkeeping.
  const borderNumerator = count();
  const borderFoci = count();
  const pairSlack = count();
  const focusSlack = count();
  let maximumFocusPairs = 0;
  const isotropicSums = count();
  const isotropicSlack = count();

  const borderCounts = new Map<number, number>();
  for (const focus of included) {
    const x = scene.positions[focus * 2];
    const y = scene.positions[focus * 2 + 1];
    const borderDistance = Math.min(
      fround(x - minX),
      fround(maxX - x),
      fround(y - minY),
      fround(maxY - y)
    );
    const scaled = fround(fround(borderDistance / maximumDistance) * radiusCount);
    const borderCount = Math.min(Math.floor(scaled), radiusCount);
    borderCounts.set(focus, borderCount);
    for (let b = 0; b < radiusCount; b++) {
      if (b < borderCount) {
        borderFoci[b]++;
      }
      if (Math.abs(scaled - (b + 1)) < EDGE_BAND * Math.max(scaled, 1)) {
        focusSlack[b]++;
      }
    }
  }
  const focusPairs = new Map<number, number>();
  forEachPair(scene, frame, 'ordered', ({focus, deltaX, deltaY, distance}) => {
    const scaled = fround(fround(distance / maximumDistance) * radiusCount);
    const annulus = Math.min(Math.max(Math.ceil(scaled) - 1, 0), radiusCount - 1);
    rawCounts[annulus]++;
    const band = EDGE_BAND * Math.max(scaled, 1);
    const nearest = Math.round(scaled);
    const nearBoundary = Math.abs(scaled - nearest) < band;
    if (nearBoundary && nearest >= 1 && nearest <= radiusCount) {
      boundaryPairs[nearest]++;
    }
    const borderCount = borderCounts.get(focus)!;
    focusPairs.set(focus, (focusPairs.get(focus) ?? 0) + 1);
    for (let b = annulus; b < borderCount; b++) {
      borderNumerator[b]++;
    }
    if (nearBoundary && nearest >= 1 && nearest <= radiusCount) {
      pairSlack[nearest - 1]++;
    }
    const x = scene.positions[focus * 2];
    const y = scene.positions[focus * 2 + 1];
    const weight = getIsotropicWeightOnCPU(
      fround(x - minX),
      fround(y - minY),
      fround(maxX - x),
      fround(maxY - y),
      distance
    );
    isotropicSums[annulus] += weight;
    if (nearBoundary && nearest >= 1 && nearest <= radiusCount) {
      isotropicSlack[nearest - 1] += weight;
    }
  });
  for (const pairs of focusPairs.values()) {
    maximumFocusPairs = Math.max(maximumFocusPairs, pairs);
  }

  const k = new Array<number>(radiusCount).fill(NaN);
  const kTolerance = new Array<number>(radiusCount).fill(NaN);
  let cumulativeCount = 0;
  let cumulativeWeight = 0;
  let cumulativeSlackCount = 0;
  const cumulativeSlackWeight = isotropicSlack.slice();
  for (let b = 0; b < radiusCount; b++) {
    cumulativeCount += rawCounts[b];
    cumulativeWeight += isotropicSums[b];
    cumulativeSlackCount = pairSlack[b];
    if (n < 2 || !(area > 0)) {
      continue;
    }
    if (mode === 'none') {
      k[b] = (area / (n * (n - 1))) * cumulativeCount;
      kTolerance[b] = (area / (n * (n - 1))) * cumulativeSlackCount;
    } else if (mode === 'isotropic') {
      k[b] = (area / (n * (n - 1))) * cumulativeWeight;
      kTolerance[b] =
        (area / (n * (n - 1))) *
        (cumulativeSlackWeight[b] +
          cumulativeCount * WEIGHT_QUANTIZATION_ERROR +
          2e-5 * cumulativeWeight);
    } else if (borderFoci[b] > 0) {
      const foci = borderFoci[b];
      k[b] = (area * borderNumerator[b]) / ((n - 1) * foci);
      kTolerance[b] =
        (area / (n - 1)) *
        ((pairSlack[b] + focusSlack[b] * maximumFocusPairs) / foci +
          (borderNumerator[b] * focusSlack[b]) / (foci * foci));
    }
    kTolerance[b] += 3e-6 * Math.abs(k[b]);
  }
  const l = k.map(value => Math.sqrt(value / Math.PI));
  const lTolerance = k.map((value, b) => {
    const tolerance = kTolerance[b];
    return (
      Math.sqrt((value + tolerance) / Math.PI) -
      Math.sqrt(Math.max(value - tolerance, 0) / Math.PI) +
      2e-6 * Math.sqrt(value / Math.PI) +
      1e-6
    );
  });
  const lMinusR = l.map((value, b) => value - radii[b]);
  const pairCorrelation = new Array<number>(radiusCount).fill(NaN);
  const pairCorrelationTolerance = new Array<number>(radiusCount).fill(NaN);
  for (let b = 0; b < radiusCount; b++) {
    const previousRadius = b === 0 ? 0 : radii[b - 1];
    const annulusArea = Math.PI * (radii[b] - previousRadius) * (radii[b] + previousRadius);
    const previousK = b === 0 ? 0 : k[b - 1];
    pairCorrelation[b] = (k[b] - previousK) / annulusArea;
    pairCorrelationTolerance[b] =
      ((kTolerance[b] + (b === 0 ? 0 : kTolerance[b - 1])) / annulusArea) * 1.05 + 1e-6;
  }
  const pairCountTolerance = rawCounts.map(
    (_, b) =>
      (b >= 1 ? boundaryPairs[b] : 0) + (b + 1 <= radiusCount - 1 ? boundaryPairs[b + 1] : 0)
  );
  return {
    n,
    area,
    radii,
    pairCounts: rawCounts,
    boundaryPairs,
    k,
    l,
    lMinusR,
    pairCorrelation,
    kTolerance,
    lTolerance,
    pairCorrelationTolerance,
    pairCountTolerance
  };
}
