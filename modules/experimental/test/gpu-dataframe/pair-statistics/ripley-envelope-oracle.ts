// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPhilox4x32, getPhiloxUnitFloat} from '../../../src/utils/philox-random';

const fround = Math.fround;

/** Second Philox key word of `GPURipleyEnvelope` (ASCII `RIPF`). */
const PHILOX_KEY_TAG = 0x52495046;

/** Statistic scale of an envelope. */
export type RipleyEnvelopeOracleStatistic = 'k' | 'l' | 'l-minus-r';

/** CPU result of {@link computeRipleyEnvelopeOnCPU}. */
export type RipleyEnvelopeOracleResult = {
  lower: number[];
  upper: number[];
  globalLower: number[];
  globalUpper: number[];
  theoretical: number[];
  /** `T_j = max_r |L_j(r) - r|` per simulation, NaN when no radius is finite. */
  deviations: number[];
  critical: number;
  observedDeviation: number;
  pValue: number;
};

/**
 * CPU mirror of the `GPURipleyEnvelope` simulation kernel: `count` uniform points in the window
 * from Philox4x32-10 with key `(seed, 'RIPF')` and counter `(row, simulationIndex, 0, 1)`, using
 * f32 arithmetic in the same order as the shader (a GPU fused multiply-add can differ by one ulp).
 */
export function simulateCSROnCPU(
  seed: number,
  simulationIndex: number,
  count: number,
  bounds: readonly [number, number, number, number]
): Float32Array {
  const [minX, minY, maxX, maxY] = bounds.map(fround);
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const words = getPhilox4x32([row, simulationIndex, 0, 1], [seed, PHILOX_KEY_TAG]);
    positions[row * 2] = fround(minX + fround(getPhiloxUnitFloat(words[0]) * fround(maxX - minX)));
    positions[row * 2 + 1] = fround(
      minY + fround(getPhiloxUnitFloat(words[1]) * fround(maxY - minY))
    );
  }
  return positions;
}

function getLFromK(kValue: number): number {
  return Math.sqrt(Math.max(kValue, 0) / Math.PI);
}

/**
 * f64 pointwise and global CSR envelopes from simulated K curves, following the definitions in
 * `GPURipleyEnvelope`: NaN entries are skipped per radius / per curve, ranks break ties by
 * simulation index, the global band is `r +- (rank-th largest max |L_j(r) - r|)`.
 */
export function computeRipleyEnvelopeOnCPU(
  simulatedK: readonly (readonly number[])[],
  observedK: readonly number[] | undefined,
  radii: readonly number[],
  rank: number,
  statistic: RipleyEnvelopeOracleStatistic
): RipleyEnvelopeOracleResult {
  const fromK = (kValue: number, radius: number) =>
    statistic === 'k' ? kValue : getLFromK(kValue) - (statistic === 'l-minus-r' ? radius : 0);
  const fromL = (lValue: number, radius: number) =>
    statistic === 'k'
      ? Math.PI * Math.max(lValue, 0) ** 2
      : lValue - (statistic === 'l-minus-r' ? radius : 0);
  const getMaximumDeviation = (curve: readonly number[]) => {
    let deviation = 0;
    let anyFinite = false;
    curve.forEach((kValue, b) => {
      if (Number.isFinite(kValue)) {
        deviation = Math.max(deviation, Math.abs(getLFromK(kValue) - radii[b]));
        anyFinite = true;
      }
    });
    return anyFinite ? deviation : NaN;
  };
  const deviations = simulatedK.map(getMaximumDeviation);
  const finiteDeviations = deviations.filter(Number.isFinite).sort((a, b) => b - a);
  const critical = finiteDeviations.length >= rank ? finiteDeviations[rank - 1] : NaN;
  const lower: number[] = [];
  const upper: number[] = [];
  for (let b = 0; b < radii.length; b++) {
    const column = simulatedK
      .map(curve => curve[b])
      .filter(Number.isFinite)
      .sort((x, y) => x - y);
    const defined = column.length >= 2 * rank;
    lower.push(defined ? fromK(column[rank - 1], radii[b]) : NaN);
    upper.push(defined ? fromK(column[column.length - rank], radii[b]) : NaN);
  }
  const observedDeviation = observedK ? getMaximumDeviation(observedK) : NaN;
  const pValue = Number.isFinite(observedDeviation)
    ? (1 + finiteDeviations.filter(value => value >= observedDeviation).length) /
      (finiteDeviations.length + 1)
    : NaN;
  return {
    lower,
    upper,
    globalLower: radii.map(radius => fromL(radius - critical, radius)),
    globalUpper: radii.map(radius => fromL(radius + critical, radius)),
    theoretical: radii.map(radius => fromL(radius, radius)),
    deviations,
    critical,
    observedDeviation,
    pValue
  };
}
