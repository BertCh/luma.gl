// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUSegregationLayout} from '../../../src/gpu-spatial-analysis/segregation/index';

/** CSR neighbor lists in float64 for the oracle. */
export type OracleWeights = {offsets: number[]; neighbors: number[]; weights: number[]};

/** Float64 result of one scale. */
export type SegregationOracle = {
  indices: number[];
  environment: number[];
  entropy: number[];
  dissimilarity: number[];
  theil: number[];
};

/**
 * Float64 reference for `GPUSegregation` at one scale (`null` weights is the aspatial scale).
 * Undefined indices and local values are NaN, as the operator writes them; a unit with an empty
 * environment has NaN local values and contributes nothing to the global sums.
 */
export function computeSegregationOracle(props: {
  unitCount: number;
  groupCount: number;
  counts: ArrayLike<number>;
  weights: OracleWeights | null;
  selfWeight?: number;
  atkinsonB?: number;
}): SegregationOracle {
  const {unitCount: units, groupCount: K, counts} = props;
  const selfWeight = props.selfWeight ?? 1;
  const b = props.atkinsonB ?? 0.5;
  const count = (unit: number, group: number) => {
    const value = counts[unit * K + group];
    return Number.isFinite(value) && value > 0 ? value : 0;
  };
  const unitTotal = (unit: number) => {
    let total = 0;
    for (let group = 0; group < K; group++) total += count(unit, group);
    return total;
  };
  const groupTotals = new Array<number>(K).fill(0);
  let total = 0;
  for (let unit = 0; unit < units; unit++) {
    for (let group = 0; group < K; group++) groupTotals[group] += count(unit, group);
    total += unitTotal(unit);
  }
  const P = groupTotals.map(value => (total > 0 ? value / total : 0));
  const entropyOf = (shares: number[]) =>
    shares.reduce((sum, share) => sum + (share > 0 ? -share * Math.log(share) : 0), 0);
  const E = entropyOf(P);
  const I = P.reduce((sum, share) => sum + share * (1 - share), 0);

  const environment = new Array<number>(units * K).fill(0);
  const shares = new Array<number[]>(units);
  const hasEnvironment = new Array<boolean>(units).fill(false);
  for (let unit = 0; unit < units; unit++) {
    const row = new Array<number>(K).fill(0);
    for (let group = 0; group < K; group++) {
      let sum = props.weights ? selfWeight * count(unit, group) : count(unit, group);
      if (props.weights) {
        for (
          let slot = props.weights.offsets[unit];
          slot < props.weights.offsets[unit + 1];
          slot++
        ) {
          sum += props.weights.weights[slot] * count(props.weights.neighbors[slot], group);
        }
      }
      row[group] = sum;
    }
    const rowTotal = row.reduce((a, c) => a + c, 0);
    hasEnvironment[unit] = rowTotal > 0;
    shares[unit] = row.map(value => (rowTotal > 0 ? value / rowTotal : 0));
    for (let group = 0; group < K; group++) {
      environment[unit * K + group] = rowTotal > 0 ? shares[unit][group] : NaN;
    }
  }

  const layout = getGPUSegregationLayout(K);
  const indices = new Array<number>(layout.stride).fill(NaN);
  const entropy = new Array<number>(units).fill(NaN);
  const theil = new Array<number>(units).fill(NaN);
  const dissimilarity = new Array<number>(units * K).fill(NaN);
  let sumEntropy = 0;
  let sumAbsolute = 0;
  const sumDissimilarity = new Array<number>(K).fill(0);
  const sumAtkinson = new Array<number>(K).fill(0);
  const sumExposure = new Array<number>(K * K).fill(0);
  for (let unit = 0; unit < units; unit++) {
    if (!hasEnvironment[unit]) continue;
    const t = unitTotal(unit);
    const unitEntropy = entropyOf(shares[unit]);
    entropy[unit] = unitEntropy;
    sumEntropy += t * unitEntropy;
    if (E > 0 && total > 0) theil[unit] = (t * (E - unitEntropy)) / (E * total);
    for (let group = 0; group < K; group++) {
      const share = shares[unit][group];
      sumAbsolute += t * Math.abs(share - P[group]);
      sumDissimilarity[group] += t * Math.abs(share - P[group]);
      const scale = 2 * total * P[group] * (1 - P[group]);
      if (scale > 0) dissimilarity[unit * K + group] = (t * Math.abs(share - P[group])) / scale;
      sumAtkinson[group] += t * Math.pow(Math.max(1 - share, 0), 1 - b) * Math.pow(share, b);
      for (let other = 0; other < K; other++) {
        sumExposure[group * K + other] += count(unit, group) * shares[unit][other];
      }
    }
  }
  indices[layout.entropy] = E > 0 && total > 0 ? 1 - sumEntropy / (E * total) : NaN;
  indices[layout.multiGroupDissimilarity] =
    I > 0 && total > 0 ? sumAbsolute / (2 * total * I) : NaN;
  indices[layout.diversity] = total > 0 ? E : NaN;
  for (let group = 0; group < K; group++) {
    const mixed = total > 0 && P[group] > 0 && P[group] < 1;
    indices[layout.dissimilarity + group] = mixed
      ? sumDissimilarity[group] / (2 * total * P[group] * (1 - P[group]))
      : NaN;
    indices[layout.atkinson + group] = mixed
      ? 1 -
        (P[group] / (1 - P[group])) *
          Math.pow(Math.abs(sumAtkinson[group] / (P[group] * total)), 1 / (1 - b))
      : NaN;
    for (let other = 0; other < K; other++) {
      const value =
        groupTotals[group] > 0 ? sumExposure[group * K + other] / groupTotals[group] : NaN;
      indices[layout.interaction + group * K + other] = value;
      if (other === group) indices[layout.isolation + group] = value;
    }
  }
  return {indices, environment, entropy, dissimilarity, theil};
}

/**
 * Textbook aspatial two-group dissimilarity `1/2 sum |a_i / A - b_i / B|` (group `g` against all
 * others), independent of the shared-term derivation above.
 */
export function computeTextbookDissimilarity(
  unitCount: number,
  groupCount: number,
  counts: ArrayLike<number>,
  group: number
): number {
  let groupTotal = 0;
  let otherTotal = 0;
  for (let unit = 0; unit < unitCount; unit++) {
    for (let m = 0; m < groupCount; m++) {
      if (m === group) groupTotal += counts[unit * groupCount + m];
      else otherTotal += counts[unit * groupCount + m];
    }
  }
  let sum = 0;
  for (let unit = 0; unit < unitCount; unit++) {
    let others = 0;
    for (let m = 0; m < groupCount; m++) if (m !== group) others += counts[unit * groupCount + m];
    sum += Math.abs(counts[unit * groupCount + group] / groupTotal - others / otherTotal);
  }
  return sum / 2;
}

/** Symmetric distance-band weights on a `columns x rows` grid: weight `1 / (1 + distance)`. */
export function createGridBandWeights(
  columns: number,
  rows: number,
  radius: number
): OracleWeights {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      for (let otherRow = 0; otherRow < rows; otherRow++) {
        for (let otherColumn = 0; otherColumn < columns; otherColumn++) {
          const distance = Math.hypot(column - otherColumn, row - otherRow);
          if (distance > 0 && distance <= radius) {
            neighbors.push(otherRow * columns + otherColumn);
            weights.push(1 / (1 + distance));
          }
        }
      }
      offsets.push(neighbors.length);
    }
  }
  return {offsets, neighbors, weights};
}

/**
 * Like the shared `expectClose`, but NaN must match NaN (an undefined index is NaN, never 0) and
 * a NaN on only one side fails.
 */
export function expectSegregationClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  label: string,
  relative = 2e-5,
  absolute = 1e-6
): void {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: length ${actual.length} != ${expected.length}`);
  }
  for (let index = 0; index < expected.length; index++) {
    const bothNaN = Number.isNaN(actual[index]) && Number.isNaN(expected[index]);
    if (
      !bothNaN &&
      !(
        Math.abs(actual[index] - expected[index]) <=
        absolute + relative * Math.abs(expected[index])
      )
    ) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}
