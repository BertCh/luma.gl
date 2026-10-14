// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getChiSquareSurvival} from '../../../src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics';

/** Double-precision homogeneity test of conditional transition matrices (Bickenbach and Bode 2003). */
export type HomogeneityOracle = {
  q: number;
  qPValue: number;
  likelihoodRatio: number;
  likelihoodRatioPValue: number;
  degreesOfFreedom: number;
};

/**
 * Independent CPU implementation of the homogeneity test of `GPUTransitionDiagnostics`, written
 * from the giddy formulas (`giddy.markov.Homogeneity_Results`). A from-state without any transition
 * contributes 0 degrees of freedom (giddy's expression contributes 1 for it).
 *
 * @param counts `conditionCount * classCount^2` counts, `(c * K + from) * K + to`.
 */
export function computeHomogeneityOracle(
  counts: ArrayLike<number>,
  classCount: number,
  conditionCount: number
): HomogeneityOracle {
  let q = 0;
  let likelihoodRatio = 0;
  let degreesOfFreedom = 0;
  for (let from = 0; from < classCount; from++) {
    const pooled = new Array<number>(classCount).fill(0);
    for (let condition = 0; condition < conditionCount; condition++) {
      for (let to = 0; to < classCount; to++) {
        pooled[to] += counts[(condition * classCount + from) * classCount + to];
      }
    }
    const pooledTotal = pooled.reduce((sum, value) => sum + value, 0);
    if (pooledTotal === 0) {
      continue;
    }
    let regimes = 0;
    for (let condition = 0; condition < conditionCount; condition++) {
      let regimeTotal = 0;
      for (let to = 0; to < classCount; to++) {
        regimeTotal += counts[(condition * classCount + from) * classCount + to];
      }
      if (regimeTotal === 0) {
        continue;
      }
      regimes++;
      for (let to = 0; to < classCount; to++) {
        const observed = counts[(condition * classCount + from) * classCount + to];
        const expected = (regimeTotal * pooled[to]) / pooledTotal;
        if (pooled[to] > 0) {
          q += (observed - expected) ** 2 / expected;
        }
        if (observed > 0) {
          likelihoodRatio += observed * Math.log(observed / expected);
        }
      }
    }
    degreesOfFreedom += (regimes - 1) * (pooled.filter(value => value > 0).length - 1);
  }
  likelihoodRatio = Math.max(2 * likelihoodRatio, 0);
  const survival = (value: number) =>
    degreesOfFreedom > 0 ? getChiSquareSurvival(value, degreesOfFreedom) : NaN;
  return {
    q,
    qPValue: survival(q),
    likelihoodRatio,
    likelihoodRatioPValue: survival(likelihoodRatio),
    degreesOfFreedom
  };
}

/**
 * Steady state of one chain by a plain linear solve (not the GTH elimination of the GPU kernel),
 * on the states with a transition in or out. NaN for every state unless each kept state has an
 * outgoing transition and the kept chain is one communicating class. Dropped states get 0.
 *
 * @param counts `classCount^2` counts of one chain, `from * K + to`.
 */
export function computeSteadyStateOracle(counts: ArrayLike<number>, classCount: number): number[] {
  const result = new Array<number>(classCount).fill(0);
  const used: number[] = [];
  let failed = false;
  for (let state = 0; state < classCount; state++) {
    let outgoing = 0;
    let incoming = 0;
    for (let other = 0; other < classCount; other++) {
      outgoing += counts[state * classCount + other];
      incoming += counts[other * classCount + state];
    }
    if (outgoing > 0 || incoming > 0) {
      used.push(state);
      failed ||= outgoing === 0;
    }
  }
  if (used.length === 0 || failed) {
    return result.fill(NaN);
  }
  const size = used.length;
  const probabilities = used.map(from => {
    const total = used.reduce((sum, to) => sum + counts[from * classCount + to], 0);
    return used.map(to => counts[from * classCount + to] / total);
  });
  // Irreducible: every state reaches every other (Floyd-Warshall closure on the support).
  const reach = probabilities.map(row => row.map(value => value > 0));
  for (let via = 0; via < size; via++) {
    for (let from = 0; from < size; from++) {
      for (let to = 0; to < size; to++) {
        reach[from][to] ||= reach[from][via] && reach[via][to];
      }
    }
  }
  if (reach.some((row, from) => row.some((value, to) => from !== to && !value))) {
    return result.fill(NaN);
  }
  // Solve (P^T - I) pi = 0 with the last equation replaced by sum(pi) = 1.
  const system = Array.from({length: size}, (_, row) =>
    Array.from({length: size + 1}, (_, column) => {
      if (row === size - 1) {
        return column === size ? 1 : 1;
      }
      return column === size ? 0 : probabilities[column][row] - (row === column ? 1 : 0);
    })
  );
  for (let pivot = 0; pivot < size; pivot++) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row++) {
      if (Math.abs(system[row][pivot]) > Math.abs(system[best][pivot])) best = row;
    }
    [system[pivot], system[best]] = [system[best], system[pivot]];
    for (let row = 0; row < size; row++) {
      if (row === pivot) continue;
      const factor = system[row][pivot] / system[pivot][pivot];
      for (let column = pivot; column <= size; column++) {
        system[row][column] -= factor * system[pivot][column];
      }
    }
  }
  used.forEach((state, index) => {
    result[state] = system[index][size] / system[index][index];
  });
  return result;
}
