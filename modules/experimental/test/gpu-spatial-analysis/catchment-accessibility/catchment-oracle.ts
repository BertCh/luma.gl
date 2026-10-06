// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUWeights} from './rig';

/** Double-precision 2SFCA / 3SFCA reference. */
export function computeCatchmentOracle(
  method: '2sfca' | '3sfca',
  supply: number[],
  demand: number[],
  facilityWeights: CPUWeights,
  demandWeights: CPUWeights
): {ratios: number[]; accessibility: number[]; reachable: number[]} {
  const selection = demand.map((_, i) => {
    let total = 0;
    for (let s = demandWeights.offsets[i]; s < demandWeights.offsets[i + 1]; s++) {
      total += demandWeights.weights[s];
    }
    return total;
  });
  const ratios = supply.map((value, j) => {
    let captured = 0;
    for (let s = facilityWeights.offsets[j]; s < facilityWeights.offsets[j + 1]; s++) {
      const i = facilityWeights.neighbors[s];
      if (method === '2sfca') captured += demand[i] * facilityWeights.weights[s];
      else if (selection[i] > 0)
        captured += (demand[i] * facilityWeights.weights[s]) / selection[i];
    }
    return captured > 0 ? value / captured : 0;
  });
  const accessibility = demand.map((_, i) => {
    let score = 0;
    for (let s = demandWeights.offsets[i]; s < demandWeights.offsets[i + 1]; s++) {
      score += demandWeights.weights[s] * ratios[demandWeights.neighbors[s]];
    }
    return method === '3sfca' ? (selection[i] > 0 ? score / selection[i] : 0) : score;
  });
  const reachable = demand.map((_, i) => demandWeights.offsets[i + 1] - demandWeights.offsets[i]);
  return {ratios, accessibility, reachable};
}

/** Double-precision Huff reference. */
export function computeHuffOracle(
  attractiveness: number[],
  demand: number[],
  alpha: number,
  demandWeights: CPUWeights,
  facilityWeights: CPUWeights
) {
  const attraction = attractiveness.map(value => (value > 0 ? value ** alpha : 0));
  const probabilities = new Array(demandWeights.neighbors.length).fill(0);
  const tradeArea: number[] = [];
  const tradeAreaProbability: number[] = [];
  const denominators: number[] = [];
  demand.forEach((_, i) => {
    let total = 0;
    for (let s = demandWeights.offsets[i]; s < demandWeights.offsets[i + 1]; s++) {
      total += attraction[demandWeights.neighbors[s]] * demandWeights.weights[s];
    }
    denominators.push(total);
    let best = 0;
    let bestFacility = 0xffffffff;
    for (let s = demandWeights.offsets[i]; s < demandWeights.offsets[i + 1]; s++) {
      const share = attraction[demandWeights.neighbors[s]] * demandWeights.weights[s];
      probabilities[s] = total > 0 ? share / total : 0;
      if (share > best) {
        best = share;
        bestFacility = demandWeights.neighbors[s];
      }
    }
    tradeArea.push(total > 0 ? bestFacility : 0xffffffff);
    tradeAreaProbability.push(total > 0 ? best / total : 0);
  });
  const expectedDemand = attractiveness.map((_, j) => {
    let captured = 0;
    for (let s = facilityWeights.offsets[j]; s < facilityWeights.offsets[j + 1]; s++) {
      const i = facilityWeights.neighbors[s];
      if (denominators[i] > 0)
        captured += (demand[i] * facilityWeights.weights[s]) / denominators[i];
    }
    return attraction[j] * captured;
  });
  return {probabilities, tradeArea, tradeAreaProbability, expectedDemand};
}
