// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NONE, type AccessibilityCSR} from '../network-accessibility/network-accessibility-oracle';

/** Parameters mirrored from `GPUMapMatchingParameters`, with the defaults resolved. */
export type MapMatchingOracleParameters = {
  sigma: number;
  beta: number;
  searchRadius: number;
  routeFactor: number;
  routeSlack: number;
};

export type MapMatchingOracleResult = {
  /** Chosen candidate edge per point or NONE. */
  edges: number[];
  fractions: number[];
  breaks: number[];
  trackLogLikelihoods: number[];
};

type Candidate = {edge: number; fraction: number; distance: number};

/**
 * Float64 Newson-Krumm Viterbi with exact bounded Dijkstra, the model `GPUMapMatching` approximates
 * only through its route node budget and f32 arithmetic.
 */
export function matchOracle(
  points: Float32Array,
  trackOffsets: readonly number[],
  nodePositions: Float32Array,
  csr: AccessibilityCSR,
  parameters: MapMatchingOracleParameters,
  candidateCount: number,
  cellSize: number
): MapMatchingOracleResult {
  const pointCount = points.length / 2;
  const radius = Math.min(parameters.searchRadius, 2 * cellSize);
  const node = (index: number) => [nodePositions[index * 2], nodePositions[index * 2 + 1]];
  const lengths = Array.from(csr.neighbors, (target, edge) => {
    const [ax, ay] = node(csr.sources[edge]);
    const [bx, by] = node(target);
    return Math.hypot(bx - ax, by - ay);
  });
  const candidates: Candidate[][] = [];
  for (let point = 0; point < pointCount; point++) {
    const px = points[point * 2];
    const py = points[point * 2 + 1];
    const found: Candidate[] = [];
    for (let edge = 0; edge < csr.neighbors.length; edge++) {
      const [ax, ay] = node(csr.sources[edge]);
      const [bx, by] = node(csr.neighbors[edge]);
      const dx = bx - ax;
      const dy = by - ay;
      const denominator = dx * dx + dy * dy;
      const fraction =
        denominator > 0
          ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / denominator))
          : 0;
      const distance = Math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy));
      if (distance <= radius) {
        found.push({edge, fraction, distance});
      }
    }
    found.sort((a, b) => a.distance - b.distance || a.edge - b.edge);
    candidates.push(found.slice(0, candidateCount));
  }
  const dijkstra = (start: number, limit: number): Map<number, number> => {
    const distances = new Map<number, number>([[start, 0]]);
    const done = new Set<number>();
    for (;;) {
      let best = -1;
      for (const [candidate, distance] of distances) {
        if (!done.has(candidate) && (best < 0 || distance < distances.get(best)!)) {
          best = candidate;
        }
      }
      if (best < 0) {
        return distances;
      }
      done.add(best);
      for (let edge = csr.offsets[best]; edge < csr.offsets[best + 1]; edge++) {
        const next = distances.get(best)! + lengths[edge];
        const target = csr.neighbors[edge];
        if (next <= limit && next < (distances.get(target) ?? Infinity)) {
          distances.set(target, next);
        }
      }
    }
  };
  const emissionConstant = -Math.log(parameters.sigma) - 0.5 * Math.log(2 * Math.PI);
  const result: MapMatchingOracleResult = {
    edges: new Array(pointCount).fill(NONE),
    fractions: new Array(pointCount).fill(-1),
    breaks: new Array(pointCount).fill(0),
    trackLogLikelihoods: []
  };
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const first = trackOffsets[track];
    const last = trackOffsets[track + 1];
    const scores: number[][] = [];
    const back: number[][] = [];
    for (let point = first; point < last; point++) {
      const here = candidates[point];
      const score = here.map(() => -Infinity);
      const pointer = here.map(() => NONE);
      let isReachable = false;
      if (point > first) {
        const straight = Math.hypot(
          points[point * 2] - points[point * 2 - 2],
          points[point * 2 + 1] - points[point * 2 - 1]
        );
        const bound = straight * parameters.routeFactor + parameters.routeSlack;
        candidates[point - 1].forEach((from, fromSlot) => {
          const fromScore = scores[point - 1 - first][fromSlot];
          if (!(fromScore > -Infinity)) {
            return;
          }
          const tail = (1 - from.fraction) * lengths[from.edge];
          const remaining = bound - tail;
          const reach = remaining >= 0 ? dijkstra(csr.neighbors[from.edge], remaining) : undefined;
          here.forEach((to, toSlot) => {
            let route = -1;
            if (to.edge === from.edge && to.fraction >= from.fraction) {
              route = (to.fraction - from.fraction) * lengths[from.edge];
            } else if (reach?.has(csr.sources[to.edge])) {
              route = tail + reach.get(csr.sources[to.edge])! + to.fraction * lengths[to.edge];
            }
            if (route < 0) {
              return;
            }
            const total =
              fromScore - Math.log(parameters.beta) - Math.abs(straight - route) / parameters.beta;
            if (total > score[toSlot]) {
              score[toSlot] = total;
              pointer[toSlot] = fromSlot;
              isReachable = true;
            }
          });
        });
      }
      here.forEach((candidate, slot) => {
        const emission = emissionConstant - 0.5 * (candidate.distance / parameters.sigma) ** 2;
        if (isReachable) {
          score[slot] += emission;
        } else {
          score[slot] = emission;
          pointer[slot] = NONE;
        }
      });
      scores.push(score);
      back.push(pointer);
    }
    let logLikelihood = 0;
    let current = NONE;
    let isPicking = true;
    for (let point = last - 1; point >= first; point--) {
      const score = scores[point - first];
      if (isPicking) {
        let bestScore = -Infinity;
        current = NONE;
        score.forEach((value, slot) => {
          if (value > bestScore) {
            bestScore = value;
            current = slot;
          }
        });
        if (current !== NONE) {
          logLikelihood += bestScore;
        }
      }
      let isBreak = current === NONE;
      isPicking = true;
      if (current !== NONE) {
        const candidate = candidates[point][current];
        result.edges[point] = candidate.edge;
        result.fractions[point] = candidate.fraction;
        const previous = back[point - first][current];
        if (previous !== NONE) {
          current = previous;
          isPicking = false;
        } else {
          isBreak = true;
        }
      }
      result.breaks[point] = isBreak && point > first ? 1 : 0;
    }
    result.trackLogLikelihoods.push(logLikelihood);
  }
  return result;
}
