// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A-5 fixture: five unit-square features on a 3 x 2 grid where feature 0 is a MULTIPART feature
// (two disjoint squares). Points -> per-feature counts -> queen contiguity W on the same features
// -> global Moran's I must chain on one feature table with no cast.

import {createSeededRandom} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';

type Square = [number, number];

/** Lower-left corners of the parts of each feature (feature 0 has two parts). */
export const CHAIN_FEATURE_PARTS: readonly (readonly Square[])[] = [
  [
    [0, 0],
    [2, 0]
  ],
  [[1, 0]],
  [[0, 1]],
  [[2, 1]],
  [[1, 1]]
];

export const CHAIN_FEATURE_COUNT = CHAIN_FEATURE_PARTS.length;

/** Four-level layout (one ring per part, closed implicitly). */
export function createChainPolygons() {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  for (const parts of CHAIN_FEATURE_PARTS) {
    for (const [x, y] of parts) {
      positions.push(x, y, x + 1, y, x + 1, y + 1, x, y + 1);
      ringOffsets.push(positions.length / 2);
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    featureOffsets: Uint32Array.from(featureOffsets)
  };
}

/** Points strictly inside the squares (never on an edge), with the feature each one falls in. */
export function createChainPoints(count = 400) {
  const random = createSeededRandom(5);
  const positions = new Float32Array(count * 2);
  const counts = new Array<number>(CHAIN_FEATURE_COUNT).fill(0);
  // Uneven intensity so Moran's I is not degenerate.
  const intensity = [6, 1, 3, 2, 5];
  const squares = CHAIN_FEATURE_PARTS.flatMap((parts, feature) =>
    parts.map(corner => ({corner, feature}))
  );
  const total = squares.reduce((sum, {feature}) => sum + intensity[feature], 0);
  for (let index = 0; index < count; index++) {
    let pick = random() * total;
    let square = squares[0];
    for (const candidate of squares) {
      pick -= intensity[candidate.feature];
      if (pick < 0) {
        square = candidate;
        break;
      }
    }
    positions[2 * index] = square.corner[0] + 0.05 + 0.9 * random();
    positions[2 * index + 1] = square.corner[1] + 0.05 + 0.9 * random();
    counts[square.feature]++;
  }
  return {positions, counts};
}

/** Queen contiguity of features: neighbors share at least one vertex of any part. */
export function getChainQueenNeighbors(): number[][] {
  const vertices = CHAIN_FEATURE_PARTS.map(
    parts =>
      new Set(
        parts.flatMap(([x, y]) => [
          `${x},${y}`,
          `${x + 1},${y}`,
          `${x + 1},${y + 1}`,
          `${x},${y + 1}`
        ])
      )
  );
  return vertices.map((own, feature) =>
    vertices.flatMap((other, neighbor) =>
      neighbor !== feature && [...own].some(vertex => other.has(vertex)) ? [neighbor] : []
    )
  );
}

/** Global Moran's I with binary weights (esda `Moran(y, w)` with `transform='B'`). */
export function getChainMoranI(values: readonly number[], neighbors: number[][]): number {
  const n = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  const deviations = values.map(value => value - mean);
  let cross = 0;
  let s0 = 0;
  for (let row = 0; row < n; row++) {
    for (const neighbor of neighbors[row]) {
      cross += deviations[row] * deviations[neighbor];
      s0 += 1;
    }
  }
  const squares = deviations.reduce((sum, value) => sum + value * value, 0);
  return (n / s0) * (cross / squares);
}
