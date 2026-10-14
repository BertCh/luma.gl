// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createNonstationarityScene} from './geographically-weighted-regression-nonstationarity-oracle';
import type {NonstationarityScene} from './geographically-weighted-regression-nonstationarity-oracle';

export {createNonstationarityScene};

/** Origin and degrees per scene unit of {@link createGeographicScene}. */
export const GEOGRAPHIC_SCENE = {
  originLongitude: 10,
  originLatitude: 45,
  degreesPerUnit: 0.1
} as const;

/**
 * The scene with `(x, y)` mapped to `(longitude, latitude)` degrees around 10E, 45N, 0.1 degree per
 * scene unit (about 8 km east and 11 km north), so the 10 x 10 domain spans one degree. Positions
 * are rounded to float32 exactly like the planar scene.
 */
export function createGeographicScene(base: NonstationarityScene): NonstationarityScene {
  const positions = new Float32Array(base.positions.length);
  for (let row = 0; row < base.positions.length / 2; row++) {
    positions[2 * row] =
      GEOGRAPHIC_SCENE.originLongitude + GEOGRAPHIC_SCENE.degreesPerUnit * base.positions[2 * row];
    positions[2 * row + 1] =
      GEOGRAPHIC_SCENE.originLatitude +
      GEOGRAPHIC_SCENE.degreesPerUnit * base.positions[2 * row + 1];
  }
  return {...base, positions};
}

/**
 * A `side x side` jittered grid with one predictor `x` and one rare dummy `d` that is 1 only for
 * the `dummyCount` rows nearest the grid centre, response `1 + 2 x + 3 d + noise`. A local fit
 * whose neighbourhood holds no dummy row has a constant `d` column and is singular, so small
 * bandwidths leave singular rows far from the centre while the centre rows stay fittable. Rows
 * farther than `noisyRadius` from the centre (the rows that go singular first) get an extra noise
 * of amplitude `noisyAmplitude`, so excluding the singular rows lowers the residual sum of squares.
 */
export function createSingularRowScene(
  side: number,
  seed: number,
  dummyCount: number,
  noisyRadius = Number.POSITIVE_INFINITY,
  noisyAmplitude = 0
): NonstationarityScene {
  let state = seed >>> 0 || 1;
  const random = () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
  const rows = side * side;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  const centre = (side / 2) * (10 / side);
  const distances: {row: number; distance: number}[] = [];
  for (let row = 0; row < rows; row++) {
    const x = ((row % side) + 0.2 + 0.6 * random()) * (10 / side);
    const y = (Math.floor(row / side) + 0.2 + 0.6 * random()) * (10 / side);
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    predictors[2 * row] = random() * 4 - 2;
    distances.push({row, distance: Math.hypot(Math.fround(x) - centre, Math.fround(y) - centre)});
  }
  distances.sort((left, right) => left.distance - right.distance);
  for (let index = 0; index < dummyCount; index++) {
    predictors[2 * distances[index].row + 1] = 1;
  }
  const distanceOf = new Map(distances.map(entry => [entry.row, entry.distance]));
  for (let row = 0; row < rows; row++) {
    const extra = distanceOf.get(row)! > noisyRadius ? noisyAmplitude * (random() - 0.5) : 0;
    response[row] =
      1 + 2 * predictors[2 * row] + 3 * predictors[2 * row + 1] + 0.4 * (random() - 0.5) + extra;
  }
  return {positions, predictors, predictorCount: 2, response};
}
