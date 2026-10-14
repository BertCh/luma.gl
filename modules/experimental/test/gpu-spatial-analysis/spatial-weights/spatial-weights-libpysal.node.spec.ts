// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The TypeScript CPU oracles (`spatial-weights-oracle.ts`, `neighbor-search-oracle.ts`) reproduce the
 * libpysal weights construction of `test/oracles/statistics-spatial-weights.py`: neighbor sets exactly,
 * weights within the tolerances recorded (and derived) by the generator.
 */

import {expect, it} from 'vitest';
import {computeNeighborSearchOracle} from '../neighbor-search/neighbor-search-oracle';
import {
  computeContiguityOracle,
  computeLatticeOracle,
  computeTransformOracle,
  type OracleCSR
} from './spatial-weights-oracle';
import {
  LIBPYSAL_CONSTRUCTION_REFERENCE as REFERENCE,
  LIBPYSAL_WEIGHTS_TOLERANCES as TOLERANCES,
  type LibpysalRows
} from './spatial-weights-u4-oracle';

/** Flattens libpysal rows to the CSR layout of the oracles. */
function toCSR(rows: LibpysalRows): OracleCSR {
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  rows.neighbors.forEach((neighbors, row) => {
    csr.neighbors.push(...neighbors);
    csr.weights.push(...rows.weights[row]);
    csr.offsets.push(csr.neighbors.length);
  });
  return csr;
}

function expectSameNeighbors(actual: OracleCSR, expected: LibpysalRows, label: string): void {
  const csr = toCSR(expected);
  expect(actual.offsets, `${label} offsets`).toEqual(csr.offsets);
  expect(actual.neighbors, `${label} neighbors`).toEqual(csr.neighbors);
}

function expectWeightsClose(actual: number[], expected: number[], relative: number, label: string) {
  expect(actual.length, `${label} length`).toBe(expected.length);
  expected.forEach((weight, slot) => {
    expect(Math.abs(actual[slot] - weight), `${label} slot ${slot}`).toBeLessThanOrEqual(
      relative * Math.abs(weight)
    );
  });
}

it('contiguity oracle equals libpysal Queen and Rook on grid, rectangles and holes', () => {
  const scenes = REFERENCE.contiguity;
  for (const [name, scene] of Object.entries(scenes)) {
    for (const criterion of ['queen', 'rook'] as const) {
      const label = `${name} ${criterion}`;
      const actual = computeContiguityOracle(scene.polygons, criterion);
      expectSameNeighbors(actual, scene[criterion], label);
      expect(actual.weights, `${label} weights`).toEqual(toCSR(scene[criterion]).weights);
      expect(actual.neighbors.length, `${label} is not empty`).toBeGreaterThan(0);
    }
  }
  // The holes scene pins the documented differences between the criteria: the hole filler is a rook
  // neighbor of the holed polygon, the corner contact only a queen neighbor.
  expect(scenes.holes.queen.neighbors[2]).toEqual([0]);
  expect(scenes.holes.rook.neighbors[2]).toEqual([]);
});

it('lattice oracle equals libpysal lat2W for rook and queen', () => {
  const {width, height} = REFERENCE.lattice;
  for (const criterion of ['rook', 'queen'] as const) {
    const actual = computeLatticeOracle({width, height, criterion});
    expectSameNeighbors(actual, REFERENCE.lattice[criterion], `lat2W ${criterion}`);
    expect(actual.weights, `lat2W ${criterion} weights`).toEqual(
      toCSR(REFERENCE.lattice[criterion]).weights
    );
  }
});

it('neighbor search oracle equals libpysal KNN and DistanceBand, binary and row standardized', () => {
  const {knn, distanceBand, rowStandardized} = REFERENCE;
  const positions = Float32Array.from(knn.positions);
  const parameters = {bounds: knn.bounds};
  const distanceTolerance = TOLERANCES.distance.relative;

  const nearest = computeNeighborSearchOracle({mode: 'knn', k: knn.k, positions, parameters});
  expectSameNeighbors(nearest, knn, 'KNN');
  expect(nearest.weights, 'KNN weights').toEqual(toCSR(knn).weights);
  expectWeightsClose(nearest.distances, knn.distances.flat(), distanceTolerance, 'KNN distances');

  const band = computeNeighborSearchOracle({
    mode: 'radius',
    positions,
    parameters: {...parameters, radius: distanceBand.threshold}
  });
  expectSameNeighbors(band, distanceBand, 'DistanceBand');
  expect(band.weights, 'DistanceBand weights').toEqual(toCSR(distanceBand).weights);
  expectWeightsClose(
    band.distances,
    distanceBand.distances.flat(),
    distanceTolerance,
    'DistanceBand distances'
  );
  // The band scene keeps islands, which libpysal reports as empty rows.
  expect(distanceBand.neighbors.some(row => row.length === 0)).toBe(true);

  // w.transform = 'r' of the KNN weights, through the search oracle and the transform oracle.
  const rowTolerance = TOLERANCES.rowStandardizedWeight.relative;
  const expectedKnn = rowStandardized.knn;
  const standardized = computeNeighborSearchOracle({
    mode: 'knn',
    k: knn.k,
    positions,
    parameters: {...parameters, rowStandardize: true}
  });
  expectSameNeighbors(standardized, expectedKnn, 'KNN row');
  expectWeightsClose(standardized.weights, toCSR(expectedKnn).weights, rowTolerance, 'KNN row');
  expectWeightsClose(
    computeTransformOracle(nearest, 'row'),
    toCSR(expectedKnn).weights,
    rowTolerance,
    'KNN row transform'
  );

  // w.transform = 'r' of the Queen square grid.
  const grid = REFERENCE.contiguity.squareGrid;
  const queen = computeContiguityOracle(grid.polygons, 'queen');
  expectSameNeighbors(queen, rowStandardized.squareGridQueen, 'grid queen row');
  expectWeightsClose(
    computeTransformOracle(queen, 'row'),
    toCSR(rowStandardized.squareGridQueen).weights,
    rowTolerance,
    'grid queen row'
  );
});

it('the kNN scene has no distance ties at the cut', () => {
  // The library breaks ties by ascending id and libpysal by tree order, so the scene must be tie free.
  const {knn, positions} = {
    knn: REFERENCE.knn,
    positions: Float32Array.from(REFERENCE.knn.positions)
  };
  const rows = positions.length / 2;
  for (let row = 0; row < rows; row++) {
    const distances: number[] = [];
    for (let other = 0; other < rows; other++) {
      if (other !== row) {
        distances.push(
          Math.hypot(
            positions[row * 2] - positions[other * 2],
            positions[row * 2 + 1] - positions[other * 2 + 1]
          )
        );
      }
    }
    distances.sort((a, b) => a - b);
    expect(distances[knn.k] - distances[knn.k - 1], `row ${row} cut gap`).toBeGreaterThan(0.005);
  }
});
