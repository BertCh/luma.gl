// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from '../../oracles/fixtures/statistics-local-join-counts-and-bivariate.json';

/** A per-row value where `null` is esda's NaN (a row that is not tested). */
type NullableRows = (number | null)[];

/**
 * esda 2.10 / libpysal 4.15 local Geary, bivariate Moran, local join counts and Lee's L, generated
 * by `test/oracles/statistics-local-join-counts-and-bivariate.py`: 60 rows, 5 nearest neighbors
 * (row standardized), seed 12345, 9999 permutations. NaN stays `null` (the specs test for it).
 */
export type NewStatisticsReference = {
  n: number;
  k: number;
  x: number[];
  y: number[];
  bx: number[];
  bz: number[];
  bv: number[];
  neighbors: number[][];
  permutations: number;
  gearyLocal: number[];
  gearyPSim: number[];
  gearyInjectedP: NullableRows;
  gearyLabels: NullableRows;
  bivariateMoranI: number[];
  bivariateMoranQuadrants: number[];
  bivariateMoranPSim: number[];
  joinCount: number[];
  joinCountPSim: NullableRows;
  joinCountBJC: number[];
  joinCountBJCPSim: NullableRows;
  joinCountCLC: number[];
  joinCountCLCPSim: NullableRows;
  joinCountBJCConsistentPSim: NullableRows;
  leeGlobal: number;
  leeGlobalSmoothingX: number;
  leeGlobalSmoothingY: number;
  leeGlobalPSim: number;
  leeLocal: number[];
  leeLocalPSim: number[];
};

const fixture = fixtureJson as unknown as {
  generator: string;
  versions: Record<string, string>;
  reference: NewStatisticsReference;
};

/** The esda reference values. */
export const NEW_STATISTICS_REFERENCE: NewStatisticsReference = fixture.reference;
