// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from '../../oracles/fixtures/statistics-max-p.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

/** One instance of `statistics-max-p.py`: spopt `MaxPHeuristic` on a lattice or Delaunay graph. */
export type MaxPFixtureInstance = {
  name: string;
  rows: number;
  /** CSR of the symmetric contiguity weights. */
  offsets: number[];
  neighbors: number[];
  /** One standardized attribute column. */
  values: number[];
  extensive: number[];
  threshold: number;
  /** `floor(sum(extensive) / threshold)`: no partition has more regions. */
  upperBound: number;
  /** Region count of each of the five spopt runs. */
  spoptRegionCounts: number[];
  spoptBestRegionCount: number;
  /** Within-region sum of squares of the best spopt run, after annealing. */
  spoptBestObjective: number;
  spoptBestLabels: number[];
};

/** The decoded `statistics-max-p.json` fixture (spopt reference). */
export const MAX_P_FIXTURE = decodeOracleFixture<{
  versions: Record<string, string>;
  instances: MaxPFixtureInstance[];
}>(fixtureJson);
