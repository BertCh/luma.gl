// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import showcase12 from '../../oracles/fixtures/parity-wp3-network-k-function-showcase-12.json';
import showcase24 from '../../oracles/fixtures/parity-wp3-network-k-function-showcase-24.json';
import showcase48 from '../../oracles/fixtures/parity-wp3-network-k-function-showcase-48.json';
import showcase96 from '../../oracles/fixtures/parity-wp3-network-k-function-showcase-96.json';
import boundary from '../../oracles/fixtures/parity-wp3-network-k-function-boundary.json';
import chicago from '../../oracles/fixtures/parity-wp3-network-k-function-chicago.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

export type KFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  positions: number[][];
  points: number[][];
  offsets: number[];
  neighbors: number[];
  weights: number[];
  networkLength: number;
  exactNetworkLength: number;
  intensity: number;
  maximumDistance: number;
  radii: number[];
  counts: number[];
  lowerCounts: number[];
  upperCounts: number[];
  ambiguousPairs: number[][][];
  distanceErrorBound: number;
  maximumHops: number;
  observed: number[];
  simulations: number[][];
  gpuSimulationCount: number;
  gpuSeeds: number[];
  tolerance: {absolute: number; relative: number};
  distributionAlpha: number;
};
export const kFixtures = [boundary, chicago].map(fixture => decodeOracleFixture<KFixture>(fixture));

export const showcaseKFixtures = [showcase12, showcase24, showcase48, showcase96].map(fixture =>
  decodeOracleFixture<KFixture>(fixture)
);

/** Two-sample empirical CDF distance; includes both sides of every jump (ties included). */
export function getDistributionDistance(
  first: readonly number[],
  second: readonly number[]
): number {
  const firstSorted = [...first].sort((left, right) => left - right);
  const secondSorted = [...second].sort((left, right) => left - right);
  let firstIndex = 0;
  let secondIndex = 0;
  let distance = 0;
  for (const threshold of [...new Set([...firstSorted, ...secondSorted])].sort(
    (left, right) => left - right
  )) {
    while (firstIndex < firstSorted.length && firstSorted[firstIndex] <= threshold) firstIndex++;
    while (secondIndex < secondSorted.length && secondSorted[secondIndex] <= threshold)
      secondIndex++;
    distance = Math.max(
      distance,
      Math.abs(firstIndex / first.length - secondIndex / second.length)
    );
  }
  return distance;
}
