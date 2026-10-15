// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import undirected from '../../../experimental/test/oracles/fixtures/parity-wp3-modularity-optimization-undirected.json';
import directed from '../../../experimental/test/oracles/fixtures/parity-wp3-modularity-optimization-directed.json';
import cliques from '../../../experimental/test/oracles/fixtures/parity-wp3-modularity-optimization-cliques.json';
import openflights from '../../../experimental/test/oracles/fixtures/parity-wp3-modularity-optimization-openflights.json';
import {decodeOracleFixture} from '../../../experimental/test/oracles/oracle-fixture';
import {evaluatePartition, type CommunityFixture} from './parity-wp3-graph-communities-fixture';

export type OptimizationFixture = Pick<
  CommunityFixture,
  | 'case'
  | 'reference'
  | 'versions'
  | 'nodeCount'
  | 'edges'
  | 'directed'
  | 'components'
  | 'tolerance'
> & {
  runs: {
    seed: number;
    resolution: number;
    initial: number[];
    initialScore: number;
    initialMaximumGain: number;
    permutation: number[];
    labels: number[];
    score: number;
    greedyLabels: number[];
    greedyScore: number;
    iterations: number;
    start: string;
  }[];
};
export const optimizationFixtures = [undirected, directed, cliques, openflights].map(value =>
  decodeOracleFixture<OptimizationFixture>(value)
);

export function scorePartition(
  fixture: OptimizationFixture,
  labels: readonly number[],
  resolution: number
): number {
  return evaluatePartition(fixture, labels, resolution).reduce((total, value) => total + value, 0);
}

/** Exhaustive objective certificate on small graphs, not an implementation of local moving. */
export function maximumMoveGain(
  fixture: OptimizationFixture,
  labels: readonly number[],
  resolution: number
): number {
  const baseline = scorePartition(fixture, labels, resolution);
  let maximum = 0;
  for (let vertex = 0; vertex < fixture.nodeCount; vertex++) {
    for (let label = 0; label < fixture.nodeCount; label++) {
      const candidate = [...labels];
      candidate[vertex] = label;
      maximum = Math.max(maximum, scorePartition(fixture, candidate, resolution) - baseline);
    }
  }
  return maximum;
}
