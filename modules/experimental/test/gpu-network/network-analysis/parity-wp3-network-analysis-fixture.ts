// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

import fixture0input from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-input.json?raw';
import fixture0reverse from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-reverse.json?raw';
import fixture0degree from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-degree.json?raw';
import fixture0inDegree from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-inDegree.json?raw';
import fixture0coreNumber from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-coreNumber.json?raw';
import fixture0pageRank from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-pageRank.json?raw';
import fixture0components from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-components.json?raw';
import fixture0zero from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-zero.json?raw';
import fixture0outgoing from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-outgoing.json?raw';
import fixture0clamped from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-clamped.json?raw';
import fixture0incoming from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-incoming.json?raw';
import fixture0empty from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-empty.json?raw';
import fixture0bounded from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-directed-bounded.json?raw';
import fixture1input from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-input.json?raw';
import fixture1reverse from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-reverse.json?raw';
import fixture1degree from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-degree.json?raw';
import fixture1inDegree from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-inDegree.json?raw';
import fixture1coreNumber from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-coreNumber.json?raw';
import fixture1pageRank from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-pageRank.json?raw';
import fixture1components from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-components.json?raw';
import fixture1zero from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-zero.json?raw';
import fixture1outgoing from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-outgoing.json?raw';
import fixture1clamped from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-clamped.json?raw';
import fixture1incoming from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-incoming.json?raw';
import fixture1empty from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-empty.json?raw';
import fixture1bounded from '../../oracles/fixtures/parity-wp3-network-analysis-synthetic-undirected-bounded.json?raw';
import fixture2input from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-input.json?raw';
import fixture2reverse from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-reverse.json?raw';
import fixture2degree from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-degree.json?raw';
import fixture2inDegree from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-inDegree.json?raw';
import fixture2coreNumber from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-coreNumber.json?raw';
import fixture2pageRank from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-pageRank.json?raw';
import fixture2components from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-components.json?raw';
import fixture2zero from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-zero.json?raw';
import fixture2outgoing from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-outgoing.json?raw';
import fixture2clamped from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-clamped.json?raw';
import fixture2incoming from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-incoming.json?raw';
import fixture2empty from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-empty.json?raw';
import fixture2bounded from '../../oracles/fixtures/parity-wp3-network-analysis-isolates-bounded.json?raw';
import fixture3input from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-input.json?raw';
import fixture3reverse from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-reverse.json?raw';
import fixture3degree from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-degree.json?raw';
import fixture3inDegree from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-inDegree.json?raw';
import fixture3coreNumber from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-coreNumber.json?raw';
import fixture3pageRank from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-pageRank.json?raw';
import fixture3components from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-components.json?raw';
import fixture3zero from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-zero.json?raw';
import fixture3outgoing from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-outgoing.json?raw';
import fixture3clamped from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-clamped.json?raw';
import fixture3incoming from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-incoming.json?raw';
import fixture3empty from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-empty.json?raw';
import fixture3bounded from '../../oracles/fixtures/parity-wp3-network-analysis-chicago-bounded.json?raw';

export type AnalysisInput = {
  case: string;
  reference: string;
  nodeCount: number;
  directed: boolean;
  offsets: number[];
  neighbors: number[];
  nodeIds: number[];
  edgeIds: number[];
  versions: Record<string, string>;
};
export type AnalysisMetric = {
  case: string;
  reference: string;
  metric: 'degree' | 'inDegree' | 'coreNumber' | 'pageRank' | 'components';
  values: number[];
  normalized: number[];
  extent: number[];
};
export type NeighborhoodCase = {
  case: string;
  reference: string;
  starts: number[];
  startCount: number;
  hops: number;
  maximumHops: number;
  incoming: boolean;
  nodeCapacity: number;
  edgeCapacity: number;
  hopDistances: number[];
  nodeMask: number[];
  edgeMask: number[];
  nodeIds: number[];
  edgeIds: number[];
  nodeCount: number;
  edgeCount: number;
  nodeOverflow: number;
  edgeOverflow: number;
};
export type AnalysisFixture = {
  input: AnalysisInput;
  reverse: Pick<AnalysisInput, 'offsets' | 'neighbors' | 'edgeIds'>;
  metrics: AnalysisMetric[];
  neighborhoods: NeighborhoodCase[];
};
// Preserve exact input/reference literals through Vite's JSON transforms.
export const analysisFixtures: AnalysisFixture[] = [
  {
    input: JSON.parse(fixture0input),
    reverse: JSON.parse(fixture0reverse),
    metrics: [
      fixture0degree,
      fixture0inDegree,
      fixture0coreNumber,
      fixture0pageRank,
      fixture0components
    ].map(value => JSON.parse(value)),
    neighborhoods: [
      fixture0zero,
      fixture0outgoing,
      fixture0clamped,
      fixture0incoming,
      fixture0empty,
      fixture0bounded
    ].map(value => JSON.parse(value))
  },
  {
    input: JSON.parse(fixture1input),
    reverse: JSON.parse(fixture1reverse),
    metrics: [
      fixture1degree,
      fixture1inDegree,
      fixture1coreNumber,
      fixture1pageRank,
      fixture1components
    ].map(value => JSON.parse(value)),
    neighborhoods: [
      fixture1zero,
      fixture1outgoing,
      fixture1clamped,
      fixture1incoming,
      fixture1empty,
      fixture1bounded
    ].map(value => JSON.parse(value))
  },
  {
    input: JSON.parse(fixture2input),
    reverse: JSON.parse(fixture2reverse),
    metrics: [
      fixture2degree,
      fixture2inDegree,
      fixture2coreNumber,
      fixture2pageRank,
      fixture2components
    ].map(value => JSON.parse(value)),
    neighborhoods: [
      fixture2zero,
      fixture2outgoing,
      fixture2clamped,
      fixture2incoming,
      fixture2empty,
      fixture2bounded
    ].map(value => JSON.parse(value))
  },
  {
    input: JSON.parse(fixture3input),
    reverse: JSON.parse(fixture3reverse),
    metrics: [
      fixture3degree,
      fixture3inDegree,
      fixture3coreNumber,
      fixture3pageRank,
      fixture3components
    ].map(value => JSON.parse(value)),
    neighborhoods: [
      fixture3zero,
      fixture3outgoing,
      fixture3clamped,
      fixture3incoming,
      fixture3empty,
      fixture3bounded
    ].map(value => JSON.parse(value))
  }
];

import showcaseNeighborhoodsJson from '../../oracles/fixtures/parity-wp3-neighborhood-showcase.json';
export const showcaseNeighborhoods =
  decodeOracleFixture<NeighborhoodCase[]>(showcaseNeighborhoodsJson);
