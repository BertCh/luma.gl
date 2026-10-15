// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import syntheticInput from '../../oracles/fixtures/parity-wp3-routing-synthetic-input.json?raw';
import syntheticSingle from '../../oracles/fixtures/parity-wp3-routing-synthetic-single.json?raw';
import syntheticMulti from '../../oracles/fixtures/parity-wp3-routing-synthetic-multi.json?raw';
import syntheticEmpty from '../../oracles/fixtures/parity-wp3-routing-synthetic-empty.json?raw';
import forwardInput from '../../oracles/fixtures/parity-wp3-routing-chicago-forward-input.json?raw';
import forwardSingle from '../../oracles/fixtures/parity-wp3-routing-chicago-forward-single.json?raw';
import forwardMulti from '../../oracles/fixtures/parity-wp3-routing-chicago-forward-multi.json?raw';
import reverseInput from '../../oracles/fixtures/parity-wp3-routing-chicago-reverse-input.json?raw';
import reverseSingle from '../../oracles/fixtures/parity-wp3-routing-chicago-reverse-single.json?raw';
import reverseMulti from '../../oracles/fixtures/parity-wp3-routing-chicago-reverse-multi.json?raw';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

export type RoutingInput = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  offsets: number[];
  neighbors: number[];
  weights: number[];
};
export type RoutingCase = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  startNodes: number[];
  startCosts: number[];
  limit: number;
  costs: number[];
  predecessors: number[];
  breaks: number[];
  bands: number[];
  bandCounts: number[];
  targets: number[];
  paths: number[][];
  edgePaths: number[][];
  pathCosts: number[];
  maximumPathLength: number;
  tightAlternativeCount: number;
};
export const routingFixtures = [
  {input: syntheticInput, cases: [syntheticSingle, syntheticMulti, syntheticEmpty]},
  {input: forwardInput, cases: [forwardSingle, forwardMulti]},
  {input: reverseInput, cases: [reverseSingle, reverseMulti]}
].map(group => ({
  input: decodeOracleFixture<RoutingInput>(JSON.parse(group.input)),
  cases: group.cases.map(value => decodeOracleFixture<RoutingCase>(JSON.parse(value)))
}));

export function mapNodeId(input: RoutingInput, row: number): number {
  return 3 * (input.nodeCount - 1 - row) + 7;
}
export function mapEdgeId(input: RoutingInput, row: number): number {
  return 5 * (input.neighbors.length - 1 - row) + 11;
}
