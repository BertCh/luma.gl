// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import syntheticInput from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-input.json?raw';
import syntheticPositions from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-positions.json?raw';
import syntheticAll from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-all.json?raw';
import syntheticAll0 from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-all-0.json?raw';
import syntheticBanned from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-banned.json?raw';
import syntheticBanned0 from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-banned-0.json?raw';
import syntheticTurnCosts from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-turn-costs.json?raw';
import syntheticTurnCosts0 from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-turn-costs-0.json?raw';
import syntheticNoUturn from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-no-uturn.json?raw';
import syntheticNoUturn0 from '../../oracles/fixtures/parity-wp3-line-graph-synthetic-no-uturn-0.json?raw';
import chicagoInput from '../../oracles/fixtures/parity-wp3-line-graph-chicago-input.json?raw';
import chicagoPositions from '../../oracles/fixtures/parity-wp3-line-graph-chicago-positions.json?raw';
import chicagoAll from '../../oracles/fixtures/parity-wp3-line-graph-chicago-all.json?raw';
import chicagoAll0 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-all-0.json?raw';
import chicagoAll1 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-all-1.json?raw';
import chicagoAll2 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-all-2.json?raw';
import chicagoAll3 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-all-3.json?raw';
import chicagoBanned from '../../oracles/fixtures/parity-wp3-line-graph-chicago-banned.json?raw';
import chicagoBanned0 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-banned-0.json?raw';
import chicagoBanned1 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-banned-1.json?raw';
import chicagoBanned2 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-banned-2.json?raw';
import chicagoBanned3 from '../../oracles/fixtures/parity-wp3-line-graph-chicago-banned-3.json?raw';

export type LineGraphInput = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  positions: number[];
};
export type LineGraphCase = {
  case: string;
  reference: string;
  parameters: number[];
  bans: number[];
  offsets: number[];
  count: number;
  costBound: number;
  exactCosts: boolean;
  neighbors: number[];
  weights: number[];
};
export const lineGraphFixtures: {input: LineGraphInput; cases: LineGraphCase[]}[] = [
  {
    input: syntheticInput,
    positions: syntheticPositions,
    cases: [
      {metadata: syntheticAll, shards: [syntheticAll0]},
      {metadata: syntheticBanned, shards: [syntheticBanned0]},
      {metadata: syntheticTurnCosts, shards: [syntheticTurnCosts0]},
      {metadata: syntheticNoUturn, shards: [syntheticNoUturn0]}
    ]
  },
  {
    input: chicagoInput,
    positions: chicagoPositions,
    cases: [
      {metadata: chicagoAll, shards: [chicagoAll0, chicagoAll1, chicagoAll2, chicagoAll3]},
      {
        metadata: chicagoBanned,
        shards: [chicagoBanned0, chicagoBanned1, chicagoBanned2, chicagoBanned3]
      }
    ]
  }
].map(group => ({
  input: {...JSON.parse(group.input), ...JSON.parse(group.positions)},
  cases: group.cases.map(fixture => {
    const shards: {neighbors: number[]; weights: number[]}[] = fixture.shards.map(value =>
      JSON.parse(value)
    );
    return {
      ...JSON.parse(fixture.metadata),
      neighbors: shards.flatMap(shard => shard.neighbors),
      weights: shards.flatMap(shard => shard.weights)
    };
  })
}));
