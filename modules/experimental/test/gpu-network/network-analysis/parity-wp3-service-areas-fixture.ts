// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import chicagoInput from '../../oracles/fixtures/parity-wp3-service-areas-chicago-input.json';
import chicagoFive from '../../oracles/fixtures/parity-wp3-service-areas-chicago-five-minutes.json';
import chicagoFifteen from '../../oracles/fixtures/parity-wp3-service-areas-chicago-fifteen-minutes.json';
import chicagoUnlimited from '../../oracles/fixtures/parity-wp3-service-areas-chicago-unlimited.json';
import syntheticInput from '../../oracles/fixtures/parity-wp3-service-areas-synthetic-input.json';
import syntheticZero from '../../oracles/fixtures/parity-wp3-service-areas-synthetic-zero.json';
import syntheticBoundary from '../../oracles/fixtures/parity-wp3-service-areas-synthetic-boundary.json';
import syntheticUnlimited from '../../oracles/fixtures/parity-wp3-service-areas-synthetic-unlimited.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

export type ServiceInput = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  facilities: number[];
  initialCosts: number[];
};
export type ServiceCase = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  budget: number;
  costs: number[];
  assignments: number[];
  allowedPredecessors: number[][];
  counts: number[];
  sums: number[];
  tolerance: {absolute: number; relative: number};
};
export const serviceFixtures = [
  {
    input: decodeOracleFixture<ServiceInput>(syntheticInput),
    cases: [syntheticZero, syntheticBoundary, syntheticUnlimited].map(value =>
      decodeOracleFixture<ServiceCase>(value)
    )
  },
  {
    input: decodeOracleFixture<ServiceInput>(chicagoInput),
    cases: [chicagoFive, chicagoFifteen, chicagoUnlimited].map(value =>
      decodeOracleFixture<ServiceCase>(value)
    )
  }
];

/** Check all predecessor links against independent allowed sets, then all roots in O(V). */
export function getPredecessorViolations(
  input: ServiceInput,
  fixture: ServiceCase,
  predecessors: readonly number[]
): number[] {
  const none = 0xffffffff;
  const violations = new Array<number>(input.nodeCount).fill(0);
  const states = new Uint8Array(input.nodeCount);
  const roots = new Array<number>(input.nodeCount).fill(none);
  for (let node = 0; node < input.nodeCount; node++) {
    const assignment = fixture.assignments[node];
    const predecessor = predecessors[node];
    if (assignment === none) {
      violations[node] = Number(predecessor !== none);
    } else if (predecessor === none) {
      violations[node] = Number(input.facilities[assignment] !== node);
    } else {
      violations[node] = Number(
        !fixture.allowedPredecessors[node].includes(predecessor) ||
          fixture.assignments[predecessor] !== assignment
      );
    }
  }
  for (let start = 0; start < input.nodeCount; start++) {
    if (states[start] || fixture.assignments[start] === none) continue;
    const path: number[] = [];
    let node = start;
    while (node < input.nodeCount && !states[node]) {
      states[node] = 1;
      path.push(node);
      if (predecessors[node] === none) break;
      node = predecessors[node];
    }
    const root =
      node >= input.nodeCount
        ? none
        : predecessors[node] === none
          ? node
          : states[node] === 2
            ? roots[node]
            : none;
    for (const member of path) {
      roots[member] = root;
      states[member] = 2;
      violations[member] ||= Number(root !== input.facilities[fixture.assignments[member]]);
    }
  }
  return violations;
}
