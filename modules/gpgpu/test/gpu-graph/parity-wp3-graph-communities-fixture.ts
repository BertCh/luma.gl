// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import undirected from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-communities-undirected.json';
import directed from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-communities-directed.json';
import cliques from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-communities-cliques.json';
import openflights from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-communities-openflights.json';
import {decodeOracleFixture} from '../../../experimental/test/oracles/oracle-fixture';

export type CommunityFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  components: number[];
  stability: number[][];
  edges: [number, number, number][];
  directed: boolean;
  exactPartition: boolean;
  scoring: {
    name: string;
    resolution: number;
    labels: number[];
    score: number;
    contributions: number[];
  }[];
  runs: {seed: number; labels: number[]; permutation: number[]; score: number}[];
  tolerance: {absolute: number; relative: number};
};
export const communityFixtures = [undirected, directed, cliques, openflights].map(value =>
  decodeOracleFixture<CommunityFixture>(value)
);

/** Published Newman community-sum definition, independently checked against every NX fixture. */
export function evaluatePartition(
  fixture: Pick<CommunityFixture, 'edges' | 'nodeCount' | 'directed'>,
  labels: readonly number[],
  resolution: number,
  weighted = true
): number[] {
  const internal = new Float64Array(fixture.nodeCount);
  const incoming = new Float64Array(fixture.nodeCount);
  const outgoing = new Float64Array(fixture.nodeCount);
  let total = 0;
  for (const [source, target, storedWeight] of fixture.edges) {
    const weight = weighted ? storedWeight : 1;
    const sourceLabel = labels[source];
    const targetLabel = labels[target];
    total += weight;
    outgoing[sourceLabel] += weight;
    incoming[targetLabel] += weight;
    if (!fixture.directed) {
      outgoing[targetLabel] += weight;
      incoming[sourceLabel] += weight;
    }
    if (sourceLabel === targetLabel) internal[sourceLabel] += weight;
  }
  return Array.from(
    internal,
    (weight, community) =>
      weight / total -
      (resolution * outgoing[community] * incoming[community]) /
        ((fixture.directed ? 1 : 4) * total ** 2)
  );
}

export function canonicalizePartition(labels: readonly number[]): number[] {
  const minimum = new Map<number, number>();
  labels.forEach((label, node) => minimum.set(label, Math.min(node, minimum.get(label) ?? node)));
  return labels.map(label => minimum.get(label)!);
}

/** Independent fixed-point check of the documented vote rule, not an LPA implementation. */
export function getUnstableVertices(
  fixture: CommunityFixture,
  labels: readonly number[],
  permutation: readonly number[]
): number[] {
  const neighbors = Array.from({length: fixture.nodeCount}, () => [] as number[]);
  for (const [source, target] of fixture.edges) {
    if (source !== target) {
      neighbors[source].push(target);
      neighbors[target].push(source);
    }
  }
  return neighbors.map((adjacent, node) => {
    const votes = new Map<number, number>();
    for (const member of [node, ...adjacent])
      votes.set(labels[member], (votes.get(labels[member]) ?? 0) + 1);
    let winner = labels[node];
    for (const [label, support] of votes) {
      if (
        support > votes.get(winner)! ||
        (support === votes.get(winner) && permutation[label] < permutation[winner])
      )
        winner = label;
    }
    return Number(winner !== labels[node]);
  });
}

/** Hubert-Arabie adjusted Rand index; checked against sklearn on all reference runs. */
export function getAdjustedRandIndex(first: readonly number[], second: readonly number[]): number {
  const rows = new Map<number, number>();
  const columns = new Map<number, number>();
  const cells = new Map<string, number>();
  for (let index = 0; index < first.length; index++) {
    rows.set(first[index], (rows.get(first[index]) ?? 0) + 1);
    columns.set(second[index], (columns.get(second[index]) ?? 0) + 1);
    const cell = `${first[index]}:${second[index]}`;
    cells.set(cell, (cells.get(cell) ?? 0) + 1);
  }
  const chooseTwo = (value: number) => (value * (value - 1)) / 2;
  const sumPairs = (counts: Iterable<number>) =>
    Array.from(counts).reduce((sum, value) => sum + chooseTwo(value), 0);
  const rowPairs = sumPairs(rows.values());
  const columnPairs = sumPairs(columns.values());
  const expected = (rowPairs * columnPairs) / chooseTwo(first.length);
  const maximum = (rowPairs + columnPairs) / 2;
  return maximum === expected ? 1 : (sumPairs(cells.values()) - expected) / (maximum - expected);
}
