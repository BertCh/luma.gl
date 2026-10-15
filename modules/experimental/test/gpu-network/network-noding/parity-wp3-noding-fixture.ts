// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import largeSnap from '../../oracles/fixtures/parity-wp3-noding-showcase-large-snap.json?raw';
import overlap from '../../oracles/fixtures/parity-wp3-noding-cross-overlap.json?raw';
import snap from '../../oracles/fixtures/parity-wp3-noding-snap-cells.json?raw';
import loop from '../../oracles/fixtures/parity-wp3-noding-chicago-loop.json?raw';
import west from '../../oracles/fixtures/parity-wp3-noding-chicago-west.json?raw';

export type NodingFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  tolerance: number;
  positions: number[];
  lineOffsets: number[];
  intersectionCapacity: number;
  coordinateBound: number;
  piecePositions: number[];
  pieceOffsets: number[];
  sourceIds: number[];
  nodePositions: number[];
  fromNodes: number[];
  toNodes: number[];
  lengths: number[];
  lengthBounds: number[];
  offsets: number[];
  neighbors: number[];
  edgeIds: number[];
  weights: number[];
  sourceLengths: number[];
  geosUnionLength: number;
  geosHausdorff: number;
  boundingBoxPairCount: number;
};
export const nodingFixtures: NodingFixture[] = [overlap, snap, loop, west, largeSnap].map(value =>
  JSON.parse(value)
);

type NodingTopology = Pick<
  NodingFixture,
  'nodePositions' | 'fromNodes' | 'toNodes' | 'offsets' | 'neighbors' | 'edgeIds' | 'weights'
>;

/**
 * Canonical node labels are first occurrence in the source-ordered piece endpoints.
 * This is a bijective relabeling, not geometry snapping or topology reconstruction.
 * Zero-tolerance GPU labels sort unsigned f32 bits; GEOS fixture labels sort numbers.
 * Keep every CSR entry in its original row order, including duplicate occurrences.
 */
export function canonicalizeNoding(topology: NodingTopology): NodingTopology {
  const nodeCount = topology.nodePositions.length / 2;
  const nodeOrder: number[] = [];
  const canonicalIds = new Array<number>(nodeCount).fill(-1);
  for (let edge = 0; edge < topology.fromNodes.length; edge++) {
    for (const node of [topology.fromNodes[edge], topology.toNodes[edge]]) {
      if (!Number.isInteger(node) || node < 0 || node >= nodeCount) {
        throw new Error('Invalid live endpoint node ID');
      }
      if (canonicalIds[node] === -1) {
        canonicalIds[node] = nodeOrder.length;
        nodeOrder.push(node);
      }
    }
  }
  if (nodeOrder.length !== nodeCount) throw new Error('Node without a piece endpoint');
  const entries = nodeOrder.flatMap(node =>
    Array.from(
      {length: topology.offsets[node + 1] - topology.offsets[node]},
      (_, entry) => topology.offsets[node] + entry
    )
  );
  const offsets = [0];
  for (const node of nodeOrder) {
    offsets.push(offsets[offsets.length - 1] + topology.offsets[node + 1] - topology.offsets[node]);
  }
  return {
    nodePositions: nodeOrder.flatMap(node => topology.nodePositions.slice(2 * node, 2 * node + 2)),
    fromNodes: topology.fromNodes.map(node => canonicalIds[node]),
    toNodes: topology.toNodes.map(node => canonicalIds[node]),
    offsets,
    neighbors: entries.map(entry => canonicalIds[topology.neighbors[entry]] ?? -1),
    edgeIds: entries.map(entry => topology.edgeIds[entry]),
    weights: entries.map(entry => topology.weights[entry])
  };
}
