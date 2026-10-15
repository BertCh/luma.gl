// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import syntheticTopology from '../../oracles/fixtures/parity-wp3-snapping-synthetic-topology.json?raw';
import syntheticPositions from '../../oracles/fixtures/parity-wp3-snapping-synthetic-positions.json?raw';
import syntheticWeights from '../../oracles/fixtures/parity-wp3-snapping-synthetic-weights.json?raw';
import syntheticQueries from '../../oracles/fixtures/parity-wp3-snapping-synthetic-queries.json?raw';
import chicagoTopology from '../../oracles/fixtures/parity-wp3-snapping-chicago-topology.json?raw';
import chicagoPositions from '../../oracles/fixtures/parity-wp3-snapping-chicago-positions.json?raw';
import chicagoWeights from '../../oracles/fixtures/parity-wp3-snapping-chicago-weights.json?raw';
import chicagoQueries from '../../oracles/fixtures/parity-wp3-snapping-chicago-queries.json?raw';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

export type SnapCandidate = {
  edge: number;
  source: number;
  target: number;
  fraction: number;
  distance: number;
  position: number[];
  length: number;
  weight: number;
  fractionBound: number;
};
export type SnapQuery = {
  point: number[];
  nearestEdges: number[];
  nearestDistance: number;
  candidates: SnapCandidate[];
};
export type SnappingFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  originalEdgeIds: number[];
  positions: number[];
  originalPointIds: number[];
  radius: number;
  coordinateBound: number;
  candidateCapacity: number;
  boundingBoxCandidateCount: number;
  exactTieBreak: boolean;
  queries: SnapQuery[];
};
// Native JSON-module transforms can perturb f64 literals by one ulp (Vite 8 observed).
// Parse source text to preserve the reference bits, especially exact f32 input coordinates.
export const snappingFixtures = [
  decodeOracleFixture<SnappingFixture>({
    ...JSON.parse(syntheticTopology),
    ...JSON.parse(syntheticPositions),
    ...JSON.parse(syntheticWeights),
    ...JSON.parse(syntheticQueries)
  }),
  decodeOracleFixture<SnappingFixture>({
    ...JSON.parse(chicagoTopology),
    ...JSON.parse(chicagoPositions),
    ...JSON.parse(chicagoWeights),
    ...JSON.parse(chicagoQueries)
  })
];
