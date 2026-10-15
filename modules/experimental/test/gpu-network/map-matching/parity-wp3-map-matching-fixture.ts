// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {compareParity, type Tolerance} from '../parity-wp3-comparison';
export {EXACT, type Tolerance} from '../parity-wp3-comparison';
import syntheticJson from '../../oracles/fixtures/parity-wp3-map-matching-synthetic.json';
import chicagoJson from '../../oracles/fixtures/parity-wp3-map-matching-chicago.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import type {GPUMapMatchingSettings} from '../../../src/gpu-network/map-matching';

export type MatchAlternative = {
  edges: number[];
  fractions: number[];
  offsets: number[];
  distances: number[];
  positions: number[];
  routes: {length: number; nodePath: number[]}[];
};
export type MatchFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  settings: GPUMapMatchingSettings;
  cellSize: number;
  entryCapacity: number;
  bounds: {minimum: [number, number]; maximum: [number, number]};
  positions: number[];
  offsets: number[];
  neighbors: number[];
  points: number[];
  trackOffsets: number[];
  expected: {logLikelihood: number; scoreGap: number; alternatives: MatchAlternative[]}[];
  candidates: {
    edge: number;
    distance: number;
    fraction: number;
    offset: number;
    position: number[];
  }[][][];
  source: {
    nodeIds?: number[];
    edgeIds?: number[];
    fullNodeCount?: number;
    fullEdgeCount?: number;
    fullGraphDistanceComparisons?: number;
    maximumSubsetCostError?: number;
  };
  diagnostics: {maximumDijkstraBall: number; nonNearestMatches: number};
  tolerances: {distance: Tolerance; fraction: Tolerance; likelihood: Tolerance};
};

export const MATCH_FIXTURES = [syntheticJson, chicagoJson].map(fixture =>
  decodeOracleFixture<MatchFixture>(fixture)
);

export function compareValues(
  fixture: MatchFixture,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: Tolerance,
  caseName = fixture.case
): void {
  compareParity('GPUMapMatching', fixture, metric, actual, expected, tolerance, caseName);
}
