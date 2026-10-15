// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />
import raw from '../../../experimental/test/oracles/fixtures/parity-wp3-graph-showcase.json?raw';
import {decodeOracleFixture} from '../../../experimental/test/oracles/oracle-fixture';
import type {GraphMetricsFixture} from './parity-wp3-graph-metrics-fixture';

export type ShowcaseGraph = GraphMetricsFixture & {
  offsets: number[];
  neighbors: number[];
  reverseOffsets: number[];
  reverseNeighbors: number[];
  components: number[];
  communityLabels: number[];
  communityEdges: [number, number, number][];
  communityScore: number;
  runs: {
    damping: number;
    iterations: number;
    ranks: number[];
    previous: number[];
    residual: number;
  }[];
};
export const showcaseGraphs = decodeOracleFixture<ShowcaseGraph[]>(JSON.parse(raw));
