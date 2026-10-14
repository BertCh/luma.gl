// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixture from '../../oracles/fixtures/network-shortest-paths.json';
import {buildCSR, type NetworkCSR, type NetworkEdge} from './network-reachability-oracle';

/** One networkx shortest-path scene (see `test/oracles/network-shortest-paths.py`). */
export type ShortestPathReferenceScene = {
  nodeCount: number;
  /** Directed `[from, to, weight]` edges; float32 weights, negative = impassable. */
  edges: NetworkEdge[];
  sources: number[];
  sourceCosts: number[];
  /** Inclusive cost limit, or `null` for none. */
  costLimit: number | null;
  /** networkx float64 cost per node, `null` when unreached. */
  costs: (number | null)[];
  /** Largest hop count of a networkx shortest path. */
  maximumHops: number;
  tolerance: {costRelative: number; reachedSet: 'exact'};
};

/** The whole fixture. */
export type ShortestPathReferenceFixture = {
  generator: string;
  versions: Record<string, string>;
  scenes: Record<string, ShortestPathReferenceScene>;
};

/** The committed networkx fixture. */
export const SHORTEST_PATH_REFERENCE = fixture as unknown as ShortestPathReferenceFixture;

/** CSR of a reference scene, edges in fixture order within each row. */
export function getShortestPathReferenceCSR(scene: ShortestPathReferenceScene): NetworkCSR {
  return buildCSR(scene.nodeCount, scene.edges);
}

/**
 * Compares float32 costs with the networkx costs of a scene: the reached set must match exactly
 * and every reached cost must lie within the scene's relative tolerance. Returns the first
 * violation as text, or `undefined`.
 */
export function compareShortestPathCosts(
  scene: ShortestPathReferenceScene,
  costs: ArrayLike<number>
): string | undefined {
  for (let node = 0; node < scene.nodeCount; node++) {
    const expected = scene.costs[node];
    const actual = costs[node];
    if (expected === null) {
      if (Number.isFinite(actual)) {
        return `node ${node}: expected unreached, got ${actual}`;
      }
      continue;
    }
    if (!Number.isFinite(actual)) {
      return `node ${node}: expected ${expected}, got unreached`;
    }
    if (Math.abs(actual - expected) > scene.tolerance.costRelative * Math.max(expected, 1e-30)) {
      return `node ${node}: expected ${expected}, got ${actual}`;
    }
  }
  return undefined;
}
