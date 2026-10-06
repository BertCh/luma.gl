// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {recommendLaneCount} from '../../../src/gpu-network/network-accessibility/network-accessibility-lanes';

it('recommendLaneCount scales lanes up for small graphs', () => {
  // 48x48 grid, 4 neighbors per node, all origins: 1M / 2304 = 434 lanes.
  expect(recommendLaneCount({rowCount: 2304, nodeCount: 2304, edgeCount: 9024})).toBe(434);
  expect(recommendLaneCount({rowCount: 100, nodeCount: 2304, edgeCount: 9024})).toBe(100);
});

it('recommendLaneCount never drops below the legacy 32 lanes for the node budget', () => {
  // 50k-node road: the node budget alone would give 19 lanes.
  expect(recommendLaneCount({rowCount: 64, nodeCount: 50176, edgeCount: 155000})).toBe(32);
  expect(recommendLaneCount({rowCount: 10, nodeCount: 50176, edgeCount: 155000})).toBe(10);
});

it('recommendLaneCount caps scratch memory', () => {
  // 32M words / (1M + 2 * 4M) = 3 lanes.
  expect(recommendLaneCount({rowCount: 64, nodeCount: 1_000_000, edgeCount: 4_000_000})).toBe(3);
  expect(
    recommendLaneCount({
      rowCount: 64,
      nodeCount: 1000,
      edgeCount: 4000,
      scratchByteBudget: 4 * 9000 * 5
    })
  ).toBe(5);
  // Always at least one lane.
  expect(recommendLaneCount({rowCount: 8, nodeCount: 1e9, edgeCount: 1e9})).toBe(1);
});

it('recommendLaneCount handles degenerate inputs and the uint32 range', () => {
  expect(recommendLaneCount({rowCount: 1, nodeCount: 1, edgeCount: 0})).toBe(1);
  expect(recommendLaneCount({rowCount: 0, nodeCount: 10, edgeCount: 0})).toBe(1);
  const lanes = recommendLaneCount({
    rowCount: 1e6,
    nodeCount: 4,
    edgeCount: 0,
    expandedNodeBudget: 1e12,
    scratchByteBudget: 1e15
  });
  expect(lanes * 4).toBeLessThan(2 ** 32 - 1);
});
