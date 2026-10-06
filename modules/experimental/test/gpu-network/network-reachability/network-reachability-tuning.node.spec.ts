// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  adaptReachabilityIterations,
  recommendReachabilityIterations
} from '../../../src/gpu-network/network-reachability/network-reachability-tuning';

it('recommends rounds from hop eccentricity and local iterations', () => {
  // ceil(224 / (0.7 * 32)) + 4 = 10 + 4
  expect(recommendReachabilityIterations({nodeCount: 50176, localIterations: 32})).toEqual({
    localIterations: 32,
    maxIterations: 14
  });
  expect(
    recommendReachabilityIterations({nodeCount: 50000, hopEccentricity: 12, localIterations: 32})
      .maxIterations
  ).toBe(5);
  expect(
    recommendReachabilityIterations({nodeCount: 490000, localIterations: 32}).maxIterations
  ).toBe(36);
});

it('recommends fewer rounds for longer chains and clamps to [1, 1024]', () => {
  const short = recommendReachabilityIterations({nodeCount: 490000, localIterations: 16});
  const long = recommendReachabilityIterations({nodeCount: 490000, localIterations: 64});
  expect(long.maxIterations).toBeLessThan(short.maxIterations);
  expect(
    recommendReachabilityIterations({nodeCount: 0, localIterations: 32}).maxIterations
  ).toBeGreaterThanOrEqual(1);
  expect(
    recommendReachabilityIterations({hopEccentricity: 1e9, nodeCount: 10, localIterations: 1})
      .maxIterations
  ).toBe(1024);
});

it('rejects invalid recommendation inputs', () => {
  expect(() => recommendReachabilityIterations({nodeCount: -1, localIterations: 32})).toThrow();
  expect(() => recommendReachabilityIterations({nodeCount: 10, localIterations: 0})).toThrow();
  expect(() =>
    recommendReachabilityIterations({nodeCount: 10, hopEccentricity: NaN, localIterations: 32})
  ).toThrow();
});

it('grows when not converged, capped at 1024', () => {
  expect(
    adaptReachabilityIterations({maxIterations: 64, iterationCount: 64, converged: false})
  ).toBe(128);
  expect(
    adaptReachabilityIterations({maxIterations: 800, iterationCount: 800, converged: false})
  ).toBe(1024);
  expect(
    adaptReachabilityIterations({maxIterations: 1024, iterationCount: 1024, converged: false})
  ).toBe(1024);
});

it('shrinks toward the rounds used when converged, with hysteresis', () => {
  // 10 * 4 < 64: shrink to ceil(15) + 2
  expect(
    adaptReachabilityIterations({maxIterations: 64, iterationCount: 10, converged: true})
  ).toBe(17);
  // 20 * 4 >= 64: keep
  expect(
    adaptReachabilityIterations({maxIterations: 64, iterationCount: 20, converged: true})
  ).toBe(64);
  expect(adaptReachabilityIterations({maxIterations: 64, iterationCount: 0, converged: true})).toBe(
    2
  );
});

it('settles: repeated converged adaptation is stable', () => {
  let maxIterations = 64;
  for (let frame = 0; frame < 5; frame++) {
    maxIterations = adaptReachabilityIterations({
      maxIterations,
      iterationCount: 10,
      converged: true
    });
  }
  expect(maxIterations).toBe(17);
});
