// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import FIXTURE from '../../oracles/fixtures/geometry-relate-near-parallel.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  evaluateRelationalPredicate,
  type OracleFeature,
  type OraclePoint
} from './spatial-predicate-oracle';
import {runScaleJoin} from './spatial-relate-scale-harness';

// Shapely relate of polygon pairs whose long edges run within a few float32 ulps of each other
// (test/oracles/geometry-relate-near-parallel.py). Locating rounded piece midpoints got 11 of the
// 400 matrices wrong; pieces are now located from an exact endpoint and exact crossings. Pairs in
// neighbouring cells may also intersect; only the pair of each cell is compared.
type NearParallelPair = {left: OraclePoint[]; right: OraclePoint[]; relate: string; mode: string};

const PAIRS = FIXTURE.pairs as NearParallelPair[];
const LEFTS = PAIRS.map(pair => toFeature(pair.left));
const RIGHTS = PAIRS.map(pair => toFeature(pair.right));

function toFeature(ring: OraclePoint[]): OracleFeature {
  return {kind: 'polygons', polygons: [[ring]]};
}

function getDiagonalPairs(pairs: string[]): string[] {
  return pairs.filter(pair => {
    const [left, right] = pair.split(',');
    return left === right;
  });
}

it('relate matrices of near-parallel polygon pairs match Shapely', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runScaleJoin(device, LEFTS, RIGHTS, 'relate', {
    engine: 'relate',
    matrix: true
  });
  const actual = new Map<string, string>();
  result.pairs.forEach((pair, slot) => actual.set(pair, result.matrices[slot]));
  const mismatches = PAIRS.flatMap((pair, index) => {
    const expected = evaluateRelationalPredicate('intersects', pair.relate, 2, 2)
      ? pair.relate
      : undefined;
    const matrix = actual.get(`${index},${index}`);
    return matrix === expected ? [] : [`${index} ${pair.mode}: ${pair.relate} != ${matrix}`];
  });
  expect(mismatches).toEqual([]);
  expect(result.uncertainCount).toBe(0);
});

it('predicates of near-parallel polygon pairs match Shapely for every engine', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const predicate of ['intersects', 'contains', 'within', 'touches', 'overlaps'] as const) {
    const expected = PAIRS.flatMap((pair, index) =>
      evaluateRelationalPredicate(predicate, pair.relate, 2, 2) ? [`${index},${index}`] : []
    );
    const engines =
      predicate === 'touches' || predicate === 'overlaps' ? ['relate'] : ['auto', 'fast', 'relate'];
    for (const engine of engines as ('auto' | 'fast' | 'relate')[]) {
      const result = await runScaleJoin(device, LEFTS, RIGHTS, predicate, {engine});
      expect(getDiagonalPairs(result.pairs).sort(), `${predicate} ${engine}`).toEqual(
        expected.sort()
      );
    }
  }
});
