// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  evaluateOracleRelate,
  evaluateRelationalPredicate,
  getOracleDimension,
  line,
  point,
  rectangle,
  type OracleFeature
} from './spatial-predicate-oracle';
import {RELATE_KINDS, getRandomRelateSeeds, getRelateSide} from './spatial-relate-scenes';
import {SHAPELY_RELATE_FIXTURES} from './shapely-relate-fixtures';

const PREDICATE_NAMES = {
  intersects: 'intersects',
  contains: 'contains',
  within: 'within',
  covers: 'covers',
  covered_by: 'coveredBy',
  touches: 'touches',
  crosses: 'crosses',
  overlaps: 'overlaps',
  equals: 'equals',
  contains_properly: 'containsProperly'
} as const;

it('oracle relate: well-known DE-9IM matrices (GEOS)', () => {
  expect(evaluateOracleRelate(rectangle(0, 0, 2, 2), rectangle(2, 0, 2, 2))).toBe('FF2F11212');
  expect(evaluateOracleRelate(rectangle(0, 0, 2, 2), rectangle(0, 0, 2, 2))).toBe('2FFF1FFF2');
  expect(evaluateOracleRelate(point(1, 1), rectangle(0, 0, 2, 2))).toBe('0FFFFF212');
  expect(evaluateOracleRelate(line([0, 0], [4, 0]), line([2, -1], [2, 1]))).toBe('0F1FF0102');
  expect(evaluateOracleRelate(point(9, 9), point(0, 0))).toBe('FF0FFF0F2');
});

it('oracle relate matches Shapely 2.1.2 on pinned hand-built and random scenes', () => {
  let compared = 0;
  let touching = 0;
  for (const leftKind of RELATE_KINDS) {
    for (const rightKind of RELATE_KINDS) {
      const fixture = SHAPELY_RELATE_FIXTURES[`${leftKind}-${rightKind}`];
      const [leftSeed, rightSeed] = getRandomRelateSeeds(leftKind, rightKind);
      const lefts: OracleFeature[] = getRelateSide(leftKind, leftSeed, true);
      const rights: OracleFeature[] = getRelateSide(rightKind, rightSeed, true);
      expect(lefts.length).toBe(fixture.left);
      expect(rights.length).toBe(fixture.right);
      for (const [leftRow, left] of lefts.entries()) {
        for (const [rightRow, right] of rights.entries()) {
          const pair = leftRow * rights.length + rightRow;
          const expected = fixture.relate.slice(pair * 9, pair * 9 + 9);
          const matrix = evaluateOracleRelate(left, right);
          expect(matrix, `${leftKind}-${rightKind} ${leftRow},${rightRow}`).toBe(expected);
          compared++;
          if (!expected.startsWith('FF*FF')) {
            touching++;
          }
          for (const [shapelyName, name] of Object.entries(PREDICATE_NAMES)) {
            expect(
              evaluateRelationalPredicate(
                name,
                matrix,
                getOracleDimension(left),
                getOracleDimension(right)
              ),
              `${leftKind}-${rightKind} ${name} ${leftRow},${rightRow}`
            ).toBe(fixture.predicates[shapelyName][pair] === '1');
          }
        }
      }
    }
  }
  expect(compared).toBe(9 * 18 * 18);
  expect(touching).toBeGreaterThan(300);
});
