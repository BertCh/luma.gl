// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {doesRelatePatternAdmitDisjoint} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {evaluateOraclePredicate, joinWithOracle} from './spatial-predicate-oracle';
import {
  PREPARED_SELF_JOIN_FEATURES,
  PREPARED_SELF_JOIN_OVERLAPS,
  PREPARED_SELF_JOIN_PATTERN
} from './spatial-join-prepared-fixture';

it('prepared self-join fixture has ten directed overlaps and only row 4 unmatched', () => {
  const features = PREPARED_SELF_JOIN_FEATURES;
  const pairs = joinWithOracle('overlaps', features, features, {excludeSameRow: true});
  expect(pairs).toEqual(PREPARED_SELF_JOIN_OVERLAPS);
  const matchedRows = new Set(pairs.map(([left]) => left));
  expect(features.flatMap((_, row) => (matchedRows.has(row) ? [] : [row]))).toEqual([4]);
  // Overlap is not reflexive, even without excludeSameRow. Intersects is reflexive:
  // the fast-path anti join must exclude the isolated row's own match to emit row 4.
  for (const feature of features) {
    expect(evaluateOraclePredicate('overlaps', feature, feature)).toBe(false);
    expect(evaluateOraclePredicate('intersects', feature, feature)).toBe(true);
  }
  expect(doesRelatePatternAdmitDisjoint([PREPARED_SELF_JOIN_PATTERN])).toBe(false);
});
