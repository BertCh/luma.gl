// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createRandom} from '../segment-intersection/segment-intersection-oracle';
import {runScaleJoin, comparePairs} from './spatial-relate-scale-harness';
import {evaluateRelationalPredicate} from './spatial-predicate-oracle';
import {
  createSmallSharedBoundaryScene,
  hashFeatureCoordinates
} from './spatial-relate-realistic-scenes';
import {SHAPELY_SMALL_SHARED_HASHES, SHAPELY_SMALL_SHARED_RELATE} from './shapely-scale-fixtures';

const SCENE = createSmallSharedBoundaryScene(createRandom);

it('small shared-boundary scene is the one pinned by the Shapely fixtures', () => {
  expect(hashFeatureCoordinates(SCENE.cells)).toBe(SHAPELY_SMALL_SHARED_HASHES.cells);
  expect(hashFeatureCoordinates(SCENE.blocks)).toBe(SHAPELY_SMALL_SHARED_HASHES.blocks);
});

it('small features sharing boundaries match Shapely for every engine', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scenes = [
    ['small-cells-blocks', SCENE.cells, SCENE.blocks],
    ['small-cells-cells', SCENE.cells, SCENE.cells]
  ] as const;
  let total = 0;
  for (const [name, lefts, rights] of scenes) {
    for (const predicate of ['intersects', 'contains', 'within'] as const) {
      const expected = Object.entries(SHAPELY_SMALL_SHARED_RELATE[name])
        .filter(([, matrix]) => evaluateRelationalPredicate(predicate, matrix, 2, 2))
        .map(([pair]) => pair)
        .sort(comparePairs);
      for (const engine of ['auto', 'fast', 'relate'] as const) {
        const result = await runScaleJoin(device, lefts, rights, predicate, {engine});
        expect(result.pairs.sort(comparePairs), `${name} ${predicate} ${engine}`).toEqual(expected);
      }
      total += expected.length;
    }
    for (const distance of [0, 0.05]) {
      for (const engine of ['auto', 'fast', 'relate'] as const) {
        const result = await runScaleJoin(device, lefts, rights, 'dwithin', {engine, distance});
        // Every Shapely-intersecting pair is within distance 0; farther pairs only for 0.05.
        const intersecting = Object.keys(SHAPELY_SMALL_SHARED_RELATE[name]).sort(comparePairs);
        const actual = result.pairs.sort(comparePairs);
        expect(actual.length, `${name} dwithin ${distance} ${engine}`).toBeGreaterThanOrEqual(
          intersecting.length
        );
        expect(actual, `${name} dwithin ${distance} ${engine}`).toEqual(
          expect.arrayContaining(intersecting)
        );
        if (distance === 0) {
          expect(actual).toEqual(intersecting);
        }
      }
    }
  }
  expect(total).toBeGreaterThan(40);
});
