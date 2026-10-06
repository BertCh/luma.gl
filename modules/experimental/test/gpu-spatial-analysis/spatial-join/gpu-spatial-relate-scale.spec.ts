// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createRandom} from '../segment-intersection/segment-intersection-oracle';
import {
  evaluateRelationalPredicate,
  getOracleDimension,
  type OracleFeature
} from './spatial-predicate-oracle';
import {
  createRealisticRelateScene,
  hashFeatureCoordinates
} from './spatial-relate-realistic-scenes';
import {comparePairs, runScaleJoin} from './spatial-relate-scale-harness';
import {SHAPELY_SCALE_RELATE, SHAPELY_SCALE_SCENE_HASHES} from './shapely-scale-fixtures';

const SCENE = createRealisticRelateScene(createRandom);

it('realistic relate scene is the one pinned by the Shapely fixtures', () => {
  for (const [name, features] of Object.entries(SCENE)) {
    expect(hashFeatureCoordinates(features), name).toBe(
      SHAPELY_SCALE_SCENE_HASHES[name as keyof typeof SHAPELY_SCALE_SCENE_HASHES]
    );
  }
});

it('GPUSpatialPredicateJoin relate matrices match Shapely on realistic polygons and lines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scenes: [string, OracleFeature[], OracleFeature[]][] = [
    ['zips-queries', SCENE.zips, SCENE.queries],
    ['zips-lines', SCENE.zips, SCENE.lines],
    ['cells-blocks', SCENE.cells, SCENE.blocks]
  ];
  for (const [name, lefts, rights] of scenes) {
    const expected = SHAPELY_SCALE_RELATE[name];
    const result = await runScaleJoin(device, lefts, rights, 'relate', {matrix: true});
    expect(result.overflow, name).toBe(0);
    expect(result.uncertainCount, name).toBe(0);
    const actual: Record<string, string> = {};
    result.pairs.forEach((pair, slot) => {
      actual[pair] = result.matrices[slot];
    });
    expect(Object.keys(expected).length, name).toBeGreaterThan(5);
    expect(actual, name).toEqual(expected);
  }
});

it('GPUSpatialPredicateJoin engines match Shapely predicates and auto picks relate for large features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scenes: [string, OracleFeature[], OracleFeature[]][] = [
    ['zips-queries', SCENE.zips, SCENE.queries],
    ['zips-lines', SCENE.zips, SCENE.lines],
    ['cells-blocks', SCENE.cells, SCENE.blocks]
  ];
  let total = 0;
  for (const [name, lefts, rights] of scenes) {
    const leftDimension = getOracleDimension(lefts[0]);
    const rightDimension = getOracleDimension(rights[0]);
    for (const predicate of ['intersects', 'contains', 'within'] as const) {
      const label = `${name} ${predicate}`;
      // Expected pairs follow from the pinned Shapely matrices (all other pairs are disjoint).
      const expected = Object.entries(SHAPELY_SCALE_RELATE[name])
        .filter(([, matrix]) =>
          evaluateRelationalPredicate(predicate, matrix, leftDimension, rightDimension)
        )
        .map(([pair]) => pair)
        .sort(comparePairs);
      const relate = await runScaleJoin(device, lefts, rights, predicate, {engine: 'relate'});
      const auto = await runScaleJoin(device, lefts, rights, predicate, {});
      expect(relate.usesRelateEngine, label).toBe(true);
      expect(auto.usesRelateEngine, label).toBe(true);
      expect(relate.pairs.sort(comparePairs), label).toEqual(expected);
      expect(auto.pairs.sort(comparePairs), label).toEqual(expected);
      const fast = await runScaleJoin(device, lefts, rights, predicate, {engine: 'fast'});
      expect(fast.usesRelateEngine, label).toBe(false);
      expect(fast.pairs.sort(comparePairs), label).toEqual(expected);
      total += expected.length;
    }
  }
  expect(total).toBeGreaterThan(40);
});

it('GPUSpatialPredicateJoin dwithin workgroup kernel matches the single-invocation kernel', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scenes: [string, OracleFeature[], OracleFeature[]][] = [
    ['zips-queries', SCENE.zips, SCENE.queries],
    ['zips-lines', SCENE.zips, SCENE.lines],
    ['cells-blocks', SCENE.cells, SCENE.blocks]
  ];
  let total = 0;
  for (const [name, lefts, rights] of scenes) {
    for (const distance of [0, 0.02, 0.4]) {
      const label = `${name} dwithin ${distance}`;
      const fast = await runScaleJoin(device, lefts, rights, 'dwithin', {engine: 'fast', distance});
      const workgroup = await runScaleJoin(device, lefts, rights, 'dwithin', {
        engine: 'relate',
        distance
      });
      const auto = await runScaleJoin(device, lefts, rights, 'dwithin', {distance});
      expect(fast.usesWorkgroupDistance, label).toBe(false);
      expect(workgroup.usesWorkgroupDistance, label).toBe(true);
      expect(auto.usesWorkgroupDistance, label).toBe(true);
      expect(workgroup.pairs, label).toEqual(fast.pairs);
      expect(auto.pairs, label).toEqual(fast.pairs);
      total += fast.pairs.length;
    }
  }
  expect(total).toBeGreaterThan(40);
});

it('GPUSpatialPredicateJoin engine auto keeps the fast kernel for small features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const squares: OracleFeature[] = [0, 1, 2].map(
    (index): OracleFeature => ({
      kind: 'polygons',
      polygons: [
        [
          [
            [index, 0],
            [index + 2, 0],
            [index + 2, 2],
            [index, 2]
          ]
        ]
      ]
    })
  );
  const auto = await runScaleJoin(device, squares, squares, 'intersects', {});
  expect(auto.usesRelateEngine).toBe(false);
  expect(auto.usesWorkgroupDistance).toBe(false);
  expect(auto.pairs.length).toBe(9);
});
