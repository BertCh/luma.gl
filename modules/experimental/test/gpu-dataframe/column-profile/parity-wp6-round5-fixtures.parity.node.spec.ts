// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import profiles from '../../oracles/fixtures/parity-wp6-offset-profiles.json';
import moments from '../../oracles/fixtures/parity-wp6-offset-moments.json';
import joins from '../../oracles/fixtures/parity-wp6-key-join.json';
import bivariate from '../../oracles/fixtures/parity-wp6-bivariate.json';
import times from '../../oracles/fixtures/parity-wp6-time-window-real.json';
import timeEdges from '../../oracles/fixtures/parity-wp6-time-window-edges.json';
import {
  decodeParityPayload,
  getParityEarthquakes
} from '../../oracles/parity-wp6-reduction-harness';

const root = resolve(import.meta.dirname, '../../../../..');
const fixtureDirectory = resolve(root, 'modules/experimental/test/oracles/fixtures');

it('WP6 round 5 references retain versions, source SHA256 and bounded strict JSON', () => {
  for (const [topic, fixture] of Object.entries({
    'offset-profiles': profiles,
    'offset-moments': moments,
    'key-join': joins,
    bivariate,
    'time-window-real': times,
    'time-window-edges': timeEdges
  })) {
    const path = resolve(fixtureDirectory, `parity-wp6-${topic}.json`);
    expect(statSync(path).size).toBeLessThanOrEqual(1500000);
    expect(JSON.parse(readFileSync(path, 'utf8')).generator).toBe('parity-wp6-round5.py');
    for (const version of Object.values(fixture.versions)) expect(version).toMatch(/^\d+\./);
    for (const [source, hash] of Object.entries(fixture.sources)) {
      expect(
        createHash('sha256')
          .update(readFileSync(resolve(root, source)))
          .digest('hex')
      ).toBe(hash);
    }
  }
});

it('WP6 shifted-moment references span partial tiles, invalid leading tiles and constants', () => {
  expect(moments.cases.map(scene => scene.values.length)).toEqual([11, 257, 2051, 4103]);
  for (const scene of moments.cases) {
    expect(scene.mask).toHaveLength(scene.values.length);
    expect(scene.expected).toHaveLength(5);
    expect(scene.expected[2]).toBeGreaterThan(0);
    const profile = profiles.cases.find(entry => entry.name === scene.name)!;
    expect(profile.expected.statistics).toHaveLength(11);
    expect(profile.expected.statistics[6]).toBeCloseTo(scene.expected[2] / scene.expected[0], 10);
  }
  expect(
    moments.cases
      .at(-1)!
      .mask.slice(0, 2048)
      .every(value => value === 0)
  ).toBe(true);
  expect(profiles.cases.at(-1)!.expected.statistics.slice(6, 9)).toEqual([0, 0, 0]);
});

it('WP6 joins retain all membership, gather, aggregate and exact-word outputs', () => {
  expect(joins.cases[0].rightKeys).toHaveLength(3109);
  for (const scene of joins.cases) {
    expect(scene.values).toHaveLength(scene.rightKeys.length);
    expect(scene.expected.rightMatched).toHaveLength(scene.rightKeys.length);
    for (const metric of [
      'rightRows',
      'matchCounts',
      'matched',
      'gathered',
      'countAll',
      'countFinite',
      'sum',
      'mean',
      'minimum',
      'maximum'
    ] as const) {
      expect(scene.expected[metric]).toHaveLength(scene.leftKeys.length);
    }
    expect(scene.expected.sumWords).toHaveLength(scene.leftKeys.length * 2);
    expect(scene.expected.meanWords).toEqual(scene.expected.sumWords);
    expect(scene.expected.ids).toHaveLength(scene.expected.count[0]);
    expect(scene.expected.ids).toEqual([...scene.expected.ids].sort((left, right) => left - right));
  }
  expect(
    joins.cases
      .find(scene => scene.name === 'duplicates-nulls-64')!
      .leftKeys.some(key => BigInt(key) > 0xffffffffn)
  ).toBe(true);
});

it('WP6 bivariate references retain every class id/color and inactive count slot on every frame', () => {
  expect(bivariate.cases[0].valuesX).toHaveLength(3109);
  for (const scene of bivariate.cases) {
    for (const frame of scene.frames) {
      expect(frame.expected.classIds).toHaveLength(scene.mask.length);
      expect(frame.expected.colors).toHaveLength(scene.mask.length);
      expect(frame.expected.classCounts).toHaveLength(16);
      expect(frame.expected.classCounts.reduce((sum, value) => sum + value, 0)).toBe(
        frame.expected.classIds.filter(value => value !== 0xffffffff).length
      );
    }
  }
});

it('WP6 time references compare full USGS masks and preserve exact Int64 keys beyond 2^53', () => {
  expect(getParityEarthquakes().seconds).toHaveLength(77231);
  for (const scene of [...times.cases, ...timeEdges.cases]) {
    const mask = decodeParityPayload(scene.expected.mask);
    const ids = decodeParityPayload(scene.expected.ids);
    expect(mask).toHaveLength(scene.rowCount);
    expect(mask.every(value => value === 0 || value === 1)).toBe(true);
    expect(mask.reduce((sum, value) => sum + value, 0)).toBe(scene.expected.requiredCount[0]);
    expect(ids).toHaveLength(scene.expected.count[0]);
    expect(ids.every(row => mask[row] === 1)).toBe(true);
    expect(ids).toEqual([...ids].sort((left, right) => left - right));
    if (scene.real) expect(scene.rowCount).toBe(77231);
  }
  const wordScene = timeEdges.cases.find(scene => scene.name === 'int64-18014398509481984')!;
  expect(BigInt(wordScene.values[3]!) - BigInt(wordScene.values[2]!)).toBe(1n);
});
