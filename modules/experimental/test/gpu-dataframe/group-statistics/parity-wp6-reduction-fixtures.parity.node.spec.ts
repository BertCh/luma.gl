// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import earthquake from '../../oracles/fixtures/parity-wp6-earthquake-input.json';
import groupStatistics from '../../oracles/fixtures/parity-wp6-group-statistics.json';
import temporal from '../../oracles/fixtures/parity-wp6-temporal-reduction.json';
import histogram from '../../oracles/fixtures/parity-wp6-histogram.json';
import {decodeParityPayload} from '../../oracles/parity-wp6-reduction-harness';

const root = resolve(import.meta.dirname, '../../../../..');
const fixtureDirectory = resolve(root, 'modules/experimental/test/oracles/fixtures');
const digest = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');

it('WP6 reduction references retain versions, original source hashes and bounded fixture sizes', () => {
  for (const [topic, fixture] of Object.entries({
    'earthquake-input': earthquake,
    'group-statistics': groupStatistics,
    'temporal-reduction': temporal,
    histogram
  })) {
    expect(
      statSync(resolve(fixtureDirectory, `parity-wp6-${topic}.json`)).size
    ).toBeLessThanOrEqual(1500000);
    for (const version of Object.values(fixture.versions)) expect(version).toMatch(/^\d+\./);
    expect(fixture.generator).toBe('parity-wp6-dataframe-reductions.py');
    for (const [source, hash] of Object.entries(fixture.sources))
      expect(digest(readFileSync(resolve(root, source)))).toBe(hash);
  }
});

it('WP6 full USGS payloads preserve every source row and little-endian byte', () => {
  expect(earthquake.rowCount).toBe(77231);
  for (const payload of [earthquake.magnitude, earthquake.seconds, earthquake.cells]) {
    const bytes = Buffer.from(payload.base64, 'base64');
    expect(bytes.length).toBe(earthquake.rowCount * 4);
    expect(digest(bytes)).toBe(payload.sha256);
    expect(decodeParityPayload(payload)).toHaveLength(earthquake.rowCount);
  }
  const sourceDirectory = resolve(
    root,
    'examples/deck/spatial-analysis-showcase/public/data/poopdeck-earthquakes'
  );
  expect(Buffer.from(earthquake.magnitude.base64, 'base64')).toEqual(
    readFileSync(resolve(sourceDirectory, 'magnitude.bin'))
  );
  expect(Buffer.from(earthquake.seconds.base64, 'base64')).toEqual(
    readFileSync(resolve(sourceDirectory, 'time.bin'))
  );
  expect(decodeParityPayload(earthquake.cells)).toEqual(
    Array.from(readFileSync(resolve(sourceDirectory, 'mag_band.bin')))
  );
});

it('WP6 fixtures contain complete output arrays, retained capacities and source-row scores', () => {
  for (const scene of groupStatistics.cases) {
    expect(scene.expected.keys).toHaveLength(scene.capacity * (scene.keyBits / 32));
    expect(scene.expected.counts).toHaveLength(scene.capacity);
    for (const columns of Object.values(scene.expected.columns)) {
      for (const column of columns) {
        for (const [field, entries] of Object.entries(column)) {
          expect(entries, `${scene.name}/${field}`).toHaveLength(
            field === 'zScores'
              ? scene.mask.length
              : scene.capacity *
                  (field === 'sums' ? 2 : field === 'percentiles' ? scene.fractions.length : 1)
          );
        }
      }
    }
  }
  for (const scene of temporal.cases) {
    for (const field of ['counts', 'minimums', 'maximums', 'first', 'last'] as const) {
      expect(scene.expected[field]).toHaveLength(scene.cellCount * scene.bucketCount);
    }
    expect(scene.expected.ids).toHaveLength(scene.expected.status[0]);
    if (scene.source)
      expect(scene.expected.counts.reduce((total, count) => total + count, 0)).toBe(
        earthquake.rowCount
      );
  }
  for (const scene of histogram.cases) {
    expect(scene.expected).toHaveLength(scene.bins);
    if (scene.source)
      expect(scene.expected.reduce((total, count) => total + count, 0)).toBe(earthquake.rowCount);
  }
});

it('WP6 int64 temporal references retain nanosecond differences beyond JS integer precision', () => {
  const relative = temporal.cases.find(scene => scene.name === 'edges-float-0')!;
  const nanoseconds = temporal.cases.find(
    scene => scene.name === 'edges-word-1577836800000000000'
  )!;
  expect(nanoseconds.expected).toEqual(relative.expected);
  const firstTimestamp = nanoseconds.timestamps[0];
  const secondTimestamp = nanoseconds.timestamps[1];
  if (typeof firstTimestamp !== 'string' || typeof secondTimestamp !== 'string') {
    throw new Error('Int64 fixture timestamps must remain decimal strings');
  }
  expect(BigInt(secondTimestamp) - BigInt(firstTimestamp)).toBe(2n);
  expect(Number(nanoseconds.timestamps[1])).toBe(Number(nanoseconds.timestamps[0]));
});
