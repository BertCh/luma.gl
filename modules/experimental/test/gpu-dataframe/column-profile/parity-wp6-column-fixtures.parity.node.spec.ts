// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import profile from '../../oracles/fixtures/parity-wp6-column-profile.json';
import quantiles from '../../oracles/fixtures/parity-wp6-column-quantiles.json';
import composite from '../../oracles/fixtures/parity-wp6-composite-score.json';
import temporal from '../../oracles/fixtures/parity-wp6-temporal-reduction.json';
import histogram from '../../oracles/fixtures/parity-wp6-histogram.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

const root = resolve(import.meta.dirname, '../../../../..');
const fixtureDirectory = resolve(root, 'modules/experimental/test/oracles/fixtures');

it('WP6 column references retain versions, complete source hashes, and bounded fixture sizes', () => {
  for (const [topic, fixture] of Object.entries({
    'column-profile': profile,
    'column-quantiles': quantiles,
    'composite-score': composite
  })) {
    expect(
      statSync(resolve(fixtureDirectory, `parity-wp6-${topic}.json`)).size
    ).toBeLessThanOrEqual(1500000);
    expect(fixture.generator).toBe('parity-wp6-column-analysis.py');
    expect(fixture.conventions).toContain('COMPOSITE REFERENCE');
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

it('WP6 profiles retain every statistic, register, bin and top-category slot', () => {
  for (const scene of profile.cases) {
    const columnCount = scene.columns.length;
    for (const column of scene.columns) expect(column.values).toHaveLength(scene.mask.length);
    expect(scene.expected.statistics).toHaveLength(columnCount * 11);
    expect(scene.expected.counts).toHaveLength(columnCount * 2);
    expect(scene.expected.histograms).toHaveLength(columnCount * scene.bins);
    expect(scene.expected.registers).toHaveLength(columnCount * 2 ** scene.precision);
    expect(scene.expected.topCategories).toHaveLength(columnCount * scene.topCount);
    expect(scene.expected.topCategoryCounts).toHaveLength(columnCount * scene.topCount);
  }
  expect(profile.cases[0].mask).toHaveLength(3109);
  const singleton = profile.cases.find(scene => scene.name === 'singleton')!;
  expect(singleton.expected.statistics[6]).toBe(0);
  expect(singleton.expected.statistics[7]).toBeNull();
  const noFinite = profile.cases.find(scene => scene.name === 'no-finite')!;
  expect(noFinite.expected.counts).toEqual([2, 1]);
  expect(noFinite.expected.statistics.slice(4, 9)).toEqual([null, null, null, null, null]);
});

it('WP6 quantiles retain five interpolation methods, half-even ties, complete filters and counts', () => {
  for (const scene of quantiles.cases) {
    expect(scene.mask).toHaveLength(scene.values.length);
    expect(scene.expected.outputMask).toHaveLength(scene.values.length);
    expect(scene.expected.filterBounds).toHaveLength(2);
    expect(Object.keys(scene.expected.quantiles)).toHaveLength(5);
    for (const values of Object.values(scene.expected.quantiles)) {
      expect(values).toHaveLength(scene.probabilities.length);
      expect(values.slice(-3)).toEqual([null, null, null]);
    }
  }
  expect(quantiles.cases[0].values).toHaveLength(3109);
  for (const name of ['nearest-half-even-upper', 'nearest-half-even-lower']) {
    expect(quantiles.cases.find(scene => scene.name === name)!.expected.quantiles.nearest[3]).toBe(
      20
    );
  }
});

it('WP6 composite fixtures retain every row component, score and requested PCA diagnostic', () => {
  for (const scene of composite.cases) {
    expect(scene.indicators).toHaveLength(scene.mask.length * scene.indicatorCount);
    expect(scene.expected.score).toHaveLength(scene.mask.length);
    expect(scene.expected.scaled).toHaveLength(scene.indicators.length);
    expect(scene.expected.columnStatistics).toHaveLength(scene.indicatorCount * 4);
    for (let row = 0; row < scene.mask.length; row++) {
      const values = scene.indicators.slice(
        row * scene.indicatorCount,
        (row + 1) * scene.indicatorCount
      );
      if (
        !scene.mask[row] ||
        values.some(
          value => value === null || (typeof value === 'string' && value.includes('Infinity'))
        )
      ) {
        expect(scene.expected.score[row]).toBeNull();
        expect(
          scene.expected.scaled.slice(row * scene.indicatorCount, (row + 1) * scene.indicatorCount)
        ).toEqual(Array(scene.indicatorCount).fill(null));
      }
    }
    if (scene.principal) {
      expect(scene.eigengap).toBeGreaterThan(0.25);
      expect(scene.expected.loadings).toHaveLength(scene.indicatorCount);
      expect(scene.expected.principalComponentSummary).toHaveLength(3);
      expect(scene.expected.principalComponentSummary![2]).toBeLessThan(1e-12);
    }
  }
  expect(composite.cases[0].mask).toHaveLength(791);
});

it('WP6 temporal E targets preserve the original f32 selections through JSON module decoding', () => {
  const raw = decodeOracleFixture<typeof temporal>(
    JSON.parse(
      readFileSync(resolve(fixtureDirectory, 'parity-wp6-temporal-reduction.json'), 'utf8')
    )
  );
  const imported = decodeOracleFixture<typeof temporal>(temporal);
  for (let sceneIndex = 0; sceneIndex < raw.cases.length; sceneIndex++) {
    for (const field of ['minimums', 'maximums', 'first', 'last'] as const) {
      // The original JSON is exactly f32. Browser module decimal transforms need only
      // preserve these same f32 bits; the GPU comparison remains exact after normalization.
      const expected = raw.cases[sceneIndex].expected[field].map(Number);
      expect(expected.map(Math.fround)).toEqual(expected);
      expect(imported.cases[sceneIndex].expected[field].map(Number).map(Math.fround)).toEqual(
        expected
      );
    }
  }
});

it('WP6 histogram regression retains exact NumPy neighbor counts and resolved auto extents', () => {
  const automatic = histogram.cases.find(scene => scene.name === 'float-auto-unmasked-extent')!;
  expect(automatic.resolvedDomain).toEqual([-1, 8]);
  // nextDown(1) belongs to [0,1); the two zeros have only one selected row.
  expect(automatic.expected).toEqual([1, 2, 2, 1, 1, 1, 0, 0, 1]);
  for (const scene of histogram.cases.filter(scene =>
    scene.name.startsWith('rounded-offset-edge-neighbors')
  )) {
    expect(scene.expected).toHaveLength(scene.bins);
    // Exactly the two outer nextafter values lie outside the inclusive domain.
    expect(scene.expected.reduce((total, count) => total + count, 0)).toBe(scene.values.length - 2);
  }
});
