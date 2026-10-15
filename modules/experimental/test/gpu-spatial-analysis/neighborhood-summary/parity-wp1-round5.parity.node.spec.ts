// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
import {expect, it} from 'vitest';
import {decodeRound4Fixture, type PackedFixture} from '../../oracles/parity-wp1-round4-fixture';
import type {
  NeighborhoodCase,
  PartitionCase,
  RegionCase
} from '../../oracles/parity-wp1-round5-fixture';

const directory = new URL('../../oracles/fixtures/', import.meta.url);
const sourceDirectory = new URL(
  '../../../../../examples/deck/spatial-analysis-showcase/public/data/',
  import.meta.url
);
function loadFixture(name: string): PackedFixture {
  return JSON.parse(readFileSync(new URL(`parity-wp1-round5-${name}.json`, directory), 'utf8'));
}
function hashBytes(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}
function hashFloats(...arrays: number[][]): string {
  return hashBytes(
    Buffer.concat(arrays.map(values => Buffer.from(Float32Array.from(values).buffer)))
  );
}
for (const name of ['neighborhood', 'partition', 'region-real', 'region-small']) {
  it(`Round 5 lossless fixture, versions and all source hashes: ${name}`, async () => {
    const bytes = readFileSync(new URL(`parity-wp1-round5-${name}.json`, directory));
    expect(bytes.byteLength).toBeLessThanOrEqual(1500000);
    const fixture = loadFixture(name);
    expect(fixture.generator).toBe('parity-wp1-round5.py');
    expect(fixture.encoding).toBe('zlib-json');
    for (const library of [
      'python',
      'numpy',
      'pandas',
      'scipy',
      'sklearn',
      'shapely',
      'pyproj',
      'libpysal'
    ])
      expect(fixture.versions[library]).toBeTruthy();
    const payload = inflateSync(Buffer.from(fixture.data, 'base64'));
    expect(hashBytes(payload)).toBe(fixture.sha256);
    const decoded = await decodeRound4Fixture<{
      cases?: {provenance: Record<string, unknown>}[];
      provenance: Record<string, unknown>;
    }>(fixture);
    for (const testCase of decoded.cases ?? [decoded]) {
      const hashes = testCase.provenance['sourceSha256'] as Record<string, string> | undefined;
      for (const [file, hash] of Object.entries(hashes ?? {}))
        expect(hashBytes(readFileSync(new URL(file, sourceDirectory))), file).toBe(hash);
    }
  });
}

it('Neighborhood fixtures retain all tracts, every output column and exact uploaded bytes', async () => {
  const {cases} = await decodeRound4Fixture<{cases: NeighborhoodCase[]}>(
    loadFixture('neighborhood')
  );
  expect(cases).toHaveLength(12);
  for (const testCase of cases) {
    expect(hashFloats(testCase.values, testCase.weights)).toBe(testCase.inputFloat32Sha256);
    expect(testCase.offsets).toHaveLength(testCase.n + 1);
    expect(testCase.offsets.at(-1)).toBe(testCase.neighbors.length);
    expect(testCase.weights).toHaveLength(testCase.neighbors.length);
    expect(testCase.expected.table).toHaveLength(testCase.n * 8);
    expect(testCase.expected.modes).toHaveLength(testCase.n);
    expect(testCase.expected.entropy).toHaveLength(testCase.n);
    expect(testCase.expected.overflow).toHaveLength(1);
  }
  for (const testCase of cases.slice(0, 4)) {
    expect(testCase.n).toBe(791);
    expect(testCase.provenance['sourcePlaces']).toBe(105808);
    expect(testCase.values.reduce((sum, value) => sum + value, 0)).toBe(
      105808 - Number(testCase.provenance['unassignedPlaces'])
    );
    expect(testCase.expected.overflow).toEqual([0]);
  }
});

it('Neighborhood independent reference distinguishes duplicate slots, weighted population deviation and categorical missing values', async () => {
  const {cases} = await decodeRound4Fixture<{cases: NeighborhoodCase[]}>(
    loadFixture('neighborhood')
  );
  const testCase = cases[4];
  expect(testCase.normalize).toBe(false);
  // Row 0 contains value 4 twice and value 8 once, weights 1,1,2.
  expect(testCase.expected.table.slice(0, 8)).toEqual([3, 4, 24, 6, 4, 8, 2, 4]);
  expect(testCase.expected.modes[0]).toBe(2); // frequency tie with category 7
  expect(testCase.expected.entropy[0]).toBeCloseTo(Math.log(2), 14);
  // Row 1's only neighbor has +Inf numeric value but a valid category.
  expect(testCase.expected.table.slice(8, 11)).toEqual([0, 0, 0]);
  expect(testCase.expected.table[11]).toBeNaN();
  expect(testCase.expected.modes[1]).toBe(9);
  expect(testCase.expected.entropy[1]).toBe(0);
  // Masked row 6 and unmasked island row 7 have different sum/weight semantics.
  expect(testCase.expected.table[6 * 8]).toBe(0);
  expect(testCase.expected.table[6 * 8 + 1]).toBeNaN();
  expect(testCase.expected.table.slice(7 * 8, 7 * 8 + 3)).toEqual([0, 0, 0]);
});

it('Neighborhood row normalization, zero focal weight and overflow retain their distinct contracts', async () => {
  const {cases} = await decodeRound4Fixture<{cases: NeighborhoodCase[]}>(
    loadFixture('neighborhood')
  );
  expect(cases[8].expected.table.slice(0, 8)).toEqual([3, 1, 6, 6, 4, 8, 2, 4]);
  for (const testCase of cases.filter(testCase => testCase.maximumNeighbors === 2)) {
    expect(testCase.expected.overflow).toEqual([1]);
    expect(testCase.expected.table[7]).toBeNaN();
    expect(Number.isFinite(testCase.expected.entropy[0])).toBe(true); // slow categorical scan stays valid
  }
  for (const testCase of cases.filter(testCase => testCase.focalWeight === 0)) {
    expect(testCase.expected.table.slice(7 * 8, 7 * 8 + 3)).toEqual([1, 0, 0]);
    expect(testCase.expected.table[7 * 8 + 3]).toBeNaN();
    expect(testCase.expected.table[7 * 8 + 7]).toBe(24);
    expect(testCase.expected.modes[7]).toBe(5);
    expect(testCase.expected.entropy[7]).toBeNaN();
  }
  for (const testCase of cases.filter(testCase => testCase.truncated)) {
    expect(testCase.expected.table.slice(0, 8).every(Number.isNaN)).toBe(true);
    expect(testCase.expected.table[6 * 8]).toBe(0);
    expect(testCase.expected.modes.every(value => value === 0xffffffff)).toBe(true);
    expect(testCase.expected.overflow).toEqual([0]);
  }
});

it('Partition reference retains full tract and observation populations and every label output', async () => {
  const {cases} = await decodeRound4Fixture<{cases: PartitionCase[]}>(loadFixture('partition'));
  expect(cases).toHaveLength(8);
  expect(cases[0].n).toBe(791);
  expect(cases[7].n).toBe(43557);
  expect(cases[7].graph).toBeUndefined();
  for (const testCase of cases) {
    expect(hashFloats(testCase.values)).toBe(testCase.inputFloat32Sha256);
    expect(testCase.values).toHaveLength(testCase.n * testCase.columns);
    expect(testCase.labels).toHaveLength(testCase.n);
    expect(testCase.expected.summary).toHaveLength(9);
    expect(testCase.expected.regionSizes).toHaveLength(testCase.labelCapacity);
    expect(testCase.expected.regionWithinSumOfSquares).toHaveLength(testCase.labelCapacity);
    const included = testCase.labels.filter(label => label < testCase.labelCapacity).length;
    expect(testCase.expected.regionSizes.reduce((sum, value) => sum + value, 0)).toBe(included);
    expect(testCase.expected.summary[7]).toBe(testCase.n - included);
    expect(testCase.expected.summary[1] + testCase.expected.summary[2]).toBeCloseTo(
      testCase.expected.summary[3],
      7
    );
  }
});

it('Partition reference distinguishes gapped labels, directed slot counts and degenerate pseudo-F', async () => {
  const {cases} = await decodeRound4Fixture<{cases: PartitionCase[]}>(loadFixture('partition'));
  const gapped = cases[2];
  expect(gapped.expected.regionSizes).toEqual([2, 0, 2, 0, 0, 0, 0, 0]);
  expect(gapped.expected.regionWithinSumOfSquares).toEqual([4, 0, 4, 0, 0, 0, 0, 0]);
  expect(gapped.expected.summary[6]).toBe(2 / 5); // zero/self/duplicate slots retained
  expect(cases[3].expected.summary[8]).toBeNaN(); // one region
  expect(cases[4].expected.summary.slice(0, 7)).toEqual([0, 0, 0, 0, 0, 0, 0]);
  expect(cases[4].expected.summary[8]).toBeNaN(); // all ignored
  expect(cases[5].expected.summary[8]).toBeNaN(); // all singleton
  expect(cases[6].expected.summary[8]).toBe(1); // sklearn convention, zero within SSD
});

for (const name of ['region-real', 'region-small']) {
  it(`Region reference stores every membership and stable ID, histogram mass and exact uploaded bytes: ${name}`, async () => {
    const testCase = await decodeRound4Fixture<RegionCase>(loadFixture(name));
    expect(hashFloats(testCase.positions, testCase.values)).toBe(testCase.inputFloat32Sha256);
    expect(testCase.positions).toHaveLength(testCase.n * 2);
    expect(testCase.values).toHaveLength(testCase.n);
    expect(testCase.sourceIds).toHaveLength(testCase.n);
    if (name === 'region-real') expect(testCase.n).toBe(43557);
    for (const query of testCase.queries) {
      expect(query.expected.mask).toHaveLength(testCase.n);
      expect(query.expected.ids).toEqual(
        testCase.sourceIds.filter((_, index) => query.expected.mask[index])
      );
      expect(query.expected.counts[0]).toBe(query.expected.ids.length);
      expect(query.expected.histogram).toHaveLength(query.binCount);
      if (query.binCount)
        expect(
          query.expected.histogram.reduce((sum, value) => sum + value, 0) + query.expected.counts[2]
        ).toBe(query.expected.counts[1]);
    }
  });
}

it('Region boundaries, empty and constant selections are explicit independent reference cases', async () => {
  const testCase = await decodeRound4Fixture<RegionCase>(loadFixture('region-small'));
  expect(testCase.queries[0].expected.counts).toEqual([6, 5, 0]);
  expect(testCase.queries[0].expected.mask.slice(0, 5)).toEqual([1, 1, 1, 1, 1]);
  expect(testCase.queries[2].expected.mask.slice(5, 9)).toEqual([1, 1, 1, 1]); // exact circle radius
  for (const query of testCase.queries) expect(query.expected.mask.slice(10, 12)).toEqual([0, 0]);
  expect(testCase.queries[3].expected.statistics).toEqual([0, 0, 0, 0]);
  expect(testCase.queries[4].expected.histogram).toEqual([1, 0, 0, 0]);
  expect(testCase.queries[5].expected.counts).toEqual([1, 0, 0]);
  expect(testCase.queries[5].expected.statistics).toEqual([0, 0, 0, 0]);
  const real = await decodeRound4Fixture<RegionCase>(loadFixture('region-real'));
  expect(real.queries[3].expected.counts[0]).toBe(43557);
  // Half-open lasso: these exact top/right-boundary rows are excluded.
  expect(real.queries[1].expected.mask[29299]).toBe(0);
  expect(real.queries[1].expected.mask[34963]).toBe(0);
  expect(real.queries[1].expected.counts[0]).toBe(2833);
});

it('Neighborhood mode resolves an exact rational tie to the smallest category', async () => {
  const fixture = await decodeRound4Fixture<{cases: NeighborhoodCase[]}>(
    loadFixture('neighborhood')
  );
  // Row 689: category 1 has 1/2 + 1/10; category 3 has 6/10.
  expect(fixture.cases[3].expected.modes[689]).toBe(1);
});
