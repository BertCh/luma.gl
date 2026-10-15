// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, readdirSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
import {expect, it} from 'vitest';
import {
  decodeRound4Fixture,
  type PackedFixture,
  type FocalCase,
  type ZonalCase,
  type GeographicCase
} from '../../oracles/parity-wp1-round4-fixture';

const directory = new URL('../../oracles/fixtures/', import.meta.url);
const files = readdirSync(directory).filter(name =>
  /^parity-wp1-(zonal|focal|geographic)-.*\.json$/.test(name)
);
function loadFixture(name: string): PackedFixture {
  return JSON.parse(readFileSync(new URL(`parity-wp1-${name}.json`, directory), 'utf8'));
}
function hashBytes(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}
function float64Bytes(values: number[]): Uint8Array {
  return new Uint8Array(Float64Array.from(values).buffer);
}

it('Round 4 has every expected compact full-output fixture', () => {
  expect(files).toHaveLength(22);
});
for (const file of files) {
  it(`Round 4 fixture provenance and lossless transport: ${file}`, async () => {
    const bytes = readFileSync(new URL(file, directory));
    expect(bytes.byteLength).toBeLessThanOrEqual(1500000);
    const fixture: PackedFixture = JSON.parse(bytes.toString());
    expect(fixture.generator).toBe('parity-wp1-round4.py');
    expect(fixture.encoding).toBe('zlib-json');
    for (const library of ['numpy', 'pandas', 'scipy', 'shapely', 'pyproj', 'pointpats'])
      expect(fixture.versions[library]).toBeTruthy();
    const payload = inflateSync(Buffer.from(fixture.data, 'base64'));
    expect(hashBytes(payload)).toBe(fixture.sha256);
    // Both Node zlib and the actual browser-capable decoder must retain NaN/Inf.
    const decoded = await decodeRound4Fixture<Record<string, unknown>>(fixture);
    expect(decoded).toHaveProperty(JSON.parse(payload.toString()).cases ? 'cases' : 'expected');
  });
}

it('Zonal reference retains every observation and every tract, including unassigned observations', async () => {
  const testCase = await decodeRound4Fixture<ZonalCase>(loadFixture('zonal-real'));
  expect(testCase.n).toBe(43557);
  expect(testCase.featureCount).toBe(791);
  expect(testCase.expected.pointFeatureRows).toHaveLength(testCase.n);
  const reconstructed = Array(testCase.featureCount).fill(0);
  for (const feature of testCase.expected.pointFeatureRows)
    if (feature < testCase.featureCount) reconstructed[feature]++;
  expect(reconstructed).toEqual(testCase.expected.counts);
  expect(testCase.expected.valueCounts).toEqual(testCase.expected.counts);
  for (const [key, values] of Object.entries(testCase.expected)) {
    if (!['extent', 'pointFeatureRows', 'overflow', 'uncertainCount'].includes(key))
      expect(values).toHaveLength(testCase.featureCount);
  }
  const positions = new Uint8Array(Float32Array.from(testCase.positions).buffer);
  const values = new Uint8Array(Float32Array.from(testCase.values).buffer);
  expect(hashBytes(Buffer.concat([positions, values]))).toBe(testCase.inputFloat32Sha256);
});

it('Zonal synthetic reference distinguishes holes, overlaps, empty features, invalid values and reliability weights', async () => {
  const {cases} = await decodeRound4Fixture<{cases: ZonalCase[]}>(loadFixture('zonal-small'));
  for (const testCase of cases) {
    const expected = testCase.expected;
    expect(expected.pointFeatureRows[2]).toBe(0xffffffff); // hole interior
    expect(expected.pointFeatureRows[4]).toBe(0); // both features; smallest wins
    expect(expected.pointFeatureRows[3]).toBe(testCase.boundaryRule === 'closed' ? 0 : 0xffffffff);
    expect(expected.counts[3]).toBe(0);
    expect(expected.means[3]).toBeNaN();
    expect(expected.valueCounts[2]).toBe(1); // second disconnected polygon has NaN
    expect(expected.variances[2]).toBeNaN();
    if (testCase.weights.length) {
      expect(expected.weightSums[4]).toBe(0);
      expect(expected.means[4]).toBeNaN();
      expect(expected.variances[4]).toBeNaN();
    }
  }
});

for (const mode of ['square', 'circle']) {
  it(`Focal ${mode} stripes cover all 512x512 cells and preserve exact full-domain SciPy seam hashes`, async () => {
    let coveredRows = 0;
    let previous: FocalCase | undefined;
    for (let stripe = 0; stripe < 8; stripe++) {
      const testCase = await decodeRound4Fixture<FocalCase>(loadFixture(`focal-${mode}-${stripe}`));
      const [top, bottom] = testCase.provenance['stripe'] as number[];
      const [start, stop] = testCase.provenance['core'] as number[];
      expect(top + start).toBe(coveredRows);
      expect(stop - start).toBe(64);
      expect(testCase.width).toBe(512);
      expect(testCase.height).toBe(bottom - top);
      expect(hashBytes(new Uint8Array(Float32Array.from(testCase.values).buffer))).toBe(
        testCase.inputFloat32Sha256
      );
      for (const [key, values] of Object.entries(testCase.expected)) {
        expect(values).toHaveLength(testCase.n);
        expect(hashBytes(float64Bytes(values.slice(start * 512, stop * 512)))).toBe(
          testCase.coreSha256![key]
        );
      }
      if (previous) {
        const [previousTop, previousBottom] = previous.provenance['stripe'] as number[];
        expect(testCase.values.slice(0, (previousBottom - top) * 512)).toEqual(
          previous.values.slice((top - previousTop) * 512)
        );
      }
      previous = testCase;
      coveredRows += stop - start;
    }
    expect(coveredRows).toBe(512);
  });
}

it('Focal synthetic cases exercise empty windows, zero radius, minimum counts and propagated nodata', async () => {
  const {cases} = await decodeRound4Fixture<{cases: FocalCase[]}>(loadFixture('focal-small'));
  expect(cases).toHaveLength(7);
  const zeroRadius = cases[0];
  expect(zeroRadius.expected.counts[28]).toBe(1); // +Infinity is a value, unlike NaN.
  expect(zeroRadius.expected.mean[28]).toBe(Infinity);
  expect(zeroRadius.expected.standardDeviation[28]).toBeNaN();
  expect(cases[5].expected.counts.every(value => value === 0)).toBe(true);
  expect(cases[5].expected.mean.every(Number.isNaN)).toBe(true);
  expect(cases[1].expected.counts).toEqual(cases[4].expected.counts);
  expect(cases[4].expected.mean[8]).toBeNaN();
  expect(cases[1].expected.mean[8]).toBeTypeOf('number');
  expect(Number.isFinite(cases[1].expected.mean[8])).toBe(true);
});

for (const mode of ['all', 'weighted-masked']) {
  it(`Geographic reference includes all full-data inputs and empty group: ${mode}`, async () => {
    const testCase = await decodeRound4Fixture<GeographicCase>(loadFixture(`geographic-${mode}`));
    expect(testCase.n).toBe(43557);
    expect(testCase.positions).toHaveLength(testCase.n * 2);
    expect(testCase.groups).toHaveLength(testCase.n);
    expect(testCase.expected.counts).toHaveLength(11);
    expect(testCase.expected.counts[10]).toBe(0);
    expect(testCase.expected.meanCenters[20]).toBeNaN();
    expect(hashBytes(new Uint8Array(Float32Array.from(testCase.positions).buffer))).toBe(
      testCase.inputFloat32Sha256
    );
    expect(testCase.expected.counts.reduce((total, value) => total + value, 0)).toBe(
      testCase.mask.filter(Boolean).length
    );
    for (let group = 0; group < 10; group++) {
      const minor = testCase.expected.ellipses[group * 3 + 1];
      const major = testCase.expected.ellipses[group * 3 + 2];
      expect(major).toBeGreaterThanOrEqual(minor);
      expect(Math.hypot(minor, major)).toBeCloseTo(testCase.expected.standardDistances[group], 10);
    }
  });
}
