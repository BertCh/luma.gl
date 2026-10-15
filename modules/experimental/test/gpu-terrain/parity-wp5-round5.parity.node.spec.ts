// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, readdirSync, statSync} from 'node:fs';
import {expect, it} from 'vitest';
import {findProfilePeaksOracle} from './terrain-features/profile-peaks-oracle';
import {
  BLEND_CASES,
  BLEND_FIXTURES,
  RELIEF_CASES,
  RELIEF_FIXTURES,
  PEAK_CASES,
  PEAK_FIXTURES,
  EXACT_TOLERANCE,
  compareRound5Values
} from './parity-wp5-round5-fixtures';

it('Round-5 fixtures identify their unchanged generator, source data, versions and size limits', () => {
  const generator = readFileSync(new URL('../oracles/parity-wp5-round5.py', import.meta.url));
  const generatorHash = createHash('sha256').update(generator).digest('hex');
  for (const fixture of [...RELIEF_FIXTURES, ...BLEND_FIXTURES, ...PEAK_FIXTURES]) {
    expect(fixture.generatorSha256).toBe(generatorHash);
    for (const library of ['python', 'numpy', 'scipy', 'rvt-py', 'rasterio', 'matplotlib']) {
      expect(fixture.versions[library]).toBeTruthy();
    }
    expect(fixture.conventions).toContain('Float64 reference on f32 inputs');
  }
  const sourceHashes = new Set<string>();
  for (const fixture of [...RELIEF_CASES, ...BLEND_CASES, ...PEAK_CASES]) {
    if (fixture.source.kind !== 'Alps DEM') continue;
    expect(fixture.source.fullShape).toEqual([2048, 2048]);
    sourceHashes.add(fixture.source.sha256!);
  }
  const sourceDirectory = new URL(
    '../../../../examples/deck/spatial-analysis-showcase/public/data/alps-dem/',
    import.meta.url
  );
  expect([...sourceHashes]).toEqual([
    createHash('sha256')
      .update(readFileSync(new URL('dem.png', sourceDirectory)))
      .digest('hex')
  ]);
  const manifestHash = createHash('sha256')
    .update(readFileSync(new URL('manifest.json', sourceDirectory)))
    .digest('hex');
  for (const fixture of [...RELIEF_CASES, ...BLEND_CASES, ...PEAK_CASES]) {
    if (fixture.source.kind === 'Alps DEM')
      expect(fixture.source.manifestSha256).toBe(manifestHash);
  }
  const fixtureDirectory = new URL('../oracles/fixtures/', import.meta.url);
  const files = readdirSync(fixtureDirectory).filter(name => name.startsWith('parity-wp5-round5-'));
  expect(files).toHaveLength(11);
  for (const file of files)
    expect(statSync(new URL(file, fixtureDirectory)).size).toBeLessThanOrEqual(1_500_000);
});

it('simple local relief retains full arrays, exact nodata masks and full-raster halo checks', () => {
  expect(RELIEF_CASES).toHaveLength(10);
  for (const fixture of RELIEF_CASES) {
    const count = fixture.width * fixture.height;
    for (const values of [
      fixture.elevation,
      fixture.validity,
      fixture.expected,
      fixture.expectedValidity
    ])
      expect(values).toHaveLength(count);
    expect(fixture.expectedValidity).toEqual(
      fixture.elevation.map((value, index) =>
        Number(Number.isFinite(value) && value !== -9999 && fixture.validity[index] !== 0)
      )
    );
    expect(fixture.expected.map(value => Number(Number.isFinite(value)))).toEqual(
      fixture.expectedValidity
    );
    if (fixture.rvtCrossCheck.performed)
      expect(fixture.rvtCrossCheck.maxAbs).toBeLessThanOrEqual(
        fixture.rvtCrossCheck.derivedAbsoluteBound
      );
    if (fixture.source.kind === 'Alps DEM') {
      expect(fixture.source.fullFilterVerified).toBe(true);
      expect(fixture.source.halo).toBeGreaterThanOrEqual(fixture.radius);
      expect(fixture.width).toBe(fixture.source.coreSize! + 2 * fixture.source.halo!);
    }
  }
  const flat = RELIEF_CASES.find(fixture => fixture.name === 'flat-high-offset')!;
  compareRound5Values(
    'GPUSimpleLocalRelief reference integrity',
    flat.name,
    'Constant surface analytic identity',
    'relief',
    flat.expected,
    flat.expected.map(() => 0),
    EXACT_TOLERANCE
  );
  const row = RELIEF_CASES.find(fixture => fixture.name === 'single-row-replicated')!;
  compareRound5Values(
    'GPUSimpleLocalRelief reference integrity',
    row.name,
    'Hand-computed radius-one normalized box mean',
    'relief',
    row.expected,
    [-1 / 3, -2, 3.5, NaN, 0.5, -1 / 3],
    {profile: 'float64 analytic check', absolute: 1e-12, relative: 0}
  );
});

it('RVT blend fixtures cover six modes, clipping thresholds, opacity and every encoded channel', () => {
  expect(BLEND_CASES).toHaveLength(22);
  for (const fixture of BLEND_CASES) {
    const count = fixture.width * fixture.height;
    for (const values of [
      ...fixture.layers,
      fixture.expected,
      fixture.expectedValidity,
      fixture.expectedColor
    ])
      expect(values).toHaveLength(count);
    expect(fixture.expectedValidity).toEqual(
      fixture.expected.map((_, index) =>
        Number(fixture.layers.every(layer => Number.isFinite(layer[index])))
      )
    );
    for (let index = 0; index < count; index++) {
      const color = fixture.expectedColor[index];
      if (!fixture.expectedValidity[index]) {
        expect(Number.isNaN(fixture.expected[index])).toBe(true);
        expect(color).toBe(0);
      } else {
        expect(fixture.expected[index]).toBeGreaterThanOrEqual(0);
        expect(fixture.expected[index]).toBeLessThanOrEqual(1);
        expect(color >>> 24).toBe(255);
        expect(color & 255).toBe((color >>> 8) & 255);
        expect(color & 255).toBe((color >>> 16) & 255);
      }
    }
    if (fixture.layers.length === 2 && fixture.settings[1].opacity === 0) {
      compareRound5Values(
        'GPUReliefBlend reference integrity',
        fixture.name,
        'Zero-opacity identity',
        'blend',
        fixture.expected,
        fixture.layers[0].map(value => Math.min(1, Math.max(0, value))),
        EXACT_TOLERANCE
      );
    }
  }
  const invalid = BLEND_CASES.find(
    fixture => fixture.name === 'five-layer-invalid-even-zero-opacity'
  )!;
  expect(invalid.expectedValidity.slice(0, 5)).toEqual([0, 0, 0, 0, 0]);
});

it('SciPy peak fixtures preserve plateau left edges, empty CSR spans, ties, seams and complete transects', () => {
  expect(PEAK_CASES).toHaveLength(5);
  for (const fixture of PEAK_CASES) {
    const count = fixture.values.length;
    expect(fixture.offsets[0]).toBe(0);
    expect(fixture.offsets.at(-1)).toBe(count);
    for (const values of [
      fixture.validity,
      fixture.expectedMask,
      ...Object.values(fixture.expected)
    ])
      expect(values).toHaveLength(count);
    expect(fixture.expectedIds).toEqual(
      fixture.expectedMask.flatMap((value, index) => (value ? [index] : []))
    );
    for (const values of Object.values(fixture.expected))
      expect(values.map(value => Number(Number.isFinite(value)))).toEqual(fixture.expectedMask);
    expect(fixture.suppressionBoundaryMargin).toBeGreaterThan(0.001);
    if (fixture.source.kind === 'Alps DEM') {
      expect(fixture.source.fullTransects).toBe(true);
      expect(count).toBe(8 * 2048);
      expect(fixture.offsets).toEqual(Array.from({length: 9}, (_, index) => index * 2048));
    }
  }
  const plateaus = PEAK_CASES.find(fixture => fixture.name === 'plateaus-gaps-empty-and-short')!;
  expect(plateaus.offsets.slice(0, 4)).toEqual([0, 0, 1, 3]);
  // Odd plateau [4,4,4] and even plateau [4,4] both use the left edge;
  // this explicitly differs from SciPy's default midpoint representative.
  expect(plateaus.expectedIds.slice(0, 2)).toEqual([5, 12]);
  const suppressed = PEAK_CASES.find(
    fixture => fixture.name === 'prominence-ranked-tied-suppression'
  )!;
  expect(suppressed.expectedIds.length).toBeLessThan(plateaus.expectedIds.length);
  const circular = PEAK_CASES.find(fixture => fixture.wrap)!;
  expect(circular.expectedMask[0]).toBe(1);
  expect(circular.expected.refinedIndex[0]).toBeCloseTo(0, 10);
});

it('independent SciPy fixtures agree with the existing CPU contract (this does not execute a GPU)', () => {
  for (const fixture of PEAK_CASES) {
    const result = findProfilePeaksOracle(
      Float32Array.from(fixture.values),
      Uint32Array.from(fixture.validity),
      fixture.offsets,
      {...fixture, minimumSide: 1}
    );
    const className = 'GPUProfilePeaks CPU contract only';
    const reference = 'Independent SciPy fixture versus existing CPU contract; no GPU execution';
    compareRound5Values(
      className,
      fixture.name,
      reference,
      'mask',
      Array.from(result.mask),
      fixture.expectedMask,
      EXACT_TOLERANCE
    );
    for (const metric of ['prominence', 'refinedIndex', 'refinedValue'] as const) {
      compareRound5Values(
        className,
        fixture.name,
        reference,
        metric,
        Array.from(result[metric]),
        fixture.expected[metric],
        fixture.tolerances[metric]
      );
    }
  }
});

it('all GPU float tolerances remain the original per-output R profile', () => {
  const comparisons = [
    ...RELIEF_CASES.map(fixture => ({values: fixture.expected, tolerance: fixture.tolerance})),
    ...BLEND_CASES.map(fixture => ({values: fixture.expected, tolerance: fixture.tolerance})),
    ...PEAK_CASES.flatMap(fixture =>
      (['prominence', 'refinedIndex', 'refinedValue'] as const).map(metric => ({
        values: fixture.expected[metric],
        tolerance: fixture.tolerances[metric]
      }))
    )
  ];
  for (const {values, tolerance} of comparisons) {
    const finite = values.filter(Number.isFinite);
    const range = finite.length ? Math.max(...finite) - Math.min(...finite) : 0;
    expect(tolerance.profile).toBe('R');
    expect(tolerance.relative).toBe(1e-4);
    // Only Python/JS float64 metadata arithmetic differs here; four double
    // epsilons cover JSON and multiplication rounding, not GPU output error.
    const expectedAbsolute = 1e-5 * Math.max(1, range);
    expect(Math.abs(tolerance.absolute - expectedAbsolute)).toBeLessThanOrEqual(
      4 * Number.EPSILON * expectedAbsolute
    );
  }
});
