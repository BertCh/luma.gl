// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, readdirSync, statSync} from 'node:fs';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {classifyCriticalPoints} from './terrain-features/terrain-critical-points-oracle';
import {computeTerrainFlowField} from './terrain-flow-field/terrain-flow-field-oracle';
import {
  compareContourSegments,
  decodeSegments
} from './terrain-analysis/parity-wp5-contour-comparison';
import {ringSymmetricDifference} from '../oracles/parity-wp4-contour-compare';
import {
  CRITICAL_CASES,
  CRITICAL_FIXTURES,
  FLOW_CASES,
  FLOW_FIXTURES,
  CONTOUR_CASES,
  CONTOUR_FIXTURES,
  EXACT_TOLERANCE,
  CRITICAL_REFERENCE,
  FLOW_REFERENCE,
  compareValues
} from './parity-wp5-round6-fixtures';

it('Round 6 fixture provenance, versions, full arrays and size budgets are intact', () => {
  const hash = (contents: Uint8Array) => createHash('sha256').update(contents).digest('hex');
  const generatorHash = hash(
    readFileSync(new URL('../oracles/parity-wp5-round6.py', import.meta.url))
  );
  for (const fixture of [...CRITICAL_FIXTURES, ...FLOW_FIXTURES, ...CONTOUR_FIXTURES]) {
    expect(fixture.generatorSha256).toBe(generatorHash);
    for (const library of ['python', 'numpy', 'scipy', 'scikit-image', 'shapely', 'rasterio'])
      expect(fixture.versions[library]).toBeTruthy();
    expect(fixture.conventions).toContain('float64 arithmetic on float32 inputs');
    for (const entry of fixture.cases) {
      expect(entry.elevation).toHaveLength(entry.width * entry.height);
      expect(hash(new Uint8Array(Float32Array.from(entry.elevation).buffer))).toBe(
        entry.elevationSha256
      );
      expect(entry.validity).toHaveLength(entry.width * entry.height);
      if (entry.source.kind === 'Alps DEM') {
        expect(entry.source.fullShape).toEqual([2048, 2048]);
        expect(entry.source.coreSize).toBe(96);
        expect(entry.source.fullReferenceVerified || entry.source.unitedReferenceVerified).toBe(
          true
        );
      }
    }
  }
  const sourceDirectory = new URL(
    '../../../../examples/deck/spatial-analysis-showcase/public/data/alps-dem/',
    import.meta.url
  );
  const sourceHash = hash(readFileSync(new URL('dem.png', sourceDirectory)));
  const manifestHash = hash(readFileSync(new URL('manifest.json', sourceDirectory)));
  for (const fixture of [...CRITICAL_CASES, ...FLOW_CASES, ...CONTOUR_CASES])
    if (fixture.source.kind === 'Alps DEM') {
      expect(fixture.source.sha256).toBe(sourceHash);
      expect(fixture.source.manifestSha256).toBe(manifestHash);
    }
  const directory = new URL('../oracles/fixtures/', import.meta.url);
  const files = readdirSync(directory).filter(file => file.startsWith('parity-wp5-round6-'));
  expect(files).toHaveLength(11);
  for (const file of files)
    expect(statSync(new URL(file, directory)).size).toBeLessThanOrEqual(1500000);
});

it('independent lower-link reference agrees with the existing CPU contract after public band calibration', () => {
  expect(CRITICAL_CASES).toHaveLength(16);
  for (const fixture of CRITICAL_CASES) {
    const validity = Uint32Array.from(fixture.elevation, (value, index) =>
      Number(Number.isFinite(value) && value !== -9999 && fixture.validity[index] !== 0)
    );
    const elevation = Float32Array.from(fixture.elevation, value =>
      Math.fround(value * fixture.scale + fixture.offset)
    );
    const actual = classifyCriticalPoints(
      elevation,
      validity,
      fixture.width,
      fixture.height,
      fixture.connectivity
    );
    for (const metric of ['classes', 'signChanges', 'counts'] as const)
      compareValues(
        'GPUTerrainCriticalPoints CPU contract only',
        fixture.name,
        CRITICAL_REFERENCE,
        metric,
        Array.from(actual[metric]),
        fixture[metric],
        EXACT_TOLERANCE
      );
    expect(fixture.counts.reduce((sum, value) => sum + value, 0)).toBe(
      fixture.width * fixture.height
    );
    expect(
      fixture.signChanges.every(value => value % 2 === 0 && value <= fixture.connectivity)
    ).toBe(true);
  }
});

it('independent 3D flow projection agrees with the existing CPU contract and unchanged R bounds', () => {
  expect(FLOW_CASES).toHaveLength(12);
  for (const fixture of FLOW_CASES) {
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const elevation = Float32Array.from(fixture.elevation, (value, index) =>
      Number.isFinite(value) && value !== -9999 && fixture.validity[index] !== 0
        ? Math.fround(value * fixture.scale + fixture.offset)
        : NaN
    );
    for (const [windIndex, wind] of fixture.winds.entries()) {
      const actual = Array.from(
        computeTerrainFlowField({
          elevation,
          grid,
          settings: {wind, verticalExaggeration: fixture.exaggeration}
        })
      );
      expect(fixture.expected[windIndex]).toHaveLength(fixture.width * fixture.height * 2);
      for (const component of [0, 1]) {
        const expected = fixture.expected[windIndex].filter((_, index) => index % 2 === component);
        const finite = expected.filter(Number.isFinite);
        const tolerance = fixture.tolerances[windIndex][component];
        const absolute =
          1e-5 * Math.max(1, finite.length ? Math.max(...finite) - Math.min(...finite) : 0);
        expect(tolerance.profile).toBe('R');
        expect(tolerance.relative).toBe(1e-4);
        // JSON parsing and the independently recomputed f64 subtraction/product
        // can differ by a few f64 ULPs. This checks metadata only; it does not
        // change the stored R tolerance used in any GPU comparison.
        expect(Math.abs(tolerance.absolute - absolute)).toBeLessThanOrEqual(
          4 * Number.EPSILON * absolute
        );
        compareValues(
          'GPUTerrainFlowField CPU contract only',
          fixture.name,
          FLOW_REFERENCE,
          `wind-${windIndex}/component-${component}`,
          actual.filter((_, index) => index % 2 === component),
          expected,
          fixture.tolerances[windIndex][component]
        );
      }
    }
  }
});

it('plane, flat, singleton and zero-exaggeration references satisfy analytic flow identities', () => {
  for (const name of ['flat', 'singleton', 'zero-exaggeration', 'zero-wind']) {
    const fixture = FLOW_CASES.find(entry => entry.name === name)!;
    for (const [windIndex, wind] of fixture.winds.entries())
      compareValues(
        'GPUTerrainFlowField reference integrity',
        name,
        'Analytic horizontal plane or zero wind',
        `wind-${windIndex}`,
        fixture.expected[windIndex],
        fixture.expected[windIndex].map((_, index) => wind[index % 2]),
        EXACT_TOLERANCE
      );
  }
  const plane = FLOW_CASES.find(entry => entry.name === 'anisotropic-plane')!;
  // z=100+3*column-2*row, dx=2, dy=3: g=(3/2,-2/3).
  const gradient = [3 / 2, -2 / 3];
  for (const [windIndex, wind] of plane.winds.entries()) {
    const projection =
      (wind[0] * gradient[0] + wind[1] * gradient[1]) / (1 + gradient[0] ** 2 + gradient[1] ** 2);
    const expected = wind.map((component, index) => component - projection * gradient[index]);
    compareValues(
      'GPUTerrainFlowField reference integrity',
      plane.name,
      'Analytic plane projection',
      `wind-${windIndex}`,
      plane.expected[windIndex],
      plane.expected[windIndex].map((_, index) => expected[index % 2]),
      {profile: 'float64 oracle check', absolute: 1e-13, relative: 0}
    );
  }
});

it('contour fixtures retain full edge topology, safe corner margins and closed-ring geometry', () => {
  expect(CONTOUR_CASES).toHaveLength(10);
  for (const fixture of CONTOUR_CASES) {
    const bound = 8 * 2 ** (Math.floor(Math.log2(Math.max(fixture.width, fixture.height))) - 23);
    expect(fixture.tolerance).toEqual({profile: 'G', absolute: bound, relative: 0});
    for (const level of fixture.expected) {
      const coordinates = level.segments.flatMap(segment => segment.slice(2));
      // Fixture topology inference is unambiguous: variable coordinates stay more
      // than two f32 ULPs from corners. Never treat an ambiguous grid corner as a pass.
      for (const coordinate of coordinates) {
        const local = coordinate - (fixture.point ? 0 : 0.5);
        if (!Number.isInteger(local))
          expect(Math.abs(local - Math.round(local))).toBeGreaterThan(bound / 4);
      }
      expect(decodeSegments(coordinates, fixture).map(segment => segment.slice(0, 2))).toEqual(
        level.segments.map(segment => segment.slice(0, 2))
      );
      compareContourSegments(fixture, level, coordinates);
    }
  }
});

it('contour area metric measures symmetric difference and detects opposite saddle pairings', () => {
  const first = [0, 0, 2, 0, 2, 2, 0, 2, 0, 0];
  const second = [1, 0, 3, 0, 3, 2, 1, 2, 1, 0];
  expect(ringSymmetricDifference(first, second)).toBe(4);
  const firstSaddle = CONTOUR_CASES.find(entry => entry.name === 'saddle-0')!;
  const secondSaddle = CONTOUR_CASES.find(entry => entry.name === 'saddle-1')!;
  expect(firstSaddle.expected[0].segments.map(segment => segment.slice(0, 2))).not.toEqual(
    secondSaddle.expected[0].segments.map(segment => segment.slice(0, 2))
  );
  expect(
    CONTOUR_CASES.find(entry => entry.name === 'closed-nested-hills')!.expected.map(
      level => level.rings.length
    )
  ).toEqual([1, 1, 1]);
});
