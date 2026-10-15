// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, readdirSync, statSync} from 'node:fs';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getGPUTerrainPeakSnapParameterValues} from '../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';
import {getGPUTerrainSummitsParameterValues} from '../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {
  computeTerrainPeakSnap,
  computeTerrainSummits
} from './terrain-features/terrain-summits-oracle';
import {
  SNAP_FIXTURES,
  SNAP_CASES,
  SUMMITS_FIXTURES,
  SUMMITS_CASES,
  SNAP_REFERENCE,
  SUMMITS_REFERENCE,
  DISTANCE_TOLERANCE,
  compareValues
} from './parity-wp5-round7-fixtures';

it('Round 7 provenance, full-raster checks, complete outputs and fixture budgets are retained', () => {
  const hash = (contents: Uint8Array) => createHash('sha256').update(contents).digest('hex');
  const generatorHash = hash(
    readFileSync(new URL('../oracles/parity-wp5-round7.py', import.meta.url))
  );
  const data = new URL(
    '../../../../examples/deck/spatial-analysis-showcase/public/data/alps-dem/',
    import.meta.url
  );
  const sourceHash = hash(readFileSync(new URL('dem.png', data)));
  const manifestHash = hash(readFileSync(new URL('manifest.json', data)));
  for (const fixture of [...SNAP_FIXTURES, ...SUMMITS_FIXTURES]) {
    expect(fixture.generatorSha256).toBe(generatorHash);
    for (const library of ['python', 'numpy', 'scipy', 'rasterio', 'gdal'])
      expect(fixture.versions[library]).toBeTruthy();
    expect(fixture.conventions).toContain('COMPOSITE');
    for (const entry of fixture.cases) {
      expect(entry.elevation).toHaveLength(entry.width * entry.height);
      expect(entry.validity).toHaveLength(entry.width * entry.height);
      expect(hash(new Uint8Array(Float32Array.from(entry.elevation).buffer))).toBe(
        entry.elevationSha256
      );
      if (entry.source.kind === 'Alps DEM') {
        expect(entry.source.sha256).toBe(sourceHash);
        expect(entry.source.manifestSha256).toBe(manifestHash);
        expect(entry.source.fullShape).toEqual([2048, 2048]);
        expect(entry.source.coreSize).toBe(96);
        expect(entry.source.halo).toBe(6);
        expect(entry.source.fullReferenceVerified).toBe(true);
      }
    }
  }
  const directory = new URL('../oracles/fixtures/', import.meta.url);
  const files = readdirSync(directory).filter(file => file.startsWith('parity-wp5-round7-'));
  expect(files).toHaveLength(6);
  for (const file of files)
    expect(statSync(new URL(file, directory)).size).toBeLessThanOrEqual(1500000);
});

function calibrateValues(fixture: {
  elevation: number[];
  validity: number[];
  scale: number;
  offset: number;
}): Float32Array {
  return Float32Array.from(fixture.elevation, (value, index) =>
    Number.isFinite(value) && value !== -9999 && fixture.validity[index] !== 0
      ? Math.fround(value * fixture.scale + fixture.offset)
      : NaN
  );
}

it('independent KD-tree and interpolation reference agrees with the existing snap CPU contract', () => {
  expect(SNAP_CASES).toHaveLength(17);
  for (const fixture of SNAP_CASES) {
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const actual = computeTerrainPeakSnap(
      calibrateValues(fixture),
      undefined,
      fixture.width,
      fixture.height,
      fixture.candidates,
      fixture.candidateHeights,
      fixture.candidateRadii,
      getGPUTerrainPeakSnapParameterValues(fixture.settings, grid),
      {maximumRadiusPixels: fixture.maximumRadiusPixels}
    );
    for (const metric of ['positions', 'heights', 'status', 'snapDistance'] as const) {
      expect(fixture.expected[metric]).toHaveLength(
        fixture.candidates.length / (metric === 'positions' ? 1 : 2)
      );
      compareValues(
        'GPUTerrainPeakSnap CPU contract only',
        fixture.name,
        SNAP_REFERENCE,
        metric,
        actual[metric],
        fixture.expected[metric],
        metric === 'heights'
          ? fixture.expected.heightTolerance
          : metric === 'snapDistance'
            ? DISTANCE_TOLERANCE
            : undefined
      );
    }
    compareValues(
      'GPUTerrainPeakSnap CPU contract only',
      fixture.name,
      SNAP_REFERENCE,
      'radiusClamped',
      [actual.overflow],
      [fixture.expected.radiusClamped]
    );
  }
});

it('independent morphology reference agrees exactly with the existing summit CPU contract', () => {
  expect(SUMMITS_CASES).toHaveLength(20);
  for (const fixture of SUMMITS_CASES) {
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const actual = computeTerrainSummits(
      calibrateValues(fixture),
      undefined,
      fixture.width,
      fixture.height,
      getGPUTerrainSummitsParameterValues(
        {radius: fixture.radius, minimumDrop: fixture.minimumDrop},
        grid
      ),
      {maximumRadiusPixels: fixture.maximumRadiusPixels, incompleteNeighborhood: fixture.policy}
    );
    for (const metric of ['mask', 'drop', 'ids'] as const)
      compareValues(
        'GPUTerrainSummits CPU contract only',
        fixture.name,
        SUMMITS_REFERENCE,
        metric,
        actual[metric],
        fixture.expected[metric]
      );
    expect(fixture.expected.mask).toHaveLength(fixture.width * fixture.height);
    expect(fixture.expected.drop).toHaveLength(fixture.width * fixture.height);
    expect(fixture.expected.ids).toEqual(
      fixture.expected.mask.flatMap((value, index) => (value ? [index] : []))
    );
    compareValues(
      'GPUTerrainSummits CPU contract only',
      fixture.name,
      SUMMITS_REFERENCE,
      'radiusClamped',
      [actual.overflow],
      [fixture.radiusClamped]
    );
  }
});

it('references exercise all snap states, plateau tie selection and empty-ring infinity', () => {
  expect([...new Set(SNAP_CASES.flatMap(fixture => fixture.expected.status))].sort()).toEqual([
    0, 1, 2, 3, 4, 5, 6
  ]);
  const plateau = SUMMITS_CASES.find(fixture => fixture.name === 'plateau-reject')!;
  expect(plateau.expected.ids).toEqual([7 * 19 + 8]);
  expect(SUMMITS_CASES.find(fixture => fixture.name === 'singleton-ignore')!.expected.drop).toEqual(
    [Infinity]
  );
  expect(SUMMITS_CASES.find(fixture => fixture.name === 'singleton-reject')!.expected.ids).toEqual(
    []
  );
  expect(SNAP_CASES.find(fixture => fixture.name === 'empty-disc')!.expected.status).toEqual([5]);
  const invalid = SNAP_CASES.find(fixture => fixture.name === 'invalid-interpolation')!;
  expect(invalid.expected.status).toEqual([0]);
  expect(invalid.expected.heights[0]).toBeNaN();
  expect(SUMMITS_CASES.some(fixture => fixture.expected.ids.length > 1)).toBe(true);
});

it('height tolerances retain the R formula and exact policies remain exact', () => {
  for (const fixture of SNAP_CASES) {
    const finite = fixture.expected.heights.filter(Number.isFinite);
    const absolute =
      1e-5 * Math.max(1, finite.length ? Math.max(...finite) - Math.min(...finite) : 0);
    const tolerance = fixture.expected.heightTolerance;
    expect(tolerance.profile).toBe('R');
    expect(tolerance.relative).toBe(1e-4);
    // Four f64 ULPs cover JSON decoding and subtraction/product recomputation.
    // This metadata check never changes the stored GPU tolerance.
    expect(Math.abs(tolerance.absolute - absolute)).toBeLessThanOrEqual(
      4 * Number.EPSILON * absolute
    );
  }
});
