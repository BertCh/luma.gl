// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {getGPUMultiScaleReliefRadii} from '../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  WATERSHED_CASES,
  WATERSHED_FIXTURES,
  RELIEF_CASES,
  RELIEF_FIXTURES,
  compareRound4Values
} from './parity-wp5-round4-fixtures';

it('Round-4 fixtures identify their generators and installed independent references', () => {
  for (const [fixtures, generator] of [
    [WATERSHED_FIXTURES, 'parity-wp5-watersheds.py'],
    [RELIEF_FIXTURES, 'parity-wp5-relief.py']
  ] as const) {
    const generatorBytes = readFileSync(new URL(`../oracles/${generator}`, import.meta.url));
    const generatorHash = createHash('sha256').update(generatorBytes).digest('hex');
    for (const fixture of fixtures) {
      expect(fixture.generatorSha256).toBe(generatorHash);
      expect(fixture.versions['numpy']).toBeTruthy();
      expect(fixture.versions['python']).toBeTruthy();
    }
  }
  for (const fixture of WATERSHED_FIXTURES) expect(fixture.versions['pysheds']).toBe('0.5');
  for (const fixture of RELIEF_FIXTURES) {
    expect(fixture.versions['rvt-py']).toBe('2.2.1');
    expect(fixture.versions['scipy']).toBeTruthy();
  }
});

it('watersheds preserve complete source graphs, exact basin coverage and terminal identities', () => {
  expect(WATERSHED_CASES).toHaveLength(7);
  expect(WATERSHED_CASES.reduce((total, fixture) => total + fixture.directions.length, 0)).toBe(
    12026
  );
  for (const fixture of WATERSHED_CASES) {
    const sourceBytes = readFileSync(
      new URL(`../oracles/fixtures/${fixture.sourceFixture}`, import.meta.url)
    );
    expect(createHash('sha256').update(sourceBytes).digest('hex')).toBe(
      fixture.sourceFixtureSha256
    );
    const source: {cases: {name: string; directions: number[]}[]} = JSON.parse(
      sourceBytes.toString()
    );
    expect(fixture.directions).toEqual(
      source.cases.find(candidate => candidate.name === fixture.name)!.directions
    );
    const count = fixture.width * fixture.height;
    expect(fixture.basinLabels).toHaveLength(count);
    expect(fixture.pourPointLabels).toHaveLength(count);
    expect(fixture.directions[0]).toBe(0xffffffff);
    for (let index = 0; index < count; index++) {
      const terminal = fixture.basinLabels[index];
      if (fixture.directions[index] === 0xffffffff) {
        expect(terminal).toBe(0xffffffff);
        expect(fixture.pourPointLabels[index]).toBe(0xffffffff);
      } else {
        expect(fixture.directions[terminal]).toBe(0);
        expect(fixture.basinLabels[terminal]).toBe(terminal);
      }
    }
    if (fixture.name.startsWith('alps')) {
      expect(fixture.source.fullGraphAccumulationVerified).toBe(true);
      expect(fixture.directions.filter(value => value !== 0xffffffff)).toHaveLength(
        fixture.source.catchmentCells!
      );
    }
  }
});

it('nested catchment masks prove nearest-point ownership and duplicate precedence', () => {
  let nestedCells = 0;
  for (const fixture of WATERSHED_CASES) {
    const counts = new Map(
      fixture.catchments.map(catchment => [
        catchment.pointIndex,
        catchment.mask.reduce((total, value) => total + value, 0)
      ])
    );
    for (const catchment of fixture.catchments) {
      expect(catchment.mask).toHaveLength(fixture.directions.length);
      const point = fixture.pourPoints[catchment.pointIndex];
      expect(catchment.mask[point]).toBe(1);
      expect(fixture.pourPointLabels[point]).toBe(fixture.pourPoints.indexOf(point));
    }
    for (let index = 0; index < fixture.directions.length; index++) {
      const enclosing = fixture.catchments.filter(catchment => catchment.mask[index]);
      const label = fixture.pourPointLabels[index];
      if (enclosing.length === 0) {
        expect(label).toBe(0xffffffff);
        continue;
      }
      expect(enclosing.some(catchment => catchment.pointIndex === label)).toBe(true);
      for (const catchment of enclosing) {
        expect(counts.get(label)!).toBeLessThanOrEqual(counts.get(catchment.pointIndex)!);
        if (counts.get(label) === counts.get(catchment.pointIndex))
          expect(label).toBeLessThanOrEqual(catchment.pointIndex);
        else nestedCells++;
      }
    }
  }
  expect(nestedCells).toBeGreaterThan(100);
});

it('relief fixtures cover all cells, exact input masks, both scale ladders and independent RVT checks', () => {
  expect(RELIEF_CASES).toHaveLength(9);
  for (const fixture of RELIEF_CASES) {
    const count = fixture.width * fixture.height;
    for (const values of [
      fixture.elevation,
      fixture.validity,
      fixture.expectedValidity,
      ...Object.values(fixture.expected)
    ]) {
      expect(values).toHaveLength(count);
    }
    expect(getGPUMultiScaleReliefRadii(fixture.scales).radii).toEqual(fixture.radii);
    expect(fixture.rvtCrossCheck.maxAbs).toBeLessThanOrEqual(
      fixture.rvtCrossCheck.derivedAbsoluteBound
    );
    for (let index = 0; index < count; index++) {
      const valid =
        Number.isFinite(fixture.elevation[index]) &&
        fixture.elevation[index] !== -9999 &&
        fixture.validity[index] !== 0;
      expect(fixture.expectedValidity[index]).toBe(Number(valid));
      for (const output of Object.values(fixture.expected))
        expect(Number.isFinite(output[index])).toBe(valid);
    }
    // Profile R is recomputed from each full reference array. This prevents a
    // fixture regeneration from silently inflating GPU comparison tolerances.
    for (const metric of ['dominance', 'relief'] as const) {
      const finite = fixture.expected[metric].filter(Number.isFinite);
      const range = finite.length ? Math.max(...finite) - Math.min(...finite) : 0;
      expect(fixture.tolerances[metric].profile).toBe('R');
      expect(fixture.tolerances[metric].relative).toBe(1e-4);
      const absolute = 1e-5 * Math.max(1, range);
      // Two f64 arithmetic steps (range subtraction and scaling) and JSON
      // transport can differ across Python/JS by rounding. Four eps bounds
      // those steps; this checks metadata only, never changes the GPU budget.
      expect(Math.abs(fixture.tolerances[metric].absolute - absolute)).toBeLessThanOrEqual(
        4 * Number.EPSILON * absolute
      );
    }
    if (fixture.name.startsWith('alps')) {
      expect(fixture.source.fullFilterVerified).toBe(true);
      expect(fixture.source.unitedDominanceVerified).toBe(true);
      expect(fixture.source.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(fixture.source.halo).toBeGreaterThanOrEqual(Math.max(...fixture.radii));
    }
  }
});

it('external relief references reproduce flat analytic values and all-invalid masks', () => {
  const flat = RELIEF_CASES.find(fixture => fixture.name === 'flat-high-offset')!;
  for (const [className, metric, constant] of [
    ['GPULocalDominance', 'dominance', 1],
    ['GPUMultiScaleRelief', 'relief', 0]
  ] as const) {
    compareRound4Values(
      className,
      'reference-flat-plane',
      'Analytic flat surface',
      metric,
      flat.expected[metric],
      new Array<number>(flat.elevation.length).fill(constant),
      flat.tolerances[metric]
    );
  }
  const invalid = RELIEF_CASES.find(fixture => fixture.name === 'all-invalid')!;
  expect(invalid.expectedValidity.every(value => value === 0)).toBe(true);
  for (const output of Object.values(invalid.expected))
    expect(output.every(Number.isNaN)).toBe(true);
});
