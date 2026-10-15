// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {HYDROLOGY_CASES, HYDROLOGY_FIXTURES} from './parity-wp5-hydrology-fixtures';

it('hydrology reference fixtures include every output and preserve full catchment provenance', () => {
  for (const fixture of HYDROLOGY_FIXTURES) {
    expect(fixture.versions['pysheds']).toBe('0.5');
    expect(fixture.generatorSha256).toMatch(/^[a-f0-9]{64}$/);
  }
  for (const fixture of HYDROLOGY_CASES) {
    const count = fixture.width * fixture.height;
    for (const values of [
      fixture.elevation,
      fixture.directions,
      fixture.streams,
      fixture.accumulation,
      ...Object.values(fixture.expected)
    ])
      expect(values).toHaveLength(count);
    for (let index = 0; index < count; index++) {
      if (fixture.streams[index]) {
        expect(fixture.expected.heightAboveDrainage[index]).toBe(0);
        expect(fixture.expected.drainageCells[index]).toBe(index);
      }
      if (fixture.directions[index] === 0xffffffff) {
        expect(fixture.expected.heightAboveDrainage[index]).toBeNaN();
        expect(fixture.expected.drainageCells[index]).toBe(0xffffffff);
        expect(fixture.expected.streamOrder[index]).toBe(0);
      }
    }
    if (fixture.name.startsWith('alps')) {
      expect(fixture.source.fullGraphAccumulationVerified).toBe(true);
      expect(fixture.source.fullGraphCellCount).toBe(2048 * 2048);
      expect(fixture.elevation.filter(Number.isFinite)).toHaveLength(
        fixture.source.catchmentCells!
      );
    }
  }
});

it('published Strahler rule distinguishes two equal maxima from all donors equal', () => {
  const binary = HYDROLOGY_CASES.find(fixture => fixture.name === 'binary-junction')!;
  const mixed = HYDROLOGY_CASES.find(fixture => fixture.name === 'mixed-2-2-1-junction')!;
  for (const fixture of [binary, mixed]) {
    expect(fixture.expected.streamOrder[5 * 11 + 5]).toBe(3);
    expect(fixture.expected.streamOrder[7 * 11 + 5]).toBe(3);
  }
  expect(mixed.pyshedsOrder[5 * 11 + 5]).toBe(2);
  expect(binary.pyshedsOrder[7 * 11 + 5]).toBe(4);
  expect(mixed.pyshedsOrderMismatchCells).toContain(5 * 11 + 5);
});

it('flat zero-area and no-drainage limits are explicit', () => {
  const fixture = HYDROLOGY_CASES.find(candidate => candidate.name === 'flat-zero-area-no-stream')!;
  const zeroArea = 2 * fixture.width + 2;
  const missingArea = zeroArea + 1;
  const positiveArea = zeroArea + 2;
  expect(fixture.expected.wetnessIndex[zeroArea]).toBe(-Infinity);
  expect(fixture.expected.wetnessIndex[positiveArea]).toBe(Infinity);
  expect(fixture.expected.wetnessIndex[missingArea]).toBeNaN();
  expect(fixture.expected.slope[positiveArea]).toBe(0);
  expect(fixture.expected.streamPowerIndex[positiveArea]).toBe(0);
  expect(fixture.expected.drainageCells.every(value => value === 0xffffffff)).toBe(true);
  expect(fixture.expected.heightAboveDrainage.every(Number.isNaN)).toBe(true);
});
