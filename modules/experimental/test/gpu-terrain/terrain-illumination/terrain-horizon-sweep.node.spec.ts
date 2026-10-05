// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getTerrainSweepLineGeometry,
  getTerrainSweepMinorOffset
} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {createRandom, createSmoothTerrain} from './terrain-horizon-oracle';
import {
  computeTerrainHorizonSweepBruteForce,
  computeTerrainHorizonSweepHull,
  type SweepOracleOptions
} from './terrain-horizon-sweep-oracle';

function createNoisyTerrain(width: number, height: number, seed: number): Float32Array {
  const random = createRandom(seed);
  const terrain = createSmoothTerrain(width, height, seed);
  for (let index = 0; index < terrain.length; index++) {
    terrain[index] += (random() - 0.5) * 12;
  }
  for (let hole = 0; hole < 9; hole++) {
    terrain[Math.floor(random() * terrain.length)] = NaN;
  }
  // A small nodata block and a nodata column stretch exercise first-valid-ahead chains.
  for (let row = 3; row < 6; row++) {
    for (let column = 4; column < 9; column++) {
      terrain[row * width + column] = NaN;
    }
  }
  return terrain;
}

function expectSameResult(
  actual: ReturnType<typeof computeTerrainHorizonSweepHull>,
  expected: ReturnType<typeof computeTerrainHorizonSweepBruteForce>
): void {
  expect(actual.validity).toEqual(expected.validity);
  for (const key of [
    'horizon',
    'skyViewFactor',
    'positiveOpenness',
    'nadirHorizon',
    'negativeOpenness'
  ] as const) {
    expect(actual[key].length).toBe(expected[key].length);
    for (const [index, value] of expected[key].entries()) {
      if (Number.isNaN(value)) {
        expect(Number.isNaN(actual[key][index]), `${key} ${index}`).toBe(true);
      } else {
        expect(Math.abs(actual[key][index] - value), `${key} ${index}`).toBeLessThan(1e-9);
      }
    }
  }
}

it('digital lines partition the grid for every direction', () => {
  for (const [width, height] of [
    [13, 9],
    [9, 13],
    [16, 16]
  ]) {
    for (let sector = 0; sector < 64; sector++) {
      const angle = (sector / 64) * 2 * Math.PI;
      const geometry = getTerrainSweepLineGeometry(
        [Math.sin(angle), -Math.cos(angle)],
        width,
        height
      );
      const majorExtent = geometry.xMajor ? width : height;
      const minorExtent = geometry.xMajor ? height : width;
      const seen = new Uint8Array(width * height);
      for (let index = 0; index < geometry.lineCount; index++) {
        const line = geometry.firstLine + index;
        let count = 0;
        for (let major = 0; major < majorExtent; major++) {
          const minor = line + getTerrainSweepMinorOffset(geometry.slopeFixed, major);
          if (minor >= 0 && minor < minorExtent) {
            seen[geometry.xMajor ? minor * width + major : major * width + minor]++;
            count++;
          }
        }
        // Every counted line contains at least one pixel.
        expect(count).toBeGreaterThan(0);
      }
      expect(seen.every(value => value === 1)).toBe(true);
    }
  }
});

it('getTerrainSweepLineGeometry derives the fixed-point slope and rejects bad input', () => {
  const axis = getTerrainSweepLineGeometry([0, -1], 10, 7);
  expect(axis).toMatchObject({xMajor: false, slopeFixed: 0, travelSign: -1, lineCount: 10});
  expect(getTerrainSweepLineGeometry([1, 1], 4, 4)).toMatchObject({
    xMajor: true,
    slopeFixed: 65536,
    lineCount: 7,
    firstLine: -3,
    stepPixelLength: Math.fround(Math.SQRT2)
  });
  expect(() => getTerrainSweepLineGeometry([0, 0], 4, 4)).toThrow(/non-zero/);
  expect(() => getTerrainSweepLineGeometry([1, 0], 40000, 4)).toThrow(/extents/);
});

it('hull sweep equals brute force on random terrain with nodata', () => {
  const width = 37;
  const height = 29;
  for (const [seed, directionCount] of [
    [1, 4],
    [2, 7],
    [3, 8],
    [4, 16],
    [5, 64]
  ] as const) {
    const elevation = createNoisyTerrain(width, height, seed);
    const base: SweepOracleOptions = {
      width,
      height,
      elevation,
      directionCount,
      cellSize: [10, 10],
      nadir: true
    };
    const variants: Partial<SweepOracleOptions>[] = [
      {},
      {maximumRadius: 3},
      {maximumRadius: 17, rowDirection: 'north'},
      {maximumRadius: 17, maximumDistance: 95},
      {curvatureCoefficient: 2e-3, zFactor: 1.7},
      {zFactor: -2, curvatureCoefficient: 1e-3, maximumRadius: 17},
      {cellSize: [10, 14], maximumDistance: 130, rowDirection: 'north'}
    ];
    for (const variant of variants) {
      const options = {...base, ...variant};
      expectSameResult(
        computeTerrainHorizonSweepHull(options),
        computeTerrainHorizonSweepBruteForce(options)
      );
    }
  }
});

it('hull sweep equals brute force with latitude-dependent cells and exact plateaus', () => {
  const width = 21;
  const height = 17;
  const elevation = createNoisyTerrain(width, height, 9);
  const options: SweepOracleOptions = {
    width,
    height,
    elevation,
    directionCount: 12,
    cellSize: [20, 20],
    cellSizeMode: 'web-mercator',
    northEdge: 0.3,
    southEdge: 0.33,
    maximumRadius: 12,
    maximumDistance: 400
  };
  // With c = 0 the argmax is exact for latitude-dependent cells.
  expectSameResult(
    computeTerrainHorizonSweepHull(options),
    computeTerrainHorizonSweepBruteForce(options)
  );
  // Integer plateaus tie exactly; nearest wins in both.
  const plateau = Float32Array.from({length: width * height}, (_, index) =>
    Math.floor(((index % width) + Math.floor(index / width)) / 7)
  );
  const plateauOptions = {...options, elevation: plateau, cellSizeMode: 'uniform' as const};
  expectSameResult(
    computeTerrainHorizonSweepHull(plateauOptions),
    computeTerrainHorizonSweepBruteForce(plateauOptions)
  );
});

it('hull sweep has small amortised cost at full radius', () => {
  const width = 96;
  const height = 80;
  const elevation = createSmoothTerrain(width, height, 5);
  const result = computeTerrainHorizonSweepHull({
    width,
    height,
    elevation,
    directionCount: 16,
    cellSize: [10, 10]
  });
  const {pixelCount, walkSteps, windowQueries} = result.statistics;
  expect(pixelCount).toBe(width * height * 16);
  // Amortised O(1): each hull edge is pushed once and popped at most once.
  expect(walkSteps / pixelCount).toBeLessThan(2);
  // A full-radius window always contains the unbounded tangent.
  expect(windowQueries).toBe(0);
});

it('hull sweep exercises the windowed query and reports nontrivial horizons', () => {
  const width = 64;
  const height = 48;
  const elevation = createNoisyTerrain(width, height, 11);
  const result = computeTerrainHorizonSweepHull({
    width,
    height,
    elevation,
    directionCount: 16,
    maximumRadius: 6,
    cellSize: [10, 10]
  });
  expect(result.statistics.windowQueries).toBeGreaterThan(1000);
  expect(result.horizon.filter(angle => angle > 5).length).toBeGreaterThan(1000);
  expect(result.horizon.filter(angle => angle < -5).length).toBeGreaterThan(1000);
  // Steps per window query stay bounded by the window.
  expect(result.statistics.windowSteps / result.statistics.windowQueries).toBeLessThan(6);
});
