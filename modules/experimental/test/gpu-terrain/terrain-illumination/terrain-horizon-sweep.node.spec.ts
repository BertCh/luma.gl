// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getTerrainSweepGroundLineGeometry,
  getTerrainSweepLineGeometry,
  getTerrainSweepMinorOffset,
  TERRAIN_SWEEP_SLOPE_SCALE
} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonStepDistances
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  getRasterGroundCellSize,
  type GPURasterCellSizeMode
} from '../../../src/analysis-core/raster-grid';
import {getOracleWebMercatorY} from '../terrain-oracle-utils';
import {computeTerrainHorizon, createRandom, createSmoothTerrain} from './terrain-horizon-oracle';
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

it('getTerrainSweepGroundLineGeometry follows a 45 degree ground azimuth on a 60 N geographic grid', () => {
  // python3 -I, WGS84 (a = 6378137, f = 1 / 298.257223563, e2 = f (2 - f), w = 1 - e2 sin^2(lat),
  // N = a / sqrt(w), M = a (1 - e2) / w^1.5): 0.001 degree cells at 60 N are
  // [55.80000157243614, 111.41228745777943] m. The 45 degree pixel slope is
  // -0.5008424370927846 = -32823.209957312734 / 65536, so slopeFixed = -32823, and that line's
  // ground azimuth is atan2(cellX, 32823 / 65536 * cellY) = 45.00018324999883 degrees. The old
  // pixel diagonal (slopeFixed -65536) was a 26.60365263545291 degree ground ray.
  const groundCell = getRasterGroundCellSize(
    'geographic',
    {cellSize: [0.001, 0.001], northEdge: 60, southEdge: 60},
    0.5,
    1
  );
  const direction = getGPUTerrainHorizonDirection(1, 8);
  const geometry = getTerrainSweepGroundLineGeometry(direction, groundCell, 48, 24);
  expect(geometry).toMatchObject({xMajor: true, slopeFixed: -32823, travelSign: 1});
  expect(geometry.stepPixelLength).toBe(Math.fround(1.1184095572334558));
  const azimuth =
    (Math.atan2(groundCell[0], (-geometry.slopeFixed / TERRAIN_SWEEP_SLOPE_SCALE) * groundCell[1]) *
      180) /
    Math.PI;
  expect(azimuth).toBeCloseTo(45.00018324999883, 9);
  // Square cells keep the float64 pixel geometry; axis sectors stay on their column or row.
  for (const sector of [0, 1, 2, 3, 5]) {
    const sectorDirection = getGPUTerrainHorizonDirection(sector, 8);
    expect(getTerrainSweepGroundLineGeometry(sectorDirection, [30, 30], 48, 24)).toEqual(
      getTerrainSweepLineGeometry(sectorDirection, 48, 24)
    );
  }
  expect(getTerrainSweepGroundLineGeometry([0, -1], groundCell, 48, 24)).toMatchObject({
    xMajor: false,
    slopeFixed: 0
  });
  expect(getTerrainSweepGroundLineGeometry([1, 0], groundCell, 48, 24)).toMatchObject({
    xMajor: true,
    slopeFixed: 0
  });
});

// Every GPU sweep fixture with non-square cells (gpu-terrain-horizon-sweep.spec.ts and
// gpu-terrain-horizon.spec.ts). The kernel derives slopeFixed from f32 cell sizes, which differ from
// the float64 oracle cells by about 1e-7 relative (about 0.01 of a fixed-point unit), so slopeFixed
// may differ by one near a .5 tie. Require each sector's unrounded slope to be at least 0.05 units
// from a tie, or else slopeFixed - 1, slopeFixed and slopeFixed + 1 to put every pixel on the same
// line, so the GPU lines equal the oracle's whatever the rounding.
it('sweep ground geometry fixtures stay clear of rounding ties', () => {
  const fixtures: {
    mode: GPURasterCellSizeMode;
    cellSize: [number, number];
    northEdge: number;
    southEdge: number;
    width: number;
    height: number;
    directionCounts: number[];
    rowDirection?: 'south' | 'north';
  }[] = [
    {
      mode: 'uniform',
      cellSize: [10, 12],
      northEdge: 0,
      southEdge: 0,
      width: 48,
      height: 40,
      directionCounts: [8, 16, 7]
    },
    {
      mode: 'web-mercator',
      // 20 km cells from normalized Web Mercator y 0.3, as in the GPU sweep and horizon specs.
      cellSize: [20000, 20000],
      northEdge: getOracleWebMercatorY(0.3),
      southEdge: getOracleWebMercatorY(0.3) + 24 * 20000,
      width: 32,
      height: 24,
      directionCounts: [12],
      rowDirection: 'north'
    },
    {
      mode: 'geographic',
      cellSize: [0.001, 0.001],
      northEdge: 60.012,
      southEdge: 59.988,
      width: 48,
      height: 24,
      directionCounts: [8, 12]
    }
  ];
  for (const fixture of fixtures) {
    const referenceRow = fixture.mode === 'uniform' ? 0 : Math.floor(fixture.height / 2);
    const [cellX, cellY] = getRasterGroundCellSize(
      fixture.mode,
      {cellSize: fixture.cellSize, northEdge: fixture.northEdge, southEdge: fixture.southEdge},
      referenceRow + 0.5,
      fixture.height
    );
    for (const directionCount of fixture.directionCounts) {
      for (let sector = 0; sector < directionCount; sector++) {
        const direction = getGPUTerrainHorizonDirection(
          sector,
          directionCount,
          fixture.rowDirection
        );
        const geometry = getTerrainSweepGroundLineGeometry(
          direction,
          [cellX, cellY],
          fixture.width,
          fixture.height
        );
        const pixelX = direction[0] / cellX;
        const pixelY = direction[1] / cellY;
        const ratio = geometry.xMajor ? pixelY / pixelX : pixelX / pixelY;
        const scaled = Math.abs(ratio * TERRAIN_SWEEP_SLOPE_SCALE);
        if (Math.abs((scaled % 1) - 0.5) > 0.05) {
          continue;
        }
        const majorExtent = geometry.xMajor ? fixture.width : fixture.height;
        for (let major = 0; major < majorExtent; major++) {
          const offset = getTerrainSweepMinorOffset(geometry.slopeFixed, major);
          for (const neighbor of [geometry.slopeFixed - 1, geometry.slopeFixed + 1]) {
            if (Math.abs(neighbor) <= TERRAIN_SWEEP_SLOPE_SCALE) {
              expect(
                getTerrainSweepMinorOffset(neighbor, major),
                `${fixture.mode} ${directionCount}:${sector} at ${major}`
              ).toBe(offset);
            }
          }
        }
      }
    }
  }
});

it('hull sweep equals brute force on a 60 N geographic grid', () => {
  const width = 32;
  const height = 24;
  const options: SweepOracleOptions = {
    width,
    height,
    elevation: createNoisyTerrain(width, height, 7),
    directionCount: 12,
    cellSize: [0.001, 0.001],
    cellSizeMode: 'geographic',
    northEdge: 60.012,
    southEdge: 59.988,
    maximumRadius: 12
  };
  const hull = computeTerrainHorizonSweepHull(options);
  expectSameResult(hull, computeTerrainHorizonSweepBruteForce(options));
  // The GPU spec 'sweep kernel follows ground azimuths on a 60 N geographic grid' uses this
  // fixture and requires more than 30 angles above 5 degrees.
  expect(hull.horizon.filter(angle => angle > 5).length).toBeGreaterThan(30);
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
    // Brute force costs O(directions x pixels x line length), so the 64-sector case (which exists
    // for its many distinct line slopes) runs the unbounded variant plus one windowed, curved,
    // north-up variant. Every option's plumbing is independent of the sector count and stays
    // covered across all seven variants by the 4, 7, 8 and 16 sector cases.
    const selectedVariants = directionCount >= 32 ? [variants[0], variants[5]] : variants;
    for (const variant of selectedVariants) {
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
    northEdge: getOracleWebMercatorY(0.3),
    southEdge: getOracleWebMercatorY(0.33),
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

it('hull sweep equals the ray march on axis sectors and stays close on diagonals', () => {
  const width = 48;
  const height = 40;
  const elevation = createSmoothTerrain(width, height, 8);
  const options = {
    width,
    height,
    elevation,
    directionCount: 16,
    maximumRadius: 20,
    cellSize: [10, 10] as [number, number]
  };
  const sweep = computeTerrainHorizonSweepHull(options);
  const march = computeTerrainHorizon({
    ...options,
    stepDistances: getGPUTerrainHorizonStepDistances(20)
  });
  // Axis lines hit pixel centres exactly like the march (identical); diagonals sample pixel
  // centres at k * sqrt(2) pixels while the march samples bilinear points at d pixels along the ray.
  for (const sector of [0, 2, 4, 6, 8, 10, 12, 14]) {
    let total = 0;
    for (let pixel = 0; pixel < width * height; pixel++) {
      total += Math.abs(sweep.horizon[pixel * 16 + sector] - march.horizon[pixel * 16 + sector]);
    }
    const tolerance = sector % 4 === 0 ? 1e-9 : 1;
    expect(total / (width * height), `sector ${sector}`).toBeLessThan(tolerance);
  }
});
