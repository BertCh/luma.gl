// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUKernelDensityWindowMass} from '../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {
  addOracleDegenerateSlots,
  ORACLE_RECTANGLE,
  ORACLE_L_SHAPE,
  ORACLE_SQUARE_WITH_HOLE,
  ORACLE_TWO_ISLANDS,
  ORACLE_CLOCKWISE_SHELL_CCW_HOLE,
  getOracleOrientedEdges,
  getOracleWindowArea,
  getOracleWindowPerimeter,
  getOracleWindowBounds,
  isOracleInsideWindow,
  getOracleBoundaryDistance,
  getOracleCircleFractionFan,
  getOracleCircleFractionBrute,
  getOracleRectangleIsotropicFraction,
  getOracleKernelMassFan,
  getOracleKernelMassBrute,
  getOracleKernelMassScheme,
  getOracleErodedArea,
  getOracleQuadratAreas,
  type OraclePoint
} from './study-window-oracle';

const kernels = ['gaussian', 'quartic', 'epanechnikov'] as const;
const windows = [
  ORACLE_RECTANGLE,
  ORACLE_L_SHAPE,
  ORACLE_SQUARE_WITH_HOLE,
  ORACLE_TWO_ISLANDS,
  ORACLE_CLOCKWISE_SHELL_CCW_HOLE
];

it('normalizes ring roles and computes geometry independently of input winding', () => {
  expect(getOracleWindowArea(ORACLE_RECTANGLE)).toBe(2000000);
  expect(getOracleWindowPerimeter(ORACLE_RECTANGLE)).toBe(6000);
  expect(getOracleWindowBounds(ORACLE_RECTANGLE)).toEqual([0, 0, 2000, 1000]);
  expect(getOracleWindowArea(ORACLE_CLOCKWISE_SHELL_CCW_HOLE)).toBe(3640000);
  expect(getOracleWindowPerimeter(ORACLE_CLOCKWISE_SHELL_CCW_HOLE)).toBe(10400);
  expect(getOracleBoundaryDistance(ORACLE_RECTANGLE, [100, 200], 99)).toBe(Infinity);
  expect(getOracleBoundaryDistance(ORACLE_RECTANGLE, [100, 200], 100)).toBe(100);
  expect(getOracleQuadratAreas(ORACLE_RECTANGLE, [0, 0, 2000, 1000], [2, 2], 32)).toEqual([
    500000, 500000, 500000, 500000
  ]);
});

it('fan fraction equals the uncapped rectangle formula, including corners and edges', () => {
  for (const centre of [
    [0, 0],
    [2000, 1000],
    [1000, 0],
    [1000, 500],
    [1e-8, 300],
    [400, 100],
    [1800, 950]
  ] as const) {
    for (const radius of [1, 100, 300, 500, 999, 1000, 1700, 2300, 10000]) {
      expect(
        Math.abs(
          getOracleCircleFractionFan(ORACLE_RECTANGLE, centre, radius) -
            getOracleRectangleIsotropicFraction([0, 0, 2000, 1000], centre, radius)
        )
      ).toBeLessThanOrEqual(1e-12);
    }
  }
});

it('fan fractions agree with independent million-angle integration across holes, concavities and islands', () => {
  const cases = [
    {window: ORACLE_L_SHAPE, centre: [600, 600] as OraclePoint, radii: [300, 900, 1600]},
    {window: ORACLE_SQUARE_WITH_HOLE, centre: [400, 1000] as OraclePoint, radii: [500, 950, 1700]},
    {window: ORACLE_TWO_ISLANDS, centre: [900, 500] as OraclePoint, radii: [700, 1300, 1800]},
    {
      window: ORACLE_CLOCKWISE_SHELL_CCW_HOLE,
      centre: [1500, 1200] as OraclePoint,
      radii: [600, 1000]
    },
    {window: ORACLE_L_SHAPE, centre: [700, 700] as OraclePoint, radii: [200, 900]}
  ];
  for (const {window, centre, radii} of cases) {
    for (const radius of radii) {
      expect(
        Math.abs(
          getOracleCircleFractionFan(window, centre, radius) -
            getOracleCircleFractionBrute(window, centre, radius)
        )
      ).toBeLessThanOrEqual(2e-6);
    }
  }
}, 30000);

it('signed fan angles sum to 2pi inside and zero outside', () => {
  for (const window of windows) {
    for (const point of [
      [300, 400],
      [1000, 1100],
      [2300, 500],
      [-100, 500]
    ] as const) {
      let angle = 0;
      for (const [startX, startY, endX, endY] of getOracleOrientedEdges(window)) {
        const firstX = startX - point[0];
        const firstY = startY - point[1];
        const secondX = endX - point[0];
        const secondY = endY - point[1];
        angle += Math.atan2(
          firstX * secondY - firstY * secondX,
          firstX * secondX + firstY * secondY
        );
      }
      expect(
        Math.abs(angle - (isOracleInsideWindow(window, point) ? 2 * Math.PI : 0))
      ).toBeLessThan(1e-12);
    }
  }
});

it('adaptive fan mass agrees with independent ray integration for every kernel', () => {
  for (const window of windows) {
    for (const kernel of kernels) {
      const centre: OraclePoint = [350, 550];
      for (const bandwidth of [650, 1400]) {
        expect(
          Math.abs(
            getOracleKernelMassFan(window, centre, bandwidth, kernel) -
              getOracleKernelMassBrute(window, centre, bandwidth, kernel)
          )
        ).toBeLessThanOrEqual(1e-6);
      }
    }
  }
}, 30000);

it('fixed WGSL scheme agrees with adaptive integration over tiny distances, support endpoints and wide fans', () => {
  let maximumError = 0;
  for (const kernel of kernels) {
    for (const ratio of [
      1e-12, 1e-9, 1e-6, 1e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.7, 0.99, 1, 1.5, 3, 4.99, 5
    ]) {
      for (const bandwidth of [100, 500, 1400]) {
        const centre: OraclePoint = [ratio * bandwidth, 500];
        const exact = getOracleKernelMassFan(ORACLE_RECTANGLE, centre, bandwidth, kernel);
        const scheme = getOracleKernelMassScheme(ORACLE_RECTANGLE, centre, bandwidth, kernel);
        maximumError = Math.max(maximumError, Math.abs(exact - scheme));
        expect(
          Math.abs(exact - scheme),
          `${kernel} p/h=${ratio}, h=${bandwidth}`
        ).toBeLessThanOrEqual(2e-6);
      }
    }
    // Distance from centre to (2000,1000) equals bandwidth exactly.
    const centre: OraclePoint = [1700, 600];
    expect(
      Math.abs(
        getOracleKernelMassFan(ORACLE_RECTANGLE, centre, 500, kernel) -
          getOracleKernelMassScheme(ORACLE_RECTANGLE, centre, 500, kernel)
      )
    ).toBeLessThanOrEqual(2e-6);
    for (const window of windows) {
      for (const point of [
        [0, 0],
        [700, 700],
        [1250, 500],
        [-100, 400]
      ] as const) {
        expect(
          Math.abs(
            getOracleKernelMassFan(window, point, 700, kernel) -
              getOracleKernelMassScheme(window, point, 700, kernel)
          )
        ).toBeLessThanOrEqual(2e-6);
      }
    }
  }
  expect(maximumError).toBeLessThanOrEqual(2e-6);
}, 30000);

it('rectangle kernel mass agrees with the existing closed forms within their documented precision', () => {
  for (const kernel of kernels) {
    for (const centre of [
      [0, 0],
      [400, 300],
      [1000, 500],
      [2500, 100]
    ] as const) {
      for (const bandwidth of [100, 500, 1200]) {
        const actual = getOracleKernelMassFan(ORACLE_RECTANGLE, centre, bandwidth, kernel);
        const rectangle = getGPUKernelDensityWindowMass(
          kernel,
          bandwidth,
          centre,
          [0, 0, 2000, 1000]
        );
        // The production rectangle Gaussian uses an erfc fit with ~1e-7 relative error.
        expect(Math.abs(actual - rectangle)).toBeLessThanOrEqual(
          kernel === 'gaussian' ? 1e-7 : 1e-9
        );
      }
    }
  }
});

it('rectangle erosion agrees with (width-2r)(height-2r) within one raster boundary band', () => {
  for (const radius of [0, 100, 225, 499, 600]) {
    const exact = Math.max(0, 2000 - 2 * radius) * Math.max(0, 1000 - 2 * radius);
    expect(
      Math.abs(getOracleErodedArea(ORACLE_RECTANGLE, radius, 400) - exact)
    ).toBeLessThanOrEqual(6000 * 5);
  }
});

it('clip-shaped empty and degenerate slots leave every oracle summary field unchanged', () => {
  for (const window of [ORACLE_L_SHAPE, ORACLE_SQUARE_WITH_HOLE]) {
    const clipped = addOracleDegenerateSlots(window);
    expect(getOracleWindowBounds(clipped)).toEqual(getOracleWindowBounds(window));
    expect(getOracleWindowArea(clipped)).toBe(getOracleWindowArea(window));
    expect(getOracleWindowPerimeter(clipped)).toBe(getOracleWindowPerimeter(window));
    expect(getOracleOrientedEdges(clipped)).toEqual(getOracleOrientedEdges(window));
  }
});
