// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createGPUPointDensityGaussianKernel,
  getGPUPointDensityHexagonCell,
  getGPUPointDensityHexagonCenter,
  getGPUPointDensityHexagonGridSize
} from '../../../src/gpu-spatial-analysis/point-density';
import {createSeededPoints, findNearestHexagon} from './point-density-oracle';

it('getGPUPointDensityHexagonCell matches the brute-force nearest center', () => {
  const radius = 1.3;
  const points = createSeededPoints(11, 500, [-10, -10, 10, 10]);
  for (let index = 0; index < points.length / 2; index++) {
    const [x, y] = [points[index * 2], points[index * 2 + 1]];
    const nearest = findNearestHexagon(x, y, -10, -10, radius, 12, 12);
    if (nearest.secondNearest - nearest.nearest < 1e-3 * radius) {
      continue;
    }
    expect(getGPUPointDensityHexagonCell(x, y, -10, -10, radius)).toEqual(nearest.cell);
  }
});

it('getGPUPointDensityHexagonCenter round-trips through the cell function', () => {
  for (let row = -3; row <= 3; row++) {
    for (let column = -3; column <= 3; column++) {
      const [x, y] = getGPUPointDensityHexagonCenter(column, row, 2, -1, 0.7);
      expect(getGPUPointDensityHexagonCell(x, y, 2, -1, 0.7)).toEqual([column, row]);
    }
  }
});

it('getGPUPointDensityHexagonGridSize covers the bounds', () => {
  expect(getGPUPointDensityHexagonGridSize([0, 0, 6, 6], 1)).toEqual([5, 5]);
  for (const radius of [0.7, 1]) {
    const [columns, rows] = getGPUPointDensityHexagonGridSize([0, 0, 6, 6], radius);
    const points = createSeededPoints(3, 1000, [0, 0, 6, 6]);
    for (let index = 0; index < points.length / 2; index++) {
      const [column, row] = getGPUPointDensityHexagonCell(
        points[index * 2],
        points[index * 2 + 1],
        0,
        0,
        radius
      );
      expect(column >= 0 && column < columns && row >= 0 && row < rows).toBe(true);
    }
  }
  for (const radius of [0, -1, Number.NaN]) {
    expect(() => getGPUPointDensityHexagonGridSize([0, 0, 1, 1], radius)).toThrow();
  }
  expect(() => getGPUPointDensityHexagonGridSize([1, 0, 0, 1], 1)).toThrow();
});

it('createGPUPointDensityGaussianKernel returns a normalized symmetric kernel', () => {
  const kernel = createGPUPointDensityGaussianKernel(2);
  expect(kernel.length).toBe(25);
  expect(kernel.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
  expect(kernel[0]).toBeCloseTo(kernel[24], 7);
  expect(Math.max(...kernel)).toBe(kernel[12]);
  expect(Array.from(createGPUPointDensityGaussianKernel(0))).toEqual([1]);
  for (const radius of [1.5, -1, 33]) {
    expect(() => createGPUPointDensityGaussianKernel(radius)).toThrow();
  }
  expect(() => createGPUPointDensityGaussianKernel(1, 0)).toThrow();
});
