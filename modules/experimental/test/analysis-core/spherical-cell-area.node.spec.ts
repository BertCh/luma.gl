// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPUAuthalicSphereRadius} from '../../src/analysis-core/contracts/spatial-context';
import {
  getRasterSphereRadius,
  getSphericalRasterCellArea
} from '../../src/analysis-core/spherical-cell-area';

describe('spherical raster cell area units', () => {
  it('matches the binary64 spherical oracle in square meters and square kilometers', () => {
    const originLatitude = 32;
    const cellWidth = 0.25;
    const cellHeight = 0.5;
    const row = 3;
    const radiansPerDegree = Math.PI / 180;
    const bottomLatitude = (originLatitude + row * cellHeight) * radiansPerDegree;
    const topLatitude = (originLatitude + (row + 1) * cellHeight) * radiansPerDegree;
    const radiusMeters = 6371007.180918475;
    const areaOracle =
      radiusMeters ** 2 *
      (cellWidth * radiansPerDegree) *
      (Math.sin(topLatitude) - Math.sin(bottomLatitude));
    const areas = (['meters', 'kilometers'] as const).map(units => {
      const radius = getGPUAuthalicSphereRadius({
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units
      });
      const area = getSphericalRasterCellArea(originLatitude, cellWidth, cellHeight, row, radius);
      const expectedArea = units === 'meters' ? areaOracle : areaOracle / 1e6;
      expect(Math.abs(area / expectedArea - 1)).toBeLessThan(1e-13);
      return area;
    });
    expect(Math.abs(areas[1] / areas[0] - 1e-6)).toBeLessThan(1e-20);
    expect(getSphericalRasterCellArea(originLatitude, cellWidth, cellHeight, row)).toBe(areas[0]);
  });

  it('retains the deprecated radius helper as an exact alias honoring kilometers', () => {
    expect(getRasterSphereRadius).toBe(getGPUAuthalicSphereRadius);
    expect(
      getRasterSphereRadius({
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'kilometers'
      })
    ).toBe(6371007.180918475 / 1000);
  });
});
