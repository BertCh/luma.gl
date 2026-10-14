// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import fixture from '../../oracles/fixtures/terrain-hydrology.json';
import {computeTerrainFlow} from './terrain-flow-oracle';

/**
 * Cross-checks the TS D8 oracle against pysheds (test/oracles/terrain-hydrology.py, versions in
 * the fixture). pysheds' -1 (flat) and -2 (pit or border outlet) are luma.gl's terminal 0; the
 * fixture's `directions` already holds the mapped codes, `pyshedsDirections` the raw ones.
 */
it('the D8 flow oracle matches pysheds directions and accumulation', () => {
  expect(fixture.versions.pysheds).toBeTruthy();
  for (const testCase of fixture.cases) {
    const oracle = computeTerrainFlow({
      elevation: Float32Array.from(testCase.elevation),
      grid: makeGPURasterMetadata({
        width: testCase.width,
        height: testCase.height,
        cellSize: testCase.cellSize as [number, number]
      }),
      settings: {}
    });
    expect(Array.from(oracle.directions), testCase.name).toEqual(testCase.directions);
    expect(Array.from(oracle.accumulation), testCase.name).toEqual(testCase.accumulation);
    // The only terminal cell of these depression-free DEMs is the border outlet (pysheds -2).
    expect(testCase.pyshedsDirections.filter(code => code < 0)).toEqual([-2]);
  }
});
