// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPUProjectionSpatialContext} from '../../src/gpu-project/projection-spatial-context';
import {
  getGPUSpatialContextCompatibility,
  resolveGPUSpatialContext
} from '../../src/analysis-core/contracts/spatial-context';

describe('GPUProjection output spatial context', () => {
  it.each(['EPSG:32633', 'EPSG:3857', 'EPSG:2263'])('describes target %s as a crs frame', crs => {
    const context = getGPUProjectionSpatialContext(crs);
    expect(context).toEqual({
      coordinateSpace: 'planar',
      metric: 'native',
      frame: {kind: 'crs', id: crs, crs}
    });
    expect(
      resolveGPUSpatialContext('op', context, {coordinateSpaces: ['planar'], measure: 'distance'})
    ).toBe(context);
    expect(
      resolveGPUSpatialContext('op', context, {coordinateSpaces: ['planar'], measure: 'area'})
    ).toBe(context);
  });

  it('keeps rows of different target CRSs apart', () => {
    const utm = getGPUProjectionSpatialContext('EPSG:32633')!;
    const mercator = getGPUProjectionSpatialContext('EPSG:3857')!;
    expect(getGPUSpatialContextCompatibility(utm, mercator).compatible).toBe(false);
    expect(getGPUSpatialContextCompatibility(utm, utm).compatible).toBe(true);
  });

  it.each([undefined, ''])('has no context without a target (%s)', targetCrs => {
    expect(getGPUProjectionSpatialContext(targetCrs)).toBeUndefined();
  });
});
