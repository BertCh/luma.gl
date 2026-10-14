// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../../src/analysis-core/contracts';

const PRIMARY: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  datum: 'EPSG:4326'
};
const PLANAR: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  frame: {id: 'primary-frame', kind: 'crs'}
};
const LABEL = 'join polygonsSpatialContext';

describe('resolveGPUSecondarySpatialContext', () => {
  it('returns the primary object when the secondary context is absent', () => {
    expect(resolveGPUSecondarySpatialContext('join', 'polygons', undefined, PRIMARY)).toBe(PRIMARY);
  });

  it('validates the secondary before checking compatibility', () => {
    expect(() =>
      resolveGPUSecondarySpatialContext('join', 'polygons', {...PRIMARY, datum: ' '}, PRIMARY)
    ).toThrowError(new Error(`${LABEL} datum must be a non-empty string`));
  });

  it('accepts WGS84 as an alias for EPSG:4326', () => {
    const secondary = {...PRIMARY, datum: 'WGS84'};
    expect(resolveGPUSecondarySpatialContext('join', 'polygons', secondary, PRIMARY)).toBe(
      secondary
    );
  });

  it('accepts an unmeasured secondary with a great-circle primary', () => {
    const secondary: GPUSpatialContext = {...PRIMARY, metric: 'none'};
    expect(resolveGPUSecondarySpatialContext('join', 'polygons', secondary, PRIMARY)).toBe(
      secondary
    );
  });

  it('rejects nonlinear longitude/latitude edges before checking compatibility', () => {
    expect(() =>
      resolveGPUSecondarySpatialContext(
        'join',
        'polygons',
        {...PRIMARY, edgeModel: 'great-circle'},
        PRIMARY,
        GPU_LINEAR_EDGE_MODELS
      )
    ).toThrowError(
      new Error(`${LABEL} needs linear edges; prepare polygons with GPULonLatPolygonPreparation`)
    );
  });

  const mismatches: {
    name: string;
    primary: GPUSpatialContext;
    secondary: GPUSpatialContext;
    reason: string;
  }[] = [
    {
      name: 'datum',
      primary: PRIMARY,
      secondary: {...PRIMARY, datum: 'EPSG:4267'},
      reason: 'datums differ (EPSG:4267 and EPSG:4326)'
    },
    {
      name: 'planar frame',
      primary: PLANAR,
      secondary: {...PLANAR, frame: {id: 'secondary-frame', kind: 'crs'}},
      reason: 'planar frames differ (secondary-frame and primary-frame)'
    },
    {
      name: 'units',
      primary: PRIMARY,
      secondary: {...PRIMARY, units: 'kilometers'},
      reason: 'units differ (kilometers and meters)'
    },
    {
      name: 'coordinate space',
      primary: PLANAR,
      secondary: PRIMARY,
      reason: 'coordinate spaces differ (longitude-latitude and planar)'
    },
    {
      name: 'metric',
      primary: PRIMARY,
      secondary: {...PRIMARY, metric: 'rhumb'},
      reason: 'metrics differ (rhumb and great-circle)'
    }
  ];
  for (const {name, primary, secondary, reason} of mismatches) {
    it(`rejects a mismatched ${name}, listing the secondary first`, () => {
      expect(() =>
        resolveGPUSecondarySpatialContext('join', 'polygons', secondary, primary)
      ).toThrowError(new Error(`${LABEL} incompatible spatial contexts: ${reason}`));
    });
  }
});
