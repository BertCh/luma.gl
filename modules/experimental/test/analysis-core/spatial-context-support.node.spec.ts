// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPUGreatCircleRadius,
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../src/analysis-core/contracts/spatial-context';

const GREAT_CIRCLE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
};

describe('resolveGPUSpatialContext support table', () => {
  it('narrows metrics per coordinate space', () => {
    const support = GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT;
    expect(resolveGPUSpatialContext('op', undefined, support).coordinateSpace).toBe('planar');
    expect(resolveGPUSpatialContext('op', GREAT_CIRCLE, support)).toBe(GREAT_CIRCLE);
    expect(
      resolveGPUSpatialContext('op', {coordinateSpace: 'planar', metric: 'none'}, support).metric
    ).toBe('none');
    for (const metric of ['none', 'rhumb', 'ellipsoidal'] as const) {
      expect(() =>
        resolveGPUSpatialContext('op', {coordinateSpace: 'longitude-latitude', metric}, support)
      ).toThrow('op longitude-latitude positions need the great-circle metric');
    }
    expect(() =>
      resolveGPUSpatialContext('op', {coordinateSpace: 'topology-only', metric: 'none'}, support)
    ).toThrow(/does not support topology-only/);
  });

  it('lets a space without an entry accept what metrics accepts', () => {
    const support: GPUSpatialContextSupport = {
      coordinateSpaces: ['planar', 'longitude-latitude'],
      metrics: ['none', 'native', 'great-circle', 'rhumb'],
      metricsByCoordinateSpace: {planar: ['native']}
    };
    expect(
      resolveGPUSpatialContext(
        'op',
        {coordinateSpace: 'longitude-latitude', metric: 'rhumb'},
        support
      ).metric
    ).toBe('rhumb');
    expect(() =>
      resolveGPUSpatialContext('op', {coordinateSpace: 'planar', metric: 'none'}, support)
    ).toThrow('op planar positions need the native metric');
    const either: GPUSpatialContextSupport = {
      coordinateSpaces: ['longitude-latitude'],
      metricsByCoordinateSpace: {'longitude-latitude': ['great-circle', 'rhumb']}
    };
    expect(() =>
      resolveGPUSpatialContext(
        'op',
        {coordinateSpace: 'longitude-latitude', metric: 'none'},
        either
      )
    ).toThrow('need the great-circle or rhumb metric');
  });

  it('rejects units outside the support table and ignores omitted units', () => {
    const meters = GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT;
    expect(resolveGPUSpatialContext('op', GREAT_CIRCLE, meters)).toBe(GREAT_CIRCLE);
    expect(resolveGPUSpatialContext('op', {...GREAT_CIRCLE, units: 'meters'}, meters).units).toBe(
      'meters'
    );
    expect(() =>
      resolveGPUSpatialContext('op', {...GREAT_CIRCLE, units: 'kilometers'}, meters)
    ).toThrow(/kilometers.*meters/);
    // The lenient constant honors kilometers through getGPUSphereRadius.
    expect(
      resolveGPUSpatialContext(
        'op',
        {...GREAT_CIRCLE, units: 'kilometers'},
        GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
      ).units
    ).toBe('kilometers');
    expect(
      resolveGPUSpatialContext(
        'op',
        {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        meters
      ).units
    ).toBe('native');
  });

  it('shares one great-circle radius helper that honors units', () => {
    expect(getGPUGreatCircleRadius({coordinateSpace: 'planar', metric: 'native'})).toBeUndefined();
    expect(getGPUGreatCircleRadius(GREAT_CIRCLE)).toBe(getGPUSphereRadius(GREAT_CIRCLE));
    expect(getGPUGreatCircleRadius(GREAT_CIRCLE)).toBe(6371008.8);
    expect(getGPUGreatCircleRadius({...GREAT_CIRCLE, units: 'kilometers'})).toBeCloseTo(
      6371.0088,
      9
    );
    expect(getGPUGreatCircleRadius({...GREAT_CIRCLE, sphereRadius: 1737400})).toBe(1737400);
  });
});
