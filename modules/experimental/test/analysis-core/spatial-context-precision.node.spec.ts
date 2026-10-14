// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {log} from '@luma.gl/core';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {
  getGPULongitudeLatitudeQuantum,
  isGPUShortLongitudeLatitudeDistance,
  warnGPUShortLongitudeLatitudeDistance,
  type GPUSpatialContext
} from '../../src/analysis-core/contracts/spatial-context';

const GREAT_CIRCLE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
const PLANAR: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};

describe('spatial context float32 longitude-latitude precision', () => {
  afterEach(() => vi.restoreAllMocks());

  it('measures the great-circle quantum in the context units and honors radius overrides', () => {
    const meters = getGPULongitudeLatitudeQuantum(GREAT_CIRCLE);
    expect(meters).toBeCloseTo(1.6967, 5);
    expect(getGPULongitudeLatitudeQuantum({...GREAT_CIRCLE, units: 'kilometers'})).toBeCloseTo(
      meters / 1000,
      12
    );
    expect(getGPULongitudeLatitudeQuantum({...GREAT_CIRCLE, sphereRadius: 1000})).toBe(
      ((2 ** -16 * Math.PI) / 180) * 1000
    );
  });

  it('uses the ellipsoid semi-major axis for ellipsoidal metrics', () => {
    expect(getGPULongitudeLatitudeQuantum({...GREAT_CIRCLE, metric: 'ellipsoidal'})).toBeCloseTo(
      1.6986,
      5
    );
  });

  it('returns zero for planar coordinates and longitude-latitude without a metric', () => {
    expect(getGPULongitudeLatitudeQuantum(PLANAR)).toBe(0);
    expect(getGPULongitudeLatitudeQuantum({...GREAT_CIRCLE, metric: 'none'})).toBe(0);
  });

  it.each([
    [16, true],
    [17, false],
    [0, false],
    [Infinity, false],
    [NaN, false],
    [-5, false]
  ] as const)('classifies distance %s as short: %s', (distance, expected) => {
    expect(isGPUShortLongitudeLatitudeDistance(GREAT_CIRCLE, distance)).toBe(expected);
  });

  it('warns without throwing and keeps the message identical across short distances', () => {
    const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
    const message = 'GPUOp distance is too short for float32 lon/lat; use createGPULocalFrame';
    expect(() => warnGPUShortLongitudeLatitudeDistance('GPUOp', GREAT_CIRCLE, 5)).not.toThrow();
    expect(() => warnGPUShortLongitudeLatitudeDistance('GPUOp', GREAT_CIRCLE, 6)).not.toThrow();
    expect(warning.mock.calls).toEqual([[message], [message]]);
  });

  it('does not warn or throw for planar, omitted or sufficiently long distances', () => {
    const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
    expect(() => warnGPUShortLongitudeLatitudeDistance('GPUOp', PLANAR, 5)).not.toThrow();
    expect(() => warnGPUShortLongitudeLatitudeDistance('GPUOp', undefined, 5)).not.toThrow();
    expect(() =>
      warnGPUShortLongitudeLatitudeDistance('GPUOp', GREAT_CIRCLE, undefined)
    ).not.toThrow();
    expect(() => warnGPUShortLongitudeLatitudeDistance('GPUOp', GREAT_CIRCLE, 50)).not.toThrow();
    expect(warning).not.toHaveBeenCalled();
  });
});
