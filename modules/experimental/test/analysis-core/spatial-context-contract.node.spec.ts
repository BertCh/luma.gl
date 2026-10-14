// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createGPULocalFrame} from '../../src/analysis-core/contracts/local-frame';
import {
  GPU_EARTH_AUTHALIC_RADIUS_METERS,
  GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE,
  GPU_WGS84_FLATTENING,
  GPU_WGS84_INVERSE_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '../../src/analysis-core/contracts/earth';
import {
  assertCompatibleGPUSpatialContexts,
  getGPUDatumKey,
  getGPUEdgeModel,
  GPU_LINEAR_EDGE_MODELS,
  getGPUAuthalicSphereRadius,
  getGPUEllipsoid,
  getGPULengthUnits,
  getGPUSpatialContextCompatibility,
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  GPU_WGS84_ELLIPSOID,
  resolveGPUSpatialContext,
  validateGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../src/analysis-core/contracts/spatial-context';

const PLANAR: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};
const GREAT_CIRCLE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
};

const LOCAL_FRAMES = ['azimuthal-equidistant', 'lambert-equal-area'] as const;

describe('spatial context measurement contract', () => {
  it('rejects the wrong local frame for each measure with the exact diagnostic', () => {
    const cases = [
      {
        kind: 'lambert-equal-area',
        measure: 'distance',
        message:
          'op measures distance; project into an azimuthal-equidistant frame, not lambert-equal-area'
      },
      {
        kind: 'azimuthal-equidistant',
        measure: 'area',
        message:
          'op measures area; project into a lambert-equal-area frame, not azimuthal-equidistant'
      }
    ] as const;
    for (const {kind, measure, message} of cases) {
      const {spatialContext} = createGPULocalFrame({origin: [12, 45], kind});
      expect(() =>
        resolveGPUSpatialContext('op', spatialContext, {
          ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
          measure
        })
      ).toThrowError(new Error(message));
    }
  });

  it('accepts matching local frames, CRS frames and frameless contexts for each measure', () => {
    for (const measure of ['distance', 'area'] as const) {
      const {spatialContext} = createGPULocalFrame({
        origin: [12, 45],
        kind: measure === 'distance' ? 'azimuthal-equidistant' : 'lambert-equal-area'
      });
      const contexts: GPUSpatialContext[] = [
        spatialContext,
        PLANAR,
        {...PLANAR, frame: {kind: 'crs', id: 'EPSG:32633', crs: 'EPSG:32633'}}
      ];
      for (const context of contexts) {
        expect(
          resolveGPUSpatialContext('op', context, {
            ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
            measure
          })
        ).toBe(context);
      }
    }
  });

  it.each(LOCAL_FRAMES)('accepts %s without a measure and declares meter output', kind => {
    const {spatialContext} = createGPULocalFrame({origin: [12, 45], kind});
    expect(spatialContext.units).toBe('meters');
    expect(resolveGPUSpatialContext('op', spatialContext)).toBe(spatialContext);
    expect(
      resolveGPUSpatialContext(
        'op',
        spatialContext,
        GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT
      )
    ).toBe(spatialContext);
  });

  it('accepts native planar meters and rejects kilometers', () => {
    expect(() => validateGPUSpatialContext('op', {...PLANAR, units: 'meters'})).not.toThrow();
    expect(() => validateGPUSpatialContext('op', {...PLANAR, units: 'kilometers'})).toThrowError(
      new Error('op native metrics must use native or meters units')
    );
  });

  it('matches planar native and meters in both directions', () => {
    const native: GPUSpatialContext = {...PLANAR, units: 'native'};
    const meters: GPUSpatialContext = {...PLANAR, units: 'meters'};
    expect(getGPUSpatialContextCompatibility(native, meters)).toEqual({compatible: true});
    expect(getGPUSpatialContextCompatibility(meters, native)).toEqual({compatible: true});
  });

  it('rejects meters and kilometers in both directions', () => {
    // Geographic contexts allow both units; native planar kilometers fail validation first.
    const meters: GPUSpatialContext = {...GREAT_CIRCLE, units: 'meters'};
    const kilometers: GPUSpatialContext = {...GREAT_CIRCLE, units: 'kilometers'};
    expect(getGPUSpatialContextCompatibility(meters, kilometers)).toEqual({
      compatible: false,
      reason: 'units differ (meters and kilometers)'
    });
    expect(getGPUSpatialContextCompatibility(kilometers, meters)).toEqual({
      compatible: false,
      reason: 'units differ (kilometers and meters)'
    });
  });
});

describe('spatial metric constants and units', () => {
  it('returns the authalic radius in the context length unit and preserves overrides', () => {
    expect(getGPUAuthalicSphereRadius(GREAT_CIRCLE)).toBe(6371007.180918475);
    expect(getGPUAuthalicSphereRadius({...GREAT_CIRCLE, units: 'meters'})).toBe(6371007.180918475);
    expect(getGPUAuthalicSphereRadius({...GREAT_CIRCLE, units: 'kilometers'})).toBe(
      6371007.180918475 / 1000
    );
    for (const units of ['meters', 'kilometers'] as const) {
      expect(getGPUAuthalicSphereRadius({...GREAT_CIRCLE, units, sphereRadius: 1737})).toBe(1737);
    }
  });

  it('matches an independent binary64 WGS84 authalic radius oracle within a micrometer', () => {
    const semiMajorAxis = 6378137;
    const flattening = 1 / 298.257223563;
    const eccentricitySquared = flattening * (2 - flattening);
    const eccentricity = Math.sqrt(eccentricitySquared);
    const polarAuthalicQ =
      (1 - eccentricitySquared) *
      (1 / (1 - eccentricitySquared) -
        Math.log((1 - eccentricity) / (1 + eccentricity)) / (2 * eccentricity));
    const authalicRadius = semiMajorAxis * Math.sqrt(polarAuthalicQ / 2);
    expect(Math.abs(GPU_EARTH_AUTHALIC_RADIUS_METERS - authalicRadius)).toBeLessThan(1e-6);
    expect(Math.abs(getGPUAuthalicSphereRadius(GREAT_CIRCLE) - authalicRadius)).toBeLessThan(1e-6);
    expect(GPU_WGS84_SEMI_MAJOR_AXIS_METERS).toBe(semiMajorAxis);
    expect(GPU_WGS84_INVERSE_FLATTENING).toBe(298.257223563);
    expect(GPU_WGS84_FLATTENING).toBe(flattening);
  });

  it('preserves the cuSpatial sinusoidal scale exactly', () => {
    const previousScale = 40_000 / 360;
    expect(GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE).toBe(previousScale);
    expect(Math.fround(GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE)).toBe(Math.fround(previousScale));
  });

  it('defaults to WGS84 ellipsoid parameters and preserves explicit overrides', () => {
    const context: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'ellipsoidal'
    };
    expect(getGPUEllipsoid(context)).toBe(GPU_WGS84_ELLIPSOID);
    expect(getGPUEllipsoid(context)).toEqual({
      semiMajorAxis: 6378137,
      flattening: 1 / 298.257223563
    });
    const ellipsoid = {semiMajorAxis: 1737400, flattening: 0};
    expect(getGPUEllipsoid({...context, ellipsoid})).toBe(ellipsoid);
  });

  it('resolves declared and default length units', () => {
    expect(getGPULengthUnits(PLANAR)).toBe('native');
    expect(getGPULengthUnits({...PLANAR, units: 'native'})).toBe('native');
    expect(getGPULengthUnits({...PLANAR, units: 'meters'})).toBe('meters');
    for (const metric of ['great-circle', 'rhumb', 'ellipsoidal'] as const) {
      expect(getGPULengthUnits({...GREAT_CIRCLE, metric})).toBe('meters');
      expect(getGPULengthUnits({...GREAT_CIRCLE, metric, units: 'meters'})).toBe('meters');
      expect(getGPULengthUnits({...GREAT_CIRCLE, metric, units: 'kilometers'})).toBe('kilometers');
    }
  });

  it('requires the native metric on planar positions when a measure is declared', () => {
    const planarNone: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'none'};
    for (const measure of ['distance', 'area'] as const) {
      expect(() =>
        resolveGPUSpatialContext('op', planarNone, {coordinateSpaces: ['planar'], measure})
      ).toThrow('op planar positions need the native metric');
    }
    expect(resolveGPUSpatialContext('op', planarNone, {coordinateSpaces: ['planar']})).toBe(
      planarNone
    );
  });
});

describe('spatial context edge model contract', () => {
  it('rejects edge models outside longitude-latitude and on ellipsoidal contexts', () => {
    expect(() => validateGPUSpatialContext('X', {...PLANAR, edgeModel: 'linear'})).toThrowError(
      new Error('X edgeModel needs a longitude-latitude coordinate space')
    );
    expect(() =>
      validateGPUSpatialContext('X', {
        ...GREAT_CIRCLE,
        edgeModel: 'unknown'
      } as unknown as GPUSpatialContext)
    ).toThrowError(new Error('X edgeModel must be linear or great-circle'));
    expect(() =>
      validateGPUSpatialContext('X', {...GREAT_CIRCLE, metric: 'ellipsoidal', edgeModel: 'linear'})
    ).toThrowError(new Error('X ellipsoidal contexts take ellipsoidalEdgeModel, not edgeModel'));
  });

  it('defaults planar and longitude-latitude edges to linear and preserves great-circle', () => {
    expect(getGPUEdgeModel(PLANAR)).toBe('linear');
    expect(getGPUEdgeModel(GREAT_CIRCLE)).toBe('linear');
    expect(getGPUEdgeModel({...GREAT_CIRCLE, edgeModel: 'great-circle'})).toBe('great-circle');
  });

  it.each([
    'none',
    'great-circle'
  ] as const)('enforces declared edge support for %s metrics', metric => {
    const support: GPUSpatialContextSupport = {
      coordinateSpaces: ['longitude-latitude'],
      metrics: ['none', 'great-circle'],
      edgeModels: GPU_LINEAR_EDGE_MODELS
    };
    const context: GPUSpatialContext = {...GREAT_CIRCLE, metric};
    const linear: GPUSpatialContext = {...context, edgeModel: 'linear'};
    const greatCircle: GPUSpatialContext = {...context, edgeModel: 'great-circle'};
    expect(resolveGPUSpatialContext('X', context, support)).toBe(context);
    expect(resolveGPUSpatialContext('X', linear, support)).toBe(linear);
    expect(() => resolveGPUSpatialContext('X', greatCircle, support)).toThrowError(
      new Error('X needs linear edges; prepare polygons with GPULonLatPolygonPreparation')
    );
    expect(
      resolveGPUSpatialContext('X', greatCircle, {
        coordinateSpaces: support.coordinateSpaces,
        metrics: support.metrics
      })
    ).toBe(greatCircle);
  });

  it('compares omitted edges as linear and reports different edge models', () => {
    const linear: GPUSpatialContext = {...GREAT_CIRCLE, edgeModel: 'linear'};
    const greatCircle: GPUSpatialContext = {...GREAT_CIRCLE, edgeModel: 'great-circle'};
    for (const context of [GREAT_CIRCLE, linear]) {
      expect(getGPUSpatialContextCompatibility(context, greatCircle)).toEqual({
        compatible: false,
        reason: 'edge models differ (linear and great-circle)'
      });
    }
    expect(getGPUSpatialContextCompatibility(GREAT_CIRCLE, linear)).toEqual({compatible: true});
    expect(getGPUSpatialContextCompatibility(linear, GREAT_CIRCLE)).toEqual({compatible: true});
  });
});

describe('spatial context horizontal datum contract', () => {
  it.each(['', '   '])('rejects an empty datum %j', datum => {
    expect(() => validateGPUSpatialContext('X', {...GREAT_CIRCLE, datum})).toThrowError(
      new Error('X datum must be a non-empty string')
    );
  });

  it('allows a datum on planar contexts', () => {
    expect(() => validateGPUSpatialContext('X', {...PLANAR, datum: 'EPSG:4258'})).not.toThrow();
  });

  it.each([
    'WGS84',
    'wgs 84',
    'epsg:4326',
    ' EPSG:6326 '
  ])('normalizes WGS84 spelling %j', datum => {
    expect(getGPUDatumKey(datum)).toBe('EPSG:4326');
  });

  it('uppercases other keys without adding aliases', () => {
    expect(getGPUDatumKey('gcj-02')).toBe('GCJ-02');
    expect(getGPUDatumKey(' nad83 ')).toBe('NAD83');
    expect(getGPUDatumKey('NAD83')).not.toBe(getGPUDatumKey('EPSG:4269'));
  });

  it('reports declared datum mismatches and the exact assertion diagnostic', () => {
    const source: GPUSpatialContext = {...GREAT_CIRCLE, datum: 'EPSG:4267'};
    const target: GPUSpatialContext = {...GREAT_CIRCLE, datum: 'EPSG:4326'};
    expect(getGPUSpatialContextCompatibility(source, target)).toEqual({
      compatible: false,
      reason: 'datums differ (EPSG:4267 and EPSG:4326)'
    });
    expect(() => assertCompatibleGPUSpatialContexts('X', source, target)).toThrowError(
      new Error('X incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)')
    );
    expect(getGPUSpatialContextCompatibility({...source, datum: 'GCJ-02'}, target)).toEqual({
      compatible: false,
      reason: 'datums differ (GCJ-02 and EPSG:4326)'
    });
  });

  it('accepts either undeclared datum and equivalent WGS84 declarations', () => {
    const declared: GPUSpatialContext = {...GREAT_CIRCLE, datum: 'EPSG:4267'};
    expect(getGPUSpatialContextCompatibility(declared, GREAT_CIRCLE)).toEqual({compatible: true});
    expect(getGPUSpatialContextCompatibility(GREAT_CIRCLE, declared)).toEqual({compatible: true});
    expect(
      getGPUSpatialContextCompatibility(
        {...GREAT_CIRCLE, datum: 'WGS84'},
        {...GREAT_CIRCLE, datum: 'epsg:4326'}
      )
    ).toEqual({compatible: true});
  });

  it('reports datum differences before metric differences', () => {
    expect(
      getGPUSpatialContextCompatibility(
        {...GREAT_CIRCLE, datum: 'EPSG:4267'},
        {...GREAT_CIRCLE, metric: 'rhumb', datum: 'EPSG:4326'}
      )
    ).toEqual({compatible: false, reason: 'datums differ (EPSG:4267 and EPSG:4326)'});
  });
});
