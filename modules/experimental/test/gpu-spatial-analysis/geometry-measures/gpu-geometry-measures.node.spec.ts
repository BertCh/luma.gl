// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGeodesicDestination,
  GPUGeodesicPairs,
  GPUGeometryMeasures,
  type GPUGeometryMeasuresProps
} from '../../../src/gpu-spatial-analysis/geometry-measures';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getCentralAngle, interpolateGreatCircle} from './geodesic-oracle';
import {
  createFlatGeometry,
  getSphericalBoxArea,
  getVincentyDistance,
  getWgs84BoxArea,
  measureFeature
} from './geometry-measures-oracle';
import {getVincentyDirect, getVincentyInverse} from './vincenty-oracle';
import {getVincentyWGSL} from '../../../src/gpu-spatial-analysis/geometry-measures/geodesic-kernels';
import {getAuthalicConstants} from '../../../src/gpu-spatial-analysis/geometry-measures/geometry-measures-kernels';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';

let viewSerial = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewSerial++}`, format, length);
}

/** Polygon props (3 features, 4 polygons, 5 rings) with any prop overridden. */
function createProps(
  graph: GPUCommandGraph,
  overrides: Record<string, unknown> = {}
): GPUGeometryMeasuresProps {
  return {
    spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
    positions: view(graph, 'float32x2', 20),
    geometryType: 'polygons',
    ringOffsets: view(graph, 'uint32', 6),
    polygonOffsets: view(graph, 'uint32', 5),
    featureOffsets: view(graph, 'uint32', 4),
    output: {areas: view(graph, 'float32', 3), centroids: view(graph, 'float32x2', 3)},
    ...overrides
  } as GPUGeometryMeasuresProps;
}

it('GPUGeometryMeasures validates layouts and emits deterministic nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'measures'});
  const contributor = new GPUGeometryMeasures({
    ...createProps(graph),
    groupIds: view(graph, 'uint32', 3),
    groupCount: 2,
    groupOutput: {areas: view(graph, 'float32', 2), featureCounts: view(graph, 'uint32', 2)}
  });
  expect(contributor.featureCount).toBe(3);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('geometry-measures-features');
  expect(ids[1]).toBe('geometry-measures-feature-output');
  expect(ids).toContain('geometry-measures-group-reduce');
  expect(ids.at(-1)).toBe('geometry-measures-group-output');

  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {
          geometryType: 'lines',
          lineOffsets: view(graph, 'uint32', 6),
          featureOffsets: view(graph, 'uint32', 4),
          output: {areas: view(graph, 'float32', 3)}
        })
      )
  ).toThrow(/requires geometryType 'polygons'/);
  expect(
    () =>
      new GPUGeometryMeasures(createProps(graph, {output: {lengths: view(graph, 'float32', 2)}}))
  ).toThrow(/must hold 3 rows/);
  expect(() => new GPUGeometryMeasures(createProps(graph, {output: {}}))).toThrow(
    /at least one output/
  );
  expect(
    () =>
      new GPUGeometryMeasures(createProps(graph, {groupOutput: {areas: view(graph, 'float32', 2)}}))
  ).toThrow(/groupOutput requires groupIds/);
  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'native'}
        })
      )
  ).toThrow(/metric/);
  // Without featureOffsets every line is a feature.
  const lines = new GPUGeometryMeasures(
    createProps(graph, {
      geometryType: 'lines',
      lineOffsets: view(graph, 'uint32', 6),
      featureOffsets: undefined,
      output: {lengths: view(graph, 'float32', 5)}
    })
  );
  expect(lines.featureCount).toBe(5);
  // Without featureOffsets every point is a feature.
  const points = new GPUGeometryMeasures(
    createProps(graph, {
      geometryType: 'points',
      featureOffsets: undefined,
      output: {vertexCounts: view(graph, 'uint32', 20)}
    })
  );
  expect(points.featureCount).toBe(20);
});

it('geometry measures oracle matches closed forms', () => {
  const box = [
    [
      [10, 20],
      [11, 20],
      [11, 21],
      [10, 21]
    ]
  ];
  const spherical = measureFeature(box, {
    isPolygon: true,
    coordinateSystem: 'spherical',
    radius: 6371008.8
  });
  expect(spherical.area / getSphericalBoxArea(10, 20, 11, 21, 6371008.8)).toBeCloseTo(1, 12);
  const wgs84 = measureFeature(box, {isPolygon: true, coordinateSystem: 'wgs84'});
  expect(wgs84.area / getWgs84BoxArea(10, 20, 11, 21)).toBeCloseTo(1, 12);
  // Vincenty: one degree of latitude at the equator is 110,574.4 m on WGS84.
  expect(getVincentyDistance([0, 0], [0, 1]).distance).toBeCloseTo(110574.389, 2);
  expect(getCentralAngle([0, 0], [90, 0])).toBeCloseTo(Math.PI / 2, 12);
  expect(interpolateGreatCircle([170, 0], [-170, 0], 0.5)[0]).toBeCloseTo(180, 10);
  const flat = createFlatGeometry([[[[0, 0]]], [], [box[0], box[0]]]);
  expect(Array.from(flat.ringOffsets)).toEqual([0, 1, 5, 9]);
  expect(Array.from(flat.featureRingOffsets)).toEqual([0, 1, 1, 3]);
  // Four-level polygon layout: one polygon per feature by default.
  expect(Array.from(flat.polygonOffsets)).toEqual([0, 1, 1, 3]);
  expect(Array.from(flat.featureOffsets)).toEqual([0, 1, 2, 3]);
  expect(Array.from(flat.pointOffsets)).toEqual([0, 1, 1, 9]);
});

it('GPUGeodesicPairs and GPUGeodesicDestination validate columns and models', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'geodesic'});
  const origins = view(graph, 'float32x2', 4);
  const targets = view(graph, 'float32x2', 4);
  const pairs = new GPUGeodesicPairs({
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', units: 'meters'},
    origins,
    targets,

    output: {distances: view(graph, 'float32', 4), status: {converged: view(graph, 'uint32', 4)}}
  });
  expect(pairs.getCommandNodes(graph).map(node => node.id)).toEqual(['geodesic-pairs-pairs']);
  expect(
    () =>
      new GPUGeodesicPairs({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        origins,
        targets,
        output: {status: {converged: view(graph, 'uint32', 4)}}
      })
  ).toThrow(/requires model 'wgs84'/);
  expect(
    () =>
      new GPUGeodesicPairs({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        origins,
        targets,
        output: {}
      })
  ).toThrow(/at least one/);
  expect(
    () =>
      new GPUGeodesicPairs({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        origins,
        targets: view(graph, 'float32x2', 3),
        output: {distances: view(graph, 'float32', 4)}
      })
  ).toThrow(/aligned/);
  expect(
    () =>
      new GPUGeodesicPairs({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'clarke' as never,
          units: 'meters'
        },
        origins,
        targets,

        output: {distances: view(graph, 'float32', 4)}
      })
  ).toThrow(/metric/);
  const destination = new GPUGeodesicDestination({
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    },
    origins,
    bearings: view(graph, 'float32', 4),
    distances: view(graph, 'float32', 4),
    output: {destinations: view(graph, 'float32x2', 4)}
  });
  expect(destination.getCommandNodes(graph)).toHaveLength(1);
  expect(
    () =>
      new GPUGeodesicDestination({
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        origins,
        bearings: view(graph, 'float32', 3),
        distances: view(graph, 'float32', 4),
        output: {destinations: view(graph, 'float32x2', 4)}
      })
  ).toThrow(/aligned/);
});

it('vincenty oracle round-trips direct and inverse solutions', () => {
  const inverse = getVincentyInverse([-0.13, 51.5], [2.35, 48.85]);
  const direct = getVincentyDirect([-0.13, 51.5], inverse.initialBearing, inverse.distance);
  expect(direct.destination[0]).toBeCloseTo(2.35, 9);
  expect(direct.destination[1]).toBeCloseTo(48.85, 9);
  expect(direct.finalBearing).toBeCloseTo(inverse.finalBearing, 9);
  expect(getVincentyInverse([0, 0], [179.7, 0.2]).converged).toBe(false);
});

it('GPUGeometryMeasures describes its measure model and validates status', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'measures-model'});
  const planar = new GPUGeometryMeasures(createProps(graph));
  expect(planar.measures).toEqual({
    lengthModel: 'planar',
    areaModel: 'planar',
    lengthUnits: 'native',
    areaUnits: 'native-squared'
  });
  const describe = (
    spatialContext: GPUGeometryMeasuresProps['spatialContext'],
    edge?: 'coordinate-linear'
  ) =>
    new GPUGeometryMeasures(createProps(graph, {spatialContext, ellipsoidalEdgeModel: edge}))
      .measures;
  expect(
    describe({coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'})
  ).toEqual({
    lengthModel: 'great-circle',
    areaModel: 'sphere-equal-area',
    lengthUnits: 'meters',
    areaUnits: 'square-meters'
  });
  expect(
    describe({coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'kilometers'})
      .areaUnits
  ).toBe('square-kilometers');
  expect(describe({coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'}).areaModel).toBe(
    'geodesic'
  );
  expect(
    describe({coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'}, 'coordinate-linear')
      .areaModel
  ).toBe('wgs84-authalic');

  const ids = new GPUGeometryMeasures(
    createProps(graph, {
      output: {
        areas: view(graph, 'float32', 3),
        status: {invalidCount: view(graph, 'uint32', 1)}
      }
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('geometry-measures-clear-invalid-count');
  expect(ids).toContain('geometry-measures-invalid-count');
  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {
          output: {
            areas: view(graph, 'float32', 3),
            status: {invalidCount: view(graph, 'float32', 1) as never}
          }
        })
      )
  ).toThrow(/invalidCount/);
});

it('GPUGeometryMeasures has no signedAreas column (areas is never negative)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'measures-no-signed'});
  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {
          output: {areas: view(graph, 'float32', 3), signedAreas: view(graph, 'float32', 3)}
        })
      )
  ).toThrow(/output\.signedAreas is not a measure column/);
});

it('geometry measures resolves native planar defaults and accepts meter frames for both measures', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  delete props.spatialContext;
  const planar = new GPUGeometryMeasures(props);
  expect(planar.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  expect(planar.coordinateSystem).toBe('planar');
  expect(planar.measures.lengthUnits).toBe('native');
  expect(planar.measures.areaUnits).toBe('native-squared');
  for (const kind of ['local-lambert-equal-area', 'local-azimuthal-equidistant'] as const) {
    const spatialContext: GPUSpatialContext = {
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'meters',
      frame: {id: 'f', kind}
    };
    const measures = new GPUGeometryMeasures(
      createProps(graph, {id: `measures-${kind}`, spatialContext})
    );
    expect(measures.measures.lengthUnits).toBe('meters');
    expect(measures.measures.areaUnits).toBe('square-meters');
    expect(measures.getCommandNodes(graph).length).toBeGreaterThan(0);
    expect(
      () =>
        new GPUGeodesicPairs({
          spatialContext,
          origins: view(graph, 'float32x2', 1),
          targets: view(graph, 'float32x2', 1),
          output: {distances: view(graph, 'float32', 1)}
        })
    ).toThrow('geodesic-pairs does not support planar coordinates');
  }
  for (const metric of ['none', 'rhumb'] as const) {
    expect(
      () =>
        new GPUGeometryMeasures(
          createProps(graph, {
            spatialContext: {coordinateSpace: 'longitude-latitude', metric}
          })
        )
    ).toThrow(
      'geometry-measures longitude-latitude positions need the great-circle or ellipsoidal metric'
    );
  }
  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'topology-only', metric: 'none'}
        })
      )
  ).toThrow('geometry-measures does not support topology-only coordinates');
  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {
          ellipsoidalEdgeModel: 'geodesic'
        })
      )
  ).toThrow('ellipsoidalEdgeModel requires an ellipsoidal spatial context');
});

it('geometry measures separates mean distance and authalic area radii in context units', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  for (const units of ['meters', 'kilometers'] as const) {
    for (const sphereRadius of [undefined, 1000]) {
      const measures = new GPUGeometryMeasures(
        createProps(graph, {
          id: `measures-${units}-${sphereRadius}`,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            units,
            sphereRadius
          }
        })
      );
      const divisor = units === 'kilometers' ? 1000 : 1;
      expect(measures.radius).toBe(sphereRadius ?? 6371008.8 / divisor);
      expect(measures.areaRadius).toBe(sphereRadius ?? 6371007.180918475 / divisor);
      expect(measures.getCommandNodes(graph).length).toBeGreaterThan(0);
    }
  }
});

it('ellipsoidal contributors honor custom ellipsoids and kilometers', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const ellipsoid = {semiMajorAxis: 6378388, flattening: 1 / 297};
  for (const units of ['meters', 'kilometers'] as const) {
    const spatialContext: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'ellipsoidal',
      units,
      ellipsoid
    };
    for (const ellipsoidalEdgeModel of ['coordinate-linear', 'geodesic'] as const) {
      const measures = new GPUGeometryMeasures(
        createProps(graph, {
          id: `measures-${units}-${ellipsoidalEdgeModel}`,
          spatialContext,
          ellipsoidalEdgeModel
        })
      );
      expect(measures.measures.lengthUnits).toBe(units);
      expect(measures.measures.areaUnits).toBe(`square-${units}`);
      expect(measures.getCommandNodes(graph).length).toBeGreaterThan(0);
    }
    const pairs = new GPUGeodesicPairs({
      spatialContext,
      origins: view(graph, 'float32x2', 1),
      targets: view(graph, 'float32x2', 1),
      output: {distances: view(graph, 'float32', 1)}
    });
    const destination = new GPUGeodesicDestination({
      spatialContext,
      origins: view(graph, 'float32x2', 1),
      bearings: view(graph, 'float32', 1),
      distances: view(graph, 'float32', 1),
      output: {destinations: view(graph, 'float32x2', 1)}
    });
    for (const contributor of [pairs, destination]) {
      expect(contributor.model).toBe('wgs84');
      expect(contributor.spatialContext.units).toBe(units);
      expect(contributor.ellipsoid).toEqual({
        ...ellipsoid,
        semiMajorAxis: ellipsoid.semiMajorAxis / (units === 'kilometers' ? 1000 : 1)
      });
      expect(contributor.getCommandNodes(graph)).toHaveLength(1);
    }
  }
});

it('custom ellipsoid authalic and Vincenty constants match binary64 formulas', () => {
  const ellipsoid = {semiMajorAxis: 6378388, flattening: 1 / 297};
  const eccentricitySquared = ellipsoid.flattening * (2 - ellipsoid.flattening);
  const eccentricity = Math.sqrt(eccentricitySquared);
  const polarQ = 1 + ((1 - eccentricitySquared) * Math.atanh(eccentricity)) / eccentricity;
  const authalic = getAuthalicConstants(ellipsoid);
  expect(authalic.eccentricitySquared).toBe(eccentricitySquared);
  expect(authalic.polarQ).toBeCloseTo(polarQ, 14);
  expect(authalic.authalicRadius / (ellipsoid.semiMajorAxis * Math.sqrt(polarQ / 2))).toBeCloseTo(
    1,
    14
  );
  expect(authalic.sinAuthalicScale).toBeCloseTo((2 * (1 - eccentricitySquared)) / polarQ, 14);
  const source = getVincentyWGSL(16, ellipsoid);
  const semiMinorAxis = ellipsoid.semiMajorAxis * (1 - ellipsoid.flattening);
  for (const [name, expected] of Object.entries({
    VINCENTY_A: ellipsoid.semiMajorAxis,
    VINCENTY_B: semiMinorAxis,
    VINCENTY_F: ellipsoid.flattening,
    VINCENTY_SECOND_ECCENTRICITY_SQUARED:
      (ellipsoid.semiMajorAxis ** 2 - semiMinorAxis ** 2) / semiMinorAxis ** 2
  })) {
    const literal = source.match(new RegExp(`const ${name}: f32 = ([^;]+);`));
    expect(literal).not.toBeNull();
    expect(Number(literal?.[1])).toBe(Math.fround(expected));
  }
  expect(getAuthalicConstants({semiMajorAxis: 1000, flattening: 0})).toEqual({
    eccentricitySquared: 0,
    polarQ: 2,
    authalicRadius: 1000,
    sinAuthalicScale: 1
  });
});
