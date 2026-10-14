// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {GPUTrajectoryMetrics} from '../../../src/gpu-spatial-analysis/trajectory-analysis/index';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample
} from '../../../src/gpu-spatial-analysis/trajectory-interpolation/index';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  ENCOUNTER_REACH_ORACLE_CASES,
  GEODESIC_ORACLE_CASES,
  GEODESIC_ORACLE_TOLERANCES,
  getCentralAngle,
  getEncounterMaximumDistance,
  getGreatCircleDistance,
  getInitialBearingDegrees,
  interpolateGreatCircle
} from './geodesic-trajectory-oracle';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const PLANAR = {coordinateSpace: 'planar', metric: 'native', units: 'native'} as const;

let uniqueIndex = 0;
const unique = (id: string) => `${id}-${uniqueIndex++}`;

function createGraph() {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: unique('graph')});
}

it('binary64 great-circle oracle functions match pyproj on the mean-radius sphere', () => {
  for (const testCase of GEODESIC_ORACLE_CASES) {
    const distance = getGreatCircleDistance(testCase.a, testCase.b);
    expect(Math.abs(distance - testCase.distance)).toBeLessThan(
      GEODESIC_ORACLE_TOLERANCES.distanceRelative * testCase.distance
    );
    const bearing = getInitialBearingDegrees(testCase.a, testCase.b);
    expect(Math.abs(bearing - testCase.bearingDegrees)).toBeLessThan(
      GEODESIC_ORACLE_TOLERANCES.bearingDegrees
    );
    for (const point of testCase.points) {
      const [longitude, latitude] = interpolateGreatCircle(testCase.a, testCase.b, point.fraction);
      expect(Math.abs(longitude - point.lon)).toBeLessThan(GEODESIC_ORACLE_TOLERANCES.pointDegrees);
      expect(Math.abs(latitude - point.lat)).toBeLessThan(GEODESIC_ORACLE_TOLERANCES.pointDegrees);
    }
    // Central angle times the radius is the distance by construction.
    expect(getCentralAngle(testCase.a, testCase.b) * 6371008.8).toBeCloseTo(distance, 6);
  }
});

it('the encounter lattice reach formula matches the pyproj-walked bound', () => {
  for (const testCase of ENCOUNTER_REACH_ORACLE_CASES) {
    expect(getEncounterMaximumDistance(testCase.cellDegrees, testCase.maximumLatitude)).toBeCloseTo(
      testCase.maximumDistance,
      6
    );
    // The walked reach stays within one cell: the bound is conservative and nearly tight.
    expect(testCase.observedLongitudeReachDegrees).toBeLessThanOrEqual(testCase.cellDegrees);
    expect(testCase.observedLongitudeReachDegrees).toBeGreaterThan(testCase.cellDegrees * 0.999999);
  }
});

it('GPUTrajectoryMetrics accepts a great-circle context and rejects other geographic contexts', () => {
  const graph = createGraph();
  const props = {
    positions: createTransientView(graph, unique('positions'), 'float32x2', 8),
    timestamps: createTransientView(graph, unique('timestamps'), 'float32', 8),
    trackOffsets: createTransientView(graph, unique('offsets'), 'uint32', 3),
    output: {
      trackLengths: createTransientView(graph, unique('lengths'), 'float32', 2),
      stepHeadings: createTransientView(graph, unique('headings'), 'float32', 8)
    }
  };
  const metrics = new GPUTrajectoryMetrics({...props, spatialContext: GREAT_CIRCLE});
  expect(metrics.spatialContext).toEqual(GREAT_CIRCLE);
  expect(metrics.getCommandNodes(graph).map(node => node.id)).toContain(
    'trajectory-metrics-step-columns'
  );
  expect(
    new GPUTrajectoryMetrics({...props, spatialContext: {...GREAT_CIRCLE, sphereRadius: 1}})
  ).toBeDefined();
  // Omitted context stays native planar.
  expect(new GPUTrajectoryMetrics(props).spatialContext).toEqual(PLANAR);
  expect(
    () =>
      new GPUTrajectoryMetrics({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/great-circle/);
  expect(
    () =>
      new GPUTrajectoryMetrics({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'}
      })
  ).toThrow(/great-circle metric/);
  expect(
    () =>
      new GPUTrajectoryMetrics({...props, spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}})
  ).not.toThrow();
  expect(
    () =>
      new GPUTrajectoryMetrics({
        ...props,
        spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
      })
  ).toThrow(/discrete-grid/);
});

it('GPUTrajectoryResample and GPUTrajectoryPlayhead accept a great-circle context', () => {
  const graph = createGraph();
  const positions = createTransientView(graph, unique('positions'), 'float32x2', 8);
  const timestamps = createTransientView(graph, unique('timestamps'), 'float32', 8);
  const trackOffsets = createTransientView(graph, unique('offsets'), 'uint32', 3);
  const resample = new GPUTrajectoryResample({
    positions,
    timestamps,
    trackOffsets,
    spacing: 'arc-length',
    sampleCount: 4,
    spatialContext: GREAT_CIRCLE,
    output: {samples: createTransientView(graph, unique('samples'), 'float32x2', 8)}
  });
  expect(resample.spatialContext).toEqual(GREAT_CIRCLE);
  expect(() => new GPUTrajectoryResample({...resample.props, spatialContext: AREA_FRAME})).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(
    () => new GPUTrajectoryResample({...resample.props, spatialContext: DISTANCE_FRAME})
  ).not.toThrow();
  expect(
    () =>
      new GPUTrajectoryResample({
        ...resample.props,
        spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}
      })
  ).not.toThrow();
  expect(resample.getCommandNodes(graph).map(node => node.id)).toContain(
    'trajectory-resample-arc-length'
  );
  const playhead = new GPUTrajectoryPlayhead({
    positions,
    timestamps,
    trackOffsets,
    parameters: createTransientView(graph, unique('parameters'), 'float32', 4),
    spatialContext: GREAT_CIRCLE,
    output: {
      currentPositions: createTransientView(graph, unique('current'), 'float32x2', 2),
      headings: createTransientView(graph, unique('headings'), 'float32', 2)
    }
  });
  expect(playhead.spatialContext).toEqual(GREAT_CIRCLE);
  expect(() => new GPUTrajectoryPlayhead({...playhead.props, spatialContext: AREA_FRAME})).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(
    () => new GPUTrajectoryPlayhead({...playhead.props, spatialContext: DISTANCE_FRAME})
  ).not.toThrow();
  expect(
    () =>
      new GPUTrajectoryPlayhead({
        ...playhead.props,
        spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}
      })
  ).not.toThrow();
  expect(playhead.getCommandNodes(graph).length).toBeGreaterThan(1);
  expect(
    () =>
      new GPUTrajectoryPlayhead({
        positions,
        timestamps,
        trackOffsets,
        parameters: createTransientView(graph, unique('parameters'), 'float32', 4),
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
        output: {currentPositions: createTransientView(graph, unique('current'), 'float32x2', 2)}
      })
  ).toThrow(/great-circle/);
});

it('GPUTrajectoryEncounters derives a conservative great-circle search distance', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const props = {
    samples: view('float32x2', 12),
    trackCount: 3,
    bucketCount: 4,
    distance: view('float32', 1),
    cellSize: 2,
    bounds: [-180, -60, 180, 60] as [number, number, number, number],
    hitCapacity: 32,
    output: {
      pairs: {
        ids: view('uint32', 8),
        minimumDistances: view('float32', 8),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      },
      status: {candidateOverflow: view('uint32', 1)}
    }
  };
  const planar = new GPUTrajectoryEncounters({
    ...props,
    cellSize: 10,
    bounds: [0, 0, 100, 100]
  });
  expect(planar.maximumDistance).toBe(10);
  expect(planar.spatialContext).toEqual(PLANAR);

  const geodesic = new GPUTrajectoryEncounters({...props, spatialContext: GREAT_CIRCLE});
  // pyproj-pinned: cell 2 degrees at latitude 60 (see ENCOUNTER_REACH_ORACLE_CASES).
  expect(geodesic.maximumDistance).toBeCloseTo(111178.03178664722, 6);
  const kilometers = new GPUTrajectoryEncounters({
    ...props,
    spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}
  });
  expect(kilometers.maximumDistance).toBeCloseTo(111.17803178664722, 9);
  expect(kilometers.maximumDistance).toBeCloseTo(geodesic.maximumDistance / 1000, 9);
  expect(() => new GPUTrajectoryEncounters({...props, spatialContext: AREA_FRAME})).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(
    () => new GPUTrajectoryEncounters({...props, spatialContext: DISTANCE_FRAME})
  ).not.toThrow();
  expect(geodesic.columns).toBe(180);
  expect(geodesic.rows).toBe(60);
  expect(geodesic.getCommandNodes(graph).length).toBeGreaterThan(10);

  // Larger latitude bounds shrink the searchable distance.
  const polar = new GPUTrajectoryEncounters({
    ...props,
    bounds: [-180, -80, 180, 80],
    spatialContext: GREAT_CIRCLE
  });
  expect(polar.maximumDistance).toBeLessThan(geodesic.maximumDistance);

  expect(
    () =>
      new GPUTrajectoryEncounters({
        ...props,
        bounds: [-180, -90, 180, 90],
        spatialContext: GREAT_CIRCLE
      })
  ).toThrow(/bounds/);
  expect(
    () =>
      new GPUTrajectoryEncounters({
        ...props,
        bounds: [-190, -60, 180, 60],
        spatialContext: GREAT_CIRCLE
      })
  ).toThrow(/bounds/);
  expect(
    () => new GPUTrajectoryEncounters({...props, cellSize: 120, spatialContext: GREAT_CIRCLE})
  ).toThrow(/cellSize/);
  expect(
    () =>
      new GPUTrajectoryEncounters({
        ...props,
        cellSize: 90,
        bounds: [0, -10, 100, 10],
        spatialContext: GREAT_CIRCLE
      })
  ).toThrow(/cellSize/);
  expect(
    () =>
      new GPUTrajectoryEncounters({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/great-circle/);
});

it('planar-only movement operators reject longitude-latitude contexts', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const zoneProps = {
    tracks: {
      kind: 'trajectories' as const,
      positions: view('float32x2', 8),
      timestamps: view('float32', 8),
      trackOffsets: view('uint32', 3)
    },
    zones: {
      kind: 'polygons' as const,
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      ringOffsets: view('uint32', 3)
    },
    candidateCapacity: 16,
    trackEventCapacity: 4,
    output: {
      events: {
        ids: view('uint32', 8),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    }
  };
  expect(new GPUZoneEvents(zoneProps).spatialContext).toEqual(PLANAR);
  // Crossing membership and dwell times do not measure distance or area.
  expect(() => new GPUZoneEvents({...zoneProps, spatialContext: DISTANCE_FRAME})).not.toThrow();
  expect(() => new GPUZoneEvents({...zoneProps, spatialContext: AREA_FRAME})).not.toThrow();
  expect(new GPUZoneEvents({...zoneProps, spatialContext: PLANAR}).spatialContext).toEqual(PLANAR);
  expect(() => new GPUZoneEvents({...zoneProps, spatialContext: GREAT_CIRCLE})).toThrow(
    /does not support longitude-latitude/
  );

  const similarityProps = {
    linesA: {
      kind: 'lines' as const,
      positions: view('float32x2', 6),
      lineOffsets: view('uint32', 3)
    },
    pairA: view('uint32', 2),
    pairB: view('uint32', 2),
    output: {hausdorff: view('float32', 2)}
  };
  expect(new GPUTrackSimilarity(similarityProps).spatialContext).toEqual(PLANAR);
  // Track similarity gained a great-circle path (unit E7); zone events stay planar.
  expect(
    new GPUTrackSimilarity({...similarityProps, spatialContext: GREAT_CIRCLE}).spatialContext
  ).toEqual(GREAT_CIRCLE);
});

const DISTANCE_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
} as const;
const AREA_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
} as const;
