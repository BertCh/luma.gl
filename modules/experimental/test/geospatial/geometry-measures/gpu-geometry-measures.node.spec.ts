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
} from '../../../src/geospatial/geometry-measures';
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

let viewSerial = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewSerial++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGeometryMeasuresProps> = {}
): GPUGeometryMeasuresProps {
  return {
    positions: view(graph, 'float32x2', 20),
    geometryType: 'polygons',
    ringOffsets: view(graph, 'uint32', 6),
    featureRingOffsets: view(graph, 'uint32', 4),
    output: {areas: view(graph, 'float32', 3), centroids: view(graph, 'float32x2', 3)},
    ...overrides
  };
}

it('GPUGeometryMeasures validates layouts and emits deterministic nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'measures'});
  const recipe = new GPUGeometryMeasures({
    ...createProps(graph),
    groupIds: view(graph, 'uint32', 3),
    groupCount: 2,
    groupOutput: {areas: view(graph, 'float32', 2), featureCounts: view(graph, 'uint32', 2)}
  });
  expect(recipe.featureCount).toBe(3);
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('geometry-measures-features');
  expect(ids[1]).toBe('geometry-measures-feature-output');
  expect(ids).toContain('geometry-measures-group-reduce');
  expect(ids.at(-1)).toBe('geometry-measures-group-output');

  expect(
    () =>
      new GPUGeometryMeasures(
        createProps(graph, {geometryType: 'lines', output: {areas: view(graph, 'float32', 3)}})
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
    () => new GPUGeometryMeasures(createProps(graph, {coordinateSystem: 'mercator' as never}))
  ).toThrow(/coordinateSystem/);
  expect(() => new GPUGeometryMeasures(createProps(graph, {holeRule: 'evenodd' as never}))).toThrow(
    /holeRule/
  );
  // Without featureRingOffsets every ring is a feature.
  const rings = new GPUGeometryMeasures(
    createProps(graph, {
      featureRingOffsets: undefined,
      output: {lengths: view(graph, 'float32', 5)}
    })
  );
  expect(rings.featureCount).toBe(5);
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
});

it('GPUGeodesicPairs and GPUGeodesicDestination validate columns and models', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'geodesic'});
  const origins = view(graph, 'float32x2', 4);
  const targets = view(graph, 'float32x2', 4);
  const pairs = new GPUGeodesicPairs({
    origins,
    targets,
    model: 'wgs84',
    output: {distances: view(graph, 'float32', 4), converged: view(graph, 'uint32', 4)}
  });
  expect(pairs.getCommandNodes(graph).map(node => node.id)).toEqual(['geodesic-pairs-pairs']);
  expect(
    () =>
      new GPUGeodesicPairs({
        origins,
        targets,
        output: {converged: view(graph, 'uint32', 4)}
      })
  ).toThrow(/requires model 'wgs84'/);
  expect(() => new GPUGeodesicPairs({origins, targets, output: {}})).toThrow(/at least one/);
  expect(
    () =>
      new GPUGeodesicPairs({
        origins,
        targets: view(graph, 'float32x2', 3),
        output: {distances: view(graph, 'float32', 4)}
      })
  ).toThrow(/aligned/);
  expect(
    () =>
      new GPUGeodesicPairs({
        origins,
        targets,
        model: 'clarke' as never,
        output: {distances: view(graph, 'float32', 4)}
      })
  ).toThrow(/model/);
  const destination = new GPUGeodesicDestination({
    origins,
    bearings: view(graph, 'float32', 4),
    distances: view(graph, 'float32', 4),
    output: {destinations: view(graph, 'float32x2', 4)}
  });
  expect(destination.getCommandNodes(graph)).toHaveLength(1);
  expect(
    () =>
      new GPUGeodesicDestination({
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
