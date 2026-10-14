// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {GPUSpatialClustering} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  GPUGroupGeometry,
  GPUGroupConvexHull
} from '../../../src/gpu-spatial-analysis/group-geometry/index';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const PLANAR = {coordinateSpace: 'planar', metric: 'native', units: 'native'} as const;
const GREAT_CIRCLE = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'kilometers'
} as const;
const LAMBERT = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'equal-area', kind: 'local-lambert-equal-area'}
} as const;

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `context-view-${serial++}`, format, length);
  return {device, graph, view};
}

it('cluster-and-outline defaults to planar and forwards the resolved context to every stage', () => {
  for (const spatialContext of [
    undefined,
    {
      ...PLANAR,
      units: 'meters',
      frame: {id: 'distance-frame', kind: 'local-azimuthal-equidistant'}
    } as const
  ]) {
    const {device, graph, view} = createFixture();
    try {
      const result = addClusterAndOutlineRecipe(graph, {
        points: {kind: 'points', positions: view('float32x2', 8)},
        spatialContext,
        clusteringParameters: view('float32', 16),
        gridSize: [4, 4],
        geometryParameters: view('float32', 16),
        maximumClusterCount: 2,
        maximumVerticesPerHull: 8,
        hullCapacity: 16
      });
      expect(result.spatialContext).toEqual(spatialContext ?? PLANAR);
      for (const contributor of result.contributors) {
        expect(
          contributor instanceof GPUSpatialClustering ||
            contributor instanceof GPUGroupGeometry ||
            contributor instanceof GPUGroupConvexHull ||
            contributor instanceof GPUGeometryMeasures
        ).toBe(true);
        if (
          contributor instanceof GPUSpatialClustering ||
          contributor instanceof GPUGroupGeometry ||
          contributor instanceof GPUGroupConvexHull ||
          contributor instanceof GPUGeometryMeasures
        ) {
          expect(contributor.spatialContext).toBe(result.spatialContext);
        }
      }
    } finally {
      device.destroy();
    }
  }
});

it('cluster-and-outline rejects equal-area frames and longitude/latitude points', () => {
  const {device, graph, view} = createFixture();
  const create = (spatialContext: GPUSpatialContext) =>
    addClusterAndOutlineRecipe(graph, {
      points: {kind: 'points', positions: view('float32x2', 8)},
      spatialContext,
      clusteringParameters: view('float32', 16),
      gridSize: [4, 4],
      geometryParameters: view('float32', 16),
      maximumClusterCount: 2,
      maximumVerticesPerHull: 8,
      hullCapacity: 16
    });
  try {
    expect(() => create(LAMBERT)).toThrow(/measures distance/);
    expect(() => create(GREAT_CIRCLE)).toThrow(
      /does not support longitude-latitude coordinates; project positions to a planar frame first/
    );
  } finally {
    device.destroy();
  }
});

function createRegression(spatialContext?: GPUSpatialContext) {
  const {device, graph, view} = createFixture();
  try {
    return addSpatialRegressionRecipe(graph, {
      predictors: view('float32', 8),
      predictorCount: 1,
      response: view('float32', 8),
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 16),
        weights: view('float32', 16)
      },
      parameters: new GPUParameterBuffer(device, {
        id: 'regression-parameters',
        format: 'float32',
        length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      }),
      localFits: {
        positions: view('float32x2', 8),
        spatialContext,
        parameters: view('float32', 64)
      }
    });
  } finally {
    device.destroy();
  }
}

it('regression local fits forward great-circle context and default to native planar', () => {
  for (const spatialContext of [undefined, GREAT_CIRCLE]) {
    const result = createRegression(spatialContext);
    const contributor = result.contributors.find(
      value => value instanceof GPUGeographicallyWeightedRegression
    );
    expect(contributor).toBeInstanceOf(GPUGeographicallyWeightedRegression);
    if (contributor instanceof GPUGeographicallyWeightedRegression) {
      expect(contributor.spatialContext).toEqual(spatialContext ?? PLANAR);
      if (spatialContext) {
        expect(contributor.spatialContext).toBe(spatialContext);
      }
    }
  }
});

it('regression local fits reject Lambert frames before using planar distances', () => {
  expect(() => createRegression(LAMBERT)).toThrow(/gwr measures distance/);
  expect(() => createRegression({...GREAT_CIRCLE, metric: 'none'})).toThrow(
    /need the great-circle metric/
  );
});
