// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {log} from '@luma.gl/core';
import {afterEach, expect, it, vi} from 'vitest';
import type {GPUSpatialContext} from '../../src/analysis-core/contracts/spatial-context';
import {getGPUNeighborSearchParameterValues} from '../../src/gpu-spatial-analysis/neighbor-search/neighbor-search-parameters';
import {getGPUSpatialClusteringParameterValues} from '../../src/gpu-spatial-analysis/spatial-clustering/spatial-clustering-parameters';
import {getGPUInverseDistanceWeightingParameterValues} from '../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {getGPUKrigingParameterValues} from '../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {getGPULineSimplificationParameterValues} from '../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';
import {getGPUOutlineGeometryParameterValues} from '../../src/gpu-spatial-analysis/outline-geometry/gpu-outline-geometry';
import {getGPUMapMatchingParameterValues} from '../../src/gpu-network/map-matching/gpu-map-matching';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import type {GPURasterMetadata} from '../../src/gpu-raster/types';

const LONGITUDE_LATITUDE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
};
const PLANAR: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native', units: 'meters'};
const GRID: GPURasterMetadata = {
  width: 2,
  height: 2,
  affine: [1, 0, 0, 0, 1, 0],
  pixelInterpretation: 'area'
};
const BOUNDS = [0, 0, 1, 1] as const;
const VARIOGRAM = {model: 'spherical', nugget: 0, sill: 1, range: 100} as const;
const WRITERS: {
  operation: string;
  field: string;
  pack: (spatialContext?: GPUSpatialContext) => Float32Array;
}[] = [
  {
    operation: 'GPUNeighborSearch',
    field: 'radius',
    pack: spatialContext =>
      getGPUNeighborSearchParameterValues({bounds: BOUNDS, radius: 5, spatialContext})
  },
  {
    operation: 'GPUSpatialClustering',
    field: 'epsilon',
    pack: spatialContext =>
      getGPUSpatialClusteringParameterValues({
        bounds: BOUNDS,
        epsilon: 5,
        minimumPoints: 2,
        spatialContext
      })
  },
  {
    operation: 'GPUInverseDistanceWeighting',
    field: 'searchRadius',
    pack: spatialContext =>
      getGPUInverseDistanceWeightingParameterValues({searchRadius: 5, spatialContext}, GRID)
  },
  {
    operation: 'GPUKriging',
    field: 'searchRadius',
    pack: spatialContext =>
      getGPUKrigingParameterValues({searchRadius: 5, variogram: VARIOGRAM, spatialContext}, GRID)
  },
  {
    operation: 'GPUKriging',
    field: 'variogram.range',
    pack: spatialContext =>
      getGPUKrigingParameterValues(
        {searchRadius: 100, variogram: {...VARIOGRAM, range: 5}, spatialContext},
        GRID
      )
  },
  {
    operation: 'GPULineSimplification',
    field: 'tolerance',
    pack: spatialContext => getGPULineSimplificationParameterValues({tolerance: 5, spatialContext})
  },
  {
    operation: 'GPUOutlineGeometry',
    field: 'distance',
    pack: spatialContext => getGPUOutlineGeometryParameterValues({distance: 5, spatialContext})
  },
  {
    operation: 'GPUMapMatching',
    field: 'sigma',
    pack: spatialContext =>
      getGPUMapMatchingParameterValues({sigma: 5, beta: 50, searchRadius: 100, spatialContext})
  },
  {
    operation: 'GPUMapMatching',
    field: 'searchRadius',
    pack: spatialContext =>
      getGPUMapMatchingParameterValues({sigma: 100, beta: 50, searchRadius: 5, spatialContext})
  },
  {
    operation: 'GPUGeographicallyWeightedRegression',
    field: 'bandwidths',
    pack: spatialContext =>
      getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [100, 5], spatialContext})
  }
];

afterEach(() => vi.restoreAllMocks());

it.each(WRITERS)('$operation warns for a short $field without changing packed values', ({
  operation,
  pack
}) => {
  const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
  const baseline = pack();
  expect(warning).not.toHaveBeenCalled();
  expect(pack(PLANAR)).toEqual(baseline);
  expect(warning).not.toHaveBeenCalled();
  expect(pack(LONGITUDE_LATITUDE)).toEqual(baseline);
  expect(warning).toHaveBeenCalledExactlyOnceWith(
    `${operation} distance is too short for float32 lon/lat; use createGPULocalFrame`
  );
});

it('does not warn for distanceFloor, Visvalingam areas or adaptive neighbor counts', () => {
  const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
  getGPUNeighborSearchParameterValues({
    bounds: BOUNDS,
    radius: 100,
    distanceFloor: 5,
    spatialContext: LONGITUDE_LATITUDE
  });
  const simplification = {tolerance: 5, method: 'visvalingam'} as const;
  expect(
    getGPULineSimplificationParameterValues({...simplification, spatialContext: LONGITUDE_LATITUDE})
  ).toEqual(getGPULineSimplificationParameterValues(simplification));
  const regression = {bandwidths: [5, 10], bandwidthMode: 'adaptive'} as const;
  expect(
    getGPUGeographicallyWeightedRegressionParameterValues({
      ...regression,
      spatialContext: LONGITUDE_LATITUDE
    })
  ).toEqual(getGPUGeographicallyWeightedRegressionParameterValues(regression));
  expect(warning).not.toHaveBeenCalled();
});

it('checks each fixed bandwidth and uses the same message for every short distance', () => {
  const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
  getGPUGeographicallyWeightedRegressionParameterValues({
    bandwidths: [5, 100, 10],
    bandwidthMode: 'fixed',
    spatialContext: LONGITUDE_LATITUDE
  });
  expect(warning.mock.calls).toEqual([
    [
      'GPUGeographicallyWeightedRegression distance is too short for float32 lon/lat; use createGPULocalFrame'
    ],
    [
      'GPUGeographicallyWeightedRegression distance is too short for float32 lon/lat; use createGPULocalFrame'
    ]
  ]);
});
