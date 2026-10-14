// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts';
import {
  GPUBufferSelection,
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin,
  GPUSpatialJoinCandidates,
  GPUSpatialPredicateJoin
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-mask';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import type {GPURegionSelection} from '../../../src/gpu-spatial-analysis/region-statistics/region-statistics-types';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createPolygonConsumer} from '../zonal-statistics/polygon-consumer-status-fixtures';
import {createPolygonStatusJoin} from './lon-lat-polygon-status-fixtures';

const OPERATIONS = [
  {name: 'GPUPointInPolygonJoin', side: 'polygons', linearEdges: true},
  {name: 'GPUSpatialPredicateJoin', side: 'right', linearEdges: true},
  {name: 'GPUSpatialJoinCandidates', side: 'right', linearEdges: false},
  {name: 'GPUNearestFeatureJoin', side: 'features', linearEdges: true},
  {name: 'GPUBufferSelection', side: 'features', linearEdges: false},
  {name: 'GPUZonalStatistics', side: 'features', linearEdges: true},
  {name: 'GPURegionMask', side: 'region', linearEdges: true},
  {name: 'GPURegionStatistics', side: 'region', linearEdges: true},
  {name: 'GPULineLengthPerPolygon', side: 'polygons', linearEdges: true}
] as const;
type OperationName = (typeof OPERATIONS)[number]['name'];

function createContextFixture(
  graph: GPUCommandGraph,
  operation: OperationName,
  spatialContext: GPUSpatialContext | undefined
) {
  if (
    operation === 'GPUPointInPolygonJoin' ||
    operation === 'GPUSpatialPredicateJoin' ||
    operation === 'GPUNearestFeatureJoin'
  ) {
    const baseline = createPolygonStatusJoin(graph, operation, spatialContext, {});
    if (baseline instanceof GPUPointInPolygonJoin) {
      return {
        baseline,
        create: (polygonsSpatialContext?: GPUSpatialContext) =>
          new GPUPointInPolygonJoin({...baseline.props, polygonsSpatialContext})
      };
    }
    if (baseline instanceof GPUSpatialPredicateJoin) {
      return {
        baseline,
        create: (rightSpatialContext?: GPUSpatialContext) =>
          new GPUSpatialPredicateJoin({...baseline.props, rightSpatialContext})
      };
    }
    return {
      baseline,
      create: (featuresSpatialContext?: GPUSpatialContext) =>
        new GPUNearestFeatureJoin({...baseline.props, featuresSpatialContext})
    };
  }
  if (operation === 'GPUBufferSelection' || operation === 'GPUSpatialJoinCandidates') {
    const points = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'points', 'float32x2', 1)
    };
    const features = {
      kind: 'points' as const,
      positions: createTransientView(graph, 'features', 'float32x2', 1)
    };
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    if (operation === 'GPUBufferSelection') {
      const baseline = new GPUBufferSelection({
        spatialContext,
        points,
        features,
        distance: createTransientView(graph, 'distance', 'float32', 1),
        candidateCapacity: 4,
        output: {outputMask: scalar('mask')}
      });
      return {
        baseline,
        create: (featuresSpatialContext?: GPUSpatialContext) =>
          new GPUBufferSelection({...baseline.props, featuresSpatialContext})
      };
    }
    const baseline = new GPUSpatialJoinCandidates({
      spatialContext,
      left: points,
      right: features,
      output: {
        leftIds: scalar('left-ids'),
        rightIds: scalar('right-ids'),
        status: {count: scalar('count'), overflow: scalar('overflow')}
      }
    });
    return {
      baseline,
      create: (rightSpatialContext?: GPUSpatialContext) =>
        new GPUSpatialJoinCandidates({...baseline.props, rightSpatialContext})
    };
  }
  const baseline = createPolygonConsumer(graph, operation, spatialContext, {});
  if (baseline instanceof GPUZonalStatistics) {
    return {
      baseline,
      create: (featuresSpatialContext?: GPUSpatialContext) =>
        new GPUZonalStatistics({...baseline.props, featuresSpatialContext})
    };
  }
  if (baseline instanceof GPULineLengthPerPolygon) {
    return {
      baseline,
      create: (polygonsSpatialContext?: GPUSpatialContext) =>
        new GPULineLengthPerPolygon({...baseline.props, polygonsSpatialContext})
    };
  }
  if (baseline instanceof GPURegionMask) {
    return {
      baseline,
      create: (regionSpatialContext?: GPUSpatialContext) =>
        new GPURegionMask({...baseline.props, regionSpatialContext})
    };
  }
  if (baseline instanceof GPURegionStatistics) {
    return {
      baseline,
      create: (regionSpatialContext?: GPUSpatialContext) =>
        new GPURegionStatistics({...baseline.props, regionSpatialContext})
    };
  }
  throw new Error(`Unexpected fixture for ${operation}`);
}

for (const {name, side, linearEdges} of OPERATIONS) {
  const primary: GPUSpatialContext =
    name === 'GPUSpatialJoinCandidates'
      ? {coordinateSpace: 'planar', metric: 'native', datum: 'EPSG:4326'}
      : {
          coordinateSpace: 'longitude-latitude',
          metric:
            name === 'GPUNearestFeatureJoin' ||
            name === 'GPUBufferSelection' ||
            name === 'GPULineLengthPerPolygon'
              ? 'great-circle'
              : 'none',
          datum: 'EPSG:4326'
        };
  describe(`${name} secondary spatial context`, () => {
    it('keeps the same command node list with omitted, undefined and matching contexts', () => {
      const device = createNullWebGPUDevice();
      try {
        for (const spatialContext of [undefined, primary]) {
          const baselineGraph = new GPUCommandGraph(device);
          const {baseline} = createContextFixture(baselineGraph, name, spatialContext);
          const expected = baseline.getCommandNodes(baselineGraph).map(node => node.id);
          expect(expected.length).toBeGreaterThan(0);
          for (const secondary of [undefined, {...baseline.spatialContext}]) {
            const graph = new GPUCommandGraph(device);
            const {create} = createContextFixture(graph, name, spatialContext);
            const operation = create(secondary);
            expect(operation.spatialContext).toBe(baseline.spatialContext);
            expect(operation.getCommandNodes(graph).map(node => node.id)).toEqual(expected);
          }
        }
      } finally {
        device.destroy();
      }
    });

    it('rejects mismatched datums at construction with the outer operator id', () => {
      const device = createNullWebGPUDevice();
      try {
        const {baseline, create} = createContextFixture(new GPUCommandGraph(device), name, primary);
        expect(() => create({...primary, datum: 'EPSG:4267'})).toThrowError(
          new Error(
            `${baseline.id} ${side}SpatialContext incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)`
          )
        );
      } finally {
        device.destroy();
      }
    });

    if (linearEdges) {
      it('rejects great-circle secondary edges at construction', () => {
        const device = createNullWebGPUDevice();
        try {
          const {baseline, create} = createContextFixture(
            new GPUCommandGraph(device),
            name,
            primary
          );
          expect(() => create({...primary, edgeModel: 'great-circle'})).toThrowError(
            new Error(
              `${baseline.id} ${side}SpatialContext needs linear edges; prepare polygons with GPULonLatPolygonPreparation`
            )
          );
        } finally {
          device.destroy();
        }
      });
    }
  });
}

for (const operation of ['GPURegionMask', 'GPURegionStatistics'] as const) {
  for (const kind of ['rectangle', 'polygon'] as const) {
    it(`${operation} rejects a secondary context for a screen-space ${kind}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const {baseline} = createContextFixture(graph, operation, undefined);
        const screenTransform = createTransientView(graph, 'transform', 'float32', 20);
        const region =
          kind === 'rectangle'
            ? {kind, screenTransform, bounds: createTransientView(graph, 'bounds', 'float32', 4)}
            : {
                kind,
                screenTransform,
                vertices: createTransientView(graph, 'vertices', 'float32x2', 4),
                vertexCount: createTransientView(graph, 'screen-vertex-count', 'uint32', 1)
              };
        if (baseline instanceof GPURegionMask) {
          expect(
            () =>
              new GPURegionMask({
                ...baseline.props,
                region,
                regionSpatialContext: baseline.spatialContext
              })
          ).toThrowError(
            new Error(`${baseline.id} regionSpatialContext needs a region in position coordinates`)
          );
        } else if (baseline instanceof GPURegionStatistics) {
          expect(
            () =>
              new GPURegionStatistics({
                ...baseline.props,
                region,
                regionSpatialContext: baseline.spatialContext
              })
          ).toThrowError(
            new Error(`${baseline.id} regionSpatialContext needs a region in position coordinates`)
          );
        } else {
          throw new Error('Expected a region fixture');
        }
      } finally {
        device.destroy();
      }
    });
  }
}

for (const kind of ['pick-region', 'mask'] as const) {
  it(`GPURegionStatistics rejects a secondary context for a ${kind} selection`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const region: GPURegionSelection =
        kind === 'pick-region'
          ? {kind, picks: createTransientView(graph, 'picks', 'uint32', 4)}
          : {kind, mask: createTransientView(graph, 'mask', 'uint32', 1)};
      const props = {
        region,
        positions: createTransientView(graph, 'points', 'float32x2', 1),
        output: {summary: createTransientView(graph, 'summary', 'uint32', 8)}
      };
      const baseline = new GPURegionStatistics(props);
      expect(baseline.getCommandNodes(graph).length).toBeGreaterThan(0);
      expect(
        () => new GPURegionStatistics({...props, regionSpatialContext: baseline.spatialContext})
      ).toThrowError(
        new Error('region-statistics regionSpatialContext needs a region in position coordinates')
      );
    } finally {
      device.destroy();
    }
  });
}
