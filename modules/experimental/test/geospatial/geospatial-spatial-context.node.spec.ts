// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUHaversineDistance,
  GPUPairwisePointDistance,
  GPUPairwisePointLinestringNearest,
  GPUPairwisePointSegmentDistance,
  GPUPointSpatialQuery,
  GPUSinusoidalProjection,
  type GPUPointSpatialQueryKind
} from '../../src/geospatial';
import type {GPUSpatialContext} from '../../src/gpu-spatial-analysis/contracts/spatial-context';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const LONGITUDE_LATITUDE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
};
const LONGITUDE_LATITUDE_NO_METRIC: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'none'
};
const PLANAR: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};

function createGraph(): GPUCommandGraph {
  return new GPUCommandGraph(createNullWebGPUDevice());
}

describe('geospatial planar kernels reject longitude/latitude', () => {
  it('GPUPairwisePointDistance and GPUPairwisePointSegmentDistance', () => {
    const graph = createGraph();
    const position = (id: string) => createTransientView(graph, id, 'float32x2', 4);
    const output = createTransientView(graph, 'output', 'float32', 4);
    const left = position('left');
    const right = position('right');
    for (const spatialContext of [LONGITUDE_LATITUDE, LONGITUDE_LATITUDE_NO_METRIC]) {
      expect(() => new GPUPairwisePointDistance({left, right, output, spatialContext})).toThrow(
        /project positions to a planar frame first/
      );
      expect(
        () =>
          new GPUPairwisePointSegmentDistance({
            points: left,
            segmentStarts: right,
            segmentEnds: position(`end-${spatialContext.metric}`),
            output,
            spatialContext
          })
      ).toThrow(/project positions to a planar frame first/);
    }
    for (const spatialContext of [undefined, PLANAR]) {
      const distance = new GPUPairwisePointDistance({left, right, output, spatialContext});
      expect(distance.spatialContext.coordinateSpace).toBe('planar');
    }
  });

  it('GPUPairwisePointLinestringNearest', () => {
    const graph = createGraph();
    expect(
      () =>
        new GPUPairwisePointLinestringNearest({
          points: createTransientView(graph, 'points', 'float32x2', 1),
          linestringPositions: createTransientView(graph, 'vertices', 'float32x2', 2),
          geometryOffsets: createTransientView(graph, 'geometry-offsets', 'uint32', 2),
          linestringOffsets: createTransientView(graph, 'linestring-offsets', 'uint32', 2),
          output: createTransientView(graph, 'output', 'float32', 1),
          spatialContext: LONGITUDE_LATITUDE
        })
    ).toThrow(/project positions to a planar frame first/);
  });

  it('GPUPointSpatialQuery rejects lon/lat radius queries but accepts lon/lat selections', () => {
    const graph = createGraph();
    const output = {
      ids: createTransientView(graph, 'ids', 'uint32', 4),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    };
    const positions = createTransientView(graph, 'positions', 'float32x2', 4);
    const create = (kind: GPUPointSpatialQueryKind, spatialContext: GPUSpatialContext) =>
      new GPUPointSpatialQuery({
        id: `query-${kind}-${spatialContext.metric}`,
        positions,
        kind,
        query: createTransientView(
          graph,
          `query-${kind}-${spatialContext.metric}`,
          'float32',
          kind === 'radius' ? 3 : 4
        ),
        output,
        spatialContext
      });
    expect(() => create('radius', LONGITUDE_LATITUDE_NO_METRIC)).toThrow(
      /project positions to a planar frame first/
    );
    expect(create('bounds', LONGITUDE_LATITUDE_NO_METRIC).spatialContext.coordinateSpace).toBe(
      'longitude-latitude'
    );
    // Bounds comparisons are not great-circle tests, so a lon/lat metric is refused.
    expect(() => create('bounds', LONGITUDE_LATITUDE)).toThrow(/great-circle metric/);
    expect(create('radius', PLANAR).spatialContext.coordinateSpace).toBe('planar');
  });
});

describe('GPUHaversineDistance units', () => {
  it('defaults to meters on the IUGG mean Earth radius', () => {
    const graph = createGraph();
    const left = createTransientView(graph, 'left', 'float32x2', 2);
    const right = createTransientView(graph, 'right', 'float32x2', 2);
    const output = createTransientView(graph, 'output', 'float32', 2);
    const meters = new GPUHaversineDistance({left, right, output});
    // R1 = (2a + b) / 3 with a = 6378137, b = a (1 - 1 / 298.257223563) = 6371008.771415 m
    // (python3 -I); the IUGG rounds it to 6371008.8 m.
    expect(meters.units).toBe('meters');
    expect(meters.radius).toBe(6371008.8);
    const kilometers = new GPUHaversineDistance({left, right, output, units: 'kilometers'});
    expect(kilometers.radius).toBeCloseTo(6371.0088, 9);
    const custom = new GPUHaversineDistance({left, right, output, units: 'kilometers', radius: 1});
    expect(custom.radius).toBe(1);
    expect(
      () =>
        new GPUHaversineDistance({
          left,
          right,
          output,
          units: 'miles' as unknown as 'meters'
        })
    ).toThrow(/units must be meters or kilometers/);
  });
});

describe('GPUSinusoidalProjection longitude wrapping', () => {
  it('keeps the cuSpatial default (no wrap) and accepts an opt-in wrap', () => {
    const graph = createGraph();
    const positions = createTransientView(graph, 'positions', 'float32x2', 2);
    const output = createTransientView(graph, 'output', 'float32x2', 2);
    expect(new GPUSinusoidalProjection({positions, output}).wrapLongitude).toBe(false);
    expect(
      new GPUSinusoidalProjection({positions, output, origin: [179, 0], wrapLongitude: true})
        .wrapLongitude
    ).toBe(true);
  });
});
