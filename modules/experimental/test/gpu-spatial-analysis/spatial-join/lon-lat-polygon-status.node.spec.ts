// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {log} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it, vi} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts';
import {getPointInPolygonContainmentContext} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  ARCTIC_RING,
  CHICAGO_SQUARE,
  FIJI_SPLIT,
  FIJI_TRANSMERIDIAN
} from '../lon-lat-polygons/fixtures';
import {createPolygonStatusJoin, getPolygonStatusCounts} from './lon-lat-polygon-status-fixtures';

for (const operation of [
  'GPUPointInPolygonJoin',
  'GPUSpatialPredicateJoin',
  'GPUNearestFeatureJoin'
] as const) {
  const context: GPUSpatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: operation === 'GPUNearestFeatureJoin' ? 'great-circle' : 'none'
  };
  it(`${operation} requires linear lon/lat edges`, () => {
    const device = createNullWebGPUDevice();
    for (const edgeModel of [undefined, 'linear', 'great-circle'] as const) {
      const graph = new GPUCommandGraph(device);
      const create = () => createPolygonStatusJoin(graph, operation, {...context, edgeModel}, {});
      if (edgeModel === 'great-circle') {
        expect(create).toThrow(
          `${operation} needs linear edges; prepare polygons with GPULonLatPolygonPreparation`
        );
      } else {
        expect(create).not.toThrow();
      }
    }
    device.destroy();
  });

  it(`${operation} builds requested checks only for lon/lat and validates scalar lengths`, () => {
    const device = createNullWebGPUDevice();
    for (const planar of [false, true]) {
      for (const fields of [
        [],
        ['antimeridianUncertainCount'],
        ['poleUncertainCount'],
        ['antimeridianUncertainCount', 'poleUncertainCount']
      ] as const) {
        const graph = new GPUCommandGraph(device);
        const status = Object.fromEntries(
          fields.map(name => [name, createTransientView(graph, name, 'uint32', 1)])
        );
        const join = createPolygonStatusJoin(
          graph,
          operation,
          planar ? undefined : context,
          status
        );
        const nodes = join.getCommandNodes(graph);
        const checks = nodes.filter(node => node.id.includes('-lon-lat-polygon-check-'));
        expect(checks).toHaveLength(!planar && fields.length ? 1 : 0);
        expect(
          nodes.filter(
            node =>
              node.id.includes('-clear-antimeridianUncertainCount') ||
              node.id.includes('-clear-poleUncertainCount')
          )
        ).toHaveLength(planar ? 0 : fields.length);
        if (checks.length) expect(nodes.indexOf(checks[0])).toBe(fields.length);
      }
    }
    for (const field of ['antimeridianUncertainCount', 'poleUncertainCount']) {
      const graph = new GPUCommandGraph(device);
      expect(() =>
        createPolygonStatusJoin(graph, operation, context, {
          [field]: createTransientView(graph, field, 'uint32', 0)
        })
      ).toThrow(`${operation} output.status.${field} needs one row`);
    }
    device.destroy();
  });

  if (operation !== 'GPUPointInPolygonJoin') {
    it(`${operation} checks both polygon sides after clearing counters once`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      try {
        const join = createPolygonStatusJoin(
          graph,
          operation,
          context,
          {antimeridianUncertainCount: createTransientView(graph, 'antimeridian', 'uint32', 1)},
          {polygonQueries: true}
        );
        const nodes = join.getCommandNodes(graph);
        expect(nodes.filter(node => node.id.includes('-lon-lat-polygon-check-'))).toHaveLength(2);
        expect(
          nodes.filter(node => node.id.includes('-clear-antimeridianUncertainCount'))
        ).toHaveLength(1);
      } finally {
        device.destroy();
      }
    });
  }
}

it('containment context preserves edge model and datum', () => {
  expect(
    getPointInPolygonContainmentContext({
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      edgeModel: 'great-circle',
      datum: 'WGS84'
    })
  ).toEqual({
    coordinateSpace: 'longitude-latitude',
    metric: 'none',
    edgeModel: 'great-circle',
    datum: 'WGS84'
  });
});

it('shared binary64 oracle counts unprepared rings', () => {
  expect(getPolygonStatusCounts(FIJI_TRANSMERIDIAN)).toEqual([1, 0]);
  expect(getPolygonStatusCounts(ARCTIC_RING)).toEqual([1, 1]);
  expect(getPolygonStatusCounts(FIJI_SPLIT)).toEqual([0, 0]);
  expect(getPolygonStatusCounts(CHICAGO_SQUARE)).toEqual([0, 0]);
});

it('numeric dwithin warns only for short great-circle lon/lat distances', () => {
  const warning = vi.spyOn(log, 'warn').mockReturnValue(() => {});
  const device = createNullWebGPUDevice();
  try {
    for (const [distance, planar, warns] of [
      [5, false, true],
      [50, false, false],
      [5, true, false]
    ] as const) {
      warning.mockClear();
      const graph = new GPUCommandGraph(device);
      createPolygonStatusJoin(
        graph,
        'GPUSpatialPredicateJoin',
        planar ? undefined : {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        {},
        {distance}
      );
      expect(warning).toHaveBeenCalledTimes(warns ? 1 : 0);
      if (warns)
        expect(warning).toHaveBeenCalledWith(
          'GPUSpatialPredicateJoin distance is too short for float32 lon/lat; use createGPULocalFrame'
        );
    }
  } finally {
    warning.mockRestore();
    device.destroy();
  }
});
