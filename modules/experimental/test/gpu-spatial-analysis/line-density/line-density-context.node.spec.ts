// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const props = {
    positions: view('float32x2', 2),
    lineOffsets: view('uint32', 2),
    output: {
      lengths: view('float32', 1),
      densities: view('float32', 1),
      status: {overflow: view('uint32', 1)}
    }
  };
  return {
    graph,
    createDensity: (spatialContext?: GPUSpatialContext) =>
      new GPULineDensity({
        ...props,
        spatialContext,
        columns: 1,
        rows: 1,
        recordCapacity: 4,
        parameters: view('float32', 4)
      }),
    createLength: (spatialContext?: GPUSpatialContext) =>
      new GPULineLengthPerPolygon({
        ...props,
        spatialContext,
        pairCapacity: 4,
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', 4),
          ringOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          featureOffsets: view('uint32', 2)
        }
      })
  };
}

it('GPULineDensity defaults to native planar and validates metric and area frames', () => {
  const {createDensity} = createFixture();
  const planar = createDensity();
  expect(planar.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  expect(planar.coordinateSystem).toBe('planar');
  expect(planar.radius).toBe(1);
  expect(planar.areaRadius).toBe(1);
  expect(() => createDensity({coordinateSpace: 'longitude-latitude', metric: 'none'})).toThrow(
    'line-density longitude-latitude positions need the great-circle metric'
  );
  expect(() =>
    createDensity({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'meters',
      frame: {id: 'distance-frame', kind: 'local-azimuthal-equidistant'}
    })
  ).toThrow(
    'line-density measures area; project into a lambert-equal-area frame, not azimuthal-equidistant'
  );
  expect(() =>
    createDensity({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'meters',
      frame: {id: 'area-frame', kind: 'local-lambert-equal-area'}
    })
  ).not.toThrow();
});

it.each([
  {units: 'meters' as const, radius: 6371008.8, areaRadius: 6371007.180918475},
  {units: 'kilometers' as const, radius: 6371.0088, areaRadius: 6371.007180918475},
  {units: 'kilometers' as const, radius: 12.5, areaRadius: 12.5, sphereRadius: 12.5}
])('GPULineDensity separates distance and area radii in $units ($radius)', options => {
  const {graph, createDensity} = createFixture();
  const contributor = createDensity({
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: options.units,
    sphereRadius: options.sphereRadius
  });
  expect(contributor.radius).toBeCloseTo(options.radius, 10);
  expect(contributor.areaRadius).toBeCloseTo(options.areaRadius, 10);
  const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const density = kernels.find(({id}) => id === 'line-density-density')!.source;
  const emit = kernels.find(({id}) => id === 'line-density-emit')!.source;
  expect(Number(density.match(/const AREA_RADIUS: f32 = ([^;]+);/)?.[1])).toBe(
    Math.fround(options.areaRadius)
  );
  expect(Number(emit.match(/const RADIUS: f32 = ([^;]+);/)?.[1])).toBe(Math.fround(options.radius));
  expect(density).toContain('AREA_RADIUS * AREA_RADIUS');
  expect(() => new WgslReflect(density)).not.toThrow();
  expect(() => new WgslReflect(emit)).not.toThrow();
});

it('GPULineLengthPerPolygon validates great-circle and distance frames; planar ignores radius', () => {
  const {graph, createLength} = createFixture();
  expect(() => createLength({coordinateSpace: 'longitude-latitude', metric: 'none'})).toThrow(
    'line-length-per-polygon longitude-latitude positions need the great-circle metric'
  );
  expect(() =>
    createLength({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'meters',
      frame: {id: 'area-frame', kind: 'local-lambert-equal-area'}
    })
  ).toThrow(
    'line-length-per-polygon measures distance; project into an azimuthal-equidistant frame, not lambert-equal-area'
  );
  const planar = createLength({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'distance-frame', kind: 'local-azimuthal-equidistant'}
  });
  expect(planar.radius).toBe(1);
  const kernels = getGPUCommandNodeKernelSources(planar.getCommandNodes(graph));
  const clip = kernels.find(({source}) => source.includes('fn getPieceLength'))!.source;
  expect(clip).toContain('return (t1 - t0) * length(b - a);');
  expect(clip).not.toContain('* RADIUS');
  expect(
    createLength({
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'kilometers'
    }).radius
  ).toBeCloseTo(6371.0088, 10);
});
