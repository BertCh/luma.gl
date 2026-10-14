// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {
  GPU_PLANAR_SPATIAL_CONTEXT,
  type GPUSpatialContext
} from '../../../src/analysis-core/contracts/spatial-context';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('nearest feature weights carry the join context and require distance-compatible frames', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `context-${serial++}`, format, length);
  const create = (spatialContext?: GPUSpatialContext) =>
    new GPUNearestFeatureWeights({
      spatialContext,
      slotCapacity: 2,
      neighborIds: view('uint32', 6),
      neighborCounts: view('uint32', 3),
      neighborDistances: view('float32', 6),
      output: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 6),
        weights: view('float32', 6),
        distances: view('float32', 6),
        status: {overflow: view('uint32', 1)}
      }
    });
  const greatCircle = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'kilometers'
  } as const;
  expect(create().outputWeights.spatialContext).toBe(GPU_PLANAR_SPATIAL_CONTEXT);
  const adapter = create(greatCircle);
  expect(adapter.outputWeights.spatialContext).toBe(greatCircle);
  expect(adapter.outputWeights.distances).toBe(adapter.props.output.distances);
  expect(adapter.outputWeights.information?.provenance).toEqual({
    source: 'GPUNearestFeatureWeights',
    criterion: 'knn',
    k: 2
  });
  const lambert = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'local', kind: 'local-lambert-equal-area'}
  } as const;
  expect(() => create(lambert)).toThrow(/measures distance/);
  expect(() =>
    create({...lambert, frame: {...lambert.frame, kind: 'local-azimuthal-equidistant'}})
  ).not.toThrow();
  device.destroy();
});
