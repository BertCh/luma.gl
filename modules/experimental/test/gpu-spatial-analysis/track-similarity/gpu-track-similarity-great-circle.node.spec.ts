// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../../src/gpu-spatial-analysis/contracts/earth';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function createProps() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'similarity-context'});
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number,
    id: string
  ) => createTransientView(graph, id, format, length);
  return {
    graph,
    props: {
      positionsA: view('float32x2', 6, 'positions'),
      offsetsA: view('uint32', 3, 'offsets'),
      pairA: view('uint32', 2, 'pair-a'),
      pairB: view('uint32', 2, 'pair-b'),
      hausdorff: view('float32', 2, 'hausdorff'),
      frechet: view('float32', 2, 'frechet'),
      maxDistance: view('float32', 2, 'max-distance')
    }
  };
}

it('GPUTrackSimilarity accepts the great-circle context and defaults the sphere to the mean radius', () => {
  const {graph, props} = createProps();
  const similarity = new GPUTrackSimilarity({...props, spatialContext: GREAT_CIRCLE});
  expect(similarity.spatialContext).toEqual(GREAT_CIRCLE);
  expect(similarity.sphereRadius).toBe(GPU_EARTH_MEAN_RADIUS_METERS);
  expect(
    new GPUTrackSimilarity({
      ...props,
      spatialContext: {...GREAT_CIRCLE, units: 'meters', sphereRadius: 1000}
    }).sphereRadius
  ).toBe(1000);
  expect(similarity.getCommandNodes(graph).map(node => node.id)).toEqual([
    'track-similarity-status',
    'track-similarity-hausdorff',
    'track-similarity-max-distance',
    'track-similarity-frechet'
  ]);
});

it('GPUTrackSimilarity stays planar without a context and rejects other longitude-latitude contexts', () => {
  const {props} = createProps();
  expect(new GPUTrackSimilarity(props).sphereRadius).toBeUndefined();
  expect(
    new GPUTrackSimilarity({
      ...props,
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'}
    }).sphereRadius
  ).toBeUndefined();
  for (const metric of ['none', 'ellipsoidal', 'rhumb'] as const) {
    expect(
      () =>
        new GPUTrackSimilarity({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric}
        })
    ).toThrow();
  }
  expect(
    () => new GPUTrackSimilarity({...props, spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}})
  ).toThrow(/kilometers/);
  expect(
    () =>
      new GPUTrackSimilarity({
        ...props,
        spatialContext: {coordinateSpace: 'planar', metric: 'great-circle'}
      })
  ).toThrow();
});
