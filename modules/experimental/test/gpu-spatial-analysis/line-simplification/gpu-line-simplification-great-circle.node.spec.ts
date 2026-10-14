// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../../src/gpu-spatial-analysis/contracts/earth';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function createLine(overrides: Record<string, unknown> = {}) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'line-context'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const line = new GPULineSimplification({
    positions: view('float32x2', 5),
    trackOffsets: view('uint32', 2),
    importance: view('float32', 5),
    timestamps: view('float32', 5),
    ...overrides
  });
  return {graph, line};
}

it('GPULineSimplification Douglas-Peucker accepts the great-circle context in meters', () => {
  const planar = createLine().line;
  expect(planar.sphereRadius).toBeUndefined();
  const {graph, line} = createLine({spatialContext: GREAT_CIRCLE});
  expect(line.spatialContext).toEqual(GREAT_CIRCLE);
  expect(line.sphereRadius).toBe(GPU_EARTH_MEAN_RADIUS_METERS);
  expect(
    createLine({spatialContext: {...GREAT_CIRCLE, units: 'meters', sphereRadius: 1000}}).line
      .sphereRadius
  ).toBe(1000);
  expect(line.getCommandNodes(graph).length).toBeGreaterThan(0);
  const timeRatio = createLine({spatialContext: GREAT_CIRCLE, metric: 'time-ratio'});
  expect(timeRatio.line.getCommandNodes(timeRatio.graph).length).toBeGreaterThan(0);
});

it('GPULineSimplification rejects other longitude-latitude contexts, kilometers and great-circle Visvalingam', () => {
  for (const metric of ['none', 'ellipsoidal', 'rhumb'] as const) {
    expect(() =>
      createLine({spatialContext: {coordinateSpace: 'longitude-latitude', metric}})
    ).toThrow();
  }
  expect(() => createLine({spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}})).toThrow(
    /kilometers/
  );
  expect(() => createLine({spatialContext: GREAT_CIRCLE, method: 'visvalingam'})).toThrow(
    /planar only/
  );
  expect(createLine({method: 'visvalingam'}).line.sphereRadius).toBeUndefined();
});
