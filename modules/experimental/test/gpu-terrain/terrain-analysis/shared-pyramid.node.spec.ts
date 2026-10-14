// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/index';
import {
  getGPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramid,
  type GPURasterExtremaPyramidOutput
} from '../../../src/gpu-raster/raster-pyramid/index';
import {
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility
} from '../../../src/gpu-terrain/point-horizon/index';
import {
  GPUTerrainCumulativeViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed
} from '../../../src/gpu-terrain/terrain-analysis/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const WIDTH = 40;
const HEIGHT = 32;
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT, cellSize: 10});

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shared-pyramid-node'});
  let viewCount = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2' | 'float32x4'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${viewCount++}`, format, length);
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('elevation', 'float32', WIDTH * HEIGHT)}
  };
  let pyramidCount = 0;
  const makeOutput = (
    width: number,
    height: number,
    options: {footprint?: 'cell' | 'bilinear'; shortBy?: number} = {}
  ): GPURasterExtremaPyramidOutput => {
    const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width, height}), {
      firstBlockSize: 4,
      footprint: options.footprint
    });
    return {
      layout,
      combined: view(
        `pyramid-${pyramidCount++}`,
        'float32',
        2 * layout.length - (options.shortBy ?? 0)
      )
    };
  };
  const consumers = (pyramid: GPURasterExtremaPyramidOutput, traversal: 'march' | 'pyramid') => ({
    viewshed: () =>
      new GPUTerrainViewshed({
        grid: GRID,
        elevation,
        parameters: view('viewshed-settings', 'float32', 10),
        output: {visibility: view('viewshed-out', 'uint32', WIDTH * HEIGHT)},
        traversal,
        pyramid
      }),
    lineOfSight: () =>
      new GPUTerrainLineOfSight({
        grid: GRID,
        elevation,
        pairs: view('pairs', 'float32x4', 4),
        parameters: view('los-settings', 'float32', 12),
        output: {visibility: view('los-out', 'uint32', 4)},
        traversal,
        pyramid
      }),
    cumulative: () =>
      new GPUTerrainCumulativeViewshed({
        grid: GRID,
        elevation,
        observers: view('observers', 'float32x2', 2),
        parameters: view('cumulative-settings', 'float32', 12),
        output: {visibleCount: view('cumulative-out', 'uint32', WIDTH * HEIGHT)},
        traversal,
        pyramid
      }),
    profile: () =>
      new GPUPointHorizonProfile({
        grid: GRID,
        elevation,
        observers: view('profile-observers', 'float32x4', 2),
        parameters: view('profile-settings', 'float32', 8),
        output: {tangent: view('tangent', 'float32', 2 * 720)},
        maximumDistance: 500,
        cellSize: 10,
        traversal,
        pyramid
      }),
    horizonVisibility: () =>
      new GPUPointHorizonVisibility({
        grid: GRID,
        elevation,
        observers: view('horizon-observers', 'float32x4', 2),
        targets: view('targets', 'float32x4', 3),
        parameters: view('horizon-settings', 'float32', 12),
        output: {visibility: view('horizon-out', 'uint32', 3)},
        maximumDistance: 500,
        cellSize: 10,
        traversal,
        pyramid
      })
  });
  return {graph, elevation, makeOutput, consumers};
}

it('every terrain consumer accepts a matching shared pyramid', () => {
  const {makeOutput, consumers} = createFixture();
  for (const create of Object.values(consumers(makeOutput(WIDTH, HEIGHT), 'pyramid'))) {
    expect(() => create()).not.toThrow();
  }
});

it('every terrain consumer rejects a mismatched shared pyramid', () => {
  const {makeOutput, consumers} = createFixture();
  const cases: [string, GPURasterExtremaPyramidOutput, RegExp][] = [
    ['grid dimensions', makeOutput(WIDTH + 4, HEIGHT), /pyramid covers 44x32 pixels/],
    ['grid height', makeOutput(WIDTH, HEIGHT - 1), /pyramid covers 40x31 pixels/],
    ['footprint', makeOutput(WIDTH, HEIGHT, {footprint: 'cell'}), /'bilinear' footprint/],
    ['combined length', makeOutput(WIDTH, HEIGHT, {shortBy: 1}), /must contain/]
  ];
  for (const [name, pyramid, message] of cases) {
    for (const [consumer, create] of Object.entries(consumers(pyramid, 'pyramid'))) {
      expect(create, `${consumer} with a ${name} mismatch`).toThrow(message);
    }
  }
});

it('a shared pyramid requires the pyramid traversal', () => {
  const {makeOutput, consumers} = createFixture();
  for (const [consumer, create] of Object.entries(consumers(makeOutput(WIDTH, HEIGHT), 'march'))) {
    expect(create, consumer).toThrow(/pyramid/);
  }
});

it('GPURasterExtremaPyramid exposes combined output and rejects mixing it with maximum', () => {
  const {graph, elevation} = createFixture();
  const layout = getGPURasterExtremaPyramidLayout(GRID, {firstBlockSize: 4});
  const combined = createTransientView(graph, 'combined', 'float32', 2 * layout.length);
  const pyramid = new GPURasterExtremaPyramid({
    grid: GRID,
    input: elevation,
    output: {combined}
  });
  expect(pyramid.output).toEqual({layout, combined});
  expect(
    () =>
      new GPURasterExtremaPyramid({
        grid: GRID,
        input: elevation,
        output: {
          combined,
          maximum: createTransientView(graph, 'maximum', 'float32', layout.length)
        }
      })
  ).toThrow(/combined cannot be used together/);
  expect(
    () =>
      new GPURasterExtremaPyramid({
        grid: GRID,
        input: elevation,
        output: {combined: createTransientView(graph, 'short', 'float32', layout.length)}
      })
  ).toThrow(/must contain/);
});
