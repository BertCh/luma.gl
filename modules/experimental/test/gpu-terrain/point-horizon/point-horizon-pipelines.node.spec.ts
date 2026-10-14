// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE,
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  getGPUPointHorizonVisibilityParameterValues
} from '../../../src/gpu-terrain/point-horizon';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {createBand} from '../terrain-test-utils';

function getChunkPipelines(kind: 'profile' | 'visibility', itemsPerDispatch: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const common = {
    grid: makeGPURasterMetadata({width: 64, height: 48, cellSize: 10}),
    elevation: createBand(graph, 'terrain', 64 * 48),
    maximumDistance: 1000,
    cellSize: 10
  };
  const nodes =
    kind === 'profile'
      ? new GPUPointHorizonProfile({
          ...common,
          observers: createTransientView(graph, 'observers', 'float32x4', 3),
          parameters: createTransientView(graph, 'settings', 'float32', 8),
          output: {tangent: createTransientView(graph, 'tangent', 'float32', 3 * 720)},
          raysPerDispatch: itemsPerDispatch
        }).getCommandNodes(graph)
      : new GPUPointHorizonVisibility({
          ...common,
          observers: createTransientView(graph, 'observers', 'float32x4', 2),
          targets: createTransientView(graph, 'targets', 'float32x4', 50),
          parameters: createTransientView(graph, 'settings', 'float32', 12),
          output: {
            visibility: createTransientView(graph, 'visibility', 'uint32', 50),
            details: createTransientView(graph, 'details', 'float32x4', 50)
          },
          targetsPerDispatch: itemsPerDispatch
        }).getCommandNodes(graph);
  const pipelines = getRasterPipelines(nodes);
  for (const pipeline of pipelines) {
    expect(() => new WgslReflect(pipeline.source), pipeline.id).not.toThrow();
  }
  return pipelines.filter(pipeline => /-(march|classify)-\d+$/.test(pipeline.id));
}

it('rule 12: point-horizon chunks read their first item from a chunk word', () => {
  for (const [kind, sizes] of [
    ['profile', [2160, 1000, 700, 333]],
    ['visibility', [50, 20, 7, 3]]
  ] as const) {
    for (const size of sizes) {
      const chunks = getChunkPipelines(kind, size);
      const itemCount = kind === 'profile' ? 2160 : 50;
      expect(chunks).toHaveLength(Math.ceil(itemCount / size));
      // Every chunk, the short last one included, has the same source and constants.
      expect(new Set(chunks.map(pipeline => pipeline.key)).size, `${kind} ${size}`).toBe(1);
      for (const pipeline of chunks) {
        expect(pipeline.source).not.toMatch(/RAY_OFFSET|TARGET_OFFSET|elevationValidity/);
      }
    }
  }
});

it('review C-05: the target ignore distance defaults to one ground pixel, 150 m is a preset', () => {
  expect(GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE).toBe(150);
  const grid = makeGPURasterMetadata({width: 64, height: 48, cellSize: 10});
  expect(getGPUPointHorizonVisibilityParameterValues({}, grid)[8]).toBe(-1);
  expect(
    getGPUPointHorizonVisibilityParameterValues(
      {targetIgnoreDistance: GPU_POINT_HORIZON_MT_IMAGE_TARGET_IGNORE_DISTANCE},
      grid
    )[8]
  ).toBe(150);
  expect(() =>
    getGPUPointHorizonVisibilityParameterValues({targetIgnoreDistance: -1}, grid)
  ).toThrow(/targetIgnoreDistance/);
  const [classify] = getChunkPipelines('visibility', 50);
  expect(classify.source).toMatch(
    /select\(packedIgnoreDistance, targetPixelLength, packedIgnoreDistance < 0\.0\)/
  );
  expect(classify.source).toMatch(/let targetPixelLength = 1\.0 \/ length\(/);
});
