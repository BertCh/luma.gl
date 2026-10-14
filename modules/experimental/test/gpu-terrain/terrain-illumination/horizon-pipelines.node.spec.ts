// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUTerrainHorizon} from '../../../src/gpu-terrain/terrain-illumination';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {createBand} from '../terrain-test-utils';

function getMarchPipelines(overrides: Record<string, unknown>) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const pixelCount = 30;
  const horizon = new GPUTerrainHorizon({
    grid: makeGPURasterMetadata({width: 6, height: 5}),
    maximumRadius: 4,
    directionCount: 8,
    elevation: createBand(graph, 'elevation', pixelCount),
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    output: {
      skyViewFactor: createTransientView(graph, 'svf', 'float32', pixelCount),
      negativeOpenness: createTransientView(graph, 'negative', 'float32', pixelCount),
      anisotropicSkyViewFactor: createTransientView(graph, 'anisotropic', 'float32', pixelCount)
    },
    ...overrides
  });
  const pipelines = getRasterPipelines(horizon.getCommandNodes(graph));
  for (const pipeline of pipelines) {
    // Every generated module parses (wgsl_reflect is laxer than Dawn; the GPU specs compile it).
    expect(() => new WgslReflect(pipeline.source), pipeline.id).not.toThrow();
  }
  return pipelines.filter(pipeline => /-horizon-(march-)?\d+$/.test(pipeline.id));
}

it('review E-06: fused march chunks read their sectors from a range word', () => {
  // A budget of 30 pixels x 4 steps x k sectors per dispatch splits 8 sectors into 8, 4, 2 or 1
  // chunks; every chunk kernel must have the same WGSL.
  for (const sectorsPerChunk of [1, 2, 3, 8]) {
    const marches = getMarchPipelines({maximumStepsPerDispatch: 30 * 4 * sectorsPerChunk});
    expect(marches.length).toBeGreaterThan(sectorsPerChunk === 1 ? 7 : 0);
    expect(new Set(marches.map(pipeline => pipeline.key)).size, String(sectorsPerChunk)).toBe(1);
    for (const pipeline of marches) {
      expect(pipeline.key).not.toMatch(/FIRST_SECTOR|END_SECTOR/);
    }
  }
});

it('review E-06: per-sector march kernels (odd unorm16) share one pipeline', () => {
  for (const directionCount of [5, 7, 9]) {
    const marches = getMarchPipelines({
      directionCount,
      horizonFormat: 'unorm16',
      horizon: undefined
    });
    expect(marches).toHaveLength(directionCount);
    expect(new Set(marches.map(pipeline => pipeline.key)).size).toBe(1);
    for (const pipeline of marches) {
      expect(pipeline.key).not.toMatch(/const SECTOR:/);
    }
  }
});

it('review E-06: sweep sectors read their line geometry from a sector record', () => {
  for (const cellSizeMode of ['uniform', 'geographic'] as const) {
    const counts = [6, 8, 12].map(directionCount => {
      const sweeps = getMarchPipelines({
        algorithm: 'sweep',
        directionCount,
        cellSizeMode
      });
      // One zenith and one nadir sweep per sector.
      expect(sweeps).toHaveLength(directionCount);
      return new Set(sweeps.map(pipeline => pipeline.key)).size;
    });
    expect(counts, cellSizeMode).toEqual([1, 1, 1]);
  }
});
