// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUDistanceField} from '../../../src/gpu-raster/distance-field';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../raster-pipeline-count';

it('singleton JFA without refinement publishes GPU initialization without an empty WGSL array', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUDistanceField({
      id: 'singleton',
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      mode: 'jump-flood',
      jumpFloodRefinementIterationCount: 0,
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      seedMask: createTransientView(graph, 'mask', 'uint32', 1),
      output: {distances: createTransientView(graph, 'distances', 'float32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.map(node => node.id)).toEqual([
      'singleton-seed-clear',
      'singleton-jump-flood-initialize',
      'singleton-finalize'
    ]);
    const pipelines = getRasterPipelines(nodes);
    expect(pipelines.length).toBeGreaterThan(0);
    for (const pipeline of pipelines) {
      expect(pipeline.source).not.toMatch(/array\s*<\s*u32\s*,\s*0\s*>/);
    }
  } finally {
    device.destroy();
  }
});
