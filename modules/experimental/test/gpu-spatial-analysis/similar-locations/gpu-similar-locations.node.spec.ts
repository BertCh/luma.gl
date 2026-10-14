// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  getGPUSimilarLocationsParameterLength,
  GPUSimilarLocations
} from '../../../src/gpu-spatial-analysis/similar-locations/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

// At and above 4096 rows `'rank'` sorts one column at a time.
const ROWS = 4096;
const RESULT_COUNT = 8;

function build(attributeCount: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `similar-${serial++}`, format, length);
  return new GPUSimilarLocations({
    id: 'similar',
    values: view('float32', ROWS * attributeCount),
    columnCount: attributeCount,
    referenceMask: view('uint32', ROWS),
    parameters: view('float32', getGPUSimilarLocationsParameterLength(attributeCount)),
    standardization: 'rank',
    resultCapacity: RESULT_COUNT,
    output: {
      ranks: view('uint32', ROWS),
      distances: view('float32', ROWS),
      topIds: view('uint32', RESULT_COUNT),
      status: {count: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
}

it('GPUSimilarLocations rank columns share one pipeline per kernel (rule 12) and parse as WGSL', () => {
  expect(getGPUCommandNodePipelineKeys(build(4)).size).toBe(
    getGPUCommandNodePipelineKeys(build(3)).size
  );
  for (const {id, source} of getGPUCommandNodeKernelSources(build(3))) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
});
