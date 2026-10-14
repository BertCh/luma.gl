// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULISAMarkov,
  GPUSpatialMarkov
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

// Rule 12: the period loop must not reach the shader text, overrides or binding offsets. Period
// transients use a 64-element-aligned stride, so every period view has the same offset residue
// and all periods share one pipeline per kernel. Distinct pipeline keys at periods 3 / 4 / 6,
// before the fix (period sub-views at `period * rows`):
//   GPULISAMarkov    rows 37 and 100: 22 / 27 / 37; rows 64 (already aligned): 12 / 12 / 12
//   GPUSpatialMarkov rows 37 and 100: 20 / 21 / 23; rows 64 (already aligned): 18 / 18 / 18
// With load-balanced lag and a separate weight reduction: LISA 13 and spatial 23 pipelines
// at every period count and row count. Splitting weight sums avoids doubling the binding size.

const ROW_COUNTS = [37, 64, 100] as const;

function createWeights(graph: GPUCommandGraph, rows: number) {
  return {
    offsets: createTransientView(graph, 'offsets', 'uint32', rows + 1),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', rows * 4),
    weights: createTransientView(graph, 'weights', 'float32', rows * 4)
  };
}

function countLisaPipelines(rows: number, periods: number): number {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `lisa-markov-${rows}-${periods}`});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `v-${Math.random()}`, format, length);
  const nodes = new GPULISAMarkov({
    values: view('float32', rows * periods),
    weights: createWeights(graph, rows),
    periodCount: periods,
    parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
    output: {
      counts: view('uint32', 25),
      probabilities: view('float32', 25),
      rowTotals: view('uint32', 5)
    }
  }).getCommandNodes(graph);
  const count = getGPUCommandNodePipelineKeys(nodes).size;
  device.destroy();
  return count;
}

function countSpatialPipelines(rows: number, periods: number): number {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `spatial-markov-${rows}-${periods}`});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `v-${Math.random()}`, format, length);
  const nodes = new GPUSpatialMarkov({
    values: view('float32', rows * periods),
    classes: view('uint32', rows * periods),
    classCount: 3,
    weights: createWeights(graph, rows),
    periodCount: periods,
    maximumLagClassCount: 2,
    lagParameters: view('float32', 16),
    output: {
      counts: view('uint32', 2 * 9),
      probabilities: view('float32', 2 * 9),
      rowTotals: view('uint32', 6),
      lagBreaks: view('float32', 3),
      lagClassCount: view('uint32', 1)
    }
  }).getCommandNodes(graph);
  const count = getGPUCommandNodePipelineKeys(nodes).size;
  device.destroy();
  return count;
}

for (const rows of ROW_COUNTS) {
  it(`GPULISAMarkov pipeline count is independent of the period count (rows ${rows})`, () => {
    const base = countLisaPipelines(rows, 3);
    expect(base).toBe(13);
    expect(countLisaPipelines(rows, 4)).toBe(base);
    expect(countLisaPipelines(rows, 6)).toBe(base);
  });

  it(`GPUSpatialMarkov pipeline count is independent of the period count (rows ${rows})`, () => {
    const base = countSpatialPipelines(rows, 3);
    // Separate lag and weight reductions share the scalar fixup and add one normalize kernel.
    expect(base).toBe(23);
    expect(countSpatialPipelines(rows, 4)).toBe(base);
    expect(countSpatialPipelines(rows, 6)).toBe(base);
  });
}
