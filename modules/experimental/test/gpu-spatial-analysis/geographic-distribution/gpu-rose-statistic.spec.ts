// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPURoseStatistic,
  GPU_ROSE_STATISTIC_SUMMARY
} from '../../../src/gpu-spatial-analysis/geographic-distribution';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('GPURoseStatistic matches directional and axial circular oracles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const starts = Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0]);
  const ends = Float32Array.from([1, 0, 0, 1, -1, 0, 0, -1]);
  for (const mode of ['directional', 'axial'] as const) {
    const graph = new GPUCommandGraph(device, {id: `rose-${mode}`});
    const startsBuffer = createInputBuffer(device, starts);
    const endsBuffer = createInputBuffer(device, ends);
    const summaryBuffer = createOutputBuffer(device, GPU_ROSE_STATISTIC_SUMMARY.stride);
    const binsBuffer = createOutputBuffer(device, 4);
    graph.add(
      new GPURoseStatistic({
        starts: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', 4),
        ends: importGraphBuffer(graph, 'ends', endsBuffer, 'float32x2', 4),
        circular: {mode, angleUnit: 'radians', origin: 'positive-x-counter-clockwise', binCount: 4},
        summary: importGraphBuffer(
          graph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_ROSE_STATISTIC_SUMMARY.stride
        ),
        bins: importGraphBuffer(graph, 'bins', binsBuffer, 'float32', 4)
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const summary = await readFloat32(summaryBuffer, GPU_ROSE_STATISTIC_SUMMARY.stride);
    const bins = await readFloat32(binsBuffer, 4);
    expect(summary[GPU_ROSE_STATISTIC_SUMMARY.count]).toBe(4);
    if (mode === 'directional') {
      expect(summary[GPU_ROSE_STATISTIC_SUMMARY.resultantLength]).toBeCloseTo(0, 5);
      expect(bins).toEqual([1, 1, 1, 1]);
    } else {
      expect(summary[GPU_ROSE_STATISTIC_SUMMARY.resultantLength]).toBeCloseTo(0, 5);
      expect(bins).toEqual([2, 0, 2, 0]);
    }
    compiled.destroy();
    startsBuffer.destroy();
    endsBuffer.destroy();
    summaryBuffer.destroy();
    binsBuffer.destroy();
  }
});
