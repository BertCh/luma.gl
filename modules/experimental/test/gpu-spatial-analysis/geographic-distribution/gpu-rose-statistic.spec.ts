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
import {requireTestDevice} from '../../utils/require-test-device';

it('GPURoseStatistic matches directional and axial circular oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
        output: {
          summary: importGraphBuffer(
            graph,
            'summary',
            summaryBuffer,
            'float32',
            GPU_ROSE_STATISTIC_SUMMARY.stride
          ),
          bins: importGraphBuffer(graph, 'bins', binsBuffer, 'float32', 4)
        }
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

it('GPURoseStatistic sorts rows into group segments for multiple groups', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Interleaved groups, one masked row and one out-of-range group ID (both excluded).
  const starts = new Float32Array(16);
  const ends = Float32Array.from([1, 0, 0, 1, 0, 1, -1, 0, 1, 0, 0, -1, 1, 0, 0, 1]);
  const groupIds = Uint32Array.from([0, 1, 0, 2, 1, 2, 9, 0]);
  const mask = Uint32Array.from([1, 1, 1, 1, 0, 1, 1, 1]);
  const groupCount = 3;
  const binCount = 4;
  const graph = new GPUCommandGraph(device, {id: 'rose-segmented'});
  const inputs = {
    starts: createInputBuffer(device, starts),
    ends: createInputBuffer(device, ends),
    groupIds: createInputBuffer(device, groupIds),
    mask: createInputBuffer(device, mask)
  };
  const summaryLength = groupCount * GPU_ROSE_STATISTIC_SUMMARY.stride;
  const summaryBuffer = createOutputBuffer(device, summaryLength);
  const binsBuffer = createOutputBuffer(device, groupCount * binCount);
  graph.add(
    new GPURoseStatistic({
      starts: importGraphBuffer(graph, 'starts', inputs.starts, 'float32x2', 8),
      ends: importGraphBuffer(graph, 'ends', inputs.ends, 'float32x2', 8),
      groupIds: importGraphBuffer(graph, 'groupIds', inputs.groupIds, 'uint32', 8),
      mask: importGraphBuffer(graph, 'mask', inputs.mask, 'uint32', 8),
      groupCount,
      circular: {
        mode: 'directional',
        angleUnit: 'radians',
        origin: 'positive-x-counter-clockwise',
        binCount
      },
      output: {
        summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'float32', summaryLength),
        bins: importGraphBuffer(graph, 'bins', binsBuffer, 'float32', groupCount * binCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const summary = await readFloat32(summaryBuffer, summaryLength);
  const bins = await readFloat32(binsBuffer, groupCount * binCount);
  const stride = GPU_ROSE_STATISTIC_SUMMARY.stride;
  // Group 0: rows 0 (east), 2 (north), 7 (north). Group 1: row 1 (north); row 4 is masked.
  // Group 2: rows 3 (west) and 5 (south).
  expect([0, 1, 2].map(group => summary[group * stride])).toEqual([3, 1, 2]);
  expect(bins).toEqual([1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]);
  for (const [group, expected] of [5 / 3, 1, 1].entries()) {
    expect(summary[group * stride + GPU_ROSE_STATISTIC_SUMMARY.rayleighZ]).toBeCloseTo(expected, 5);
  }
  expect(summary[stride + GPU_ROSE_STATISTIC_SUMMARY.meanAngle]).toBeCloseTo(Math.PI / 2, 5);
  expect(summary[stride + GPU_ROSE_STATISTIC_SUMMARY.resultantLength]).toBeCloseTo(1, 5);
  expect(summary[GPU_ROSE_STATISTIC_SUMMARY.resultantLength]).toBeCloseTo(Math.sqrt(5) / 3, 5);
  compiled.destroy();
  Object.values(inputs).forEach(buffer => buffer.destroy());
  summaryBuffer.destroy();
  binsBuffer.destroy();
});

it('GPURoseStatistic keeps Rayleigh z independent of descriptive weight units and excludes invalid rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['directional', 'axial'] as const) {
    for (const unitScale of [1, 1000]) {
      const graph = new GPUCommandGraph(device);
      // The first four directions are east, east, north, west. Remaining rows are a
      // masked north, zero-length line, zero-weight north and non-finite-weight north.
      const starts = createInputBuffer(device, new Float32Array(16));
      const ends = createInputBuffer(
        device,
        Float32Array.from([1, 0, 1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
      );
      const weights = createInputBuffer(
        device,
        Float32Array.from([1, 2, 4, 8, 1, 1, 0, NaN].map(weight => weight * unitScale))
      );
      const mask = createInputBuffer(device, Uint32Array.from([1, 1, 1, 1, 0, 1, 1, 1]));
      const summaryBuffer = createOutputBuffer(device, 5);
      const binsBuffer = createOutputBuffer(device, 4);
      graph.add(
        new GPURoseStatistic({
          starts: importGraphBuffer(graph, 'starts', starts, 'float32x2', 8),
          ends: importGraphBuffer(graph, 'ends', ends, 'float32x2', 8),
          weights: importGraphBuffer(graph, 'weights', weights, 'float32', 8),
          mask: importGraphBuffer(graph, 'mask', mask, 'uint32', 8),
          circular: {
            mode,
            angleUnit: 'radians',
            origin: 'positive-x-counter-clockwise',
            binCount: 4
          },
          output: {
            summary: importGraphBuffer(graph, 'summary', summaryBuffer, 'float32', 5),
            bins: importGraphBuffer(graph, 'bins', binsBuffer, 'float32', 4)
          }
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      const summary = await readFloat32(summaryBuffer, 5);
      const bins = await readFloat32(binsBuffer, 4);
      expect(summary[GPU_ROSE_STATISTIC_SUMMARY.count]).toBe(4);
      expect(summary[GPU_ROSE_STATISTIC_SUMMARY.rayleighZ]).toBeCloseTo(
        mode === 'axial' ? 1 : 0.5,
        5
      );
      expect(summary[GPU_ROSE_STATISTIC_SUMMARY.resultantLength]).toBeCloseTo(
        mode === 'axial' ? 7 / 15 : Math.sqrt(41) / 15,
        5
      );
      expect(bins).toEqual(
        (mode === 'axial' ? [11, 0, 4, 0] : [3, 4, 8, 0]).map(weight => weight * unitScale)
      );
      mask.write(new Uint32Array(8));
      submitGraph(device, compiled, undefined);
      const emptySummary = await readFloat32(summaryBuffer, 5);
      expect(emptySummary[GPU_ROSE_STATISTIC_SUMMARY.count]).toBe(0);
      expect(emptySummary[GPU_ROSE_STATISTIC_SUMMARY.rayleighZ]).toBeNaN();
      expect(await readFloat32(binsBuffer, 4)).toEqual([0, 0, 0, 0]);
      compiled.destroy();
      for (const buffer of [starts, ends, weights, mask, summaryBuffer, binsBuffer])
        buffer.destroy();
    }
  }
});
