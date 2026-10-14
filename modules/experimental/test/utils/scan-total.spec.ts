// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GPUScan} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {createSaturatedScanTotalNodes, SATURATED_SCAN_TOTAL} from '../../src/utils/scan-total';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

/** True sum of the counts, saturated like the GPU total. */
function getExpectedTotal(counts: readonly number[]): number {
  const sum = counts.reduce((total, count) => total + count, 0);
  return sum >= 2 ** 32 ? SATURATED_SCAN_TOTAL : sum;
}

// S-8: GPUScan wraps modulo 2^32, so a count total of 2^32 or more used to read back as a small
// number and pass every capacity check. The saturated total must equal the true sum below 2^32 and
// 0xffffffff from 2^32 on, also when the wrap happens deep inside a multi-block scan.
it('saturates exclusive count-scan totals that reach 2^32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const many = (length: number, count: number) => Array.from({length}, () => count);
  const cases: readonly (readonly number[])[] = [
    [1, 2, 3],
    [0xfffffffe, 1],
    [0xffffffff, 1],
    [0x80000000, 0x80000000, 7],
    [5, 0xffffffff, 0, 3],
    many(5000, 850_000),
    many(5000, 900_000),
    [...many(4095, 0), 0xffffffff, ...many(4096, 1)]
  ];
  for (const counts of cases) {
    const graph = new GPUCommandGraph(device, {id: 'scan-total'});
    const countsBuffer = createInputBuffer(device, Uint32Array.from(counts));
    const offsetsBuffer = createOutputBuffer(device, counts.length);
    const totalBuffer = createOutputBuffer(device, 2);
    const countsView = importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', counts.length);
    const offsetsView = importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', counts.length);
    const totalView = importGraphBuffer(graph, 'total', totalBuffer, 'uint32', 2);
    for (const node of new GPUScan({
      id: 'scan',
      input: countsView,
      output: offsetsView,
      mode: 'exclusive'
    }).getCommandNodes(graph)) {
      graph.add(node);
    }
    for (const node of createSaturatedScanTotalNodes(graph, {
      id: 'scan-total',
      operation: 'ScanTotalTest',
      counts: countsView,
      offsets: offsetsView,
      rowCount: counts.length,
      total: totalView,
      totalRow: 1
    })) {
      graph.add(node);
    }
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const [, total] = await readUint32(totalBuffer, 2);
    compiled.destroy();
    for (const buffer of [countsBuffer, offsetsBuffer, totalBuffer]) buffer.destroy();
    expect(total, `counts of length ${counts.length}`).toBe(getExpectedTotal(counts));
  }
});
