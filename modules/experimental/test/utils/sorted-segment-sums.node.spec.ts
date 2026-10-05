// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {getSortedSegmentSumNodes} from '../../src/utils/sorted-segment-sums';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

it('getSortedSegmentSumNodes orders sort, scan, then gather and segment sum per reduction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const segmentKeys = createTransientView(graph, 'keys', 'uint32', 8);
  const segmentCounts = createTransientView(graph, 'counts', 'uint32', 3);
  const contributionsX = createTransientView(graph, 'contributions-x', 'float32', 8);
  const contributionsY = createTransientView(graph, 'contributions-y', 'float32', 8);
  const sumsX = createTransientView(graph, 'sums-x', 'float32', 3);
  const sumsY = createTransientView(graph, 'sums-y', 'float32', 3);

  const nodes = getSortedSegmentSumNodes(graph, {
    id: 'demo',
    operation: 'GPUDemo',
    segmentCount: 3,
    segmentKeys,
    segmentCounts,
    reductions: [
      {name: 'x', contributions: contributionsX, output: sumsX},
      {name: 'y', contributions: contributionsY, output: sumsY}
    ]
  });

  const ids = nodes.map(node => node.id);
  expect(ids.indexOf('demo-sort-prepare')).toBe(0);
  expect(ids.indexOf('demo-segment-total')).toBeGreaterThan(ids.indexOf('demo-sort-prepare'));
  const tail = ids.slice(ids.indexOf('demo-segment-total') + 1);
  expect(tail).toEqual(['demo-gather-x', 'demo-reduce-x', 'demo-gather-y', 'demo-reduce-y']);
  expect(new Set(ids).size).toBe(ids.length);
});
