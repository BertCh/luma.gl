// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_SPLAT_STAGE_PREFIXES,
  getGPUSplatStage,
  getGPUSplatStageTimings
} from '../src/splat-stage-timings';

/** Builds a timing report shaped like one the command graph produces. */
function makeReport(
  nodes: {id: string; cpu: number; gpu?: number}[]
): Parameters<typeof getGPUSplatStageTimings>[0] {
  return {
    cpuEncodeTimeMilliseconds: nodes.reduce((total, node) => total + node.cpu, 0),
    ...(nodes.some(node => node.gpu !== undefined)
      ? {
          gpuTimeMilliseconds: nodes.reduce((total, node) => total + (node.gpu ?? 0), 0)
        }
      : {}),
    nodes: nodes.map(node => ({
      id: node.id,
      type: 'compute' as const,
      cpuEncodeTimeMilliseconds: node.cpu,
      hasGPUTimestamps: node.gpu !== undefined,
      ...(node.gpu === undefined ? {} : {gpuTimeMilliseconds: node.gpu})
    }))
  };
}

it('every node the splat graph emits is attributed to a stage', () => {
  const expected: [string, string][] = [
    ['gaussian-splat-initialize', 'initialize'],
    ['gaussian-splat-project-batch-0', 'projection'],
    ['gaussian-splat-project-batch-17', 'projection'],
    ['gaussian-splat-features-batch-3', 'features'],
    ['gaussian-splat-global-depth-sort-radix-digit-0-histogram', 'sort'],
    ['gaussian-splat-global-depth-sort-radix-digit-4-scatter', 'sort'],
    ['gaussian-splat-gather-sorted-records', 'gather'],
    ['gaussian-splat-indirect-render', 'raster']
  ];
  for (const [nodeId, stage] of expected) {
    expect(getGPUSplatStage(nodeId), nodeId).toBe(stage);
  }
  expect(getGPUSplatStage('something-else'), 'a foreign node belongs to no stage').toBe(undefined);
});

it('stage prefixes are disjoint, so the gather pass is never mistaken for part of the sort', () => {
  // Classification takes the first matching prefix. That is only order-independent if no prefix is
  // a prefix of another; otherwise a reordering could silently move a node into the wrong budget.
  for (const [prefix] of GPU_SPLAT_STAGE_PREFIXES) {
    const otherPrefixes = GPU_SPLAT_STAGE_PREFIXES.filter(([other]) => other !== prefix);
    for (const [other] of otherPrefixes) {
      expect(prefix.startsWith(other), `${prefix} does not start with ${other}`).toBe(false);
    }
  }
  expect(
    getGPUSplatStage('gaussian-splat-gather-sorted-records'),
    'gather is its own stage, although its id mentions sorting'
  ).toBe('gather');
  expect(
    getGPUSplatStage('gaussian-splat-sorted-records'),
    'a sort-adjacent resource id is not attributed to the sort'
  ).toBe(undefined);
});

it('stage timings sum the nodes that belong to each stage', () => {
  const timings = getGPUSplatStageTimings(
    makeReport([
      {id: 'gaussian-splat-initialize', cpu: 0.1, gpu: 0.02},
      {id: 'gaussian-splat-project-batch-0', cpu: 0.4, gpu: 1.5},
      {id: 'gaussian-splat-project-batch-1', cpu: 0.3, gpu: 1.1},
      {id: 'gaussian-splat-global-depth-sort-radix-digit-0-histogram', cpu: 0.2, gpu: 0.3},
      {id: 'gaussian-splat-global-depth-sort-radix-digit-0-scatter', cpu: 0.2, gpu: 0.5},
      {id: 'gaussian-splat-indirect-render', cpu: 0.05, gpu: 4.2}
    ])
  );

  expect(timings.hasGPUTimings, 'timestamps were available').toBe(true);
  expect(timings.stages.projection?.nodeCount, 'both projection batches are counted').toBe(2);
  expect(timings.stages.projection?.gpuTimeMilliseconds, 'and their GPU time summed').toBeCloseTo(
    2.6,
    6
  );
  expect(timings.stages.sort?.gpuTimeMilliseconds, 'the sort covers all of its passes').toBeCloseTo(
    0.8,
    6
  );
  expect(timings.stages.raster?.gpuTimeMilliseconds, 'and the draw stands alone').toBeCloseTo(
    4.2,
    6
  );
  expect(timings.stages.features, 'a stage with no nodes is absent rather than zero').toBe(
    undefined
  );

  // This is the whole point of splitting them: on this frame the answer is "raster", and no
  // amount of work on the sort would help.
  const slowest = Object.entries(timings.stages).sort(
    ([, left], [, right]) => (right.gpuTimeMilliseconds ?? 0) - (left.gpuTimeMilliseconds ?? 0)
  )[0];
  expect(slowest[0], 'the dominant stage is identifiable').toBe('raster');
});

it('without timestamp queries the stages still report their encoding cost', () => {
  const timings = getGPUSplatStageTimings(
    makeReport([
      {id: 'gaussian-splat-project-batch-0', cpu: 0.4},
      {id: 'gaussian-splat-indirect-render', cpu: 0.05}
    ])
  );

  expect(timings.hasGPUTimings, 'no device support means no GPU durations').toBe(false);
  expect(
    timings.stages.projection?.cpuEncodeTimeMilliseconds,
    'but CPU encoding cost is always available'
  ).toBeCloseTo(0.4, 6);
  expect(
    timings.stages.projection?.gpuTimeMilliseconds,
    'and a missing GPU duration is absent rather than zero, which would read as free'
  ).toBe(undefined);
});
