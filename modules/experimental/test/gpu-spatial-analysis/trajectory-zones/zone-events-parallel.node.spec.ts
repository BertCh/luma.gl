// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';

let uniqueIndex = 0;
const unique = (id: string) => `${id}-${uniqueIndex++}`;

function getZoneEventsNodes(trackCount: number, withSpans: boolean) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: unique('graph')});
  const view = <Format extends 'float32' | 'uint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const cellCount = trackCount * 2;
  const compact = (capacity: number) => ({
    ids: view('uint32', capacity),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  });
  return new GPUZoneEvents({
    tracks: {
      kind: 'trajectories',
      positions: view('float32x2', 8 * trackCount),
      timestamps: view('float32', 8 * trackCount),
      trackOffsets: view('uint32', trackCount + 1)
    },
    zones: {
      kind: 'polygons',
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      ringOffsets: view('uint32', 3)
    },
    candidateCapacity: 64,
    trackEventCapacity: 4,
    output: {
      events: {...compact(16), eventTypes: view('uint32', 16), eventTimes: view('float32', 16)},
      dwellTimes: view('float32', cellCount),
      visitCounts: view('uint32', cellCount),
      trackEventCounts: view('uint32', trackCount),
      visitTable: withSpans
        ? {
            ...compact(cellCount),
            firstEnterTimes: view('float32', cellCount),
            lastExitTimes: view('float32', cellCount)
          }
        : undefined
    }
  }).getCommandNodes(graph);
}

it('GPUZoneEvents replaces the per-track walk with a cell sort and per-cell kernels', () => {
  const ids = getZoneEventsNodes(3, true).map(node => node.id);
  for (const removed of ['zone-events-seed', 'zone-events-walk', 'zone-events-close']) {
    expect(ids, removed).not.toContain(removed);
  }
  const order = [
    'zone-events-sorted-columns',
    'zone-events-event-ranks',
    'zone-events-track-counts',
    'zone-events-cell-keys',
    'zone-events-cell-sort',
    'zone-events-cell-offsets',
    'zone-events-cell-events',
    'zone-events-dwell',
    'zone-events-cell-finalize',
    'zone-events-cell-spans',
    'zone-events-keep-flags'
  ];
  let previous = -1;
  for (const prefix of order) {
    const index = ids.findIndex(id => id === prefix || id.startsWith(`${prefix}-`));
    expect(index, prefix).toBeGreaterThan(previous);
    previous = index;
  }
  // Without the visit table spans there is no span kernel.
  expect(getZoneEventsNodes(3, false).map(node => node.id)).not.toContain('zone-events-cell-spans');
});

it('GPUZoneEvents parallel kernels parse, stay within 8 storage bindings and add no size-varying variants', () => {
  const nodes = getZoneEventsNodes(3, true);
  for (const [kernelId, source] of getParsedKernelSources(nodes)) {
    expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
  }
  // Rule 12: no kernel family appears more than once per build, so a different track count gives
  // the same number of distinct pipelines.
  expect(getGPUCommandNodePipelineKeys(getZoneEventsNodes(4, true)).size).toBe(
    getGPUCommandNodePipelineKeys(nodes).size
  );
});

/**
 * JS model of the parallel formulas (cell-events, dwell contribution, cell finalize and spans)
 * against a direct replay of the sequential walk they replace, on random event streams.
 */
it('parallel cell formulas equal the sequential enter/exit walk', () => {
  const random = createRandom(7);
  for (let trial = 0; trial < 200; trial++) {
    const seed = random() < 0.4 ? 1 : 0;
    const count = Math.floor(random() * 9);
    const times: number[] = [];
    let time = random() * 3;
    for (let event = 0; event < count; event++) {
      time += random() < 0.2 ? 0 : random() * 5;
      times.push(time);
    }
    const duration = time + random() * 4;
    // Sequential walk (the removed kernels): seed, walk, dwell, span, close.
    let state = seed;
    let enterTime = 0;
    let dwell = 0;
    let visits = seed;
    let firstEnter = seed === 1 ? 0 : -1;
    let lastExit = 0;
    const types: number[] = [];
    for (const eventTime of times) {
      state ^= 1;
      types.push(state === 1 ? 0 : 1);
      if (state === 1) {
        enterTime = eventTime;
        visits++;
        if (firstEnter < 0) {
          firstEnter = eventTime;
        }
      } else {
        dwell += eventTime - enterTime;
        lastExit = eventTime;
      }
    }
    if (state === 1) {
      dwell += duration - enterTime;
      lastExit = duration;
    }
    // Parallel formulas.
    const parallelTypes = times.map((_, rank) => ((seed ^ ((rank + 1) & 1)) === 1 ? 0 : 1));
    let parallelDwell = 0;
    times.forEach((eventTime, rank) => {
      if ((seed ^ ((rank + 1) & 1)) === 0) {
        parallelDwell += eventTime - (rank > 0 ? times[rank - 1] : 0);
      }
    });
    const finalInside = seed ^ (count & 1);
    const enters = seed === 1 ? Math.floor(count / 2) : Math.floor((count + 1) / 2);
    if (finalInside === 1) {
      parallelDwell += duration - (count > 0 ? times[count - 1] : 0);
    }
    const parallelFirstEnter = seed === 1 ? 0 : count > 0 ? times[0] : -1;
    const parallelLastExit = finalInside === 1 ? duration : count > 0 ? times[count - 1] : 0;
    expect(parallelTypes).toEqual(types);
    expect(seed + enters).toBe(visits);
    expect(parallelDwell).toBeCloseTo(dwell, 9);
    expect(parallelFirstEnter).toBe(firstEnter);
    expect(parallelLastExit).toBe(lastExit);
  }
});
