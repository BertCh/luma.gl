// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTimeWindowFilter,
  type GPUTimeWindowFilterProps
} from '../../../src/gpu-dataframe/time-window-filter';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

function createProps(
  graph: GPUCommandGraph,
  rows: number = 4,
  overrides: Partial<GPUTimeWindowFilterProps> = {}
): GPUTimeWindowFilterProps {
  return {
    timestamps: createTransientView(graph, 'timestamps', 'float32', rows),
    window: createTransientView(graph, 'window', 'float32', 8),
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', rows),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    },
    ...overrides
  };
}

it('GPUTimeWindowFilter schedules classify, visibility, and publish nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUTimeWindowFilter(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids[0]).toBe('time-window-filter-classify');
  expect(ids.at(-1)).toBe('time-window-filter-publish');
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.every(id => id.startsWith('time-window-filter-'))).toBe(true);
  expect(ids.some(id => id.includes('-compose'))).toBe(false);

  const graphWithPredicate = new GPUCommandGraph(device);
  const predicateIds = new GPUTimeWindowFilter(
    createProps(graphWithPredicate, 4, {
      additionalPredicates: [
        {
          kind: 'selection',
          mask: createTransientView(graphWithPredicate, 'sel', 'uint32', 4)
        }
      ]
    })
  )
    .getCommandNodes(graphWithPredicate)
    .map(node => node.id);
  expect(predicateIds.some(id => id.startsWith('time-window-filter-visibility-compose'))).toBe(
    true
  );
  device.destroy();
});

it('GPUTimeWindowFilter wires track counts and vector chunks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const timestamps = createVectorView('timestamps', 'float32', [
    createTransientView(graph, 't0', 'float32', 3),
    createTransientView(graph, 't1', 'float32', 0),
    createTransientView(graph, 't2', 'float32', 2)
  ]);
  const ids = new GPUTimeWindowFilter({
    ...createProps(graph, 5),
    id: 'tracks',
    timestamps,
    trackIds: createTransientView(graph, 'track-ids', 'uint32', 5),
    trackVisibleCounts: createTransientView(graph, 'track-counts', 'uint32', 2)
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('tracks-classify-chunk-0');
  expect(ids).toContain('tracks-classify-chunk-2');
  expect(ids).not.toContain('tracks-classify-chunk-1');
  const trackIndex = ids.findIndex(id => id.startsWith('tracks-track-counts'));
  expect(trackIndex).toBeGreaterThan(0);
  expect(trackIndex).toBeLessThan(ids.indexOf('tracks-publish'));

  const emptyGraph = new GPUCommandGraph(device);
  const emptyIds = new GPUTimeWindowFilter(createProps(emptyGraph, 0))
    .getCommandNodes(emptyGraph)
    .map(node => node.id);
  expect(emptyIds.at(-1)).toBe('time-window-filter-publish');
  device.destroy();
});

it('GPUTimeWindowFilter validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUTimeWindowFilterProps>) =>
    new GPUTimeWindowFilter({...base, ...overrides});

  expect(() => create({window: createTransientView(graph, 'w4', 'float32', 4)})).toThrow(/window/);
  expect(() => create({timestampsLow: createTransientView(graph, 'low', 'float32', 3)})).toThrow(
    /topology/
  );
  expect(() =>
    create({
      clipFractions: createTransientView(graph, 'clip', 'float32x2', 4)
    })
  ).toThrow(/requires endTimestamps/);
  expect(() =>
    create({
      endTimestampsLow: createTransientView(graph, 'end-low', 'float32', 4)
    })
  ).toThrow(/requires endTimestamps/);
  expect(() =>
    create({
      trackVisibleCounts: createTransientView(graph, 'counts', 'uint32', 2)
    })
  ).toThrow(/together/);
  expect(() => create({trackIds: createTransientView(graph, 'tracks', 'uint32', 4)})).toThrow(
    /together/
  );
  expect(() => create({sourceIds: createTransientView(graph, 'source', 'uint32', 3)})).toThrow(
    /sourceIds/
  );

  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 64,
    usage: 128
  });
  expect(() =>
    create({
      timestamps: graph.createDataView(shared, {
        format: 'float32',
        length: 4
      }),
      output: {
        ...base.output,
        ids: graph.createDataView(shared, {format: 'uint32', length: 4})
      }
    })
  ).toThrow(/share buffers/);

  const strided = graph.createTransientBuffer({
    id: 'strided',
    byteLength: 64,
    usage: 128
  });
  expect(() =>
    create({
      drawInstanceCount: graph.createDataView(strided, {
        format: 'uint32',
        length: 1,
        byteStride: 16,
        byteOffset: 4,
        rowByteLength: 4
      })
    })
  ).toThrow(/drawInstanceCount/);

  const otherGraph = new GPUCommandGraph(device);
  const foreign = new GPUTimeWindowFilter(createProps(otherGraph));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/belong to the target graph/);
  device.destroy();
});
