// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUFlowPairKey,
  getGPUFlowPairZones,
  GPUFlowAggregation,
  type GPUFlowAggregationProps
} from '../../../src/gpu-network/flow-aggregation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let defaultViewSerial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUFlowAggregationProps> = {}
): GPUFlowAggregationProps {
  const createDefault = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `default-${defaultViewSerial++}`, format, length);
  return {
    zones: {kind: 'ids', zoneCount: 4},
    originZoneIds: createDefault('uint32', 10),
    destinationZoneIds: createDefault('uint32', 10),
    pairCapacity: 8,
    output: {
      ids: createDefault('uint32', 4),
      count: createDefault('uint32', 1),
      overflow: createDefault('uint32', 1)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUFlowAggregationProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUFlowAggregation(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('GPUFlowAggregation pair keys round trip', () => {
  for (const [origin, destination, zoneCount] of [
    [0, 0, 1],
    [2, 3, 4],
    [65534, 65534, 65535],
    [12, 40000, 65535]
  ]) {
    const key = getGPUFlowPairKey(origin, destination, zoneCount);
    expect(key).toBeLessThan(0xffffffff);
    expect(getGPUFlowPairZones(key, zoneCount)).toEqual([origin, destination]);
  }
});

it('GPUFlowAggregation rejects invalid properties', () => {
  expectThrows(() => ({pairCapacity: 6}), /pairCapacity must be a positive power of two/);
  expectThrows(() => ({pairCapacity: 0}), /pairCapacity must be a positive power of two/);
  expectThrows(() => ({maxProbeCount: 9}), /maxProbeCount/);
  expectThrows(() => ({zones: {kind: 'ids', zoneCount: 65536}}), /zone count/);
  expectThrows(() => ({zones: {kind: 'ids', zoneCount: 0}}), /zone count/);
  expectThrows(
    () => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [300, 300]}
    }),
    /zone count/
  );
  expectThrows(() => ({originZoneIds: undefined}), /ids zones require/);
  expectThrows(
    graph => ({
      origins: createTransientView(graph, 'origins', 'float32x2', 10)
    }),
    /do not accept origins/
  );
  expectThrows(
    graph => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'origins', 'float32x2', 10)
    }),
    /require origins and destinations/
  );
  expectThrows(
    graph => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      origins: createTransientView(graph, 'origins', 'float32x2', 10),
      destinations: createTransientView(graph, 'destinations', 'float32x2', 10)
    }),
    /do not accept zone IDs/
  );
  expectThrows(
    graph => ({
      destinationZoneIds: createTransientView(graph, 'short', 'uint32', 9)
    }),
    /destinationZoneIds length/
  );
  expectThrows(
    graph => ({
      zones: {
        kind: 'hexagon',
        bounds: [0, 0, 1, 1],
        gridSize: [2, 2],
        radius: -1
      },
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'origins', 'float32x2', 10),
      destinations: createTransientView(graph, 'destinations', 'float32x2', 10)
    }),
    /radius/
  );
  expectThrows(
    graph => ({
      zoneOutWeights: createTransientView(graph, 'zone-out-weights', 'float32', 4)
    }),
    /zoneOutWeights requires weights/
  );
  expectThrows(
    graph => ({
      zoneOutCounts: createTransientView(graph, 'zone-out-counts', 'uint32', 3)
    }),
    /zoneOutCounts length/
  );
  expectThrows(
    graph => ({
      flowCounts: createTransientView(graph, 'flow-counts', 'uint32', 5)
    }),
    /flowCounts length/
  );
  expectThrows(
    graph => ({
      flowWeights: createTransientView(graph, 'flow-weights', 'float32', 3)
    }),
    /flowWeights length/
  );
  expectThrows(
    graph => ({
      weights: createTransientView(graph, 'weights', 'float32', 9)
    }),
    /weights length/
  );
  expectThrows(
    graph => ({
      output: {
        ids: createTransientView(graph, 'empty-ids', 'uint32', 0),
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }),
    /at least one row/
  );
});

it('GPUFlowAggregation rejects outputs that share buffers with inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const shared = createTransientView(graph, 'shared', 'uint32', 4);
  expect(
    () =>
      new GPUFlowAggregation(
        createProps(graph, {
          originZoneIds: createTransientView(graph, 'origin-ids', 'uint32', 4),
          destinationZoneIds: shared,
          output: {
            ids: shared,
            count: createTransientView(graph, 'count', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          }
        })
      )
  ).toThrow(/must not share buffers/);
  device.destroy();
});

it('GPUFlowAggregation rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const otherGraph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(createProps(graph));
  expect(() => contributor.getCommandNodes(otherGraph)).toThrow(/target graph/);
  device.destroy();
});

it('GPUFlowAggregation schedules ids zones in order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUFlowAggregation(
    createProps(graph, {
      sumOrder: 'atomic',
      weights: createTransientView(graph, 'weights', 'float32', 10),
      flowCounts: createTransientView(graph, 'flow-counts', 'uint32', 4),
      flowOriginZoneIds: createTransientView(graph, 'flow-origin', 'uint32', 4),
      zoneOutCounts: createTransientView(graph, 'zone-out-counts', 'uint32', 4),
      zoneInWeights: createTransientView(graph, 'zone-in-weights', 'float32', 4)
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.every(id => id.startsWith('flow-aggregation-'))).toBe(true);
  expect(new Set(ids).size).toBe(ids.length);
  const prefixes = [
    'flow-aggregation-pair-keys',
    'flow-aggregation-pair-index',
    'flow-aggregation-pair-query',
    'flow-aggregation-pair-counts',
    'flow-aggregation-pair-weights',
    'flow-aggregation-slot-keys',
    'flow-aggregation-sort-pairs',
    'flow-aggregation-weight-keys',
    'flow-aggregation-sort-weights',
    'flow-aggregation-gather-ids',
    'flow-aggregation-gather-values',
    'flow-aggregation-decode-zones',
    'flow-aggregation-pair-overflow',
    'flow-aggregation-publish',
    'flow-aggregation-zone-out-counts',
    'flow-aggregation-zone-in-weights'
  ];
  let cursor = -1;
  for (const prefix of prefixes) {
    const position = ids.findIndex((id, index) => index > cursor && id.startsWith(prefix));
    expect(position, prefix).toBeGreaterThan(cursor);
    cursor = position;
  }
  expect(ids.some(id => id.includes('time-classify'))).toBe(false);
  expect(ids.some(id => id.includes('-origin-zones'))).toBe(false);
  device.destroy();
});

it('GPUFlowAggregation schedules grid zone and time nodes first', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUFlowAggregation(
    createProps(graph, {
      id: 'flows',
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      originZoneIds: undefined,
      destinationZoneIds: undefined,
      origins: createTransientView(graph, 'origins', 'float32x2', 10),
      destinations: createTransientView(graph, 'destinations', 'float32x2', 10),
      timeWindow: {
        timestamps: createTransientView(graph, 'timestamps', 'float32', 10),
        window: createTransientView(graph, 'window', 'float32', 8)
      }
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.slice(0, 4)).toEqual([
    'flows-origin-zones',
    'flows-destination-zones',
    'flows-time-classify',
    'flows-pair-keys'
  ]);
  expect(ids.every(id => id.startsWith('flows-'))).toBe(true);
  device.destroy();
});

it('GPUFlowAggregation validates word time windows and sumOrder', () => {
  const timeWindow = (
    graph: GPUCommandGraph,
    overrides: Partial<GPUFlowAggregationProps['timeWindow'] & object>
  ): GPUFlowAggregationProps['timeWindow'] => ({
    timestamps: createTransientView(graph, 'word-timestamps', 'uint32x2', 10),
    window: createTransientView(graph, 'word-window', 'uint32', 8),
    ...overrides
  });
  expectThrows(() => ({sumOrder: 'fast' as 'sorted'}), /sumOrder must be atomic or sorted/);
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        window: createTransientView(graph, 'float-window', 'float32', 8)
      })
    }),
    /word window/
  );
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        window: createTransientView(graph, 'short-word-window', 'uint32', 7)
      })
    }),
    /word window must hold 8 uint32/
  );
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        timestampsLow: createTransientView(graph, 'low', 'float32', 10)
      })
    }),
    /timestampsLow requires float32 timestamps/
  );
  expectThrows(
    graph => ({
      timeWindow: timeWindow(graph, {
        timestamps: createTransientView(graph, 'short-word-timestamps', 'uint32x2', 9)
      })
    }),
    /timestamps length/
  );
  expectThrows(
    graph => ({
      timeWindow: {
        timestamps: createTransientView(graph, 'float-timestamps', 'float32', 10),
        window: createTransientView(graph, 'uint-window', 'uint32', 8)
      }
    }),
    /window/
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUFlowAggregation(
    createProps(graph, {
      timeWindow: timeWindow(graph, {}),
      sumOrder: 'sorted'
    })
  );
  expect(contributor.sumOrder).toBe('sorted');
  expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
  const defaultGraph = new GPUCommandGraph(device);
  const defaultContributor = new GPUFlowAggregation(createProps(defaultGraph, {}));
  expect(defaultContributor.sumOrder).toBe('sorted');
  device.destroy();
});

it('GPUFlowAggregation sorted sums add sort, scan, and gather nodes', () => {
  const countNodes = (sumOrder: 'atomic' | 'sorted') => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const createDefault = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${sumOrder}-${name}`, format, length);
    const contributor = new GPUFlowAggregation({
      sumOrder,
      zones: {kind: 'ids', zoneCount: 4},
      originZoneIds: createDefault('origins', 'uint32', 10),
      destinationZoneIds: createDefault('destinations', 'uint32', 10),
      weights: createDefault('weights', 'float32', 10),
      pairCapacity: 16,
      output: {
        ids: createDefault('ids', 'uint32', 4),
        count: createDefault('count', 'uint32', 1),
        overflow: createDefault('overflow', 'uint32', 1)
      },
      flowWeights: createDefault('flow-weights', 'float32', 4),
      zoneOutWeights: createDefault('zone-out-weights', 'float32', 4),
      zoneInWeights: createDefault('zone-in-weights', 'float32', 4)
    });
    const count = contributor.getCommandNodes(graph).length;
    device.destroy();
    return count;
  };
  const atomicCount = countNodes('atomic');
  const sortedCount = countNodes('sorted');
  expect(sortedCount).toBeGreaterThan(atomicCount);
  console.log(
    `flow-aggregation nodes (3 weight sums): atomic=${atomicCount} sorted=${sortedCount}`
  );
});
