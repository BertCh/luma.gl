// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  createLoadBalancedForEachNode,
  createLoadBalancedSegmentSumNodes,
  getLoadBalancedTileCount,
  LOAD_BALANCED_TILE_SIZE,
  type LoadBalancedAccumulatorType
} from '../../src/utils/load-balanced-for-each';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from './gpu-contributor-test-utils';

const CONTRIBUTION_WGSL = (
  type: string
) => `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> ${type} {
  return ${type}(itemValues[itemValuesOffset + item]) + ${type}(f32(lane) * 0.0);
}`;

function buildSegmentSum(
  rowCount: number,
  itemCapacity: number,
  laneCount = 1,
  accumulatorType: LoadBalancedAccumulatorType = 'f32',
  extraBindingCount = 0
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'load-balanced'});
  const offsets = createTransientView(graph, 'offsets', 'uint32', rowCount + 1);
  const itemValues = createTransientView(
    graph,
    'item-values',
    'float32',
    Math.max(itemCapacity, 1)
  );
  const components = accumulatorType.startsWith('vec4')
    ? 4
    : accumulatorType.startsWith('vec2')
      ? 2
      : 1;
  const isFloat = accumulatorType.includes('f32');
  const output = createTransientView(
    graph,
    'output',
    isFloat ? 'float32' : 'uint32',
    rowCount * laneCount * components
  );
  const extras = Array.from({length: extraBindingCount}, (_, index) => ({
    name: `extra${index}`,
    view: createTransientView(graph, `extra-${index}`, 'uint32', 4),
    type: 'u32' as const,
    access: 'read' as const
  }));
  const nodes = createLoadBalancedSegmentSumNodes(graph, {
    id: 'sum',
    operation: 'LoadBalancedSpec',
    offsets,
    itemCapacity,
    laneCount,
    accumulatorType,
    output,
    bindings: [{name: 'itemValues', view: itemValues, type: 'f32', access: 'read'}, ...extras],
    declarations: isFloat
      ? CONTRIBUTION_WGSL(accumulatorType)
      : `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> ${accumulatorType} {
  return ${accumulatorType}(u32(itemValues[itemValuesOffset + item]) + lane);
}`
  });
  return {graph, nodes};
}

it('createLoadBalancedSegmentSumNodes builds a walk and a fixup that parse within core limits', () => {
  for (const accumulatorType of [
    'f32',
    'vec2<f32>',
    'vec4<f32>',
    'u32',
    'vec2<u32>',
    'vec4<u32>'
  ] as const) {
    const {nodes} = buildSegmentSum(1000, 100_000, 3, accumulatorType, 4);
    expect(nodes.map(node => node.id)).toEqual(['sum', 'sum-fixup']);
    const sources = getParsedKernelSources(nodes);
    expect([...sources.keys()]).toEqual(['sum', 'sum-fixup']);
    for (const [id, source] of sources) {
      // Utility bindings: offsets, output and carries (walk), carries and output (fixup).
      expect(getStorageBindingCount(source), `${id} bindings`).toBeLessThanOrEqual(8);
      expect(getWorkgroupMemoryBytes(source), `${id} workgroup`).toBeLessThanOrEqual(
        CORE_WORKGROUP_STORAGE_LIMIT_BYTES
      );
      expect(getPrivateMemoryBytes(source), `${id} private`).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
    expect(getStorageBindingCount(sources.get('sum')!)).toBe(8);
  }
});

it('createLoadBalancedSegmentSumNodes dispatches one workgroup per tile of rows plus items', () => {
  const rowCount = 10_001;
  const itemCapacity = 100_000;
  const {nodes} = buildSegmentSum(rowCount, itemCapacity, 2);
  const tileCount = getLoadBalancedTileCount(rowCount, itemCapacity);
  expect(tileCount).toBe(Math.ceil((rowCount + itemCapacity) / LOAD_BALANCED_TILE_SIZE));
  const [walk, fixup] = nodes as {workload: {maximumInvocationCount: number}}[];
  expect(walk.workload.maximumInvocationCount).toBe(2 * tileCount * 256);
  expect(fixup.workload.maximumInvocationCount).toBe(2 * tileCount);
  const [walkKernel, fixupKernel] = getGPUCommandNodeKernelSources(nodes);
  expect(walkKernel.constants).toMatchObject({
    LOAD_BALANCED_ROW_COUNT: rowCount,
    LOAD_BALANCED_ITEM_CAPACITY: itemCapacity,
    LOAD_BALANCED_LANE_COUNT: 2,
    LOAD_BALANCED_TILES_PER_LANE: tileCount
  });
  expect(fixupKernel.constants).toMatchObject({LOAD_BALANCED_TILES_PER_LANE: tileCount});
});

it('load-balanced kernels share one module across sizes (rule 12)', () => {
  // Sizes are override constants; the WGSL text is the same for every row and item count, so a
  // loop that builds one sum per round compiles one module per kernel kind.
  const sources = new Set<string>();
  for (const [rowCount, itemCapacity] of [
    [10, 100],
    [11, 101],
    [5000, 1_000_000]
  ]) {
    const {nodes} = buildSegmentSum(rowCount, itemCapacity);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      sources.add(kernel.source);
    }
  }
  expect(sources.size).toBe(2);
  // Same rounds count twice: N and N + 1 copies of one shape add no pipelines.
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rounds'});
  const buildRounds = (roundCount: number) => {
    const nodes = [];
    for (let round = 0; round < roundCount; round++) {
      nodes.push(
        createLoadBalancedForEachNode(graph, {
          id: `for-each-${roundCount}-${round}`,
          operation: 'LoadBalancedSpec',
          offsets: createTransientView(graph, `offsets-${roundCount}-${round}`, 'uint32', 65),
          itemCapacity: 4096,
          bindings: [
            {
              name: 'visits',
              view: createTransientView(graph, `visits-${roundCount}-${round}`, 'uint32', 4096),
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          declarations: `fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  atomicAdd(&visits[visitsOffset + item], row + lane + 1u);
}`
        })
      );
    }
    return getGPUCommandNodePipelineKeys(nodes).size;
  };
  const threeRounds = buildRounds(3);
  expect(threeRounds).toBe(1);
  expect(buildRounds(4)).toBe(threeRounds);
});

it('createLoadBalancedForEachNode parses and binds the offsets first', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'for-each'});
  const node = createLoadBalancedForEachNode(graph, {
    id: 'visit',
    operation: 'LoadBalancedSpec',
    offsets: createTransientView(graph, 'offsets', 'uint32', 101),
    itemCapacity: 1000,
    bindings: [
      {
        name: 'rowOfItem',
        view: createTransientView(graph, 'row-of-item', 'uint32', 1000),
        type: 'u32',
        access: 'read_write'
      }
    ],
    declarations: `fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  rowOfItem[rowOfItemOffset + item] = row + lane;
}`
  });
  const source = getParsedKernelSources([node]).get('visit')!;
  expect(source).toMatch(/@binding\(0\) var<storage, read> loadBalancedOffsets/);
  expect(getWorkgroupMemoryBytes(source)).toBe((LOAD_BALANCED_TILE_SIZE + 1) * 4 + 8);
});

it('load-balanced loops reject topologies past uint32 (rule 16)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'limits'});
  const offsets = createTransientView(graph, 'offsets', 'uint32', 11);
  const props = {
    id: 'too-large',
    operation: 'LoadBalancedSpec',
    offsets,
    bindings: [],
    declarations: 'fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {}'
  };
  expect(() =>
    createLoadBalancedForEachNode(graph, {...props, itemCapacity: 2 ** 31, laneCount: 2})
  ).toThrow(/invalid load-balanced topology/);
  expect(() => createLoadBalancedForEachNode(graph, {...props, itemCapacity: -1})).toThrow(
    /invalid load-balanced topology/
  );
});
