// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkCostMatrix,
  type GPUNetworkCostMatrixProps
} from '../../../src/gpu-network/network-accessibility';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const NODE_COUNT = 8;
const EDGE_COUNT = 12;

let viewCount = 0;

function createView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  rowCount: number,
  overrides: Partial<GPUNetworkCostMatrixProps> = {},
  columnCount: number = NODE_COUNT
): GPUNetworkCostMatrixProps {
  return {
    id: 'matrix',
    topology: {
      offsets: createView(graph, 'uint32', NODE_COUNT + 1),
      neighbors: createView(graph, 'uint32', EDGE_COUNT),
      weights: createView(graph, 'float32', EDGE_COUNT)
    },
    startNodes: createView(graph, 'uint32', rowCount),
    output: {
      costs: createView(graph, 'float32', rowCount * columnCount),
      status: {converged: createView(graph, 'uint32', 1)}
    },
    laneCount: 2,
    maximumIterations: 3,
    ...overrides
  };
}

function getNodeIds(rowCount: number, targetCount?: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostMatrix(
    createProps(
      graph,
      rowCount,
      targetCount === undefined ? {} : {targetNodes: createView(graph, 'uint32', targetCount)},
      targetCount ?? NODE_COUNT
    )
  );
  return {contributor, nodes: contributor.getCommandNodes(graph)};
}

/** Node IDs before `targetNodes` existed; the default path must keep them. */
const SNAPSHOT_IDS = [
  'matrix-expand-seeds',
  'matrix-batch-info',
  'matrix-batch-0-initialize',
  'matrix-batch-0-seed',
  'matrix-batch-0-relax-0',
  'matrix-batch-0-relax-1',
  'matrix-batch-0-relax-2',
  'matrix-batch-0-finalize',
  'matrix-batch-0-copy',
  'matrix-batch-1-initialize',
  'matrix-batch-1-seed',
  'matrix-batch-1-relax-0',
  'matrix-batch-1-relax-1',
  'matrix-batch-1-relax-2',
  'matrix-batch-1-finalize',
  'matrix-batch-1-copy',
  'matrix-batch-2-initialize',
  'matrix-batch-2-seed',
  'matrix-batch-2-relax-0',
  'matrix-batch-2-relax-1',
  'matrix-batch-2-relax-2',
  'matrix-batch-2-finalize',
  'matrix-batch-2-copy',
  'matrix-converged'
];

it('GPUNetworkCostMatrix without targetNodes keeps its node list', () => {
  const {contributor, nodes} = getNodeIds(5);
  expect(contributor.columnCount).toBe(NODE_COUNT);
  expect(nodes.map(node => node.id)).toEqual(SNAPSHOT_IDS);
});

it('GPUNetworkCostMatrix validates targetNodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const withTargets = (
    targets: GPUNetworkCostMatrixProps['targetNodes'],
    costsLength: number,
    costsFormat: 'float32' | 'uint32' = 'float32'
  ) => {
    const props = createProps(graph, 4, {targetNodes: targets});
    return new GPUNetworkCostMatrix({
      ...props,
      output: {...props.output, costs: createView(graph, costsFormat, costsLength) as never}
    });
  };
  expect(withTargets(createView(graph, 'uint32', 3), 12).rowCount).toBe(4);
  expect(withTargets(createView(graph, 'uint32', 3), 12).columnCount).toBe(3);
  expect(() => withTargets(createView(graph, 'uint32', 3), 13)).toThrow(/multiple of the target/);
  expect(() => withTargets(createView(graph, 'uint32', 3), 0)).toThrow(/multiple of the target/);
  expect(() => withTargets(createView(graph, 'uint32', 0), 12)).toThrow(/targetNodes/);
  expect(() => withTargets(createView(graph, 'float32', 3) as never, 12)).toThrow(/targetNodes/);
  const shared = createView(graph, 'uint32', 1);
  expect(
    () =>
      new GPUNetworkCostMatrix({
        ...createProps(graph, 4),
        targetNodes: shared,
        output: {costs: createView(graph, 'float32', 4), status: {converged: shared}}
      })
  ).toThrow(/outputs/);
});

it('GPUNetworkCostMatrix with targetNodes searches into scratch and gathers every batch', () => {
  const {contributor, nodes} = getNodeIds(5, 3);
  expect(contributor.rowCount).toBe(5);
  expect(contributor.columnCount).toBe(3);
  expect(contributor.batchCount).toBe(3);
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('matrix-expand-seeds');
  expect(ids).toContain('matrix-batch-info');
  for (const batch of [0, 1, 2]) {
    const seed = ids.indexOf(`matrix-batch-${batch}-seed`);
    const relax = ids.indexOf(`matrix-batch-${batch}-relax-2`);
    const gather = ids.indexOf(`matrix-batch-${batch}-gather`);
    expect(ids.indexOf('matrix-batch-info')).toBeLessThan(seed);
    expect(seed).toBeLessThan(relax);
    expect(relax).toBeLessThan(gather);
    expect(ids).not.toContain(`matrix-batch-${batch}-copy`);
  }
  expect(ids.at(-1)).toBe('matrix-converged');
});

it('GPUNetworkCostMatrix kernels parse and stay within eight storage bindings', () => {
  const sources = getParsedKernelSources(getNodeIds(5, 3).nodes);
  const gatherSources = [...sources.entries()].filter(([id]) => id.includes('gather'));
  expect(gatherSources.length).toBeGreaterThan(0);
  for (const [, source] of sources) {
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
});

it('GPUNetworkCostMatrix with targetNodes compiles the same pipelines for every batch count', () => {
  const pipelineCount = (rowCount: number) =>
    getGPUCommandNodePipelineKeys(getNodeIds(rowCount, 3).nodes, {includeShaderLayout: true}).size;
  // Fixed laneCount 2: 6 rows are 3 batches, 7 rows are 4 batches.
  expect(getNodeIds(6, 3).contributor.batchCount).toBe(3);
  expect(getNodeIds(7, 3).contributor.batchCount).toBe(4);
  expect(pipelineCount(7)).toBe(pipelineCount(6));
  expect(pipelineCount(2)).toBe(pipelineCount(6));
});
