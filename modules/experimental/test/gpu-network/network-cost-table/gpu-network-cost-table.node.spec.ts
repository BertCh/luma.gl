// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkCostTable,
  type GPUNetworkCostTableProps
} from '../../../src/gpu-network/network-cost-table/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';

const EDGE_COUNT = 24;
let viewCount = 0;

function createView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `cost-table-view-${viewCount++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  nodeCount: number,
  overrides: Partial<GPUNetworkCostTableProps> = {},
  capacity = 100
): GPUNetworkCostTableProps {
  return {
    id: 'table',
    topology: {
      offsets: createView(graph, 'uint32', nodeCount + 1),
      neighbors: createView(graph, 'uint32', EDGE_COUNT),
      weights: createView(graph, 'float32', EDGE_COUNT)
    },
    costLimit: createView(graph, 'float32', 1),
    laneCount: 10,
    maximumIterations: 4,
    output: {
      offsets: createView(graph, 'uint32', nodeCount + 1),
      targets: createView(graph, 'uint32', capacity),
      costs: createView(graph, 'float32', capacity),
      status: {
        overflow: createView(graph, 'uint32', 1),
        requiredCount: createView(graph, 'uint32', 1),
        converged: createView(graph, 'uint32', 1)
      }
    },
    ...overrides
  };
}

function build(nodeCount: number, overrides: Partial<GPUNetworkCostTableProps> = {}) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `table-${nodeCount}`});
  const contributor = new GPUNetworkCostTable(createProps(graph, nodeCount, overrides));
  return {graph, contributor, nodes: contributor.getCommandNodes(graph)};
}

it('GPUNetworkCostTable resolves lanes and batches', () => {
  const {contributor} = build(35);
  expect(contributor.nodeCount).toBe(35);
  expect(contributor.laneCount).toBe(10);
  expect(contributor.batchCount).toBe(4);
  expect(contributor.capacity).toBe(100);
  // Default lane count: recommendLaneCount with one row per node, at most nodeCount.
  const defaults = build(35, {laneCount: undefined}).contributor;
  expect(defaults.laneCount).toBe(35);
  expect(defaults.batchCount).toBe(1);
});

it('GPUNetworkCostTable lists the init, per-batch and finalize nodes', () => {
  const {nodes} = build(35);
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('table-init');
  expect(ids.at(-1)).toBe('table-finalize');
  for (const batch of [0, 3]) {
    const seeds = ids.indexOf(`table-batch-${batch}-seeds`);
    const count = ids.indexOf(`table-batch-${batch}-count`);
    const scan = ids.indexOf(`table-batch-${batch}-scan`);
    const fill = ids.indexOf(`table-batch-${batch}-fill`);
    expect(seeds).toBeGreaterThan(0);
    expect(count).toBeGreaterThan(seeds);
    expect(scan).toBeGreaterThan(count);
    expect(fill).toBeGreaterThan(scan);
  }
  expect(ids.some(id => id.startsWith('table-batch-3-') && id.includes('relax'))).toBe(true);
  expect(ids).not.toContain('table-batch-4-count');
});

it('GPUNetworkCostTable kernels parse and stay within the core limits', () => {
  const {nodes} = build(35);
  const sources = getParsedKernelSources(nodes);
  for (const variant of ['init', 'finalize']) {
    expect(sources.has(`table-${variant}`)).toBe(true);
  }
  for (const [id, source] of sources) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  // Counts, the scan and the fill never use float atomics; the saturating cursor is plain u32.
  const fill = sources.get('table-batch-0-fill') ?? '';
  expect(fill).toContain('rank < CAPACITY - base');
  expect(fill).not.toContain('atomic');
});

it('GPUNetworkCostTable batches share pipelines (rule 12)', () => {
  const few = build(35);
  const many = build(45);
  expect(few.contributor.batchCount).toBe(4);
  expect(many.contributor.batchCount).toBe(5);
  const count = getGPUCommandNodePipelineKeys(few.nodes, {includeShaderLayout: true}).size;
  expect(getGPUCommandNodePipelineKeys(many.nodes, {includeShaderLayout: true}).size).toBe(count);
  // One set of seeds, count, scan and fill kernels serves every batch.
  for (const variant of ['seeds', 'count', 'scan', 'fill']) {
    expect(
      getGPUCommandNodePipelineKeys(many.nodes, {
        includeShaderLayout: true,
        idPattern: new RegExp(`-batch-\\d+-${variant}$`)
      }).size
    ).toBe(1);
  }
});

it('GPUNetworkCostTable validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'table-validation'});
  const make =
    (overrides: Partial<GPUNetworkCostTableProps>, nodeCount = 35) =>
    () =>
      new GPUNetworkCostTable(createProps(graph, nodeCount, overrides));
  expect(make({costLimit: createView(graph, 'float32', 2)})).toThrow(
    /costLimit must contain exactly one row/
  );
  expect(make({laneCount: 36})).toThrow(/laneCount must be an integer in \[1, 35\]/);
  expect(make({laneCount: 0})).toThrow(/laneCount/);
  const base = createProps(graph, 35);
  expect(make({output: {...base.output, offsets: createView(graph, 'uint32', 35)}})).toThrow(
    /output.offsets must contain nodeCount \+ 1 rows/
  );
  expect(make({output: {...base.output, costs: createView(graph, 'float32', 99)}})).toThrow(
    /output.costs length must equal output.targets length/
  );
  expect(
    make({
      output: {...base.output, status: {overflow: createView(graph, 'uint32', 2)}}
    })
  ).toThrow(/overflow/);
  expect(
    make({
      output: {...base.output, status: {converged: createView(graph, 'uint32', 2)}}
    })
  ).toThrow(/converged must contain exactly one row/);
  expect(
    make({
      topology: {
        offsets: createView(graph, 'uint32', 36),
        neighbors: createView(graph, 'uint32', EDGE_COUNT)
      } as never
    })
  ).toThrow(/weights/);
  // Outputs may not alias an input or each other.
  expect(
    make({
      topology: base.topology,
      output: {
        ...base.output,
        targets: base.topology.neighbors,
        costs: createView(graph, 'float32', EDGE_COUNT)
      }
    })
  ).toThrow(/outputs must not share buffers with inputs/);
  expect(make({output: {...base.output, costs: base.output.targets as never}})).toThrow();
});

it('GPUNetworkCostTable omits status kernels bindings it was not asked for', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'table-minimal'});
  const props = createProps(graph, 35);
  const contributor = new GPUNetworkCostTable({
    ...props,
    output: {
      offsets: props.output.offsets,
      targets: props.output.targets,
      costs: props.output.costs
    }
  });
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  const finalize = sources.get('table-finalize') ?? '';
  expect(getStorageBindingCount(finalize)).toBe(1);
  expect(sources.get('table-batch-0-scan')).not.toContain('batchConverged');
});
