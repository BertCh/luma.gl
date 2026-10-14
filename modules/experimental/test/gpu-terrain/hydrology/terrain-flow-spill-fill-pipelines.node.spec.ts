// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {createTerrainFlowSpillFillNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-fill';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

function buildSpillFill(
  width: number,
  height: number,
  options: {device?: Device; outputs?: boolean} = {}
) {
  const device = options.device ?? createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const cellCount = width * height;
  const result = createTerrainFlowSpillFillNodes(graph, {
    id: 'flow',
    width,
    height,
    cellSizeMode: 'uniform',
    maxIterations: 512,
    elevation: createTransientView(graph, 'elevation', 'float32', cellCount),
    filled: createTransientView(graph, 'filled', 'float32', cellCount),
    settings: createTransientView(graph, 'settings', 'float32', 8),
    converged: options.outputs ? createTransientView(graph, 'converged', 'uint32', 1) : undefined,
    iterationCount: options.outputs ? createTransientView(graph, 'rounds', 'uint32', 1) : undefined
  });
  return result;
}

// 6 x 5 = 30 cells needs 6 Boruvka rounds; 40 x 36 = 1440 cells needs 12.
const SIZES = [
  [6, 5],
  [40, 36]
] as const;
// 7 x 7 = 49 cells needs 7 Boruvka rounds. The arm statements of a loop gate hold the bounded
// dispatch layout of its shape (workgroup counts), so kernels over larger shapes legitimately form
// new pipelines; with every shape inside one 256-wide workgroup (5 n <= 256) only the rounds differ.
const ROUND_SIZES = [
  [6, 5],
  [7, 7]
] as const;

it('compiles the same pipelines at grid sizes with different round counts (rule 12)', () => {
  const counts = ROUND_SIZES.map(([width, height]) => {
    const {nodes} = buildSpillFill(width, height);
    return {
      nodeCount: nodes.length,
      keys: getGPUCommandNodePipelineKeys(nodes).size,
      // Every round of a family shares one pipeline, whatever its index or buffer direction.
      boruvka: [
        /-spill-boruvka-\d+-clear$/,
        /-spill-boruvka-\d+-pick-key$/,
        /-spill-boruvka-\d+-pick-edge$/,
        /-spill-boruvka-\d+-hook$/,
        /-spill-boruvka-\d+-record$/,
        /-spill-boruvka-\d+-apply$/,
        /-spill-boruvka-\d+-jump-\d+$/,
        /-spill-tree-rank-round-\d+$/,
        /-spill-tree-path-round-\d+$/
      ].map(idPattern => getGPUCommandNodePipelineKeys(nodes, {idPattern}).size)
    };
  });
  expect(counts[1].nodeCount).toBeGreaterThan(counts[0].nodeCount);
  expect(counts[1].keys).toBe(counts[0].keys);
  expect(counts[0].boruvka).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
  expect(counts[1].boruvka).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
});

it('every round family shares one pipeline at larger grids too', () => {
  const {nodes} = buildSpillFill(40, 36);
  for (const idPattern of [
    /-spill-boruvka-\d+-pick-key$/,
    /-spill-boruvka-\d+-hook$/,
    /-spill-boruvka-\d+-jump-\d+$/,
    /-spill-tree-rank-round-\d+$/,
    /-spill-tree-path-round-\d+$/
  ]) {
    expect(getGPUCommandNodePipelineKeys(nodes, {idPattern}).size, String(idPattern)).toBe(1);
  }
});

it('builds the documented node families with the documented bindings and gates', () => {
  const {nodes, status} = buildSpillFill(40, 36, {outputs: true});
  const ids = nodes.map(node => node.id);
  expect(status.length).toBe(4);
  for (const family of [
    /flow-spill-select$/,
    /flow-fill-init$/,
    /flow-spill-tile-fill$/,
    /flow-spill-init$/,
    /flow-spill-boruvka-0-pick-key$/,
    /flow-spill-tree-lists$/,
    /flow-spill-tree-link$/,
    /flow-spill-tree-root$/,
    /flow-spill-final$/,
    /flow-spill-publish$/
  ]) {
    expect(
      ids.some(id => family.test(id)),
      String(family)
    ).toBe(true);
  }
  // The epsilon relaxation path keeps its names and rounds.
  expect(ids.some(id => /flow-fill-0$/.test(id))).toBe(false);
  expect(ids.some(id => /flow-fill-relax-0$/.test(id))).toBe(true);
  // Every large kernel is gated on the GPU (rules 13 and 14): spill kernels on the mode gate or
  // on a loop gate, the relaxation initialization on the mode gate.
  const gated = nodes as readonly {id: string; condition?: {source?: string; mode?: string}}[];
  for (const node of gated.filter(({id}) =>
    /-spill-(tile-fill|init|final|boruvka-\d+-(clear|pick-key|pick-edge|hook|record|apply|jump-\d+)|tree-(lists|link|root|rank-round-\d+|path-round-\d+))$|flow-fill-init$|flow-fill-relax-\d+$/.test(
      id
    )
  )) {
    expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
});

it('every spill kernel parses and respects the core limits and the kernel audit', () => {
  for (const [width, height] of SIZES) {
    const {nodes} = buildSpillFill(width, height, {outputs: true});
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(15);
    for (const [id, source] of sources) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
      expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
        CORE_WORKGROUP_STORAGE_LIMIT_BYTES
      );
      expect(auditWGSLModule(source), id).toEqual([]);
      // A loop or round index never reaches the WGSL text.
      expect(source, id).not.toMatch(/-round-\d+|boruvka-\d+|jump-\d+/);
    }
  }
});

it('the tile kernel uses 12 KiB of workgroup memory, or 16 x 16 tiles on a small device', () => {
  const kernelSource = (device?: Device) => {
    const {nodes} = buildSpillFill(70, 70, {device});
    const tile = [...getParsedKernelSources(nodes)].find(([id]) => /spill-tile-fill$/.test(id));
    expect(tile).toBeTruthy();
    return (tile as [string, string])[1];
  };
  const large = kernelSource();
  expect(large).toMatch(/const TILE: u32 = 32u;/);
  expect(getWorkgroupMemoryBytes(large)).toBeLessThanOrEqual(12296);

  const small = createNullWebGPUDevice();
  Object.defineProperty(small, 'limits', {
    value: {...small.limits, maxComputeWorkgroupStorageSize: 8192}
  });
  const smallSource = kernelSource(small);
  expect(smallSource).toMatch(/const TILE: u32 = 16u;/);
  expect(getWorkgroupMemoryBytes(smallSource)).toBeLessThanOrEqual(8192);
});

it('grids whose tour arrays exceed one binding use the relaxation alone', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 8192}
  });
  // 40 x 36 cells need 8 * 1441 bytes for the tour; the limit is 8 KiB.
  const {nodes} = buildSpillFill(40, 36, {device});
  expect(nodes.some(node => /-spill-/.test(node.id))).toBe(false);
  expect(nodes.some(node => /flow-fill-init$/.test(node.id))).toBe(true);
});
