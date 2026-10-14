// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  recordComputePipelines
} from '../../utils/gpu-contributor-test-utils';

import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

const REDUCTION_WORKGROUP_SIZE = 256;

function createNodes(pointCount: number, withRule = true) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const kde = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', pointCount),
    weights: createTransientView(graph, 'weights', 'float32', pointCount),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(4, 3, [0, 0, 4, 3]),
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 4, 3],
    bandwidthRule: withRule ? 'scott' : undefined,
    edgeCorrection: 'diggle',
    output: {
      values: createTransientView(graph, 'values', 'float32', 12),
      bandwidth: withRule ? createTransientView(graph, 'bandwidth', 'float32', 1) : undefined
    }
  });
  const nodes = kde.getCommandNodes(graph);
  return {device, nodes};
}

it('bandwidth rule moments are a two-stage hierarchical reduction, ordered before the density kernels', () => {
  const pointCount = 5000;
  const {device, nodes} = createNodes(pointCount);
  const ids = nodes.map(node => node.id);
  const firstUse = Math.min(
    ids.indexOf('kernel-density-point-weights'),
    ids.indexOf('kernel-density-gather')
  );
  const ruleIndex = ids.indexOf('kernel-density-bandwidth-rule');
  expect(ruleIndex).toBeGreaterThan(-1);
  expect(ruleIndex).toBeLessThan(firstUse);
  expect(ids.indexOf('kernel-density-bandwidth-terms')).toBeLessThan(
    ids.indexOf('kernel-density-bandwidth-means')
  );
  expect(ids.indexOf('kernel-density-bandwidth-means')).toBeLessThan(
    ids.indexOf('kernel-density-bandwidth-spread')
  );
  for (const name of ['weight', 'weight-square', 'x', 'y', 'spread-x', 'spread-y']) {
    expect(ids.some(id => id.startsWith(`kernel-density-bandwidth-sum-${name}`))).toBe(true);
  }
  // Every reduction over n rows has partials per workgroup (ceil(n / 256) of them) and a combine.
  const pipelines = recordComputePipelines(nodes);
  const bandwidthPipelines = pipelines.filter(pipeline => pipeline.id.includes('bandwidth'));
  for (const pipeline of bandwidthPipelines) {
    const count = pipeline.constants['INVOCATION_COUNT'];
    if (count !== undefined && count >= pointCount) {
      expect(count / REDUCTION_WORKGROUP_SIZE).toBeGreaterThan(1);
      expect(pipeline.source).toMatch(/@workgroup_size\((64|128|256)\)/);
    }
  }
  // Only the three row kernels use n invocations; the reductions' first level is also over n but
  // writes one partial per workgroup, and the two finalize kernels use one invocation.
  const single = bandwidthPipelines.filter(
    pipeline => pipeline.constants['INVOCATION_COUNT'] === 1
  );
  expect(single.map(pipeline => pipeline.id).sort()).toHaveLength(2);
  expect(
    nodes.filter(node => node.id.includes('bandwidth-sum-weight')).length
  ).toBeGreaterThanOrEqual(2);
  device.destroy();
});

it('bandwidth rule kernels parse as WGSL and the density kernels read the state slot', () => {
  const {device, nodes} = createNodes(300);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
  }
  const byId = (fragment: string) => sources.find(entry => entry.id.includes(fragment))?.source;
  for (const fragment of ['point-weights', 'gather']) {
    const source = byId(fragment) ?? '';
    expect(source, fragment).toContain('bandwidthState[bandwidthStateOffset]');
    expect(source, fragment).not.toContain('params[paramsOffset + 4u]');
  }
  const rule = byId('bandwidth-rule') ?? '';
  expect(rule).toContain('pow(effective, -1.0 / 6.0)');
  // NaN tests go through the shared helpers, never x == x.
  expect(rule).not.toMatch(/(\w+) == \1\b/);
  device.destroy();
});

it('bandwidth rule pipeline count does not grow with the row count', () => {
  const small = createNodes(2000);
  const large = createNodes(4000);
  const smallPipelines = recordComputePipelines(small.nodes);
  const largePipelines = recordComputePipelines(large.nodes);
  expect(largePipelines.length).toBe(smallPipelines.length);
  // The same shaders (sources) are used; only the row-count constants differ.
  expect(new Set(largePipelines.map(pipeline => pipeline.source)).size).toBe(
    new Set(smallPipelines.map(pipeline => pipeline.source)).size
  );
  small.device.destroy();
  large.device.destroy();
});

it('without a rule the bandwidth comes from the parameters and no moments are scheduled', () => {
  const {device, nodes} = createNodes(300, false);
  expect(nodes.some(node => node.id.includes('bandwidth'))).toBe(false);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.find(entry => entry.id.includes('gather'))?.source).toContain(
    'params[paramsOffset + 4u]'
  );
  device.destroy();
});

it('bandwidthRule and output.bandwidth are validated', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = {
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(4, 3, [0, 0, 4, 3]),
    indexGridSize: [2, 2] as const,
    indexBounds: [0, 0, 4, 3] as const
  };
  const values = createTransientView(graph, 'values', 'float32', 12);
  expect(
    () => new GPUKernelDensity({...base, bandwidthRule: 'cv' as never, output: {values}})
  ).toThrow(/bandwidthRule/);
  expect(
    () =>
      new GPUKernelDensity({
        ...base,
        output: {values, bandwidth: createTransientView(graph, 'bandwidth', 'float32', 1)}
      })
  ).toThrow(/needs bandwidthRule/);
  expect(
    () =>
      new GPUKernelDensity({
        ...base,
        bandwidthRule: 'silverman',
        output: {values, bandwidth: createTransientView(graph, 'bandwidth-two', 'float32', 2)}
      })
  ).toThrow(/exactly one/);
  device.destroy();
});
