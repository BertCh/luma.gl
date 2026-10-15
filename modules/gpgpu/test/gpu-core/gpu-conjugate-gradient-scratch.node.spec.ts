// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {Buffer} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {GPUProgramCompiler, createGPUConjugateGradientProgram} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {afterEach, expect, test, vi} from 'vitest';

afterEach(() => vi.restoreAllMocks());

function compileSolver(size: number, maximumIterations: number, recording = false) {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  const sources: string[] = [];
  vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
    sources.push(props!.shader.source);
    return {id: props?.id, hash: '', destroy() {}} as never;
  });
  const submit = vi.spyOn(device, 'submit');
  const solver = createGPUConjugateGradientProgram({
    id: 'solver',
    size,
    nonZeros: size,
    maxIterations: maximumIterations,
    toleranceSquared: 1e-8
  });
  const inputs = [
    solver.rowOffsets,
    solver.columnIndices,
    solver.values,
    solver.rhs,
    solver.solution
  ];
  const data = inputs.map(
    input =>
      new GPUData({
        buffer: device.createBuffer({byteLength: input.length * 4, usage: Buffer.STORAGE}),
        ownsBuffer: true,
        format: input.format,
        length: input.length
      })
  );
  try {
    const compilation = new GPUProgramCompiler(device).compile(
      solver.program,
      {
        vectors: Object.fromEntries(inputs.map((input, index) => [input.id, data[index]]))
      },
      recording ? {recording: {sourceText: true}} : {}
    );
    const compiled = compilation.graph.compile();
    try {
      const description = compiled.describe();
      expect(submit).not.toHaveBeenCalled();
      return {
        sources,
        order: compiled.stats.nodeOrder,
        edges: description.graph.edges,
        resources: description.graph.resources ?? [],
        nodes: description.graph.nodes.map(node => ({
          id: node.id,
          kind: node.kind,
          reads: node.reads,
          writes: node.writes,
          workload: node.workload,
          condition: node.condition,
          workgroups: node.dispatch?.workgroups
        }))
      };
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const input of data) input.destroy();
    device.destroy();
  }
}

test.each([
  [1, 'c8b696468a537f4ede4e28ae48cbfdec61f314f8d7216455e37f22d6e12968e3'],
  [256, '6e809ca6752e4a3b07a62c9d1f2eaa80d8342aad63f90f14b33c06a6b5a8d31d'],
  [257, '3a7bef069a21bc3731af4c22cf7a52970285fcf9c03dae3956b685a651b0c8e3'],
  [65537, '641ecf55020d4f490249ed0484c9a99f0a12be9922325078adc3b6c7262a513f']
] as const)('CG size %s preserves the previously working production graph', (size, fingerprint) => {
  // Freeze pre-fix nodes, resource layouts, dependencies, workloads and compiled WGSL.
  const result = compileSolver(size, size <= 256 ? 3 : 1);
  expect(createHash('sha256').update(JSON.stringify(result)).digest('hex')).toBe(fingerprint);
});

test.each([
  257, 65537
])('CG size %s allocates distinct reduction scratch for every iteration', size => {
  const result = compileSolver(size, 3);
  const scratch = result.resources.filter(resource => resource.id.includes('-partials-'));
  const levels = size === 257 ? 1 : 2;
  expect(scratch).toHaveLength(7 * levels);
  expect(new Set(scratch.map(resource => resource.id)).size).toBe(scratch.length);
  for (const resource of scratch) {
    expect(result.nodes.filter(node => node.writes?.includes(resource.id))).toHaveLength(1);
    expect(result.nodes.filter(node => node.reads?.includes(resource.id))).toHaveLength(1);
  }
  expect(compileSolver(size, 3, true)).toEqual(result);
  expect(compileSolver(size, 3)).toEqual(result);
});
