// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/test-utils';
import {
  GPUCommandGraph,
  GPUProgramCompiler,
  createGPUConjugateGradientProgram,
  createTransientView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {getGPUScalarReductionNodes} from '../../src/gpu-core/gpu-reduction-substrate';
import {createGPUScalar} from '../../src/gpu-core/gpu-scalar';
import {afterEach, expect, test, vi} from 'vitest';

afterEach(() => vi.restoreAllMocks());

function createDevice(sources: string[]) {
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
  vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
    sources.push(props!.shader.source);
    return {id: props?.id, hash: '', destroy() {}} as never;
  });
  return device;
}

test('CG recording preserves production node lists, dependencies, resources, workloads and WGSL', () => {
  const results = [false, true].map(recording => {
    const sources: string[] = [];
    const device = createDevice(sources);
    const submit = vi.spyOn(device, 'submit');
    const solver = createGPUConjugateGradientProgram({
      id: 'solver',
      size: 256,
      nonZeros: 256,
      maxIterations: 2,
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
          buffer: device.createBuffer({byteLength: 257 * 4, usage: Buffer.STORAGE}),
          format:
            input === solver.rowOffsets || input === solver.columnIndices ? 'uint32' : 'float32',
          length: input === solver.rowOffsets ? 257 : 256
        })
    );
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
      if (recording) {
        const kernels = description.graph.nodes.filter(node => node.source);
        expect(kernels.some(node => node.id.includes('spmv'))).toBe(true);
        expect(kernels.some(node => node.id.includes('level-0'))).toBe(true);
        for (const node of kernels) {
          expect(sources).toContain(
            description.sources.find(source => source.id === node.source)?.text
          );
          expect(node.dispatch?.indirect).toBe(node.condition?.source === 'gpu');
        }
      } else {
        expect(description.sources).toEqual([]);
        expect(description.graph.nodes.every(node => !node.source && !node.workgroupSize)).toBe(
          true
        );
      }
      expect(submit).not.toHaveBeenCalled();
      return {
        sources,
        order: compiled.stats.nodeOrder,
        edges: description.graph.edges,
        resources: description.graph.resources,
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
      for (const input of data) input.destroy();
      device.destroy();
    }
  });
  expect(results[1]).toEqual(results[0]);
});

test.each([
  false,
  true
])('scalar reduction records exact bindings and sources (square=%s)', square => {
  const results = [false, true].map(recording => {
    const sources: string[] = [];
    const device = createDevice(sources);
    const graph = new GPUCommandGraph(device, {
      id: 'reduction',
      ...(recording && {recording: {sourceText: true}})
    });
    const input = createTransientView(graph, 'left', 'float32', 257);
    const inputB = square ? input : createTransientView(graph, 'right', 'float32', 257);
    const output = createGPUScalar(graph, 'total', 'float32');
    const nodes = getGPUScalarReductionNodes(graph, {
      id: 'dot',
      input,
      inputB,
      output,
      accumulate: true
    });
    graph.add(nodes);
    const compiled = graph.compile();
    try {
      if (recording) {
        expect(nodes.map(node => node.annotation?.source?.text)).toEqual(sources);
        expect(nodes.map(node => node.annotation?.dispatch?.invocationCount)).toEqual([257, 2]);
        expect(
          nodes.map(node =>
            node.annotation?.bindings?.map(binding => [binding.name, binding.access])
          )
        ).toEqual([
          [
            ['inputValues', 'read'],
            ...(square ? [] : [['inputBValues', 'read']]),
            ['outputValues', 'write']
          ],
          [
            ['inputValues', 'read'],
            ['gpuValues', 'read_write']
          ]
        ]);
        expect(sources[1]).toContain('bitcast<f32>(gpuValues[0u]) + total');
      } else {
        for (const node of nodes) expect(node).not.toHaveProperty('annotation');
      }
      return {
        sources,
        order: compiled.stats.nodeOrder,
        workloads: nodes.map(node => node.workload)
      };
    } finally {
      compiled.destroy();
      device.destroy();
    }
  });
  expect(results[1]).toEqual(results[0]);
});
