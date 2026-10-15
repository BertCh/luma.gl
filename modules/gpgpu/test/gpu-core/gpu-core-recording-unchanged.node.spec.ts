// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  GPUScan,
  GPUFlagOffsets,
  GPUCompaction,
  GPUMask,
  createTransientView
} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {expect, test, vi} from 'vitest';

const elements = ['GPUScan', 'GPUFlagOffsets', 'GPUCompaction', 'GPUMask'] as const;

test.each(
  elements
)('%s preserves production nodes, resources, workloads and compiled WGSL', element => {
  for (const length of [0, 1, 1024]) {
    const results = [false, true].map(recording => {
      const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
      Object.defineProperty(device, 'type', {value: 'webgpu'});
      Object.defineProperty(device, 'limits', {
        value: {...device.limits, maxComputeWorkgroupsPerDimension: 65535}
      });
      // NullDevice has no compute pipelines. Record pipeline source without GPU execution.
      const sources: string[] = [];
      vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
        sources.push(props!.shader.source);
        return {id: props?.id, hash: '', destroy() {}} as never;
      });
      const graph = new GPUCommandGraph(device, {
        id: 'unchanged',
        ...(recording && {recording: {sourceText: true}})
      });
      const input = createTransientView(graph, 'input', 'uint32', length);
      const flags = createTransientView(graph, 'flags', 'uint32', length);
      const output = createTransientView(graph, 'output', 'uint32', length);
      const count = createTransientView(graph, 'count', 'uint32', 1);
      const contributor =
        element === 'GPUScan'
          ? new GPUScan({input, output})
          : element === 'GPUFlagOffsets'
            ? new GPUFlagOffsets({flags, offsets: output, count})
            : element === 'GPUCompaction'
              ? new GPUCompaction({input, flags, output, count})
              : new GPUMask({inputs: [input, flags], output, operation: 'and'});
      const nodes = contributor.getCommandNodes(graph);
      for (const node of nodes) {
        if (!recording) expect(node).not.toHaveProperty('annotation');
      }
      graph.add(nodes);
      const compiled = graph.compile();
      try {
        const description = compiled.describe();
        if (recording) {
          for (const node of description.graph.nodes) {
            expect(node.op).toBeTruthy();
            expect(node.source).toBeTruthy();
            expect(node.workgroupSize).toHaveLength(3);
            expect(node.dispatch?.workgroups).toHaveLength(3);
          }
          expect(nodes.map(node => node.annotation?.source?.text)).toEqual(sources);
          if (length === 1024 && element !== 'GPUMask') {
            const scans = nodes.filter(node => node.annotation?.op === 'GPUScan');
            expect(scans.map(node => node.annotation?.loop?.iteration)).toEqual([0, 1, 0]);
            expect(scans.map(node => node.annotation?.dedupName)).toEqual([
              'level-scan',
              'level-scan',
              'add-offsets'
            ]);
            expect(scans.every(node => node.annotation?.loop?.maximumIterations === 2)).toBe(true);
          }
        }
        return {
          order: compiled.stats.nodeOrder,
          sources,
          nodes: nodes.map(node => ({
            id: node.id,
            type: node.type,
            workload: node.workload,
            resources: node.resources.map(resource =>
              'buffer' in resource
                ? {
                    usage: resource.usage,
                    id:
                      'buffer' in resource.buffer ? resource.buffer.buffer.id : resource.buffer.id,
                    ...('buffer' in resource.buffer && {
                      byteOffset: resource.buffer.byteOffset,
                      length: resource.buffer.length
                    })
                  }
                : resource
            )
          }))
        };
      } finally {
        compiled.destroy();
        device.destroy();
      }
    });
    expect(results[1]).toEqual(results[0]);
  }
});

import {
  GPUElementwise,
  GPUGather,
  GPUHistogram,
  GPUReduction,
  GPUSegmentedReduction,
  GPUSegmentedSort,
  GPUSort,
  type GPUCommandNode
} from '@luma.gl/gpgpu/gpu-core';

type ReductionCase = {
  name: string;
  lengths: readonly number[];
  /** Recorded workloads are rewritten by the sort recording hooks, so they are not compared. */
  comparesWorkload: boolean;
  build(graph: GPUCommandGraph<undefined>, length: number): readonly GPUCommandNode<undefined>[];
};

const SORT_CASES: readonly ReductionCase[] = [
  {
    name: 'GPUReduction',
    lengths: [0, 1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUReduction({
        input: createTransientView(graph, 'input', 'float32', length),
        output: createTransientView(graph, 'output', 'float32', 1),
        operation: 'max'
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUSegmentedReduction',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUSegmentedReduction({
        input: createTransientView(graph, 'input', 'float32', length),
        segmentOffsets: createTransientView(graph, 'offsets', 'uint32', 3),
        output: createTransientView(graph, 'output', 'float32', 2),
        operation: 'sum'
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUSort radix',
    lengths: [1, 1024],
    comparesWorkload: false,
    build: (graph, length) =>
      new GPUSort({
        keys: createTransientView(graph, 'keys', 'uint32', length),
        values: createTransientView(graph, 'values', 'uint32', length),
        outputKeys: createTransientView(graph, 'outputKeys', 'uint32', length),
        outputValues: createTransientView(graph, 'outputValues', 'uint32', length),
        algorithm: 'radix',
        direction: 'descending',
        keyBits: 12
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUSort bitonic',
    lengths: [1, 200, 1024],
    comparesWorkload: false,
    build: (graph, length) =>
      new GPUSort({
        keys: createTransientView(graph, 'keys', 'uint32', length),
        values: createTransientView(graph, 'values', 'uint32', length),
        outputKeys: createTransientView(graph, 'outputKeys', 'uint32', length),
        outputValues: createTransientView(graph, 'outputValues', 'uint32', length),
        algorithm: 'bitonic'
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUSegmentedSort',
    lengths: [1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUSegmentedSort({
        keys: createTransientView(graph, 'keys', 'uint32', length),
        values: createTransientView(graph, 'values', 'uint32', length),
        outputKeys: createTransientView(graph, 'outputKeys', 'uint32', length),
        outputValues: createTransientView(graph, 'outputValues', 'uint32', length),
        segments: [
          {keysOffset: 0, valuesOffset: 0, outputKeysOffset: 0, outputValuesOffset: 0, length: 61},
          {
            keysOffset: 61,
            valuesOffset: 61,
            outputKeysOffset: 61,
            outputValuesOffset: 61,
            length: 134
          }
        ]
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUGather',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUGather({
        source: createTransientView(graph, 'source', 'float32', length),
        indices: createTransientView(graph, 'indices', 'uint32', 16),
        output: createTransientView(graph, 'output', 'float32', 16)
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUHistogram',
    lengths: [0, 1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUHistogram({
        input: createTransientView(graph, 'input', 'float32', length),
        output: createTransientView(graph, 'output', 'uint32', 16),
        domain: [0, 320]
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUElementwise',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUElementwise({
        input: createTransientView(graph, 'input', 'float32', length),
        inputB: createTransientView(graph, 'inputB', 'float32', length),
        output: createTransientView(graph, 'output', 'float32', length),
        operation: 'add'
      }).getCommandNodes(graph)
  }
];

test.each(SORT_CASES)('$name preserves production nodes, resources and compiled WGSL', ({
  lengths,
  comparesWorkload,
  build
}) => {
  for (const length of lengths) {
    const results = [false, true].map(recording => {
      const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
      Object.defineProperty(device, 'type', {value: 'webgpu'});
      Object.defineProperty(device, 'limits', {
        value: {
          ...device.limits,
          maxComputeWorkgroupsPerDimension: 65535,
          maxStorageBufferBindingSize: 2 ** 30
        }
      });
      const sources: string[] = [];
      vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
        sources.push(props!.shader.source);
        return {id: props?.id, hash: '', destroy() {}} as never;
      });
      const graph = new GPUCommandGraph<undefined>(device, {
        id: 'unchanged',
        ...(recording && {recording: {sourceText: true}})
      });
      const nodes = build(graph, length);
      for (const node of nodes) {
        if (!recording) expect(node).not.toHaveProperty('annotation');
      }
      graph.add(nodes);
      const compiled = graph.compile();
      try {
        const description = compiled.describe();
        if (recording) {
          for (const node of description.graph.nodes) {
            expect(node.op).toBeTruthy();
            expect(node.source).toBeTruthy();
            expect(node.workgroupSize).toHaveLength(3);
            expect(node.dispatch?.workgroups).toHaveLength(3);
          }
          expect(nodes.map(node => node.annotation?.source?.text)).toEqual(sources);
        }
        return {
          order: compiled.stats.nodeOrder,
          sources,
          nodes: nodes.map(node => ({
            id: node.id,
            type: node.type,
            ...(comparesWorkload && {workload: node.workload}),
            resources: node.resources.map(resource =>
              'buffer' in resource
                ? {
                    usage: resource.usage,
                    id:
                      'buffer' in resource.buffer ? resource.buffer.buffer.id : resource.buffer.id,
                    ...('buffer' in resource.buffer && {
                      byteOffset: resource.buffer.byteOffset,
                      length: resource.buffer.length
                    })
                  }
                : resource
            )
          }))
        };
      } finally {
        compiled.destroy();
        device.destroy();
      }
    });
    expect(results[1]).toEqual(results[0]);
  }
});

test('GPUSort and GPUReduction record unrolled passes and levels as loop coordinates', () => {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => ({id: props?.id, hash: '', destroy() {}}) as never
  );
  const graph = new GPUCommandGraph<undefined>(device, {
    id: 'loops',
    recording: {sourceText: true}
  });
  const sortNodes = SORT_CASES[2].build(graph, 1024);
  const passes = sortNodes.filter(node => node.annotation?.loop?.id === 'gpu-sort/passes');
  // Histogram and scatter of each of the three 4-bit radix passes carry the pass index.
  expect(passes.map(node => node.annotation?.loop?.iteration)).toEqual([0, 0, 1, 1, 2, 2]);
  expect(passes.every(node => node.annotation?.loop?.maximumIterations === 3)).toBe(true);
  expect(new Set(passes.map(node => node.annotation?.dedupName))).toEqual(
    new Set(['radix-histogram', 'radix-scatter'])
  );
  const reductionNodes = SORT_CASES[0].build(graph, 1024);
  expect(reductionNodes.map(node => node.annotation?.loop?.iteration)).toEqual([0, 1, undefined]);
  expect(reductionNodes.map(node => node.annotation?.formula)).toEqual([
    'gpu-core.reduction',
    'gpu-core.reduction',
    'gpu-core.reduction'
  ]);
  device.destroy();
});

import {
  GPUGridAggregation,
  GPUGridBinning,
  GPUGridIndex,
  GPUGroupAggregation,
  GPUHashIndex,
  GPUHashJoin
} from '@luma.gl/gpgpu/gpu-core';

const GRID_AND_HASH_CASES: readonly (Omit<ReductionCase, 'lengths'> & {
  op: string;
  lengths: readonly number[];
})[] = [
  {
    name: 'GPUGridIndex',
    op: 'GPUGridIndex',
    lengths: [1, 1024],
    // Recording adds byte counts to the index workloads.
    comparesWorkload: false,
    build: (graph, length) =>
      new GPUGridIndex({
        positions: createTransientView(graph, 'positions', 'float32x2', length),
        gridSize: [18, 16],
        bounds: [-92, 24, -74, 40],
        cellOffsets: createTransientView(graph, 'cellOffsets', 'uint32', 289),
        objectIds: createTransientView(graph, 'objectIds', 'uint32', length),
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUGridBinning',
    op: 'GPUGridBinning',
    lengths: [1, 1024],
    comparesWorkload: false,
    build: (graph, length) =>
      new GPUGridBinning({
        positions: createTransientView(graph, 'positions', 'float32x2', length),
        output: createTransientView(graph, 'output', 'uint32', 288),
        gridSize: [18, 16],
        bounds: [-92, 24, -74, 40]
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUGridAggregation mean',
    op: 'GPUGridAggregation',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUGridAggregation({
        positions: createTransientView(graph, 'positions', 'float32x2', length),
        weights: createTransientView(graph, 'weights', 'float32', length),
        output: createTransientView(graph, 'output', 'float32', 288),
        operation: 'mean',
        gridSize: [18, 16],
        bounds: [-92, 24, -74, 40]
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUGroupAggregation count',
    op: 'GPUGroupAggregation',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUGroupAggregation({
        keys: createTransientView(graph, 'keys', 'uint32', length),
        output: createTransientView(graph, 'output', 'uint32', 23)
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUGroupAggregation mean',
    op: 'GPUGroupAggregation',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUGroupAggregation({
        keys: createTransientView(graph, 'keys', 'uint32', length),
        values: createTransientView(graph, 'values', 'float32', length),
        output: createTransientView(graph, 'output', 'float32', 23),
        operation: 'mean'
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUHashIndex',
    op: 'GPUHashIndex',
    lengths: [1, 1024],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUHashIndex({
        keys: createTransientView(graph, 'keys', 'uint32', length),
        tableKeys: createTransientView(graph, 'tableKeys', 'uint32', 2048),
        tableValues: createTransientView(graph, 'tableValues', 'uint32', 2048),
        statistics: createTransientView(graph, 'statistics', 'uint32', 6)
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUHashJoin',
    op: 'GPUHashJoin',
    lengths: [1, 143],
    comparesWorkload: true,
    build: (graph, length) => {
      const index = new GPUHashIndex({
        keys: createTransientView(graph, 'rightKeys', 'uint32', 1024),
        tableKeys: createTransientView(graph, 'tableKeys', 'uint32', 2048),
        tableValues: createTransientView(graph, 'tableValues', 'uint32', 2048),
        statistics: createTransientView(graph, 'indexStatistics', 'uint32', 6)
      });
      return [
        ...index.getCommandNodes(graph),
        ...new GPUHashJoin({
          index,
          keys: createTransientView(graph, 'leftKeys', 'uint32', length),
          outputLeftRows: createTransientView(graph, 'outputLeftRows', 'uint32', length),
          outputRightRows: createTransientView(graph, 'outputRightRows', 'uint32', length),
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1),
          statistics: createTransientView(graph, 'lookupStatistics', 'uint32', 4)
        }).getCommandNodes(graph)
      ];
    }
  }
];

test.each(GRID_AND_HASH_CASES)('$name preserves production nodes and annotates its own kernels', ({
  op,
  lengths,
  comparesWorkload,
  build
}) => {
  for (const length of lengths) {
    const results = [false, true].map(recording => {
      const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
      Object.defineProperty(device, 'type', {value: 'webgpu'});
      Object.defineProperty(device, 'limits', {
        value: {
          ...device.limits,
          maxComputeWorkgroupsPerDimension: 65535,
          maxStorageBufferBindingSize: 2 ** 30
        }
      });
      const sources: string[] = [];
      vi.spyOn(device, 'createComputePipeline').mockImplementation(props => {
        sources.push(props!.shader.source);
        return {id: props?.id, hash: '', destroy() {}} as never;
      });
      const graph = new GPUCommandGraph<undefined>(device, {
        id: 'unchanged',
        ...(recording && {recording: {sourceText: true}})
      });
      const nodes = build(graph, length);
      for (const node of nodes) {
        if (!recording) expect(node).not.toHaveProperty('annotation');
      }
      graph.add(nodes);
      const compiled = graph.compile();
      try {
        compiled.describe();
        if (recording) {
          const annotated = nodes.filter(node => node.annotation?.op === op);
          expect(annotated.length).toBeGreaterThan(0);
          for (const node of annotated) {
            expect(node.annotation?.source?.text).toBeTruthy();
            expect(node.annotation?.workgroupSize).toHaveLength(3);
            expect(node.annotation?.dispatch?.workgroups).toHaveLength(3);
            expect(sources).toContain(node.annotation!.source!.text);
          }
        }
        return {
          order: compiled.stats.nodeOrder,
          sources,
          nodes: nodes.map(node => ({
            id: node.id,
            type: node.type,
            ...(comparesWorkload && {workload: node.workload}),
            resources: node.resources.map(resource =>
              'buffer' in resource
                ? {
                    usage: resource.usage,
                    id:
                      'buffer' in resource.buffer ? resource.buffer.buffer.id : resource.buffer.id,
                    ...('buffer' in resource.buffer && {
                      byteOffset: resource.buffer.byteOffset,
                      length: resource.buffer.length
                    })
                  }
                : resource
            )
          }))
        };
      } finally {
        compiled.destroy();
        device.destroy();
      }
    });
    expect(results[1]).toEqual(results[0]);
  }
});

import {GPUConvolution, GPUFFT1D, GPUFFT2D, GPUMatMul, GPUMatVec} from '@luma.gl/gpgpu/gpu-core';

const DENSE_AND_SPECTRAL_CASES: readonly (Omit<ReductionCase, 'lengths'> & {
  op: string;
  lengths: readonly number[];
})[] = [
  {
    name: 'GPUMatVec',
    op: 'GPUMatVec',
    lengths: [8, 40],
    comparesWorkload: true,
    build: (graph, size) =>
      new GPUMatVec({
        matrix: createTransientView(graph, 'matrix', 'float32', size * size),
        vector: createTransientView(graph, 'vector', 'float32', size),
        output: createTransientView(graph, 'output', 'float32', size),
        rows: size,
        columns: size
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUMatMul',
    op: 'GPUMatMul',
    lengths: [8, 40],
    comparesWorkload: true,
    build: (graph, size) =>
      new GPUMatMul({
        left: createTransientView(graph, 'left', 'float32', size * size),
        right: createTransientView(graph, 'right', 'float32', size * size),
        output: createTransientView(graph, 'output', 'float32', size * size),
        m: size,
        k: size,
        n: size
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUConvolution direct',
    op: 'GPUConvolution',
    lengths: [8, 64],
    comparesWorkload: true,
    build: (graph, size) =>
      new GPUConvolution({
        input: createTransientView(graph, 'input', 'float32', size * size),
        kernel: createTransientView(graph, 'kernel', 'float32', 9),
        output: createTransientView(graph, 'output', 'float32', size * size),
        width: size,
        height: size,
        kernelWidth: 3,
        kernelHeight: 3,
        strategy: 'direct'
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUConvolution fft',
    op: 'GPUConvolution',
    lengths: [8],
    comparesWorkload: true,
    build: (graph, size) =>
      new GPUConvolution({
        input: createTransientView(graph, 'input', 'float32', size * size),
        kernel: createTransientView(graph, 'kernel', 'float32', 9),
        output: createTransientView(graph, 'output', 'float32', size * size),
        width: size,
        height: size,
        kernelWidth: 3,
        kernelHeight: 3,
        strategy: 'fft'
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUFFT1D',
    op: 'GPUFFT1D',
    lengths: [8, 64],
    comparesWorkload: true,
    build: (graph, length) =>
      new GPUFFT1D({
        input: createTransientView(graph, 'input', 'float32x2', length),
        output: createTransientView(graph, 'output', 'float32x2', length),
        length
      }).getCommandNodes(graph)
  },
  {
    name: 'GPUFFT2D',
    op: 'GPUFFT2D',
    lengths: [8, 64],
    comparesWorkload: true,
    build: (graph, size) =>
      new GPUFFT2D({
        input: createTransientView(graph, 'input', 'float32x2', size * size),
        output: createTransientView(graph, 'output', 'float32x2', size * size),
        width: size,
        height: size
      }).getCommandNodes(graph)
  }
];

test.each(
  DENSE_AND_SPECTRAL_CASES
)('$name preserves production nodes and annotates its own kernels', ({
  op,
  lengths,
  comparesWorkload,
  build
}) => {
  for (const length of lengths) {
    const results = [false, true].map(recording => {
      const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
      Object.defineProperty(device, 'type', {value: 'webgpu'});
      Object.defineProperty(device, 'limits', {
        value: {
          ...device.limits,
          maxComputeWorkgroupsPerDimension: 65535,
          maxComputeInvocationsPerWorkgroup: 256,
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 256,
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
      const graph = new GPUCommandGraph<undefined>(device, {
        id: 'unchanged',
        ...(recording && {recording: {sourceText: true}})
      });
      const nodes = build(graph, length);
      for (const node of nodes) {
        if (!recording) expect(node).not.toHaveProperty('annotation');
      }
      graph.add(nodes);
      const compiled = graph.compile();
      try {
        compiled.describe();
        if (recording) {
          const annotated = nodes.filter(node => node.annotation?.op === op);
          expect(annotated.length).toBeGreaterThan(0);
          for (const node of annotated) {
            expect(node.annotation?.source?.text).toBeTruthy();
            expect(node.annotation?.workgroupSize).toHaveLength(3);
            expect(node.annotation?.dispatch?.workgroups).toHaveLength(3);
            expect(sources).toContain(node.annotation!.source!.text);
          }
          // The documented result is computed by a node that carries the curated formula.
          expect(annotated.some(node => node.annotation?.formula?.startsWith('gpu-core.'))).toBe(
            true
          );
        }
        return {
          order: compiled.stats.nodeOrder,
          sources,
          nodes: nodes.map(node => ({
            id: node.id,
            type: node.type,
            ...(comparesWorkload && {workload: node.workload}),
            resources: node.resources.map(resource =>
              'buffer' in resource
                ? {
                    usage: resource.usage,
                    id:
                      'buffer' in resource.buffer ? resource.buffer.buffer.id : resource.buffer.id,
                    ...('buffer' in resource.buffer && {
                      byteOffset: resource.buffer.byteOffset,
                      length: resource.buffer.length
                    })
                  }
                : resource
            )
          }))
        };
      } finally {
        compiled.destroy();
        device.destroy();
      }
    });
    expect(results[1]).toEqual(results[0]);
  }
});
