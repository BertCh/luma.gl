// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addGPUCommandNodes,
  type GPUCommandGraph,
  type GPUCommandGraphComputeNode,
  GraphBufferHandle,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, test} from 'vitest';
import {
  GPUSpatialScanStatistic,
  type GPUSpatialScanModel
} from '../../../src/gpu-spatial-analysis/scan-statistics';

// WebGPU's default (and the 'core' device) limit of storage buffers per shader stage.
const CORE_STORAGE_BUFFER_LIMIT = 8;

type RecordingGraph = GPUCommandGraph & {passes: GPUCommandGraphComputeNode<unknown>[]};

describe('GPUSpatialScanStatistic storage buffer bindings', () => {
  test.each<GPUSpatialScanModel>([
    'poisson',
    'bernoulli',
    'space-time-permutation'
  ])('%s binds at most 8 storage buffers per pass with maximal options', model => {
    const graph = makeRecordingGraph(`scan-bindings-${model}`, CORE_STORAGE_BUFFER_LIMIT);
    const zones = 12;
    const timeBuckets = 3;
    const cells = zones * timeBuckets;
    const maximumPermutations = 9;
    const maximumClusters = 4;
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => makeView(graph, `scan-${name}`, format, length);
    const statistic = new GPUSpatialScanStatistic({
      positions: view('positions', 'float32x2', zones),
      model,
      cases: view('cases', 'uint32', cells),
      ...(model === 'poisson' ? {baseline: view('baseline', 'float32', cells)} : {}),
      ...(model === 'bernoulli' ? {controls: view('controls', 'uint32', cells)} : {}),
      timeBuckets,
      maximumWindowZones: 8,
      maximumPermutations,
      maximumClusters,
      parameters: view('parameters', 'uint32', 8),
      output: {
        clusterIndices: view('cluster-indices', 'uint32', maximumClusters * 4),
        clusterStatistics: view('cluster-statistics', 'float32', maximumClusters * 8),
        statistics: view('statistics', 'float32', maximumPermutations + 1),
        summary: view('summary', 'uint32', 4),
        zoneStatistics: view('zone-statistics', 'float32', zones * 4)
      }
    });

    // The kernel declaration itself throws above the device limit; assert the counts too.
    addGPUCommandNodes(graph, statistic.getCommandNodes(graph));
    expect(graph.passes.length).toBeGreaterThan(0);
    const evaluate = graph.passes.find(pass => pass.id.endsWith('-evaluate'));
    expect(evaluate).toBeDefined();
    for (const pass of graph.passes) {
      const storageBindings = (pass.resources ?? []).filter(resource =>
        resource.usage.startsWith('storage')
      );
      expect(storageBindings.length, pass.id).toBeLessThanOrEqual(CORE_STORAGE_BUFFER_LIMIT);
    }
    expect(evaluate!.resources!.length).toBe(CORE_STORAGE_BUFFER_LIMIT);
  });
});

function makeRecordingGraph(id: string, maxStorageBuffersPerShaderStage: number): RecordingGraph {
  const passes: GPUCommandGraphComputeNode<unknown>[] = [];
  const graph = {
    id,
    device: {
      type: 'webgpu',
      limits: {
        maxTextureDimension2D: 8192,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupSizeY: 256,
        maxComputeWorkgroupsPerDimension: 65535,
        maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: 134217728,
        maxBufferSize: 134217728,
        minStorageBufferOffsetAlignment: 256
      }
    },
    passes,
    addComputePass(pass: GPUCommandGraphComputeNode<unknown>): void {
      passes.push(pass);
    },
    createTransientBuffer(descriptor: {
      id: string;
      byteLength: number;
      usage: number;
    }): GraphBufferHandle {
      return new GraphBufferHandle(graph as never, descriptor, true);
    },
    createDataView(
      buffer: GraphBufferHandle,
      props: {format: 'float32' | 'uint32'; length: number; byteOffset?: number}
    ): GraphDataView<'float32' | 'uint32'> {
      return new GraphDataView(buffer, {
        format: props.format,
        length: props.length,
        byteOffset: props.byteOffset ?? 0,
        byteStride: 4,
        rowByteLength: 4
      });
    }
  };
  return graph as unknown as RecordingGraph;
}

function makeView<Format extends 'float32' | 'uint32' | 'float32x2'>(
  graph: RecordingGraph,
  id: string,
  format: Format,
  length: number
): GraphDataView<Format> {
  const rowByteLength = format === 'float32x2' ? 8 : 4;
  const buffer = new GraphBufferHandle(
    graph,
    {id, byteLength: Math.max(length, 1) * rowByteLength, usage: 0},
    false
  );
  return new GraphDataView(buffer, {
    format,
    length,
    byteOffset: 0,
    byteStride: rowByteLength,
    rowByteLength
  });
}
