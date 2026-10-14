// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join';
import {getNearestNeighborNodes} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const BINDING_LIMIT_BYTES = 134217728;

/** Compiles `nodes` on a recording device and returns each pipeline's source and constants. */
function recordPipelines(nodes: readonly unknown[]): {id: string; source: string; key: string}[] {
  const device: Device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const pipelines: {id: string; source: string; key: string}[] = [];
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string; shader: {source: string}; constants?: object}) => {
      pipelines.push({
        id: props.id ?? '',
        source: props.shader.source,
        key: `${props.shader.source}\n${JSON.stringify(props.constants ?? null)}`
      });
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  for (const node of nodes as {compile?: (context: {device: Device}) => unknown}[]) {
    node.compile?.({device});
  }
  device.destroy();
  return pipelines;
}

type ScratchOptions = {
  queryCount: number;
  k?: number;
  footPoints?: boolean;
  scratchChunkQueryCount?: number;
};

function buildNeighborNodes(graph: GPUCommandGraph, options: ScratchOptions, prefix = 'n') {
  const k = options.k ?? 8;
  const {queryCount} = options;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${prefix}-${name}`, format, length);
  return getNearestNeighborNodes(graph, {
    id: prefix,
    operation: 'GPUNearestFeatureJoin',
    queries: {kind: 'points', points: view('points', 'float32x2', queryCount)},
    queryCount,
    features: {
      kind: 'segments',
      starts: view('starts', 'float32x2', 100),
      ends: view('ends', 'float32x2', 100)
    },
    featureCount: 100,
    k,
    capacity: k,
    leafCapacity: 128,
    spatialSort: false,
    neighborIds: view('ids', 'uint32', queryCount * k),
    neighborCounts: view('counts', 'uint32', queryCount),
    neighborDistances: view('distances', 'float32', queryCount * k),
    ...(options.footPoints
      ? {
          neighborFootPoints: view('foot', 'float32x2', queryCount * k),
          neighborQueryPoints: view('query-points', 'float32x2', queryCount * k),
          neighborSegmentIndices: view('segments', 'uint32', queryCount * k)
        }
      : {}),
    overflow: view('overflow', 'uint32', 1),
    scratchChunkQueryCount: options.scratchChunkQueryCount
  }).nodes;
}

it('nearest neighbors build at 900k queries with k = 8 and every scratch binding under the limit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = buildNeighborNodes(graph, {queryCount: 900_000, footPoints: true});
  const ids = nodes.map(node => node.id);
  // Two scratch chunks: the cursor advances once between them.
  expect(ids.filter(id => id.endsWith('-advance-cursor'))).toHaveLength(1);
  expect(ids).toContain('n-clear-overflow');
  // Every storage binding of every node, scratch included, fits one binding.
  for (const node of nodes) {
    for (const resource of (
      node as {resources?: {buffer?: {length: number; rowByteLength: number}}[]}
    ).resources ?? []) {
      if (resource.buffer) {
        expect(resource.buffer.length * resource.buffer.rowByteLength).toBeLessThanOrEqual(
          BINDING_LIMIT_BYTES
        );
      }
    }
  }
  // Compiling parses and records every kernel; the chunked kernels are valid WGSL.
  for (const {source} of recordPipelines(nodes)) {
    expect(() => new WgslReflect(source)).not.toThrow();
  }
  device.destroy();
});

it('nearest neighbors keep one scratch when it fits and share pipelines across chunks', () => {
  const countDistinctPipelines = (queryCount: number) => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const nodes = buildNeighborNodes(graph, {
      queryCount,
      footPoints: true,
      scratchChunkQueryCount: 128
    });
    const pipelines = recordPipelines(nodes);
    const chunkCount = nodes.filter(node => node.id.endsWith('-advance-cursor')).length + 1;
    device.destroy();
    return {
      distinct: new Set(pipelines.map(pipeline => pipeline.key)).size,
      distinctSources: new Set(pipelines.map(pipeline => pipeline.source)).size,
      chunkCount
    };
  };
  // Query counts that differ by a whole chunk (and have the same remainder rows) so that the
  // constants of the chunked kernels (query count of the dispatch) are the only difference.
  const two = countDistinctPipelines(128 + 5);
  const three = countDistinctPipelines(2 * 128 + 5);
  const four = countDistinctPipelines(3 * 128 + 5);
  expect([two.chunkCount, three.chunkCount, four.chunkCount]).toEqual([2, 3, 4]);
  expect(two.distinct).toBeGreaterThan(6);
  expect(three.distinct).toBe(two.distinct);
  expect(four.distinct).toBe(two.distinct);
  expect(three.distinctSources).toBe(two.distinctSources);
  expect(four.distinctSources).toBe(two.distinctSources);
});

it('nearest neighbors do not chunk when the scratch fits one binding', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = buildNeighborNodes(graph, {queryCount: 1000, footPoints: true}).map(node => node.id);
  expect(ids).toContain('n-clear');
  expect(ids.some(id => id.includes('scratch') || id.includes('cursor'))).toBe(false);
  device.destroy();
});

it('nearest neighbors reject a scratch chunk size that is not a multiple of 64', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => buildNeighborNodes(graph, {queryCount: 1000, scratchChunkQueryCount: 100})).toThrow(
    /multiple of 64/
  );
  device.destroy();
});

it('GPUNearestFeatureJoin neighbors mode builds at 900k queries with k = 8', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const queryCount = 900_000;
  const k = 8;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `j-${name}`, format, length);
  const nodes = new GPUNearestFeatureJoin({
    id: 'j',
    queries: {kind: 'points', positions: view('points', 'float32x2', queryCount)},
    features: {
      kind: 'segments',
      starts: view('starts', 'float32x2', 10),
      ends: view('ends', 'float32x2', 10)
    },
    k,
    output: {
      neighborIds: view('ids', 'uint32', queryCount * k),
      neighborCounts: view('counts', 'uint32', queryCount),
      neighborFootPoints: view('foot', 'float32x2', queryCount * k),
      status: {overflow: view('overflow', 'uint32', 1)}
    }
  }).getCommandNodes(graph);
  expect(nodes.some(node => node.id.endsWith('-advance-cursor'))).toBe(true);
  device.destroy();
});
