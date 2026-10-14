// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUTerrainFlow, GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function buildFlow(width: number, height: number, runoff: boolean) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const cellCount = width * height;
  const grid = makeGPURasterMetadata({width, height});
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length = cellCount
  ) => createTransientView(graph, name, format, length);
  return new GPUTerrainFlow({
    grid,
    elevation: makeGPURasterBand(view('elevation', 'float32'), grid, {id: 'elevation'}),
    parameters: view('settings', 'float32', 8),
    runoff: runoff ? view('runoff', 'float32') : undefined,
    output: {
      accumulation: view('accumulation', 'float32'),
      streams: view('streams', 'uint32'),
      status: {
        accumulationConverged: view('converged', 'uint32', 1),
        accumulationIterationCount: view('rounds', 'uint32', 1)
      }
    }
  }).getCommandNodes(graph);
}

function buildOrder(width: number, height: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const cellCount = width * height;
  const view = (name: string, length = cellCount) =>
    createTransientView(graph, name, 'uint32', length);
  return new GPUTerrainStreamOrder({
    grid: makeGPURasterMetadata({width, height}),
    flowDirections: view('directions'),
    streams: view('streams'),
    output: {
      streamOrder: view('order'),
      status: {converged: view('converged', 1), iterationCount: view('rounds', 1)}
    }
  }).getCommandNodes(graph);
}

// 6 x 5 needs 8 rounds and 2 levels; 40 x 36 needs 16 rounds and 10 levels.
const SIZES = [
  [6, 5],
  [40, 36]
] as const;

it('D8 cells accumulation compiles the same pipelines at grid sizes with different round counts (rule 12)', () => {
  const counts = SIZES.map(([width, height]) => {
    const nodes = buildFlow(width, height, false);
    const tourNodes = nodes.filter(node => /-accumulate-/.test(node.id));
    return {
      nodeCount: tourNodes.length,
      keys: getGPUCommandNodePipelineKeys(tourNodes).size,
      roundKeys: getGPUCommandNodePipelineKeys(nodes, {idPattern: /-tour-round-\d+$/}).size
    };
  });
  expect(counts[1].nodeCount).toBeGreaterThan(counts[0].nodeCount);
  expect(counts[1].keys).toBe(counts[0].keys);
  // Even and odd rounds read opposite buffers but share one pipeline.
  expect(counts[0].roundKeys).toBe(1);
  expect(counts[1].roundKeys).toBe(1);
  // link, reset, slot table, round, group gate, count, finalize.
  expect(counts[0].keys).toBeLessThanOrEqual(8);
});

it('the tour path ignores the iteration limit and the pull path (runoff) stays', () => {
  const tour = buildFlow(6, 5, false).map(node => node.id);
  const pull = buildFlow(6, 5, true).map(node => node.id);
  expect(tour.some(id => /-accumulate-tour-link$/.test(id))).toBe(true);
  expect(tour.some(id => /-accumulate-count$/.test(id))).toBe(true);
  expect(tour.some(id => /-accumulate-round-\d+$/.test(id))).toBe(false);
  expect(pull.some(id => /-accumulate-round-\d+$/.test(id))).toBe(true);
  expect(pull.some(id => /-tour-/.test(id))).toBe(false);
});

it('stream order compiles the same pipelines whatever the grid size (rules 12 and 13)', () => {
  const counts = SIZES.map(([width, height]) => {
    // The scans (GPUScan) add block levels with the problem size; they are not loop-indexed.
    const nodes = buildOrder(width, height).filter(node => !/-scan-/.test(node.id));
    return {
      nodeCount: nodes.length,
      keys: getGPUCommandNodePipelineKeys(nodes).size,
      levelKeys: getGPUCommandNodePipelineKeys(nodes, {idPattern: /-order-level-\d+-scatter$/}).size
    };
  });
  expect(counts[1].nodeCount).toBeGreaterThan(counts[0].nodeCount);
  expect(counts[1].keys).toBe(counts[0].keys);
  expect(counts[0].levelKeys).toBe(1);
  expect(counts[0].keys).toBeLessThanOrEqual(40);
});

it('every tour kernel parses as WGSL within the core binding limit', () => {
  for (const nodes of [buildFlow(6, 5, false), buildOrder(6, 5), buildOrder(40, 36)]) {
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(5);
    for (const [id, source] of sources) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      // A loop or round index never reaches the WGSL text.
      expect(source, id).not.toMatch(/-round-\d+|level-\d+/);
    }
  }
});

it('gated level kernels carry the indirect GPU condition and the scans do not', () => {
  const nodes = buildOrder(6, 5) as readonly {
    id: string;
    condition?: {source?: string; mode?: string};
  }[];
  for (const node of nodes.filter(node => /-order-level-\d+-(scatter|gather)$/.test(node.id))) {
    expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
  for (const node of nodes.filter(node => /-order-unknown-/.test(node.id))) {
    expect(node.condition, node.id).toBeUndefined();
  }
  expect(getGPUCommandNodeKernelSources(nodes).length).toBeGreaterThan(0);
});
