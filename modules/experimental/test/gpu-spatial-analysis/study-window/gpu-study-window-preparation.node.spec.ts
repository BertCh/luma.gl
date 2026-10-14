// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GPUScan} from '@luma.gl/gpgpu/gpu-core';
import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {getStudyWindowPackedLayout} from '../../../src/utils/study-window-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createFixture<Parameters = void>() {
  const graph = new GPUCommandGraph<Parameters>(createNullWebGPUDevice());
  const geometry = {
    positions: createTransientView(graph, 'positions', 'float32x2', 9),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 3)
  };
  return {graph, geometry};
}

it('GPUStudyWindowPreparation validates context, declared layouts and sampling topology', () => {
  const {graph, geometry} = createFixture();
  expect(
    () =>
      new GPUStudyWindowPreparation({
        ...geometry,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(
    'study-window does not support longitude-latitude coordinates; project positions to a planar frame first'
  );
  expect(
    () =>
      new GPUStudyWindowPreparation({
        ...geometry,
        featureOffsets: createTransientView(graph, 'two-features', 'uint32', 3)
      })
  ).toThrow('study-window study window needs exactly one feature (featureOffsets of length 2)');
  for (const samplingGrid of [
    [1.5, 2],
    [0, 2],
    [1, 1048577]
  ] as const) {
    expect(() => new GPUStudyWindowPreparation({...geometry, samplingGrid})).toThrow(
      'study-window samplingGrid must be two positive integers with a product of at most 1048576'
    );
  }
  for (const invalid of [
    {positions: createTransientView(graph, 'few-vertices', 'float32x2', 2)},
    {ringOffsets: createTransientView(graph, 'no-rings', 'uint32', 1)},
    {polygonOffsets: createTransientView(graph, 'no-polygons', 'uint32', 1)}
  ]) {
    expect(() => new GPUStudyWindowPreparation({...geometry, ...invalid})).toThrow(
      'study-window study window needs at least one ring of 3 or more vertices'
    );
  }
  expect(new GPUStudyWindowPreparation(geometry).sampleGrid).toEqual([256, 256]);
  expect(
    () => new GPUStudyWindowPreparation({...geometry, samplingGrid: [1024, 1024]})
  ).not.toThrow();
  expect(
    () =>
      new GPUStudyWindowPreparation({
        ...geometry,
        output: {
          summary: createTransientView(graph, 'short-summary', 'float32', 7)
        }
      })
  ).toThrow('output.summary must contain 8 float32 values');
});

it('GPUStudyWindowPreparation registers persistent graph-local views and orders eight-binding kernels', () => {
  const {graph, geometry} = createFixture();
  const producer = new GPUStudyWindowPreparation({
    ...geometry,
    samplingGrid: [8, 4],
    output: {
      summary: createTransientView(graph, 'summary-copy', 'float32', 8)
    }
  });
  const message = 'study-window study window needs graph.add before its consumers';
  expect(() => producer.getViews(graph)).toThrow(message);
  const nodes = producer.getCommandNodes(graph);
  const views = producer.getViews(graph);
  expect(() => producer.getViews(new GPUCommandGraph(graph.device))).toThrow(message);
  const ids = nodes.map(node => node.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.slice(0, 7)).toEqual([
    'study-window-ring-terms',
    'study-window-ring-reduce-1',
    'study-window-ring-reduce-2',
    'study-window-ring-reduce-4',
    'study-window-ring-reduce-8',
    'study-window-ring-orientation',
    'study-window-orient-edges'
  ]);
  expect(ids.indexOf('study-window-summarize')).toBeLessThan(
    ids.indexOf('study-window-slab-params')
  );
  expect(ids.indexOf('study-window-slab-edges-fill')).toBeLessThan(
    ids.indexOf('study-window-sample-parameters')
  );
  expect(ids.indexOf('study-window-sample-parameters')).toBeLessThan(
    ids.indexOf('study-window-sample-count')
  );
  expect(ids.indexOf('study-window-sample-compact')).toBeLessThan(
    ids.indexOf('study-window-pack-summary')
  );
  expect(ids.at(-1)).toBe('study-window-copy-summary');
  expect(views.packed.length).toBe(
    getStudyWindowPackedLayout({
      vertexCount: 9,
      slabOffsetCount: views.slabs.slabOffsets.length,
      slabEntryCount: views.slabs.slabEntries.length,
      sampleCellCapacity: 32
    }).length
  );
  for (const source of getParsedKernelSources(nodes).values()) {
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
  }
  expect(() => producer.getCommandNodes(graph)).toThrow('study-window is already added to a graph');
  expect(() => producer.getCommandNodes(new GPUCommandGraph(graph.device))).toThrow(
    'study-window is already added to a graph'
  );
  for (const view of [
    views.edges,
    views.summary,
    views.sampleCells,
    views.sampleCellCount,
    views.packed,
    views.slabs.params,
    views.slabs.slabOffsets,
    views.slabs.slabEntries
  ]) {
    expect(view.buffer.transient).toBe(false);
  }
  producer.destroy();
});

it('GPUStudyWindowPreparation latches rebuild decisions once per encoding and releases owned storage', () => {
  const {graph, geometry} = createFixture<{changed: boolean}>();
  const rebuildWhen = vi.fn((parameters: {changed: boolean}) => parameters.changed);
  const producer = new GPUStudyWindowPreparation({...geometry, rebuildWhen});
  expect(producer.isDeclaredIn(graph)).toBe(false);
  const nodes = producer.getCommandNodes(graph);
  expect(producer.isDeclaredIn(graph)).toBe(true);
  expect(producer.isDeclaredIn(new GPUCommandGraph(graph.device))).toBe(false);
  const encode = (changed: boolean, expected: boolean) => {
    for (const node of nodes) {
      expect(node.condition?.id).toBe('study-window-rebuild');
      expect(node.condition?.source).toBe('cpu');
      if (node.condition?.source === 'cpu') {
        expect(node.condition.evaluate({changed})).toBe(expected);
      }
    }
  };
  expect(producer.encodedBuildCount).toBe(0);
  encode(false, true);
  expect(producer.encodedBuildCount).toBe(1);
  encode(false, false);
  expect(rebuildWhen).toHaveBeenCalledTimes(1);
  producer.invalidate();
  encode(false, true);
  expect(producer.encodedBuildCount).toBe(2);
  encode(false, false);
  encode(true, true);
  expect(producer.encodedBuildCount).toBe(3);
  encode(false, false);
  const views = producer.getViews(graph);
  const owned = [
    views.edges,
    views.summary,
    views.sampleCells,
    views.sampleCellCount,
    views.packed,
    views.slabs.params,
    views.slabs.slabOffsets,
    views.slabs.slabEntries
  ];
  const destroyed = owned.map(view => {
    expect(view.buffer.usage).toBe(Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST);
    const buffer = view.buffer.defaultBuffer;
    expect(buffer).toBeInstanceOf(Buffer);
    return vi.spyOn(buffer as Buffer, 'destroy');
  });
  producer.destroy();
  destroyed.forEach(destroy => expect(destroy).toHaveBeenCalledOnce());
  expect(producer.isDeclaredIn(graph)).toBe(false);
  expect(() => producer.getViews(graph)).toThrow(
    'study-window study window needs graph.add before its consumers'
  );
  // A new graph can import the same geometry buffers; use new graph-local views for this fixture.
  const next = createFixture<{changed: boolean}>();
  Object.assign(producer.props, next.geometry);
  const nextNodes = producer.getCommandNodes(next.graph);
  expect(producer.getViews(next.graph).packed).not.toBe(views.packed);
  for (const node of nextNodes) {
    if (node.condition?.source === 'cpu') {
      expect(node.condition.evaluate({changed: false})).toBe(true);
    }
  }
  expect(producer.encodedBuildCount).toBe(4);
  producer.destroy();
});

it('GPUStudyWindowPreparation combines nested scan conditions without losing the encoding latch', () => {
  const original = GPUScan.prototype.getCommandNodes;
  const nested = vi.fn(() => false);
  const scan = vi.spyOn(GPUScan.prototype, 'getCommandNodes').mockImplementation(function (
    this: GPUScan,
    graph
  ) {
    return original.call(this, graph).map(node => ({
      ...node,
      condition: {id: 'nested-scan', source: 'cpu' as const, evaluate: nested}
    }));
  });
  const {graph, geometry} = createFixture();
  const producer = new GPUStudyWindowPreparation({...geometry, samplingGrid: [8, 4]});
  try {
    const nodes = producer.getCommandNodes(graph);
    for (let encoding = 0; encoding < 2; encoding++) {
      for (const node of nodes) {
        expect(node.condition?.id).toBe('study-window-rebuild');
        if (node.condition?.source === 'cpu') {
          const result = node.condition.evaluate(undefined);
          if (encoding === 1) {
            expect(result).toBe(false);
          }
        }
      }
      expect(producer.encodedBuildCount).toBe(1);
    }
    expect(nested).toHaveBeenCalled();
  } finally {
    scan.mockRestore();
    producer.destroy();
  }
});
