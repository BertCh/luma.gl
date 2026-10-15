// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GraphVectorView, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCrossfilter,
  type GPUCrossfilterExternalDimension
} from '@luma.gl/experimental/gpu-crossfilter';
import {expect, test, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'external'});
  const input = createTransientView(graph, 'input', 'float32', 4);
  const maskBuffer = device.createBuffer({
    id: 'flags',
    data: new Uint32Array([0, 2, 0, 3]),
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const handle = graph.importBuffer(
    {id: 'flags', byteLength: maskBuffer.byteLength, usage: maskBuffer.usage},
    maskBuffer
  );
  const mask = graph.createDataView(handle, {format: 'uint32', length: 4});
  const external: GPUCrossfilterExternalDimension = {id: 'external', kind: 'external', mask};
  return {device, graph, input, mask, maskBuffer, external};
}

test('external dimensions borrow flags, allocate no controls, and validate their API', () => {
  const fixture = createFixture();
  const {device, graph, input, mask, maskBuffer, external} = fixture;
  const createBuffer = vi.spyOn(device, 'createBuffer');
  const writeMask = vi.spyOn(maskBuffer, 'write');
  const controller = new GPUCrossfilter(graph, {dimensions: [external]});
  try {
    expect(createBuffer).not.toHaveBeenCalled();
    expect(controller.getDimensionMask('external')).toBe(mask);
    expect(controller.setExternalActive('external', true)).toBe(controller);
    expect(controller.setExternalActive('external', false)).toBe(controller);
    expect(controller.clear('external').clearAll().invalidate('external')).toBe(controller);
    expect(writeMask).not.toHaveBeenCalled();
    expect(() => controller.setRange('external', [0, 1])).toThrow(/scalar range/);
    expect(() => controller.setBounds('external', [0, 0, 1, 1])).toThrow(/bounds/);
    expect(() => controller.setExternalActive('missing', true)).toThrow(/does not contain/);
    const range = new GPUCrossfilter(graph, {
      id: 'range',
      dimensions: [{id: 'range', kind: 'range', input}]
    });
    expect(() => range.setExternalActive('range', true)).toThrow(/not external/);
    range.destroy();
    controller.destroy();
    expect(maskBuffer.destroyed).toBe(false);
    expect(() => controller.setExternalActive('external', true)).toThrow(/destroyed/);
  } finally {
    controller.destroy();
    createBuffer.mockRestore();
    writeMask.mockRestore();
    maskBuffer.destroy();
    device.destroy();
  }
});

test('external masks reject non-uint32 formats and mismatched row or chunk topology', () => {
  const {device, graph, input, maskBuffer, external} = createFixture();
  try {
    expect(
      () =>
        new GPUCrossfilter(graph, {
          dimensions: [
            // @ts-expect-error Runtime validation rejects incorrectly typed caller data too.
            {...external, mask: input}
          ]
        })
    ).toThrow(/uint32/);
    expect(
      () =>
        new GPUCrossfilter(graph, {
          id: 'short',
          dimensions: [
            {id: 'range', kind: 'range', input},
            {...external, mask: createTransientView(graph, 'short-mask', 'uint32', 3)}
          ]
        })
    ).toThrow(/same row count|same chunk topology/);
    const vector = (id: string, lengths: number[]) =>
      new GraphVectorView({
        id,
        name: id,
        format: 'uint32',
        length: 4,
        valueLength: 4,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: lengths.map((length, index) =>
          createTransientView(graph, `${id}-${index}`, 'uint32', length)
        )
      });
    expect(
      () =>
        new GPUCrossfilter(graph, {
          id: 'chunks',
          dimensions: [
            {id: 'first', kind: 'external', mask: vector('first', [2, 0, 2])},
            {id: 'second', kind: 'external', mask: vector('second', [1, 0, 3])}
          ]
        })
    ).toThrow(/same chunk topology/);
    expect(
      () =>
        new GPUCrossfilter(graph, {
          id: 'view-kind',
          dimensions: [external, {id: 'vector', kind: 'external', mask: vector('vector', [4])}]
        })
    ).toThrow(/same view kind/);
  } finally {
    maskBuffer.destroy();
    device.destroy();
  }
});

test('external activation and invalidation drive composition and leave-one-out skip conditions', () => {
  const {device, graph, input, maskBuffer, external} = createFixture();
  const passes = vi.spyOn(graph, 'addComputePass');
  const controller = new GPUCrossfilter(graph, {
    id: 'filter',
    skipUnchangedDimensions: true,
    dimensions: [{id: 'range', kind: 'range', input}, external],
    views: [
      {
        id: 'range-count',
        kind: 'count',
        dimension: 'range',
        includeOwnSelection: false,
        output: createTransientView(graph, 'range-count', 'uint32', 1)
      },
      {
        id: 'external-count',
        kind: 'count',
        dimension: 'external',
        includeOwnSelection: false,
        output: createTransientView(graph, 'external-count', 'uint32', 1)
      }
    ]
  });
  try {
    controller.addToGraph();
    const nodes = passes.mock.calls.map(([pass]) => pass);
    const encode = () =>
      nodes
        .filter(node => node.condition?.source !== 'cpu' || node.condition.evaluate(undefined))
        .map(node => node.id);
    expect(nodes.some(node => node.id === 'filter-external')).toBe(false);
    expect(
      nodes.some(node =>
        node.resources?.some(
          resource =>
            'buffer' in resource &&
            resource.buffer === external.mask &&
            resource.usage !== 'storage-read'
        )
      )
    ).toBe(false);
    const first = encode();
    expect(first.some(id => id.includes('/inactive'))).toBe(true);
    expect(first.some(id => id.endsWith('/active'))).toBe(false);
    expect(encode()).toEqual([]);
    controller.setExternalActive('external', true);
    const active = encode();
    expect(active.some(id => id.endsWith('/active'))).toBe(true);
    expect(active.some(id => id.includes('/inactive'))).toBe(false);
    expect(active.some(id => id.startsWith('filter/view/11:range-count'))).toBe(true);
    expect(active.some(id => id.startsWith('filter/view/14:external-count'))).toBe(false);
    expect(active).not.toContain('filter-range');
    expect(encode()).toEqual([]);
    controller.invalidate('external');
    expect(encode()).toEqual(active);
    controller.clearAll();
    expect(encode().some(id => id.includes('/inactive'))).toBe(true);
    const sources = getGPUCommandNodeKernelSources(nodes);
    const activeSource = sources.find(source => source.id.endsWith('/external-0/active'))!;
    expect(activeSource.source).toContain('!= 0u');
  } finally {
    controller.destroy();
    passes.mockRestore();
    maskBuffer.destroy();
    device.destroy();
  }
});

test('a sole external cube mask is composed and invalidates the cube build', () => {
  const {device, graph, input, maskBuffer, external} = createFixture();
  const passes = vi.spyOn(graph, 'addComputePass');
  const controller = new GPUCrossfilter(graph, {
    id: 'cube',
    skipUnchangedDimensions: true,
    histogramCubes: true,
    dimensions: [{id: 'range', kind: 'range', input}, external],
    views: [
      {
        id: 'active',
        kind: 'histogram',
        dimension: 'range',
        input,
        domain: [0, 4],
        output: createTransientView(graph, 'active', 'uint32', 2)
      },
      {
        id: 'linked',
        kind: 'histogram',
        input,
        domain: [0, 4],
        output: createTransientView(graph, 'linked', 'uint32', 2)
      }
    ]
  });
  try {
    controller.addToGraph();
    const nodes = passes.mock.calls.map(([pass]) => pass);
    const encode = () =>
      nodes
        .filter(node => node.condition?.source !== 'cpu' || node.condition.evaluate(undefined))
        .map(node => node.id);
    controller.activateView('active').setRange('range', [0, 1]);
    expect(encode().some(id => id.includes('/cube/') && id.includes('/inactive'))).toBe(true);
    expect(encode().some(id => id.endsWith('/clear'))).toBe(false);
    controller.setExternalActive('external', true);
    expect(encode().some(id => id.includes('/cube/') && id.endsWith('/active'))).toBe(true);
    controller.invalidate('external');
    expect(encode().some(id => id.includes('/cube/') && id.endsWith('/clear'))).toBe(true);
  } finally {
    controller.destroy();
    passes.mockRestore();
    maskBuffer.destroy();
    device.destroy();
  }
});
