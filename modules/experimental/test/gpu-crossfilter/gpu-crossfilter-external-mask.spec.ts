// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

test('external masks compose nonzero flags and refresh self-excluding views after invalidation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const graph = new GPUCommandGraph(device, {id: 'external-mask'});
  const buffers: Buffer[] = [];
  const importValues = <Format extends 'float32' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return {buffer, view: graph.createDataView(handle, {format, length: values.length})};
  };
  const input = importValues('input', new Float32Array([0, 1, 2, 3, 4, 5]), 'float32');
  const external = importValues('external', new Uint32Array([3, 0, 2, 7, 0, 1]), 'uint32');
  const output = importValues('output', new Uint32Array(6), 'uint32');
  const count = importValues('count', new Uint32Array(1), 'uint32');
  const ownCount = importValues('own-count', new Uint32Array(1), 'uint32');
  const controller = new GPUCrossfilter(graph, {
    id: 'filter',
    skipUnchangedDimensions: true,
    dimensions: [
      {id: 'range', kind: 'range', input: input.view},
      {id: 'external', kind: 'external', mask: external.view}
    ],
    outputMask: output.view,
    views: [
      {
        id: 'range-count',
        kind: 'count',
        dimension: 'range',
        includeOwnSelection: false,
        output: count.view
      },
      {
        id: 'external-count',
        kind: 'count',
        dimension: 'external',
        includeOwnSelection: false,
        output: ownCount.view
      }
    ]
  });
  controller.addToGraph();
  const compiled = graph.compile();
  const read = async (buffer: Buffer) => {
    const bytes = await buffer.readAsync();
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
  };
  const encode = () => {
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
  };
  try {
    controller.setRange('range', [1, 4]);
    encode();
    expect(await read(output.buffer)).toEqual([0, 1, 1, 1, 1, 0]);
    expect(await read(count.buffer)).toEqual([6]);

    controller.setExternalActive('external', true);
    encode();
    expect(await read(output.buffer)).toEqual([0, 0, 1, 1, 0, 0]);
    expect(await read(count.buffer)).toEqual([4]);
    expect(await read(ownCount.buffer)).toEqual([4]);
    encode();
    expect(await read(output.buffer)).toEqual([0, 0, 1, 1, 0, 0]);

    external.buffer.write(new Uint32Array([0, 8, 0, 0, 5, 0]));
    controller.invalidate('external');
    encode();
    expect(await read(output.buffer)).toEqual([0, 1, 0, 0, 1, 0]);
    expect(await read(count.buffer)).toEqual([2]);
    expect(await read(ownCount.buffer)).toEqual([4]);

    controller.clear('external');
    encode();
    expect(await read(output.buffer)).toEqual([0, 1, 1, 1, 1, 0]);
    expect(await read(count.buffer)).toEqual([6]);
    expect(await read(external.buffer)).toEqual([0, 8, 0, 0, 5, 0]);
  } finally {
    compiled.destroy();
    controller.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
