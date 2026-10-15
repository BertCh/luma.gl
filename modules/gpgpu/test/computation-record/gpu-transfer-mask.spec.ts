// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, addGPUCommandNodes} from '@luma.gl/gpgpu/gpu-core';
import {GPUTransferMask} from '@luma.gl/gpgpu';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

for (const layout of ['index', 'csr'] as const) {
  for (const mode of ['any', 'count'] as const) {
    test(`GPUTransferMask ${layout}-${mode} matches a CPU relation oracle`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) return;
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const byteOffset = 260;
      const makeView = (id: string, values: number[]) => {
        const data = new Uint32Array(byteOffset / 4 + Math.max(values.length, 1));
        data.fill(99);
        data.set(values, byteOffset / 4);
        const buffer = device.createBuffer({
          data,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        const handle = graph.importBuffer(
          {id, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        );
        return graph.createDataView(handle, {format: 'uint32', length: values.length, byteOffset});
      };
      const flags = [1, 0, 7, 1, 1, 1];
      const targets = [2, 1, 2, 0xffffffff, 8, 0];
      const offsets = [0, 2, 3, 5, 6, 6, 8];
      const neighbors = [0, 2, 1, 1, 2, 0xffffffff, 2, 9];
      const source = makeView('source', flags);
      const target = makeView('target', [99, 99, 99, 99]);
      const relation =
        layout === 'index'
          ? {layout, targets: makeView('targets', targets)}
          : {
              layout,
              offsets: makeView('offsets', offsets),
              neighbors: makeView('neighbors', neighbors)
            };
      addGPUCommandNodes(
        graph,
        new GPUTransferMask({id: 'transfer', source, relation, target, mode}).getCommandNodes(graph)
      );
      const compiled = graph.compile();
      try {
        const expected = new Array<number>(target.length).fill(0);
        for (const [sourceIndex, flag] of flags.entries()) {
          if (!flag) continue;
          const reached =
            layout === 'index'
              ? [targets[sourceIndex]]
              : neighbors.slice(offsets[sourceIndex], offsets[sourceIndex + 1]);
          for (const targetIndex of reached) {
            if (targetIndex < expected.length) {
              expected[targetIndex] = mode === 'any' ? 1 : expected[targetIndex] + 1;
            }
          }
        }
        // Repeated runs must clear old counts and leave bytes outside the target view alone.
        for (let execution = 0; execution < 2; execution++) {
          const commandEncoder = device.createCommandEncoder();
          compiled.encode(commandEncoder, {parameters: undefined});
          device.submit(commandEncoder.finish());
          const bytes = await buffers[1].readAsync();
          const values = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
          expect(Array.from(values.slice(byteOffset / 4))).toEqual(expected);
          expect(Array.from(values.slice(0, byteOffset / 4))).toEqual(
            new Array(byteOffset / 4).fill(99)
          );
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}
