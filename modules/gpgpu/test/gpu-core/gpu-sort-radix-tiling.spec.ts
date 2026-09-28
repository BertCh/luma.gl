// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUSort,
  type GPUSortDirection,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';

/**
 * Multi-element radix tiles must stay stable.
 *
 * A tile addresses its keys in a striped layout so that ranking one slot at a time across all 256
 * threads visits keys in ascending index order. These cases pack heavy duplicates into rows that
 * straddle tile boundaries, which is where a blocked layout would reorder equal keys.
 */
it('GPUSort radix keeps equal keys stable across multi-element tiles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }

  for (const length of [255, 2048, 2049, 5000]) {
    for (const elementsPerThread of [1, 8]) {
      for (const direction of ['ascending', 'descending'] as const) {
        const keys = Uint32Array.from({length}, (_, index) => (index * 7) % 11);
        const values = Uint32Array.from({length}, (_, index) => 1000 + index);
        const result = await runTiledSort(device, keys, values, {
          direction,
          keyBits: 8,
          elementsPerThread
        });
        const expected = getStableSortedPairs(keys, values, direction, 8);

        expect(result.keys, `${length} rows x ${elementsPerThread} keys/thread sorts`).toEqual(
          expected.keys
        );
        expect(
          result.values,
          `${length} rows x ${elementsPerThread} keys/thread preserves input order for ties`
        ).toEqual(expected.values);
      }
    }
  }
});

it('GPUSort radix sorts a full 32-bit key with eight-bit digits', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }

  const length = 4096;
  const keys = Uint32Array.from({length}, (_, index) =>
    index % 97 === 0 ? 0xffffffff : (index * 2654435761) >>> 0
  );
  const values = Uint32Array.from({length}, (_, index) => index);
  const result = await runTiledSort(device, keys, values, {
    direction: 'ascending',
    keyBits: 32,
    digitBits: 8,
    elementsPerThread: 8
  });
  const expected = getStableSortedPairs(keys, values, 'ascending', 32);

  expect(result.keys, 'four eight-bit passes order the full key').toEqual(expected.keys);
  expect(result.values, 'payloads follow their keys').toEqual(expected.values);
});

type TiledSortOptions = {
  direction: GPUSortDirection;
  keyBits: number;
  digitBits?: 4 | 8;
  elementsPerThread?: number;
};

async function runTiledSort(
  device: Device,
  keys: Uint32Array,
  values: Uint32Array,
  options: TiledSortOptions
): Promise<{keys: Uint32Array; values: Uint32Array}> {
  const byteLength = keys.length * Uint32Array.BYTES_PER_ELEMENT;
  const keysBuffer = device.createBuffer({
    id: 'tiled-sort-keys',
    data: keys,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const valuesBuffer = device.createBuffer({
    id: 'tiled-sort-values',
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const outputKeysBuffer = device.createBuffer({
    id: 'tiled-sort-output-keys',
    byteLength,
    usage: Buffer.STORAGE | Buffer.COPY_SRC
  });
  const outputValuesBuffer = device.createBuffer({
    id: 'tiled-sort-output-values',
    byteLength,
    usage: Buffer.STORAGE | Buffer.COPY_SRC
  });

  const graph = new GPUCommandGraph(device, {id: 'tiled-sort'});
  const sort = new GPUSort({
    id: 'tiled-sort',
    keys: importView(graph, 'keys', keysBuffer, keys.length),
    values: importView(graph, 'values', valuesBuffer, values.length),
    outputKeys: importView(graph, 'output-keys', outputKeysBuffer, keys.length),
    outputValues: importView(graph, 'output-values', outputValuesBuffer, values.length),
    algorithm: 'radix',
    ...options
  });
  graph.add(sort);

  const compiled = graph.compile();
  const commandEncoder = device.createCommandEncoder({id: 'tiled-sort-encoder'});
  compiled.encode(commandEncoder, {parameters: undefined});
  device.submit(commandEncoder.finish());

  const [keyBytes, valueBytes] = await Promise.all([
    outputKeysBuffer.readAsync(),
    outputValuesBuffer.readAsync()
  ]);
  const result = {
    keys: new Uint32Array(keyBytes.buffer, keyBytes.byteOffset, keys.length).slice(),
    values: new Uint32Array(valueBytes.buffer, valueBytes.byteOffset, values.length).slice()
  };

  compiled.destroy();
  for (const buffer of [keysBuffer, valuesBuffer, outputKeysBuffer, outputValuesBuffer]) {
    buffer.destroy();
  }
  return result;
}

function importView(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  length: number
): GraphDataView<'uint32'> {
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {format: 'uint32', length});
}

function getStableSortedPairs(
  keys: Uint32Array,
  values: Uint32Array,
  direction: GPUSortDirection,
  keyBits: number
): {keys: Uint32Array; values: Uint32Array} {
  const mask = keyBits === 32 ? 0xffffffff : 2 ** keyBits - 1;
  const order = Array.from(keys, (_, index) => index).sort((left, right) => {
    const leftKey = (keys[left] & mask) >>> 0;
    const rightKey = (keys[right] & mask) >>> 0;
    if (leftKey !== rightKey) {
      return direction === 'descending' ? rightKey - leftKey : leftKey - rightKey;
    }
    return left - right;
  });
  return {
    keys: Uint32Array.from(order, index => keys[index]),
    values: Uint32Array.from(order, index => values[index])
  };
}
