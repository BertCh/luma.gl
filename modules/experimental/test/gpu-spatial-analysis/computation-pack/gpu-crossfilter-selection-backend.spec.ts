// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTransferMask} from '@luma.gl/gpgpu';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';
import {GPUCrossfilter} from '../../../src/gpu-crossfilter/gpu-crossfilter';
import {GPUCrossfilterSelectionBackend} from '../../../src/computation-views/coordinator/gpu-crossfilter-selection-backend';

// Written for the main session: never run this browser/GPU spec in an S1 worker.
test('GPU selection uploads external flags, transfers CSR counts, and reads packed nonzero counts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const graph = new GPUCommandGraph(device, {id: 'selection-backend'});
  const buffers: Buffer[] = [];
  function importValues(id: string, values: Uint32Array) {
    const buffer = device!.createBuffer({
      id,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return {buffer, view: graph.createDataView(handle, {format: 'uint32', length: values.length})};
  }
  const external = importValues('external', new Uint32Array(3));
  const selected = importValues('selected', new Uint32Array(3));
  const groups = importValues('groups', new Uint32Array(2));
  const offsets = importValues('offsets', new Uint32Array([0, 2, 3, 4]));
  const neighbors = importValues('neighbors', new Uint32Array([0, 1, 0, 1]));
  const ownCount = importValues('own-count', new Uint32Array(1));
  const crossfilter = new GPUCrossfilter(graph, {
    id: 'spatial-brush',
    dimensions: [{kind: 'external', id: 'brush', mask: external.view}],
    outputMask: selected.view,
    skipUnchangedDimensions: true,
    views: [
      {
        id: 'map',
        kind: 'count',
        dimension: 'brush',
        includeOwnSelection: false,
        output: ownCount.view
      }
    ]
  });
  crossfilter.addToGraph();
  const transfer = new GPUTransferMask({
    id: 'row-to-group',
    source: selected.view,
    target: groups.view,
    relation: {layout: 'csr', offsets: offsets.view, neighbors: neighbors.view},
    mode: 'count'
  });
  const backend = new GPUCrossfilterSelectionBackend({
    crossfilter,
    minimumKeyCount: 0,
    keySpaces: {
      row: {keyCount: 3, dimensions: {map: 'brush'}},
      group: {keyCount: 2, dimensions: {}}
    },
    relations: {groups: {from: 'row', to: 'group', transfer, output: groups.view}}
  });
  const read = async (buffer: Buffer) => {
    const bytes = await buffer.readAsync();
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
  };
  try {
    backend.setClause({
      source: 'map',
      keySpace: 'row',
      label: 'two rows',
      predicate: {kind: 'keys', keys: [0, 2]}
    });
    const source = backend.getMask('row');
    expect(source.buffer).toBe(selected.view);
    expect(await read(selected.buffer)).toEqual([1, 0, 1]);
    expect(backend.transferMask(source, 'groups').buffer).toBe(groups.view);
    expect(await read(groups.buffer)).toEqual([1, 2]);
    expect(await backend.readCounts()).toEqual({row: 2, group: 2});
    expect(backend.getMask('row', {excludeSource: 'map'}).keyCount).toBe(3);
    expect(await read(ownCount.buffer)).toEqual([3]);
    backend.setClause({
      source: 'map',
      keySpace: 'row',
      label: 'one row',
      predicate: {kind: 'keys', keys: [1]}
    });
    expect(await backend.readCounts()).toEqual({row: 1, group: 1});
    expect(await read(groups.buffer)).toEqual([1, 0]);
    backend.clearSource('map');
    expect(await backend.readCounts()).toEqual({row: 3, group: 2});
  } finally {
    backend.destroy();
    crossfilter.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
