// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUTransferMask} from '@luma.gl/gpgpu';
import {describe, expect, test, vi} from 'vitest';
import type {GPUCrossfilter} from '../../../src/gpu-crossfilter/gpu-crossfilter';
import {GPUCrossfilterSelectionBackend} from '../../../src/computation-views/coordinator/gpu-crossfilter-selection-backend';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function makeFixture(keyCount = 8, useHook = false) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'backend-test'});
  const buffers: Buffer[] = [];
  function importValues(id: string, values: Uint32Array) {
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
    return {buffer, view: graph.createDataView(handle, {format: 'uint32', length: values.length})};
  }
  const input = importValues('input', new Uint32Array(keyCount));
  const output = importValues('output', new Uint32Array(keyCount));
  const compiled = {encode: vi.fn(), destroy: vi.fn()} as unknown as CompiledGPUCommandGraph;
  const compile = vi.spyOn(graph, 'compile').mockReturnValue(compiled);
  const submit = vi.spyOn(device, 'submit').mockImplementation(() => {});
  const crossfilter = {
    id: 'fake-filter',
    graph,
    mask: output.view,
    dimensions: [
      {id: 'external', kind: 'external', mask: input.view},
      {id: 'range', kind: 'range', input: input.view}
    ],
    views: [{id: 'exclude-map', kind: 'mask', dimension: 'external', output: output.view}],
    getDimensionMask: vi.fn(() => input.view),
    getViewMask: vi.fn(() => undefined),
    setExternalActive: vi.fn(),
    invalidate: vi.fn(),
    setRange: vi.fn(),
    clear: vi.fn()
  } as unknown as GPUCrossfilter;
  const writeMask = vi.fn();
  const keySpaces = {
    row: {keyCount, dimensions: {map: 'external', chart: 'range'}, ...(useHook ? {writeMask} : {})}
  };
  const backend = new GPUCrossfilterSelectionBackend({crossfilter, keySpaces, minimumKeyCount: 0});
  return {
    device,
    graph,
    crossfilter,
    backend,
    input,
    output,
    compile,
    submit,
    compiled,
    importValues,
    writeMask,
    cleanup: () => {
      backend.destroy();
      buffers.forEach(buffer => buffer.destroy());
      device.destroy();
    }
  };
}

describe('GPUCrossfilterSelectionBackend', () => {
  test('maps source clauses, uploads u32 flags, excludes own dimension, and advances generations', () => {
    const fixture = makeFixture();
    const {backend, crossfilter, input} = fixture;
    const write = vi.spyOn(input.buffer, 'write');
    try {
      backend.setClause({
        source: 'map',
        keySpace: 'row',
        label: 'rows',
        predicate: {kind: 'keys', keys: [1, 4, 4]}
      });
      expect(write).toHaveBeenCalledWith(new Uint32Array([0, 1, 0, 0, 1, 0, 0, 0]), 0);
      expect(crossfilter.setExternalActive).toHaveBeenCalledWith('external', true);
      expect(crossfilter.invalidate).toHaveBeenCalledWith('external');
      const mask = backend.getMask('row');
      expect(mask.buffer).toBe(fixture.output.view);
      expect(mask.generation).toBe(1);
      expect(backend.getMask('row').generation).toBe(1);
      expect(fixture.submit).toHaveBeenCalledTimes(1);
      expect(backend.getMask('row', {excludeSource: 'map'}).keyCount).toBe(8);
      expect(crossfilter.getViewMask).toHaveBeenCalledWith('exclude-map');
      backend.setClause({
        source: 'chart',
        keySpace: 'row',
        label: 'range',
        predicate: {kind: 'range', field: 'value', range: [2, 6]}
      });
      expect(crossfilter.setRange).toHaveBeenCalledWith('range', [2, 6]);
      expect(crossfilter.setExternalActive).toHaveBeenCalledTimes(1);
      expect(backend.getMask('row').generation).toBe(2);
      backend.clearSource('map');
      expect(crossfilter.setExternalActive).toHaveBeenLastCalledWith('external', false);
      backend.clearSource('chart');
      expect(crossfilter.clear).toHaveBeenCalledWith('range');
      expect(backend.getMask('row').generation).toBe(4);
      expect(fixture.compile).toHaveBeenCalledTimes(1);
    } finally {
      fixture.cleanup();
    }
  });

  test('delegates writes, enforces the default budget and reports unsupported requests', () => {
    const fixture = makeFixture(8, true);
    const {backend, crossfilter} = fixture;
    try {
      const defaultBackend = new GPUCrossfilterSelectionBackend({
        crossfilter,
        keySpaces: {
          large: {keyCount: 65537, dimensions: {}},
          small: {keyCount: 65536, dimensions: {}}
        }
      });
      expect(defaultBackend.accepts('large', 65537)).toBe(true);
      expect(defaultBackend.accepts('large', 65538)).toBe(false);
      expect(defaultBackend.accepts('small', 65536)).toBe(false);
      expect(defaultBackend.accepts('missing', 65537)).toBe(false);
      expect(defaultBackend.accepts('large', Infinity)).toBe(false);
      backend.setClause({
        source: 'map',
        keySpace: 'row',
        label: 'one',
        predicate: {kind: 'keys', keys: [2]}
      });
      expect(fixture.writeMask).toHaveBeenCalledWith('row', [2], 'map');
      expect(() =>
        backend.setClause({source: 'map', keySpace: 'row', label: 'all', predicate: {kind: 'all'}})
      ).toThrow(/Unsupported all/);
      expect(() =>
        backend.setClause({
          source: 'chart',
          keySpace: 'row',
          label: 'keys',
          predicate: {kind: 'keys', keys: [2]}
        })
      ).toThrow(/Unsupported keys/);
      expect(() =>
        backend.setClause({
          source: 'missing',
          keySpace: 'row',
          label: 'keys',
          predicate: {kind: 'keys', keys: [2]}
        })
      ).toThrow(/No crossfilter dimension/);
      expect(() =>
        backend.setClause({
          source: 'map',
          keySpace: 'row',
          label: 'bad key',
          predicate: {kind: 'keys', keys: [8]}
        })
      ).toThrow(/outside/);
      expect(() => backend.getMask('missing')).toThrow(/Unknown selection key space/);
      expect(() => backend.getMask('row', {excludeSource: 'missing'})).toThrow(/No self-excluding/);
      backend.destroy();
      expect(() => backend.getMask('row')).toThrow(/destroyed/);
      expect(fixture.input.buffer.destroyed).toBe(false);
    } finally {
      fixture.cleanup();
    }
  });

  test('submits a fixed CSR transfer, returns its output and performs one packed counts readback', async () => {
    const fixture = makeFixture(3);
    const target = fixture.importValues('target', new Uint32Array(2));
    const offsets = fixture.importValues('offsets', new Uint32Array([0, 1, 2, 3]));
    const neighbors = fixture.importValues('neighbors', new Uint32Array([0, 1, 0]));
    const transfer = new GPUTransferMask({
      id: 'transfer',
      source: fixture.output.view,
      target: target.view,
      relation: {layout: 'csr', offsets: offsets.view, neighbors: neighbors.view},
      mode: 'count'
    });
    const backend = new GPUCrossfilterSelectionBackend({
      crossfilter: fixture.crossfilter,
      minimumKeyCount: 0,
      keySpaces: {
        row: {keyCount: 3, dimensions: {map: 'external'}},
        group: {keyCount: 2, dimensions: {}}
      },
      relations: {groups: {from: 'row', to: 'group', transfer, output: target.view}}
    });
    const createBuffer = vi.spyOn(fixture.device, 'createBuffer');
    try {
      const mask = backend.getMask('row');
      const transferred = backend.transferMask(mask, 'groups');
      expect(transferred.buffer).toBe(target.view);
      expect(transferred.generation).toBe(mask.generation + 1);
      expect(() => backend.transferMask(mask, 'missing')).toThrow(/Unknown or incompatible/);
      expect(() => backend.transferMask({...mask, buffer: target.view}, 'groups')).toThrow(
        /declared source mask/
      );
      const countBuffer = createBuffer.mock.results
        .map(result => result.value as Buffer)
        .find(buffer => buffer.id.endsWith('-counts'))!;
      const read = vi
        .spyOn(countBuffer, 'readAsync')
        .mockResolvedValue(new Uint8Array(new Uint32Array([2, 1]).buffer));
      expect(await backend.readCounts()).toEqual({row: 2, group: 1});
      expect(read).toHaveBeenCalledTimes(1);
    } finally {
      backend.destroy();
      fixture.cleanup();
    }
  });
});
