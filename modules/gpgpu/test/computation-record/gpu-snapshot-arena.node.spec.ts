// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {describe, expect, test, vi} from 'vitest';
import {
  GPUSnapshotArena,
  type GPUSnapshotSlot
} from '../../src/computation-record/gpu-snapshot-arena';

function createFixture(slots: readonly GPUSnapshotSlot[], capacity = 2) {
  let storage = new Uint8Array(0);
  const readAsync = vi.fn(async () => storage);
  const destroy = vi.fn();
  const createBuffer = vi.fn((props: {byteLength: number}) => {
    storage = new Uint8Array(props.byteLength);
    return {byteLength: props.byteLength, readAsync, destroy};
  });
  const device = {createBuffer} as unknown as Device;
  const arena = new GPUSnapshotArena(device, {id: 'history', slots, capacity});
  const copyBufferToBuffer = vi.fn(
    (props: {sourceOffset: number; destinationOffset: number; size: number}) => {
      storage.set(new Uint8Array(props.size).fill(props.sourceOffset), props.destinationOffset);
    }
  );
  const encoder = {copyBufferToBuffer} as unknown as CommandEncoder;
  const source = {byteLength: 64} as Buffer;
  return {arena, encoder, source, createBuffer, copyBufferToBuffer, readAsync, destroy, device};
}

describe('GPUSnapshotArena', () => {
  test('lays out frames, maps once, and returns independent typed copies and metadata', async () => {
    const fixture = createFixture([
      {name: 'position', byteLength: 8, elementType: 'f32', components: 2},
      {name: 'mask', byteLength: 4, elementType: 'u32'},
      {name: 'signed', byteLength: 4, elementType: 'i32'}
    ]);
    const {arena, encoder, source} = fixture;
    expect(arena.frameByteLength).toBe(16);
    expect(fixture.createBuffer).toHaveBeenCalledWith({
      id: 'history',
      byteLength: 32,
      usage: Buffer.MAP_READ | Buffer.COPY_DST
    });
    const sources = {
      position: {buffer: source},
      mask: {buffer: source, byteOffset: 4},
      signed: {buffer: source}
    };
    expect(
      arena.encodeSnapshot(encoder, sources, {iteration: 7, afterNode: 'advance', label: 'ready'})
    ).toBe(0);
    expect(arena.encodeSnapshot(encoder, sources)).toBe(1);
    expect(fixture.copyBufferToBuffer.mock.calls.map(([copy]) => copy.destinationOffset)).toEqual([
      0, 8, 12, 16, 24, 28
    ]);
    const captures = await arena.read({
      graph: 'compiled',
      resources: {mask: 'mask-buffer'},
      keySpaces: {mask: 'rows'}
    });
    expect(captures).toHaveLength(6);
    expect(captures[0].values).toBeInstanceOf(Float32Array);
    expect(captures[1].values).toBeInstanceOf(Uint32Array);
    expect(captures[2].values).toBeInstanceOf(Int32Array);
    expect(captures[1]).toMatchObject({
      id: 'history-mask-0',
      name: 'mask',
      iteration: 7,
      afterNode: 'advance',
      meaning: 'ready',
      graph: 'compiled',
      resource: 'mask-buffer',
      keySpace: 'rows',
      provenance: 'gpu-readback'
    });
    expect(captures[3].iteration).toBe(1);
    expect(Object.values(captures[3])).not.toContain(undefined);
    (captures[1].values as Uint32Array)[0] = 0;
    const repeated = await arena.read();
    expect(repeated[1].values[0]).toBe(0x04040404);
    expect(fixture.readAsync).toHaveBeenCalledTimes(1);
    expect(() => arena.encodeSnapshot(encoder, sources)).toThrow('history');
    arena.destroy();
    arena.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
    expect(repeated[1].values[0]).toBe(0x04040404);
    await expect(arena.read()).rejects.toThrow('history');
  });

  test('rejects missing, misaligned and undersized sources before copying; rejects full arenas', () => {
    const {arena, encoder, source, copyBufferToBuffer} = createFixture(
      [
        {name: 'first', byteLength: 4, elementType: 'u32'},
        {name: 'second', byteLength: 4, elementType: 'u32'}
      ],
      1
    );
    expect(() => arena.encodeSnapshot(encoder, {first: {buffer: source}})).toThrow(
      'history missing source'
    );
    expect(() =>
      arena.encodeSnapshot(encoder, {
        first: {buffer: source},
        second: {buffer: source, byteOffset: 2}
      })
    ).toThrow('multiple of 4');
    expect(() =>
      arena.encodeSnapshot(encoder, {
        first: {buffer: source},
        second: {buffer: source, byteOffset: 64}
      })
    ).toThrow('too small');
    expect(copyBufferToBuffer).not.toHaveBeenCalled();
    expect(arena.count).toBe(0);
    const sources = {first: {buffer: source}, second: {buffer: source}};
    arena.encodeSnapshot(encoder, sources);
    expect(() => arena.encodeSnapshot(encoder, sources)).toThrow('history snapshot arena is full');
    arena.destroy();
  });

  test('validates slot alignment, capacity, component counts and the byte budget before allocation', () => {
    const createBuffer = vi.fn();
    const device = {createBuffer} as unknown as Device;
    const slot: GPUSnapshotSlot = {name: 'value', byteLength: 4, elementType: 'u32'};
    const make = (overrides: Partial<ConstructorParameters<typeof GPUSnapshotArena>[1]>) =>
      new GPUSnapshotArena(device, {id: 'bounded', slots: [slot], capacity: 1, ...overrides});
    expect(() => make({slots: [{...slot, byteLength: 3}]})).toThrow('bounded');
    expect(() => make({slots: [{...slot, byteLength: -4}]})).toThrow('bounded');
    expect(() => make({slots: [{...slot, components: 2}]})).toThrow('whole components');
    expect(() => make({slots: [slot, slot]})).toThrow('duplicate slot');
    expect(() => make({capacity: -1})).toThrow('capacity');
    expect(() => make({capacity: 1.5})).toThrow('capacity');
    expect(() => make({maximumByteLength: 3})).toThrow('bounded exceeds maximumByteLength');
    expect(() => make({capacity: 16 * 1024 * 1024 + 1})).toThrow('maximumByteLength');
    expect(createBuffer).not.toHaveBeenCalled();
  });

  test('empty frames and empty executions need no map', async () => {
    const {arena, encoder, source, readAsync} = createFixture([
      {name: 'empty', byteLength: 0, elementType: 'u32'}
    ]);
    arena.encodeSnapshot(encoder, {empty: {buffer: source}});
    expect((await arena.read())[0].values.length).toBe(0);
    expect(readAsync).not.toHaveBeenCalled();
    arena.destroy();
    const empty = createFixture([], 0);
    expect(await empty.arena.read()).toEqual([]);
    expect(empty.readAsync).not.toHaveBeenCalled();
    empty.arena.destroy();
  });
});
