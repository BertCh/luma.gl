// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, test} from 'vitest';
import {createChangedBindingResolver} from '../../src/geospatial/gpu-point-spatial-query';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

test('createChangedBindingResolver returns bindings only when a resolved buffer changes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'binding-resolver'});
  const first = createTransientView(graph, 'first', 'uint32', 4);
  const second = createTransientView(graph, 'second', 'uint32', 4);
  const buffers = new Map<unknown, Buffer>([
    [first, {id: 'a', byteLength: 1024} as unknown as Buffer],
    [second, {id: 'b', byteLength: 1024} as unknown as Buffer]
  ]);
  const getBuffer = (view: unknown) => buffers.get(view)!;
  const resolve = createChangedBindingResolver({first, second});

  const initial = resolve(getBuffer);
  expect(Object.keys(initial!)).toEqual(['first', 'second']);
  expect(resolve(getBuffer), 'unchanged buffers yield no new binding objects').toBeNull();

  buffers.set(second, {id: 'c', byteLength: 1024} as unknown as Buffer);
  const changed = resolve(getBuffer);
  expect(changed).not.toBeNull();
  expect(changed!['first']).not.toBe(initial!['first']);
  expect(resolve(getBuffer)).toBeNull();
  device.destroy();
});
