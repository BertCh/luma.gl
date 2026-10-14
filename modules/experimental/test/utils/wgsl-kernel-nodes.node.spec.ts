// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import type {Buffer, Device, TextureView} from '@luma.gl/core';
import {expect, test} from 'vitest';
import {
  createChangedBindingResolver,
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize
} from '../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';

test('createWGSLKernelNode accepts an omitted workload variant', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-variant'});
  const output = createTransientView(graph, 'output', 'uint32', 4);

  const node = createWGSLKernelNode(graph, {
    id: 'no-variant',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
    invocationCount: 4,
    body: 'output[outputOffset + index] = index;'
  });

  expect(node.workload && 'variant' in node.workload, 'no undefined variant key').toBe(false);
  expect(() => graph.add(node)).not.toThrow();
  device.destroy();
});

test('generated guarded kernels scale their default workgroup size', () => {
  expect(getAutomaticWGSLKernelWorkgroupSize(0)).toBe(32);
  expect(getAutomaticWGSLKernelWorkgroupSize(1)).toBe(32);
  expect(getAutomaticWGSLKernelWorkgroupSize(33)).toBe(64);
  expect(getAutomaticWGSLKernelWorkgroupSize(129)).toBe(256);
  expect(getAutomaticWGSLKernelWorkgroupSize(4096)).toBe(256);
  expect(() => getAutomaticWGSLKernelWorkgroupSize(-1)).toThrow(/non-negative safe integer/);
});

function createTextureKernelFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-texture'});
  const output = createTransientView(graph, 'output', 'uint32', 4);
  const source = graph.createTransientTexture({
    id: 'source-texture',
    format: 'rgba8unorm',
    width: 2,
    height: 2,
    usage: 0x4 /* TEXTURE_BINDING */
  } as never);
  const sourceView = graph.createTextureView(source);
  const props = {
    id: 'textured',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'output', view: output, type: 'u32' as const, access: 'read_write' as const}],
    invocationCount: 4,
    body: 'output[outputOffset + index] = index;'
  };
  return {device, graph, output, sourceView, props};
}

test('createWGSLKernelNode declares textures after storage bindings', () => {
  const {device, graph, sourceView, props} = createTextureKernelFixture();
  const plainNode = createWGSLKernelNode(graph, props);
  const texturedNode = createWGSLKernelNode(graph, {
    ...props,
    id: 'textured-2',
    textures: [{name: 'sourceTexture', texture: sourceView, sampleType: 'float'}]
  });
  const [plain] = getGPUCommandNodeKernelSources([plainNode]);
  const [textured] = getGPUCommandNodeKernelSources([texturedNode]);

  expect(plain!.source).not.toContain('texture_');
  expect(textured!.source).toContain('@group(0) @binding(1) var sourceTexture: texture_2d<f32>;');
  expect(textured!.source).toContain('_ = sourceTexture;');
  expect(textured!.source).toContain('_ = &output;');
  // Everything outside the texture additions is unchanged.
  expect(
    textured!.source
      .replace('\n@group(0) @binding(1) var sourceTexture: texture_2d<f32>;', '')
      .replace('\n  _ = sourceTexture;', '')
  ).toBe(plain!.source);
  expect(texturedNode.resources).toContainEqual({texture: sourceView, usage: 'sampled'});
  expect(plainNode.resources?.some(resource => 'texture' in resource)).toBe(false);
  device.destroy();
});

test('createWGSLKernelNode rejects more sampled textures than the device allows', () => {
  const {device, graph, sourceView, props} = createTextureKernelFixture();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxSampledTexturesPerShaderStage: 0}
  });
  expect(() =>
    createWGSLKernelNode(graph, {
      ...props,
      textures: [{name: 'sourceTexture', texture: sourceView, sampleType: 'float'}]
    })
  ).toThrow(/sampled textures/);
  device.destroy();
});

test('createWGSLKernelNode reuses the bind group until a resolved resource changes', () => {
  const {device, graph, output, sourceView, props} = createTextureKernelFixture();
  const node = createWGSLKernelNode(graph, {
    ...props,
    textures: [{name: 'sourceTexture', texture: sourceView, sampleType: 'float'}]
  });
  const cacheKeys: object[] = [];
  const bindingObjects: Record<string, unknown>[] = [];
  const fakeDevice = Object.create(device) as Device;
  Object.defineProperty(fakeDevice, 'createComputePipeline', {
    value: (pipelineProps: {id?: string}) => ({
      id: pipelineProps.id ?? '',
      hash: '',
      destroy: () => {}
    })
  });
  Object.defineProperty(fakeDevice, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const executable = node.compile!({device: fakeDevice}) as unknown as {
    encode: (context: unknown) => void;
  };
  let buffer = {id: 'buffer-a', byteLength: 1024} as unknown as Buffer;
  let textureView = {id: 'view-a'};
  const computePass = {
    setPipeline: () => {},
    setBindings: (
      bindings: Record<string, unknown>,
      options?: {_bindGroupCacheKeys?: Record<number, object>}
    ) => {
      bindingObjects.push(bindings);
      cacheKeys.push(options!._bindGroupCacheKeys![0]!);
    },
    dispatch: () => {}
  };
  const encode = () =>
    executable.encode({
      computePass,
      getBuffer: (view: unknown) => (view === output ? buffer : undefined),
      getTextureView: () => textureView
    });

  encode();
  encode();
  expect(cacheKeys[1], 'unchanged resources keep the cache key').toBe(cacheKeys[0]);
  expect(bindingObjects[1]).toBe(bindingObjects[0]);
  expect(bindingObjects[0]!['sourceTexture']).toBe(textureView);

  textureView = {id: 'view-b'};
  encode();
  expect(cacheKeys[2], 'a new texture view invalidates the bind group').not.toBe(cacheKeys[1]);
  encode();
  expect(cacheKeys[3]).toBe(cacheKeys[2]);

  buffer = {id: 'buffer-b', byteLength: 1024} as unknown as Buffer;
  encode();
  expect(cacheKeys[4], 'a new buffer invalidates the bind group').not.toBe(cacheKeys[3]);
  device.destroy();
});

test('createChangedBindingResolver returns records only when a buffer or texture view changes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'changed-binding-resolver'});
  const values = createTransientView(graph, 'values', 'uint32', 4);
  const texture = {id: 'texture'} as unknown as NonNullable<
    Parameters<typeof createChangedBindingResolver>[1]
  >[string];
  let buffer = {id: 'a', byteLength: 1024} as unknown as Buffer;
  let textureView = {id: 'view-a'} as unknown as TextureView;
  const getBuffer = () => buffer;
  const getTextureView = () => textureView;
  const resolve = createChangedBindingResolver({values}, {sampled: texture});

  const initial = resolve(getBuffer, getTextureView);
  expect(Object.keys(initial!)).toEqual(['values', 'sampled']);
  expect(initial!['sampled']).toBe(textureView);
  expect(resolve(getBuffer, getTextureView), 'unchanged resources yield null').toBeNull();

  textureView = {id: 'view-b'} as unknown as TextureView;
  const afterTexture = resolve(getBuffer, getTextureView);
  expect(afterTexture!['sampled'], 'a new texture view yields a new record').toBe(textureView);
  expect(afterTexture!['values']).not.toBe(initial!['values']);
  expect(resolve(getBuffer, getTextureView)).toBeNull();

  buffer = {id: 'b', byteLength: 1024} as unknown as Buffer;
  expect(resolve(getBuffer, getTextureView), 'a new buffer yields a new record').not.toBeNull();
  expect(resolve(getBuffer, getTextureView)).toBeNull();

  expect(createChangedBindingResolver({})(getBuffer), 'first call always resolves').toEqual({});
  device.destroy();
});

test('createWGSLKernelNode rejects a binding above maxStorageBufferBindingSize at declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-binding-size'});
  // A 6000 x 6000 float32 raster is 144 MB, above the 128 MiB default binding limit.
  const cellCount = 6000 * 6000;
  const large = createTransientView(graph, 'large', 'float32', cellCount);
  const props = {
    id: 'large-raster',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'values', view: large, type: 'f32' as const, access: 'read_write' as const}],
    invocationCount: cellCount,
    body: 'values[valuesOffset + index] = 0.0;'
  };
  expect(() => createWGSLKernelNode(graph, props)).toThrow(
    /kernel large-raster binding values needs 144000000 bytes, above the device limit of 134217728/
  );
  // 128 MiB exactly fits.
  const limit = createTransientView(graph, 'limit', 'float32', 134217728 / 4);
  expect(() =>
    createWGSLKernelNode(graph, {
      ...props,
      bindings: [{name: 'values', view: limit, type: 'f32', access: 'read_write'}]
    })
  ).not.toThrow();
  device.destroy();
});
