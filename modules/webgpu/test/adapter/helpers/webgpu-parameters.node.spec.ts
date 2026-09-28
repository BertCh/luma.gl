// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {applyParametersToRenderPipelineDescriptor} from '../../../src/adapter/helpers/webgpu-parameters';

function makeDescriptor(): GPURenderPipelineDescriptor {
  return {
    layout: 'auto',
    vertex: {module: {} as GPUShaderModule, entryPoint: 'vertexMain'},
    fragment: {module: {} as GPUShaderModule, entryPoint: 'fragmentMain', targets: []}
  };
}

const BLEND_FACTORS = {
  blendColorOperation: 'add',
  blendColorSrcFactor: 'one',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaOperation: 'add',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

it('WebGPU parameters: blend factors enable blending', () => {
  const descriptor = makeDescriptor();
  applyParametersToRenderPipelineDescriptor(descriptor, {...BLEND_FACTORS});
  const target = [...descriptor.fragment!.targets][0]!;
  expect(target.blend, 'every factor lands in the first color target').toEqual({
    color: {operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha'},
    alpha: {operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha'}
  });
});

it('WebGPU parameters: blend: true with factors blends', () => {
  const descriptor = makeDescriptor();
  applyParametersToRenderPipelineDescriptor(descriptor, {blend: true, ...BLEND_FACTORS});
  const target = [...descriptor.fragment!.targets][0]!;
  expect(target.blend?.color.srcFactor, 'explicit blending keeps its factors').toBe('one');
});

it('WebGPU parameters: an explicit blend: false wins over inherited blend factors', () => {
  for (const parameters of [
    {...BLEND_FACTORS, blend: false},
    {blend: false, ...BLEND_FACTORS}
  ]) {
    const descriptor = makeDescriptor();
    applyParametersToRenderPipelineDescriptor(descriptor, parameters);
    const targets = [...descriptor.fragment!.targets];
    expect(targets.length, 'the color target itself is kept').toBe(1);
    expect(targets[0], 'the color target itself is kept').toBeTruthy();
    expect(targets[0]?.blend, 'no blend state, whatever order the parameters arrive in').toBe(
      undefined
    );
  }
});

it('WebGPU parameters: blend: false keeps the rest of the color target', () => {
  const descriptor = makeDescriptor();
  applyParametersToRenderPipelineDescriptor(descriptor, {
    ...BLEND_FACTORS,
    colorMask: 0x7,
    blend: false
  });
  const target = [...descriptor.fragment!.targets][0]!;
  expect(target.writeMask, 'the write mask survives').toBe(0x7);
  expect(target.blend, 'only the blend state is dropped').toBe(undefined);
});

it('WebGPU parameters: blend: false clears blend state already on the target', () => {
  const descriptor = makeDescriptor();
  applyParametersToRenderPipelineDescriptor(descriptor, {...BLEND_FACTORS});
  applyParametersToRenderPipelineDescriptor(descriptor, {blend: false});
  expect([...descriptor.fragment!.targets][0]?.blend, 'a later blend: false wins').toBe(undefined);
});

it('WebGPU parameters: blend: false without a fragment target does not throw', () => {
  const descriptor: GPURenderPipelineDescriptor = {
    layout: 'auto',
    vertex: {module: {} as GPUShaderModule, entryPoint: 'vertexMain'}
  };
  applyParametersToRenderPipelineDescriptor(descriptor, {blend: false});
  expect(descriptor.fragment?.targets?.[0]?.blend, 'and adds no blend state').toBe(undefined);
});
