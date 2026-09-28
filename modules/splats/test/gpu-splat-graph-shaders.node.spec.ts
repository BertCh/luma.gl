// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPU_SPLAT_COMPATIBLE_RENDER_SHADER,
  GPU_SPLAT_COMPATIBLE_RENDER_SHADER_LAYOUT,
  GPU_SPLAT_FEATURE_FLAGS,
  GPU_SPLAT_FEATURE_SHADER,
  GPU_SPLAT_FEATURE_SHADER_LAYOUT,
  GPU_SPLAT_GATHER_SHADER,
  GPU_SPLAT_GRAPH_FEATURE_UNIFORM_BYTE_LENGTH,
  GPU_SPLAT_GRAPH_UNIFORM_BYTE_LENGTH,
  GPU_SPLAT_PROJECTION_SHADER,
  GPU_SPLAT_PROJECTION_SHADER_LAYOUT,
  GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH,
  GPU_SPLAT_RENDER_SHADER,
  GPU_SPLAT_RENDER_SHADER_LAYOUT
} from '../src/gpu-splat-graph-shaders';
import {
  GPU_PAGED_SPLAT_FEATURE_SHADER,
  GPU_PAGED_SPLAT_PROJECTION_SHADER,
  GPU_PAGED_SPLAT_RENDER_SHADER
} from '../src/gpu-paged-splat-shaders';
import {
  GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER,
  GPU_SPLAT_GRAPH_PICKING_SHADER
} from '../src/gpu-splat-graph-interaction';
import {getSplatMaximumDepthKey, packSplatDepthKey} from '../src/splat-depth-key';

const ALL_SPLAT_SHADERS = {
  projection: GPU_SPLAT_PROJECTION_SHADER,
  feature: GPU_SPLAT_FEATURE_SHADER,
  render: GPU_SPLAT_RENDER_SHADER,
  compatibleRender: GPU_SPLAT_COMPATIBLE_RENDER_SHADER,
  gather: GPU_SPLAT_GATHER_SHADER,
  picking: GPU_SPLAT_GRAPH_PICKING_SHADER,
  compatiblePicking: GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER,
  pagedProjection: GPU_PAGED_SPLAT_PROJECTION_SHADER,
  pagedFeature: GPU_PAGED_SPLAT_FEATURE_SHADER,
  pagedRender: GPU_PAGED_SPLAT_RENDER_SHADER
} as const;

it('every Gaussian shader parses and agrees on the packed record and uniform block', () => {
  for (const [name, source] of Object.entries(ALL_SPLAT_SHADERS)) {
    const reflect = new WgslReflect(source);
    const projectedRecord = reflect.structs.find(struct => struct.name === 'ProjectedSplat');
    expect(projectedRecord?.size, `${name}: the packed record is 32 bytes`).toBe(
      GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH
    );
    if (name === 'gather') {
      continue;
    }
    const graphUniforms = reflect.structs.find(struct => struct.name === 'GraphSplatUniforms');
    expect(graphUniforms?.size, `${name}: the uniform block is 160 bytes`).toBe(
      GPU_SPLAT_GRAPH_UNIFORM_BYTE_LENGTH
    );
  }
});

it('the packed projected record keeps the exact offsets the uniform writer assumes', () => {
  const projection = new WgslReflect(GPU_SPLAT_PROJECTION_SHADER);
  const projectedRecord = projection.structs.find(struct => struct.name === 'ProjectedSplat');

  expect(
    projectedRecord?.members.map(member => ({name: member.name, offset: member.offset})),
    'a single-precision clip center, half-precision axes, and half-precision HDR color'
  ).toEqual([
    {name: 'clipCenter', offset: 0},
    {name: 'packedAxis0', offset: 16},
    {name: 'packedAxis1', offset: 20},
    {name: 'packedColorRG', offset: 24},
    {name: 'packedColorBA', offset: 28}
  ]);
});

it('the graph uniform block keeps the exact offsets the uniform writer assumes', () => {
  const projection = new WgslReflect(GPU_SPLAT_PROJECTION_SHADER);
  const graphUniforms = projection.uniforms.find(uniform => uniform.name === 'graphUniforms');

  // Every offset here is written by hand in `writeBatchUniforms`, so a change on either side that
  // is not mirrored on the other silently corrupts the camera rather than failing to compile.
  expect(
    graphUniforms?.members?.map(member => ({name: member.name, offset: member.offset})),
    'camera, styling, filtering, depth-key and per-level fields stay at their written offsets'
  ).toEqual([
    {name: 'modelViewProjectionMatrix', offset: 0},
    {name: 'viewportSize', offset: 64},
    {name: 'depthRange', offset: 72},
    {name: 'radiusScale', offset: 80},
    {name: 'alphaScale', offset: 84},
    {name: 'alphaCutoff', offset: 88},
    {name: 'screenSizeCutoffPixels', offset: 92},
    {name: 'gaussianSupportRadius', offset: 96},
    {name: 'screenFilterVariance', offset: 100},
    {name: 'levelFilterVariance', offset: 104},
    {name: 'levelFadeOpacity', offset: 108},
    {name: 'maxScreenSpaceSplatSize', offset: 112},
    {name: 'exposure', offset: 116},
    {name: 'toneMapping', offset: 120},
    {name: 'featureFlags', offset: 124},
    {name: 'depthKeyMode', offset: 128},
    {name: 'maximumDepthKey', offset: 132},
    {name: 'batchOffset', offset: 136},
    {name: 'rowCount', offset: 140},
    {name: 'frameIndex', offset: 144},
    {name: 'pickingAlphaThreshold', offset: 148},
    {name: 'hasActiveRows', offset: 152},
    {name: 'sourceRowOffset', offset: 156}
  ]);
});

it('GPU Gaussian projection stays within guaranteed WebGPU storage binding limits', () => {
  const projection = new WgslReflect(GPU_SPLAT_PROJECTION_SHADER);
  expect(projection.storage.length, 'projection consumes exactly eight storage buffers').toBe(8);
  expect(
    projection.storage.map(resource => ({name: resource.name, location: resource.binding})),
    'source columns and renderer-owned derived outputs match declared binding order'
  ).toEqual(
    GPU_SPLAT_PROJECTION_SHADER_LAYOUT.bindings
      .filter(binding => binding.type !== 'uniform')
      .map(binding => ({name: binding.name, location: binding.location}))
  );
  expect(
    projection.uniforms.map(resource => ({name: resource.name, location: resource.binding})),
    'the clip region rides in a second uniform buffer, not a ninth storage binding'
  ).toEqual([
    {name: 'graphUniforms', location: 8},
    {name: 'clipUniforms', location: 9}
  ]);
  expect(
    projection.storage.find(resource => resource.name === 'projectedRecords')?.stride,
    'camera-projected records have a tightly packed 32-byte storage stride'
  ).toBe(GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH);

  const render = new WgslReflect(GPU_SPLAT_RENDER_SHADER);
  expect(
    [
      ...render.uniforms.map(resource => ({name: resource.name, location: resource.binding})),
      ...render.storage.map(resource => ({name: resource.name, location: resource.binding}))
    ],
    'one draw consumes only camera uniforms, projected records, and globally sorted IDs'
  ).toEqual(
    GPU_SPLAT_RENDER_SHADER_LAYOUT.bindings.map(binding => ({
      name: binding.name,
      location: binding.location
    }))
  );
});

it('the compatibility render path binds no storage buffers in the vertex stage', () => {
  const compatible = new WgslReflect(GPU_SPLAT_COMPATIBLE_RENDER_SHADER);

  expect(
    compatible.storage.length,
    'WebGPU compatibility mode reports zero vertex-stage storage buffers'
  ).toBe(0);
  expect(
    compatible.uniforms.map(resource => ({name: resource.name, location: resource.binding})),
    'only the camera uniform block remains bound'
  ).toEqual(
    GPU_SPLAT_COMPATIBLE_RENDER_SHADER_LAYOUT.bindings.map(binding => ({
      name: binding.name,
      location: binding.location
    }))
  );
  expect(
    compatible.entry.vertex[0]?.inputs?.map(input => input.name).sort(),
    'the sorted records arrive as an instance stream instead'
  ).toEqual(['instanceClipCenter', 'instancePackedRecord', 'vertexIndex']);
  expect(
    GPU_SPLAT_GATHER_SHADER,
    'and a gather pass materializes that stream from the global sort'
  ).toMatch(/sortedRecords\[sortedIndex\]\s*=\s*projectedRecords\[sortedIds\[sortedIndex\]\]/);
});

it('GPU Gaussian enhancement evaluates harmonics and semantic filters within WebGPU limits', () => {
  const feature = new WgslReflect(GPU_SPLAT_FEATURE_SHADER);
  expect(feature.storage.length, 'directional radiance and semantics need only seven buffers').toBe(
    7
  );
  expect(
    feature.storage.map(resource => ({name: resource.name, location: resource.binding})),
    'borrows separate source-owned coefficient and semantic buffers without packing source rows'
  ).toEqual(
    GPU_SPLAT_FEATURE_SHADER_LAYOUT.bindings
      .filter(binding => binding.type !== 'uniform')
      .map(binding => ({name: binding.name, location: binding.location}))
  );
  const featureUniforms = feature.uniforms.find(uniform => uniform.name === 'featureUniforms');
  expect(
    featureUniforms?.size,
    'keeps optional view-dependent feature controls in one compact 48-byte uniform'
  ).toBe(GPU_SPLAT_GRAPH_FEATURE_UNIFORM_BYTE_LENGTH);
  expect(
    GPU_SPLAT_FEATURE_SHADER,
    'removes filtered rows from the existing GPU-owned indirect draw count'
  ).toMatch(/atomicSub\(&drawCommands\[1u\],\s*1u\)/);
  expect(
    GPU_SPLAT_FEATURE_SHADER,
    'evaluates every Khronos/GraphDECO spherical-harmonic basis through degree three'
  ).toMatch(/case 14u:[\s\S]*?-0\.5900435899266435/);
  expect(
    GPU_SPLAT_FEATURE_SHADER,
    'spherical harmonics change chroma only; the stored opacity already sized the quad'
  ).toMatch(
    /evaluateGraphSphericalHarmonics\(projectedColor\.rgb, batchRowIndex\)[\s\S]*?projectedColor\.a/
  );
});

it('GPU Gaussian projection compensates dilation and publishes a distributed depth key', () => {
  expect(
    GPU_SPLAT_PROJECTION_SHADER,
    'the dilated covariance is compensated against its pre-dilation determinant'
  ).toMatch(
    /getSplatDilationCompensation\(\s*baseCovariance00,\s*covariance01,\s*baseCovariance11,\s*addedVariance\s*\)/
  );
  expect(
    GPU_SPLAT_PROJECTION_SHADER,
    'per-level refiltering shares the screen-space filter and its compensation'
  ).toMatch(/graphUniforms\.screenFilterVariance \+ graphUniforms\.levelFilterVariance/);
  expect(
    GPU_SPLAT_PROJECTION_SHADER,
    'the size clamp no longer rescales the covariance without compensating for it'
  ).toMatch(/getSplatClampCompensation\(clampScale\)/);
  expect(
    GPU_SPLAT_PROJECTION_SHADER,
    'depth keys come from a selectable distribution rather than hyperbolic device depth'
  ).toMatch(/depthKeys\[projectedRowIndex\] = packSplatDepthKey\(/);
  expect(
    GPU_SPLAT_PROJECTION_SHADER,
    'visible rows increment the indirect draw instance-count word'
  ).toMatch(/atomicAdd\(&drawCommands\[1u\],\s*1u\)/);
});

it('back-to-front keys stay ordered and never collide with the culled sentinel', () => {
  const maximumKey = getSplatMaximumDepthKey(16);
  const near = packSplatDepthKey(1, {mode: 'float16'});
  const far = packSplatDepthKey(1000, {mode: 'float16'});

  expect(far < near, 'ascending keys render far Gaussians first').toBe(true);
  expect(near <= maximumKey, 'no visible row reaches the culled sentinel').toBe(true);
  expect(
    packSplatDepthKey(0.5, {mode: 'linear', depthMin: 0, depthMax: 1}) >
      packSplatDepthKey(0.9, {mode: 'linear', depthMin: 0, depthMax: 1}),
    'the linear distribution is monotone in view-space distance'
  ).toBe(true);
  expect(
    packSplatDepthKey(1000, {mode: 'float32'}) < packSplatDepthKey(1, {mode: 'float32'}),
    'the single-precision distribution orders the full 32-bit range'
  ).toBe(true);
});

it('the paged shaders keep their sparse-source and separate-depth-domain invariants', () => {
  expect(
    GPU_PAGED_SPLAT_PROJECTION_SHADER.includes('atomicAdd(&drawCommands'),
    'paged projection leaves the visible count to its feature pass'
  ).toBe(false);
  expect(
    GPU_PAGED_SPLAT_FEATURE_SHADER,
    'which publishes it once, after both harmonics and semantic filtering'
  ).toMatch(/atomicAdd\(&drawCommands\[1u\],\s*1u\)/);
  expect(
    GPU_PAGED_SPLAT_PROJECTION_SHADER,
    'paged rows are reached through an active-row index'
  ).toMatch(/batchRowIndex = activeRows\[projectedRowIndex\]/);
  expect(
    GPU_PAGED_SPLAT_PROJECTION_SHADER,
    'depth keys are global while projected records stay page-local'
  ).toMatch(/depthKeys\[graphUniforms\.batchOffset \+ projectedRowIndex\] = packSplatDepthKey\(/);
  expect(
    GPU_PAGED_SPLAT_PROJECTION_SHADER,
    'paged projection differentiates the perspective divide instead of finite differencing it'
  ).toMatch(/fn getProjectedScreenAxis\(/);
  expect(
    GPU_PAGED_SPLAT_RENDER_SHADER.includes('sortedIds'),
    'gathered paged records are already in painter order'
  ).toBe(false);
});

it('radiance-field and antialiasing behavior are feature flags, not forked shaders', () => {
  expect(
    Object.values(GPU_SPLAT_FEATURE_FLAGS).every(flag => Number.isInteger(flag) && flag > 0),
    'every flag is a distinct positive bit'
  ).toBe(true);
  expect(new Set(Object.values(GPU_SPLAT_FEATURE_FLAGS)).size, 'no two features share a bit').toBe(
    Object.keys(GPU_SPLAT_FEATURE_FLAGS).length
  );
  for (const [name, source] of Object.entries(ALL_SPLAT_SHADERS)) {
    if (name === 'gather') {
      continue;
    }
    expect(source, `${name}: reads the shared feature-flag word`).toMatch(/featureFlags/);
  }
  // Every stage that sizes or samples a Gaussian resolves its support radius through the one
  // shared function, so a flag cannot mean one thing during projection and another when drawing.
  for (const name of [
    'render',
    'compatibleRender',
    'picking',
    'compatiblePicking',
    'pagedRender'
  ]) {
    expect(
      ALL_SPLAT_SHADERS[name as keyof typeof ALL_SPLAT_SHADERS],
      `${name}: resolves the support radius through the shared helper`
    ).toMatch(/getSplatResolvedSupportRadius\(/);
  }
});
