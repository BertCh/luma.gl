// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUReliefShadingParameterValues,
  GPU_RELIEF_SHADING_CONTRAST_PIVOT,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPUReliefShading
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('getGPUReliefShadingParameterValues packs the Imhof, curvature, and contrast slots', () => {
  expect(GPU_RELIEF_SHADING_PARAMETER_LENGTH).toBe(80);
  expect(GPU_RELIEF_SHADING_CONTRAST_PIVOT).toBe(0.72);
  const defaults = getGPUReliefShadingParameterValues({cellSize: [1, 1]});
  // Slot 19 is zero for non-swing weighting, curvature strength defaults to 1, contrast is off.
  expect(defaults[6]).toBe(1);
  expect(Array.from(defaults.slice(19, 20))).toEqual([0]);
  expect(Array.from(defaults.slice(76, 80))).toEqual([1, 0, 0, 0]);
  const swing = getGPUReliefShadingParameterValues({
    cellSize: [1, 1],
    lights: [{azimuthDegrees: 315}],
    lightWeighting: 'imhof-swing'
  });
  expect(swing[6]).toBe(2);
  expect(swing[19]).toBe(65);
  const custom = getGPUReliefShadingParameterValues({
    cellSize: [1, 1],
    lightWeighting: 'imhof-swing',
    imhofSwingDegrees: 0,
    curvatureStrength: 2.5,
    contrastLowElevation: 100,
    contrastHighElevation: 900,
    contrastStrength: 0.4
  });
  expect(custom[19]).toBe(0);
  expect(Array.from(custom.slice(76, 80))).toEqual([2.5, 100, 900, Math.fround(0.4)]);
  // Slots 20-75 are untouched by the new options.
  const plain = getGPUReliefShadingParameterValues({cellSize: [1, 1]});
  expect(Array.from(custom.slice(20, 76))).toEqual(Array.from(plain.slice(20, 76)));
});

it('GPUReliefShading wires the swing and detail nodes and validates curvature', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const width = 6;
  const height = 5;
  const view = (id: string, length = width * height) =>
    createTransientView(graph, id, 'float32', length);
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('elevation')}
  };
  const settings = view('settings', GPU_RELIEF_SHADING_PARAMETER_LENGTH);
  const withoutCurvature = new GPUReliefShading({
    id: 'plain',
    width,
    height,
    elevation,
    settings,
    hillshade: view('plain-hillshade'),
    relief: view('plain-relief')
  }).getCommandNodes(graph);
  const withCurvature = new GPUReliefShading({
    id: 'curved',
    width,
    height,
    elevation,
    settings,
    curvature: view('curvature'),
    imhofSwing: true,
    textureShade: view('texture-shade'),
    skyViewFactor: view('svf'),
    hillshade: view('curved-hillshade'),
    relief: view('curved-relief'),
    color: createTransientView(graph, 'curved-color', 'uint32', width * height),
    validity: createTransientView(graph, 'curved-validity', 'uint32', width * height)
  }).getCommandNodes(graph);
  // The default node list is unchanged; the detail pre-kernel and swing node are additions.
  expect(withoutCurvature.some(node => node.id.includes('swing'))).toBe(false);
  expect(withoutCurvature.some(node => node.id.includes('detail'))).toBe(false);
  expect(withCurvature.length).toBe(withoutCurvature.length + 2);
  expect(withCurvature.some(node => node.id === 'curved-detail')).toBe(true);
  expect(withCurvature.some(node => node.id === 'curved-hillshade-swing')).toBe(true);
  expect(
    () =>
      new GPUReliefShading({
        width,
        height,
        elevation,
        settings,
        curvature: view('short-curvature', 3),
        relief: view('short-relief')
      })
  ).toThrow(/curvature/);
  // Curvature alone fits in the compose kernel and needs no detail node.
  const direct = new GPUReliefShading({
    id: 'direct',
    width,
    height,
    elevation,
    settings,
    curvature: view('direct-curvature'),
    relief: view('direct-relief')
  }).getCommandNodes(graph);
  expect(direct.length).toBe(withoutCurvature.length);
  expect(direct.some(node => node.id.includes('detail'))).toBe(false);
});
