// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUShapeGenerator,
  GPU_SHAPE_GENERATOR_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/grid-generators';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GPUShapeGenerator kernels parse and share the finite-bits helper', () => {
  for (const shape of ['circle', 'ellipse', 'sector'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `shape-${shape}`});
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `${shape}-view-${serial++}`, format, length);
    const nodes = new GPUShapeGenerator({
      shape,
      centers: view('float32x2', 2),
      radii: shape === 'ellipse' ? view('float32x2', 2) : view('float32', 2),
      ...(shape === 'sector' ? {bearings: view('float32x2', 2)} : {}),
      ...(shape === 'ellipse' ? {rotations: view('float32', 2)} : {}),
      maximumSegments: 8,
      parameters: view('float32', GPU_SHAPE_GENERATOR_PARAMETER_LENGTH),
      output: {
        positions: view('float32x2', 64),
        offsets: view('uint32', 3),
        vertexCount: view('uint32', 1)
      }
    } as never).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.length).toBeGreaterThan(0);
    for (const {id, source} of sources) {
      expect(() => new WgslReflect(source), id).not.toThrow();
      expect(source, id).not.toContain('isFinite32');
      expect(source.split('fn isFiniteBits(').length - 1, id).toBeLessThanOrEqual(1);
    }
    expect(sources.some(({source}) => source.includes('fn isFiniteBits('))).toBe(true);
  }
});
