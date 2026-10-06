// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPURasterSieveParameterValues,
  GPURasterPatchMetrics,
  GPURasterSieve
} from '../../../src/gpu-raster/raster-patches/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('GPURasterPatchMetrics and GPURasterSieve validate props and declare nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'raster-patches-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const labels = view('uint32', 12);
  const base = {width: 4, height: 3, labels};
  expect(() => new GPURasterPatchMetrics({...base, output: {}})).toThrow(/at least one column/);
  expect(
    () => new GPURasterPatchMetrics({...base, width: 5, output: {pixelCounts: view('uint32', 4)}})
  ).toThrow(/width \* height/);
  expect(
    () =>
      new GPURasterPatchMetrics({
        ...base,
        output: {pixelCounts: view('uint32', 4), areas: view('float32', 5)}
      })
  ).toThrow(/identical lengths/);
  expect(
    () =>
      new GPURasterPatchMetrics({
        ...base,
        affine: [1, 0, 0, 0, NaN, 0],
        output: {pixelCounts: view('uint32', 4)}
      })
  ).toThrow(/affine/);
  const metrics = new GPURasterPatchMetrics({
    ...base,
    output: {
      pixelCounts: view('uint32', 4),
      perimeters: view('float32', 4),
      minColumns: view('uint32', 4)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(metrics[0]).toBe('raster-patch-metrics-keys');
  expect(metrics).toContain('raster-patch-metrics-faces');
  expect(metrics).toContain('raster-patch-metrics-extents');
  expect(metrics).not.toContain('raster-patch-metrics-areas');

  const sieveBase = {
    ...base,
    patchCapacity: 4,
    parameters: view('uint32', 4),
    output: {labels: view('uint32', 12)}
  };
  expect(() => new GPURasterSieve({...sieveBase, patchCapacity: 0})).toThrow(/patchCapacity/);
  expect(() => new GPURasterSieve({...sieveBase, mode: 'fill' as 'merge'})).toThrow(/mode/);
  expect(() => new GPURasterSieve({...sieveBase, connectivity: 6 as 4})).toThrow(/connectivity/);
  expect(() => getGPURasterSieveParameterValues({minimumPixels: 0})).toThrow(/minimumPixels/);
  const remove = new GPURasterSieve(sieveBase).getCommandNodes(graph).map(node => node.id);
  const merge = new GPURasterSieve({...sieveBase, id: 'merge', mode: 'merge'})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(remove).not.toContain('raster-sieve-merge-targets');
  expect(merge).toContain('merge-merge-targets');
  expect(remove.at(-1)).toBe('raster-sieve-relabel');
});
