// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
  GPUProfilePeaks,
  getGPUProfilePeaksParameterValues
} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  concatenateProfiles,
  createGaussianBumps,
  findProfilePeaksOracle
} from './profile-peaks-oracle';

it('packs settings and validates them', () => {
  expect(GPU_PROFILE_PEAKS_PARAMETER_LENGTH).toBe(1);
  expect(Array.from(getGPUProfilePeaksParameterValues({minProminence: 2.5}))).toEqual([2.5]);
  expect(() => getGPUProfilePeaksParameterValues({minProminence: Number.NaN})).toThrow(/NaN/);
  expect(() => getGPUProfilePeaksParameterValues({minProminence: 1}, new Float32Array(0))).toThrow(
    /target/
  );
});

it('oracle reproduces the mt-image reference cases', () => {
  const offsets = [0, 200];
  const summits = findProfilePeaksOracle(
    createGaussianBumps(200, [
      [50.3, 10, 6],
      [120.7, 6, 6]
    ]),
    undefined,
    offsets,
    {window: 30, minProminence: 2}
  );
  expect(summits.peaks).toHaveLength(2);
  expect(summits.peaks[0].index).toBeCloseTo(50.3, 1);
  expect(summits.peaks[1].index).toBeCloseTo(120.7, 1);
  // Plateau: the first sample wins.
  const plateau = new Float32Array(60);
  plateau.fill(5, 25, 31);
  expect(
    findProfilePeaksOracle(plateau, undefined, [0, 60], {window: 15, minProminence: 1}).peaks.map(
      peak => peak.local
    )
  ).toEqual([25]);
  // Tiny and empty profiles.
  expect(
    findProfilePeaksOracle(Float32Array.of(1, 2), undefined, [0, 2], {window: 5, minProminence: 0})
      .peaks
  ).toEqual([]);
  expect(
    findProfilePeaksOracle(new Float32Array(0), undefined, [0], {window: 5, minProminence: 0}).peaks
  ).toEqual([]);
  // Chain: A > B > C keeps A and C.
  const chain = findProfilePeaksOracle(
    createGaussianBumps(80, [
      [30, 10, 2],
      [38, 8, 2],
      [46, 6, 2]
    ]),
    undefined,
    [0, 80],
    {window: 12, minProminence: 1, nms: 10}
  );
  expect(chain.peaks.map(peak => peak.local)).toEqual([30, 46]);
  // A gap with a sentinel value never contributes.
  const sentinel = createGaussianBumps(100, [[50, 10, 4]]);
  const validity = new Uint32Array(100).fill(1);
  sentinel[56] = 1e30;
  validity[56] = 0;
  const withSentinel = findProfilePeaksOracle(sentinel, validity, [0, 100], {
    window: 20,
    minProminence: 1
  });
  expect(withSentinel.peaks.map(peak => peak.local)).toEqual([50]);
  expect(withSentinel.peaks[0].prominence).toBeLessThan(11);
  // Wrap: a seam-straddling summit is found only when circular.
  const circular = Float32Array.from(
    {length: 60},
    (_, index) => 10 * Math.exp(-(Math.min(index, 60 - index) ** 2) / 8)
  );
  expect(
    findProfilePeaksOracle(circular, undefined, [0, 60], {window: 15, minProminence: 1}).peaks
  ).toEqual([]);
  const wrapped = findProfilePeaksOracle(circular, undefined, [0, 60], {
    window: 15,
    minProminence: 1,
    wrap: true
  });
  expect(wrapped.peaks.map(peak => peak.local)).toEqual([0]);
  expect(wrapped.peaks[0].index).toBeGreaterThanOrEqual(0);
  expect(wrapped.peaks[0].index).toBeLessThan(60);
  // Concatenation helper.
  expect(Array.from(concatenateProfiles([[1], [], [2, 3]]).offsets)).toEqual([0, 1, 1, 3]);
});

it('GPUProfilePeaks validates props and schedules its nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Record<string, unknown> = {}) => {
    instance++;
    return new GPUProfilePeaks({
      values: createTransientView(graph, `values-${instance}`, 'float32', 40),
      offsets: createTransientView(graph, `offsets-${instance}`, 'uint32', 3),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 1),
      window: 8,
      nmsRounds: 2,
      peakMask: createTransientView(graph, `mask-${instance}`, 'uint32', 40),
      ...overrides
    });
  };
  const graphForNodes = () => new GPUCommandGraph(device);
  const build = (overrides: Record<string, unknown> = {}) => {
    const target = graphForNodes();
    return new GPUProfilePeaks({
      values: createTransientView(target, 'values', 'float32', 40),
      offsets: createTransientView(target, 'offsets', 'uint32', 3),
      settings: createTransientView(target, 'settings', 'float32', 1),
      window: 8,
      nmsRounds: 2,
      peakMask: createTransientView(target, 'mask', 'uint32', 40),
      ...overrides
    }).getCommandNodes(target);
  };
  expect(create().recipe).toBe('profile-peaks');
  expect(create().nms).toBe(2);
  expect(create({window: 40}).nms).toBe(10);
  expect(build().map(node => node.id)).toEqual([
    'profile-peaks-candidates',
    'profile-peaks-nms-0',
    'profile-peaks-nms-1',
    'profile-peaks-outputs'
  ]);
  const target = graphForNodes();
  const full = new GPUProfilePeaks({
    id: 'peaks',
    values: createTransientView(target, 'values', 'float32', 40),
    validity: createTransientView(target, 'validity', 'uint32', 40),
    offsets: createTransientView(target, 'offsets', 'uint32', 3),
    settings: createTransientView(target, 'settings', 'float32', 1),
    window: 8,
    nmsRounds: 1,
    prominence: createTransientView(target, 'prominence', 'float32', 40),
    refinedIndex: createTransientView(target, 'refined-index', 'float32', 40),
    refinedValue: createTransientView(target, 'refined-value', 'float32', 40),
    converged: createTransientView(target, 'converged', 'uint32', 1),
    output: {
      ids: createTransientView(target, 'ids', 'uint32', 5),
      count: createTransientView(target, 'count', 'uint32', 1),
      overflow: createTransientView(target, 'overflow', 'uint32', 1)
    }
  });
  const ids = full.getCommandNodes(target).map(node => node.id);
  expect(ids.slice(0, 5)).toEqual([
    'peaks-candidates',
    'peaks-nms-0',
    'peaks-converged-reset',
    'peaks-converged',
    'peaks-outputs'
  ]);
  expect(ids).toContain('peaks-refined-index');
  expect(ids).toContain('peaks-publish');
  expect(ids.some(id => id.startsWith('peaks-compaction'))).toBe(true);

  expect(() => create({peakMask: undefined})).toThrow(/at least one output/);
  expect(() => create({window: 0})).toThrow(/window/);
  expect(() => create({window: 2.5})).toThrow(/window/);
  expect(() => create({window: 5000})).toThrow(/window/);
  expect(() => create({minSide: -1})).toThrow(/minSide/);
  expect(() => create({nms: -1})).toThrow(/nms/);
  expect(() => create({nms: Number.NaN})).toThrow(/nms/);
  expect(() => create({nmsRounds: 2000})).toThrow(/nmsRounds/);
  expect(() => create({nmsRounds: 1.5})).toThrow(/nmsRounds/);
  expect(() => create({offsets: createTransientView(graph, 'empty-offsets', 'uint32', 0)})).toThrow(
    /offsets/
  );
  expect(() =>
    create({validity: createTransientView(graph, 'short-validity', 'uint32', 39)})
  ).toThrow(/validity/);
  expect(() => create({peakMask: createTransientView(graph, 'short-mask', 'uint32', 39)})).toThrow(
    /one value per sample/
  );
  expect(() =>
    create({prominence: createTransientView(graph, 'wrong-format', 'uint32', 40)})
  ).toThrow(/prominence/);
  expect(() =>
    create({settings: createTransientView(graph, 'empty-settings', 'float32', 0)})
  ).toThrow(/settings/);
  const shared = createTransientView(graph, 'shared', 'uint32', 40);
  expect(() => create({validity: shared, peakMask: shared})).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUProfilePeaks({
      values: createTransientView(otherGraph, 'foreign-values', 'float32', 40),
      offsets: createTransientView(graph, 'foreign-offsets', 'uint32', 3),
      settings: createTransientView(graph, 'foreign-settings', 'float32', 1),
      window: 8,
      peakMask: createTransientView(graph, 'foreign-mask', 'uint32', 40)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
